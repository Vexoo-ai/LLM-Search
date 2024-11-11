import os
import json
import asyncio
import numpy as np
from functools import lru_cache
from dotenv import load_dotenv
from aiohttp import ClientSession, TCPConnector
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMRanker
from sentence_transformers import SentenceTransformer
from serpapi import GoogleSearch
import aiohttp
import ssl
import certifi
from anthropic import AsyncAnthropicBedrock

load_dotenv()

# Load environment variables
SERPAPI_API_KEY = os.getenv('serpapi_api_key')
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

anthropic_client = AsyncAnthropicBedrock(
    aws_access_key=AWS_ACCESS_KEY_ID,
    aws_secret_key=AWS_SECRET_ACCESS_KEY,
    aws_region=AWS_REGION
)


class NewsSearchEngine:
    def __init__(self):
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    @lru_cache(maxsize=1000)
    def get_news_embedding(self, text):
        return self.sentence_model.encode(text)
    
    async def fetch_news_results(self, query, num_results=10):
        serpapi_api_key = os.getenv('serpapi_api_key')
        
        params = {
            "api_key": serpapi_api_key,
            "engine": "google_news",
            "q": query,
            "num": num_results
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        return results.get("news_results", [])
    
    def format_news_results(self, search_data):
        formatted_results = []
        for result in search_data:
            formatted_result = {
                'source': result.get('source', ''),
                'date': result.get('date', ''),
                'title': result.get('title', ''),
                'snippet': result.get('snippet', ''),
                'link': result.get('link', '')
            }
            formatted_results.append(formatted_result)
        return formatted_results
    
    async def rank_news_results(self, results, query):
        query_embedding = await asyncio.to_thread(self.get_news_embedding, query)
        
        features = []
        for i, result in enumerate(results):
            title = result.get('title', '')
            snippet = result.get('snippet', '')
            
            title_embedding = await asyncio.to_thread(self.get_news_embedding, title)
            snippet_embedding = await asyncio.to_thread(self.get_news_embedding, snippet)
            
            semantic_similarity_title = cosine_similarity([query_embedding], [title_embedding])[0][0]
            semantic_similarity_snippet = cosine_similarity([query_embedding], [snippet_embedding])[0][0]
            
            features.append([
                i,
                semantic_similarity_snippet,
                semantic_similarity_title
            ])
        
        X = np.array(features)
        X = MinMaxScaler().fit_transform(X)
        
        y = np.arange(len(features))[::-1]
        y = MinMaxScaler().fit_transform(y.reshape(-1, 1)).ravel()
        y = (y * 20).astype(int)
        
        group = [len(features)]
        
        model = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            boosting_type="dart",
            n_estimators=100,
            importance_type="gain",
            max_position=21
        )
        model.fit(X, y, group=group)
        
        scores = model.predict(X)
        
        sorted_indices = np.argsort(scores)[::-1]
        ranked_results = [results[i] for i in sorted_indices]
        
        return ranked_results

async def call_mistral_news_stream(query, search_results, news_search_engine):
    try:
        url = os.getenv("AZURE_AI_ENDPOINT")
        api_key = os.getenv("AZURE_AI_API_KEY")
        ssl_verify = os.getenv("SSL_VERIFY", "True").lower() == "true"

        if not url or not api_key:
            raise ValueError("AZURE_AI_ENDPOINT or AZURE_AI_API_KEY environment variable is missing")

        formatted_results = news_search_engine.format_news_results(search_results)
        context = '\n\n'.join([f"Source: {r['source']}\nDate: {r['date']}\nTitle: {r['title']}\nSnippet: {r['snippet']}\nURL: {r['link']}" for r in formatted_results])

        system_message = """You are an advanced AI assistant specializing in providing informative responses based on news articles. Your task is to synthesize the given search results and generate a comprehensive response to the user's query, focusing on the most relevant and up-to-date information. Your responses should be characterized by their depth, accuracy, and attention to detail. Adapt your approach based on the nature of the query:

        1. For current events:
           - Provide a chronological overview of the event, including key developments and turning points.
           - Analyze the implications from multiple angles: social, economic, political, etc.
           - Discuss expert opinions and predictions about future developments.
           - Address any conflicting information or perspectives from different sources.

        2. For trend analysis:
           - Identify and explain emerging trends in the news.
           - Discuss the factors driving these trends and their potential long-term impacts.
           - Provide context by relating current trends to historical patterns or similar events.

        3. For fact-checking queries:
           - Clearly state the claim being examined.
           - Present evidence from reliable sources to support or refute the claim.
           - Explain any nuances or complexities surrounding the issue.
           - Provide a clear conclusion about the accuracy of the claim.

        4. For in-depth topic exploration:
           - Offer a comprehensive overview of the topic, exploring its various facets.
           - Discuss different perspectives and debates surrounding the issue.
           - Provide relevant background information and historical context.
           - Analyze potential future developments or implications.

        General guidelines:
        - Use clear, concise language while maintaining depth and nuance in your explanations.
        - Cite sources for key information, using inline citations [1], [2], etc.
        - Address potential biases in news reporting and present a balanced view when appropriate.
        - Highlight any limitations in current knowledge or areas where information is still developing.
        - Organize your response logically, using paragraphs, bullet points, or numbering as appropriate.
        - Anticipate and address potential follow-up questions in your response.

        Your goal is to provide a response that not only answers the query comprehensively but also enhances the user's understanding of the news topic, offering insights and context they might not have gathered from the individual articles alone."""
        
        user_message = f"""Query: {query}

        News Search Results:
        {context}

        Based on these news search results and your extensive knowledge, provide a comprehensive and insightful response to the query. Your answer should synthesize information from multiple sources, highlight key developments, and provide a nuanced analysis of the topic. Ensure your response is complete and addresses all aspects of the query comprehensively."""

        data = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": 2000,
            "temperature": 0.5,
            "stream": True,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        ssl_context = ssl.create_default_context(cafile=certifi.where()) if ssl_verify else ssl.create_default_context()
        if not ssl_verify:
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers, ssl=ssl_context) as resp:
                if resp.status != 200:
                    error_msg = await resp.text()
                    raise Exception(f"API request failed with status {resp.status}: {error_msg}")

                async for line in resp.content:
                    if line:
                        try:
                            json_line = json.loads(line.decode('utf-8').replace('data: ', '', 1))
                            content = json_line['choices'][0]['delta'].get('content', '')
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            yield f"\nError processing stream: {str(e)}\n"

    except Exception as e:
        yield f"\nError calling Azure AI: {str(e)}\n"
        yield await news_fallback_response(query, search_results, news_search_engine)

async def call_claude_news_stream(query, search_results, news_search_engine):
    try:
        formatted_results = news_search_engine.format_news_results(search_results)
        context = '\n\n'.join([f"Source: {r['source']}\nDate: {r['date']}\nTitle: {r['title']}\nSnippet: {r['snippet']}\nURL: {r['link']}" for r in formatted_results])

        system_message = """You are an advanced AI assistant specializing in providing informative responses based on news articles. Your task is to synthesize the given search results and generate a comprehensive response to the user's query, focusing on the most relevant and up-to-date information. Your responses should be characterized by their depth, accuracy, and attention to detail. Adapt your approach based on the nature of the query:

        1. For current events:
           - Provide a chronological overview of the event, including key developments and turning points.
           - Analyze the implications from multiple angles: social, economic, political, etc.
           - Discuss expert opinions and predictions about future developments.
           - Address any conflicting information or perspectives from different sources.

        2. For trend analysis:
           - Identify and explain emerging trends in the news.
           - Discuss the factors driving these trends and their potential long-term impacts.
           - Provide context by relating current trends to historical patterns or similar events.

        3. For fact-checking queries:
           - Clearly state the claim being examined.
           - Present evidence from reliable sources to support or refute the claim.
           - Explain any nuances or complexities surrounding the issue.
           - Provide a clear conclusion about the accuracy of the claim.

        4. For in-depth topic exploration:
           - Offer a comprehensive overview of the topic, exploring its various facets.
           - Discuss different perspectives and debates surrounding the issue.
           - Provide relevant background information and historical context.
           - Analyze potential future developments or implications.

        General guidelines:
        - Use clear, concise language while maintaining depth and nuance in your explanations.
        - Cite sources for key information, using inline citations [1], [2], etc.
        - Address potential biases in news reporting and present a balanced view when appropriate.
        - Highlight any limitations in current knowledge or areas where information is still developing.
        - Organize your response logically, using paragraphs, bullet points, or numbering as appropriate.
        - Anticipate and address potential follow-up questions in your response.

        Your goal is to provide a response that not only answers the query comprehensively but also enhances the user's understanding of the news topic, offering insights and context they might not have gathered from the individual articles alone."""

        user_message = f"""Query: {query}

        News Search Results:
        {context}

        Based on these news search results and your extensive knowledge, provide a comprehensive and insightful response to the query. Your answer should synthesize information from multiple sources, highlight key developments, and provide a nuanced analysis of the topic. Include relevant citations and suggest areas for further reading or monitoring."""

        async with anthropic_client.messages.stream(
            max_tokens=2046,
            messages=[
                {
                    "role": "user",
                    "content": user_message,
                }
            ],
            model="anthropic.claude-3-5-sonnet-20240620-v1:0",
            system=system_message,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    except Exception as e:
        yield f"\nError calling Claude API: {str(e)}\n"
        yield await news_fallback_response(query, search_results, news_search_engine)

async def news_fallback_response(query, search_results, news_search_engine):
    formatted_results = news_search_engine.format_news_results(search_results)
    response = f"I apologize, but I couldn't access the AI language models due to an error. However, I can provide you with a summary of the top news search results for your query: '{query}'\n\n"
    
    for idx, result in enumerate(formatted_results[:5], 1):
        response += f"{idx}. {result['title']}\n   Source: {result['source']}\n   Date: {result['date']}\n   Snippet: {result['snippet'][:150]}...\n   URL: {result['link']}\n\n"
    
    response += "For more detailed information, please visit the source websites or try your query again later when the AI models are available."
    return response