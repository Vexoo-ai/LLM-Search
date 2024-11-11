# scholar.py

import os
from datetime import datetime
from dotenv import load_dotenv
from serpapi import GoogleSearch
import aiohttp
import ssl
import certifi
from anthropic import AsyncAnthropicBedrock
import json

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

class ScholarSearchEngine:
    def __init__(self):
        self.api_key = os.getenv('serpapi_api_key')

    async def fetch_google_scholar_results(self, query, num_results=10):
        params = {
            "api_key": self.api_key,
            "engine": "google_scholar",
            "q": query,
            "num": num_results,
            "sort": "date",
            "as_ylo": datetime.now().year - 5
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        return results.get("organic_results", [])

    async def format_scholar_results(self, search_data):
        formatted_results = []
        for result in search_data:
            formatted_result = {
                'source': result.get('publication_info', {}).get('summary', ''),
                'date': result.get('publication_info', {}).get('summary', ''),
                'title': result.get('title', ''),
                'snippet': result.get('snippet', ''),
                'link': result.get('link', ''),
                'citations': result.get('inline_links', {}).get('cited_by', {}).get('total', 0)
            }
            formatted_results.append(formatted_result)
        return formatted_results

async def call_mistral_scholar_stream(query, search_results, scholar_search_engine):
    try:
        url = os.getenv("AZURE_AI_ENDPOINT")
        api_key = os.getenv("AZURE_AI_API_KEY")
        ssl_verify = os.getenv("SSL_VERIFY", "True").lower() == "true"

        if not url or not api_key:
            raise ValueError("AZURE_AI_ENDPOINT or AZURE_AI_API_KEY environment variable is missing")

        formatted_results = await scholar_search_engine.format_scholar_results(search_results)
        context = '\n\n'.join([f"Title: {r['title']}\nAuthors: {r['source']}\nSnippet: {r['snippet']}\nCitations: {r['citations']}\nURL: {r['link']}" for r in formatted_results])

        system_message = """You are an advanced AI assistant specializing in academic and scientific research. Your task is to provide comprehensive and insightful responses to queries based on Google Scholar search results. Your knowledge spans a wide range of academic disciplines. Your responses should be characterized by their depth, precision, and attention to scholarly details..."""
        
        user_message = f"""Query: {query}

        Google Scholar Search Results:
        {context}

        Based on these search results and your extensive knowledge of academic literature, provide a comprehensive and insightful response to the query. Ensure your response is complete and addresses all aspects of the query comprehensively."""

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
        yield await scholar_fallback_response(query, search_results, scholar_search_engine)

async def call_claude_scholar_stream(query, search_results, scholar_search_engine):
    try:
        formatted_results = await scholar_search_engine.format_scholar_results(search_results)
        context = '\n\n'.join([f"Title: {r['title']}\nAuthors: {r['source']}\nSnippet: {r['snippet']}\nCitations: {r['citations']}\nURL: {r['link']}" for r in formatted_results])

        system_message = """You are an advanced AI assistant specializing in academic and scientific research. Your task is to provide comprehensive and insightful responses to queries based on Google Scholar search results. Your knowledge spans a wide range of academic disciplines. Your responses should be characterized by their depth, precision, and attention to scholarly details..."""

        user_message = f"""Query: {query}

        Google Scholar Search Results:
        {context}

        Based on these search results and your extensive knowledge of academic literature, provide a comprehensive and insightful response to the query. Your answer should synthesize information from multiple sources, highlight key findings and methodologies, and provide a critical analysis of the current state of research on this topic. Include relevant citations and suggest directions for future research."""

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
        yield f"\nError calling Azure AI: {str(e)}\n"
        fallback = await scholar_fallback_response(query, search_results, scholar_search_engine)
        yield fallback

async def scholar_fallback_response(query, search_results, scholar_search_engine):
    formatted_results = await scholar_search_engine.format_scholar_results(search_results)
    response = f"I apologize, but I couldn't access the AI language models due to an error. However, I can provide you with a summary of the top Google Scholar results for your query: '{query}'\n\n"
    
    for idx, result in enumerate(formatted_results[:5], 1):
        response += f"{idx}. {result['title']}\n   Authors: {result['source']}\n   Snippet: {result['snippet'][:150]}...\n   Citations: {result['citations']}\n   URL: {result['link']}\n\n"
    
    response += "For more detailed information, please visit the source websites or try your query again later when the AI models are available."
    return response