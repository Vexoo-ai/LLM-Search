import os
import re
import json
import asyncio
import numpy as np
import pandas as pd
from typing import List, Dict, Any, AsyncIterator
from datetime import datetime, timedelta
from dotenv import load_dotenv
from serpapi import GoogleSearch, BingSearch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMRanker
from sentence_transformers import SentenceTransformer
from urllib.parse import urlparse
from functools import lru_cache
from anthropic import AsyncAnthropicBedrock

load_dotenv()

serpapi_api_key = os.getenv('serpapi_api_key')

# Load environment variables
SERPAPI_API_KEY = os.getenv('serpapi_api_key')
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

anthropic_client = AsyncAnthropicBedrock(aws_access_key=AWS_ACCESS_KEY_ID,
                                            aws_secret_key=AWS_SECRET_ACCESS_KEY,
                                            aws_region=AWS_REGION)

class SearchProcessor:
    def __init__(self):
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

    @lru_cache(maxsize=1000)
    def get_domain_authority(self, domain: str) -> float:
        return np.random.random()

    @lru_cache(maxsize=1000)
    def get_content_freshness(self, result: str) -> float:
        return np.random.random()

    def process_and_rank_results(self, all_results: Dict[str, List[Dict[str, Any]]], query: str) -> Dict[str, List[Dict[str, Any]]]:
        combined_results = []
        for engine, results in all_results.items():
            for result in results:
                result['engine'] = engine
                combined_results.append(result)
        
        if len(combined_results) < 2:
            return {'organic_results': combined_results}
        
        query_embedding = self.sentence_model.encode([query])[0]
        
        features = []
        for result in combined_results:
            snippet = result.get('snippet', '')
            title = result.get('title', '')
            url = result.get('link', '')
            
            snippet_embedding = self.sentence_model.encode([snippet])[0]
            title_embedding = self.sentence_model.encode([title])[0]
            
            semantic_similarity_snippet = cosine_similarity([query_embedding], [snippet_embedding])[0][0]
            semantic_similarity_title = cosine_similarity([query_embedding], [title_embedding])[0][0]
            
            domain = urlparse(url).netloc
            domain_authority = self.get_domain_authority(domain)
            
            features.append([
                result.get('position', 0),
                len(snippet),
                snippet.count(query),
                semantic_similarity_snippet,
                semantic_similarity_title,
                domain_authority,
                self.get_content_freshness(json.dumps(result)),
                int(engine == 'google'),
                int(engine == 'bing'),
                int(engine == 'duckduckgo')
            ])
        
        X = np.array(features)
        
        if X.shape[0] >= 2:
            X = MinMaxScaler().fit_transform(X)
            y = np.arange(len(features))[::-1]
            y = MinMaxScaler().fit_transform(y.reshape(-1, 1)).ravel()
            y = (y * 30).astype(int)
            group = [len(features)]
            
            model = LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                boosting_type="dart",
                n_estimators=100,
                importance_type="gain",
                max_position=31
            )
            model.fit(X, y, group=group)
            scores = model.predict(X)
            sorted_indices = np.argsort(scores)[::-1]
            ranked_results = [combined_results[i] for i in sorted_indices]
            diverse_results = self.ensure_diversity(ranked_results)
        else:
            diverse_results = combined_results
        
        return {'organic_results': diverse_results}

    def ensure_diversity(self, results: List[Dict[str, Any]], diversity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        diverse_results = []
        domains_included = set()
        
        for result in results:
            domain = urlparse(result.get('link', '')).netloc
            if domain not in domains_included or len(diverse_results) < 5:
                diverse_results.append(result)
                domains_included.add(domain)
            
            if len(diverse_results) >= 10:
                break
        
        return diverse_results

    def format_search_results(self, search_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        formatted_results = []
        for result in search_data.get('organic_results', []):
            displayed_link = result.get('displayed_link', '')
            source = None
            if displayed_link:
                if '://' in displayed_link:
                    displayed_link = displayed_link.split('://', 1)[-1]
                source = displayed_link.split('/')[0]

            formatted_result = {
                'source': source,
                'date': None,
                'title': result.get('title', ''),
                'snippet': result.get('snippet', ''),
                'highlight': result.get('snippet_highlighted_words', ''),
                'engine': result.get('engine', '')
            }
            formatted_results.append(formatted_result)
        return formatted_results
    
    def get_context_search(self, query: str):
        params = {
            "q": query, 
            "engine": "google", 
            "hl": "en", 
            "gl": "us",
            "num": "3",
            "google_domain": "google.com", 
            "api_key": serpapi_api_key,
        }
        search = GoogleSearch(params)
        search_results = search.get_dict()
        formatted_results = self.format_search_results(search_results)
        context = '\n\n'.join([f"Source: {r['source']}\nTitle: {r['title']}\nSnippet: {r['snippet']}" for r in formatted_results])
        print("----------------------Context-----------------------")
        print(context)
        print("----------------------------------------")
        return context

    def call_search_engines(self, query: str) -> Dict[str, Any]:
        def search(engine: str, params: Dict[str, Any]) -> Dict[str, Any]:
            try:
                if engine == "google":
                    search = GoogleSearch(params)
                elif engine == "bing":
                    search = BingSearch(params)
                else:
                    search = GoogleSearch(params)  # DuckDuckGo uses GoogleSearch
                return search.get_dict()
            except Exception as e:
                print(f"Error in {engine} search: {str(e)}")
                return {"organic_results": []}

        results = [
            search("google", {
                "q": query, "engine": "google", "hl": "en", "gl": "us",
                "google_domain": "google.com", "api_key": serpapi_api_key,
            }),
            search("bing", {
                "q": query, "engine": "bing", "cc": "US", "api_key": serpapi_api_key,
            }),
            search("duckduckgo", {
                "q": query, "engine": "duckduckgo", "api_key": serpapi_api_key,
            })
        ]
        
        all_results = {
            "google": results[0].get('organic_results', []),
            "bing": results[1].get('organic_results', []),
            "duckduckgo": results[2].get('organic_results', [])
        }
        
        search_results = self.process_and_rank_results(all_results, query)
        formatted_results = self.format_search_results(search_results)
        return formatted_results
    
    def extract_all_questions(self, results):
        all_questions = []
        
        for category, questions in results.items():
            all_questions.extend(questions)
        
        return all_questions
    
    def merge_list_and_dict(self, data_list, data_dict):
        merged_result = {}
        
        for i, (key, questions) in enumerate(data_dict.items()):
            merged_key = f"Section_{i+1}"
            start_index = i * 9  # Each section has 3 sources
            end_index = start_index + 9
            merged_result[merged_key] = {
                "research_area": key,
                "questions": questions,
                "sources": data_list[start_index:end_index] if start_index < len(data_list) else []
            }
        
        return json.dumps(merged_result, indent=2)
    
    def extract_source(self, data):
        sources = []
        for section in data.values():
            if 'sources' in section:
                for source in section['sources']:
                    if 'source' in source and 'title' in source:
                        sources.append(f"Source: {source['source']}\nTitle: {source['title']}\n")
        return "\n".join(sources)

async def generate_queries_and_sections(query):
    search_processor = SearchProcessor()

    context = search_processor.get_context_search(query)

    system_message = """
        You are a search engine specialist with expertise in crafting precise, search engine-friendly queries to retrieve the most relevant and insightful information on any given topic. 
        Based on the user query and internet search results obtained, generate a set of 3 search-focused search queries. Each query should be optimized for search engines, covering key aspects such as historical context, current developments, expert opinions, and diverse perspectives. 
        Ensure that the queries are specific, actionable, and designed to yield high-quality search results.

        Following are the examples:
        Latest news and articles on the current status of topic.
        Expert opinions and analyses on the implications of topic.
        Impact of topic on global/regional politics, economy, and society

        Now remember these 3 research-focused search queries:
        search-focused query 1,
        search-focused query 2,
        search-focused query 3

        Your next task is
        Given the set of 3 search-focused search queries generate a set of related questions that dive deeper into the topic, aiming to uncover various aspects, perspectives, or details. 
        The generated questions should be relevant, clear, and designed to prompt further exploration or clarification on the topic. 
        Provide at least 3 related questions for each set of query.

        You should analyse each of the 3 queries and for each query you will generate a list of 3 questions. In total, you 
        will generate a new set of 9 questions.

        Output format:
        {
        "search-focused search query 1": [
            "related question 1 based on search-focused search query 1",
            "related question 2 based on search-focused search query 1",
            "related question 3 based on search-focused search query 1"
        ],
        "search-focused search query 2": [
            "related question 1 based on search-focused search query 2",
            "related question 2 based on search-focused search query 2",
            "related question 3 based on search-focused search query 2"
        ],
        "search-focused search query 3": [
            "related question 1 based on search-focused search query 3",
            "related question 2 based on search-focused search query 3",
            "related question 3 based on search-focused search query 3"
        ]
        }
    """

    user_message = f"""Query: {query}

    Internet Search Results:
    {context}
    """
    message = await anthropic_client.messages.create(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        max_tokens=500,
        system=system_message,
        messages=[
                {"role": "user", "content": user_message}
        ]
    )
    json_match = re.search(r'\{[\s\S]*?\}', message.content[0].text)
    print("----------------------- Extracted JSON Response -------------------------------")
    print(json_match.group(0))
    print("-------------------------------------------------------")
    print("----------------------- Extracted Questions -------------------------------")
    generated_json_queries = json.loads(json_match.group(0))
    extracted_questions = search_processor.extract_all_questions(generated_json_queries)
    print(extracted_questions)
    print("-------------------------------------------------------")
    print("--------------------------------Search Process from SERP -----------------------")
    all_results = []
    for questions in extracted_questions:
        results = search_processor.call_search_engines(questions)
        all_results.extend(results)

    merged_data = search_processor.merge_list_and_dict(all_results, generated_json_queries)
    parsed_data = json.loads(merged_data)

    return parsed_data

async def vexoo_claude_pro_search(query, data):
    search_processor = SearchProcessor()
    
    web_results = search_processor.extract_source(data)
    
    system_prompt = f"""
    You are an advanced AI assistant with real-time internet search capabilities, designed to provide exceptionally detailed, comprehensive, and insightful responses to
    any type of query. Your knowledge spans a wide range of topics including but not limited to general knowledge, current events, science, technology, programming, arts,
    and more. Your responses should be characterized by their depth, precision, and attention to nuanced details.    
    
    Instructions:
    Analyze the Query: Understand the user's question or inquiry.
    Evaluate the Sources: Review the provided web search results, noting the relevance, credibility, and citation count of each source.
    Synthesize Information: Combine insights from the most credible and relevant sources to form a comprehensive and detailed answer.
    Cite Sources: Where necessary, cite the sources in your response to support your statements.

    Output:
    Provide a clear, concise, and well-reasoned answer to the user query based on the information from the web search results. 
    """

    user_message = f"""Query: {query}

    Internet Search Results:
    {web_results}

    Based on these search results and your extensive knowledge, provide an exceptionally detailed, comprehensive, and insightful response to the query. Your answer should be characterized by its depth, precision, and attention to nuanced details. Ensure your response is directly relevant, incorporates the latest information, and is tailored to the specific nature and depth of the question. If the query is coding-related, include detailed code examples with explanations. Anticipate and address potential follow-up questions in your response. Remember to include a detailed "Sources" section at the end of your response, with numbered references and URLs for further reading."""

    async with anthropic_client.messages.stream(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        max_tokens=2046,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_message}
        ]
    ) as stream:
        async for text in stream.text_stream:
            yield text


