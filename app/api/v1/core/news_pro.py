import os
import json
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from serpapi import GoogleSearch
from anthropic import AsyncAnthropicBedrock
import logging
import re

# Load environment variables and set up logging
load_dotenv()

# Load environment variables
SERPAPI_API_KEY = os.getenv('serpapi_api_key')
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

anthropic_client = AsyncAnthropicBedrock(aws_access_key=AWS_ACCESS_KEY_ID,
                                            aws_secret_key=AWS_SECRET_ACCESS_KEY,
                                            aws_region=AWS_REGION)

class NewsProSearch:
    def __init__(self):
        self.serpapi_key = os.getenv('serpapi_api_key')
        self.anthropic_client = anthropic_client


    async def fetch_news_results(self, query, num_results=10):
        params = {
            "api_key": self.serpapi_key,
            "engine": "google_news",
            "q": query,
            "num": num_results
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        return results.get("news_results", [])

    def format_search_results(self, search_data):
        return [{
            'source': result.get('source', ''),
            'date': result.get('date', ''),
            'title': result.get('title', ''),
            'snippet': result.get('snippet', ''),
            'link': result.get('link', '')
        } for result in search_data]

    async def generate_research_areas_and_queries(self, query):
        system_prompt = """
            You are a search engine specialist with expertise in crafting precise, search engine-friendly queries to retrieve the most relevant and insightful information on any given topic. 
            Based on the user query, generate a set of 3 research-focused search queries. Each query should be optimized for search engines, covering key aspects such as historical context, current developments, expert opinions, and diverse perspectives. 
            Ensure that the queries are specific, actionable, and designed to yield high-quality search results.

            Following are the examples:
            Latest news and articles on the current status of topic.
            Expert opinions and analyses on the implications of topic.
            Impact of topic on global/regional politics, economy, and society

            Now remember these 3 research-focused search queries:
            research-focused search query 1,
            research-focused search query 2,
            research-focused search query 3

            Your next task is
            Given the set of 3 research-focused search queries generate a set of related questions that dive deeper into the topic, aiming to uncover various aspects, perspectives, or details. 
            The generated questions should be relevant, clear, and designed to prompt further exploration or clarification on the topic. 
            Provide at least 3 related questions for each set of query.

            You should analyse each of the 3 queries and for each query you will generate a list of 3 questions. In total, you 
            will generate a new set of 9 questions.

            Output format:
            {
            "research-focused search query 1": [
                "related question 1 based on research-focused search query 1",
                "related question 2 based on research-focused search query 1",
                "related question 3 based on research-focused search query 1"
            ],
            "research-focused search query 2": [
                "related question 1 based on research-focused search query 2",
                "related question 2 based on research-focused search query 2",
                "related question 3 based on research-focused search query 2"
            ],
            "research-focused search query 3": [
                "related question 1 based on research-focused search query 3",
                "related question 2 based on research-focused search query 3",
                "related question 3 based on research-focused search query 3"
            ]
            }
            """

        user_message = f"User query: {query}"

        try:
            message = await self.anthropic_client.messages.create(
                model="anthropic.claude-3-5-sonnet-20240620-v1:0",
                max_tokens=500,
                system=system_prompt,
                messages=[
                        {"role": "user", "content": user_message}
                ]
            )
            
            if message:
                json_match = re.search(r'\{[\s\S]*\}', message.content[0].text)
                if json_match:
                    try:
                        generated_json_queries = json.loads(json_match.group(0))
                        result = []
                        for i, (key, questions) in enumerate(generated_json_queries.items(), 1):
                            results = await self.fetch_news_results(" ".join(questions), num_results=3)
                            formatted_results = self.format_search_results(results)

                            result.append({
                                f"Section_{i}": {
                                    "research_area": key,
                                    "questions": questions,
                                    "sources": formatted_results
                                }
                            })
                        return result
                    except json.JSONDecodeError:
                        logging.error("Invalid JSON content found in the response.")
                        return None
                else:
                    logging.error("No JSON content found in the response.")
                    return None
            else:
                logging.error("No message received from Claude API.")
                return None
        except Exception as e:
            logging.error(f"Error in generate_research_areas_and_queries: {e}")
            return None

    async def call_claude_llm_stream(self, query, sections):
        try:
            context = self.extract_source(sections)

            system_prompt = """You are an advanced AI assistant specializing in providing informative responses based on news articles. Your task is to synthesize the given search results and generate a comprehensive response to the user's query, focusing on the most relevant and up-to-date information. Your responses should be characterized by their depth, accuracy, and attention to detail. Adapt your approach based on the nature of the query:

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

            async with self.anthropic_client.messages.stream(
                model="anthropic.claude-3-5-sonnet-20240620-v1:0",
                max_tokens=4096,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            logging.error(f"Error calling Claude API: {str(e)}")
            yield await self.fallback_response(query, sections)

    async def fallback_response(self, query, sections):
        response = f"I apologize, but I couldn't access the AI language model due to an error. However, I can provide you with a summary of the top news results for your query: '{query}'\n\n"
        
        for section in sections:
            for key, value in section.items():
                response += f"Research Area: {value['research_area']}\n"
                for idx, source in enumerate(value['sources'][:3], 1):
                    response += f"{idx}. {source['title']}\n   Source: {source['source']}\n   Date: {source['date']}\n   Snippet: {source['snippet'][:150]}...\n   URL: {source['link']}\n\n"
        
        response += "For more detailed information, please visit the source websites or try your query again later when the AI model is available."
        return response

    @staticmethod
    def extract_source(sections):
        sources = []
        for section in sections:
            for key, value in section.items():
                for source in value['sources']:
                    sources.append(f"Source: {source['source']}\nTitle: {source['title']}\nSnippet: {source['snippet']}\nURL: {source['link']}")
        return "\n\n".join(sources)