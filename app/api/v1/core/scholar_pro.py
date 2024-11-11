import logging
from serpapi import GoogleSearch
from datetime import datetime
from anthropic import AsyncAnthropicBedrock
from dotenv import load_dotenv
import os
import json
import re
import asyncio
import aiohttp

# Load environment variables and set up logging
load_dotenv()
logging.basicConfig(level=logging.INFO)

# Load environment variables
SERPAPI_API_KEY = os.getenv('serpapi_api_key')
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

anthropic_client = AsyncAnthropicBedrock(aws_access_key=AWS_ACCESS_KEY_ID,
                                            aws_secret_key=AWS_SECRET_ACCESS_KEY,
                                            aws_region=AWS_REGION)

class ScholarProEngine:
    def __init__(self):
        self.serpapi_key = os.getenv('serpapi_api_key')
        self.anthropic_client = anthropic_client
        self.session = None

    async def initialize(self):
        self.session = aiohttp.ClientSession()

    async def close(self):
        if self.session:
            await self.session.close()

    async def fetch_google_scholar_results(self, query, num_results):
        if not self.session:
            await self.initialize()

        params = {
            "api_key": self.serpapi_key,
            "engine": "google_scholar",
            "q": query,
            "num": num_results,
            "sort": "date",
            "as_ylo": datetime.now().year - 5
        }
        
        try:
            async with self.session.get('https://serpapi.com/search', params=params) as response:
                if response.status == 200:
                    results = await response.json()
                    return results.get("organic_results", [])
                else:
                    logging.error(f"SerpAPI request failed with status {response.status}")
                    return []
        except aiohttp.ClientError as e:
            logging.error(f"Error fetching Google Scholar results: {e}")
            return []

    def format_scholar_results(self, search_data):
        return [{
            'source': result.get('publication_info', {}).get('summary', ''),
            'date': result.get('publication_info', {}).get('summary', ''),
            'title': result.get('title', ''),
            'snippet': result.get('snippet', ''),
            'link': result.get('link', ''),
            'citations': result.get('inline_links', {}).get('cited_by', {}).get('total', 0)
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
                            tasks = [self.fetch_google_scholar_results(question, num_results=1) for question in questions]
                            results = await asyncio.gather(*tasks)
                            
                            formatted_results = [
                                result 
                                for search_results in results 
                                for result in self.format_scholar_results(search_results)
                            ]

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

    async def vexoo_claude_scholar(self, query):
        try:
            web_results = []
            progress_updates = []
            research_areas = await self.generate_research_areas_and_queries(query)
            
            if research_areas is None:
                yield "Sorry, I couldn't generate a response due to an error in processing the query."
                return

            for section in research_areas:
                web_results.append(self.extract_source(section))
                progress_updates.append(f"Section {list(section.keys())[0]} completed.")

            web_results_str = "\n".join(web_results)
            
            system_prompt = """You are an advanced AI assistant with real-time internet search capabilities, designed to provide exceptionally detailed, comprehensive, and insightful responses to any type of query. Your knowledge spans a wide range of topics including but not limited to general knowledge, current events, science, technology, programming, arts, and more. Your responses should be characterized by their depth, precision, and attention to nuanced details. Adapt your approach based on the nature of the query:

        1. For general queries:
           - Provide an extensive and nuanced overview of the topic, exploring its various facets and subtleties.
           - Offer in-depth analysis, including historical context, current relevance, and future implications.
           - Include multiple relevant examples, case studies, or analogies to illustrate complex points.
           - Discuss any debates or controversies surrounding the topic, presenting various perspectives.

        2. For technical or coding questions:
           - Explain concepts thoroughly, breaking down complex ideas into digestible parts.
           - Provide detailed code examples, including comments explaining each significant part of the code.
           - Discuss the underlying principles, not just the surface-level implementation.
           - Cover edge cases, potential optimizations, and alternative approaches.
           - Include information about best practices, common pitfalls, and performance considerations.

        3. For current events or evolving topics:
           - Provide a detailed timeline of events, including key developments and turning points.
           - Analyze the implications from multiple angles: social, economic, political, etc.
           - Discuss expert opinions and predictions about future developments.
           - Address any misinformation or common misconceptions related to the topic.

        4. For how-to or problem-solving queries:
           - Provide extremely detailed, step-by-step instructions or explanations.
           - Anticipate and address potential issues or questions at each step.
           - Explain the reasoning behind each step or recommendation.
           - Offer multiple methods or solutions when applicable, comparing their pros and cons.
           - Include troubleshooting tips for common problems that might arise.

        General guidelines for all responses:
        - Dive deep into the subject matter, exploring subtopics and related concepts thoroughly.
        - Use precise language and technical terms where appropriate, but always provide clear explanations.
        - Incorporate relevant statistics, data, or quantitative information to support your points.
        - Address the nuances and complexities of the topic, avoiding oversimplification.
        - Organize information logically, using paragraphs, bullet points, or numbering as appropriate.
        - Use analogies or metaphors to explain complex concepts when helpful.
        - Anticipate follow-up questions and proactively address them in your response.
        - Acknowledge limitations in current knowledge or areas of ongoing research/debate.
        - Use an engaging and professional tone, making even the most complex information accessible.
        - Use inline citations [1], [2], etc., extensively, and include a detailed "Sources" section at the end with references and URLs.

        Your goal is to provide a response that not only answers the query comprehensively but also significantly enhances the user's understanding of the topic, offering insights and details they might not have even known to ask about. Prioritize depth and completeness over brevity, ensuring that your response is a thorough exploration of the subject matter.
        
        IMPORTANT: Always end your response with a "Sources" section, even if you need to truncate some of the main content to fit within the token limit."""

            user_message = f"""Query: {query}

        Internet Search Results:
        {web_results_str}

        Based on these search results and your extensive knowledge, provide an exceptionally detailed, comprehensive, and insightful response to the query. Your answer should be characterized by its depth, precision, and attention to nuanced details. Ensure your response is directly relevant, incorporates the latest information, and is tailored to the specific nature and depth of the question. If the query is coding-related, include detailed code examples with explanations. Anticipate and address potential follow-up questions in your response. Remember to include a detailed "Sources" section at the end of your response, with numbered references and URLs for further reading."""

            full_response = ""
            async with self.anthropic_client.messages.stream(
                model="anthropic.claude-3-5-sonnet-20240620-v1:0",
                max_tokens=4096,  # Increased token limit
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            ) as stream:
                async for text in stream.text_stream:
                    yield text
                    full_response += text

            # Check if the response ends with a Sources section
            if not full_response.strip().lower().endswith('sources:'):
                yield "\n\nNote: The response may have been truncated. For a complete answer, consider breaking down your query into more specific questions."

            # Log progress updates instead of yielding them
            logging.info("Research progress: " + ", ".join(progress_updates))

        except Exception as e:
            logging.error(f"Error in vexoo_claude_scholar: {e}")
            yield "Sorry, I encountered an error while generating the response."

    @staticmethod
    def extract_source(data):
        sources = []
        for section in data.values():
            for source in section.get('sources', []):
                if 'source' in source and 'title' in source:
                    sources.append(f"Source: {source['source']}\nTitle: {source['title']}\nSnippet: {source.get('snippet', 'N/A')}\nURL: {source.get('link', 'N/A')}")
        return "\n\n".join(sources)