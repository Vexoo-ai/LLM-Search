import os
import json
import ssl
import certifi
import aiohttp
from dotenv import load_dotenv

load_dotenv()

async def generate_related_questions(query):
    try:
        url = os.getenv("AZURE_AI_ENDPOINT")
        api_key = os.getenv("AZURE_AI_API_KEY")
        ssl_verify = os.getenv("SSL_VERIFY", "True").lower() == "true"

        if not url or not api_key:
            raise ValueError("AZURE_AI_ENDPOINT or AZURE_AI_API_KEY environment variable is missing")

        system_message = """You are a search engine specialist with expertise in crafting precise, search engine-friendly queries.
        Based on the user query, generate a set of 5 search-focused queries. Each query should be optimized for search engines, covering key aspects of the topic.
        Ensure that the queries are specific, actionable, and designed to yield high-quality search results.

        Output format:
        {
        "questions": 
        [
        "related question 1 based on search-focused query",
        "related question 2 based on search-focused query",
        "related question 3 based on search-focused query",
        "related question 4 based on search-focused query",
        "related question 5 based on search-focused query"
        ]
        }"""
        
        user_message = f"Generate 5 related questions for the following query: {query}"

        data = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": 500,
            "temperature": 0.7,
            "stream": False,
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

                response_json = await resp.json()
                content = response_json['choices'][0]['message']['content']
                questions_dict = json.loads(content)
                return questions_dict

    except Exception as e:
        print(f"Error generating related questions: {str(e)}")
        return None