import os
import json
from dotenv import load_dotenv
from mistralai.async_client import MistralAsyncClient
from mistralai.models.chat_completion import ChatMessage
from anthropic import AsyncAnthropicBedrock
import aiohttp
import ssl
import certifi

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


async def call_mistral_llm_stream(query, search_results):
    try:
        url = os.getenv("AZURE_AI_ENDPOINT")
        api_key = os.getenv("AZURE_AI_API_KEY")
        ssl_verify = os.getenv("SSL_VERIFY", "True").lower() == "true"

        if not url or not api_key:
            raise ValueError("AZURE_AI_ENDPOINT or AZURE_AI_API_KEY environment variable is missing")

        formatted_results = format_web_search_results(search_results)
        context = '\n\n'.join([f"Source: {r['source']}\nTitle: {r['title']}\nSnippet: {r['snippet']}\nURL: {r.get('link', 'N/A')}" for r in formatted_results])

        system_message = """You are an advanced AI assistant with real-time internet search capabilities, designed to provide exceptionally detailed, comprehensive, and insightful responses to any type of query. Your knowledge spans a wide range of topics including but not limited to general knowledge, current events, science, technology, programming, arts, and more. Your responses should be characterized by their depth, precision, and attention to nuanced details."""
        
        user_message = f"""Query: {query}

        Internet Search Results:
        {context}

        Based on these search results and your extensive knowledge, provide an exceptionally detailed, comprehensive, and insightful response to the query."""

        data = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": 1000,
            "temperature": 0.7,
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
        yield await fallback_response(query, search_results)

async def call_claude_llm_stream(query, search_results):
    try:
        formatted_results = format_web_search_results(search_results)
        context = '\n\n'.join([f"Source: {r['source']}\nTitle: {r['title']}\nSnippet: {r['snippet']}\nURL: {r['link']}" for r in formatted_results])

        system_message = """You are an advanced AI assistant with real-time internet search capabilities, designed to provide exceptionally detailed, comprehensive, and insightful responses to any type of query. Your knowledge spans a wide range of topics including but not limited to general knowledge, current events, science, technology, programming, arts, and more. Your responses should be characterized by their depth, precision, and attention to nuanced details."""

        user_message = f"""Query: {query}

        Internet Search Results:
        {context}

        Based on these search results and your extensive knowledge, provide an exceptionally detailed, comprehensive, and insightful response to the query. Your answer should be characterized by its depth, precision, and attention to nuanced details. Ensure your response is directly relevant, incorporates the latest information, and is tailored to the specific nature and depth of the question. If the query is coding-related, include detailed code examples with explanations. Anticipate and address potential follow-up questions in your response. Remember to include a detailed "Sources" section at the end of your response, with numbered references and URLs for further reading."""

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
        yield await fallback_response(query, search_results)

async def fallback_response(query, search_results):
    formatted_results = format_web_search_results(search_results)
    response = f"I apologize, but I couldn't access the AI language model due to an error. However, I can provide you with a summary of the top search results for your query: '{query}'\n\n"
    
    for idx, result in enumerate(formatted_results[:5], 1):
        response += f"{idx}. {result['title']}\n   Source: {result['source']}\n   Snippet: {result['snippet'][:150]}...\n\n"
    
    response += "For more detailed information, please visit the source websites or try your query again later when the AI model is available."
    return response

def format_web_search_results(search_data):
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
            'engine': result.get('engine', ''),
            'link': result.get('link', '')
        }
        formatted_results.append(formatted_result)
    return formatted_results

def format_scholar_results(search_data):
    formatted_results = []
    # Check if search_data is already a list
    results = search_data if isinstance(search_data, list) else search_data.get('organic_results', [])
    for result in results:
        source = result.get('source') or result.get('publication_info', {}).get('summary', '')
        formatted_result = {
            'source': source,
            'date': result.get('date') or result.get('publication_info', {}).get('summary', ''),
            'title': result.get('title', ''),
            'snippet': result.get('snippet', ''),
            'highlight': result.get('snippet_highlighted_words', ''),
            'link': result.get('link', ''),
            'citations': result.get('inline_links', {}).get('cited_by', {}).get('total', 0),
            'engine': result.get('engine', '')
        }
        formatted_results.append(formatted_result)
    return formatted_results