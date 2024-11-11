import os
import re
import json
import asyncio
import aiohttp
import ssl
import certifi
from dateutil import parser
from statistics import mean, median
from dotenv import load_dotenv
from serpapi import GoogleSearch
from anthropic import AnthropicBedrock

load_dotenv()

class FinanceSearchEngine:
    def __init__(self):
        self.api_key = os.getenv("serpapi_api_key")

    async def fetch_finance_results(self, ticker):
        params = {
            "engine": "google_finance",
            "q": ticker,
            "api_key": self.api_key
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        return self.format_finance_results(results, ticker)

    def format_finance_results(self, search_data, ticker):
        summary = search_data.get('summary', {})
        
        formatted_result = {
            'ticker': ticker,
            'title': summary.get('title', 'N/A'),
            'price': summary.get('price', 'N/A'),
            'currency': summary.get('currency', 'N/A'),
            'exchange': summary.get('exchange', 'N/A'),
            'market_cap': summary.get('market_cap', 'N/A'),
            'pe_ratio': summary.get('pe_ratio', 'N/A'),
            'dividend_yield': summary.get('dividend_yield', 'N/A'),
            'eps': summary.get('eps', 'N/A'),
            'beta': summary.get('beta', 'N/A')
        }
        
        if 'price_movement' in summary:
            formatted_result['price_change'] = summary['price_movement'].get('value', 'N/A')
            formatted_result['price_change_percentage'] = summary['price_movement'].get('percentage', 'N/A')
        
        if 'top_news' in search_data:
            formatted_result['top_news'] = search_data['top_news']
        
        return formatted_result

async def process_natural_language_input(query):
    system_message = """You are an AI assistant specializing in global finance. Your task is to extract or suggest the most relevant stock ticker from the user's natural language query. Respond with the ticker symbol only (e.g., AAPL, TATASTEEL, HDFCBANK, 005930). Ticker symbols can be of varying lengths and may include numbers. If you're not sure about the exact ticker, make your best guess based on the information provided."""
    
    try:
        client = AnthropicBedrock()
        
        response = await client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": f"Extract or suggest the most relevant stock ticker from this query: {query}"
                }
            ],
            system=system_message
        )
        
        if response.content:
            extracted_text = response.content[0].text.strip()
            match = re.search(r'\b[A-Z0-9.]+\b', extracted_text)
            if match:
                return match.group()
            else:
                print(f"Couldn't automatically extract a ticker. AI response: {extracted_text}")
        else:
            print("No content in the AI response.")
        
        return None
    except Exception as e:
        print(f"Error in processing natural language input: {str(e)}")
        return None

def format_context(search_results):
    context = "Available financial data:\n"
    for key, value in search_results.items():
        if value != 'N/A' and key != 'top_news':
            context += f"{key.replace('_', ' ').title()}: {value}\n"
    
    if 'top_news' in search_results:
        context += f"Top News: {search_results['top_news'].get('snippet', 'N/A')}\n"
    
    return context

async def call_mistral_finance_stream(query, search_results, finance_search_engine):
    try:
        url = os.getenv("AZURE_AI_ENDPOINT")
        api_key = os.getenv("AZURE_AI_API_KEY")
        ssl_verify = os.getenv("SSL_VERIFY", "True").lower() == "true"

        if not url or not api_key:
            raise ValueError("AZURE_AI_ENDPOINT or AZURE_AI_API_KEY environment variable is missing")

        context = "Available financial data:\n"
        for key, value in search_results.items():
            if value != 'N/A' and key != 'top_news':
                context += f"{key.replace('_', ' ').title()}: {value}\n"
        
        if 'top_news' in search_results:
            context += f"Top News: {search_results['top_news'].get('snippet', 'N/A')}\n"

        system_message = """You are an advanced AI assistant specializing in comprehensive financial analysis and stock market insights. Your expertise covers a wide range of financial topics, including fundamental analysis, technical analysis, market trends, economic factors, and trading strategies. Your task is to provide detailed, insightful responses to queries based on available financial data, while also leveraging your broad knowledge to make informed approximations when specific data is limited.

        Key Capabilities:

        1. Fundamental Analysis:
           - Analyze available financial metrics (e.g., P/E ratio, EPS, market cap) to assess a company's financial health.
           - Compare these metrics to industry averages and historical data, even if using approximations.
           - Discuss the company's business model, competitive position, and growth prospects based on available information and industry knowledge.

        2. Technical Analysis:
           - Perform technical analysis based on recent price movements, trading volumes, and any available chart patterns.
           - Discuss key technical indicators (e.g., moving averages, RSI, MACD) even if using approximated data.
           - Identify potential support and resistance levels based on recent price action.

        3. Market Trend Analysis:
           - Analyze broader market trends and their potential impact on the stock.
           - Discuss sector-specific trends and how they might affect the company.
           - Consider macroeconomic factors and their influence on the stock and its sector.

        4. Buy/Sell Recommendations:
           - Provide clear buy, sell, or hold recommendations based on your analysis.
           - Explain the rationale behind your recommendation, considering both fundamental and technical factors.
           - Discuss potential risks and rewards associated with the recommendation.
           - Always include a disclaimer that this is for educational purposes and not professional financial advice.

        5. Risk Assessment:
           - Evaluate potential risks, including company-specific, industry-wide, and macroeconomic factors.
           - Discuss the stock's volatility and how it compares to the broader market or sector.

        6. Future Outlook:
           - Provide insights into the company's potential future performance.
           - Discuss upcoming events or catalysts that could impact the stock price.

        7. Comparative Analysis:
           - Compare the stock to its peers or competitors, even if using approximations or general industry knowledge.

        8. Trading Strategies:
           - Suggest potential trading strategies suitable for the stock based on its characteristics and market conditions.
           - Discuss appropriate entry and exit points, stop-loss levels, and position sizing considerations.

        General Guidelines:
        - Use clear, concise language while maintaining depth and sophistication in your explanations.
        - When specific data is missing, make reasonable approximations based on industry averages, recent news, or general market knowledge. Clearly state when you are making approximations.
        - Organize your response logically, using headings, bullet points, or numbering as appropriate.
        - Anticipate and address potential follow-up questions in your response.
        - Always emphasize the importance of due diligence and personal research in investment decisions.

        Your goal is to provide a comprehensive, insightful, and actionable analysis that enhances the user's understanding of the stock, its potential, and the broader market context. Blend available data with your extensive knowledge to offer valuable insights, even when specific information is limited."""
        
        user_message = f"""Query: {query}

        Available Financial Data:
        {context}

        Based on this financial data (or lack thereof) and your extensive knowledge of finance and stock markets, provide a comprehensive and sophisticated analysis addressing the query. Your response should include:

        1. A summary of the available data and any key approximations or assumptions you're making.
        2. Fundamental analysis of the company and its financial health.
        3. Technical analysis based on recent price movements and any available chart patterns.
        4. Market and sector trend analysis.
        5. A clear buy, sell, or hold recommendation with detailed rationale.
        6. Risk assessment and future outlook.
        7. Comparative analysis with peers or sector (using approximations if necessary).
        8. Suggested trading strategies or investment approaches.

        If specific data is missing, use your knowledge to make reasonable approximations and provide a general market analysis. Ensure your response is thorough, insightful, and actionable, while clearly stating any limitations or assumptions in your analysis."""

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
        yield await finance_fallback_response(query, search_results, finance_search_engine)

async def call_claude_finance_stream(query, search_results, finance_search_engine):
    try:
        client = AnthropicBedrock()
        
        context = format_context(search_results)

        system_message = """You are an advanced AI assistant specializing in comprehensive financial analysis and stock market insights. Your expertise covers a wide range of financial topics, including fundamental analysis, technical analysis, market trends, economic factors, and trading strategies. Your task is to provide detailed, insightful responses to queries based on available financial data, while also leveraging your broad knowledge to make informed approximations when specific data is limited.

        Key Capabilities:

        1. Fundamental Analysis
        2. Technical Analysis
        3. Market Trend Analysis
        4. Buy/Sell Recommendations
        5. Risk Assessment
        6. Future Outlook
        7. Comparative Analysis
        8. Trading Strategies

        Provide a comprehensive, insightful, and actionable analysis that enhances the user's understanding of the stock, its potential, and the broader market context. Blend available data with your extensive knowledge to offer valuable insights, even when specific information is limited."""

        user_message = f"""Query: {query}

        Available Financial Data:
        {context}

        Based on this financial data (or lack thereof) and your extensive knowledge of finance and stock markets, provide a comprehensive and sophisticated analysis addressing the query. Your response should include:

        1. A summary of the available data and any key approximations or assumptions you're making.
        2. Fundamental analysis of the company and its financial health.
        3. Technical analysis based on recent price movements and any available chart patterns.
        4. Market and sector trend analysis.
        5. A clear buy, sell, or hold recommendation with detailed rationale.
        6. Risk assessment and future outlook.
        7. Comparative analysis with peers or sector (using approximations if necessary).
        8. Suggested trading strategies or investment approaches.

        If specific data is missing, use your knowledge to make reasonable approximations and provide a general market analysis. Ensure your response is thorough, insightful, and actionable, while clearly stating any limitations or assumptions in your analysis."""

        with client.messages.stream(
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
            for text in stream.text_stream:
                yield text

    except Exception as e:
        yield f"\nError calling Claude API: {str(e)}\n"
        yield await finance_fallback_response(query, search_results, finance_search_engine)

async def finance_fallback_response(query, search_results, finance_search_engine):
    response = f"I apologize, but I couldn't access the AI language models due to an error. However, I can provide you with a summary of the finance search results for your query: '{query}'\n\n"
    
    context = format_context(search_results)
    response += context
    
    response += "\nFor more detailed financial analysis, please try your query again later when the AI models are available."
    return response