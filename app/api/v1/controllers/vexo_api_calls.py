from fastapi import HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from app.api.v1.models import SerpRequest, SerpAPIResponseBody
from app.api.v1.web_crawler.search import call_search_engines, fetch_google_scholar_results, process_and_rank_results, process_and_rank_scholar_results, format_scholar_results
from app.api.v1.core.llm import call_mistral_llm_stream, call_claude_llm_stream
from app.api.v1.core.news import NewsSearchEngine, call_mistral_news_stream, call_claude_news_stream
from app.api.v1.core.finance import FinanceSearchEngine, process_natural_language_input, call_mistral_finance_stream, call_claude_finance_stream
from app.api.v1.core.scholar import ScholarSearchEngine, call_mistral_scholar_stream, call_claude_scholar_stream
from app.api.v1.core.scholar_pro import ScholarProEngine
from app.api.v1.core.news_pro import NewsProSearch
from app.api.v1.models import RelatedQuestionsResponse, RelatedQuestion
from app.api.v1.core.relatedques import generate_related_questions
from app.api.v1.core.websearch_pro import generate_queries_and_sections, vexoo_claude_pro_search


# Web Search
async def get_mistral_response(request: SerpRequest) -> StreamingResponse:
    args = request.input
    if not args or not args.query:
        raise HTTPException(status_code=400, detail="Invalid input: query is required")
    try:
        search_results = await call_search_engines(args.query)
        async def generate():
            async for chunk in call_mistral_llm_stream(args.query, search_results):
                yield chunk
        return StreamingResponse(generate(), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

async def get_claude_response(request: SerpRequest) -> StreamingResponse:
    args = request.input
    if not args or not args.query:
        raise HTTPException(status_code=400, detail="Invalid input: query is required")
    try:
        search_results = await call_search_engines(args.query)
        async def generate():
            async for chunk in call_claude_llm_stream(args.query, search_results):
                yield chunk
        return StreamingResponse(generate(), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

async def get_search_evidence(request: SerpRequest) -> JSONResponse:
    args = request.input
    if args:
        question = args.query
        results = await call_search_engines(question)
        response_body = SerpAPIResponseBody(response=results)
        return JSONResponse(content={"success": True, "response": response_body.dict()})
    return JSONResponse(content={"success": False, "response": {}})


# Scholar Search
async def get_scholar_mistral_response(request: SerpRequest) -> StreamingResponse:
    args = request.input
    if not args or not args.query:
        raise HTTPException(status_code=400, detail="Invalid input: query is required")
    try:
        scholar_search_engine = ScholarSearchEngine()
        results = await scholar_search_engine.fetch_google_scholar_results(args.query)
        async def generate():
            async for chunk in call_mistral_scholar_stream(args.query, results, scholar_search_engine):
                yield chunk
        return StreamingResponse(generate(), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

async def get_scholar_claude_response(request: SerpRequest) -> StreamingResponse:
    args = request.input
    if not args or not args.query:
        raise HTTPException(status_code=400, detail="Invalid input: query is required")
    try:
        scholar_search_engine = ScholarSearchEngine()
        results = await scholar_search_engine.fetch_google_scholar_results(args.query)
        async def generate():
            async for chunk in call_claude_scholar_stream(args.query, results, scholar_search_engine):
                yield chunk
        return StreamingResponse(generate(), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

async def get_scholar_evidence(request: SerpRequest) -> JSONResponse:
    args = request.input
    if args and args.query:
        query = args.query
        scholar_search_engine = ScholarSearchEngine()
        try:
            results = await scholar_search_engine.fetch_google_scholar_results(query)
            # Await the format_scholar_results method
            formatted_results = await scholar_search_engine.format_scholar_results(results)
            response_body = SerpAPIResponseBody(response={"organic_results": formatted_results})
            return JSONResponse(content={"success": True, "response": response_body.dict()})
        except Exception as e:
            return JSONResponse(content={"success": False, "error": str(e)})
    return JSONResponse(content={"success": False, "error": "Invalid input: query is required"})

# News Search
async def get_news_mistral_response(request: SerpRequest) -> StreamingResponse:
    args = request.input
    if not args or not args.query:
        raise HTTPException(status_code=400, detail="Invalid input: query is required")
    try:
        news_search_engine = NewsSearchEngine()
        news_results = await news_search_engine.fetch_news_results(args.query)
        ranked_results = await news_search_engine.rank_news_results(news_results, args.query)
        async def generate():
            async for chunk in call_mistral_news_stream(args.query, ranked_results, news_search_engine):
                yield chunk
        return StreamingResponse(generate(), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

async def get_news_claude_response(request: SerpRequest) -> StreamingResponse:
    args = request.input
    if not args or not args.query:
        raise HTTPException(status_code=400, detail="Invalid input: query is required")
    try:
        news_search_engine = NewsSearchEngine()
        news_results = await news_search_engine.fetch_news_results(args.query)
        ranked_results = await news_search_engine.rank_news_results(news_results, args.query)
        async def generate():
            async for chunk in call_claude_news_stream(args.query, ranked_results, news_search_engine):
                yield chunk
        return StreamingResponse(generate(), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

async def get_news_evidence(request: SerpRequest) -> JSONResponse:
    args = request.input
    if args and args.query:
        query = args.query
        news_search_engine = NewsSearchEngine()
        news_results = await news_search_engine.fetch_news_results(query)
        ranked_results = await news_search_engine.rank_news_results(news_results, query)
        
        response_data = {"organic_results": ranked_results}
        
        response_body = SerpAPIResponseBody(response=response_data)
        return JSONResponse(content={"success": True, "response": response_body.dict()})
    return JSONResponse(content={"success": False, "response": {}})

# Finance Search
async def get_finance_mistral_response(request: SerpRequest) -> StreamingResponse:
    args = request.input
    if not args or not args.query:
        raise HTTPException(status_code=400, detail="Invalid input: query is required")
    try:
        finance_search_engine = FinanceSearchEngine()
        processed_query = await process_natural_language_input(args.query)
        if processed_query is None:
            raise HTTPException(status_code=400, detail="Unable to process the query. Please provide a more specific question about a company or stock.")
        
        finance_results = await finance_search_engine.fetch_finance_results(processed_query)
        if not finance_results:
            raise HTTPException(status_code=404, detail="No financial data could be retrieved at this moment. Please try again later.")
        
        async def generate():
            async for chunk in call_mistral_finance_stream(args.query, finance_results, finance_search_engine):
                yield chunk
        return StreamingResponse(generate(), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

async def get_finance_claude_response(request: SerpRequest) -> StreamingResponse:
    args = request.input
    if not args or not args.query:
        raise HTTPException(status_code=400, detail="Invalid input: query is required")
    try:
        finance_search_engine = FinanceSearchEngine()
        processed_query = await process_natural_language_input(args.query)
        if processed_query is None:
            raise HTTPException(status_code=400, detail="Unable to process the query. Please provide a more specific question about a company or stock.")
        
        finance_results = await finance_search_engine.fetch_finance_results(processed_query)
        if not finance_results:
            raise HTTPException(status_code=404, detail="No financial data could be retrieved at this moment. Please try again later.")
        
        async def generate():
            async for chunk in call_claude_finance_stream(args.query, finance_results, finance_search_engine):
                yield chunk
        return StreamingResponse(generate(), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

async def get_finance_evidence(request: SerpRequest) -> JSONResponse:
    args = request.input
    if args and args.query:
        try:
            finance_search_engine = FinanceSearchEngine()
            processed_query = await process_natural_language_input(args.query)
            if processed_query is None:
                return JSONResponse(content={"success": False, "error": "Unable to process the query. Please provide a more specific question about a company or stock."})
            
            finance_results = await finance_search_engine.fetch_finance_results(processed_query)
            if not finance_results:
                return JSONResponse(content={"success": False, "error": "No financial data could be retrieved at this moment. Please try again later."})
            
            response_body = SerpAPIResponseBody(response=finance_results)
            return JSONResponse(content={"success": True, "response": response_body.dict()})
        except Exception as e:
            return JSONResponse(content={"success": False, "error": str(e)})
    return JSONResponse(content={"success": False, "error": "Invalid input: query is required"})

# ScholarPro Search
async def get_scholar_pro_subqueries(request: SerpRequest) -> JSONResponse:
    args = request.input
    if not args or not args.query:
        raise HTTPException(status_code=400, detail="Invalid input: query is required")
    try:
        scholar_pro_engine = ScholarProEngine()
        result = await scholar_pro_engine.generate_research_areas_and_queries(args.query)
        if result is None:
            raise HTTPException(status_code=500, detail="Failed to generate research areas and queries")
        return JSONResponse(content={"success": True, "response": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

async def get_scholar_pro_claude(request: SerpRequest) -> StreamingResponse:
    args = request.input
    if not args or not args.query:
        raise HTTPException(status_code=400, detail="Invalid input: query is required")
    try:
        scholar_pro_engine = ScholarProEngine()
        async def generate():
            async for chunk in scholar_pro_engine.vexoo_claude_scholar(args.query):
                yield chunk
        return StreamingResponse(generate(), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
# NewsPro Search
async def get_news_pro_subqueries(request: SerpRequest) -> JSONResponse:
    args = request.input
    if not args or not args.query:
        raise HTTPException(status_code=400, detail="Invalid input: query is required")
    try:
        news_pro = NewsProSearch()
        sections = await news_pro.generate_research_areas_and_queries(args.query)
        if sections is None:
            raise HTTPException(status_code=500, detail="Failed to generate research areas and queries")
        return JSONResponse(content={"success": True, "sections": sections})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

async def get_news_pro_claude(request: SerpRequest) -> StreamingResponse:
    args = request.input
    if not args or not args.query:
        raise HTTPException(status_code=400, detail="Invalid input: query is required")
    try:
        news_pro = NewsProSearch()
        sections = await news_pro.generate_research_areas_and_queries(args.query)
        if sections is None:
            raise HTTPException(status_code=500, detail="Failed to generate research areas and queries")
        
        async def generate():
            async for chunk in news_pro.call_claude_llm_stream(args.query, sections):
                yield chunk.encode('utf-8')

        return StreamingResponse(generate(), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
# Related Questions
async def get_related_questions(request: SerpRequest) -> JSONResponse:
    args = request.input
    if not args or not args.query:
        raise HTTPException(status_code=400, detail="Invalid input: query is required")
    try:
        questions_dict = await generate_related_questions(args.query)
        if questions_dict is None or "questions" not in questions_dict:
            raise HTTPException(status_code=500, detail="Failed to generate related questions")
        
        questions = questions_dict["questions"]
        response = RelatedQuestionsResponse(success=True, related_questions=questions)
        return JSONResponse(content=response.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
# Web Search Pro
async def get_searchpro_subqueries_response(request: SerpRequest) -> JSONResponse:
    args = request.input
    if not args or not args.query:
        raise HTTPException(status_code=400, detail="Invalid input: query is required")
    
    try:
        data = await generate_queries_and_sections(args.query)
        return JSONResponse(content={"success": True, "response": data})
    except Exception as e:
        return JSONResponse(content={"success": False, "error": f"Error generating subqueries: {str(e)}"})

async def get_searchpro_claude_response(request: SerpRequest) -> StreamingResponse:
    args = request.input
    if not args or not args.query:
        raise HTTPException(status_code=400, detail="Invalid input: query is required")
    
    try:
        data = await generate_queries_and_sections(args.query)
        
        async def generate():
            async for chunk in vexoo_claude_pro_search(args.query, data):
                yield chunk
        
        return StreamingResponse(generate(), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")