from fastapi import APIRouter, Body
from fastapi.responses import StreamingResponse, JSONResponse
from app.api.v1.controllers.vexo_api_calls import (
    get_mistral_response, get_claude_response, get_search_evidence,
    get_scholar_mistral_response, get_scholar_claude_response, get_scholar_evidence,
    get_news_mistral_response, get_news_claude_response, get_news_evidence,
    get_finance_mistral_response, get_finance_claude_response, get_finance_evidence,
    get_scholar_pro_subqueries, get_scholar_pro_claude,
    get_news_pro_subqueries, get_news_pro_claude,
    get_related_questions,
    get_searchpro_subqueries_response,
    get_searchpro_claude_response

)
from app.api.v1.models import SerpRequest, RelatedQuestionsResponse


vexo_api_router = APIRouter()

## --------------------Standard Endpoints---------------------##

# Existing endpoints
@vexo_api_router.post("/SearchMistral")
async def vexoo_search_mistral(request: SerpRequest = Body(...)) -> StreamingResponse:
    return await get_mistral_response(request)

@vexo_api_router.post("/SearchClaude")
async def vexoo_search_claude(request: SerpRequest = Body(...)) -> StreamingResponse:
    return await get_claude_response(request)

@vexo_api_router.post("/SearchEvidence")
async def vexoo_search_evidence(request: SerpRequest = Body(...)) -> JSONResponse:
    return await get_search_evidence(request)

#Scholar endpoints
@vexo_api_router.post("/ScholarMistral")
async def scholar_search_mistral(request: SerpRequest = Body(...)) -> StreamingResponse:
    return await get_scholar_mistral_response(request)

@vexo_api_router.post("/ScholarClaude")
async def scholar_search_claude(request: SerpRequest = Body(...)) -> StreamingResponse:
    return await get_scholar_claude_response(request)

@vexo_api_router.post("/ScholarEvidence")
async def scholar_search_evidence(request: SerpRequest = Body(...)) -> JSONResponse:
    return await get_scholar_evidence(request)

# News endpoints
@vexo_api_router.post("/NewsMistral")
async def news_search_mistral(request: SerpRequest = Body(...)) -> StreamingResponse:
    return await get_news_mistral_response(request)

@vexo_api_router.post("/NewsClaude")
async def news_search_claude(request: SerpRequest = Body(...)) -> StreamingResponse:
    return await get_news_claude_response(request)

@vexo_api_router.post("/NewsEvidence")
async def news_search_evidence(request: SerpRequest = Body(...)) -> JSONResponse:
    return await get_news_evidence(request)

# Finance endpoints
@vexo_api_router.post("/FinanceMistral")
async def finance_search_mistral(request: SerpRequest = Body(...)) -> StreamingResponse:
    return await get_finance_mistral_response(request)

@vexo_api_router.post("/FinanceClaude")
async def finance_search_claude(request: SerpRequest = Body(...)) -> StreamingResponse:
    return await get_finance_claude_response(request)

@vexo_api_router.post("/FinanceEvidence")
async def finance_search_evidence(request: SerpRequest = Body(...)) -> JSONResponse:
    return await get_finance_evidence(request)


##--------------------PRO Endpoints---------------------##

# Web Search Pro 
@vexo_api_router.post("/SearchProSubQueries")
async def vexoo_searchprosubqueries(request: SerpRequest = Body(...)) -> JSONResponse:
    return await get_searchpro_subqueries_response(request)

@vexo_api_router.post("/SearchProClaude")
async def vexoo_searchproclaude(request: SerpRequest = Body(...)) -> StreamingResponse:
    return await get_searchpro_claude_response(request)

# ScholarPro endpoints
@vexo_api_router.post("/ScholarProSubQueries")
async def scholar_pro_subqueries(request: SerpRequest = Body(...)) -> JSONResponse:
    return await get_scholar_pro_subqueries(request)

@vexo_api_router.post("/ScholarProClaude")
async def scholar_pro_claude(request: SerpRequest = Body(...)) -> StreamingResponse:
    return await get_scholar_pro_claude(request)

# NewsPro endpoints
@vexo_api_router.post("/NewsProSubQueries")
async def news_pro_subqueries(request: SerpRequest = Body(...)) -> JSONResponse:
    return await get_news_pro_subqueries(request)

@vexo_api_router.post("/NewsProClaude")
async def news_pro_claude(request: SerpRequest = Body(...)) -> StreamingResponse:
    return await get_news_pro_claude(request)

#Related Questions
@vexo_api_router.post("/RelatedQuestions", response_model=RelatedQuestionsResponse)
async def vexo_related_questions(request: SerpRequest = Body(...)) -> JSONResponse:
    return await get_related_questions(request)

