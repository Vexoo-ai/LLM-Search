from typing import Any, Optional, Dict, List
from pydantic import BaseModel
from app.utils.models import SerpAPIResponseBody, LLMGeneratedResponse

class Response(BaseModel):
    success: bool
    response: SerpAPIResponseBody

class LLMResponse(BaseModel):
    success: bool
    response: LLMGeneratedResponse

class LLMSummaryArgs(BaseModel):
    query: str

class SerpArgs(BaseModel):
    query: str

class LLMSummaryRequest(BaseModel):
    input: LLMSummaryArgs

class SerpRequest(BaseModel):
    input: SerpArgs
    
class SerpAPIResponseBody(BaseModel):
    response: Dict[str, List[Dict[str, Any]]]
    
class RelatedQuestion(BaseModel):
    question: str

class RelatedQuestionsResponse(BaseModel):
    success: bool
    related_questions: List[str]