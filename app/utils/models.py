import datetime
from enum import Enum
from typing import Any, Optional, Dict, List

from fastapi import HTTPException, Request
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

class SerpAPIResponseBody(BaseModel):
    response: Dict[str, Any]

class LLMGeneratedResponse(BaseModel):
    answer: str
    evidences: str
    links: List[str]
    links_and_evidences: Dict[str, str]

class RootResponse(BaseModel):
    message: str
    timestamp: datetime.datetime
    running_time: str

class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    def _init_(self, app, max_request_size: int):
        super()._init_(app)
        self.max_request_size = max_request_size

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            raise HTTPException(status_code=413, detail="Request body too large")
        response = await call_next(request)
        return response