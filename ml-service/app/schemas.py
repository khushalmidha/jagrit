from pydantic import BaseModel
from typing import List

class RecommendRequest(BaseModel):
    user_id: str
    top_k: int = 10

class RecommendedArticle(BaseModel):
    news_id: str
    title: str
    abstract: str = ""
    category: str
    image_url: str = ""
    score: float
    rank: int

class RecommendResponse(BaseModel):
    recommendations: List[RecommendedArticle]
    model_version: str
    latency_ms: float
    is_cold_start: bool

class RAGRequest(BaseModel):
    query: str

class RAGResponse(BaseModel):
    answer: str
    context_articles: List[RecommendedArticle]
    latency_ms: float
