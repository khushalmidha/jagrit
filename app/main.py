import os
from fastapi import FastAPI, HTTPException
import logging
import json
from app.schemas import RecommendRequest, RecommendResponse, RecommendedArticle
from app.inference import RankerService

# Setup structured JSON logging
logger = logging.getLogger("ranking_api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}')
handler.setFormatter(formatter)
logger.addHandler(handler)

app = FastAPI(title="Real-Time Content Ranking API")

# Global state for metrics
metrics = {
    "total_requests": 0,
    "total_latency_ms": 0,
    "cold_starts": 0
}

ranker_service = None

@app.on_event("startup")
def startup_event():
    global ranker_service
    logger.info("Initializing RankerService...")
    # Read redis host from env if running in docker
    redis_host = os.environ.get("REDIS_HOST", "localhost")
    ranker_service = RankerService(redis_host=redis_host)
    logger.info("RankerService initialized.")

@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    global metrics
    
    logger.info(f"Received recommendation request for user: {request.user_id}")
    
    try:
        recs_data, latency, is_cold_start = ranker_service.recommend(request.user_id, request.top_k)
        
        # Update metrics
        metrics["total_requests"] += 1
        metrics["total_latency_ms"] += latency
        if is_cold_start:
            metrics["cold_starts"] += 1
            
        articles = [RecommendedArticle(**r) for r in recs_data]
        
        logger.info(f"Served {len(articles)} recommendations in {latency:.2f}ms. Cold start: {is_cold_start}")
        
        return RecommendResponse(
            recommendations=articles,
            model_version=ranker_service.model_version,
            latency_ms=latency,
            is_cold_start=is_cold_start
        )
    except Exception as e:
        logger.error(f"Error serving recommendation: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_version": ranker_service.model_version if ranker_service else "not_loaded",
        "redis_connected": ranker_service.is_redis_available() if ranker_service else False
    }

@app.get("/metrics")
def get_metrics():
    reqs = metrics["total_requests"]
    avg_lat = metrics["total_latency_ms"] / reqs if reqs > 0 else 0
    cold_pct = (metrics["cold_starts"] / reqs * 100) if reqs > 0 else 0
    
    return {
        "total_requests": reqs,
        "avg_latency_ms": round(avg_lat, 2),
        "cold_start_percentage": round(cold_pct, 2)
    }
