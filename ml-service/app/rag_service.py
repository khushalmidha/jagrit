import os
import time
import logging
import redis
import google.generativeai as genai
from typing import List, Dict, Any

logger = logging.getLogger("rag_service")

class RAGService:
    def __init__(self, redis_host='localhost', redis_port=6379, redis_url=None):
        try:
            if redis_url:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
            else:
                self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)
        except Exception as e:
            logger.error(f"Error initializing Redis client for RAG: {e}")
            self.redis_client = None

        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            # Use gemini-1.5-flash as it's fast and suitable for this task
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.is_llm_ready = True
        else:
            logger.warning("GEMINI_API_KEY not found in environment. RAG generation will be disabled.")
            self.model = None
            self.is_llm_ready = False

    def is_redis_available(self):
        try:
            return self.redis_client.ping()
        except (redis.ConnectionError, AttributeError):
            return False

    def retrieve_context(self, top_k=5) -> List[Dict[str, Any]]:
        results = []
        if self.is_redis_available():
            try:
                live_ids = self.redis_client.zrevrange('news:recent', 0, top_k - 1)
                for rank, news_id in enumerate(live_ids, 1):
                    title = self.redis_client.hget(f"article:{news_id}", "title") or "Live News"
                    category = self.redis_client.hget(f"article:{news_id}", "category") or "news"
                    abstract = self.redis_client.hget(f"article:{news_id}", "abstract") or ""
                    image_url = self.redis_client.hget(f"article:{news_id}", "image_url") or ""
                    
                    results.append({
                        "news_id": news_id,
                        "title": title,
                        "category": category,
                        "abstract": abstract,
                        "image_url": image_url,
                        "score": 1.0 / rank,
                        "rank": rank
                    })
                return results
            except Exception as e:
                logger.error(f"Redis retrieval error in RAG: {e}")
                
        # Mock fallback if Redis not available
        return [
            {
                "news_id": "M1",
                "title": "Global Markets Rally as Tech Stocks Surge",
                "category": "business",
                "abstract": "Stock markets around the world have seen a significant increase today.",
                "image_url": "",
                "score": 1.0,
                "rank": 1
            },
            {
                "news_id": "M2",
                "title": "New Breakthrough in Renewable Energy Tech",
                "category": "technology",
                "abstract": "Scientists have announced a major breakthrough in solar panel efficiency.",
                "image_url": "",
                "score": 0.5,
                "rank": 2
            }
        ]

    def ask(self, query: str) -> tuple[str, List[Dict[str, Any]], float]:
        start_time = time.time()
        
        # 1. Retrieve Context
        articles = self.retrieve_context(top_k=5)
        
        # 2. Check LLM availability
        if not self.is_llm_ready:
            latency = (time.time() - start_time) * 1000
            return "I am currently unable to answer questions as my language model is not configured (Missing GEMINI_API_KEY).", articles, latency
            
        # 3. Format Context
        context_str = "Recent News Context:\n\n"
        for article in articles:
            context_str += f"Title: {article['title']}\n"
            context_str += f"Category: {article['category']}\n"
            context_str += f"Abstract: {article['abstract']}\n\n"
            
        prompt = f"""
You are an intelligent AI news assistant. Your job is to answer the user's question based ONLY on the provided recent news context. 
If the context does not contain relevant information to answer the question, say "I don't have enough information in the latest news to answer that." 
Keep your answer concise, engaging, and objective.

{context_str}

User Question: {query}
"""
        
        # 4. Generate Answer
        try:
            response = self.model.generate_content(prompt)
            answer = response.text
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            answer = "Sorry, I encountered an error while trying to process your request."
            
        latency = (time.time() - start_time) * 1000
        return answer, articles, latency
