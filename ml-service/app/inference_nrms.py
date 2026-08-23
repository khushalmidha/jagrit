import os
import torch
import numpy as np
import pickle
import time

class NRMSRankerService:
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.model_version = "nrms-v1"
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # In a real production system, these vectors would be loaded from a vector database or Redis.
        # For this implementation, we simulate the "offline precomputed cache" by keeping them in memory.
        self.cached_news_vectors = {}
        self.cached_user_vectors = {}
        
        # Load vocab for fallback/demo purposes
        vocab_path = os.path.join(base_dir, 'dl', 'data', 'word2idx.pkl')
        if os.path.exists(vocab_path):
            with open(vocab_path, 'rb') as f:
                self.word2idx = pickle.load(f)
        else:
            self.word2idx = {}
            
    def _get_cached_news_vectors(self, candidates):
        """
        Simulates fetching precomputed news vectors (e.g. from Redis/Vector Store).
        Returns a matrix of size [num_candidates, embed_dim]
        """
        embed_dim = 200 # hidden_dim of NRMS
        vectors = []
        for c in candidates:
            if c in self.cached_news_vectors:
                vectors.append(self.cached_news_vectors[c])
            else:
                # Fallback: if not cached, we would normally run the news_encoder here.
                # For this demo of the serving pattern, we simulate a cached vector.
                vectors.append(np.random.normal(size=embed_dim).astype(np.float32))
        return np.vstack(vectors)
        
    def _get_cached_user_vector(self, user_id):
        """
        Simulates fetching a precomputed user vector.
        User vectors are periodically updated offline (e.g. on new interaction).
        """
        embed_dim = 200
        if user_id in self.cached_user_vectors:
            return self.cached_user_vectors[user_id]
        else:
            return np.random.normal(size=embed_dim).astype(np.float32)
            
    def recommend(self, user_id: str, top_k: int = 10):
        start_time = time.time()
        
        # For the demo, we generate some mock candidates since we don't want to couple heavily 
        # with the XGBoost CandidateGenerator which relies on full dataframes loaded in memory.
        candidates = [f"N{i}" for i in range(100)]
        is_cold_start = user_id not in self.cached_user_vectors
        
        # 1. Fetch precomputed user vector
        user_vector = self._get_cached_user_vector(user_id)
        
        # 2. Fetch precomputed news vectors
        news_vectors = self._get_cached_news_vectors(candidates)
        
        # 3. Fast dot product scoring (NO neural network forward pass!)
        scores = np.dot(news_vectors, user_vector)
        
        # 4. Rank
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        recommendations = []
        for idx in top_indices:
            news_id = candidates[idx]
            score = float(scores[idx])
            recommendations.append({
                "news_id": news_id,
                "score": score,
                "category": "sports", # mock for demo output
                "title": f"Cached Article {news_id}"
            })
            
        latency = (time.time() - start_time) * 1000
        return recommendations, latency, is_cold_start
