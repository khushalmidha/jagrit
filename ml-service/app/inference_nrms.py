import os
import numpy as np
import pickle
import time

class NRMSRankerService:
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.model_version = "nrms-v1"
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Load precomputed vectors if available
        self.cached_news_vectors = {}
        self.cached_user_vectors = {}
        
        news_vec_path = os.path.join(base_dir, 'dl', 'data', 'nrms_news_vectors.pkl')
        user_vec_path = os.path.join(base_dir, 'dl', 'data', 'nrms_user_vectors.pkl')
        
        if os.path.exists(news_vec_path):
            with open(news_vec_path, 'rb') as f:
                self.cached_news_vectors = pickle.load(f)
                
        if os.path.exists(user_vec_path):
            with open(user_vec_path, 'rb') as f:
                self.cached_user_vectors = pickle.load(f)
        
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
                # If not cached, fallback to zero vector (unscored)
                vectors.append(np.zeros(embed_dim, dtype=np.float32))
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
            return np.zeros(embed_dim, dtype=np.float32)
            
    def recommend(self, user_id: str, top_k: int = 10):
        start_time = time.time()
        
        # Fetch real candidate pool from Redis or fallback to random IDs if empty
        # In a fully integrated prod system, this list comes from the same CandidateGenerator as XGBoost
        # For the prototype, we assume the frontend requests /recommend/v2 which we want to test for latency
        candidates = [f"N{i}" for i in range(100)] if not self.cached_news_vectors else list(self.cached_news_vectors.keys())[:100]
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
                "category": "news", # Usually fetched from metadata store
                "title": f"Article {news_id}"
            })
            
        latency = (time.time() - start_time) * 1000
        return recommendations, latency, is_cold_start
