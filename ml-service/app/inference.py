import os
import pickle
import pandas as pd
import redis
import time
from src.candidate_generation import CandidateGenerator
from src.features import FeatureEngineer

class RankerService:
    def __init__(self, redis_host='localhost', redis_port=6379, redis_url=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Load Model
        model_path = os.path.join(base_dir, "models", "ranker.pkl")
        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
        except FileNotFoundError:
            print("WARNING: Model ranker.pkl not found. Running in fallback mode.")
            self.model = None
            
        self.model_version = "v1-xgboost" if self.model else "v0-fallback"
            
        # Load Static Data for Candidate Gen & Features
        news_path = os.path.join(base_dir, "data", "processed", "news.parquet")
        try:
            self.df_news = pd.read_parquet(news_path)
        except (FileNotFoundError, OSError):
            print("WARNING: news.parquet not found. Using mock news data.")
            self.df_news = pd.DataFrame([
                {"news_id": "M1", "category": "business", "title": "Global Markets Rally as Tech Stocks Surge", "abstract": ""},
                {"news_id": "M2", "category": "technology", "title": "New Breakthrough in Renewable Energy Tech", "abstract": ""},
                {"news_id": "M3", "category": "sports", "title": "World Cup Finals: An Unforgettable Match", "abstract": ""},
                {"news_id": "M4", "category": "politics", "title": "Elections 2026: What You Need to Know", "abstract": ""},
                {"news_id": "M5", "category": "technology", "title": "The Future of AI in Healthcare", "abstract": ""}
            ])
            
        self.news_dict = self.df_news.set_index('news_id').to_dict('index')
        
        # Initialize Feature Engineer and Candidate Generator
        self.fe = FeatureEngineer(self.df_news)
        
        # Static interactions for history fallback if Redis missing
        interactions_path = os.path.join(base_dir, "data", "processed", "interactions.parquet")
        try:
            self.df_interactions = pd.read_parquet(interactions_path)
        except (FileNotFoundError, OSError):
            print("WARNING: interactions.parquet not found. Using empty interactions.")
            self.df_interactions = pd.DataFrame(columns=["user_id", "history", "timestamp", "clicked", "news_id"])
            
        self.cg = CandidateGenerator(self.df_news, self.df_interactions)
        
        # Redis Client
        try:
            if redis_url:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
            else:
                self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)
        except Exception as e:
            print(f"Error initializing Redis client: {e}")
            self.redis_client = None
        
    def is_redis_available(self):
        try:
            return self.redis_client.ping()
        except (redis.ConnectionError, AttributeError):
            return False
            
    def _get_user_history(self, user_id):
        # Try Redis first
        if self.is_redis_available():
            history = self.redis_client.lrange(f"user:{user_id}:last_10_clicked", 0, -1)
            if history:
                return history, False # Not cold start
                
        # Fallback to static data
        user_hist = self.df_interactions[self.df_interactions['user_id'] == user_id]
        if not user_hist.empty:
            hist_str = user_hist.iloc[-1]['history']
            if pd.notna(hist_str):
                return str(hist_str).split(), False
                
        return [], True # Cold start
        
    def _cold_start_recommendation(self, top_k):
        """
        Return trending + fresh live articles from Redis if available, otherwise fallback.
        """
        results = []
        
        # Try getting live news first for cold start to ensure we have enough fresh articles
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
                if len(results) >= top_k:
                    return results
            except Exception as e:
                print(f"Redis cold start error: {e}")

        # Fallback to static popular news
        popular = self.cg.popular_news[:top_k]
        for rank, news_id in enumerate(popular, len(results) + 1):
            if len(results) >= top_k:
                break
            if news_id in self.news_dict:
                article = self.news_dict[news_id]
                results.append({
                    "news_id": news_id,
                    "title": article['title'],
                    "category": article['category'],
                    "abstract": article.get('abstract', ''),
                    "score": 1.0 / rank, # Mock score
                    "rank": rank
                })
                
        # Fill remaining slots with dynamic placeholders if we still don't have top_k
        while len(results) < top_k:
            rank = len(results) + 1
            results.append({
                "news_id": f"DYN{rank}",
                "title": f"Breaking Global News {rank}",
                "category": "world",
                "abstract": "Stay tuned for more updates on this developing story.",
                "score": 1.0 / rank,
                "rank": rank
            })
            
        return results

    def recommend(self, user_id: str, top_k: int = 10):
        start_time = time.time()
        
        # 1. Get History
        user_history, is_cold_start = self._get_user_history(user_id)
        
        if is_cold_start:
            recs = self._cold_start_recommendation(top_k)
            latency = (time.time() - start_time) * 1000
            return recs, latency, True
            
        # 2. Candidate Generation
        use_live = self.is_redis_available()
        redis_conn = self.redis_client if use_live else None
        
        candidates = []
        if use_live:
            # Fetch 50 most recent live articles from Redis
            live_candidates = redis_conn.zrevrange('news:recent', 0, 49)
            if live_candidates:
                candidates = live_candidates
                
        if not candidates:
            # Fallback to static candidates
            candidates = self.cg.generate_candidates(user_history, num_candidates=50)
        
        # 3. Features
        df_feats = self.fe.build_features(user_id, user_history, candidates, live_redis_client=redis_conn)
        
        if df_feats.empty:
            # Fallback
            recs = self._cold_start_recommendation(top_k)
            latency = (time.time() - start_time) * 1000
            return recs, latency, True
            
        feature_cols = [
            'category_match_score', 'article_popularity', 'article_recency_hours',
            'title_length', 'abstract_length', 'user_total_clicks_in_history', 'user_category_diversity'
        ]
        
        # 4. Rank
        if self.model is None:
            # Fallback if no model exists (e.g. data missing)
            recs = self._cold_start_recommendation(top_k)
            latency = (time.time() - start_time) * 1000
            return recs, latency, True
            
        preds = self.model.predict_proba(df_feats[feature_cols])[:, 1]
        df_feats['score'] = preds
        
        top_n = df_feats.sort_values(by='score', ascending=False).head(top_k)
        
        # 5. Format Response
        recs = []
        for rank, row in enumerate(top_n.itertuples(), 1):
            if row.news_id in self.news_dict:
                article = self.news_dict[row.news_id]
                title = article['title']
                category = article['category']
                abstract = article.get('abstract', '')
            elif use_live:
                title = redis_conn.hget(f"article:{row.news_id}", "title") or "Live News"
                category = redis_conn.hget(f"article:{row.news_id}", "category") or "news"
                abstract = redis_conn.hget(f"article:{row.news_id}", "abstract") or ""
                image_url = redis_conn.hget(f"article:{row.news_id}", "image_url") or ""
            else:
                continue
                
            recs.append({
                "news_id": row.news_id,
                "title": title,
                "abstract": abstract,
                "category": category,
                "image_url": image_url,
                "score": float(row.score),
                "rank": rank
            })
            
        latency = (time.time() - start_time) * 1000
        return recs, latency, False
