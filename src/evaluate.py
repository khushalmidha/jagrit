import os
import argparse
import pandas as pd
import pickle
import redis
from src.candidate_generation import CandidateGenerator
from src.features import FeatureEngineer

def evaluate_user(user_id, use_live_features=False):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    news_path = os.path.join(base_dir, "data", "processed", "news.parquet")
    interactions_path = os.path.join(base_dir, "data", "processed", "interactions.parquet")
    model_path = os.path.join(base_dir, "models", "ranker.pkl")
    
    if not os.path.exists(news_path) or not os.path.exists(model_path):
        print("Required files not found. Ensure preprocess.py and train_ranker.py have been run.")
        return
        
    df_news = pd.read_parquet(news_path)
    df_interactions = pd.read_parquet(interactions_path)
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    redis_client = None
    if use_live_features:
        try:
            redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            redis_client.ping()
        except redis.ConnectionError:
            print("Redis not available, falling back to static features.")
            redis_client = None
            
    # Get user history
    user_hist = df_interactions[df_interactions['user_id'] == user_id]
    if user_hist.empty:
        print(f"User {user_id} not found in history. Simulating cold-start.")
        user_history_news_ids = []
    else:
        # Get the history from the last interaction
        hist_str = user_hist.iloc[-1]['history']
        user_history_news_ids = str(hist_str).split() if pd.notna(hist_str) else []
        
    print(f"User {user_id} history length: {len(user_history_news_ids)}")
    
    # Generate Candidates
    cg = CandidateGenerator(df_news, df_interactions)
    candidates = cg.generate_candidates(user_history_news_ids, num_candidates=50)
    print(f"Generated {len(candidates)} candidates.")
    
    if not candidates:
        print("No candidates generated.")
        return
        
    # Extract Features
    fe = FeatureEngineer(df_news)
    df_feats = fe.build_features(user_id, user_history_news_ids, candidates, live_redis_client=redis_client)
    
    if df_feats.empty:
        print("Feature generation failed.")
        return
        
    feature_cols = [
        'category_match_score', 'article_popularity', 'article_recency_hours',
        'title_length', 'abstract_length', 'user_total_clicks_in_history', 'user_category_diversity'
    ]
    
    # Rank
    preds = model.predict_proba(df_feats[feature_cols])[:, 1]
    df_feats['score'] = preds
    
    # Sort and get top 10
    top_10 = df_feats.sort_values(by='score', ascending=False).head(10)
    
    print("\n--- TOP 10 RECOMMENDED ARTICLES ---")
    news_dict = df_news.set_index('news_id').to_dict('index')
    
    for rank, row in enumerate(top_10.itertuples(), 1):
        news_id = row.news_id
        score = row.score
        article = news_dict[news_id]
        print(f"[{rank}] Score: {score:.4f} | Category: {article['category']:<12} | Title: {article['title']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_id", type=str, required=True, help="User ID to evaluate")
    parser.add_argument("--use-live-features", action="store_true", help="Use Redis for live features")
    args = parser.parse_args()
    
    evaluate_user(args.user_id, args.use_live_features)
