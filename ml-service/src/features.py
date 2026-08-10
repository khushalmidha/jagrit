import pandas as pd
import numpy as np
from scipy.stats import entropy

class FeatureEngineer:
    def __init__(self, df_news):
        self.df_news = df_news
        # Create lookups for fast feature building
        self.news_dict = df_news.set_index('news_id').to_dict('index')
        
    def _compute_category_diversity(self, history_cats):
        if not history_cats:
            return 0.0
        counts = pd.Series(history_cats).value_counts()
        probs = counts / len(history_cats)
        return entropy(probs)
        
    def build_features(self, user_id, user_history_news_ids, candidate_news_ids, live_redis_client=None, static_popularity=None):
        """
        Build features for a user and a list of candidates.
        live_redis_client is used in Phase 2 for live features.
        """
        features_list = []
        
        # User side features
        user_total_clicks = len(user_history_news_ids) if user_history_news_ids else 0
        
        history_cats = []
        if user_history_news_ids:
            for nid in user_history_news_ids:
                if nid in self.news_dict:
                    history_cats.append(self.news_dict[nid]['category'])
                    
        user_cat_diversity = self._compute_category_diversity(history_cats)
        
        for news_id in candidate_news_ids:
            cat = "news"
            title_len = 0
            abs_len = 0
            
            if news_id in self.news_dict:
                article = self.news_dict[news_id]
                cat = article['category']
                title_len = article.get('title_word_count', 0)
                abs_len = len(str(article.get('abstract', '')).split())
            elif live_redis_client:
                cat = live_redis_client.hget(f"article:{news_id}", "category") or "news"
                title = live_redis_client.hget(f"article:{news_id}", "title") or ""
                abstract = live_redis_client.hget(f"article:{news_id}", "abstract") or ""
                title_len = len(title.split())
                abs_len = len(abstract.split())
            else:
                continue
            
            # category_match_score
            cat_match = 0.0
            if history_cats:
                cat_match = history_cats.count(cat) / len(history_cats)
                
            # article_popularity (CTR)
            # In Phase 1 static, we use precalculated CTR to avoid train/serve skew.
            # In Phase 2 live, we pull from Redis.
            article_popularity = 0.0
            if live_redis_client:
                # Example Redis pull: live CTR
                clicks = live_redis_client.hget(f"article:{news_id}", "clicks")
                imps = live_redis_client.hget(f"article:{news_id}", "impressions")
                if clicks and imps and int(imps) > 0:
                    article_popularity = int(clicks) / int(imps)
            elif static_popularity:
                article_popularity = static_popularity.get(news_id, 0.0)
            
            # Recency (MIND news.tsv lacks timestamps, using 0 or mock)
            article_recency_hours = 24.0 
            
            features_list.append({
                'user_id': user_id,
                'news_id': news_id,
                'category_match_score': cat_match,
                'article_popularity': article_popularity,
                'article_recency_hours': article_recency_hours,
                'title_length': title_len,
                'abstract_length': abs_len,
                'user_total_clicks_in_history': user_total_clicks,
                'user_category_diversity': user_cat_diversity
            })
            
        return pd.DataFrame(features_list)

def create_training_dataset(interactions_path, news_path, output_path):
    """
    Build features for the entire interaction dataset to train the ranker.
    Maintains the time-based split inherently present in interactions data.
    """
    df_interactions = pd.read_parquet(interactions_path)
    df_news = pd.read_parquet(news_path)
    
    fe = FeatureEngineer(df_news)
    
    # We will build features for a sample of the data to save time and memory for this project.
    # In a real system, we'd use Spark.
    print(f"Building features for {len(df_interactions)} interactions... this may take a moment.")
    
    # Pre-calculate global article popularity for static training
    clicks_count = df_interactions.groupby('news_id')['clicked'].sum()
    imps_count = df_interactions.groupby('news_id')['clicked'].count()
    global_ctr = (clicks_count / imps_count).to_dict()
    
    features_list = []
    
    for idx, row in df_interactions.iterrows():
        user_id = row['user_id']
        news_id = row['news_id']
        clicked = row['clicked']
        split = row['split']
        impression_id = row['impression_id']
        history = str(row['history']).split() if pd.notna(row['history']) else []
        
        # Build base features
        df_feats = fe.build_features(user_id, history, [news_id])
        if not df_feats.empty:
            feat_dict = df_feats.iloc[0].to_dict()
            # Override popularity with static precalculated
            feat_dict['article_popularity'] = global_ctr.get(news_id, 0.0)
            feat_dict['clicked'] = clicked
            feat_dict['split'] = split
            feat_dict['impression_id'] = impression_id
            features_list.append(feat_dict)
            
        if idx % 10000 == 0 and idx > 0:
            print(f"Processed {idx} interactions...")
            
    df_train_dev = pd.DataFrame(features_list)
    df_train_dev.to_parquet(output_path, index=False)
    print(f"Saved {len(df_train_dev)} feature vectors to {output_path}")

if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    interactions_path = os.path.join(base_dir, "data", "processed", "interactions.parquet")
    news_path = os.path.join(base_dir, "data", "processed", "news.parquet")
    output_path = os.path.join(base_dir, "data", "processed", "features.parquet")
    
    if os.path.exists(interactions_path) and os.path.exists(news_path):
        create_training_dataset(interactions_path, news_path, output_path)
    else:
        print("Preprocessed data not found. Run preprocess.py first.")
