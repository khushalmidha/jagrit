import pandas as pd
import numpy as np

class CandidateGenerator:
    def __init__(self, df_news, df_interactions=None):
        """
        Initializes CandidateGenerator.
        - df_news: DataFrame of all available news articles.
        - df_interactions: DataFrame of historical interactions (for calculating popularity).
        """
        self.df_news = df_news
        self.df_interactions = df_interactions
        
        # Precompute popular articles if interactions are provided
        if df_interactions is not None:
            clicks = df_interactions[df_interactions['clicked'] == 1]
            self.popular_news = clicks['news_id'].value_counts().head(100).index.tolist()
        else:
            self.popular_news = self.df_news.head(100)['news_id'].tolist()
            
        # Recent articles (mocked using the last 100 in the dataset for now since MIND news.tsv lacks timestamps)
        # In a real system, we'd use article publish time.
        self.recent_news = self.df_news['news_id'].tail(100).tolist()
        
        # Create a mapping of category -> list of news_ids for fast lookup
        self.category_to_news = self.df_news.groupby('category')['news_id'].apply(list).to_dict()
        
    def generate_candidates(self, user_history_news_ids, num_candidates=50):
        """
        Generate candidates based on:
        a) Same-category articles as user's recent clicks
        b) Most popular articles
        c) Most recent articles
        d) Exploration (random diverse categories)
        
        Documented reasoning:
        - Same-category captures primary user interests.
        - Popularity captures global trends (wisdom of the crowd).
        - Recency ensures fresh content, critical for news.
        - Exploration avoids filter bubbles and discovers new interests.
        """
        candidates = set()
        
        # 1. Same-category articles
        user_categories = []
        if user_history_news_ids:
            hist_news = self.df_news[self.df_news['news_id'].isin(user_history_news_ids)]
            user_categories = hist_news['category'].value_counts().index.tolist()
            
            for cat in user_categories[:3]: # top 3 categories from history
                if cat in self.category_to_news:
                    cat_news = self.category_to_news[cat]
                    # Take some random samples from these categories
                    sampled = np.random.choice(cat_news, min(10, len(cat_news)), replace=False)
                    candidates.update(sampled)
                    
        # 2. Most popular articles (add ~15)
        candidates.update(self.popular_news[:15])
        
        # 3. Most recent articles (add ~15)
        candidates.update(self.recent_news[:15])
        
        # 4. Exploration (add ~10 from random categories not in user's top)
        all_cats = list(self.category_to_news.keys())
        explore_cats = [c for c in all_cats if c not in user_categories]
        if explore_cats:
            sampled_cats = np.random.choice(explore_cats, min(3, len(explore_cats)), replace=False)
            for cat in sampled_cats:
                cat_news = self.category_to_news[cat]
                sampled = np.random.choice(cat_news, min(5, len(cat_news)), replace=False)
                candidates.update(sampled)
                
        # Remove history from candidates so we don't recommend already clicked items
        if user_history_news_ids:
            candidates.difference_update(user_history_news_ids)
            
        return list(candidates)[:num_candidates]
