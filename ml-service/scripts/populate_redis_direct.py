import os
import pandas as pd
import redis
import time

def populate_redis(redis_url, speed_multiplier=100.0):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    interactions_path = os.path.join(base_dir, "data", "processed", "interactions.parquet")
    news_path = os.path.join(base_dir, "data", "processed", "news.parquet")
    
    if not os.path.exists(interactions_path) or not os.path.exists(news_path):
        print("Data files not found. Ensure preprocess.py was run.")
        return
        
    df_interactions = pd.read_parquet(interactions_path)
    df_news = pd.read_parquet(news_path)
    
    # Sort by timestamp
    df_interactions = df_interactions.sort_values('timestamp')
    
    # Create lookup for category
    news_cat = df_news.set_index('news_id')['category'].to_dict()
    
    print(f"Connecting to Redis at {redis_url.split('@')[-1] if '@' in redis_url else 'localhost'}...")
    r = redis.from_url(redis_url, decode_responses=True)
    
    try:
        r.ping()
        print("Connected to Upstash Redis successfully!")
    except Exception as e:
        print(f"Failed to connect to Redis: {e}")
        return
        
    print(f"Starting direct population of {len(df_interactions)} events to Redis (bypassing Kafka)...")
    
    # Use pipelining for massive speedup
    pipe = r.pipeline(transaction=False)
    
    for idx, row in df_interactions.iterrows():
        news_id = row['news_id']
        category = news_cat.get(news_id, "unknown")
        user_id = row['user_id']
        clicked = int(row['clicked'])
        
        # 1. Update article impressions
        pipe.hincrby(f"article:{news_id}", "impressions", 1)
        if clicked == 1:
            pipe.hincrby(f"article:{news_id}", "clicks", 1)
            
            # 2. Update user's category clicks
            pipe.hincrby(f"user:{user_id}:category_clicks", category, 1)
            
            # 3. Update user's last 10 clicked articles
            pipe.lpush(f"user:{user_id}:last_10_clicked", news_id)
            pipe.ltrim(f"user:{user_id}:last_10_clicked", 0, 9)
            
        # Execute pipeline every 10,000 commands
        if idx % 10000 == 0 and idx > 0:
            pipe.execute()
            print(f"Processed {idx} events...")
            
    pipe.execute()
    print("Redis population completed! All ML features are now in Upstash Cloud.")

if __name__ == "__main__":
    url = os.environ.get("REDIS_URL")
    if not url:
        print("Please set REDIS_URL environment variable.")
    else:
        populate_redis(url)
