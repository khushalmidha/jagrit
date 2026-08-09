import os
import time
import argparse
import pandas as pd
from kafka import KafkaProducer
import json
import datetime

def replay_events(speed_multiplier, bootstrap_servers):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    interactions_path = os.path.join(base_dir, "data", "processed", "interactions.parquet")
    news_path = os.path.join(base_dir, "data", "processed", "news.parquet")
    
    if not os.path.exists(interactions_path) or not os.path.exists(news_path):
        print("Data files not found. Run preprocess.py first.")
        return
        
    df_interactions = pd.read_parquet(interactions_path)
    df_news = pd.read_parquet(news_path)
    
    # Sort by timestamp
    df_interactions = df_interactions.sort_values('timestamp')
    
    # Create lookup for category
    news_cat = df_news.set_index('news_id')['category'].to_dict()
    
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    print(f"Starting replay of {len(df_interactions)} events at {speed_multiplier}x speed...")
    
    first_event_time = df_interactions.iloc[0]['timestamp']
    if pd.isna(first_event_time):
        # Fallback if no valid timestamp
        first_event_time = datetime.datetime.now()
        
    real_start_time = time.time()
    
    for idx, row in df_interactions.iterrows():
        event_time = row['timestamp']
        if pd.isna(event_time):
            continue
            
        # Calculate time to wait based on speed multiplier
        # (event_time - first_event_time) / multiplier
        simulated_elapsed = (event_time - first_event_time).total_seconds() / speed_multiplier
        real_elapsed = time.time() - real_start_time
        
        sleep_time = simulated_elapsed - real_elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
            
        news_id = row['news_id']
        category = news_cat.get(news_id, "unknown")
        
        event = {
            'user_id': row['user_id'],
            'news_id': news_id,
            'category': category,
            'timestamp': event_time.isoformat(),
            'clicked': int(row['clicked'])
        }
        
        producer.send('user-events', event)
        
        if idx % 1000 == 0:
            print(f"Replayed {idx} events. Last event time: {event_time}")
            producer.flush()
            
    producer.flush()
    print("Replay completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-speed", type=float, default=100.0, help="Speed multiplier (e.g. 100x)")
    parser.add_argument("--bootstrap-servers", type=str, default="localhost:9092", help="Kafka broker")
    args = parser.parse_args()
    
    replay_events(args.replay_speed, args.bootstrap_servers)
