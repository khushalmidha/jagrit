import json
import argparse
from kafka import KafkaConsumer
import redis

def consume_and_update(bootstrap_servers, redis_host, redis_port):
    consumer = KafkaConsumer(
        'user-events',
        bootstrap_servers=bootstrap_servers,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='feature-updater-group'
    )
    
    r = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)
    
    print("Listening for events to update Redis features...")
    
    try:
        r.ping()
    except redis.ConnectionError:
        print("Redis is not available. Please ensure Redis is running.")
        return
        
    for message in consumer:
        event = message.value
        user_id = event['user_id']
        news_id = event['news_id']
        category = event['category']
        clicked = event['clicked']
        
        # 1. Update article impressions and clicks (CTR)
        r.hincrby(f"article:{news_id}", "impressions", 1)
        if clicked == 1:
            r.hincrby(f"article:{news_id}", "clicks", 1)
            
            # 2. Update user's category clicks
            r.hincrby(f"user:{user_id}:category_clicks", category, 1)
            
            # 3. Update user's last 10 clicked articles
            # LPUSH adds to the head (most recent first)
            r.lpush(f"user:{user_id}:last_10_clicked", news_id)
            # LTRIM keeps only the first 10 elements
            r.ltrim(f"user:{user_id}:last_10_clicked", 0, 9)
            
        print(f"Processed event for User {user_id} -> News {news_id} (Clicked: {clicked})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-servers", type=str, default="localhost:9092")
    parser.add_argument("--redis-host", type=str, default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    args = parser.parse_args()
    
    consume_and_update(args.bootstrap_servers, args.redis_host, args.redis_port)
