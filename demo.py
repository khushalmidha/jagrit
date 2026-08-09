import requests
import json
import time

def run_demo():
    print("--- REAL-TIME CONTENT RANKING DEMO ---")
    print("Assuming docker-compose is up and running on localhost:8000\n")
    
    # 1. Warm-up and active user
    print("Test 1: Active User (U13740)")
    # U13740 is a real MIND user_id, assuming it's in the dev set
    payload = {"user_id": "U13740", "top_k": 5}
    try:
        start = time.time()
        res = requests.post("http://localhost:8000/recommend", json=payload)
        res.raise_for_status()
        data = res.json()
        print(f"Latency: {data['latency_ms']:.2f}ms | Cold Start: {data['is_cold_start']}")
        for r in data['recommendations']:
            print(f"[{r['rank']}] ({r['score']:.4f}) {r['category']:<12} | {r['title']}")
    except Exception as e:
        print(f"Error: {e}")
        
    print("\n-----------------------------------\n")
    
    # 2. Another active user
    print("Test 2: Active User (U91836)")
    payload = {"user_id": "U91836", "top_k": 5}
    try:
        res = requests.post("http://localhost:8000/recommend", json=payload)
        res.raise_for_status()
        data = res.json()
        print(f"Latency: {data['latency_ms']:.2f}ms | Cold Start: {data['is_cold_start']}")
        for r in data['recommendations']:
            print(f"[{r['rank']}] ({r['score']:.4f}) {r['category']:<12} | {r['title']}")
    except Exception as e:
        print(f"Error: {e}")
        
    print("\n-----------------------------------\n")
    
    # 3. Cold Start User
    print("Test 3: Cold Start User (NEW_USER_999)")
    payload = {"user_id": "NEW_USER_999", "top_k": 5}
    try:
        res = requests.post("http://localhost:8000/recommend", json=payload)
        res.raise_for_status()
        data = res.json()
        print(f"Latency: {data['latency_ms']:.2f}ms | Cold Start: {data['is_cold_start']}")
        print("Since this is a cold start, returning diversified popular articles:")
        for r in data['recommendations']:
            print(f"[{r['rank']}] ({r['score']:.4f}) {r['category']:<12} | {r['title']}")
    except Exception as e:
        print(f"Error: {e}")
        
    print("\n-----------------------------------\n")
    
    # Metrics
    print("Fetching Global Metrics...")
    try:
        res = requests.get("http://localhost:8000/metrics")
        res.raise_for_status()
        metrics = res.json()
        print(json.dumps(metrics, indent=2))
    except Exception as e:
        print(f"Error fetching metrics: {e}")
        
if __name__ == "__main__":
    run_demo()
