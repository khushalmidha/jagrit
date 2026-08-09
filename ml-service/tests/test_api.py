import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_version" in data

def test_recommend_valid_user():
    # Use a dummy user_id that might exist, or simulate
    payload = {"user_id": "U12345", "top_k": 5}
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert len(data["recommendations"]) <= 5
    assert "latency_ms" in data

def test_recommend_cold_start():
    # Provide a completely unheard of user_id
    payload = {"user_id": "COLD_START_USER_999", "top_k": 10}
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_cold_start"] == True
    assert len(data["recommendations"]) > 0

def test_recommend_invalid_input():
    payload = {"user_id": 12345} # Should be string
    # Depending on pydantic version, it might auto-cast, let's omit user_id
    response = client.post("/recommend", json={"top_k": 5})
    assert response.status_code == 422 # Validation error

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "avg_latency_ms" in data
    assert "cold_start_percentage" in data
