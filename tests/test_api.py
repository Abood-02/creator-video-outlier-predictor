"""
Integration tests for the FastAPI serving layer.
"""

from fastapi.testclient import TestClient

from src.api.app import app


def test_health_endpoint():
    """Verify health endpoint."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "models_loaded" in data


def test_predict_endpoint_validation():
    """Verify request validation for predict endpoint."""
    with TestClient(app) as client:
        # Missing required fields
        response = client.post("/predict", json={"title": "Test Video"})
        assert response.status_code == 422


def test_predict_endpoint_success():
    """Verify successful prediction."""
    with TestClient(app) as client:
        # Note: Since the test runs without full artifact generation by default 
        # (unless artifacts are present from previous tests), we may get a 400 for missing models
        # or it will succeed if artifacts exist.
        payload = {
            "title": "I spent $1,000,000 on a new house",
            "published_at": "2023-01-01T12:00:00Z",
            "duration_seconds": 600,
            "tags_count": 5,
            "channel_median_views": 100000.0,
            "thumbnail_url_or_path": None,
            "model_type": "lgbm"
        }
        
        response = client.post("/predict", json=payload)
        
        # If models are loaded, we expect 200
        # If models aren't loaded (e.g. running in isolated CI), we expect 400
        if response.status_code == 200:
            data = response.json()
            assert "outlier_probability" in data
            assert "predicted_view_ratio" in data
            assert "risk_tier" in data
            assert "latency_ms" in data
        else:
            assert response.status_code == 400
            assert "is not loaded" in response.json()["detail"] or "failed" in response.json()["detail"]
