import pytest
from fastapi.testclient import TestClient
from src.serving.app import app

client = TestClient(app)

def test_health_endpoint():
    """Verify that the health check endpoint responds successfully."""
    response = client.get("/health")
    # If your model isn't fully loaded in the test environment, 
    # it might return a 503, which is an expected handled state.
    assert response.status_code in [200, 503]

def test_predict_endpoint_validation():
    """Verify that sending a malformed payload triggers a validation error (422)."""
    bad_payload = {
        "user_id": "USER_123",
        # Missing transaction_id, amount, and device_type
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422