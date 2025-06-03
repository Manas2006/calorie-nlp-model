"""Tests for the FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient
from calorie_nlp.api import app

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_single_prediction():
    """Test single food prediction endpoint."""
    response = client.post(
        "/predict",
        json={"name": "chocolate milk"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "food" in data
    assert "calories_per_100g" in data
    assert data["food"] == "chocolate milk"
    assert isinstance(data["calories_per_100g"], float)

def test_batch_prediction():
    """Test batch prediction endpoint."""
    response = client.post(
        "/predict/batch",
        json={"items": ["chocolate milk", "apple", "pizza"]}
    )
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 3
    
    for pred in data["predictions"]:
        assert "food" in pred
        assert "calories_per_100g" in pred
        assert isinstance(pred["calories_per_100g"], float)

def test_invalid_input():
    """Test error handling for invalid input."""
    response = client.post(
        "/predict",
        json={"invalid": "input"}
    )
    assert response.status_code == 422  # Validation error

def test_empty_batch():
    """Test error handling for empty batch."""
    response = client.post(
        "/predict/batch",
        json={"items": []}
    )
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 0 