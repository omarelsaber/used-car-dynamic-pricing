"""
API Integration Tests
====================

Tests the FastAPI endpoints for car price prediction.
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from app.api import app, load_artifacts

# Load artifacts before running tests (TestClient doesn't trigger lifespan events)
try:
    load_artifacts()
except Exception as e:
    pytest.skip(f"Could not load artifacts for testing: {e}", allow_module_level=True)

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint returns API info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["preprocessor_loaded"] is True


def test_predict_toyota():
    """Test prediction for Toyota Corolla."""
    payload = {
        "name": "Maruti Swift Dzire VDI",
        "year": 2014,
        "km_driven": 145500,
        "fuel": "Diesel",
        "seller_type": "Individual",
        "transmission": "Manual",
        "owner": "First Owner",
        "mileage": 23.4,
        "engine": 1248,
        "max_power": 74.0,
        "seats": 5
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "predicted_price" in data
    assert "currency" in data
    assert data["currency"] == "INR"
    assert data["predicted_price"] > 0


def test_predict_invalid_year():
    """Test validation for invalid year."""
    payload = {
        "name": "Maruti Swift Dzire VDI",
        "year": 2030,  # Future year (invalid)
        "km_driven": 145500,
        "fuel": "Diesel",
        "seller_type": "Individual",
        "transmission": "Manual",
        "owner": "First Owner",
        "mileage": 23.4,
        "engine": 1248,
        "max_power": 74.0,
        "seats": 5
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error


def test_predict_invalid_mileage():
    """Test validation for invalid mileage."""
    payload = {
        "name": "Maruti Swift Dzire VDI",
        "year": 2020,
        "km_driven": 145500,
        "fuel": "Diesel",
        "seller_type": "Individual",
        "transmission": "Manual",
        "owner": "First Owner",
        "mileage": -5,  # Invalid negative mileage
        "engine": 1248,
        "max_power": 74.0,
        "seats": 5
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error


def test_model_info():
    """Test model info endpoint."""
    response = client.get("/model/info")
    assert response.status_code == 200
    
    data = response.json()
    assert "model_type" in data
    assert "model_version" in data
    assert "n_features" in data
