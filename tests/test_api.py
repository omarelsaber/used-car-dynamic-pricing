"""
API Integration Tests
====================

Tests the FastAPI endpoints for car price prediction.
Gracefully skips if model artifacts are not available.
"""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@pytest.fixture(scope="module")
def api_client():
    """Create API client for testing."""
    from fastapi.testclient import TestClient
    from app.api import app, load_artifacts
    
    # Try to load artifacts, but don't fail if they don't exist
    try:
        load_artifacts()
    except FileNotFoundError as e:
        pytest.skip(f"Model artifacts not found: {e}")
    
    return TestClient(app)


# Check if model artifacts exist before running API tests
def model_exists():
    """Check if model artifacts exist."""
    project_root = Path(__file__).parent.parent
    model_path = project_root / "models" / "model.pkl"
    preprocessor_path = project_root / "models" / "preprocessor.pkl"
    return model_path.exists() and preprocessor_path.exists()


pytestmark = pytest.mark.skipif(
    not model_exists(),
    reason="Model artifacts not found - skipping API tests"
)


@pytest.mark.requires_model
def test_root_endpoint(api_client):
    """Test root endpoint returns API info."""
    response = api_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


@pytest.mark.requires_model
def test_health_check(api_client):
    """Test health check endpoint."""
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["preprocessor_loaded"] is True


@pytest.mark.requires_model
def test_predict_toyota(api_client):
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
    
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "predicted_price" in data
    assert "currency" in data
    assert data["currency"] == "INR"
    assert data["predicted_price"] > 0


@pytest.mark.requires_model
def test_predict_invalid_year(api_client):
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
    
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error


@pytest.mark.requires_model
def test_predict_invalid_mileage(api_client):
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
    
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error


@pytest.mark.requires_model
def test_model_info(api_client):
    """Test model info endpoint."""
    response = api_client.get("/model/info")
    assert response.status_code == 200
    
    data = response.json()
    assert "model_type" in data
    assert "model_version" in data
    assert "n_features" in data
