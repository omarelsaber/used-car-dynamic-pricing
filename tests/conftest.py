"""
Pytest Configuration and Fixtures
==================================

Provides shared fixtures and configuration for all tests.
"""

import pytest
from pathlib import Path
import sys

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@pytest.fixture(scope="session")
def model_artifacts_exist():
    """
    Check if model artifacts exist on disk.
    
    Returns:
        bool: True if both model.pkl and preprocessor.pkl exist
    """
    project_root = Path(__file__).parent.parent
    model_path = project_root / "models" / "model.pkl"
    preprocessor_path = project_root / "models" / "preprocessor.pkl"
    
    return model_path.exists() and preprocessor_path.exists()


@pytest.fixture(scope="session")
def test_client(model_artifacts_exist):
    """
    Create FastAPI TestClient with proper error handling.
    
    Returns:
        TestClient: FastAPI test client if artifacts exist, else None
    """
    if not model_artifacts_exist:
        pytest.skip("Model artifacts not found - skipping API tests", allow_module_level=False)
    
    from fastapi.testclient import TestClient
    from app.api import app, load_artifacts
    
    try:
        load_artifacts()
    except Exception as e:
        pytest.skip(f"Could not load artifacts for testing: {e}", allow_module_level=False)
    
    return TestClient(app)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", 
        "requires_model: mark test as requiring model artifacts"
    )
