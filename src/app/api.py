"""
FastAPI Application for Used Car Price Prediction
=================================================

Production-grade REST API for real-time car price predictions.
Uses XGBoost model with log-transformed predictions.

Author: Omar Elsaber
Date: Feb 2026
"""

import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional
import re

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .schemas import CarInput, PredictionResponse, HealthResponse, ErrorResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Global variables for model artifacts (loaded at startup)
model = None
preprocessor = None
MODEL_VERSION = "xgboost_v5.0_cardekho"


def load_artifacts():
    """
    Load ML model and preprocessor artifacts at startup.
    
    This function is called once when the API starts, avoiding
    the overhead of loading artifacts on every request.
    
    Returns:
        tuple: (model, preprocessor)
        
    Raises:
        FileNotFoundError: If artifacts are not found
    """
    global model, preprocessor
    
    logger.info("Loading ML artifacts...")
    
    # Define paths (relative to project root)
    project_root = Path(__file__).parent.parent.parent
    model_path = project_root / "models" / "model.pkl"
    preprocessor_path = project_root / "models" / "preprocessor.pkl"
    
    # Check if files exist
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}")
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    logger.info(f"✅ Model loaded: {type(model).__name__}")
    
    # Load preprocessor
    logger.info(f"Loading preprocessor from {preprocessor_path}")
    preprocessor = joblib.load(preprocessor_path)
    logger.info(f"✅ Preprocessor loaded: {type(preprocessor).__name__}")
    
    logger.info("🎉 All artifacts loaded successfully!")
    
    return model, preprocessor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    
    Handles startup and shutdown events.
    Loads model artifacts at startup.
    """
    # Startup
    logger.info("="*80)
    logger.info("STARTING USED CAR PRICE PREDICTION API")
    logger.info("="*80)
    
    try:
        load_artifacts()
        logger.info("API ready to accept requests!")
    except Exception as e:
        logger.error(f"Failed to load artifacts: {str(e)}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down API...")


# Initialize FastAPI app
app = FastAPI(
    title="Used Car Price Prediction API",
    description="""
    Production-grade REST API for predicting used car prices using XGBoost.
    
    ## Features
    * Real-time price predictions
    * Input validation with Pydantic
    * Automatic preprocessing pipeline
    * Log-transformed predictions for accuracy
    * Health check endpoint
    
    ## Model Details
    * **Model Type**: XGBoost Regressor
    * **Target Transform**: log1p (natural log + 1)
    * **Performance**: R² ≈ 0.40
    * **Version**: 2.0
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def clean_mileage_string(miles_str: str) -> int:
    """
    Clean mileage string and convert to integer.
    
    Matches the preprocessing logic from data processing pipeline.
    
    Args:
        miles_str (str): Mileage string (e.g., "45,000 miles")
        
    Returns:
        int: Cleaned mileage value
        
    Example:
        >>> clean_mileage_string("45,000 miles")
        45000
    """
    # Remove 'miles', commas, and whitespace
    cleaned = miles_str.replace('miles', '').replace('Miles', '').replace(',', '').strip()
    
    try:
        return int(float(cleaned))
    except ValueError as e:
        raise ValueError(f"Could not convert mileage '{miles_str}' to integer: {str(e)}")


def extract_brand(car_name: str) -> str:
    """Extract brand from car name."""
    if not car_name or car_name.strip() == "":
        return "Unknown"
    return car_name.strip().split()[0]


def create_features(input_data: CarInput) -> pd.DataFrame:
    """
    Create features from input data - Cardekho V5.0.
    
    Args:
        input_data (CarInput): Validated input
        
    Returns:
        pd.DataFrame: Features matching training pipeline
    """
    logger.info("Creating features (Cardekho V5.0)...")
    
    # Extract brand
    brand = extract_brand(input_data.name)
    
    # Calculate car age
    current_year = datetime.now().year
    car_age = current_year - input_data.year
    
    logger.info(f"Extracted brand: {brand}")
    logger.info(f"Car age: {car_age} years")
    
    # Create DataFrame - EXACT order as training
    df = pd.DataFrame([{
        'brand': brand,
        'fuel': input_data.fuel,
        'seller_type': input_data.seller_type,
        'transmission': input_data.transmission,
        'owner': input_data.owner,
        'year': input_data.year,
        'mileage_driven': input_data.km_driven,
        'mileage': input_data.mileage,
        'engine': input_data.engine,
        'max_power': input_data.max_power,
        'seats': input_data.seats,
        'car_age': car_age
    }])
    
    logger.info(f"✅ Features created: {df.columns.tolist()}")
    
    return df


def preprocess_and_predict(features_df: pd.DataFrame) -> float:
    """
    Preprocess features and make prediction.
    
    Steps:
    1. Transform features using fitted preprocessor
    2. Predict using XGBoost model (in log-space)
    3. Apply inverse transform (expm1) to get real price
    
    Args:
        features_df (pd.DataFrame): DataFrame with engineered features
        
    Returns:
        float: Predicted price in USD
        
    Raises:
        RuntimeError: If model or preprocessor not loaded
    """
    global model, preprocessor
    
    if model is None or preprocessor is None:
        raise RuntimeError("Model or preprocessor not loaded. Check API startup logs.")
    
    logger.info("Preprocessing features...")
    
    # Transform features using fitted preprocessor
    features_transformed = preprocessor.transform(features_df)
    logger.info(f"Features transformed: shape {features_transformed.shape}")
    
    # Predict (model outputs log-transformed price)
    logger.info("Making prediction...")
    price_log = model.predict(features_transformed)[0]
    logger.info(f"Prediction (log-space): {price_log:.4f}")
    
    # CRITICAL: Apply inverse transform to get real price
    # Model was trained on log1p(price), so we use expm1 to reverse
    price_real = np.expm1(price_log)
    logger.info(f"Prediction (real price): ${price_real:,.2f}")
    
    return float(price_real)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get(
    "/",
    summary="Root endpoint",
    description="Welcome message and API information"
)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Used Car Price Prediction API",
        "version": "1.0.0",
        "model_version": MODEL_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if API and ML artifacts are loaded and ready"
)
async def health_check():
    """
    Health check endpoint.
    
    Returns API status and whether ML artifacts are loaded.
    """
    model_loaded = model is not None
    preprocessor_loaded = preprocessor is not None
    
    status_value = "healthy" if (model_loaded and preprocessor_loaded) else "unhealthy"
    
    return HealthResponse(
        status=status_value,
        model_loaded=model_loaded,
        preprocessor_loaded=preprocessor_loaded,
        version="1.0.0"
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict car price",
    description="Predict the price of a used car based on its features",
    responses={
        200: {
            "description": "Successful prediction",
            "content": {
                "application/json": {
                    "example": {
                        "predicted_price": 15988.50,
                        "currency": "USD",
                        "model_version": "xgboost_v2.0"
                    }
                }
            }
        },
        400: {
            "description": "Invalid input",
            "model": ErrorResponse
        },
        500: {
            "description": "Server error",
            "model": ErrorResponse
        }
    }
)
async def predict_price(input_data: CarInput):
    """
    Predict used car price.
    
    **Process:**
    1. Validate input data (automatic via Pydantic)
    2. Clean mileage string
    3. Calculate car age and derived features
    4. Preprocess using fitted pipeline
    5. Predict using XGBoost model
    6. Apply inverse log transform
    7. Return predicted price
    
    **Example Request:**
    ```json
    {
        "name": "Toyota Corolla",
        "year": 2020,
        "miles": "45,000 miles",
        "color": "Black",
        "condition": "No accidents reported, 1 Owner"
    }
    ```
    
    **Example Response:**
    ```json
    {
        "predicted_price": 15988.50,
        "currency": "USD",
        "model_version": "xgboost_v2.0"
    }
    ```
    """
    try:
        logger.info("="*80)
        logger.info("NEW PREDICTION REQUEST")
        logger.info("="*80)
        logger.info(f"Input: {input_data.model_dump()}")
        
        # Create features (matching training pipeline exactly)
        features_df = create_features(input_data)
        
        # Preprocess and predict
        predicted_price = preprocess_and_predict(features_df)
        
        # Ensure price is positive
        if predicted_price < 0:
            logger.warning(f"Negative prediction ({predicted_price}), clipping to 0")
            predicted_price = 0.0
        
        logger.info(f"✅ Final prediction: ₹{predicted_price:,.2f}")
        logger.info("="*80)
        
        return PredictionResponse(
            predicted_price=round(predicted_price, 2),
            currency="INR",
            model_version=MODEL_VERSION
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get(
    "/model/info",
    summary="Model information",
    description="Get information about the loaded model"
)
async def model_info():
    """
    Get information about the loaded ML model.
    """
    global model, preprocessor
    
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model or preprocessor not loaded"
        )
    
    # Get feature names from preprocessor
    feature_names = preprocessor.get_feature_names_out()
    
    return {
        "model_type": type(model).__name__,
        "model_version": MODEL_VERSION,
        "n_features": len(feature_names),
        "sample_features": list(feature_names[:10]),
        "hyperparameters": {
            "n_estimators": model.n_estimators if hasattr(model, 'n_estimators') else None,
            "learning_rate": model.learning_rate if hasattr(model, 'learning_rate') else None,
            "max_depth": model.max_depth if hasattr(model, 'max_depth') else None
        },
        "target_transform": "log1p",
        "preprocessor_type": type(preprocessor).__name__
    }


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler for unexpected errors."""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    
    # Run the API
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
