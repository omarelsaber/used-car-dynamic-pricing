"""
Pydantic Schemas for Cardekho V3 Dataset
========================================
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime


class CarInput(BaseModel):
    """Input schema for car price prediction - Cardekho V3."""
    
    name: str = Field(
        ...,
        description="Car make and model",
        examples=["Maruti Swift Dzire VDI", "Hyundai i20 Sportz Diesel"]
    )
    
    year: int = Field(
        ...,
        description="Manufacturing year",
        ge=2000,
        le=datetime.now().year,
        examples=[2014, 2017, 2020]
    )
    
    km_driven: int = Field(
        ...,
        description="Total kilometers driven",
        ge=0,
        examples=[45000, 37000, 21000]
    )
    
    fuel: str = Field(
        ...,
        description="Fuel type",
        examples=["Petrol", "Diesel", "CNG", "LPG", "Electric"]
    )
    
    seller_type: str = Field(
        ...,
        description="Seller type",
        examples=["Individual", "Dealer", "Trustmark Dealer"]
    )
    
    transmission: str = Field(
        ...,
        description="Transmission type",
        examples=["Manual", "Automatic"]
    )
    
    owner: str = Field(
        ...,
        description="Owner type",
        examples=["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"]
    )
    
    mileage: float = Field(
        ...,
        description="Fuel efficiency in kmpl",
        gt=0,
        examples=[23.4, 18.9, 17.3]
    )
    
    engine: int = Field(
        ...,
        description="Engine size in CC",
        gt=0,
        examples=[1248, 1197, 1582]
    )
    
    max_power: float = Field(
        ...,
        description="Maximum power in bhp",
        gt=0,
        examples=[74.0, 81.86, 103.52]
    )
    
    seats: int = Field(
        ...,
        description="Number of seats",
        ge=2,
        le=10,
        examples=[5, 7, 8]
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Output schema for price prediction."""
    
    predicted_price: float = Field(..., gt=0)
    currency: str = Field(default="INR")  # Changed to INR for Indian dataset
    model_version: Optional[str] = Field(default=None)


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    model_loaded: bool
    preprocessor_loaded: bool
    version: str = Field(default="5.0")


class ErrorResponse(BaseModel):
    """Error response."""
    
    error: str
    detail: Optional[str] = None
