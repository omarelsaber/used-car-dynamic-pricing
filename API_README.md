# Used Car Price Prediction API

Production-grade REST API for predicting used car prices using XGBoost.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the API

**Option A: Using Makefile (Recommended)**
```bash
make api-dev
```

**Option B: Using uvicorn directly**
```bash
cd src/app
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

**Option C: From project root**
```bash
uvicorn src.app.api:app --reload --host 0.0.0.0 --port 8000
```

### 3. Access API Documentation

Once the API is running, open your browser:

- **Interactive Docs (Swagger UI)**: http://localhost:8000/docs
- **Alternative Docs (ReDoc)**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## API Endpoints

### `GET /`
Root endpoint with API information.

### `GET /health`
Health check endpoint. Returns API status and whether ML artifacts are loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "preprocessor_loaded": true,
  "version": "1.0.0"
}
```

### `POST /predict`
Predict used car price.

**Request Body:**
```json
{
  "name": "Toyota Corolla",
  "year": 2020,
  "miles": "45,000 miles",
  "color": "Black",
  "condition": "No accidents reported, 1 Owner"
}
```

**Response:**
```json
{
  "predicted_price": 15988.50,
  "currency": "USD",
  "model_version": "xgboost_v2.0"
}
```

### `GET /model/info`
Get information about the loaded ML model.

## Testing

### Run API Tests
```bash
pytest tests/test_api.py -v
```

### Test with cURL

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Toyota Corolla",
    "year": 2020,
    "miles": "45,000 miles",
    "color": "Black",
    "condition": "No accidents reported, 1 Owner"
  }'
```

### Test with Python

```python
import requests

url = "http://localhost:8000/predict"
payload = {
    "name": "Toyota Corolla",
    "year": 2020,
    "miles": "45,000 miles",
    "color": "Black",
    "condition": "No accidents reported, 1 Owner"
}

response = requests.post(url, json=payload)
print(response.json())
```

## Requirements

- Python 3.8+
- Trained model: `models/model.pkl`
- Preprocessor: `models/preprocessor.pkl`

Make sure you've run the training pipeline first:
```bash
dvc repro
```

## Model Details

- **Model Type**: XGBoost Regressor
- **Target Transform**: log1p (natural log + 1)
- **Performance**: R² ≈ 0.40
- **Version**: xgboost_v2.0

## Troubleshooting

### Model/Preprocessor Not Found
Make sure you're running the API from the project root, or that the model files exist in `models/` directory.

### Port Already in Use
Change the port:
```bash
uvicorn src.app.api:app --reload --port 8001
```

### Import Errors
Make sure you've installed all dependencies:
```bash
pip install -r requirements.txt
```
