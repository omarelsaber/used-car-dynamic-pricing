# Deployment Guide – Plan B: Docker Ready, Run Locally

This project is **GitHub-ready** with full Docker support. You can run everything locally without Docker (API + Streamlit) or use Docker when available.

---

## Prerequisites

- **Python 3.10+** (for local run)
- **Docker & Docker Compose** (optional, for containerized run)
- **Trained artifacts**: `models/model.pkl` and `models/preprocessor.pkl`  
  → If missing, run: `dvc repro` (or run the pipeline steps manually)

---

## Option 1: Run Locally (No Docker)

### API (FastAPI)
```bash
# From project root
python -m uvicorn src.app.api:app --reload --host 127.0.0.1 --port 8000
```
- API: http://localhost:8000  
- Docs: http://localhost:8000/docs  

### Frontend (Streamlit)
```bash
# From project root; API must be running
set API_URL=http://localhost:8000
streamlit run src/frontend/app.py --server.port=8501
```
- UI: http://localhost:8501  

---

## Option 2: Run with Docker Compose

Ensures all dependencies (XGBoost, Streamlit-Lottie, FastAPI, etc.) are included via `requirements.txt`.

```bash
# Build and start API + Frontend
docker-compose up -d --build

# API:  http://localhost:8000
# UI:   http://localhost:8501
```

### Verify
```bash
# Health checks
curl http://localhost:8000/health
curl -s http://localhost:8501/_stcore/health

# Stop
docker-compose down
```

---

## Docker Layout

| File | Purpose |
|------|--------|
| `Dockerfile` | API image (FastAPI, XGBoost, uvicorn). Port 8000. |
| `Dockerfile.frontend` | Frontend image (Streamlit, streamlit-lottie). Port 8501. |
| `docker-compose.yaml` | Runs `api` + `frontend`; frontend waits for API health. |
| `requirements.txt` | Single source of dependencies for both images. |

---

## Dependencies (requirements.txt)

- **ML**: pandas, numpy, scikit-learn, **xgboost**
- **API**: fastapi, uvicorn, pydantic, python-multipart
- **Frontend**: **streamlit**, **streamlit-lottie**, requests, plotly
- **Utils**: joblib, pyyaml, python-dotenv
- **Dev/Quality**: pytest, black, flake8, etc.

All Docker builds use this same `requirements.txt`; no separate file needed.

---

## Troubleshooting

| Issue | Action |
|-------|--------|
| `models/model.pkl` not found | Run `dvc repro` or train the model and save to `models/`. |
| Port 8000 or 8501 in use | Change port in `docker-compose.yaml` or use `--port` when running locally. |
| Frontend can't reach API | In Docker, `API_URL=http://api:8000` is set automatically. Locally, set `API_URL=http://localhost:8000`. |
| Docker build fails | Ensure Docker daemon is running and `requirements.txt` has no typos. |

---

## GitHub Readiness

- All Docker configs are in the repo.
- Single `requirements.txt` for API, frontend, and pipeline.
- Plan B: clone repo → install deps → run API + Streamlit locally.
- When Docker is available: `docker-compose up -d --build`.
