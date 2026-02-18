# Docker - Used Car Price Prediction API

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Trained model and preprocessor in `models/` (run `dvc repro` first if needed)

### Production (single container)

```bash
# Build image
docker build -t used-car-price-api:latest .

# Run container
docker run -d --name car-price-api -p 8000:8000 used-car-price-api:latest

# Test
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"name":"Toyota Corolla","year":2020,"miles":"45,000 miles","color":"Black","condition":"No accidents reported, 1 Owner"}'

# Stop
docker stop car-price-api && docker rm car-price-api
```

### Docker Compose (recommended)

```bash
# Start API
docker-compose up -d --build

# Logs
docker-compose logs -f api

# Stop
docker-compose down
```

### Development (hot-reload)

```bash
# Start dev environment
docker-compose -f docker-compose.dev.yaml up -d --build

# API at http://localhost:8000 with code hot-reload
docker-compose -f docker-compose.dev.yaml down
```

## Makefile shortcuts

| Command | Description |
|---------|-------------|
| `make docker-build` | Build production image |
| `make docker-run` | Run container (port 8000) |
| `make docker-stop` | Stop and remove container |
| `make docker-logs` | Follow container logs |
| `make docker-test` | POST /predict sample |
| `make compose-up` | Start with docker-compose |
| `make compose-down` | Stop docker-compose |
| `make compose-dev` | Start dev (hot-reload) |
| `make compose-dev-down` | Stop dev environment |

## Health check

```bash
# Bash (Git Bash / WSL / Linux)
bash docker-healthcheck.sh

# Or manually
curl http://localhost:8000/health
```

## Image details

- **Base**: `python:3.10-slim`
- **Multi-stage**: builder + runtime (smaller image)
- **User**: non-root `appuser`
- **Healthcheck**: `curl -f http://localhost:8000/health` every 30s
- **Port**: 8000

## Troubleshooting

**Models not found**  
Ensure `models/model.pkl` and `models/preprocessor.pkl` exist before building.

**Port 8000 in use**  
Stop local API or use another port: `docker run -p 8001:8000 ...`

**Permission errors**  
Build without cache: `docker build --no-cache -t used-car-price-api:latest .`
