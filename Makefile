.PHONY: help install install-dev lint format format-check test test-cov test-fast train train-force metrics metrics-diff dag status mlflow-ui api-dev api-test docker-build docker-run docker-stop docker-logs docker-test docker-clean compose-up compose-down compose-logs compose-dev compose-dev-down check ci all

# Python interpreter
PYTHON := python
PIP := pip

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help:
	@echo "$(BLUE)Available commands:$(NC)"
	@echo ""
	@echo "$(GREEN)Setup:$(NC)"
	@echo "  make install           Install production dependencies"
	@echo "  make install-dev       Install development dependencies"
	@echo ""
	@echo "$(GREEN)Code Quality:$(NC)"
	@echo "  make lint              Run flake8 linting"
	@echo "  make format            Format code with black and isort"
	@echo "  make format-check      Check formatting without changes"
	@echo ""
	@echo "$(GREEN)Testing:$(NC)"
	@echo "  make test              Run all tests"
	@echo "  make test-cov          Run tests with coverage report"
	@echo "  make test-fast         Run tests without coverage (fastest)"
	@echo ""
	@echo "$(GREEN)ML Pipeline:$(NC)"
	@echo "  make train             Run DVC pipeline"
	@echo "  make train-force       Force rerun entire DVC pipeline"
	@echo "  make metrics           Show current metrics"
	@echo "  make metrics-diff      Compare metrics between commits"
	@echo "  make dag               Show DVC pipeline DAG"
	@echo "  make status            Show DVC pipeline status"
	@echo "  make mlflow-ui         Launch MLflow UI"
	@echo ""
	@echo "$(GREEN)API Development:$(NC)"
	@echo "  make api-dev           Run FastAPI development server"
	@echo "  make api-test          Test API with sample request"
	@echo ""
	@echo "$(GREEN)Docker:$(NC)"
	@echo "  make docker-build      Build Docker image"
	@echo "  make docker-run        Run Docker container"
	@echo "  make docker-stop       Stop Docker container"
	@echo "  make docker-logs       View Docker logs"
	@echo "  make docker-test       Test API in Docker"
	@echo "  make compose-up       Start services with docker-compose"
	@echo "  make compose-down     Stop docker-compose services"
	@echo "  make compose-dev       Start development environment (hot-reload)"
	@echo "  make compose-dev-down Stop development environment"
	@echo ""
	@echo "$(GREEN)Cleanup:$(NC)"
	@echo "  make clean             Remove cache and temporary files"
	@echo "  make clean-data        Remove DVC pipeline outputs"
	@echo "  make clean-models      Remove trained models"
	@echo ""
	@echo "$(GREEN)Quality Gates:$(NC)"
	@echo "  make check             Run lint + test (pre-commit check)"
	@echo "  make ci                Run full CI pipeline (format-check + lint + test)"
	@echo ""
	@echo "$(GREEN)Workflows:$(NC)"
	@echo "  make all               Run complete workflow (install, test, train)"
	@echo ""

# Setup targets
install:
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Installation complete$(NC)"

install-dev:
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Development installation complete$(NC)"

# Code Quality targets
lint:
	@echo "$(BLUE)Running flake8 linter...$(NC)"
	$(PYTHON) -m flake8 src tests
	@echo "$(GREEN)✓ Linting complete - no issues found$(NC)"

format:
	@echo "$(BLUE)Formatting code with black and isort...$(NC)"
	$(PYTHON) -m isort src tests
	$(PYTHON) -m black src tests --line-length 120
	@echo "$(GREEN)✓ Code formatting complete$(NC)"

format-check:
	@echo "$(BLUE)Checking code formatting...$(NC)"
	$(PYTHON) -m isort --check-only src tests
	$(PYTHON) -m black --check src tests --line-length 120
	@echo "$(GREEN)✓ Code formatting check passed$(NC)"

# Testing targets
test:
	@echo "$(BLUE)Running all tests...$(NC)"
	$(PYTHON) -m pytest tests -v --tb=short
	@echo "$(GREEN)✓ All tests passed$(NC)"

test-cov:
	@echo "$(BLUE)Running tests with coverage report...$(NC)"
	$(PYTHON) -m pytest tests -v --cov=used-car-dynamic-pricing --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)✓ Tests complete - coverage report generated (htmlcov/index.html)$(NC)"

test-fast:
	@echo "$(BLUE)Running tests quickly...$(NC)"
	$(PYTHON) -m pytest tests -q
	@echo "$(GREEN)✓ Tests passed$(NC)"

# ML Pipeline targets
train:
	@echo "$(BLUE)Running DVC pipeline...$(NC)"
	dvc repro
	@echo "$(GREEN)✓ Pipeline execution complete$(NC)"

train-force:
	@echo "$(BLUE)Force rerunning DVC pipeline...$(NC)"
	dvc repro --force
	@echo "$(GREEN)✓ Pipeline force execution complete$(NC)"

metrics:
	@echo "$(BLUE)Current metrics:$(NC)"
	dvc metrics show
	@echo ""

metrics-diff:
	@echo "$(BLUE)Metrics comparison:$(NC)"
	dvc metrics diff
	@echo ""

dag:
	@echo "$(BLUE)DVC Pipeline DAG:$(NC)"
	dvc dag
	@echo ""

status:
	@echo "$(BLUE)DVC Pipeline Status:$(NC)"
	dvc status
	@echo ""

mlflow-ui:
	@echo "$(BLUE)Launching MLflow UI...$(NC)"
	mlflow ui
	@echo "$(GREEN)✓ MLflow UI running at http://localhost:5000$(NC)"

# API Development targets
api-dev:
	@echo "$(BLUE)Starting FastAPI development server...$(NC)"
	@echo "$(GREEN)📊 API docs: http://localhost:8000/docs$(NC)"
	@echo "$(GREEN)📋 Health: http://localhost:8000/health$(NC)"
	cd src/app && $(PYTHON) -m uvicorn api:app --reload --host 0.0.0.0 --port 8000

api-test:
	@echo "$(BLUE)Testing API...$(NC)"
	curl -X POST "http://localhost:8000/predict" \
	  -H "Content-Type: application/json" \
	  -d "{\"name\":\"Toyota Corolla\",\"year\":2020,\"miles\":\"45,000 miles\",\"color\":\"Black\",\"condition\":\"No accidents reported, 1 Owner\"}"

# Docker targets
docker-build:
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t used-car-price-api:latest .
	@echo "$(GREEN)✓ Image built successfully$(NC)"

docker-run:
	@echo "$(BLUE)Starting Docker container...$(NC)"
	docker run -d \
	  --name car-price-api \
	  -p 8000:8000 \
	  --restart unless-stopped \
	  -v $(PWD)/logs:/app/logs \
	  used-car-price-api:latest
	@echo "$(GREEN)✓ Container started$(NC)"
	@echo "$(GREEN)📊 API: http://localhost:8000$(NC)"
	@echo "$(GREEN)📚 Docs: http://localhost:8000/docs$(NC)"

docker-stop:
	@echo "$(BLUE)Stopping Docker container...$(NC)"
	-docker stop car-price-api
	-docker rm car-price-api
	@echo "$(GREEN)✓ Container stopped$(NC)"

docker-logs:
	docker logs -f car-price-api

docker-shell:
	docker exec -it car-price-api bash

docker-test:
	@echo "$(BLUE)Testing API in Docker...$(NC)"
	curl -X POST "http://localhost:8000/predict" \
	  -H "Content-Type: application/json" \
	  -d '{"name":"Toyota Corolla","year":2020,"miles":"45,000 miles","color":"Black","condition":"No accidents reported, 1 Owner"}'

docker-clean:
	@echo "$(BLUE)Cleaning Docker resources...$(NC)"
	-docker-compose down -v
	docker system prune -f
	@echo "$(GREEN)✓ Cleaned$(NC)"

# Docker Compose targets
compose-up:
	@echo "$(BLUE)Starting services...$(NC)"
	docker-compose up -d --build
	@echo "$(GREEN)✓ Services started$(NC)"
	@echo "$(GREEN)📊 API: http://localhost:8000$(NC)"

compose-down:
	@echo "$(BLUE)Stopping services...$(NC)"
	docker-compose down
	@echo "$(GREEN)✓ Services stopped$(NC)"

compose-logs:
	docker-compose logs -f api

compose-dev:
	@echo "$(BLUE)Starting development environment...$(NC)"
	docker-compose -f docker-compose.dev.yaml up -d --build
	@echo "$(GREEN)✓ Development environment started$(NC)"
	@echo "$(GREEN)📊 API: http://localhost:8000 (hot-reload enabled)$(NC)"

compose-dev-down:
	docker-compose -f docker-compose.dev.yaml down
	@echo "$(GREEN)✓ Development environment stopped$(NC)"

# Cleanup targets
clean:
	@echo "$(BLUE)Cleaning cache and temporary files...$(NC)"
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .coverage -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .eggs -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

clean-data:
	@echo "$(BLUE)Removing DVC pipeline outputs...$(NC)"
	rm -rf data/processed
	rm -rf models/*.pkl
	dvc gc
	@echo "$(GREEN)✓ Data cleanup complete$(NC)"

clean-models:
	@echo "$(BLUE)Removing trained models...$(NC)"
	rm -rf models/*.pkl
	rm -rf mlruns
	@echo "$(GREEN)✓ Models cleanup complete$(NC)"

# Quality gates
check: lint test
	@echo "$(GREEN)✓ All quality checks passed$(NC)"

ci: format-check lint test
	@echo "$(GREEN)✓ CI pipeline passed - ready to merge$(NC)"

# Workflow targets
all: clean install test train
	@echo "$(GREEN)✓ Complete workflow finished successfully$(NC)"
	@echo ""
	@echo "$(BLUE)Next steps:$(NC)"
	@echo "  - Review metrics: make metrics"
	@echo "  - Launch MLflow: make mlflow-ui"
	@echo "  - View coverage: open htmlcov/index.html"
	@echo ""

#—————————————————————————————————————————————————————————————————————————————
# FRONTEND COMMANDS
#—————————————————————————————————————————————————————————————————————————————

frontend-dev:  ## Run Streamlit frontend locally
	@echo "🎨 Starting Streamlit frontend..."
	@echo "🌐 Frontend: http://localhost:8501"
	cd src/frontend && API_URL=http://localhost:8000 streamlit run app.py

#—————————————————————————————————————————————————————————————————————————————
# FULL STACK COMMANDS
#—————————————————————————————————————————————————————————————————————————————

stack-up:  ## Start full stack (API + Frontend)
	@echo "🚀 Starting full stack..."
	docker-compose up -d --build
	@echo "✅ Full stack started!"
	@echo ""
	@echo "📊 Services:"
	@echo "   - API:      http://localhost:8000"
	@echo "   - API Docs: http://localhost:8000/docs"
	@echo "   - Frontend: http://localhost:8501"
	@echo ""
	@echo "View logs: make stack-logs"

stack-down:  ## Stop full stack
	@echo "🛑 Stopping full stack..."
	docker-compose down
	@echo "✅ Full stack stopped"

stack-logs:  ## View full stack logs
	docker-compose logs -f

stack-restart:  ## Restart full stack
	@echo "🔄 Restarting full stack..."
	docker-compose restart
	@echo "✅ Full stack restarted"

stack-rebuild:  ## Rebuild and restart full stack
	@echo "🔨 Rebuilding full stack..."
	docker-compose down
	docker-compose up -d --build
	@echo "✅ Full stack rebuilt and started"

stack-status:  ## Show status of all services
	@echo "📊 Service Status:"
	docker-compose ps


.DEFAULT_GOAL := help
