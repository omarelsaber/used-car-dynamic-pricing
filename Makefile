.PHONY: help install install-dev lint format format-check test test-cov test-fast train train-force metrics metrics-diff dag status mlflow-ui clean clean-data clean-models check ci all

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

.DEFAULT_GOAL := help
