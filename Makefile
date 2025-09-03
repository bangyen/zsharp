.PHONY: help install test lint format clean run-experiments run-train

help: ## Show this help message
	@echo "ZSharp: Sharpness-Aware Minimization with Z-Score Gradient Filtering"
	@echo "=================================================================="
	@echo ""
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies
	pip install -e ".[dev]"

install-prod: ## Install production dependencies only
	pip install -e .

test: ## Run tests with coverage
	python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing --cov-fail-under=95

test-fast: ## Run tests without coverage
	python -m pytest tests/ -v

lint: ## Run linting checks
	ruff check src/ tests/ scripts/
	ruff format --check src/ tests/ scripts/
	mypy src/
	interrogate src/ --fail-under=100

format: ## Format code with ruff
	ruff format src/ tests/ scripts/
	ruff check --fix src/ tests/ scripts/

clean: ## Clean up generated files
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

run-experiments: ## Run comprehensive experiments
	python -m scripts.experiment

run-train: ## Run single training experiment
	python -m scripts.train --config configs/zsharp_baseline.yaml

setup-dev: install ## Setup development environment
	pre-commit install

build: ## Build package
	python -m build

publish: build ## Build and publish to PyPI
	twine upload dist/*
