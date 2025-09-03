.PHONY: help install test lint format clean run-experiments run-train

help: ## Show this help message
	@echo "ZSharp: Sharpness-Aware Minimization with Z-Score Gradient Filtering"
	@echo "=================================================================="
	@echo ""
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies
	pip install -r requirements.txt

test: ## Run tests with coverage
	python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing --cov-fail-under=95

test-fast: ## Run tests without coverage
	python -m pytest tests/ -v

lint: ## Run linting checks
	black --check --line-length=79 src/ tests/ scripts/
	flake8 src/ tests/ scripts/
	mypy src/ tests/ scripts/
	interrogate src/ --fail-under=100

format: ## Format code with black
	black --line-length=79 src/ tests/ scripts/

clean: ## Clean up generated files
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

run-experiments: ## Run comprehensive experiments
	python -m scripts.experiment

run-experiments-fast: ## Run experiments in fast mode
	python -m scripts.experiment --fast

run-hp-study: ## Run hyperparameter study
	python -m scripts.experiment --hp-study

run-train: ## Run single training experiment
	python -m scripts.train --config configs/zsharp_baseline.yaml

run-train-sgd: ## Run SGD baseline training
	python -m scripts.train --config configs/sgd_baseline.yaml

run-train-quick: ## Run quick training experiment
	python -m scripts.train --config configs/zsharp_quick.yaml

setup-dev: install ## Setup development environment
	pip install pre-commit
	pre-commit install

check-all: lint test ## Run all checks

# Development shortcuts
dev: setup-dev ## Setup development environment
	@echo "Development environment ready!"

quick-test: test-fast ## Quick test run
	@echo "Tests completed!"

demo: run-experiments-fast ## Run quick demo
	@echo "Demo completed!"
