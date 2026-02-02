# Task runner for ZSharp

# Auto-detect uv - falls back to plain python if not available
PYTHON := `command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python"`

# Show help message
help:
    @echo "ZSharp: Sharpness-Aware Minimization with Z-Score Gradient Filtering"
    @echo "=================================================================="
    @echo ""
    @echo "Available commands:"
    @just --list

# Setup development environment
init:
    #!/usr/bin/env bash
    if command -v uv >/dev/null 2>&1; then
        echo "Using uv..."
        uv sync --extra dev
        uv run pre-commit install
    else
        echo "Using pip..."
        python -m pip install -U pip
        pip install -e ".[dev]"
        pre-commit install
    fi

# Format code with ruff
fmt:
    {{PYTHON}} -m ruff format src/ tests/ scripts/
    {{PYTHON}} -m ruff check --fix src/ tests/ scripts/

# Lint code
lint:
    {{PYTHON}} -m ruff check src/ tests/ scripts/
    {{PYTHON}} -m ruff format --check src/ tests/ scripts/
    {{PYTHON}} -m mypy src/
    {{PYTHON}} -m interrogate src/ --fail-under=100

# Type-check
type:
    {{PYTHON}} -m mypy src/

# Run tests with coverage
test:
    {{PYTHON}} -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing --cov-fail-under=95

# Run tests without coverage
test-fast:
    {{PYTHON}} -m pytest tests/ -v

# Verify dependencies
verify-deps:
    {{PYTHON}} -m deptry .

# Check for dead code
dead-code:
    {{PYTHON}} -m vulture src/

# Verify architectural contracts
arch:
    {{PYTHON}} -c "from importlinter.cli import lint_imports_command; lint_imports_command()"

# Run mutation testing (slow)
mutation:
    {{PYTHON}} -m mutmut run

# Run extended quality checks (dependencies, dead code, architecture)
quality: verify-deps dead-code arch

# Run all checks (fmt, lint, type, test, quality)
all: fmt lint type test quality
    @echo "All checks completed!"

# Clean up generated files
clean:
    #!/usr/bin/env bash
    rm -rf htmlcov/
    rm -rf .pytest_cache/
    rm -rf __pycache__/
    rm -rf build/
    rm -rf dist/
    rm -rf *.egg-info/
    find . -name "*.pyc" -delete
    find . -name "__pycache__" -delete

# Run comprehensive experiments
run-experiments:
    {{PYTHON}} -m scripts.experiment

# Run single training experiment
run-train:
    {{PYTHON}} -m scripts.train --config configs/zsharp_baseline.yaml
