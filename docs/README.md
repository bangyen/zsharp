# ZSharp Documentation

This directory contains comprehensive documentation for the ZSharp project.

## Project Structure

```
zsharp/
├── src/                    # Main source code
│   ├── __init__.py
│   ├── constants.py       # Constants and configuration
│   ├── data.py           # Data loading and preprocessing
│   ├── eval.py           # Evaluation utilities
│   ├── experiments.py    # Experiment management
│   ├── models.py         # Model definitions
│   ├── optimizer.py      # ZSharp optimizer implementation
│   ├── train.py          # Training loop
│   └── utils.py          # Utility functions
├── tests/                 # Unit tests
│   ├── test_optimizer.py # Optimizer tests
│   ├── test_models.py    # Model tests
│   ├── test_data.py      # Data tests
│   ├── test_train.py     # Training tests
│   ├── test_eval.py      # Evaluation tests
│   ├── test_experiments.py # Experiment tests
│   └── test_utils.py     # Utility tests
├── configs/              # Configuration files
│   ├── zsharp_baseline.yaml
│   ├── sgd_baseline.yaml
│   ├── zsharp_quick.yaml
│   ├── sgd_quick.yaml
│   ├── cifar100_zsharp.yaml
│   └── vit_zsharp.yaml
├── data/                  # Dataset storage
│   ├── cifar-10-batches-py/
│   └── cifar-100-python/
├── results/              # Experiment results
│   ├── comparison_results.json
│   ├── hyperparameter_study.json
│   └── individual experiment results
├── run_experiments.py    # Experiment runner
├── scripts/              # Training and experiment scripts
│   ├── train.py         # Individual training script
│   └── experiment.py    # Comprehensive experiment runner
├── docs/                 # Documentation (this directory)
│   ├── README.md         # Project structure (this file)
│   ├── api.md            # API documentation
│   └── algorithm.md      # Algorithm details
├── venv/                 # Virtual environment
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Project configuration
├── Makefile             # Build and development tasks
├── mypy.ini            # Type checking configuration
├── setup.cfg           # Additional project settings
└── README.md            # Main project README
```

## Documentation Overview

### 📚 **Essential Documentation**
- **[README.md](README.md)** - Project structure and overview (this file)
- **[api.md](api.md)** - Complete API reference for all modules
- **[algorithm.md](algorithm.md)** - Detailed ZSharp algorithm explanation

### 🔧 **Development Resources**
- **Main README.md** - Quickstart guide and project overview
- **Makefile** - Available commands and development tasks
- **requirements.txt** - Python dependencies
- **configs/** - Configuration examples for different scenarios

## Key Files

- `README.md` - Main project documentation with quickstart guide
- `requirements.txt` - Python dependencies and versions
- `pyproject.toml` - Project configuration and metadata
- `Makefile` - Build and development tasks
- `mypy.ini` - Type checking configuration
- `setup.cfg` - Additional project settings

## Development Workflow

1. **Setup**: `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
2. **Testing**: `make test` or `pytest tests/`
3. **Training**: `python -m scripts.train --config configs/zsharp_baseline.yaml`
4. **Experiments**: `python -m scripts.experiment`
5. **Code Quality**: `make lint` and `make type-check`
6. **All Checks**: `make check-all`

## Available Commands

```bash
# Development
make install          # Install dependencies
make test            # Run tests with coverage
make test-fast       # Run tests without coverage
make lint            # Run linting checks
make format          # Format code with black
make clean           # Clean up generated files

# Experiments
make run-experiments      # Run comprehensive experiments
make run-experiments-fast # Run experiments in fast mode
make run-hp-study         # Run hyperparameter study

# Training
make run-train            # Run ZSharp training
make run-train-sgd        # Run SGD baseline
make run-train-quick      # Run quick experiment

# Setup
make setup-dev       # Setup development environment
make check-all       # Run all checks (lint + test)
make dev             # Setup development environment
make quick-test      # Quick test run
make demo            # Run quick demo
```

## Configuration Examples

The project includes several pre-configured setups:

- **zsharp_baseline.yaml** - Standard ZSharp configuration
- **sgd_baseline.yaml** - SGD baseline for comparison
- **zsharp_quick.yaml** - Fast ZSharp training
- **sgd_quick.yaml** - Fast SGD training
- **cifar100_zsharp.yaml** - ZSharp on CIFAR-100
- **vit_zsharp.yaml** - ZSharp with Vision Transformer
