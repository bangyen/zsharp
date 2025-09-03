# ZSharp Documentation

This directory contains documentation for the ZSharp project.

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
├── configs/              # Configuration files
├── data/                  # Dataset storage
├── results/              # Experiment results
├── run_experiments.py    # Experiment runner
├── docs/                 # Documentation (this directory)
│   ├── README.md         # Project structure (this file)
│   ├── api.md            # API documentation
│   └── algorithm.md      # Algorithm details
├── venv/                 # Virtual environment
└── README.md             # Main project README
```

## Documentation Overview

### 📚 **Essential Documentation**
- **[README.md](README.md)** - Project structure and overview (this file)
- **[api.md](api.md)** - API reference for all modules
- **[algorithm.md](algorithm.md)** - ZSharp algorithm explanation

## Key Files

- `README.md` - Main project documentation
- `requirements.txt` - Python dependencies
- `pyproject.toml` - Project configuration
- `Makefile` - Build and development tasks
- `.pre-commit-config.yaml` - Code quality hooks
- `mypy.ini` - Type checking configuration
- `setup.cfg` - Additional project settings

## Development Workflow

1. **Setup**: `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
2. **Testing**: `make test` or `pytest tests/`
3. **Training**: `python -m src.train --config configs/zsharp_baseline.yaml`
4. **Experiments**: `python run_experiments.py`
5. **Code Quality**: `make lint` and `make type-check`
6. **All Checks**: `make check-all`
