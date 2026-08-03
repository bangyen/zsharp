# Scripts Directory

This directory contains executable scripts for training and running experiments with the ZSharp algorithm. For detailed usage instructions, see the [main README](../README.md) and [API documentation](../docs/api.md).

## Available Scripts

### `train.py`
Individual training script for running single experiments.

**Usage:**
```bash
python -m scripts.train --config configs/zsharp_baseline.yaml
python -m scripts.train --config configs/sgd_baseline.yaml --verbose
```

**Features:**
- Loads configuration from YAML files
- Supports both ZSharp and SGD optimizers
- Provides verbose output option
- Saves results automatically

### `experiment.py`
Comprehensive experiment runner for paper reproduction.

**Usage:**
```bash
python -m scripts.experiment                    # Run comparison experiments
python -m scripts.experiment --fast            # Run in fast mode
python -m scripts.experiment --hp-study       # Run hyperparameter study
```

**Features:**
- Runs multiple experiments for comparison
- Supports hyperparameter optimization
- Fast mode for testing
- Comprehensive logging and result saving

## Task Runner Integration

All scripts are integrated with the project's task runner (`just`):

```bash
just run-train        # Run ZSharp training (configs/zsharp_baseline.yaml)
just run-experiments  # Run comprehensive experiments
```

Run `just` (or `just --list`) to see the full set of targets.

## Configuration Files

Scripts use configuration files from the `configs/` directory:
- `zsharp_baseline.yaml` - Standard ZSharp configuration
- `sgd_baseline.yaml` - SGD baseline for comparison
- `zsharp_quick.yaml` - Fast ZSharp training
- `sgd_quick.yaml` - Fast SGD training
