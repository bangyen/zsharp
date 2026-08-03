# ZSharp API Documentation

This document provides a comprehensive API reference for the ZSharp project.
For algorithm details, see [algorithm.md](algorithm.md). For quickstart
instructions, see the [main README](../README.md).

## Core Modules

### `src.optimizer`

Optimizer implementations for SAM and ZSharp.

**Classes:**

- `SAM(base_optimizer, rho=0.05, **kwargs)` — Sharpness-Aware Minimization.
  Subclass of `torch.optim.Optimizer`.
- `ZSharp(base_optimizer, rho=0.05, percentile=70, **kwargs)` — SAM with
  Z-Score gradient filtering. Subclass of `SAM`.

Both take `params` (an iterable of parameters) as their first positional
argument.

**Key Methods:**

- `first_step()`: Apply gradient filtering (ZSharp) and SAM perturbation
- `second_step()`: Remove the perturbation and update parameters
- `step(closure)`: Combined first/second step; requires a closure that
  re-evaluates the model and returns the loss

**Parameters:**

- `params`: Model parameters
- `base_optimizer`: Base optimizer class (e.g., `torch.optim.SGD`)
- `rho`: SAM perturbation radius (default: 0.05)
- `percentile`: Global filtering threshold in percent (default: 70)
- `lr`: Learning rate (default: 0.01)
- `momentum`: Momentum coefficient (default: 0.9)
- `weight_decay`: Weight decay (default: 5e-4)

### `src.trainer`

Training utilities and the main training loop.

**Functions:**

- `train(config) -> ExperimentResults | None`: Train a model using the given
  `TrainingConfig`. Returns `None` if interrupted.
- `set_seed(seed=42)`: Set random seeds for reproducibility
- `get_device(config) -> torch.device`: Resolve the best available device
  (`cuda`, `mps`, or `cpu`) from a config

**Dataclasses:**

- `TrainingContext`: Encapsulates model, optimizer, criterion, device, and
  flags used during training
- `TrainingHistory`: Accumulated per-epoch metrics and final results

### `src.data`

Data loading and preprocessing utilities for CIFAR-10 and CIFAR-100.

**Functions:**

- `get_dataset(dataset_name, batch_size=128, num_workers=2, *, pin_memory=False)`:
  Get train/test data loaders by name
- `get_cifar10(batch_size=128, num_workers=2, *, pin_memory=False)`:
  CIFAR-10 data loaders
- `get_cifar100(batch_size=128, num_workers=2, *, pin_memory=False)`:
  CIFAR-100 data loaders

**Data:**

- `DATASET_METADATA`: Registry of normalization statistics, class counts,
  image sizes, and crop padding for each dataset

**Supported Datasets:**

- `cifar10`: CIFAR-10 dataset
- `cifar100`: CIFAR-100 dataset

### `src.models`

Model loading utilities.

**Functions:**

- `get_model(model_name="resnet18", num_classes=10) -> nn.Module`: Get a
  PyTorch model by name

**Supported Models:**

- `resnet18`: ResNet-18 architecture
- `vgg11`: VGG-11 architecture
- `vit_b_16`: Vision Transformer B-16

### `src.constants`

Configuration models and default values.

**Pydantic Models:**

- `TrainingConfig`: Overall training configuration
  - `train: TrainingSubConfig`
  - `optimizer: OptimizerConfig`
  - `dataset: str`
  - `model: str`
- `TrainingSubConfig`: Training parameters (`device`, `batch_size`, `epochs`,
  `num_workers`, `pin_memory`, `use_mixed_precision`)
- `OptimizerConfig`: Optimizer parameters (`type`, `lr`, `momentum`,
  `weight_decay`, `rho`, `percentile`)
- `ExperimentResults`: Results from an experiment (config, accuracies, losses,
  training time, device, optimizer type)

**Key Constants:**

- `DEFAULT_SEED`: 42
- `DEFAULT_LEARNING_RATE`: 0.01
- `DEFAULT_MOMENTUM`: 0.9
- `DEFAULT_RHO`: 0.05
- `DEFAULT_PERCENTILE`: 70
- `DEFAULT_WEIGHT_DECAY`: 5e-4
- `DEFAULT_BATCH_SIZE`: 128
- `RESULTS_DIR`: "results"

## Configuration

Configuration files are stored in `configs/` and follow YAML format, validated
against `TrainingConfig`:

```yaml
dataset: cifar10
model: resnet18
optimizer:
  type: zsharp
  rho: 0.05
  percentile: 70
  lr: 0.01
  momentum: 0.9
  weight_decay: 5e-4
train:
  batch_size: 128
  epochs: 20
  device: auto
  num_workers: 4
  pin_memory: false
  use_mixed_precision: false
```

The optimizer `type` accepts `zsharp` or `sgd`; any other value is rejected
at validation time rather than silently defaulting to `zsharp`. Numeric fields
are also range-checked (`percentile` in [0, 100], positive `lr`, `rho`, and
`epochs`, etc.).

## Usage Examples

### Basic Training

```python
from src.constants import TrainingConfig
from src.trainer import train

config = TrainingConfig.model_validate(
    {"dataset": "cifar10", "model": "resnet18"}
)
results = train(config)
print(f"Final test accuracy: {results.final_test_accuracy:.2f}%")
```

### Command Line Training

```bash
# Train with ZSharp
python -m scripts.train --config configs/zsharp_baseline.yaml

# Train with SGD baseline
python -m scripts.train --config configs/sgd_baseline.yaml

# Verbose output
python -m scripts.train --config configs/zsharp_baseline.yaml --verbose
```

### Custom Optimizer

```python
from src.optimizer import ZSharp
import torch

# Create ZSharp optimizer
optimizer = ZSharp(
    list(model.parameters()),
    base_optimizer=torch.optim.SGD,
    rho=0.05,
    percentile=70,
    lr=0.01,
    momentum=0.9,
)

# Training loop
for batch_x, batch_y in dataloader:
    # Forward pass
    outputs = model(batch_x)
    loss = criterion(outputs, batch_y)

    # Backward pass
    loss.backward()
    optimizer.first_step()

    # Second forward-backward pass
    criterion(model(batch_x), batch_y).backward()
    optimizer.second_step()
```

### Running Experiments

```bash
# Run comparison experiments
python -m scripts.experiment

# Run hyperparameter study
python -m scripts.experiment --hp-study

# Fast mode for testing
python -m scripts.experiment --fast
```

Results are saved as JSON under `results/`.
