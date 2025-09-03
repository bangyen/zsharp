# ZSharp API Documentation

## Core Modules

### `src.optimizer`
The main ZSharp optimizer implementation.

**Classes:**
- `ZSharp`: Sharpness-Aware Minimization with Z-Score Gradient Filtering

**Key Methods:**
- `first_step()`: Apply gradient filtering and SAM perturbation
- `second_step()`: Update parameters with filtered gradients
- `step()`: Combined first and second step for convenience

**Parameters:**
- `params`: Model parameters
- `base_optimizer`: Base optimizer class (e.g., torch.optim.SGD)
- `rho`: SAM perturbation radius (default: 0.05)
- `percentile`: Gradient filtering threshold (default: 70)
- `lr`: Learning rate (default: 0.01)
- `momentum`: Momentum coefficient (default: 0.9)
- `weight_decay`: Weight decay (default: 1e-4)

### `src.train`
Training utilities and main training loop.

**Functions:**
- `train(config)`: Main training function
- `validate(model, dataloader, criterion, device)`: Validation loop

**Returns:**
- Dictionary containing training metrics, losses, and final results

### `src.data`
Data loading and preprocessing utilities.

**Functions:**
- `get_dataloaders(dataset_name, batch_size, num_workers, pin_memory)`: Get train/val dataloaders
- `get_dataset(dataset_name, train=True)`: Get dataset with transforms

**Supported Datasets:**
- `cifar10`: CIFAR-10 dataset
- `cifar100`: CIFAR-100 dataset

### `src.models`
Model definitions and architectures.

**Functions:**
- `get_model(model_name, num_classes)`: Get model by name
- `get_num_parameters(model)`: Count model parameters

**Supported Models:**
- `resnet18`: ResNet-18 architecture
- `vgg16`: VGG-16 architecture
- `vit`: Vision Transformer

### `src.eval`
Evaluation and metrics computation.

**Functions:**
- `compute_accuracy(outputs, targets)`: Compute classification accuracy
- `compute_metrics(model, dataloader, criterion, device)`: Compute comprehensive metrics

### `src.experiments`
Experiment management and execution.

**Functions:**
- `run_comparison_experiments()`: Run ZSharp vs SGD comparison
- `run_hyperparameter_study()`: Run hyperparameter optimization
- `save_results(results, filename)`: Save experiment results

### `src.constants`
Project constants and default values.

**Key Constants:**
- `DEFAULT_LEARNING_RATE`: 0.01
- `DEFAULT_MOMENTUM`: 0.9
- `DEFAULT_RHO`: 0.05
- `DEFAULT_PERCENTILE`: 70
- `RESULTS_DIR`: "results"
- `DEFAULT_BATCH_SIZE`: 32
- `DEFAULT_EPOCHS`: 20

### `src.utils`
Utility functions for logging and file operations.

**Functions:**
- `setup_logging(level)`: Configure logging
- `ensure_dir(directory)`: Create directory if it doesn't exist
- `get_device()`: Get available device (CPU/MPS/CUDA)

## Configuration

Configuration files are stored in `configs/` and follow YAML format:

```yaml
dataset: cifar10
model: resnet18
optimizer:
  type: zsharp
  rho: 0.05
  percentile: 70
  lr: 0.01
  momentum: 0.9
  weight_decay: 1e-4
train:
  batch_size: 32
  epochs: 20
  device: mps
  num_workers: 0
```

## Usage Examples

### Basic Training

```python
from src.train import train
import yaml

# Load configuration
with open('configs/zsharp_baseline.yaml') as f:
    config = yaml.safe_load(f)

# Train model
results = train(config)
print(f"Final accuracy: {results['final_test_accuracy']:.2f}%")
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
    model.parameters(),
    base_optimizer=torch.optim.SGD,
    rho=0.05,
    percentile=70,
    lr=0.01,
    momentum=0.9
)

# Training loop
for epoch in range(num_epochs):
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

```python
from src.experiments import run_comparison_experiments

# Run comparison experiments
results = run_comparison_experiments()
print("Experiment results:", results)
```

### Command Line Experiments

```bash
# Run comprehensive experiments
python -m scripts.experiment

# Run hyperparameter study
python -m scripts.experiment --hp-study

# Fast mode for testing
python -m scripts.experiment --fast
```

## Error Handling

The API includes comprehensive error handling:

- **Invalid configuration**: Raises `ValueError` with descriptive message
- **Missing datasets**: Automatically downloads CIFAR datasets
- **Device issues**: Falls back to CPU if MPS/CUDA unavailable
- **Memory issues**: Provides helpful error messages for OOM scenarios

## Performance Notes

- **Apple Silicon**: Optimized for MPS with 4.39x speedup over CPU
- **Memory usage**: Efficient gradient filtering reduces memory footprint
- **Convergence**: ZSharp typically converges in fewer epochs than SGD
- **Stability**: Robust to hyperparameter variations
