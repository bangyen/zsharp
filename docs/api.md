# ZSharp API Documentation

## Core Modules

### `src.optimizer`
The main ZSharp optimizer implementation.

**Classes:**
- `ZSharp`: Sharpness-Aware Minimization with Z-Score Gradient Filtering

**Key Methods:**
- `first_step()`: Apply gradient filtering and SAM perturbation
- `second_step()`: Update parameters with filtered gradients

### `src.train`
Training utilities and main training loop.

**Functions:**
- `train(config)`: Main training function
- `validate(model, dataloader, criterion, device)`: Validation loop

### `src.data`
Data loading and preprocessing utilities.

**Functions:**
- `get_dataloaders(dataset_name, batch_size, num_workers, pin_memory)`: Get train/val dataloaders
- `get_dataset(dataset_name, train=True)`: Get dataset with transforms

### `src.models`
Model definitions and architectures.

**Functions:**
- `get_model(model_name, num_classes)`: Get model by name
- `get_num_parameters(model)`: Count model parameters

### `src.eval`
Evaluation and metrics computation.

**Functions:**
- `compute_accuracy(outputs, targets)`: Compute classification accuracy
- `compute_metrics(model, dataloader, criterion, device)`: Compute comprehensive metrics

## Configuration

Configuration files are stored in `configs/` and follow YAML format:

```yaml
dataset: cifar10
model: resnet18
optimizer:
  type: zsharp
  rho: 0.05
  percentile: 50
  lr: 0.01
  momentum: 0.9
train:
  batch_size: 128
  epochs: 20
  device: mps
```

## Constants

Key constants are defined in `src.constants`:

- `DEFAULT_LEARNING_RATE`: 0.01
- `DEFAULT_MOMENTUM`: 0.9
- `DEFAULT_RHO`: 0.05
- `DEFAULT_PERCENTILE`: 50
- `RESULTS_DIR`: "results"
