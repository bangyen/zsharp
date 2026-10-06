# ZSharp API Documentation

This document provides a comprehensive API reference for the ZSharp project.
For algorithm details, see [algorithm.md](algorithm.md). For quickstart
instructions, see the [main README](../README.md).

## Core Modules

### `zsharp.optimizer`

Optimizer implementations for SAM and ZSharp.

**Classes:**

- `SAM(base_optimizer, rho=0.05, **kwargs)` — Sharpness-Aware Minimization.
  Subclass of `torch.optim.Optimizer`.
- `ZSharp(base_optimizer, rho=0.05, percentile=95, **kwargs)` — SAM with
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
- `base_optimizer`: Base optimizer class. The trainer uses
  `torch.optim.AdamW`, matching the paper.
- `rho`: SAM perturbation radius (default: 0.05)
- `percentile`: Global filtering threshold in percent (default: 95)
- `lr`: Learning rate (default: 0.001)
- `weight_decay`: Weight decay (default: 5e-5)

Any further keyword arguments are forwarded to `base_optimizer`, so
`momentum` is accepted only when the base optimizer is SGD.

### `zsharp.trainer`

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

### `zsharp.data`

Data loading and preprocessing utilities for CIFAR-10, CIFAR-100, and
Tiny-ImageNet.

**Functions:**

- `get_dataset(dataset_name, batch_size=128, num_workers=2, *, pin_memory=False)`:
  Get train/test data loaders by name
- `get_cifar10(batch_size=128, num_workers=2, *, pin_memory=False)`:
  CIFAR-10 data loaders
- `get_cifar100(batch_size=128, num_workers=2, *, pin_memory=False)`:
  CIFAR-100 data loaders

**Classes:**

- `TinyImageNet(root, *, train=True, download=True, transform=None)`:
  Tiny-ImageNet-200 dataset. Not distributed through torchvision, so the
  archive is downloaded and extracted on first use. The validation split
  ships as a flat directory, and its labels are resolved from
  `val_annotations.txt`.

**Data:**

- `DATASET_METADATA`: Registry of normalization statistics, class counts,
  image sizes, and crop padding for each dataset

**Supported Datasets:**

- `cifar10`: CIFAR-10 dataset (10 classes, 32x32)
- `cifar100`: CIFAR-100 dataset (100 classes, 32x32)
- `tiny_imagenet`: Tiny-ImageNet-200 (200 classes, 64x64)

The paper does not state normalization statistics or augmentation for any
dataset. CIFAR uses the conventional per-dataset statistics; Tiny-ImageNet
uses the commonly cited Tiny-ImageNet values. All three apply the same
augmentation as the author's reference implementation: random crop with
padding, horizontal flip, then normalization.

### `zsharp.models`

Model loading utilities.

**Functions:**

- `get_model(model_name="resnet18", num_classes=10, image_size=32) -> nn.Module`:
  Get a PyTorch model by name. `image_size` is used by the paper's ViT
  variants to derive their patch size.

**Classes:**

- `CifarResNet(blocks_per_stage, num_classes=10)`: CIFAR-style ResNet with
  `6n+2` layers (He et al., Sec. 4.2)
- `PaperViT(num_classes=10, image_size=32, *, ...)`: the paper's compact
  Vision Transformer

**Supported Models:**

Architectures used in the ZSharp paper:

- `resnet56`, `resnet110`: CIFAR-style ResNets — three stages of basic
  blocks at 16/32/64 channels with parameter-free (option A) shortcuts.
  These depths exist only in the CIFAR family; torchvision does not ship
  them, so they are implemented here.
- `vgg16_bn`: VGG-16 with batch normalization
- `vit_7_8_8_384`, `vit_7_8_12_768`: compact ViTs with 7 layers, 8 heads,
  and an embedding width of 384

Also available:

- `resnet18`: ResNet-18 architecture
- `vgg11`: VGG-11 architecture
- `vit_b_16`: Vision Transformer B-16

> **On the ViT naming**: the paper writes `ViT-7/8/8-384` and
> `ViT-7/8/12-768`, and its text reads the fields as layers / heads /
> patch size / MLP dimension. That reading is not self-consistent — it
> makes both variants 8-headed with patch size 8, leaving the differing
> third field unexplained, and 12 patches per side does not divide a 32x32
> input. The author's reference implementation fixes patches at 8 per side
> and varies heads (8 and 12) at a constant embedding width of 384, which
> is what is implemented here. The second field is patches *per side*, not
> pixels: on a 32x32 input, 8 per side gives 4x4 pixel patches.

### `zsharp.constants`

Configuration models and default values.

**Pydantic Models:**

- `TrainingConfig`: Overall training configuration
  - `train: TrainingSubConfig`
  - `optimizer: OptimizerConfig`
  - `dataset: str`
  - `model: str`
  - `seed: int` (default 42)
- `TrainingSubConfig`: Training parameters (`device`, `batch_size`, `epochs`,
  `num_workers`, `pin_memory`, `use_mixed_precision`, `checkpoint_dir`).
  When `checkpoint_dir` is set, training state is saved there after every
  epoch as `<dataset>_<model>_<optimizer>_seed<seed>.pt`, and rerunning the
  same config resumes from it (a finished run is only re-evaluated).
- `OptimizerConfig`: Optimizer parameters (`type`, `lr`, `momentum`,
  `weight_decay`, `rho`, `percentile`)
- `ExperimentResults`: Results from an experiment (config, accuracies, losses,
  training time, device, optimizer type)

**Key Constants:**

- `DEFAULT_SEED`: 42
- `DEFAULT_LEARNING_RATE`: 0.001
- `DEFAULT_MOMENTUM`: 0.9 (SGD baseline only)
- `DEFAULT_RHO`: 0.05
- `DEFAULT_PERCENTILE`: 95
- `DEFAULT_WEIGHT_DECAY`: 5e-5
- `DEFAULT_BATCH_SIZE`: 256
- `DEFAULT_LR_STEP_SIZE`: 10
- `DEFAULT_LR_GAMMA`: 0.75
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
  percentile: 95
  lr: 0.001
  momentum: 0.9  # unused by zsharp; AdamW is the base optimizer
  weight_decay: 5e-5
train:
  batch_size: 256
  epochs: 200
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
from zsharp.constants import TrainingConfig
from zsharp.trainer import train

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
from zsharp.optimizer import ZSharp
import torch

# Create ZSharp optimizer
optimizer = ZSharp(
    list(model.parameters()),
    base_optimizer=torch.optim.AdamW,
    rho=0.05,
    percentile=95,
    lr=0.001,
    weight_decay=5e-5,
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
