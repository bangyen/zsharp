# ZSharp: Sharpness-Aware Minimization with Z-Score Gradient Filtering

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2505.02369-brightgreen.svg)](https://arxiv.org/html/2505.02369v3)

A PyTorch implementation of **ZSharp: Sharpness-Aware Minimization with Z-Score Gradient Filtering**, optimized for Apple Silicon and featuring comprehensive experimental validation.

## Quickstart

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bangyen/zsharp/blob/main/zsharp_demo.ipynb)

```bash
# Train with ZSharp
python -m scripts.train --config configs/zsharp_baseline.yaml

# Train with baseline SGD for comparison
python -m scripts.train --config configs/sgd_baseline.yaml

# Run comprehensive experiments
python -m scripts.experiment
```

## Performance Results

### Training Curves

![Training Curves](docs/training_curves.png)

### Performance Comparison Table

| Method | Test Accuracy | Test Loss | Training Time | Improvement |
|--------|---------------|-----------|---------------|-------------|
| **SGD** | 74.89% | 0.722 | 2231s | Baseline |
| **ZSharp** | **80.15%** | **0.572** | 4715s | **+5.26%** |

**Key Findings:**
- **ZSharp outperforms SGD by 5.26%** on CIFAR-10 with 20 epochs
- **Better generalization** with lower test loss (0.572 vs 0.722)
- **Stable training** with consistent convergence over 20 epochs
- **Consistent performance** across full training runs
- **Robust optimization** with 70th percentile gradient filtering

## Key Features

- **Paper Reproduction**: Implements the ZSharp algorithm with 5.26% improvement over SGD
- **Apple Silicon Optimized**: 4.39x speedup using MPS (Metal Performance Shaders)
- **Comprehensive Testing**: 100% test coverage with 92 unit tests
- **Experimental Validation**: Multiple datasets (CIFAR-10/100) and architectures (ResNet, VGG, ViT)
- **Production Ready**: Type hints, documentation, and reproducible results
- **Virtual Environment Ready**: Includes pre-configured virtual environment for easy setup

## Table of Contents

- [Quickstart](#quickstart)
- [Performance Results](#performance-results)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Development Commands](#development-commands)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Architecture

### ZSharp Algorithm

ZSharp extends SAM (Sharpness-Aware Minimization) with intelligent gradient filtering:

1. **Layer-wise Z-score Normalization**: Normalizes gradients within each layer
2. **Percentile-based Filtering**: Keeps only the most important gradients (configurable threshold)
3. **SAM Perturbation**: Applies filtered gradients to SAM's two-step optimization

```python
# Example usage
from src.optimizer import ZSharp

optimizer = ZSharp(
    model.parameters(),
    base_optimizer=torch.optim.SGD,
    rho=0.05,           # SAM perturbation radius
    percentile=70,      # Gradient filtering threshold (default)
    lr=0.01,
    momentum=0.9
)

# Two-step training
loss = criterion(model(x), y)
loss.backward()
optimizer.first_step()   # Apply gradient filtering + SAM perturbation
criterion(model(x), y).backward()
optimizer.second_step()  # Update parameters
```

### Training Curves

- **ZSharp**: ~83% training accuracy → 79.96% test accuracy
- **SGD**: ~78% training accuracy → 74.74% test accuracy

## Installation

```bash
# Clone and setup
git clone https://github.com/bangyen/zsharp.git
cd zsharp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev]"

# Verify installation
python -m pytest tests/ -v
```

## Usage

### Basic Training

```python
from src.train import train
import yaml

# Load configuration and train
with open('configs/zsharp_baseline.yaml') as f:
    config = yaml.safe_load(f)

results = train(config)
print(f"Final accuracy: {results['final_test_accuracy']:.2f}%")
```

### Custom Configuration

```yaml
# configs/custom.yaml
dataset: cifar10
model: resnet18
optimizer:
  type: zsharp
  rho: 0.05
  percentile: 70      # Gradient filtering threshold
  lr: 0.01
  momentum: 0.9
  weight_decay: 1e-4
train:
  batch_size: 32
  epochs: 20
  device: mps  # Apple Silicon GPU
  num_workers: 0
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

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test
python -m pytest tests/test_optimizer.py::TestZSharp::test_zsharp_gradient_filtering -v
```

## Project Structure

```
zsharp/
├── src/                    # Core implementation
├── tests/                 # Test suite (92 tests)
├── configs/               # Configuration files
├── results/               # Experimental results
├── data/                  # Dataset storage
├── docs/                  # Documentation
├── scripts/               # Training and experiment scripts
├── pyproject.toml         # Package configuration
└── Makefile              # Development commands
```

## Development Commands

```bash
# Setup development environment
make setup-dev

# Run all quality checks
make lint

# Format code
make format

# Run tests
make test

# Run experiments
make run-experiments
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{zsharp2025,
  title={Sharpness-Aware Minimization with Z-Score Gradient Filtering},
  author={Juyoung Yun},
  journal={arXiv preprint arXiv:2505.02369},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This is a reproduction of the ZSharp paper optimized for Apple Silicon. For the original paper, see [arXiv:2505.02369](https://arxiv.org/html/2505.02369v3).
