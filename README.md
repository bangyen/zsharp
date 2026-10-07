# ZSharp

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bangyen/zsharp/blob/main/zsharp_demo.ipynb)
[![CI](https://github.com/bangyen/zsharp/actions/workflows/ci.yml/badge.svg)](https://github.com/bangyen/zsharp/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/bangyen/zsharp)](LICENSE)

**Sharpness-Aware Minimization with Z-Score Gradient Filtering: +0.87% accuracy over SGD at 200 epochs (+2.65% at 20), Apple Silicon optimized, fully reproducible**

<p align="center">
  <img src="docs/training_curves.png" alt="Training curves comparison" width="600">
</p>

## Quickstart

Clone the repo and run the demo:

```bash
git clone https://github.com/bangyen/zsharp.git
cd zsharp
pip install -e .
pytest   # optional: run tests
python -m scripts.train --config configs/zsharp_baseline.yaml
```

Or open in Colab: [Colab Notebook](https://colab.research.google.com/github/bangyen/zsharp/blob/main/zsharp_demo.ipynb).

## Results

| Scenario / Dataset | Epochs | Baseline (SGD) | This Project | Δ Improvement |
|--------------------|--------|----------------|--------------|---------------|
| CIFAR-10 ResNet-18 | 20     | 78.19 ± 0.23%  | **80.84 ± 0.54%** | +2.65%   |
| CIFAR-10 ResNet-18 | 200    | 84.22%         | **85.09%**   | +0.87%        |

*Final test accuracy. 20 epochs: mean ± std over 3 seeds (42, 1, 2).
200 epochs: single seed (42). Baseline: `configs/sgd_baseline.yaml`. ZSharp:
`configs/zsharp_baseline.yaml` (paper hyperparameters: AdamW, batch size 256,
$Q_p = 0.95$), with `epochs` set as shown. Measured on CPU.*

> **Note**: the gap shrinks with training — ZSharp reaches its plateau
> sooner, but both end within about a point. Each ZSharp step costs two
> forward/backward passes; the 200-epoch run took ~4.3x the wall-clock time
> of SGD (13.2 h vs 3.1 h).

## Features

- **Z-Score Gradient Filtering** — Layer-wise Z-score normalization with a global 95th percentile threshold (configurable), matching the paper's $Q_p = 0.95$.
- **Apple Silicon Optimization** — Up to 4.39x speedup using MPS (Metal Performance Shaders) for faster training on Mac.
- **Paper Architectures** — CIFAR-style ResNet-56/110, VGG-16BN, and the paper's compact ViTs, on CIFAR-10/100 and Tiny-ImageNet.
- **Comprehensive Testing** — 95%+ test coverage with 85 unit tests ensuring reliability and reproducibility.

## Repo Structure

```plaintext
zsharp/
├── zsharp_demo.ipynb  # Colab notebook demo
├── scripts/           # Training and experiment scripts
├── tests/             # Unit/integration tests (85 tests)
├── docs/              # Documentation and training curves
├── configs/           # Configuration files
├── results/           # Experimental results
└── zsharp/               # Core implementation
```

## Validation

- ✅ 95%+ test coverage (`pytest`)
- ✅ Reproducible seeds for experiments
- ✅ Benchmark scripts included

## References

- [Sharpness-Aware Minimization with Z-Score Gradient Filtering](https://arxiv.org/html/2505.02369v3) — Original research paper by Juyoung Yun. The optimizer and default hyperparameters follow this paper: $Q_p = 0.95$, $\rho = 0.05$, AdamW base optimizer (lr 1e-3, weight decay 5e-5), and an LR step decay of 0.75 every 10 epochs.
- [Sharpness-Aware Minimization](https://arxiv.org/abs/2010.01412) — Foundation SAM algorithm research.

## License

This project is licensed under the [MIT License](LICENSE).
