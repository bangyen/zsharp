# ZSharp

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bangyen/zsharp/blob/main/zsharp_demo.ipynb)
[![CI](https://github.com/bangyen/zsharp/actions/workflows/ci.yml/badge.svg)](https://github.com/bangyen/zsharp/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/bangyen/zsharp)](LICENSE)

**Sharpness-Aware Minimization with Z-Score Gradient Filtering: +0.87% accuracy over SGD at 200 epochs (+2.65% at 20), fully reproducible**

<p align="center">
  <img src="docs/training_curves.png" alt="Test accuracy of ZSharp and SGD per epoch and per wall-clock hour" width="600">
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
| CIFAR-10 ResNet-18 | 50     | 81.99%         | **83.99%**   | +2.00%        |
| CIFAR-10 ResNet-18, strong reg.† | 50 | 80.13% | **83.23%** | +3.10%        |
| CIFAR-10 ResNet-18 | 200    | 84.22%         | **85.09%**   | +0.87%        |

*Final test accuracy. 20 epochs: mean ± std over 3 seeds (42, 1, 2).
50 and 200 epochs: single seed (42); the plain 50-epoch row is epoch 50 of
the 200-epoch runs, which match a 50-epoch run exactly since the LR schedule
does not depend on the total. Baseline: `configs/sgd_baseline.yaml`. ZSharp:
`configs/zsharp_baseline.yaml` (paper hyperparameters: AdamW, batch size 256,
$Q_p = 0.95$), with `epochs` set as shown. Measured on CPU.
† `label_smoothing: 0.1` and `strong_augmentation: true` (TrivialAugmentWide
plus random erasing). Both models underfit at 50 epochs under this
regularization (~70% train accuracy), so it lowers accuracy for both, but
the per-epoch gap widens. ZSharp took ~3.2x SGD's wall-clock time here
(before the faster threshold computation).*

> **Note**: the gap shrinks with training — ZSharp reaches its plateau
> sooner, but both end within about a point. Each ZSharp step costs two
> forward/backward passes plus the Z-score filtering; one ZSharp epoch takes
> ~2.4x as long as an SGD epoch (155 s vs 66 s on 4 CPU threads). At equal
> wall-clock SGD leads for the first ~1.8 h; after that ZSharp is ahead, and
> at SGD's full 200-epoch budget (3.1 h) ZSharp reaches 85.12% (epoch 85)
> vs SGD's 84.22%. See the right panel of the figure above. The 200-epoch
> ZSharp run itself predates the faster threshold computation and took
> 13.2 h; at the current per-epoch cost it would take ~7.3 h.

## Features

- **Z-Score Gradient Filtering** — Layer-wise Z-score normalization with a global 95th percentile threshold (configurable), matching the paper's $Q_p = 0.95$.
- **Device Support** — Runs on CUDA, Apple Silicon (MPS) or CPU, with optional half precision on MPS.
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
