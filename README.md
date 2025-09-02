# ZSharp Reproduction

This repository reproduces **ZSharp: Gradient-Filtered Sharpness-Aware Minimization** with a simple PyTorch implementation.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py --config configs/default.yaml
```

## Results

* CIFAR-10 with ResNet18, 5 epochs: ~70% accuracy (CPU-friendly demo).

## Citation

```bibtex
@article{zsharp2025,
  title={Sharpness-Aware Minimization with Z-Score Gradient Filtering},
  author={...},
  journal={arXiv preprint arXiv:2505.02369},
  year={2025}
}
```

## Reproducibility checklist
- [x] `requirements.txt`
- [x] Random seeds (set inside `utils.py` if extended)
- [x] Example training commands
- [x] Example run JSON
- [ ] Colab notebook (optional, to be added)
