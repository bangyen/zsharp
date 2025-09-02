# ZSharp Reproduction

This repository reproduces **ZSharp: Sharpness-Aware Minimization with Z-Score Gradient Filtering** with a faithful PyTorch implementation optimized for Apple Silicon.

## Quickstart

```bash
# Activate virtual environment
source venv/bin/activate

# Train with ZSharp (default)
PYTHONPATH=/Users/bangyen/Documents/repos/zsharp python src/train.py --config configs/default.yaml

# Train with baseline SGD for comparison
PYTHONPATH=/Users/bangyen/Documents/repos/zsharp python src/train.py --config configs/baseline_sgd.yaml

# Run comprehensive experiments
python run_experiments.py
```

## Paper Faithfulness

This implementation is faithful to the [ZSharp paper](https://arxiv.org/html/2505.02369v3) with:

✅ **Core Algorithm**: Layer-wise Z-score normalization and percentile-based gradient filtering  
✅ **Two-step Training**: Proper SAM ascent-descent structure  
✅ **Multiple Datasets**: CIFAR-10, CIFAR-100 support  
✅ **Multiple Architectures**: ResNet, VGG, Vision Transformers  
✅ **Hyperparameter Study**: Percentile threshold analysis  
✅ **Comparison Experiments**: Baseline SGD vs ZSharp  

## Apple Silicon Optimization

This repository is optimized for Apple Silicon MacBooks (M1/M2/M3) with:
- **MPS (Metal Performance Shaders)** for 4.39x speedup over CPU
- **Mixed precision training** for additional performance
- **Optimized data loading** with parallel workers

## Configuration

The default configuration uses ZSharp with recommended hyperparameters:
```yaml
optimizer:
  type: zsharp  # ZSharp optimizer
  rho: 0.05     # Perturbation radius
  percentile: 70 # Gradient filtering threshold
  lr: 0.01
  momentum: 0.9
  weight_decay: 5e-4
train:
  device: mps  # Apple Silicon GPU
  use_mixed_precision: true
```

## Experiments

### Comparison Experiments
- **SGD Baseline**: Standard SGD optimizer
- **ZSharp CIFAR-10**: ZSharp on CIFAR-10 with ResNet18
- **ZSharp CIFAR-100**: ZSharp on CIFAR-100 with ResNet18  
- **ZSharp ViT**: ZSharp with Vision Transformer on CIFAR-10

### Hyperparameter Study
Tests percentile thresholds: [50, 60, 70, 80, 90] to find optimal gradient filtering.

## Results

* CIFAR-10 with ResNet18, 10 epochs: ~72% accuracy with ZSharp
* Training time: ~280s (4.39x faster than CPU)
* Results saved to `results/` directory with detailed metrics

## Citation

```bibtex
@article{zsharp2025,
  title={Sharpness-Aware Minimization with Z-Score Gradient Filtering},
  author={Juyoung Yun},
  journal={arXiv preprint arXiv:2505.02369},
  year={2025}
}
```

## Reproducibility checklist
- [x] `requirements.txt`
- [x] Random seeds (set in `utils.py`)
- [x] Example training commands
- [x] Example run JSON
- [x] Apple Silicon optimization
- [x] Multiple dataset support
- [x] Multiple architecture support
- [x] Hyperparameter study
- [x] Comparison experiments
- [ ] Colab notebook (optional, to be added)
