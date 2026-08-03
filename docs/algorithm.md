# ZSharp Algorithm

## Overview

ZSharp extends SAM (Sharpness-Aware Minimization) with intelligent gradient
filtering to improve training stability and generalization. The algorithm
combines the benefits of SAM's sharpness-aware optimization with selective
gradient filtering based on layer-wise Z-score normalization.

## Core Algorithm

### Two-Step Optimization

```python
# Step 1: Compute gradients and apply filtering + perturbation
loss = criterion(model(x), y)
loss.backward()
optimizer.first_step()  # Gradient filtering + SAM perturbation

# Step 2: Recompute gradients and update parameters
criterion(model(x), y).backward()
optimizer.second_step()  # Parameter update
```

### Gradient Filtering Process

1. **Z-score Normalization**: Normalize gradients within each layer
2. **Global Threshold**: Compute a single threshold over the absolute Z-scores
   of all layers combined
3. **Masking**: Zero out gradients whose absolute Z-score falls below the
   threshold
4. **SAM Perturbation**: Apply filtered gradients to SAM's perturbation

### Mathematical Formulation

For each layer $l$ with gradients $g_l$:

1. **Z-score computation** (per layer):
   $$z_l = \frac{g_l - \mu_l}{\sigma_l + \epsilon}$$
   where $\mu_l$ and $\sigma_l$ are the mean and standard deviation of
   gradients in layer $l$ and $\epsilon$ is a small stability constant
   (layers with fewer than 2 elements are skipped).

2. **Global filtering threshold**:
   $$t = \text{quantile}\left(\bigcup_l |z_l|,\; p\right)$$
   where $p$ is the percentile (default: 70). The threshold is computed over
   the absolute Z-scores of **all layers concatenated**, not per layer.

3. **Masking**:
   $$g_l^{filtered} = g_l \odot \mathbb{I}[|z_l| \geq t]$$
   If no component in a layer passes the threshold, the top
   $\lceil 0.2 \cdot \text{numel}(g_l) \rceil$ components are kept so the
   layer is never fully zeroed.

4. **SAM perturbation**:
   $$\epsilon = \rho \frac{g^{filtered}}{\|g^{filtered}\|_2}$$
   where $\rho$ is the perturbation radius.

5. **Parameter update**:
   $$\theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t + \epsilon)$$
   where $\alpha$ is the learning rate.

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rho` | 0.05 | SAM perturbation radius |
| `percentile` | 70 | Global filtering threshold (%) |
| `lr` | 0.01 | Learning rate |
| `momentum` | 0.9 | Momentum coefficient |
| `weight_decay` | 5e-4 | Weight decay |

## Key Benefits

1. **Reduced Gradient Noise**: Filtering removes noisy gradients that can destabilize training
2. **Better Convergence**: More focused parameter updates lead to faster convergence
3. **Improved Generalization**: Smaller train/test gap due to sharpness-aware optimization
4. **Training Stability**: Less sensitive to hyperparameter choices and learning rate

## Algorithm Comparison

### ZSharp vs SAM
- **ZSharp**: Adds gradient filtering to SAM's two-step process
- **SAM**: Uses all gradients for perturbation
- **Result**: ZSharp achieves better generalization with similar computational cost

### ZSharp vs SGD
- **ZSharp**: Sharpness-aware optimization with gradient filtering
- **SGD**: Standard gradient descent
- **Result**: ZSharp shows +5.26% improvement in test accuracy (CIFAR-10)

## Implementation Details

### Gradient Filtering Implementation

The reference implementation lives in `src/optimizer.py`
(`ZSharp.first_step`). The filtering logic:

```python
def _compute_layer_zscores(layer_grads):
    zscores = []
    for grad in layer_grads:
        if grad.numel() < MIN_NUM_FOR_STD:
            zscores.append(torch.zeros_like(grad))
        else:
            mean = grad.mean()
            std = grad.std() + EPSILON_STD
            zscores.append((grad - mean) / std)
    return zscores

def _compute_filtering_threshold(zscores_list, percentile):
    all_zscores = torch.cat(zscores_list)
    return torch.quantile(all_zscores.abs(), percentile / 100)
```

### SAM Perturbation

```python
grad_norm = torch.norm(stacked_gradients)
scale = rho / (grad_norm + EPSILON)
parameters += parameters.grad * scale  # first_step
parameters -= state["e"]               # second_step (after re-backward)
```

## Experimental Results

### Performance Metrics

| Metric | SGD | ZSharp | Improvement |
|--------|-----|--------|-------------|
| Test Accuracy | 74.89% | 80.15% | +5.26% |
| Training Time | Baseline | ~4.39x faster on MPS | Speedup |

### Hyperparameter Sensitivity

ZSharp is robust to hyperparameter variations:

- **Percentile (50-90%)**: Consistent performance across range
- **Rho (0.01-0.1)**: Stable convergence with optimal at 0.05
- **Learning Rate**: Less sensitive than SGD to learning rate choice

## Computational Complexity

- **Time Complexity**: O(n) where n is the number of parameters
- **Space Complexity**: O(n) for gradient storage
- **Memory Overhead**: Minimal due to efficient filtering
- **GPU Utilization**: Optimized for Apple Silicon MPS

## Best Practices

1. **Start with defaults**: Use default hyperparameters for initial experiments
2. **Adjust percentile**: Lower percentile (50-60%) for noisy datasets
3. **Monitor convergence**: ZSharp typically converges in fewer epochs
4. **Use appropriate batch size**: 128 works well for most cases
5. **Enable MPS**: Use Apple Silicon GPU for up to 4.39x speedup
