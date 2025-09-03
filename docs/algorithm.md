# ZSharp Algorithm

## Overview

ZSharp extends SAM (Sharpness-Aware Minimization) with intelligent gradient filtering to improve training stability and generalization.

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

### Gradient Filtering

1. **Z-score Normalization**: Normalize gradients within each layer
2. **Percentile-based Filtering**: Keep only top gradients above threshold
3. **SAM Perturbation**: Apply filtered gradients to SAM's perturbation

### Mathematical Formulation

For each layer \(l\) with gradients \(g_l\):

1. **Z-score computation**:
   \[ z_l = \frac{g_l - \mu_l}{\sigma_l} \]

2. **Percentile filtering**:
   \[ g_l^{filtered} = g_l \odot \mathbb{I}[z_l > \text{percentile}(z_l, p)] \]

3. **SAM perturbation**:
   \[ \epsilon = \rho \frac{g_l^{filtered}}{\|g_l^{filtered}\|_2} \]

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rho` | 0.05 | SAM perturbation radius |
| `percentile` | 50 | Gradient filtering threshold (%) |
| `lr` | 0.01 | Learning rate |
| `momentum` | 0.9 | Momentum coefficient |

## Key Benefits

1. **Reduced Gradient Noise**: Filtering removes noisy gradients
2. **Better Convergence**: More focused parameter updates
3. **Improved Generalization**: Smaller train/test gap
4. **Training Stability**: Less sensitive to hyperparameter choices
