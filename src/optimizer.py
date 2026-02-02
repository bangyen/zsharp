"""Optimizer implementations for SAM and ZSharp.

This module provides implementations of SAM (Sharpness-Aware Minimization)
and ZSharp optimizers for deep learning training with gradient filtering.
"""
from __future__ import annotations

from typing import Callable, Union, cast, overload

import torch
import torch.nn
import torch.optim
from torch.optim import Optimizer

from src.constants import (
    DEFAULT_PERCENTILE,
    DEFAULT_RHO,
    DEFAULT_TOP_K_RATIO,
    EPSILON,
    EPSILON_STD,
    PERCENTAGE_MULTIPLIER,
)

# Type for optimizer kwargs
OptimizerKwargs = Union[float, int, bool]
"""Type alias for optimizer keyword arguments."""

# Type for base optimizer constructor
BaseOptimizerConstructor = Callable[..., Optimizer]
"""Type alias for base optimizer constructor functions."""


class SAM(Optimizer):
    """Sharpness-Aware Minimization (SAM) optimizer.

    SAM is a two-step optimizer that first perturbs parameters in the direction
    of the gradient to find a sharp minimum, then updates parameters using the
    base optimizer.

    Args:
        params: Parameters to optimize
        base_optimizer: Base optimizer class (e.g., torch.optim.SGD)
        rho: Perturbation radius for SAM
        **kwargs: Additional arguments passed to base_optimizer

    """

    def __init__(
        self,
        params: list[torch.nn.Parameter],
        base_optimizer: BaseOptimizerConstructor,
        rho: float = DEFAULT_RHO,
        **kwargs: OptimizerKwargs,
    ) -> None:
        """Initialize SAM optimizer.

        Args:
            params: Parameters to optimize
            base_optimizer: Base optimizer class (e.g., torch.optim.SGD)
            rho: Perturbation radius for SAM
            **kwargs: Additional arguments passed to base_optimizer

        """
        defaults = {"rho": rho, **kwargs}
        super().__init__(params, defaults)
        self.base_optimizer: Optimizer = base_optimizer(
            self.param_groups,
            **kwargs,
        )
        self.rho = rho

    @torch.no_grad()
    def first_step(self) -> None:
        """First step of SAM: perturb parameters in gradient direction.

        This step adds a perturbation to each parameter in the direction of
        its gradient, scaled by the perturbation radius rho.
        """
        grad_norm = torch.norm(
            torch.stack(
                [
                    p.grad.norm(p=2)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ],
            ),
            p=2,
        )
        scale = self.rho / (grad_norm + EPSILON)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e = p.grad * scale
                p.add_(e)
                if not hasattr(p, "state"):
                    p.state = {}
                p.state["e"] = e

    @torch.no_grad()
    def second_step(self) -> None:
        """Second step of SAM: remove perturbation and update parameters.

        This step removes the perturbation added in first_step and then
        updates parameters using the base optimizer.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if hasattr(p, "state") and "e" in p.state:
                    p.sub_(p.state["e"])
        self.base_optimizer.step()

    @overload
    def step(self, closure: None = None) -> None:
        ...

    @overload
    def step(self, closure: Callable[[], float]) -> float:
        ...

    def step(
        self,
        closure: Callable[[], float] | None = None,
    ) -> float | None:
        """Raise an error since SAM requires two-step calls.

        SAM must be used with explicit first_step() and second_step() calls
        rather than the standard step() method.
        """
        _ = closure
        msg = "SAM requires two-step calls: first_step and second_step"
        raise RuntimeError(msg)


class ZSharp(SAM):
    """ZSharp: Sharpness-Aware Minimization with Z-Score Gradient Filtering.

    ZSharp extends SAM by applying layer-wise Z-score normalization and
    percentile-based gradient filtering before the SAM perturbation step.
    This helps focus on the most important gradients and improves training
    stability.

    Args:
        params: Parameters to optimize
        base_optimizer: Base optimizer class (e.g., torch.optim.SGD)
        rho: Perturbation radius for SAM
        percentile: Percentile threshold for gradient filtering (0-100)
        **kwargs: Additional arguments passed to base_optimizer

    """

    def __init__(
        self,
        params: list[torch.nn.Parameter],
        base_optimizer: BaseOptimizerConstructor,
        rho: float = 0.05,
        percentile: int = DEFAULT_PERCENTILE,
        **kwargs: OptimizerKwargs,
    ) -> None:
        """Initialize ZSharp optimizer.

        Args:
            params: Parameters to optimize
            base_optimizer: Base optimizer class (e.g., torch.optim.SGD)
            rho: Perturbation radius for SAM
            percentile: Percentile threshold for gradient filtering (0-100)
            **kwargs: Additional arguments passed to base_optimizer

        """
        super().__init__(params, base_optimizer, rho=rho, **kwargs)
        self.percentile = percentile

    @torch.no_grad()
    def first_step(self) -> None:
        """First step of ZSharp: apply gradient filtering and SAM perturbation.

        This step:
        1. Computes layer-wise Z-scores for all gradients
        2. Applies percentile-based filtering to keep only important gradients
        3. Applies SAM perturbation with filtered gradients
        """
        # Collect gradients by layer/parameter group for layer-wise Z-score
        # computation
        info_type = tuple[torch.nn.Parameter, torch.Tensor, int, int]
        layer_grad_info: list[info_type] = []
        layer_grads: list[torch.Tensor] = []
        # Store (param, grad, start_idx, end_idx) for each layer

        start_idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad_flat = p.grad.detach().flatten()

                # Handle mixed precision
                if grad_flat.dtype == torch.float16:
                    grad_flat = grad_flat.float()

                layer_grads.append(grad_flat)
                end_idx = start_idx + grad_flat.numel()
                layer_grad_info.append((p, p.grad, start_idx, end_idx))
                start_idx = end_idx

        if not layer_grads:
            return

        # Compute Z-scores layer-wise as described in the paper
        zscores_list: list[torch.Tensor] = []
        for grad_flat in layer_grads:
            # Layer-wise Z-score normalization with numerical stability
            layer_mean, layer_std = (
                torch.mean(grad_flat),
                torch.std(grad_flat) + EPSILON_STD,
            )
            layer_zscores = (grad_flat - layer_mean) / layer_std
            zscores_list.append(layer_zscores)

        # Concatenate Z-scores for percentile computation
        all_zscores = torch.cat(zscores_list)

        # Use absolute Z-scores for percentile computation as per paper
        threshold = torch.quantile(
            all_zscores.abs(),
            self.percentile / PERCENTAGE_MULTIPLIER,
        ).item()

        # Apply filtering to each layer
        for i, (p, original_grad, _, _) in enumerate(layer_grad_info):
            # Get Z-scores for this layer
            layer_zscores = zscores_list[i]

            # Create mask based on absolute Z-score threshold
            mask = layer_zscores.abs() >= threshold

            # Ensure at least some gradients are kept (numerical stability)
            if not mask.any():
                # Keep top 20% if no gradients pass threshold
                top_k = max(1, int(DEFAULT_TOP_K_RATIO * mask.numel()))
                _, indices = torch.topk(layer_zscores.abs(), top_k)
                mask = torch.zeros_like(mask)
                mask[indices] = True

            # Reshape mask to match original gradient shape
            mask = mask.view_as(original_grad)

            # Apply masking to gradients
            # Note: p.grad is guaranteed to be not None here because we only
            # process parameters with gradients in the collection phase above
            p.grad = cast("torch.Tensor", p.grad) * mask

        # Apply SAM perturbation with filtered gradients
        grad_norm = torch.norm(
            torch.stack(
                [
                    p.grad.norm(p=2)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ],
            ),
            p=2,
        )

        # Add numerical stability to gradient scaling
        scale = self.rho / grad_norm

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue  # Skip parameters without gradients
                e = p.grad * scale
                p.add_(e)
                if not hasattr(p, "state"):
                    p.state = {}
                p.state["e"] = e
