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
# Type for base optimizer constructor
BaseOptimizerConstructor = Callable[..., Optimizer]  # type: ignore[explicit-any]
"""Type alias for base optimizer constructor functions."""


class SAM(Optimizer):  # type: ignore[misc]
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

    def _get_grad_norm(self) -> torch.Tensor:
        """Compute the norm of all gradients."""
        norms = [
            p.grad.norm(p=2)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        return torch.norm(torch.stack(norms), p=2)

    def _apply_to_param(self, p: torch.nn.Parameter, scale: float) -> None:
        """Apply perturbation to a single parameter."""
        if p.grad is not None:
            e = p.grad * scale
            p.add_(e)
            if not hasattr(p, "state"):
                p.state = {}
            p.state["e"] = e

    def _apply_perturbation(self, scale: float) -> None:
        """Add e to each parameter."""
        for group in self.param_groups:
            for p in group["params"]:
                self._apply_to_param(p, scale)

    def first_step(self) -> None:
        """First step of SAM: perturb parameters in gradient direction."""
        with torch.no_grad():
            grad_norm = self._get_grad_norm()
            scale = self.rho / (grad_norm + EPSILON)
            self._apply_perturbation(float(scale))

    def second_step(self) -> None:
        """Second step of SAM: remove perturbation and update parameters."""
        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    if hasattr(p, "state") and "e" in p.state:
                        p.sub_(p.state["e"])
            self.base_optimizer.step()

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

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

    def first_step(self) -> None:
        """First step of ZSharp: apply gradient filtering and perturbation."""
        with torch.no_grad():
            layer_grad_info, layer_grads = (
                self._collect_gradients_for_filtering()
            )
            if not layer_grads:
                return

            zscores_list = self._compute_layer_zscores(layer_grads)
            threshold = self._compute_filtering_threshold(zscores_list)

            self._apply_gradient_filtering(
                layer_grad_info, zscores_list, threshold
            )

            self._apply_sam_perturbation()

    def _collect_for_param(
        self, p: torch.nn.Parameter, s_idx: int
    ) -> tuple[torch.Tensor, int, int] | None:
        """Process a single parameter for gradient collection."""
        if p.grad is None:
            return None
        gf = p.grad.detach().flatten()
        if gf.dtype == torch.float16:
            gf = gf.float()
        return gf, s_idx, s_idx + gf.numel()

    def _collect_gradients_for_filtering(
        self,
    ) -> tuple[
        list[tuple[torch.nn.Parameter, torch.Tensor, int, int]],
        list[torch.Tensor],
    ]:
        """Collect and flatten gradients for each layer."""
        info, grads = [], []
        curr_idx = 0
        for g in self.param_groups:
            for p in g["params"]:
                res = self._collect_for_param(p, curr_idx)
                if res:
                    gf, start, end = res
                    grads.append(gf)
                    info.append((p, p.grad, start, end))
                    curr_idx = end
        return info, grads

    def _compute_layer_zscores(
        self,
        layer_grads: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Compute Z-score normalization for each layer independently.

        Args:
            layer_grads: List of flattened gradients for each layer.

        Returns:
            list: Normalized Z-scores for each layer.
        """
        zscores_list: list[torch.Tensor] = []
        for grad_flat in layer_grads:
            layer_mean = torch.mean(grad_flat)
            layer_std = torch.std(grad_flat) + EPSILON_STD
            layer_zscores = (grad_flat - layer_mean) / layer_std
            zscores_list.append(layer_zscores)
        return zscores_list

    def _compute_filtering_threshold(
        self,
        zscores_list: list[torch.Tensor],
    ) -> float:
        """Compute the global threshold based on absolute Z-scores.

        Args:
            zscores_list: List of layer-wise normalized Z-scores.

        Returns:
            float: The absolute Z-score threshold for the requested percentile.
        """
        all_zscores = torch.cat(zscores_list)
        return float(
            torch.quantile(
                all_zscores.abs(),
                self.percentile / PERCENTAGE_MULTIPLIER,
            ).item(),
        )

    def _apply_gradient_filtering(
        self,
        layer_grad_info: list[
            tuple[torch.nn.Parameter, torch.Tensor, int, int]
        ],
        zscores_list: list[torch.Tensor],
        threshold: float,
    ) -> None:
        """Apply filtering mask to gradients based on threshold.

        Args:
            layer_grad_info: Metadata to map back to parameters.
            zscores_list: Precomputed Z-scores.
            threshold: Absolute Z-score threshold.
        """
        for i, (p, original_grad, _, _) in enumerate(layer_grad_info):
            layer_zscores = zscores_list[i]
            mask = layer_zscores.abs() >= threshold

            if not mask.any():
                top_k = max(1, int(DEFAULT_TOP_K_RATIO * mask.numel()))
                _, indices = torch.topk(layer_zscores.abs(), top_k)
                mask = torch.zeros_like(mask)
                mask[indices] = True

            mask = mask.view_as(original_grad)
            p.grad = cast("torch.Tensor", p.grad) * mask

    def _apply_sam_perturbation(self) -> None:
        """Apply the final SAM perturbation with filtered gradients."""
        grad_norm = self._get_grad_norm()
        scale = self.rho / (grad_norm + EPSILON)
        self._apply_perturbation(float(scale))
