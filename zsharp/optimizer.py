# Copyright (c) 2025 Bangyen Pham
"""Optimizer implementations for SAM and ZSharp.

This module provides implementations of SAM (Sharpness-Aware Minimization)
and ZSharp optimizers for deep learning training with gradient filtering.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional, cast, overload

if TYPE_CHECKING:
    from collections.abc import Callable

import torch
import torch.optim
from torch.optim import Optimizer

from zsharp.constants import (
    DEFAULT_PERCENTILE,
    DEFAULT_RHO,
    EPSILON,
    EPSILON_STD,
    MIN_NUM_FOR_STD,
    PREFILTER_MIN_NUMEL,
    QUANTILE_SAMPLE_SIZE,
)

# Type for optimizer kwargs
OptimizerKwargs = Any
"""Type alias for optimizer keyword arguments."""


def _quantile_by_selection(values: torch.Tensor, q: float) -> float:
    """Return ``torch.quantile(values, q)`` without sorting every element.

    torch.quantile fully sorts its input, which dominates a ZSharp step on
    CPU, and it rejects inputs above 2**24 elements. Only the two order
    statistics around the quantile are needed. Large inputs are first
    narrowed to the values above a bound read off a strided sample, which
    is exact whenever the bound lies below the target rank; otherwise, and
    for small inputs, the ``n - floor(q * (n - 1))`` largest values are
    kept with ``topk`` and the order statistics read off their bottom.

    Args:
        values: Non-empty 1-D tensor.
        q: Quantile in [0, 1].

    Returns:
        float: The interpolated quantile. It agrees with ``torch.quantile``
        to float32 precision; the interpolation position is computed in
        float64 here, whereas torch.quantile rounds it to float32, which
        snaps it to an integer rank for multi-million-element inputs.
    """
    n = values.numel()
    pos = q * (n - 1)
    lo = int(pos)
    frac = pos - lo
    order_stats = None
    if n >= PREFILTER_MIN_NUMEL:
        order_stats = _order_stats_via_sample(values, q, lo, frac)
    if order_stats is None:
        upper = values.topk(n - lo, sorted=False).values
        order_stats = upper.topk(min(2, n - lo), largest=False).values
        order_stats = order_stats.sort().values
    if frac == 0:
        return float(order_stats[0].item())
    v_lo, v_hi = order_stats[0], order_stats[1]
    return float((v_lo + (v_hi - v_lo) * frac).item())


def _order_stats_via_sample(
    values: torch.Tensor, q: float, lo: int, frac: float
) -> Optional[torch.Tensor]:
    """Select order statistics ``lo`` (and ``lo + 1``) above a safe bound.

    A strided sample (deterministic, so the global RNG is untouched) gives
    a bound a few standard errors below the target quantile. Values below
    the bound are counted and dropped, and the remaining few percent are
    searched directly.

    Returns:
        The needed order statistics in ascending order, or None if the
        bound overshot the target rank and the caller must fall back.
    """
    n = values.numel()
    sample = values[:: max(1, n // QUANTILE_SAMPLE_SIZE)]
    m = sample.numel()
    margin = int(4 * math.sqrt(m * q * (1 - q))) + 1
    bound = sample.kthvalue(max(0, int(q * (m - 1)) - margin) + 1).values
    below = int((values < bound).sum().item())
    if below > lo:
        return None
    candidates = values[values >= bound]
    rank = lo - below
    stats = [candidates.kthvalue(rank + 1).values]
    if frac:
        stats.append(candidates.kthvalue(rank + 2).values)
    return torch.stack(stats)


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
        base_optimizer: type[Optimizer],
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
        if not norms:
            return torch.tensor(0.0)
        return cast("torch.Tensor", torch.norm(torch.stack(norms), p=2))

    def first_step(self) -> None:
        """First step of SAM: perturb parameters in gradient direction."""
        with torch.no_grad():
            grad_norm = self._get_grad_norm()
            scale = float(self.rho / (grad_norm + EPSILON))
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        e = p.grad * scale
                        p.add_(e)
                        self.state[p]["e"] = e

    def second_step(self) -> None:
        """Second step of SAM: remove perturbation and update parameters."""
        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    if "e" in self.state[p]:
                        p.sub_(self.state[p]["e"])
            self.base_optimizer.step()

    @overload
    def step(self, closure: None = None) -> None:  # noqa: D418
        """Perform a step with no closure."""

    @overload
    def step(self, closure: Callable[[], float]) -> float:  # noqa: D418
        """Perform a step with a closure."""

    def step(
        self,
        closure: Optional[Callable[[], float]] = None,
    ) -> Optional[float]:
        """Perform a standard optimizer step with SAM.

        This method allows SAM to be used like a standard PyTorch optimizer.
        It requires a closure that re-evaluates the model and returns the loss,
        as SAM needs to compute gradients twice (at the current point and at
        the perturbed point).

        Args:
            closure: A closure that re-evaluates the model and returns the loss.

        Returns:
            Optional[float]: The loss value from the closure.
        """
        if closure is None:
            msg = "SAM requires a closure that returns the loss"
            raise RuntimeError(msg)

        self.first_step()
        loss = closure()
        self.second_step()
        return loss


class ZSharp(SAM):
    """ZSharp: Sharpness-Aware Minimization with Z-Score Gradient Filtering.

    ZSharp extends SAM by applying layer-wise Z-score normalization and
    percentile-based gradient filtering before the SAM perturbation step.
    This helps focus on the most important gradients and improves training
    stability.

    Following the paper (arXiv:2505.02369), Z-scores are normalized within
    each layer, the threshold is the ``percentile``-th quantile of the
    absolute Z-scores pooled across all layers, and filtering is applied to
    the ascent step only. If the filtered gradient vanishes everywhere, the
    unfiltered gradient is used instead (Eq. 9).

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
        base_optimizer: type[Optimizer],
        rho: float = DEFAULT_RHO,
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

            super().first_step()

    def _collect_gradients_for_filtering(
        self,
    ) -> tuple[
        list[tuple[torch.nn.Parameter, torch.Tensor]],
        list[torch.Tensor],
    ]:
        """Collect and flatten gradients for each layer."""
        info: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
        grads: list[torch.Tensor] = []
        for g in self.param_groups:
            for p in g["params"]:
                gf = self._get_flattened_grad(p)
                if gf is not None:
                    grads.append(gf)
                    info.append((p, p.grad))
        return info, grads

    def _get_flattened_grad(
        self, p: torch.nn.Parameter
    ) -> Optional[torch.Tensor]:
        """Extract and flatten gradient from a parameter."""
        if p.grad is None:
            return None
        gf = p.grad.detach().flatten()
        if gf.dtype == torch.float16:
            gf = gf.float()
        return gf

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
            if grad_flat.numel() < MIN_NUM_FOR_STD:
                # Handle edge case for single-element or empty tensors
                layer_zscores = torch.zeros_like(grad_flat)
            else:
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
        all_zscores = torch.cat(zscores_list).abs()
        n = all_zscores.numel()
        if n == 0:
            return 0.0
        return _quantile_by_selection(all_zscores, self.percentile / 100)

    def _apply_gradient_filtering(
        self,
        layer_grad_info: list[tuple[torch.nn.Parameter, torch.Tensor]],
        zscores_list: list[torch.Tensor],
        threshold: float,
    ) -> None:
        """Apply filtering mask to gradients based on threshold.

        Components whose absolute Z-score does not exceed the threshold are
        zeroed. Whole layers may be zeroed out, which the paper permits: the
        threshold is pooled across the network, so a layer with uniformly
        small Z-scores contributes nothing to the ascent direction. Only if
        *every* layer is zeroed does the filtering back off, restoring the
        unfiltered gradients per Eq. 9.

        Args:
            layer_grad_info: Metadata to map back to parameters.
            zscores_list: Precomputed Z-scores.
            threshold: Absolute Z-score threshold.
        """
        retained = 0
        for i, (p, original_grad) in enumerate(layer_grad_info):
            mask = (zscores_list[i].abs() > threshold).view_as(original_grad)
            retained += int(mask.any())
            p.grad = cast("torch.Tensor", p.grad) * mask

        if not retained:
            self._restore_unfiltered_gradients(layer_grad_info)

    @staticmethod
    def _restore_unfiltered_gradients(
        layer_grad_info: list[tuple[torch.nn.Parameter, torch.Tensor]],
    ) -> None:
        """Undo filtering when it zeroed the gradient everywhere (Eq. 9).

        Args:
            layer_grad_info: Parameters paired with their pre-filter
                gradients. Masking rebinds ``p.grad`` rather than mutating
                it, so the originals are still intact.
        """
        for p, original_grad in layer_grad_info:
            p.grad = original_grad
