"""Training utilities for deep learning models.

This module provides comprehensive training functionality including
device management, data loading, model training, and result saving.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

import torch
import torch.nn
import torch.optim
from torch import nn, optim
from tqdm import tqdm

from src.constants import (
    CIFAR100_DATASET,
    CIFAR100_NUM_CLASSES,
    CPU_DEVICE,
    CUDA_DEVICE,
    DEFAULT_SEED,
    MAX_GRADIENT_NORM,
    MPS_DEVICE,
    PERCENTAGE_MULTIPLIER,
    RESULTS_DIR,
    SGD_OPTIMIZER,
    ExperimentResults,
    TrainingConfig,
)
from src.data import get_dataset
from src.models import get_model
from src.optimizer import ZSharp
from src.utils import set_seed

logger = logging.getLogger(__name__)


def get_device(config: TrainingConfig) -> torch.device:
    """Get the best available device for training.

    Args:
        config: Configuration dictionary containing device settings

    Returns:
        torch.device: Best available device (mps/cuda/cpu)

    """
    device_config = config.train.device

    if device_config == MPS_DEVICE and torch.backends.mps.is_available():
        return torch.device(MPS_DEVICE)
    if device_config == CUDA_DEVICE and torch.cuda.is_available():
        return torch.device(CUDA_DEVICE)
    return torch.device(CPU_DEVICE)


@dataclass(frozen=True)
class TrainingContext:
    """Encapsulates training components and flags."""

    model: nn.Module
    optimizer: torch.optim.Optimizer
    criterion: nn.Module
    device: torch.device
    use_zsharp: bool
    use_half: bool


def _setup_optimizer(
    config: TrainingConfig,
    model: nn.Module,
) -> tuple[torch.optim.Optimizer, bool]:
    """Initialize the optimizer based on configuration."""
    opt_config = config.optimizer
    opt_type = opt_config.type
    params = list(model.parameters())
    lr = float(opt_config.lr)
    momentum = float(opt_config.momentum)
    wd = float(opt_config.weight_decay)

    if opt_type == SGD_OPTIMIZER:
        optimizer: torch.optim.Optimizer = optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=wd
        )
        return optimizer, False

    # ZSharp optimizer
    optimizer = ZSharp(
        params,
        base_optimizer=optim.SGD,
        rho=float(opt_config.rho),
        lr=lr,
        momentum=momentum,
        weight_decay=wd,
        percentile=int(opt_config.percentile),
    )
    return optimizer, True


def _run_train_step(
    ctx: TrainingContext,
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """Perform a single training step."""
    if ctx.use_zsharp:
        # ZSharp two-step training
        loss = ctx.criterion(ctx.model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            ctx.model.parameters(), MAX_GRADIENT_NORM
        )
        zsharp_opt = cast("ZSharp", ctx.optimizer)
        zsharp_opt.first_step()
        ctx.criterion(ctx.model(x), y).backward()
        zsharp_opt.second_step()
    else:
        # Standard SGD training
        ctx.optimizer.zero_grad()
        loss = ctx.criterion(ctx.model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            ctx.model.parameters(), MAX_GRADIENT_NORM
        )
        ctx.optimizer.step()
    return float(loss.item())


def _validate(
    ctx: TrainingContext,
    loader: torch.utils.data.DataLoader[torch.Tensor],
) -> tuple[float, float]:
    """Evaluate model on a dataset."""
    ctx.model.eval()
    correct, total, total_loss = 0, 0, 0.0
    pbar = tqdm(loader, desc="Evaluating")
    with torch.no_grad():
        for x, y in pbar:
            x, y = x.to(ctx.device), y.to(ctx.device)
            if ctx.use_half:
                x = x.half()
            outputs = ctx.model(x)
            loss = ctx.criterion(outputs, y)
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == y).sum().item()
            total += y.size(0)
            pbar.set_postfix(
                {"Acc": f"{PERCENTAGE_MULTIPLIER * correct / total:.2f}%"}
            )
    acc = PERCENTAGE_MULTIPLIER * correct / total if total > 0 else 0.0
    return acc, total_loss / len(loader) if len(loader) > 0 else 0.0


def _run_epoch(
    ctx: TrainingContext,
    epoch: int,
    loader: torch.utils.data.DataLoader[torch.Tensor],
) -> tuple[float, float]:
    """Run a single training epoch."""
    ctx.model.train()
    epoch_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}")

    for x, y in pbar:
        x, y = x.to(ctx.device), y.to(ctx.device)
        if ctx.use_half:
            x = x.half()
        loss = _run_train_step(ctx, x, y)
        epoch_loss += loss
        correct += (ctx.model(x).argmax(dim=1) == y).sum().item()
        total += y.size(0)
        pbar.set_postfix({"Loss": f"{loss:.4f}"})
    return epoch_loss / len(loader), PERCENTAGE_MULTIPLIER * correct / total


def _save_results(
    results: ExperimentResults,
    dataset_name: str,
    model_name: str,
    opt_type: str,
) -> None:
    """Save results to a JSON file."""
    path = Path(RESULTS_DIR)
    file_path = path / f"zsharp_{dataset_name}_{model_name}_{opt_type}.json"
    path.mkdir(parents=True, exist_ok=True)
    with file_path.open("w") as f:
        json.dump(results.model_dump(), f, indent=2)


def _init_components(
    config: TrainingConfig,
    device: torch.device,
) -> tuple[nn.Module, torch.optim.Optimizer, bool]:
    """Initialize model and optimizer components."""
    ds_name = config.dataset
    classes = CIFAR100_NUM_CLASSES if ds_name == CIFAR100_DATASET else 10
    model_name = config.model
    model = get_model(model_name, num_classes=classes).to(device)
    optimizer, use_zs = _setup_optimizer(config, model)
    return model, optimizer, use_zs


def _prepare_training(
    config: TrainingConfig,
    device: torch.device,
) -> tuple[TrainingContext, tuple[DataLoader[Any], DataLoader[Any]], int]:
    """Prepare training context and loaders."""
    cfg = config.train
    m, opt, uz = _init_components(config, device)
    uh = bool(device.type == "mps" and cfg.use_mixed_precision)
    if uh:
        m = m.half()
    ctx = TrainingContext(m, opt, nn.CrossEntropyLoss(), device, uz, uh)
    ldrs = get_dataset(
        dataset_name=config.dataset,
        batch_size=int(cfg.batch_size),
        num_workers=int(cfg.num_workers),
        pin_memory=cfg.pin_memory,
    )
    return ctx, ldrs, int(cfg.epochs)


def _create_results(
    config: TrainingConfig,
    ctx: TrainingContext,
    metrics: tuple[list[float], list[float], list[float]],
    final: tuple[float, float],
    duration: float,
) -> ExperimentResults:
    """Consolidate results into dictionary."""
    fa, fl = final
    return ExperimentResults(
        config=config,
        final_test_accuracy=fa,
        final_test_loss=fl,
        train_losses=metrics[0],
        train_accuracies=metrics[1],
        test_accuracies=metrics[2],
        total_training_time=duration,
        device=str(ctx.device),
        optimizer_type="zsharp",
    )


def train(config: TrainingConfig) -> ExperimentResults | None:
    """Train a model using the provided configuration."""
    set_seed(DEFAULT_SEED)
    device = get_device(config)
    ctx, (train_ldr, test_ldr), epochs = _prepare_training(config, device)
    start_time = time.time()
    l_list, t_list, v_list = [], [], []

    try:
        for epoch in range(epochs):
            e_loss, a = _run_epoch(ctx, epoch, train_ldr)
            va, _ = _validate(ctx, test_ldr)
            l_list.append(e_loss)
            t_list.append(a)
            v_list.append(va)
            logger.info(
                "Epoch %d: Acc: %.2f%%, Test: %.2f%%", epoch + 1, a, va
            )
    except KeyboardInterrupt:
        return None

    res = _create_results(
        config,
        ctx,
        (l_list, t_list, v_list),
        _validate(ctx, test_ldr),
        time.time() - start_time,
    )
    _save_results(res, config.dataset, config.model, "zsharp")
    return res
