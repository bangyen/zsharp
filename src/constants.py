# Copyright (c) 2025 Bangyen Pham
"""Constants used throughout the ZSharp codebase.

This module defines all the magic numbers and configuration values
that were previously hardcoded throughout the codebase.
"""

from pydantic import BaseModel, Field, field_validator

# Random seed for reproducibility
DEFAULT_SEED = 42

# Math constants
MIN_NUM_FOR_STD = 2

# Dataset names
CIFAR10_DATASET = "cifar10"
CIFAR100_DATASET = "cifar100"

# Default batch and training parameters
DEFAULT_BATCH_SIZE = 128
DEFAULT_NUM_WORKERS = 2
DEFAULT_PIN_MEMORY = False

# Optimizer constants
DEFAULT_RHO = 0.05
DEFAULT_PERCENTILE = 70
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_MOMENTUM = 0.9
DEFAULT_WEIGHT_DECAY = 5e-4

# Numerical stability constants
EPSILON = 1e-12
EPSILON_STD = 1e-8

# Gradient clipping
MAX_GRADIENT_NORM = 1.0

# Z-score filtering constants
DEFAULT_TOP_K_RATIO = 0.2  # Keep top 20% if no gradients pass threshold

# Model architecture constants
RESNET18_NAME = "resnet18"

# Optimizer types
SGD_OPTIMIZER = "sgd"
ZSHARP_OPTIMIZER = "zsharp"

# Device types
MPS_DEVICE = "mps"
CUDA_DEVICE = "cuda"
CPU_DEVICE = "cpu"
AUTO_DEVICE = "auto"

# File paths
DATA_ROOT = "./data"
RESULTS_DIR = "results"

# Configuration parameter keys
# Removed unused *_KEY constants


# Type definitions for configuration
class OptimizerConfig(BaseModel):
    """Configuration for the optimizer."""

    type: str = Field(default=ZSHARP_OPTIMIZER)
    lr: float = Field(default=DEFAULT_LEARNING_RATE, gt=0)
    momentum: float = Field(default=DEFAULT_MOMENTUM, ge=0, lt=1)
    weight_decay: float = Field(default=DEFAULT_WEIGHT_DECAY, ge=0)
    rho: float = Field(default=DEFAULT_RHO, gt=0)
    percentile: int = Field(default=DEFAULT_PERCENTILE, ge=0, le=100)

    @field_validator("type")
    @classmethod
    def _validate_type(cls, value: str) -> str:
        """Reject unknown optimizer types instead of silently using ZSharp."""
        if value not in (SGD_OPTIMIZER, ZSHARP_OPTIMIZER):
            msg = f"Unknown optimizer type: {value}"
            raise ValueError(msg)
        return value


class TrainingSubConfig(BaseModel):
    """Sub-configuration for training parameters."""

    device: str = Field(default=AUTO_DEVICE)
    batch_size: int = Field(default=DEFAULT_BATCH_SIZE, gt=0)
    epochs: int = Field(default=10, gt=0)
    num_workers: int = Field(default=DEFAULT_NUM_WORKERS, ge=0)
    pin_memory: bool = Field(default=DEFAULT_PIN_MEMORY)
    use_mixed_precision: bool = Field(default=False)


class TrainingConfig(BaseModel):
    """Overall training configuration."""

    train: TrainingSubConfig = Field(default_factory=TrainingSubConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    dataset: str = Field(default=CIFAR10_DATASET)
    model: str = Field(default=RESNET18_NAME)


class ExperimentResults(BaseModel):
    """Results from an experiment."""

    config: TrainingConfig
    final_test_accuracy: float
    final_test_loss: float
    train_losses: list[float]
    train_accuracies: list[float]
    test_accuracies: list[float]
    total_training_time: float
    device: str
    optimizer_type: str
