# Copyright (c) 2025 Bangyen Pham
"""Constants used throughout the ZSharp codebase.

This module defines all the magic numbers and configuration values
that were previously hardcoded throughout the codebase.
"""

from typing import Optional

from pydantic import BaseModel, Field, field_validator

# Random seed for reproducibility
DEFAULT_SEED = 42

# Math constants
MIN_NUM_FOR_STD = 2

# torch.quantile rejects inputs larger than 2**24 elements; above this we
# fall back to kthvalue, which computes the same order statistic unbounded.
MAX_QUANTILE_NUMEL = 2**24

# Dataset names
CIFAR10_DATASET = "cifar10"
CIFAR100_DATASET = "cifar100"
TINY_IMAGENET_DATASET = "tiny_imagenet"

# Tiny-ImageNet is not distributed through torchvision; it is downloaded
# from the canonical Stanford CS231n mirror and extracted under DATA_ROOT.
TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
TINY_IMAGENET_DIRNAME = "tiny-imagenet-200"

# Default batch and training parameters. Batch size matches the paper; the
# epoch default below stays low deliberately, since the paper's 200 epochs
# is a poor default for an unattended run. The shipped configs set it.
DEFAULT_BATCH_SIZE = 256
DEFAULT_NUM_WORKERS = 2
DEFAULT_PIN_MEMORY = False

# Optimizer constants
# Paper defaults (arXiv:2505.02369, "Experimental Settings"): AdamW with
# lr 1e-3 and weight decay 5e-5, and Q_p = 0.95, which keeps the top 5% of
# gradient components by absolute Z-score.
DEFAULT_RHO = 0.05
DEFAULT_PERCENTILE = 95
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_MOMENTUM = 0.9
DEFAULT_WEIGHT_DECAY = 5e-5

# Learning rate schedule: multiplied by 0.75 every 10 epochs.
DEFAULT_LR_STEP_SIZE = 10
DEFAULT_LR_GAMMA = 0.75

# Numerical stability constant (delta in the paper).
EPSILON = 1e-8
EPSILON_STD = 1e-8

# Model architecture constants
RESNET18_NAME = "resnet18"

# CIFAR-style ResNets (He et al., Sec. 4.2): three stages of basic blocks
# starting at 16 channels and doubling, for a depth of 6n + 2.
CIFAR_RESNET_BASE_WIDTH = 16
CIFAR_RESNET_STAGES = 3

# The paper's compact ViTs: 7 layers, 8 heads, an embedding width of 384,
# and 8 patches per side. Taken from the author's reference implementation
# (github.com/YUNBLAK/Sharpness-Aware-Minimization-with-Z-Score-Gradient-Filtering),
# since the paper itself does not state the embedding dimension.
VIT_PAPER_LAYERS = 7
VIT_PAPER_HEADS = 8
VIT_PAPER_HIDDEN = 384
VIT_PAPER_PATCHES_PER_SIDE = 8

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
    # When set, training state is saved here after every epoch and an
    # existing checkpoint for the same run is resumed on the next start.
    checkpoint_dir: Optional[str] = Field(default=None)


class TrainingConfig(BaseModel):
    """Overall training configuration."""

    train: TrainingSubConfig = Field(default_factory=TrainingSubConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    dataset: str = Field(default=CIFAR10_DATASET)
    model: str = Field(default=RESNET18_NAME)
    seed: int = Field(default=DEFAULT_SEED, ge=0)


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
