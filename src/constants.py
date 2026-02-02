"""Constants used throughout the ZSharp codebase.

This module defines all the magic numbers and configuration values
that were previously hardcoded throughout the codebase.
"""

from pydantic import BaseModel, Field

# Random seed for reproducibility
DEFAULT_SEED = 42

# Dataset constants
CIFAR10_NUM_CLASSES = 10
CIFAR100_NUM_CLASSES = 100

# CIFAR-10 normalization values
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

# CIFAR-100 normalization values
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

# Image dimensions
CIFAR_IMAGE_SIZE = 32
CIFAR_CROP_PADDING = 4

# Default batch and training parameters
DEFAULT_BATCH_SIZE = 128
DEFAULT_NUM_WORKERS = 2
DEFAULT_PIN_MEMORY = False

# Optimizer constants
DEFAULT_RHO = 0.05
DEFAULT_PERCENTILE = 70
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_MOMENTUM = 0.9
DEFAULT_WEIGHT_DECAY = 5e-4

# Numerical stability constants
EPSILON = 1e-12
EPSILON_STD = 1e-8

# Gradient clipping
MAX_GRADIENT_NORM = 1.0

# Z-score filtering constants
DEFAULT_TOP_K_RATIO = 0.2  # Keep top 20% if no gradients pass threshold

# Percentage conversion
PERCENTAGE_MULTIPLIER = 100

# Model architecture constants
RESNET18_NAME = "resnet18"
VIT_NAME = "vit"

# Dataset names
CIFAR10_DATASET = "cifar10"
CIFAR100_DATASET = "cifar100"

# Optimizer types
SGD_OPTIMIZER = "sgd"
ZSHARP_OPTIMIZER = "zsharp"

# Device types
MPS_DEVICE = "mps"
CUDA_DEVICE = "cuda"
CPU_DEVICE = "cpu"

# File paths
DATA_ROOT = "./data"
RESULTS_DIR = "results"

# Training configuration keys
TRAIN_CONFIG_KEY = "train"
OPTIMIZER_CONFIG_KEY = "optimizer"
DATASET_CONFIG_KEY = "dataset"
MODEL_CONFIG_KEY = "model"

# Configuration parameter keys
BATCH_SIZE_KEY = "batch_size"
NUM_WORKERS_KEY = "num_workers"
PIN_MEMORY_KEY = "pin_memory"
USE_MIXED_PRECISION_KEY = "use_mixed_precision"
EPOCHS_KEY = "epochs"
DEVICE_KEY = "device"
TYPE_KEY = "type"
LR_KEY = "lr"
MOMENTUM_KEY = "momentum"
WEIGHT_DECAY_KEY = "weight_decay"
RHO_KEY = "rho"
PERCENTILE_KEY = "percentile"


# Type definitions for configuration
class OptimizerConfig(BaseModel):
    """Configuration for the optimizer."""

    type: str = Field(default=ZSHARP_OPTIMIZER)
    lr: float = Field(default=DEFAULT_LEARNING_RATE)
    momentum: float = Field(default=DEFAULT_MOMENTUM)
    weight_decay: float = Field(default=DEFAULT_WEIGHT_DECAY)
    rho: float = Field(default=DEFAULT_RHO)
    percentile: int = Field(default=DEFAULT_PERCENTILE)


class TrainingSubConfig(BaseModel):
    """Sub-configuration for training parameters."""

    device: str = Field(default=CPU_DEVICE)
    batch_size: int = Field(default=DEFAULT_BATCH_SIZE)
    epochs: int = Field(default=10)
    num_workers: int = Field(default=DEFAULT_NUM_WORKERS)
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
