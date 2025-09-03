"""Utility functions for reproducibility and common operations.

This module provides utility functions for setting random seeds
and other common operations used throughout the project.
"""

# Utility functions
import random

import numpy as np
import torch

from src.constants import DEFAULT_SEED


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value for all random number generators
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
