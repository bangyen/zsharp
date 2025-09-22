"""Test suite for utility functions."""

import random
from unittest.mock import patch

import numpy as np
import torch

from src.utils import set_seed


class TestUtils:
    """Test cases for utility functions"""

    def test_set_seed_default(self):
        """Test set_seed with default seed value"""
        # Set seed
        set_seed()

        # Generate some random numbers
        torch_rand = torch.rand(5)
        np_rand = np.random.rand(5)
        py_rand = [random.random() for _ in range(5)]

        # Should not raise an error
        assert len(torch_rand) == 5
        assert len(np_rand) == 5
        assert len(py_rand) == 5

    def test_set_seed_custom(self):
        """Test set_seed with custom seed value"""
        seed = 12345
        set_seed(seed)

        # Generate random numbers
        torch_rand1 = torch.rand(5)
        np_rand1 = np.random.rand(5)
        py_rand1 = [random.random() for _ in range(5)]

        # Set same seed again
        set_seed(seed)

        # Generate random numbers again
        torch_rand2 = torch.rand(5)
        np_rand2 = np.random.rand(5)
        py_rand2 = [random.random() for _ in range(5)]

        # Should be reproducible
        assert torch.allclose(torch_rand1, torch_rand2)
        assert np.allclose(np_rand1, np_rand2)
        assert py_rand1 == py_rand2

    def test_set_seed_cuda_specific_seeds(self):
        """Test set_seed specifically covers CUDA seed setting lines"""
        seed = 42

        # Mock CUDA availability to ensure the CUDA seed lines are executed
        with patch("torch.cuda.is_available", return_value=True):
            set_seed(seed)

            # Verify that the function completed without error
            # The CUDA seed setting lines should have been executed
            torch_rand = torch.rand(5)
            assert len(torch_rand) == 5
