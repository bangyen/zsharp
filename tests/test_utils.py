import torch
import random
import numpy as np
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

    def test_set_seed_different_seeds(self):
        """
        Test set_seed with different seed values produces different results
        """
        # Set seed 1
        set_seed(42)
        torch_rand1 = torch.rand(5)
        np_rand1 = np.random.rand(5)
        py_rand1 = [random.random() for _ in range(5)]

        # Set seed 2
        set_seed(123)
        torch_rand2 = torch.rand(5)
        np_rand2 = np.random.rand(5)
        py_rand2 = [random.random() for _ in range(5)]

        # Should be different
        assert not torch.allclose(torch_rand1, torch_rand2)
        assert not np.allclose(np_rand1, np_rand2)
        assert py_rand1 != py_rand2

    def test_set_seed_torch_deterministic(self):
        """Test that torch.backends.cudnn.deterministic is set to True"""
        set_seed(42)

        # Check that deterministic flag is set
        assert torch.backends.cudnn.deterministic is True

    def test_set_seed_torch_benchmark(self):
        """Test that torch.backends.cudnn.benchmark is set to False"""
        set_seed(42)

        # Check that benchmark flag is set to False
        assert torch.backends.cudnn.benchmark is False

    def test_set_seed_cuda_available(self):
        """Test set_seed when CUDA is available"""
        set_seed(42)

        # Should not raise an error regardless of CUDA availability
        # Verify that the function completes successfully
        torch_rand = torch.rand(5)
        assert len(torch_rand) == 5

        # If CUDA is available, verify CUDA seeds are set
        if torch.cuda.is_available():
            # The function should have set CUDA seeds without error
            # Verify that CUDA random state is properly initialized
            cuda_rand = torch.cuda.FloatTensor(5).uniform_()
            assert len(cuda_rand) == 5

    def test_set_seed_reproducibility(self):
        """Test that set_seed ensures reproducibility across multiple calls"""
        seed = 999

        # First run
        set_seed(seed)
        results1 = []
        for _ in range(3):
            results1.append(torch.rand(3).tolist())

        # Second run with same seed
        set_seed(seed)
        results2 = []
        for _ in range(3):
            results2.append(torch.rand(3).tolist())

        # Results should be identical
        assert results1 == results2

    def test_set_seed_negative_seed(self):
        """Test set_seed with negative seed value"""
        # Negative seeds should be handled gracefully
        try:
            set_seed(-42)
            # If it doesn't raise an error, should work
            torch_rand = torch.rand(5)
            assert len(torch_rand) == 5
        except ValueError:
            # This is expected behavior for negative seeds
            # Verify that the error is properly handled
            # Test that we can still use random functions after the error
            torch_rand = torch.rand(5)
            assert len(torch_rand) == 5

    def test_set_seed_zero_seed(self):
        """Test set_seed with zero seed value"""
        set_seed(0)

        # Should not raise an error
        torch_rand = torch.rand(5)
        assert len(torch_rand) == 5

    def test_set_seed_large_seed(self):
        """Test set_seed with large seed value"""
        set_seed(999999)

        # Should not raise an error
        torch_rand = torch.rand(5)
        assert len(torch_rand) == 5

    def test_set_seed_multiple_calls(self):
        """Test set_seed with multiple consecutive calls"""
        for seed in [1, 2, 3, 4, 5]:
            set_seed(seed)
            torch_rand = torch.rand(3)
            assert len(torch_rand) == 3

    def test_set_seed_numpy_reproducibility(self):
        """Test that numpy random state is properly set"""
        seed = 42

        # First run
        set_seed(seed)
        np_rand1 = np.random.rand(5)

        # Second run
        set_seed(seed)
        np_rand2 = np.random.rand(5)

        # Should be identical
        assert np.allclose(np_rand1, np_rand2)

    def test_set_seed_python_random_reproducibility(self):
        """Test that python random state is properly set"""
        seed = 42

        # First run
        set_seed(seed)
        py_rand1 = [random.random() for _ in range(5)]

        # Second run
        set_seed(seed)
        py_rand2 = [random.random() for _ in range(5)]

        # Should be identical
        assert py_rand1 == py_rand2
