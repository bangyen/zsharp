"""Test suite for experiment running functions."""

import json
import os
import tempfile
from unittest.mock import patch

import yaml

from src.experiments import run_experiment


class TestExperiments:
    """Test cases for experiments module"""

    def test_run_experiment_basic(self):
        """Test basic experiment execution"""
        # Create a temporary config file
        config = {
            "dataset": "cifar10",
            "model": "resnet18",
            "train": {
                "epochs": 1,
                "batch_size": 32,
                "device": "cpu",
                "num_workers": 0,
            },
            "optimizer": {
                "type": "sgd",
                "lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0001,
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config, f)
            config_path = f.name

        try:
            # Create temporary results file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                results_path = f.name

            # Mock the train function to avoid actual training
            with patch("src.experiments.train") as mock_train:
                mock_train.return_value = {"status": "mocked"}

                results = run_experiment(config_path, results_path)

                # Check that train was called
                mock_train.assert_called_once()

                # Check that results were returned
                assert isinstance(results, dict)
                assert "config" in results
                assert "status" in results
                assert results["status"] == "completed"

                # Check that results file was created
                assert os.path.exists(results_path)

                # Check that results were written to file
                with open(results_path) as f:
                    saved_results = json.load(f)
                    assert saved_results["status"] == "completed"

        finally:
            # Clean up temporary files
            if os.path.exists(config_path):
                os.unlink(config_path)
            if os.path.exists(results_path):
                os.unlink(results_path)
