"""Test suite for experiment running functions."""

import json
import os
import tempfile
from unittest.mock import patch

import pytest
import yaml

from src.experiments import run_experiment


class TestExperiments:
    """Test cases for experiments module"""

    def test_run_experiment_basic(self):
        """Test basic experiment execution"""
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

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        results_path = tempfile.mktemp(suffix=".json")

        try:
            with patch("src.experiments.train") as mock_train:
                mock_train.return_value = {
                    "config": config,
                    "final_test_accuracy": 0.9,
                    "final_test_loss": 0.1,
                    "train_losses": [0.1],
                    "train_accuracies": [0.9],
                    "test_accuracies": [0.9],
                    "total_training_time": 1.0,
                    "device": "cpu",
                    "optimizer_type": "sgd",
                }

                results = run_experiment(config_path, results_path)

                mock_train.assert_called_once()
                assert isinstance(results, dict)
                assert "final_test_accuracy" in results
                assert os.path.exists(results_path)
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
            if os.path.exists(results_path):
                os.unlink(results_path)

    def test_run_experiment_failure(self):
        """Test experiment execution failure when train returns None"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"dataset": "c10"}, f)
            config_path = f.name
        try:
            with patch("src.experiments.train", return_value=None):
                with pytest.raises(RuntimeError, match="Training failed to return results"):
                    run_experiment(config_path, "results_failed.json")
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
            if os.path.exists("results_failed.json"):
                os.unlink("results_failed.json")
