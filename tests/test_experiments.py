import pytest
import yaml
import json
import tempfile
import os
from unittest.mock import patch, MagicMock
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
                with open(results_path, "r") as f:
                    saved_results = json.load(f)
                    assert saved_results["status"] == "completed"

        finally:
            # Clean up temporary files
            if os.path.exists(config_path):
                os.unlink(config_path)
            if os.path.exists(results_path):
                os.unlink(results_path)

    def test_run_experiment_with_zsharp(self):
        """Test experiment execution with ZSharp optimizer"""
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
                "type": "zsharp",
                "lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0001,
                "rho": 0.05,
                "percentile": 70,
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config, f)
            config_path = f.name

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                results_path = f.name

            with patch("src.experiments.train") as mock_train:
                mock_train.return_value = {"status": "mocked"}

                results = run_experiment(config_path, results_path)

                mock_train.assert_called_once()
                assert isinstance(results, dict)
                assert results["status"] == "completed"

        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
            if os.path.exists(results_path):
                os.unlink(results_path)

    def test_run_experiment_invalid_config_file(self):
        """Test experiment execution with invalid config file"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                results_path = f.name

            # Should raise an error for invalid YAML
            with pytest.raises(yaml.YAMLError):
                run_experiment(config_path, results_path)

        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
            if os.path.exists(results_path):
                os.unlink(results_path)

    def test_run_experiment_nonexistent_config_file(self):
        """Test experiment execution with nonexistent config file"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            results_path = f.name

        try:
            # Should raise an error for nonexistent file
            with pytest.raises(FileNotFoundError):
                run_experiment("nonexistent_config.yaml", results_path)

        finally:
            if os.path.exists(results_path):
                os.unlink(results_path)

    def test_run_experiment_results_directory_creation(self):
        """Test that results directory is created if it doesn't exist"""
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
            # Use a path in a directory that might not exist
            results_path = "temp_results/test_experiment.json"

            # Mock the train function to avoid actual training
            with patch("src.experiments.train") as mock_train:
                mock_train.return_value = {"status": "mocked"}

                results = run_experiment(config_path, results_path)

                # Check that results were returned
                assert isinstance(results, dict)
                assert results["status"] == "completed"

                # Check that the directory was created
                assert os.path.exists("temp_results")

        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
            if os.path.exists("temp_results"):
                import shutil
                shutil.rmtree("temp_results")

    def test_run_experiment_json_serialization(self):
        """Test that results are properly serialized to JSON"""
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
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                results_path = f.name

            with patch("src.experiments.train") as mock_train:
                mock_train.return_value = {"status": "mocked"}

                results = run_experiment(config_path, results_path)

                # Check that results file contains valid JSON
                with open(results_path, "r") as f:
                    saved_results = json.load(f)

                    # Check structure
                    assert "config" in saved_results
                    assert "status" in saved_results
                    assert saved_results["status"] == "completed"

                    # Check that config was preserved
                    assert saved_results["config"]["dataset"] == "cifar10"
                    assert saved_results["config"]["model"] == "resnet18"

        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
            if os.path.exists(results_path):
                os.unlink(results_path)

    def test_run_experiment_train_function_integration(self):
        """Test integration with train function"""
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
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                results_path = f.name

            with patch("src.experiments.train") as mock_train:
                # Mock train to return realistic results
                mock_train.return_value = {
                    "final_test_accuracy": 85.5,
                    "final_test_loss": 0.45,
                    "train_losses": [1.2, 0.8, 0.6],
                    "train_accuracies": [45.2, 67.8, 78.9],
                    "total_training_time": 120.5,
                    "device": "cpu",
                    "optimizer_type": "sgd",
                }

                results = run_experiment(config_path, results_path)

                # Check that train was called with the correct config
                mock_train.assert_called_once()
                called_config = mock_train.call_args[0][0]
                assert called_config["dataset"] == "cifar10"
                assert called_config["model"] == "resnet18"

                # Check that results were returned
                assert isinstance(results, dict)
                assert results["status"] == "completed"

        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
            if os.path.exists(results_path):
                os.unlink(results_path)

    def test_run_experiment_error_handling(self):
        """Test error handling in experiment execution"""
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
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                results_path = f.name

            with patch("src.experiments.train") as mock_train:
                # Mock train to raise an exception
                mock_train.side_effect = RuntimeError("Training failed")

                # Should propagate the exception
                with pytest.raises(RuntimeError, match="Training failed"):
                    run_experiment(config_path, results_path)

        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
            if os.path.exists(results_path):
                os.unlink(results_path)
