"""Experiments module for running different configurations.

This module provides utilities for running experiments with different
configurations and saving results to JSON files.
"""

import json
from pathlib import Path

import yaml

from src.constants import ExperimentResults, TrainingConfig
from src.train import train


def run_experiment(
    config_path: str,
    results_path: str = "results/experiment.json",
) -> ExperimentResults:
    """Run a single experiment and save results.

    Args:
        config_path: Path to YAML configuration file
        results_path: Path to save experiment results

    Returns:
        dict: Experiment results and configuration

    """
    with Path(config_path).open() as f:
        config: TrainingConfig = yaml.safe_load(f)

    # Run training
    results = train(config)

    if results is None:
        msg = "Training failed to return results"
        raise RuntimeError(msg)

    # Create directory if it doesn't exist
    res_path = Path(results_path)
    res_path.parent.mkdir(parents=True, exist_ok=True)

    with res_path.open("w") as f:
        json.dump(results, f, indent=2)

    return results
