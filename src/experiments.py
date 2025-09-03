# Experiments module for running different configurations
import json
import os

import yaml

from src.train import train


def run_experiment(config_path, results_path="results/experiment.json"):
    """Run a single experiment and save results"""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Run training
    train(config)

    # Save results (placeholder)
    results = {"config": config, "status": "completed"}

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results
