# Experiments module for running different configurations
import yaml
import json
from src.train import train


def run_experiment(config_path, results_path="results/experiment.json"):
    """Run a single experiment and save results"""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Run training
    train(config)

    # Save results (placeholder)
    results = {"config": config, "status": "completed"}

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results
