"""Comprehensive experiment runner for ZSharp paper reproduction."""

import argparse
import json
import logging
import os
import signal
import sys
import time

import yaml

from src.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MOMENTUM,
    DEFAULT_NUM_WORKERS,
    DEFAULT_RHO,
    DEFAULT_WEIGHT_DECAY,
    RESULTS_DIR,
)

# Import the training function directly
from src.train import train

# Configure logging without prefix
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("experiments.log"),
    ],
)
logger = logging.getLogger(__name__)


def run_experiment(config_path):
    """Run a single experiment and save results"""
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    try:
        # Run training directly using the imported function
        results = train(config)
        if results is None:
            logger.warning("Training was interrupted")
            return None
        return results

    except KeyboardInterrupt:
        logger.warning("Experiment interrupted by user")
        return None
    except Exception as e:
        logger.error(f"Experiment failed with error: {e}")
        return None


def run_comparison_experiments(fast_mode=False):
    """Run comparison experiments as described in the ZSharp paper"""
    if fast_mode:
        experiments = [
            ("configs/sgd_quick.yaml", "SGD Baseline (Quick)"),
            ("configs/zsharp_quick.yaml", "ZSharp (Quick)"),
        ]
    else:
        experiments = [
            ("configs/sgd_baseline.yaml", "SGD Baseline"),
            ("configs/zsharp_baseline.yaml", "ZSharp"),
            # Temporarily disabled for testing:
            # ("configs/cifar100_zsharp.yaml", "ZSharp CIFAR-100"),
            # ("configs/vit_zsharp.yaml", "ZSharp ViT"),
        ]

    results = {}

    logger.info("=" * 50)
    logger.info(f"Running {len(experiments)} experiments")
    logger.info("=" * 50)

    # Add progress bar for comparison experiments
    for i, (config_path, experiment_name) in enumerate(experiments, 1):
        logger.info(f"\n{i}. {experiment_name}")
        logger.info("-" * 30)

        start_time = time.time()
        output = run_experiment(config_path)
        end_time = time.time()

        if output:
            results[experiment_name] = {
                "config_path": config_path,
                "final_test_accuracy": output.get("final_test_accuracy", 0),
                "final_test_loss": output.get("final_test_loss", 0),
                "runtime": end_time - start_time,
                "status": "success",
                "history": [
                    {
                        "epoch": i + 1,
                        "train_accuracy": output.get("train_accuracies", [])[i]
                        if i < len(output.get("train_accuracies", []))
                        else 0,
                        "test_accuracy": output.get("test_accuracies", [])[i]
                        if i < len(output.get("test_accuracies", []))
                        else 0,
                    }
                    for i in range(len(output.get("test_accuracies", [])))
                ],
            }
        else:
            results[experiment_name] = {
                "config_path": config_path,
                "status": "failed",
            }

    # Print summary
    logger.info("=" * 50)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 50)
    for exp_name, result in results.items():
        if result["status"] == "success":
            logger.info(
                f"{exp_name}: {result['final_test_accuracy']:.2f}% accuracy"
            )
        else:
            logger.warning(f"{exp_name}: Failed")

    # Save comparison results
    comparison_file = f"{RESULTS_DIR}/comparison_results.json"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(comparison_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Detailed results saved to {comparison_file}")
    return results


def run_hyperparameter_study():
    """Run hyperparameter study for percentile threshold as mentioned in the paper"""
    percentiles = [50, 60, 70, 80, 90]
    results = {}

    for percentile in percentiles:
        logger.info("=" * 50)
        logger.info(f"Testing percentile: {percentile}")
        logger.info("=" * 50)

        # Create temporary config
        config = {
            "dataset": "cifar10",
            "model": "resnet18",
            "optimizer": {
                "type": "zsharp",
                "rho": DEFAULT_RHO,
                "percentile": percentile,
                "lr": DEFAULT_LEARNING_RATE,
                "momentum": DEFAULT_MOMENTUM,
                "weight_decay": DEFAULT_WEIGHT_DECAY,
            },
            "train": {
                "batch_size": DEFAULT_BATCH_SIZE,
                "epochs": 10,
                "device": "mps",
                "num_workers": DEFAULT_NUM_WORKERS,
                "pin_memory": False,
                "use_mixed_precision": True,
            },
        }

        # Save temporary config
        temp_config_path = f"configs/temp_percentile_{percentile}.yaml"
        with open(temp_config_path, "w") as f:
            yaml.dump(config, f)

        # Run experiment
        start_time = time.time()
        output = run_experiment(temp_config_path)
        end_time = time.time()

        if output:
            results[f"percentile_{percentile}"] = {
                "percentile": percentile,
                "final_test_accuracy": output.get("final_test_accuracy", 0),
                "final_test_loss": output.get("final_test_loss", 0),
                "runtime": end_time - start_time,
                "status": "success",
            }

        # Clean up
        os.remove(temp_config_path)
        logger.info("")

    # Print summary
    logger.info("=" * 50)
    logger.info("HYPERPARAMETER STUDY SUMMARY")
    logger.info("=" * 50)
    for percentile in percentiles:
        key = f"percentile_{percentile}"
        if key in results and results[key]["status"] == "success":
            acc = results[key]["final_test_accuracy"]
            logger.info(f"Percentile {percentile}%: {acc:.2f}% accuracy")

    # Save hyperparameter study results
    hp_file = f"{RESULTS_DIR}/hyperparameter_study.json"
    with open(hp_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Detailed results saved to {hp_file}")
    return results


def signal_handler(_sig, _frame):
    """Handle Ctrl+C gracefully by logging and exiting cleanly."""
    logger.warning("Interrupted by user. Cleaning up...")
    sys.exit(0)


if __name__ == "__main__":
    """Main entry point for running ZSharp paper reproduction experiments."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="ZSharp Paper Reproduction Experiments"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run in fast mode using test configs",
    )
    parser.add_argument(
        "--hp-study",
        action="store_true",
        help="Run hyperparameter study instead of comparison experiments",
    )
    args = parser.parse_args()

    # Set up signal handler for graceful interruption
    signal.signal(signal.SIGINT, signal_handler)

    try:
        if args.hp_study:
            # Run hyperparameter study
            hp_results = run_hyperparameter_study()
        else:
            # Run comparison experiments
            comparison_results = run_comparison_experiments(
                fast_mode=args.fast
            )

    except KeyboardInterrupt:
        logger.warning("Experiments interrupted by user.")
        sys.exit(0)
