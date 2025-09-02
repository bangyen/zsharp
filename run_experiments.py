# Comprehensive experiment runner for ZSharp paper reproduction
import yaml
import json
import os
import time
import signal
import sys
import argparse

# Import the training function directly
from src.train import train


def run_experiment(config_path, results_dir="results"):
    """Run a single experiment and save results"""
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    try:
        # Run training directly using the imported function
        results = train(config)
        return results

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        return None
    except Exception as e:
        print(f"Experiment failed with error: {e}")
        return None


def run_comparison_experiments(fast_mode=False):
    """Run comparison experiments as described in the ZSharp paper"""
    if fast_mode:
        experiments = [
            ("configs/test_sgd.yaml", "SGD Baseline (Test)"),
            ("configs/test.yaml", "ZSharp (Test)"),
        ]
    else:
        experiments = [
            ("configs/baseline_sgd.yaml", "SGD Baseline"),
            ("configs/default.yaml", "ZSharp"),
            # Temporarily disabled for testing:
            # ("configs/cifar100.yaml", "ZSharp CIFAR-100"),
            # ("configs/vit_experiment.yaml", "ZSharp ViT"),
        ]

    results = {}

    print(f"{'='*50}")
    print(f"Running {len(experiments)} experiments")
    print(f"{'='*50}")

    # Add progress bar for comparison experiments
    for i, (config_path, experiment_name) in enumerate(experiments, 1):
        print(f"\n{i}. {experiment_name}")
        print("-" * 30)

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
            }
        else:
            results[experiment_name] = {
                "config_path": config_path,
                "status": "failed",
            }

    # Print summary
    print(f"\n{'='*50}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*50}")
    for exp_name, result in results.items():
        if result["status"] == "success":
            print(f"{exp_name}: {result['final_test_accuracy']:.2f}% accuracy")
        else:
            print(f"{exp_name}: Failed")

    # Save comparison results
    comparison_file = "results/comparison_results.json"
    os.makedirs("results", exist_ok=True)
    with open(comparison_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to {comparison_file}")
    return results


def run_hyperparameter_study():
    """
    Run hyperparameter study for percentile threshold as mentioned in the paper
    """
    percentiles = [50, 60, 70, 80, 90]
    results = {}

    for percentile in percentiles:
        print(f"{'='*50}")
        print(f"Testing percentile: {percentile}")
        print(f"{'='*50}")

        # Create temporary config
        config = {
            "dataset": "cifar10",
            "model": "resnet18",
            "optimizer": {
                "type": "zsharp",
                "rho": 0.05,
                "percentile": percentile,
                "lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 5e-4,
            },
            "train": {
                "batch_size": 256,
                "epochs": 10,
                "device": "mps",
                "num_workers": 4,
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
        print("")

    # Print summary
    print(f"\n{'='*50}")
    print("HYPERPARAMETER STUDY SUMMARY")
    print(f"{'='*50}")
    for percentile in percentiles:
        key = f"percentile_{percentile}"
        if key in results and results[key]["status"] == "success":
            acc = results[key]["final_test_accuracy"]
            print(f"Percentile {percentile}%: {acc:.2f}% accuracy")

    # Save hyperparameter study results
    hp_file = "results/hyperparameter_study.json"
    with open(hp_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to {hp_file}")
    return results


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\nInterrupted by user. Cleaning up...")
    sys.exit(0)


if __name__ == "__main__":
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
        print("\n\nExperiments interrupted by user.")
        sys.exit(0)
