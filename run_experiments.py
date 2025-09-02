# Comprehensive experiment runner for ZSharp paper reproduction
import yaml
import json
import os
import time
import signal
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

# Import the training function directly
from src.train import train


def run_experiment(config_path, results_dir="results"):
    """Run a single experiment and save results"""
    print(f"Running experiment with config: {config_path}")
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    try:
        # Run training directly using the imported function
        # This will preserve all progress bars and output
        train(config)
        
        print(f"Experiment completed successfully")
        return "Experiment completed successfully"
        
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
    
    # Add progress bar for comparison experiments
    for config_path, experiment_name in experiments:
        print(f"\n{'='*50}")
        print(f"Running: {experiment_name}")
        print(f"{'='*50}")
        
        start_time = time.time()
        output = run_experiment(config_path)
        end_time = time.time()
        
        if output:
            results[experiment_name] = {
                "config_path": config_path,
                "output": output,
                "runtime": end_time - start_time,
                "status": "success"
            }
        else:
            results[experiment_name] = {
                "config_path": config_path,
                "status": "failed"
            }
    
    # Save comparison results
    comparison_file = "results/comparison_results.json"
    os.makedirs("results", exist_ok=True)
    with open(comparison_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nComparison results saved to {comparison_file}")
    return results


def run_hyperparameter_study():
    """Run hyperparameter study for percentile threshold as mentioned in the paper"""
    percentiles = [50, 60, 70, 80, 90]
    results = {}
    
    # Add progress bar for hyperparameter study
    for percentile in tqdm(percentiles, desc="Hyperparameter Study", unit="percentile"):
        print(f"\n{'='*50}")
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
                "weight_decay": 5e-4
            },
            "train": {
                "batch_size": 256,
                "epochs": 10,
                "device": "mps",
                "num_workers": 4,
                "pin_memory": False,
                "use_mixed_precision": True
            }
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
                "output": output,
                "runtime": end_time - start_time,
                "status": "success"
            }
        
        # Clean up
        os.remove(temp_config_path)
    
    # Save hyperparameter study results
    hp_file = "results/hyperparameter_study.json"
    with open(hp_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nHyperparameter study results saved to {hp_file}")
    return results


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print('\n\nInterrupted by user. Cleaning up...')
    sys.exit(0)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="ZSharp Paper Reproduction Experiments")
    parser.add_argument("--fast", action="store_true", help="Run in fast mode using test configs")
    parser.add_argument("--hp-study", action="store_true", help="Run hyperparameter study instead of comparison experiments")
    args = parser.parse_args()
    
    # Set up signal handler for graceful interruption
    signal.signal(signal.SIGINT, signal_handler)
    
    print("ZSharp Paper Reproduction Experiments")
    print("="*50)
    if args.fast:
        print("Running in FAST MODE (using test configs)")
        print("="*50)
    if args.hp_study:
        print("Running HYPERPARAMETER STUDY")
        print("="*50)
    
    try:
        if args.hp_study:
            # Run hyperparameter study
            print("\nRunning hyperparameter study...")
            hp_results = run_hyperparameter_study()
            print("\nHyperparameter study completed!")
        else:
            # Run comparison experiments
            print("\n1. Running comparison experiments...")
            comparison_results = run_comparison_experiments(fast_mode=args.fast)
            print("\nComparison experiments completed!")
        
        print("Check the 'results' directory for detailed outputs.")
        
    except KeyboardInterrupt:
        print('\n\nExperiments interrupted by user.')
        sys.exit(0)
