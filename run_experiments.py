# Comprehensive experiment runner for ZSharp paper reproduction
import yaml
import json
import os
import subprocess
import time
from pathlib import Path


def run_experiment(config_path, results_dir="results"):
    """Run a single experiment and save results"""
    print(f"Running experiment with config: {config_path}")
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Run training
    cmd = f"PYTHONPATH=/Users/bangyen/Documents/repos/zsharp python src/train.py --config {config_path}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Experiment failed: {result.stderr}")
        return None
    
    print(f"Experiment completed successfully")
    return result.stdout


def run_comparison_experiments():
    """Run comparison experiments as described in the ZSharp paper"""
    experiments = [
        ("configs/baseline_sgd.yaml", "SGD Baseline"),
        ("configs/default.yaml", "ZSharp"),
        ("configs/cifar100.yaml", "ZSharp CIFAR-100"),
        ("configs/vit_experiment.yaml", "ZSharp ViT"),
    ]
    
    results = {}
    
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
    
    for percentile in percentiles:
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


if __name__ == "__main__":
    print("ZSharp Paper Reproduction Experiments")
    print("="*50)
    
    # Run comparison experiments
    print("\n1. Running comparison experiments...")
    comparison_results = run_comparison_experiments()
    
    # Run hyperparameter study
    print("\n2. Running hyperparameter study...")
    hp_results = run_hyperparameter_study()
    
    print("\nAll experiments completed!")
    print("Check the 'results' directory for detailed outputs.")
