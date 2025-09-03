import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

from src.data import get_dataset
from src.models import get_model
from src.optimizer import ZSharp
from src.utils import set_seed


def get_device(config):
    """Get the best available device for training"""
    device_config = config["train"]["device"]

    if device_config == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    elif device_config == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def train(config):
    # Set seed for reproducibility
    set_seed(42)

    device = get_device(config)

    # Get training parameters
    batch_size = int(config["train"]["batch_size"])
    num_workers = int(config["train"].get("num_workers", 2))
    pin_memory = config["train"].get("pin_memory", False)
    use_mixed_precision = config["train"].get("use_mixed_precision", False)

    # Get dataset
    dataset_name = config.get("dataset", "cifar10")
    trainloader, testloader = get_dataset(
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Get model
    model_name = config.get("model", "resnet18")
    num_classes = 100 if dataset_name == "cifar100" else 10
    model = get_model(model_name, num_classes=num_classes).to(device)

    # Enable mixed precision for faster training on Apple Silicon (optional)
    if device.type == "mps" and use_mixed_precision:
        model = model.half()  # Use float16 for better performance

    # Setup optimizer based on config
    optimizer_type = config["optimizer"].get("type", "zsharp")

    if optimizer_type == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=float(config["optimizer"]["lr"]),
            momentum=float(config["optimizer"]["momentum"]),
            weight_decay=float(config["optimizer"]["weight_decay"]),
        )
        use_zsharp = False
    else:
        # ZSharp optimizer
        base_opt = optim.SGD
        optimizer = ZSharp(
            model.parameters(),
            base_optimizer=base_opt,
            rho=float(config["optimizer"]["rho"]),
            lr=float(config["optimizer"]["lr"]),
            momentum=float(config["optimizer"]["momentum"]),
            weight_decay=float(config["optimizer"]["weight_decay"]),
            percentile=int(config["optimizer"]["percentile"]),
        )
        use_zsharp = True

    criterion = nn.CrossEntropyLoss()

    # Training loop with timing
    start_time = time.time()
    train_losses = []
    train_accuracies = []

    for epoch in range(int(config["train"]["epochs"])):
        model.train()
        epoch_loss = 0.0
        correct, total = 0, 0

        desc = f"Epoch {epoch+1}/{config['train']['epochs']}"
        pbar = tqdm(trainloader, desc=desc)

        for i, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)

            # Convert to half precision if using MPS and mixed precision is
            # enabled
            if device.type == "mps" and use_mixed_precision:
                x = x.half()

            if use_zsharp:
                # ZSharp two-step training
                loss = criterion(model(x), y)
                loss.backward()

                # Add gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )

                optimizer.first_step()

                criterion(model(x), y).backward()
                optimizer.second_step()
            else:
                # Standard SGD training
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()

                # Add gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )

                optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            # Update progress bar
            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Acc": f"{100 * correct / total:.2f}%",
                }
            )

        # Record epoch metrics
        avg_loss = epoch_loss / len(trainloader)
        train_accuracy = 100 * correct / total
        train_losses.append(avg_loss)
        train_accuracies.append(train_accuracy)

    total_time = time.time() - start_time

    # Evaluate on test set
    model.eval()
    test_correct, test_total = 0, 0
    test_loss = 0.0

    with torch.no_grad():
        eval_pbar = tqdm(testloader, desc="Evaluating")
        for x, y in eval_pbar:
            x, y = x.to(device), y.to(device)
            if device.type == "mps" and use_mixed_precision:
                x = x.half()

            outputs = model(x)
            loss = criterion(outputs, y)
            test_loss += loss.item()

            preds = outputs.argmax(dim=1)
            test_correct += (preds == y).sum().item()
            test_total += y.size(0)

            eval_pbar.set_postfix(
                {"Test Acc": f"{100 * test_correct / test_total:.2f}%"}
            )

    test_accuracy = 100 * test_correct / test_total
    avg_test_loss = test_loss / len(testloader)

    # Save results
    results = {
        "config": config,
        "final_test_accuracy": test_accuracy,
        "final_test_loss": avg_test_loss,
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "total_training_time": total_time,
        "device": str(device),
        "optimizer_type": optimizer_type,
    }

    # Save results to file
    results_file = (
        f"results/zsharp_{dataset_name}_{model_name}_{optimizer_type}.json"
    )

    os.makedirs("results", exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/zsharp_baseline.yaml"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config)
