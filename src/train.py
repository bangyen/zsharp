import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse

from src.models import get_model
from src.data import get_cifar10
from src.optimizer import ZSharp


def train(config):
    device = config["train"]["device"] if torch.cuda.is_available() else "cpu"
    trainloader, testloader = get_cifar10(config["train"]["batch_size"])
    model = get_model(config["model"]).to(device)

    base_opt = optim.SGD
    optimizer = ZSharp(
        model.parameters(),
        base_optimizer=base_opt,
        rho=config["optimizer"]["rho"],
        lr=config["optimizer"]["lr"],
        momentum=config["optimizer"]["momentum"],
        weight_decay=config["optimizer"]["weight_decay"],
        percentile=config["optimizer"]["percentile"],
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(config["train"]["epochs"]):
        model.train()
        for i, (x, y) in enumerate(trainloader):
            x, y = x.to(device), y.to(device)

            # first forward-backward pass
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.first_step()

            # second forward-backward pass
            criterion(model(x), y).backward()
            optimizer.second_step()

        print(
            f"Epoch {epoch+1}/{config['train']['epochs']}"
            f": Loss {loss.item():.4f}"
        )

    # Evaluate
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    print(f"Test Accuracy: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config)
