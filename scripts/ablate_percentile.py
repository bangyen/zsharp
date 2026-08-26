# Copyright (c) 2025 Bangyen Pham
"""A/B the Z-score percentile threshold with everything else held fixed.

Isolates the hyperparameter that moved most when the implementation was
aligned to the paper (70 -> 95), so the effect can be read without the
cost of a full 200-epoch reproduction. Each arm re-seeds identically, so
the only difference between runs is the threshold.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from zsharp.constants import RESULTS_DIR, TrainingConfig
from zsharp.trainer import train

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _build_config(
    percentile: int, epochs: int, model: str, dataset: str
) -> TrainingConfig:
    """Build a ZSharp config that varies only in the percentile.

    Args:
        percentile: Z-score filtering threshold.
        epochs: Number of training epochs.
        model: Model name.
        dataset: Dataset name.

    Returns:
        TrainingConfig: The validated configuration.
    """
    return TrainingConfig.model_validate(
        {
            "dataset": dataset,
            "model": model,
            "optimizer": {"type": "zsharp", "percentile": percentile},
            "train": {
                "epochs": epochs,
                "batch_size": 256,
                "device": "auto",
                "num_workers": 4,
            },
        }
    )


def main() -> None:
    """Run one arm per percentile and report the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--percentiles",
        type=int,
        nargs="+",
        default=[70, 95],
        help="Thresholds to compare (default: the old and new defaults)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model", default="resnet56")
    parser.add_argument("--dataset", default="cifar10")
    args = parser.parse_args()

    results = {}
    for percentile in args.percentiles:
        logger.info("=" * 60)
        logger.info(
            "percentile=%d  model=%s  dataset=%s  epochs=%d",
            percentile,
            args.model,
            args.dataset,
            args.epochs,
        )
        logger.info("=" * 60)

        start = time.time()
        output = train(
            _build_config(percentile, args.epochs, args.model, args.dataset)
        )
        if output is None:
            logger.warning("percentile=%d interrupted", percentile)
            continue

        results[percentile] = {
            "final_test_accuracy": output.final_test_accuracy,
            "final_test_loss": output.final_test_loss,
            "test_accuracies": output.test_accuracies,
            "runtime": time.time() - start,
        }

    logger.info("=" * 60)
    logger.info("PERCENTILE ABLATION")
    logger.info("=" * 60)
    for percentile, result in sorted(results.items()):
        logger.info(
            "percentile %3d: %.2f%%  (%.0fs)",
            percentile,
            result["final_test_accuracy"],
            result["runtime"],
        )
    if len(results) > 1:
        best = max(results, key=lambda k: results[k]["final_test_accuracy"])
        worst = min(results, key=lambda k: results[k]["final_test_accuracy"])
        delta = (
            results[best]["final_test_accuracy"]
            - results[worst]["final_test_accuracy"]
        )
        logger.info("best: %d (+%.2f%% over %d)", best, delta, worst)

    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "percentile_ablation.json"
    with out_path.open("w") as f:
        json.dump(
            {
                "model": args.model,
                "dataset": args.dataset,
                "epochs": args.epochs,
                "results": results,
            },
            f,
            indent=2,
        )
    logger.info("saved to %s", out_path)


if __name__ == "__main__":
    main()
