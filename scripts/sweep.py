# Copyright (c) 2025 Bangyen Pham
"""Multi-seed sweep comparing SGD and ZSharp, meant for a GPU machine.

Each (config, seed) run is checkpointed and its result written to
``results/sweep/``; rerunning the same command skips finished runs and
resumes interrupted ones, so it survives preemption. A mean ± std table is
printed at the end.

Example (the 200-epoch comparison, with and without strong regularization):

    python -m scripts.sweep --epochs 200 --seeds 42 1 2
    python -m scripts.sweep --epochs 200 --seeds 42 1 2 --regularize
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

import yaml

from zsharp.constants import RESULTS_DIR, TrainingConfig
from zsharp.trainer import train

logger = logging.getLogger(__name__)

DEFAULT_CONFIGS = ["configs/sgd_baseline.yaml", "configs/zsharp_baseline.yaml"]


def build_config(path, seed, epochs, *, regularize, checkpoint_dir):
    """Load a YAML config and apply the sweep's overrides."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    raw["seed"] = seed
    train_cfg = raw.setdefault("train", {})
    train_cfg["epochs"] = epochs
    train_cfg["checkpoint_dir"] = str(checkpoint_dir)
    if regularize:
        train_cfg["label_smoothing"] = 0.1
        train_cfg["strong_augmentation"] = True
    return TrainingConfig.model_validate(raw)


def run_name(path, epochs, *, regularize):
    """Name a sweep variant, e.g. ``zsharp_baseline_e200_reg``."""
    suffix = "_reg" if regularize else ""
    return f"{Path(path).stem}_e{epochs}{suffix}"


def run_sweep(configs, seeds, epochs, *, regularize, out_dir):
    """Run every (config, seed) pair, skipping ones already finished.

    Returns:
        dict: Final test accuracies per variant name, in seed order.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    finals = {}
    for path in configs:
        name = run_name(path, epochs, regularize=regularize)
        # Checkpoint names don't encode epochs or regularization, so each
        # variant gets its own directory.
        ckpt_dir = out_dir / "checkpoints" / name
        for seed in seeds:
            result_path = out_dir / f"{name}_s{seed}.json"
            if not result_path.exists():
                logger.info("Running %s seed %d", name, seed)
                config = build_config(
                    path,
                    seed,
                    epochs,
                    regularize=regularize,
                    checkpoint_dir=ckpt_dir,
                )
                results = train(config)
                if results is None:
                    logger.warning("Interrupted; rerun to resume")
                    return finals
                result_path.write_text(
                    json.dumps(results.model_dump(), indent=2)
                )
            accuracy = json.loads(result_path.read_text())[
                "final_test_accuracy"
            ]
            finals.setdefault(name, []).append(accuracy)
    return finals


def format_table(finals):
    """Render final accuracies as mean ± std per variant."""
    lines = ["| Run | Seeds | Final test accuracy |", "|---|---|---|"]
    for name, accs in finals.items():
        std = statistics.stdev(accs) if len(accs) > 1 else 0.0
        mean = statistics.mean(accs)
        lines.append(f"| {name} | {len(accs)} | {mean:.2f} ± {std:.2f}% |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--regularize",
        action="store_true",
        help="Add label smoothing 0.1 and strong augmentation",
    )
    parser.add_argument("--out-dir", default=str(Path(RESULTS_DIR) / "sweep"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    finals = run_sweep(
        args.configs,
        args.seeds,
        args.epochs,
        regularize=args.regularize,
        out_dir=args.out_dir,
    )
    sys.stdout.write(format_table(finals) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
