# Copyright (c) 2025 Bangyen Pham
"""Plot the README's CIFAR-10 training curves (``docs/training_curves.png``).

Left panel: test accuracy per epoch. Right panel: the same curves against
wall-clock time, each cut at SGD's 200-epoch budget.

Usage:
    python -m scripts.plot_curves --baseline results/sweep \
        --fast results/sweep_fast200 --out docs/training_curves.png \
        --sgd-seconds 12329.35 12478.16
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

EPOCHS = 200
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1a9e6e"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load(directory: Path, pattern: str) -> list[dict]:
    """Load every sweep result in ``directory`` matching ``pattern``."""
    return [json.loads(p.read_text()) for p in sorted(directory.glob(pattern))]


def hours_per_epoch(runs: list[dict]) -> float:
    """Mean wall-clock hours per epoch across runs."""
    return (
        float(np.mean([r["total_training_time"] for r in runs]))
        / EPOCHS
        / 3600
    )


def accuracies(runs: list[dict]) -> np.ndarray:
    """Per-epoch test accuracy, one row per run."""
    return np.array([r["test_accuracies"] for r in runs])


def plot(
    baseline: Path, fast: Path, out: Path, sgd_seconds: list[float]
) -> None:
    """Draw both panels and save the figure to ``out``."""
    zsharp = load(baseline, f"zsharp_baseline_e{EPOCHS}_s*.json")
    sgd = load(baseline, f"sgd_baseline_e{EPOCHS}_s*.json")
    every4 = load(fast, f"zsharp_fast_e{EPOCHS}_s*.json")

    sgd_h = float(np.mean(sgd_seconds)) / EPOCHS / 3600
    z_h, k_h = hours_per_epoch(zsharp), hours_per_epoch(every4)
    z_epochs, k_epochs = int(EPOCHS * sgd_h / z_h), int(EPOCHS * sgd_h / k_h)
    z, s, k = accuracies(zsharp), accuracies(sgd), accuracies(every4)

    plt.rcParams.update(
        {
            "font.size": 11,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": GRID,
        }
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4.6), sharey=True, facecolor=SURFACE
    )
    ep = np.arange(1, EPOCHS + 1)
    panels = [
        (
            axes[0],
            f"Per epoch ({EPOCHS} epochs each)",
            "Epoch",
            [
                ("ZSharp", "ZSharp", ep, z, BLUE),
                ("ZSharp, SAM every 4th batch", "Every 4th", ep, k, GREEN),
                ("SGD", "SGD", ep, s, ORANGE),
            ],
        ),
        (
            axes[1],
            (
                f"Equal compute (SGD {EPOCHS} epochs, ZSharp {z_epochs}, "
                f"every-4th {k_epochs})"
            ),
            "Wall-clock hours (CPU)",
            [
                (
                    "ZSharp",
                    "ZSharp",
                    ep[:z_epochs] * z_h,
                    z[:, :z_epochs],
                    BLUE,
                ),
                (
                    "ZSharp, SAM every 4th batch",
                    "Every 4th",
                    ep[:k_epochs] * k_h,
                    k[:, :k_epochs],
                    GREEN,
                ),
                ("SGD", "SGD", ep * sgd_h, s, ORANGE),
            ],
        ),
    ]
    for ax, title, xlabel, series in panels:
        ax.set_facecolor(SURFACE)
        ax.grid(visible=True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(title, loc="left", fontsize=12, color=INK)
        ax.set_xlabel(xlabel)
        finals = []
        for label, short, xs, ys, color in series:
            mean, std = ys.mean(0), ys.std(0, ddof=1)
            ax.fill_between(
                xs, mean - std, mean + std, color=color, alpha=0.18, lw=0
            )
            ax.plot(
                xs,
                mean,
                color=color,
                lw=1.6 if short == "ZSharp" else 2,
                label=label,
            )
            finals.append(f"{short} {mean[-1]:.2f}%")
        ax.text(
            0.98,
            0.32,
            "\n".join(finals),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            color=INK,
            linespacing=1.6,
        )
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_ylim(45, 88)
    axes[0].legend(frameon=False, loc="lower left", bbox_to_anchor=(0.06, 0.0))
    fig.suptitle(
        "CIFAR-10 ResNet-18, mean ± std over 3 seeds",
        x=0.06,
        ha="left",
        fontsize=13,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    logger.info(
        "Equal-compute epochs: ZSharp %d, every-4th %d", z_epochs, k_epochs
    )


def main() -> None:
    """Parse arguments and plot."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", type=Path, default=Path("results/sweep"))
    parser.add_argument(
        "--fast", type=Path, default=Path("results/sweep_fast200")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("docs/training_curves.png")
    )
    parser.add_argument(
        "--sgd-seconds",
        type=float,
        nargs="+",
        required=True,
        help="SGD 200-epoch training times to use as the compute budget",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    plot(args.baseline, args.fast, args.out, args.sgd_seconds)


if __name__ == "__main__":
    main()
