from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


MODEL_CONFIGS = [
    {
        "name": "baseline",
        "eval_path": "runs/moxing/jichu/evaluation.csv",
        "format_key": "raw_text_format_rate",
        "acrostic_key": "raw_acrostic_rate",
        "color": "#4C78A8",
    },
    {
        "name": "weighted",
        "eval_path": "runs/moxing/jiaquan/evaluation.csv",
        "format_key": "raw_text_format_rate",
        "acrostic_key": "raw_acrostic_rate",
        "color": "#72B7B2",
    },
    {
        "name": "structured",
        "eval_path": "runs/moxing/jiegou/evaluation.csv",
        "format_key": "constrained_format_rate",
        "acrostic_key": "constrained_acrostic_rate",
        "color": "#F58518",
    },
]

STRATEGY_NOTE = "stable: T=0.7 | balanced: T=1.0 | creative: T=1.3"


def load_rows(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def annotate(ax, bars, offset: float = 0.02) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:.3f}", ha="center", va="bottom", fontsize=8)


def plot_model(config: dict, out_dir: Path) -> None:
    rows = load_rows(config["eval_path"])
    labels = [row["strategy"] for row in rows]
    format_values = [float(row[config["format_key"]]) for row in rows]
    acrostic_values = [float(row[config["acrostic_key"]]) for row in rows]

    x = list(range(len(labels)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.6, 4.8))

    bars1 = ax.bar([i - width / 2 for i in x], format_values, width=width, label="format rate", color=config["color"], alpha=0.9)
    bars2 = ax.bar([i + width / 2 for i in x], acrostic_values, width=width, label="acrostic rate", color="#E45756", alpha=0.9)

    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.20)
    ax.set_ylabel("Rate")
    ax.set_title(f"{config['name'].title()} Model: Three Sampling Strategies")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.text(0.5, -0.18, STRATEGY_NOTE, transform=ax.transAxes, ha="center", va="top", fontsize=7)

    annotate(ax, bars1, offset=0.02)
    annotate(ax, bars2, offset=0.06)

    fig.tight_layout(rect=[0, 0.08, 0.88, 1])
    fig.savefig(out_dir / f"{config['name']}_san_caiyang_duibi.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="runs/duibi/tupian")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for config in MODEL_CONFIGS:
        plot_model(config, out_dir)


if __name__ == "__main__":
    main()
