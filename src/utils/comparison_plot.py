from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_comparison(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_metrics(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def pretty_label(setting: str) -> str:
    mapping = {
        "baseline": "Baseline",
        "weighted": "Weighted",
        "structured": "Structured",
        "plain_baseline": "Plain\nbaseline",
        "structured_training": "Structured\nraw",
        "structured_plus_constrained": "Structured\nconstrained",
    }
    return mapping.get(setting, setting.replace("_", "\n"))


def annotate_bars(ax, bars, fmt: str, offset: float = 0.01) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            format(height, fmt),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_main_comparison(rows: list[dict], out_path: str | Path) -> None:
    labels = [pretty_label(row["setting"]) for row in rows]
    fmt_rate = [float(row["text_format_rate"]) for row in rows]
    acro = [float(row["acrostic_rate"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.4))
    colors = ["#4C78A8", "#72B7B2", "#F58518"]

    bars = axes[0].bar(labels, fmt_rate, color=colors)
    axes[0].set_title("Format Rate")
    axes[0].set_ylim(0, 1.12)
    annotate_bars(axes[0], bars, ".3f", offset=0.02)

    bars = axes[1].bar(labels, acro, color=colors)
    axes[1].set_title("Acrostic Rate")
    axes[1].set_ylim(0, 1.12)
    annotate_bars(axes[1], bars, ".3f", offset=0.02)

    for ax in axes:
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle(f"Three Models Comparison ({rows[0]['strategy']})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_structure_comparison(rows: list[dict], out_path: str | Path) -> None:
    labels = [pretty_label(row["setting"]) for row in rows]
    fmt_rate = [float(row["text_format_rate"]) for row in rows]
    acro = [float(row["acrostic_rate"]) for row in rows]
    leak = [float(row.get("control_leak_rate", 0.0)) for row in rows]

    x = list(range(len(labels)))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.2, 4.8))

    bars1 = ax.bar([i - width for i in x], fmt_rate, width=width, label="format", color="#4C78A8")
    bars2 = ax.bar(x, acro, width=width, label="acrostic", color="#F58518")
    bars3 = ax.bar([i + width for i in x], leak, width=width, label="control leak", color="#E45756")

    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.14)
    ax.set_title("Generation Quality Comparison")
    ax.legend(ncols=3, frameon=False, loc="upper center")
    ax.grid(axis="y", alpha=0.2)

    annotate_bars(ax, bars1, ".3f", offset=0.02)
    annotate_bars(ax, bars2, ".3f", offset=0.02)
    annotate_bars(ax, bars3, ".3f", offset=0.02)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_val_ppl_compare(metric_paths: list[tuple[str, str | Path]], out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    colors = ["#4C78A8", "#72B7B2", "#F58518"]
    all_epochs: list[int] = []
    for idx, (label, metrics_path) in enumerate(metric_paths):
        rows = load_metrics(metrics_path)
        epochs = [int(row["epoch"]) for row in rows]
        ppl = [float(row["val_ppl"]) for row in rows]
        all_epochs.extend(epochs)
        best = min(zip(epochs, ppl), key=lambda item: item[1])
        color = colors[idx % len(colors)]
        ax.plot(epochs, ppl, marker="o", markersize=3, label=label, color=color)
        ax.scatter(*best, color=color, s=40, zorder=3)
        ax.text(best[0], best[1] + 0.8, f"{best[1]:.3f}", ha="center", fontsize=9)

    min_epoch = min(all_epochs)
    max_epoch = max(all_epochs)
    tick_step = 5 if max_epoch > 15 else 1
    ax.set_xlim(min_epoch - 0.5, max_epoch + 0.5)
    ax.set_xticks(list(range(min_epoch, max_epoch + 1, tick_step)))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation PPL")
    ax.set_title("Validation PPL Curves")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", default="runs/duibi/biaoge/san_moxing_duibi.json")
    parser.add_argument("--baseline_metrics", default="runs/moxing/jichu/metrics.csv")
    parser.add_argument("--weighted_metrics", default="runs/moxing/jiaquan/metrics.csv")
    parser.add_argument("--structured_metrics", default="runs/moxing/jiegou/metrics.csv")
    parser.add_argument("--out_dir", default="runs/duibi/tupian")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_comparison(args.comparison)
    plot_main_comparison(rows, out_dir / "san_moxing_zhuyao_duibi.png")
    plot_structure_comparison(rows, out_dir / "san_moxing_zhiliang_duibi.png")
    plot_val_ppl_compare(
        [
            ("baseline", args.baseline_metrics),
            ("weighted", args.weighted_metrics),
            ("structured", args.structured_metrics),
        ],
        out_dir / "san_moxing_ppl_duibi.png",
    )


if __name__ == "__main__":
    main()
