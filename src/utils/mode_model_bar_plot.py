from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from src.engine.evaluate import STRATEGIES
from src.engine.generate import generate_poem, load_checkpoint
from src.metrics.poem_metrics import acrostic_ok, strict_format_ok
from src.utils.common import choose_device, read_jsonl, seed_everything


MODELS = [
    {"name": "baseline", "checkpoint": "checkpoints/gru_plain_best.pt", "structure_constraint": False},
    {"name": "weighted", "checkpoint": "checkpoints/gru_plain_weighted_best.pt", "structure_constraint": False},
    {"name": "structured", "checkpoint": "checkpoints/gru_best.pt", "structure_constraint": True},
]

STRATEGY_NOTE = "stable: T=0.7 | balanced: T=1.0 | creative: T=1.3"


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def collect_rows(data_dir: Path, seed: int, n_prompts: int) -> tuple[list[dict], list[dict]]:
    test_poems = read_jsonl(data_dir / "test.jsonl")[:n_prompts]
    continue_prompts = [poem["lines"][0] for poem in test_poems]
    acrostic_prompts = ["".join(line[0] for line in poem["lines"]) for poem in test_poems]

    device = choose_device("auto")
    continue_rows: list[dict] = []
    acrostic_rows: list[dict] = []

    for model_idx, model_cfg in enumerate(MODELS):
        model, vocab, _ = load_checkpoint(model_cfg["checkpoint"], device)
        for strategy_idx, (strategy_name, strategy) in enumerate(STRATEGIES.items()):
            generated_continue = []
            for prompt_idx, prompt in enumerate(continue_prompts):
                seed_everything(seed + model_idx * 10000 + strategy_idx * 1000 + prompt_idx)
                generated_continue.append(
                    generate_poem(
                        model,
                        vocab,
                        "continue",
                        prompt,
                        strategy,
                        structure_constraint=model_cfg["structure_constraint"],
                    )
                )

            generated_acrostic = []
            for prompt_idx, prompt in enumerate(acrostic_prompts):
                seed_everything(seed + 50000 + model_idx * 10000 + strategy_idx * 1000 + prompt_idx)
                generated_acrostic.append(
                    generate_poem(
                        model,
                        vocab,
                        "acrostic",
                        prompt,
                        strategy,
                        structure_constraint=model_cfg["structure_constraint"],
                    )
                )

            continue_rows.append(
                {
                    "model": model_cfg["name"],
                    "strategy": strategy_name,
                    "label": f"{model_cfg['name']}\n{strategy_name}",
                    "format_rate": mean([float(strict_format_ok(lines)) for lines in generated_continue]),
                }
            )
            acrostic_rows.append(
                {
                    "model": model_cfg["name"],
                    "strategy": strategy_name,
                    "label": f"{model_cfg['name']}\n{strategy_name}",
                    "format_rate": mean([float(strict_format_ok(lines)) for lines in generated_acrostic]),
                    "acrostic_rate": mean(
                        [float(acrostic_ok(lines, prompt)) for lines, prompt in zip(generated_acrostic, acrostic_prompts)]
                    ),
                }
            )

    return continue_rows, acrostic_rows


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def annotate(ax, bars, offset: float = 0.02) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:.3f}", ha="center", va="bottom", fontsize=8)


def plot_continue(rows: list[dict], out_path: Path) -> None:
    labels = [row["label"] for row in rows]
    values = [float(row["format_rate"]) for row in rows]
    colors = {"baseline": "#4C78A8", "weighted": "#72B7B2", "structured": "#F58518"}

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    bars = ax.bar(range(len(labels)), values, color=[colors[row["model"]] for row in rows])
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylim(0, 1.20)
    ax.set_ylabel("Format Rate")
    ax.set_title("Continue Mode: 3 Models x 3 Sampling Strategies")
    ax.grid(axis="y", alpha=0.2)
    ax.text(0.5, -0.18, STRATEGY_NOTE, transform=ax.transAxes, ha="center", va="top", fontsize=7)
    annotate(ax, bars, offset=0.02)
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_acrostic(rows: list[dict], out_path: Path) -> None:
    labels = [row["label"] for row in rows]
    format_values = [float(row["format_rate"]) for row in rows]
    acrostic_values = [float(row["acrostic_rate"]) for row in rows]

    x = list(range(len(labels)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    bars1 = ax.bar([i - width / 2 for i in x], format_values, width=width, label="format rate", color="#4C78A8")
    bars2 = ax.bar([i + width / 2 for i in x], acrostic_values, width=width, label="acrostic rate", color="#E45756")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.20)
    ax.set_ylabel("Rate")
    ax.set_title("Acrostic Mode: 3 Models x 3 Sampling Strategies")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.text(0.5, -0.18, STRATEGY_NOTE, transform=ax.transAxes, ha="center", va="top", fontsize=7)
    annotate(ax, bars1, offset=0.02)
    annotate(ax, bars2, offset=0.06)
    fig.tight_layout(rect=[0, 0.08, 0.88, 1])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed/qijue")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_prompts", type=int, default=100)
    parser.add_argument("--table_dir", default="runs/duibi/biaoge")
    parser.add_argument("--out_dir", default="runs/duibi/tupian")
    args = parser.parse_args()

    continue_rows, acrostic_rows = collect_rows(Path(args.data_dir), args.seed, args.n_prompts)
    table_dir = Path(args.table_dir)
    out_dir = Path(args.out_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_csv(table_dir / "xuxie_moshi_jiu_zuhe.csv", continue_rows)
    save_csv(table_dir / "cangtou_moshi_jiu_zuhe.csv", acrostic_rows)
    plot_continue(continue_rows, out_dir / "xuxie_moshi_jiu_zuhe.png")
    plot_acrostic(acrostic_rows, out_dir / "cangtou_moshi_jiu_zuhe.png")


if __name__ == "__main__":
    main()
