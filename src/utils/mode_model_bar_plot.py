from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


MODELS = [
    {
        "name": "baseline",
        "eval_path": "runs/moxing/jichu/evaluation.csv",
        "format_key": "raw_text_format_rate",
        "acrostic_key": "raw_acrostic_rate",
    },
    {
        "name": "weighted",
        "eval_path": "runs/moxing/jiaquan/evaluation.csv",
        "format_key": "raw_text_format_rate",
        "acrostic_key": "raw_acrostic_rate",
    },
    {
        "name": "structured",
        "eval_path": "runs/moxing/jiegou/evaluation.csv",
        "format_key": "constrained_format_rate",
        "acrostic_key": "constrained_acrostic_rate",
    },
]

STRATEGY_NOTE = "stable: T=0.7 | balanced: T=1.0 | creative: T=1.3"


def load_rows(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def annotate(ax, bars, offset: float = 0.02) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:.3f}", ha="center", va="bottom", fontsize=8)


def collect_rows() -> tuple[list[dict], list[dict]]:
    continue_rows: list[dict] = []
    acrostic_rows: list[dict] = []
    for model_cfg in MODELS:
        eval_rows = load_rows(model_cfg["eval_path"])
        for row in eval_rows:
            label = f"{model_cfg['name']}\n{row['strategy']}"
            continue_rows.append(
                {
                    "model": model_cfg["name"],
                    "strategy": row["strategy"],
                    "label": label,
                    "format_rate": row[model_cfg["format_key"]],
                }
            )
            acrostic_rows.append(
                {
                    "model": model_cfg["name"],
                    "strategy": row["strategy"],
                    "label": label,
                    "format_rate": row[model_cfg["format_key"]],
                    "acrostic_rate": row[model_cfg["acrostic_key"]],
                }
            )
    return continue_rows, acrostic_rows


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
    parser.add_argument("--table_dir", default="runs/duibi/biaoge")
    parser.add_argument("--out_dir", default="runs/duibi/tupian")
    args = parser.parse_args()

    continue_rows, acrostic_rows = collect_rows()
    table_dir = Path(args.table_dir)
    out_dir = Path(args.out_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_csv(table_dir / "xuxie_moshi_jiu_zuhe.csv", continue_rows, ["model", "strategy", "label", "format_rate"])
    save_csv(
        table_dir / "cangtou_moshi_jiu_zuhe.csv",
        acrostic_rows,
        ["model", "strategy", "label", "format_rate", "acrostic_rate"],
    )
    plot_continue(continue_rows, out_dir / "xuxie_moshi_jiu_zuhe.png")
    plot_acrostic(acrostic_rows, out_dir / "cangtou_moshi_jiu_zuhe.png")


if __name__ == "__main__":
    main()
