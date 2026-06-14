from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def plot_metrics(csv_path: str | Path, out_dir: str | Path) -> None:
    with Path(csv_path).open("r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    train_ppl = [float(row["train_ppl"]) for row in rows]
    val_loss = [float(row["val_loss"]) for row in rows]
    val_ppl = [float(row["val_ppl"]) for row in rows]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_loss, label="train loss")
    plt.plot(epochs, val_loss, label="validation loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "loss_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_ppl, label="training PPL")
    best_epoch, best_ppl = min(zip(epochs, train_ppl), key=lambda item: item[1])
    plt.scatter([best_epoch], [best_ppl], color="#d62728", zorder=3, label=f"best {best_ppl:.3f}")
    plt.annotate(
        f"{best_ppl:.3f}",
        xy=(best_epoch, best_ppl),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        color="#d62728",
    )
    plt.xlabel("epoch")
    plt.ylabel("PPL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "train_ppl_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, val_ppl, label="validation PPL")
    best_epoch, best_ppl = min(zip(epochs, val_ppl), key=lambda item: item[1])
    plt.scatter([best_epoch], [best_ppl], color="#d62728", zorder=3, label=f"best {best_ppl:.3f}")
    plt.annotate(
        f"{best_ppl:.3f}",
        xy=(best_epoch, best_ppl),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        color="#d62728",
    )
    plt.xlabel("epoch")
    plt.ylabel("PPL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "val_ppl_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.4, 4.2))
    plt.plot(epochs, train_ppl, marker="o", markersize=3, label="train PPL", color="#4C78A8")
    plt.plot(epochs, val_ppl, marker="o", markersize=3, label="val PPL", color="#F58518")
    best_train_epoch, best_train_ppl = min(zip(epochs, train_ppl), key=lambda item: item[1])
    best_val_epoch, best_val_ppl = min(zip(epochs, val_ppl), key=lambda item: item[1])
    plt.scatter([best_train_epoch], [best_train_ppl], color="#4C78A8", s=36, zorder=3)
    plt.scatter([best_val_epoch], [best_val_ppl], color="#F58518", s=36, zorder=3)
    plt.annotate(f"{best_train_ppl:.3f}", xy=(best_train_epoch, best_train_ppl), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9, color="#4C78A8")
    plt.annotate(f"{best_val_ppl:.3f}", xy=(best_val_epoch, best_val_ppl), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9, color="#F58518")
    plt.xlabel("epoch")
    plt.ylabel("PPL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "train_val_ppl_curve.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    plot_metrics(args.metrics, args.out_dir)


if __name__ == "__main__":
    main()
