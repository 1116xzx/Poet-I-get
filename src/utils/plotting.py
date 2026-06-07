from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def plot_metrics(csv_path: str | Path, out_dir: str | Path) -> None:
    with Path(csv_path).open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
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
    plt.plot(epochs, val_ppl, label="validation PPL")
    plt.xlabel("epoch")
    plt.ylabel("PPL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "val_ppl_curve.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    plot_metrics(args.metrics, args.out_dir)


if __name__ == "__main__":
    main()
