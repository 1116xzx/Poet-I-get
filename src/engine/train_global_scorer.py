from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.global_scorer_dataset import GlobalMaskDataset, collate_global_mask_examples
from src.data.tokenizer import Vocab
from src.models.global_bigru_scorer import GlobalBiGRUScorer
from src.utils.common import choose_device, load_yaml, read_jsonl, seed_everything


def run_epoch(model, loader, device, optimizer=None, grad_clip: float = 1.0):
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss(reduction="none")
    total_loss = 0.0
    total_weight = 0.0
    total_correct = 0
    total_count = 0

    for batch in tqdm(loader, leave=False):
        input_ids = batch["input_ids"].to(device)
        row_ids = batch["row_ids"].to(device)
        col_ids = batch["col_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        target_pos = batch["target_pos"].to(device)
        target_weights = batch["target_weights"].to(device)
        with torch.set_grad_enabled(training):
            logits = model(input_ids, row_ids, col_ids, target_pos)
            losses = criterion(logits, target_ids)
            weighted_loss = (losses * target_weights).sum() / torch.clamp(target_weights.sum(), min=1.0)
            if training:
                optimizer.zero_grad(set_to_none=True)
                weighted_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        total_loss += float((losses * target_weights).sum().item())
        total_weight += float(target_weights.sum().item())
        total_correct += int((logits.argmax(dim=-1) == target_ids).sum().item())
        total_count += int(target_ids.numel())

    return total_loss / max(total_weight, 1.0), total_correct / max(total_count, 1)


def save_checkpoint(path: Path, model: GlobalBiGRUScorer, vocab: Vocab, config: dict, epoch: int, val_loss: float, val_acc: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": config["model"],
            "vocab_tokens": vocab.tokens,
            "epoch": epoch,
            "val_loss": val_loss,
            "val_acc": val_acc,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a BiGRU global position scorer for 4x7 qijue grids.")
    parser.add_argument("--config", default="configs/global_bigru_scorer.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)
    seed_everything(config["seed"])
    device = choose_device(config.get("device", "auto"))
    data_dir = Path(config["data"]["data_dir"])
    vocab = Vocab.load(config["data"]["vocab_path"])
    train_poems = read_jsonl(data_dir / "train.jsonl")
    valid_poems = read_jsonl(data_dir / "valid.jsonl")
    max_train_poems = int(config["data"].get("max_train_poems", 0))
    max_valid_poems = int(config["data"].get("max_valid_poems", 0))
    if max_train_poems > 0:
        train_poems = train_poems[:max_train_poems]
    if max_valid_poems > 0:
        valid_poems = valid_poems[:max_valid_poems]

    train_set = GlobalMaskDataset(
        train_poems,
        vocab,
        samples_per_poem=int(config["data"].get("train_samples_per_poem", 2)),
        seed=int(config["seed"]),
    )
    valid_set = GlobalMaskDataset(
        valid_poems,
        vocab,
        samples_per_poem=int(config["data"].get("valid_samples_per_poem", 1)),
        seed=int(config["seed"]) + 999,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["train"]["num_workers"]),
        collate_fn=collate_global_mask_examples,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["train"]["num_workers"]),
        collate_fn=collate_global_mask_examples,
    )

    model = GlobalBiGRUScorer(vocab_size=len(vocab.tokens), **config["model"]).to(device)
    optimizer = AdamW(model.parameters(), lr=float(config["train"]["learning_rate"]), weight_decay=float(config["train"]["weight_decay"]))
    run_dir = Path(config["train"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(config["train"]["checkpoint_path"])
    best_val_loss = float("inf")

    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writeheader()
        for epoch in range(1, int(config["train"]["epochs"]) + 1):
            train_loss, train_acc = run_epoch(model, train_loader, device, optimizer, float(config["train"]["grad_clip"]))
            val_loss, val_acc = run_epoch(model, valid_loader, device)
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
            )
            f.flush()
            print(
                f"epoch={epoch:02d} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(checkpoint_path, model, vocab, config, epoch, val_loss, val_acc)
                print(f"saved best checkpoint -> {checkpoint_path}")


if __name__ == "__main__":
    main()
