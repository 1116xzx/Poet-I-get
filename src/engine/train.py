from __future__ import annotations

import argparse
import csv
import math
from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import PoemDataset, collate_examples
from src.data.tokenizer import Vocab
from src.models.char_rnn import CharPoemLM
from src.utils.common import choose_device, load_yaml, read_jsonl, seed_everything


def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer=None, grad_clip: float = 1.0) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_nll, total_tokens = 0.0, 0
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

    for batch in tqdm(loader, leave=False):
        inputs = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        loss_weights = batch["loss_weights"].to(device)
        with torch.set_grad_enabled(training):
            logits = model(inputs)
            token_loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1)).reshape_as(labels)
            valid = labels != -100
            nll = token_loss[valid].sum()
            tokens = int(valid.sum().item())
            if training:
                weighted_nll = (token_loss * loss_weights).sum()
                weight_sum = loss_weights[valid].sum()
                loss = weighted_nll / torch.clamp(weight_sum, min=1.0)
            else:
                loss = nll / max(tokens, 1)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        total_nll += float(nll.item())
        total_tokens += tokens

    mean_nll = total_nll / max(total_tokens, 1)
    return mean_nll, math.exp(min(mean_nll, 20.0))


def save_checkpoint(path: Path, model: CharPoemLM, vocab: Vocab, config: dict, epoch: int, val_ppl: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": config["model"],
            "vocab_tokens": vocab.tokens,
            "epoch": epoch,
            "val_ppl": val_ppl,
        },
        path,
    )


def load_model_weights(path: Path, model: CharPoemLM, device: torch.device) -> tuple[int, float]:
    bundle = torch.load(path, map_location=device)
    model.load_state_dict(bundle["model_state"])
    return int(bundle.get("epoch", 0)), float(bundle.get("val_ppl", "inf"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a unified multi-task character GRU/LSTM.")
    parser.add_argument("--config", default="configs/gru_base.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)
    seed_everything(config["seed"])
    device = choose_device(config.get("device", "auto"))
    data_dir = Path(config["data"]["data_dir"])
    vocab_path = Path(config["data"]["vocab_path"])
    vocab = Vocab.load(vocab_path)
    tasks = config["data"].get("tasks", ["free", "continue", "acrostic"])
    use_line_markers = config["data"].get("use_line_markers", True)
    acrostic_head_weight = float(config["data"].get("acrostic_head_weight", 1.0))

    train_set = PoemDataset(read_jsonl(data_dir / "train.jsonl"), vocab, tasks, use_line_markers=use_line_markers, acrostic_head_weight=acrostic_head_weight)
    valid_set = PoemDataset(read_jsonl(data_dir / "valid.jsonl"), vocab, tasks, use_line_markers=use_line_markers)
    collate = partial(collate_examples, pad_id=vocab.pad_id)
    train_loader = DataLoader(train_set, batch_size=config["train"]["batch_size"], shuffle=True, num_workers=config["train"]["num_workers"], collate_fn=collate)
    valid_loader = DataLoader(valid_set, batch_size=config["train"]["batch_size"], shuffle=False, num_workers=config["train"]["num_workers"], collate_fn=collate)

    model = CharPoemLM(vocab_size=len(vocab.tokens), **config["model"]).to(device)
    resume_from = config["train"].get("resume_from")
    start_epoch = 1
    best_ppl = float("inf")
    if resume_from:
        resumed_epoch, best_ppl = load_model_weights(Path(resume_from), model, device)
        start_epoch = resumed_epoch + 1
        print(f"resumed model weights from {resume_from} at epoch={resumed_epoch}")
    optimizer = AdamW(model.parameters(), lr=config["train"]["learning_rate"], weight_decay=config["train"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["train"]["epochs"])
    for _ in range(start_epoch - 1):
        scheduler.step()
    run_dir = Path(config["train"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(config["train"]["checkpoint_path"])

    metrics_path = run_dir / "metrics.csv"
    append_metrics = start_epoch > 1 and metrics_path.exists()
    with metrics_path.open("a" if append_metrics else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_ppl", "val_loss", "val_ppl", "learning_rate"])
        if not append_metrics:
            writer.writeheader()
        for epoch in range(start_epoch, config["train"]["epochs"] + 1):
            train_loss, train_ppl = run_epoch(model, train_loader, device, optimizer, config["train"]["grad_clip"])
            val_loss, val_ppl = run_epoch(model, valid_loader, device)
            scheduler.step()
            row = {"epoch": epoch, "train_loss": train_loss, "train_ppl": train_ppl, "val_loss": val_loss, "val_ppl": val_ppl, "learning_rate": scheduler.get_last_lr()[0]}
            writer.writerow(row)
            f.flush()
            print(f"epoch={epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_ppl={val_ppl:.3f}")
            if val_ppl < best_ppl:
                best_ppl = val_ppl
                save_checkpoint(checkpoint_path, model, vocab, config, epoch, val_ppl)
                print(f"saved best checkpoint -> {checkpoint_path}")


if __name__ == "__main__":
    main()
