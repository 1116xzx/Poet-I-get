from __future__ import annotations

import argparse
import csv
import math
import random
from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import PoemDataset, collate_examples
from src.data.tokenizer import PUNCT_TOKENS
from src.engine.generate import SamplingConfig, generate_poem, generate_poem_trace, load_checkpoint
from src.metrics.poem_metrics import CopyRiskIndex, acrostic_ok, distinct_n, repeat_rate, strict_format_ok
from src.utils.common import choose_device, read_jsonl, save_json, seed_everything

STRATEGIES = {
    "stable": SamplingConfig(temperature=0.7, top_k=0, top_p=1.0),
    "balanced": SamplingConfig(temperature=0.9, top_k=20, top_p=1.0),
    "creative": SamplingConfig(temperature=1.1, top_k=0, top_p=0.95),
}


def has_line_markers(vocab) -> bool:
    return all(token in vocab.stoi for token in ("<L1>", "<L2>", "<L3>", "<L4>"))


def test_ppl(model, vocab, poems, device) -> float:
    dataset = PoemDataset(poems, vocab, tasks=["free"], use_line_markers=has_line_markers(vocab))
    loader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=partial(collate_examples, pad_id=vocab.pad_id))
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    total_nll, total_tokens = 0.0, 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)
            logits = model(batch["input_ids"].to(device))
            total_nll += float(criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1)).item())
            total_tokens += int((labels != -100).sum().item())
    return math.exp(min(total_nll / max(total_tokens, 1), 20.0))


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def expected_puncts_for_mode(poem: dict, mode: str) -> list[str]:
    puncts = list(poem.get("puncts", ["", "", "", ""]))
    puncts = (puncts + ["", "", "", ""])[:4]
    return puncts[1:] if mode == "continue" else puncts


def strict_raw_text_format_ok(mode: str, prompt: str, trace, expected_puncts: list[str]) -> bool:
    if trace.invalid_token:
        return False
    puncts = [token for token in trace.tokens if token in PUNCT_TOKENS]
    if mode == "continue":
        lines = [prompt, *trace.lines]
    else:
        lines = trace.lines
    return puncts == expected_puncts and strict_format_ok(lines)


def evaluate_strategy(model, vocab, prompts: list[dict], train_index: CopyRiskIndex, strategy: SamplingConfig, seed: int) -> dict:
    rows = []
    for i, poem in enumerate(prompts):
        for mode, prompt in (("continue", poem["lines"][0]), ("acrostic", "".join(line[0] for line in poem["lines"]))):
            seed_everything(seed + i)
            raw_trace = generate_poem_trace(model, vocab, mode, prompt, strategy)
            raw_lines = [prompt, *raw_trace.lines] if mode == "continue" else raw_trace.lines
            constrained_lines = generate_poem(model, vocab, mode, prompt, strategy, structure_constraint=True)
            text = "".join(raw_lines)
            rows.append(
                {
                    "mode": mode,
                    "raw_text_format_ok": float(strict_raw_text_format_ok(mode, prompt, raw_trace, expected_puncts_for_mode(poem, mode))),
                    "raw_control_leak": float(raw_trace.leaked_control_token),
                    "constrained_format_ok": float(strict_format_ok(constrained_lines)),
                    "raw_acrostic_ok": float(acrostic_ok(raw_lines, prompt)) if mode == "acrostic" else None,
                    "constrained_acrostic_ok": float(acrostic_ok(constrained_lines, prompt)) if mode == "acrostic" else None,
                    "distinct_2": distinct_n(text, 2),
                    "repeat_rate": repeat_rate(text, 2),
                    "nearest_4gram_jaccard": train_index.max_jaccard(text),
                }
            )
    raw_acro = [row["raw_acrostic_ok"] for row in rows if row["raw_acrostic_ok"] is not None]
    constrained_acro = [row["constrained_acrostic_ok"] for row in rows if row["constrained_acrostic_ok"] is not None]
    return {
        "raw_text_format_rate": mean([row["raw_text_format_ok"] for row in rows]),
        "raw_control_leak_rate": mean([row["raw_control_leak"] for row in rows]),
        "constrained_format_rate": mean([row["constrained_format_ok"] for row in rows]),
        "raw_acrostic_rate": mean(raw_acro),
        "constrained_acrostic_rate": mean(constrained_acro),
        "distinct_2": mean([row["distinct_2"] for row in rows]),
        "repeat_rate": mean([row["repeat_rate"] for row in rows]),
        "nearest_4gram_jaccard": mean([row["nearest_4gram_jaccard"] for row in rows]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PPL and constrained-generation metrics.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default="data/processed/qijue")
    parser.add_argument("--n_prompts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="runs/evaluation.csv")
    args = parser.parse_args()

    device = choose_device(args.device)
    model, vocab, _ = load_checkpoint(args.checkpoint, device)
    data_dir = Path(args.data_dir)
    train_poems = read_jsonl(data_dir / "train.jsonl")
    test_poems = read_jsonl(data_dir / "test.jsonl")
    rng = random.Random(args.seed)
    prompts = rng.sample(test_poems, k=min(args.n_prompts, len(test_poems)))
    ppl = test_ppl(model, vocab, test_poems, device)
    copy_index = CopyRiskIndex([item["text"] for item in train_poems])

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, strategy in STRATEGIES.items():
        row = {"strategy": name, "test_ppl": ppl, **evaluate_strategy(model, vocab, prompts, copy_index, strategy, args.seed)}
        rows.append(row)
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(rows, output.with_suffix(".json"))
    print(f"saved evaluation -> {output}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
