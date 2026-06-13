from __future__ import annotations

import math
from pathlib import Path

import torch

from src.data.global_scorer_dataset import COL_IDS, ROW_IDS, SEQ_LEN
from src.data.tokenizer import Vocab
from src.models.global_bigru_scorer import GlobalBiGRUScorer


def load_global_scorer(path: str | Path, device: torch.device):
    bundle = torch.load(Path(path), map_location=device)
    vocab = Vocab(bundle["vocab_tokens"])
    model = GlobalBiGRUScorer(vocab_size=len(vocab.tokens), **bundle["model_config"]).to(device)
    model.load_state_dict(bundle["model_state"])
    model.eval()
    return model, vocab, bundle


@torch.no_grad()
def pseudo_log_likelihood(lines: list[str], model: GlobalBiGRUScorer, vocab: Vocab, device: torch.device) -> float:
    chars = [ch for line in lines for ch in line]
    if len(chars) != SEQ_LEN:
        raise ValueError("expected four lines of seven characters each")
    ids = vocab.encode(chars)
    row_ids = torch.tensor([ROW_IDS], dtype=torch.long, device=device)
    col_ids = torch.tensor([COL_IDS], dtype=torch.long, device=device)
    total_log_prob = 0.0
    for pos, target_id in enumerate(ids):
        masked = list(ids)
        masked[pos] = model.mask_id
        logits = model(
            torch.tensor([masked], dtype=torch.long, device=device),
            row_ids,
            col_ids,
            torch.tensor([pos], dtype=torch.long, device=device),
        )
        log_probs = torch.log_softmax(logits, dim=-1)
        total_log_prob += float(log_probs[0, target_id].item())
    return total_log_prob / SEQ_LEN


def perplexity_from_pll(pll: float) -> float:
    return math.exp(-pll)


@torch.no_grad()
def prefix_pseudo_log_likelihood(
    columns: list[str] | tuple[str, ...],
    model: GlobalBiGRUScorer,
    vocab: Vocab,
    device: torch.device,
) -> float:
    if not columns:
        raise ValueError("expected at least one column")
    if any(len(col) != 4 for col in columns):
        raise ValueError("each column must contain four characters")
    if len(columns) > 7:
        raise ValueError("expected at most seven columns")

    filled: list[int] = [model.mask_id] * SEQ_LEN
    known_positions: list[int] = []
    for col in range(7):
        if col < len(columns):
            current = columns[col]
            for row in range(4):
                pos = row * 7 + col
                filled[pos] = vocab.stoi[current[row]]
                known_positions.append(pos)

    row_ids = torch.tensor([ROW_IDS], dtype=torch.long, device=device)
    col_ids = torch.tensor([COL_IDS], dtype=torch.long, device=device)
    total_log_prob = 0.0
    for pos in known_positions:
        target_id = filled[pos]
        masked = list(filled)
        masked[pos] = model.mask_id
        logits = model(
            torch.tensor([masked], dtype=torch.long, device=device),
            row_ids,
            col_ids,
            torch.tensor([pos], dtype=torch.long, device=device),
        )
        log_probs = torch.log_softmax(logits, dim=-1)
        total_log_prob += float(log_probs[0, target_id].item())
    return total_log_prob / max(len(known_positions), 1)
