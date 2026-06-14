from __future__ import annotations

from pathlib import Path

import torch

from src.data.global_prefix_scorer_dataset import COL_IDS, LINE_LEN, NUM_LINES, ROW_IDS, SEQ_LEN
from src.data.tokenizer import Vocab
from src.models.global_prefix_bigru_scorer import GlobalPrefixBiGRUScorer


def load_global_prefix_scorer(path: str | Path, device: torch.device):
    bundle = torch.load(Path(path), map_location=device)
    vocab = Vocab(bundle["vocab_tokens"])
    model = GlobalPrefixBiGRUScorer(vocab_size=len(vocab.tokens), **bundle["model_config"]).to(device)
    model.load_state_dict(bundle["model_state"])
    model.eval()
    return model, vocab, bundle


def _prefix_input_for_target(
    columns: tuple[str, ...] | list[str],
    vocab: Vocab,
    model: GlobalPrefixBiGRUScorer,
    target_row: int,
) -> tuple[list[int], int, int]:
    target_col = len(columns) - 1
    target_pos = target_row * LINE_LEN + target_col
    target_id = vocab.stoi[columns[target_col][target_row]]
    input_ids: list[int] = []

    for pos in range(SEQ_LEN):
        row = ROW_IDS[pos]
        col = COL_IDS[pos]
        if col < target_col:
            input_ids.append(vocab.stoi[columns[col][row]])
        elif col == target_col:
            input_ids.append(model.mask_id if row == target_row else model.col_pad_id)
        else:
            input_ids.append(model.prefix_pad_id)
    return input_ids, target_pos, target_id


@torch.no_grad()
def prefix_column_log_likelihood(
    columns: tuple[str, ...] | list[str],
    model: GlobalPrefixBiGRUScorer,
    vocab: Vocab,
    device: torch.device,
) -> float:
    if not columns:
        raise ValueError("expected at least one column")
    if len(columns) > LINE_LEN:
        raise ValueError("expected at most seven columns")
    if any(len(column) != NUM_LINES for column in columns):
        raise ValueError("each column must contain four characters")

    row_ids = torch.tensor([ROW_IDS], dtype=torch.long, device=device)
    col_ids = torch.tensor([COL_IDS], dtype=torch.long, device=device)
    total_log_prob = 0.0

    for target_row in range(NUM_LINES):
        input_ids, target_pos, target_id = _prefix_input_for_target(columns, vocab, model, target_row)
        logits = model(
            torch.tensor([input_ids], dtype=torch.long, device=device),
            row_ids,
            col_ids,
            torch.tensor([target_pos], dtype=torch.long, device=device),
        )
        log_probs = torch.log_softmax(logits, dim=-1)
        total_log_prob += float(log_probs[0, target_id].item())
    return total_log_prob / NUM_LINES
