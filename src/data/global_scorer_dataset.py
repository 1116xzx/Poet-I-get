from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import torch
from torch.utils.data import Dataset

from src.data.tokenizer import Vocab

NUM_LINES = 4
LINE_LEN = 7
SEQ_LEN = NUM_LINES * LINE_LEN


def flatten_poem_lines(poem: dict) -> list[str]:
    lines = poem["lines"]
    if len(lines) != NUM_LINES or any(len(line) != LINE_LEN for line in lines):
        raise ValueError("expected a 4x7 qijue poem")
    return [ch for line in lines for ch in line]


def position_metadata() -> tuple[list[int], list[int]]:
    row_ids: list[int] = []
    col_ids: list[int] = []
    for row in range(NUM_LINES):
        for col in range(LINE_LEN):
            row_ids.append(row)
            col_ids.append(col)
    return row_ids, col_ids


ROW_IDS, COL_IDS = position_metadata()


def mask_weight_for_pos(pos: int) -> float:
    col = COL_IDS[pos]
    row = ROW_IDS[pos]
    weight = 1.0
    if col == 0:
        weight = max(weight, 2.0)
    if col == LINE_LEN - 1:
        weight = max(weight, 1.8)
    if col == LINE_LEN - 1 and row in (1, 3):
        weight = max(weight, 2.2)
    return weight


MASK_POSITION_WEIGHTS = [mask_weight_for_pos(pos) for pos in range(SEQ_LEN)]


@dataclass(frozen=True)
class GlobalMaskExample:
    input_ids: list[int]
    row_ids: list[int]
    col_ids: list[int]
    target_id: int
    target_pos: int
    target_weight: float


class GlobalMaskDataset(Dataset[GlobalMaskExample]):
    def __init__(
        self,
        poems: Sequence[dict],
        vocab: Vocab,
        *,
        samples_per_poem: int = 1,
        seed: int = 42,
    ) -> None:
        self.vocab = vocab
        self.samples_per_poem = samples_per_poem
        self.seed = seed
        self.mask_id = len(vocab.tokens)
        self.sequences = [vocab.encode(flatten_poem_lines(poem)) for poem in poems]

    def __len__(self) -> int:
        return len(self.sequences) * self.samples_per_poem

    def __getitem__(self, index: int) -> GlobalMaskExample:
        poem_index = index // self.samples_per_poem
        sample_index = index % self.samples_per_poem
        ids = list(self.sequences[poem_index])
        rng = random.Random(self.seed + poem_index * 1009 + sample_index * 9173)
        target_pos = rng.choices(range(SEQ_LEN), weights=MASK_POSITION_WEIGHTS, k=1)[0]
        target_id = ids[target_pos]
        ids[target_pos] = self.mask_id
        return GlobalMaskExample(
            input_ids=ids,
            row_ids=ROW_IDS,
            col_ids=COL_IDS,
            target_id=target_id,
            target_pos=target_pos,
            target_weight=MASK_POSITION_WEIGHTS[target_pos],
        )


def collate_global_mask_examples(batch: list[GlobalMaskExample]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([item.input_ids for item in batch], dtype=torch.long),
        "row_ids": torch.tensor([item.row_ids for item in batch], dtype=torch.long),
        "col_ids": torch.tensor([item.col_ids for item in batch], dtype=torch.long),
        "target_ids": torch.tensor([item.target_id for item in batch], dtype=torch.long),
        "target_pos": torch.tensor([item.target_pos for item in batch], dtype=torch.long),
        "target_weights": torch.tensor([item.target_weight for item in batch], dtype=torch.float),
    }
