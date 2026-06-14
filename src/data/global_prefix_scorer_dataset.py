from __future__ import annotations

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


@dataclass(frozen=True)
class PrefixMaskExample:
    input_ids: list[int]
    row_ids: list[int]
    col_ids: list[int]
    target_id: int
    target_pos: int
    target_weight: float
    prefix_cols: int
    target_row: int


class GlobalPrefixMaskDataset(Dataset[PrefixMaskExample]):
    """Prefix scorer data.

    Each poem is expanded into 28 examples: for every target column k and
    target row r, columns before k are visible, the target is <MASK>, other
    cells in column k are <COL_PAD>, and future columns are <PREFIX_PAD>.
    """

    examples_per_poem = NUM_LINES * LINE_LEN

    def __init__(self, poems: Sequence[dict], vocab: Vocab) -> None:
        self.vocab = vocab
        self.mask_id = len(vocab.tokens)
        self.prefix_pad_id = len(vocab.tokens) + 1
        self.col_pad_id = len(vocab.tokens) + 2
        self.sequences = [vocab.encode(flatten_poem_lines(poem)) for poem in poems]

    def __len__(self) -> int:
        return len(self.sequences) * self.examples_per_poem

    def __getitem__(self, index: int) -> PrefixMaskExample:
        poem_index = index // self.examples_per_poem
        local_index = index % self.examples_per_poem
        target_col = local_index // NUM_LINES
        target_row = local_index % NUM_LINES
        target_pos = target_row * LINE_LEN + target_col
        ids = self.sequences[poem_index]
        input_ids: list[int] = []

        for pos, token_id in enumerate(ids):
            row = ROW_IDS[pos]
            col = COL_IDS[pos]
            if col < target_col:
                input_ids.append(token_id)
            elif col == target_col:
                input_ids.append(self.mask_id if row == target_row else self.col_pad_id)
            else:
                input_ids.append(self.prefix_pad_id)

        return PrefixMaskExample(
            input_ids=input_ids,
            row_ids=ROW_IDS,
            col_ids=COL_IDS,
            target_id=ids[target_pos],
            target_pos=target_pos,
            target_weight=1.0,
            prefix_cols=target_col + 1,
            target_row=target_row,
        )


def collate_global_prefix_examples(batch: list[PrefixMaskExample]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([item.input_ids for item in batch], dtype=torch.long),
        "row_ids": torch.tensor([item.row_ids for item in batch], dtype=torch.long),
        "col_ids": torch.tensor([item.col_ids for item in batch], dtype=torch.long),
        "target_ids": torch.tensor([item.target_id for item in batch], dtype=torch.long),
        "target_pos": torch.tensor([item.target_pos for item in batch], dtype=torch.long),
        "target_weights": torch.tensor([item.target_weight for item in batch], dtype=torch.float),
        "prefix_cols": torch.tensor([item.prefix_cols for item in batch], dtype=torch.long),
        "target_rows": torch.tensor([item.target_row for item in batch], dtype=torch.long),
    }
