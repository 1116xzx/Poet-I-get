from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch.utils.data import Dataset

from src.data.tokenizer import Vocab

TASKS = ("free", "continue", "acrostic")


def structured_poem(lines: list[str]) -> list[str]:
    tokens: list[str] = []
    for i, line in enumerate(lines, start=1):
        tokens.append(f"<L{i}>")
        tokens.extend(line)
    return tokens


def make_sequence(lines: list[str], task: str) -> tuple[list[str], int]:
    if task == "free":
        prefix = ["<BOS>", "<TASK_FREE>", "<SEP>"]
        body = structured_poem(lines)
    elif task == "continue":
        prefix = ["<BOS>", "<TASK_CONT>", "<SEP>", *lines[0], "<SEP>", "<L2>"]
        body = [*lines[1], "<L3>", *lines[2], "<L4>", *lines[3]]
    elif task == "acrostic":
        heads = [line[0] for line in lines]
        prefix = ["<BOS>", "<TASK_ACRO>", "<SEP>", *heads, "<SEP>", "<L1>"]
        body = [*lines[0], "<L2>", *lines[1], "<L3>", *lines[2], "<L4>", *lines[3]]
    else:
        raise ValueError(f"unknown task: {task}")
    return [*prefix, *body, "<EOS>"], len(prefix)


@dataclass
class EncodedExample:
    input_ids: list[int]
    labels: list[int]


class PoemDataset(Dataset[EncodedExample]):
    def __init__(self, poems: list[dict], vocab: Vocab, tasks: Iterable[str] = TASKS) -> None:
        self.poems = poems
        self.vocab = vocab
        self.tasks = tuple(tasks)
        if not self.tasks:
            raise ValueError("at least one task is required")

    def __len__(self) -> int:
        return len(self.poems) * len(self.tasks)

    def __getitem__(self, index: int) -> EncodedExample:
        poem = self.poems[index // len(self.tasks)]
        task = self.tasks[index % len(self.tasks)]
        tokens, target_start = make_sequence(poem["lines"], task)
        ids = self.vocab.encode(tokens)
        inputs, labels = ids[:-1], ids[1:]
        labels = [token_id if i + 1 >= target_start else -100 for i, token_id in enumerate(labels)]
        return EncodedExample(inputs, labels)


def collate_examples(batch: list[EncodedExample], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(item.input_ids) for item in batch)
    inputs = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for row, item in enumerate(batch):
        inputs[row, : len(item.input_ids)] = torch.tensor(item.input_ids)
        labels[row, : len(item.labels)] = torch.tensor(item.labels)
    return {"input_ids": inputs, "labels": labels}
