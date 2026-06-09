from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch.utils.data import Dataset

from src.data.tokenizer import Vocab

TASKS = ("free", "continue", "acrostic")


def poem_puncts(poem: dict) -> list[str]:
    puncts = list(poem.get("puncts", ["", "", "", ""]))
    return (puncts + ["", "", "", ""])[:4]


def punctuated_line(line: str, punct: str) -> list[str]:
    return [*line, *([punct] if punct else [])]


def structured_poem(lines: list[str], puncts: list[str], use_line_markers: bool = True) -> list[str]:
    tokens: list[str] = []
    for i, line in enumerate(lines, start=1):
        if use_line_markers:
            tokens.append(f"<L{i}>")
        tokens.extend(punctuated_line(line, puncts[i - 1]))
    return tokens


def make_sequence(poem: dict, task: str, use_line_markers: bool = True) -> tuple[list[str], int, list[int]]:
    lines = poem["lines"]
    puncts = poem_puncts(poem)
    head_positions: list[int] = []
    if task == "free":
        prefix = ["<BOS>", "<TASK_FREE>", "<SEP>"]
        body = structured_poem(lines, puncts, use_line_markers)
    elif task == "continue":
        prefix = ["<BOS>", "<TASK_CONT>", "<SEP>", *lines[0], "<SEP>"]
        if use_line_markers:
            prefix.append("<L2>")
            body = [
                *punctuated_line(lines[1], puncts[1]),
                "<L3>",
                *punctuated_line(lines[2], puncts[2]),
                "<L4>",
                *punctuated_line(lines[3], puncts[3]),
            ]
        else:
            body = [
                *punctuated_line(lines[1], puncts[1]),
                *punctuated_line(lines[2], puncts[2]),
                *punctuated_line(lines[3], puncts[3]),
            ]
    elif task == "acrostic":
        heads = [line[0] for line in lines]
        prefix = ["<BOS>", "<TASK_ACRO>", "<SEP>", *heads, "<SEP>"]
        if use_line_markers:
            prefix.append("<L1>")
            body = [
                *punctuated_line(lines[0], puncts[0]),
                "<L2>",
                *punctuated_line(lines[1], puncts[1]),
                "<L3>",
                *punctuated_line(lines[2], puncts[2]),
                "<L4>",
                *punctuated_line(lines[3], puncts[3]),
            ]
        else:
            body = [
                *punctuated_line(lines[0], puncts[0]),
                *punctuated_line(lines[1], puncts[1]),
                *punctuated_line(lines[2], puncts[2]),
                *punctuated_line(lines[3], puncts[3]),
            ]
        offset = len(prefix)
        if use_line_markers:
            head_positions = [offset]
            cursor = offset + len(punctuated_line(lines[0], puncts[0])) + 1
            head_positions.append(cursor)
            cursor += len(punctuated_line(lines[1], puncts[1])) + 1
            head_positions.append(cursor)
            cursor += len(punctuated_line(lines[2], puncts[2])) + 1
            head_positions.append(cursor)
        else:
            cursor = offset
            for line, punct in zip(lines, puncts):
                head_positions.append(cursor)
                cursor += len(punctuated_line(line, punct))
    else:
        raise ValueError(f"unknown task: {task}")
    return [*prefix, *body, "<EOS>"], len(prefix), head_positions


@dataclass
class EncodedExample:
    input_ids: list[int]
    labels: list[int]
    loss_weights: list[float]


class PoemDataset(Dataset[EncodedExample]):
    def __init__(
        self,
        poems: list[dict],
        vocab: Vocab,
        tasks: Iterable[str] = TASKS,
        use_line_markers: bool = True,
        acrostic_head_weight: float = 1.0,
    ) -> None:
        self.poems = poems
        self.vocab = vocab
        self.tasks = tuple(tasks)
        self.use_line_markers = use_line_markers
        self.acrostic_head_weight = acrostic_head_weight
        if not self.tasks:
            raise ValueError("at least one task is required")

    def __len__(self) -> int:
        return len(self.poems) * len(self.tasks)

    def __getitem__(self, index: int) -> EncodedExample:
        poem = self.poems[index // len(self.tasks)]
        task = self.tasks[index % len(self.tasks)]
        tokens, target_start, head_positions = make_sequence(poem, task, self.use_line_markers)
        ids = self.vocab.encode(tokens)
        inputs, labels = ids[:-1], ids[1:]
        labels = [token_id if i + 1 >= target_start else -100 for i, token_id in enumerate(labels)]
        loss_weights = [1.0 if label != -100 else 0.0 for label in labels]
        if task == "acrostic" and self.acrostic_head_weight != 1.0:
            for token_pos in head_positions:
                label_pos = token_pos - 1
                if 0 <= label_pos < len(loss_weights) and labels[label_pos] != -100:
                    loss_weights[label_pos] = self.acrostic_head_weight
        return EncodedExample(inputs, labels, loss_weights)


def collate_examples(batch: list[EncodedExample], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(item.input_ids) for item in batch)
    inputs = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    loss_weights = torch.zeros((len(batch), max_len), dtype=torch.float)
    for row, item in enumerate(batch):
        inputs[row, : len(item.input_ids)] = torch.tensor(item.input_ids)
        labels[row, : len(item.labels)] = torch.tensor(item.labels)
        loss_weights[row, : len(item.loss_weights)] = torch.tensor(item.loss_weights)
    return {"input_ids": inputs, "labels": labels, "loss_weights": loss_weights}
