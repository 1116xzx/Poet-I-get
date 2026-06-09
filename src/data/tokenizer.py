from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.utils.common import read_jsonl

BASE_SPECIAL_TOKENS = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>",
    "<SEP>",
    "<TASK_FREE>",
    "<TASK_CONT>",
    "<TASK_ACRO>",
]
LINE_TOKENS = ["<L1>", "<L2>", "<L3>", "<L4>"]
PUNCT_TOKENS = ["，", "。", "！", "？", "；", "：", "、", ",", ".", "!", "?", ";", ":"]


def special_tokens(include_line_tokens: bool = True) -> list[str]:
    return [*BASE_SPECIAL_TOKENS, *(LINE_TOKENS if include_line_tokens else []), *PUNCT_TOKENS]


@dataclass(frozen=True)
class Vocab:
    tokens: list[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "stoi", {token: i for i, token in enumerate(self.tokens)})

    @classmethod
    def build(cls, poems: list[dict]) -> "Vocab":
        return cls.build_with_options(poems)

    @classmethod
    def build_with_options(cls, poems: list[dict], include_line_tokens: bool = True) -> "Vocab":
        counter = Counter("".join(item["text"] for item in poems))
        chars = [ch for ch, _ in counter.most_common()]
        return cls(special_tokens(include_line_tokens) + chars)

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["tokens"])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"tokens": self.tokens}, ensure_ascii=False, indent=2), encoding="utf-8")

    def id(self, token: str) -> int:
        return self.stoi.get(token, self.unk_id)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.id(token) for token in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.tokens[i] for i in ids]

    @property
    def pad_id(self) -> int:
        return self.stoi["<PAD>"]

    @property
    def unk_id(self) -> int:
        return self.stoi["<UNK>"]

    @property
    def bos_id(self) -> int:
        return self.stoi["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.stoi["<EOS>"]

    @property
    def char_ids(self) -> list[int]:
        return [
            idx
            for idx, token in enumerate(self.tokens)
            if len(token) == 1 and ("\u4e00" <= token <= "\u9fff" or token == "〇")
        ]

    @property
    def punct_ids(self) -> list[int]:
        return [self.stoi[token] for token in PUNCT_TOKENS if token in self.stoi]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a training-only character vocabulary.")
    parser.add_argument("--data_dir", default="data/processed/qijue")
    parser.add_argument("--out", default="data/processed/qijue/vocab.json")
    parser.add_argument("--without-line-tokens", action="store_true")
    args = parser.parse_args()
    vocab = Vocab.build_with_options(
        read_jsonl(Path(args.data_dir) / "train.jsonl"),
        include_line_tokens=not args.without_line_tokens,
    )
    vocab.save(args.out)
    print(f"saved {len(vocab.tokens)} tokens -> {args.out}")


if __name__ == "__main__":
    main()
