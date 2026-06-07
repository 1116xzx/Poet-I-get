from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable

FILE_ID = "1YcP6B28KsOwacr7C_j1tcstkA-QtPd61"
SENTENCE_SPLIT_RE = re.compile(r"[，。！？；：、,.!?;:|/／\\]+")
BLANK_BLOCK_RE = re.compile(r"\n\s*\n")


def download_raw(out_path: str | Path) -> None:
    """Download chinese_poem.txt from the officiahl Google Drive link."""
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("Please install gdown first: pip install gdown") from exc
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = gdown.download(id=FILE_ID, output=str(out), quiet=False)
    if result is None:
        raise RuntimeError("Dataset download failed. Open the official page and download manually.")


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).replace("\u3000", " ").replace("\xa0", " ")


def is_han(ch: str) -> bool:
    return ("\u4e00" <= ch <= "\u9fff") or ch == "〇"


def only_han(text: str) -> str:
    return "".join(ch for ch in normalize_text(text) if is_han(ch))


def split_if_quatrain(record: str) -> list[str] | None:
    """Try common layouts: four lines, punctuation-separated text, or flat 20/28 chars."""
    record = normalize_text(record).strip()
    if not record:
        return None

    line_parts = [only_han(x) for x in record.splitlines()]
    line_parts = [x for x in line_parts if x]
    if len(line_parts) == 4:
        return line_parts

    punct_parts = [only_han(x) for x in SENTENCE_SPLIT_RE.split(record)]
    punct_parts = [x for x in punct_parts if x]
    if len(punct_parts) == 4:
        return punct_parts

    flat = only_han(record)
    if len(flat) in (20, 28):
        width = len(flat) // 4
        return [flat[i : i + width] for i in range(0, len(flat), width)]
    return None


def iter_candidate_quatrains(raw_text: str) -> Iterable[list[str]]:
    """Yield quatrain candidates without assuming a single raw-file layout."""
    raw_text = normalize_text(raw_text).replace("\r\n", "\n").replace("\r", "\n")

    for block in BLANK_BLOCK_RE.split(raw_text):
        parsed = split_if_quatrain(block)
        if parsed:
            yield parsed

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    for line in lines:
        parsed = split_if_quatrain(line)
        if parsed:
            yield parsed

    buffer: list[str] = []
    width: int | None = None
    for line in lines:
        cleaned = only_han(line)
        if len(cleaned) not in (5, 7):
            buffer, width = [], None
            continue
        if width is None or len(cleaned) == width:
            buffer.append(cleaned)
            width = len(cleaned)
        else:
            buffer, width = [cleaned], len(cleaned)
        if len(buffer) == 4:
            yield buffer
            buffer, width = [], None


def poem_kind(lines: list[str]) -> str:
    lengths = [len(x) for x in lines]
    if lengths == [7, 7, 7, 7]:
        return "qijue"
    if lengths == [5, 5, 5, 5]:
        return "wujue"
    return "other_" + "-".join(map(str, lengths))


def split_dataset(items: list[dict], seed: int = 42) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    items = items[:]
    rng.shuffle(items)
    n = len(items)
    n_train = int(n * 0.90)
    n_valid = int(n * 0.05)
    return {
        "train": items[:n_train],
        "valid": items[n_train : n_train + n_valid],
        "test": items[n_train + n_valid :],
    }


def save_jsonl(items: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def preprocess(raw_path: str | Path, out_dir: str | Path, seed: int = 42) -> dict:
    raw = Path(raw_path).read_text(encoding="utf-8", errors="ignore")
    form_counter: Counter[str] = Counter()
    seen: set[str] = set()
    qijue: list[dict] = []

    for lines in iter_candidate_quatrains(raw):
        kind = poem_kind(lines)
        form_counter[kind] += 1
        if kind != "qijue":
            continue
        flat = "".join(lines)
        if flat in seen:
            continue
        seen.add(flat)
        qijue.append({"text": flat, "lines": lines})

    splits = split_dataset(qijue, seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        save_jsonl(items, out / f"{name}.jsonl")
        (out / f"{name}.flat.txt").write_text(
            "\n".join(item["text"] for item in items) + ("\n" if items else ""),
            encoding="utf-8",
        )

    char_counter = Counter("".join(item["text"] for item in qijue))
    stats = {
        "strict_qijue_count": len(qijue),
        "parsed_form_distribution": dict(form_counter),
        "split_sizes": {name: len(items) for name, items in splits.items()},
        "unique_chars": len(char_counter),
        "top_50_chars": char_counter.most_common(50),
    }
    (out / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "top_chars.csv").write_text(
        "char,freq\n" + "\n".join(f"{ch},{freq}" for ch, freq in char_counter.most_common()),
        encoding="utf-8-sig",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract strict 4x7 Chinese quatrains.")
    parser.add_argument("--raw_path", default="data/raw/chinese_poem.txt")
    parser.add_argument("--out_dir", default="data/processed/qijue")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.download:
        download_raw(args.raw_path)
    preprocess(args.raw_path, args.out_dir, args.seed)


if __name__ == "__main__":
    main()
