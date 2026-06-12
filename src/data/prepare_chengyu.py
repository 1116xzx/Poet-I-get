from __future__ import annotations

import argparse
import json
from pathlib import Path


def is_chinese_word(text: str) -> bool:
    return bool(text) and all("\u4e00" <= ch <= "\u9fff" for ch in text)


def prepare_idioms(raw_path: str | Path, out_path: str | Path) -> list[str]:
    raw_path = Path(raw_path)
    out_path = Path(out_path)
    data = json.loads(raw_path.read_text(encoding="utf-8"))
    idioms = sorted({item["word"].strip() for item in data if len(item.get("word", "").strip()) == 4 and is_chinese_word(item["word"].strip())})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(idioms) + "\n", encoding="utf-8")
    return idioms


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract four-character idioms from chinese-xinhua idiom.json.")
    parser.add_argument("--raw", default="data/raw/idiom.json")
    parser.add_argument("--out", default="data/processed/chengyu/idioms.txt")
    args = parser.parse_args()
    idioms = prepare_idioms(args.raw, args.out)
    print(f"saved {len(idioms)} idioms -> {args.out}")


if __name__ == "__main__":
    main()
