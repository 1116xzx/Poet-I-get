from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data.dataset import punctuated_line, structured_poem
from src.engine.generate import load_checkpoint
from src.utils.common import choose_device, seed_everything

DEFAULT_PUNCTS = ["，", "。", "，", "。"]
POETIC_CHARS = set("山水风月花云雨雪江河湖海天日夜春秋寒清明落流水烟霞楼舟梦影声色")
DEFAULT_PHRASE_STATS_PATH = Path("data/processed/qijue/train.jsonl")
FUNCTION_WORDS = set("有无是不一")
IMAGE_CHARS = set("山水风月天日云江湖海河雨雪秋春")


@dataclass(frozen=True)
class GridCandidate:
    columns: tuple[str, ...]
    score: float
    nll: float
    repeat_penalty: float
    phrase_penalty: float
    style_penalty: float

    @property
    def lines(self) -> list[str]:
        return ["".join(column[row] for column in self.columns) for row in range(4)]


@dataclass(frozen=True)
class PhraseStats:
    bigram_freq: dict[str, int]
    trigram_freq: dict[str, int]
    common_bigram_threshold: int
    common_trigram_threshold: int


def is_chinese_word(text: str) -> bool:
    return bool(text) and all("\u4e00" <= ch <= "\u9fff" for ch in text)


def has_line_markers(vocab) -> bool:
    return all(token in vocab.stoi for token in ("<L1>", "<L2>", "<L3>", "<L4>"))


def _ngrams(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(max(len(text) - size + 1, 0))]


def _percentile_threshold(counter: Counter[str], percentile: float, floor: int) -> int:
    if not counter:
        return floor
    values = sorted(counter.values())
    idx = int((len(values) - 1) * percentile)
    return max(floor, values[idx])


@lru_cache(maxsize=4)
def load_phrase_stats(path: str | Path = DEFAULT_PHRASE_STATS_PATH) -> PhraseStats:
    bigram_freq: Counter[str] = Counter()
    trigram_freq: Counter[str] = Counter()
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            item = json.loads(raw)
            for line in item.get("lines", []):
                bigram_freq.update(_ngrams(line, 2))
                trigram_freq.update(_ngrams(line, 3))
    return PhraseStats(
        bigram_freq=dict(bigram_freq),
        trigram_freq=dict(trigram_freq),
        common_bigram_threshold=_percentile_threshold(bigram_freq, 0.8, 4),
        common_trigram_threshold=_percentile_threshold(trigram_freq, 0.8, 2),
    )


def load_idioms(path: str | Path, vocab, limit: int = 300, poetic_only: bool = True, allow_repeated_idioms: bool = False) -> list[str]:
    idioms = []
    seen = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        idiom = line.strip()
        if len(idiom) != 4 or idiom in seen or not is_chinese_word(idiom):
            continue
        if not allow_repeated_idioms and len(set(idiom)) < 4:
            continue
        if poetic_only and sum(ch in POETIC_CHARS for ch in idiom) < 2:
            continue
        if all(ch in vocab.stoi for ch in idiom):
            idioms.append(idiom)
            seen.add(idiom)

    def idiom_commonness(idiom: str) -> tuple[int, str]:
        poetic_bonus = sum(ch in POETIC_CHARS for ch in idiom)
        return (-poetic_bonus, sum(vocab.stoi.get(ch, len(vocab.tokens)) for ch in idiom), idiom)

    idioms.sort(key=idiom_commonness)
    return idioms[:limit]


def repetition_penalty(lines: list[str]) -> float:
    chars = "".join(lines)
    if not chars:
        return 0.0
    char_counts = {ch: chars.count(ch) for ch in set(chars)}
    char_repeat = sum(max(count - 1, 0) for count in char_counts.values()) / len(chars)
    line_char_repeat = 0.0
    for line in lines:
        if line:
            counts = {ch: line.count(ch) for ch in set(line)}
            line_char_repeat += sum(max(count - 1, 0) for count in counts.values()) / len(line)
    line_char_repeat /= max(len(lines), 1)
    bigrams = [line[i : i + 2] for line in lines for i in range(max(len(line) - 1, 0))]
    bigram_repeat = 0.0 if not bigrams else 1.0 - len(set(bigrams)) / len(bigrams)
    trigrams = [line[i : i + 3] for line in lines for i in range(max(len(line) - 2, 0))]
    trigram_repeat = 0.0 if not trigrams else 1.0 - len(set(trigrams)) / len(trigrams)
    adjacent = [line[i] == line[i + 1] for line in lines for i in range(max(len(line) - 1, 0))]
    adjacent_repeat = 0.0 if not adjacent else sum(adjacent) / len(adjacent)
    skip_repeat = [line[i] == line[i + 2] for line in lines for i in range(max(len(line) - 2, 0))]
    skip_repeat_rate = 0.0 if not skip_repeat else sum(skip_repeat) / len(skip_repeat)
    return (
        char_repeat
        + 2.0 * line_char_repeat
        + 1.5 * bigram_repeat
        + 2.0 * trigram_repeat
        + 2.0 * adjacent_repeat
        + 3.0 * skip_repeat_rate
    )


def phrase_penalty(lines: list[str], stats: PhraseStats) -> float:
    reward = 0.0
    penalty = 0.0
    total_units = 0
    rewarded_bigrams: set[str] = set()
    rewarded_trigrams: set[str] = set()
    for line in lines:
        for bigram in _ngrams(line, 2):
            total_units += 1
            count = stats.bigram_freq.get(bigram, 0)
            if count == 0:
                penalty += 1.2
            elif count == 1:
                penalty += 0.4
            elif count >= stats.common_bigram_threshold and bigram not in rewarded_bigrams:
                reward += min(1.6, 0.35 + math.log1p(count) / 3.2)
                rewarded_bigrams.add(bigram)
        for trigram in _ngrams(line, 3):
            total_units += 1
            count = stats.trigram_freq.get(trigram, 0)
            if count == 0:
                penalty += 1.6
            elif count == 1:
                penalty += 0.6
            elif count >= stats.common_trigram_threshold and trigram not in rewarded_trigrams:
                reward += min(2.2, 0.5 + math.log1p(count) / 3.0)
                rewarded_trigrams.add(trigram)
    return (penalty - reward) / max(total_units, 1)


def style_penalty(lines: list[str]) -> float:
    penalty = 0.0
    for line in lines:
        if not line:
            continue
        function_count = sum(ch in FUNCTION_WORDS for ch in line)
        if function_count:
            penalty += 0.9 * function_count
        tail = line[-1]
        if tail in FUNCTION_WORDS:
            penalty += 1.8
        image_count = sum(ch in IMAGE_CHARS for ch in line)
        unique_image_count = len({ch for ch in line if ch in IMAGE_CHARS})
        if image_count >= 4:
            penalty += 0.6 * (image_count - 3)
        if image_count >= 5 and unique_image_count <= 3:
            penalty += 1.2
        for i in range(len(line) - 1):
            pair = line[i : i + 2]
            if pair in {"山水", "水山", "山日", "水日", "日山", "月日", "日月", "山天", "水天"}:
                penalty += 0.8
    return penalty / max(len(lines), 1)


def poem_tokens(lines: list[str], vocab, acrostic: str) -> list[str]:
    prefix = ["<BOS>", "<TASK_ACRO>", "<SEP>", *acrostic, "<SEP>"]
    complete = all(len(line) == 7 for line in lines)
    if complete:
        body = structured_poem(lines, DEFAULT_PUNCTS, use_line_markers=has_line_markers(vocab))
        return [*prefix, *body, "<EOS>"]
    body: list[str] = []
    for line in lines:
        body.extend(line)
    return [*prefix, *body]


@torch.no_grad()
def batch_nll(model, vocab, line_batch: list[list[str]], acrostic: str, batch_size: int = 256) -> list[float]:
    device = next(model.parameters()).device
    target_start = 8
    scores: list[float] = []
    for start in range(0, len(line_batch), batch_size):
        batch_lines = line_batch[start : start + batch_size]
        encoded = [vocab.encode(poem_tokens(lines, vocab, acrostic)) for lines in batch_lines]
        max_len = max(len(ids) for ids in encoded)
        inputs = torch.full((len(encoded), max_len - 1), vocab.pad_id, dtype=torch.long, device=device)
        labels = torch.full((len(encoded), max_len - 1), -100, dtype=torch.long, device=device)
        for row, ids in enumerate(encoded):
            input_ids = ids[:-1]
            label_ids = ids[1:]
            inputs[row, : len(input_ids)] = torch.tensor(input_ids, dtype=torch.long, device=device)
            for idx, token_id in enumerate(label_ids):
                if idx + 1 >= target_start:
                    labels[row, idx] = token_id
        logits = model(inputs)
        losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100, reduction="none").reshape_as(labels)
        valid = labels != -100
        per_item = losses.sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        scores.extend(float(value) for value in per_item.cpu())
    return scores


def expand_candidates(
    beam: list[GridCandidate],
    idioms: list[str],
    model,
    vocab,
    acrostic: str,
    stats: PhraseStats,
    nll_weight: float,
    repeat_weight: float,
    phrase_weight: float,
    style_weight: float,
    beam_size: int,
) -> list[GridCandidate]:
    columns: list[tuple[str, ...]] = []
    for candidate in beam:
        used = set(candidate.columns[1:])
        for idiom in idioms:
            if idiom not in used:
                columns.append((*candidate.columns, idiom))

    line_batch = [["".join(column[row] for column in cols) for row in range(4)] for cols in columns]
    nll_values = batch_nll(model, vocab, line_batch, acrostic)
    expanded: list[GridCandidate] = []
    for cols, lines, nll in zip(columns, line_batch, nll_values):
        repeat = repetition_penalty(lines)
        phrase = phrase_penalty(lines, stats)
        style = style_penalty(lines)
        expanded.append(
            GridCandidate(
                cols,
                nll_weight * nll + repeat_weight * repeat + phrase_weight * phrase + style_weight * style,
                nll,
                repeat,
                phrase,
                style,
            )
        )
    expanded.sort(key=lambda item: item.score)
    return expanded[:beam_size]


def search_acrostic_grid(
    acrostic: str,
    model,
    vocab,
    idioms: list[str],
    beam_size: int = 20,
    top_k: int = 5,
    nll_weight: float = 5.5,
    repeat_weight: float = 5.0,
    phrase_weight: float = 4.0,
    style_weight: float = 3.0,
    phrase_stats_path: str | Path = DEFAULT_PHRASE_STATS_PATH,
) -> list[GridCandidate]:
    if len(acrostic) != 4 or not is_chinese_word(acrostic):
        raise ValueError("acrostic must contain exactly four Chinese characters")
    if any(ch not in vocab.stoi for ch in acrostic):
        raise ValueError("acrostic contains characters outside the model vocabulary")

    stats = load_phrase_stats(phrase_stats_path)
    beam = [GridCandidate((acrostic,), 0.0, 0.0, 0.0, 0.0, 0.0)]
    for _ in range(6):
        beam = expand_candidates(
            beam,
            idioms,
            model,
            vocab,
            acrostic,
            stats,
            nll_weight,
            repeat_weight,
            phrase_weight,
            style_weight,
            beam_size,
        )
    return beam[:top_k]


def write_markdown(path: str | Path, acrostic: str, candidates: list[GridCandidate]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = [f"# 成语藏头格诗样例：{acrostic}", ""]
    for idx, candidate in enumerate(candidates, start=1):
        rows.extend(
            [
                f"## 候选 {idx}",
                "",
                f"- 综合得分：{candidate.score:.4f}",
                f"- GRU_NLL：{candidate.nll:.4f}",
                f"- 重复惩罚：{candidate.repeat_penalty:.4f}",
                f"- 短语惩罚：{candidate.phrase_penalty:.4f}",
                f"- 风格惩罚：{candidate.style_penalty:.4f}",
                "",
                "|  | 1 | 2 | 3 | 4 | 5 | 6 | 7 |",
                "|---|---|---|---|---|---|---|---|",
            ]
        )
        for row_idx, line in enumerate(candidate.lines, start=1):
            rows.append(f"| 第{row_idx}句 | " + " | ".join(line) + " |")
        rows.extend(["", "横向诗：", ""])
        rows.extend(candidate.lines)
        rows.extend(["", "纵向约束：", ""])
        rows.append(f"- 第1列：{candidate.columns[0]}（用户藏头）")
        for col_idx, idiom in enumerate(candidate.columns[1:], start=2):
            rows.append(f"- 第{col_idx}列：{idiom}")
        rows.append("")
    path.write_text("\n".join(rows), encoding="utf-8")


def write_csv(path: str | Path, acrostic: str, candidates: list[GridCandidate]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rank", "acrostic", "score", "nll", "repeat_penalty", "phrase_penalty", "style_penalty", "lines", "columns"],
        )
        writer.writeheader()
        for idx, candidate in enumerate(candidates, start=1):
            writer.writerow(
                {
                    "rank": idx,
                    "acrostic": acrostic,
                    "score": f"{candidate.score:.6f}",
                    "nll": f"{candidate.nll:.6f}",
                    "repeat_penalty": f"{candidate.repeat_penalty:.6f}",
                    "phrase_penalty": f"{candidate.phrase_penalty:.6f}",
                    "style_penalty": f"{candidate.style_penalty:.6f}",
                    "lines": " / ".join(candidate.lines),
                    "columns": " / ".join(candidate.columns),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate acrostic idiom-grid poems with beam search.")
    parser.add_argument("--acrostic", default="春江花月")
    parser.add_argument("--checkpoint", default="checkpoints/gru_best.pt")
    parser.add_argument("--idioms", default="data/processed/chengyu/idioms.txt")
    parser.add_argument("--candidate_limit", type=int, default=300)
    parser.add_argument("--all_idioms", action="store_true", help="Disable poetic idiom filtering.")
    parser.add_argument("--allow_repeated_idioms", action="store_true", help="Allow idioms with repeated internal characters, such as 风风雨雨.")
    parser.add_argument("--beam_size", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--nll_weight", type=float, default=5.5)
    parser.add_argument("--repeat_weight", type=float, default=5.0)
    parser.add_argument("--phrase_weight", type=float, default=4.0)
    parser.add_argument("--style_weight", type=float, default=3.0)
    parser.add_argument("--phrase_stats", default=str(DEFAULT_PHRASE_STATS_PATH))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out_md", default="runs/chengyu_grid/acrostic_samples.md")
    parser.add_argument("--out_csv", default="runs/chengyu_grid/acrostic_samples.csv")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = choose_device(args.device)
    model, vocab, _ = load_checkpoint(args.checkpoint, device)
    idioms = load_idioms(args.idioms, vocab, args.candidate_limit, poetic_only=not args.all_idioms, allow_repeated_idioms=args.allow_repeated_idioms)
    if not idioms:
        raise ValueError("no idioms available after filtering")
    candidates = search_acrostic_grid(
        args.acrostic,
        model,
        vocab,
        idioms,
        args.beam_size,
        args.top_k,
        args.nll_weight,
        args.repeat_weight,
        args.phrase_weight,
        args.style_weight,
        args.phrase_stats,
    )
    write_markdown(args.out_md, args.acrostic, candidates)
    write_csv(args.out_csv, args.acrostic, candidates)

    print(f"idiom candidates: {len(idioms)}")
    print(f"saved markdown -> {args.out_md}")
    print(f"saved csv -> {args.out_csv}")
    for idx, candidate in enumerate(candidates, start=1):
        print(
            f"\n#{idx} score={candidate.score:.4f} "
            f"nll={candidate.nll:.4f} repeat={candidate.repeat_penalty:.4f} "
            f"phrase={candidate.phrase_penalty:.4f} style={candidate.style_penalty:.4f}"
        )
        for line in candidate.lines:
            print(line)
        print("columns:", " / ".join(candidate.columns))


if __name__ == "__main__":
    main()
