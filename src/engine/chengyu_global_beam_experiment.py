from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import torch

from src.engine.chengyu_grid import (
    DEFAULT_PHRASE_STATS_PATH,
    GridCandidate,
    load_idioms,
    load_phrase_stats,
    phrase_penalty,
    repetition_penalty,
    style_penalty,
    batch_nll,
)
from src.engine.generate import load_checkpoint
from src.engine.global_prefix_score import load_global_prefix_scorer, prefix_column_log_likelihood
from src.metrics.poem_metrics import rhyme_score
from src.utils.common import choose_device, seed_everything


@dataclass(frozen=True)
class PrefixBeamCandidate:
    columns: tuple[str, ...]
    prefix_global_pll: float
    beam_score: float = 0.0

    @property
    def lines(self) -> list[str]:
        return ["".join(column[row] for column in self.columns) for row in range(4)]


def expand_with_prefix_global(
    beam: list[PrefixBeamCandidate],
    idioms: list[str],
    global_model,
    global_vocab,
    device: torch.device,
    beam_size: int,
    prefix_repeat_weight: float,
    prefix_phrase_weight: float,
    prefix_style_weight: float,
    phrase_stats,
) -> list[PrefixBeamCandidate]:
    expanded: list[PrefixBeamCandidate] = []
    for candidate in beam:
        used = set(candidate.columns[1:])
        for idiom in idioms:
            if idiom in used:
                continue
            columns = (*candidate.columns, idiom)
            pll = prefix_column_log_likelihood(columns, global_model, global_vocab, device)
            lines = ["".join(column[row] for column in columns) for row in range(4)]
            beam_score = (
                pll
                - prefix_repeat_weight * repetition_penalty(lines)
                - prefix_phrase_weight * phrase_penalty(lines, phrase_stats)
                - prefix_style_weight * style_penalty(lines)
            )
            expanded.append(PrefixBeamCandidate(columns=columns, prefix_global_pll=pll, beam_score=beam_score))
    expanded.sort(key=lambda item: item.beam_score, reverse=True)
    return expanded[:beam_size]


def finalize_candidates(
    beam: list[PrefixBeamCandidate],
    model,
    vocab,
    acrostic: str,
    *,
    nll_weight: float,
    repeat_weight: float,
    phrase_weight: float,
    style_weight: float,
    rhyme_weight: float,
    phrase_stats_path: str | Path,
) -> list[GridCandidate]:
    stats = load_phrase_stats(phrase_stats_path)
    line_batch = [candidate.lines for candidate in beam]
    nll_values = batch_nll(model, vocab, line_batch, acrostic)
    finals: list[GridCandidate] = []
    for candidate, nll in zip(beam, nll_values):
        lines = candidate.lines
        repeat = repetition_penalty(lines)
        phrase = phrase_penalty(lines, stats)
        style = style_penalty(lines)
        rhyme = rhyme_score(lines)
        score = (
            nll_weight * nll
            + repeat_weight * repeat
            + phrase_weight * phrase
            + style_weight * style
            - rhyme_weight * (rhyme / 100.0)
        )
        finals.append(
            GridCandidate(
                columns=candidate.columns,
                score=score,
                nll=nll,
                repeat_penalty=repeat,
                phrase_penalty=phrase,
                style_penalty=style,
                rhyme_score_value=rhyme,
                global_pll=candidate.prefix_global_pll,
            )
        )
    finals.sort(key=lambda item: item.score)
    return finals


def search_with_global_beam(
    acrostic: str,
    model,
    vocab,
    idioms: list[str],
    global_model,
    global_vocab,
    device: torch.device,
    *,
    beam_size: int = 5,
    top_k: int = 5,
    nll_weight: float = 7.0,
    repeat_weight: float = 11.0,
    phrase_weight: float = 6.0,
    style_weight: float = 4.5,
    rhyme_weight: float = 12.0,
    prefix_repeat_weight: float = 1.5,
    prefix_phrase_weight: float = 1.6,
    prefix_style_weight: float = 0.8,
    phrase_stats_path: str | Path = DEFAULT_PHRASE_STATS_PATH,
) -> list[GridCandidate]:
    phrase_stats = load_phrase_stats(phrase_stats_path)
    beam = [PrefixBeamCandidate(columns=(acrostic,), prefix_global_pll=0.0, beam_score=0.0)]
    for _ in range(6):
        beam = expand_with_prefix_global(
            beam,
            idioms,
            global_model,
            global_vocab,
            device,
            beam_size,
            prefix_repeat_weight,
            prefix_phrase_weight,
            prefix_style_weight,
            phrase_stats,
        )
    finals = finalize_candidates(
        beam,
        model,
        vocab,
        acrostic,
        nll_weight=nll_weight,
        repeat_weight=repeat_weight,
        phrase_weight=phrase_weight,
        style_weight=style_weight,
        rhyme_weight=rhyme_weight,
        phrase_stats_path=phrase_stats_path,
    )
    return finals[:top_k]


def write_csv(path: str | Path, acrostic: str, candidates: list[GridCandidate]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "acrostic",
                "score",
                "nll",
                "repeat_penalty",
                "phrase_penalty",
                "style_penalty",
                "rhyme_score",
                "prefix_global_pll",
                "lines",
                "columns",
            ],
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
                    "rhyme_score": f"{candidate.rhyme_score_value:.6f}",
                    "prefix_global_pll": f"{candidate.global_pll:.6f}",
                    "lines": " / ".join(candidate.lines),
                    "columns": " / ".join(candidate.columns),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimental: beam search guided by prefix global PLL, final ranking by local GRU+rules.")
    parser.add_argument("--acrostic", default="清风明月")
    parser.add_argument("--checkpoint", default="checkpoints/gru_best.pt")
    parser.add_argument("--global_scorer", default="checkpoints/global_prefix_bigru_20e.pt")
    parser.add_argument("--idioms", default="data/processed/chengyu/idioms.txt")
    parser.add_argument("--candidate_limit", type=int, default=240)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--nll_weight", type=float, default=7.0)
    parser.add_argument("--repeat_weight", type=float, default=11.0)
    parser.add_argument("--phrase_weight", type=float, default=6.0)
    parser.add_argument("--style_weight", type=float, default=4.5)
    parser.add_argument("--rhyme_weight", type=float, default=12.0)
    parser.add_argument("--prefix_repeat_weight", type=float, default=1.5)
    parser.add_argument("--prefix_phrase_weight", type=float, default=1.6)
    parser.add_argument("--prefix_style_weight", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out_csv", default="runs/chengyu_global_beam_exp/candidates.csv")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = choose_device(args.device)
    model, vocab, _ = load_checkpoint(args.checkpoint, device)
    global_model, global_vocab, _ = load_global_prefix_scorer(args.global_scorer, device)
    idioms = load_idioms(args.idioms, vocab, limit=args.candidate_limit, poetic_only=True, allow_repeated_idioms=False)
    candidates = search_with_global_beam(
        args.acrostic,
        model,
        vocab,
        idioms,
        global_model,
        global_vocab,
        device,
        beam_size=args.beam_size,
        top_k=args.top_k,
        nll_weight=args.nll_weight,
        repeat_weight=args.repeat_weight,
        phrase_weight=args.phrase_weight,
        style_weight=args.style_weight,
        rhyme_weight=args.rhyme_weight,
        prefix_repeat_weight=args.prefix_repeat_weight,
        prefix_phrase_weight=args.prefix_phrase_weight,
        prefix_style_weight=args.prefix_style_weight,
    )
    write_csv(args.out_csv, args.acrostic, candidates)
    print(f"saved -> {args.out_csv}")
    for idx, candidate in enumerate(candidates, start=1):
        print(
            f"#{idx} score={candidate.score:.4f} prefix_pll={candidate.global_pll:.4f} "
            f"nll={candidate.nll:.4f} rhyme={candidate.rhyme_score_value:.2f}"
        )
        for line in candidate.lines:
            print(line)
        print()


if __name__ == "__main__":
    main()
