from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data.tokenizer import Vocab
from src.models.char_rnn import CharPoemLM
from src.utils.common import choose_device, seed_everything


# ── Rhyme index ──
def _rhyme_index(vocab: Vocab) -> dict[str, list[int]]:
    try:
        from pypinyin import Style, lazy_pinyin
    except ImportError:
        return {}
    index: dict[str, list[int]] = {}
    for cid in vocab.char_ids:
        ch = vocab.tokens[cid]
        if len(ch) == 1 and "一" <= ch <= "鿿":
            py = lazy_pinyin(ch, style=Style.FINALS)
            if py and py[0]:
                index.setdefault(py[0], []).append(cid)
    return {f: ids for f, ids in index.items() if len(ids) >= 3}

_rhyme_cache: dict[str, list[int]] | None = None

def get_rhyme_candidates(vocab: Vocab, char: str) -> list[int]:
    global _rhyme_cache
    if _rhyme_cache is None:
        _rhyme_cache = _rhyme_index(vocab)
    try:
        from pypinyin import Style, lazy_pinyin
        py = lazy_pinyin(char, style=Style.FINALS)
        final = py[0] if py else ""
        return _rhyme_cache.get(final, [])
    except ImportError:
        return []


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 0.9
    top_k: int = 20
    top_p: float = 1.0


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.numel():
        return logits
    cutoff = torch.topk(logits, k).values[-1]
    return logits.masked_fill(logits < cutoff, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    remove = torch.cumsum(sorted_probs, dim=-1) > p
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    filtered = logits.clone()
    filtered[sorted_idx[remove]] = float("-inf")
    return filtered


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[CharPoemLM, Vocab, dict]:
    bundle = torch.load(path, map_location=device)
    vocab = Vocab(bundle["vocab_tokens"])
    model = CharPoemLM(vocab_size=len(vocab.tokens), **bundle["model_config"])
    model.load_state_dict(bundle["model_state"])
    model.to(device).eval()
    return model, vocab, bundle


class IncrementalDecoder:
    def __init__(self, model: CharPoemLM, prefix_ids: list[int]) -> None:
        self.model = model
        self.device = next(model.parameters()).device
        self.position = 0
        self.state = None
        self.logits: torch.Tensor | None = None
        for token_id in prefix_ids:
            self.push(token_id)

    def push(self, token_id: int) -> None:
        token = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        logits, self.state = self.model.step(token, self.position, self.state)
        self.logits = logits[0]
        self.position += 1


def sample_char(logits: torch.Tensor, vocab: Vocab, sampling: SamplingConfig) -> int:
    allowed = torch.zeros_like(logits, dtype=torch.bool)
    allowed[vocab.char_ids] = True
    logits = logits.masked_fill(~allowed, float("-inf"))
    logits = logits / max(sampling.temperature, 1e-5)
    logits = top_k_filter(logits, sampling.top_k)
    logits = top_p_filter(logits, sampling.top_p)
    return int(torch.multinomial(F.softmax(logits, dim=-1), 1).item())


def validate_prompt(mode: str, prompt: str) -> None:
    expected = 7 if mode == "continue" else 4
    if len(prompt) != expected or not all("一" <= ch <= "鿿" or ch == "〇" for ch in prompt):
        raise ValueError(f"{mode} prompt must contain exactly {expected} Chinese characters: {prompt!r}")


@torch.no_grad()
def generate_poem(
    model: CharPoemLM,
    vocab: Vocab,
    mode: str,
    prompt: str,
    sampling: SamplingConfig,
    rhyme_constraint: bool = False,
) -> list[str]:
    validate_prompt(mode, prompt)
    if mode == "continue":
        lines = [prompt]
        prefix = ["<BOS>", "<TASK_CONT>", "<SEP>", *prompt, "<SEP>", "<L2>"]
        start_line, heads = 1, None
    elif mode == "acrostic":
        lines = []
        prefix = ["<BOS>", "<TASK_ACRO>", "<SEP>", *prompt, "<SEP>", "<L1>"]
        start_line, heads = 0, prompt
    else:
        raise ValueError("mode must be 'continue' or 'acrostic'")

    rhyme_char = None
    rhyme_candidates: list[int] = []

    decoder = IncrementalDecoder(model, vocab.encode(prefix))
    for line_idx in range(start_line, 4):
        line_chars: list[str] = []
        for char_idx in range(7):
            if heads is not None and char_idx == 0:
                visible_char = heads[line_idx]
                token_id = vocab.id(visible_char)
            elif rhyme_constraint and line_idx == 3 and char_idx == 6 and rhyme_candidates:
                token_id = sample_char_rhyme(decoder.logits, vocab, sampling, rhyme_candidates)
                visible_char = vocab.tokens[token_id]
            else:
                assert decoder.logits is not None
                token_id = sample_char(decoder.logits, vocab, sampling)
                visible_char = vocab.tokens[token_id]
            decoder.push(token_id)
            line_chars.append(visible_char)

        lines.append("".join(line_chars))

        if rhyme_constraint and line_idx == 1:
            rhyme_char = line_chars[-1]
            rhyme_candidates = get_rhyme_candidates(vocab, rhyme_char)

        if line_idx < 3:
            decoder.push(vocab.id(f"<L{line_idx + 2}>"))
    return lines


def sample_char_rhyme(logits, vocab, sampling, rhyme_candidates):
    allowed = torch.zeros(logits.numel(), dtype=torch.bool, device=logits.device)
    for cid in rhyme_candidates:
        allowed[cid] = True
    logits = logits.clone()
    logits = logits.masked_fill(~allowed, float("-inf"))
    logits = logits / max(sampling.temperature, 1e-5)
    logits = top_k_filter(logits, sampling.top_k)
    logits = top_p_filter(logits, sampling.top_p)
    return int(torch.multinomial(F.softmax(logits, dim=-1), 1).item())


def format_poem(lines: list[str]) -> str:
    return "，\n".join(lines) + "。"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate constrained seven-character quatrains.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=["continue", "acrostic"], required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    model, vocab, _ = load_checkpoint(args.checkpoint, device)
    sampling = SamplingConfig(args.temperature, args.top_k, args.top_p)
    rows = []
    for offset in range(args.num_return_sequences):
        seed_everything(args.seed + offset)
        lines = generate_poem(model, vocab, args.mode, args.prompt, sampling)
        rows.append({"mode": args.mode, "prompt": args.prompt, "lines": lines})
        print(format_poem(lines), "\n")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
