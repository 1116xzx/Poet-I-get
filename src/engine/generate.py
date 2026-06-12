from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data.tokenizer import PUNCT_TOKENS, Vocab
from src.models.char_rnn import CharPoemLM
from src.utils.common import choose_device, seed_everything

LINE_TOKENS = ("<L1>", "<L2>", "<L3>", "<L4>")
DEFAULT_LINE_PUNCT = ("，", "。", "，", "。")
LINE_PUNCT = DEFAULT_LINE_PUNCT
_rhyme_cache: dict[str, list[int]] | None = None


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 0.9
    top_k: int = 20
    top_p: float = 1.0


@dataclass(frozen=True)
class GenerationTrace:
    lines: list[str]
    tokens: list[str]
    invalid_token: bool = False
    leaked_control_token: bool = False


def is_chinese_char(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff" or ch == "〇"


def has_line_markers(vocab: Vocab) -> bool:
    return all(token in vocab.stoi for token in LINE_TOKENS)


def _rhyme_index(vocab: Vocab) -> dict[str, list[int]]:
    try:
        from pypinyin import Style, lazy_pinyin
    except ImportError:
        return {}
    index: dict[str, list[int]] = {}
    for cid in vocab.char_ids:
        ch = vocab.tokens[cid]
        py = lazy_pinyin(ch, style=Style.FINALS)
        if py and py[0]:
            index.setdefault(py[0], []).append(cid)
    return {final: ids for final, ids in index.items() if len(ids) >= 3}


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


def char_final(char: str) -> str:
    try:
        from pypinyin import Style, lazy_pinyin

        py = lazy_pinyin(char, style=Style.FINALS)
        return py[0] if py else ""
    except ImportError:
        return ""


def rhyme_ready_ids(vocab: Vocab) -> list[int]:
    if _rhyme_cache is None:
        _ = _rhyme_index(vocab)
    return sorted({cid for group in (_rhyme_cache or {}).values() for cid in group})


def avoid_same_rhyme_ids(vocab: Vocab, char: str) -> list[int]:
    target_final = char_final(char)
    if not target_final:
        return vocab.char_ids
    ids = [cid for cid in vocab.char_ids if char_final(vocab.tokens[cid]) != target_final]
    return ids or vocab.char_ids


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


def sample_token(logits: torch.Tensor, sampling: SamplingConfig, allowed_ids: list[int]) -> int:
    allowed = torch.zeros_like(logits, dtype=torch.bool)
    allowed[allowed_ids] = True
    filtered = logits.masked_fill(~allowed, float("-inf"))
    filtered = filtered / max(sampling.temperature, 1e-5)
    filtered = top_k_filter(filtered, sampling.top_k)
    filtered = top_p_filter(filtered, sampling.top_p)
    return int(torch.multinomial(F.softmax(filtered, dim=-1), 1).item())


def sample_char(logits: torch.Tensor, vocab: Vocab, sampling: SamplingConfig) -> int:
    return sample_token(logits, sampling, vocab.char_ids)


def sample_char_rhyme(logits: torch.Tensor, sampling: SamplingConfig, rhyme_candidates: list[int]) -> int:
    return sample_token(logits, sampling, rhyme_candidates)


def validate_prompt(mode: str, prompt: str) -> None:
    expected = 7 if mode == "continue" else 4
    if len(prompt) != expected or not all(is_chinese_char(ch) for ch in prompt):
        raise ValueError(f"{mode} prompt must contain exactly {expected} Chinese characters: {prompt!r}")


def push_punctuation(decoder: IncrementalDecoder, vocab: Vocab, line_idx: int) -> None:
    punct = DEFAULT_LINE_PUNCT[line_idx]
    if punct in vocab.stoi:
        decoder.push(vocab.id(punct))


def parse_generation_trace(tokens: list[str]) -> GenerationTrace:
    lines: list[str] = []
    current: list[str] = []
    invalid = False
    leaked_control = False
    for token in tokens:
        if token == "<EOS>":
            break
        if token in LINE_TOKENS:
            leaked_control = True
            if current:
                lines.append("".join(current))
                current = []
            continue
        if token in PUNCT_TOKENS:
            if current:
                lines.append("".join(current))
                current = []
            continue
        if len(token) == 1 and is_chinese_char(token):
            current.append(token)
        else:
            invalid = True
    if current:
        lines.append("".join(current))
    return GenerationTrace(lines=lines, tokens=tokens, invalid_token=invalid, leaked_control_token=leaked_control)


def run_trace(model: CharPoemLM, vocab: Vocab, prefix: list[str], sampling: SamplingConfig, allowed_ids: list[int], max_steps: int = 48) -> GenerationTrace:
    decoder = IncrementalDecoder(model, vocab.encode(prefix))
    generated_tokens: list[str] = []
    for _ in range(max_steps):
        assert decoder.logits is not None
        token_id = sample_token(decoder.logits, sampling, allowed_ids)
        token = vocab.tokens[token_id]
        decoder.push(token_id)
        generated_tokens.append(token)
        if token == "<EOS>":
            break
    return parse_generation_trace(generated_tokens)


@torch.no_grad()
def generate_poem(
    model: CharPoemLM,
    vocab: Vocab,
    mode: str,
    prompt: str,
    sampling: SamplingConfig,
    structure_constraint: bool = True,
    rhyme_constraint: bool = False,
    truncate_to_four: bool = True,
) -> list[str]:
    validate_prompt(mode, prompt)
    if has_line_markers(vocab):
        if structure_constraint:
            return generate_poem_structured(model, vocab, mode, prompt, sampling, rhyme_constraint)
        return generate_poem_structured_raw(model, vocab, mode, prompt, sampling, truncate_to_four)
    if structure_constraint:
        return generate_poem_plain_constrained(model, vocab, mode, prompt, sampling)
    return generate_poem_plain_raw(model, vocab, mode, prompt, sampling, truncate_to_four)


@torch.no_grad()
def generate_poem_trace(model: CharPoemLM, vocab: Vocab, mode: str, prompt: str, sampling: SamplingConfig) -> GenerationTrace:
    validate_prompt(mode, prompt)
    if has_line_markers(vocab):
        return generate_poem_structured_raw_trace(model, vocab, mode, prompt, sampling)
    return generate_poem_plain_raw_trace(model, vocab, mode, prompt, sampling)


@torch.no_grad()
def generate_poem_structured(model: CharPoemLM, vocab: Vocab, mode: str, prompt: str, sampling: SamplingConfig, rhyme_constraint: bool = False) -> list[str]:
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

    rhyme_candidates: list[int] = []
    line2_rhyme_char = ""
    decoder = IncrementalDecoder(model, vocab.encode(prefix))
    for line_idx in range(start_line, 4):
        line_chars: list[str] = []
        for char_idx in range(7):
            if heads is not None and char_idx == 0:
                token_id = vocab.id(heads[line_idx])
            elif rhyme_constraint and line_idx == 1 and char_idx == 6:
                assert decoder.logits is not None
                token_id = sample_token(decoder.logits, sampling, rhyme_ready_ids(vocab) or vocab.char_ids)
            elif rhyme_constraint and line_idx == 2 and char_idx == 6 and line2_rhyme_char:
                assert decoder.logits is not None
                token_id = sample_token(decoder.logits, sampling, avoid_same_rhyme_ids(vocab, line2_rhyme_char))
            elif rhyme_constraint and line_idx == 3 and char_idx == 6 and rhyme_candidates:
                assert decoder.logits is not None
                token_id = sample_char_rhyme(decoder.logits, sampling, rhyme_candidates)
            else:
                assert decoder.logits is not None
                token_id = sample_char(decoder.logits, vocab, sampling)
            decoder.push(token_id)
            line_chars.append(vocab.tokens[token_id])
        lines.append("".join(line_chars))
        if rhyme_constraint and line_idx == 1:
            line2_rhyme_char = line_chars[-1]
            rhyme_candidates = get_rhyme_candidates(vocab, line_chars[-1])
        push_punctuation(decoder, vocab, line_idx)
        if line_idx < 3:
            decoder.push(vocab.id(f"<L{line_idx + 2}>"))
    return lines


@torch.no_grad()
def generate_poem_structured_raw(model: CharPoemLM, vocab: Vocab, mode: str, prompt: str, sampling: SamplingConfig, truncate_to_four: bool = True) -> list[str]:
    base_lines = [prompt] if mode == "continue" else []
    trace = generate_poem_structured_raw_trace(model, vocab, mode, prompt, sampling)
    lines = [*base_lines, *trace.lines]
    return lines[:4] if truncate_to_four else lines


@torch.no_grad()
def generate_poem_plain_raw(model: CharPoemLM, vocab: Vocab, mode: str, prompt: str, sampling: SamplingConfig, truncate_to_four: bool = True) -> list[str]:
    base_lines = [prompt] if mode == "continue" else []
    trace = generate_poem_plain_raw_trace(model, vocab, mode, prompt, sampling)
    lines = [*base_lines, *trace.lines]
    return lines[:4] if truncate_to_four else lines


@torch.no_grad()
def generate_poem_structured_raw_trace(model: CharPoemLM, vocab: Vocab, mode: str, prompt: str, sampling: SamplingConfig) -> GenerationTrace:
    if mode == "continue":
        prefix = ["<BOS>", "<TASK_CONT>", "<SEP>", *prompt, "<SEP>", "<L2>"]
    elif mode == "acrostic":
        prefix = ["<BOS>", "<TASK_ACRO>", "<SEP>", *prompt, "<SEP>", "<L1>"]
    else:
        raise ValueError("mode must be 'continue' or 'acrostic'")
    extra_ids = [vocab.eos_id, *vocab.punct_ids, *[vocab.id(tok) for tok in LINE_TOKENS]]
    return run_trace(model, vocab, prefix, sampling, sorted(set(vocab.char_ids + extra_ids)))


@torch.no_grad()
def generate_poem_plain_raw_trace(model: CharPoemLM, vocab: Vocab, mode: str, prompt: str, sampling: SamplingConfig) -> GenerationTrace:
    if mode == "continue":
        prefix = ["<BOS>", "<TASK_CONT>", "<SEP>", *prompt, "<SEP>"]
    elif mode == "acrostic":
        prefix = ["<BOS>", "<TASK_ACRO>", "<SEP>", *prompt, "<SEP>"]
    else:
        raise ValueError("mode must be 'continue' or 'acrostic'")
    allowed_ids = sorted(set(vocab.char_ids + vocab.punct_ids + [vocab.eos_id]))
    return run_trace(model, vocab, prefix, sampling, allowed_ids)


@torch.no_grad()
def generate_poem_plain_constrained(model: CharPoemLM, vocab: Vocab, mode: str, prompt: str, sampling: SamplingConfig) -> list[str]:
    if mode == "continue":
        prefix = ["<BOS>", "<TASK_CONT>", "<SEP>", *prompt, "<SEP>"]
        lines = [prompt]
        start_line, heads = 1, None
    elif mode == "acrostic":
        prefix = ["<BOS>", "<TASK_ACRO>", "<SEP>", *prompt, "<SEP>"]
        lines = []
        start_line, heads = 0, prompt
    else:
        raise ValueError("mode must be 'continue' or 'acrostic'")

    decoder = IncrementalDecoder(model, vocab.encode(prefix))
    for line_idx in range(start_line, 4):
        line_chars: list[str] = []
        for char_idx in range(7):
            if heads is not None and char_idx == 0:
                token_id = vocab.id(heads[line_idx])
            else:
                assert decoder.logits is not None
                token_id = sample_char(decoder.logits, vocab, sampling)
            decoder.push(token_id)
            line_chars.append(vocab.tokens[token_id])
        lines.append("".join(line_chars))
        push_punctuation(decoder, vocab, line_idx)
    return lines


def format_poem(lines: list[str]) -> str:
    return "\n".join(f"{line}{DEFAULT_LINE_PUNCT[i] if i < len(DEFAULT_LINE_PUNCT) else ''}" for i, line in enumerate(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate seven-character quatrains.")
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
