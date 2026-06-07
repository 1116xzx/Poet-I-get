from __future__ import annotations

from pathlib import Path

import torch

from src.data.dataset import PoemDataset, collate_examples
from src.data.preprocess import preprocess
from src.data.tokenizer import Vocab
from src.engine.generate import SamplingConfig, generate_poem
from src.metrics.poem_metrics import acrostic_ok, strict_format_ok
from src.models.char_rnn import CharPoemLM


def test_end_to_end_smoke(tmp_path: Path) -> None:
    raw = tmp_path / "raw.txt"
    raw.write_text(
        "春风又过江南岸，细雨轻沾客子衣。芳草连天人未返，一篙烟水送斜晖。\n"
        "月落乌啼霜满天，江枫渔火照孤眠。姑苏城外疏钟动，客梦随声到远船。\n"
        "山寺微钟落暮林，高窗灯火照书琴。水边一雁归云急，长夜清霜入客襟。\n",
        encoding="utf-8",
    )
    out = tmp_path / "processed"
    stats = preprocess(raw, out)
    assert stats["strict_qijue_count"] == 3

    poems = [
        {"text": "春风又过江南岸细雨轻沾客子衣芳草连天人未返一篙烟水送斜晖", "lines": ["春风又过江南岸", "细雨轻沾客子衣", "芳草连天人未返", "一篙烟水送斜晖"]}
    ]
    vocab = Vocab.build(poems)
    dataset = PoemDataset(poems, vocab)
    batch = collate_examples([dataset[0], dataset[1], dataset[2]], vocab.pad_id)
    model = CharPoemLM(len(vocab.tokens), emb_dim=16, pos_dim=4, hidden_size=32, num_layers=1, dropout=0.0)
    logits = model(batch["input_ids"])
    assert logits.shape[:2] == batch["input_ids"].shape

    lines = generate_poem(model, vocab, "continue", "春风又过江南岸", SamplingConfig(0.9, 10, 1.0))
    assert strict_format_ok(lines)
    acro = generate_poem(model, vocab, "acrostic", "春江花月", SamplingConfig(0.9, 10, 1.0))
    assert acrostic_ok(acro, "春江花月")
