"""Flask backend — serves the best checkpoint behind a clean Apple-style UI."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template, request

# Make sure sibling packages are importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.engine.generate import (
    IncrementalDecoder,
    SamplingConfig,
    generate_poem,
    load_checkpoint,
    top_k_filter,
    top_p_filter,
    validate_prompt,
)
from src.metrics.poem_metrics import rhyme_report, rhyme_score
from src.utils.common import choose_device

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

# ── Bootstrap: load checkpoint once at startup ──────────────────────
CHECKPOINT = Path("checkpoints/gru_best.pt")
device = choose_device("auto")
model, vocab, meta = load_checkpoint(CHECKPOINT, device)
print(f"[OK] Checkpoint loaded -- epoch {meta.get('epoch','?')}  val_ppl={meta.get('val_ppl','?'):.2f}")


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def api_generate() -> tuple:
    body = request.get_json(force=True)
    mode: str = body.get("mode", "continue")
    prompt: str = body.get("prompt", "")
    temperature: float = float(body.get("temperature", 0.9))
    top_k: int = int(body.get("top_k", 20))
    top_p: float = float(body.get("top_p", 1.0))
    seed: int = int(body.get("seed", random.randint(0, 2**31)))
    structure_constraint: bool = bool(body.get("structure_constraint", False))

    try:
        validate_prompt(mode, prompt)
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    rhyme_constraint: bool = bool(body.get("rhyme_constraint", False))
    sampling = SamplingConfig(temperature=temperature, top_k=top_k, top_p=top_p)
    torch.manual_seed(seed)
    lines = generate_poem(
        model,
        vocab,
        mode,
        prompt,
        sampling,
        structure_constraint=structure_constraint,
        rhyme_constraint=structure_constraint and rhyme_constraint,
    )

    return jsonify({
        "ok": True,
        "mode": mode,
        "prompt": prompt,
        "lines": lines,
        "seed": seed,
        "structure_constraint": structure_constraint,
        "rhyme_constraint": rhyme_constraint,
    })


@app.route("/api/generate_best", methods=["POST"])
def api_generate_best() -> tuple:
    """Generate N candidates and return the best according to rhyme score."""
    body = request.get_json(force=True)
    mode: str = body.get("mode", "continue")
    prompt: str = body.get("prompt", "")
    temperature: float = float(body.get("temperature", 0.9))
    top_k: int = int(body.get("top_k", 20))
    top_p: float = float(body.get("top_p", 1.0))
    num_candidates: int = int(body.get("num_candidates", 10))
    base_seed: int = int(body.get("seed", random.randint(0, 2**31)))
    structure_constraint: bool = bool(body.get("structure_constraint", False))
    rhyme_constraint: bool = bool(body.get("rhyme_constraint", True))

    try:
        validate_prompt(mode, prompt)
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    if num_candidates < 1:
        num_candidates = 1
    if num_candidates > 50:
        num_candidates = 50  # safety cap

    sampling = SamplingConfig(temperature=temperature, top_k=top_k, top_p=top_p)

    best_lines: list[str] | None = None
    best_score: float = -1.0
    all_candidates: list[dict] = []

    for i in range(num_candidates):
        torch.manual_seed(base_seed + i)
        lines = generate_poem(
            model,
            vocab,
            mode,
            prompt,
            sampling,
            structure_constraint=structure_constraint,
            rhyme_constraint=structure_constraint and rhyme_constraint,
        )
        score = rhyme_score(lines)
        rhyme_info = rhyme_report(lines)
        candidate = {
            "lines": lines,
            "rhyme_score": round(score, 2),
            "rhyme_info": rhyme_info,
        }
        all_candidates.append(candidate)
        if score > best_score:
            best_score = score
            best_lines = lines

    # Sort by rhyme score descending
    all_candidates.sort(key=lambda c: c["rhyme_score"], reverse=True)

    return jsonify({
        "ok": True,
        "mode": mode,
        "prompt": prompt,
        "lines": best_lines,  # ← best candidate (backward-compatible field)
        "seed": base_seed,
        "rhyme_score": round(best_score, 2),
        "num_candidates": num_candidates,
        "all_candidates": all_candidates[:5],  # top 5 for optional display
        "structure_constraint": structure_constraint,
        "rhyme_constraint": rhyme_constraint,
    })


@app.route("/api/pinyin", methods=["POST"])
def api_pinyin() -> tuple:
    """Convert Chinese text to pinyin-annotated characters."""
    body = request.get_json(force=True)
    text: str = body.get("text", "")

    try:
        from pypinyin import Style, lazy_pinyin

        chars = list(text)
        pinyins = lazy_pinyin(text, style=Style.TONE)
        result = [
            {"char": ch, "pinyin": py}
            for ch, py in zip(chars, pinyins)
        ]
        return jsonify({"ok": True, "annotations": result})
    except ImportError:
        return jsonify({"ok": False, "error": "pypinyin not installed"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5100, debug=False)
