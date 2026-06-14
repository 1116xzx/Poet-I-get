"""Flask backend — serves the best checkpoint behind a clean Apple-style UI."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template, request

# Make sure sibling packages are importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
STATIC_DIR = PROJECT_ROOT / "static"

from src.engine.chengyu_grid import load_idioms, sample_diverse_idioms, search_acrostic_grid
from src.engine.chengyu_global_beam_experiment import search_with_global_beam
from src.engine.generate import SamplingConfig, generate_poem, load_checkpoint, validate_prompt
from src.engine.global_prefix_score import load_global_prefix_scorer
from src.metrics.poem_metrics import rhyme_report, rhyme_score
from src.utils.common import choose_device

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
app.config["TEMPLATES_AUTO_RELOAD"] = True

# ── Bootstrap: load checkpoint once at startup ──────────────────────
device = choose_device("auto")
MODEL_CONFIGS = {
    "baseline": {"checkpoint": PROJECT_ROOT / "checkpoints/gru_plain_best.pt", "structure_constraint": False},
    "weighted": {"checkpoint": PROJECT_ROOT / "checkpoints/gru_plain_weighted_best.pt", "structure_constraint": False},
    "structured": {"checkpoint": PROJECT_ROOT / "checkpoints/gru_best.pt", "structure_constraint": True},
}
IDIOMS_PATH = PROJECT_ROOT / "data/processed/chengyu/idioms.txt"
GLOBAL_SCORER_PATH = PROJECT_ROOT / "checkpoints/global_bigru_scorer_20e.pt"
GLOBAL_PREFIX_SCORER_PATH = PROJECT_ROOT / "checkpoints/global_prefix_bigru_20e.pt"
MODEL_BUNDLES: dict[str, dict] = {}
for model_name, cfg in MODEL_CONFIGS.items():
    loaded_model, loaded_vocab, loaded_meta = load_checkpoint(cfg["checkpoint"], device)
    MODEL_BUNDLES[model_name] = {
        "model": loaded_model,
        "vocab": loaded_vocab,
        "meta": loaded_meta,
        **cfg,
    }
    print(
        f"[OK] Loaded {model_name} -- epoch {loaded_meta.get('epoch', '?')} "
        f"val_ppl={loaded_meta.get('val_ppl', '?'):.2f}"
    )

GLOBAL_PREFIX_BUNDLE: dict | None = None
if GLOBAL_PREFIX_SCORER_PATH.exists():
    prefix_model, prefix_vocab, prefix_meta = load_global_prefix_scorer(GLOBAL_PREFIX_SCORER_PATH, device)
    GLOBAL_PREFIX_BUNDLE = {"model": prefix_model, "vocab": prefix_vocab, "meta": prefix_meta}
    print(
        f"[OK] Loaded global prefix scorer -- epoch {prefix_meta.get('epoch', '?')} "
        f"val_loss={prefix_meta.get('val_loss', '?')}"
    )


def get_model_bundle(model_name: str) -> dict:
    return MODEL_BUNDLES.get(model_name, MODEL_BUNDLES["structured"])


def resolve_model_name(model_name: str, rhyme_constraint: bool, mode: str | None = None) -> str:
    if mode == "chengyu":
        return "structured"
    if rhyme_constraint:
        return "structured"
    return model_name if model_name in MODEL_BUNDLES else "structured"


def validate_prompt_for_mode(mode: str, prompt: str) -> None:
    if mode == "chengyu":
        if len(prompt) != 4 or any(not ("\u4e00" <= ch <= "\u9fff") for ch in prompt):
            raise ValueError(f"{mode} prompt must contain exactly 4 Chinese characters: {prompt!r}")
        return
    validate_prompt(mode, prompt)


def generate_chengyu_candidates(body: dict, mode: str, prompt: str, model_name: str) -> dict:
    bundle = get_model_bundle(model_name)
    num_candidates = int(body.get("num_candidates", 5))
    num_candidates = max(1, min(num_candidates, 10))
    candidate_limit = int(body.get("candidate_limit", 240))
    beam_size = int(body.get("beam_size", 8))
    nll_weight = float(body.get("nll_weight", 9.0))
    repeat_weight = float(body.get("repeat_weight", 11.0))
    phrase_weight = float(body.get("phrase_weight", 10.0))
    style_weight = float(body.get("style_weight", 5.0))
    rhyme_weight = float(body.get("rhyme_weight", 12.0))
    prefix_repeat_weight = float(body.get("prefix_repeat_weight", 2.0))
    prefix_phrase_weight = float(body.get("prefix_phrase_weight", 2.5))
    prefix_style_weight = float(body.get("prefix_style_weight", 1.2))
    enable_global_rerank = bool(body.get("enable_global_rerank", True))
    global_weight = float(body.get("global_weight", 4.0))
    global_rerank_top_n = int(body.get("global_rerank_top_n", 12))
    poetic_only = not bool(body.get("all_idioms", False))
    diverse_idioms = bool(body.get("diverse_idioms", True))
    full_idiom_pool = bool(body.get("full_idiom_pool", True))
    pool_seed = int(body.get("seed", random.randint(0, 2**31)))

    idioms = load_idioms(
        IDIOMS_PATH,
        bundle["vocab"],
        limit=0 if full_idiom_pool else candidate_limit,
        poetic_only=poetic_only,
        allow_repeated_idioms=False,
        diverse=diverse_idioms,
    )
    source_idiom_count = len(idioms)
    if full_idiom_pool:
        idioms = sample_diverse_idioms(idioms, candidate_limit, seed=pool_seed)
    if not idioms:
        raise ValueError("成语库为空，暂时无法生成成积似涵结果。")
    if GLOBAL_PREFIX_BUNDLE is not None:
        candidates = search_with_global_beam(
            prompt,
            bundle["model"],
            bundle["vocab"],
            idioms,
            GLOBAL_PREFIX_BUNDLE["model"],
            GLOBAL_PREFIX_BUNDLE["vocab"],
            device,
            beam_size=beam_size,
            top_n=num_candidates,
            nll_weight=nll_weight,
            repeat_weight=repeat_weight,
            phrase_weight=phrase_weight,
            style_weight=style_weight,
            rhyme_weight=rhyme_weight,
            prefix_repeat_weight=prefix_repeat_weight,
            prefix_phrase_weight=prefix_phrase_weight,
            prefix_style_weight=prefix_style_weight,
        )
    else:
        candidates = search_acrostic_grid(
            prompt,
            bundle["model"],
            bundle["vocab"],
            idioms,
            beam_size=beam_size,
            top_n=num_candidates,
            nll_weight=nll_weight,
            repeat_weight=repeat_weight,
            phrase_weight=phrase_weight,
            style_weight=style_weight,
            rhyme_weight=rhyme_weight,
            global_scorer_path=GLOBAL_SCORER_PATH if enable_global_rerank and GLOBAL_SCORER_PATH.exists() else None,
            global_rerank_top_n=global_rerank_top_n,
            global_weight=global_weight,
        )
    best = candidates[0]
    all_candidates = [
        {
            "lines": candidate.lines,
            "columns": list(candidate.columns),
            "score": round(candidate.score, 4),
            "nll": round(candidate.nll, 4),
            "repeat_penalty": round(candidate.repeat_penalty, 4),
            "phrase_penalty": round(candidate.phrase_penalty, 4),
            "style_penalty": round(candidate.style_penalty, 4),
            "rhyme_score": round(candidate.rhyme_score_value, 2),
            "global_pll": round(candidate.global_pll, 4),
            "base_rank": candidate.base_rank,
            "rerank_delta": round(candidate.rerank_delta, 4),
        }
        for candidate in candidates
    ]
    return {
        "ok": True,
        "model": model_name,
        "requested_model": model_name,
        "mode": mode,
        "prompt": prompt,
        "seed": None,
        "lines": best.lines,
        "grid_columns": list(best.columns),
        "score": round(best.score, 4),
        "nll": round(best.nll, 4),
        "repeat_penalty": round(best.repeat_penalty, 4),
        "phrase_penalty": round(best.phrase_penalty, 4),
        "style_penalty": round(best.style_penalty, 4),
        "rhyme_score": round(best.rhyme_score_value, 2),
        "global_pll": round(best.global_pll, 4),
        "base_rank": best.base_rank,
        "rerank_delta": round(best.rerank_delta, 4),
        "num_candidates": num_candidates,
        "source_idiom_count": source_idiom_count,
        "active_idiom_count": len(idioms),
        "all_candidates": all_candidates,
    }


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def api_generate() -> tuple:
    body = request.get_json(force=True)
    mode: str = body.get("mode", "continue")
    prompt: str = body.get("prompt", "")
    requested_model: str = body.get("model", "structured")
    temperature: float = float(body.get("temperature", 1.0))
    seed: int = int(body.get("seed", random.randint(0, 2**31)))
    rhyme_constraint: bool = bool(body.get("rhyme_constraint", False))
    model_name = resolve_model_name(requested_model, rhyme_constraint, mode)
    bundle = get_model_bundle(model_name)
    structure_constraint: bool = bool(bundle["structure_constraint"])

    try:
        validate_prompt_for_mode(mode, prompt)
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    if mode == "chengyu":
        try:
            return jsonify(generate_chengyu_candidates(body, mode, prompt, model_name))
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    sampling = SamplingConfig(temperature=temperature)
    torch.manual_seed(seed)
    lines = generate_poem(
        bundle["model"],
        bundle["vocab"],
        mode,
        prompt,
        sampling,
        structure_constraint=structure_constraint,
        rhyme_constraint=structure_constraint and rhyme_constraint,
    )

    return jsonify({
        "ok": True,
        "model": model_name,
        "requested_model": requested_model,
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
    requested_model: str = body.get("model", "structured")
    temperature: float = float(body.get("temperature", 1.0))
    num_candidates: int = int(body.get("num_candidates", 10))
    base_seed: int = int(body.get("seed", random.randint(0, 2**31)))
    rhyme_constraint: bool = bool(body.get("rhyme_constraint", True))
    model_name = resolve_model_name(requested_model, rhyme_constraint, mode)
    bundle = get_model_bundle(model_name)
    structure_constraint: bool = bool(bundle["structure_constraint"])

    try:
        validate_prompt_for_mode(mode, prompt)
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    if mode == "chengyu":
        try:
            return jsonify(generate_chengyu_candidates(body, mode, prompt, model_name))
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    if num_candidates < 1:
        num_candidates = 1
    if num_candidates > 50:
        num_candidates = 50  # safety cap

    sampling = SamplingConfig(temperature=temperature)

    best_lines: list[str] | None = None
    best_score: float = -1.0
    all_candidates: list[dict] = []

    for i in range(num_candidates):
        torch.manual_seed(base_seed + i)
        lines = generate_poem(
            bundle["model"],
            bundle["vocab"],
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
        "model": model_name,
        "requested_model": requested_model,
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
