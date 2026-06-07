from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.engine.evaluate import STRATEGIES
from src.engine.generate import format_poem, generate_poem, load_checkpoint
from src.utils.common import choose_device, seed_everything

CONTINUE_PROMPTS = ["春风又过江南岸", "月落乌啼霜满天", "空山新雨晚来秋", "孤舟夜泊寒江雪", "长安回望绣成堆"]
ACROSTIC_PROMPTS = ["春江花月", "山高水长", "江山如画", "风花雪月", "天地人和"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the required 5+5 report examples.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", default="runs/demo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    model, vocab, _ = load_checkpoint(args.checkpoint, device)
    rows = []
    for mode, prompts in (("continue", CONTINUE_PROMPTS), ("acrostic", ACROSTIC_PROMPTS)):
        for prompt_idx, prompt in enumerate(prompts):
            for strategy_idx, (strategy_name, strategy) in enumerate(STRATEGIES.items()):
                seed_everything(args.seed + 100 * prompt_idx + strategy_idx)
                lines = generate_poem(model, vocab, mode, prompt, strategy)
                rows.append({"mode": mode, "prompt": prompt, "strategy": strategy_name, "lines": lines})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "samples.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    md = ["# 七言绝句生成样例", ""]
    for mode in ("continue", "acrostic"):
        md.extend([f"## {mode}", ""])
        for prompt in CONTINUE_PROMPTS if mode == "continue" else ACROSTIC_PROMPTS:
            md.extend([f"### 输入：{prompt}", ""])
            for row in rows:
                if row["mode"] == mode and row["prompt"] == prompt:
                    md.extend([f"**{row['strategy']}**", "", format_poem(row["lines"]), ""])
    (out_dir / "samples.md").write_text("\n".join(md), encoding="utf-8")
    print(f"saved samples -> {out_dir}")


if __name__ == "__main__":
    main()
