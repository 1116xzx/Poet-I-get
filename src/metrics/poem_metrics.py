from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


def is_chinese_char(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff" or ch == "〇"


def strict_format_ok(lines: list[str]) -> bool:
    return len(lines) == 4 and all(len(line) == 7 and all(is_chinese_char(ch) for ch in line) for line in lines)


def ngrams(text: str, n: int) -> set[str]:
    return {text[i : i + n] for i in range(max(0, len(text) - n + 1))}


def distinct_n(text: str, n: int = 2) -> float:
    grams = [text[i : i + n] for i in range(max(0, len(text) - n + 1))]
    return len(set(grams)) / len(grams) if grams else 0.0


def repeat_rate(text: str, n: int = 2) -> float:
    return 1.0 - distinct_n(text, n)


def acrostic_ok(lines: list[str], heads: str) -> bool:
    return strict_format_ok(lines) and "".join(line[0] for line in lines) == heads


def final_rhyme(ch: str) -> str:
    try:
        from pypinyin import Style, lazy_pinyin

        result = lazy_pinyin(ch, style=Style.FINALS)
        return normalize_final(result[0]) if result else ""
    except ImportError:
        return ""


def normalize_final(final: str) -> str:
    if not final:
        return ""
    mapping = {
        "iu": "ou",
        "iou": "ou",
        "ui": "ei",
        "uei": "ei",
        "un": "en",
        "uen": "en",
        "ve": "ue",
        "ue": "ue",
        "iang": "ang",
        "uang": "ang",
    }
    return mapping.get(final, final)


def rhyme_tone(ch: str) -> int:
    try:
        from pypinyin import Style, lazy_pinyin

        result = lazy_pinyin(ch, style=Style.TONE3)
        if result:
            digit = result[0][-1]
            if digit.isdigit():
                return int(digit)
        return 0
    except ImportError:
        return 0


def _rhymes(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    return len(a) >= 2 and len(b) >= 2 and (a.endswith(b) or b.endswith(a))


def _near_rhymes(a: str, b: str) -> bool:
    if _rhymes(a, b):
        return False
    if not a or not b:
        return False
    return a[-1] == b[-1]


def rhyme_report(lines: list[str]) -> dict:
    safe_lines = [line for line in lines if line]
    finals = [final_rhyme(line[-1]) for line in safe_lines]
    tones = [rhyme_tone(line[-1]) for line in safe_lines]
    endings = []
    for i, line in enumerate(safe_lines):
        label = f"{line[-1]}({finals[i]})"
        if tones[i] > 0:
            label += str(tones[i])
        endings.append(label)

    if len(safe_lines) < 4:
        return {
            "endings": endings,
            "line2_line4_rhyme": False,
            "line2_line4_near_rhyme": False,
            "line1_into_rhyme": False,
            "line3_avoids_rhyme": False,
            "tone_bonus": 0.0,
            "components": {},
            "score": 0.0,
        }

    score_24 = 0.0
    score_1 = 0.0
    score_3 = 0.0
    tone_bonus = 0.0

    if _rhymes(finals[1], finals[3]):
        score_24 = 50.0
    elif _near_rhymes(finals[1], finals[3]):
        score_24 = 20.0

    if _rhymes(finals[0], finals[1]) or _rhymes(finals[0], finals[3]):
        score_1 = 15.0

    if not _rhymes(finals[2], finals[1]) and not _rhymes(finals[2], finals[3]):
        score_3 = 20.0
    else:
        score_3 = -15.0

    for i, j in [(1, 3), (0, 1), (0, 3)]:
        if _rhymes(finals[i], finals[j]) and tones[i] == tones[j] and tones[i] > 0:
            tone_bonus += 3.0
    tone_bonus = min(tone_bonus, 10.0)
    score = max(0.0, min(100.0, score_24 + score_1 + score_3 + tone_bonus))
    return {
        "endings": endings,
        "line2_line4_rhyme": _rhymes(finals[1], finals[3]),
        "line2_line4_near_rhyme": _near_rhymes(finals[1], finals[3]),
        "line1_into_rhyme": score_1 > 0,
        "line3_avoids_rhyme": score_3 > 0,
        "tone_bonus": tone_bonus,
        "components": {
            "line2_line4": score_24,
            "line1_bonus": score_1,
            "line3_avoid": score_3,
            "tone_bonus": tone_bonus,
        },
        "score": score,
    }


def rhyme_score(lines: list[str]) -> float:
    if len(lines) != 4 or any(not line for line in lines):
        return 0.0
    return float(rhyme_report(lines)["score"])


@dataclass
class CopyRiskIndex:
    texts: list[str]
    n: int = 4

    def __post_init__(self) -> None:
        self.text_grams = [ngrams(text, self.n) for text in self.texts]
        self.postings: dict[str, set[int]] = defaultdict(set)
        for idx, grams in enumerate(self.text_grams):
            for gram in grams:
                self.postings[gram].add(idx)

    def max_jaccard(self, text: str) -> float:
        query = ngrams(text, self.n)
        candidates: set[int] = set()
        for gram in query:
            candidates.update(self.postings.get(gram, ()))
        best = 0.0
        for idx in candidates:
            ref = self.text_grams[idx]
            union = len(query | ref)
            if union:
                best = max(best, len(query & ref) / union)
        return best
