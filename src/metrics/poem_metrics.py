from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


def strict_format_ok(lines: list[str]) -> bool:
    return len(lines) == 4 and all(len(line) == 7 and all("\u4e00" <= ch <= "\u9fff" or ch == "〇" for ch in line) for line in lines)


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
    """Return the final (韵母) of a Chinese character.  E.g. 桥->iao, 潮->ao."""
    try:
        from pypinyin import Style, lazy_pinyin

        result = lazy_pinyin(ch, style=Style.FINALS)
        return result[0] if result else ""
    except ImportError:
        return ""


def rhyme_tone(ch: str) -> int:
    """Return the tone (1-5, 5=轻声) of a Chinese character."""
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


def rhyme_score(lines: list[str]) -> float:
    """Rate how well a 4-line quatrain rhymes. 0–100 scale.

    绝句 rules:
      - lines 2 & 4 MUST share the same final (最高权重)
      - line 1 may also rhyme with 2/4 (bonus)
      - matching tone (同平仄) on rhyming lines adds precision
      - line 3 should NOT rhyme with 2/4 (绝句第三句不押韵)
    """
    if len(lines) != 4:
        return 0.0

    finals = [final_rhyme(line[-1]) for line in lines]
    tones = [rhyme_tone(line[-1]) for line in lines]

    def _rhymes(a: str, b: str) -> bool:
        if not a or not b:
            return False
        if a == b:
            return True
        if len(a) >= 2 and len(b) >= 2 and (a.endswith(b) or b.endswith(a)):
            return True
        return False

    score = 0.0

    # ── Core: line 2 & 4 (权重 50 分) ──
    if _rhymes(finals[1], finals[3]):
        score += 40.0
        # tone match bonus (同声调更佳)
        if tones[1] == tones[3] and tones[1] > 0:
            score += 10.0
    elif finals[1] and finals[3]:
        # at least share last vowel
        if finals[1][-1] == finals[3][-1]:
            score += 12.0
            # tone match bonus
            if tones[1] == tones[3] and tones[1] > 0:
                score += 3.0

    # ── Bonus: line 1 also rhyming (20 分) ──
    if _rhymes(finals[0], finals[1]):
        score += 15.0
        if tones[0] == tones[1] and tones[0] > 0:
            score += 5.0
    if _rhymes(finals[0], finals[3]):
        score += 15.0
        if tones[0] == tones[3] and tones[0] > 0:
            score += 5.0

    # ── Bonus: line 3 should NOT rhyme (绝句第三句不押韵, 20 分) ──
    if not _rhymes(finals[2], finals[1]) and not _rhymes(finals[2], finals[3]):
        score += 20.0

    # ── Penalty: line 3 wrongly rhymes with 2 or 4 ──
    if _rhymes(finals[2], finals[1]) or _rhymes(finals[2], finals[3]):
        score -= 15.0

    # ── Small bonus: each rhyming pair shares same tone ──
    pair_count = 0
    for i, j in [(1, 3), (0, 1), (0, 3)]:
        if _rhymes(finals[i], finals[j]) and tones[i] == tones[j] and tones[i] > 0:
            pair_count += 1
    score += float(pair_count) * 3.0

    return max(0.0, min(100.0, score))


def rhyme_report(lines: list[str]) -> dict:
    """Return a human-readable rhyme diagnosis for the UI."""
    finals = [final_rhyme(line[-1]) for line in lines]
    tones = [rhyme_tone(line[-1]) for line in lines]
    endings = []
    for i in range(4):
        label = f"{lines[i][-1]}({finals[i]})"
        if tones[i] > 0:
            label += str(tones[i])
        endings.append(label)
    return {
        "endings": endings,
        "line2_line4_rhyme": finals[1] == finals[3] if len(finals) >= 4 else False,
        "score": rhyme_score(lines),
    }


@dataclass
class CopyRiskIndex:
    """Fast nearest-neighbour overlap check using an inverted 4-gram index."""

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
