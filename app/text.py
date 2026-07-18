"""OCR text normalization, quality checks, and helper predicates.

Extracted from app.controller. Holds the seven module-level regex constants
that gate CJK / Latin / punct classification, plus the small functions that
consume them. None of these touch Controller state; only `app.config`
tuning constants are referenced.
"""

from __future__ import annotations

import re

from app import config

_CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿가-힯]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_ASCII_PUNCT_PREFIX_RE = re.compile(r"^[,.;:!?/\\|+*=~`-]+")
_SHORT_MIXED_CJK_LATIN_TAIL_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿가-힯].*[A-Za-z]{1,3}$")
_SHORT_MIXED_CJK_LATIN_ANY_RE = re.compile(
    r"(?=.*[぀-ヿ㐀-䶿一-鿿豈-﫿가-힯])(?=.*[A-Za-z])[A-Za-z぀-ヿ㐀-䶿一-鿿豈-﫿가-힯0-9]{2,8}$"
)
_DIALOGUE_PUNCT_RE = re.compile(r"[。！？…「」『』：!?]")
_PUNCT_RE = re.compile(r"[^\w぀-ヿ㐀-䶿一-鿿豈-﫿가-힯\s]")


def _mean_confidence_from_words(words: list[object]) -> float:
    values: list[float] = []
    for word in words or []:
        conf = getattr(word, "confidence", None)
        try:
            if conf is not None:
                values.append(float(conf))
        except Exception:
            pass
    if not values:
        return 1.0
    return sum(values) / max(1, len(values))


def _compact_text_len(text: str) -> int:
    return len(re.sub(r"\s+", "", text or ""))


def _contains_dialogue_punct(text: str) -> bool:
    return bool(_DIALOGUE_PUNCT_RE.search(text or ""))


def _looks_like_upper_suffix_junk(norm: str, avg_confidence: float) -> bool:
    compact = re.sub(r"\s+", "", norm or "")
    if not compact or len(compact) > 10:
        return False
    max_conf = float(getattr(config, "OCR_UPPER_SUFFIX_MAX_CONFIDENCE", 0.96))
    if re.search(r"[぀-ヿ㐀-䶿一-鿿][A-Z]{2,}$", compact):
        return avg_confidence < max_conf
    if re.search(r"[A-Z]{2,}[぀-ヿ㐀-䶿一-鿿]$", compact):
        return avg_confidence < max_conf
    if re.search(r"[A-Z]{2,}", compact) and len(_CJK_RE.findall(compact)) <= 2:
        return avg_confidence < max_conf
    return False


def _looks_like_short_mixed_junk(norm: str, avg_confidence: float) -> bool:
    compact = re.sub(r"\s+", "", norm or "")
    if not compact or len(compact) > 8:
        return False
    if _SHORT_MIXED_CJK_LATIN_TAIL_RE.search(compact):
        latin_count = len(_LATIN_RE.findall(compact))
        tail_max_conf = float(getattr(config, "OCR_MIXED_TAIL_MAX_CONFIDENCE", 0.93))
        return latin_count <= 3 and avg_confidence < tail_max_conf
    if _SHORT_MIXED_CJK_LATIN_ANY_RE.search(compact):
        latin_count = len(_LATIN_RE.findall(compact))
        any_max_conf = float(getattr(config, "OCR_MIXED_ANY_MAX_CONFIDENCE", 0.97))
        return latin_count <= 4 and avg_confidence < any_max_conf
    return False


def _text_quality_ok(s: str, avg_confidence: float = 1.0) -> bool:
    if not config.OCR_FILTER_ENABLED:
        return True
    norm = s.replace("\r", "").strip()
    if len(norm) < config.MIN_TEXT_LEN:
        return False
    compact = re.sub(r"\s+", "", norm)
    if len(compact) < config.MIN_TEXT_LEN:
        return False
    cjk = len(_CJK_RE.findall(norm))
    if cjk < config.MIN_CJK_CHARS:
        return False
    punct = len(_PUNCT_RE.findall(norm))
    nonspace = len(compact)
    if nonspace > 0 and (punct / nonspace) > config.MAX_PUNCT_RATIO:
        return False
    if bool(getattr(config, "OCR_REJECT_LEADING_PUNCT_SHORT", True)) and len(compact) <= int(
        getattr(config, "OCR_SHORT_TEXT_MAX_LEN", 12)
    ):
        if _ASCII_PUNCT_PREFIX_RE.match(compact):
            return False
    if bool(getattr(config, "OCR_REJECT_MIXED_SHORT_LATIN_CJK", True)) and _looks_like_short_mixed_junk(
        compact, avg_confidence
    ):
        return False
    if bool(getattr(config, "OCR_REJECT_UPPER_SUFFIX_JUNK", True)) and _looks_like_upper_suffix_junk(
        compact, avg_confidence
    ):
        return False
    min_conf = float(getattr(config, "OCR_MIN_LINE_AVG_CONFIDENCE", 0.0) or 0.0)
    if (
        min_conf > 0
        and avg_confidence < min_conf
        and len(compact) <= int(getattr(config, "OCR_SHORT_TEXT_MAX_LEN", 12))
    ):
        return False
    return True


def _normalize_ocr_text(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = "\n".join(line.strip() for line in t.split("\n"))
    t = re.sub(
        r"(?<=[぀-ヿ㐀-䶿一-鿿豈-﫿＀-￯가-힯])\s+(?=[぀-ヿ㐀-䶿一-鿿豈-﫿＀-￯가-힯])",
        "",
        t,
    )
    t = re.sub(r"\s+([、。！？：；…」』】])", r"\1", t)
    t = re.sub(r"([「『【])\s+", r"\1", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _cleanup_translation(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    if bool(getattr(config, "CLEANUP_STRIP_MARKDOWN", False)):
        t = t.replace("**", "")
    t = "\n".join(line.strip() for line in t.split("\n"))
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _contains_vertical_script(text: str) -> bool:
    """True when the text contains any CJK character.

    Used by the renderer to decide whether a translation can be laid out
    vertically. English / Latin output rendered with writing_mode=vertical
    gets verticalized character-by-character ("Hello" -> H\\ne\\nl\\nl\\no)
    and disables wrap, which overflows narrow source columns. When the
    translation has no CJK chars there is no vertical layout to preserve,
    so the renderer should treat the track as horizontal.
    """
    return bool(_CJK_RE.search(text or ""))


def _join_word_texts(parts: list[str]) -> str:
    cleaned = [str(part).strip() for part in parts if str(part).strip()]
    if not cleaned:
        return ""
    if any(_CJK_RE.search(part) for part in cleaned):
        return "".join(cleaned)
    return " ".join(cleaned)


def _contains_timestamp_like(normalized: str) -> bool:
    text = normalized or ""
    if not text:
        return False
    if re.search(r"\d{4}年\d{1,2}月\d{1,2}日", text):
        return True
    if re.search(r"\d{1,2}時\d{1,2}分", text):
        return True
    if re.search(r"\d{1,2}:\d{2}", text):
        return True
    if re.search(r"[月火水木金土日]\)", text) or re.search(r"[月火水木金土日]曜", text):
        return True
    return False


def _contains_ui_keyword(normalized: str) -> bool:
    text = (normalized or "").replace("\n", " ")
    keywords = (
        "ロールバック",
        "ヒストリー",
        "スキップ",
        "オート",
        "セーブ",
        "ロード",
        "設定",
        "ヘルプ",
        "終了",
        "戻る",
        "ページ",
        "スロット",
        "クイックセーブ",
        "クイックロード",
        "Q.セーブ",
        "Q.ロード",
        "バージョン情報",
        "メインメニュー",
        "オフィス",
        "使用する",
    )
    return any(k in text for k in keywords)


__all__ = [
    "_ASCII_PUNCT_PREFIX_RE",
    "_CJK_RE",
    "_DIALOGUE_PUNCT_RE",
    "_LATIN_RE",
    "_PUNCT_RE",
    "_SHORT_MIXED_CJK_LATIN_ANY_RE",
    "_SHORT_MIXED_CJK_LATIN_TAIL_RE",
    "_cleanup_translation",
    "_compact_text",
    "_compact_text_len",
    "_contains_dialogue_punct",
    "_contains_timestamp_like",
    "_contains_ui_keyword",
    "_contains_vertical_script",
    "_join_word_texts",
    "_looks_like_short_mixed_junk",
    "_looks_like_upper_suffix_junk",
    "_mean_confidence_from_words",
    "_normalize_ocr_text",
    "_text_quality_ok",
]
