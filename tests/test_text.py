"""OCR text normalization, quality, and predicate tests."""

from __future__ import annotations

from app.text import (
    _CJK_RE,
    _DIALOGUE_PUNCT_RE,
    _LATIN_RE,
    _cleanup_translation,
    _compact_text,
    _compact_text_len,
    _contains_dialogue_punct,
    _contains_timestamp_like,
    _contains_ui_keyword,
    _join_word_texts,
    _looks_like_short_mixed_junk,
    _looks_like_upper_suffix_junk,
    _mean_confidence_from_words,
    _normalize_ocr_text,
    _text_quality_ok,
)

# --- regex constants --------------------------------------------------------


def test_cjk_regex_matches_hiragana() -> None:
    assert _CJK_RE.search("あ")


def test_cjk_regex_matches_kanji() -> None:
    assert _CJK_RE.search("漢")


def test_cjk_regex_rejects_latin() -> None:
    assert _CJK_RE.search("ABC") is None


def test_latin_regex_matches() -> None:
    assert _LATIN_RE.search("Z")


def test_dialogue_punct_regex() -> None:
    assert _DIALOGUE_PUNCT_RE.search("こんにちは。")
    assert _DIALOGUE_PUNCT_RE.search("hello?") is not None
    assert _DIALOGUE_PUNCT_RE.search("hello") is None


# --- normalization / cleanup ------------------------------------------------


def test_normalize_collapses_multispace() -> None:
    assert _normalize_ocr_text("a    b   c") == "a b c"


def test_normalize_strips_per_line() -> None:
    assert _normalize_ocr_text("  hi  \n  there  ") == "hi\nthere"


def test_normalize_joins_cjk_split_by_whitespace() -> None:
    # The regex glues CJK chars across stray spaces — common OCR artifact.
    assert _normalize_ocr_text("こん にちは") == "こんにちは"


def test_normalize_handles_empty() -> None:
    assert _normalize_ocr_text("") == ""


def test_cleanup_translation_collapses_blank_lines() -> None:
    out = _cleanup_translation("hi\n\n\n\nthere")
    assert out == "hi\n\nthere"


def test_cleanup_translation_empty() -> None:
    assert _cleanup_translation("") == ""


# --- compact / join ---------------------------------------------------------


def test_compact_text_strips_whitespace() -> None:
    assert _compact_text("a b\tc\n") == "abc"


def test_compact_text_len_counts_nonspace() -> None:
    assert _compact_text_len("a b c") == 3
    assert _compact_text_len("   ") == 0


def test_join_word_texts_cjk_no_spaces() -> None:
    assert _join_word_texts(["こん", "にちは"]) == "こんにちは"


def test_join_word_texts_latin_with_spaces() -> None:
    assert _join_word_texts(["hello", "world"]) == "hello world"


def test_join_word_texts_filters_empty() -> None:
    assert _join_word_texts(["", "  ", "x"]) == "x"


# --- predicates -------------------------------------------------------------


def test_contains_dialogue_punct() -> None:
    assert _contains_dialogue_punct("これは何？") is True
    assert _contains_dialogue_punct("普通の文") is False


def test_contains_timestamp_like_iso() -> None:
    assert _contains_timestamp_like("12:34") is True


def test_contains_timestamp_like_jp() -> None:
    assert _contains_timestamp_like("2024年5月17日") is True


def test_contains_timestamp_like_none() -> None:
    assert _contains_timestamp_like("hello world") is False


def test_contains_ui_keyword_hits() -> None:
    assert _contains_ui_keyword("セーブする") is True
    assert _contains_ui_keyword("ロード") is True


def test_contains_ui_keyword_miss() -> None:
    assert _contains_ui_keyword("彼は走った") is False


def test_mean_confidence_empty_is_one() -> None:
    assert _mean_confidence_from_words([]) == 1.0


def test_mean_confidence_averaged() -> None:
    class W:
        def __init__(self, c: float) -> None:
            self.confidence = c

    assert abs(_mean_confidence_from_words([W(0.5), W(0.9)]) - 0.7) < 1e-9


# --- quality gates ----------------------------------------------------------


def test_text_quality_ok_short_latin_rejected() -> None:
    # Default OCR_FILTER_ENABLED + MIN_CJK_CHARS=1 means Latin-only short text
    # is rejected.
    assert _text_quality_ok("ab", avg_confidence=1.0) is False


def test_text_quality_ok_long_jp_accepted() -> None:
    assert _text_quality_ok("これは普通の日本語の文章です。", avg_confidence=1.0) is True


def test_looks_like_short_mixed_junk_low_conf() -> None:
    # Short CJK+Latin tail with low confidence is junk.
    assert _looks_like_short_mixed_junk("漢AB", avg_confidence=0.5) is True


def test_looks_like_short_mixed_junk_long_text_passes() -> None:
    # Above the 8-char compact threshold => not junk.
    assert _looks_like_short_mixed_junk("これはとても長い文章AB", avg_confidence=0.1) is False


def test_looks_like_upper_suffix_junk_low_conf() -> None:
    assert _looks_like_upper_suffix_junk("漢XX", avg_confidence=0.5) is True


def test_looks_like_upper_suffix_junk_high_conf_pass() -> None:
    # High confidence overrides the junk classification.
    assert _looks_like_upper_suffix_junk("漢XX", avg_confidence=0.99) is False
