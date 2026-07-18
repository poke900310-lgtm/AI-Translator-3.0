"""Vertical-script detection for translation rendering.

Bug repro from debug/logs/render.log (run 20260601-062245-531):
patches like ``patch=710,277,48x134 text_h=126 wrap=False align=center``
showed Latin translations of vertically-oriented CJK source bleeding
past narrow source-column patch widths. The renderer's vertical path:

  1. inherits writing_mode="vertical" from the track (source was CJK column)
  2. verticalizes the English translation char-by-char ("Hello" -> "H\\ne\\nl\\nl\\no")
  3. forces allow_wrap=False, alignment="center"
  4. Qt's drawText with no TextWordWrap doesn't clip — text renders past
     the rect bounds because there's nothing to break the line

Fix: when a translation contains no CJK characters there's no vertical
layout to preserve, so the renderer should treat the track as horizontal.
This test pins ``_contains_vertical_script`` since the controller uses it
to short-circuit the vertical path.
"""

from __future__ import annotations

from app.text import _contains_vertical_script


def test_pure_english_returns_false() -> None:
    assert _contains_vertical_script("Hello world") is False


def test_pure_japanese_returns_true() -> None:
    assert _contains_vertical_script("こんにちは") is True


def test_japanese_kanji_returns_true() -> None:
    assert _contains_vertical_script("日本語") is True


def test_korean_hangul_returns_true() -> None:
    assert _contains_vertical_script("안녕하세요") is True


def test_chinese_returns_true() -> None:
    assert _contains_vertical_script("你好世界") is True


def test_mixed_cjk_and_latin_returns_true() -> None:
    """Even one CJK char anywhere flips the result, since the original
    intent for vertical rendering may still apply to the CJK portion."""
    assert _contains_vertical_script("Hello 世界 world") is True


def test_punctuation_only_returns_false() -> None:
    assert _contains_vertical_script("...!?") is False


def test_empty_string_returns_false() -> None:
    assert _contains_vertical_script("") is False
    assert _contains_vertical_script(None) is False  # type: ignore[arg-type]
