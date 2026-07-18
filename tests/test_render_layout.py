"""Render-layout tests: alignment defaults, font-shrink cap, outline scaling.

User-driven changes:
- The translated overlay text should sit in the patch's top-left corner
  (not centered), so it follows reading order and doesn't visually overlap
  the source row's vertical midline during cross-frame transitions.
- Font size across nearby boxes should look uniform; the prior shrink-to-
  TEXT_MIN_PIXEL behavior produced wildly inconsistent text sizes when one
  translation grew slightly longer than another.
- Outline weight should scale with font size so large text gets a thicker
  halo instead of the same 1px stroke as 12px sign captions.
"""

from __future__ import annotations

import sys

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from app import config


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


# --- alignment ------------------------------------------------------------


def test_default_alignment_is_top_left() -> None:
    """Pin the user-facing default per their explicit request."""
    assert config.TEXT_DEFAULT_ALIGNMENT == "top_left"


def test_alignment_flags_top_right(qapp: QApplication) -> None:
    from app.qt_render import _alignment_flags

    flags = _alignment_flags("top_right", allow_wrap=False)
    assert flags & int(Qt.AlignmentFlag.AlignRight)
    assert flags & int(Qt.AlignmentFlag.AlignTop)
    assert not (flags & int(Qt.AlignmentFlag.AlignVCenter))


def test_alignment_flags_top_left(qapp: QApplication) -> None:
    from app.qt_render import _alignment_flags

    flags = _alignment_flags("top_left", allow_wrap=False)
    assert flags & int(Qt.AlignmentFlag.AlignLeft)
    assert flags & int(Qt.AlignmentFlag.AlignTop)


def test_alignment_flags_legacy_left_keeps_vcenter(qapp: QApplication) -> None:
    """The legacy "left" mode still anchors vertically centered — used by
    code paths that haven't migrated to top_left yet."""
    from app.qt_render import _alignment_flags

    flags = _alignment_flags("left", allow_wrap=False)
    assert flags & int(Qt.AlignmentFlag.AlignLeft)
    assert flags & int(Qt.AlignmentFlag.AlignVCenter)


def test_alignment_flags_word_wrap_bit(qapp: QApplication) -> None:
    from app.qt_render import _alignment_flags

    flags = _alignment_flags("top_right", allow_wrap=True)
    assert flags & int(Qt.TextFlag.TextWordWrap)


def test_render_alignment_uses_configured_default() -> None:
    """Controller._render_alignment must defer to TEXT_DEFAULT_ALIGNMENT for
    multi-line / multi-char tracks (the common dialog case)."""
    from app.controller import _render_alignment
    from app.types import Rect, Track, VisualStyle

    style = VisualStyle((0, 0, 0), (0, 0, 0), (0, 0, 0))
    # A multi-line dialog-class track — was hardcoded to "left" before.
    track = Track(
        track_id=1,
        text="hello",
        normalized="hello",
        rect=Rect(0, 0, 200, 80),
        word_rects=[],
        style=style,
        line_count=2,
        dialogue_hint=True,
    )
    assert _render_alignment(track) == config.TEXT_DEFAULT_ALIGNMENT


def test_render_alignment_repeated_label_uses_top_left() -> None:
    """Repeated UI labels keep top-left so they read like the source sign."""
    from app.controller import _render_alignment
    from app.types import Rect, Track, VisualStyle

    style = VisualStyle((0, 0, 0), (0, 0, 0), (0, 0, 0))
    track = Track(
        track_id=2,
        text="MENU",
        normalized="menu",
        rect=Rect(0, 0, 80, 20),
        word_rects=[],
        style=style,
        repeated_label_hint=True,
    )
    assert _render_alignment(track) == "top_left"


# --- shrink cap -----------------------------------------------------------


def test_shrink_ratio_caps_below_preferred() -> None:
    """The shrink cap must not be > 1.0 (no growth) and not absurdly low."""
    ratio = float(config.TEXT_MAX_SHRINK_RATIO)
    assert 0.6 <= ratio <= 1.0


def test_resolve_layout_does_not_drop_below_shrink_floor(qapp: QApplication) -> None:
    """A translation that doesn't fit at the preferred size and is too long
    to fit at the shrink floor either should still render at the floor —
    NOT collapse to TEXT_MIN_PIXEL."""
    from app.qt_render import _resolve_layout

    item: dict[str, object] = {
        "text": "An absurdly long English translation that will not fit horizontally in this small patch",
        "text_rect": _MockRect(0, 0, 80, 28),
        "preferred_pixel_size": 22,
        "alignment": "top_right",
        "allow_wrap": False,
        "font_family": None,
        "font_weight": None,
        "writing_mode": "horizontal",
    }
    font, _flags, _draw_rect, _txt = _resolve_layout(item)
    preferred = 22
    shrink_floor = int(round(preferred * float(config.TEXT_MAX_SHRINK_RATIO)))
    min_px = int(config.TEXT_MIN_PIXEL)
    assert font.pixelSize() >= shrink_floor, (
        f"font shrank to {font.pixelSize()}px, below the configured "
        f"shrink_floor {shrink_floor}px (preferred {preferred}px). "
        f"That defeats the size-uniformity guarantee."
    )
    # And of course it must not be smaller than the absolute floor either.
    assert font.pixelSize() >= min_px


def test_resolve_layout_keeps_preferred_when_fits(qapp: QApplication) -> None:
    from app.qt_render import _resolve_layout

    item: dict[str, object] = {
        "text": "ok",
        "text_rect": _MockRect(0, 0, 200, 40),
        "preferred_pixel_size": 22,
        "alignment": "top_right",
        "allow_wrap": True,
        "font_family": None,
        "font_weight": None,
        "writing_mode": "horizontal",
    }
    font, _flags, _draw_rect, _txt = _resolve_layout(item)
    assert font.pixelSize() == 22


# --- font hinting + outline scale ----------------------------------------


def test_make_font_sets_full_hinting(qapp: QApplication) -> None:
    from PyQt6.QtGui import QFont

    from app.qt_render import _FONT_CACHE, _make_font

    _FONT_CACHE.clear()
    font = _make_font(14)
    assert font.hintingPreference() == QFont.HintingPreference.PreferFullHinting


def test_outline_offsets_cached_by_radius(qapp: QApplication) -> None:
    from app.qt_render import _OUTLINE_OFFSET_CACHE, _outline_offsets

    _OUTLINE_OFFSET_CACHE.clear()
    a = _outline_offsets(1)
    b = _outline_offsets(1)
    assert a is b, "ring offsets for radius=1 should be cached"
    assert len(a) == 8, "radius=1 should produce 8 ring offsets"


def test_outline_offsets_grow_with_radius(qapp: QApplication) -> None:
    from app.qt_render import _outline_offsets

    r1 = _outline_offsets(1)
    r2 = _outline_offsets(2)
    # Radius 2 produces a ring of 16 cells (the 5×5 ring).
    assert len(r1) < len(r2)


# --- support classes for the layout tests ---------------------------------


class _MockRect:
    """Mimics the Rect attribute-shape expected by _resolve_layout. We don't
    use app.types.Rect because that's frozen and the test only needs the
    four attribute accesses."""

    __slots__ = ("left", "top", "width", "height")

    def __init__(self, left: int, top: int, width: int, height: int) -> None:
        self.left = left
        self.top = top
        self.width = width
        self.height = height
