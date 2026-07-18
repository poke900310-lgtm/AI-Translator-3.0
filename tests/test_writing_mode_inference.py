"""Writing-mode classification for OCR fragment groups.

The renderer derives ``allow_wrap`` and ``alignment`` from
``Observation.writing_mode``. A vertical observation paints with
``allow_wrap=False`` and ``alignment="center"``, which is correct for
CJK column text but wrong for a stack of horizontal lines.

Regression: a multi-line horizontal menu (each line wider than tall,
but the union of all lines taller than wide) was being mis-classified
as vertical. The renderer then drew the whole translation as a single
non-wrapping line that overflowed or shrunk to fit. Pin the fix:

    Real vertical text never contains a fragment wider than tall, so
    the presence of any wide fragment forces ``horizontal`` regardless
    of how tall+narrow the union sums to.
"""

from __future__ import annotations

from app.controller import _infer_fragment_writing_mode
from app.types import Rect


def _frag(left: int, top: int, width: int, height: int) -> tuple[str, str, Rect]:
    return ("x", "x", Rect(left, top, width, height))


def test_single_horizontal_line_is_horizontal() -> None:
    fragments = [_frag(100, 200, 300, 30)]
    assert _infer_fragment_writing_mode(fragments) == "horizontal"


def test_single_vertical_column_is_vertical() -> None:
    """One tall+narrow rect (a column of stacked CJK glyphs)."""
    fragments = [_frag(100, 100, 30, 200)]
    assert _infer_fragment_writing_mode(fragments) == "vertical"


def test_multi_line_horizontal_menu_is_horizontal() -> None:
    """6-line menu, each line 378x50, left-aligned. Without the wide-
    fragment guard this used to read as 'vertical' because the union
    (378 wide x ~400 tall) is taller than wide and y_span dominates
    x_span (lines share roughly the same left edge)."""
    fragments = [_frag(321, 215 + i * 60, 378, 50) for i in range(6)]
    assert _infer_fragment_writing_mode(fragments) == "horizontal"


def test_stacked_vertical_columns_is_vertical() -> None:
    """Three narrow+tall columns side-by-side (CJK manga panel).
    No fragment is wider than tall, so the wide-fragment guard doesn't
    fire and the vertical inference path runs."""
    fragments = [_frag(100 + i * 40, 100, 30, 200) for i in range(3)]
    assert _infer_fragment_writing_mode(fragments) == "vertical"


def test_empty_returns_horizontal() -> None:
    assert _infer_fragment_writing_mode([]) == "horizontal"


def test_one_wide_fragment_overrides_tall_union() -> None:
    """Even when most fragments look vertical, a single wide line in the
    group means the union is mixed and 'horizontal' is the safer call
    for rendering (so the renderer wraps instead of single-lining)."""
    fragments = [
        _frag(100, 100, 30, 200),
        _frag(100, 320, 30, 200),
        _frag(100, 540, 300, 30),  # one wide fragment breaks the column read
    ]
    assert _infer_fragment_writing_mode(fragments) == "horizontal"
