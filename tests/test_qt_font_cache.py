"""Regression test for commit 31e41f1.

Before the fix, app/qt_render.py:_font_metrics() recursively called itself
on cache miss instead of constructing QtGui.QFontMetrics(font). The very
first call after a cold cache (which is every fresh process) raised
RecursionError and crashed the overlay paint event.

These tests pin the fix in place.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.usefixtures("qapp")


def test_font_metrics_does_not_recurse_on_cold_cache() -> None:
    from PyQt6.QtGui import QFontMetrics

    from app.qt_render import _FONT_CACHE, _FONT_METRICS_CACHE, _font_metrics, _make_font

    _FONT_CACHE.clear()
    _FONT_METRICS_CACHE.clear()

    font = _make_font(14)
    metrics = _font_metrics(font)  # would RecursionError before the fix
    assert isinstance(metrics, QFontMetrics)
    assert metrics.height() > 0


def test_font_metrics_returns_cached_instance() -> None:
    from app.qt_render import _FONT_METRICS_CACHE, _font_metrics, _make_font

    _FONT_METRICS_CACHE.clear()
    font = _make_font(14)
    first = _font_metrics(font)
    second = _font_metrics(font)
    assert first is second


def test_make_font_returns_cached_instance() -> None:
    from app.qt_render import _FONT_CACHE, _make_font

    _FONT_CACHE.clear()
    f1 = _make_font(14)
    f2 = _make_font(14)
    assert f1 is f2
