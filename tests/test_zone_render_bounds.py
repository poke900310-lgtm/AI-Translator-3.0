"""Translated text must stay inside the user-drawn translation zone.

Bug: even after the grouped-zone-rect fix, long translations of dialogue-
class tracks could spill past the user's zone boundary because
``_planned_text_rect`` is allowed to grow the text rect up to
``TEXT_DIALOGUE_MAX_HEIGHT_RATIO`` × the source height. The downstream
``_move_rect_inside`` only clamps to image bounds, not zone bounds.

Fix: ``_render_bounds_for_rect(rect, image, expand_to_zone=True)``
expands the bounds to the enclosing user zone (clipped to image) so the
downstream text-rect clamp ends up bounded by the zone — a long
translation wraps within the zone instead of overflowing.
"""

from __future__ import annotations

import sys

import pytest
from PIL import Image
from PyQt6.QtWidgets import QApplication

from app.controller import Controller
from app.types import Rect, TranslationZone


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _ctl_with_zones(qapp: QApplication, zones: list[TranslationZone]) -> Controller:
    ctl = Controller(lang_tag=None)
    ctl._translation_zones = list(zones)
    return ctl


def test_bounds_clip_to_zone_when_rect_inside_zone(qapp: QApplication) -> None:
    """A rect inside the zone gets render_bounds clipped to that zone,
    so any downstream growth is capped by the zone box."""
    img = Image.new("RGB", (800, 600), (0, 0, 0))
    zone = TranslationZone(rect=Rect(50, 400, 400, 100))  # 50,400 -> 450,500
    ctl = _ctl_with_zones(qapp, [zone])
    rect = Rect(100, 420, 200, 30)  # firmly inside the zone
    bounds = ctl._render_bounds_for_rect(rect, img, expand_to_zone=True)
    # The bounds should be the zone rect, not the whole image.
    assert bounds.left == 50
    assert bounds.top == 400
    assert bounds.width == 400
    assert bounds.height == 100


def test_bounds_match_legacy_when_no_zones(qapp: QApplication) -> None:
    """When the user hasn't drawn any zones, the helper falls back to
    the legacy ``_render_bounds_for_rect`` behaviour — rect intersected
    with the allowed edge-region (not the whole image)."""
    img = Image.new("RGB", (800, 600), (0, 0, 0))
    ctl = _ctl_with_zones(qapp, [])
    rect = Rect(100, 420, 200, 30)
    bounds = ctl._render_bounds_for_rect(rect, img, expand_to_zone=True)
    legacy = ctl._render_bounds_for_rect(rect, img)
    assert bounds == legacy


def test_bounds_match_legacy_when_track_outside_every_zone(qapp: QApplication) -> None:
    """A track that doesn't overlap any zone falls back to the legacy
    bounds — the user only constrained certain regions, not the entire
    screen, so non-zone tracks render normally."""
    img = Image.new("RGB", (800, 600), (0, 0, 0))
    zone = TranslationZone(rect=Rect(0, 400, 800, 100))
    ctl = _ctl_with_zones(qapp, [zone])
    rect = Rect(100, 50, 200, 30)  # well above the zone
    bounds = ctl._render_bounds_for_rect(rect, img, expand_to_zone=True)
    legacy = ctl._render_bounds_for_rect(rect, img)
    assert bounds == legacy


def test_bounds_pick_best_overlapping_zone(qapp: QApplication) -> None:
    """When two zones exist and the rect straddles both, the zone with
    the most overlap wins (same rule as _zone_index_for_rect)."""
    img = Image.new("RGB", (800, 600), (0, 0, 0))
    zones = [
        TranslationZone(rect=Rect(0, 0, 400, 600)),  # left half
        TranslationZone(rect=Rect(400, 0, 400, 600)),  # right half
    ]
    ctl = _ctl_with_zones(qapp, zones)
    rect = Rect(450, 100, 100, 30)  # 100px wide, all in right half
    bounds = ctl._render_bounds_for_rect(rect, img, expand_to_zone=True)
    assert bounds.left == 400
    assert bounds.width == 400
