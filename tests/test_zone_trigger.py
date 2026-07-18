"""Regression tests for the per-zone trigger gate (commit 7619c6f).

Before the fix, a pixel delta in zone A caused OCR re-runs that produced
fresh observations for zone B too. OCR jitter in B's text then reset its
stable_frames and re-queued B for translation — so editing zone A's
contents perturbed every other zone.

The fix: per-zone pixel-change signatures, plus `_track_zone_changed`
which returns False (== don't requeue) when the track's enclosing zone
didn't actually move pixels this frame.

These tests exercise both methods against a real Controller with two
synthetic zones.
"""

from __future__ import annotations

import sys

import pytest
from PIL import Image
from PyQt6.QtWidgets import QApplication

from app.controller import Controller
from app.types import Rect, Track, TranslationZone, VisualStyle


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _controller(qapp: QApplication) -> Controller:
    return Controller(lang_tag=None)


def _solid(color: tuple[int, int, int], size: tuple[int, int] = (300, 200)) -> Image.Image:
    return Image.new("RGB", size, color)


def _make_track(track_id: int, rect: Rect) -> Track:
    return Track(
        track_id=track_id,
        text="x",
        normalized="x",
        rect=rect,
        word_rects=[],
        style=VisualStyle((0, 0, 0), (0, 0, 0), (0, 0, 0)),
    )


def test_no_zones_means_every_track_reports_changed(qapp: QApplication) -> None:
    """When no user zones are configured, _track_zone_changed must return
    True for every track (preserving prior whole-screen behavior)."""
    ctl = _controller(qapp)
    assert ctl._translation_zones == []
    track = _make_track(1, Rect(10, 10, 50, 30))
    assert ctl._track_zone_changed(track) is True


def test_zone_signature_first_frame_marks_all_as_changed(qapp: QApplication) -> None:
    """First call after zones are defined treats every zone as changed
    so initial OCR populates tracks."""
    ctl = _controller(qapp)
    ctl._translation_zones = [
        TranslationZone(rect=Rect(0, 0, 100, 100)),
        TranslationZone(rect=Rect(150, 0, 100, 100)),
    ]
    img = _solid((30, 30, 30))
    ctl._update_per_zone_change_state(img, frame_index=1)
    assert ctl._zones_changed_this_frame == {0, 1}


def test_only_changed_zone_reports_changed(qapp: QApplication) -> None:
    """When pixels change in zone A only, zone B's track must NOT be marked
    as changed."""
    ctl = _controller(qapp)
    ctl._translation_zones = [
        TranslationZone(rect=Rect(0, 0, 100, 100)),  # zone A
        TranslationZone(rect=Rect(150, 0, 100, 100)),  # zone B
    ]
    # Frame 1: baseline.
    base = _solid((30, 30, 30))
    ctl._update_per_zone_change_state(base, frame_index=1)
    # Frame 2: modify only zone A's pixels (top-left region).
    modified = base.copy()
    for x in range(0, 100):
        for y in range(0, 100):
            modified.putpixel((x, y), (200, 200, 200))
    ctl._update_per_zone_change_state(modified, frame_index=2)
    # Exactly zone A should be flagged.
    assert 0 in ctl._zones_changed_this_frame
    assert 1 not in ctl._zones_changed_this_frame
    # And _track_zone_changed reflects that for tracks inside each zone.
    track_in_a = _make_track(1, Rect(20, 20, 30, 20))
    track_in_b = _make_track(2, Rect(170, 20, 30, 20))
    assert ctl._track_zone_changed(track_in_a) is True
    assert ctl._track_zone_changed(track_in_b) is False


def test_track_outside_all_zones_is_treated_as_changed(qapp: QApplication) -> None:
    """A track whose rect lies outside every defined zone keeps the
    pre-zone behavior: never gets suppressed by the gate."""
    ctl = _controller(qapp)
    ctl._translation_zones = [TranslationZone(rect=Rect(0, 0, 100, 100))]
    img = _solid((30, 30, 30))
    ctl._update_per_zone_change_state(img, frame_index=1)
    # Now reset _zones_changed_this_frame so only "outside" semantics matter.
    ctl._zones_changed_this_frame = set()
    track_outside = _make_track(7, Rect(200, 200, 50, 30))
    assert ctl._track_zone_changed(track_outside) is True


def test_clearing_zones_clears_per_zone_state(qapp: QApplication) -> None:
    """set_translation_zones([]) must wipe the per-zone signature cache so
    a later zone reconfiguration starts from scratch."""
    ctl = _controller(qapp)
    ctl._translation_zones = [TranslationZone(rect=Rect(0, 0, 100, 100))]
    img = _solid((30, 30, 30))
    ctl._update_per_zone_change_state(img, frame_index=1)
    assert ctl._per_zone_pixel_signatures  # non-empty
    ctl.clear_translation_region()
    assert ctl._per_zone_pixel_signatures == {}
    assert ctl._zones_changed_this_frame == set()
