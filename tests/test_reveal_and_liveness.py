"""Regression tests for the render-flicker + reveal-hold fixes.

Two distinct problems addressed:

1. A small pulsing effect (cursor blink, sparkle, status dot) inside the
   track rect was triggering ``_fast_render_liveness`` to hide the overlay
   every 2 frames at delta>=8.0. Fix: separate the liveness path's
   threshold (LIVENESS_PIXEL_CHANGE_THRESHOLD) from the OCR retrigger
   threshold (PIXEL_CHANGE_THRESHOLD), and ignore the wide-context band of
   the signature when scoring for liveness.

2. Type-on dialogue reveal animations let stable_frames tick up between
   character additions, so the translator queued mid-reveal text. Fix: a
   text-growth detector + REVEAL_HOLD_STREAK_FRAMES gate that holds
   translation while OCR keeps reporting prefix-of-prior + longer text.
"""

from __future__ import annotations

import sys

import pytest
from PIL import Image
from PyQt6.QtWidgets import QApplication

from app import config
from app.controller import Controller
from app.types import Rect, Track, VisualStyle


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _make_track(track_id: int, rect: Rect, *, text: str = "x", normalized: str = "x") -> Track:
    return Track(
        track_id=track_id,
        text=text,
        normalized=normalized,
        rect=rect,
        word_rects=[],
        style=VisualStyle((0, 0, 0), (0, 0, 0), (0, 0, 0)),
    )


# --- liveness threshold separation ----------------------------------------


def test_liveness_fraction_threshold_is_stricter_than_ocr() -> None:
    """The render-hide path must require a larger fraction of cells to
    change than the OCR retrigger path, so cursor blinks etc. don't
    flicker the overlay."""
    assert float(config.LIVENESS_CHANGED_FRACTION_THRESHOLD) > float(config.PIXEL_CHANGE_FRACTION_THRESHOLD)


def test_liveness_cell_delta_min_is_stricter() -> None:
    """A higher per-cell delta floor for liveness so noise cells don't
    count even if many of them exist."""
    assert int(config.LIVENESS_CELL_DELTA_MIN) >= int(config.PIXEL_CHANGE_CELL_DELTA_MIN)


def test_liveness_delete_after_is_more_patient() -> None:
    assert int(config.LIVENESS_PIXEL_CHANGE_DELETE_AFTER_FRAMES) >= int(config.PIXEL_CHANGE_DELETE_AFTER_FRAMES)


def test_small_local_change_does_not_trip_liveness(qapp: QApplication) -> None:
    """A pulsing cursor / small ball changing < 5% of the region's cells
    must NOT trip the liveness gate. This is the regression test for the
    flicker the user reported."""
    ctl = Controller(lang_tag=None)
    # Background frame: solid gray text region.
    img_a = Image.new("RGB", (400, 80), (40, 40, 40))
    # Same region with a small ~20x20 px "pulse" injected. In a 400x80 rect
    # (32000 px) this is ~1.25% of pixels — well below any reasonable
    # fraction threshold.
    img_b = img_a.copy()
    for x in range(360, 380):
        for y in range(30, 50):
            img_b.putpixel((x, y), (240, 240, 240))
    rect = Rect(0, 0, 400, 80)
    track = _make_track(1, rect)
    ctl._latest_image = img_a
    from app.controller import _region_signature

    track.region_signature = _region_signature(img_a, rect, None)
    liv_changed, liv_fraction, _, liv_mean = ctl._region_visually_changed(img_b, track, for_liveness=True)
    assert not liv_changed, (
        f"Small local pulse must not trip liveness gate. "
        f"fraction={liv_fraction:.4f}, mean_delta={liv_mean:.2f}, "
        f"threshold={config.LIVENESS_CHANGED_FRACTION_THRESHOLD}"
    )
    # Fraction-of-cells must report as a small percentage, not a huge mean-delta.
    assert liv_fraction < 0.10, f"small local change shouldn't read as >10% cells changed (got {liv_fraction:.4f})"


def test_large_region_change_does_trip_liveness(qapp: QApplication) -> None:
    """A real scene change (e.g. most of the region's content swapped) must
    trip the liveness gate. This is the negative regression — we don't want
    the gate so permissive that real changes get missed."""
    ctl = Controller(lang_tag=None)
    img_a = Image.new("RGB", (400, 80), (40, 40, 40))
    # Replace the entire region with a different color — a hard scene cut.
    img_b = Image.new("RGB", (400, 80), (200, 60, 100))
    rect = Rect(0, 0, 400, 80)
    track = _make_track(1, rect)
    ctl._latest_image = img_a
    from app.controller import _region_signature

    track.region_signature = _region_signature(img_a, rect, None)
    liv_changed, liv_fraction, _, liv_mean = ctl._region_visually_changed(img_b, track, for_liveness=True)
    assert liv_changed, (
        f"Full-region color swap must trip liveness gate. " f"fraction={liv_fraction:.4f}, mean_delta={liv_mean:.2f}"
    )


# --- reveal-hold gate -----------------------------------------------------


def test_reveal_growth_streak_increments_on_prefix_extension(qapp: QApplication) -> None:
    """When OCR reports a strict prefix-extension of the prior text, the
    growth streak ticks up."""
    ctl = Controller(lang_tag=None)
    rect = Rect(10, 10, 200, 30)
    track = _make_track(1, rect, text="Hello", normalized="hello")
    # Stub _latest_image so _update_track's region_signature call works.
    ctl._latest_image = Image.new("RGB", (300, 100), (0, 0, 0))
    ctl._tracks[track.track_id] = track
    from app.types import Observation

    style = VisualStyle((0, 0, 0), (0, 0, 0), (0, 0, 0))
    obs = Observation(text="Hello wor", normalized="hello wor", rect=rect, word_rects=[], style=style)
    ctl._update_track(frame_index=10, track=track, obs=obs)
    assert track.text_growth_streak == 1
    # Next frame extends further.
    obs2 = Observation(text="Hello world!", normalized="hello world!", rect=rect, word_rects=[], style=style)
    ctl._update_track(frame_index=11, track=track, obs=obs2)
    assert track.text_growth_streak == 2


def test_reveal_growth_streak_resets_on_non_growth(qapp: QApplication) -> None:
    """A frame where text stops growing (same text, or a shrink, or a
    full replacement) resets the streak."""
    ctl = Controller(lang_tag=None)
    rect = Rect(10, 10, 200, 30)
    track = _make_track(1, rect, text="abc", normalized="abc")
    ctl._latest_image = Image.new("RGB", (300, 100), (0, 0, 0))
    ctl._tracks[track.track_id] = track
    from app.types import Observation

    style = VisualStyle((0, 0, 0), (0, 0, 0), (0, 0, 0))
    obs_growth = Observation(text="abcdef", normalized="abcdef", rect=rect, word_rects=[], style=style)
    ctl._update_track(frame_index=1, track=track, obs=obs_growth)
    assert track.text_growth_streak == 1
    # Same text → no growth.
    obs_same = Observation(text="abcdef", normalized="abcdef", rect=rect, word_rects=[], style=style)
    ctl._update_track(frame_index=2, track=track, obs=obs_same)
    assert track.text_growth_streak == 0


def test_reveal_hold_blocks_translation_request(qapp: QApplication, monkeypatch) -> None:
    """When text_growth_streak >= REVEAL_HOLD_STREAK_FRAMES, the request
    path emits QUEUE_HOLD_REVEALING and returns without queuing."""
    ctl = Controller(lang_tag=None)
    rect = Rect(10, 10, 200, 30)
    track = _make_track(1, rect, text="Hello world!", normalized="hello world!")
    track.text_growth_streak = int(config.REVEAL_HOLD_STREAK_FRAMES)
    track.stable_frames = 99  # otherwise stable_required would block first
    ctl._tracks[track.track_id] = track

    captured: list[dict[str, object]] = []
    real_channel = ctl.logger.channel

    def _capture(channel: str, **fields: object) -> None:
        captured.append({"channel": channel, **fields})
        real_channel(channel, **fields)

    monkeypatch.setattr(ctl.logger, "channel", _capture)
    initial_queue_size = ctl._tr_q.qsize()
    ctl._maybe_request_translation(frame_index=100, track=track)
    assert ctl._tr_q.qsize() == initial_queue_size, "reveal hold must not enqueue"
    hold_msgs = [e for e in captured if e.get("message") == "QUEUE_HOLD_REVEALING"]
    assert len(hold_msgs) == 1
    assert hold_msgs[0]["track_id"] == 1


def test_reveal_hold_does_not_block_after_streak_resets(qapp: QApplication, monkeypatch) -> None:
    """Once the reveal stops (streak == 0) the gate releases."""
    ctl = Controller(lang_tag=None)
    rect = Rect(10, 10, 200, 30)
    track = _make_track(1, rect, text="Hello world!", normalized="hello world!")
    track.text_growth_streak = 0
    track.stable_frames = 99
    ctl._tracks[track.track_id] = track

    captured: list[dict[str, object]] = []
    real_channel = ctl.logger.channel

    def _capture(channel: str, **fields: object) -> None:
        captured.append({"channel": channel, **fields})
        real_channel(channel, **fields)

    monkeypatch.setattr(ctl.logger, "channel", _capture)
    ctl._maybe_request_translation(frame_index=100, track=track)
    hold_msgs = [e for e in captured if e.get("message") == "QUEUE_HOLD_REVEALING"]
    assert hold_msgs == []
