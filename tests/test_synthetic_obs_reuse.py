"""OCR-skip-on-identical-CRC: reuse the last observation set instead.

Audit finding: ``_needs_ocr_on_unchanged_capture`` would force a fresh
OCR run on a byte-identical frame just to advance a track's stability
counter. OCR is deterministic on the same masked image, so the fresh
observations are identical to the cached ``_last_observations`` — the
~50 ms OCR call is pure waste.

The fix lets ``_tick`` skip the OCR dispatch and re-apply the cached
observations via ``_update_tracks`` directly, but ONLY when nothing
visibly moved:

  * CRC matches the previous capture (image_changed=False)
  * No dialogue-zone pixel change
  * No fast-liveness change
  * No forced OCR reason (periodic rescan, drought, ui-confirmation, ...)
  * We actually have prior observations to reuse
  * No OCR currently in flight or queued

These tests pin the predicate (``_can_synthesize_observations``) so a
future edit can't silently re-introduce the redundant OCR work, and so
the gate stays restrictive enough that we never reuse a stale set on a
frame that has any actual reason to refresh.
"""

from __future__ import annotations

import sys

import pytest
from PIL import Image
from PyQt6.QtWidgets import QApplication

from app.controller import Controller
from app.types import Observation, Rect, VisualStyle


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _ctl(qapp: QApplication) -> Controller:
    return Controller(lang_tag=None)


def _obs(text: str = "hi", rect: Rect | None = None) -> Observation:
    return Observation(
        text=text,
        normalized=text,
        rect=rect or Rect(10, 10, 80, 24),
        word_rects=[rect or Rect(10, 10, 80, 24)],
        style=VisualStyle((0, 0, 0), (0, 0, 0), (0, 0, 0)),
        line_count=1,
        median_char_height=20,
    )


# --- happy path: every gate is False, reuse is allowed --------------------


def test_synthesize_allowed_when_nothing_changed(qapp: QApplication) -> None:
    ctl = _ctl(qapp)
    ctl._last_observations = [_obs()]
    ctl._ocr_inflight_frame_index = 0
    assert ctl._can_synthesize_observations(
        image_changed=False,
        dialogue_zone_changed=False,
        liveness_changed=False,
        force_reason=None,
    )


# --- each gate independently blocks the reuse path -------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"image_changed": True},
        {"dialogue_zone_changed": True},
        {"liveness_changed": True},
        {"force_reason": "periodic_full_rescan"},
    ],
)
def test_any_change_signal_blocks_synthesis(qapp: QApplication, kwargs: dict[str, object]) -> None:
    """Any single change signal must force a real OCR pass — reusing
    cached observations against a moved screen would silently render
    stale results."""
    ctl = _ctl(qapp)
    ctl._last_observations = [_obs()]
    ctl._ocr_inflight_frame_index = 0
    base = {
        "image_changed": False,
        "dialogue_zone_changed": False,
        "liveness_changed": False,
        "force_reason": None,
    }
    base.update(kwargs)
    assert not ctl._can_synthesize_observations(**base)  # type: ignore[arg-type]


def test_empty_observation_cache_blocks_synthesis(qapp: QApplication) -> None:
    """On the first frame ``_last_observations`` is empty — we have
    nothing to reuse so a real OCR pass is required."""
    ctl = _ctl(qapp)
    ctl._last_observations = []
    ctl._ocr_inflight_frame_index = 0
    assert not ctl._can_synthesize_observations(
        image_changed=False,
        dialogue_zone_changed=False,
        liveness_changed=False,
        force_reason=None,
    )


def test_inflight_ocr_blocks_synthesis(qapp: QApplication) -> None:
    """If an OCR call is in flight, fresh observations are imminent —
    don't apply the stale set just before they arrive (would cause a
    spurious sameish->changed->sameish ripple)."""
    ctl = _ctl(qapp)
    ctl._last_observations = [_obs()]
    ctl._ocr_inflight_frame_index = 42
    assert not ctl._can_synthesize_observations(
        image_changed=False,
        dialogue_zone_changed=False,
        liveness_changed=False,
        force_reason=None,
    )


# --- end-to-end: synthesis actually advances stable_frames -----------------


def test_synthetic_update_advances_stable_frames(qapp: QApplication) -> None:
    """The point of the reuse path is to let stability tick up without an
    OCR call. Push an initial observation, then re-apply the same set via
    ``_update_tracks`` (the same call the synthesis path makes) and check
    that the track's stable_frames advanced."""
    ctl = _ctl(qapp)
    ctl._latest_image = Image.new("RGB", (200, 100), (40, 40, 40))
    obs = _obs(text="hello", rect=Rect(20, 20, 100, 30))
    # First update creates the track at stable_frames=1.
    ctl._update_tracks(1, [obs])
    track = next(iter(ctl._tracks.values()))
    initial_stable = track.stable_frames
    # Second update with the SAME observation = sameish → stable++.
    ctl._update_tracks(2, [obs])
    assert track.stable_frames == initial_stable + 1
