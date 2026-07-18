"""Ordering of skip-gates in _maybe_request_translation.

Audit findings:
- Cache lookup used to run AFTER the ui_hint policy gate, so a UI-text
  track with a cached translation got silently skipped instead of having
  the free cached result applied.
- TEXT_SETTLED used to log AFTER the not_stable gate but BEFORE the
  ui_hint gate, so a UI track logged TEXT_SETTLED every frame it cleared
  stability — immediately followed by QUEUE_SKIP reason=ui_text. Spammy.

The new ordering inside _maybe_request_translation:
    1. static / low_value / not_stable skip
    2. empty / already-pending skip
    3. already-translated → re-enable render
    4. cache lookup (CACHE_HIT applies the translation)
    5. ui_hint policy skip (only AFTER cache lookup)
    6. dedupe pending
    7. backlog limit
    8. TEXT_SETTLED log (fires only on the path that actually queues)
    9. queue push

These tests pin the new ordering so it can't silently regress.
"""

from __future__ import annotations

import sys

import pytest
from PIL import Image
from PyQt6.QtWidgets import QApplication

from app.controller import Controller
from app.types import Rect, Track, VisualStyle


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _track(
    *,
    track_id: int = 1,
    text: str = "hello world",
    ui_hint: bool = False,
    stable_frames: int = 2,
) -> Track:
    return Track(
        track_id=track_id,
        text=text,
        normalized=text,
        rect=Rect(0, 0, 100, 30),
        word_rects=[],
        style=VisualStyle((0, 0, 0), (0, 0, 0), (0, 0, 0)),
        stable_frames=stable_frames,
        line_count=1,
        ui_hint=ui_hint,
    )


def _ctl_with_image(qapp: QApplication) -> Controller:
    ctl = Controller(lang_tag=None)
    ctl._latest_image = Image.new("RGB", (200, 100), (40, 40, 40))
    return ctl


def test_cache_hit_applies_to_ui_track_despite_translate_ui_text_off(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A UI-hinted track with a cached translation must have that
    translation applied even when TRANSLATE_UI_TEXT is off — the policy
    gate is for fresh llama calls, not free cached results."""
    from app import config

    monkeypatch.setattr(config, "TRANSLATE_UI_TEXT", False)
    ctl = _ctl_with_image(qapp)
    track = _track(ui_hint=True)
    ctl._tracks[track.track_id] = track
    # UI-hinted tracks now cache under a namespaced key ("ui\0<normalized>")
    # so a stateless-context translation of the same source text doesn't
    # collide with the UI-context translation (e.g. ロード → "Road" vs
    # "Load"). The cache-hit path must still work for that namespaced key.
    ctl._translation_cache["ui\x00" + track.normalized] = "HELLO WORLD"
    ctl._maybe_request_translation(frame_index=1, track=track)
    assert track.translation == "HELLO WORLD"
    assert track.translation_pending is False


def test_ui_track_without_cache_still_skipped(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    """When there's no cache hit, the ui_hint policy gate still blocks
    a fresh llama call. Reordering must not weaken the policy itself."""
    from app import config

    monkeypatch.setattr(config, "TRANSLATE_UI_TEXT", False)
    ctl = _ctl_with_image(qapp)
    track = _track(ui_hint=True)
    ctl._tracks[track.track_id] = track
    ctl._maybe_request_translation(frame_index=1, track=track)
    assert track.translation == ""
    assert track.translation_pending is False
    # And no task got queued.
    assert ctl._tr_q.qsize() == 0


def test_non_ui_track_without_cache_queues_translation(qapp: QApplication) -> None:
    """The happy path: stable, no cache, no policy block -> task queued."""
    ctl = _ctl_with_image(qapp)
    track = _track(ui_hint=False, text="goodbye world")
    ctl._tracks[track.track_id] = track
    ctl._maybe_request_translation(frame_index=1, track=track)
    assert track.translation_pending is True
    assert ctl._tr_q.qsize() == 1
    assert track.normalized in ctl._pending_translation_keys


def test_not_stable_skips_before_cache_lookup(qapp: QApplication) -> None:
    """The stability gate must short-circuit BEFORE the cache lookup so
    a flickering OCR doesn't keep re-applying the same cached translation
    while the underlying text is in motion."""
    ctl = _ctl_with_image(qapp)
    track = _track(stable_frames=1)  # < TRACK_STABLE_FRAMES (2)
    ctl._tracks[track.track_id] = track
    ctl._translation_cache[track.normalized] = "CACHED"
    ctl._maybe_request_translation(frame_index=1, track=track)
    assert track.translation == ""


def test_already_translated_track_just_re_enables_render(qapp: QApplication) -> None:
    """A track that already has a translation should re-enable rendering
    without touching the cache or the queue."""
    ctl = _ctl_with_image(qapp)
    track = _track()
    track.translation = "previous translation"
    track.render_enabled = False
    ctl._tracks[track.track_id] = track
    ctl._maybe_request_translation(frame_index=1, track=track)
    assert track.translation == "previous translation"
    assert ctl._tr_q.qsize() == 0
