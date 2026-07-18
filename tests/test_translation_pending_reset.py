"""Audit findings: translation_pending must reset on every drop path.

Bug #1 (DROP_STALE leak): when a translation result was dropped because
``track.text != source_text`` or ``track.source_version != source_version``
— common with 1-char OCR jitter on long lines — the handler returned
without resetting ``track.translation_pending``. The next
``_maybe_request_translation`` call then early-returned on the
``translation_pending`` guard, blocking every subsequent translation
attempt forever until a hard text change naturally bumped
``source_version`` (which also clears ``translation_pending``).

Bug #2 (7-tuple consistency): the success-path ``_translationReady.emit``
omitted ``task.generation``, so generation-based stale-drop only worked
on the error path. A late-arriving successful translation from a
previous pipeline generation got applied to the new session's tracks.

These tests pin both fixes.
"""

from __future__ import annotations

import sys

import pytest
from PIL import Image
from PyQt6.QtWidgets import QApplication

from app.controller import Controller
from app.types import Rect, Track, TranslationResult, VisualStyle


def _payload(*, frame_index, track_id, source_version, source_text, cache_key, translated, generation):
    return TranslationResult(
        frame_index=frame_index,
        track_id=track_id,
        source_version=source_version,
        source_text=source_text,
        cache_key=cache_key,
        translated=translated,
        generation=generation,
    )


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
    source_version: int = 0,
    translation_pending: bool = True,
) -> Track:
    return Track(
        track_id=track_id,
        text=text,
        normalized=text,
        rect=Rect(0, 0, 100, 30),
        word_rects=[],
        style=VisualStyle((0, 0, 0), (0, 0, 0), (0, 0, 0)),
        stable_frames=2,
        line_count=1,
        source_version=source_version,
        translation_pending=translation_pending,
    )


def _ctl(qapp: QApplication) -> Controller:
    ctl = Controller(lang_tag=None)
    ctl._latest_image = Image.new("RGB", (200, 100), (40, 40, 40))
    return ctl


# --- Bug #1: translation_pending leak on DROP_STALE ------------------------


def test_drop_stale_text_drift_resets_translation_pending(qapp: QApplication) -> None:
    """OCR jitter between queue-time and result-arrival changes
    track.text. The result handler must drop the stale translation AND
    reset translation_pending so subsequent requests can queue again."""
    ctl = _ctl(qapp)
    track = _track(text="hello world")
    ctl._tracks[track.track_id] = track
    # Track.text drifted between queue-time and result arrival.
    track.text = "hella world"  # 1-char OCR jitter
    track.normalized = "hella world"
    payload = _payload(
        frame_index=1,
        track_id=track.track_id,
        source_version=track.source_version,
        source_text="hello world",
        cache_key="hello world",
        translated="TRANSLATED",
        generation=ctl._pipeline_generation,
    )
    ctl._handle_translation_result(payload)
    # Translation was dropped (text drifted) but pending must clear so a
    # fresh request can queue against the new text.
    assert track.translation_pending is False
    assert track.translation == ""  # not applied — text mismatched


def test_drop_stale_source_version_resets_translation_pending(qapp: QApplication) -> None:
    """Same fix on the source_version-mismatch branch."""
    ctl = _ctl(qapp)
    track = _track(text="hello", source_version=2)
    ctl._tracks[track.track_id] = track
    payload = _payload(
        frame_index=1,
        track_id=track.track_id,
        source_version=1,  # task source_version
        source_text=track.text,
        cache_key=track.normalized,
        translated="TRANSLATED",
        generation=ctl._pipeline_generation,
    )
    ctl._handle_translation_result(payload)
    assert track.translation_pending is False
    assert track.translation == ""


def test_successful_translation_applies_and_clears_pending(qapp: QApplication) -> None:
    """The happy path: text matched -> translation applied and
    pending cleared."""
    ctl = _ctl(qapp)
    track = _track(text="hello")
    ctl._tracks[track.track_id] = track
    payload = _payload(
        frame_index=1,
        track_id=track.track_id,
        source_version=track.source_version,
        source_text=track.text,
        cache_key=track.normalized,
        translated="GREETINGS",
        generation=ctl._pipeline_generation,
    )
    ctl._handle_translation_result(payload)
    assert track.translation_pending is False
    assert track.translation == "GREETINGS"


# --- Bug #2: 7-tuple shape is uniform across all emit sites ----------------


def test_handler_accepts_translation_result_dataclass(qapp: QApplication) -> None:
    """The handler consumes a TranslationResult dataclass. The producer
    emits the same shape from _emit_translation_result, so the bug class
    fixed in 8ade7ab (positional tuple with a field omitted on the error
    path) is now structurally impossible."""
    ctl = _ctl(qapp)
    track = _track(text="hello")
    ctl._tracks[track.track_id] = track
    payload = _payload(
        frame_index=1,
        track_id=track.track_id,
        source_version=track.source_version,
        source_text=track.text,
        cache_key=track.normalized,
        translated="TRANSLATED",
        generation=ctl._pipeline_generation,
    )
    # Should not raise.
    ctl._handle_translation_result(payload)


def test_stale_generation_drops_even_with_matching_text(qapp: QApplication) -> None:
    """A late-arriving translation from a previous pipeline generation
    must be dropped even when the track's text happens to match."""
    ctl = _ctl(qapp)
    track = _track(text="hello")
    ctl._tracks[track.track_id] = track
    stale_generation = ctl._pipeline_generation - 1  # before current
    payload = _payload(
        frame_index=1,
        track_id=track.track_id,
        source_version=track.source_version,
        source_text=track.text,
        cache_key=track.normalized,
        translated="STALE_TRANSLATION_FROM_OLD_GEN",
        generation=stale_generation,
    )
    ctl._handle_translation_result(payload)
    # Generation mismatch -> dropped before any state mutation.
    assert track.translation == ""
    # And pending stays unchanged (the result is from an old generation
    # we no longer care about; the new generation's request manages
    # its own pending flag).
    assert track.translation_pending is True
