"""Regression tests for the text-stability trigger (commit e63539b).

The fix:
- tightened the sameish iou floor 0.55 → 0.70 so sliding text doesn't
  accumulate stable_frames;
- added a Track.unstable_frames counter incremented on every non-sameish
  observation;
- replaced the literal-1 stable_required in dialogue/sign/name fast paths
  with TRACK_STABLE_FRAMES_DIALOGUE / _SIGN / _NAME (default 2);
- added TEXT_UNSTABLE_HOLD_FRAMES so the stale translation can be cleared
  before the new one arrives.

The decision logic is now a pure helper (_compute_stability_update) so we
can drive it directly without a Controller.
"""

from __future__ import annotations

from app import config
from app.controller import (
    SAMEISH_IOU_MIN,
    SAMEISH_TEXT_RATIO_MIN,
    StabilityDecision,
    _compute_stability_update,
)


def _decision(
    *,
    text_ratio: float = 1.0,
    iou: float = 1.0,
    class_changed: bool = False,
    track_normalized: str = "hi",
    obs_normalized: str = "hi",
    track_stable_frames: int = 1,
    track_unstable_frames: int = 0,
    zone_changed: bool = True,
) -> StabilityDecision:
    return _compute_stability_update(
        text_ratio=text_ratio,
        iou=iou,
        class_changed=class_changed,
        track_normalized=track_normalized,
        obs_normalized=obs_normalized,
        track_stable_frames=track_stable_frames,
        track_unstable_frames=track_unstable_frames,
        zone_changed=zone_changed,
    )


# --- sameish path -----------------------------------------------------------


def test_sameish_increments_stable_and_clears_unstable() -> None:
    d = _decision(text_ratio=1.0, iou=1.0, track_stable_frames=3, track_unstable_frames=2)
    assert d.stable_frames == 4
    assert d.unstable_frames == 0
    assert d.bump_source_version is False
    assert d.clear_translation is False
    assert d.suppressed_by_zone_jitter is False


def test_high_iou_below_floor_is_not_sameish() -> None:
    """The fix raised the iou floor from 0.55 to 0.70. iou=0.6 should NOT
    count as stable, even if text is identical."""
    d = _decision(text_ratio=1.0, iou=0.6, track_stable_frames=5, track_unstable_frames=0)
    # text_ratio 1.0 >= 0.99 so stable_frames is preserved (not reset to 1).
    assert d.stable_frames == 5
    # But unstable counter must tick.
    assert d.unstable_frames == 1


def test_iou_exactly_at_floor_counts_as_sameish() -> None:
    d = _decision(
        text_ratio=SAMEISH_TEXT_RATIO_MIN,
        iou=SAMEISH_IOU_MIN,
        track_stable_frames=2,
        track_unstable_frames=3,
    )
    assert d.stable_frames == 3
    assert d.unstable_frames == 0


# --- unstable path ---------------------------------------------------------


def test_text_changes_with_zone_change_bumps_source_version() -> None:
    d = _decision(
        text_ratio=0.5,
        iou=0.4,
        class_changed=False,
        track_normalized="hello",
        obs_normalized="goodbye",
        zone_changed=True,
    )
    assert d.bump_source_version is True
    assert d.clear_translation is True
    assert d.reset_unchanged_frames is True
    assert d.suppressed_by_zone_jitter is False


def test_text_changes_without_zone_change_is_suppressed() -> None:
    """OCR jitter in a zone whose pixels didn't move shouldn't requeue
    translation. This is the per-zone gate from commit 7619c6f."""
    d = _decision(text_ratio=0.5, iou=0.4, track_normalized="hello", obs_normalized="goodbye", zone_changed=False)
    assert d.bump_source_version is False
    assert d.clear_translation is False
    assert d.suppressed_by_zone_jitter is True


def test_changed_but_same_normalized_skips_source_version() -> None:
    """Geometry shift on identical text should not requeue."""
    d = _decision(text_ratio=0.5, iou=0.4, track_normalized="same", obs_normalized="same", zone_changed=True)
    assert d.bump_source_version is False
    assert d.clear_translation is False


# --- held path (not sameish, not changed) ----------------------------------


def test_held_preserves_stable_counter() -> None:
    """text_ratio in [0.95, 0.985] AND iou in [0.30, 0.70) = OCR jitter.

    Previously this branch reset stable_frames to 1, so a single 1-char
    OCR flip on a long string (ratio ≈ 0.96) wiped out previous stability
    progress and the track never reached stable_required. Real repro from
    debug/logs: track text alternating between '今日しか' and '今同しか'
    accumulated 169 QUEUE_SKIP not_stable events on one run. Held is OCR
    jitter, not a real text change — preserve the counter so the next
    sameish frame can push stable_frames past the threshold.
    """
    d = _decision(
        text_ratio=0.96,
        iou=0.50,
        track_normalized="hi",
        obs_normalized="hi",
        track_stable_frames=4,
        track_unstable_frames=1,
    )
    # Held no longer wipes the counter.
    assert d.stable_frames == 4
    # Unstable counter still ticks so TEXT_UNSTABLE_HOLD_FRAMES can fire.
    assert d.unstable_frames == 2
    assert d.bump_source_version is False
    assert d.reset_unchanged_frames is False


def test_changed_with_low_text_ratio_does_reset_stable_counter() -> None:
    """A true text change (ratio < 0.95) must still reset stable_frames
    to 1 so we don't keep stale stability against new content."""
    d = _decision(
        text_ratio=0.40,
        iou=0.20,
        track_normalized="hello",
        obs_normalized="goodbye",
        track_stable_frames=5,
        track_unstable_frames=0,
    )
    assert d.stable_frames == 1
    assert d.bump_source_version is True


# --- accumulation over multiple frames -------------------------------------


def test_unstable_frames_accumulate_until_translation_hold_threshold() -> None:
    """After TEXT_UNSTABLE_HOLD_FRAMES (default 4) consecutive unstable
    updates, the controller layer clears the prior translation. Test that
    the unstable counter ramps as expected; the actual clear is on the
    controller side."""
    unstable = 0
    for _ in range(int(config.TEXT_UNSTABLE_HOLD_FRAMES) + 1):
        d = _decision(text_ratio=0.6, iou=0.5, track_normalized="a", obs_normalized="a", track_unstable_frames=unstable)
        unstable = d.unstable_frames
    assert unstable >= int(config.TEXT_UNSTABLE_HOLD_FRAMES)


# --- config wired through ---------------------------------------------------


def test_config_thresholds_have_expected_values() -> None:
    """Pin the values referenced by _maybe_request_translation's fast paths.
    Previously these were literal `1` in the source; commit e63539b made
    them config-driven (default 2)."""
    assert int(config.TRACK_STABLE_FRAMES_DIALOGUE) >= 2
    assert int(config.TRACK_STABLE_FRAMES_SIGN) >= 2
    assert int(config.TRACK_STABLE_FRAMES_NAME) >= 2
    assert int(config.TEXT_UNSTABLE_HOLD_FRAMES) >= 1
