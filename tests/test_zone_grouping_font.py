"""Regression for zone-grouping font sizing.

When a translation zone has `group_all=True`, multiple source observations
get merged into a single grouped Observation that the renderer then sizes.
The user reported: even though each source row has its own font height,
the rendered output picked an inconsistent size across rows.

Root cause: ``_merge_zone_observations`` was not propagating
``median_char_height`` to the merged Observation, so the renderer fell
through to ``_median_word_height`` which medians across ALL word rects
from all rows — and with multiple rows of different heights, that lands
between the rows instead of on any single one.

Fix: ``_merge_zone_observations`` now sets ``median_char_height`` from
the median of each member's own single-row height, so the merged
observation reports a single representative row height that every
rendered row in the grouped zone shares.
"""

from __future__ import annotations

import sys

import pytest
from PIL import Image
from PyQt6.QtWidgets import QApplication

from app.controller import Controller
from app.types import Observation, Rect, TranslationZone, VisualStyle


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _obs(rect: Rect, *, line_count: int = 1, median_char_height: int = 0, text: str = "x") -> Observation:
    return Observation(
        text=text,
        normalized=text,
        rect=rect,
        word_rects=[rect],
        style=VisualStyle((0, 0, 0), (0, 0, 0), (0, 0, 0)),
        line_count=line_count,
        median_char_height=median_char_height,
    )


def test_merge_zone_observations_sets_median_char_height(qapp: QApplication) -> None:
    """Three single-row members of similar heights should produce a merged
    observation whose median_char_height equals their median, NOT a value
    derived from the merged rect / merged line_count."""
    ctl = Controller(lang_tag=None)
    img = Image.new("RGB", (400, 400), (0, 0, 0))
    zone = TranslationZone(rect=Rect(0, 0, 400, 300))
    members = [
        _obs(Rect(0, 0, 200, 24), median_char_height=20),
        _obs(Rect(0, 30, 200, 26), median_char_height=22),
        _obs(Rect(0, 60, 200, 24), median_char_height=21),
    ]
    merged = ctl._merge_zone_observations(frame_index=1, zone=zone, members=members, image=img)
    assert merged is not None
    # Median of [20, 22, 21] sorted = [20, 21, 22] -> 21.
    assert merged.median_char_height == 21


def test_merge_zone_observations_falls_back_to_rect_height(qapp: QApplication) -> None:
    """When members don't carry their own median_char_height (= 0), the
    merger derives single-row height from ``rect.height / line_count``."""
    ctl = Controller(lang_tag=None)
    img = Image.new("RGB", (400, 400), (0, 0, 0))
    zone = TranslationZone(rect=Rect(0, 0, 400, 300))
    members = [
        _obs(Rect(0, 0, 200, 60), line_count=2, median_char_height=0),  # 60/2 = 30
        _obs(Rect(0, 70, 200, 30), line_count=1, median_char_height=0),  # 30/1 = 30
    ]
    merged = ctl._merge_zone_observations(frame_index=1, zone=zone, members=members, image=img)
    assert merged is not None
    # Both inferred row heights are 30; median is 30.
    assert merged.median_char_height == 30


def test_merge_zone_observations_outlier_row_does_not_skew(qapp: QApplication) -> None:
    """One outlier row with a much larger height should NOT pull the merged
    size away from the bulk-of-the-rows height. Median-of-members is the
    right summary; mean or max would skew."""
    ctl = Controller(lang_tag=None)
    img = Image.new("RGB", (400, 400), (0, 0, 0))
    zone = TranslationZone(rect=Rect(0, 0, 400, 300))
    members = [
        _obs(Rect(0, 0, 200, 24), median_char_height=20),
        _obs(Rect(0, 30, 200, 26), median_char_height=22),
        # outlier: a larger heading row that happens to fall in the zone.
        _obs(Rect(0, 80, 200, 80), median_char_height=64),
    ]
    merged = ctl._merge_zone_observations(frame_index=1, zone=zone, members=members, image=img)
    assert merged is not None
    # Median of sorted [20, 22, 64] = 22. (Mean would be 35.3; max would be 64.)
    assert merged.median_char_height == 22


def test_merge_zone_observations_empty_returns_none(qapp: QApplication) -> None:
    ctl = Controller(lang_tag=None)
    img = Image.new("RGB", (400, 400), (0, 0, 0))
    zone = TranslationZone(rect=Rect(0, 0, 400, 300))
    assert ctl._merge_zone_observations(frame_index=1, zone=zone, members=[], image=img) is None
