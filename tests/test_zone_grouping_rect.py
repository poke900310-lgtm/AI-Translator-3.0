"""Merged-zone observation rect must wrap the source text, not the whole zone.

Bug repro from debug/logs/render.log (run 20260601-051803-478): a full-width
dialogue zone produced merged Observations with rect=zone.rect (800x150) even
when the actual OCR'd source text only occupied a sub-region within. The
renderer then painted a 791x149 translation patch across the entire strip,
covering neighbouring scene content the source text never touched.

Fix: ``_merge_zone_observations`` now sets the merged Observation's rect to
the union of member rects, intersected with the zone rect so a drifted OCR
rect can't extend past the user's configured area.
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


def _obs(rect: Rect, text: str = "x") -> Observation:
    return Observation(
        text=text,
        normalized=text,
        rect=rect,
        word_rects=[rect],
        style=VisualStyle((0, 0, 0), (0, 0, 0), (0, 0, 0)),
        line_count=1,
        median_char_height=20,
    )


def test_merged_rect_wraps_text_not_zone(qapp: QApplication) -> None:
    """Source text occupies a narrow centred sub-region of a wide zone.
    The merged rect must hug the text, NOT spread to the zone width."""
    ctl = Controller(lang_tag=None)
    img = Image.new("RGB", (800, 600), (0, 0, 0))
    zone = TranslationZone(rect=Rect(0, 423, 800, 150))  # full-width bottom strip
    members = [
        _obs(Rect(200, 430, 400, 30)),
        _obs(Rect(200, 465, 400, 30)),
        _obs(Rect(200, 500, 400, 30)),
    ]
    merged = ctl._merge_zone_observations(frame_index=1, zone=zone, members=members, image=img)
    assert merged is not None
    # Union of member rects: left=200, top=430, right=600, bottom=530.
    assert merged.rect.left == 200
    assert merged.rect.top == 430
    assert merged.rect.width == 400
    assert merged.rect.height == 100


def test_merged_rect_clips_to_zone(qapp: QApplication) -> None:
    """If a member rect extends past the zone (drifted OCR), the merged
    rect is clipped to the zone so the paint patch can't escape the
    user's configured area."""
    ctl = Controller(lang_tag=None)
    img = Image.new("RGB", (800, 600), (0, 0, 0))
    zone = TranslationZone(rect=Rect(100, 100, 400, 200))  # 100,100 → 500,300
    members = [
        _obs(Rect(120, 110, 300, 30)),
        _obs(Rect(120, 280, 600, 30)),  # extends to right=720, well past zone right=500
    ]
    merged = ctl._merge_zone_observations(frame_index=1, zone=zone, members=members, image=img)
    assert merged is not None
    # Union right would be 720; clipped to zone right of 500.
    assert merged.rect.left == 120
    assert merged.rect.left + merged.rect.width <= 500
    # And top/bottom: union top=110, bottom=310 → clipped to zone bottom 300.
    assert merged.rect.top == 110
    assert merged.rect.top + merged.rect.height <= 300


def test_merged_rect_single_member_equals_member(qapp: QApplication) -> None:
    """One member's rect IS the union, so the merged rect equals it."""
    ctl = Controller(lang_tag=None)
    img = Image.new("RGB", (800, 600), (0, 0, 0))
    zone = TranslationZone(rect=Rect(0, 0, 800, 600))
    member_rect = Rect(150, 200, 300, 40)
    merged = ctl._merge_zone_observations(frame_index=1, zone=zone, members=[_obs(member_rect)], image=img)
    assert merged is not None
    assert merged.rect == member_rect
