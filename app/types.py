"""Pure data classes for the overlay translator pipeline.

Extracted from app.controller to keep that file focused on the Controller
class and main loop. These types are imported by controller.py and re-exported
from there for back-compat with any in-tree code that still imports them
from app.controller.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image

from app.one_ocr import StructuredOcrResult


@dataclass(slots=True, frozen=True)
class Rect:
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


@dataclass(slots=True)
class VisualStyle:
    fill_color: tuple[int, int, int]
    outline_color: tuple[int, int, int]
    background_color: tuple[int, int, int]


@dataclass(slots=True)
class LineRecord:
    raw: str
    normalized: str
    rect: Rect
    word_rects: list[Rect]
    avg_confidence: float
    line_id: str
    word_ids: list[str]


@dataclass(slots=True)
class Observation:
    text: str
    normalized: str
    rect: Rect
    word_rects: list[Rect]
    style: VisualStyle
    frame_index: int = 0
    obs_index: int = 0
    obs_id: str = ""
    source_line_ids: list[str] = field(default_factory=list)
    source_word_ids: list[str] = field(default_factory=list)
    line_count: int = 1
    hud_hint: bool = False
    low_value_hint: bool = False
    avg_confidence: float = 1.0
    dialogue_hint: bool = False
    ui_hint: bool = False
    name_hint: bool = False
    repeated_label_hint: bool = False
    repeat_cluster_id: str = ""
    repeated_consensus_text: str = ""
    repaired_from_cluster: bool = False
    render_anchor_left: int = 0
    estimated_full_width: int = 0
    median_char_height: int = 0
    writing_mode: str = "horizontal"
    # For grouped-zone merged observations: the individual per-line source
    # rects that got unioned into ``rect``. Empty for non-merged observations.
    # Used by the renderer to draw per-line patches (tight to each source
    # line) instead of one big union patch that also covers the gaps between
    # source lines.
    member_source_rects: list[Rect] = field(default_factory=list)


@dataclass(slots=True)
class Track:
    track_id: int
    text: str
    normalized: str
    rect: Rect
    word_rects: list[Rect]
    style: VisualStyle
    dialogue_generation: int = 0
    stable_frames: int = 1
    unchanged_frames: int = 0
    missing_frames: int = 0
    is_static: bool = False
    source_version: int = 0
    translation: str = ""
    translation_pending: bool = False
    line_count: int = 1
    hud_hint: bool = False
    low_value_hint: bool = False
    held_frames: int = 0
    change_frames: int = 0
    region_signature: bytes = b""
    render_enabled: bool = False
    render_z_index: int = 0
    render_suppressed_until_match: bool = False
    last_matched_frame: int = 0
    dialogue_hint: bool = False
    ui_hint: bool = False
    name_hint: bool = False
    repeated_label_hint: bool = False
    repeat_cluster_id: str = ""
    repeated_consensus_text: str = ""
    repaired_from_cluster: bool = False
    render_anchor_left: int = 0
    estimated_full_width: int = 0
    median_char_height: int = 0
    writing_mode: str = "horizontal"
    liveness_change_frames: int = 0
    liveness_hold_frames: int = 0
    unstable_frames: int = 0
    text_growth_streak: int = 0
    created_frame_index: int = 0
    created_from_obs_id: str = ""
    last_obs_id: str = ""
    last_obs_frame_index: int = 0
    # Mirror of Observation.member_source_rects for grouped-zone tracks.
    member_source_rects: list[Rect] = field(default_factory=list)


@dataclass(slots=True)
class FramePacket:
    frame_index: int
    image: Image.Image
    generation: int = 0


@dataclass(slots=True)
class OcrFrame:
    frame_index: int
    image: Image.Image
    layout: StructuredOcrResult
    generation: int = 0


@dataclass(slots=True)
class TranslationTask:
    frame_index: int
    track_id: int
    source_version: int
    source_text: str
    cache_key: str = ""
    priority: int = 50
    queued_at: float = 0.0
    generation: int = 0
    # Track classification carried into the translator so the LLM prompt
    # knows this is a UI/menu label vs. dialogue. Fixes cases like
    # クイックロード → "Quick Road" (correct standalone reading) instead of
    # "Quick Load" (correct game-menu reading) once row-merging is turned
    # down and each menu item is translated in isolation.
    ui_hint: bool = False
    name_hint: bool = False
    dialogue_hint: bool = False


@dataclass(slots=True, frozen=True)
class TranslationResult:
    """Payload emitted by ``Controller._translationReady`` and consumed by
    ``_handle_translation_result``. Previously a positional 7-tuple; a
    field omission on the error path silently dropped ``generation`` and
    bypassed the stale-generation drop (fixed in 8ade7ab). Making it a
    typed dataclass means the producer / consumer contract is checked by
    mypy and any future field addition fails fast at the type level
    instead of silently lopping off a value.
    """

    frame_index: int
    track_id: int
    source_version: int
    source_text: str
    cache_key: str
    translated: str
    generation: int


@dataclass(slots=True)
class RegionMemo:
    source_text: str
    normalized: str
    translation: str
    line_count: int
    dialogue_hint: bool
    ui_hint: bool
    style: VisualStyle
    frame_index: int


@dataclass(slots=True)
class TranslationZone:
    rect: Rect
    group_all: bool = False


@dataclass(slots=True)
class RenderedItem:
    patch_rect: Rect
    patch_image: Image.Image
    text_rect: Rect
    text: str
    fill_color: tuple[int, int, int]
    outline_color: tuple[int, int, int]
    allow_wrap: bool = True
    preferred_pixel_size: int = 0
    alignment: str = "top_left"
    font_family: str = ""
    font_weight: int = 0
    writing_mode: str = "horizontal"
    render_id: str = ""
    source_track_id: int = 0
    source_obs_id: str = ""
    source_frame_index: int = 0
    z_index: int = 0
    # Additional (rect, patch_image) pairs painted alongside ``patch_rect``.
    # Grouped-zone rendering uses this to draw one tight patch per source
    # line so gaps between lines aren't blanketed by a single union patch.
    extra_patches: list[tuple[Rect, Image.Image]] = field(default_factory=list)


@dataclass(slots=True)
class OverlayScene:
    size: tuple[int, int]
    items: list[RenderedItem] = field(default_factory=list)
    allowed_regions: list[Rect] = field(default_factory=list)
    ignore_regions: list[Rect] = field(default_factory=list)


__all__ = [
    "FramePacket",
    "LineRecord",
    "Observation",
    "OcrFrame",
    "OverlayScene",
    "Rect",
    "RegionMemo",
    "RenderedItem",
    "Track",
    "TranslationTask",
    "TranslationZone",
    "VisualStyle",
]
