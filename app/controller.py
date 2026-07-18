"""Core controller for the full-window in-place translation workflow.

Workflow:
- attach to one visible top-level window
- capture the full client area every tick
- run the bundled One OCR engine over the whole captured image
- group OCR lines into translatable text blocks with bounding boxes
- track blocks over time to classify static versus dynamic text
- translate only dynamic, stable blocks
- paint translations back over their original positions with a reconstructed
  local background patch and sampled source text colors
"""

from __future__ import annotations

import itertools
import json
import math
import queue
import re
import threading
import zlib
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from PyQt6 import QtCore, QtGui
from rapidfuzz.fuzz import ratio as _fuzz_ratio

from app import config
from app.color import (
    _build_style,
    _crop,
    _delta_e,
    _median_float,
    _median_int,
    _sample_background_color,
    _sample_foreground_color,
    _sample_rect_mean_rgb,
)
from app.geometry import (
    _clamp,
    _expand_rect,
    _horizontal_gap,
    _horizontal_overlap_ratio,
    _is_vertical_rect,
    _move_rect_inside,
    _rect_area,
    _rect_center_distance,
    _rect_contains_point,
    _rect_intersection,
    _rect_intersection_area,
    _rect_iou,
    _rect_vertical_overlap_ratio,
    _same_text_row,
    _union_rect,
    _vertical_overlap_ratio,
)
from app.llama_server import cache_fingerprint, translate_llama_server
from app.logging import Logger
from app.one_ocr import BoundingBox, OneOcr, StructuredOcrResult
from app.qt_render import paint_overlay_items, pil_to_qimage, qimage_to_pil
from app.screen_capture import Region, grab_window_client
from app.text import (
    _CJK_RE,
    _LATIN_RE,
    _cleanup_translation,
    _compact_text,
    _compact_text_len,
    _contains_dialogue_punct,
    _contains_timestamp_like,
    _contains_ui_keyword,
    _contains_vertical_script,
    _join_word_texts,
    _looks_like_short_mixed_junk,
    _looks_like_upper_suffix_junk,
    _mean_confidence_from_words,
    _normalize_ocr_text,
    _text_quality_ok,
)
from app.types import (
    FramePacket,
    LineRecord,
    Observation,
    OcrFrame,
    OverlayScene,
    Rect,
    RegionMemo,
    RenderedItem,
    Track,
    TranslationResult,
    TranslationTask,
    TranslationZone,
    VisualStyle,
)
from app.window_binding import (
    ClientRect,
    detect_window_occlusion,
    format_window_diagnostics,
    get_client_rect_screen,
    get_window_diagnostics,
    is_window_usable,
    set_window_enabled,
)

LANG_HINT = {
    None: None,
    "ja": "Japanese",
    "ko": "Korean",
    "zh-Hans": "Chinese",
    "zh-Hant": "Chinese",
    "en": "English",
}


def _preview_text(text: str, limit: int | None = None) -> str:
    """Truncate long text for log/debug field display.

    Free-function form so module-level helpers (e.g.
    `_merge_dialogue_render_cluster`) can call it without an instance.
    Controller._preview_text is a thin @staticmethod wrapper around this.
    """
    if limit is None or limit <= 0:
        effective_limit = int(getattr(config, "DEBUG_MAX_TEXT_PREVIEW", 400))
    else:
        effective_limit = int(limit)
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    return text if len(text) <= effective_limit else (text[:effective_limit] + "…")


# Track-stability classification thresholds (used by _compute_stability_update).
# Each frame's (track, observation) pair lands in one of three classes:
#   sameish — text-similar AND rect-overlapping enough to count toward stability
#   changed — text-similarity OR rect-overlap dropped enough to be a real change
#   held    — everything in between: light OCR jitter that mustn't wipe progress
# These constants give those thresholds names so the magic floats don't
# drift between the function, its docstring, and the tests that pin them.
SAMEISH_TEXT_RATIO_MIN = 0.985
SAMEISH_IOU_MIN = 0.70
CHANGED_TEXT_RATIO_FLOOR = 0.95
CHANGED_IOU_FLOOR = 0.30
HELD_PRESERVE_TEXT_RATIO = 0.95


@dataclass(slots=True, frozen=True)
class _SceneLayoutCfg:
    """Snapshot of every config value the scene-build hot path reads.

    Built once per ``_build_scene`` call via ``from_config`` and threaded
    through ``_build_render_item_for_track``. Config is immutable at
    runtime in this build, so a single getattr pass replaces ~18
    per-track lookups and keeps the per-track method body free of
    ``getattr(config, ...)`` noise.
    """

    small_text_rescan_max_height: int
    text_margin_x: int
    text_margin_y: int
    text_dialogue_extra_margin_x: int
    text_dialogue_extra_margin_y: int
    text_min_pixel: int
    text_max_pixel: int
    text_source_height_scale: float
    small_sign_text_scale: float
    small_sign_max_extra_px: int
    small_sign_max_pixel: int
    text_sign_font_family: str
    text_sign_font_weight: int
    text_dialogue_font_family: str
    text_dialogue_font_weight: int
    text_name_font_family: str
    text_name_font_weight: int
    text_name_scale: float
    render_vertical_text: bool
    text_vertical_extra_margin_x: int
    text_vertical_extra_margin_y: int
    hud_thin_max_height: int
    thin_label_wrap_ratio: float

    @classmethod
    def from_config(cls) -> "_SceneLayoutCfg":
        sign_family = str(getattr(config, "TEXT_SIGN_FONT_FAMILY", getattr(config, "TEXT_FONT_FAMILY", "Segoe UI")))
        sign_weight = int(getattr(config, "TEXT_SIGN_FONT_WEIGHT", getattr(config, "TEXT_FONT_WEIGHT", 600)))
        return cls(
            small_text_rescan_max_height=int(getattr(config, "SMALL_TEXT_RESCAN_MAX_HEIGHT", 50)),
            text_margin_x=int(config.TEXT_MARGIN_X),
            text_margin_y=int(config.TEXT_MARGIN_Y),
            text_dialogue_extra_margin_x=int(getattr(config, "TEXT_DIALOGUE_EXTRA_MARGIN_X", 14)),
            text_dialogue_extra_margin_y=int(getattr(config, "TEXT_DIALOGUE_EXTRA_MARGIN_Y", 12)),
            text_min_pixel=int(getattr(config, "TEXT_MIN_PIXEL", 10)),
            text_max_pixel=int(getattr(config, "TEXT_MAX_PIXEL", 26)),
            text_source_height_scale=float(getattr(config, "TEXT_SOURCE_HEIGHT_SCALE", 0.9)),
            small_sign_text_scale=float(getattr(config, "SMALL_SIGN_TEXT_SCALE", 0.98)),
            small_sign_max_extra_px=int(getattr(config, "SMALL_SIGN_MAX_EXTRA_PX", 1)),
            small_sign_max_pixel=int(getattr(config, "SMALL_SIGN_MAX_PIXEL", 16)),
            text_sign_font_family=sign_family,
            text_sign_font_weight=sign_weight,
            text_dialogue_font_family=str(getattr(config, "TEXT_DIALOGUE_FONT_FAMILY", sign_family)),
            text_dialogue_font_weight=int(getattr(config, "TEXT_DIALOGUE_FONT_WEIGHT", sign_weight)),
            text_name_font_family=str(getattr(config, "TEXT_NAME_FONT_FAMILY", sign_family)),
            text_name_font_weight=int(getattr(config, "TEXT_NAME_FONT_WEIGHT", sign_weight)),
            text_name_scale=float(getattr(config, "TEXT_NAME_SCALE", 0.92)),
            render_vertical_text=bool(getattr(config, "RENDER_VERTICAL_TEXT", True)),
            text_vertical_extra_margin_x=int(getattr(config, "TEXT_VERTICAL_EXTRA_MARGIN_X", 6)),
            text_vertical_extra_margin_y=int(getattr(config, "TEXT_VERTICAL_EXTRA_MARGIN_Y", 10)),
            hud_thin_max_height=int(getattr(config, "HUD_THIN_MAX_HEIGHT", 24)),
            thin_label_wrap_ratio=float(getattr(config, "THIN_LABEL_WRAP_RATIO", 1.75)),
        )


@dataclass(slots=True, frozen=True)
class StabilityDecision:
    """The pure decision output of the per-frame track stability update.

    Returned by `_compute_stability_update` and applied by `_update_track`.
    Extracting this lets us unit-test the stable/unstable/source-version
    transitions without a live Controller, OCR pipeline, or PIL image.
    """

    stable_frames: int  # new value to assign to track.stable_frames
    unstable_frames: int  # new value to assign to track.unstable_frames
    reset_unchanged_frames: bool  # True → unchanged_frames := 0; else += 1
    bump_source_version: bool  # True → track.source_version += 1
    clear_translation: bool  # True → clear translation + pending + render
    suppressed_by_zone_jitter: bool  # True → emit ZONE_JITTER_SUPPRESSED


def _compute_stability_update(
    *,
    text_ratio: float,
    iou: float,
    class_changed: bool,
    track_normalized: str,
    obs_normalized: str,
    track_stable_frames: int,
    track_unstable_frames: int,
    zone_changed: bool,
) -> StabilityDecision:
    """Compute the next stability state for a track given a new observation.

    Pure function — no Controller state, no PIL access. The thresholds
    encode the text-stability gate that landed in commit e63539b:

    - ``sameish``  = text_ratio ≥ SAMEISH_TEXT_RATIO_MIN AND iou ≥ SAMEISH_IOU_MIN
    - ``changed``  = text_ratio < CHANGED_TEXT_RATIO_FLOOR OR iou < CHANGED_IOU_FLOOR OR class flipped
    - Otherwise: held — light OCR jitter; stable_frames is preserved when
      text_ratio ≥ HELD_PRESERVE_TEXT_RATIO so a 1-char flip on a long line
      doesn't wipe earlier sameish progress.

    Zone gating: if the track's enclosing zone didn't actually move pixels
    this frame, a text change is treated as OCR jitter — the
    source_version bump is suppressed so the overlay doesn't re-translate
    against a noisy read.
    """
    sameish = text_ratio >= SAMEISH_TEXT_RATIO_MIN and iou >= SAMEISH_IOU_MIN
    changed = text_ratio < CHANGED_TEXT_RATIO_FLOOR or iou < CHANGED_IOU_FLOOR or class_changed
    if sameish:
        return StabilityDecision(
            stable_frames=track_stable_frames + 1,
            unstable_frames=0,
            reset_unchanged_frames=False,
            bump_source_version=False,
            clear_translation=False,
            suppressed_by_zone_jitter=False,
        )
    # Held = "not sameish AND not changed" — OCR jitter range. Preserve
    # track_stable_frames when text_ratio is at least HELD_PRESERVE_TEXT_RATIO
    # so a 1-char flip (e.g. 日 ↔ 同) on a long line doesn't wipe progress.
    new_stable = max(1, track_stable_frames if text_ratio >= HELD_PRESERVE_TEXT_RATIO else 1)
    new_unstable = track_unstable_frames + 1
    if not changed:
        return StabilityDecision(
            stable_frames=new_stable,
            unstable_frames=new_unstable,
            reset_unchanged_frames=False,
            bump_source_version=False,
            clear_translation=False,
            suppressed_by_zone_jitter=False,
        )
    if track_normalized == obs_normalized:
        return StabilityDecision(
            stable_frames=new_stable,
            unstable_frames=new_unstable,
            reset_unchanged_frames=True,
            bump_source_version=False,
            clear_translation=False,
            suppressed_by_zone_jitter=False,
        )
    if not zone_changed:
        return StabilityDecision(
            stable_frames=new_stable,
            unstable_frames=new_unstable,
            reset_unchanged_frames=True,
            bump_source_version=False,
            clear_translation=False,
            suppressed_by_zone_jitter=True,
        )
    return StabilityDecision(
        stable_frames=new_stable,
        unstable_frames=new_unstable,
        reset_unchanged_frames=True,
        bump_source_version=True,
        clear_translation=True,
        suppressed_by_zone_jitter=False,
    )


def _dict_int(d: dict[str, object], key: str, default: int = 0) -> int:
    """Narrow a ``dict[str, object]`` field to int.

    Zone JSON payloads and debug-manifest dicts are typed loosely so they can
    carry mixed values, but the individual int fields can't be cast with bare
    ``int(d.get(key, 0))`` because ``int`` has no overload for ``object``.
    This helper isinstance-narrows and falls back on anything weird.
    """
    value = d.get(key, default)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


@dataclass
class AppState:
    target_hwnd: int | None = None
    client_region: Region | None = None
    paused: bool = True


def _same_region(a: Region | None, b: Region | None) -> bool:
    if a is None or b is None:
        return a is b
    return (a.left, a.top, a.width, a.height) == (b.left, b.top, b.width, b.height)


def _bbox_to_rect(bbox: BoundingBox, max_w: int, max_h: int) -> Rect:
    xs = [bbox.top_left.x, bbox.top_right.x, bbox.bottom_right.x, bbox.bottom_left.x]
    ys = [bbox.top_left.y, bbox.top_right.y, bbox.bottom_right.y, bbox.bottom_left.y]
    left = _clamp(int(math.floor(min(xs))), 0, max_w - 1)
    top = _clamp(int(math.floor(min(ys))), 0, max_h - 1)
    right = _clamp(int(math.ceil(max(xs))), left + 1, max_w)
    bottom = _clamp(int(math.ceil(max(ys))), top + 1, max_h)
    return Rect(left, top, max(1, right - left), max(1, bottom - top))


def _bbox_points_dict(bbox: BoundingBox) -> dict[str, float]:
    return {
        "x1": round(float(bbox.top_left.x), 3),
        "y1": round(float(bbox.top_left.y), 3),
        "x2": round(float(bbox.top_right.x), 3),
        "y2": round(float(bbox.top_right.y), 3),
        "x3": round(float(bbox.bottom_right.x), 3),
        "y3": round(float(bbox.bottom_right.y), 3),
        "x4": round(float(bbox.bottom_left.x), 3),
        "y4": round(float(bbox.bottom_left.y), 3),
    }


def _bbox_schema_dict(bbox: BoundingBox, max_w: int, max_h: int) -> dict[str, object]:
    rect = _bbox_to_rect(bbox, max_w, max_h)
    return {
        "rect": {"x": rect.left, "y": rect.top, "width": rect.width, "height": rect.height},
        "quad": _bbox_points_dict(bbox),
    }


def _obs_is_translation_candidate(obs: Observation) -> bool:
    return bool((obs.text or "").strip())


def _rect_touches_edge(rect: Rect, max_w: int, max_h: int, margin: int) -> bool:
    return (
        rect.left <= margin or rect.top <= margin or rect.right >= (max_w - margin) or rect.bottom >= (max_h - margin)
    )


def _line_geometry_ok(norm: str, rect: Rect, max_w: int, max_h: int, avg_confidence: float) -> bool:
    compact = re.sub(r"\s+", "", norm or "")
    area = _rect_area(rect)
    if area < int(getattr(config, "OCR_MIN_LINE_RECT_AREA", 0) or 0):
        return False
    tiny_edge_max_area = int(getattr(config, "OCR_TINY_EDGE_MAX_AREA", 40) or 40)
    if area <= tiny_edge_max_area and len(compact) <= int(getattr(config, "OCR_TINY_EDGE_MAX_CHARS", 4) or 4):
        if _rect_touches_edge(
            rect, max_w, max_h, int(getattr(config, "OCR_TINY_EDGE_MARGIN_PX", 24) or 24)
        ) and avg_confidence <= float(getattr(config, "OCR_TINY_EDGE_MAX_CONFIDENCE", 0.9) or 0.9):
            return False
    return True


def _infer_fragment_writing_mode(fragments: list[tuple[str, str, Rect]]) -> str:
    """Classify a group of OCR fragments as vertical (CJK column) or horizontal text.

    The previous y_span/x_span fallback flagged any tall-narrow stack of
    horizontal lines as vertical text — e.g. a 6-line menu where each line
    is wider than tall, but the union of all six lines is taller than wide.
    That mis-classification made the renderer pass ``allow_wrap=False`` and
    ``alignment="center"`` so the translation rendered as one giant single
    line. True vertical text is composed of NARROW fragments (each glyph or
    short word stacks top-to-bottom with width <= height per fragment), so
    the presence of any width > height fragment is a hard "horizontal" tell.
    """
    if not fragments:
        return "horizontal"
    union = fragments[0][2]
    centers_x: list[float] = []
    centers_y: list[float] = []
    for _, _, rect in fragments:
        union = _union_rect(union, rect)
        centers_x.append(rect.left + (rect.width / 2.0))
        centers_y.append(rect.top + (rect.height / 2.0))
    x_span = (max(centers_x) - min(centers_x)) if centers_x else 0.0
    y_span = (max(centers_y) - min(centers_y)) if centers_y else 0.0
    vertical_bias = float(getattr(config, "VERTICAL_TEXT_SCORE_BIAS", 1.1))
    # Real vertical text never contains a fragment wider than tall. If any
    # fragment is wide (one OCR line of horizontal script), the union is a
    # stack of horizontal lines regardless of how tall+narrow it sums to.
    if any(rect.width > rect.height for _, _, rect in fragments):
        return "horizontal"
    if _is_vertical_rect(union):
        return "vertical"
    if y_span >= max(8.0, x_span * vertical_bias) and union.height > union.width:
        return "vertical"
    return "horizontal"


def _join_fragments_by_row(fragments: list[tuple[str, str, Rect]]) -> tuple[str, str, int]:
    if not fragments:
        return "", "", 0
    writing_mode = _infer_fragment_writing_mode(fragments)
    if writing_mode == "vertical":
        columns: list[dict[str, Any]] = []
        for raw, normalized, rect in sorted(fragments, key=lambda item: (item[2].left, item[2].top), reverse=True):
            placed = False
            center_x = rect.left + (rect.width / 2.0)
            for column in columns:
                col_rect = column["rect"]
                col_center_x = col_rect.left + (col_rect.width / 2.0)
                same_col = _horizontal_overlap_ratio(col_rect, rect) >= 0.22 or abs(col_center_x - center_x) <= max(
                    col_rect.width, rect.width
                )
                if same_col:
                    column["items"].append((raw, normalized, rect))
                    column["rect"] = _union_rect(col_rect, rect)
                    placed = True
                    break
            if not placed:
                columns.append({"rect": rect, "items": [(raw, normalized, rect)]})
        text_cols: list[str] = []
        norm_cols: list[str] = []
        for column in sorted(columns, key=lambda item: item["rect"].left, reverse=True):
            col_items = column["items"]
            assert isinstance(col_items, list)
            items = sorted(col_items, key=lambda item: (item[2].top, item[2].left))
            text_cols.append("".join((raw or "").strip() for raw, _, _ in items if (raw or "").strip()))
            norm_cols.append(
                "".join((normalized or "").strip() for _, normalized, _ in items if (normalized or "").strip())
            )
        text_cols = [col for col in text_cols if col]
        norm_cols = [col for col in norm_cols if col]
        return "\n".join(text_cols), "\n".join(norm_cols), max(len(text_cols), len(norm_cols))
    rows: list[dict[str, Any]] = []
    for raw, normalized, rect in sorted(fragments, key=lambda item: (item[2].top, item[2].left)):
        placed = False
        for row in rows:
            row_rect = row["rect"]
            if _same_text_row(row_rect, rect):
                row["items"].append((raw, normalized, rect))
                row["rect"] = _union_rect(row_rect, rect)
                placed = True
                break
        if not placed:
            rows.append({"rect": rect, "items": [(raw, normalized, rect)]})
    text_rows: list[str] = []
    normalized_rows: list[str] = []
    for row in rows:
        row_items = row["items"]
        assert isinstance(row_items, list)
        items = sorted(row_items, key=lambda item: (item[2].left, item[2].top))
        text_rows.append("".join((raw or "").strip() for raw, _, _ in items if (raw or "").strip()))
        normalized_rows.append(
            "".join((normalized or "").strip() for _, normalized, _ in items if (normalized or "").strip())
        )
    text_rows = [row for row in text_rows if row]
    normalized_rows = [row for row in normalized_rows if row]
    return "\n".join(text_rows), "\n".join(normalized_rows), max(len(text_rows), len(normalized_rows))


def _gap_strip_rect(left_rect: Rect, right_rect: Rect, image: Image.Image) -> Rect | None:
    gap_left = max(0, left_rect.right)
    gap_right = min(image.width, right_rect.left)
    if gap_right <= gap_left:
        return None
    pad_y = max(1, int(max(left_rect.height, right_rect.height) * 0.2))
    top = max(0, min(left_rect.top, right_rect.top) - pad_y)
    bottom = min(image.height, max(left_rect.bottom, right_rect.bottom) + pad_y)
    if bottom <= top:
        return None
    return Rect(gap_left, top, gap_right - gap_left, bottom - top)


def _gap_has_background_seam(
    image: Image.Image, left_rect: Rect, right_rect: Rect, left_bg: tuple[int, int, int], right_bg: tuple[int, int, int]
) -> tuple[bool, float]:
    gap_rect = _gap_strip_rect(left_rect, right_rect, image)
    if gap_rect is None:
        return (False, 0.0)
    gap_rgb = _sample_rect_mean_rgb(image, gap_rect)
    seam_strength = min(_delta_e(gap_rgb, left_bg), _delta_e(gap_rgb, right_bg))
    seam_threshold = float(getattr(config, "OCR_COLOR_GROUP_SEAM_DELTA_E_MAX", 12.0))
    wide_gap_threshold = max(
        6,
        int(max(left_rect.height, right_rect.height) * float(getattr(config, "OCR_COLOR_GROUP_WIDE_GAP_FACTOR", 0.75))),
    )
    has_seam = seam_strength >= seam_threshold or gap_rect.width >= wide_gap_threshold
    return (has_seam, seam_strength)


def _vertical_gap_strip_rect(top_rect: Rect, bottom_rect: Rect, image: Image.Image) -> Rect | None:
    gap_top = max(0, top_rect.bottom)
    gap_bottom = min(image.height, bottom_rect.top)
    if gap_bottom <= gap_top:
        return None
    pad_x = max(1, int(max(top_rect.width, bottom_rect.width) * 0.08))
    left = max(0, min(top_rect.left, bottom_rect.left) - pad_x)
    right = min(image.width, max(top_rect.right, bottom_rect.right) + pad_x)
    if right <= left:
        return None
    return Rect(left, gap_top, right - left, gap_bottom - gap_top)


def _vertical_gap_has_background_seam(
    image: Image.Image, top_rect: Rect, bottom_rect: Rect, top_bg: tuple[int, int, int], bottom_bg: tuple[int, int, int]
) -> tuple[bool, float]:
    gap_rect = _vertical_gap_strip_rect(top_rect, bottom_rect, image)
    if gap_rect is None:
        return (False, 0.0)
    gap_rgb = _sample_rect_mean_rgb(image, gap_rect)
    seam_strength = min(_delta_e(gap_rgb, top_bg), _delta_e(gap_rgb, bottom_bg))
    seam_threshold = float(getattr(config, "REPEAT_LABEL_VERTICAL_SEAM_DELTA_E_MAX", 10.0))
    wide_gap_threshold = max(4, int(max(top_rect.height, bottom_rect.height) * 0.22))
    has_seam = seam_strength >= seam_threshold or gap_rect.height >= wide_gap_threshold
    return (has_seam, seam_strength)


def _repeat_similarity_stats(a: str, b: str) -> tuple[float, int, int]:
    aa = _compact_text(a)
    bb = _compact_text(b)
    if not aa or not bb:
        return (0.0, 0, 0)
    ratio = _fuzz_ratio(aa, bb) / 100.0
    prefix = 0
    while prefix < min(len(aa), len(bb)) and aa[prefix] == bb[prefix]:
        prefix += 1
    suffix = 0
    while suffix < min(len(aa), len(bb)) and aa[-(suffix + 1)] == bb[-(suffix + 1)]:
        suffix += 1
    return (ratio, prefix, suffix)


def _repeat_similarity_pass(a: str, b: str, min_similarity: float) -> bool:
    ratio, prefix, suffix = _repeat_similarity_stats(a, b)
    if ratio >= min_similarity:
        return True
    compact_a = _compact_text(a)
    compact_b = _compact_text(b)
    min_len = min(len(compact_a), len(compact_b))
    if min_len <= 0:
        return False
    edge_min = max(1, int(math.ceil(min_len * float(getattr(config, "REPEAT_LABEL_EDGE_PREFIX_RATIO", 0.6)))))
    return abs(len(compact_a) - len(compact_b)) <= int(getattr(config, "REPEAT_LABEL_MAX_LENGTH_DELTA", 2)) and (
        prefix >= edge_min or suffix >= edge_min
    )


def _line_record_is_compact_label(item: LineRecord, image: Image.Image) -> bool:
    return (
        not _contains_dialogue_punct(item.normalized)
        and not _contains_timestamp_like(item.normalized)
        and not _contains_ui_keyword(item.normalized)
        and _is_compact_scene_label_geometry(item.normalized, item.rect, 1, image)
    )


def _line_record_repeat_support(item: LineRecord, compact_items: list[LineRecord], min_similarity: float) -> int:
    count = 1
    item_compact = _compact_text(item.normalized)
    for other in compact_items:
        if other is item:
            continue
        if not _repeat_similarity_pass(item_compact, _compact_text(other.normalized), min_similarity):
            continue
        same_row = _rect_vertical_overlap_ratio(item.rect, other.rect) >= float(
            getattr(config, "REPEAT_LABEL_ROW_OVERLAP_MIN", 0.52)
        )
        same_col = _horizontal_overlap_ratio(item.rect, other.rect) >= float(
            getattr(config, "REPEAT_LABEL_COL_OVERLAP_MIN", 0.32)
        )
        if same_row or same_col:
            count += 1
    return count


def _split_repeated_label_groups(
    frame_index: int,
    groups: list[list[LineRecord]],
    per_line: list[LineRecord],
    image: Image.Image,
    logger: Logger | None = None,
) -> list[list[LineRecord]]:
    if not groups:
        return groups
    min_similarity = float(getattr(config, "REPEAT_LABEL_MIN_TEXT_SIMILARITY", 0.60))
    compact_items = [item for item in per_line if _line_record_is_compact_label(item, image)]
    if len(compact_items) < int(getattr(config, "REPEAT_LABEL_MIN_OBS", 3)):
        if logger is not None:
            logger.channel(
                "repeat",
                "SPLIT_SKIPPED",
                frame_index=frame_index,
                reason="not_enough_compact_items",
                compact_item_count=len(compact_items),
            )
        return groups
    split_groups: list[list[LineRecord]] = []
    applied = 0
    rejected = 0
    for group in groups:
        if len(group) <= 1 or len(group) > int(getattr(config, "REPEAT_LABEL_MAX_VERTICAL_SPLIT_LINES", 4)):
            split_groups.append(group)
            continue
        eligible = [_line_record_is_compact_label(item, image) for item in group]
        supports = [
            _line_record_repeat_support(item, compact_items, min_similarity) if ok else 0
            for item, ok in zip(group, eligible, strict=True)
        ]
        seam_votes = 0
        for prev, curr in itertools.pairwise(group):
            top_bg = _sample_background_color(image, prev.rect)
            bottom_bg = _sample_background_color(image, curr.rect)
            seam, _ = _vertical_gap_has_background_seam(image, prev.rect, curr.rect, top_bg, bottom_bg)
            if seam:
                seam_votes += 1
        should_split = (
            sum(1 for ok in eligible if ok) >= 2 and max(supports or [0]) >= 2 and (seam_votes > 0 or all(eligible))
        )
        if should_split:
            applied += 1
            if logger is not None:
                logger.channel(
                    "repeat",
                    "COLOR_SPLIT_APPLIED",
                    frame_index=frame_index,
                    group_size=len(group),
                    support=max(supports or [0]),
                    seam_votes=seam_votes,
                    texts=[item.normalized for item in group],
                )
            for item in group:
                split_groups.append([item])
        else:
            rejected += 1
            if logger is not None:
                logger.channel(
                    "repeat",
                    "COLOR_SPLIT_REJECTED",
                    frame_index=frame_index,
                    group_size=len(group),
                    eligible_count=sum(1 for ok in eligible if ok),
                    max_support=max(supports or [0]),
                    seam_votes=seam_votes,
                    texts=[item.normalized for item in group],
                )
            split_groups.append(group)
    if logger is not None:
        logger.channel(
            "repeat",
            "SPLIT_SUMMARY",
            frame_index=frame_index,
            input_group_count=len(groups),
            output_group_count=len(split_groups),
            applied=applied,
            rejected=rejected,
        )
    return split_groups


def _infer_word_writing_mode(words: list[dict[str, Any]], line_rect: Rect) -> str:
    if not words:
        return "vertical" if _is_vertical_rect(line_rect) else "horizontal"
    if _is_vertical_rect(line_rect):
        return "vertical"
    centers_x = [item["rect"].left + (item["rect"].width / 2.0) for item in words]
    centers_y = [item["rect"].top + (item["rect"].height / 2.0) for item in words]
    x_span = (max(centers_x) - min(centers_x)) if centers_x else 0.0
    y_span = (max(centers_y) - min(centers_y)) if centers_y else 0.0
    vertical_pairs = 0
    ordered = sorted(words, key=lambda item: (item["rect"].top, item["rect"].left))
    for prev, curr in itertools.pairwise(ordered):
        prev_rect = prev["rect"]
        curr_rect = curr["rect"]
        center_dx = abs((prev_rect.left + prev_rect.width / 2.0) - (curr_rect.left + curr_rect.width / 2.0))
        if center_dx <= max(prev_rect.width, curr_rect.width):
            vertical_pairs += 1
    if y_span >= max(8.0, x_span * float(getattr(config, "VERTICAL_TEXT_SCORE_BIAS", 1.1))) and vertical_pairs >= max(
        1, len(words) // 2
    ):
        return "vertical"
    return "horizontal"


def _iter_line_word_items(
    frame_index: int, line_index: int, line: Any, img_w: int, img_h: int, image: Image.Image
) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for word_index, word in enumerate(getattr(line, "words", []) or [], start=1):
        text = (getattr(word, "text", "") or "").strip()
        if not text:
            continue
        rect = _bbox_to_rect(word.bounding_box, img_w, img_h)
        if rect.width < int(getattr(config, "OCR_MIN_WORD_DIM_PX", 2)) or rect.height < int(
            getattr(config, "OCR_MIN_WORD_DIM_PX", 2)
        ):
            continue
        if _rect_area(rect) < int(getattr(config, "OCR_MIN_WORD_RECT_AREA", 6)):
            continue
        bg = _sample_background_color(image, rect)
        fg = _sample_foreground_color(image, [rect], bg)
        conf = getattr(word, "confidence", None)
        try:
            confidence = float(conf) if conf is not None else 1.0
        except Exception:
            confidence = 1.0
        words.append(
            {
                "text": text,
                "rect": rect,
                "bg": bg,
                "fg": fg,
                "confidence": confidence,
                "word_id": f"F{int(frame_index):06d}-L{int(line_index):03d}-W{int(word_index):02d}",
            }
        )
    return words


def _split_line_into_records(
    frame_index: int, line_index: int, line: Any, image: Image.Image, img_w: int, img_h: int
) -> list[LineRecord]:
    raw_text = (getattr(line, "text", "") or "").strip()
    normalized = _normalize_ocr_text(raw_text)
    avg_confidence = _mean_confidence_from_words(
        [word for word in getattr(line, "words", []) or [] if (getattr(word, "text", "") or "").strip()]
    )
    if not normalized or not _text_quality_ok(normalized, avg_confidence=avg_confidence):
        return []
    line_rect = _bbox_to_rect(line.bounding_box, img_w, img_h)
    if not _line_geometry_ok(normalized, line_rect, img_w, img_h, avg_confidence):
        return []
    words = _iter_line_word_items(frame_index, line_index, line, img_w, img_h, image)
    line_id = f"F{int(frame_index):06d}-L{int(line_index):03d}"
    if not words:
        return [
            LineRecord(
                raw=raw_text,
                normalized=normalized,
                rect=line_rect,
                word_rects=[line_rect],
                avg_confidence=avg_confidence,
                line_id=line_id,
                word_ids=[],
            )
        ]
    writing_mode = _infer_word_writing_mode(words, line_rect)
    ordered_words = sorted(
        words,
        key=(lambda w: (w["rect"].top, w["rect"].left))
        if writing_mode == "vertical"
        else (lambda w: (w["rect"].left, w["rect"].top)),
    )
    if not bool(getattr(config, "OCR_COLOR_GROUP_ENABLE", True)) or len(ordered_words) <= 1:
        joined_text = _join_word_texts([w["text"] for w in ordered_words])
        joined_rect = ordered_words[0]["rect"]
        for w in ordered_words[1:]:
            joined_rect = _union_rect(joined_rect, w["rect"])
        joined_norm = _normalize_ocr_text(joined_text)
        if not joined_norm or not _text_quality_ok(joined_norm, avg_confidence=avg_confidence):
            return []
        return [
            LineRecord(
                raw=joined_text,
                normalized=joined_norm,
                rect=joined_rect,
                word_rects=[w["rect"] for w in ordered_words],
                avg_confidence=sum(float(w["confidence"]) for w in ordered_words) / max(1, len(ordered_words)),
                line_id=line_id,
                word_ids=[str(w["word_id"]) for w in ordered_words],
            )
        ]
    groups: list[list[dict[str, Any]]] = [[ordered_words[0]]]
    bg_delta_max = float(getattr(config, "OCR_COLOR_BG_DELTA_E_MAX", 14.0))
    fg_delta_max = float(getattr(config, "OCR_COLOR_FG_DELTA_E_MAX", 18.0))
    for word in ordered_words[1:]:
        prev = groups[-1][-1]
        prev_rect = prev["rect"]
        rect = word["rect"]
        bg_delta = _delta_e(tuple(prev["bg"]), tuple(word["bg"]))
        fg_delta = _delta_e(tuple(prev["fg"]), tuple(word["fg"]))
        if writing_mode == "vertical":
            center_dx = abs((prev_rect.left + prev_rect.width / 2.0) - (rect.left + rect.width / 2.0))
            vertical_gap = max(0, rect.top - prev_rect.bottom)
            same_column = _horizontal_overlap_ratio(prev_rect, rect) >= float(
                getattr(config, "OCR_COLOR_GROUP_HORIZONTAL_OVERLAP_MIN", 0.45)
            ) or center_dx <= max(prev_rect.width, rect.width)
            gap_limit = max(
                6,
                int(
                    max(prev_rect.height, rect.height)
                    * float(getattr(config, "OCR_COLOR_GROUP_VERTICAL_GAP_FACTOR", 0.85))
                ),
            )
            should_split = (not same_column) or (
                vertical_gap > gap_limit and (bg_delta > bg_delta_max or fg_delta > fg_delta_max)
            )
        else:
            gap_px = max(0, rect.left - prev_rect.right)
            same_row = _rect_vertical_overlap_ratio(prev_rect, rect) >= float(
                getattr(config, "OCR_COLOR_GROUP_VERTICAL_OVERLAP_MIN", 0.55)
            )
            gap_limit = max(
                4, int(max(prev_rect.height, rect.height) * float(getattr(config, "OCR_COLOR_GROUP_GAP_FACTOR", 0.68)))
            )
            seam, seam_strength = _gap_has_background_seam(image, prev_rect, rect, prev["bg"], word["bg"])
            huge_gap = gap_px >= max(
                gap_limit + 6,
                int(
                    max(prev_rect.height, rect.height) * float(getattr(config, "OCR_COLOR_GROUP_HUGE_GAP_FACTOR", 1.15))
                ),
            )
            should_split = (
                (not same_row)
                or (huge_gap and seam)
                or (
                    gap_px > gap_limit
                    and (
                        bg_delta > bg_delta_max
                        or fg_delta > fg_delta_max
                        or seam_strength >= float(getattr(config, "OCR_COLOR_GROUP_SEAM_DELTA_E_MAX", 12.0))
                    )
                )
            )
        if should_split:
            groups.append([word])
        else:
            groups[-1].append(word)
    records: list[LineRecord] = []
    for group in groups:
        group_text = _join_word_texts([str(w["text"]) for w in group])
        group_normalized = _normalize_ocr_text(group_text)
        group_conf = sum(float(w["confidence"]) for w in group) / max(1, len(group))
        if not group_normalized or not _text_quality_ok(group_normalized, avg_confidence=group_conf):
            continue
        rect = group[0]["rect"]
        for w in group[1:]:
            rect = _union_rect(rect, w["rect"])
        if not _line_geometry_ok(group_normalized, rect, img_w, img_h, group_conf):
            continue
        records.append(
            LineRecord(
                raw=group_text,
                normalized=group_normalized,
                rect=rect,
                word_rects=[w["rect"] for w in group],
                avg_confidence=group_conf,
                line_id=line_id,
                word_ids=[str(w["word_id"]) for w in group],
            )
        )
    if not records:
        return [
            LineRecord(
                raw=raw_text,
                normalized=normalized,
                rect=line_rect,
                word_rects=[w["rect"] for w in ordered_words],
                avg_confidence=avg_confidence,
                line_id=line_id,
                word_ids=[str(w["word_id"]) for w in ordered_words],
            )
        ]
    return records


def _refresh_observation_hints(observations: list[Observation], image: Image.Image) -> None:
    for obs in observations:
        obs.hud_hint = False
        obs.low_value_hint = False
        obs.dialogue_hint = False
        obs.ui_hint = False
        obs.name_hint = False
    if bool(getattr(config, "ENABLE_DIALOGUE_HINTS", True)):
        for obs in observations:
            obs.dialogue_hint = _obs_is_dialogue_like(obs, image)
    if bool(getattr(config, "ENABLE_UI_HINTS", True)):
        for obs in observations:
            obs.ui_hint = _obs_is_ui_like(obs, image) and not obs.dialogue_hint
    if bool(getattr(config, "ENABLE_NAME_HINTS", True)):
        for obs in observations:
            obs.name_hint = _obs_is_name_like(obs, image) and not obs.dialogue_hint and not obs.ui_hint


def _obs_repeat_candidate(obs: Observation) -> bool:
    compact = _compact_text(obs.normalized)
    return (
        obs.line_count == 1
        and not obs.hud_hint
        and not obs.dialogue_hint
        and not obs.ui_hint
        and not obs.name_hint
        and 2 <= len(compact) <= int(getattr(config, "REPEAT_LABEL_MAX_CHARS", 24))
        and obs.rect.height <= int(getattr(config, "REPEAT_LABEL_MAX_HEIGHT", 72))
    )


def _repeat_grid_geometry_match(a: Observation, b: Observation) -> bool:
    ax = a.rect.left + (a.rect.width / 2.0)
    ay = a.rect.top + (a.rect.height / 2.0)
    bx = b.rect.left + (b.rect.width / 2.0)
    by = b.rect.top + (b.rect.height / 2.0)
    dx = abs(ax - bx)
    dy = abs(ay - by)
    max_width = max(1.0, float(max(a.rect.width, b.rect.width)))
    max_height = max(1.0, float(max(a.rect.height, b.rect.height)))
    center_dist = math.hypot(dx, dy)
    return (
        dx <= (max_width * float(getattr(config, "REPEAT_LABEL_GRID_MAX_X_MULT", 1.55)))
        and dy <= (max_height * float(getattr(config, "REPEAT_LABEL_GRID_MAX_Y_MULT", 4.25)))
        and center_dist
        <= (max(max_width, max_height) * float(getattr(config, "REPEAT_LABEL_GRID_MAX_DIST_MULT", 4.75)))
    )


def _obs_repeat_compatible(a: Observation, b: Observation) -> bool:
    compact_a = _compact_text(a.normalized)
    compact_b = _compact_text(b.normalized)
    if not compact_a or not compact_b:
        return False
    min_similarity = float(getattr(config, "REPEAT_LABEL_MIN_TEXT_SIMILARITY", 0.60))
    ratio, prefix, suffix = _repeat_similarity_stats(compact_a, compact_b)
    if not _repeat_similarity_pass(compact_a, compact_b, min_similarity):
        return False
    height_ratio = max(a.rect.height, b.rect.height) / max(1.0, float(min(a.rect.height, b.rect.height)))
    if height_ratio > float(getattr(config, "REPEAT_LABEL_MAX_HEIGHT_RATIO", 1.45)):
        return False
    width_ratio = max(a.rect.width, b.rect.width) / max(1.0, float(min(a.rect.width, b.rect.width)))
    if width_ratio > float(getattr(config, "REPEAT_LABEL_MAX_WIDTH_RATIO", 2.6)):
        return False
    bg_delta = _delta_e(a.style.background_color, b.style.background_color)
    fg_delta = _delta_e(a.style.fill_color, b.style.fill_color)
    same_row = _rect_vertical_overlap_ratio(a.rect, b.rect) >= float(
        getattr(config, "REPEAT_LABEL_ROW_OVERLAP_MIN", 0.52)
    )
    same_col = _horizontal_overlap_ratio(a.rect, b.rect) >= float(getattr(config, "REPEAT_LABEL_COL_OVERLAP_MIN", 0.32))
    same_grid = _repeat_grid_geometry_match(a, b)
    if not (same_row or same_col or same_grid):
        return False
    if bg_delta <= float(getattr(config, "REPEAT_LABEL_BG_DELTA_E_MAX", 28.0)) and fg_delta <= float(
        getattr(config, "REPEAT_LABEL_FG_DELTA_E_MAX", 36.0)
    ):
        return True
    min_len = min(len(compact_a), len(compact_b))
    edge_min = max(1, int(math.ceil(min_len * float(getattr(config, "REPEAT_LABEL_EDGE_PREFIX_RATIO", 0.6)))))
    return ratio >= min_similarity and (prefix >= edge_min or suffix >= edge_min)


def _should_repair_from_cluster(obs_text: str, canonical_text: str, avg_confidence: float) -> bool:
    obs_compact = _compact_text(obs_text)
    canonical_compact = _compact_text(canonical_text)
    if not obs_compact or not canonical_compact:
        return False
    if obs_compact == canonical_compact:
        return True
    ratio, prefix, suffix = _repeat_similarity_stats(obs_compact, canonical_compact)
    edge_min = max(
        1,
        int(
            math.ceil(
                min(len(obs_compact), len(canonical_compact))
                * float(getattr(config, "REPEAT_LABEL_EDGE_PREFIX_RATIO", 0.6))
            )
        ),
    )
    localized_edge_damage = prefix >= edge_min or suffix >= edge_min
    small_length_delta = abs(len(canonical_compact) - len(obs_compact)) <= int(
        getattr(config, "REPEAT_LABEL_MAX_LENGTH_DELTA", 2)
    )
    return (
        small_length_delta
        and _repeat_similarity_pass(
            obs_compact, canonical_compact, float(getattr(config, "REPEAT_LABEL_MIN_TEXT_SIMILARITY", 0.60))
        )
        and localized_edge_damage
        and (
            avg_confidence <= float(getattr(config, "REPEAT_LABEL_REPAIR_MAX_CONFIDENCE", 0.98))
            or len(obs_compact) < len(canonical_compact)
        )
    )


def _apply_repeated_text_consensus(
    frame_index: int, image: Image.Image, observations: list[Observation], logger: Logger | None = None
) -> int:
    if not bool(getattr(config, "REPEAT_LABEL_CONSENSUS_ENABLED", True)):
        return 0
    candidates = [obs for obs in observations if _obs_repeat_candidate(obs)]
    if logger is not None:
        logger.channel(
            "repeat",
            "CONSENSUS_CANDIDATES",
            frame_index=frame_index,
            candidate_count=len(candidates),
            observation_count=len(observations),
            candidates=[obs.obs_id for obs in candidates],
        )
    if len(candidates) < int(getattr(config, "REPEAT_LABEL_MIN_OBS", 3)):
        return 0
    adjacency: dict[str, set[str]] = {obs.obs_id: set() for obs in candidates}
    obs_by_id = {obs.obs_id: obs for obs in candidates}
    for index, obs in enumerate(candidates):
        for other in candidates[index + 1 :]:
            if _obs_repeat_compatible(obs, other):
                adjacency[obs.obs_id].add(other.obs_id)
                adjacency[other.obs_id].add(obs.obs_id)
    repairs = 0
    cluster_seq = 0
    seen: set[str] = set()
    for obs in candidates:
        if obs.obs_id in seen:
            continue
        stack = [obs.obs_id]
        component_ids: list[str] = []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component_ids.append(current)
            stack.extend(sorted(adjacency.get(current, ())))
        if len(component_ids) < int(getattr(config, "REPEAT_LABEL_MIN_OBS", 3)):
            continue
        component = [obs_by_id[item_id] for item_id in component_ids]
        text_counts = Counter(_compact_text(item.normalized) for item in component if _compact_text(item.normalized))
        if not text_counts:
            continue

        # Bind text_counts via default arg so ruff B023 / closure-over-loop-var
        # is satisfied; the function is consumed in the same iteration anyway.
        def _canonical_score(item: Observation, _counts: Counter = text_counts) -> tuple[int, int, float, float]:
            compact = _compact_text(item.normalized)
            return (int(_counts.get(compact, 0)), len(compact), float(item.avg_confidence), float(item.rect.width))

        canonical_obs = max(component, key=_canonical_score)
        canonical_text = canonical_obs.text
        canonical_norm = canonical_obs.normalized
        canonical_compact = _compact_text(canonical_norm)
        if not canonical_compact:
            continue
        exact_support = sum(1 for item in component if _compact_text(item.normalized) == canonical_compact)
        fuzzy_support = sum(
            1
            for item in component
            if _repeat_similarity_pass(
                item.normalized, canonical_norm, float(getattr(config, "REPEAT_LABEL_MIN_TEXT_SIMILARITY", 0.60))
            )
        )
        if fuzzy_support < int(getattr(config, "REPEAT_LABEL_MIN_FUZZY_SUPPORT", 3)) and exact_support < int(
            getattr(config, "REPEAT_LABEL_MIN_STRONG_SUPPORT", 2)
        ):
            if logger is not None:
                logger.channel(
                    "repeat",
                    "REPEAT_CLUSTER_REJECT",
                    frame_index=frame_index,
                    component_ids=component_ids,
                    canonical_text=canonical_text,
                    exact_support=exact_support,
                    fuzzy_support=fuzzy_support,
                    reason="insufficient_support",
                )
            continue
        cluster_seq += 1
        cluster_id = f"F{int(frame_index):06d}-C{int(cluster_seq):03d}"
        if logger is not None:
            logger.channel(
                "repeat",
                "REPEAT_CLUSTER_CREATE",
                frame_index=frame_index,
                cluster_id=cluster_id,
                component_ids=component_ids,
                canonical_text=canonical_text,
                exact_support=exact_support,
                fuzzy_support=fuzzy_support,
            )
        char_heights = [max(1, rect.height) for item in component for rect in item.word_rects if rect.height > 0]
        median_char_height = _median_int(char_heights, default=max(1, canonical_obs.rect.height))
        char_width_samples = [
            float(item.rect.width) / max(1, len(_compact_text(item.normalized)))
            for item in component
            if _compact_text(item.normalized)
        ]
        median_char_width = _median_float(
            char_width_samples, default=float(canonical_obs.rect.width) / max(1, len(canonical_compact))
        )
        expected_width = max(canonical_obs.rect.width, int(round(median_char_width * max(1, len(canonical_compact)))))
        for item in component:
            similarity = _fuzz_ratio(_compact_text(item.normalized), canonical_compact) / 100.0
            if not _repeat_similarity_pass(
                item.normalized, canonical_compact, float(getattr(config, "REPEAT_LABEL_MIN_TEXT_SIMILARITY", 0.60))
            ):
                if logger is not None:
                    logger.channel(
                        "repeat",
                        "CONSENSUS_MEMBER_REJECT",
                        frame_index=frame_index,
                        cluster_id=cluster_id,
                        obs_id=item.obs_id,
                        similarity=round(similarity, 3),
                        text=item.normalized,
                    )
                continue
            item.repeated_label_hint = True
            item.repeat_cluster_id = cluster_id
            item.repeated_consensus_text = canonical_text
            item.render_anchor_left = min([r.left for r in item.word_rects] or [item.rect.left])
            item.estimated_full_width = max(item.rect.width, expected_width)
            item.median_char_height = median_char_height
            if _should_repair_from_cluster(item.normalized, canonical_norm, item.avg_confidence):
                if item.normalized != canonical_norm:
                    repairs += 1
                item.text = canonical_text
                item.normalized = canonical_norm
                item.repaired_from_cluster = True
                right = min(image.width, item.render_anchor_left + max(item.rect.width, expected_width))
                item.rect = Rect(
                    item.render_anchor_left, item.rect.top, max(1, right - item.render_anchor_left), item.rect.height
                )
                if logger is not None:
                    logger.channel(
                        "repeat",
                        "CONSENSUS_REPAIR_APPLIED",
                        frame_index=frame_index,
                        cluster_id=cluster_id,
                        obs_id=item.obs_id,
                        canonical_text=canonical_text,
                        estimated_full_width=item.estimated_full_width,
                        render_anchor_left=item.render_anchor_left,
                    )
            else:
                item.repaired_from_cluster = False
    if logger is not None:
        logger.channel(
            "repeat", "CONSENSUS_SUMMARY", frame_index=frame_index, repair_count=repairs, cluster_count=cluster_seq
        )
    return repairs


def _build_patch(
    image: Image.Image,
    rect: Rect,
    background_color: tuple[int, int, int],
    extra_expand_x: int = 0,
    extra_expand_y: int = 0,
    extra_patch_x: int | None = None,
    extra_patch_y: int | None = None,
) -> tuple[Rect, Image.Image]:
    """Build the background patch for translated text.

    Always uses the blur+solid blend path: crop the underlying scene, blur
    it, then blend toward the sampled background colour. The cv2.inpaint
    branch was removed after live use showed it producing visible artifacts
    (stretched glyph remnants, wrong edge colours on busy scenes).
    """
    if extra_patch_x is not None:
        extra_expand_x = int(extra_patch_x)
    if extra_patch_y is not None:
        extra_expand_y = int(extra_patch_y)
    patch_rect = _expand_rect(
        rect,
        image.width,
        image.height,
        int(config.PATCH_EXPAND_X) + int(extra_expand_x),
        int(config.PATCH_EXPAND_Y) + int(extra_expand_y),
    )
    crop = _crop(image, patch_rect).convert("RGB")
    blurred = crop.filter(ImageFilter.GaussianBlur(radius=float(config.PATCH_BLUR_RADIUS)))
    solid = Image.new("RGB", crop.size, background_color)
    crop = Image.blend(blurred, solid, float(config.PATCH_SOLID_BLEND))
    patch = crop.convert("RGBA")
    alpha_mask = Image.new("L", crop.size, 0)
    draw = ImageDraw.Draw(alpha_mask)
    draw.rounded_rectangle(
        [0, 0, crop.size[0] - 1, crop.size[1] - 1],
        radius=max(2, min(crop.size) // 8),
        fill=235,
    )
    alpha_mask = alpha_mask.filter(ImageFilter.GaussianBlur(radius=float(config.PATCH_EDGE_FEATHER)))
    patch.putalpha(alpha_mask)
    return patch_rect, patch


def _signature_crop_bytes(image: Image.Image, rect: Rect, side: int, edge_enhanced: bool = False) -> bytes:
    crop = _crop(image, rect).convert("L").resize((side, side), Image.Resampling.BILINEAR)
    if edge_enhanced:
        crop = crop.filter(ImageFilter.FIND_EDGES)
    return crop.tobytes()


def _word_union_rect(rect: Rect, word_rects: list[Rect] | None, image: Image.Image) -> Rect:
    if not word_rects:
        base = rect
    else:
        base = word_rects[0]
        for word_rect in word_rects[1:]:
            base = _union_rect(base, word_rect)
    return _expand_rect(
        base,
        image.width,
        image.height,
        int(getattr(config, "PIXEL_CHANGE_WORD_EXPAND_X", 2)),
        int(getattr(config, "PIXEL_CHANGE_WORD_EXPAND_Y", 2)),
    )


def _region_signature(image: Image.Image, rect: Rect, word_rects: list[Rect] | None = None) -> bytes:
    probe_rect = _expand_rect(
        rect,
        image.width,
        image.height,
        int(getattr(config, "PIXEL_CHANGE_EXPAND_X", 6)),
        int(getattr(config, "PIXEL_CHANGE_EXPAND_Y", 4)),
    )
    context_side = max(6, int(getattr(config, "PIXEL_CHANGE_SIGNATURE_SIZE", 24)))
    text_side = max(context_side, int(getattr(config, "PIXEL_CHANGE_TEXT_SIGNATURE_SIZE", 32)))
    detail_rect = _word_union_rect(rect, word_rects, image)
    context_bytes = _signature_crop_bytes(image, probe_rect, context_side)
    detail_gray = _crop(image, detail_rect).convert("L").resize((text_side, text_side), Image.Resampling.BILINEAR)
    detail_bytes = detail_gray.tobytes()
    edge_bytes = detail_gray.filter(ImageFilter.FIND_EDGES).tobytes()
    return context_bytes + detail_bytes + edge_bytes


def _region_signature_delta(a: bytes, b: bytes) -> float:
    if not a or not b or len(a) != len(b):
        return 999.0
    arr_a = np.frombuffer(a, dtype=np.uint8).astype(np.int16)
    arr_b = np.frombuffer(b, dtype=np.uint8).astype(np.int16)
    return float(np.mean(np.abs(arr_a - arr_b)))


def _region_signature_metrics(a: bytes, b: bytes, cell_delta_min: int) -> tuple[float, float]:
    """Return (fraction_of_cells_changed, mean_abs_delta).

    fraction_of_cells_changed is 0..1: the share of downscaled signature
    cells whose absolute brightness delta exceeded cell_delta_min. This is
    the primary "did this region change" signal — fraction-based so a small
    pulsing effect can't trip the gate just because it changed a few cells
    by a lot.

    mean_abs_delta is the legacy mean(|a-b|) score, kept for the
    PIXEL_CHANGE_HARD_CUT_THRESHOLD fallback that catches scenarios the
    fraction gate misses (e.g. global brightness shift where every cell
    moves by a small amount).
    """
    if not a or not b or len(a) != len(b):
        return 1.0, 999.0
    arr_a = np.frombuffer(a, dtype=np.uint8).astype(np.int16)
    arr_b = np.frombuffer(b, dtype=np.uint8).astype(np.int16)
    abs_diff = np.abs(arr_a - arr_b)
    fraction = float((abs_diff >= max(1, int(cell_delta_min))).sum()) / max(1, len(arr_a))
    mean_delta = float(np.mean(abs_diff))
    return fraction, mean_delta


def _obs_is_bottom_hud(obs: Observation, image: Image.Image) -> bool:
    bottom_band_px = int(getattr(config, "HUD_BOTTOM_BAND_PX", 32))
    thin_max_height = int(getattr(config, "HUD_THIN_MAX_HEIGHT", 24))
    return (
        obs.line_count <= 1
        and obs.rect.height <= thin_max_height
        and obs.rect.bottom >= (image.height - bottom_band_px)
    )


def _is_compact_scene_label_geometry(normalized: str, rect: Rect, line_count: int, image: Image.Image) -> bool:
    compact = _compact_text_len(normalized)
    area_ratio = _rect_area(rect) / max(1, image.width * image.height)
    return (
        line_count <= 2
        and compact <= int(getattr(config, "LOW_VALUE_DUPLICATE_MAX_CHARS", 18))
        and rect.height <= int(getattr(config, "LOW_VALUE_DUPLICATE_MAX_HEIGHT", 30))
        and area_ratio <= float(getattr(config, "LOW_VALUE_DUPLICATE_MAX_AREA_RATIO", 0.04))
        and rect.bottom <= int(image.height * float(getattr(config, "LOW_VALUE_SCENE_MAX_BOTTOM_RATIO", 0.82)))
    )


def _obs_is_edge_menu_cluster(obs: Observation, image: Image.Image) -> bool:
    edge_margin = int(getattr(config, "HUD_EDGE_MARGIN_PX", 72))
    compact = _compact_text_len(obs.normalized)
    return (
        obs.line_count >= int(getattr(config, "LOW_VALUE_EDGE_MENU_MIN_LINES", 3))
        and compact <= int(getattr(config, "LOW_VALUE_EDGE_MENU_MAX_CHARS", 64))
        and obs.rect.width <= int(image.width * float(getattr(config, "LOW_VALUE_EDGE_MENU_MAX_WIDTH_RATIO", 0.42)))
        and (obs.rect.left <= edge_margin or obs.rect.right >= (image.width - edge_margin))
        and obs.rect.bottom <= int(image.height * float(getattr(config, "LOW_VALUE_SCENE_MAX_BOTTOM_RATIO", 0.82)))
    )


def _dialogue_box_top(image: Image.Image) -> int:
    return int(image.height * float(getattr(config, "DIALOGUE_BOX_MIN_TOP_RATIO", 0.78)))


def _rect_in_dialogue_box(rect: Rect, image: Image.Image) -> bool:
    bottom_margin = int(getattr(config, "DIALOGUE_BOX_MAX_BOTTOM_MARGIN_PX", 88))
    return rect.top >= _dialogue_box_top(image) and rect.bottom <= (image.height - max(0, bottom_margin))


def _style_contrast(style: VisualStyle) -> float:
    fill = tuple(int(v) for v in style.fill_color)
    bg = tuple(int(v) for v in style.background_color)
    return max(abs(fill[0] - bg[0]), abs(fill[1] - bg[1]), abs(fill[2] - bg[2]))


def _obs_is_slot_like(obs: Observation, image: Image.Image) -> bool:
    compact = _compact_text_len(obs.normalized)
    has_slot_metadata = _contains_timestamp_like(obs.normalized) or _contains_ui_keyword(obs.normalized)
    return (
        has_slot_metadata
        and not _contains_dialogue_punct(obs.normalized)
        and obs.line_count >= int(getattr(config, "UI_SLOT_MIN_LINES", 2))
        and compact <= int(getattr(config, "UI_SLOT_MAX_CHARS", 120))
        and obs.rect.top <= int(image.height * float(getattr(config, "UI_SLOT_MAX_TOP_RATIO", 0.86)))
        and obs.rect.left <= int(image.width * float(getattr(config, "UI_SLOT_MAX_LEFT_RATIO", 0.26)))
        and obs.rect.width <= int(image.width * float(getattr(config, "UI_SLOT_MAX_WIDTH_RATIO", 0.82)))
    )


def _obs_is_ui_like(obs: Observation, image: Image.Image) -> bool:
    if obs.hud_hint or _obs_is_edge_menu_cluster(obs, image):
        return True
    if _contains_timestamp_like(obs.normalized):
        return True
    if _contains_ui_keyword(obs.normalized):
        return True
    if _obs_is_slot_like(obs, image):
        return True
    return False


def _obs_is_dialogue_like(obs: Observation, image: Image.Image) -> bool:
    if (
        obs.low_value_hint
        or obs.hud_hint
        or _contains_timestamp_like(obs.normalized)
        or _contains_ui_keyword(obs.normalized)
    ):
        return False
    compact = _compact_text_len(obs.normalized)
    lower_half = obs.rect.top >= int(image.height * float(getattr(config, "DIALOGUE_PRIORITY_MIN_TOP_RATIO", 0.35)))
    wide_enough = obs.rect.width >= int(image.width * float(getattr(config, "DIALOGUE_PRIORITY_MIN_WIDTH_RATIO", 0.18)))
    multiline = obs.line_count >= int(getattr(config, "DIALOGUE_PRIORITY_MIN_LINES", 2))
    punctuated = _contains_dialogue_punct(obs.normalized) and compact >= int(
        getattr(config, "DIALOGUE_PRIORITY_MIN_CHARS", 8)
    )
    centered = obs.rect.left > int(image.width * 0.1) and obs.rect.right < int(image.width * 0.9)
    bottom_box = _rect_in_dialogue_box(obs.rect, image)
    bottom_narration = bottom_box and compact >= int(getattr(config, "DIALOGUE_BOX_MIN_SINGLELINE_CHARS", 16))
    strong_contrast = _style_contrast(obs.style) >= float(getattr(config, "DIALOGUE_BOX_MIN_CONTRAST", 72.0))
    return (
        lower_half and centered and wide_enough and (multiline or punctuated or (bottom_narration and strong_contrast))
    )


def _obs_is_name_like(obs: Observation, image: Image.Image) -> bool:
    compact = _compact_text_len(obs.normalized)
    return (
        not obs.low_value_hint
        and not obs.hud_hint
        and not obs.ui_hint
        and not _contains_timestamp_like(obs.normalized)
        and not _contains_ui_keyword(obs.normalized)
        and not _contains_dialogue_punct(obs.normalized)
        and obs.line_count == 1
        and 1 < compact <= int(getattr(config, "NAME_MAX_CHARS", 16))
        and obs.rect.top >= int(image.height * float(getattr(config, "NAME_MIN_TOP_RATIO", 0.55)))
        and obs.rect.width <= int(image.width * float(getattr(config, "NAME_MAX_WIDTH_RATIO", 0.35)))
        and obs.rect.left > int(image.width * 0.12)
        and obs.rect.right < int(image.width * 0.88)
    )


def _zone_index_for_rect(rect: Rect, zones: list[Rect]) -> int:
    """Return the index of the zone with the most overlap with rect, or -1 if none."""
    best_idx = -1
    best_area = 0
    for idx, zone in enumerate(zones):
        area = _rect_intersection_area(rect, zone)
        if area > best_area:
            best_area = area
            best_idx = idx
    return best_idx


def _records_should_merge(prev: LineRecord, item: LineRecord, image: Image.Image) -> bool:
    prev_rect = prev.rect
    rect = item.rect
    prev_mode = _infer_fragment_writing_mode([(prev.raw, prev.normalized, prev_rect)])
    item_mode = _infer_fragment_writing_mode([(item.raw, item.normalized, rect)])
    prev_bg = _sample_background_color(image, prev_rect)
    item_bg = _sample_background_color(image, rect)
    bg_close = _delta_e(prev_bg, item_bg) <= (float(getattr(config, "OCR_COLOR_BG_DELTA_E_MAX", 14.0)) * 1.8)
    if prev_mode == "vertical" or item_mode == "vertical":
        gap = _horizontal_gap(prev_rect, rect)
        overlap = _vertical_overlap_ratio(prev_rect, rect)
        aligned = overlap >= 0.18 or abs(
            (prev_rect.top + prev_rect.height / 2.0) - (rect.top + rect.height / 2.0)
        ) <= max(prev_rect.height, rect.height)
        return gap <= max(18, int(max(prev_rect.width, rect.width) * 1.4)) and aligned and bg_close
    # Two rects that overlap vertically by more than the "actually stacked"
    # margin are almost certainly side-by-side columns, not consecutive
    # rows in one column. Bail before the gap/alignment checks even look
    # at them — otherwise negative-gap always passes ``close_vertically``,
    # and any two columns whose centers are within 300px used to fuse into
    # one garbled multi-column "sentence". This was the primary reason a
    # config menu with a left+right column layout came out completely
    # merged even with the OCR_LINE_MERGE knobs at their minimum.
    vertical_overlap = min(prev_rect.bottom, rect.bottom) - max(prev_rect.top, rect.top)
    if vertical_overlap > int(0.3 * min(prev_rect.height, rect.height)):
        return False
    gap = rect.top - prev_rect.bottom
    overlap = _horizontal_overlap_ratio(prev_rect, rect)
    # Vertical-gap allowance was ``max(18, 0.8 * max_h)``; both terms are now
    # runtime tunables so a menu list can be stopped from getting fused
    # into one multi-line Observation. Require gap >= 0 so a negative gap
    # (rects that overlap vertically) never satisfies the check — the
    # vertical_overlap gate above already handles that case, this is
    # belt-and-suspenders.
    min_gap = max(0, int(getattr(config, "OCR_LINE_MERGE_MIN_GAP_PX", 4)))
    height_factor = float(getattr(config, "OCR_LINE_MERGE_LINE_HEIGHT_FACTOR", 0.35))
    close_vertically = 0 <= gap <= max(min_gap, int(height_factor * max(prev_rect.height, rect.height)))
    # Actual horizontal overlap required. The previous fallback
    # ``center_dist < MERGE_MAX_CENTER_DIST_PX=300`` fired on totally-non-
    # overlapping columns whose centers happened to be within 300px — a
    # 96px-wide "サウンド設定" (left column) got fused with a 211px-wide
    # "マウスカーソル自動移動設定" (right column) because their centers
    # were 199px apart. Overlap-ratio is the safer signal for row
    # continuation; a wrapped paragraph line stays in the same column so
    # its overlap with the previous line is essentially the full smaller
    # width.
    overlap_min = float(getattr(config, "OCR_LINE_MERGE_OVERLAP_MIN", 0.5))
    aligned = overlap >= overlap_min
    return close_vertically and aligned and bg_close


def _group_lines(
    frame_index: int,
    layout: StructuredOcrResult,
    image: Image.Image,
    zone_rects: list[Rect] | None = None,
    logger: Logger | None = None,
) -> list[Observation]:
    per_line: list[LineRecord] = []
    img_w, img_h = layout.image_size
    for line_index, line in enumerate(layout.lines, start=1):
        per_line.extend(_split_line_into_records(frame_index, line_index, line, image, img_w, img_h))
    per_line.sort(key=lambda item: (item.rect.top, item.rect.left))
    if not per_line:
        return []
    groups: list[list[LineRecord]] = []
    current = [per_line[0]]
    for item in per_line[1:]:
        prev = current[-1]
        # Never merge lines that belong to different translation zones
        if zone_rects and _zone_index_for_rect(prev.rect, zone_rects) != _zone_index_for_rect(item.rect, zone_rects):
            groups.append(current)
            current = [item]
            continue
        if _records_should_merge(prev, item, image):
            current.append(item)
        else:
            groups.append(current)
            current = [item]
    groups.append(current)
    groups = _split_repeated_label_groups(frame_index, groups, per_line, image, logger=logger)
    observations: list[Observation] = []
    for obs_index, group in enumerate(groups, start=1):
        fragments = [(item.raw, item.normalized, item.rect) for item in group]
        text, normalized, visual_line_count = _join_fragments_by_row(fragments)
        writing_mode = _infer_fragment_writing_mode(fragments)
        rect = group[0].rect
        word_rects: list[Rect] = []
        avg_confidence = sum(item.avg_confidence for item in group) / max(1, len(group))
        source_line_ids: list[str] = []
        source_word_ids: list[str] = []
        for rec in group:
            rect = _union_rect(rect, rec.rect)
            word_rects.extend(rec.word_rects)
            source_line_ids.append(rec.line_id)
            source_word_ids.extend(rec.word_ids)
        # Median of per-LineRecord (row) heights. OCR line rects are
        # uniform across visually-identical menu rows (all ~24px), whereas
        # per-word rects vary with the specific glyphs in each row (kanji
        # vs hiragana vs punctuation) — the renderer's fallback path
        # ``_median_word_height`` picked up that per-glyph noise and made
        # otherwise-identical menu items render at different font sizes.
        # Setting median_char_height here at the row level gives the
        # renderer a stable, per-row measurement and skips the noisy
        # fallback entirely.
        row_heights = sorted(max(1, int(rec.rect.height)) for rec in group if rec.rect.height > 0)
        median_char_height = row_heights[len(row_heights) // 2] if row_heights else 0
        style = _build_style(image, rect, word_rects)
        observations.append(
            Observation(
                text=text,
                normalized=normalized,
                rect=rect,
                word_rects=word_rects,
                style=style,
                frame_index=frame_index,
                obs_index=obs_index,
                obs_id=f"F{int(frame_index):06d}-O{int(obs_index):03d}",
                source_line_ids=source_line_ids,
                source_word_ids=source_word_ids,
                line_count=max(1, int(visual_line_count or 0)),
                avg_confidence=avg_confidence,
                writing_mode=writing_mode,
                median_char_height=median_char_height,
            )
        )
    _apply_repeated_text_consensus(frame_index, image, observations, logger=logger)
    _refresh_observation_hints(observations, image)
    return observations


def _median_word_height(word_rects: list[Rect], fallback: Rect, line_count: int) -> int:
    heights = sorted(max(1, int(r.height)) for r in word_rects if r.height > 0)
    if heights:
        return heights[len(heights) // 2]
    return max(1, int(fallback.height / max(1, line_count)))


def _render_alignment(track: Track) -> str:
    """Pick a default alignment for a track's rendered text.

    The user-facing default lives in TEXT_DEFAULT_ALIGNMENT (currently
    "top_left" — anchors the translation at the patch's upper-left so it
    follows reading order). Some track classes override that:

    - vertical writing mode → "center" (matches column-text expectations)
    - thin single-line labels (HUD strings, sign captions, repeated
      labels) → "top_left" so they read like the original sign

    Dialogue / name tracks follow the global default so two adjacent
    speech boxes are visually consistent.
    """
    if track.writing_mode == "vertical":
        return "center"
    if track.repeated_label_hint:
        return "top_left"
    thin_max = int(getattr(config, "HUD_THIN_MAX_HEIGHT", 24))
    if track.line_count <= 1 and track.rect.height <= thin_max:
        return "top_left"
    return str(getattr(config, "TEXT_DEFAULT_ALIGNMENT", "top_left"))


def _qt_font_weight(weight_override: int | None = None) -> QtGui.QFont.Weight:
    weight = int(weight_override if weight_override is not None else getattr(config, "TEXT_FONT_WEIGHT", 600))
    if weight <= 450:
        return QtGui.QFont.Weight.Medium
    if weight <= 650:
        return QtGui.QFont.Weight.DemiBold
    return QtGui.QFont.Weight.Bold


def _qt_make_font(
    pixel_size: int, family_override: str | None = None, weight_override: int | None = None
) -> QtGui.QFont:
    family = str(family_override or getattr(config, "TEXT_FONT_FAMILY", "Segoe UI"))
    font = QtGui.QFont(family)
    font.setWeight(_qt_font_weight(weight_override))
    font.setPixelSize(max(1, int(pixel_size)))
    font.setStyleHint(QtGui.QFont.StyleHint.SansSerif)
    font.setStyleStrategy(QtGui.QFont.StyleStrategy.PreferAntialias)
    return font


def _qt_alignment_flags(mode: str, allow_wrap: bool) -> int:
    mode = (mode or "center").lower()
    flags = (
        int(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        if mode == "left"
        else int(QtCore.Qt.AlignmentFlag.AlignCenter)
    )
    if allow_wrap:
        flags |= int(QtCore.Qt.TextFlag.TextWordWrap)
    return flags


def _planned_text_rect(
    base_rect: Rect,
    text: str,
    preferred_px: int,
    family: str,
    weight: int,
    alignment: str,
    allow_wrap: bool,
    image: Image.Image,
    track: Track,
    max_width: int | None = None,
    max_height: int | None = None,
) -> Rect:
    if not (text or "").strip() or not allow_wrap or track.writing_mode == "vertical":
        return base_rect
    if track.dialogue_hint:
        max_ratio = float(getattr(config, "TEXT_DIALOGUE_MAX_HEIGHT_RATIO", 3.25))
    elif track.name_hint:
        max_ratio = float(getattr(config, "TEXT_NAME_MAX_HEIGHT_RATIO", 2.1))
    else:
        max_ratio = float(getattr(config, "TEXT_LAYOUT_MAX_HEIGHT_RATIO", 1.75))
    metrics = QtGui.QFontMetrics(_qt_make_font(preferred_px, family_override=family, weight_override=weight))
    flags = _qt_alignment_flags(alignment, allow_wrap)
    line_h = max(1, metrics.height())
    pad = int(getattr(config, "TEXT_LAYOUT_WRAP_EXTRA_PAD_PX", 6))

    # Horizontal widening. The pre-fix behaviour widened text_rect up to
    # ``max_width`` (the render bounds — the whole translation zone in
    # grouped mode), which made rendered translations extend to the left
    # and right edges of the group box even when the source text sat in a
    # tight corner. That was the "renders too far up and to the right at
    # the very edge of the group box" bug. Widening is now off by default
    # (``TEXT_LAYOUT_WIDEN_MAX_RATIO=1.0``) — text wraps within the source
    # rect's own width. Bump the knob (e.g. 1.5) to let translations spill
    # sideways up to that ratio × base_rect.width instead of wrapping into
    # extra rows.
    widen_ratio = float(getattr(config, "TEXT_LAYOUT_WIDEN_MAX_RATIO", 1.0))
    target_width = base_rect.width
    if widen_ratio > 1.0 and max_width and max_width > base_rect.width:
        width_cap = max(base_rect.width, int(round(base_rect.width * widen_ratio)))
        upper = min(int(max_width), image.width, width_cap)
        if upper > base_rect.width:
            huge_probe = QtCore.QRect(0, 0, upper, line_h * 32)
            natural = metrics.boundingRect(huge_probe, flags, text)
            want = min(upper, natural.width() + pad)
            target_width = max(base_rect.width, want)

    # Centre widened text_rect around the source rect's horizontal centre
    # so translations track the visual position of the source they cover.
    if target_width > base_rect.width:
        centre_x = base_rect.left + base_rect.width // 2
        target_left = centre_x - target_width // 2
    else:
        target_left = base_rect.left

    # Vertical growth cap. Strictly ``base_rect.height * max_ratio`` — the
    # scale-was-too-high fix that used to allow growth up to
    # ``render_bounds`` was letting text_rect fill the zone vertically and
    # then get shifted UP by ``_move_rect_inside`` when it hit the zone
    # bottom (the "renders too far up" tail of the same bug). If a user's
    # ``TEXT_SOURCE_HEIGHT_SCALE`` produces text that doesn't fit at the
    # per-hint ``max_ratio``, the fix is now to raise the ratio directly
    # via ``TEXT_LAYOUT_MAX_HEIGHT_RATIO`` / ``TEXT_DIALOGUE_MAX_HEIGHT_RATIO``
    # / ``TEXT_NAME_MAX_HEIGHT_RATIO`` — a predictable knob instead of
    # implicit zone-fill.
    image_ceiling = max(1, image.height - base_rect.top)
    height_cap = min(image_ceiling, max(base_rect.height, int(round(base_rect.height * max_ratio))))
    probe_rect = QtCore.QRect(target_left, base_rect.top, target_width, height_cap)
    required = metrics.boundingRect(probe_rect, flags, text)
    required_height = required.height() + pad
    target_height = max(base_rect.height, min(height_cap, required_height))
    return Rect(target_left, base_rect.top, target_width, max(1, target_height))


def _track_is_dialogue_box_render_candidate(track: Track, image: Image.Image) -> bool:
    compact = _compact_text_len(track.normalized)
    return (
        bool((track.translation or "").strip())
        and track.render_enabled
        and not track.is_static
        and not track.render_suppressed_until_match
        and int(track.missing_frames or 0) == 0
        and compact >= 2
        and _rect_in_dialogue_box(track.rect, image)
        and not track.hud_hint
        and not track.low_value_hint
        and not track.ui_hint
        and not track.repeated_label_hint
        and not _contains_timestamp_like(track.normalized)
        and not _contains_ui_keyword(track.normalized)
    )


def _track_is_dialogue_box_inline_name(track: Track, image: Image.Image) -> bool:
    compact = _compact_text_len(track.normalized)
    return (
        _track_is_dialogue_box_render_candidate(track, image)
        and int(track.line_count or 1) == 1
        and compact <= int(getattr(config, "DIALOGUE_BOX_INLINE_NAME_MAX_CHARS", 8))
        and track.name_hint
    )


def _track_has_dialogue_mass(track: Track) -> bool:
    compact = _compact_text_len(track.normalized)
    return bool(
        track.dialogue_hint
        or _contains_dialogue_punct(track.normalized)
        or compact >= int(getattr(config, "DIALOGUE_BOX_MIN_SINGLELINE_CHARS", 16))
        or int(track.line_count or 1) >= 2
    )


def _dialogue_box_track_pair_mergeable(left: Track, right: Track, image: Image.Image) -> bool:
    if not (
        _track_is_dialogue_box_render_candidate(left, image) and _track_is_dialogue_box_render_candidate(right, image)
    ):
        return False
    frame_delta = int(getattr(config, "DIALOGUE_BOX_RENDER_MERGE_FRAME_DELTA", 16))
    if abs(int(left.last_obs_frame_index or 0) - int(right.last_obs_frame_index or 0)) > frame_delta:
        return False
    if _same_text_row(left.rect, right.rect):
        return not (
            _track_is_dialogue_box_inline_name(left, image) and _track_is_dialogue_box_inline_name(right, image)
        )
    gap = right.rect.top - left.rect.bottom
    if gap < 0 or gap > int(getattr(config, "DIALOGUE_BOX_CHAIN_GAP_MAX_PX", 34)):
        return False
    tolerance = int(getattr(config, "DIALOGUE_BOX_CHAIN_ALIGN_TOLERANCE_PX", 132))
    overlap = _horizontal_overlap_ratio(left.rect, right.rect)
    aligned = (
        overlap >= 0.08
        or abs(left.rect.left - right.rect.left) <= tolerance
        or abs(left.rect.right - right.rect.right) <= tolerance
        or abs((left.rect.left + left.rect.width / 2) - (right.rect.left + right.rect.width / 2)) <= tolerance
    )
    if not aligned:
        return False
    if _track_is_dialogue_box_inline_name(left, image) and _track_is_dialogue_box_inline_name(right, image):
        return False
    return True


def _join_dialogue_translations(parts: list[str]) -> str:
    cleaned = [re.sub(r"\s+", " ", (part or "").strip()) for part in parts if (part or "").strip()]
    return "\n".join(cleaned)


def _merge_dialogue_render_cluster(cluster: list[Track], image: Image.Image) -> Track:
    lead = cluster[0]
    rect = lead.rect
    word_rects: list[Rect] = []
    for track in cluster:
        rect = _union_rect(rect, track.rect)
        word_rects.extend(list(track.word_rects or []))
    style = _build_style(image, rect, word_rects)
    # Pick a single representative row height for the merged cluster instead
    # of the max. `max` let one tall outlier track (e.g. a heading row that
    # happened to chain in) drive `preferred_px` up for the whole cluster,
    # which then failed to fit in `_resolve_layout`'s shrink window and
    # rendered clipped past the cluster's bounds. Median mirrors the same
    # rule `_merge_zone_observations` uses for grouped zones.
    per_track_heights = sorted(int(track.median_char_height or 0) for track in cluster)
    merged_char_height = per_track_heights[len(per_track_heights) // 2] if per_track_heights else 0
    merged = replace(
        lead,
        text="\n".join((track.text or "").strip() for track in cluster if (track.text or "").strip()),
        normalized="\n".join((track.normalized or "").strip() for track in cluster if (track.normalized or "").strip()),
        rect=rect,
        word_rects=word_rects,
        style=style,
        translation=_join_dialogue_translations([track.translation for track in cluster]),
        line_count=sum(max(1, int(track.line_count or 1)) for track in cluster),
        dialogue_hint=any(track.dialogue_hint for track in cluster) or len(cluster) >= 2,
        name_hint=False,
        render_z_index=min(int(track.render_z_index or 0) for track in cluster),
        last_obs_frame_index=max(int(track.last_obs_frame_index or 0) for track in cluster),
        last_obs_id=next((track.last_obs_id for track in reversed(cluster) if track.last_obs_id), lead.last_obs_id),
        median_char_height=merged_char_height,
    )
    return merged


def _apply_scene_uniformity(tracks: list[Track]) -> list[Track]:
    """Snap visually-similar same-class tracks to their cluster median.

    OCR sampling produces slightly different fill colors (215 vs 218 vs
    206 vs 214) and slightly different bounding-box heights (17 vs 18
    vs 19) for menu items that are, visibly, identical. The user sees
    those as inconsistent rendering. This groups tracks by (dialogue,
    ui, name) hint class, then within each class clusters by fill-color
    delta-E; clusters with >= SCENE_UNIFORMITY_MIN_CLUSTER members get
    a shared median fill / outline / median_char_height. Legitimately
    different items (a highlighted yellow menu row, a title at 25px)
    stay separate because their delta-E to the cluster is above the
    threshold.

    Returns a new list of (possibly ``dataclasses.replace``-copied)
    tracks. The originals in ``self._tracks`` are NOT mutated, so
    per-track color/size sampling stays accurate across frames — the
    uniformity snap is a per-frame render-time overlay.
    """
    if not tracks or not bool(getattr(config, "SCENE_UNIFORMITY_ENABLE", True)):
        return tracks
    min_cluster = max(2, int(getattr(config, "SCENE_UNIFORMITY_MIN_CLUSTER", 3)))
    delta_max = float(getattr(config, "SCENE_UNIFORMITY_COLOR_DELTA_E", 25.0))
    # Split by class hint. Multi-line and grouped-zone tracks are excluded
    # so a paragraph doesn't get flattened toward the surrounding menu
    # rows' color.
    by_class: dict[tuple[bool, bool, bool], list[Track]] = {}
    for t in tracks:
        if int(t.line_count or 1) > 1 or int(t.median_char_height or 0) <= 0:
            continue
        by_class.setdefault(
            (bool(t.dialogue_hint), bool(t.ui_hint), bool(t.name_hint)), []
        ).append(t)
    overrides: dict[int, tuple[tuple[int, int, int], tuple[int, int, int], int]] = {}
    for members in by_class.values():
        if len(members) < min_cluster:
            continue
        clusters: list[list[Track]] = []
        for t in members:
            placed = False
            for cluster in clusters:
                if _delta_e(t.style.fill_color, cluster[0].style.fill_color) <= delta_max:
                    cluster.append(t)
                    placed = True
                    break
            if not placed:
                clusters.append([t])
        for cluster in clusters:
            if len(cluster) < min_cluster:
                continue
            fills = sorted(cluster, key=lambda x: sum(x.style.fill_color))
            outlines = sorted(cluster, key=lambda x: sum(x.style.outline_color))
            heights = sorted(int(x.median_char_height) for x in cluster)
            med_fill = fills[len(fills) // 2].style.fill_color
            med_outline = outlines[len(outlines) // 2].style.outline_color
            med_height = heights[len(heights) // 2]
            for t in cluster:
                overrides[t.track_id] = (med_fill, med_outline, med_height)
    if not overrides:
        return tracks
    result: list[Track] = []
    for t in tracks:
        override = overrides.get(t.track_id)
        if override is None:
            result.append(t)
            continue
        fill, outline, mch = override
        new_style = VisualStyle(
            fill_color=fill, outline_color=outline, background_color=t.style.background_color
        )
        result.append(replace(t, style=new_style, median_char_height=mch))
    return result


def _prepare_render_tracks(tracks: list[Track], image: Image.Image, logger: Logger | None = None) -> list[Track]:
    if not bool(getattr(config, "DIALOGUE_BOX_RENDER_MERGE", True)):
        return tracks
    if len(tracks) < 2:
        return tracks
    max_cluster = max(2, int(getattr(config, "DIALOGUE_BOX_RENDER_MERGE_MAX_TRACKS", 4)))
    min_compact = int(getattr(config, "DIALOGUE_BOX_CHAIN_MIN_COMPACT_CHARS", 20))
    out: list[Track] = []
    cluster: list[Track] = []
    merged_any = False

    def flush_cluster() -> None:
        nonlocal merged_any
        if not cluster:
            return
        compact_total = sum(_compact_text_len(track.normalized) for track in cluster)
        dialogue_mass = any(_track_has_dialogue_mass(track) for track in cluster)
        inline_name_present = any(_track_is_dialogue_box_inline_name(track, image) for track in cluster)
        if (
            len(cluster) >= 2
            and compact_total >= min_compact
            and dialogue_mass
            and (inline_name_present or any(track.dialogue_hint for track in cluster) or len(cluster) >= 3)
        ):
            merged = _merge_dialogue_render_cluster(cluster, image)
            out.append(merged)
            merged_any = True
            if logger is not None:
                logger.channel(
                    "render",
                    message="MERGE_DIALOGUE_RENDER_CLUSTER",
                    source_track_ids=[track.track_id for track in cluster],
                    source_track_refs=[f"T{int(track.track_id):04d}" for track in cluster],
                    source_frames=[int(track.last_obs_frame_index or 0) for track in cluster],
                    merged_track_id=merged.track_id,
                    text_preview=_preview_text(merged.translation),
                )
        else:
            out.extend(cluster)
        cluster.clear()

    for track in tracks:
        if not cluster:
            if _track_is_dialogue_box_render_candidate(track, image):
                cluster.append(track)
            else:
                out.append(track)
            continue
        if len(cluster) < max_cluster and _dialogue_box_track_pair_mergeable(cluster[-1], track, image):
            prospective = cluster + [track]
            prospective_compact = sum(_compact_text_len(item.normalized) for item in prospective)
            prospective_dialogue = any(_track_has_dialogue_mass(item) for item in prospective)
            if prospective_dialogue and prospective_compact >= min_compact:
                cluster.append(track)
                continue
            if _track_is_dialogue_box_inline_name(track, image):
                cluster.append(track)
                continue
        flush_cluster()
        if _track_is_dialogue_box_render_candidate(track, image):
            cluster.append(track)
        else:
            out.append(track)
    flush_cluster()
    return out if merged_any else tracks


class Controller(QtCore.QObject):
    """Owns capture, OCR, temporal tracking, translation, and overlay scenes."""

    statusUpdated = QtCore.pyqtSignal(str)
    clientRegionUpdated = QtCore.pyqtSignal(int, int, int, int)
    targetAttached = QtCore.pyqtSignal(bool)
    pausedStateChanged = QtCore.pyqtSignal(bool)
    overlaySceneUpdated = QtCore.pyqtSignal(object)
    _ocrFrameReady = QtCore.pyqtSignal(object)
    _translationReady = QtCore.pyqtSignal(object)

    def __init__(self, lang_tag: str | None):
        super().__init__()
        self.state = AppState()
        self.lang_tag = lang_tag
        self.logger = Logger(
            config.DEBUG_LOG,
            config.DEBUG_LOG_PATH,
            getattr(config, "DEBUG_LOG_DIR", None),
            queue_size=int(getattr(config, "DEBUG_LOGGER_QUEUE_SIZE", 4096)),
        )
        self._last_capture_error = ""
        self._last_capture_error_key: tuple[str, int | None] = ("", None)
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._tick)
        self._ocrFrameReady.connect(self._handle_ocr_frame)
        self._translationReady.connect(self._handle_translation_result)
        self._frame_q: "queue.Queue[FramePacket]" = queue.Queue(maxsize=1)
        self._tr_q: "queue.PriorityQueue[tuple[int, int, TranslationTask]]" = queue.PriorityQueue(maxsize=64)
        self._stop = threading.Event()
        self._ocr_thread = threading.Thread(target=self._ocr_worker, daemon=True)
        self._tr_thread = threading.Thread(target=self._translate_worker, daemon=True)
        self._ocr = OneOcr(lang_tag)
        self._frame_counter = 0
        self._next_track_id = 1
        self._tracks: dict[int, Track] = {}
        self._translation_cache: dict[str, str] = {}
        # Rolling (source, translation) window fed to the LLM as few-shot
        # context on each translation call. Stops per-line stateless
        # translation from mangling pronouns / dropped subjects / callbacks
        # by giving the model the last few things that were said. Cleared
        # on scene / window reset so context from one scene doesn't leak
        # into the next.
        history_max = max(0, int(getattr(config, "TRANSLATION_CONTEXT_MAX_LINES", 4)))
        self._translation_history: deque[tuple[str, str]] = deque(maxlen=history_max or 1)
        self._translation_history_max = history_max
        # Rolling per-zone char-height samples that feed a running median
        # used as the STABLE render font-size baseline for that zone. The
        # game's dialogue font is uniform; without this, OCR / merge-time
        # measurement noise makes the rendered translation size wobble
        # from one line to the next in the same zone. Cleared on runtime
        # reset so a new scene doesn't inherit the prior scene's font.
        self._zone_baseline_window = max(0, int(getattr(config, "ZONE_FONT_BASELINE_WINDOW", 10)))
        self._zone_baseline_samples: dict[int, deque[int]] = {}
        self._translation_memory_name = "global"
        self._translation_memory_dirty = False
        self._translation_memory_last_save_ms = 0.0
        self._latest_image: Image.Image | None = None
        self._last_observations: list[Observation] = []
        self._last_scene_frame_index = 0
        self._last_annotated_frame_index = -1
        self._latest_captured_frame_index = 0
        self._latest_requested_ocr_frame_index = 0
        self._latest_applied_ocr_frame_index = 0
        self._latest_presented_frame_index = 0
        self._next_render_z_index = 1
        self._last_capture_crc = ""
        self._ocr_inflight_frame_index = 0
        self._translation_seq = itertools.count()
        self._pending_translation_keys: set[str] = set()
        self._region_memo: "OrderedDict[str, RegionMemo]" = OrderedDict()
        self._last_status_message = ""
        self._last_ocr_request_at = 0.0
        self._frame_debug_index: dict[int, dict[str, object]] = {}
        self._last_dialogue_zone_signature: bytes = b""
        self._last_dialogue_zone_crc = ""
        self._dialogue_generation = 0
        self._dialogue_transition_active = False
        self._dialogue_transition_frame_index = 0
        self._last_dialogue_seen_frame = 0
        self._last_translation_activity_frame = 0
        self._ui_only_confirmation_budget = 0
        self._ui_only_confirmation_last_request_frame = 0
        self._translation_drought_recheck_last_frame = 0
        self._capture_occluded = False
        self._last_occlusion_foreign_hwnd = 0
        self._force_next_ocr_reason: str | None = None
        self._last_full_rescan_ts: float = 0.0
        self._ignore_regions: list[Rect] = []
        self._translation_zones: list[TranslationZone] = []
        # Per-zone pixel-change signatures keyed by zone index. Used to
        # suppress cross-zone retranslation when only one zone's pixels
        # actually moved between frames.
        self._per_zone_pixel_signatures: dict[int, bytes] = {}
        self._zones_changed_this_frame: set[int] = set()
        self._pipeline_generation = 1
        self._region_edit_active = False
        self._region_edit_paused_before = False

    def set_language_hint(self, lang_tag: str | None):
        if lang_tag in (None, "", "0"):
            self.lang_tag = None
            label: str = "Auto detect / no hint"
        else:
            self.lang_tag = str(lang_tag)
            label = LANG_HINT.get(self.lang_tag) or self.lang_tag
        self.logger.channel("session", message="LANGUAGE_HINT_CHANGED", lang_tag=self.lang_tag or "0")
        self._set_status(f"Source language hint: {label}.")

    def set_overlay_window(self, hwnd: int):
        self._overlay_hwnd = int(hwnd or 0)

    def _zone_bounds(self) -> tuple[int, int]:
        region = self.state.client_region
        if region is not None:
            return max(1, int(region.width)), max(1, int(region.height))
        image = self._latest_image
        if image is not None:
            return max(1, int(image.width)), max(1, int(image.height))
        return 1, 1

    def _normalize_zone_rect(self, left: int, top: int, width: int, height: int) -> Rect | None:
        max_w, max_h = self._zone_bounds()
        width = max(0, int(width))
        height = max(0, int(height))
        left = max(0, min(max_w - 1, int(left)))
        top = max(0, min(max_h - 1, int(top)))
        right = max(left + 1, min(max_w, left + width))
        bottom = max(top + 1, min(max_h, top + height))
        rect = Rect(left, top, max(1, right - left), max(1, bottom - top))
        if rect.width < 8 or rect.height < 8:
            return None
        return rect

    def _region_policy_changed(self, status: str) -> None:
        self._reset_runtime_state(
            clear_overlay=True, clear_translation_cache=False, clear_region_memo=True, preserve_regions=True
        )
        self.logger.channel(
            "session",
            message="REGION_POLICY_CHANGED",
            ignore_regions=[self._rect_to_dict(r) for r in self._ignore_regions],
            translation_zones=[
                {"rect": self._rect_to_dict(z.rect), "group_all": bool(z.group_all)} for z in self._translation_zones
            ],
            edge_ignore_padding=int(getattr(config, "EDGE_IGNORE_PADDING", 0)),
        )
        self._set_status(status)

    def begin_region_edit(self, kind: str) -> None:
        if self.state.target_hwnd is None:
            self._set_status("Attach to a target window before editing regions.")
            return
        if self._region_edit_active:
            return
        self._region_edit_active = True
        self._region_edit_paused_before = bool(self.state.paused)
        self._pipeline_generation += 1
        self.state.paused = True
        self._timer.stop()
        self._reset_runtime_state(
            clear_overlay=True, clear_translation_cache=False, clear_region_memo=False, preserve_regions=True
        )
        try:
            set_window_enabled(int(self.state.target_hwnd or 0), False)
        except Exception:
            pass
        self.pausedStateChanged.emit(True)
        self.logger.channel(
            "session", message="REGION_EDIT_BEGIN", kind=kind, generation=int(self._pipeline_generation)
        )
        self._set_status(
            f"Editing {kind} region. Drag on the hooked window, then click Lock Region. Press Esc to cancel."
        )

    def end_region_edit(self, kind: str, applied: bool) -> None:
        if not self._region_edit_active:
            return
        self._pipeline_generation += 1
        self._region_edit_active = False
        try:
            set_window_enabled(int(self.state.target_hwnd or 0), True)
        except Exception:
            pass
        self._reset_runtime_state(
            clear_overlay=True, clear_translation_cache=False, clear_region_memo=False, preserve_regions=True
        )
        restore_paused = bool(self._region_edit_paused_before)
        self.state.paused = restore_paused
        if not restore_paused and not self._stop.is_set():
            self._timer.start(int(getattr(config, "CAPTURE_INTERVAL_MS", config.OCR_INTERVAL_MS)))
        self.pausedStateChanged.emit(bool(restore_paused))
        self.logger.channel(
            "session",
            message="REGION_EDIT_END",
            kind=kind,
            applied=bool(applied),
            generation=int(self._pipeline_generation),
            paused=bool(restore_paused),
        )
        if applied:
            self._set_status(f"{kind.title()} region updated. Capture resumed.")
        elif restore_paused:
            self._set_status("Region edit cancelled. App remains paused.")
        else:
            self._set_status("Region edit cancelled. Capture resumed.")

    def set_edge_ignore_padding(self, padding: int) -> None:
        config.EDGE_IGNORE_PADDING = max(0, int(padding))
        self._region_policy_changed(f"Edge ignore padding set to {config.EDGE_IGNORE_PADDING}px.")

    def set_ignore_regions(self, regions: list[tuple[int, int, int, int]] | list[dict[str, int]]) -> None:
        normalized: list[Rect] = []
        for item in regions or []:
            if isinstance(item, dict):
                rect = self._normalize_zone_rect(
                    int(item.get("left", 0)),
                    int(item.get("top", 0)),
                    int(item.get("width", 0)),
                    int(item.get("height", 0)),
                )
            else:
                left, top, width, height = item
                rect = self._normalize_zone_rect(left, top, width, height)
            if rect is not None:
                normalized.append(rect)
        self._ignore_regions = normalized
        self._region_policy_changed("Ignore regions updated.")

    def clear_ignore_region(self) -> None:
        self._ignore_regions = []
        self._region_policy_changed("Ignore regions cleared.")

    def set_translation_zones(self, zones: list[dict[str, object]]) -> None:
        normalized: list[TranslationZone] = []
        for item in zones or []:
            rect = self._normalize_zone_rect(
                _dict_int(item, "left"),
                _dict_int(item, "top"),
                _dict_int(item, "width"),
                _dict_int(item, "height"),
            )
            if rect is None:
                continue
            normalized.append(TranslationZone(rect=rect, group_all=bool(item.get("group_all", False))))
        self._translation_zones = normalized
        self._per_zone_pixel_signatures.clear()
        self._zones_changed_this_frame.clear()
        self._region_policy_changed("Translation zones updated.")

    def clear_translation_region(self) -> None:
        self._translation_zones = []
        self._per_zone_pixel_signatures.clear()
        self._zones_changed_this_frame.clear()
        self._region_policy_changed("Translation zones cleared.")

    def _edge_base_region(self, image: Image.Image) -> Rect:
        full = Rect(0, 0, image.width, image.height)
        pad = max(0, int(getattr(config, "EDGE_IGNORE_PADDING", 0)))
        if pad > 0 and (image.width > pad * 2) and (image.height > pad * 2):
            return Rect(pad, pad, image.width - (pad * 2), image.height - (pad * 2))
        return full

    def _active_base_regions(self, image: Image.Image) -> list[Rect]:
        if not self._translation_zones:
            return [self._edge_base_region(image)]
        full = Rect(0, 0, image.width, image.height)
        regions: list[Rect] = []
        for zone in self._translation_zones:
            clipped = _rect_intersection(full, zone.rect)
            if clipped is not None and clipped.width > 0 and clipped.height > 0:
                regions.append(clipped)
        return regions

    def _ignore_overlap_ratio(self, rect: Rect) -> float:
        area = max(1, _rect_area(rect))
        overlap = 0
        for ignore in self._ignore_regions:
            overlap += _rect_intersection_area(rect, ignore)
        return min(1.0, overlap / area)

    def _word_ignore_ratio(self, word_rects: list[Rect]) -> float:
        if not word_rects:
            return 0.0
        ignored = 0
        for rect in word_rects:
            cx = rect.left + (rect.width / 2.0)
            cy = rect.top + (rect.height / 2.0)
            if any(_rect_contains_point(ignore, cx, cy) for ignore in self._ignore_regions):
                ignored += 1
        return ignored / max(1, len(word_rects))

    def _rect_center_ignored(self, rect: Rect) -> bool:
        cx = rect.left + (rect.width / 2.0)
        cy = rect.top + (rect.height / 2.0)
        return any(_rect_contains_point(ignore, cx, cy) for ignore in self._ignore_regions)

    def _zone_index_for_observation(self, obs: Observation) -> int | None:
        best_idx: int | None = None
        best_score = 0.0
        for idx, zone in enumerate(self._translation_zones):
            if not bool(getattr(zone, "group_all", False)):
                continue
            overlap = _rect_intersection_area(obs.rect, zone.rect)
            if overlap <= 0:
                continue
            score = overlap / max(1, _rect_area(obs.rect))
            if score > best_score:
                best_idx = idx
                best_score = score
        return best_idx

    def _zone_baseline_font_size(self, zone_idx: int | None, sample: int) -> int:
        """Return the stable render font size for a zone. Feeds a rolling
        median over the last N per-zone samples so a genuine game-font
        change eventually pulls the baseline with it, but a single noisy
        OCR frame does not. Passing ``zone_idx=None`` (e.g. from tests
        that don't use zone identity) or a disabled window falls back to
        the raw sample.
        """
        if zone_idx is None or self._zone_baseline_window <= 0:
            return int(sample)
        samples = self._zone_baseline_samples.setdefault(
            int(zone_idx), deque(maxlen=self._zone_baseline_window)
        )
        if int(sample) > 0:
            samples.append(int(sample))
        if not samples:
            return int(sample)
        ordered = sorted(samples)
        return ordered[len(ordered) // 2]

    def _merge_zone_observations(
        self,
        frame_index: int,
        zone: TranslationZone,
        members: list[Observation],
        image: Image.Image,
        zone_idx: int | None = None,
    ) -> Observation | None:
        if not members:
            return None
        vertical_votes = sum(1 for obs in members if obs.writing_mode == "vertical")
        writing_mode = "vertical" if vertical_votes >= max(1, len(members) / 2.0) else "horizontal"
        if writing_mode == "vertical":
            ordered = sorted(members, key=lambda obs: (-obs.rect.left, obs.rect.top, obs.obs_index))
        else:
            ordered = sorted(members, key=lambda obs: (obs.rect.top, obs.rect.left, obs.obs_index))
        text_parts: list[str] = []
        normalized_parts: list[str] = []
        word_rects: list[Rect] = []
        source_line_ids: list[str] = []
        source_word_ids: list[str] = []
        fill_color = ordered[0].style.fill_color
        outline_color = ordered[0].style.outline_color
        bg_color = ordered[0].style.background_color
        avg_conf = 0.0
        for obs in ordered:
            if obs.text.strip():
                text_parts.append(obs.text.strip())
            if obs.normalized.strip():
                normalized_parts.append(obs.normalized.strip())
            word_rects.extend(obs.word_rects)
            source_line_ids.extend(obs.source_line_ids)
            source_word_ids.extend(obs.source_word_ids)
            avg_conf += float(obs.avg_confidence or 1.0)
        if not normalized_parts:
            return None
        separator = "\n"
        merged_text = separator.join(text_parts) if text_parts else separator.join(normalized_parts)
        merged_norm = separator.join(normalized_parts)
        if not merged_norm.strip():
            return None
        style = VisualStyle(fill_color=fill_color, outline_color=outline_color, background_color=bg_color)
        # Tight bbox around the actual source text, NOT the whole zone. Using
        # zone.rect meant the translation patch covered the entire user zone
        # (e.g. a full-width dialogue strip) — wiping out neighbouring scene
        # content the source text never occupied. The union of member rects
        # gives the smallest box that contains every merged line; clipping to
        # the zone keeps a wandering OCR rect from extending past the user's
        # configured area.
        union_rect = ordered[0].rect
        for obs in ordered[1:]:
            union_rect = _union_rect(union_rect, obs.rect)
        clipped_zone = self._clip_rect_to_allowed(zone.rect, image) or zone.rect
        rect = _rect_intersection(union_rect, clipped_zone) or union_rect
        # Derive the merged observation's per-row character height from a
        # SINGLE representative source row, not from aggregates across the
        # whole merged block. This is what the user wants: every rendered
        # row in a grouped zone uses the same font size regardless of how
        # many source rows ended up merged. We take the median of each
        # member's own single-row height to be robust against one outlier
        # row pulling the size up or down.
        per_member_row_heights: list[int] = []
        for obs in ordered:
            if int(obs.median_char_height or 0) > 0:
                per_member_row_heights.append(int(obs.median_char_height))
                continue
            lines = max(1, int(obs.line_count or 1))
            per_member_row_heights.append(max(1, int(obs.rect.height / lines)))
        per_member_row_heights.sort()
        merged_row_height = per_member_row_heights[len(per_member_row_heights) // 2] if per_member_row_heights else 0
        # Stabilise the merged height against a rolling per-zone baseline
        # so the rendered font size doesn't wobble between dialogue lines
        # in the same zone just because OCR measured a couple of pixels
        # differently. Skipped when zone_idx is None (unit tests) so the
        # tests can assert the raw median directly.
        merged_row_height = self._zone_baseline_font_size(zone_idx, merged_row_height)
        return Observation(
            text=merged_text,
            normalized=merged_norm,
            rect=rect,
            word_rects=word_rects,
            style=style,
            frame_index=frame_index,
            obs_index=min(obs.obs_index for obs in ordered),
            obs_id=f"{frame_index:06d}:zone:{source_line_ids[0] if source_line_ids else len(ordered)}",
            source_line_ids=source_line_ids,
            source_word_ids=source_word_ids,
            line_count=sum(max(1, int(obs.line_count or 1)) for obs in ordered),
            avg_confidence=(avg_conf / max(1, len(ordered))),
            median_char_height=merged_row_height,
            writing_mode=writing_mode,
            # Preserve each source line's own rect so the renderer can paint
            # per-line patches tight to each source row instead of one union
            # patch that also covers the gaps between rows.
            member_source_rects=[obs.rect for obs in ordered],
        )

    def _apply_grouped_translation_zones(
        self, frame_index: int, image: Image.Image, observations: list[Observation]
    ) -> list[Observation]:
        if not observations or not any(bool(getattr(zone, "group_all", False)) for zone in self._translation_zones):
            return observations
        passthrough: list[Observation] = []
        grouped: dict[int, list[Observation]] = {}
        for obs in observations:
            zone_idx = self._zone_index_for_observation(obs)
            if zone_idx is None:
                passthrough.append(obs)
            else:
                grouped.setdefault(zone_idx, []).append(obs)
        for zone_idx, members in grouped.items():
            if zone_idx < 0 or zone_idx >= len(self._translation_zones):
                passthrough.extend(members)
                continue
            merged = self._merge_zone_observations(
                frame_index, self._translation_zones[zone_idx], members, image, zone_idx=zone_idx
            )
            if merged is not None:
                passthrough.append(merged)
        return passthrough

    def _observation_allowed(self, obs: Observation, image: Image.Image) -> bool:
        if not self._rect_is_allowed(obs.rect, image):
            return False
        if obs.word_rects:
            if self._word_ignore_ratio(obs.word_rects) >= 0.95:
                return False
        elif self._rect_center_ignored(obs.rect):
            return False
        return True

    def _clip_rect_to_allowed(self, rect: Rect, image: Image.Image) -> Rect | None:
        best: Rect | None = None
        best_area = 0
        for base in self._active_base_regions(image):
            clipped = _rect_intersection(rect, base)
            if clipped is None:
                continue
            area = _rect_area(clipped)
            if area > best_area:
                best = clipped
                best_area = area
        return best

    def _effective_render_regions(self, image: Image.Image) -> list[Rect]:
        return list(self._active_base_regions(image))

    def _render_bounds_for_rect(self, rect: Rect, image: Image.Image, *, expand_to_zone: bool = False) -> Rect:
        """Render bounds for ``rect`` clipped to allowed regions.

        Default behaviour: return rect ∩ allowed (the smallest box that's
        guaranteed safe to paint into). This matches the OCR-rect path
        where the rendered text shouldn't grow past the source-text bbox.

        With ``expand_to_zone=True``: if a user translation zone encloses
        ``rect``, return the whole zone (clipped to image) instead. The
        scene builder uses this so a long translation can grow within
        the user-drawn zone up to its edge, but no further. Without zones
        defined, or when ``rect`` sits outside every zone, this folds back
        to the default rect-clipped behaviour — long translations get the
        safer narrow clamp rather than spilling across the screen.
        """
        if expand_to_zone and self._translation_zones:
            zone_idx = _zone_index_for_rect(rect, [z.rect for z in self._translation_zones])
            if zone_idx >= 0:
                zone_rect = self._translation_zones[zone_idx].rect
                full = Rect(0, 0, image.width, image.height)
                clipped = _rect_intersection(zone_rect, full)
                if clipped is not None and clipped.width > 0 and clipped.height > 0:
                    return clipped
        best = self._clip_rect_to_allowed(rect, image)
        if best is not None:
            return best
        full = Rect(0, 0, image.width, image.height)
        return _rect_intersection(rect, full) or full

    def _rect_is_allowed(self, rect: Rect, image: Image.Image) -> bool:
        clipped = self._clip_rect_to_allowed(rect, image)
        if clipped is None:
            return False
        visible_ratio = _rect_area(clipped) / max(1, _rect_area(rect))
        if visible_ratio < 0.20:
            return False
        if self._rect_center_ignored(rect):
            return False
        if self._ignore_overlap_ratio(rect) >= 0.90:
            return False
        return True

    def _mask_image_for_ocr(self, image: Image.Image) -> Image.Image:
        bases = self._active_base_regions(image)
        if not bases:
            return Image.new("RGB", image.size, (0, 0, 0))
        masked = Image.new("RGB", image.size, (0, 0, 0))
        rgb = image.convert("RGB")
        for base in bases:
            crop = rgb.crop((base.left, base.top, base.right, base.bottom))
            masked.paste(crop, (base.left, base.top))
        draw = ImageDraw.Draw(masked)
        for ignore in self._ignore_regions:
            clipped = _rect_intersection(ignore, Rect(0, 0, image.width, image.height))
            if clipped is None:
                continue
            draw.rectangle([clipped.left, clipped.top, clipped.right - 1, clipped.bottom - 1], fill=(0, 0, 0))
        return masked

    def start(self):
        self._stop.clear()
        if not self._ocr_thread.is_alive():
            self._ocr_thread.start()
        if not self._tr_thread.is_alive():
            self._tr_thread.start()
        self._timer.start(int(getattr(config, "CAPTURE_INTERVAL_MS", config.OCR_INTERVAL_MS)))
        self.logger.channel(
            "session",
            message="START",
            capture_interval_ms=int(getattr(config, "CAPTURE_INTERVAL_MS", config.OCR_INTERVAL_MS)),
            ocr_interval_ms=int(getattr(config, "OCR_MIN_INTERVAL_MS", config.OCR_INTERVAL_MS)),
            lang_tag=self.lang_tag,
            llama_base=config.LLAMA_SERVER_BASE_URL,
            llama_model=getattr(config, "LLAMA_SERVER_MODEL", "local-model"),
            debug_log_dir=getattr(config, "DEBUG_LOG_DIR", "debug_artifacts"),
        )
        self._set_status(
            "Choose a target window and attach. Capture starts immediately; use Pause to reset and hide the overlay."
        )

    def stop(self):
        self.logger.channel("session", message="STOP")
        self._set_status("Stopping...")
        self._stop.set()
        self._timer.stop()
        # Force a final memory save so debounced-but-pending entries from
        # the last few seconds of translations aren't lost on shutdown.
        if self._translation_memory_dirty:
            self._save_translation_memory(force=True)
        try:
            set_window_enabled(int(self.state.target_hwnd or 0), True)
        except Exception:
            pass
        self.logger.close(wait=True, timeout_s=float(getattr(config, "DEBUG_LOGGER_CLOSE_TIMEOUT_S", 15.0)))

    def attach_target(self, hwnd: int):
        if not hwnd:
            self._set_status("No target window selected.")
            return
        rect = self._client_rect_to_region(get_client_rect_screen(hwnd))
        if rect is None:
            self._set_status("Unable to attach to that window. Make sure it is visible and not minimized.")
            self.state.target_hwnd = None
            self.state.client_region = None
            self.targetAttached.emit(False)
            return
        # Flush the previous window's cache to its bucket BEFORE switching
        # buckets and wiping in-memory state. Without this, recent
        # translations from the prior attached window get lost.
        if self._translation_memory_dirty:
            self._save_translation_memory(force=True)
        self.state.target_hwnd = hwnd
        self.state.client_region = rect
        self.state.paused = False
        self._reset_runtime_state(clear_overlay=False, clear_translation_cache=True, preserve_regions=False)
        try:
            diag_obj = get_window_diagnostics(hwnd)
            diag = format_window_diagnostics(diag_obj)
            self.logger.log("ATTACH_TARGET: " + diag)
            title = getattr(diag_obj, "title", "")
            window_class = getattr(diag_obj, "class_name", "")
            self._translation_memory_name = self._build_translation_memory_name(window_class, title)
            self._load_translation_memory()
            self.logger.channel(
                "capture",
                message="ATTACH_TARGET",
                hwnd=f"0x{int(hwnd):08X}",
                pid=getattr(diag_obj, "pid", None),
                title=title,
                window_class=window_class,
                translation_memory=self._translation_memory_name,
                rect={"left": rect.left, "top": rect.top, "width": rect.width, "height": rect.height},
            )
        except Exception as e:
            self.logger.log(f"ATTACH_TARGET_DIAGNOSTICS_ERROR: {e!r}")
        self.overlaySceneUpdated.emit(OverlayScene(size=(rect.width, rect.height), items=[]))
        self.clientRegionUpdated.emit(rect.left, rect.top, rect.width, rect.height)
        self.targetAttached.emit(True)
        self.pausedStateChanged.emit(False)
        self._set_status(
            "Attached. Full-window capture is active. Use Pause to hide the overlay and flush the pipeline."
        )

    def _reset_runtime_state(
        self,
        clear_overlay: bool = True,
        clear_translation_cache: bool = False,
        clear_region_memo: bool = False,
        preserve_regions: bool = True,
    ):
        self._tracks.clear()
        self._latest_image = None
        self._last_observations = []
        self._last_scene_frame_index = 0
        self._last_annotated_frame_index = -1
        self._last_capture_crc = ""
        self._last_ocr_request_at = 0.0
        self._latest_captured_frame_index = 0
        self._latest_requested_ocr_frame_index = 0
        self._latest_applied_ocr_frame_index = 0
        self._latest_presented_frame_index = 0
        self._next_render_z_index = 1
        self._ocr_inflight_frame_index = 0
        self._last_dialogue_zone_signature = b""
        self._last_dialogue_zone_crc = ""
        self._dialogue_generation = 0
        self._dialogue_transition_active = False
        self._dialogue_transition_frame_index = 0
        self._last_dialogue_seen_frame = 0
        self._last_translation_activity_frame = 0
        self._ui_only_confirmation_budget = 0
        self._ui_only_confirmation_last_request_frame = 0
        self._translation_drought_recheck_last_frame = 0
        self._capture_occluded = False
        self._last_occlusion_foreign_hwnd = 0
        self._force_next_ocr_reason = None
        self._last_full_rescan_ts = 0.0
        if not preserve_regions:
            self._ignore_regions = []
            self._translation_zones = []
        self._drain_queue(self._frame_q)
        self._drain_queue(self._tr_q)
        self._pending_translation_keys.clear()
        # Rolling context is per-scene. Always drop it on runtime reset
        # (window switch, pause/resume, region change) so prior-scene
        # dialogue doesn't get injected into a fresh scene's few-shot.
        self._translation_history.clear()
        # Zone font baselines are also per-scene; the new window may use
        # a different in-game font size.
        self._zone_baseline_samples.clear()
        if clear_translation_cache:
            self._translation_cache = {}
            clear_region_memo = True
        if clear_region_memo:
            self._region_memo.clear()
        if clear_overlay:
            if self.state.client_region is not None:
                self.overlaySceneUpdated.emit(
                    OverlayScene(size=(self.state.client_region.width, self.state.client_region.height), items=[])
                )
            else:
                self.overlaySceneUpdated.emit(OverlayScene(size=(1, 1), items=[]))

    def set_paused(self, paused: bool):
        if self.state.target_hwnd is None and not paused:
            self._set_status("Attach to a target window first.")
            return
        # Flush any pending debounced memory write BEFORE the in-memory
        # cache is cleared, otherwise translations from the last
        # TRANSLATION_MEMORY_SAVE_INTERVAL_MS would be lost across the
        # pause/resume cycle.
        if self._translation_memory_dirty:
            self._save_translation_memory(force=True)
        self.state.paused = bool(paused)
        self.logger.channel(
            "session",
            message="PAUSE_STATE",
            paused=bool(paused),
            hwnd=(f"0x{self.state.target_hwnd:08X}" if self.state.target_hwnd else None),
        )
        self._pipeline_generation += 1
        self._reset_runtime_state(clear_overlay=True, clear_translation_cache=True, preserve_regions=True)
        if paused:
            self.pausedStateChanged.emit(True)
            self._set_status("Paused. Overlay hidden and pipeline flushed.")
        else:
            self._load_translation_memory()
            self.pausedStateChanged.emit(False)
            self._set_status("Resumed. Capturing the full target window and painting translations in place.")

    @staticmethod
    def _rect_to_dict(rect: Rect | Region) -> dict[str, int]:
        # Rect (app.types) and Region (app.screen_capture) both expose
        # left/top/width/height — same attribute shape, so one helper
        # serves both call sites (frame-level rects and the capture
        # client region in the FRAME log).
        return {"left": rect.left, "top": rect.top, "width": rect.width, "height": rect.height}

    @staticmethod
    def _style_to_dict(style: VisualStyle) -> dict[str, tuple[int, int, int]]:
        def _t(c: tuple[int, int, int]) -> tuple[int, int, int]:
            return (int(c[0]), int(c[1]), int(c[2]))

        return {
            "fill": _t(style.fill_color),
            "outline": _t(style.outline_color),
            "background": _t(style.background_color),
        }

    @staticmethod
    def _preview_text(text: str, limit: int | None = None) -> str:
        return _preview_text(text, limit)

    @staticmethod
    def _safe_fs_name(text: str) -> str:
        text = re.sub(r"[^A-Za-z0-9._-]+", "_", (text or "").strip())
        return text[:120].strip("._-") or "global"

    def _build_translation_memory_name(self, window_class: str, title: str) -> str:
        parts = [self._safe_fs_name(window_class)]
        title = (title or "").strip()
        if title:
            parts.append(self._safe_fs_name(title[:80]))
        return "__".join([p for p in parts if p]) or "global"

    def _translation_memory_path(self) -> Path:
        memory_dir_name = str(getattr(config, "TRANSLATION_MEMORY_DIR", "memory") or "memory")
        memory_dir = self.logger.base_dir / memory_dir_name
        memory_dir.mkdir(parents=True, exist_ok=True)
        return memory_dir / f"{self._translation_memory_name}.json"

    def _current_cache_fingerprint(self) -> str:
        """Fingerprint of the build's prompt + schema + model + source-hint.
        Stored alongside every cached entries blob so a prompt/model change
        auto-invalidates stale translations instead of returning them.
        """
        return cache_fingerprint(
            LANG_HINT.get(self.lang_tag),
            getattr(config, "LLAMA_SERVER_MODEL", "local-model"),
        )

    def _load_translation_memory(self) -> None:
        if not getattr(config, "TRANSLATION_MEMORY_ENABLED", True):
            return
        path = self._translation_memory_path()
        loaded: Any = None
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                self.logger.channel("translate", message="MEMORY_LOAD_ERROR", path=str(path), error=repr(e))
                loaded = None
        current_fp = self._current_cache_fingerprint()
        entries: dict[str, str] = {}
        stored_fp: str | None = None
        if isinstance(loaded, dict) and isinstance(loaded.get("entries"), dict):
            stored_fp = str(loaded.get("fingerprint") or "")
            if stored_fp == current_fp:
                entries = {str(k): str(v) for k, v in loaded["entries"].items() if str(k).strip() and str(v).strip()}
        elif isinstance(loaded, dict):
            stored_fp = "<legacy-flat>"
        if stored_fp is not None and stored_fp != current_fp:
            self.logger.channel(
                "translate",
                message="MEMORY_FINGERPRINT_MISMATCH",
                bucket=self._translation_memory_name,
                stored_fingerprint=stored_fp,
                current_fingerprint=current_fp,
                path=str(path),
            )
        self._translation_cache = entries
        self.logger.channel(
            "translate",
            message="MEMORY_LOADED",
            bucket=self._translation_memory_name,
            entry_count=len(self._translation_cache),
            fingerprint=current_fp,
            path=str(path),
        )

    def _save_translation_memory(self, *, force: bool = False) -> None:
        """Persist the translation cache to disk, debounced.

        Previously every successful translation triggered a full file
        write. With a steady-state cache of N entries that's an O(N) JSON
        write per translation, scaling linearly with cache size; on a
        long session with hundreds of cached entries the disk traffic
        becomes significant. Debouncing batches saves so the file is
        written at most once per
        ``TRANSLATION_MEMORY_SAVE_INTERVAL_MS`` regardless of how many
        translations arrived in between, but cumulative writes still
        capture every new entry. ``force=True`` bypasses the debounce
        (used on shutdown / pipeline reset).
        """
        if not (
            getattr(config, "TRANSLATION_MEMORY_ENABLED", True) and getattr(config, "TRANSLATION_MEMORY_PERSIST", True)
        ):
            return
        if not force:
            interval_ms = max(0, int(getattr(config, "TRANSLATION_MEMORY_SAVE_INTERVAL_MS", 2000)))
            now = perf_counter() * 1000.0
            if interval_ms > 0 and (now - self._translation_memory_last_save_ms) < float(interval_ms):
                self._translation_memory_dirty = True
                return
            self._translation_memory_last_save_ms = now
        path = self._translation_memory_path()
        try:
            entries = dict(sorted(self._translation_cache.items()))
            fingerprint = self._current_cache_fingerprint()
            payload = {"fingerprint": fingerprint, "version": 2, "entries": entries}
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self._translation_memory_dirty = False
            self.logger.channel(
                "translate",
                message="MEMORY_SAVED",
                bucket=self._translation_memory_name,
                entry_count=len(entries),
                fingerprint=fingerprint,
                path=str(path),
            )
        except Exception as e:
            self.logger.channel(
                "translate",
                message="MEMORY_SAVE_ERROR",
                bucket=self._translation_memory_name,
                path=str(path),
                error=repr(e),
            )

    @staticmethod
    def _image_crc32(image: Image.Image) -> str:
        rgb = image.convert("RGB")
        return f"{zlib.crc32(rgb.tobytes()) & 0xFFFFFFFF:08x}"

    def _dialogue_zone_rect(self, image: Image.Image) -> Rect:
        top = max(0, _dialogue_box_top(image) - 12)
        bottom_margin = max(0, int(getattr(config, "DIALOGUE_BOX_MAX_BOTTOM_MARGIN_PX", 88)) - 12)
        bottom = max(top + 1, image.height - bottom_margin)
        return Rect(0, top, image.width, max(1, bottom - top))

    def _dialogue_zone_signature(self, image: Image.Image) -> tuple[str, bytes]:
        rect = self._dialogue_zone_rect(image)
        crop = image.crop((rect.left, rect.top, rect.right, rect.bottom))
        return self._image_crc32(crop), _region_signature(image, rect)

    def _update_per_zone_change_state(self, image: Image.Image, frame_index: int) -> None:
        """Refresh per-translation-zone pixel signatures and record which
        zones moved this frame. When no user zones are defined the cross-zone
        suppression is disabled (set is left empty; the helper treats absence
        of zones as 'always changed' so behavior matches the pre-zone path).

        Uses the same fraction-of-cells gate as the OCR-retrigger path
        (PIXEL_CHANGE_CELL_DELTA_MIN + PIXEL_CHANGE_FRACTION_THRESHOLD).
        The legacy mean-delta gate (PIXEL_CHANGE_THRESHOLD=8.0) was too
        coarse for text-region zones: a complete dialogue text swap might
        only change ~5-10% of signature cells, and on a dark zone with
        sparse light text the mean of |new-old| across all cells could
        stay below 8 even though every visible glyph changed — which then
        suppressed the source_version bump and left fresh tracks stuck at
        stable_frames=1 forever (visible as ZONE_JITTER_SUPPRESSED firing
        on dialogue lines that were obviously different).
        """
        if not self._translation_zones:
            self._per_zone_pixel_signatures.clear()
            self._zones_changed_this_frame.clear()
            return
        changed: set[int] = set()
        cell_min = int(getattr(config, "PIXEL_CHANGE_CELL_DELTA_MIN", 16))
        frac_threshold = float(getattr(config, "PIXEL_CHANGE_FRACTION_THRESHOLD", 0.01))
        for idx, zone in enumerate(self._translation_zones):
            zone_rect = _rect_intersection(Rect(0, 0, image.width, image.height), zone.rect)
            if zone_rect is None or zone_rect.width <= 0 or zone_rect.height <= 0:
                continue
            sig = _region_signature(image, zone_rect)
            prev = self._per_zone_pixel_signatures.get(idx)
            self._per_zone_pixel_signatures[idx] = sig
            if prev is None:
                # First observation of this zone — count as changed so initial
                # OCR is allowed to populate tracks.
                changed.add(idx)
                continue
            fraction, _ = _region_signature_metrics(prev, sig, cell_min)
            if fraction >= frac_threshold:
                changed.add(idx)
        self._zones_changed_this_frame = changed
        if self.logger is not None and changed:
            self.logger.channel(
                "ocr",
                message="ZONE_PIXEL_CHANGE",
                frame_index=frame_index,
                changed_zones=sorted(changed),
                total_zones=len(self._translation_zones),
            )

    def _track_zone_changed(self, track: Track) -> bool:
        """Return True if the track's enclosing zone changed pixels this
        frame, or True when no user zones are defined (preserving prior
        cross-screen behavior)."""
        if not self._translation_zones:
            return True
        zone_idx = _zone_index_for_rect(track.rect, [z.rect for z in self._translation_zones])
        if zone_idx < 0:
            # Track outside all zones — still treat as changed since the
            # per-zone gate only governs user-defined zones.
            return True
        return zone_idx in self._zones_changed_this_frame

    def _obs_in_dialogue_zone(self, obs: Observation, image: Image.Image | None) -> bool:
        if image is None:
            return bool(obs.dialogue_hint or obs.name_hint)
        zone_rect = self._dialogue_zone_rect(image)
        return bool(
            obs.dialogue_hint
            or obs.name_hint
            or (_rect_vertical_overlap_ratio(obs.rect, zone_rect) >= 0.45 and obs.rect.bottom >= zone_rect.top)
        )

    def _track_in_dialogue_zone(self, track: Track, image: Image.Image | None = None) -> bool:
        image = image or self._latest_image
        if image is None:
            return bool(track.dialogue_hint or track.name_hint)
        zone_rect = self._dialogue_zone_rect(image)
        return bool(
            track.dialogue_hint
            or track.name_hint
            or (_rect_vertical_overlap_ratio(track.rect, zone_rect) >= 0.45 and track.rect.bottom >= zone_rect.top)
        )

    def _current_dialogue_generation_for_obs(self, obs: Observation) -> int:
        if self._obs_in_dialogue_zone(obs, self._latest_image):
            return max(1, int(self._dialogue_generation))
        return 0

    def _dialogue_transition_blocks_track(self, track: Track) -> bool:
        if not self._dialogue_transition_active:
            return False
        if not self._track_in_dialogue_zone(track):
            return False
        if track.dialogue_generation < max(1, int(self._dialogue_generation)):
            return True
        return int(track.last_obs_frame_index or 0) < int(self._dialogue_transition_frame_index or 0)

    def _dialogue_cache_quarantine(self, track: Track) -> bool:
        if not bool(getattr(config, "DIALOGUE_CACHE_QUARANTINE_ENABLED", True)):
            return False
        return self._dialogue_transition_active and self._track_in_dialogue_zone(track)

    def _purge_dialogue_renders_on_zone_change(self, frame_index: int, image: Image.Image) -> bool:
        if not bool(getattr(config, "DIALOGUE_ZONE_PURGE_ENABLED", True)):
            return False
        zone_crc, zone_sig = self._dialogue_zone_signature(image)
        if not self._last_dialogue_zone_signature:
            self._last_dialogue_zone_signature = zone_sig
            self._last_dialogue_zone_crc = zone_crc
            return False
        score = _region_signature_delta(self._last_dialogue_zone_signature, zone_sig)
        changed = zone_crc != self._last_dialogue_zone_crc and score >= float(
            getattr(config, "DIALOGUE_ZONE_PURGE_MIN_DELTA", 22.0)
        )
        self._last_dialogue_zone_signature = zone_sig
        self._last_dialogue_zone_crc = zone_crc
        if not changed:
            return False
        self._dialogue_generation = max(1, int(self._dialogue_generation) + 1)
        self._dialogue_transition_active = True
        self._dialogue_transition_frame_index = int(frame_index)
        scene_changed = False
        purged = []
        zone_rect = self._dialogue_zone_rect(image)
        for track in self._tracks.values():
            if not track.render_enabled or not (track.translation or "").strip():
                continue
            is_dialogue_zone_track = (
                track.dialogue_hint
                or track.name_hint
                or (_rect_vertical_overlap_ratio(track.rect, zone_rect) >= 0.45 and track.rect.bottom >= zone_rect.top)
            )
            if not is_dialogue_zone_track:
                continue
            if self._set_track_render_enabled(track, False):
                scene_changed = True
            track.render_suppressed_until_match = True
            track.translation_pending = False
            purged.append(track.track_id)
        if purged:
            self.logger.channel(
                "tracks",
                message="DIALOGUE_ZONE_PURGE",
                frame_index=frame_index,
                track_ids=purged,
                zone_crc=zone_crc,
                diff_score=round(score, 3),
            )
        return scene_changed

    @staticmethod
    def _frame_ref(frame_index: int | None) -> str:
        if frame_index is None:
            return ""
        try:
            return f"F{int(frame_index):06d}"
        except (TypeError, ValueError):
            return ""

    @staticmethod
    def _track_ref(track_id: int | None) -> str:
        if track_id is None:
            return ""
        try:
            return f"T{int(track_id):04d}"
        except (TypeError, ValueError):
            return ""

    @staticmethod
    def _obs_ref(frame_index: int, obs_index: int) -> str:
        return f"F{int(frame_index):06d}-O{int(obs_index):03d}"

    @staticmethod
    def _line_ref(frame_index: int, line_index: int) -> str:
        return f"F{int(frame_index):06d}-L{int(line_index):03d}"

    @staticmethod
    def _word_ref(frame_index: int, line_index: int, word_index: int) -> str:
        return f"F{int(frame_index):06d}-L{int(line_index):03d}-W{int(word_index):02d}"

    @staticmethod
    def _render_ref(frame_index: int, render_index: int) -> str:
        return f"F{int(frame_index):06d}-R{int(render_index):03d}"

    def _frame_debug_record(self, frame_index: int, **fields):
        if not getattr(config, "DEBUG_SAVE_FRAME_MANIFESTS", True):
            return
        entry = self._frame_debug_index.setdefault(
            int(frame_index), {"frame_index": int(frame_index), "frame_ref": self._frame_ref(frame_index)}
        )
        for key, value in fields.items():
            if value is not None:
                entry[key] = value
        max_entries = 256
        if len(self._frame_debug_index) > max_entries:
            for old_key in sorted(self._frame_debug_index.keys())[:-max_entries]:
                self._frame_debug_index.pop(old_key, None)

    def _track_snapshot(self, track: Track) -> dict:
        return {
            "track_id": track.track_id,
            "track_ref": self._track_ref(track.track_id),
            "text": self._preview_text(track.text),
            "normalized": self._preview_text(track.normalized),
            "rect": self._rect_to_dict(track.rect),
            "style": self._style_to_dict(track.style),
            "stable_frames": track.stable_frames,
            "unchanged_frames": track.unchanged_frames,
            "missing_frames": track.missing_frames,
            "is_static": track.is_static,
            "source_version": track.source_version,
            "has_translation": bool((track.translation or "").strip()),
            "translation_pending": track.translation_pending,
            "line_count": track.line_count,
            "hud_hint": track.hud_hint,
            "low_value_hint": track.low_value_hint,
            "dialogue_hint": track.dialogue_hint,
            "ui_hint": track.ui_hint,
            "name_hint": track.name_hint,
            "repeated_label_hint": track.repeated_label_hint,
            "repeat_cluster_id": track.repeat_cluster_id,
            "repeated_consensus_text": self._preview_text(track.repeated_consensus_text),
            "repaired_from_cluster": track.repaired_from_cluster,
            "render_anchor_left": track.render_anchor_left,
            "estimated_full_width": track.estimated_full_width,
            "median_char_height": track.median_char_height,
            "held_frames": track.held_frames,
            "change_frames": track.change_frames,
            "render_enabled": track.render_enabled,
            "render_z_index": track.render_z_index,
            "dialogue_generation": track.dialogue_generation,
            "render_suppressed_until_match": track.render_suppressed_until_match,
            "created_frame_index": track.created_frame_index,
            "created_frame_ref": self._frame_ref(track.created_frame_index),
            "created_from_obs_id": track.created_from_obs_id,
            "last_obs_id": track.last_obs_id,
            "last_obs_frame_index": track.last_obs_frame_index,
            "last_obs_frame_ref": self._frame_ref(track.last_obs_frame_index),
            "last_matched_frame": track.last_matched_frame,
            "last_matched_frame_ref": self._frame_ref(track.last_matched_frame),
        }

    def _log_track_event(self, event: str, frame_index: int, track: Track, **extra):
        payload = {
            "event": event,
            "frame_index": frame_index,
            **self._track_snapshot(track),
        }
        payload.update(extra)
        self.logger.channel("tracks", **payload)

    def _log_timing(self, stage: str, duration_ms: float, **extra):
        if not getattr(config, "TIMING_LOGGING", True):
            return
        self.logger.channel("timing", message="STEP", stage=stage, duration_ms=round(float(duration_ms), 2), **extra)

    def _save_capture_artifact(
        self, frame_index: int, img: Image.Image, *, image_changed: bool = False, needs_ocr: bool = False
    ) -> str:
        if str(getattr(config, "DEBUG_MODE", "lite")).lower() == "off":
            return ""
        if not getattr(config, "DEBUG_SAVE_CAPTURE_IMAGES", False):
            return ""
        every_n = max(1, int(getattr(config, "DEBUG_SAVE_CAPTURE_EVERY_N", 1)))
        heartbeat_n = max(every_n, int(getattr(config, "DEBUG_SAVE_HEARTBEAT_EVERY_N", 24)))
        only_interesting = bool(getattr(config, "DEBUG_SAVE_ONLY_INTERESTING_FRAMES", False))
        if only_interesting:
            if (
                bool(getattr(config, "DEBUG_SAVE_CAPTURE_ON_CHANGED_ONLY", True))
                and not image_changed
                and not needs_ocr
            ):
                if frame_index % heartbeat_n != 0:
                    return ""
            elif frame_index % every_n != 0:
                return ""
        elif frame_index % every_n != 0:
            return ""
        path = self.logger.save_image("1.Capture", f"{self._frame_ref(frame_index)}_capture", img)
        self._frame_debug_record(frame_index, capture_image=path)
        return path

    def _save_text_snapshot(
        self, category: str, stem: str, content: str, suffix: str = ".txt", frame_index: int | None = None
    ) -> str:
        if str(getattr(config, "DEBUG_MODE", "lite")).lower() == "off":
            return ""
        if not getattr(config, "DEBUG_SAVE_TEXT_SNAPSHOTS", False):
            return ""
        path = self.logger.save_text(category, stem, content, suffix=suffix)
        if frame_index is not None:
            frame_entry = dict(self._frame_debug_index.get(int(frame_index), {}))
            raw_files = frame_entry.get("text_files", [])
            text_files: list[str] = list(raw_files) if isinstance(raw_files, list) else []
            text_files.append(path)
            self._frame_debug_record(int(frame_index), text_files=text_files)
        return path

    def _should_save_scene_artifact(
        self, frame_index: int, *, reason: str | None = None, item_count: int = 0, observation_count: int = 0
    ) -> bool:
        every_n = max(1, int(getattr(config, "DEBUG_SAVE_RENDER_EVERY_N", 1)))
        heartbeat_n = max(every_n, int(getattr(config, "DEBUG_SAVE_HEARTBEAT_EVERY_N", 24)))
        if not bool(getattr(config, "DEBUG_SAVE_ONLY_INTERESTING_FRAMES", False)):
            return frame_index % every_n == 0
        if item_count > 0:
            return frame_index % every_n == 0
        if (reason == "ocr_result" and observation_count > 0) or reason in {"translation_result", "forced_recheck"}:
            return frame_index % every_n == 0
        return frame_index % heartbeat_n == 0

    def _draw_debug_regions(self, draw: ImageDraw.ImageDraw, image_size: tuple[int, int], *, font=None) -> None:
        width, height = image_size
        pad = max(0, int(getattr(config, "EDGE_IGNORE_PADDING", 0)))
        if pad > 0 and width > pad * 2 and height > pad * 2:
            draw.rectangle([pad, pad, width - pad - 1, height - pad - 1], outline=(255, 255, 255, 180), width=1)
            if font is not None:
                try:
                    draw.text((pad + 4, pad + 4), f"Edge ignore: {pad}px", fill=(255, 255, 255, 220), font=font)
                except Exception:
                    pass
        for index, rect in enumerate(self._ignore_regions, start=1):
            draw.rectangle([rect.left, rect.top, rect.right, rect.bottom], outline=(255, 96, 96, 255), width=2)
            if font is not None:
                try:
                    draw.text(
                        (rect.left + 2, max(0, rect.top - 12)), f"Ignore {index}", fill=(255, 96, 96, 255), font=font
                    )
                except Exception:
                    pass
        for index, zone in enumerate(self._translation_zones, start=1):
            rect = zone.rect
            draw.rectangle([rect.left, rect.top, rect.right, rect.bottom], outline=(64, 200, 255, 255), width=2)
            label = f"Translation {index}" + (" [group]" if bool(zone.group_all) else "")
            if font is not None:
                try:
                    draw.text((rect.left + 2, max(0, rect.top - 12)), label, fill=(64, 200, 255, 255), font=font)
                except Exception:
                    pass

    def _save_annotated_frame(
        self,
        frame_index: int,
        image: Image.Image,
        observations: list[Observation],
        *,
        reason: str | None = None,
        scene_item_count: int = 0,
    ) -> str:
        if str(getattr(config, "DEBUG_MODE", "lite")).lower() == "off":
            return ""
        if not getattr(config, "DEBUG_SAVE_ANNOTATED_IMAGES", False):
            return ""
        every_n = max(1, int(getattr(config, "DEBUG_SAVE_ANNOTATED_EVERY_N", 1)))
        if frame_index % every_n != 0:
            return ""
        if not self._should_save_scene_artifact(
            frame_index, reason=reason, item_count=scene_item_count, observation_count=len(observations)
        ):
            return ""
        preview = image.convert("RGBA").copy()
        draw = ImageDraw.Draw(preview)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        if bool(getattr(config, "DEBUG_ANNOTATED_SHOW_OBSERVATIONS", True)):
            for obs in observations:
                draw.rectangle(
                    [obs.rect.left, obs.rect.top, obs.rect.right, obs.rect.bottom], outline=(0, 255, 0, 255), width=2
                )
                if bool(getattr(config, "DEBUG_OVERLAY_DEBUG_LABELS", True)):
                    try:
                        obs_label = obs.obs_id or self._obs_ref(frame_index, obs.obs_index)
                        draw.text(
                            (obs.rect.left + 2, max(0, obs.rect.top - 24)), obs_label, fill=(0, 255, 0, 255), font=font
                        )
                    except Exception:
                        pass
        show_stale = bool(getattr(config, "DEBUG_ANNOTATED_SHOW_STALE_TRACKS", False))
        show_only_render = bool(getattr(config, "DEBUG_ANNOTATED_ONLY_RENDERED_TRACKS", False))
        for track in self._tracks.values():
            matched_now = track.last_matched_frame == frame_index
            if not matched_now and not show_stale:
                continue
            if show_only_render and not (track.render_enabled or track.is_static):
                continue
            color = (80, 180, 255, 255) if track.is_static else (255, 170, 30, 255)
            if not matched_now:
                color = (180, 80, 255, 255)
            r = track.rect
            draw.rectangle([r.left, r.top, r.right, r.bottom], outline=color, width=3)
            status = []
            if track.is_static:
                status.append("S")
            if track.render_enabled:
                status.append("R")
            if not matched_now:
                status.append("H")
            suffix = (":" + "".join(status)) if status else ""
            label = f"{self._track_ref(track.track_id)}{suffix}"
            if bool(getattr(config, "DEBUG_OVERLAY_DEBUG_LABELS", True)) and track.last_obs_id:
                label = f"{label} <- {track.last_obs_id}"
            try:
                draw.text((r.left + 2, max(0, r.top - 12)), label, fill=color, font=font)
            except Exception:
                pass
        self._draw_debug_regions(draw, preview.size, font=font)
        self._last_annotated_frame_index = frame_index
        path = self.logger.save_image("2.Annotated", f"{self._frame_ref(frame_index)}_annotated", preview)
        self._frame_debug_record(frame_index, annotated_image=path)
        return path

    def _save_render_preview(self, frame_index: int, scene: OverlayScene, *, reason: str | None = None) -> str:
        if str(getattr(config, "DEBUG_MODE", "lite")).lower() == "off":
            return ""
        if not getattr(config, "DEBUG_SAVE_RENDER_PREVIEWS", False):
            return ""
        every_n = max(1, int(getattr(config, "DEBUG_SAVE_RENDER_EVERY_N", 1)))
        if frame_index % every_n != 0:
            return ""
        if bool(getattr(config, "DEBUG_SAVE_RENDER_ONLY_WITH_ITEMS", False)) and not scene.items:
            if not self._should_save_scene_artifact(
                frame_index, reason=reason, item_count=0, observation_count=len(self._last_observations)
            ):
                return ""
        elif not self._should_save_scene_artifact(
            frame_index, reason=reason, item_count=len(scene.items), observation_count=len(self._last_observations)
        ):
            return ""
        image = self._latest_image
        if image is None:
            return ""
        preview_qimg = pil_to_qimage(image.convert("RGBA"))
        painter = QtGui.QPainter(preview_qimg)
        items = []
        for item in scene.items:
            extra_pairs = [
                (rect, pil_to_qimage(img)) for rect, img in getattr(item, "extra_patches", []) or []
            ]
            items.append(
                {
                    "patch_rect": item.patch_rect,
                    "patch_image": pil_to_qimage(item.patch_image),
                    "extra_patches": extra_pairs,
                    "text_rect": item.text_rect,
                    "text": item.text,
                    "fill": item.fill_color,
                    "outline": item.outline_color,
                    "allow_wrap": getattr(item, "allow_wrap", True),
                    "preferred_pixel_size": getattr(item, "preferred_pixel_size", 0),
                    "alignment": getattr(item, "alignment", "center"),
                    "writing_mode": getattr(item, "writing_mode", "horizontal"),
                }
            )
        allowed_regions = [
            QtCore.QRect(r.left, r.top, r.width, r.height) for r in getattr(scene, "allowed_regions", [])
        ]
        ignore_regions = [QtCore.QRect(r.left, r.top, r.width, r.height) for r in getattr(scene, "ignore_regions", [])]
        if allowed_regions:
            clip_path = QtGui.QPainterPath()
            for rect in allowed_regions:
                clip_path.addRect(QtCore.QRectF(rect))
            for rect in ignore_regions:
                ignore_path = QtGui.QPainterPath()
                ignore_path.addRect(QtCore.QRectF(rect))
                clip_path = clip_path.subtracted(ignore_path)
            painter.setClipPath(clip_path)
        paint_overlay_items(painter, items)
        painter.end()
        preview = qimage_to_pil(preview_qimg).convert("RGBA")
        draw = ImageDraw.Draw(preview)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        self._draw_debug_regions(draw, preview.size, font=font)
        if bool(getattr(config, "DEBUG_OVERLAY_DEBUG_LABELS", True)):
            for item in scene.items:
                label = item.render_id or self._render_ref(frame_index, 0)
                if item.source_track_id:
                    label = f"{label} {self._track_ref(item.source_track_id)}"
                if item.source_obs_id:
                    label = f"{label} {item.source_obs_id}"
                try:
                    draw.text(
                        (item.text_rect.left + 2, max(0, item.text_rect.top - 12)),
                        label,
                        fill=(255, 255, 0, 255),
                        font=font,
                    )
                except Exception:
                    pass
        path = self.logger.save_image("3.Rendered", f"{self._frame_ref(frame_index)}_rendered", preview)
        self._frame_debug_record(frame_index, render_preview=path)
        return path

    def _save_frame_manifest(self, frame_index: int, scene: OverlayScene, reason: str | None = None) -> str:
        if str(getattr(config, "DEBUG_MODE", "lite")).lower() == "off":
            return ""
        if not getattr(config, "DEBUG_SAVE_FRAME_MANIFESTS", True):
            return ""
        every_n = max(1, int(getattr(config, "DEBUG_FRAME_MANIFEST_EVERY_N", 1)))
        if frame_index % every_n != 0:
            return ""
        if bool(
            getattr(config, "DEBUG_SAVE_MANIFESTS_ONLY_INTERESTING", False)
        ) and not self._should_save_scene_artifact(
            frame_index, reason=reason, item_count=len(scene.items), observation_count=len(self._last_observations)
        ):
            return ""
        image = self._latest_image
        manifest = {
            "frame_index": frame_index,
            "frame_ref": self._frame_ref(frame_index),
            "reason": reason,
            "image_size": None if image is None else {"width": image.width, "height": image.height},
            "ignore_regions": [self._rect_to_dict(r) for r in self._ignore_regions],
            "translation_zones": [
                {"rect": self._rect_to_dict(z.rect), "group_all": bool(z.group_all)} for z in self._translation_zones
            ],
            "edge_ignore_padding": int(getattr(config, "EDGE_IGNORE_PADDING", 0)),
            "artifacts": dict(self._frame_debug_index.get(int(frame_index), {})),
            "observations": [
                {
                    "obs_id": obs.obs_id,
                    "frame_index": obs.frame_index,
                    "frame_ref": self._frame_ref(obs.frame_index),
                    "text": self._preview_text(obs.text),
                    "normalized": self._preview_text(obs.normalized),
                    "rect": self._rect_to_dict(obs.rect),
                    "line_count": obs.line_count,
                    "dialogue_hint": obs.dialogue_hint,
                    "ui_hint": obs.ui_hint,
                    "name_hint": obs.name_hint,
                    "low_value_hint": obs.low_value_hint,
                    "source_line_ids": list(obs.source_line_ids),
                    "source_word_ids": list(obs.source_word_ids),
                    "repeated_label_hint": obs.repeated_label_hint,
                    "repeat_cluster_id": obs.repeat_cluster_id,
                    "repeated_consensus_text": self._preview_text(obs.repeated_consensus_text),
                    "repaired_from_cluster": obs.repaired_from_cluster,
                    "render_anchor_left": obs.render_anchor_left,
                    "estimated_full_width": obs.estimated_full_width,
                    "median_char_height": obs.median_char_height,
                }
                for obs in self._last_observations
            ],
            "tracks": [
                self._track_snapshot(track) for track in sorted(self._tracks.values(), key=lambda t: t.track_id)
            ],
            "render_items": [
                {
                    "render_id": item.render_id,
                    "source_track_id": item.source_track_id,
                    "source_track_ref": self._track_ref(item.source_track_id),
                    "source_obs_id": item.source_obs_id,
                    "source_frame_index": item.source_frame_index,
                    "source_frame_ref": self._frame_ref(item.source_frame_index),
                    "z_index": item.z_index,
                    "text": self._preview_text(item.text),
                    "patch_rect": self._rect_to_dict(item.patch_rect),
                    "text_rect": self._rect_to_dict(item.text_rect),
                    "allow_wrap": bool(getattr(item, "allow_wrap", True)),
                    "preferred_pixel_size": int(getattr(item, "preferred_pixel_size", 0) or 0),
                    "alignment": getattr(item, "alignment", "center"),
                }
                for item in scene.items
            ],
        }
        path = self._save_text_snapshot(
            "frame_manifest",
            f"{self._frame_ref(frame_index)}_manifest",
            json.dumps(manifest, ensure_ascii=False, indent=2),
            suffix=".json",
            frame_index=frame_index,
        )
        self._frame_debug_record(frame_index, frame_manifest=path)
        return path

    def _build_native_ocr_payload(
        self,
        frame_index: int,
        frame_ref: str,
        layout: StructuredOcrResult,
        image: Image.Image,
        observation_count: int,
        image_angle: float,
    ) -> tuple[dict[str, object], list[str], int]:
        """Build the Oneocr.debug.v2 JSON payload from the OCR layout in a
        single pass.

        Returns ``(payload, raw_lines, tiny_native)`` so the caller doesn't
        need to walk ``layout.lines`` separately to compute the raw-text
        snapshot or count tiny line rects. ``word_rect`` is cached per
        word and reused for both the ``rect`` field and the schema dict's
        ``boundingBox``, matching the cache the line loop has used since
        the helper was extracted.
        """
        lines_payload: list[dict[str, object]] = []
        payload: dict[str, object] = {
            "schemaVersion": "Oneocr.debug.v2",
            "coordinateSpace": "frame_pixels",
            "frame_index": frame_index,
            "frame_ref": frame_ref,
            "image": {"width": image.width, "height": image.height, "angle": image_angle},
            "summary": {"rawLineCount": 0, "acceptedBlockCount": observation_count},
            "lines": lines_payload,
        }
        raw_lines: list[str] = []
        tiny_native = 0
        for line_index, line in enumerate(layout.lines, start=1):
            line_text = (line.text or "").strip()
            if not line_text:
                continue
            raw_lines.append(line_text)
            line_id = self._line_ref(frame_index, line_index)
            line_rect = _bbox_to_rect(line.bounding_box, image.width, image.height)
            if _rect_area(line_rect) <= 16:
                tiny_native += 1
            words_list: list[dict[str, object]] = []
            for word_index, word in enumerate(line.words, start=1):
                word_text = (word.text or "").strip()
                if not word_text:
                    continue
                word_rect = _bbox_to_rect(word.bounding_box, image.width, image.height)
                words_list.append(
                    {
                        "word_id": self._word_ref(frame_index, line_index, word_index),
                        "line_id": line_id,
                        "text": word_text,
                        "confidence": word.confidence,
                        "boundingBox": _bbox_schema_dict(word.bounding_box, image.width, image.height),
                        "rect": self._rect_to_dict(word_rect),
                    }
                )
            lines_payload.append(
                {
                    "line_id": line_id,
                    "text": line_text,
                    "boundingBox": _bbox_schema_dict(line.bounding_box, image.width, image.height),
                    "rect": self._rect_to_dict(line_rect),
                    "wordCount": len(words_list),
                    "words": words_list,
                }
            )
        summary = payload["summary"]
        assert isinstance(summary, dict)
        summary["rawLineCount"] = len(raw_lines)
        return payload, raw_lines, tiny_native

    def _log_ocr_frame(
        self, frame_index: int, layout: StructuredOcrResult, observations: list[Observation], image: Image.Image
    ):
        """Persist the per-frame OCR debug artifacts (raw / accepted text
        files, native JSON dump, annotated overlay) and emit the FRAME
        breadcrumb on the ``ocr`` channel.
        """
        frame_ref = self._frame_ref(frame_index)
        accepted_text = "\n\n".join(obs.text for obs in observations)
        # Cache values reused by the JSON payload AND the FRAME log so the
        # CRC / image_angle calculations don't run twice per frame.
        image_crc32 = self._image_crc32(image)
        image_angle = float(getattr(layout, "image_angle", 0.0) or 0.0)
        # Single walk of layout.lines: payload, raw_lines, and tiny_native
        # all fall out of the same pass.
        native_payload, raw_lines, tiny_native = self._build_native_ocr_payload(
            frame_index, frame_ref, layout, image, len(observations), image_angle
        )
        raw_text = "\n".join(raw_lines)
        raw_path = self._save_text_snapshot("ocr_raw", f"{frame_ref}_raw", raw_text, frame_index=frame_index)
        accepted_path = self._save_text_snapshot(
            "ocr_blocks", f"{frame_ref}_accepted", accepted_text, frame_index=frame_index
        )
        native_path = self._save_text_snapshot(
            "ocr_native",
            f"{frame_ref}_native",
            json.dumps(native_payload, ensure_ascii=False, indent=2),
            suffix=".json",
            frame_index=frame_index,
        )
        tiny_obs = sum(1 for obs in observations if _rect_area(obs.rect) <= 16)
        annotated_path = self._save_annotated_frame(frame_index, image, observations)
        self._frame_debug_record(
            frame_index,
            raw_text_file=raw_path,
            accepted_text_file=accepted_path,
            native_json_file=native_path,
            annotated_image=annotated_path,
            image_crc32=image_crc32,
            observation_count=len(observations),
        )
        ocr_payload: dict[str, Any] = dict(
            message="FRAME",
            frame_index=frame_index,
            frame_ref=frame_ref,
            image_size={"width": image.width, "height": image.height},
            image_crc32=image_crc32,
            image_angle=image_angle,
            raw_line_count=len(raw_lines),
            accepted_block_count=len(observations),
            tiny_native_line_rect_count=tiny_native,
            tiny_observation_rect_count=tiny_obs,
            raw_text_preview=self._preview_text(raw_text),
            accepted_text_preview=self._preview_text(accepted_text),
            raw_text_file=raw_path,
            accepted_text_file=accepted_path,
            native_json_file=native_path,
            annotated_image=annotated_path,
        )
        if bool(getattr(config, "DEBUG_LOG_OCR_OBSERVATIONS", False)):
            ocr_payload["observations"] = [
                {
                    "obs_id": obs.obs_id,
                    "obs_index": obs.obs_index,
                    "frame_index": obs.frame_index,
                    "frame_ref": self._frame_ref(obs.frame_index),
                    "text": self._preview_text(obs.text),
                    "normalized": self._preview_text(obs.normalized),
                    "rect": self._rect_to_dict(obs.rect),
                    "word_rects": [self._rect_to_dict(r) for r in obs.word_rects[:16]],
                    "style": self._style_to_dict(obs.style),
                    "line_count": obs.line_count,
                    "hud_hint": obs.hud_hint,
                    "low_value_hint": obs.low_value_hint,
                    "dialogue_hint": obs.dialogue_hint,
                    "ui_hint": obs.ui_hint,
                    "name_hint": obs.name_hint,
                    "avg_confidence": round(float(obs.avg_confidence), 3),
                    "source_line_ids": list(obs.source_line_ids),
                    "source_word_ids": list(obs.source_word_ids),
                    "repeated_label_hint": obs.repeated_label_hint,
                    "repeat_cluster_id": obs.repeat_cluster_id,
                    "repeated_consensus_text": self._preview_text(obs.repeated_consensus_text),
                    "repaired_from_cluster": obs.repaired_from_cluster,
                    "render_anchor_left": obs.render_anchor_left,
                    "estimated_full_width": obs.estimated_full_width,
                    "median_char_height": obs.median_char_height,
                }
                for obs in observations
            ]
        self.logger.channel("ocr", **ocr_payload)
        if tiny_obs and observations:
            self.logger.channel(
                "ocr",
                message="WARNING_TINY_RECTS",
                frame_index=frame_index,
                tiny_observation_rect_count=tiny_obs,
                accepted_block_count=len(observations),
                native_json_file=native_path,
            )

    def _ocr_request_due(self) -> bool:
        min_interval_ms = max(1.0, float(getattr(config, "OCR_MIN_INTERVAL_MS", config.OCR_INTERVAL_MS)))
        now = perf_counter()
        if self._last_ocr_request_at <= 0.0:
            return True
        return ((now - self._last_ocr_request_at) * 1000.0) >= min_interval_ms

    def _effective_missing_params(self, track: Track) -> tuple[int, int, int, int, float]:
        hold_max_frames = int(getattr(config, "PIXEL_HOLD_MAX_FRAMES", 240))
        disappear_confirm_frames = int(getattr(config, "OCR_DISAPPEAR_CONFIRM_FRAMES", 10))
        replaced_confirm_frames = int(getattr(config, "OCR_REPLACED_CONFIRM_FRAMES", 2))
        pixel_delete_after_frames = max(1, int(getattr(config, "PIXEL_CHANGE_DELETE_AFTER_FRAMES", 2)))
        hard_cut_threshold = float(getattr(config, "PIXEL_CHANGE_HARD_CUT_THRESHOLD", 48.0))
        ui_hold_max = int(getattr(config, "UI_HOLD_MAX_FRAMES", 4))
        ui_disappear_confirm = int(getattr(config, "UI_DISAPPEAR_CONFIRM_FRAMES", 2))
        ui_replaced_confirm = int(getattr(config, "UI_REPLACED_CONFIRM_FRAMES", 1))
        dialogue_hold_max = int(getattr(config, "DIALOGUE_HOLD_MAX_FRAMES", 6))
        dialogue_disappear_confirm = int(getattr(config, "DIALOGUE_DISAPPEAR_CONFIRM_FRAMES", 1))
        dialogue_replaced_confirm = int(getattr(config, "DIALOGUE_REPLACED_CONFIRM_FRAMES", 1))
        effective_hold_max = (
            min(hold_max_frames, int(getattr(config, "LOW_VALUE_HOLD_MAX_FRAMES", 6)))
            if track.low_value_hint
            else hold_max_frames
        )
        effective_disappear_confirm = (
            min(disappear_confirm_frames, int(getattr(config, "LOW_VALUE_DISAPPEAR_CONFIRM_FRAMES", 3)))
            if track.low_value_hint
            else disappear_confirm_frames
        )
        effective_replaced_confirm = (
            min(replaced_confirm_frames, int(getattr(config, "LOW_VALUE_REPLACED_CONFIRM_FRAMES", 1)))
            if track.low_value_hint
            else replaced_confirm_frames
        )
        effective_pixel_delete_after = pixel_delete_after_frames
        if track.ui_hint:
            effective_hold_max = min(effective_hold_max, ui_hold_max)
            effective_disappear_confirm = min(effective_disappear_confirm, ui_disappear_confirm)
            effective_replaced_confirm = min(effective_replaced_confirm, ui_replaced_confirm)
        if track.dialogue_hint or track.name_hint:
            effective_hold_max = min(effective_hold_max, dialogue_hold_max)
            effective_disappear_confirm = min(effective_disappear_confirm, dialogue_disappear_confirm)
            effective_replaced_confirm = min(effective_replaced_confirm, dialogue_replaced_confirm)
            effective_pixel_delete_after = 1
        return (
            effective_hold_max,
            effective_disappear_confirm,
            effective_replaced_confirm,
            effective_pixel_delete_after,
            hard_cut_threshold,
        )

    def _can_show_track_render(self, track: Track) -> bool:
        render_stable = int(getattr(config, "RENDER_STABLE_FRAMES_REQUIRED", 2))
        return (
            bool((track.translation or "").strip())
            and (not track.is_static)
            and track.missing_frames == 0
            and (not track.render_suppressed_until_match)
            and (not self._dialogue_transition_blocks_track(track))
            and track.stable_frames >= render_stable
        )

    def _set_track_render_enabled(self, track: Track, enabled: bool) -> bool:
        enabled = bool(enabled)
        was_enabled = bool(track.render_enabled)
        track.render_enabled = enabled
        if enabled and ((not was_enabled) or track.render_z_index <= 0):
            track.render_z_index = self._next_render_z_index
            self._next_render_z_index += 1
        return was_enabled != enabled

    def _fast_render_liveness(self, frame_index: int, image: Image.Image) -> bool:
        if not bool(getattr(config, "FAST_RENDER_LIVENESS_ENABLED", True)):
            return False
        scene_changed = False
        max_tracks = max(1, int(getattr(config, "FAST_RENDER_LIVENESS_MAX_TRACKS", 24)))
        candidates = [
            track
            for track in self._tracks.values()
            if track.render_enabled and (track.translation or "").strip() and not track.is_static
        ]
        candidates.sort(key=lambda track: (0 if (track.dialogue_hint or track.name_hint) else 1, track.track_id))
        liveness_delete_after = max(1, int(getattr(config, "LIVENESS_PIXEL_CHANGE_DELETE_AFTER_FRAMES", 4)))
        for track in candidates[:max_tracks]:
            changed, fraction, current_sig, mean_delta = self._region_visually_changed(image, track, for_liveness=True)
            _, _, _, _, hard_cut_threshold = self._effective_missing_params(track)
            # Hard-cut fallback: even if the fraction gate misses (e.g. all
            # cells moved by a small amount), a catastrophic mean delta
            # still triggers an immediate hide.
            hard_cut = mean_delta >= hard_cut_threshold
            if changed or hard_cut:
                track.liveness_change_frames += 1
                track.liveness_hold_frames = 0
                if hard_cut or track.liveness_change_frames >= liveness_delete_after:
                    if track.render_enabled:
                        self._set_track_render_enabled(track, False)
                        track.render_suppressed_until_match = True
                        track.translation_pending = False
                        scene_changed = True
                    self.logger.channel(
                        "tracks",
                        message="FAST_LIVENESS_HIDE",
                        frame_index=frame_index,
                        track_id=track.track_id,
                        change_frames=track.liveness_change_frames,
                        fraction_changed=round(fraction, 4),
                        mean_delta=round(mean_delta, 3),
                        hard_cut=bool(hard_cut),
                        text=self._preview_text(track.text),
                    )
                else:
                    self.logger.channel(
                        "tracks",
                        message="FAST_LIVENESS_PENDING",
                        frame_index=frame_index,
                        track_id=track.track_id,
                        change_frames=track.liveness_change_frames,
                        fraction_changed=round(fraction, 4),
                        mean_delta=round(mean_delta, 3),
                        text=self._preview_text(track.text),
                    )
                continue
            track.liveness_change_frames = 0
            track.liveness_hold_frames += 1
            track.region_signature = current_sig
        return scene_changed

    def _tick(self):
        """Main capture-and-dispatch tick (called by QTimer).

        Guard → grab → log FRAME → delegate the OCR-vs-skip-vs-synth
        decision to _dispatch_after_capture. Wrapped in the capture
        error handler so any failure (Win32 grab, image conversion,
        downstream logging) is surfaced via the status bar without
        crashing the timer.
        """
        if self._stop.is_set() or self.state.target_hwnd is None:
            return
        if not self._refresh_target_geometry() or self.state.paused:
            return
        hwnd = self.state.target_hwnd
        if self._target_is_occluded(int(hwnd)):
            return
        started = perf_counter()
        try:
            img = grab_window_client(hwnd)
            elapsed_ms = round((perf_counter() - started) * 1000.0, 2)
            self._log_timing("capture", elapsed_ms, hwnd=f"0x{int(hwnd):08X}")
            self._last_capture_error = ""
            self._last_capture_error_key = ("", None)
            self._frame_counter += 1
            frame_index = self._frame_counter
            self._latest_captured_frame_index = frame_index
            self._latest_image = img
            self._last_scene_frame_index = frame_index
            ocr_img = self._mask_image_for_ocr(img)
            image_crc = self._image_crc32(ocr_img)
            self._maybe_periodic_full_rescan(started, frame_index)
            image_changed = image_crc != self._last_capture_crc
            dialogue_zone_changed = self._purge_dialogue_renders_on_zone_change(frame_index, img)
            liveness_changed = self._fast_render_liveness(frame_index, img) or dialogue_zone_changed
            needs_ocr = image_changed or dialogue_zone_changed or self._needs_ocr_on_unchanged_capture(frame_index)
            capture_path = self._save_capture_artifact(
                frame_index, ocr_img, image_changed=image_changed, needs_ocr=needs_ocr
            )
            self._log_capture_frame(frame_index, hwnd, elapsed_ms, img, image_crc, capture_path)
            self._dispatch_after_capture(
                frame_index=frame_index,
                img=img,
                ocr_img=ocr_img,
                image_crc=image_crc,
                image_changed=image_changed,
                dialogue_zone_changed=dialogue_zone_changed,
                liveness_changed=liveness_changed,
                needs_ocr=needs_ocr,
            )
            # Every successful capture updates the CRC. Hoisted out of the
            # per-branch assignments in _dispatch_after_capture so a future
            # branch addition can't forget the invariant.
            self._last_capture_crc = image_crc
        except Exception as e:
            self._handle_capture_exception(hwnd, e)

    def _log_capture_frame(
        self,
        frame_index: int,
        hwnd: int,
        elapsed_ms: float,
        img: Image.Image,
        image_crc: str,
        capture_path: str | None,
    ) -> None:
        """Emit the per-tick FRAME breadcrumb on the capture channel."""
        region = self.state.client_region
        self.logger.channel(
            "capture",
            message="FRAME",
            frame_index=frame_index,
            hwnd=f"0x{int(hwnd):08X}",
            duration_ms=elapsed_ms,
            size={"width": img.width, "height": img.height},
            region=None if region is None else self._rect_to_dict(region),
            image_crc32=image_crc,
            capture_image=capture_path,
        )

    def _dispatch_after_capture(
        self,
        *,
        frame_index: int,
        img: Image.Image,
        ocr_img: Image.Image,
        image_crc: str,
        image_changed: bool,
        dialogue_zone_changed: bool,
        liveness_changed: bool,
        needs_ocr: bool,
    ) -> None:
        """Decide what to do with the frame after capture + change detection.

        Branches (mutually exclusive, in priority order):
          UNCHANGED_SKIP_OCR — nothing moved AND no pending track needs a tick.
          SYNTHETIC_OBS_REUSE — CRC matches and OCR is deterministic, so
              re-apply _last_observations instead of paying a real OCR call.
          OCR_THROTTLED — would have OCR'd but min-interval hasn't elapsed.
          Queue path — push a FramePacket onto _frame_q and let the OCR
              worker pick it up; emit forced_recheck or capture_refresh
              once dispatched.
        """
        if not needs_ocr:
            self.logger.channel("capture", message="UNCHANGED_SKIP_OCR", frame_index=frame_index, image_crc32=image_crc)
            self._age_tracks_without_ocr(frame_index, img, reason="unchanged_skip")
            self._emit_scene(frame_index, reason="unchanged_skip")
            return
        force_reason = self._forced_ocr_reason(
            frame_index,
            image_changed=image_changed,
            dialogue_zone_changed=dialogue_zone_changed,
            liveness_changed=liveness_changed,
        )
        if self._can_synthesize_observations(
            image_changed=image_changed,
            dialogue_zone_changed=dialogue_zone_changed,
            liveness_changed=liveness_changed,
            force_reason=force_reason,
        ):
            self._synthesize_observation_reuse(frame_index, image_crc)
            return
        if (not force_reason) and (not self._ocr_request_due()):
            self.logger.channel(
                "capture",
                message="OCR_THROTTLED",
                frame_index=frame_index,
                image_crc32=image_crc,
                image_changed=bool(image_changed),
                liveness_changed=bool(liveness_changed),
            )
            self._age_tracks_without_ocr(frame_index, img, reason="capture_refresh")
            self._emit_scene(frame_index, reason="capture_refresh")
            return
        self._enqueue_ocr_packet(
            frame_index=frame_index,
            ocr_img=ocr_img,
            image_crc=image_crc,
            image_changed=image_changed,
            liveness_changed=liveness_changed,
            force_reason=force_reason,
        )

    def _synthesize_observation_reuse(self, frame_index: int, image_crc: str) -> None:
        """OCR-skip path: re-apply _last_observations to advance track
        stability without paying a real OCR call. Safe only when the
        _can_synthesize_observations predicate already holds.
        """
        self.logger.channel(
            "capture",
            message="SYNTHETIC_OBS_REUSE",
            frame_index=frame_index,
            image_crc32=image_crc,
            observation_count=len(self._last_observations),
        )
        synth_started = perf_counter()
        self._update_tracks(frame_index, self._last_observations)
        self._log_timing(
            "update_tracks_synthetic",
            (perf_counter() - synth_started) * 1000.0,
            frame_index=frame_index,
            observation_count=len(self._last_observations),
        )
        self._emit_scene(frame_index, reason="synthetic_no_change")

    def _enqueue_ocr_packet(
        self,
        *,
        frame_index: int,
        ocr_img: Image.Image,
        image_crc: str,
        image_changed: bool,
        liveness_changed: bool,
        force_reason: str | None,
    ) -> None:
        """Push a FramePacket onto _frame_q; handle the queue-full drop-
        oldest replacement. Emits FORCE_OCR_BYPASS / WAITING_OCR /
        QUEUE_PUSH / QUEUE_REPLACE log breadcrumbs around the push and
        consumes the relevant force_reason budgets after a successful
        enqueue.
        """
        if force_reason:
            self.logger.channel(
                "capture",
                message="FORCE_OCR_BYPASS",
                frame_index=frame_index,
                image_crc32=image_crc,
                reason=force_reason,
                image_changed=bool(image_changed),
                liveness_changed=bool(liveness_changed),
            )
        if bool(self._ocr_inflight_frame_index) or self._frame_q.qsize() > 0:
            self.logger.channel(
                "capture",
                message="WAITING_OCR",
                frame_index=frame_index,
                image_crc32=image_crc,
                image_changed=bool(image_changed),
                inflight_frame=self._ocr_inflight_frame_index or None,
                queue_size=self._frame_q.qsize(),
            )
        packet = FramePacket(frame_index=frame_index, image=ocr_img, generation=int(self._pipeline_generation))
        self._latest_requested_ocr_frame_index = frame_index
        self._last_ocr_request_at = perf_counter()
        try:
            self._frame_q.put_nowait(packet)
            self.logger.channel(
                "capture",
                message="QUEUE_PUSH",
                frame_index=frame_index,
                queue_size=self._frame_q.qsize(),
                image_changed=bool(image_changed),
            )
        except queue.Full:
            dropped = None
            try:
                dropped = self._frame_q.get_nowait()
            except queue.Empty:
                pass
            self._frame_q.put_nowait(packet)
            self.logger.channel(
                "capture",
                message="QUEUE_REPLACE",
                frame_index=frame_index,
                dropped_frame_index=(None if dropped is None else dropped.frame_index),
                queue_size=self._frame_q.qsize(),
                latest_only=bool(getattr(config, "OCR_QUEUE_LATEST_ONLY", True)),
            )
        if force_reason == self._force_next_ocr_reason:
            self._force_next_ocr_reason = None
        if force_reason == "ui_only_confirmation" and self._ui_only_confirmation_budget > 0:
            self._ui_only_confirmation_budget = max(0, int(self._ui_only_confirmation_budget) - 1)
            self._ui_only_confirmation_last_request_frame = frame_index
        elif force_reason == "translation_drought":
            self._translation_drought_recheck_last_frame = frame_index
        self._emit_scene(frame_index, reason=("forced_recheck" if force_reason else "capture_refresh"))

    def _handle_capture_exception(self, hwnd: int, exc: Exception) -> None:
        """Surface a capture-loop exception via the status bar and the
        capture log channel. Coalesces repeated errors by exception class
        + Win32 error code so a transient cause with jittering args
        (DXGI surface lost mid-grab, BitBlt to a resizing window, DPI
        change) doesn't flood the debug log on every tick. The
        user-facing status bar still updates every time so the user sees
        the latest message.
        """
        try:
            diag = format_window_diagnostics(get_window_diagnostics(hwnd))
        except Exception as diag_error:
            diag = f"<window diagnostics unavailable: {diag_error!r}>"
        msg = f"Capture error: {exc!r}. See overlay_translator_debug.log"
        # Stable coalescing key: exception class + winerror (when present).
        # repr(exc) embeds jittering hwnd/pointer values for some OSError
        # subclasses, so comparing full strings dedupes only when the args
        # tuple stabilises (rare during a transient cause).
        coalesce_key = (type(exc).__name__, getattr(exc, "winerror", None))
        self.logger.channel("capture", message="ERROR", hwnd=f"0x{int(hwnd):08X}", error=repr(exc), diagnostics=diag)
        if coalesce_key != self._last_capture_error_key:
            self.logger.log(f"CAPTURE_ERROR: {exc!r}")
            self.logger.log("CAPTURE_ERROR_TARGET: " + diag)
            self._last_capture_error = msg
            self._last_capture_error_key = coalesce_key
        self._set_status(msg)

    def _refresh_target_geometry(self) -> bool:
        hwnd = self.state.target_hwnd
        if hwnd is None:
            return False
        if not is_window_usable(hwnd):
            self.state.target_hwnd = None
            self.state.client_region = None
            self.state.paused = True
            self.overlaySceneUpdated.emit(OverlayScene(size=(1, 1), items=[]))
            self.targetAttached.emit(False)
            self.logger.channel("capture", message="TARGET_UNAVAILABLE", hwnd=f"0x{int(hwnd):08X}")
            self._set_status("Target window is no longer available. Attach again.")
            return False
        rect = self._client_rect_to_region(get_client_rect_screen(hwnd))
        if rect is None:
            self.logger.channel("capture", message="CLIENT_RECT_UNAVAILABLE", hwnd=f"0x{int(hwnd):08X}")
            self._set_status("Unable to read the target window client area.")
            return False
        if not _same_region(rect, self.state.client_region):
            old = self.state.client_region
            size_changed = bool(old is not None and (old.width != rect.width or old.height != rect.height))
            self.state.client_region = rect
            if size_changed and bool(getattr(config, "SIZE_CHANGE_FLUSH_ENABLED", True)):
                self._handle_client_size_change(hwnd, old, rect)
            self.clientRegionUpdated.emit(rect.left, rect.top, rect.width, rect.height)
            self.logger.channel(
                "capture",
                message="CLIENT_RECT_CHANGED",
                hwnd=f"0x{int(hwnd):08X}",
                old=(
                    None
                    if old is None
                    else {"left": old.left, "top": old.top, "width": old.width, "height": old.height}
                ),
                new={"left": rect.left, "top": rect.top, "width": rect.width, "height": rect.height},
                size_changed=bool(size_changed),
            )
        return True

    def _handle_client_size_change(self, hwnd: int, old: Region | None, new: Region) -> None:
        self.logger.channel(
            "capture",
            message="CLIENT_SIZE_FLUSH",
            hwnd=f"0x{int(hwnd):08X}",
            old=(None if old is None else {"left": old.left, "top": old.top, "width": old.width, "height": old.height}),
            new={"left": new.left, "top": new.top, "width": new.width, "height": new.height},
            flushed_tracks=len(self._tracks),
        )
        self._reset_runtime_state(
            clear_overlay=True, clear_translation_cache=False, clear_region_memo=True, preserve_regions=True
        )
        self._set_status("Target size changed. Overlay flushed and rescanning from scratch.")

    def _client_rect_to_region(self, rect: ClientRect | None) -> Region | None:
        if rect is None:
            return None
        return Region(rect.left, rect.top, rect.width, rect.height)

    def _set_status(self, msg: str):
        if msg == self._last_status_message:
            return
        self._last_status_message = msg
        self.logger.log("STATUS: " + msg)
        self.logger.channel("session", message="STATUS", status=msg)
        self.statusUpdated.emit(msg)

    def _has_recent_dialogue_context(self, frame_index: int) -> bool:
        lookback = max(1, int(getattr(config, "OCR_FORCE_ON_CHANGED_DIALOGUE_LOOKBACK_FRAMES", 36)))
        if self._last_dialogue_seen_frame and (frame_index - int(self._last_dialogue_seen_frame)) <= lookback:
            return True
        for track in self._tracks.values():
            if not (track.dialogue_hint or track.name_hint):
                continue
            if (frame_index - int(track.last_obs_frame_index or 0)) <= lookback:
                return True
        return False

    def _target_is_occluded(self, hwnd: int) -> bool:
        if not bool(getattr(config, "CAPTURE_PAUSE_WHEN_OCCLUDED", True)):
            if self._capture_occluded:
                self._capture_occluded = False
                self._last_occlusion_foreign_hwnd = 0
            return False
        ignore_hwnds: tuple[int, ...] = (int(self._overlay_hwnd),) if int(self._overlay_hwnd or 0) else ()
        occlusion = detect_window_occlusion(
            hwnd,
            ignore_hwnds=ignore_hwnds,
            sample_cols=max(1, int(getattr(config, "CAPTURE_OCCLUSION_SAMPLE_COLS", 3))),
            sample_rows=max(1, int(getattr(config, "CAPTURE_OCCLUSION_SAMPLE_ROWS", 3))),
            inset_px=max(0, int(getattr(config, "CAPTURE_OCCLUSION_INSET_PX", 12))),
        )
        if occlusion.occluded:
            if (not self._capture_occluded) or int(occlusion.foreign_hwnd or 0) != int(
                self._last_occlusion_foreign_hwnd or 0
            ):
                self.logger.channel(
                    "capture",
                    message="TARGET_OCCLUDED",
                    hwnd=f"0x{int(hwnd):08X}",
                    foreign_hwnd=(f"0x{int(occlusion.foreign_hwnd):08X}" if occlusion.foreign_hwnd else None),
                    foreign_title=occlusion.foreign_title,
                    sampled_points=occlusion.sampled_points,
                    foreign_samples=occlusion.foreign_samples,
                )
                title = (occlusion.foreign_title or "another window").strip() or "another window"
                self._set_status(f"Target occluded by {title!r}. Capture paused until the window is clear.")
            self._capture_occluded = True
            self._last_occlusion_foreign_hwnd = int(occlusion.foreign_hwnd or 0)
            return True
        if self._capture_occluded:
            self._capture_occluded = False
            self._last_occlusion_foreign_hwnd = 0
            self._force_next_ocr_reason = "occlusion_cleared"
            self._last_capture_crc = ""
            self.logger.channel(
                "capture",
                message="TARGET_OCCLUSION_CLEARED",
                hwnd=f"0x{int(hwnd):08X}",
                sampled_points=occlusion.sampled_points,
            )
            self._set_status("Target visible again. Capture resumed.")
        return False

    def _maybe_periodic_full_rescan(self, now_ts: float, frame_index: int) -> None:
        """Force a full rescan at most once per FORCE_FULL_RESCAN_INTERVAL_MS.

        Resets the capture CRC and per-zone pixel signatures so the next
        change-gate evaluation sees the frame as "changed", and primes
        ``_force_next_ocr_reason`` so the OCR throttle is bypassed even when
        nothing visibly moved. Acts as a safety net against the gate
        suppressing translations on regions that genuinely never animate.
        """
        interval_ms = int(getattr(config, "FORCE_FULL_RESCAN_INTERVAL_MS", 1000))
        if interval_ms <= 0:
            return
        if self._last_full_rescan_ts <= 0.0:
            self._last_full_rescan_ts = now_ts
            return
        if (now_ts - self._last_full_rescan_ts) * 1000.0 < float(interval_ms):
            return
        self._last_full_rescan_ts = now_ts
        self._last_capture_crc = ""
        self._per_zone_pixel_signatures.clear()
        if not self._force_next_ocr_reason:
            self._force_next_ocr_reason = "periodic_full_rescan"
        if self.logger is not None:
            self.logger.channel(
                "capture",
                message="PERIODIC_FULL_RESCAN",
                frame_index=frame_index,
                interval_ms=interval_ms,
            )

    def _can_synthesize_observations(
        self,
        *,
        image_changed: bool,
        dialogue_zone_changed: bool,
        liveness_changed: bool,
        force_reason: str | None,
    ) -> bool:
        """True when we can skip OCR and reuse the cached observations.

        Safe to short-circuit only when nothing visibly moved (CRC matches,
        no dialogue zone change, no liveness change, no forced reason) AND
        we have a prior observation set to reuse AND no OCR is in flight
        or queued. OCR is deterministic on the same masked image, so a
        re-run on byte-identical input produces the same observation set
        — the savings (avoid a ~50 ms OCR call) translate directly into
        lower latency on tracks that are still ticking up stable_frames.
        """
        return (
            (not image_changed)
            and (not dialogue_zone_changed)
            and (not liveness_changed)
            and (not force_reason)
            and bool(self._last_observations)
            and not self._ocr_inflight_frame_index
            and self._frame_q.qsize() == 0
        )

    def _forced_ocr_reason(
        self, frame_index: int, *, image_changed: bool, dialogue_zone_changed: bool, liveness_changed: bool
    ) -> str | None:
        if self._force_next_ocr_reason:
            return str(self._force_next_ocr_reason)
        if self._ui_only_confirmation_budget > 0:
            interval = max(1, int(getattr(config, "OCR_UI_ONLY_CONFIRMATION_INTERVAL_FRAMES", 6)))
            if (frame_index - int(self._ui_only_confirmation_last_request_frame or 0)) >= interval:
                return "ui_only_confirmation"
        if bool(getattr(config, "OCR_FORCE_ON_CHANGED_RECENT_DIALOGUE", True)):
            if (image_changed or dialogue_zone_changed or liveness_changed) and self._has_recent_dialogue_context(
                frame_index
            ):
                return "changed_recent_dialogue"
        if bool(getattr(config, "OCR_TRANSLATION_DROUGHT_FORCE_RECHECK", True)) and self._has_recent_dialogue_context(
            frame_index
        ):
            drought_frames = max(1, int(getattr(config, "OCR_TRANSLATION_DROUGHT_FRAMES", 18)))
            interval = max(1, int(getattr(config, "OCR_TRANSLATION_DROUGHT_INTERVAL_FRAMES", 10)))
            no_items = not any((t.translation or "").strip() and t.render_enabled for t in self._tracks.values())
            no_candidate_tracks = not any(
                (t.dialogue_hint or t.name_hint or (not t.ui_hint and not t.low_value_hint and not t.is_static))
                and t.missing_frames == 0
                for t in self._tracks.values()
            )
            if (
                no_items
                and no_candidate_tracks
                and self._last_translation_activity_frame
                and (frame_index - int(self._last_translation_activity_frame)) >= drought_frames
                and (frame_index - int(self._translation_drought_recheck_last_frame or 0)) >= interval
            ):
                return "translation_drought"
        return None

    def _needs_ocr_on_unchanged_capture(self, frame_index: int | None = None) -> bool:
        frame_index = int(frame_index or self._frame_counter)
        stable_required = int(getattr(config, "TRACK_STABLE_FRAMES", 2))
        if not self._tracks:
            return not bool(self._ocr_inflight_frame_index) and self._frame_q.qsize() == 0
        for track in self._tracks.values():
            if not track.text.strip():
                continue
            if track.translation_pending:
                return True
            if track.is_static:
                continue
            if track.stable_frames < stable_required:
                return True
        return (
            self._forced_ocr_reason(
                frame_index, image_changed=False, dialogue_zone_changed=False, liveness_changed=False
            )
            is not None
        )

    def _age_tracks_without_ocr(self, frame_index: int, image: Image.Image, *, reason: str) -> None:
        if not bool(getattr(config, "TRACK_AGE_SUPPRESSED_ON_NON_OCR_FRAMES", True)):
            return
        dead_ids: list[int] = []
        aged = 0
        for track in self._tracks.values():
            should_age = bool(track.render_suppressed_until_match or track.missing_frames > 0)
            if not should_age:
                continue
            track.missing_frames += 1
            track.translation_pending = False if track.missing_frames > 1 else track.translation_pending
            aged += 1
            if image is not None and track.region_signature:
                try:
                    changed, _, current_sig, _ = self._region_visually_changed(image, track)
                    if not changed:
                        track.region_signature = current_sig
                except Exception:
                    pass
            forget_after = int(getattr(config, "TRACK_FORGET_FRAMES", 12))
            if track.dialogue_hint or track.name_hint:
                forget_after = min(forget_after, max(2, int(getattr(config, "OCR_DISAPPEAR_CONFIRM_FRAMES", 10))))
            if track.missing_frames > forget_after:
                dead_ids.append(track.track_id)
        for track_id in sorted(set(dead_ids)):
            doomed = self._tracks.get(track_id)
            if doomed is None:
                continue
            self._log_track_event("DELETE", frame_index, doomed, delete_reason=reason)
            del self._tracks[track_id]
        if aged or dead_ids:
            self.logger.channel(
                "tracks",
                message="NON_OCR_AGE",
                frame_index=frame_index,
                reason=reason,
                aged_track_count=aged,
                deleted_track_ids=dead_ids,
            )

    def _drain_queue(self, q: queue.Queue):
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                return

    def _ocr_worker(self):
        while not self._stop.is_set():
            try:
                packet = self._frame_q.get(timeout=0.2)
            except queue.Empty:
                continue
            started = perf_counter()
            self._ocr_inflight_frame_index = packet.frame_index
            try:
                layout = self._ocr.recognize_layout(packet.image)
                elapsed_ms = round((perf_counter() - started) * 1000.0, 2)
                self._log_timing("ocr", elapsed_ms, frame_index=packet.frame_index, line_count=len(layout.lines))
                self.logger.channel(
                    "ocr",
                    message="WORKER_DONE",
                    frame_index=packet.frame_index,
                    duration_ms=elapsed_ms,
                    line_count=len(layout.lines),
                )
                self._ocrFrameReady.emit(OcrFrame(packet.frame_index, packet.image, layout, packet.generation))
            except Exception as e:
                self.logger.log(f"OCR_ERROR: {e!r}")
                self.logger.channel("ocr", message="ERROR", frame_index=packet.frame_index, error=repr(e))
            finally:
                if self._ocr_inflight_frame_index == packet.frame_index:
                    self._ocr_inflight_frame_index = 0

    def _translate_worker(self):
        """Background thread that drains _tr_q and runs translations.

        For each task:
          1. Compute queue-wait time + save the source-text snapshot.
          2. Log REQUEST on the translate channel.
          3. Call translate_llama_server; on success log RESPONSE and
             emit the result. On exception log ERROR and emit an empty
             translation (preserving the same shape so the handler can
             still discard pending state).

        _tr_q is typed (priority, seq, TranslationTask); the task is the
        last element of the tuple. The per-task call is wrapped in a
        broad except so a failure in the surrounding bookkeeping (disk
        full during _save_text_snapshot, a logger channel write that
        raises, etc.) cannot kill the worker thread and stall the
        translate queue.
        """
        while not self._stop.is_set():
            try:
                queued = self._tr_q.get(timeout=0.2)
            except queue.Empty:
                continue
            task = queued[-1]
            try:
                self._run_translation_task(task)
            except Exception as e:
                self.logger.log(f"TRANSLATE_WORKER_ERROR: {e!r}")
                self.logger.channel(
                    "translate",
                    message="WORKER_ERROR",
                    frame_index=task.frame_index,
                    track_id=task.track_id,
                    source_version=task.source_version,
                    error=repr(e),
                )
                # Make sure the handler unblocks the track even if the
                # task aborted before its own emit could fire.
                try:
                    self._emit_translation_result(task, "")
                except Exception as emit_err:
                    self.logger.log(f"TRANSLATE_EMIT_RECOVERY_FAILED: {emit_err!r}")

    def _run_translation_task(self, task: TranslationTask) -> None:
        started = perf_counter()
        queue_wait_ms = round(max(0.0, (started - float(task.queued_at or 0.0)) * 1000.0), 2) if task.queued_at else 0.0
        priority = int(task.priority)
        source_file = self._save_text_snapshot(
            "translate_source",
            f"{self._frame_ref(task.frame_index)}_{self._track_ref(task.track_id)}_v{task.source_version}_source",
            task.source_text,
            frame_index=task.frame_index,
        )
        self.logger.channel(
            "translate",
            message="REQUEST",
            frame_index=task.frame_index,
            track_id=task.track_id,
            source_version=task.source_version,
            priority=priority,
            queue_wait_ms=queue_wait_ms,
            source_hint=LANG_HINT.get(self.lang_tag),
            source_text=self._preview_text(task.source_text),
            source_file=source_file,
        )
        # Snapshot context BEFORE calling the LLM so a concurrent
        # _apply_new_translation append can't mutate what we send. deque
        # __iter__ is atomic under CPython's GIL; list() copies out a
        # consistent view.
        history_snapshot: list[tuple[str, str]] = (
            list(self._translation_history) if self._translation_history_max > 0 else []
        )
        context_kind = "ui" if task.ui_hint else ("name" if task.name_hint else "")
        try:
            translated = translate_llama_server(
                task.source_text,
                source_hint=LANG_HINT.get(self.lang_tag),
                base_url=config.LLAMA_SERVER_BASE_URL,
                model=getattr(config, "LLAMA_SERVER_MODEL", "local-model"),
                timeout_s=getattr(config, "LLAMA_SERVER_TIMEOUT_S", 60),
                max_tokens=getattr(config, "LLAMA_SERVER_MAX_TOKENS", 512),
                history=history_snapshot,
                context_kind=context_kind,
            )
            translated = _cleanup_translation(translated)
            elapsed_ms = round((perf_counter() - started) * 1000.0, 2)
            self._log_timing("translate", elapsed_ms, frame_index=task.frame_index, track_id=task.track_id)
            translated_file = self._save_text_snapshot(
                "translate_result",
                f"{self._frame_ref(task.frame_index)}_{self._track_ref(task.track_id)}_v{task.source_version}_result",
                translated,
                frame_index=task.frame_index,
            )
            self.logger.channel(
                "translate",
                message="RESPONSE",
                frame_index=task.frame_index,
                track_id=task.track_id,
                source_version=task.source_version,
                duration_ms=elapsed_ms,
                priority=priority,
                queue_wait_ms=queue_wait_ms,
                translated_text=self._preview_text(translated),
                translated_file=translated_file,
            )
            self._emit_translation_result(task, translated)
        except Exception as e:
            self.logger.log(f"TRANSLATE_ERROR: {e!r}")
            self.logger.channel(
                "translate",
                message="ERROR",
                frame_index=task.frame_index,
                track_id=task.track_id,
                source_version=task.source_version,
                priority=priority,
                error=repr(e),
            )
            self._emit_translation_result(task, "")

    def _emit_translation_result(self, task: TranslationTask, translated: str) -> None:
        """Emit the TranslationResult consumed by _handle_translation_result.

        Both success and error paths go through here. The dataclass shape
        guarantees that every emitted result includes ``generation`` so the
        stale-drop check in the handler always runs. Before the
        ``TranslationResult`` extraction, this was a positional 7-tuple
        and the error path silently dropped the field (8ade7ab).
        """
        self._translationReady.emit(
            TranslationResult(
                frame_index=task.frame_index,
                track_id=task.track_id,
                source_version=task.source_version,
                source_text=task.source_text,
                cache_key=task.cache_key,
                translated=translated,
                generation=task.generation,
            )
        )

    def _small_text_rescan_score(self, normalized: str) -> int:
        compact = re.sub(r"\s+", "", normalized or "")
        if not compact:
            return 0
        cjk = len(_CJK_RE.findall(compact))
        latin = len(_LATIN_RE.findall(compact))
        digits = len(re.findall(r"\d", compact))
        punct = len(re.findall(r"[^\w\s]", compact))
        return (3 * cjk) + latin + digits - punct

    def _should_rescan_small_text(self, obs: Observation, image: Image.Image) -> bool:
        if obs.dialogue_hint or obs.name_hint:
            return False
        edge_margin = int(getattr(config, "SMALL_TEXT_EDGE_MARGIN", 36))
        near_edge = obs.rect.left <= edge_margin or obs.rect.right >= (image.width - edge_margin)
        short_sign = obs.rect.height <= int(getattr(config, "SMALL_TEXT_RESCAN_MAX_HEIGHT", 50))
        low_conf = obs.avg_confidence < float(getattr(config, "SMALL_TEXT_RESCAN_MAX_CONFIDENCE", 0.78))
        return (short_sign or near_edge) and low_conf

    def _refine_small_text_observations(
        self, frame_index: int, image: Image.Image, observations: list[Observation]
    ) -> int:
        if not bool(getattr(config, "SMALL_TEXT_RESCAN_ENABLED", True)):
            return 0
        max_per_frame = int(getattr(config, "SMALL_TEXT_RESCAN_MAX_PER_FRAME", 2))
        target_height = int(getattr(config, "SMALL_TEXT_TARGET_HEIGHT", 80))
        pad_x = int(getattr(config, "SMALL_TEXT_PADDING_X", 16))
        pad_y = int(getattr(config, "SMALL_TEXT_PADDING_Y", 12))
        max_total_ms = float(getattr(config, "SMALL_TEXT_RESCAN_MAX_TOTAL_MS", 120.0))
        applied = 0
        started = perf_counter()
        for obs in observations:
            if applied >= max_per_frame:
                break
            if ((perf_counter() - started) * 1000.0) >= max_total_ms:
                break
            if not self._should_rescan_small_text(obs, image):
                continue
            crop_rect = _expand_rect(obs.rect, image.width, image.height, pad_x, pad_y)
            crop = _crop(image, crop_rect).convert("RGB")
            scale = max(1.0, float(target_height) / max(1.0, float(crop.height)))
            if scale > 1.01:
                crop = crop.resize(
                    (max(1, int(round(crop.width * scale))), max(1, int(round(crop.height * scale)))),
                    Image.Resampling.LANCZOS,
                )
            rescanned = (self._ocr.recognize(crop) or "").strip()
            rescanned_norm = _normalize_ocr_text(rescanned)
            if (
                not rescanned_norm
                or rescanned_norm == obs.normalized
                or not _text_quality_ok(rescanned_norm, avg_confidence=obs.avg_confidence)
            ):
                continue
            if _looks_like_short_mixed_junk(rescanned_norm, obs.avg_confidence) or _looks_like_upper_suffix_junk(
                rescanned_norm, obs.avg_confidence
            ):
                continue
            old_score = self._small_text_rescan_score(obs.normalized)
            new_score = self._small_text_rescan_score(rescanned_norm)
            if new_score < old_score + int(
                getattr(config, "SMALL_TEXT_RESCAN_MIN_SCORE_GAIN", 2)
            ) and not _looks_like_short_mixed_junk(obs.normalized, obs.avg_confidence):
                continue
            self.logger.channel(
                "ocr",
                message="SMALL_TEXT_RESCAN_APPLIED",
                frame_index=frame_index,
                old_text=self._preview_text(obs.text),
                new_text=self._preview_text(rescanned),
                rect=self._rect_to_dict(obs.rect),
                original_confidence=round(float(obs.avg_confidence), 3),
                crop_rect=self._rect_to_dict(crop_rect),
                upscale_ratio=round(float(scale), 3),
            )
            obs.text = rescanned
            obs.normalized = rescanned_norm
            applied += 1
        self._log_timing(
            "small_text_rescan", (perf_counter() - started) * 1000.0, frame_index=frame_index, rescans_applied=applied
        )
        return applied

    def _ocr_result_drop_reason(self, result: OcrFrame) -> str | None:
        """Return the channel message name for an OCR-result drop, or None
        to accept the result. Drop cases: stale generation (pipeline reset),
        region edit in progress, out-of-order result superseded by a newer
        frame. The fourth case (LATE_RESULT_APPLYING) is a log breadcrumb
        only — it doesn't drop — and is handled inline in _handle_ocr_frame.
        """
        if int(result.generation or 0) != int(self._pipeline_generation):
            self.logger.channel(
                "ocr",
                message="DROP_STALE_GENERATION",
                frame_index=result.frame_index,
                result_generation=int(result.generation or 0),
                current_generation=int(self._pipeline_generation),
            )
            return "DROP_STALE_GENERATION"
        if self._region_edit_active:
            self.logger.channel(
                "ocr",
                message="DROP_REGION_EDIT_ACTIVE",
                frame_index=result.frame_index,
                generation=int(result.generation or 0),
            )
            return "DROP_REGION_EDIT_ACTIVE"
        if result.frame_index < self._latest_applied_ocr_frame_index:
            self.logger.channel(
                "ocr",
                message="DROP_OUT_OF_ORDER_RESULT",
                frame_index=result.frame_index,
                latest_applied_frame=self._latest_applied_ocr_frame_index,
            )
            return "DROP_OUT_OF_ORDER_RESULT"
        return None

    def _process_ocr_observations(self, result: OcrFrame, base_image: Image.Image) -> tuple[list[Observation], int]:
        """Run OCR layout -> grouped+filtered Observations and the
        repeated-text consensus pass. Returns the final observation list
        plus the count of observations repaired by consensus (used by
        REPEAT_LABEL_CONSENSUS_APPLIED log).
        """
        started = perf_counter()
        zone_rects = [z.rect for z in self._translation_zones] if self._translation_zones else None
        observations = _group_lines(
            result.frame_index, result.layout, base_image, zone_rects=zone_rects, logger=self.logger
        )
        observations = [obs for obs in observations if self._observation_allowed(obs, base_image)]
        observations = self._apply_grouped_translation_zones(result.frame_index, base_image, observations)
        observations = [obs for obs in observations if self._observation_allowed(obs, base_image)]
        self._log_timing(
            "group_lines",
            (perf_counter() - started) * 1000.0,
            frame_index=result.frame_index,
            observation_count=len(observations),
        )
        self._refine_small_text_observations(result.frame_index, base_image, observations)
        repaired_count = _apply_repeated_text_consensus(
            result.frame_index, base_image, observations, logger=self.logger
        )
        _refresh_observation_hints(observations, base_image)
        return observations, repaired_count

    def _update_ui_only_confirmation_budget(self, frame_index: int, observations: list[Observation]) -> None:
        """Arm or clear the ui-only-confirmation budget based on whether
        this frame's observations are exclusively UI/HUD/low-value (and
        we recently had real dialogue context). When armed, the capture
        loop will force an extra OCR pass every
        OCR_UI_ONLY_CONFIRMATION_INTERVAL_FRAMES frames until the budget
        depletes — catching the case where dialogue text briefly drops
        below confidence threshold.
        """
        candidate_count = sum(1 for obs in observations if _obs_is_translation_candidate(obs))
        ui_only_scene = (candidate_count == 0) and (
            not observations or all(obs.ui_hint or obs.low_value_hint or obs.hud_hint for obs in observations)
        )
        if ui_only_scene and self._has_recent_dialogue_context(frame_index):
            confirm_budget = max(0, int(getattr(config, "OCR_UI_ONLY_CONFIRMATION_SCANS", 2)))
            if confirm_budget > 0:
                self._ui_only_confirmation_budget = max(int(self._ui_only_confirmation_budget or 0), confirm_budget)
                self.logger.channel(
                    "ocr",
                    message="UI_ONLY_CONFIRMATION_ARMED",
                    frame_index=frame_index,
                    observation_count=len(observations),
                    budget=self._ui_only_confirmation_budget,
                )
        else:
            self._ui_only_confirmation_budget = 0

    def _handle_ocr_frame(self, result: OcrFrame):
        """Apply one OCR result: drop checks → observation processing →
        track update → scene emit. Each phase is delegated to a focused
        helper; this orchestrator just sequences them.
        """
        if self._ocr_result_drop_reason(result) is not None:
            return
        if result.frame_index < self._latest_requested_ocr_frame_index:
            self.logger.channel(
                "ocr",
                message="LATE_RESULT_APPLYING",
                frame_index=result.frame_index,
                latest_requested_frame=self._latest_requested_ocr_frame_index,
                latest_applied_frame=self._latest_applied_ocr_frame_index,
            )
        self._latest_applied_ocr_frame_index = max(self._latest_applied_ocr_frame_index, result.frame_index)
        base_image = self._latest_image if self._latest_image is not None else result.image
        self._update_per_zone_change_state(base_image, result.frame_index)
        observations, repaired_count = self._process_ocr_observations(result, base_image)
        if any(obs.dialogue_hint or obs.name_hint for obs in observations):
            self._last_dialogue_seen_frame = max(int(self._last_dialogue_seen_frame or 0), int(result.frame_index))
        self._update_ui_only_confirmation_budget(result.frame_index, observations)
        self._last_observations = list(observations)
        started = perf_counter()
        self._update_tracks(result.frame_index, observations)
        self._log_timing(
            "update_tracks",
            (perf_counter() - started) * 1000.0,
            frame_index=result.frame_index,
            observation_count=len(observations),
            track_count=len(self._tracks),
        )
        started = perf_counter()
        self._log_ocr_frame(result.frame_index, result.layout, observations, base_image)
        if repaired_count:
            self.logger.channel(
                "ocr",
                message="REPEAT_LABEL_CONSENSUS_APPLIED",
                frame_index=result.frame_index,
                repaired_count=repaired_count,
            )
        self._log_timing("log_ocr", (perf_counter() - started) * 1000.0, frame_index=result.frame_index)
        started = perf_counter()
        self._emit_scene(self._last_scene_frame_index, reason="ocr_result")
        self._consume_dialogue_transition(result.frame_index)
        self._log_timing("emit_scene", (perf_counter() - started) * 1000.0, frame_index=result.frame_index)

    def _consume_dialogue_transition(self, frame_index: int) -> None:
        """Log DIALOGUE_TRANSITION_OCR_APPLIED and clear the active flag
        in one place. Called once per OCR-result handler after the scene
        has been emitted (so the breadcrumb is co-located with the
        flag-clear, and a future change to the predicate can't drift
        between two sites).
        """
        if not (self._dialogue_transition_active and frame_index >= int(self._dialogue_transition_frame_index or 0)):
            return
        self.logger.channel(
            "tracks",
            message="DIALOGUE_TRANSITION_OCR_APPLIED",
            frame_index=frame_index,
            generation=self._dialogue_generation,
        )
        self._dialogue_transition_active = False

    def _match_score(self, track: Track, obs: Observation) -> float:
        text_ratio = (
            _fuzz_ratio(track.normalized, obs.normalized) / 100.0 if track.normalized and obs.normalized else 0.0
        )
        iou = _rect_iou(track.rect, obs.rect)
        dist = _rect_center_distance(track.rect, obs.rect)
        distance_factor = max(0.0, 1.0 - (dist / max(1.0, float(config.MATCH_DISTANCE_PX))))
        dialogue_conflict = track.dialogue_hint != obs.dialogue_hint and (track.dialogue_hint or obs.dialogue_hint)
        ui_conflict = track.ui_hint != obs.ui_hint and (track.ui_hint or obs.ui_hint)
        if dialogue_conflict and text_ratio < float(getattr(config, "MATCH_DIALOGUE_CLASS_MIN_TEXT_RATIO", 0.93)):
            return 0.0
        if ui_conflict and text_ratio < float(getattr(config, "MATCH_UI_CLASS_MIN_TEXT_RATIO", 0.90)):
            return 0.0
        if iou < config.MATCH_IOU_THRESHOLD and distance_factor <= 0.0 and text_ratio < 0.70:
            return 0.0
        text_reuse_max_distance = max(
            float(getattr(config, "MATCH_TEXT_REUSE_MAX_DISTANCE_PX", 84)),
            float(max(track.rect.width, track.rect.height, obs.rect.width, obs.rect.height, 1))
            * float(getattr(config, "MATCH_TEXT_REUSE_MAX_DISTANCE_RATIO", 0.85)),
        )
        text_reuse_iou_floor = float(getattr(config, "MATCH_TEXT_REUSE_IOU_FLOOR", 0.08))
        if text_ratio >= 0.97 and iou < text_reuse_iou_floor and dist > text_reuse_max_distance:
            return 0.0
        score = (0.52 * text_ratio) + (0.30 * iou) + (0.18 * distance_factor)
        if dialogue_conflict:
            score *= float(getattr(config, "MATCH_DIALOGUE_CLASS_PENALTY", 0.30))
        elif ui_conflict:
            score *= float(getattr(config, "MATCH_UI_CLASS_PENALTY", 0.55))
        if text_ratio >= 0.94 and iou < max(text_reuse_iou_floor, 0.12) and dist > (text_reuse_max_distance * 0.8):
            score *= float(getattr(config, "MATCH_TEXT_REUSE_PENALTY", 0.42))
        if text_ratio >= 0.985 and dist <= text_reuse_max_distance:
            score = max(score, 0.88 if iou < 0.18 else 0.92)
        elif text_ratio >= 0.95 and dist <= min(float(config.MATCH_DISTANCE_PX) * 1.2, text_reuse_max_distance):
            score = max(score, 0.72)
        return score

    def _create_track(self, frame_index: int, obs: Observation) -> Track:
        image = self._latest_image
        immediate_static = False
        track = Track(
            track_id=self._next_track_id,
            text=obs.text,
            normalized=obs.normalized,
            rect=obs.rect,
            word_rects=list(obs.word_rects),
            style=obs.style,
            line_count=obs.line_count,
            hud_hint=obs.hud_hint,
            low_value_hint=obs.low_value_hint,
            is_static=immediate_static,
            region_signature=(b"" if image is None else _region_signature(image, obs.rect, obs.word_rects)),
            last_matched_frame=frame_index,
            dialogue_generation=self._current_dialogue_generation_for_obs(obs),
            dialogue_hint=obs.dialogue_hint,
            ui_hint=obs.ui_hint,
            name_hint=obs.name_hint,
            repeated_label_hint=obs.repeated_label_hint,
            repeat_cluster_id=obs.repeat_cluster_id,
            repeated_consensus_text=obs.repeated_consensus_text,
            repaired_from_cluster=obs.repaired_from_cluster,
            render_anchor_left=obs.render_anchor_left,
            estimated_full_width=obs.estimated_full_width,
            median_char_height=obs.median_char_height,
            writing_mode=obs.writing_mode,
            created_frame_index=frame_index,
            created_from_obs_id=obs.obs_id,
            last_obs_id=obs.obs_id,
            last_obs_frame_index=obs.frame_index,
            member_source_rects=list(obs.member_source_rects),
        )
        self._next_track_id += 1
        self._tracks[track.track_id] = track
        self._log_track_event("CREATE", frame_index, track)
        self._apply_region_memo(frame_index, track)
        self._maybe_request_translation(frame_index, track)
        return track

    def _copy_observation_to_track(
        self, frame_index: int, track: Track, obs: Observation, *, freeze_geometry: bool = False
    ) -> None:
        """Copy ``obs`` field-by-field onto ``track`` and reset the per-
        match counters. Pure mechanical assignment — the stability
        decision must already have been applied before this runs so the
        prev_* values captured at the top of _update_track still reflect
        the previous track state.

        ``freeze_geometry`` preserves the previous render geometry
        (``rect``, ``word_rects``, ``median_char_height``) even though a
        fresh observation is being merged in. Callers set this when the
        text is effectively unchanged and the OCR bbox only drifted by a
        few pixels — the "sameish" case in ``_compute_stability_update``.
        Without freezing, the rendered overlay would visibly jitter each
        frame as OCR bboxes wobble, and the font size would tick up and
        down as the median row height rocked between neighbouring pixel
        values.
        """
        track.text = obs.text
        track.normalized = obs.normalized
        if not freeze_geometry:
            track.rect = obs.rect
            track.word_rects = list(obs.word_rects)
            track.median_char_height = obs.median_char_height
            track.member_source_rects = list(obs.member_source_rects)
        track.style = obs.style
        track.line_count = obs.line_count
        track.hud_hint = obs.hud_hint
        track.low_value_hint = obs.low_value_hint
        track.dialogue_hint = obs.dialogue_hint
        track.ui_hint = obs.ui_hint
        track.name_hint = obs.name_hint
        track.dialogue_generation = self._current_dialogue_generation_for_obs(obs)
        track.repeated_label_hint = obs.repeated_label_hint
        track.repeat_cluster_id = obs.repeat_cluster_id
        track.repeated_consensus_text = obs.repeated_consensus_text
        track.repaired_from_cluster = obs.repaired_from_cluster
        track.render_anchor_left = obs.render_anchor_left
        track.estimated_full_width = obs.estimated_full_width
        track.writing_mode = obs.writing_mode
        track.last_obs_id = obs.obs_id
        track.last_obs_frame_index = obs.frame_index
        track.missing_frames = 0
        track.held_frames = 0
        track.change_frames = 0
        track.render_suppressed_until_match = False
        track.last_matched_frame = frame_index

    def _update_track(self, frame_index: int, track: Track, obs: Observation):
        image = self._latest_image
        prev_text = track.text
        prev_norm = track.normalized
        prev_rect = track.rect
        prev_static = track.is_static
        was_suppressed = bool(track.render_suppressed_until_match)
        was_liveness_active = was_suppressed and int(track.liveness_change_frames or 0) > 0
        text_ratio = (
            _fuzz_ratio(track.normalized, obs.normalized) / 100.0 if track.normalized and obs.normalized else 0.0
        )
        iou = _rect_iou(track.rect, obs.rect)
        class_changed = ((track.dialogue_hint != obs.dialogue_hint) and (track.dialogue_hint or obs.dialogue_hint)) or (
            (track.ui_hint != obs.ui_hint) and (track.ui_hint or obs.ui_hint)
        )
        # Same "changed" predicate _compute_stability_update uses internally;
        # named constants keep the two call sites from drifting on a tune.
        changed = text_ratio < CHANGED_TEXT_RATIO_FLOOR or iou < CHANGED_IOU_FLOOR or class_changed
        zone_changed = self._track_zone_changed(track)
        decision = _compute_stability_update(
            text_ratio=text_ratio,
            iou=iou,
            class_changed=class_changed,
            track_normalized=track.normalized,
            obs_normalized=obs.normalized,
            track_stable_frames=track.stable_frames,
            track_unstable_frames=track.unstable_frames,
            zone_changed=zone_changed,
        )
        sameish = decision.stable_frames == track.stable_frames + 1
        track.stable_frames = decision.stable_frames
        track.unstable_frames = decision.unstable_frames
        # Text-growth (type-on reveal) detector. If the new compact text strictly
        # extends the prior compact text and grew by at least the configured
        # ratio, count it as a reveal frame so _maybe_request_translation can
        # hold off until the reveal stops.
        prev_compact = _compact_text(track.normalized)
        new_compact = _compact_text(obs.normalized)
        min_growth_ratio = float(getattr(config, "REVEAL_HOLD_MIN_GROWTH_RATIO", 1.02))
        is_growth = (
            bool(prev_compact)
            and len(new_compact) > len(prev_compact)
            and new_compact.startswith(prev_compact)
            and len(new_compact) >= len(prev_compact) * min_growth_ratio
        )
        if is_growth:
            track.text_growth_streak += 1
        else:
            track.text_growth_streak = 0
        if sameish:
            track.unchanged_frames += 1
        elif decision.reset_unchanged_frames:
            track.unchanged_frames = 0
        if decision.bump_source_version:
            track.source_version += 1
        if decision.clear_translation:
            track.translation = ""
            track.translation_pending = False
            self._set_track_render_enabled(track, False)
        if decision.suppressed_by_zone_jitter:
            self.logger.channel(
                "tracks",
                message="ZONE_JITTER_SUPPRESSED",
                frame_index=frame_index,
                track_id=track.track_id,
                prev_text=self._preview_text(track.text),
                obs_text=self._preview_text(obs.text),
            )
        # Freeze render geometry when the text is effectively unchanged
        # AND the OCR bbox only drifted slightly (same predicate the
        # stability logic already uses for the "sameish" branch). This
        # stops the overlay from wobbling and the font from ticking every
        # frame while the source content is standing still.
        freeze_geometry = (
            not class_changed
            and text_ratio >= SAMEISH_TEXT_RATIO_MIN
            and iou >= SAMEISH_IOU_MIN
        )
        self._copy_observation_to_track(frame_index, track, obs, freeze_geometry=freeze_geometry)
        if image is not None:
            track.region_signature = _region_signature(image, track.rect, track.word_rects)
        # If this track was suppressed (stale content hidden) and the text changed, the old
        # translation must not flash in before a fresh one arrives. Clear it so the
        # re-enable below and in _maybe_request_translation find no stale content.
        if was_suppressed and prev_norm != track.normalized and (track.translation or "").strip():
            track.translation = ""
            track.translation_pending = False
        if track.translation.strip() and not track.is_static:
            self._set_track_render_enabled(track, True)
        # If liveness was actively detecting pixel changes on this track when this OCR
        # result arrived, the OCR image is from a stale frame — don't let it override
        # the liveness suppression. Keep suppressed until the next OCR cycle, which will
        # see current pixels and make the correct call. Only applies when OCR reports
        # the text as unchanged (changed=True already cleared translation above).
        if was_liveness_active and not changed and (track.translation or "").strip():
            track.render_suppressed_until_match = True
            self._set_track_render_enabled(track, False)
        track.liveness_change_frames = 0
        track.liveness_hold_frames = 0
        if track.low_value_hint and track.unchanged_frames >= int(
            getattr(config, "LOW_VALUE_STATIC_FRAME_THRESHOLD", 2)
        ):
            track.is_static = True
            track.translation = ""
            track.translation_pending = False
            self._set_track_render_enabled(track, False)
        elif changed:
            if image is None or not _obs_is_bottom_hud(obs, image):
                track.is_static = False
        if prev_norm != track.normalized:
            self._log_track_event(
                "SOURCE_CHANGED",
                frame_index,
                track,
                previous_text=self._preview_text(prev_text),
                previous_rect=self._rect_to_dict(prev_rect),
            )
        elif prev_rect != track.rect or prev_static != track.is_static:
            self._log_track_event("UPDATE", frame_index, track, previous_rect=self._rect_to_dict(prev_rect))
        if track.is_static and not prev_static:
            self._log_track_event("PROMOTED_STATIC", frame_index, track)
        self._apply_region_memo(frame_index, track)
        self._maybe_request_translation(frame_index, track)

    def _region_memo_key(self, rect: Rect, region_signature: bytes) -> str:
        quantum = max(1, int(getattr(config, "REGION_MEMO_RECT_QUANTUM", 12)))
        return f"{int(rect.left / quantum)}:{int(rect.top / quantum)}:{int(rect.width / quantum)}:{int(rect.height / quantum)}:{region_signature.hex()}"

    def _remember_region_memo(self, frame_index: int, track: Track) -> None:
        if not bool(getattr(config, "REGION_MEMO_ENABLED", True)):
            return
        if track.is_static or track.low_value_hint:
            return
        if track.ui_hint and not bool(getattr(config, "REGION_MEMO_ALLOW_UI", False)):
            return
        if not track.region_signature or not (track.translation or "").strip():
            return
        key = self._region_memo_key(track.rect, track.region_signature)
        self._region_memo[key] = RegionMemo(
            source_text=track.text,
            normalized=track.normalized,
            translation=track.translation,
            line_count=track.line_count,
            dialogue_hint=track.dialogue_hint,
            ui_hint=track.ui_hint,
            style=track.style,
            frame_index=frame_index,
        )
        self._region_memo.move_to_end(key)
        max_entries = max(16, int(getattr(config, "REGION_MEMO_MAX_ENTRIES", 512)))
        while len(self._region_memo) > max_entries:
            self._region_memo.popitem(last=False)

    def _apply_region_memo(self, frame_index: int, track: Track) -> bool:
        if not bool(getattr(config, "REGION_MEMO_ENABLED", True)):
            return False
        if track.is_static or track.low_value_hint or not track.region_signature:
            return False
        if self._dialogue_cache_quarantine(track):
            self.logger.channel(
                "translate",
                message="REGION_MEMO_QUARANTINED",
                frame_index=frame_index,
                track_id=track.track_id,
                source_version=track.source_version,
                text=self._preview_text(track.text),
            )
            return False
        if track.ui_hint and not bool(getattr(config, "REGION_MEMO_ALLOW_UI", False)):
            return False
        key = self._region_memo_key(track.rect, track.region_signature)
        memo = self._region_memo.get(key)
        if memo is None or not (memo.translation or "").strip():
            return False
        similarity = (
            _fuzz_ratio(memo.normalized or "", track.normalized or "") / 100.0
            if (memo.normalized or track.normalized)
            else 1.0
        )
        min_similarity = float(getattr(config, "REGION_MEMO_MIN_TEXT_SIMILARITY", 0.55))
        if track.normalized and memo.normalized and similarity < min_similarity:
            return False
        if memo.ui_hint != track.ui_hint and (memo.ui_hint or track.ui_hint):
            return False
        if (
            bool(getattr(config, "REGION_MEMO_REQUIRE_DIALOGUE_CLASS_MATCH", True))
            and (memo.dialogue_hint != track.dialogue_hint)
            and (memo.dialogue_hint or track.dialogue_hint)
        ):
            return False
        track.translation = memo.translation
        track.translation_pending = False
        track.render_suppressed_until_match = False
        self._set_track_render_enabled(track, True)
        self._translation_cache.setdefault(track.normalized, memo.translation)
        self._last_translation_activity_frame = max(int(self._last_translation_activity_frame or 0), int(frame_index))
        self.logger.channel(
            "translate",
            message="REGION_MEMO_HIT",
            frame_index=frame_index,
            track_id=track.track_id,
            source_version=track.source_version,
            similarity=round(similarity, 3),
            memo_frame_index=memo.frame_index,
            text=self._preview_text(track.text),
            translated_text=self._preview_text(memo.translation),
        )
        return True

    def _translation_class_settings(self, track: Track) -> tuple[int, int]:
        """Per-track-class (stable_required, priority) for the translation
        queue. Dialogue / name / sign fast paths shorten stability and
        bump priority; low-line-count fallbacks get default stability and
        the low-priority queue slot.
        """
        stable_required = int(config.TRACK_STABLE_FRAMES)
        priority = int(getattr(config, "TRANSLATION_NORMAL_PRIORITY", 50))
        if bool(getattr(config, "DIALOGUE_FAST_PATH_ENABLED", True)) and track.dialogue_hint:
            stable_required = int(getattr(config, "TRACK_STABLE_FRAMES_DIALOGUE", 2))
            priority = int(getattr(config, "TRANSLATION_DIALOGUE_PRIORITY", 0))
        elif track.name_hint:
            stable_required = int(getattr(config, "TRACK_STABLE_FRAMES_NAME", 2))
            priority = int(getattr(config, "TRANSLATION_NAME_PRIORITY", 10))
        elif (
            bool(getattr(config, "SIGN_FAST_PATH_ENABLED", True))
            and (not track.ui_hint)
            and (not track.low_value_hint)
            and (not track.dialogue_hint)
            and track.line_count <= 2
        ):
            stable_required = int(getattr(config, "TRACK_STABLE_FRAMES_SIGN", 2))
            priority = int(getattr(config, "TRANSLATION_SIGN_PRIORITY", 20))
        elif track.line_count <= 1:
            priority = int(getattr(config, "TRANSLATION_LOW_PRIORITY", 100))
        return stable_required, priority

    def _handle_reveal_or_unstable_hold(self, frame_index: int, track: Track, stable_required: int) -> bool:
        """Apply the reveal-hold and unstable-hold gates.

        Returns True if the gate fired (caller should return without
        queuing). Reveal hold: text growing as a typewriter reveal — wait
        for it to stop. Unstable hold: track failed to settle for
        TEXT_UNSTABLE_HOLD_FRAMES frames AND a translation exists or is
        pending — wipe the stale render rather than holding it visible
        while the text churns.
        """
        reveal_streak_required = int(getattr(config, "REVEAL_HOLD_STREAK_FRAMES", 3))
        if track.text_growth_streak >= reveal_streak_required:
            self.logger.channel(
                "translate",
                message="QUEUE_HOLD_REVEALING",
                frame_index=frame_index,
                track_id=track.track_id,
                source_version=track.source_version,
                text_growth_streak=track.text_growth_streak,
                reveal_streak_required=reveal_streak_required,
                text=self._preview_text(track.text),
            )
            return True
        unstable_hold = int(getattr(config, "TEXT_UNSTABLE_HOLD_FRAMES", 4))
        if (
            track.stable_frames < stable_required
            and track.unstable_frames >= unstable_hold
            and ((track.translation or "").strip() or track.translation_pending)
        ):
            track.translation = ""
            track.translation_pending = False
            self._set_track_render_enabled(track, False)
            self.logger.channel(
                "translate",
                message="QUEUE_HOLD_UNSTABLE",
                frame_index=frame_index,
                track_id=track.track_id,
                source_version=track.source_version,
                stable_frames=track.stable_frames,
                stable_required=stable_required,
                unstable_frames=track.unstable_frames,
                unstable_hold=unstable_hold,
                text=self._preview_text(track.text),
            )
            return True
        return False

    def _translation_skip_reason(self, track: Track, stable_required: int) -> str | None:
        """Returns the QUEUE_SKIP reason for static / low_value / not_stable
        tracks (or None if none apply). The ui_text policy gate is checked
        separately after the cache lookup so a cached UI translation can
        still apply.
        """
        if track.is_static:
            return "static"
        if track.low_value_hint and bool(getattr(config, "TRANSLATION_SKIP_LOW_VALUE", False)):
            return "low_value"
        if track.stable_frames < stable_required:
            return "not_stable"
        return None

    def _apply_cached_translation(self, frame_index: int, track: Track, cache_key: str) -> bool:
        """If a cached translation exists for ``cache_key``, apply it and
        return True. Honours the dialogue-cache quarantine. Returns False
        for cache miss; the caller continues to the fresh-translation
        path.
        """
        cached = self._translation_cache.get(cache_key)
        if cached is not None and self._dialogue_cache_quarantine(track):
            self.logger.channel(
                "translate",
                message="CACHE_HIT_QUARANTINED",
                frame_index=frame_index,
                track_id=track.track_id,
                source_version=track.source_version,
                text=self._preview_text(track.text),
                bucket=self._translation_memory_name,
            )
            cached = None
        if cached is None:
            return False
        track.translation = cached
        track.translation_pending = False
        if not track.render_suppressed_until_match:
            self._set_track_render_enabled(track, True)
        self._last_translation_activity_frame = max(int(self._last_translation_activity_frame or 0), int(frame_index))
        self.logger.channel(
            "translate",
            message="CACHE_HIT",
            frame_index=frame_index,
            track_id=track.track_id,
            source_version=track.source_version,
            text=self._preview_text(track.text),
            translated_text=self._preview_text(cached),
            bucket=self._translation_memory_name,
        )
        self._remember_region_memo(frame_index, track)
        return True

    def _enqueue_translation(
        self,
        frame_index: int,
        track: Track,
        *,
        cache_key: str,
        stable_required: int,
        priority: int,
    ) -> None:
        """Final stage of the translation request: push a TranslationTask
        onto ``self._tr_q``. Emits TEXT_SETTLED on the first stability-
        crossing frame, QUEUE_PUSH on success, QUEUE_FULL on backpressure.
        """
        if track.stable_frames == stable_required:
            self.logger.channel(
                "translate",
                message="TEXT_SETTLED",
                frame_index=frame_index,
                track_id=track.track_id,
                source_version=track.source_version,
                stable_frames=track.stable_frames,
                stable_required=stable_required,
                text=self._preview_text(track.text),
            )
        task = TranslationTask(
            frame_index=frame_index,
            track_id=track.track_id,
            source_version=track.source_version,
            source_text=track.text,
            cache_key=cache_key,
            priority=priority,
            queued_at=perf_counter(),
            generation=int(self._pipeline_generation),
            ui_hint=bool(track.ui_hint),
            name_hint=bool(track.name_hint),
            dialogue_hint=bool(track.dialogue_hint),
        )
        try:
            queued_item = (priority, next(self._translation_seq), task)
            self._tr_q.put_nowait(queued_item)
            self._pending_translation_keys.add(cache_key)
            track.translation_pending = True
            self._last_translation_activity_frame = max(
                int(self._last_translation_activity_frame or 0), int(frame_index)
            )
            self.logger.channel(
                "translate",
                message="QUEUE_PUSH",
                frame_index=frame_index,
                track_id=track.track_id,
                source_version=track.source_version,
                priority=priority,
                stable_required=stable_required,
                queue_size=self._tr_q.qsize(),
                text=self._preview_text(track.text),
            )
            self._set_status("Active. Translating dynamic text blocks…")
        except queue.Full:
            self.logger.log("TRANSLATE_QUEUE_FULL")
            self.logger.channel(
                "translate",
                message="QUEUE_FULL",
                frame_index=frame_index,
                track_id=track.track_id,
                source_version=track.source_version,
                priority=priority,
            )

    def _maybe_request_translation(self, frame_index: int, track: Track):
        """Top-level translation-request gate.

        Pipeline (in order):
          1. class settings (dialogue / name / sign / default)
          2. reveal hold + unstable hold
          3. static / low_value / not_stable skip
          4. empty / already-pending early return
          5. already-translated → re-enable render
          6. cache lookup (CACHE_HIT applies the translation)
          7. ui_hint policy skip (only after cache lookup)
          8. dedupe pending
          9. backlog limit (non-dialogue)
         10. TEXT_SETTLED log + queue push

        Each branch returns immediately on a hit; the orchestrator just
        sequences them so the path is readable end-to-end.
        """
        stable_required, priority = self._translation_class_settings(track)
        if self._handle_reveal_or_unstable_hold(frame_index, track, stable_required):
            return
        skip_reason = self._translation_skip_reason(track, stable_required)
        if skip_reason is not None:
            self.logger.channel(
                "translate",
                message="QUEUE_SKIP",
                frame_index=frame_index,
                track_id=track.track_id,
                source_version=track.source_version,
                reason=skip_reason,
                stable_frames=track.stable_frames,
                stable_required=stable_required,
                unstable_frames=track.unstable_frames,
                text=self._preview_text(track.text),
            )
            return
        if not track.text.strip() or track.translation_pending:
            return
        if (track.translation or "").strip():
            if not track.render_suppressed_until_match and not self._dialogue_transition_blocks_track(track):
                self._set_track_render_enabled(track, True)
            return
        # Namespace cache by hint kind so a menu label cached as "Road" from
        # a stateless translation pass doesn't get returned for the same
        # source text seen as UI. Keeps dialogue/UI translations separate
        # buckets.
        cache_key = track.normalized
        if track.ui_hint:
            cache_key = "ui\x00" + cache_key
        elif track.name_hint:
            cache_key = "name\x00" + cache_key
        if self._apply_cached_translation(frame_index, track, cache_key):
            return
        if track.ui_hint and not bool(getattr(config, "TRANSLATE_UI_TEXT", False)):
            self.logger.channel(
                "translate",
                message="QUEUE_SKIP",
                frame_index=frame_index,
                track_id=track.track_id,
                source_version=track.source_version,
                reason="ui_text",
                text=self._preview_text(track.text),
            )
            return
        if (
            bool(getattr(config, "TRANSLATION_DEDUPLICATE_PENDING", True))
            and cache_key in self._pending_translation_keys
        ):
            self.logger.channel(
                "translate",
                message="QUEUE_DEDUPED_PENDING",
                frame_index=frame_index,
                track_id=track.track_id,
                source_version=track.source_version,
                priority=priority,
                text=self._preview_text(track.text),
            )
            return
        backlog_limit = int(getattr(config, "TRANSLATION_LOW_PRIORITY_BACKLOG_LIMIT", 1))
        queue_size = self._tr_q.qsize()
        if not track.dialogue_hint:
            if (
                (not track.ui_hint)
                and (not track.name_hint)
                and track.stable_frames >= int(getattr(config, "SIGN_FORCE_QUEUE_STABLE_FRAMES", 3))
            ):
                backlog_limit = max(backlog_limit, int(getattr(config, "TRANSLATION_SIGN_BACKLOG_LIMIT", 4)))
            if queue_size >= backlog_limit:
                self.logger.channel(
                    "translate",
                    message="QUEUE_DEFER_LOW_PRIORITY",
                    frame_index=frame_index,
                    track_id=track.track_id,
                    source_version=track.source_version,
                    priority=priority,
                    queue_size=queue_size,
                    backlog_limit=backlog_limit,
                    text=self._preview_text(track.text),
                )
                return
        self._enqueue_translation(
            frame_index,
            track,
            cache_key=cache_key,
            stable_required=stable_required,
            priority=priority,
        )

    def _region_visually_changed(
        self, image: Image.Image, track: Track, *, for_liveness: bool = False
    ) -> tuple[bool, float, bytes, float]:
        """Decide whether the track region changed, fraction-of-cells style.

        Returns ``(changed, fraction, current_sig, mean_delta)``:

        - ``changed`` — True iff the fraction-of-cells gate fired. This is
          the primary "should I act" signal.
        - ``fraction`` — 0..1, fraction of downscaled signature cells whose
          per-cell brightness delta exceeded the cell-delta-min threshold.
          Loggers report this as a percentage so a small effect on a long
          region reads as "3% changed" rather than as a mean-delta of 0.4.
        - ``current_sig`` — the fresh signature bytes; caller stores into
          the track on confirmed match.
        - ``mean_delta`` — 0..255 mean absolute delta across all cells.
          Used by callers for the PIXEL_CHANGE_HARD_CUT_THRESHOLD fallback
          that catches global-brightness-shift scenarios the fraction gate
          would otherwise miss.

        When ``for_liveness=True`` (the render-hide path) we use a stricter
        cell-delta floor + a higher fraction threshold, and optionally drop
        the wide-context band of the signature so small pulses outside the
        actual text glyphs (cursor blink, sparkle next to a name tag) don't
        trigger a hide.
        """
        current_sig = _region_signature(image, track.rect, track.word_rects)
        stored = track.region_signature
        if for_liveness and bool(getattr(config, "LIVENESS_WORD_FOCUSED_SIGNATURE", True)):
            context_size = int(getattr(config, "PIXEL_CHANGE_SIGNATURE_SIZE", 24)) ** 2
            if stored and len(stored) > context_size and len(current_sig) > context_size:
                stored_eff = stored[context_size:]
                current_eff = current_sig[context_size:]
            else:
                stored_eff = stored
                current_eff = current_sig
        else:
            stored_eff = stored
            current_eff = current_sig
        if for_liveness:
            cell_min = int(getattr(config, "LIVENESS_CELL_DELTA_MIN", 24))
            frac_threshold = float(getattr(config, "LIVENESS_CHANGED_FRACTION_THRESHOLD", 0.20))
        else:
            cell_min = int(getattr(config, "PIXEL_CHANGE_CELL_DELTA_MIN", 16))
            frac_threshold = float(getattr(config, "PIXEL_CHANGE_FRACTION_THRESHOLD", 0.10))
        fraction, mean_delta = _region_signature_metrics(stored_eff, current_eff, cell_min)
        return fraction >= frac_threshold, fraction, current_sig, mean_delta

    def _find_replacement_observation(self, track: Track, observations: list[Observation]) -> tuple[bool, float, float]:
        best_iou = 0.0
        best_text_ratio = 0.0
        max_distance = max(float(config.MATCH_DISTANCE_PX) * 1.25, float(max(track.rect.width, track.rect.height, 1)))
        iou_min = float(getattr(config, "REPLACEMENT_OBS_IOU_MIN", 0.28))
        text_ratio_max = float(getattr(config, "REPLACEMENT_OBS_TEXT_MAX_RATIO", 0.74))
        horiz_min = float(getattr(config, "REPLACEMENT_OBS_HORIZONTAL_OVERLAP_MIN", 0.55))
        vert_min = float(getattr(config, "REPLACEMENT_OBS_VERTICAL_OVERLAP_MIN", 0.55))
        for obs in observations:
            iou = _rect_iou(track.rect, obs.rect)
            dist = _rect_center_distance(track.rect, obs.rect)
            if iou < 0.08 and dist > max_distance:
                continue
            text_ratio = (
                _fuzz_ratio(track.normalized, obs.normalized) / 100.0 if track.normalized and obs.normalized else 0.0
            )
            best_iou = max(best_iou, iou)
            best_text_ratio = max(best_text_ratio, text_ratio)
            if text_ratio >= text_ratio_max:
                continue
            horiz_overlap = _horizontal_overlap_ratio(track.rect, obs.rect)
            vert_overlap = max(0, min(track.rect.bottom, obs.rect.bottom) - max(track.rect.top, obs.rect.top)) / max(
                1, min(track.rect.height, obs.rect.height)
            )
            if iou >= iou_min or (horiz_overlap >= horiz_min and vert_overlap >= vert_min):
                return True, iou, text_ratio
        return False, best_iou, best_text_ratio

    def _match_observations_to_tracks(self, frame_index: int, observations: list[Observation]) -> set[int]:
        """Run the per-observation matching pass.

        For each observation pick the highest-scoring unmatched track and
        either update it (score ≥ 0.24) or create a fresh track. Returns
        the set of matched track ids so the caller can identify the
        unmatched (missing-this-frame) tracks.
        """
        matched_ids: set[int] = set()
        for obs in observations:
            best_track: Track | None = None
            best_score = 0.0
            for track in self._tracks.values():
                if track.track_id in matched_ids:
                    continue
                score = self._match_score(track, obs)
                if score > best_score:
                    best_score = score
                    best_track = track
            if best_track is None or best_score < 0.24:
                track = self._create_track(frame_index, obs)
            else:
                track = best_track
                self._update_track(frame_index, track, obs)
            matched_ids.add(track.track_id)
        return matched_ids

    def _stale_missing_limit_for_track(self, track: Track) -> int:
        """Per-class missing-frame limit for the "held but no longer
        observed" path. Picks the strictest applicable limit among the
        track's class hints; default for unhinted tracks.
        """
        limit = max(1, int(getattr(config, "STALE_RENDER_MAX_MISSING_FRAMES", 3)))
        if track.low_value_hint:
            limit = min(limit, max(1, int(getattr(config, "STALE_RENDER_MAX_MISSING_FRAMES_LOW_VALUE", 2))))
        if track.ui_hint:
            limit = min(limit, max(1, int(getattr(config, "STALE_RENDER_MAX_MISSING_FRAMES_UI", 1))))
        if track.dialogue_hint or track.name_hint:
            limit = min(limit, max(1, int(getattr(config, "STALE_RENDER_MAX_MISSING_FRAMES_DIALOGUE", 1))))
        return limit

    def _process_missing_track(
        self,
        frame_index: int,
        track: Track,
        image: Image.Image | None,
        observations: list[Observation],
        forget_frames: int,
    ) -> bool:
        """Handle a track that wasn't matched by any observation this frame.

        Updates missing_frames, decides whether the track holds (text
        likely still on-screen), got replaced (a new observation supplanted
        it), or vanished (waiting for OCR confirm). Returns True if the
        track should be marked dead and removed.
        """
        track.missing_frames += 1
        if track.missing_frames > 1:
            track.translation_pending = False
        has_render_state = bool(track.render_enabled or track.is_static or bool((track.translation or "").strip()))
        if image is None or not has_render_state:
            self.logger.channel(
                "tracks",
                message="MISS",
                frame_index=frame_index,
                track_id=track.track_id,
                missing_frames=track.missing_frames,
                text=self._preview_text(track.text),
            )
            return track.missing_frames > forget_frames
        (
            effective_hold_max,
            effective_disappear_confirm,
            _effective_replaced_confirm,
            effective_pixel_delete_after,
            hard_cut_threshold,
        ) = self._effective_missing_params(track)
        changed, fraction, current_sig, mean_delta = self._region_visually_changed(image, track)
        replacement_found, replacement_iou, replacement_text_ratio = self._find_replacement_observation(
            track, observations
        )
        hard_cut = mean_delta >= hard_cut_threshold
        changed = changed or hard_cut
        score = fraction  # alias for the existing log fields below
        if not changed:
            track.held_frames += 1
            track.change_frames = 0
            track.region_signature = current_sig
            stale_missing_limit = self._stale_missing_limit_for_track(track)
            if track.missing_frames >= stale_missing_limit:
                self._set_track_render_enabled(track, False)
                track.render_suppressed_until_match = True
                track.translation_pending = False
                self.logger.channel(
                    "tracks",
                    message="MISS_HELD_EXPIRE",
                    frame_index=frame_index,
                    track_id=track.track_id,
                    missing_frames=track.missing_frames,
                    held_frames=track.held_frames,
                    missing_limit=stale_missing_limit,
                    diff_score=round(score, 3),
                    text=self._preview_text(track.text),
                )
                return track.missing_frames > max(forget_frames, stale_missing_limit)
            if (track.translation or "").strip() and not track.is_static and not track.render_suppressed_until_match:
                self._set_track_render_enabled(track, True)
            self.logger.channel(
                "tracks",
                message="MISS_HELD",
                frame_index=frame_index,
                track_id=track.track_id,
                missing_frames=track.missing_frames,
                held_frames=track.held_frames,
                diff_score=round(score, 3),
                render_suppressed=bool(track.render_suppressed_until_match),
                missing_limit=stale_missing_limit,
                text=self._preview_text(track.text),
            )
            return track.held_frames > effective_hold_max
        track.change_frames += 1
        if replacement_found:
            self._set_track_render_enabled(track, False)
            track.render_suppressed_until_match = True
            track.translation_pending = False
            self.logger.channel(
                "tracks",
                message="MISS_REPLACED",
                frame_index=frame_index,
                track_id=track.track_id,
                missing_frames=track.missing_frames,
                change_frames=track.change_frames,
                diff_score=round(score, 3),
                replacement_iou=round(replacement_iou, 3),
                replacement_text_ratio=round(replacement_text_ratio, 3),
                render_disabled=True,
                text=self._preview_text(track.text),
            )
            return track.missing_frames > forget_frames
        if hard_cut or track.change_frames >= effective_pixel_delete_after:
            self._set_track_render_enabled(track, False)
            track.render_suppressed_until_match = True
            track.translation_pending = False
        self.logger.channel(
            "tracks",
            message="MISS_CHANGED_WAIT_OCR",
            frame_index=frame_index,
            track_id=track.track_id,
            missing_frames=track.missing_frames,
            change_frames=track.change_frames,
            diff_score=round(score, 3),
            hard_cut=bool(hard_cut),
            render_disabled=bool(hard_cut or track.change_frames >= effective_pixel_delete_after),
            render_suppressed=bool(track.render_suppressed_until_match),
            text=self._preview_text(track.text),
        )
        if track.change_frames < effective_disappear_confirm:
            if (
                (track.translation or "").strip()
                and not track.is_static
                and (not track.render_suppressed_until_match)
                and not track.render_enabled
            ):
                self._set_track_render_enabled(track, True)
            return False
        self._set_track_render_enabled(track, False)
        track.render_suppressed_until_match = True
        return track.missing_frames > max(forget_frames, effective_disappear_confirm)

    def _update_tracks(self, frame_index: int, observations: list[Observation]):
        """Track lifecycle for one OCR frame.

        Three phases:
          1. _match_observations_to_tracks — assign observations to tracks,
             create new tracks for unmatched observations.
          2. For each track NOT matched this frame, _process_missing_track
             decides hold / replace / wait / delete.
          3. Sweep dead tracks and emit FRAME_SUMMARY.
        """
        matched_ids = self._match_observations_to_tracks(frame_index, observations)
        dead_ids: list[int] = []
        image = self._latest_image
        forget_frames = int(config.TRACK_FORGET_FRAMES)
        for track_id, track in self._tracks.items():
            if track_id in matched_ids:
                continue
            if self._process_missing_track(frame_index, track, image, observations, forget_frames):
                dead_ids.append(track_id)
        for track_id in sorted(set(dead_ids)):
            doomed = self._tracks.get(track_id)
            if doomed is None:
                continue
            self._log_track_event("DELETE", frame_index, doomed)
            del self._tracks[track_id]
        self.logger.channel(
            "tracks",
            message="FRAME_SUMMARY",
            frame_index=frame_index,
            active_track_count=len(self._tracks),
            observation_count=len(observations),
            tracks=[self._track_snapshot(t) for t in sorted(self._tracks.values(), key=lambda x: x.track_id)],
        )

    def _handle_translation_result(self, payload: TranslationResult):
        """Apply a translation worker's result to the matching track.

        Thin orchestrator: delegates validation to ``_resolve_translation_result``
        (returns the track to apply against, or None if the result is to be
        dropped) and the side-effects to ``_apply_translation_to_track``.
        Payload is a TranslationResult dataclass so generation is always
        present and the stale-drop check (8ade7ab) cannot be bypassed by a
        truncated tuple.
        """
        track = self._resolve_translation_result(
            frame_index=payload.frame_index,
            track_id=payload.track_id,
            source_version=payload.source_version,
            source_text=payload.source_text,
            cache_key=payload.cache_key,
            generation=payload.generation,
        )
        if track is None:
            return
        self._apply_translation_to_track(payload.frame_index, track, payload.source_version, payload.translated)
        self._emit_scene(self._last_scene_frame_index, reason="translation_result")

    def _resolve_translation_result(
        self,
        *,
        frame_index: int,
        track_id: int,
        source_version: int,
        source_text: str,
        cache_key: str,
        generation: int,
    ) -> Track | None:
        """Validate a translation result and return the track to apply it to.

        Returns None for every drop path (stale generation, region edit
        active, no such track, or stale source text/version). Always
        clears ``track.translation_pending`` before the stale-source
        check so a DROP_STALE result doesn't permanently block future
        translation requests on the track (bug fixed in 8ade7ab).
        """
        if int(generation) != int(self._pipeline_generation):
            self.logger.channel(
                "translate",
                message="DROP_STALE_GENERATION",
                frame_index=frame_index,
                track_id=track_id,
                source_version=source_version,
                result_generation=int(generation),
                current_generation=int(self._pipeline_generation),
            )
            return None
        if self._region_edit_active:
            self.logger.channel(
                "translate",
                message="DROP_REGION_EDIT_ACTIVE",
                frame_index=frame_index,
                track_id=track_id,
                source_version=source_version,
            )
            return None
        if cache_key:
            self._pending_translation_keys.discard(cache_key)
        track = self._tracks.get(track_id)
        if track is None:
            self.logger.channel(
                "translate",
                message="DROP_NO_TRACK",
                frame_index=frame_index,
                track_id=track_id,
                source_version=source_version,
            )
            return None
        track.translation_pending = False
        if track.source_version != source_version or track.text != source_text:
            self.logger.channel(
                "translate",
                message="DROP_STALE",
                frame_index=frame_index,
                track_id=track_id,
                source_version=source_version,
                current_source_version=track.source_version,
                requested_text=self._preview_text(source_text),
                current_text=self._preview_text(track.text),
            )
            return None
        return track

    def _apply_translation_to_track(self, frame_index: int, track: Track, source_version: int, translated: str) -> None:
        """Side-effects of a validated translation result: set the
        translation, propagate to peers, persist memory, update render
        gates, and emit the log breadcrumb.
        """
        if not translated:
            self.logger.channel(
                "translate",
                message="EMPTY_RESULT",
                frame_index=frame_index,
                track_id=track.track_id,
                source_version=source_version,
            )
            return
        track.translation = translated
        self._last_translation_activity_frame = max(int(self._last_translation_activity_frame or 0), int(frame_index))
        self._set_track_render_enabled(track, self._can_show_track_render(track))
        track.held_frames = 0
        track.change_frames = 0
        track.liveness_change_frames = 0
        track.liveness_hold_frames = 0
        self._translation_cache[track.normalized] = translated
        self._append_translation_history(track.normalized, translated)
        self._propagate_translation_to_peers(track, translated)
        self._remember_region_memo(frame_index, track)
        self._save_translation_memory()
        self._set_status("Active. Capturing the full target window and painting translations in place.")
        self._log_track_event("TRANSLATED", frame_index, track, translated_text=self._preview_text(translated))

    def _append_translation_history(self, source: str, translated: str) -> None:
        """Push a (source, translation) pair onto the rolling context
        window used by later translations as few-shot examples. Skips
        empties and skips a duplicate-of-last append so the same line
        held across many frames doesn't flood the window with itself
        (which would push out real prior context and give the model
        nothing useful).
        """
        if self._translation_history_max <= 0:
            return
        src = (source or "").strip()
        tgt = (translated or "").strip()
        if not src or not tgt:
            return
        if self._translation_history and self._translation_history[-1][0] == src:
            return
        self._translation_history.append((src, tgt))

    def _propagate_translation_to_peers(self, track: Track, translated: str) -> None:
        """Apply ``translated`` to peer tracks that match by text OR by
        region signature + rect. Single iteration over self._tracks; the
        two predicates used to be evaluated in two separate passes (one
        per ``match_by_text`` boolean), iterating the track set twice.
        OR-ing them in one loop produces identical end state — any peer
        matched by either rule still gets the translation.
        """
        for other in self._tracks.values():
            if other.track_id == track.track_id:
                continue
            if other.is_static or other.low_value_hint:
                continue
            matches_text = other.normalized == track.normalized
            matches_region = other.region_signature == track.region_signature and other.rect == track.rect
            if not (matches_text or matches_region):
                continue
            other.translation_pending = False
            if not (other.translation or "").strip():
                other.translation = translated
            if (
                other.missing_frames == 0
                and not other.render_suppressed_until_match
                and self._can_show_track_render(other)
            ):
                self._set_track_render_enabled(other, True)

    def _build_scene(self) -> OverlayScene:
        """Build the next OverlayScene from current track state.

        Three phases:
          1. Snapshot scene-wide config once (_SceneLayoutCfg.from_config).
          2. Sort + prepare the eligible tracks.
          3. _build_render_item_for_track per track, dropping any that
             fail the allowed-rect or empty-result checks.
        """
        image = self._latest_image
        if image is None:
            region = self.state.client_region
            size = (region.width, region.height) if region else (1, 1)
            return OverlayScene(size=size, items=[], allowed_regions=[], ignore_regions=[])
        started = perf_counter()
        scene_cfg = _SceneLayoutCfg.from_config()
        scene_frame_index = int(self._last_scene_frame_index or self._latest_captured_frame_index or 0)
        seen_compact_labels: set[str] = set()
        tracks = sorted(
            self._tracks.values(),
            key=lambda t: (
                int(t.render_z_index),
                int(t.last_matched_frame),
                t.rect.top,
                t.rect.left,
                t.track_id,
            ),
        )
        tracks = _prepare_render_tracks(tracks, image, logger=self.logger)
        tracks = _apply_scene_uniformity(tracks)
        items: list[RenderedItem] = []
        render_index = 0
        for track in tracks:
            item = self._build_render_item_for_track(
                track,
                image,
                scene_cfg=scene_cfg,
                scene_frame_index=scene_frame_index,
                next_render_index=render_index + 1,
                seen_compact_labels=seen_compact_labels,
            )
            if item is None:
                continue
            render_index += 1
            items.append(item)
        self._log_timing("build_scene", (perf_counter() - started) * 1000.0, item_count=len(items))
        return OverlayScene(
            size=image.size,
            items=items,
            allowed_regions=self._effective_render_regions(image),
            ignore_regions=list(self._ignore_regions),
        )

    def _build_render_item_for_track(
        self,
        track: Track,
        image: Image.Image,
        *,
        scene_cfg: "_SceneLayoutCfg",
        scene_frame_index: int,
        next_render_index: int,
        seen_compact_labels: set[str],
    ) -> RenderedItem | None:
        """Build one RenderedItem for ``track``, or None to skip.

        Mirrors the original inline `continue` paths in _build_scene:
        static / no-translation / render-disabled / dialogue-blocked /
        duplicate-compact-label / not-allowed-rect → None. Otherwise
        compute the layout from ``scene_cfg`` and return the item.
        ``seen_compact_labels`` is mutated to dedupe across calls within
        the same _build_scene pass.
        """
        if track.is_static:
            return None
        translation = (track.translation or "").strip()
        if not translation or not track.render_enabled:
            return None
        if self._dialogue_transition_blocks_track(track):
            return None
        if (not track.repeated_label_hint) and _is_compact_scene_label_geometry(
            track.normalized, track.rect, track.line_count, image
        ):
            if track.normalized in seen_compact_labels:
                return None
            seen_compact_labels.add(track.normalized)
        render_rect = track.rect
        if track.repeated_label_hint and int(track.estimated_full_width or 0) > 0:
            anchor_left = max(
                0,
                min(image.width - 1, int(track.render_anchor_left or track.rect.left)),
            )
            desired_width = max(track.rect.width, int(track.estimated_full_width or track.rect.width))
            desired_right = min(image.width, anchor_left + desired_width)
            render_rect = Rect(anchor_left, track.rect.top, max(1, desired_right - anchor_left), track.rect.height)
        if not self._rect_is_allowed(render_rect, image):
            return None
        margin_x = scene_cfg.text_margin_x
        margin_y = scene_cfg.text_margin_y
        if track.dialogue_hint or track.name_hint:
            margin_x += scene_cfg.text_dialogue_extra_margin_x
            margin_y += scene_cfg.text_dialogue_extra_margin_y
        source_h = int(track.median_char_height or 0)
        if source_h <= 0:
            source_h = _median_word_height(track.word_rects, render_rect, track.line_count)
        min_px = scene_cfg.text_min_pixel
        max_px = scene_cfg.text_max_pixel
        # ``text_source_height_scale`` used to only apply in the fallback branch
        # (source_h == 0), so with OCR reporting a char height on every frame
        # the "Scale of source char height" spinbox did nothing. Applying it in
        # both branches makes the knob actually reduce the rendered font when
        # OCR did report a size — the main lever the user has to shrink text
        # across the board.
        if source_h > 0:
            scaled = int(round(source_h * scene_cfg.text_source_height_scale))
            preferred_px = max(min_px, min(max_px, scaled))
        else:
            h = max(1, int(track.rect.height / max(1, track.line_count)))
            preferred_px = max(min_px, min(max_px, int(round(h * scene_cfg.text_source_height_scale))))
        # small_sign was firing on render_rect.height alone (≤50px), which
        # captured every normal single-line label — a "Yasumi" name-tag at
        # 26px source rendered at 15px next to dialogue at 25px because
        # this branch hard-capped preferred_px at SMALL_SIGN_MAX_PIXEL=16
        # and then TEXT_NAME_SCALE=0.92 multiplied on top. Real UI signs
        # have a genuinely small source (source_h itself is tiny); use
        # that as the gate instead. Name tags are excluded — they're
        # dialogue-adjacent labels, not tiny HUD signs.
        small_sign = (
            (not track.dialogue_hint)
            and (not track.ui_hint)
            and (not track.name_hint)
            and render_rect.height <= scene_cfg.small_text_rescan_max_height
            and 0 < source_h <= scene_cfg.small_sign_max_pixel
        )
        if small_sign:
            preferred_px = int(round(preferred_px * scene_cfg.small_sign_text_scale))
            preferred_px = min(preferred_px, max(source_h + scene_cfg.small_sign_max_extra_px, min_px))
            preferred_px = min(preferred_px, scene_cfg.small_sign_max_pixel)
        font_family = scene_cfg.text_sign_font_family
        font_weight = scene_cfg.text_sign_font_weight
        if track.dialogue_hint:
            font_family = scene_cfg.text_dialogue_font_family
            font_weight = scene_cfg.text_dialogue_font_weight
        elif track.name_hint:
            font_family = scene_cfg.text_name_font_family
            font_weight = scene_cfg.text_name_font_weight
            preferred_px = int(round(preferred_px * scene_cfg.text_name_scale))
        writing_mode = track.writing_mode if scene_cfg.render_vertical_text else "horizontal"
        if writing_mode == "vertical" and not _contains_vertical_script(translation):
            # Source was vertical CJK but the translation has no CJK chars.
            # The vertical renderer would verticalize char-by-char and
            # disable wrap, overflowing narrow source columns since
            # Qt drawText doesn't clip non-wrapped text. Treat as
            # horizontal so the text wraps inside the (tall narrow) rect.
            writing_mode = "horizontal"
        render_bounds = self._render_bounds_for_rect(render_rect, image, expand_to_zone=True)
        if writing_mode == "vertical":
            text_rect = _expand_rect(
                render_rect,
                image.width,
                image.height,
                scene_cfg.text_vertical_extra_margin_x,
                scene_cfg.text_vertical_extra_margin_y,
            )
            allow_wrap = False
            alignment = "center"
        else:
            text_rect = _expand_rect(render_rect, image.width, image.height, margin_x, margin_y)
            # Clip width to render_bounds BEFORE wrap planning. Otherwise
            # _planned_text_rect measures wrap against the (image-limited)
            # expanded width while _move_rect_inside below clamps the final
            # rect to the narrower render_bounds — leaving the planned height
            # short for the actual wrapped line count, so text overflows.
            text_rect = _rect_intersection(text_rect, render_bounds) or text_rect
            thin_label = track.line_count <= 1 and track.rect.height <= scene_cfg.hud_thin_max_height
            expansion_ratio = len(re.sub(r"\s+", "", translation)) / max(1, len(re.sub(r"\s+", "", track.normalized)))
            allow_wrap = (not thin_label) or (expansion_ratio >= scene_cfg.thin_label_wrap_ratio)
            alignment = _render_alignment(track)
            text_rect = _planned_text_rect(
                text_rect,
                translation,
                preferred_px,
                font_family,
                font_weight,
                alignment,
                allow_wrap,
                image,
                track,
                max_width=render_bounds.width,
                max_height=max(1, render_bounds.bottom - text_rect.top),
            )
        text_rect = _move_rect_inside(text_rect, render_bounds)
        cover_rect = _union_rect(render_rect, text_rect)
        if not self._rect_is_allowed(cover_rect, image):
            return None
        # Grouped-zone tracks carry the individual per-line source rects that
        # got unioned into ``track.rect``. Paint one patch per source line so
        # gaps between lines aren't blanketed by a single tall union patch —
        # that was the "size is larger when grouped than ungrouped" complaint.
        # The translation still renders into ``text_rect`` (which spans the
        # union), so text can bleed over gaps between patches — that's fine
        # because gap pixels are the untouched source background.
        member_rects = [
            r for r in (track.member_source_rects or []) if self._rect_is_allowed(r, image)
        ]
        extra_patches: list[tuple[Rect, Image.Image]] = []
        if len(member_rects) >= 2:
            patch_rects: list[tuple[Rect, Image.Image]] = []
            for line_rect in member_rects:
                p_rect, p_img = _build_patch(
                    image,
                    line_rect,
                    track.style.background_color,
                    extra_patch_x=0,
                    extra_patch_y=0,
                )
                patch_rects.append((p_rect, p_img))
            # Cover any text_rect region that extends beyond the union of the
            # source lines (e.g. a translation that wraps into more rows than
            # the source had). Otherwise the tail would render over the raw
            # scene with no patch underneath.
            src_union = member_rects[0]
            for r in member_rects[1:]:
                src_union = _union_rect(src_union, r)
            tail_top = src_union.bottom
            if text_rect.bottom > tail_top + int(config.PATCH_EXPAND_Y):
                tail_rect = Rect(
                    max(text_rect.left, src_union.left),
                    tail_top,
                    max(1, min(text_rect.right, src_union.right + int(config.PATCH_EXPAND_X)) - max(text_rect.left, src_union.left)),
                    max(1, text_rect.bottom - tail_top),
                )
                p_rect, p_img = _build_patch(
                    image,
                    tail_rect,
                    track.style.background_color,
                    extra_patch_x=0,
                    extra_patch_y=0,
                )
                patch_rects.append((p_rect, p_img))
            patch_rect, patch_img = patch_rects[0]
            extra_patches = patch_rects[1:]
        else:
            patch_rect, patch_img = _build_patch(
                image,
                cover_rect,
                track.style.background_color,
                extra_patch_x=0,
                extra_patch_y=0,
            )
        return RenderedItem(
            patch_rect=patch_rect,
            patch_image=patch_img,
            text_rect=text_rect,
            text=translation,
            fill_color=track.style.fill_color,
            outline_color=track.style.outline_color,
            allow_wrap=allow_wrap,
            preferred_pixel_size=preferred_px,
            alignment=alignment,
            font_family=font_family,
            font_weight=font_weight,
            writing_mode=writing_mode,
            render_id=self._render_ref(scene_frame_index, next_render_index),
            source_track_id=track.track_id,
            source_obs_id=track.last_obs_id,
            source_frame_index=track.last_obs_frame_index,
            z_index=int(track.render_z_index),
            extra_patches=extra_patches,
        )

    def _emit_scene(self, frame_index: int | None = None, reason: str | None = None):
        started = perf_counter()
        scene = self._build_scene()
        if frame_index is None:
            frame_index = self._last_scene_frame_index
        if frame_index < self._latest_presented_frame_index and bool(
            getattr(config, "PRESENTATION_STRICT_MONOTONIC", True)
        ):
            self.logger.channel(
                "render",
                message="DROP_STALE_SCENE",
                frame_index=frame_index,
                latest_presented_frame=self._latest_presented_frame_index,
                reason=reason,
                item_count=len(scene.items),
            )
            return
        if self._latest_image is not None and frame_index != self._last_annotated_frame_index:
            try:
                self._save_annotated_frame(
                    frame_index,
                    self._latest_image,
                    self._last_observations,
                    reason=reason,
                    scene_item_count=len(scene.items),
                )
            except Exception as exc:
                self.logger.channel(
                    "ocr",
                    message="ANNOTATED_SAVE_FAILED",
                    frame_index=frame_index,
                    error=str(exc),
                )
        try:
            preview_path = self._save_render_preview(frame_index, scene, reason=reason)
        except Exception as exc:
            preview_path = ""
            self.logger.channel(
                "render",
                message="PREVIEW_SAVE_FAILED",
                frame_index=frame_index,
                error=str(exc),
            )
        try:
            manifest_path = self._save_frame_manifest(frame_index, scene, reason=reason)
        except Exception as exc:
            manifest_path = ""
            self.logger.channel(
                "render",
                message="MANIFEST_SAVE_FAILED",
                frame_index=frame_index,
                error=str(exc),
            )
        render_payload: dict[str, Any] = dict(
            message="SCENE",
            frame_index=frame_index,
            reason=reason,
            item_count=len(scene.items),
            scene_size={"width": scene.size[0], "height": scene.size[1]},
            preview_image=preview_path,
            frame_manifest=manifest_path,
        )
        if bool(getattr(config, "DEBUG_LOG_RENDER_ITEMS", False)):
            render_payload["items"] = [
                {
                    "render_id": item.render_id,
                    "source_track_id": item.source_track_id,
                    "source_track_ref": self._track_ref(item.source_track_id),
                    "source_obs_id": item.source_obs_id,
                    "source_frame_index": item.source_frame_index,
                    "source_frame_ref": self._frame_ref(item.source_frame_index),
                    "z_index": item.z_index,
                    "text": self._preview_text(item.text),
                    "patch_rect": self._rect_to_dict(item.patch_rect),
                    "text_rect": self._rect_to_dict(item.text_rect),
                    "fill": tuple(item.fill_color),
                    "outline": tuple(item.outline_color),
                    "allow_wrap": bool(getattr(item, "allow_wrap", True)),
                    "preferred_pixel_size": int(getattr(item, "preferred_pixel_size", 0) or 0),
                    "alignment": getattr(item, "alignment", "center"),
                }
                for item in scene.items
            ]
        self.logger.channel("render", **render_payload)
        self.overlaySceneUpdated.emit(scene)
        self._latest_presented_frame_index = max(self._latest_presented_frame_index, frame_index)
        self._log_timing(
            "overlay_emit",
            (perf_counter() - started) * 1000.0,
            frame_index=frame_index,
            item_count=len(scene.items),
            reason=reason,
        )
