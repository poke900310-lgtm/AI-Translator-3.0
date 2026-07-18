"""Pure geometry helpers operating on Rect (from app.types).

Extracted from app.controller. None of these functions touch Controller
state — only Rect arithmetic and a couple of config lookups for tuning
thresholds. Kept underscore-prefixed to preserve the prior in-tree convention.
"""

from __future__ import annotations

import math

from app import config
from app.types import Rect


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _rect_area(rect: Rect) -> int:
    return int(rect.width * rect.height)


def _union_rect(a: Rect, b: Rect) -> Rect:
    left = min(a.left, b.left)
    top = min(a.top, b.top)
    right = max(a.right, b.right)
    bottom = max(a.bottom, b.bottom)
    return Rect(left, top, right - left, bottom - top)


def _rect_iou(a: Rect, b: Rect) -> float:
    inter_left = max(a.left, b.left)
    inter_top = max(a.top, b.top)
    inter_right = min(a.right, b.right)
    inter_bottom = min(a.bottom, b.bottom)
    inter_w = max(0, inter_right - inter_left)
    inter_h = max(0, inter_bottom - inter_top)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    union = (a.width * a.height) + (b.width * b.height) - inter
    return inter / max(1, union)


def _rect_intersection(a: Rect, b: Rect) -> Rect | None:
    inter_left = max(a.left, b.left)
    inter_top = max(a.top, b.top)
    inter_right = min(a.right, b.right)
    inter_bottom = min(a.bottom, b.bottom)
    if inter_right <= inter_left or inter_bottom <= inter_top:
        return None
    return Rect(inter_left, inter_top, inter_right - inter_left, inter_bottom - inter_top)


def _rect_intersection_area(a: Rect, b: Rect) -> int:
    inter = _rect_intersection(a, b)
    return 0 if inter is None else _rect_area(inter)


def _rect_contains_point(rect: Rect, x: float, y: float) -> bool:
    return rect.left <= x < rect.right and rect.top <= y < rect.bottom


def _rect_center_distance(a: Rect, b: Rect) -> float:
    ax = a.left + (a.width / 2.0)
    ay = a.top + (a.height / 2.0)
    bx = b.left + (b.width / 2.0)
    by = b.top + (b.height / 2.0)
    return math.hypot(ax - bx, ay - by)


def _horizontal_overlap_ratio(a: Rect, b: Rect) -> float:
    overlap = max(0, min(a.right, b.right) - max(a.left, b.left))
    return overlap / max(1, min(a.width, b.width))


def _vertical_overlap_ratio(a: Rect, b: Rect) -> float:
    overlap = max(0, min(a.bottom, b.bottom) - max(a.top, b.top))
    return overlap / max(1, min(a.height, b.height))


def _horizontal_gap(a: Rect, b: Rect) -> int:
    if a.right < b.left:
        return b.left - a.right
    if b.right < a.left:
        return a.left - b.right
    return 0


def _same_text_row(a: Rect, b: Rect) -> bool:
    min_overlap = float(getattr(config, "DIALOGUE_BOX_SAME_ROW_OVERLAP_MIN", 0.5))
    baseline_tol = int(getattr(config, "DIALOGUE_BOX_SAME_ROW_BASELINE_TOLERANCE_PX", 8))
    top_tol = int(getattr(config, "DIALOGUE_BOX_SAME_ROW_TOP_TOLERANCE_PX", 6))
    gap_max = int(getattr(config, "DIALOGUE_BOX_SAME_ROW_GAP_MAX_PX", 140))
    overlap = _vertical_overlap_ratio(a, b)
    same_baseline = abs(a.bottom - b.bottom) <= baseline_tol or abs(a.top - b.top) <= top_tol
    return _horizontal_gap(a, b) <= gap_max and (
        overlap >= min_overlap or (overlap >= max(0.25, min_overlap * 0.72) and same_baseline)
    )


def _is_vertical_rect(rect: Rect) -> bool:
    return rect.height >= max(12, int(rect.width * float(getattr(config, "VERTICAL_TEXT_ASPECT_RATIO", 1.35))))


def _rect_vertical_overlap_ratio(a: Rect, b: Rect) -> float:
    overlap = max(0, min(a.bottom, b.bottom) - max(a.top, b.top))
    return overlap / max(1, min(a.height, b.height))


def _expand_rect(rect: Rect, width_limit: int, height_limit: int, dx: int, dy: int) -> Rect:
    left = max(0, rect.left - dx)
    top = max(0, rect.top - dy)
    right = min(width_limit, rect.right + dx)
    bottom = min(height_limit, rect.bottom + dy)
    return Rect(left, top, max(1, right - left), max(1, bottom - top))


def _move_rect_inside(rect: Rect, bounds: Rect) -> Rect:
    width = min(max(1, rect.width), max(1, bounds.width))
    height = min(max(1, rect.height), max(1, bounds.height))
    left = rect.left
    top = rect.top
    if left < bounds.left:
        left = bounds.left
    if top < bounds.top:
        top = bounds.top
    if left + width > bounds.right:
        left = bounds.right - width
    if top + height > bounds.bottom:
        top = bounds.bottom - height
    left = max(bounds.left, left)
    top = max(bounds.top, top)
    return Rect(left, top, width, height)


__all__ = [
    "_clamp",
    "_expand_rect",
    "_horizontal_gap",
    "_horizontal_overlap_ratio",
    "_is_vertical_rect",
    "_move_rect_inside",
    "_rect_area",
    "_rect_center_distance",
    "_rect_contains_point",
    "_rect_intersection",
    "_rect_intersection_area",
    "_rect_iou",
    "_rect_vertical_overlap_ratio",
    "_same_text_row",
    "_union_rect",
    "_vertical_overlap_ratio",
]
