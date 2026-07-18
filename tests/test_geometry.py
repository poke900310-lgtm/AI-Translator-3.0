"""Pure Rect-math helper tests."""

from __future__ import annotations

import math

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
from app.types import Rect

# Sample rects used across multiple cases. left, top, width, height.
A = Rect(0, 0, 10, 10)
B = Rect(5, 5, 10, 10)
C = Rect(100, 100, 5, 5)


def test_clamp_inside_range() -> None:
    assert _clamp(5, 0, 10) == 5


def test_clamp_below_low() -> None:
    assert _clamp(-3, 0, 10) == 0


def test_clamp_above_high() -> None:
    assert _clamp(20, 0, 10) == 10


def test_rect_area() -> None:
    assert _rect_area(A) == 100
    assert _rect_area(C) == 25


def test_union_rect_overlapping() -> None:
    u = _union_rect(A, B)
    assert (u.left, u.top, u.width, u.height) == (0, 0, 15, 15)


def test_union_rect_disjoint() -> None:
    u = _union_rect(A, C)
    assert (u.left, u.top, u.right, u.bottom) == (0, 0, 105, 105)


def test_rect_iou_partial_overlap() -> None:
    iou = _rect_iou(A, B)
    # Intersection 5×5 = 25; union = 100 + 100 - 25 = 175 -> 25/175 ≈ 0.142857
    assert math.isclose(iou, 25 / 175, rel_tol=1e-6)


def test_rect_iou_no_overlap() -> None:
    assert _rect_iou(A, C) == 0.0


def test_rect_iou_identical() -> None:
    assert _rect_iou(A, A) == 1.0


def test_rect_intersection_partial() -> None:
    inter = _rect_intersection(A, B)
    assert inter is not None
    assert (inter.left, inter.top, inter.width, inter.height) == (5, 5, 5, 5)


def test_rect_intersection_disjoint() -> None:
    assert _rect_intersection(A, C) is None


def test_rect_intersection_area() -> None:
    assert _rect_intersection_area(A, B) == 25
    assert _rect_intersection_area(A, C) == 0


def test_rect_contains_point_inside() -> None:
    assert _rect_contains_point(A, 5, 5) is True
    assert _rect_contains_point(A, 0, 0) is True


def test_rect_contains_point_outside() -> None:
    # right/bottom edges are exclusive
    assert _rect_contains_point(A, 10, 5) is False
    assert _rect_contains_point(A, -1, 5) is False


def test_rect_center_distance() -> None:
    # A center (5,5), B center (10,10) -> dist sqrt(50)
    d = _rect_center_distance(A, B)
    assert math.isclose(d, math.sqrt(50), rel_tol=1e-9)


def test_horizontal_overlap_ratio() -> None:
    # A:[0,10], B:[5,15] -> overlap 5, min width 10 -> 0.5
    assert math.isclose(_horizontal_overlap_ratio(A, B), 0.5, rel_tol=1e-9)


def test_vertical_overlap_ratio() -> None:
    assert math.isclose(_vertical_overlap_ratio(A, B), 0.5, rel_tol=1e-9)


def test_rect_vertical_overlap_ratio_matches() -> None:
    # Same shape as _vertical_overlap_ratio for the typical case.
    assert _rect_vertical_overlap_ratio(A, B) == _vertical_overlap_ratio(A, B)


def test_horizontal_gap_no_gap() -> None:
    assert _horizontal_gap(A, B) == 0  # overlap


def test_horizontal_gap_left_then_right() -> None:
    left = Rect(0, 0, 5, 5)
    right = Rect(10, 0, 5, 5)
    assert _horizontal_gap(left, right) == 5
    assert _horizontal_gap(right, left) == 5  # symmetric


def test_is_vertical_rect_tall() -> None:
    tall = Rect(0, 0, 10, 30)
    assert _is_vertical_rect(tall) is True


def test_is_vertical_rect_wide() -> None:
    wide = Rect(0, 0, 30, 10)
    assert _is_vertical_rect(wide) is False


def test_expand_rect_clamped_to_bounds() -> None:
    r = Rect(5, 5, 10, 10)
    expanded = _expand_rect(r, width_limit=20, height_limit=20, dx=3, dy=3)
    # Bottom-right would have been (18, 18) -> stays within 20.
    assert (expanded.left, expanded.top, expanded.right, expanded.bottom) == (2, 2, 18, 18)


def test_expand_rect_clamped_at_origin() -> None:
    r = Rect(1, 1, 5, 5)
    expanded = _expand_rect(r, width_limit=100, height_limit=100, dx=10, dy=10)
    # Top-left can't go below (0, 0).
    assert expanded.left == 0
    assert expanded.top == 0


def test_move_rect_inside_already_inside() -> None:
    r = Rect(5, 5, 3, 3)
    bounds = Rect(0, 0, 20, 20)
    moved = _move_rect_inside(r, bounds)
    assert moved == r


def test_move_rect_inside_off_top_left() -> None:
    r = Rect(-5, -5, 3, 3)
    bounds = Rect(0, 0, 20, 20)
    moved = _move_rect_inside(r, bounds)
    assert (moved.left, moved.top) == (0, 0)
    assert (moved.width, moved.height) == (3, 3)


def test_move_rect_inside_off_bottom_right() -> None:
    r = Rect(18, 18, 5, 5)
    bounds = Rect(0, 0, 20, 20)
    moved = _move_rect_inside(r, bounds)
    assert moved.right <= bounds.right
    assert moved.bottom <= bounds.bottom


def test_same_text_row_vertically_aligned() -> None:
    left = Rect(0, 100, 30, 20)
    right = Rect(50, 100, 30, 20)
    assert _same_text_row(left, right) is True


def test_same_text_row_different_rows() -> None:
    top = Rect(0, 100, 30, 20)
    bottom = Rect(0, 200, 30, 20)
    assert _same_text_row(top, bottom) is False
