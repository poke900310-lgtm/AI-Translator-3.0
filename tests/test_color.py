"""Color and style helper tests against synthetic PIL images."""

from __future__ import annotations

import math

from PIL import Image

from app.color import (
    _build_style,
    _color_distance,
    _contrast_color,
    _crop,
    _delta_e,
    _luminance,
    _median_float,
    _median_int,
    _rgb_to_lab,
    _sample_background_color,
    _sample_rect_mean_rgb,
    _srgb_channel_to_linear,
)
from app.types import Rect, VisualStyle


def _solid(color: tuple[int, int, int], size: tuple[int, int] = (64, 64)) -> Image.Image:
    return Image.new("RGB", size, color)


def _two_band(
    left_color: tuple[int, int, int],
    right_color: tuple[int, int, int],
    size: tuple[int, int] = (64, 64),
) -> Image.Image:
    img = Image.new("RGB", size, left_color)
    for x in range(size[0] // 2, size[0]):
        for y in range(size[1]):
            img.putpixel((x, y), right_color)
    return img


# --- luminance / contrast ---------------------------------------------------


def test_luminance_white() -> None:
    assert _luminance((255, 255, 255)) > 250


def test_luminance_black() -> None:
    assert _luminance((0, 0, 0)) == 0


def test_contrast_color_dark_bg_returns_light() -> None:
    assert _contrast_color((10, 10, 10)) == (245, 245, 245)


def test_contrast_color_light_bg_returns_dark() -> None:
    assert _contrast_color((250, 250, 250)) == (20, 20, 20)


# --- color distance ---------------------------------------------------------


def test_color_distance_zero() -> None:
    assert _color_distance((100, 100, 100), (100, 100, 100)) == 0.0


def test_color_distance_symmetric() -> None:
    a, b = (50, 100, 150), (200, 30, 75)
    assert math.isclose(_color_distance(a, b), _color_distance(b, a), rel_tol=1e-9)


# --- crop ------------------------------------------------------------------


def test_crop_matches_rect() -> None:
    img = _two_band((0, 0, 0), (255, 255, 255))
    crop = _crop(img, Rect(40, 0, 10, 10))
    # The right half is white.
    assert crop.size == (10, 10)
    assert crop.getpixel((0, 0)) == (255, 255, 255)


# --- sampling --------------------------------------------------------------


def test_sample_rect_mean_rgb_solid() -> None:
    img = _solid((100, 150, 200))
    mean = _sample_rect_mean_rgb(img, Rect(0, 0, 32, 32))
    assert mean == (100, 150, 200)


def test_sample_rect_mean_rgb_empty_returns_black() -> None:
    img = _solid((100, 100, 100))
    assert _sample_rect_mean_rgb(img, Rect(0, 0, 0, 0)) == (0, 0, 0)


def test_sample_background_color_uses_surround() -> None:
    # Center 24x24 rect of red text on a uniform gray background — sampler
    # masks out the center and returns the surround mean.
    img = _solid((128, 128, 128))
    text_rect = Rect(20, 20, 24, 24)
    # Inject "text" into the rect.
    for x in range(20, 44):
        for y in range(20, 44):
            img.putpixel((x, y), (200, 0, 0))
    bg = _sample_background_color(img, text_rect)
    # Should still report gray, not the red text.
    r, g, b = bg
    assert abs(r - 128) <= 5
    assert abs(g - 128) <= 5
    assert abs(b - 128) <= 5


# --- median helpers --------------------------------------------------------


def test_median_int_basic() -> None:
    assert _median_int([3, 1, 2]) == 2


def test_median_int_default_on_empty() -> None:
    assert _median_int([], default=7) == 7


def test_median_float_basic() -> None:
    assert _median_float([1.0, 2.0, 3.0]) == 2.0


# --- color spaces ---------------------------------------------------------


def test_srgb_linear_zero_is_zero() -> None:
    assert _srgb_channel_to_linear(0.0) == 0.0


def test_srgb_linear_one_is_one() -> None:
    assert math.isclose(_srgb_channel_to_linear(1.0), 1.0, rel_tol=1e-9)


def test_rgb_to_lab_identity_zero_distance() -> None:
    c = (123, 45, 200)
    lab1 = _rgb_to_lab(c)
    lab2 = _rgb_to_lab(c)
    assert lab1 == lab2


def test_delta_e_identity() -> None:
    c = (123, 45, 200)
    assert math.isclose(_delta_e(c, c), 0.0, abs_tol=1e-9)


def test_delta_e_symmetric() -> None:
    a, b = (100, 50, 30), (200, 180, 90)
    assert math.isclose(_delta_e(a, b), _delta_e(b, a), rel_tol=1e-9)


# --- style build -----------------------------------------------------------


def test_build_style_returns_visual_style() -> None:
    img = _solid((50, 50, 50))
    style = _build_style(img, Rect(10, 10, 20, 20), word_rects=[Rect(12, 12, 16, 16)])
    assert isinstance(style, VisualStyle)
    assert len(style.fill_color) == 3
    assert len(style.outline_color) == 3
    assert len(style.background_color) == 3
