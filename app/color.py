"""Color sampling, luminance, and style helpers.

Extracted from app.controller. Pure functions: take PIL Image + Rect, return
RGB tuples or VisualStyle. No Controller state; only a couple of standard
library functions (functools.lru_cache, math) and numpy for the rect-mean
samplers.
"""

from __future__ import annotations

import functools
import math

import numpy as np
from PIL import Image

from app.types import Rect, VisualStyle


def _luminance(rgb: tuple[int, int, int]) -> float:
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast_color(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return (20, 20, 20) if _luminance(rgb) > 150 else (245, 245, 245)


@functools.lru_cache(maxsize=512)
def _color_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b, strict=True)))


def _crop(image: Image.Image, rect: Rect) -> Image.Image:
    return image.crop((rect.left, rect.top, rect.right, rect.bottom))


def _sample_background_color(image: Image.Image, rect: Rect) -> tuple[int, int, int]:
    outer_left = max(0, rect.left - 8)
    outer_top = max(0, rect.top - 6)
    outer_right = min(image.width, rect.right + 8)
    outer_bottom = min(image.height, rect.bottom + 6)
    arr = np.array(image)[outer_top:outer_bottom, outer_left:outer_right, :3]
    if arr.size == 0:
        return (0, 0, 0)
    h, w = arr.shape[:2]
    mask = np.ones((h, w), dtype=bool)
    il = max(0, rect.left - outer_left)
    it = max(0, rect.top - outer_top)
    ir = min(w, rect.right - outer_left)
    ib = min(h, rect.bottom - outer_top)
    if il < ir and it < ib:
        mask[it:ib, il:ir] = False
    pixels = arr[mask]
    if len(pixels) == 0:
        return (0, 0, 0)
    mean = pixels.mean(axis=0)
    return (int(round(float(mean[0]))), int(round(float(mean[1]))), int(round(float(mean[2]))))


def _sample_foreground_color(
    image: Image.Image,
    word_rects: list[Rect],
    background: tuple[int, int, int],
) -> tuple[int, int, int]:
    candidates: list[tuple[int, tuple[int, int, int]]] = []
    for rect in word_rects[:12]:
        crop = _crop(image, rect).convert("RGB")
        try:
            palette = crop.quantize(colors=4, method=Image.Quantize.MEDIANCUT).convert("RGB")
            colors = palette.getcolors(maxcolors=64) or []
        except Exception:
            colors = []
        for count, rgb in colors:
            rgb_tuple: tuple[int, int, int] = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
            candidates.append((int(count), rgb_tuple))
    if not candidates:
        return _contrast_color(background)
    best_color = None
    best_score = -1.0
    for count, rgb in candidates:
        dist = _color_distance(rgb, background)
        score = dist * max(1, count)
        if score > best_score:
            best_score = score
            best_color = rgb
    if best_color is None or _color_distance(best_color, background) < 28:
        return _contrast_color(background)
    return best_color


def _sample_outline_color(fill: tuple[int, int, int], background: tuple[int, int, int]) -> tuple[int, int, int]:
    if _color_distance(fill, background) < 70:
        return _contrast_color(fill)
    return (10, 10, 10) if _luminance(fill) > 160 else (245, 245, 245)


def _build_style(image: Image.Image, rect: Rect, word_rects: list[Rect]) -> VisualStyle:
    background = _sample_background_color(image, rect)
    fill = _sample_foreground_color(image, word_rects or [rect], background)
    outline = _sample_outline_color(fill, background)
    return VisualStyle(fill_color=fill, outline_color=outline, background_color=background)


def _median_int(values: list[int], default: int = 0) -> int:
    cleaned = sorted(int(v) for v in values if int(v) > 0)
    if not cleaned:
        return int(default)
    return cleaned[len(cleaned) // 2]


def _median_float(values: list[float], default: float = 0.0) -> float:
    cleaned = sorted(float(v) for v in values if float(v) > 0.0)
    if not cleaned:
        return float(default)
    return cleaned[len(cleaned) // 2]


def _sample_rect_mean_rgb(image: Image.Image, rect: Rect) -> tuple[int, int, int]:
    if rect.width <= 0 or rect.height <= 0:
        return (0, 0, 0)
    arr = np.array(image)[rect.top : rect.bottom, rect.left : rect.right, :3]
    if arr.size == 0:
        return (0, 0, 0)
    mean = arr.mean(axis=(0, 1))
    return (int(round(float(mean[0]))), int(round(float(mean[1]))), int(round(float(mean[2]))))


def _srgb_channel_to_linear(value: float) -> float:
    value = max(0.0, min(1.0, float(value)))
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def _rgb_to_lab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    r = _srgb_channel_to_linear(rgb[0] / 255.0)
    g = _srgb_channel_to_linear(rgb[1] / 255.0)
    b = _srgb_channel_to_linear(rgb[2] / 255.0)
    x = (r * 0.4124564) + (g * 0.3575761) + (b * 0.1804375)
    y = (r * 0.2126729) + (g * 0.7151522) + (b * 0.0721750)
    z = (r * 0.0193339) + (g * 0.1191920) + (b * 0.9503041)
    xr = x / 0.95047
    yr = y / 1.0
    zr = z / 1.08883

    def _f(t: float) -> float:
        if t > 0.008856:
            return t ** (1.0 / 3.0)
        return (7.787 * t) + (16.0 / 116.0)

    fx = _f(xr)
    fy = _f(yr)
    fz = _f(zr)
    return ((116.0 * fy) - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))


@functools.lru_cache(maxsize=512)
def _delta_e(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    al, aa, ab = _rgb_to_lab(a)
    bl, ba, bb = _rgb_to_lab(b)
    return math.sqrt(((al - bl) ** 2) + ((aa - ba) ** 2) + ((ab - bb) ** 2))


__all__ = [
    "_build_style",
    "_color_distance",
    "_contrast_color",
    "_crop",
    "_delta_e",
    "_luminance",
    "_median_float",
    "_median_int",
    "_rgb_to_lab",
    "_sample_background_color",
    "_sample_foreground_color",
    "_sample_outline_color",
    "_sample_rect_mean_rgb",
    "_srgb_channel_to_linear",
]
