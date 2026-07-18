from __future__ import annotations

from io import BytesIO
from typing import Any

from PIL import Image
from PyQt6 import QtCore, QtGui

from app import config


def pil_to_qimage(image: Image.Image) -> QtGui.QImage:
    rgba = image.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    qimg = QtGui.QImage(data, rgba.width, rgba.height, rgba.width * 4, QtGui.QImage.Format.Format_RGBA8888)
    return qimg.copy()


def qimage_to_pil(image: QtGui.QImage) -> Image.Image:
    qimg = image.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    width = qimg.width()
    height = qimg.height()
    stride = qimg.bytesPerLine()
    size = qimg.sizeInBytes()
    ptr = qimg.bits()

    # PyQt6 stubs type QImage.bits() as ``voidptr | None`` and QImage.save() as
    # taking ``str``, but at runtime both forms work. The except: branch is the
    # belt-and-suspenders fallback if the fast path's voidptr handling fails.
    try:
        if hasattr(ptr, "setsize"):
            ptr.setsize(size)  # type: ignore[union-attr]
        buf = bytes(ptr)  # type: ignore[call-overload]
        return Image.frombuffer("RGBA", (width, height), buf, "raw", "RGBA", stride, 1).copy()
    except Exception:
        buffer = QtCore.QBuffer()
        buffer.open(QtCore.QIODevice.OpenModeFlag.ReadWrite)
        qimg.save(buffer, b"PNG")  # type: ignore[call-overload]
        data = bytes(buffer.data())  # type: ignore[call-overload]
        buffer.close()
        return Image.open(BytesIO(data)).convert("RGBA")


def _font_weight(weight_override: int | None = None) -> QtGui.QFont.Weight:
    weight = int(weight_override if weight_override is not None else getattr(config, "TEXT_FONT_WEIGHT", 600))
    if weight <= 450:
        return QtGui.QFont.Weight.Medium
    if weight <= 650:
        return QtGui.QFont.Weight.DemiBold
    return QtGui.QFont.Weight.Bold


_FONT_CACHE: dict[tuple[str, int, int], QtGui.QFont] = {}
_FONT_METRICS_CACHE: dict[tuple[str, int, int], QtGui.QFontMetrics] = {}


def _make_font(pixel_size: int, family_override: str | None = None, weight_override: int | None = None) -> QtGui.QFont:
    family = str(family_override or getattr(config, "TEXT_FONT_FAMILY", "Segoe UI"))
    px = max(1, int(pixel_size))
    weight = int(weight_override if weight_override is not None else getattr(config, "TEXT_FONT_WEIGHT", 600))
    key = (family, px, weight)
    cached = _FONT_CACHE.get(key)
    if cached is not None:
        return cached
    font = QtGui.QFont(family)
    font.setWeight(_font_weight(weight_override))
    font.setPixelSize(px)
    font.setStyleHint(QtGui.QFont.StyleHint.SansSerif)
    font.setStyleStrategy(QtGui.QFont.StyleStrategy.PreferAntialias)
    # PreferFullHinting sharpens stroke edges, especially at small pixel
    # sizes where the overlay text otherwise looks blurry against the
    # underlying scene. The cost is negligible at the cache hit rate we
    # see (a few dozen distinct (family,px,weight) tuples per session).
    font.setHintingPreference(QtGui.QFont.HintingPreference.PreferFullHinting)
    _FONT_CACHE[key] = font
    return font


def _font_metrics(font: QtGui.QFont) -> QtGui.QFontMetrics:
    key = (font.family(), font.pixelSize(), int(font.weight()))
    cached = _FONT_METRICS_CACHE.get(key)
    if cached is not None:
        return cached
    metrics = QtGui.QFontMetrics(font)
    _FONT_METRICS_CACHE[key] = metrics
    return metrics


def _alignment_flags(mode: str, allow_wrap: bool) -> int:
    """Translate an alignment-name string into Qt flag bits.

    Supported modes:
      "top_left"  — anchor at the patch's top-left corner (default).
      "top_right" — anchor at the patch's top-right corner.
      "left"      — left + vertical-center (legacy mode kept for dialogue).
      "center"    — both axes centered (legacy).

    `allow_wrap=True` adds TextWordWrap.
    """
    mode = (mode or "top_left").lower()
    if mode == "top_right":
        flags = QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignTop
    elif mode == "top_left":
        flags = QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop
    elif mode == "left":
        flags = QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
    else:  # "center"
        flags = QtCore.Qt.AlignmentFlag.AlignCenter
    if allow_wrap:
        flags |= QtCore.Qt.TextFlag.TextWordWrap
    return flags


def _verticalize_text(text: str) -> str:
    lines: list[str] = []
    for block in (text or "").splitlines() or [text or ""]:
        compact = block.strip()
        if not compact:
            continue
        chars = [ch for ch in compact if ch != " "]
        lines.append("\n".join(chars))
    return "\n\n".join(lines) if lines else (text or "")


def _fits(metrics: QtGui.QFontMetrics, rect: QtCore.QRect, flags: int, text: str) -> bool:
    br = metrics.boundingRect(rect, flags, text)
    return br.width() <= rect.width() and br.height() <= rect.height()


def _resolve_layout(item: dict[str, Any]) -> tuple[QtGui.QFont, int, QtCore.QRect, str]:
    text_rect = item["text_rect"]
    draw_rect = QtCore.QRect(text_rect.left, text_rect.top, text_rect.width, text_rect.height)
    writing_mode = str(item.get("writing_mode") or "horizontal")
    text = item["text"] if writing_mode != "vertical" else _verticalize_text(item["text"])
    min_px = int(getattr(config, "TEXT_MIN_PIXEL", getattr(config, "TEXT_MIN_POINT", 10)))
    max_px = int(getattr(config, "TEXT_MAX_PIXEL", getattr(config, "TEXT_MAX_POINT", 28)))
    preferred = int(item.get("preferred_pixel_size") or max_px)
    preferred = max(min_px, min(max_px, preferred))
    # Cap how far the fit loop can shrink from the source-derived preferred.
    # TEXT_MAX_SHRINK_RATIO=0.85 means a 22px source can drop to 18px but not
    # to 12px. Keeps font sizes visually uniform across adjacent translation
    # boxes — the prior unbounded shrink made one dialog read at 22px and an
    # adjacent name tag at 10px just because the latter's translation was
    # longer.
    shrink_ratio = float(getattr(config, "TEXT_MAX_SHRINK_RATIO", 0.85))
    shrink_floor = max(min_px, int(round(preferred * shrink_ratio)))
    allow_wrap = bool(item.get("allow_wrap", True)) and writing_mode != "vertical"
    alignment = str(item.get("alignment") or getattr(config, "TEXT_DEFAULT_ALIGNMENT", "top_left"))
    flags = int(_alignment_flags(alignment, allow_wrap))

    family = item.get("font_family") or None
    weight = item.get("font_weight") or None
    font = _make_font(preferred, family_override=family, weight_override=weight)
    metrics = _font_metrics(font)
    if _fits(metrics, draw_rect, flags, text):
        return font, flags, draw_rect, text

    for px in range(preferred - 1, shrink_floor - 1, -1):
        font = _make_font(px, family_override=family, weight_override=weight)
        metrics = _font_metrics(font)
        if _fits(metrics, draw_rect, flags, text):
            return font, flags, draw_rect, text

    if writing_mode != "vertical" and not allow_wrap and (" " in text or "-" in text):
        flags = int(_alignment_flags(item.get("alignment") or "top_left", True))
        for px in range(preferred, shrink_floor - 1, -1):
            font = _make_font(px, family_override=family, weight_override=weight)
            metrics = _font_metrics(font)
            if _fits(metrics, draw_rect, flags, text):
                item["allow_wrap"] = True
                item["alignment"] = item.get("alignment") or "top_left"
                return font, flags, draw_rect, text

    # Last resort — render at the shrink floor and accept the clip rather
    # than collapsing to TEXT_MIN_PIXEL. The user explicitly asked for size
    # uniformity; a slightly clipped overlay reads better than a wildly
    # downsized one.
    return _make_font(shrink_floor, family_override=family, weight_override=weight), flags, draw_rect, text


def _outline_offsets(radius: int) -> list[tuple[int, int]]:
    """All ring offsets at distance == radius (cached per radius)."""
    cached = _OUTLINE_OFFSET_CACHE.get(radius)
    if cached is not None:
        return cached
    offsets = [
        (dx, dy)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        if (dx != 0 or dy != 0) and max(abs(dx), abs(dy)) == radius
    ]
    _OUTLINE_OFFSET_CACHE[radius] = offsets
    return offsets


_OUTLINE_OFFSET_CACHE: dict[int, list[tuple[int, int]]] = {}


def paint_overlay_items(painter: QtGui.QPainter, items: list[dict[str, Any]]) -> None:
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)
    painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
    base_outline = max(1, int(getattr(config, "TEXT_OUTLINE_WIDTH", 1)))
    outline_divisor = max(1, int(getattr(config, "TEXT_OUTLINE_SCALE_DIVISOR", 14)))

    for item in items:
        patch_rect = item["patch_rect"]
        patch_img = item["patch_image"]
        painter.drawImage(
            QtCore.QRect(patch_rect.left, patch_rect.top, patch_rect.width, patch_rect.height),
            patch_img,
        )
        for extra_rect, extra_img in item.get("extra_patches") or []:
            painter.drawImage(
                QtCore.QRect(extra_rect.left, extra_rect.top, extra_rect.width, extra_rect.height),
                extra_img,
            )

        text = item["text"]
        if not text:
            continue

        fill = item["fill"]
        outline = item["outline"]
        font, flags, draw_rect, draw_text = _resolve_layout(item)
        # Outline radius scales with font px size so a 26px headline gets a
        # proportional 2px halo instead of the same 1px halo as 12px sign
        # text. Keeps the visual weight of the outline matched to the body.
        outline_radius = max(base_outline, int(round(font.pixelSize() / outline_divisor)))
        offsets = _outline_offsets(outline_radius)
        painter.setFont(font)
        painter.setPen(QtGui.QPen(QtGui.QColor(*outline), 1))
        for dx, dy in offsets:
            painter.drawText(draw_rect.translated(dx, dy), flags, draw_text)
        painter.setPen(QtGui.QPen(QtGui.QColor(*fill), 1))
        painter.drawText(draw_rect, flags, draw_text)
