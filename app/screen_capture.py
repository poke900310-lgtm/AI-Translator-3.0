"""Window-client capture for the translator.

This build uses one single capture workflow:
- bind to one target window
- resolve its client area in screen coordinates
- capture those pixels directly from the desktop compositor

This avoids the `PrintWindow` failures that occur with many GPU-accelerated
windows while still keeping the whole app attached to a single target window.
The tradeoff is that the target window must actually be visible on screen.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass

from PIL import Image

from app import config
from app.window_binding import get_client_rect_screen

SRCCOPY = 0x00CC0020
CAPTUREBLT = 0x40000000
DIB_RGB_COLORS = 0
BI_RGB = 0

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [  # noqa: RUF012  # ctypes introspects this; ClassVar wrapper is unnecessary
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [  # noqa: RUF012  # ctypes introspects this; ClassVar wrapper is unnecessary
        ("bmiHeader", BITMAPINFOHEADER),
        ("bmiColors", wintypes.DWORD * 3),
    ]


@dataclass(frozen=True)
class Region:
    left: int
    top: int
    width: int
    height: int


def _raise_last_error(prefix: str) -> None:
    code = ctypes.GetLastError() or ctypes.get_last_error()
    if code:
        raise RuntimeError(f"{prefix} (winerr={code}: {ctypes.FormatError(code).strip()})")
    raise RuntimeError(prefix)


def grab_window_client(hwnd: int) -> Image.Image:
    """Capture the full client area of the attached target window.

    The target window must be visible because pixels are copied from the
    composed desktop at the target window's client-area coordinates.
    """

    rect = get_client_rect_screen(int(hwnd))
    if rect is None:
        raise RuntimeError("Target client rect unavailable (window hidden, minimized, or invalid)")
    width = max(1, int(rect.width))
    height = max(1, int(rect.height))

    screen_dc = user32.GetDC(0)
    if not screen_dc:
        _raise_last_error("GetDC(0) failed")

    mem_dc = gdi32.CreateCompatibleDC(screen_dc)
    if not mem_dc:
        user32.ReleaseDC(0, screen_dc)
        _raise_last_error("CreateCompatibleDC failed")

    bitmap = gdi32.CreateCompatibleBitmap(screen_dc, width, height)
    if not bitmap:
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(0, screen_dc)
        _raise_last_error("CreateCompatibleBitmap failed")

    old_obj = gdi32.SelectObject(mem_dc, bitmap)
    if not old_obj:
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(0, screen_dc)
        _raise_last_error("SelectObject failed")

    try:
        ok = gdi32.BitBlt(
            mem_dc,
            0,
            0,
            width,
            height,
            screen_dc,
            int(rect.left),
            int(rect.top),
            SRCCOPY | (CAPTUREBLT if bool(getattr(config, "CAPTURE_INCLUDE_LAYERED_WINDOWS", False)) else 0),
        )
        if not ok:
            _raise_last_error("BitBlt failed")

        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = width
        bmi.bmiHeader.biHeight = -height  # top-down
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = BI_RGB
        bmi.bmiHeader.biSizeImage = width * height * 4

        buffer = ctypes.create_string_buffer(width * height * 4)
        lines = gdi32.GetDIBits(
            mem_dc,
            bitmap,
            0,
            height,
            buffer,
            ctypes.byref(bmi),
            DIB_RGB_COLORS,
        )
        if lines != height:
            _raise_last_error(f"GetDIBits failed (got {lines} of {height} scan lines)")

        return Image.frombuffer("RGBA", (width, height), buffer, "raw", "BGRA", 0, 1).convert("RGB")
    finally:
        gdi32.SelectObject(mem_dc, old_obj)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(0, screen_dc)
