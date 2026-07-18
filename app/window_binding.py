"""Minimal Win32 helpers for binding capture/overlay state to a target window.

This module keeps the app on one workflow:
- choose one top-level target window (HWND)
- read its client area in screen coordinates
- capture that client area from the desktop compositor
- align the translation overlay to the same client area

It also exposes small diagnostics helpers so capture failures can be logged with
useful window metadata instead of a single opaque exception string.
"""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from dataclasses import dataclass

user32 = ctypes.windll.user32

dwmapi: ctypes.WinDLL | None
try:
    dwmapi = ctypes.windll.dwmapi
except Exception:
    dwmapi = None


DWMWA_CLOAKED = 14
WDA_EXCLUDEFROMCAPTURE = 0x11

GWL_EXSTYLE = -20
GWLP_HWNDPARENT = -8
WS_EX_TOOLWINDOW = 0x00000080
HWND_TOP = wintypes.HWND(0)

GA_ROOT = 2
GW_HWNDPREV = 3
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040
SWP_NOOWNERZORDER = 0x0200
SWP_NOSENDCHANGING = 0x0400

HWND_TOPMOST = wintypes.HWND(-1)
HWND_NOTOPMOST = wintypes.HWND(-2)


class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]  # noqa: RUF012  # ctypes pattern


class RECT(ctypes.Structure):
    _fields_ = [  # noqa: RUF012  # ctypes pattern
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    pid: int
    title: str


@dataclass(frozen=True)
class ClientRect:
    left: int
    top: int
    width: int
    height: int


@dataclass(frozen=True)
class WindowDiagnostics:
    hwnd: int
    pid: int
    title: str
    class_name: str
    visible: bool
    iconic: bool
    cloaked: bool
    tool_window: bool
    client_rect: ClientRect | None


@dataclass(frozen=True)
class OcclusionState:
    occluded: bool
    sampled_points: int
    foreign_samples: int
    foreign_hwnd: int
    foreign_title: str


def _window_text(hwnd: int) -> str:
    length = user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, len(buf))
    return buf.value.strip()


def _window_pid(hwnd: int) -> int:
    pid = wintypes.DWORD(0)
    user32.GetWindowThreadProcessId(wintypes.HWND(hwnd), ctypes.byref(pid))
    return int(pid.value)


def _window_class(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(256)
    n = user32.GetClassNameW(wintypes.HWND(hwnd), buf, len(buf))
    if n <= 0:
        return ""
    return buf.value.strip()


def _is_cloaked(hwnd: int) -> bool:
    if dwmapi is None:
        return False
    cloaked = wintypes.DWORD(0)
    try:
        hr = dwmapi.DwmGetWindowAttribute(
            wintypes.HWND(hwnd),
            wintypes.DWORD(DWMWA_CLOAKED),
            ctypes.byref(cloaked),
            ctypes.sizeof(cloaked),
        )
        return hr == 0 and bool(cloaked.value)
    except Exception:
        return False


def _is_tool_window(hwnd: int) -> bool:
    exstyle = user32.GetWindowLongW(wintypes.HWND(hwnd), GWL_EXSTYLE)
    return bool(exstyle & WS_EX_TOOLWINDOW)


def _root_window(hwnd: int) -> int:
    if not hwnd:
        return 0
    try:
        root = int(user32.GetAncestor(wintypes.HWND(hwnd), GA_ROOT) or 0)
    except Exception:
        root = 0
    return root or int(hwnd)


def set_owner_window(window_hwnd: int, owner_hwnd: int) -> bool:
    """Bind a top-level overlay window to the target as an owned popup.

    Owned popups stay above their owner in the normal z-order band while still
    remaining below unrelated windows that are brought to the foreground.
    Passing ``owner_hwnd=0`` clears the owner.
    """

    if not window_hwnd:
        return False
    try:
        setter = getattr(user32, "SetWindowLongPtrW", None)
        if setter is None:
            setter = user32.SetWindowLongW
        setter(wintypes.HWND(window_hwnd), GWLP_HWNDPARENT, wintypes.HWND(int(owner_hwnd or 0)))
        user32.SetWindowPos(
            wintypes.HWND(window_hwnd),
            HWND_TOP,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_NOOWNERZORDER | SWP_NOSENDCHANGING,
        )
        return True
    except Exception:
        return False


def _iter_sample_points(rect: ClientRect, cols: int, rows: int, inset_px: int):
    if rect.width <= 0 or rect.height <= 0:
        return
    inset_x = max(1, min(int(inset_px), max(1, rect.width // 4)))
    inset_y = max(1, min(int(inset_px), max(1, rect.height // 4)))
    left = rect.left + inset_x
    right = rect.left + rect.width - inset_x - 1
    top = rect.top + inset_y
    bottom = rect.top + rect.height - inset_y - 1
    if right < left:
        left = right = rect.left + max(0, rect.width // 2)
    if bottom < top:
        top = bottom = rect.top + max(0, rect.height // 2)
    xs = [left] if cols <= 1 else [int(round(left + (right - left) * i / max(1, cols - 1))) for i in range(cols)]
    ys = [top] if rows <= 1 else [int(round(top + (bottom - top) * j / max(1, rows - 1))) for j in range(rows)]
    seen: set[tuple[int, int]] = set()
    for y in ys:
        for x in xs:
            pt = (int(x), int(y))
            if pt in seen:
                continue
            seen.add(pt)
            yield pt


def list_candidate_windows() -> list[WindowInfo]:
    """Return visible, titled top-level windows suitable for capture binding."""

    current_pid = os.getpid()
    out: list[WindowInfo] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _enum_proc(hwnd, lparam):
        ihwnd = int(hwnd)
        if not user32.IsWindowVisible(hwnd):
            return True
        if user32.IsIconic(hwnd):
            return True
        if _is_cloaked(ihwnd):
            return True
        if _is_tool_window(ihwnd):
            return True
        title = _window_text(ihwnd)
        if not title:
            return True
        pid = _window_pid(ihwnd)
        if pid == current_pid:
            return True
        out.append(WindowInfo(hwnd=ihwnd, pid=pid, title=title))
        return True

    user32.EnumWindows(_enum_proc, 0)
    out.sort(key=lambda w: w.title.lower())
    return out


def is_window_usable(hwnd: int) -> bool:
    return bool(
        hwnd
        and user32.IsWindow(wintypes.HWND(hwnd))
        and user32.IsWindowVisible(wintypes.HWND(hwnd))
        and not user32.IsIconic(wintypes.HWND(hwnd))
    )


def get_client_rect_screen(hwnd: int) -> ClientRect | None:
    """Return the target window's client area in absolute screen pixels."""

    if not is_window_usable(hwnd):
        return None

    rect = RECT()
    if not user32.GetClientRect(wintypes.HWND(hwnd), ctypes.byref(rect)):
        return None

    tl = POINT(rect.left, rect.top)
    br = POINT(rect.right, rect.bottom)
    if not user32.ClientToScreen(wintypes.HWND(hwnd), ctypes.byref(tl)):
        return None
    if not user32.ClientToScreen(wintypes.HWND(hwnd), ctypes.byref(br)):
        return None

    width = max(1, int(br.x - tl.x))
    height = max(1, int(br.y - tl.y))
    return ClientRect(left=int(tl.x), top=int(tl.y), width=width, height=height)


def detect_window_occlusion(
    hwnd: int,
    *,
    ignore_hwnds: tuple[int, ...] = (),
    sample_cols: int = 3,
    sample_rows: int = 3,
    inset_px: int = 12,
) -> OcclusionState:
    """Detect whether another top-level window is covering the target client area.

    This is intentionally lightweight: it samples a handful of points inside the
    target client rect and checks which top-level window owns those pixels.
    """

    rect = get_client_rect_screen(hwnd)
    if rect is None:
        return OcclusionState(False, 0, 0, 0, "")

    target_root = _root_window(int(hwnd))
    ignored_roots = {_root_window(int(h)) for h in ignore_hwnds if h}
    foreign_counts: dict[int, int] = {}
    sampled = 0

    for x, y in _iter_sample_points(rect, max(1, int(sample_cols)), max(1, int(sample_rows)), max(0, int(inset_px))):
        hit = int(user32.WindowFromPoint(POINT(int(x), int(y))) or 0)
        if not hit:
            continue
        sampled += 1
        root = _root_window(hit)
        if not root:
            continue
        if root in ignored_roots:
            continue
        if root == target_root:
            continue
        if user32.IsChild(wintypes.HWND(hwnd), wintypes.HWND(hit)):
            continue
        foreign_counts[root] = foreign_counts.get(root, 0) + 1

    if not foreign_counts:
        return OcclusionState(False, sampled, 0, 0, "")

    foreign_hwnd = max(foreign_counts.items(), key=lambda item: item[1])[0]
    foreign_samples = int(sum(foreign_counts.values()))
    return OcclusionState(
        True,
        sampled,
        foreign_samples,
        foreign_hwnd,
        _window_text(foreign_hwnd),
    )


def stack_window_above_target(window_hwnd: int, target_hwnd: int) -> bool:
    """Place an overlay window directly above the target in z-order, without topmost.

    SetWindowPos expects ``hWndInsertAfter`` to identify the window *behind* the
    positioned window. So to land the overlay immediately above the target, we
    must insert it behind the window that is currently above the target. If the
    target is already at the top of its band, we use HWND_TOP.
    """

    if not window_hwnd or not target_hwnd:
        return False
    try:
        prev_hwnd = int(user32.GetWindow(wintypes.HWND(target_hwnd), GW_HWNDPREV) or 0)
        insert_after = wintypes.HWND(prev_hwnd) if prev_hwnd else HWND_TOP
        return bool(
            user32.SetWindowPos(
                wintypes.HWND(window_hwnd),
                insert_after,
                0,
                0,
                0,
                0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW | SWP_NOOWNERZORDER | SWP_NOSENDCHANGING,
            )
        )
    except Exception:
        return False


def set_window_topmost(window_hwnd: int, topmost: bool) -> bool:
    """Toggle a window in/out of the Windows topmost z-order band.

    Used to raise the overlay above unrelated foreground programs while the
    target is active, and to drop it back to normal band when the user
    switches away. Owned-popup parenting alone can't cross the topmost
    boundary, so this is the mechanism that fixes the "overlay renders
    beneath another program" case.
    """

    if not window_hwnd:
        return False
    try:
        insert_after = HWND_TOPMOST if topmost else HWND_NOTOPMOST
        return bool(
            user32.SetWindowPos(
                wintypes.HWND(window_hwnd),
                insert_after,
                0,
                0,
                0,
                0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_NOOWNERZORDER | SWP_NOSENDCHANGING,
            )
        )
    except Exception:
        return False


def is_target_foreground(target_hwnd: int) -> bool:
    """Return True if the target window (or one of its ancestors/descendants)
    is currently the Windows foreground window.

    ``target_hwnd`` is the top-level HWND the overlay is bound to. We compare
    root ancestors so a foreground child window (dialog, MDI child) of the
    same top-level still counts as "target is active."
    """

    if not target_hwnd:
        return False
    try:
        fg = int(user32.GetForegroundWindow() or 0)
    except Exception:
        return False
    if not fg:
        return False
    return _root_window(fg) == _root_window(int(target_hwnd))


def get_window_diagnostics(hwnd: int) -> WindowDiagnostics:
    """Return a snapshot of target-window metadata useful for debug logs."""

    return WindowDiagnostics(
        hwnd=int(hwnd),
        pid=_window_pid(int(hwnd)) if hwnd else 0,
        title=_window_text(int(hwnd)) if hwnd else "",
        class_name=_window_class(int(hwnd)) if hwnd else "",
        visible=bool(hwnd and user32.IsWindowVisible(wintypes.HWND(hwnd))),
        iconic=bool(hwnd and user32.IsIconic(wintypes.HWND(hwnd))),
        cloaked=_is_cloaked(int(hwnd)) if hwnd else False,
        tool_window=_is_tool_window(int(hwnd)) if hwnd else False,
        client_rect=get_client_rect_screen(int(hwnd)) if hwnd else None,
    )


def format_window_diagnostics(diag: WindowDiagnostics) -> str:
    rect = diag.client_rect
    rect_text = (
        f"left={rect.left}, top={rect.top}, width={rect.width}, height={rect.height}" if rect is not None else "None"
    )
    return (
        f"hwnd=0x{diag.hwnd:08X}, pid={diag.pid}, title={diag.title!r}, class={diag.class_name!r}, "
        f"visible={diag.visible}, iconic={diag.iconic}, cloaked={diag.cloaked}, "
        f"tool_window={diag.tool_window}, client_rect={rect_text}"
    )


def apply_exclude_from_capture(hwnd: int) -> bool:
    """Hide a top-level overlay window from supported Windows capture paths."""

    try:
        return bool(user32.SetWindowDisplayAffinity(wintypes.HWND(hwnd), WDA_EXCLUDEFROMCAPTURE))
    except Exception:
        return False


def set_window_enabled(hwnd: int, enabled: bool) -> bool:
    """Enable or disable a top-level window safely."""

    if not hwnd:
        return False
    try:
        return bool(user32.EnableWindow(wintypes.HWND(int(hwnd)), wintypes.BOOL(bool(enabled))))
    except Exception:
        return False
