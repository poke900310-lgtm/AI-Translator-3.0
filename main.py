"""Main entry point for the full-window in-place translator.

This build keeps the embedded One OCR engine and redesigns the rest of the
workflow around one attached target window. The whole client area is captured,
text is tracked over time, and translations are painted back over the original
text regions.
"""

from __future__ import annotations

import atexit
import ctypes
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

try:
    import keyboard as _keyboard

    _KEYBOARD_AVAILABLE = True
except ImportError:
    _keyboard = None  # type: ignore[assignment]
    _KEYBOARD_AVAILABLE = False

from PyQt6 import QtCore, QtWidgets

from app import VERSION, config
from app.controller import Controller
from app.windows import ControlPanel, RegionSelectionWindow, TranslationOverlay, ZoneManagerWindow

_SOURCE_LANGUAGE_CHOICES: list[tuple[str, str]] = [
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("zh-Hans", "Chinese (Simplified)"),
    ("zh-Hant", "Chinese (Traditional)"),
    ("en", "English"),
    ("0", "Auto detect / no hint"),
]


def initial_language_hint() -> str | None:
    env_lang = (os.environ.get("AI_TRANSLATE_LANG") or "").strip()
    valid = {tag for tag, _ in _SOURCE_LANGUAGE_CHOICES}
    if env_lang not in valid:
        return "ja"
    return None if env_lang == "0" else env_lang


def _enable_dpi_awareness() -> None:
    try:
        try:
            ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
            return
        except Exception:
            pass
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
            return
        except Exception:
            pass
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
    except Exception:
        pass


_LLAMA_PROC = None


def _health_ok(base_url: str, timeout_s: int = 2) -> bool:
    try:
        req = urllib.request.Request(base_url.rstrip("/") + "/health")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.status == 200
    except Exception:
        return False


def _running_server_model_ids(base_url: str, timeout_s: int = 2) -> list[str] | None:
    """Return the list of model ids served by an already-running llama-server,
    or None if the endpoint wasn't reachable. llama.cpp's OpenAI-compatible
    ``/v1/models`` returns ``{"object": "list", "data": [{"id": "...", ...}]}``.
    """
    try:
        req = urllib.request.Request(base_url.rstrip("/") + "/v1/models")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            if resp.status != 200:
                return None
            import json as _json

            payload = _json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return None
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return None
    ids: list[str] = []
    for entry in data:
        if isinstance(entry, dict):
            entry_id = entry.get("id")
            if isinstance(entry_id, str) and entry_id.strip():
                ids.append(entry_id.strip())
    return ids


def _warn_if_running_server_model_mismatches(base_url: str, model_path: str) -> None:
    """If an externally-managed llama-server is already running, sanity-check
    that it's serving the model we expected. A stale server left over from a
    previous session may be loaded with a different GGUF — translations would
    silently come back from the wrong model. We do not abort (the user may
    have intentionally pointed at a custom server), just print a clear
    warning that explains how to fix it.
    """
    served = _running_server_model_ids(base_url)
    if served is None:
        return  # /v1/models unavailable — older llama.cpp build, skip silently.
    expected_stem = Path(model_path).stem if model_path else ""
    if not expected_stem:
        return
    for served_id in served:
        served_stem = Path(served_id).stem
        if expected_stem == served_stem or expected_stem in served_id:
            return
    print(
        "WARNING: llama-server at "
        f"{base_url} is serving {served!r} but the project is configured "
        f"for model stem {expected_stem!r}. If translations come out in "
        "the wrong language or garbled, stop the stale server and let "
        "this launcher start a fresh one with the configured model.",
        file=sys.stderr,
    )


def _find_llama_server_exe(project_root: Path) -> Path:
    binaries_root = project_root / "llama_cpp" / "binaries"
    if not binaries_root.exists():
        raise FileNotFoundError(f"Missing llama_cpp binaries folder: {binaries_root}")
    for p in binaries_root.rglob("llama-server.exe"):
        return p
    raise FileNotFoundError(f"Could not find llama-server.exe under: {binaries_root}")


def ensure_llama_server_running(project_root: Path, base_url: str, model_path: str) -> None:
    global _LLAMA_PROC
    if _health_ok(base_url):
        # Accept an already-running server but warn loudly if it's serving a
        # different model than we expect (stale process from a prior session).
        _warn_if_running_server_model_mismatches(base_url, model_path)
        return

    exe = _find_llama_server_exe(project_root)
    mp = Path(model_path).expanduser()
    if not mp.is_absolute():
        mp = (project_root / mp).resolve()
    else:
        mp = mp.resolve()
    if not mp.exists():
        raise FileNotFoundError(f"Model file not found: {mp}")

    u = urlparse(base_url)
    port = u.port or 8080
    host = u.hostname or "127.0.0.1"
    args = [
        str(exe),
        "-m",
        str(mp),
        "-c",
        str(getattr(config, "LLAMA_SERVER_CTX", 2048)),
        "-ngl",
        str(getattr(config, "LLAMA_SERVER_NGL", "99")),
        "-fa",
        str(getattr(config, "LLAMA_SERVER_FA", "on")),
        "--host",
        host,
        "--port",
        str(port),
    ]

    _LLAMA_PROC = subprocess.Popen(args, cwd=str(exe.parent))

    def _cleanup() -> None:
        global _LLAMA_PROC
        try:
            if _LLAMA_PROC and _LLAMA_PROC.poll() is None:
                _LLAMA_PROC.terminate()
        except Exception:
            pass

    atexit.register(_cleanup)

    deadline = time.time() + 120
    while time.time() < deadline:
        if _health_ok(base_url, timeout_s=2):
            return
        if _LLAMA_PROC.poll() is not None:
            raise RuntimeError("llama-server exited during startup. Check the console output for details.")
        time.sleep(0.5)
    raise TimeoutError(f"llama-server did not become ready in time at {base_url}/health")


def main() -> None:
    _enable_dpi_awareness()

    backend = getattr(config, "TRANSLATION_BACKEND", "llama_server").lower().strip()
    if backend != "llama_server":
        raise RuntimeError(f"Unsupported TRANSLATION_BACKEND: {backend} (this build supports llama_server only)")

    project_root = Path(__file__).resolve().parent
    # app/config.py already reads LLAMA_SERVER_MODEL_PATH from the env at
    # import time. Consume it directly here so there's a single source of
    # truth and the env var doesn't end up read twice on every startup.
    ensure_llama_server_running(project_root, config.LLAMA_SERVER_BASE_URL, config.LLAMA_SERVER_MODEL_PATH)

    env_model = os.environ.get("AI_TRANSLATE_MODEL") or os.environ.get("LLAMA_SERVER_MODEL")
    if env_model:
        config.LLAMA_SERVER_MODEL = env_model

    print(f"OverlayTranslator2 v{VERSION}")
    print(
        f"Using llama-server: {config.LLAMA_SERVER_BASE_URL} "
        f"(model label: {getattr(config, 'LLAMA_SERVER_MODEL', 'local-model')})"
    )

    lang_tag = initial_language_hint()
    os.environ["AI_TRANSLATE_LANG"] = lang_tag or "0"

    app = QtWidgets.QApplication(sys.argv)

    controls = ControlPanel(language_choices=_SOURCE_LANGUAGE_CHOICES, current_lang=(lang_tag or "0"))
    controls.move(80, 80)
    controls.show()

    overlay = TranslationOverlay()
    region_editor = RegionSelectionWindow()
    zone_manager = ZoneManagerWindow()

    ctl = Controller(lang_tag=lang_tag)
    ctl.set_overlay_window(int(overlay.winId()))
    app.aboutToQuit.connect(ctl.stop)
    controls.attachRequested.connect(overlay.set_target_window)
    controls.attachRequested.connect(ctl.attach_target)
    controls.pauseToggled.connect(ctl.set_paused)
    controls.languageChanged.connect(ctl.set_language_hint)
    controls.settingsApplied.connect(ctl.set_edge_ignore_padding)
    controls.zonesRequested.connect(lambda: (zone_manager.show(), zone_manager.raise_(), zone_manager.activateWindow()))

    ctl.clientRegionUpdated.connect(overlay.set_client_region)
    ctl.clientRegionUpdated.connect(region_editor.set_client_region)
    ctl.targetAttached.connect(lambda attached: overlay.set_target_window(ctl.state.target_hwnd if attached else 0))
    ctl.targetAttached.connect(
        lambda attached: region_editor.set_target_window(ctl.state.target_hwnd if attached else 0)
    )
    ctl.overlaySceneUpdated.connect(overlay.set_scene)
    ctl.statusUpdated.connect(controls.set_status)
    ctl.targetAttached.connect(controls.set_attached)
    ctl.pausedStateChanged.connect(controls.set_paused_state)

    def _apply_zone_lists(ignore_regions, translation_zones) -> None:
        ctl.set_ignore_regions(ignore_regions)
        ctl.set_translation_zones(translation_zones)
        region_editor.set_ignore_regions(ignore_regions)
        region_editor.set_translation_zones(translation_zones)
        zone_manager.set_ignore_regions(ignore_regions)
        zone_manager.set_translation_zones(translation_zones)

    def _begin_zone_edit(kind: str, index, rect) -> None:
        ctl.begin_region_edit(kind)
        if ctl.state.target_hwnd and getattr(ctl, "_region_edit_active", False):
            region_editor.set_ignore_regions(zone_manager.ignore_regions())
            region_editor.set_translation_zones(zone_manager.translation_zones())
            region_editor.begin_region_edit(kind, rect)

    def _apply_region_edit(kind: str, left: int, top: int, width: int, height: int) -> None:
        zone_manager.apply_region_edit(kind, left, top, width, height)
        ctl.end_region_edit(kind, True)

    def _cancel_region_edit(kind: str) -> None:
        zone_manager.finish_region_edit(cancelled=True)
        ctl.end_region_edit(kind, False)

    zone_manager.zoneListsChanged.connect(_apply_zone_lists)
    zone_manager.beginZoneEditRequested.connect(_begin_zone_edit)
    zone_manager.lockZoneEditRequested.connect(region_editor.lock_region_edit)
    zone_manager.cancelZoneEditRequested.connect(region_editor.cancel_region_edit)
    region_editor.regionEdited.connect(_apply_region_edit)
    region_editor.regionEditCancelled.connect(_cancel_region_edit)

    _apply_zone_lists([], [])

    ctl.start()

    hotkey_str = str(getattr(config, "HOTKEY_PAUSE_RESUME", "F9")).strip()
    if hotkey_str and _KEYBOARD_AVAILABLE:
        try:

            def _toggle_pause_hotkey():
                QtCore.QMetaObject.invokeMethod(
                    controls,
                    "toggle_pause",
                    QtCore.Qt.ConnectionType.QueuedConnection,
                )

            _keyboard.add_hotkey(hotkey_str, _toggle_pause_hotkey)
            ctl.logger.channel("session", message="HOTKEY_REGISTERED", hotkey=hotkey_str)
        except Exception as exc:
            ctl.logger.channel(
                "session",
                message="HOTKEY_REGISTER_FAILED",
                hotkey=hotkey_str,
                error=repr(exc),
            )
            print(f"Warning: could not register hotkey '{hotkey_str}': {exc}")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
