"""Multi-channel debug logging utilities.

This build keeps the legacy session log file for quick reading, but also writes
separate timestamped channel logs and optionally saves image/text artifacts.
Writes are queued onto a single background worker so capture/OCR hot paths do
not block on filesystem I/O.
"""

from __future__ import annotations

import datetime
import itertools
import json
import queue
import re
import threading
from pathlib import Path
from typing import Any


class Logger:
    DEFAULT_LOG_PATH = "overlay_translator_debug.log"
    DEFAULT_LOG_DIR = "debug_artifacts"

    def __init__(self, enabled: bool, path: str, log_dir: str | None = None, queue_size: int = 4096):
        self.enabled = enabled
        resolved = (path or "").strip() or self.DEFAULT_LOG_PATH
        self.path = Path(resolved)
        self.base_dir = Path((log_dir or "").strip() or self.DEFAULT_LOG_DIR)
        if not self.base_dir.is_absolute():
            self.base_dir = self.path.parent / self.base_dir
        self.logs_dir = self.base_dir / "logs"
        self.images_dir = self.base_dir / "images"
        self.text_dir = self.base_dir / "text"
        self.run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        self._initialized_channels: set[str] = set()
        self._closed = False
        self._closing = False
        self._queue: "queue.Queue[tuple[str, tuple[Any, ...]]]" = queue.Queue(maxsize=max(256, int(queue_size or 4096)))
        self._worker: threading.Thread | None = None
        self._seq = itertools.count(1)
        self._seq_lock = threading.Lock()
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            self.images_dir.mkdir(parents=True, exist_ok=True)
            self.text_dir.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                f.write("OverlayTranslator2 debug log\n")
            self._worker = threading.Thread(target=self._worker_loop, name="overlay-logger", daemon=True)
            self._worker.start()

    @staticmethod
    def _ts() -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    @staticmethod
    def _safe_name(value: str) -> str:
        value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
        return value[:180].strip("._-") or "item"

    @staticmethod
    def _frame_ref(value: Any) -> str:
        try:
            return f"F{int(value):06d}"
        except Exception:
            return ""

    @staticmethod
    def _track_ref(value: Any) -> str:
        try:
            return f"T{int(value):04d}"
        except Exception:
            return ""

    def _next_seq(self) -> int:
        with self._seq_lock:
            return int(next(self._seq))

    def _ensure_channel(self, channel: str) -> Path:
        path = self.logs_dir / f"{self._safe_name(channel)}.log"
        if channel not in self._initialized_channels:
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"OverlayTranslator2 {channel} log\n")
            self._initialized_channels.add(channel)
        return path

    def _enqueue(self, op: str, *args: Any) -> None:
        if not self.enabled or self._closed or self._closing:
            return
        try:
            self._queue.put_nowait((op, args))
        except queue.Full:
            if op in {"save_image", "save_text"}:
                return
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait((op, args))
            except queue.Full:
                return

    def _worker_loop(self) -> None:
        while True:
            op = ""
            try:
                op, args = self._queue.get(timeout=0.25)
            except queue.Empty:
                if self._closing and self._queue.empty():
                    self._closed = True
                    return
                continue
            try:
                if op == "log":
                    line = args[0]
                    with open(self.path, "a", encoding="utf-8") as f:
                        f.write(line)
                elif op == "channel":
                    channel, line = args
                    path = self._ensure_channel(channel)
                    with open(path, "a", encoding="utf-8") as f:
                        f.write(line + "\n")
                elif op == "save_text":
                    path, content = args
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(content)
                elif op == "save_image":
                    path, image, fmt = args
                    path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(path, format=fmt)
            except Exception:
                pass
            finally:
                self._queue.task_done()

    def close(self, wait: bool = True, timeout_s: float = 10.0) -> None:
        if not self.enabled or self._closed or self._closing:
            return
        self._closing = True
        if wait and self._worker is not None:
            try:
                self._queue.join()
            except Exception:
                pass
            self._worker.join(timeout=max(0.1, float(timeout_s or 10.0)))
        self._closed = True

    def log(self, msg: str) -> None:
        if not self.enabled:
            return
        ts = self._ts()
        line = f"[{ts}] {msg}\n"
        self._enqueue("log", line)

    def channel(self, channel: str, message: str | None = None, **fields: Any) -> None:
        if not self.enabled:
            return
        ts = self._ts()
        payload: dict[str, Any] = {"ts": ts, "run_id": self.run_id, "log_seq": self._next_seq()}
        if message is not None:
            payload["message"] = message
        if fields:
            payload.update(fields)
        if "frame_index" in payload and "frame_ref" not in payload:
            frame_ref = self._frame_ref(payload.get("frame_index"))
            if frame_ref:
                payload["frame_ref"] = frame_ref
        if "track_id" in payload and "track_ref" not in payload:
            track_ref = self._track_ref(payload.get("track_id"))
            if track_ref:
                payload["track_ref"] = track_ref
        derived_pairs = {
            "dropped_frame_index": "dropped_frame_ref",
            "inflight_frame": "inflight_frame_ref",
            "latest_requested_frame": "latest_requested_frame_ref",
            "latest_applied_frame": "latest_applied_frame_ref",
            "source_track_id": "source_track_ref",
            "current_track_id": "current_track_ref",
        }
        for source_key, ref_key in derived_pairs.items():
            if source_key in payload and ref_key not in payload:
                if source_key.endswith("track_id"):
                    ref = self._track_ref(payload.get(source_key))
                else:
                    ref = self._frame_ref(payload.get(source_key))
                if ref:
                    payload[ref_key] = ref
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        self._enqueue("channel", channel, line)

    def save_text(self, category: str, stem: str, content: str, suffix: str = ".txt") -> str:
        if not self.enabled:
            return ""
        safe_category = self._safe_name(category)
        safe_stem = self._safe_name(stem)
        target_dir = self.text_dir / safe_category
        path = target_dir / f"{safe_stem}{suffix}"
        self._enqueue("save_text", path, content)
        return str(path.relative_to(self.path.parent))

    def save_image(self, category: str, stem: str, image, fmt: str = "PNG") -> str:
        if not self.enabled:
            return ""
        safe_category = self._safe_name(category)
        safe_stem = self._safe_name(stem)
        target_dir = self.images_dir / safe_category
        ext = ".png" if fmt.upper() == "PNG" else ".jpg"
        path = target_dir / f"{safe_stem}{ext}"
        try:
            payload_image = image.copy()
        except Exception:
            payload_image = image
        self._enqueue("save_image", path, payload_image, fmt)
        return str(path.relative_to(self.path.parent))
