"""Embedded One OCR adapter.

This adapter now passes already-decoded PIL images directly into the OCR prep
layer, which avoids the previous PIL -> PNG bytes -> decode round-trip.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
_ONE_ROOT = _PROJECT_ROOT / "vendor" / "Oneocr"
_ONE_PYTHON_ROOT = _ONE_ROOT / "python"

if str(_ONE_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_ONE_PYTHON_ROOT))

# These imports intentionally follow the sys.path insertion above; the vendored
# OneOCR package is not on the normal import path until we extend sys.path.
from Oneocr.common import ensure_windows_64_bit, load_app_context  # noqa: E402
from Oneocr.native import BoundingBox, OcrLine, OcrResult, OcrWord, OneOcrEngine  # noqa: E402
from Oneocr.runtime_cache import prepare_local_runtime_dir  # noqa: E402


@dataclass(slots=True)
class StructuredOcrResult:
    lines: list[OcrLine]
    image_size: tuple[int, int]
    image_angle: float = 0.0


class OneOcr:
    """Thin adapter for the embedded One OCR runtime."""

    def __init__(self, lang_tag: str | None):
        self.requested_tag = lang_tag
        self.actual_tag = lang_tag

        ensure_windows_64_bit()
        self._context = load_app_context(_ONE_ROOT)
        self._runtime_dir = prepare_local_runtime_dir(self._context)
        self._engine = OneOcrEngine(self._runtime_dir, self._context)

    def recognize(self, pil_img: Image.Image) -> str:
        return (self._engine.recognize_text_fast(pil_img) or "").strip()

    def recognize_layout(self, pil_img: Image.Image) -> StructuredOcrResult:
        result: OcrResult = self._engine.recognize_result(pil_img, include_words=True)
        return StructuredOcrResult(
            lines=result.lines,
            image_size=pil_img.size,
            image_angle=result.image_angle,
        )

    def close(self) -> None:
        try:
            self._engine.close()
        except Exception:
            pass


__all__ = [
    "BoundingBox",
    "OcrLine",
    "OcrWord",
    "OneOcr",
    "StructuredOcrResult",
]
