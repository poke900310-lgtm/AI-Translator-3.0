"""Python file-loading helpers for OneOcr.

This module mirrors `OneOcr.Image.psm1` in PowerShell.  It intentionally stays
small: it only normalizes a path, validates it, and returns the raw image bytes.

The actual image-to-BGRA conversion is handled by the native OCR wrapper module.
That split keeps the language-specific image processing code close to the native
interop code that needs it.
"""

from __future__ import annotations

from .common import OcrError, resolve_image_path


def get_image_bytes(path: str) -> tuple[str, bytes]:
    """Read the target image from disk and return `(normalized_path, bytes)`."""

    normalized = resolve_image_path(path)
    if not normalized.is_file():
        raise OcrError(f'Image file not found: {normalized}')

    try:
        data = normalized.read_bytes()
    except OSError as exc:
        raise OcrError(str(exc)) from exc

    if not data:
        raise OcrError(f'Image file is empty: {normalized}')

    return str(normalized), data
