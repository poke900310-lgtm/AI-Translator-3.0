"""OneOcr v1.0 Python image-preparation helpers.

This module validates image dimensions, optionally upscales small images to the
configured minimum size, and converts image bytes or in-memory PIL images into
packed BGRA data for oneocr.dll.

Version: 1.1
"""

from __future__ import annotations

import math
from io import BytesIO
from typing import Any

from .workflow import OcrError


def import_upscale_bridge() -> None:
    """Initialize the Python upscale path."""

    return None


def _get_pillow_image_module():
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise OcrError(
            'Python dependency missing: Pillow. Install it with: '
            'python -m pip install -r requirements.txt'
        ) from exc
    return Image


def _clone_source_image(source: Any):
    Image = _get_pillow_image_module()
    if isinstance(source, (bytes, bytearray, memoryview)):
        with Image.open(BytesIO(bytes(source))) as opened:
            opened.load()
            return opened.copy()
    if hasattr(source, 'copy') and hasattr(source, 'size') and hasattr(source, 'convert'):
        return source.copy()
    raise OcrError(f'Unsupported image source type: {type(source)!r}')


def convert_to_prepared_image(
    input_source: Any,
    min_size: int,
    max_size: int,
    upscale_small_images: bool,
) -> tuple[bytes, int, int, int]:
    """Return packed BGRA bytes plus width, height, and step for oneocr.dll.

    ``input_source`` may be raw image bytes or an already-decoded PIL image.
    Accepting PIL images avoids the costly PNG encode/decode round-trip in the
    live translator path.
    """

    import_upscale_bridge()
    Image = _get_pillow_image_module()

    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS

    source = _clone_source_image(input_source)
    try:
        width, height = source.size

        if width > max_size or height > max_size:
            raise OcrError(
                f'Unsupported image size. Keep width and height between {min_size} and {max_size} pixels.'
            )

        if width < min_size or height < min_size:
            if not upscale_small_images:
                raise OcrError(
                    f'Image is smaller than the configured minimum size of {min_size} pixels and auto-upscaling is disabled.'
                )

            scale = max(min_size / float(width), min_size / float(height))
            width = int(math.ceil(width * scale))
            height = int(math.ceil(height * scale))

            if width > max_size or height > max_size:
                raise OcrError(
                    f'Upscaled image would exceed the configured maximum size of {max_size} pixels.'
                )

            source = source.resize((width, height), resample)

        bitmap = source.convert('RGBA')
        bgra_bytes = bitmap.tobytes('raw', 'BGRA')
        step = width * 4
        return bgra_bytes, width, height, step
    finally:
        try:
            source.close()
        except Exception:
            pass
