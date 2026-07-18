"""Compatibility shim for older OneOcr module layouts.

The embedded translator uses the direct Python OCR engine, not the CLI bridge.
Some helper modules still import ``oneocr.workflow.OcrError`` from the older
layout, so this shim re-exports the shared exception type from ``common.py``.
"""

from __future__ import annotations

from .common import OcrError

__all__ = ["OcrError"]
