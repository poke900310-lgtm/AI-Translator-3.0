"""Canary: every app.* module imports cleanly under a live QApplication.

Catches the "extraction broke a circular import" / "import-time side effect
failed" class of regression.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.usefixtures("qapp")
@pytest.mark.parametrize(
    "module",
    [
        "app",
        "app.color",
        "app.config",
        "app.controller",
        "app.geometry",
        "app.llama_server",
        "app.logging",
        "app.one_ocr",
        "app.qt_render",
        "app.screen_capture",
        "app.text",
        "app.types",
        "app.window_binding",
        "app.windows",
    ],
)
def test_module_imports(module: str) -> None:
    importlib.import_module(module)


def test_version_is_string() -> None:
    from app import VERSION

    assert isinstance(VERSION, str)
    assert VERSION.count(".") >= 1
