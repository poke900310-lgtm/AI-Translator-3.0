"""Shared pytest fixtures.

A session-scoped QApplication is needed for any test that touches PyQt6 font
construction (QFont / QFontMetrics will assert if no QGuiApplication exists).
pytest-qt provides this implicitly when using `qtbot`, but a plain fixture is
clearer here and lets non-Qt tests stay non-Qt.
"""

from __future__ import annotations

import sys

import pytest
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app
