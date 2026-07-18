"""Shared Python helpers for the OneOcr unified environment.

This module deliberately mirrors the responsibility of the PowerShell
`OneOcr.Common.psm1` module.  It owns:
- locating the project root
- loading the shared JSON configuration
- common process checks such as Windows + 64-bit validation
- common console output behavior

Keeping these responsibilities in one place makes the Python and PowerShell
entry scripts look nearly identical at a high level.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class OcrError(RuntimeError):
    """Raised when setup, file loading, or native OCR execution fails."""


@dataclass(frozen=True)
class AppContext:
    """Resolved project information shared by all Python modules.

    Attributes:
        project_root:
            Top-level folder that contains `runtime`, `shared`, `powershell`, and
            `python`.
        config_path:
            Path to `shared/config.json`.
        config:
            Parsed JSON dictionary from the shared configuration file.
    """

    project_root: Path
    config_path: Path
    config: dict[str, Any]


def resolve_project_root(anchor_path: Path) -> Path:
    """Walk upward until the unified environment root is found.

    We identify the root by looking for the shared config file and the runtime
    folder.  This makes the code resilient even if the app entry scripts are moved
    around inside the `python` or `powershell` subtrees.
    """

    current = anchor_path.resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / 'shared' / 'config.json').is_file() and (candidate / 'runtime').is_dir():
            return candidate

    raise OcrError(
        f'Could not find the OneOcr project root above: {anchor_path}'
    )


def load_app_context(anchor_path: Path) -> AppContext:
    """Load the shared configuration for the unified environment."""

    project_root = resolve_project_root(anchor_path)
    config_path = project_root / 'shared' / 'config.json'

    try:
        config = json.loads(config_path.read_text(encoding='utf-8'))
    except FileNotFoundError as exc:
        raise OcrError(f'Shared config file not found: {config_path}') from exc
    except json.JSONDecodeError as exc:
        raise OcrError(f'Shared config file is not valid JSON: {config_path}') from exc

    return AppContext(project_root=project_root, config_path=config_path, config=config)


def ensure_windows_64_bit() -> None:
    """Enforce the same runtime rule as the PowerShell scripts.

    The bundled oneocr runtime is a 64-bit Windows native dependency.  Running on
    another OS or in a 32-bit process will fail later in much more confusing ways,
    so we fail fast here with a clear message.
    """

    if os.name != 'nt':
        raise OcrError('This script requires Windows.')
    if ctypes.sizeof(ctypes.c_void_p) != 8:
        raise OcrError('Run this in 64-bit Python on Windows.')


def configure_stdout_utf8() -> None:
    """Make stdout explicitly UTF-8 when the host supports reconfiguration."""

    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')


def resolve_image_path(path: str) -> Path:
    """Normalize a possibly quoted or expanded image path.

    UNC paths work fine here as long as Python can access them.  We expand
    environment variables, expand `~`, strip surrounding quotes, and normalize the
    path separators before the file is opened.
    """

    if not path or not path.strip():
        raise OcrError('Image path is empty.')

    expanded = os.path.expandvars(os.path.expanduser(path.strip().strip('"')))
    normalized = os.path.normpath(expanded)
    return Path(normalized)
