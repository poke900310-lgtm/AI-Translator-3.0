"""Python runtime-cache module for the unified OneOcr environment.

This module mirrors `OneOcr.RuntimeCache.psm1` in PowerShell.

The core rule is simple:
- never load the OCR runtime directly from a UNC path when it can be avoided
- copy the native files to a per-user local cache under LOCALAPPDATA
- reuse existing cached files when size and write time still match exactly

The cache path includes both the runtime version and a hash of the source folder.
That keeps different runtime sources isolated while still making repeat runs cheap.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .common import AppContext, OcrError


def get_cache_root(config: dict[str, Any]) -> Path:
    """Return the per-user cache root used by both app variants."""

    local_appdata = os.environ.get('LOCALAPPDATA')
    base = Path(local_appdata) if local_appdata else Path(tempfile.gettempdir())
    return base / config['cache_root_name'] / config['runtime_cache_subdirectory']


def get_stable_hash(value: str) -> str:
    """Generate a stable lowercase SHA-256 hash for a folder identity string."""

    return hashlib.sha256(value.lower().encode('utf-8')).hexdigest()


def runtime_file_is_current(source: Path, destination: Path) -> bool:
    """Treat a cached file as current only when size and nanosecond mtime match."""

    try:
        source_stat = source.stat()
        destination_stat = destination.stat()
    except FileNotFoundError:
        return False

    return (
        destination.is_file()
        and source_stat.st_size == destination_stat.st_size
        and source_stat.st_mtime_ns == destination_stat.st_mtime_ns
    )


def get_runtime_source_dir(context: AppContext) -> Path:
    """Resolve the shared runtime folder from the shared configuration."""

    source_dir = (context.project_root / context.config['runtime_relative_path']).resolve()
    if not source_dir.is_dir():
        raise OcrError(f'Runtime folder not found: {source_dir}')
    return source_dir


def read_runtime_manifest(runtime_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Load the runtime manifest stored next to the native runtime files."""

    import json

    manifest_path = runtime_dir / config['runtime_manifest_name']
    try:
        return json.loads(manifest_path.read_text(encoding='utf-8'))
    except FileNotFoundError as exc:
        raise OcrError(f'Runtime manifest not found: {manifest_path}') from exc
    except json.JSONDecodeError as exc:
        raise OcrError(f'Runtime manifest is not valid JSON: {manifest_path}') from exc


def prepare_local_runtime_dir(context: AppContext) -> Path:
    """Ensure the shared runtime exists in the local per-user cache.

    This is the critical UNC/network-share portability step.  Both the CLI and GUI
    app variants call this before any native DLL is loaded.
    """

    config = context.config
    runtime_dir = get_runtime_source_dir(context)
    manifest = read_runtime_manifest(runtime_dir, config)
    required_files = list(config['required_runtime_files'])

    missing = [name for name in required_files if not (runtime_dir / name).is_file()]
    if missing:
        raise OcrError(
            f"Missing required file(s) in runtime folder: {', '.join(missing)}"
        )

    version = str(manifest.get('version', 'unknown'))
    platform = str(manifest.get('platform', 'unknown'))
    source_hash = get_stable_hash(str(runtime_dir))

    cache_dir = get_cache_root(config) / platform / version / source_hash
    cache_dir.mkdir(parents=True, exist_ok=True)

    for name in required_files:
        source = runtime_dir / name
        destination = cache_dir / name

        if runtime_file_is_current(source, destination):
            continue

        shutil.copy2(source, destination)

        # copy2 normally preserves mtime, but we explicitly force nanosecond parity
        # so the Python and PowerShell cache-current checks behave as similarly as
        # practical.
        source_stat = source.stat()
        os.utime(destination, ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns))

    # Keep a copy of the manifest in the cache folder for troubleshooting.
    manifest_destination = cache_dir / config['runtime_manifest_name']
    shutil.copy2(runtime_dir / config['runtime_manifest_name'], manifest_destination)

    return cache_dir
