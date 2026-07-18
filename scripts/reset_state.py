from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR_NAMES = {
    "__pycache__",
    "debug",
    "debug_artifacts",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}
FILE_NAMES = {
    "overlay_translator_debug.log",
    "debug.log",
    "config.runtime.json",
    # Scratch files this codebase tends to accumulate during audit sessions:
    "_commit_msg.txt",
    "_audit_imports.py",
    "_audit_full.py",
    "_fix_docstring_order.py",
    "_smoke_font.py",
    "_ruff.log",
    "_mypy_full.log",
    "_mypy_priority.log",
}
FILE_SUFFIXES = {".pyc", ".pyo"}


def remove_path(path: Path) -> None:
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=False)
        else:
            path.unlink(missing_ok=True)
        print(f"removed: {path.relative_to(ROOT)}")
    except FileNotFoundError:
        pass


EXCLUDE_DIRS = {".venv", "vendor", "llama_cpp", "models"}


def _is_excluded(path: Path) -> bool:
    """Skip third-party / generated trees so resets don't nuke working
    state inside .venv, vendor/Oneocr, the extracted llama.cpp binaries,
    or the GGUF model cache."""
    try:
        rel_parts = path.relative_to(ROOT).parts
    except ValueError:
        return True
    return any(part in EXCLUDE_DIRS for part in rel_parts)


def main() -> int:
    print(f"Cleaning project tree: {ROOT}")
    removed = 0

    for path in sorted(ROOT.rglob("*")):
        if _is_excluded(path):
            continue
        name = path.name
        if path.is_dir() and name in DIR_NAMES:
            remove_path(path)
            removed += 1
            continue
        if path.is_file() and (name in FILE_NAMES or path.suffix.lower() in FILE_SUFFIXES):
            remove_path(path)
            removed += 1

    print(f"Done. Removed {removed} paths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
