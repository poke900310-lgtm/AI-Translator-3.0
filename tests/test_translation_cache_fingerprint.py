"""Translation-memory cache fingerprint discipline.

The Controller persists per-window translation memory to disk so repeated
lines (HUD strings, menu labels, recurring names) don't re-hit the llama
server. Without a fingerprint, an edit to the system prompt / model /
source-language hint silently keeps returning stale cache entries.

This test pins the new contract from ``llama_server.cache_fingerprint``:

  * The fingerprint changes when PROMPT_VERSION, SCHEMA_VERSION, the
    model id, or the source-language hint changes.
  * A cache file whose stored fingerprint matches the current build is
    loaded verbatim.
  * A cache file whose stored fingerprint mismatches the current build
    is dropped on load (so users can never serve stale translations
    just because the file exists).
  * Legacy flat-dict files (no fingerprint field) are treated as a
    mismatch and dropped, so the upgrade path is "save once, rebuild
    on miss".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PyQt6.QtWidgets import QApplication

from app import llama_server
from app.controller import Controller


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_fingerprint_changes_with_each_input() -> None:
    base = llama_server.cache_fingerprint("ja", "local-model")
    assert base != llama_server.cache_fingerprint("ko", "local-model")
    assert base != llama_server.cache_fingerprint("ja", "different-model")
    # Empty / None source_hint folds to "auto".
    assert llama_server.cache_fingerprint(None, "m") == llama_server.cache_fingerprint("", "m")
    # Stable across calls with identical inputs.
    assert llama_server.cache_fingerprint("ja", "local-model") == base


def test_fingerprint_includes_prompt_and_schema_versions() -> None:
    fp = llama_server.cache_fingerprint("ja", "local-model")
    assert llama_server.PROMPT_VERSION in fp
    assert llama_server.SCHEMA_VERSION in fp


def _write_cache_file(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_load_uses_entries_when_fingerprint_matches(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    ctl = Controller(lang_tag=None)
    ctl._translation_memory_name = "test_bucket"
    expected_fp = ctl._current_cache_fingerprint()
    _write_cache_file(
        ctl._translation_memory_path(),
        {"fingerprint": expected_fp, "version": 2, "entries": {"hola": "hello", "adiós": "goodbye"}},
    )
    ctl._load_translation_memory()
    assert ctl._translation_cache == {"hola": "hello", "adiós": "goodbye"}


def test_load_drops_entries_on_fingerprint_mismatch(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    ctl = Controller(lang_tag=None)
    ctl._translation_memory_name = "test_bucket"
    _write_cache_file(
        ctl._translation_memory_path(),
        {"fingerprint": "v0::v0::stale-model::other", "version": 2, "entries": {"hola": "stale"}},
    )
    ctl._load_translation_memory()
    assert ctl._translation_cache == {}


def test_load_drops_legacy_flat_dict(qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A pre-v2 flat ``{src: dst}`` file has no fingerprint to verify; the
    safe default is to drop it on load so users can't accidentally serve
    pre-prompt-edit translations forever."""
    monkeypatch.chdir(tmp_path)
    ctl = Controller(lang_tag=None)
    ctl._translation_memory_name = "test_bucket"
    _write_cache_file(ctl._translation_memory_path(), {"hola": "legacy"})
    ctl._load_translation_memory()
    assert ctl._translation_cache == {}


def test_save_then_load_roundtrips(qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    ctl = Controller(lang_tag=None)
    ctl._translation_memory_name = "test_bucket"
    ctl._translation_cache = {"hola": "hello", "adiós": "goodbye"}
    ctl._save_translation_memory()
    ctl2 = Controller(lang_tag=None)
    ctl2._translation_memory_name = "test_bucket"
    ctl2._load_translation_memory()
    assert ctl2._translation_cache == {"hola": "hello", "adiós": "goodbye"}


def test_saved_file_has_fingerprint_field(qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    ctl = Controller(lang_tag=None)
    ctl._translation_memory_name = "test_bucket"
    ctl._translation_cache = {"hola": "hello"}
    ctl._save_translation_memory()
    on_disk = json.loads(ctl._translation_memory_path().read_text(encoding="utf-8"))
    assert on_disk["fingerprint"] == ctl._current_cache_fingerprint()
    assert on_disk["version"] == 2
    assert on_disk["entries"] == {"hola": "hello"}
