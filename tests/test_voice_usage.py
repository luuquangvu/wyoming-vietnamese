"""Persistent recent voice preference tests."""

from pathlib import Path

import pytest

import wyoming_vietnamese.voice_usage as voice_usage_module
from wyoming_vietnamese.voice_usage import RecentVoiceStore


def test_recent_voice_store_tracks_order_and_survives_restart(tmp_path: Path) -> None:
    """Persist latest selected voice first and reload it in a new store instance."""
    path = tmp_path / "recent.json"
    store = RecentVoiceStore(path)
    assert store.get_recent() == ()
    store.record("Voice A")
    store.record("Voice B")
    store.record("Voice A")
    assert RecentVoiceStore(path).get_recent() == ("Voice A", "Voice B")


def test_recent_voice_store_handles_invalid_data(tmp_path: Path) -> None:
    """Ignore invalid persisted state and do not record empty voice names."""
    path = tmp_path / "recent.json"
    path.write_text("{invalid", encoding="utf-8")
    store = RecentVoiceStore(path)
    assert store.get_recent() == ()
    store.record("")
    assert path.read_text(encoding="utf-8") == "{invalid"


def test_recent_voice_store_deduplicates_loaded_names(tmp_path: Path) -> None:
    """Remove duplicates from existing data while preserving the newest ordering."""
    path = tmp_path / "recent.json"
    path.write_text('["A", "B", "A", 1, ""]', encoding="utf-8")
    assert RecentVoiceStore(path).get_recent() == ("A", "B")


def test_recent_voice_store_suppresses_persistence_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A filesystem failure while recording a voice does not disrupt inference."""
    store = RecentVoiceStore(tmp_path / "recent.json")

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("read-only filesystem")

    monkeypatch.setattr(voice_usage_module.os, "replace", fail_replace)

    store.record("Voice A")

    assert store.get_recent() == ()
