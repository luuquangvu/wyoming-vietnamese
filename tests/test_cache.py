"""Bounded inference result cache tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import wyoming_vietnamese.cache as cache_module
from wyoming_vietnamese.cache import BoundedLruCache, PersistentAudioCache


def _cache(
    *,
    max_entries: int = 2,
    max_bytes: int = 10,
    max_item_bytes: int = 8,
    max_idle_seconds: float = 60,
) -> BoundedLruCache[str, str]:
    """Build test support for  cache."""
    return BoundedLruCache(
        max_entries=max_entries,
        max_bytes=max_bytes,
        max_item_bytes=max_item_bytes,
        max_idle_seconds=max_idle_seconds,
    )


def test_cache_evicts_least_recently_used_entry() -> None:
    """Test cache evicts least recently used entry."""
    cache = _cache(max_bytes=30)
    assert cache.put("a", "A", size_bytes=3)
    assert cache.put("b", "B", size_bytes=3)
    assert cache.get("a") == "A"
    assert cache.put("c", "C", size_bytes=3)
    assert cache.get("b") is None
    assert cache.get("a") == "A"
    assert cache.get("c") == "C"


def test_cache_evicts_to_byte_limit_and_accounts_replacements() -> None:
    """Test cache evicts to byte limit and accounts replacements."""
    cache = _cache(max_entries=10, max_bytes=6)
    assert cache.put("a", "A", size_bytes=3)
    assert cache.put("b", "B", size_bytes=3)
    assert cache.put("c", "C", size_bytes=3)
    assert cache.get("a") is None
    assert cache.total_bytes == 6

    assert cache.put("b", "new", size_bytes=2)
    assert cache.total_bytes == 5
    assert cache.get("b") == "new"


def test_cache_rejects_oversized_items_and_invalid_sizes() -> None:
    """Test cache rejects oversized items and invalid sizes."""
    cache = _cache(max_bytes=20, max_item_bytes=4)
    assert cache.put("large", "value", size_bytes=5) is False
    assert len(cache) == 0
    with pytest.raises(ValueError, match="must not be negative"):
        cache.put("bad", "value", size_bytes=-1)


def test_cache_expires_entries_after_idle_period(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test cache expires entries after idle period."""
    current_time = 10.0
    monkeypatch.setattr(cache_module, "monotonic", lambda: current_time)
    cache = _cache(max_idle_seconds=5)
    assert cache.put("key", "value", size_bytes=5)

    current_time = 14.0
    assert cache.get("key") == "value"
    current_time = 20.0
    assert cache.get("key") is None
    assert cache.total_bytes == 0


def test_cache_can_disable_expiration_and_clear_all_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test cache can disable expiration and clear all values."""
    current_time = 1.0
    monkeypatch.setattr(cache_module, "monotonic", lambda: current_time)
    cache = _cache(max_idle_seconds=0)
    assert cache.put("key", "value", size_bytes=5)
    current_time = 1_000_000.0
    assert cache.get("key") == "value"
    assert cache.clear() == (1, 5)
    assert len(cache) == 0
    assert cache.total_bytes == 0


@pytest.mark.parametrize(
    "limits",
    [
        {"max_entries": 0, "max_bytes": 10, "max_item_bytes": 10},
        {"max_entries": 1, "max_bytes": 0, "max_item_bytes": 10},
        {"max_entries": 1, "max_bytes": 10, "max_item_bytes": 0},
    ],
)
def test_cache_zero_capacity_disables_storage(limits: dict[str, int]) -> None:
    """Test cache zero capacity disables storage."""
    cache = BoundedLruCache[str, str](max_idle_seconds=10, **limits)
    assert cache.enabled is False
    assert cache.put("key", "value", size_bytes=1) is False
    assert cache.get("key") is None


def test_cache_rejects_negative_limits() -> None:
    """Test cache rejects negative limits."""
    with pytest.raises(ValueError, match="limits must not be negative"):
        BoundedLruCache[str, str](
            max_entries=-1,
            max_bytes=1,
            max_item_bytes=1,
            max_idle_seconds=1,
        )
    with pytest.raises(ValueError, match="duration must not be negative"):
        BoundedLruCache[str, str](
            max_entries=1,
            max_bytes=1,
            max_item_bytes=1,
            max_idle_seconds=-1,
        )


def test_persistent_audio_cache_survives_recreation(tmp_path: Path) -> None:
    """Persist disk entries across cache objects while weak devices retain no RAM copy."""
    directory = tmp_path / "audio"
    memory_cache = BoundedLruCache[bytes, bytes](
        max_entries=0,
        max_bytes=0,
        max_item_bytes=0,
        max_idle_seconds=60,
    )
    first = PersistentAudioCache(
        directory,
        memory_cache,
        max_entries=2,
        max_bytes=20,
        max_item_bytes=20,
        max_idle_seconds=60,
    )
    assert first.put(b"key", b"audio!", size_bytes=9)
    assert memory_cache.total_bytes == 0

    second = PersistentAudioCache(
        directory,
        BoundedLruCache[bytes, bytes](
            max_entries=0,
            max_bytes=0,
            max_item_bytes=0,
            max_idle_seconds=60,
        ),
        max_entries=2,
        max_bytes=20,
        max_item_bytes=20,
        max_idle_seconds=60,
    )
    assert second.get(b"key") == b"audio!"
    assert second.clear() == (0, 0)
    assert second.get(b"key") == b"audio!"


def test_persistent_audio_cache_prunes_oldest_entry(tmp_path: Path) -> None:
    """Keep disk results within entry and byte limits using LRU file age."""
    cache = PersistentAudioCache(
        tmp_path,
        BoundedLruCache[bytes, bytes](
            max_entries=0,
            max_bytes=0,
            max_item_bytes=0,
            max_idle_seconds=60,
        ),
        max_entries=1,
        max_bytes=40,
        max_item_bytes=40,
        max_idle_seconds=60,
    )
    assert cache.put(b"a", b"first", size_bytes=6)
    assert cache.put(b"b", b"second", size_bytes=7)
    assert cache.get(b"a") is None
    assert cache.get(b"b") == b"second"


def _disk_only_cache(
    directory: Path,
    *,
    max_bytes: int,
    max_item_bytes: int,
) -> PersistentAudioCache:
    """Build a persistent cache without a RAM tier for disk-budget tests."""
    return PersistentAudioCache(
        directory,
        BoundedLruCache[bytes, bytes](
            max_entries=0,
            max_bytes=0,
            max_item_bytes=0,
            max_idle_seconds=60,
        ),
        max_entries=10,
        max_bytes=max_bytes,
        max_item_bytes=max_item_bytes,
        max_idle_seconds=60,
    )


def test_persistent_cache_rejects_item_over_disk_item_limit(tmp_path: Path) -> None:
    """Do not persist one result larger than the configured per-item budget."""
    cache = _disk_only_cache(tmp_path, max_bytes=100, max_item_bytes=6)

    assert cache.put(b"k", b"audio", size_bytes=7) is False
    assert not list(tmp_path.glob("*.pcm"))


def test_persistent_cache_prunes_to_combined_disk_byte_budget(tmp_path: Path) -> None:
    """Evict least-recently-used entries when their combined accounted size is too big."""
    cache = _disk_only_cache(tmp_path, max_bytes=8, max_item_bytes=8)

    assert cache.put(b"a", b"1234", size_bytes=5)
    assert cache.put(b"b", b"5678", size_bytes=5)

    assert cache.get(b"a") is None
    assert cache.get(b"b") == b"5678"
    assert len(list(tmp_path.glob("*.pcm"))) == 1


def test_persistent_cache_accounts_for_key_in_item_limit(tmp_path: Path) -> None:
    """Reject payloads whose key plus audio exceed the per-item budget."""
    cache = _disk_only_cache(tmp_path, max_bytes=20, max_item_bytes=6)

    assert cache.put(b"long", b"123", size_bytes=len(b"long") + len(b"123")) is False
    assert not list(tmp_path.glob("*.pcm"))


def test_persistent_cache_removes_odd_sized_pcm_entry(tmp_path: Path) -> None:
    """Treat a disk entry ending halfway through a 16-bit PCM sample as corrupt."""
    cache = _disk_only_cache(tmp_path, max_bytes=20, max_item_bytes=20)
    path = tmp_path / f"{b'key'.hex()}.pcm"
    path.write_bytes(b"123")

    assert cache.get(b"key") is None
    assert path.exists() is False


def test_persistent_cache_removes_zero_length_pcm_entry(tmp_path: Path) -> None:
    """Treat an empty disk entry as corrupt and evict it on lookup."""
    cache = _disk_only_cache(tmp_path, max_bytes=20, max_item_bytes=20)
    path = tmp_path / f"{b'key'.hex()}.pcm"
    path.write_bytes(b"")

    assert cache.get(b"key") is None
    assert path.exists() is False


def test_persistent_cache_put_syncs_file_before_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flush and sync the temporary file to durable storage before atomic replace."""
    cache = _disk_only_cache(tmp_path, max_bytes=20, max_item_bytes=20)
    events: list[str] = []
    original_fsync = os.fsync
    original_replace = cache_module.os.replace

    def tracked_fsync(fd: int) -> None:
        events.append("fsync")
        original_fsync(fd)

    def tracked_replace(src: Path | str, dst: Path | str) -> None:
        events.append("replace")
        original_replace(src, dst)

    monkeypatch.setattr(cache_module.os, "fsync", tracked_fsync)
    monkeypatch.setattr(cache_module.os, "replace", tracked_replace)

    assert cache.put(b"key", b"12", size_bytes=5) is True
    assert events == ["fsync", "replace"]


def test_persistent_cache_read_failure_is_a_miss_without_deleting_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient read error returns a miss and leaves the entry for a later retry."""
    cache = _disk_only_cache(tmp_path, max_bytes=20, max_item_bytes=20)
    path = tmp_path / f"{b'key'.hex()}.pcm"
    path.write_bytes(b"12")
    original_read_bytes = Path.read_bytes

    def fail_entry_read(candidate: Path) -> bytes:
        if candidate == path:
            raise PermissionError("access denied")
        return original_read_bytes(candidate)

    monkeypatch.setattr(Path, "read_bytes", fail_entry_read)

    assert cache.get(b"key") is None
    assert path.exists() is True


def test_persistent_cache_suppresses_atomic_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed atomic disk write is contained and reported as not persisted."""
    cache = _disk_only_cache(tmp_path, max_bytes=20, max_item_bytes=20)

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("read-only filesystem")

    monkeypatch.setattr(cache_module.os, "replace", fail_replace)

    assert cache.put(b"key", b"12", size_bytes=5) is False
    assert not list(tmp_path.iterdir())


def test_persistent_cache_memory_hit_refreshes_disk_age(tmp_path: Path) -> None:
    """A RAM hit also refreshes the persistent entry's idle expiration age."""
    directory = tmp_path / "audio"
    cache = PersistentAudioCache(
        directory,
        BoundedLruCache[bytes, bytes](
            max_entries=2,
            max_bytes=100,
            max_item_bytes=100,
            max_idle_seconds=60,
        ),
        max_entries=2,
        max_bytes=100,
        max_item_bytes=100,
        max_idle_seconds=60,
    )
    assert cache.put(b"key", b"value", size_bytes=8)
    entry = directory / f"{b'key'.hex()}.pcm"
    os.utime(entry, (1, 1))
    assert cache.get(b"key") == b"value"
    assert entry.stat().st_mtime > 1


def test_persistent_audio_cache_expires_and_can_disable_disk(tmp_path: Path) -> None:
    """Expire idle disk entries and continue serving RAM when disk caching is off."""
    directory = tmp_path / "audio"
    cache = PersistentAudioCache(
        directory,
        BoundedLruCache[bytes, bytes](
            max_entries=0,
            max_bytes=0,
            max_item_bytes=0,
            max_idle_seconds=1,
        ),
        max_entries=2,
        max_bytes=100,
        max_item_bytes=100,
        max_idle_seconds=1,
    )
    assert cache.put(b"expire", b"old", size_bytes=9)
    old_path = directory / f"{b'expire'.hex()}.pcm"
    os.utime(old_path, (1, 1))
    assert cache.get(b"expire") is None
    assert old_path.exists() is False

    memory_only = PersistentAudioCache(
        tmp_path / "memory-only",
        BoundedLruCache[bytes, bytes](
            max_entries=2,
            max_bytes=100,
            max_item_bytes=100,
            max_idle_seconds=60,
        ),
        max_entries=0,
        max_bytes=0,
        max_item_bytes=0,
        max_idle_seconds=60,
    )
    assert memory_only.put(b"key", b"value", size_bytes=8)
    assert memory_only.get(b"key") == b"value"


def test_persistent_audio_cache_rejects_negative_limits(tmp_path: Path) -> None:
    """Reject invalid disk budgets during cache construction."""
    with pytest.raises(ValueError, match="disk cache limits must not be negative"):
        PersistentAudioCache(
            tmp_path,
            BoundedLruCache[bytes, bytes](
                max_entries=0,
                max_bytes=0,
                max_item_bytes=0,
                max_idle_seconds=0,
            ),
            max_entries=-1,
            max_bytes=1,
            max_item_bytes=1,
            max_idle_seconds=1,
        )


def test_persistent_cache_prunes_existing_entries_on_init(tmp_path: Path) -> None:
    """Enforce disk limits during construction so a tier change takes effect immediately."""
    for i in range(5):
        (tmp_path / f"{bytes([i]).hex()}.pcm").write_bytes(b"ab")
    assert len(list(tmp_path.glob("*.pcm"))) == 5

    PersistentAudioCache(
        tmp_path,
        BoundedLruCache[bytes, bytes](
            max_entries=0,
            max_bytes=0,
            max_item_bytes=0,
            max_idle_seconds=60,
        ),
        max_entries=2,
        max_bytes=100,
        max_item_bytes=100,
        max_idle_seconds=60,
    )
    assert len(list(tmp_path.glob("*.pcm"))) == 2


def test_persistent_cache_utime_failure_still_returns_cached_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A utime failure during cache hit does not fail the read or discard cached audio."""
    cache = _disk_only_cache(tmp_path, max_bytes=100, max_item_bytes=100)
    assert cache.put(b"key", b"12345678", size_bytes=12)

    def fail_utime(_path: Path | str, _times: object = None) -> None:
        raise PermissionError("operation not permitted")

    monkeypatch.setattr(cache_module.os, "utime", fail_utime)
    assert cache.get(b"key") == b"12345678"


def test_persistent_cache_chmod_failure_on_init_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Filesystems not supporting chmod do not prevent persistent cache construction."""

    def fail_chmod(_self: Path, _mode: int) -> None:
        raise OSError("operation not permitted")

    monkeypatch.setattr(Path, "chmod", fail_chmod)
    cache = PersistentAudioCache(
        tmp_path / "cache",
        BoundedLruCache[bytes, bytes](
            max_entries=0,
            max_bytes=0,
            max_item_bytes=0,
            max_idle_seconds=60,
        ),
        max_entries=2,
        max_bytes=100,
        max_item_bytes=100,
        max_idle_seconds=60,
    )
    assert cache.enabled is True


def test_persistent_cache_prunes_orphaned_temporary_files(tmp_path: Path) -> None:
    """Interrupted atomic writes leave temp files that are pruned after expiring."""
    stale_temp = tmp_path / ".tts-oldtemp"
    stale_temp.write_bytes(b"interrupted")
    os.utime(stale_temp, (1, 1))

    fresh_temp = tmp_path / ".tts-freshtemp"
    fresh_temp.write_bytes(b"active")

    cache = _disk_only_cache(tmp_path, max_bytes=100, max_item_bytes=100)
    cache._prune_disk_entries()

    assert stale_temp.exists() is False
    assert fresh_temp.exists() is True


def test_persistent_cache_strict_write_containment_prunes_before_publishing(tmp_path: Path) -> None:
    """Enforce strict write-time containment so aggregate storage never exceeds max_bytes."""
    # Key hex is 4 chars -> stem//2 = 2 bytes overhead.
    # With 12-byte even PCM value, item_size is 14 bytes.
    cache = _disk_only_cache(tmp_path, max_bytes=30, max_item_bytes=30)
    assert cache.put(b"k1", b"0123456789ab", size_bytes=14)
    assert cache.disk_total_bytes == 14
    assert cache.disk_entries == 1

    assert cache.put(b"k2", b"0123456789cd", size_bytes=14)
    assert cache.disk_total_bytes == 28
    assert cache.disk_entries == 2

    # Third item (14 bytes) would exceed 30 bytes (28 + 14 = 42).
    # Pre-write pruning must evict k1 before k3 is published.
    assert cache.put(b"k3", b"0123456789ef", size_bytes=14)
    assert cache.disk_total_bytes == 28
    assert cache.disk_entries == 2
    assert cache.get(b"k1") is None
    assert cache.get(b"k2") == b"0123456789cd"
    assert cache.get(b"k3") == b"0123456789ef"


def test_persistent_cache_failed_deletion_does_not_deduct_and_rejects_overbudget_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failed unlinks during pruning do not deduct bytes and reject writes exceeding budget."""
    cache = _disk_only_cache(tmp_path, max_bytes=30, max_item_bytes=30)
    assert cache.put(b"k1", b"0123456789ab", size_bytes=14)
    assert cache.put(b"k2", b"0123456789cd", size_bytes=14)
    assert cache.disk_total_bytes == 28

    def fail_unlink(_self: Path, *, missing_ok: bool = False) -> None:
        raise OSError("disk write protected")

    monkeypatch.setattr(Path, "unlink", fail_unlink)

    # Attempt to write k3: needs to prune k1, but unlink fails.
    # Write must be rejected and not persisted to disk.
    assert cache.put(b"k3", b"0123456789ef", size_bytes=14) is False
    assert cache.disk_total_bytes == 28
    assert cache.get(b"k3") is None
