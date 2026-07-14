"""Bounded inference result cache tests."""

from __future__ import annotations

import pytest

import wyoming_vietnamese.cache as cache_module
from wyoming_vietnamese.cache import BoundedLruCache


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
