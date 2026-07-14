"""Bounded in-memory LRU cache for deterministic inference results."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from time import monotonic


@dataclass(slots=True)
class _CacheEntry[Value]:
    """Store one cached value and its accounting metadata."""

    value: Value
    size_bytes: int
    expires_at: float


class BoundedLruCache[Key, Value]:
    """Keep recently used values within entry, byte, item, and idle-age limits.

    The cache is intentionally synchronous because all server access occurs on the
    asyncio event-loop thread. Values are immutable strings or bytes, so returning
    them does not expose mutable cache state.
    """

    def __init__(
        self,
        *,
        max_entries: int,
        max_bytes: int,
        max_item_bytes: int,
        max_idle_seconds: float,
    ) -> None:
        """Initialize fixed resource limits; any zero size/count disables caching."""
        if min(max_entries, max_bytes, max_item_bytes) < 0:
            raise ValueError("cache limits must not be negative")
        if max_idle_seconds < 0:
            raise ValueError("cache idle duration must not be negative")

        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.max_item_bytes = max_item_bytes
        self.max_idle_seconds = max_idle_seconds
        self._entries: OrderedDict[Key, _CacheEntry[Value]] = OrderedDict()
        self._total_bytes = 0

    @property
    def enabled(self) -> bool:
        """Return whether all required capacity limits permit storing entries."""
        return bool(self.max_entries and self.max_bytes and self.max_item_bytes)

    @property
    def total_bytes(self) -> int:
        """Return the accounted bytes currently retained by the cache."""
        return self._total_bytes

    def __len__(self) -> int:
        """Return the number of live entries after removing expired values."""
        self.prune()
        return len(self._entries)

    def get(self, key: Key) -> Value | None:
        """Return and refresh a cached value, or ``None`` when it is unavailable."""
        if not self.enabled:
            return None

        now = monotonic()
        self.prune(now)
        entry = self._entries.pop(key, None)
        if entry is None:
            return None

        entry.expires_at = self._expires_at(now)
        self._entries[key] = entry
        return entry.value

    def put(self, key: Key, value: Value, *, size_bytes: int) -> bool:
        """Store a value if it fits, evicting least-recently-used entries as needed."""
        if size_bytes < 0:
            raise ValueError("cache item size must not be negative")
        if not self.enabled or size_bytes > self.max_item_bytes or size_bytes > self.max_bytes:
            return False

        now = monotonic()
        self.prune(now)
        previous = self._entries.pop(key, None)
        if previous is not None:
            self._total_bytes -= previous.size_bytes

        self._entries[key] = _CacheEntry(
            value=value,
            size_bytes=size_bytes,
            expires_at=self._expires_at(now),
        )
        self._total_bytes += size_bytes
        while len(self._entries) > self.max_entries or self._total_bytes > self.max_bytes:
            _, evicted = self._entries.popitem(last=False)
            self._total_bytes -= evicted.size_bytes
        return True

    def prune(self, now: float | None = None) -> int:
        """Remove idle entries and return how many values were discarded."""
        if not self._entries or not self.max_idle_seconds:
            return 0

        current_time = monotonic() if now is None else now
        removed = 0
        while self._entries:
            first_key = next(iter(self._entries))
            entry = self._entries[first_key]
            if entry.expires_at > current_time:
                break
            self._entries.pop(first_key)
            self._total_bytes -= entry.size_bytes
            removed += 1
        return removed

    def clear(self) -> tuple[int, int]:
        """Discard every value and return the released entry and byte counts."""
        released = (len(self._entries), self._total_bytes)
        self._entries.clear()
        self._total_bytes = 0
        return released

    def _expires_at(self, now: float) -> float:
        """Calculate an idle deadline, using infinity when expiration is disabled."""
        if not self.max_idle_seconds:
            return float("inf")
        return now + self.max_idle_seconds
