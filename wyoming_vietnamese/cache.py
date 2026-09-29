"""Bounded memory and persistent disk caches for deterministic inference results."""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import RLock
from time import monotonic, time
from typing import Protocol

from .const import TtsCacheFile

_LOGGER = logging.getLogger(__name__)


class ResultCache(Protocol):
    """Cache interface used by the asynchronous TTS handler."""

    @property
    def enabled(self) -> bool:
        """Return whether this cache can retain values."""
        ...

    @property
    def max_item_bytes(self) -> int:
        """Return the largest cache entry this cache accepts."""
        ...

    def get(self, key: bytes) -> bytes | None:
        """Return a cached value, if available."""
        ...

    def put(self, key: bytes, value: bytes, *, size_bytes: int) -> bool:
        """Store a cache value and report whether it was retained."""
        ...

    def clear(self) -> tuple[int, int]:
        """Clear volatile cache state and return released entry and byte counts."""
        ...


@dataclass(slots=True)
class _CacheEntry[ValueT]:
    """Store one cached value and its accounting metadata."""

    value: ValueT
    size_bytes: int
    expires_at: float


class BoundedLruCache[Key, Value]:
    """Keep recently used values within entry, byte, item, and idle-age limits.

    The cache is synchronous and thread-safe; callers can dispatch its methods to an
    executor when they must not block an asyncio event loop.
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
        self._lock = RLock()

    @property
    def enabled(self) -> bool:
        """Return whether all required capacity limits permit storing entries."""
        return bool(self.max_entries and self.max_bytes and self.max_item_bytes)

    @property
    def total_bytes(self) -> int:
        """Return the accounted bytes currently retained by the cache."""
        with self._lock:
            return self._total_bytes

    def __len__(self) -> int:
        """Return the number of live entries after removing expired values."""
        with self._lock:
            self.prune()
            return len(self._entries)

    def get(self, key: Key) -> Value | None:
        """Return and refresh a cached value, or ``None`` when it is unavailable."""
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
            released = (len(self._entries), self._total_bytes)
            self._entries.clear()
            self._total_bytes = 0
            return released

    def _expires_at(self, now: float) -> float:
        """Calculate an idle deadline, using infinity when expiration is disabled."""
        return now + self.max_idle_seconds if self.max_idle_seconds else float("inf")


class PersistentAudioCache:
    """Keep bounded PCM results on disk and optionally mirror them in RAM."""

    def __init__(
        self,
        directory: Path,
        memory_cache: BoundedLruCache[bytes, bytes],
        *,
        max_entries: int,
        max_bytes: int,
        max_item_bytes: int,
        max_idle_seconds: float,
    ) -> None:
        """Initialize the persistent cache under an application cache directory."""
        if min(max_entries, max_bytes, max_item_bytes) < 0 or max_idle_seconds < 0:
            raise ValueError("disk cache limits must not be negative")
        self.directory = directory
        self.memory_cache = memory_cache
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.disk_max_item_bytes = max_item_bytes
        self.max_idle_seconds = max_idle_seconds
        self._lock = RLock()
        self._puts_since_prune = 0
        self._disk_total_bytes = 0
        self._disk_entries = 0
        self.directory.mkdir(parents=True, exist_ok=True)
        with suppress(OSError):
            self.directory.chmod(0o700)
        if self._disk_enabled:
            self._prune_disk_entries()

    @property
    def enabled(self) -> bool:
        """Return whether either disk or memory capacity enables caching."""
        return self.memory_cache.enabled or bool(
            self.max_entries and self.max_bytes and self.disk_max_item_bytes
        )

    @property
    def max_item_bytes(self) -> int:
        """Return the largest item accepted by either cache tier."""
        return max(self.memory_cache.max_item_bytes, self.disk_max_item_bytes)

    @property
    def disk_total_bytes(self) -> int:
        """Return the accounted bytes currently retained on disk."""
        with self._lock:
            return self._disk_total_bytes

    @property
    def disk_entries(self) -> int:
        """Return the number of entries currently retained on disk."""
        with self._lock:
            return self._disk_entries

    def _evict_file_from_disk(self, path: Path, item_size: int) -> None:
        """Remove a stale or invalid disk cache file and update accounted metrics."""
        with suppress(OSError):
            path.unlink(missing_ok=True)
            self._disk_total_bytes = max(0, self._disk_total_bytes - item_size)
            self._disk_entries = max(0, self._disk_entries - 1)

    def get(self, key: bytes) -> bytes | None:
        """Load a memory or disk hit and refresh its idle and LRU age."""
        with self._lock:
            if (value := self.memory_cache.get(key)) is not None:
                if self._disk_enabled:
                    with suppress(OSError):
                        os.utime(self._path_for(key), None)
                return value
            if not self._disk_enabled:
                return None
            path = self._path_for(key)
            try:
                stat = path.stat()
                item_size = stat.st_size + len(path.stem) // 2
                if self.max_idle_seconds and time() - stat.st_mtime >= self.max_idle_seconds:
                    self._evict_file_from_disk(path, item_size)
                    return None
                if stat.st_size + len(key) > min(self.disk_max_item_bytes, self.max_bytes):
                    self._evict_file_from_disk(path, item_size)
                    return None
                # TTS results are signed 16-bit PCM, so a zero-length file or an odd byte count
                # can only represent corrupt audio and cannot be a valid cache hit.
                if stat.st_size == 0 or stat.st_size % 2:
                    self._evict_file_from_disk(path, item_size)
                    return None
                value = path.read_bytes()
                with suppress(OSError):
                    os.utime(path, None)
            except FileNotFoundError:
                return None
            except OSError as err:
                _LOGGER.warning("Could not read TTS disk cache entry %s: %s", path.name, err)
                return None
            self.memory_cache.put(key, value, size_bytes=len(key) + len(value))
            return value

    def put(self, key: bytes, value: bytes, *, size_bytes: int) -> bool:
        """Persist a completed result atomically and refresh the RAM copy if allowed."""
        if size_bytes < 0:
            raise ValueError("cache item size must not be negative")
        with self._lock:
            memory_stored = self.memory_cache.put(key, value, size_bytes=size_bytes)
            if (
                not self._disk_enabled
                or size_bytes > self.disk_max_item_bytes
                or size_bytes > self.max_bytes
            ):
                return memory_stored

            path = self._path_for(key)
            existing_size = 0
            try:
                stat = path.stat()
                existing_size = stat.st_size + len(path.stem) // 2
            except OSError:
                existing_size = 0

            net_byte_increase = size_bytes - existing_size
            net_entry_increase = 0 if existing_size > 0 else 1

            if (
                self._disk_total_bytes + net_byte_increase > self.max_bytes
                or self._disk_entries + net_entry_increase > self.max_entries
            ):
                target_bytes = max(0, self.max_bytes - net_byte_increase)
                target_entries = max(0, self.max_entries - net_entry_increase)
                self._prune_disk_entries(target_bytes=target_bytes, target_entries=target_entries)

            if (
                self._disk_total_bytes + net_byte_increase > self.max_bytes
                or self._disk_entries + net_entry_increase > self.max_entries
            ):
                _LOGGER.warning(
                    "Disk cache limit reached and space could not be reclaimed; "
                    "skipping persistent cache write"
                )
                return memory_stored

            temporary_path: Path | None = None
            try:
                with NamedTemporaryFile(
                    dir=self.directory, prefix=TtsCacheFile.TEMP_PREFIX, delete=False
                ) as stream:
                    temporary_path = Path(stream.name)
                    stream.write(value)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_path, path)
                self._disk_total_bytes += net_byte_increase
                self._disk_entries += net_entry_increase
                self._puts_since_prune += 1
                if self._puts_since_prune >= max(1, self.max_entries // 16):
                    self._prune_disk_entries()
                    self._puts_since_prune = 0
                return True
            except OSError as err:
                _LOGGER.warning("Could not write TTS disk cache entry %s: %s", path.name, err)
                if temporary_path is not None:
                    with suppress(OSError):
                        temporary_path.unlink(missing_ok=True)
                return memory_stored

    def clear(self) -> tuple[int, int]:
        """Release the in-memory mirror while preserving restart-persistent entries."""
        return self.memory_cache.clear()

    @property
    def _disk_enabled(self) -> bool:
        """Return whether configured disk limits permit storing entries."""
        return bool(self.max_entries and self.max_bytes and self.disk_max_item_bytes)

    def _path_for(self, key: bytes) -> Path:
        """Return the content-addressed path for a privacy-preserving cache key."""
        return self.directory / f"{key.hex()}{TtsCacheFile.PCM_SUFFIX}"

    def _cleanup_stale_temp_files(self, temp_files: list[Path], current_time: float) -> None:
        """Remove unfinished temporary files older than five minutes."""
        for temp_path in temp_files:
            try:
                stat = temp_path.stat()
                if current_time - stat.st_mtime >= 300:
                    temp_path.unlink(missing_ok=True)
            except OSError:
                continue

    def _prune_disk_entries(
        self,
        *,
        target_bytes: int | None = None,
        target_entries: int | None = None,
    ) -> None:
        """Remove expired and least-recently-used disk entries within configured limits."""
        effective_max_bytes = self.max_bytes if target_bytes is None else target_bytes
        effective_max_entries = self.max_entries if target_entries is None else target_entries
        try:
            files = list(self.directory.glob(TtsCacheFile.PCM_GLOB))
            temp_files = list(self.directory.glob(f"{TtsCacheFile.TEMP_PREFIX}*"))
        except OSError as err:
            _LOGGER.warning("Could not list TTS disk cache: %s", err)
            return

        current_time = time()
        self._cleanup_stale_temp_files(temp_files, current_time)

        live: list[tuple[float, Path, int]] = []
        total_bytes = 0
        for path in files:
            try:
                stat = path.stat()
                if self.max_idle_seconds and current_time - stat.st_mtime >= self.max_idle_seconds:
                    path.unlink(missing_ok=True)
                    continue
            except OSError:
                continue
            item_size = stat.st_size + len(path.stem) // 2
            live.append((stat.st_mtime, path, item_size))
            total_bytes += item_size

        live.sort(key=lambda item: item[0])
        while live and (len(live) > effective_max_entries or total_bytes > effective_max_bytes):
            _, oldest_path, size_bytes = live.pop(0)
            try:
                oldest_path.unlink(missing_ok=True)
                total_bytes -= size_bytes
            except OSError as err:
                _LOGGER.warning(
                    "Could not delete TTS disk cache entry %s: %s", oldest_path.name, err
                )
                live.append((0.0, oldest_path, size_bytes))
                break

        self._disk_total_bytes = total_bytes
        self._disk_entries = len(live)
