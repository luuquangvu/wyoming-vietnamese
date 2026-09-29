"""Persist recently used TTS voices to guide startup warmup."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import RLock

_LOGGER = logging.getLogger(__name__)
_MAX_RECENT_VOICES = 32


class RecentVoiceStore:
    """Keep a small most-recently-used voice list across process restarts."""

    def __init__(self, path: Path) -> None:
        """Initialize a recent voice store at the supplied persistent path."""
        self.path = path
        self._lock = RLock()

    def get_recent(self) -> tuple[str, ...]:
        """Return voice names ordered from most to least recently used."""
        with self._lock:
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                return ()
            except (OSError, UnicodeError, json.JSONDecodeError) as err:
                _LOGGER.warning("Could not read recent TTS voices from %s: %s", self.path, err)
                return ()
            if not isinstance(data, list):
                return ()
            voices = [voice for voice in data if isinstance(voice, str) and voice]
            return tuple(dict.fromkeys(voices))

    def record(self, voice_name: str) -> None:
        """Move a successfully selected voice to the newest position atomically."""
        if not voice_name:
            return
        with self._lock:
            current = self.get_recent()
            if current[:1] == (voice_name,):
                return
            recent = [voice for voice in current if voice != voice_name]
            recent.insert(0, voice_name)
            temporary_path: Path | None = None
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=self.path.parent,
                    prefix=".recent-voices-",
                    delete=False,
                ) as stream:
                    temporary_path = Path(stream.name)
                    json.dump(recent[:_MAX_RECENT_VOICES], stream, ensure_ascii=False)
                    stream.write("\n")
                os.replace(temporary_path, self.path)
            except OSError as err:
                _LOGGER.warning("Could not persist recent TTS voice %s: %s", voice_name, err)
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
