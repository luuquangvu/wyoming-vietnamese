"""Vietnamese text-to-speech engines and Wyoming event handler."""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import random
import re
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import Executor
from contextlib import suppress
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Protocol

import numpy as np
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error as WyomingError
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
    SynthesizeTextFormat,
    SynthesizeVoice,
)

from .cache import BoundedLruCache
from .config import resolve_cpu_threads
from .const import (
    DEFAULT_EVENT_TIMEOUT,
    DEFAULT_TTS_CLAUSE_SILENCE_MS,
    DEFAULT_TTS_SENTENCE_SILENCE_MS,
    DEFAULT_TTS_SILENCE_JITTER_PERCENT,
    DEFAULT_WRITE_TIMEOUT,
    MAX_TTS_SILENCE_MS,
    PROGRAM_NAME,
    TTS_CHUNK_SIZE,
    TTS_CONFIG_FILE,
    TTS_MODEL_FILE,
    TTS_OUTPUT_QUEUE_CHUNKS,
    TTS_SAMPLE_CHANNELS,
    TTS_SAMPLE_RATE,
    TTS_SAMPLE_WIDTH,
    TTS_TOKENS_FILE,
    VIETNAMESE_LANGUAGE,
)
from .inference import run_inference
from .protocol import ProtocolWriteError, SafeAsyncEventHandler, is_vietnamese_language
from .tts_model import DEFAULT_TTS_VOICE, TTS_VOICES_BY_ID, TtsVoiceSpec

_LOGGER = logging.getLogger(__name__)


class TTSEngine(Protocol):
    """NghiTTS engine surface used by the server."""

    @property
    def _preset_voices(self) -> Mapping[str, object]:
        """Mapping of voice identifiers to preset metadata."""
        ...

    @property
    def sample_rate(self) -> int:
        """PCM sample rate produced by the selected engine."""
        ...

    def infer_stream(self, text: str, *, voice: str | None = None) -> Iterator[np.ndarray]:
        """Yield a synthesized waveform incrementally."""
        ...

    def close(self) -> None:
        """Release model resources."""


class _SherpaGeneratedAudio(Protocol):
    """Audio result returned by Sherpa-ONNX offline synthesis."""

    samples: np.ndarray
    sample_rate: int


class _SherpaOfflineTts(Protocol):
    """Native Sherpa-ONNX TTS surface used by the NghiTTS adapter."""

    sample_rate: int
    num_speakers: int

    def generate(
        self,
        text: str,
        sid: int = 0,
        speed: float = 1.0,
        callback: Callable[[np.ndarray, float], int] | None = None,
    ) -> _SherpaGeneratedAudio:
        """Generate speech and optionally report completed sentence audio."""
        ...


class _NghiDone:
    """Mark completion in the bounded NghiTTS callback queue."""


_NGHI_DONE = _NghiDone()


class _NghiAudioIterator(Iterator[np.ndarray]):
    """Bridge Sherpa's native callback API to a bounded blocking iterator."""

    def __init__(self, tts: _SherpaOfflineTts, text: str, speaker_id: int) -> None:
        """Start one native synthesis worker and expose its sentence chunks."""
        self._tts = tts
        self._text = text
        self._speaker_id = speaker_id
        self._queue: queue.Queue[np.ndarray | Exception | _NghiDone] = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._emitted_audio = False
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="NghiTTS inference",
            daemon=True,
        )
        self._thread.start()

    def __iter__(self) -> _NghiAudioIterator:
        """Return this callback-backed iterator."""
        return self

    def __next__(self) -> np.ndarray:
        """Wait for the next completed sentence or propagate worker failure."""
        if self._closed:
            raise StopIteration
        item = self._queue.get()
        if isinstance(item, np.ndarray):
            return item
        if isinstance(item, Exception):
            raise item
        self._closed = True
        self._thread.join()
        raise StopIteration

    def _put(self, item: np.ndarray | Exception | _NghiDone) -> bool:
        """Put an item without allowing cancellation to deadlock the worker."""
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _on_audio(self, samples: np.ndarray, _progress: float) -> int:
        """Copy native callback memory and request continued sentence generation."""
        audio = np.array(samples, dtype=np.float32, copy=True).ravel()
        if audio.size:
            self._emitted_audio = True
            if not self._put(audio):
                return 0
        return 0 if self._stop.is_set() else 1

    def _run(self) -> None:
        """Run the blocking Sherpa call and terminate the iterator exactly once."""
        try:
            generated = self._tts.generate(
                self._text,
                sid=self._speaker_id,
                speed=1.0,
                callback=self._on_audio,
            )
            if not self._emitted_audio and not self._stop.is_set():
                fallback = np.array(generated.samples, dtype=np.float32, copy=True).ravel()
                if fallback.size:
                    self._put(fallback)
        except Exception as err:
            self._put(err)
        finally:
            self._put(_NGHI_DONE)

    def close(self) -> None:
        """Stop after the current native unit and release queued callback chunks."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        while self._thread.is_alive():
            with suppress(queue.Empty):
                self._queue.get_nowait()
            self._thread.join(timeout=0.1)


class NghiTTSEngine:
    """Adapt a non-autoregressive NghiTTS voice to the service API."""

    def __init__(self, tts: _SherpaOfflineTts, voice: TtsVoiceSpec) -> None:
        """Validate the native model contract and expose one named preset voice."""
        if tts.sample_rate != TTS_SAMPLE_RATE:
            raise RuntimeError("NghiTTS returned an unexpected sample rate")
        if tts.num_speakers != 1:
            raise RuntimeError("NghiTTS returned an unexpected speaker count")
        self._tts: _SherpaOfflineTts | None = tts
        self.voice = voice
        self.sample_rate = tts.sample_rate
        self._preset_voices: Mapping[str, Mapping[str, object]] = {
            voice.name: {"description": "NghiTTS voice"}
        }

    def _native_tts(self) -> _SherpaOfflineTts:
        """Return the live native engine or reject use after shutdown."""
        if self._tts is None:
            raise RuntimeError("NghiTTS engine is closed")
        return self._tts

    def _speaker_id(self, voice: str | None) -> int:
        """Map the single Wyoming voice identifier to the native speaker ID."""
        if voice not in (None, self.voice.name):
            raise ValueError(f"Unsupported NghiTTS voice: {voice!r}")
        return 0

    def infer_stream(
        self,
        text: str,
        *,
        voice: str | None = None,
    ) -> Iterator[np.ndarray]:
        """Stream completed native sentence chunks through a bounded iterator."""
        return _NghiAudioIterator(
            self._native_tts(),
            text,
            self._speaker_id(voice),
        )

    def close(self) -> None:
        """Release the native model reference."""
        self._tts = None


class MultiVoiceTTSEngine:
    """Route synthesis requests to one initialized engine per configured voice."""

    def __init__(self, engines: Iterable[NghiTTSEngine]) -> None:
        """Validate and retain a non-empty collection of compatible engines."""
        engine_list = list(engines)
        if not engine_list:
            raise ValueError("At least one TTS voice must be configured")
        sample_rates = {engine.sample_rate for engine in engine_list}
        if len(sample_rates) != 1:
            raise ValueError("Configured TTS voices must use the same sample rate")
        self._engines = {engine.voice.name: engine for engine in engine_list}
        if len(self._engines) != len(engine_list):
            raise ValueError("Configured TTS voices must have unique names")
        self._default_engine = engine_list[0]
        self.sample_rate = self._default_engine.sample_rate
        self._preset_voices: Mapping[str, object] = {
            engine.voice.name: {"description": "NghiTTS voice"} for engine in engine_list
        }

    def _engine(self, voice: str | None) -> NghiTTSEngine:
        """Resolve an explicit voice name or return the first configured engine."""
        if voice is None:
            return self._default_engine
        try:
            return self._engines[voice]
        except KeyError as err:
            raise ValueError(f"Unsupported NghiTTS voice: {voice!r}") from err

    def infer_stream(
        self,
        text: str,
        *,
        voice: str | None = None,
    ) -> Iterator[np.ndarray]:
        """Stream a waveform with the selected voice."""
        return self._engine(voice).infer_stream(text)

    def close(self) -> None:
        """Release every initialized voice engine."""
        for engine in self._engines.values():
            engine.close()


def get_tts_sample_rate(tts: TTSEngine) -> int:
    """Read and validate the sample rate advertised by an initialized engine."""
    sample_rate = getattr(tts, "sample_rate", TTS_SAMPLE_RATE)
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError(f"Invalid TTS engine sample rate: {sample_rate!r}")
    return sample_rate


@dataclass(slots=True)
class _TTSProduction:
    """Metrics and cache state collected during one synthesis segment."""

    inference_api: str = "unknown"
    total_bytes: int = 0
    engine_chunk_count: int = 0
    engine_seconds: float = 0.0
    pcm_seconds: float = 0.0
    cleanup_seconds: float = 0.0
    cache_parts: list[bytes] | None = None


class _OutputBackpressureError(RuntimeError):
    """Raised when network backpressure outlives the bounded output pipeline."""


def _next_pcm_chunk(
    iterator: Iterator[np.ndarray],
) -> tuple[bool, bytes | None, float, float]:
    """Advance inference by one chunk and convert it, timing both blocking steps.

    Callers dispatch this to the inference executor so neither step runs on the event
    loop, and report the returned durations separately from their own dispatch overhead.
    """
    engine_started = perf_counter()
    try:
        audio = next(iterator)
    except StopIteration:
        return False, None, perf_counter() - engine_started, 0.0

    engine_seconds = perf_counter() - engine_started
    pcm_started = perf_counter()
    audio_bytes = _audio_to_pcm_bytes(audio)
    return True, audio_bytes, engine_seconds, perf_counter() - pcm_started


def _audio_to_pcm_bytes(audio: np.ndarray) -> bytes:
    """Convert arbitrary numeric audio safely into clipped little-endian PCM16."""
    samples = np.asarray(audio, dtype=np.float32).ravel()
    if not samples.size:
        return b""
    # Copy once: the remaining conversion is in place and must not touch engine memory.
    normalized = np.nan_to_num(samples, copy=True, nan=0.0, posinf=1.0, neginf=-1.0)
    np.clip(normalized, -1.0, 1.0, out=normalized)
    normalized *= 32767.0
    return normalized.astype("<i2").tobytes()


_SENTENCE_END_CHARS = frozenset(".!?…")
_CLAUSE_END_CHARS = frozenset(",;:\u2013\u2014-")
_LINE_BREAK_CHARS = frozenset("\n")
# Subset of the delimiters above that also separate digits: "33,8", "1.000", "10:30".
_NUMERIC_SEPARATOR_CHARS = frozenset(",.:\u2013\u2014-")
# Subset of the delimiters above that also joins one token: "Wi-Fi", "e-mail", "TP-HCM".
_TOKEN_JOINING_CHARS = frozenset("\u2013\u2014-")
# Closing punctuation may follow a boundary character; rstrip() requires text.
_TRAILING_WRAPPER_CHARS = "\"')]}\u00bb\u201d\u2019"
# Shortest clause a punctuation boundary may produce, and shortest standalone final tail.
_MIN_CLAUSE_CHARS = 15


def _delimiter_pattern(boundary_chars: frozenset[str]) -> re.Pattern[str]:
    """Match a run of boundary characters and the whitespace that follows it."""
    char_class = re.escape("".join(sorted(boundary_chars)))
    return re.compile(f"([{char_class}]+)(\\s*)")


_CLAUSE_DELIM_RE = _delimiter_pattern(_CLAUSE_END_CHARS | _LINE_BREAK_CHARS)
_SENTENCE_DELIM_RE = _delimiter_pattern(_SENTENCE_END_CHARS | _LINE_BREAK_CHARS)


def _is_numeric_separator(text: str, start: int, end: int) -> bool:
    """Return whether one delimiter sits between the digits of a single number."""
    if start == 0 or not text[start - 1].isdigit():
        return False
    # A delimiter closing the buffer may still be waiting for the rest of its number.
    return end >= len(text) or text[end].isdigit()


def _is_token_joining_dash(text: str, start: int, trailing_space: str) -> bool:
    """Return whether one dash joins a hyphenated token instead of ending a clause.

    A spaced dash ("Sài Gòn - Hà Nội") separates clauses, while an unspaced dash
    ("Wi-Fi") belongs to the token. The delimiter pattern consumes trailing
    whitespace greedily, so an empty ``trailing_space`` means the dash is either
    followed by a word or is still waiting for the rest of its token.
    """
    return not trailing_space and start > 0 and not text[start - 1].isspace()


def _is_intra_token_separator(text: str, match: re.Match[str]) -> bool:
    """Return whether a delimiter sits inside one token instead of ending a clause."""
    delimiter = match.group(1)
    if len(delimiter) != 1:
        return False
    start, end = match.start(1), match.end(1)
    if delimiter in _NUMERIC_SEPARATOR_CHARS and _is_numeric_separator(text, start, end):
        return True
    return delimiter in _TOKEN_JOINING_CHARS and _is_token_joining_dash(text, start, match.group(2))


def _silence_pcm(milliseconds: int, sample_rate: int) -> bytes:
    """Build little-endian PCM16 silence of the requested duration."""
    if milliseconds < 0:
        raise ValueError("Silence duration must not be negative")
    samples = round(sample_rate * milliseconds / 1000)
    return bytes(samples * TTS_SAMPLE_WIDTH * TTS_SAMPLE_CHANNELS)


def _jittered_silence_ms(base_ms: int, jitter_percent: int, rng: random.Random) -> int:
    """Spread a pause around its configured duration so boundaries stop sounding metronomic."""
    if base_ms < 0:
        raise ValueError("Silence duration must not be negative")
    if jitter_percent < 0:
        raise ValueError("Silence jitter must not be negative")
    if not base_ms or not jitter_percent:
        return base_ms
    spread = base_ms * jitter_percent / 100
    jittered = round(rng.uniform(base_ms - spread, base_ms + spread))
    return min(max(jittered, 0), MAX_TTS_SILENCE_MS)


def _boundary_kind(text: str) -> str:
    """Classify the punctuation that a synthesized segment ends on."""
    trimmed = text.rstrip().rstrip(_TRAILING_WRAPPER_CHARS).rstrip()
    if not trimmed:
        return "none"
    last_char = trimmed[-1]
    if last_char in _SENTENCE_END_CHARS:
        return "sentence"
    return "clause" if last_char in _CLAUSE_END_CHARS else "none"


class StreamClauseDetector:
    """Incremental text boundary detector for real-time TTS synthesis.

    Yields chunks incrementally at configured sentence, clause, and line boundaries.
    Unpunctuated text remains intact until the stream finishes so synthesis does not
    sound clipped.
    """

    def __init__(self, min_clause_chars: int = _MIN_CLAUSE_CHARS) -> None:
        """Initialize the minimum length for a clause-level punctuation boundary."""
        self.min_clause_chars = min_clause_chars
        self.buffer = ""
        # Whitespace discarded after the most recent boundary, kept so a merged
        # final tail can be restored without rewriting the requested text.
        self.trailing_separator = ""

    def add_chunk(self, text: str) -> Iterable[str]:
        """Buffer incoming streaming text and yield synthesis-ready clauses."""
        self.buffer += text
        return self._drain()

    def finish(self) -> str:
        """Flush all remaining text at stream end."""
        text = self.buffer.strip()
        self.buffer = ""
        return text

    @staticmethod
    def split_clause_text(text: str) -> list[str]:
        """Split a static block only at configured text boundaries."""
        trimmed = text.strip()
        if not trimmed:
            return []
        detector = StreamClauseDetector()
        clauses = list(detector.add_chunk(trimmed))
        remainder = detector.finish()
        if not remainder:
            return clauses
        if not clauses or len(remainder) >= _MIN_CLAUSE_CHARS:
            clauses.append(remainder)
            return clauses
        # Re-attach a short tail using the separator the boundary discarded, so the
        # engine receives the requested text rather than a reconstruction of it.
        clauses[-1] = f"{clauses[-1]}{detector.trailing_separator}{remainder}"
        return clauses

    def _drain(self) -> list[str]:
        """Extract configured text boundaries without length-based splitting."""
        yielded: list[str] = []
        while self.buffer:
            match = self._find_natural_boundary()
            if match is None:
                break

            window = self.buffer
            end_pos = match.end()
            head = window[:end_pos]
            leading = len(head) - len(head.lstrip())
            piece = head.strip()
            self.buffer = window[end_pos:].lstrip()
            self.trailing_separator = window[leading + len(piece) : len(window) - len(self.buffer)]
            if piece:
                yielded.append(piece)
        return yielded

    def _first_text_boundary(self, matches: Iterable[re.Match[str]]) -> re.Match[str] | None:
        """Return the earliest delimiter that ends a clause rather than joining a token."""
        return next(
            (match for match in matches if not _is_intra_token_separator(self.buffer, match)),
            None,
        )

    def _find_natural_boundary(self) -> re.Match[str] | None:
        """Find the earliest sentence or sufficiently long clause delimiter."""
        sentence_match = self._first_text_boundary(_SENTENCE_DELIM_RE.finditer(self.buffer))
        clause_match = self._first_text_boundary(
            match
            for match in _CLAUSE_DELIM_RE.finditer(self.buffer)
            if match.end() >= self.min_clause_chars
        )
        if sentence_match is not None and clause_match is not None:
            return sentence_match if sentence_match.start() < clause_match.start() else clause_match
        return sentence_match or clause_match


class TTSEventHandler(SafeAsyncEventHandler):
    """Handle bounded one-shot and streaming synthesis using a shared engine."""

    def __init__(
        self,
        tts: TTSEngine,
        wyoming_info_event: Event,
        inference_lock: asyncio.Lock,
        result_cache: BoundedLruCache[bytes, bytes],
        max_text_chars: int,
        queue_timeout: float,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        event_timeout: float = DEFAULT_EVENT_TIMEOUT,
        write_timeout: float = DEFAULT_WRITE_TIMEOUT,
        sentence_silence_ms: int = DEFAULT_TTS_SENTENCE_SILENCE_MS,
        clause_silence_ms: int = DEFAULT_TTS_CLAUSE_SILENCE_MS,
        silence_jitter_percent: int = DEFAULT_TTS_SILENCE_JITTER_PERCENT,
        rng: random.Random | None = None,
        inference_executor: Executor | None = None,
    ) -> None:
        """Initialize per-connection state around the shared TTS engine."""
        super().__init__(
            reader,
            writer,
            event_timeout=event_timeout,
            write_timeout=write_timeout,
        )
        self.tts = tts
        self.inference_executor = inference_executor
        self.wyoming_info_event = wyoming_info_event
        self.inference_lock = inference_lock
        self.result_cache = result_cache
        self.max_text_chars = max_text_chars
        self.queue_timeout = queue_timeout
        self.sample_rate = get_tts_sample_rate(tts)
        self.sentence_silence_ms = sentence_silence_ms
        self.clause_silence_ms = clause_silence_ms
        self.silence_jitter_percent = silence_jitter_percent
        self._rng = rng if rng is not None else random.Random()
        self.preset_voices = dict(getattr(tts, "_preset_voices", {}))
        self.stream_voice_name: str | None
        # Survives _reset_stream() so a late compatibility request cannot restart a
        # lifecycle that already ended. Reset only when the connection is released.
        self.stream_lifecycle_served = False
        self._reset_stream()

    async def handle_event(self, event: Event) -> bool:
        """Handle service discovery and one-shot or streaming text requests."""
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent TTS service information")
            return True

        if SynthesizeStart.is_type(event.type):
            return await self._handle_stream_start(event)

        if SynthesizeChunk.is_type(event.type):
            return await self._handle_stream_chunk(event)

        if SynthesizeStop.is_type(event.type):
            return await self._handle_stream_stop()

        if Synthesize.is_type(event.type):
            if self.is_streaming or self.stream_lifecycle_served:
                # Streaming clients send the complete request for compatibility, either
                # inside the stream or after it. Re-synthesizing it would duplicate the
                # whole reply on a lifecycle that already sent synthesize-stopped.
                return True
            return await self._handle_synthesize(event)

        return True

    async def _handle_synthesize(self, event: Event) -> bool:
        """Validate and execute a one-shot synthesis request."""
        try:
            synthesize = Synthesize.from_event(event)
            if not isinstance(synthesize.text, str):
                raise TypeError("text must be a string")
            voice_name = self._resolve_voice_name(synthesize.voice)
        except KeyError, TypeError, ValueError:
            return await self._fail("Malformed synthesis request", "invalid-request")

        text = synthesize.text.strip()
        if not text:
            return await self._fail("Synthesis text must not be empty", "empty-text")
        if len(text) > self.max_text_chars:
            return await self._fail("Synthesis text exceeds the configured limit", "text-too-long")
        if not self._is_supported_text_format(synthesize.text_format):
            return await self._fail(
                "Only plain text synthesis is supported",
                "unsupported-text-format",
            )

        _LOGGER.info(
            "Starting TTS request: text_chars=%d, voice=%s",
            len(text),
            voice_name or "default",
        )
        request_started = perf_counter()
        production = await self._synthesize(text, voice_name)
        success = production is not None
        audio_seconds = (
            production.total_bytes / (self.sample_rate * TTS_SAMPLE_WIDTH * TTS_SAMPLE_CHANNELS)
            if production is not None
            else 0.0
        )
        engine_seconds = production.engine_seconds if production is not None else 0.0
        real_time_factor = engine_seconds / audio_seconds if audio_seconds else 0.0
        _LOGGER.info(
            "Completed TTS request: text_chars=%d voice=%s success=%s audio_seconds=%.3f "
            "engine_ms=%.3f handle_ms=%.3f real_time_factor=%.3f",
            len(text),
            voice_name or "default",
            success,
            audio_seconds,
            engine_seconds * 1000,
            (perf_counter() - request_started) * 1000,
            real_time_factor,
        )
        return False

    async def _handle_stream_start(self, event: Event) -> bool:
        """Initialize a Wyoming streaming-text synthesis lifecycle."""
        if self.is_streaming:
            return await self._fail_stream(
                "Synthesis stream already started",
                "invalid-event-order",
            )

        try:
            stream_start = SynthesizeStart.from_event(event)
            voice_name = self._resolve_voice_name(stream_start.voice)
        except KeyError, TypeError, ValueError:
            return await self._fail_stream(
                "Malformed synthesis stream start",
                "invalid-request",
            )
        if not self._is_supported_text_format(stream_start.text_format):
            return await self._fail_stream(
                "Only plain text synthesis is supported",
                "unsupported-text-format",
            )

        self._reset_stream()
        self.is_streaming = True
        self.stream_started_at = perf_counter()
        self.stream_voice_name = voice_name
        _LOGGER.debug(
            "TTS step=text-stream-start voice=%s max_text_chars=%d",
            voice_name or "default",
            self.max_text_chars,
        )
        return True

    async def _handle_stream_chunk(self, event: Event) -> bool:
        """Buffer arbitrary text fragments and synthesize complete text segments."""
        if not self.is_streaming:
            return await self._fail_stream(
                "Synthesis chunk received before stream start",
                "invalid-event-order",
            )

        try:
            stream_chunk = SynthesizeChunk.from_event(event)
            if not isinstance(stream_chunk.text, str):
                raise TypeError("text must be a string")
        except KeyError, TypeError, ValueError:
            return await self._fail_stream(
                "Malformed synthesis stream chunk",
                "invalid-request",
            )

        self.stream_chunk_count += 1
        self.stream_text_chars += len(stream_chunk.text)
        if self.stream_text_chars > self.max_text_chars:
            return await self._fail_stream(
                "Synthesis text exceeds the configured limit",
                "text-too-long",
            )

        for clause in self.sentence_boundary.add_chunk(stream_chunk.text):
            self.stream_segment_count += 1
            _LOGGER.debug(
                "TTS step=sentence-ready segment=%d text_chars=%d",
                self.stream_segment_count,
                len(clause),
            )
            if not await self._synthesize(
                clause,
                self.stream_voice_name,
                send_start=not self.stream_audio_started,
                send_stop=False,
            ):
                await self.write_event(SynthesizeStopped().event())
                self._reset_stream()
                return False
            self.stream_audio_started = True

        return True

    async def _handle_stream_stop(self) -> bool:
        """Synthesize buffered text and terminate the streaming response."""
        if not self.is_streaming:
            return await self._fail_stream(
                "Synthesis stop received before stream start",
                "invalid-event-order",
            )

        stop_started = perf_counter()
        self.stream_lifecycle_served = True
        final_text = self.sentence_boundary.finish()
        _LOGGER.debug(
            "TTS step=text-stream-stop chunks=%d completed_segments=%d final_text_chars=%d",
            self.stream_chunk_count,
            self.stream_segment_count,
            len(final_text),
        )
        if final_text:
            clauses = StreamClauseDetector.split_clause_text(final_text)
            for idx, clause in enumerate(clauses):
                self.stream_segment_count += 1
                is_last_clause = idx == len(clauses) - 1
                _LOGGER.debug(
                    "TTS step=sentence-ready segment=%d text_chars=%d final=%s",
                    self.stream_segment_count,
                    len(clause),
                    is_last_clause,
                )
                if not await self._synthesize(
                    clause,
                    self.stream_voice_name,
                    send_start=not self.stream_audio_started,
                    send_stop=is_last_clause,
                ):
                    await self.write_event(SynthesizeStopped().event())
                    self._reset_stream()
                    return False
                self.stream_audio_started = True
        elif self.stream_audio_started:
            write_started = perf_counter()
            await self.write_event(AudioStop().event())
            self.stream_output_seconds += perf_counter() - write_started
        else:
            return await self._fail_stream("Synthesis text must not be empty", "empty-text")

        write_started = perf_counter()
        await self.write_event(SynthesizeStopped().event())
        self.stream_output_seconds += perf_counter() - write_started
        completed_at = perf_counter()
        stream_started = self.stream_started_at or stop_started
        audio_seconds = self.stream_audio_bytes / (
            self.sample_rate * TTS_SAMPLE_WIDTH * TTS_SAMPLE_CHANNELS
        )
        real_time_factor = self.stream_engine_seconds / audio_seconds if audio_seconds else 0.0
        first_audio_ms = (
            (self.stream_first_audio_at - stream_started) * 1000
            if self.stream_first_audio_at is not None
            else 0.0
        )
        _LOGGER.info(
            "Completed streaming TTS request: text_chunks=%d segments=%d text_chars=%d "
            "audio_bytes=%d audio_seconds=%.3f queue_ms=%.3f engine_ms=%.3f "
            "pcm_ms=%.3f output_ms=%.3f receive_ms=%.3f handle_ms=%.3f "
            "first_audio_ms=%.3f total_ms=%.3f real_time_factor=%.3f",
            self.stream_chunk_count,
            self.stream_segment_count,
            self.stream_text_chars,
            self.stream_audio_bytes,
            audio_seconds,
            self.stream_queue_seconds * 1000,
            self.stream_engine_seconds * 1000,
            self.stream_pcm_seconds * 1000,
            self.stream_output_seconds * 1000,
            (stop_started - stream_started) * 1000,
            self.stream_synthesis_seconds * 1000,
            first_audio_ms,
            (completed_at - stream_started) * 1000,
            real_time_factor,
        )
        self._reset_stream()
        return True

    @staticmethod
    def _is_supported_text_format(text_format: object) -> bool:
        """Return whether the requested Wyoming text format is supported."""
        return text_format in (None, "text", SynthesizeTextFormat.TEXT)

    def _resolve_voice_name(self, voice: SynthesizeVoice | None) -> str | None:
        """Validate Wyoming voice selectors and resolve a preset voice name."""
        if voice is None:
            return None

        voice_name = voice.name
        if voice_name is not None:
            if not isinstance(voice_name, str):
                raise TypeError("voice name must be text")
            catalog_voice = TTS_VOICES_BY_ID.get(voice_name)
            if catalog_voice is not None:
                voice_name = catalog_voice.name
        if voice_name is not None and voice_name not in self.preset_voices:
            raise ValueError(f"Requested TTS voice is unavailable: {voice_name}")

        language = voice.language
        if language is not None and not isinstance(language, str):
            raise TypeError("voice language must be text")
        if language is not None and not is_vietnamese_language(language):
            raise ValueError(f"Requested TTS language is unavailable: {language}")

        speaker = voice.speaker
        if speaker is not None:
            if not isinstance(speaker, str):
                raise TypeError("voice speaker must be text")
            raise ValueError("This TTS service does not expose selectable speakers")
        return voice_name

    def _silence_pcm(self, base_ms: int) -> bytes:
        """Draw one randomized pause of the configured length as PCM silence."""
        return _silence_pcm(
            _jittered_silence_ms(base_ms, self.silence_jitter_percent, self._rng),
            self.sample_rate,
        )

    def _boundary_silence_pcm(self, text: str) -> bytes:
        """Return padding for a completed segment when another segment follows.

        Non-final segments end at a recognized sentence, clause, or line boundary;
        the boundary type selects the configured pause.
        """
        if _boundary_kind(text) == "clause":
            return self._silence_pcm(self.clause_silence_ms)
        return self._silence_pcm(self.sentence_silence_ms)

    async def _write_silence(self, silence: bytes) -> float:
        """Write padding silence as protocol audio chunks and report elapsed time."""
        write_started = perf_counter()
        for offset in range(0, len(silence), TTS_CHUNK_SIZE):
            await self.write_event(
                AudioChunk(
                    rate=self.sample_rate,
                    width=TTS_SAMPLE_WIDTH,
                    channels=TTS_SAMPLE_CHANNELS,
                    audio=silence[offset : offset + TTS_CHUNK_SIZE],
                ).event()
            )
        return perf_counter() - write_started

    def _cache_key(self, text: str, voice_name: str | None) -> bytes:
        """Hash text and voice without retaining request text in cache keys."""
        voice_bytes = (voice_name or "").encode("utf-8")
        digest = sha256()
        digest.update(len(voice_bytes).to_bytes(4, "big"))
        digest.update(voice_bytes)
        digest.update(text.encode("utf-8"))
        return digest.digest()

    async def _synthesize(
        self,
        text: str,
        voice_name: str | None,
        *,
        send_start: bool = True,
        send_stop: bool = True,
    ) -> _TTSProduction | None:
        """Pipeline inference and bounded network output around the shared lock."""
        segment_started = perf_counter()
        segment_number = self.stream_segment_count if self.is_streaming else 1
        _LOGGER.debug(
            "TTS step=segment-start segment=%d streaming=%s text_chars=%d voice=%s "
            "send_start=%s send_stop=%s",
            segment_number,
            self.is_streaming,
            len(text),
            voice_name or "default",
            send_start,
            send_stop,
        )
        lock_acquired = False
        audio_started = not send_start
        production = _TTSProduction()
        output_chunk_count = 0
        pause_bytes = 0
        queue_seconds = 0.0
        output_seconds = 0.0
        cache_lookup_seconds = 0.0
        first_audio_at: float | None = None
        cache_status = "disabled"
        cache_key = self._cache_key(text, voice_name) if self.result_cache.enabled else None
        cached_audio: bytes | None = None
        if cache_key is not None:
            cache_lookup_started = perf_counter()
            cached_audio = self.result_cache.get(cache_key)
            cache_lookup_seconds += perf_counter() - cache_lookup_started
            cache_status = "hit" if cached_audio is not None else "miss"

        production.cache_parts = [] if cache_key is not None else None
        producer: asyncio.Task[None] | None = None
        output_queue: asyncio.Queue[bytes | None] | None = None
        output_slots: asyncio.Semaphore | None = None
        abort_production: asyncio.Event | None = None
        try:
            if self.writer.is_closing():
                _LOGGER.debug(
                    "TTS step=inference-skipped segment=%d reason=client-closed",
                    segment_number,
                )
                return None

            if cached_audio is None:
                queue_started = perf_counter()
                _LOGGER.debug("TTS step=queue-wait timeout_seconds=%.3f", self.queue_timeout)
                try:
                    await asyncio.wait_for(
                        self.inference_lock.acquire(),
                        timeout=self.queue_timeout,
                    )
                except TimeoutError:
                    _LOGGER.debug(
                        "TTS step=queue-timeout segment=%d wait_ms=%.3f",
                        segment_number,
                        (perf_counter() - queue_started) * 1000,
                    )
                    await self._write_audio_failure(
                        "TTS inference queue is busy",
                        "server-busy",
                        audio_started=not send_start,
                    )
                    return None
                lock_acquired = True
                queue_seconds = perf_counter() - queue_started
                _LOGGER.debug(
                    "TTS step=queue-acquired segment=%d wait_ms=%.3f",
                    segment_number,
                    queue_seconds * 1000,
                )
                if self.writer.is_closing():
                    _LOGGER.debug(
                        "TTS step=inference-skipped segment=%d reason=client-closed",
                        segment_number,
                    )
                    return None
                if cache_key is not None:
                    cache_lookup_started = perf_counter()
                    cached_audio = self.result_cache.get(cache_key)
                    cache_lookup_seconds += perf_counter() - cache_lookup_started
                    if cached_audio is not None:
                        cache_status = "hit-after-wait"
                        self.inference_lock.release()
                        lock_acquired = False

            if cached_audio is not None:
                production.inference_api = "cache"
                production.total_bytes = len(cached_audio)
                production.cache_parts = None
            else:
                output_queue = asyncio.Queue(maxsize=TTS_OUTPUT_QUEUE_CHUNKS + 1)
                output_slots = asyncio.Semaphore(TTS_OUTPUT_QUEUE_CHUNKS)
                abort_production = asyncio.Event()
                # Eager start hands the inference lock over safely: the producer enters
                # its try block before this returns, so any later cancellation is
                # guaranteed to run the cleanup that releases the lock and unblocks the
                # consumer below. A task cancelled before its first step never would.
                producer = asyncio.Task(
                    self._produce_audio(
                        text,
                        voice_name,
                        cache_key,
                        output_queue,
                        output_slots,
                        abort_production,
                        production,
                    ),
                    loop=asyncio.get_running_loop(),
                    name="TTS inference producer",
                    eager_start=True,
                )
                lock_acquired = False

            if send_start:
                write_started = perf_counter()
                await self.write_event(
                    AudioStart(
                        rate=self.sample_rate,
                        width=TTS_SAMPLE_WIDTH,
                        channels=TTS_SAMPLE_CHANNELS,
                    ).event()
                )
                output_seconds += perf_counter() - write_started
                audio_started = True

            if cached_audio is not None:
                for offset in range(0, len(cached_audio), TTS_CHUNK_SIZE):
                    chunk = cached_audio[offset : offset + TTS_CHUNK_SIZE]
                    if chunk:
                        write_started = perf_counter()
                        await self.write_event(
                            AudioChunk(
                                rate=self.sample_rate,
                                width=TTS_SAMPLE_WIDTH,
                                channels=TTS_SAMPLE_CHANNELS,
                                audio=chunk,
                            ).event()
                        )
                        output_seconds += perf_counter() - write_started
                        output_chunk_count += 1
                        if first_audio_at is None:
                            first_audio_at = perf_counter()
            elif output_queue is not None and output_slots is not None and producer is not None:
                while True:
                    chunk = await output_queue.get()
                    if chunk is None:
                        break
                    try:
                        write_started = perf_counter()
                        await self.write_event(
                            AudioChunk(
                                rate=self.sample_rate,
                                width=TTS_SAMPLE_WIDTH,
                                channels=TTS_SAMPLE_CHANNELS,
                                audio=chunk,
                            ).event()
                        )
                        output_seconds += perf_counter() - write_started
                    finally:
                        output_slots.release()
                    output_chunk_count += 1
                    if first_audio_at is None:
                        first_audio_at = perf_counter()
                await producer

            if not send_stop:
                pause_pcm = self._boundary_silence_pcm(text)
                pause_bytes = len(pause_pcm)
                output_seconds += await self._write_silence(pause_pcm)

            if send_stop:
                write_started = perf_counter()
                await self.write_event(AudioStop().event())
                output_seconds += perf_counter() - write_started

            if cache_key is not None and cached_audio is None:
                if production.cache_parts is None:
                    cache_status = "miss-too-large"
                else:
                    cached_result = b"".join(production.cache_parts)
                    stored = self.result_cache.put(
                        cache_key,
                        cached_result,
                        size_bytes=len(cache_key) + len(cached_result),
                    )
                    cache_status = "miss-stored" if stored else "miss-not-stored"
        except ProtocolWriteError as err:
            if (
                producer is not None
                and output_queue is not None
                and output_slots is not None
                and abort_production is not None
            ):
                await self._abort_audio_producer(
                    producer,
                    output_queue,
                    output_slots,
                    abort_production,
                )
            _LOGGER.warning(
                "TTS output failed: segment=%d api=%s detail=%s",
                segment_number,
                production.inference_api,
                err,
            )
            return None
        except asyncio.CancelledError:
            if (
                producer is not None
                and output_queue is not None
                and output_slots is not None
                and abort_production is not None
            ):
                await self._abort_audio_producer(
                    producer,
                    output_queue,
                    output_slots,
                    abort_production,
                )
            raise
        except Exception:
            if (
                producer is not None
                and output_queue is not None
                and output_slots is not None
                and abort_production is not None
            ):
                await self._abort_audio_producer(
                    producer,
                    output_queue,
                    output_slots,
                    abort_production,
                )
            _LOGGER.exception(
                "TTS inference failed: segment=%d api=%s queue_ms=%.3f engine_ms=%.3f "
                "pcm_ms=%.3f output_ms=%.3f cache=%s elapsed_ms=%.3f",
                segment_number,
                production.inference_api,
                queue_seconds * 1000,
                production.engine_seconds * 1000,
                production.pcm_seconds * 1000,
                output_seconds * 1000,
                cache_status,
                (perf_counter() - segment_started) * 1000,
            )
            with suppress(ProtocolWriteError):
                await self._write_audio_failure(
                    "Speech synthesis failed",
                    "inference-failed",
                    audio_started=audio_started,
                )
            return None
        finally:
            if lock_acquired:
                self.inference_lock.release()

        completed_at = perf_counter()
        segment_seconds = completed_at - segment_started
        audio_seconds = production.total_bytes / (
            self.sample_rate * TTS_SAMPLE_WIDTH * TTS_SAMPLE_CHANNELS
        )
        real_time_factor = production.engine_seconds / audio_seconds if audio_seconds else 0.0
        first_audio_ms = (
            (first_audio_at - segment_started) * 1000 if first_audio_at is not None else 0.0
        )
        if self.is_streaming:
            self.stream_audio_bytes += production.total_bytes
            self.stream_queue_seconds += queue_seconds
            self.stream_engine_seconds += production.engine_seconds
            self.stream_pcm_seconds += production.pcm_seconds
            self.stream_output_seconds += output_seconds
            self.stream_synthesis_seconds += segment_seconds
            if self.stream_first_audio_at is None:
                self.stream_first_audio_at = first_audio_at
        _LOGGER.debug(
            "TTS step=segment-complete segment=%d api=%s text_chars=%d engine_chunks=%d "
            "protocol_chunks=%d audio_bytes=%d pause_bytes=%d audio_seconds=%.3f queue_ms=%.3f "
            "engine_ms=%.3f pcm_ms=%.3f output_ms=%.3f cleanup_ms=%.3f "
            "cache_lookup_ms=%.3f cache=%s first_audio_ms=%.3f handle_ms=%.3f "
            "real_time_factor=%.3f",
            segment_number,
            production.inference_api,
            len(text),
            production.engine_chunk_count,
            output_chunk_count,
            production.total_bytes,
            pause_bytes,
            audio_seconds,
            queue_seconds * 1000,
            production.engine_seconds * 1000,
            production.pcm_seconds * 1000,
            output_seconds * 1000,
            production.cleanup_seconds * 1000,
            cache_lookup_seconds * 1000,
            cache_status,
            first_audio_ms,
            segment_seconds * 1000,
            real_time_factor,
        )
        return production

    async def _produce_audio(
        self,
        text: str,
        voice_name: str | None,
        cache_key: bytes | None,
        output_queue: asyncio.Queue[bytes | None],
        output_slots: asyncio.Semaphore,
        abort: asyncio.Event,
        production: _TTSProduction,
    ) -> None:
        """Run engine iteration under the inference lock and fill a bounded queue."""
        iterator: Iterator[np.ndarray] | None = None
        try:
            production.inference_api = "infer_stream"
            _LOGGER.debug("TTS step=inference-start api=%s", production.inference_api)
            engine_started = perf_counter()
            try:
                iterator = await run_inference(
                    self.inference_executor,
                    self.tts.infer_stream,
                    text,
                    voice=voice_name,
                )
            finally:
                production.engine_seconds += perf_counter() - engine_started

            while not abort.is_set():
                worker_step_started = perf_counter()
                has_chunk, audio_bytes, engine_seconds, pcm_seconds = await run_inference(
                    self.inference_executor,
                    _next_pcm_chunk,
                    iterator,
                )
                worker_step_seconds = perf_counter() - worker_step_started
                dispatch_seconds = max(
                    worker_step_seconds - engine_seconds - pcm_seconds,
                    0.0,
                )
                production.engine_seconds += engine_seconds + dispatch_seconds
                production.pcm_seconds += pcm_seconds
                if not has_chunk:
                    break
                if audio_bytes is None:
                    continue
                if production.engine_chunk_count and self.sentence_silence_ms:
                    # Native chunks are whole sentences that otherwise abut each other.
                    audio_bytes = self._silence_pcm(self.sentence_silence_ms) + audio_bytes
                production.engine_chunk_count += 1
                production.total_bytes += len(audio_bytes)
                if (
                    production.cache_parts is not None
                    and cache_key is not None
                    and len(cache_key) + production.total_bytes <= self.result_cache.max_item_bytes
                ):
                    production.cache_parts.append(audio_bytes)
                else:
                    production.cache_parts = None

                for offset in range(0, len(audio_bytes), TTS_CHUNK_SIZE):
                    chunk = audio_bytes[offset : offset + TTS_CHUNK_SIZE]
                    if not chunk:
                        continue
                    try:
                        await asyncio.wait_for(
                            output_slots.acquire(),
                            timeout=self.write_timeout,
                        )
                    except TimeoutError as err:
                        raise _OutputBackpressureError(
                            "TTS output queue remained full past the write timeout"
                        ) from err
                    if abort.is_set():
                        output_slots.release()
                        break
                    output_queue.put_nowait(chunk)

            if not abort.is_set() and not production.total_bytes:
                raise RuntimeError("TTS engine returned no audio")
        finally:
            cleanup_started = perf_counter()
            try:
                if iterator is not None:
                    close = getattr(iterator, "close", None)
                    if callable(close):
                        with suppress(Exception):
                            await run_inference(self.inference_executor, close)
            finally:
                # Awaiting above re-raises CancelledError if this task is cancelled, so
                # the lock release and the completion sentinel must not sit behind it:
                # skipping them would strand the shared lock and the waiting consumer.
                production.cleanup_seconds = perf_counter() - cleanup_started
                self.inference_lock.release()
                output_queue.put_nowait(None)

    @staticmethod
    def _drain_audio_queue(
        output_queue: asyncio.Queue[bytes | None],
        output_slots: asyncio.Semaphore,
    ) -> None:
        """Discard queued audio and release producer slots during teardown."""
        while True:
            try:
                chunk = output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if chunk is not None:
                output_slots.release()

    async def _abort_audio_producer(
        self,
        producer: asyncio.Task[None],
        output_queue: asyncio.Queue[bytes | None],
        output_slots: asyncio.Semaphore,
        abort: asyncio.Event,
    ) -> None:
        """Stop queueing, preserve native-call cleanup, and consume task failures."""
        abort.set()
        self._drain_audio_queue(output_queue, output_slots)
        with suppress(Exception):
            await asyncio.shield(producer)
        self._drain_audio_queue(output_queue, output_slots)

    async def _write_audio_failure(self, text: str, code: str, *, audio_started: bool) -> None:
        """Write an error and deterministically close the audio lifecycle."""
        _LOGGER.debug(
            "TTS step=audio-failure code=%s detail=%s audio_started=%s",
            code,
            text,
            audio_started,
        )
        await self.write_event(WyomingError(text=text, code=code).event())
        if not audio_started:
            await self.write_event(
                AudioStart(
                    rate=self.sample_rate,
                    width=TTS_SAMPLE_WIDTH,
                    channels=TTS_SAMPLE_CHANNELS,
                ).event()
            )
        await self.write_event(AudioStop().event())

    async def _fail(self, text: str, code: str) -> bool:
        """Return an error plus an empty audio lifecycle for client compatibility."""
        _LOGGER.debug("TTS step=request-failed code=%s detail=%s", code, text)
        await self._write_audio_failure(text, code, audio_started=False)
        return False

    async def _fail_stream(self, text: str, code: str) -> bool:
        """Return a terminal error for a streaming synthesis lifecycle."""
        elapsed_seconds = (
            perf_counter() - self.stream_started_at if self.stream_started_at is not None else 0.0
        )
        _LOGGER.debug(
            "TTS step=stream-failed code=%s detail=%s text_chunks=%d segments=%d "
            "text_chars=%d elapsed_ms=%.3f",
            code,
            text,
            self.stream_chunk_count,
            self.stream_segment_count,
            self.stream_text_chars,
            elapsed_seconds * 1000,
        )
        self.stream_lifecycle_served = True
        await self._write_audio_failure(text, code, audio_started=self.stream_audio_started)
        await self.write_event(SynthesizeStopped().event())
        self._reset_stream()
        return False

    def _reset_stream(self) -> None:
        """Release all per-connection streaming text state."""
        self.is_streaming = False
        self.stream_audio_started = False
        self.stream_started_at: float | None = None
        self.stream_first_audio_at: float | None = None
        self.stream_chunk_count = 0
        self.stream_segment_count = 0
        self.stream_text_chars = 0
        self.stream_audio_bytes = 0
        self.stream_queue_seconds = 0.0
        self.stream_engine_seconds = 0.0
        self.stream_pcm_seconds = 0.0
        self.stream_output_seconds = 0.0
        self.stream_synthesis_seconds = 0.0
        self.stream_voice_name = None
        self.sentence_boundary = StreamClauseDetector()

    async def disconnect(self) -> None:
        """Release buffered streaming text when the client disconnects."""
        if self.stream_started_at is not None:
            _LOGGER.debug(
                "TTS step=disconnect text_chunks=%d segments=%d text_chars=%d elapsed_ms=%.3f",
                self.stream_chunk_count,
                self.stream_segment_count,
                self.stream_text_chars,
                (perf_counter() - self.stream_started_at) * 1000,
            )
        self.stream_lifecycle_served = False
        self._reset_stream()


def warm_up_tts(tts: TTSEngine) -> None:
    """Run a complete streaming synthesis to warm up the TTS engine."""
    warmup_started = perf_counter()
    iterator = iter(
        tts.infer_stream(
            "Xin chào.",
            voice=None,
        )
    )
    has_audio = False
    try:
        for audio in iterator:
            if np.asarray(audio).size:
                has_audio = True
        if not has_audio:
            raise RuntimeError("TTS warm-up returned no audio")
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            with suppress(Exception):
                close()

    _LOGGER.info(
        "Warmed up TTS streaming inference: duration_ms=%.3f",
        (perf_counter() - warmup_started) * 1000,
    )


def get_tts_info(voices: TtsVoiceSpec | Iterable[TtsVoiceSpec]) -> Info:
    """Build Wyoming discovery information for the configured NghiTTS voices."""
    voice_specs = (voices,) if isinstance(voices, TtsVoiceSpec) else tuple(voices)
    if not voice_specs:
        raise ValueError("At least one TTS voice must be configured")
    attribution = Attribution(
        name="nghimestudio",
        url="https://github.com/nghimestudio/nghitts",
    )
    return Info(
        tts=[
            TtsProgram(
                name=PROGRAM_NAME,
                description="NghiTTS Vietnamese (Sherpa-ONNX)",
                attribution=attribution,
                installed=True,
                version="medium",
                voices=[
                    TtsVoice(
                        name=voice.name,
                        description=voice.name,
                        attribution=attribution,
                        installed=True,
                        version="medium",
                        languages=[VIETNAMESE_LANGUAGE],
                    )
                    for voice in voice_specs
                ],
                supports_synthesize_streaming=True,
            )
        ]
    )


def initialize_tts(
    model_dir: Path,
    num_threads: int = 0,
    voice: TtsVoiceSpec = DEFAULT_TTS_VOICE,
) -> TTSEngine:
    """Initialize one shared NghiTTS engine from required local assets."""
    model_path = model_dir / TTS_MODEL_FILE
    config_path = model_dir / TTS_CONFIG_FILE
    tokens_path = model_dir / TTS_TOKENS_FILE
    for label, path in (
        ("ONNX model", model_path),
        ("model configuration", config_path),
        ("token table", tokens_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required NghiTTS {label} missing: {path!s}")
    espeak_data_dir = _resolve_nghitts_espeak_data(model_dir)

    import sherpa_onnx

    threads = resolve_cpu_threads(num_threads)
    _LOGGER.info(
        "Initializing NghiTTS: voice=%s threads=%d model=%s espeak_data=%s",
        voice.name,
        threads,
        model_path,
        espeak_data_dir,
    )
    initialization_started = perf_counter()
    native_tts: _SherpaOfflineTts = sherpa_onnx.OfflineTts(
        sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(model_path.resolve()),
                    tokens=str(tokens_path.resolve()),
                    data_dir=str(espeak_data_dir),
                ),
                num_threads=threads,
                debug=False,
                provider="cpu",
            ),
            max_num_sentences=1,
        )
    )
    engine = NghiTTSEngine(native_tts, voice)
    _LOGGER.info(
        "Initialized NghiTTS: voice=%s sample_rate=%d duration_ms=%.3f",
        voice.name,
        engine.sample_rate,
        (perf_counter() - initialization_started) * 1000,
    )
    return engine


def initialize_tts_voices(
    model_root: Path,
    voices: tuple[TtsVoiceSpec, ...],
    num_threads: int = 0,
) -> TTSEngine:
    """Initialize and combine all configured voices from voice-specific directories."""
    if not voices:
        raise ValueError("At least one TTS voice must be configured")
    engines: list[NghiTTSEngine] = []
    try:
        for voice in voices:
            engine = initialize_tts(model_root / voice.id, num_threads, voice)
            if not isinstance(engine, NghiTTSEngine):
                raise TypeError("initialize_tts returned an incompatible engine")
            engines.append(engine)
        return MultiVoiceTTSEngine(engines)
    except Exception:
        for engine in engines:
            engine.close()
        raise


def _resolve_nghitts_espeak_data(model_dir: Path) -> Path:
    """Locate a complete architecture-independent eSpeak NG data directory."""
    required_paths = (
        Path("phondata"),
        Path("phontab"),
        Path("vi_dict"),
        Path("lang/aav/vi"),
    )

    if configured := os.environ.get("NGHITTS_ESPEAK_DATA_DIR", "").strip():
        configured_path = Path(configured).expanduser()
        if not all((configured_path / name).is_file() for name in required_paths):
            raise FileNotFoundError(
                "NGHITTS_ESPEAK_DATA_DIR does not contain complete eSpeak NG data: "
                f"{configured_path!s}"
            )
        return configured_path.resolve()

    candidates = [
        model_dir / "espeak-ng-data",
        Path("/usr/share/espeak-ng-data"),
        Path("/usr/lib/espeak-ng-data"),
        *sorted(Path("/usr/lib").glob("*-linux-gnu/espeak-ng-data")),
    ]
    for candidate in candidates:
        if all((candidate / name).is_file() for name in required_paths):
            return candidate.resolve()

    raise FileNotFoundError(
        "NghiTTS eSpeak NG data was not found; install espeak-ng-data or set "
        "NGHITTS_ESPEAK_DATA_DIR"
    )
