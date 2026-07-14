"""Sherpa-ONNX speech-to-text service and Wyoming event handler."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import Executor
from pathlib import Path
from time import perf_counter
from typing import Protocol

import numpy as np
import wyoming.pyaudioop as wyoming_audioop
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.error import Error as WyomingError
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info

from .config import resolve_cpu_threads
from .const import (
    DEFAULT_EVENT_TIMEOUT,
    DEFAULT_WRITE_TIMEOUT,
    MAX_INPUT_SAMPLE_RATE,
    MIN_INPUT_SAMPLE_RATE,
    PROGRAM_NAME,
    STT_CONVERSION_SLICE_SECONDS,
    STT_INLINE_CONVERSION_SECONDS,
    STT_SAMPLE_CHANNELS,
    STT_SAMPLE_WIDTH,
    SUPPORTED_INPUT_CHANNELS,
    SUPPORTED_INPUT_WIDTHS,
    TARGET_SAMPLE_RATE,
    VIETNAMESE_LANGUAGE,
)
from .inference import run_inference
from .protocol import ByteBudget, SafeAsyncEventHandler, is_vietnamese_language

_LOGGER = logging.getLogger(__name__)


class _RecognitionResult(Protocol):
    """Result shape exposed by Sherpa-ONNX offline streams."""

    text: str


class _OfflineStream(Protocol):
    """Subset of the Sherpa-ONNX stream API used by the server."""

    result: _RecognitionResult

    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> None:
        """Accept normalized waveform samples."""


class STTRecognizer(Protocol):
    """Subset of the Sherpa-ONNX recognizer API used by request handlers."""

    def create_stream(self) -> _OfflineStream:
        """Create an isolated recognition stream."""
        ...

    def decode_stream(self, s: _OfflineStream) -> None:
        """Decode an accepted waveform."""


class SherpaSTTEventHandler(SafeAsyncEventHandler):
    """Handle bounded, offline Wyoming transcription requests."""

    def __init__(
        self,
        recognizer: STTRecognizer,
        wyoming_info_event: Event,
        inference_lock: asyncio.Lock,
        model_name: str,
        max_audio_seconds: float,
        queue_timeout: float,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        buffer_budget: ByteBudget | None = None,
        event_timeout: float = DEFAULT_EVENT_TIMEOUT,
        write_timeout: float = DEFAULT_WRITE_TIMEOUT,
        inference_executor: Executor | None = None,
    ) -> None:
        """Initialize per-connection audio state around a shared recognizer."""
        super().__init__(
            reader,
            writer,
            event_timeout=event_timeout,
            write_timeout=write_timeout,
        )
        self.recognizer = recognizer
        self.inference_executor = inference_executor
        self.wyoming_info_event = wyoming_info_event
        self.inference_lock = inference_lock
        self.model_name = model_name
        self.queue_timeout = queue_timeout
        self.max_audio_seconds = max_audio_seconds
        self.max_audio_bytes = int(
            max_audio_seconds * TARGET_SAMPLE_RATE * STT_SAMPLE_WIDTH * STT_SAMPLE_CHANNELS
        )
        self.audio_bytes = bytearray()
        self.buffer_budget = buffer_budget
        self.reserved_audio_bytes = 0
        self.audio_started = False
        self.audio_started_at: float | None = None
        self.audio_chunk_count = 0
        self.input_audio_bytes = 0
        self.input_audio_frames = 0
        self.audio_conversion_seconds = 0.0
        self.source_format: tuple[int, int, int] | None = None
        self.converter: AudioChunkConverter | None = None

    async def handle_event(self, event: Event) -> bool:
        """Handle service discovery and one transcription lifecycle."""
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent STT service information")
            return True

        if Transcribe.is_type(event.type):
            return await self._handle_transcribe(event)

        if AudioStart.is_type(event.type):
            return await self._handle_audio_start(event)

        if AudioChunk.is_type(event.type):
            return await self._handle_audio_chunk(event)

        if AudioStop.is_type(event.type):
            return await self._handle_audio_stop()

        return True

    async def _handle_transcribe(self, event: Event) -> bool:
        """Validate requested model and language before audio arrives."""
        if self.audio_started:
            return await self._fail("Audio is already streaming", "invalid-event-order")
        try:
            request = Transcribe.from_event(event)
        except KeyError, TypeError, ValueError:
            return await self._fail("Malformed transcription request", "invalid-request")

        if request.name is not None and not isinstance(request.name, str):
            return await self._fail("STT model name must be text", "invalid-request")
        if request.name and request.name != self.model_name:
            return await self._fail("Requested STT model is not available", "unsupported-model")
        if request.language is not None and not isinstance(request.language, str):
            return await self._fail("STT language must be text", "invalid-request")
        if request.language is not None and not is_vietnamese_language(request.language):
            return await self._fail("Only Vietnamese STT is supported", "unsupported-language")

        _LOGGER.debug("Accepted transcription request")
        return True

    async def _handle_audio_start(self, event: Event) -> bool:
        """Initialize streaming conversion after validating the source format."""
        if self.audio_started:
            return await self._fail("Audio stream already started", "invalid-event-order")
        try:
            audio_start = AudioStart.from_event(event)
            self._validate_audio_format(
                audio_start.rate,
                audio_start.width,
                audio_start.channels,
            )
        except (KeyError, TypeError, ValueError) as err:
            _LOGGER.warning("Rejected STT audio format: %s", err)
            return await self._fail("Unsupported or malformed audio format", "unsupported-audio")

        self._clear_audio_buffer()
        self.audio_started = True
        self.audio_started_at = perf_counter()
        self.audio_chunk_count = 0
        self.input_audio_bytes = 0
        self.input_audio_frames = 0
        self.audio_conversion_seconds = 0.0
        self.source_format = (audio_start.rate, audio_start.width, audio_start.channels)
        self.converter = AudioChunkConverter(
            rate=TARGET_SAMPLE_RATE,
            width=STT_SAMPLE_WIDTH,
            channels=STT_SAMPLE_CHANNELS,
        )
        _LOGGER.debug(
            "STT step=audio-start source_rate=%d source_width=%d source_channels=%d "
            "max_audio_seconds=%.3f",
            audio_start.rate,
            audio_start.width,
            audio_start.channels,
            self.max_audio_bytes / (TARGET_SAMPLE_RATE * STT_SAMPLE_WIDTH * STT_SAMPLE_CHANNELS),
        )
        return True

    async def _handle_audio_chunk(self, event: Event) -> bool:
        """Validate and convert one source chunk into bounded 16-kHz mono PCM."""
        if not self.audio_started or self.source_format is None or self.converter is None:
            return await self._fail(
                "Audio chunk received before audio start", "invalid-event-order"
            )

        try:
            chunk = AudioChunk.from_event(event)
        except KeyError, TypeError, ValueError:
            return await self._fail("Malformed audio chunk", "invalid-audio")

        if (chunk.rate, chunk.width, chunk.channels) != self.source_format:
            return await self._fail("Audio format changed during the stream", "invalid-audio")
        self.audio_chunk_count += 1
        if not chunk.audio:
            return True

        frame_width = chunk.width * chunk.channels
        if len(chunk.audio) % frame_width:
            return await self._fail("Audio chunk contains a partial PCM frame", "invalid-audio")

        source_frames = len(chunk.audio) // frame_width
        max_source_frames = int(self.max_audio_seconds * chunk.rate)
        if self.input_audio_frames + source_frames > max_source_frames:
            return await self._fail(
                "Audio exceeds the configured duration limit",
                "audio-too-long",
            )
        self.input_audio_frames += source_frames
        self.input_audio_bytes += len(chunk.audio)

        conversion_started = perf_counter()
        try:
            if self.source_format == (
                TARGET_SAMPLE_RATE,
                STT_SAMPLE_WIDTH,
                STT_SAMPLE_CHANNELS,
            ):
                append_error = self._append_audio(chunk.audio)
                if append_error is not None:
                    text, code = append_error
                    return await self._fail(text, code)
            elif source_frames / chunk.rate <= STT_INLINE_CONVERSION_SECONDS:
                converted = self._convert_chunk(chunk)
                append_error = self._append_audio(converted.audio)
                if append_error is not None:
                    text, code = append_error
                    return await self._fail(text, code)
            else:
                frames_per_slice = max(
                    1,
                    int(chunk.rate * STT_CONVERSION_SLICE_SECONDS),
                )
                bytes_per_slice = frames_per_slice * frame_width
                converted_parts = await asyncio.to_thread(
                    self._convert_chunk_slices,
                    chunk,
                    bytes_per_slice,
                )
                for converted_audio in converted_parts:
                    append_error = self._append_audio(converted_audio)
                    if append_error is not None:
                        text, code = append_error
                        return await self._fail(text, code)
        except (TypeError, ValueError) as err:
            _LOGGER.warning("Failed to convert STT audio chunk: %s", err)
            return await self._fail("Audio conversion failed", "invalid-audio")
        conversion_seconds = perf_counter() - conversion_started
        self.audio_conversion_seconds += conversion_seconds
        return True

    def _append_audio(self, converted_audio: bytes) -> tuple[str, str] | None:
        """Append normalized PCM after enforcing local and process-wide bounds."""
        if len(self.audio_bytes) + len(converted_audio) > self.max_audio_bytes:
            return "Audio exceeds the configured duration limit", "audio-too-long"
        if self.buffer_budget is not None and not self.buffer_budget.try_reserve(
            len(converted_audio)
        ):
            return "STT buffer capacity is busy", "server-busy"
        self.reserved_audio_bytes += len(converted_audio)
        self.audio_bytes.extend(converted_audio)
        return None

    def _convert_chunk(self, chunk: AudioChunk) -> AudioChunk:
        """Downmix safely and run the stateful Wyoming converter on one chunk."""
        converter = self.converter
        if converter is None:
            raise RuntimeError("audio converter is not initialized")
        if chunk.channels == 2:
            chunk = AudioChunk(
                rate=chunk.rate,
                width=chunk.width,
                channels=1,
                audio=bytes(wyoming_audioop.tomono(chunk.audio, chunk.width, 0.5, 0.5)),
                timestamp=chunk.timestamp,
            )
        return converter.convert(chunk)

    def _convert_chunk_slices(
        self,
        chunk: AudioChunk,
        bytes_per_slice: int,
    ) -> list[bytes]:
        """Convert all slices of a large event in one worker dispatch."""
        converted_parts: list[bytes] = []
        for offset in range(0, len(chunk.audio), bytes_per_slice):
            source_chunk = AudioChunk(
                rate=chunk.rate,
                width=chunk.width,
                channels=chunk.channels,
                audio=chunk.audio[offset : offset + bytes_per_slice],
                timestamp=chunk.timestamp,
            )
            converted_parts.append(self._convert_chunk(source_chunk).audio)
        return converted_parts

    def _clear_audio_buffer(self) -> None:
        """Release this handler's process-wide reservation and local PCM buffer."""
        if self.buffer_budget is not None and self.reserved_audio_bytes:
            self.buffer_budget.release(self.reserved_audio_bytes)
        self.reserved_audio_bytes = 0
        # Rebind instead of clear(): a NumPy view over this buffer may still be alive,
        # and resizing an exported bytearray raises BufferError.
        self.audio_bytes = bytearray()

    async def _handle_audio_stop(self) -> bool:
        """Finish the audio lifecycle and run recognition outside the event loop."""
        if not self.audio_started:
            return await self._fail("Audio stop received before audio start", "invalid-event-order")

        stop_started = perf_counter()
        request_started = self.audio_started_at or stop_started
        receive_seconds = stop_started - request_started
        self.audio_started = False
        self.source_format = None
        self.converter = None
        audio_size = len(self.audio_bytes)
        audio_seconds = audio_size / (TARGET_SAMPLE_RATE * STT_SAMPLE_WIDTH * STT_SAMPLE_CHANNELS)
        _LOGGER.debug(
            "STT step=audio-stop chunks=%d input_bytes=%d normalized_bytes=%d "
            "audio_seconds=%.3f receive_ms=%.3f conversion_ms=%.3f",
            self.audio_chunk_count,
            self.input_audio_bytes,
            audio_size,
            audio_seconds,
            receive_seconds * 1000,
            self.audio_conversion_seconds * 1000,
        )
        if not audio_size:
            response_started = perf_counter()
            await self.write_event(Transcript(text="", language=VIETNAMESE_LANGUAGE).event())
            completed_at = perf_counter()
            _LOGGER.info(
                "Completed STT request: chunks=%d input_bytes=%d audio_bytes=0 "
                "audio_seconds=0.000 receive_ms=%.3f handle_ms=%.3f response_ms=%.3f "
                "total_ms=%.3f real_time_factor=0.000",
                self.audio_chunk_count,
                self.input_audio_bytes,
                receive_seconds * 1000,
                (completed_at - stop_started) * 1000,
                (completed_at - response_started) * 1000,
                (completed_at - request_started) * 1000,
            )
            self.audio_started_at = None
            return False

        preprocess_started = perf_counter()
        audio_float = np.multiply(
            np.frombuffer(self.audio_bytes, dtype="<i2"),
            np.float32(1.0 / 32768.0),
            dtype=np.float32,
        )
        self._clear_audio_buffer()
        preprocess_seconds = perf_counter() - preprocess_started
        _LOGGER.debug(
            "STT step=preprocess samples=%d duration_ms=%.3f",
            audio_float.size,
            preprocess_seconds * 1000,
        )

        queue_started = perf_counter()
        _LOGGER.debug("STT step=queue-wait timeout_seconds=%.3f", self.queue_timeout)
        try:
            await asyncio.wait_for(self.inference_lock.acquire(), timeout=self.queue_timeout)
        except TimeoutError:
            _LOGGER.debug(
                "STT step=queue-timeout wait_ms=%.3f",
                (perf_counter() - queue_started) * 1000,
            )
            return await self._fail("STT inference queue is busy", "server-busy")
        queue_seconds = perf_counter() - queue_started
        _LOGGER.debug("STT step=queue-acquired wait_ms=%.3f", queue_seconds * 1000)

        inference_seconds = 0.0
        try:
            if self.writer.is_closing():
                _LOGGER.debug("STT step=inference-skipped reason=client-closed")
                self.audio_started_at = None
                return False
            inference_started = perf_counter()
            try:
                text = await run_inference(
                    self.inference_executor,
                    self._recognize,
                    audio_float,
                )
            finally:
                inference_seconds = perf_counter() - inference_started
        except Exception:
            _LOGGER.exception(
                "STT inference failed after %.3f ms",
                inference_seconds * 1000,
            )
            return await self._fail("Speech recognition failed", "inference-failed")
        finally:
            self.inference_lock.release()

        if self.writer.is_closing():
            self.audio_started_at = None
            return False
        real_time_factor = inference_seconds / audio_seconds
        _LOGGER.debug(
            "STT step=inference-complete inference_ms=%.3f real_time_factor=%.3f text_chars=%d",
            inference_seconds * 1000,
            real_time_factor,
            len(text),
        )

        response_started = perf_counter()
        await self.write_event(Transcript(text=text, language=VIETNAMESE_LANGUAGE).event())
        completed_at = perf_counter()
        _LOGGER.info(
            "Completed STT request: chunks=%d input_bytes=%d audio_bytes=%d "
            "audio_seconds=%.3f text_chars=%d conversion_ms=%.3f preprocess_ms=%.3f "
            "queue_ms=%.3f inference_ms=%.3f response_ms=%.3f receive_ms=%.3f "
            "handle_ms=%.3f total_ms=%.3f real_time_factor=%.3f",
            self.audio_chunk_count,
            self.input_audio_bytes,
            audio_size,
            audio_seconds,
            len(text),
            self.audio_conversion_seconds * 1000,
            preprocess_seconds * 1000,
            queue_seconds * 1000,
            inference_seconds * 1000,
            (completed_at - response_started) * 1000,
            receive_seconds * 1000,
            (completed_at - stop_started) * 1000,
            (completed_at - request_started) * 1000,
            real_time_factor,
        )
        self.audio_started_at = None
        return False

    def _recognize(self, audio_float: np.ndarray) -> str:
        """Execute one recognition stream and normalize its transcript."""
        stream = self.recognizer.create_stream()
        stream.accept_waveform(TARGET_SAMPLE_RATE, audio_float)
        self.recognizer.decode_stream(stream)
        return stream.result.text.strip().lower()

    @staticmethod
    def _validate_audio_format(rate: int, width: int, channels: int) -> None:
        """Validate supported PCM format metadata without coercion."""
        if type(rate) is not int or not MIN_INPUT_SAMPLE_RATE <= rate <= MAX_INPUT_SAMPLE_RATE:
            raise ValueError(
                f"sample rate must be {MIN_INPUT_SAMPLE_RATE}-{MAX_INPUT_SAMPLE_RATE} Hz"
            )
        if type(width) is not int or width not in SUPPORTED_INPUT_WIDTHS:
            raise ValueError("sample width must be 1, 2, 3, or 4 bytes")
        if type(channels) is not int or channels not in SUPPORTED_INPUT_CHANNELS:
            raise ValueError("audio must be mono or stereo")

    async def _fail(self, text: str, code: str) -> bool:
        """Return a Wyoming error and terminal empty transcript."""
        now = perf_counter()
        elapsed_seconds = now - self.audio_started_at if self.audio_started_at is not None else 0.0
        _LOGGER.debug(
            "STT step=request-failed code=%s detail=%s chunks=%d input_bytes=%d "
            "normalized_bytes=%d elapsed_ms=%.3f",
            code,
            text,
            self.audio_chunk_count,
            self.input_audio_bytes,
            len(self.audio_bytes),
            elapsed_seconds * 1000,
        )
        self.audio_started = False
        self.audio_started_at = None
        self.audio_chunk_count = 0
        self.input_audio_bytes = 0
        self.input_audio_frames = 0
        self.audio_conversion_seconds = 0.0
        self.source_format = None
        self.converter = None
        self._clear_audio_buffer()
        await self.write_event(WyomingError(text=text, code=code).event())
        await self.write_event(Transcript(text="", language=VIETNAMESE_LANGUAGE).event())
        return False

    async def disconnect(self) -> None:
        """Release buffered audio immediately when a client disconnects."""
        if self.audio_started_at is not None:
            _LOGGER.debug(
                "STT step=disconnect chunks=%d normalized_bytes=%d elapsed_ms=%.3f",
                self.audio_chunk_count,
                len(self.audio_bytes),
                (perf_counter() - self.audio_started_at) * 1000,
            )
        self._clear_audio_buffer()
        self.audio_started_at = None
        self.converter = None


def get_stt_info(model_name: str, model_version: str | None = None) -> Info:
    """Build Wyoming discovery information for the STT service."""
    return Info(
        asr=[
            AsrProgram(
                name=PROGRAM_NAME,
                description="Sherpa-ONNX Offline Transducer STT",
                attribution=Attribution(name="k2-fsa", url="https://github.com/k2-fsa/sherpa-onnx"),
                installed=True,
                version="1.13",
                models=[
                    AsrModel(
                        name=model_name,
                        description="Zipformer RNNT STT model",
                        attribution=Attribution(
                            name=model_name.split("/", maxsplit=1)[0],
                            url=f"https://huggingface.co/{model_name}",
                        ),
                        installed=True,
                        version=model_version,
                        languages=[VIETNAMESE_LANGUAGE],
                    )
                ],
                supports_transcript_streaming=False,
                requires_external_vad=True,
            )
        ]
    )


def warm_up_stt(recognizer: STTRecognizer) -> None:
    """Exercise the recognizer once so the first live utterance avoids cold start."""
    warmup_started = perf_counter()
    stream = recognizer.create_stream()
    stream.accept_waveform(
        TARGET_SAMPLE_RATE,
        np.zeros(TARGET_SAMPLE_RATE // 10, dtype=np.float32),
    )
    recognizer.decode_stream(stream)
    _LOGGER.info(
        "Warmed up Sherpa-ONNX inference: duration_ms=%.3f",
        (perf_counter() - warmup_started) * 1000,
    )


def initialize_stt(
    model_dir: Path,
    num_threads: int = 0,
) -> STTRecognizer:
    """Initialize one shared Sherpa-ONNX recognizer from required local files."""
    encoder = model_dir / "encoder.onnx"
    decoder = model_dir / "decoder.onnx"
    joiner = model_dir / "joiner.onnx"
    tokens = model_dir / "tokens.txt"

    for file_path in (encoder, decoder, joiner, tokens):
        if not file_path.is_file():
            raise FileNotFoundError(f"Required Sherpa-ONNX model file missing: {file_path!s}")

    import sherpa_onnx

    threads = resolve_cpu_threads(num_threads)
    _LOGGER.info(
        "Initializing Sherpa-ONNX: threads=%d, model_dir=%s",
        threads,
        model_dir,
    )
    initialization_started = perf_counter()
    recognizer: STTRecognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(encoder),
        decoder=str(decoder),
        joiner=str(joiner),
        tokens=str(tokens),
        num_threads=threads,
        sample_rate=TARGET_SAMPLE_RATE,
        feature_dim=80,
        decoding_method="greedy_search",
        provider="cpu",
    )
    _LOGGER.info(
        "Initialized Sherpa-ONNX: duration_ms=%.3f",
        (perf_counter() - initialization_started) * 1000,
    )
    return recognizer
