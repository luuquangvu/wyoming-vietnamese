"""Constants shared by the Wyoming Vietnamese services."""

from enum import Enum, IntEnum, StrEnum

DEFAULT_PORT = 10300
PROGRAM_NAME = "wyoming_vietnamese"
NGHITTS_MODEL_BASE_URL = "https://nghitts.app/api/model"
HUGGINGFACE_BASE_URL = "https://huggingface.co"
HUGGINGFACE_API_MODELS_URL = f"{HUGGINGFACE_BASE_URL}/api/models"

SHERPA_ONNX_REPO_URL = "https://github.com/k2-fsa/sherpa-onnx"
NGHITTS_REPO_URL = "https://github.com/nghimestudio/nghitts"
ZEROTTS_REPO_URL = "https://github.com/zeroweight-ai/ZeroTTS"
ZEROTTS_SOURCE_COMMIT = "319c3e07c8575c1e38617f07c773ec76a780a1f1"

DEFAULT_CACHE_DIR = "/app/.cache"
DEFAULT_DOWNLOAD_DIR = "/app/models"
VIETNAMESE_LANGUAGE = "vi"


class TtsEngine(StrEnum):
    """Supported text-to-speech synthesis engines."""

    NGHITTS = "nghitts"
    ZEROTTS = "zerotts"


DEFAULT_TTS_ENGINE = TtsEngine.NGHITTS


class TtsProvider(StrEnum):
    """Provider attribution labels for text-to-speech engines."""

    NGHITTS = "NghiTTS"
    ZEROTTS = "ZeroTTS"


class NghiTtsFile(StrEnum):
    """File names for NghiTTS ONNX model assets."""

    MODEL = "model.onnx"
    CONFIG = "model.onnx.json"
    TOKENS = "tokens.txt"


class ZeroTtsFile(StrEnum):
    """File names for ZeroTTS runtime and model assets."""

    CONFIG = "config.json"
    TOKENIZER = "tokenizer.json"
    NULL_VOICE_EMB = "null_voice_emb.npy"
    DEFAULT_GGUF_MODEL = "zerotts-q8_0.gguf"
    LIBZEROTTS_SO = "libzerotts.so"


class ZeroTtsDirectory(StrEnum):
    """Directory names for ZeroTTS runtime and model assets."""

    GGUF = "gguf"


class ZeroTtsVariant(StrEnum):
    """Supported CPU architecture optimization variants for ZeroTTS GGML runtime."""

    COMPAT = "compat"
    SSE4 = "sse4"
    AVX = "avx"
    AVX2 = "avx2"
    AVX512 = "avx512"
    ARM_DOTPROD = "arm_dotprod"
    NATIVE = "native"


ZEROTTS_REPO_ID = "zeroweight-ai/ZeroTTS-GGUF"


class EventLimit(IntEnum):
    """Byte bounds for Wyoming protocol frames."""

    MAX_DATA_BYTES = 64 * 1024
    MAX_HEADER_BYTES = 64 * 1024
    MAX_PAYLOAD_BYTES = 4 * 1024 * 1024


class Timeout(float, Enum):
    """Default operational timeouts in seconds."""

    INFERENCE_QUEUE = 30.0
    EVENT = 60.0
    WRITE = 5.0
    SHUTDOWN_DRAIN = 10.0


class ConnectionLimit(IntEnum):
    """Connection limits and reserve bounds."""

    MAX_ACTIVE = 64
    LOCAL_RESERVE = 2


DEFAULT_MAX_STT_BUFFER_MB = 256
DEFAULT_MAX_STT_AUDIO_SECONDS = 120.0
DEFAULT_MAX_TTS_TEXT_CHARS = 2_000


class TtsSilenceMs(IntEnum):
    """Synthesized speech boundary silence durations in milliseconds."""

    PARAGRAPH = 600
    SENTENCE = 400
    CLAUSE = 200
    MAX = 3_000


DEFAULT_TTS_CACHE_IDLE_SECONDS = 86_400.0


class TtsCacheLimit(IntEnum):
    """Default capacity bounds for the TTS audio cache."""

    MAX_ENTRIES = 128
    MAX_MB = 64
    MAX_ITEM_MB = 4


# Inference is serialized by the STT and TTS locks, so a small pool is enough: one STT
# recognition, one TTS chunk, and their iterator cleanup never exceed it.
INFERENCE_EXECUTOR_WORKERS = 4


class SttAudio(IntEnum):
    """Audio stream specifications for speech-to-text."""

    TARGET_SAMPLE_RATE = 16000
    SAMPLE_WIDTH = 2
    SAMPLE_CHANNELS = 1


class InputSampleRate(IntEnum):
    """Supported input audio sample rate bounds in Hz."""

    MIN = 8_000
    MAX = 192_000


SUPPORTED_INPUT_WIDTHS = frozenset({1, 2, 3, 4})
SUPPORTED_INPUT_CHANNELS = frozenset({1, 2})


class SttConversion(float, Enum):
    """Audio conversion timing thresholds in seconds."""

    INLINE_SECONDS = 0.1
    SLICE_SECONDS = 0.25


class TtsAudio(IntEnum):
    """Audio stream specifications for text-to-speech."""

    SAMPLE_WIDTH = 2
    SAMPLE_CHANNELS = 1
    CHUNK_SIZE = 4096
    # Bounds the audio a producer may run ahead of a draining client. Inference releases the
    # shared TTS lock only once the engine iterator is exhausted, so a capacity below one
    # utterance couples the lock to client read speed. 128 chunks is 512 KiB, about 11.9
    # seconds at 22.05 kHz, and the TTS lock keeps at most one producer filling a queue.
    OUTPUT_QUEUE_CHUNKS = 128


class NghiTtsAudio(IntEnum):
    """Audio stream specifications for NghiTTS."""

    SAMPLE_RATE = 22050


class ZeroTtsAudio(IntEnum):
    """Audio stream specifications for ZeroTTS."""

    SAMPLE_RATE = 48000
