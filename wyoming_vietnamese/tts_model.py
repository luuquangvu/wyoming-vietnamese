"""Checksum-pinned NghiTTS voices supported by the service.

Each voice is part of the service definition rather than a deployment setting, exactly
like the Sherpa-ONNX speech-to-text model. Pinning every file's SHA-256 digest means an
upstream replacement under the same name stops startup instead of
silently changing voice quality or executing unreviewed model content.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, overload

from .const import (
    DEFAULT_TTS_ENGINE,
    ZEROTTS_REPO_ID,
    NghiTtsFile,
    TtsEngine,
    ZeroTtsFile,
)

_NGHITTS_COMMON_CONFIG_SHA256 = "971f57f8d504223fee5b40d664f503cf769baf7db21f7d2ae0554a75d07de2f8"
_NGHITTS_SOUTH_CONFIG_SHA256 = "77c49591a8786c1842e06c328360cb5ccd6aa8b57322b2373e19c69f81452a0a"

DEFAULT_NGHITTS_VOICE_ID = "ngoc-huyen-moi"


@dataclass(frozen=True, slots=True)
class NghiTtsArtifact:
    """Immutable identity and integrity metadata for one pinned voice file."""

    remote_name: str
    local_name: str
    sha256: str


@dataclass(frozen=True, slots=True)
class NghiTtsVoiceSpec:
    """Immutable identity of one NghiTTS voice and every file it needs."""

    id: str
    name: str
    model: NghiTtsArtifact
    config: NghiTtsArtifact

    @property
    def artifacts(self) -> tuple[NghiTtsArtifact, ...]:
        """Return every pinned file for this voice, the inference graph first."""
        return (self.model, self.config)


def _voice(
    voice_id: str,
    name: str,
    model_sha256: str,
    *,
    config_sha256: str = _NGHITTS_COMMON_CONFIG_SHA256,
) -> NghiTtsVoiceSpec:
    """Build one voice from the API file names NghiTTS derives from its display name."""
    return NghiTtsVoiceSpec(
        voice_id,
        name,
        NghiTtsArtifact(f"{name}.onnx", NghiTtsFile.MODEL, model_sha256),
        NghiTtsArtifact(f"{name}.onnx.json", NghiTtsFile.CONFIG, config_sha256),
    )


def _south_voice(voice_id: str, name: str, model_sha256: str) -> NghiTtsVoiceSpec:
    """Build a voice using NghiTTS's Southern Vietnamese configuration."""
    return _voice(
        voice_id,
        name,
        model_sha256,
        config_sha256=_NGHITTS_SOUTH_CONFIG_SHA256,
    )


NGHITTS_VOICES = (
    _voice(
        "ban-mai", "Ban Mai", "9f599998f244511edd3df757febc0653ee94bfdbe673740f25c98c9b9b374984"
    ),
    _south_voice(
        "chieu-thanh",
        "Chiếu Thành",
        "5b3059395e1d2f3d7499deaf3463615b5281c607465ea274d82ce84d93d61ac3",
    ),
    _voice(
        "duy-onyx-moi",
        "Duy Onyx (mới)",
        "b4ebfc949c3330cf3d44520133cb1ba3460bb4218ff5286e28ba10fcfa898ac8",
    ),
    _voice(
        "duy-oryx",
        "Duy Oryx",
        "0d6f6b95eee7256f18115c09ad1cbdb69acddf87761549ec1a7decd19a4eba5c",
    ),
    _voice(
        "lac-phi",
        "Lạc Phi",
        "c6e496c69c05f6043efd9967f0fb98d34b7cf94b0e698de43169253d34b318b2",
    ),
    _voice(
        "mai-phuong",
        "Mai Phương",
        "3d8384d05a0569be1a0a4da12d7582e7d0500b1f55139584e70f310dac92b01b",
    ),
    _voice(
        "minh-khang",
        "Minh Khang",
        "3c95962901b3c3dd03d6de636625f6f3608877d69d7fb5cb20f9c26b202f3453",
    ),
    _voice(
        "minh-quang",
        "Minh Quang",
        "0a559df4c4eaab442e0d2007f32888f87567900dfe1a4e9f1632f64070090e17",
    ),
    _voice(
        "manh-dung",
        "Mạnh Dũng",
        "b5e4d3b55e02847082a07ddca34bccaca201bc83a4441b974accac699645bfe0",
    ),
    _voice(
        "my-tam",
        "Mỹ Tâm",
        "9a0c576d3366aa336ab56d782e2a0db174a26f6a1629cf418d2cf059d45ff6c1",
    ),
    _south_voice(
        "my-tam-real",
        "Mỹ Tâm Real",
        "140f4b6f3c487338613f7c1b2842950110178404f7e9b5abed7640acd8956267",
    ),
    _voice(
        "ngoc-huyen-moi",
        "Ngọc Huyền (mới)",
        "88bc21477831dd99759cff164f0ab270e258faf435d01741f0211ff8620255e2",
    ),
    _voice(
        "ngoc-ngan",
        "Ngọc Ngạn",
        "c03a71201f51336fe937f12e54781520cb3bf629fa6adb3943314e29a58a9133",
    ),
    _voice(
        "phuong-trang",
        "Phương Trang",
        "78e7f514503652a666e3e4b4d0d8fdc42f03d81a775a521b3119d1d7e6fc303f",
    ),
    _voice(
        "thanh-phuong-viettel",
        "Thanh Phương Viettel",
        "f7d806e82e785729caf950893c2caa19a94ca79acfa1cd0deb2fc654cd61df68",
    ),
    _south_voice(
        "thien-tam",
        "Thiện Tâm",
        "2ac4616193b00c68d10997670ad617c6a3798a05262638b4811b2448e77c85dc",
    ),
    _voice(
        "tran-thanh",
        "Trấn Thành",
        "c3235ad51d99f5ce7d147ceb28db2ad2fe3500131d8ec77e804e6451ec293759",
    ),
    _voice(
        "tai-an",
        "Tài An",
        "74e911f0c6d32e3e14811271f5da896befadbdea418272b72c7ad228c6322498",
    ),
    _voice(
        "viet-thao",
        "Việt Thảo",
        "f8cad266cfed6018390752326373379411b85efcfca21441e41e31a5aa4a6daf",
    ),
    _voice(
        "adam",
        "adam",
        "90e73d171447825fa8442fea8bf39c54bcfb206958f05170361e0fa3ba5c48eb",
    ),
)

NGHITTS_VOICES_BY_ID = {voice.id: voice for voice in NGHITTS_VOICES}
DEFAULT_ZEROTTS_VOICE_ID = "maichi"


@dataclass(frozen=True, slots=True)
class ZeroTtsArtifact:
    """Immutable identity and integrity metadata for one pinned ZeroTTS repository file."""

    remote_name: str
    local_name: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ZeroTtsModelSpec:
    """Immutable identity of the ZeroTTS speech synthesis model and core assets."""

    repo: str
    revision: str
    config: ZeroTtsArtifact
    tokenizer: ZeroTtsArtifact
    null_voice_emb: ZeroTtsArtifact
    gguf_model: ZeroTtsArtifact
    codec_files: tuple[ZeroTtsArtifact, ...]

    @property
    def core_artifacts(self) -> tuple[ZeroTtsArtifact, ...]:
        """Return the core model assets."""
        return (self.config, self.tokenizer, self.null_voice_emb, self.gguf_model)

    @property
    def artifacts(self) -> tuple[ZeroTtsArtifact, ...]:
        """Return all pinned model and runtime files."""
        return (*self.core_artifacts, *self.codec_files)

    @property
    def allow_patterns(self) -> tuple[str, ...]:
        """Return the repository file patterns the service downloads."""
        return (
            self.config.remote_name,
            self.tokenizer.remote_name,
            self.null_voice_emb.remote_name,
            self.gguf_model.remote_name,
            "onnx/codec/*",
            "voices/*",
        )


# Verified against the repository at the pinned commit on 2026-09-17.
ZEROTTS_MODEL = ZeroTtsModelSpec(
    repo=ZEROTTS_REPO_ID,
    revision="92ca8651645d4733df56620b3aadc768a76f7c46",
    config=ZeroTtsArtifact(
        ZeroTtsFile.CONFIG,
        ZeroTtsFile.CONFIG,
        "3676da6a9f4dba7a8f2d106d5c1fb02c491391dc19c91dae893133a7e219d666",
    ),
    tokenizer=ZeroTtsArtifact(
        ZeroTtsFile.TOKENIZER,
        ZeroTtsFile.TOKENIZER,
        "4fd646e8a1fd6694cb9c876c914a516ef8dd84b4f84db2605be1a7388885ae91",
    ),
    null_voice_emb=ZeroTtsArtifact(
        ZeroTtsFile.NULL_VOICE_EMB,
        ZeroTtsFile.NULL_VOICE_EMB,
        "ec014c14f79e8fc16f4ca6ea557f33fa2cbd7d448b5de5f467aace9ca0321e9f",
    ),
    gguf_model=ZeroTtsArtifact(
        f"gguf/{ZeroTtsFile.DEFAULT_GGUF_MODEL}",
        f"gguf/{ZeroTtsFile.DEFAULT_GGUF_MODEL}",
        "2841287bed42440a0d0b373639acc0206b07e2895be5e48f3eaab46570f1820c",
    ),
    codec_files=(
        ZeroTtsArtifact(
            "onnx/codec/codec_browser_onnx_meta.json",
            "onnx/codec/codec_browser_onnx_meta.json",
            "32009d6ac1cd2663bbbf5c06d6835b3a862c02f7de121663071938f2edac8e92",
        ),
        ZeroTtsArtifact(
            "onnx/codec/moss_audio_tokenizer_decode_full.onnx",
            "onnx/codec/moss_audio_tokenizer_decode_full.onnx",
            "0fbbafe3fd4afa2a019af5c5ced204af6e2d1db044fa40f021525d2aee95b4ac",
        ),
        ZeroTtsArtifact(
            "onnx/codec/moss_audio_tokenizer_decode_shared.data",
            "onnx/codec/moss_audio_tokenizer_decode_shared.data",
            "e69d52e0f4e84ca27850557ee54face46632d3a5a16c89bd246c7c408466dcad",
        ),
        ZeroTtsArtifact(
            "onnx/codec/moss_audio_tokenizer_decode_step.onnx",
            "onnx/codec/moss_audio_tokenizer_decode_step.onnx",
            "9527c86a29e1837edec1f74db57d5eeaadb3a715af3382703566460afed25855",
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class ZeroTtsVoiceSpec:
    """Immutable identity and metadata of one ZeroTTS voice."""

    id: str
    name: str
    gender: str
    description: str
    artifact: ZeroTtsArtifact

    @property
    def artifacts(self) -> tuple[ZeroTtsArtifact, ...]:
        """Return every pinned file for this voice."""
        return (self.artifact,)


def _zerotts_voice(
    voice_id: str,
    name: str,
    gender: str,
    description: str,
    sha256: str,
) -> ZeroTtsVoiceSpec:
    """Build one pinned ZeroTTS voice specification."""
    return ZeroTtsVoiceSpec(
        voice_id,
        name,
        gender,
        description,
        ZeroTtsArtifact(f"voices/{voice_id}/voice.npz", "voice.npz", sha256),
    )


ZEROTTS_VOICES = (
    _zerotts_voice(
        "maichi",
        "Mai Chi",
        "nữ",
        "Nữ miền Bắc (nhẹ nhàng, thân thiện, tự nhiên)",
        "34bbc62df623764c5cf50b840bb27eb590b2385d7a72b9728494636ef677c6e6",
    ),
    _zerotts_voice(
        "baotrang",
        "Bảo Trang",
        "nữ",
        "Nữ miền Bắc (trưởng thành, tin tức, rõ ràng, trung tính)",
        "abbc5807cc4767e7c62cc59f7b7cd04b4b7e0dba176086447a0c114578cf695a",
    ),
    _zerotts_voice(
        "giahuy",
        "Gia Huy",
        "nam",
        "Nam miền Bắc (trẻ, kể chuyện, trầm ấm, tâm tình)",
        "3979a8369000eaaee1130801c1cacdadddc12e8d9a4a604be7430c4bcc6df32e",
    ),
    _zerotts_voice(
        "hamy",
        "Hà My",
        "nữ",
        "Nữ miền Bắc (trẻ, hoạt hình, cao, biểu cảm)",
        "a864c127386f74976154c840875f6ad615d74c377768bc6edcb75e3278f228b7",
    ),
    _zerotts_voice(
        "huuduc",
        "Hữu Đức",
        "nam",
        "Nam miền Bắc (lớn tuổi, kể chuyện, trầm, điềm đạm)",
        "d7a6370180263093eb9dd009661ba24b5309f0ba94ec18d8ff2323c80616a7eb",
    ),
    _zerotts_voice(
        "kimoanh",
        "Kim Oanh",
        "nữ",
        "Nữ miền Bắc (trung niên, kể chuyện, ấm áp, truyền cảm)",
        "0aee413f25eae30123f930bd525d31a3b5a20c0803b4899305c2ff4a69466946",
    ),
    _zerotts_voice(
        "quangminh",
        "Quang Minh",
        "nam",
        "Nam miền Bắc (trẻ, tin tức, rõ ràng, dứt khoát)",
        "4d2acb18f831ade23ed2d02e7b749fa745c95e9ee305795c5ddeb75f93d35fdb",
    ),
    _zerotts_voice(
        "tiendat",
        "Tiến Đạt",
        "nam",
        "Nam miền Bắc (trẻ, bình luận, sôi nổi, năng lượng cao)",
        "a076fe7b938ad057652ffc4614678395edad90ab5cdc74546f3977636de15d0a",
    ),
)

ZEROTTS_VOICES_BY_ID = {voice.id: voice for voice in ZEROTTS_VOICES}


def get_nghitts_voice(voice_id: str) -> NghiTtsVoiceSpec:
    """Resolve one supported NghiTTS voice ID or report the available choices."""
    try:
        return NGHITTS_VOICES_BY_ID[voice_id]
    except KeyError as err:
        choices = ", ".join(NGHITTS_VOICES_BY_ID)
        raise ValueError(f"TTS_VOICE must be one of: {choices}") from err


def get_zerotts_voice(voice_id: str) -> ZeroTtsVoiceSpec:
    """Resolve one supported ZeroTTS voice ID or report the available choices."""
    try:
        return ZEROTTS_VOICES_BY_ID[voice_id]
    except KeyError as err:
        choices = ", ".join(ZEROTTS_VOICES_BY_ID)
        raise ValueError(f"TTS_VOICE must be one of: {choices}") from err


@overload
def get_voice(
    voice_id: str,
    engine: Literal[TtsEngine.NGHITTS] = TtsEngine.NGHITTS,
) -> NghiTtsVoiceSpec: ...


@overload
def get_voice(
    voice_id: str,
    engine: Literal[TtsEngine.ZEROTTS],
) -> ZeroTtsVoiceSpec: ...


@overload
def get_voice(voice_id: str, engine: str) -> NghiTtsVoiceSpec | ZeroTtsVoiceSpec: ...


def get_voice(
    voice_id: str,
    engine: str = DEFAULT_TTS_ENGINE,
) -> NghiTtsVoiceSpec | ZeroTtsVoiceSpec:
    """Resolve a voice ID, raising ``ValueError`` for an unsupported voice or engine."""
    if engine == TtsEngine.ZEROTTS:
        return get_zerotts_voice(voice_id)
    if engine == TtsEngine.NGHITTS:
        return get_nghitts_voice(voice_id)
    raise ValueError(f"Unknown TTS engine: {engine}")


DEFAULT_NGHITTS_VOICE = get_nghitts_voice(DEFAULT_NGHITTS_VOICE_ID)
DEFAULT_ZEROTTS_VOICE = get_zerotts_voice(DEFAULT_ZEROTTS_VOICE_ID)
