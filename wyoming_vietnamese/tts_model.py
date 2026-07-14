"""Checksum-pinned NghiTTS voices supported by the service.

Each voice is part of the service definition rather than a deployment setting, exactly
like the Sherpa-ONNX speech-to-text model. Pinning every file's SHA-256 digest means an
upstream replacement under the same name stops startup instead of
silently changing voice quality or executing unreviewed model content.
"""

from __future__ import annotations

from dataclasses import dataclass

from .const import TTS_CONFIG_FILE, TTS_MODEL_FILE

_COMMON_CONFIG_SHA256 = "971f57f8d504223fee5b40d664f503cf769baf7db21f7d2ae0554a75d07de2f8"
_SOUTH_CONFIG_SHA256 = "77c49591a8786c1842e06c328360cb5ccd6aa8b57322b2373e19c69f81452a0a"

DEFAULT_TTS_VOICE_ID = "ngoc-huyen-moi"


@dataclass(frozen=True, slots=True)
class TtsArtifact:
    """Immutable identity and integrity metadata for one pinned voice file."""

    remote_name: str
    local_name: str
    sha256: str


@dataclass(frozen=True, slots=True)
class TtsVoiceSpec:
    """Immutable identity of one NghiTTS voice and every file it needs."""

    id: str
    name: str
    model: TtsArtifact
    config: TtsArtifact

    @property
    def artifacts(self) -> tuple[TtsArtifact, ...]:
        """Return every pinned file for this voice, the inference graph first."""
        return (self.model, self.config)


def _voice(
    voice_id: str,
    name: str,
    model_sha256: str,
    *,
    config_sha256: str = _COMMON_CONFIG_SHA256,
) -> TtsVoiceSpec:
    """Build one voice from the API file names NghiTTS derives from its display name."""
    return TtsVoiceSpec(
        voice_id,
        name,
        TtsArtifact(f"{name}.onnx", TTS_MODEL_FILE, model_sha256),
        TtsArtifact(f"{name}.onnx.json", TTS_CONFIG_FILE, config_sha256),
    )


def _south_voice(voice_id: str, name: str, model_sha256: str) -> TtsVoiceSpec:
    """Build a voice using NghiTTS's Southern Vietnamese configuration."""
    return _voice(
        voice_id,
        name,
        model_sha256,
        config_sha256=_SOUTH_CONFIG_SHA256,
    )


# Verified against the NghiTTS model API on 2026-08-11.
TTS_VOICES = (
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
)

TTS_VOICES_BY_ID = {voice.id: voice for voice in TTS_VOICES}


def get_voice(voice_id: str) -> TtsVoiceSpec:
    """Resolve one supported voice ID or report the available choices."""
    try:
        return TTS_VOICES_BY_ID[voice_id]
    except KeyError as err:
        choices = ", ".join(TTS_VOICES_BY_ID)
        raise ValueError(f"TTS_VOICE must be one of: {choices}") from err


DEFAULT_TTS_VOICE = get_voice(DEFAULT_TTS_VOICE_ID)
