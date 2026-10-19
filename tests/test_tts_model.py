"""Checksum-pinned NghiTTS and ZeroTTS voice catalogue tests."""

from pathlib import Path

import pytest

from wyoming_vietnamese.const import (
    DEFAULT_ZEROTTS_AUTO,
    NghiTtsFile,
    TtsEngine,
    ZeroTtsCodebook,
    ZeroTtsDirectory,
    ZeroTtsFile,
    ZeroTtsQuality,
    ZeroTtsVariant,
)
from wyoming_vietnamese.tts_model import (
    DEFAULT_NGHITTS_VOICE,
    DEFAULT_NGHITTS_VOICE_ID,
    DEFAULT_ZEROTTS_VOICE,
    DEFAULT_ZEROTTS_VOICE_ID,
    NGHITTS_VOICES,
    NGHITTS_VOICES_BY_ID,
    ZEROTTS_BASE_MODEL,
    ZEROTTS_CODEC_INT8_ARTIFACT,
    ZEROTTS_CODEC_INT8_SHA256,
    ZEROTTS_EDGE_VARIANTS,
    ZEROTTS_MODEL,
    ZEROTTS_Q4_GGUF_ARTIFACT,
    ZEROTTS_Q4_MODEL,
    ZEROTTS_VOICES,
    ZEROTTS_VOICES_BY_ID,
    ZeroTtsModelSpec,
    ZeroTtsVoiceSpec,
    get_voice,
    get_zerotts_voice,
    is_zerotts_edge_variant,
    normalize_zerotts_model,
    reset_zerotts_cpu_variant,
    resolve_quality_codebooks,
    resolve_quality_zerotts_model,
    resolve_zerotts_codec_int8,
    resolve_zerotts_cpu_variant,
    resolve_zerotts_model,
    resolve_zerotts_model_spec,
)


def _assert_valid_sha256(digest: str) -> None:
    """Assert that a string is a valid lowercase 64-character SHA-256 hex digest."""
    assert len(digest) == 64
    assert digest == digest.lower()
    assert int(digest, 16) >= 0


def _assert_valid_zerotts_voice(voice: ZeroTtsVoiceSpec) -> None:
    """Assert basic validity invariants for a ZeroTTS voice specification."""
    assert voice.id in ZEROTTS_VOICES_BY_ID
    assert voice.name
    assert voice.gender in {"nữ", "nam"}
    assert voice.description
    assert voice.artifacts == (voice.artifact,)
    assert voice.artifact.remote_name == f"voices/{voice.id}/voice.npz"
    assert voice.artifact.local_name == "voice.npz"
    _assert_valid_sha256(voice.artifact.sha256)


def test_voice_catalog_has_unique_valid_entries() -> None:
    """Test every selectable voice has a unique ID and pinned artifacts."""
    assert len(NGHITTS_VOICES) == 20
    assert len(NGHITTS_VOICES_BY_ID) == len(NGHITTS_VOICES)
    assert len({voice.name for voice in NGHITTS_VOICES}) == len(NGHITTS_VOICES)
    assert all(voice.id and voice.name for voice in NGHITTS_VOICES)


def test_voice_artifacts_are_fully_pinned() -> None:
    """Test each voice pins a SHA-256 digest for every file it downloads."""
    for voice in NGHITTS_VOICES:
        assert voice.artifacts == (voice.model, voice.config)
        assert voice.model.local_name == NghiTtsFile.MODEL
        assert voice.config.local_name == NghiTtsFile.CONFIG
        assert voice.model.remote_name == f"{voice.name}.onnx"
        assert voice.config.remote_name == f"{voice.name}.onnx.json"
        for artifact in voice.artifacts:
            _assert_valid_sha256(artifact.sha256)


def test_get_voice_resolves_default_and_rejects_unknown_id() -> None:
    """Test catalog lookup is deterministic and reports supported choices."""
    assert get_voice(DEFAULT_NGHITTS_VOICE_ID) is DEFAULT_NGHITTS_VOICE
    with pytest.raises(ValueError, match="TTS_VOICE must be one of"):
        get_voice("missing")


def test_zerotts_voice_catalog() -> None:
    """Test ZeroTTS voice catalogue entries and helper lookups."""
    assert len(ZEROTTS_VOICES) == 8
    assert len(ZEROTTS_VOICES_BY_ID) == 8
    assert DEFAULT_ZEROTTS_VOICE_ID == "maichi"
    assert DEFAULT_ZEROTTS_VOICE.id == "maichi"
    assert DEFAULT_NGHITTS_VOICE_ID == "ngoc-huyen-moi"

    for voice in ZEROTTS_VOICES:
        _assert_valid_zerotts_voice(voice)

    mai_chi = get_zerotts_voice("maichi")
    assert mai_chi.name == "Mai Chi"
    assert mai_chi.gender == "nữ"

    assert get_voice("maichi", engine=TtsEngine.ZEROTTS) is mai_chi
    assert get_voice("ngoc-huyen-moi", engine=TtsEngine.NGHITTS) is DEFAULT_NGHITTS_VOICE

    with pytest.raises(ValueError, match="TTS_VOICE must be one of"):
        get_zerotts_voice("nonexistent")

    with pytest.raises(ValueError, match="TTS_VOICE must be one of"):
        get_voice("ngoc-huyen-moi", engine=TtsEngine.ZEROTTS)

    with pytest.raises(ValueError, match="TTS_VOICE must be one of"):
        get_voice("maichi", engine=TtsEngine.NGHITTS)

    with pytest.raises(ValueError, match="Unknown TTS engine"):
        get_voice("maichi", engine="invalid")


def test_zerotts_model_specification_is_fully_pinned() -> None:
    """Test the ZeroTTS model specification and all its artifacts are pinned."""
    assert ZEROTTS_MODEL.repo == "zeroweight-ai/ZeroTTS-GGUF"
    assert len(ZEROTTS_MODEL.revision) == 40
    assert len(ZEROTTS_MODEL.core_artifacts) == 4
    assert len(ZEROTTS_MODEL.codec_files) == 5
    assert len(ZEROTTS_MODEL.artifacts) == 9
    assert len(ZEROTTS_MODEL.allow_patterns) == 6

    for artifact in ZEROTTS_MODEL.artifacts:
        assert artifact.remote_name
        assert artifact.local_name
        _assert_valid_sha256(artifact.sha256)


def test_zerotts_codec_int8_artifact() -> None:
    """Test that the ZeroTTS INT8 codec artifact specification is valid."""
    assert (
        ZEROTTS_CODEC_INT8_ARTIFACT.remote_name == f"onnx/codec/{ZeroTtsFile.DECODE_STEP_INT8_ONNX}"
    )
    assert (
        ZEROTTS_CODEC_INT8_ARTIFACT.local_name == f"onnx/codec/{ZeroTtsFile.DECODE_STEP_INT8_ONNX}"
    )
    assert ZEROTTS_CODEC_INT8_ARTIFACT.sha256 == ZEROTTS_CODEC_INT8_SHA256
    _assert_valid_sha256(ZEROTTS_CODEC_INT8_SHA256)
    assert ZEROTTS_CODEC_INT8_ARTIFACT in ZEROTTS_MODEL.codec_files


def test_resolve_zerotts_codec_int8() -> None:
    """Test resolution of ZeroTTS INT8 neural codec precision across CPU architectures."""
    # Explicit booleans are preserved directly
    assert resolve_zerotts_codec_int8(True) is True
    assert resolve_zerotts_codec_int8(False) is False

    # Strict string parsing
    assert resolve_zerotts_codec_int8("true") is True
    assert resolve_zerotts_codec_int8("1") is True
    assert resolve_zerotts_codec_int8("yes") is True
    assert resolve_zerotts_codec_int8("on") is True
    assert resolve_zerotts_codec_int8("false") is False
    assert resolve_zerotts_codec_int8("0") is False
    assert resolve_zerotts_codec_int8("no") is False
    assert resolve_zerotts_codec_int8("off") is False

    with pytest.raises(ValueError, match="codec_int8 must be one of"):
        resolve_zerotts_codec_int8("invalid")

    # Model name hint resolution
    assert resolve_zerotts_codec_int8(model_name="q4_hybrid") is True
    assert resolve_zerotts_codec_int8(model_name="q8_0") is False
    assert resolve_zerotts_codec_int8(model_name=ZeroTtsFile.Q4_HYBRID_GGUF_MODEL) is True
    assert resolve_zerotts_codec_int8(model_name=ZeroTtsFile.Q8_0_GGUF_MODEL) is False

    # Auto resolution on edge CPU variants (SSE4, AVX, COMPAT)
    for edge_var in (ZeroTtsVariant.SSE4, ZeroTtsVariant.AVX, ZeroTtsVariant.COMPAT):
        assert resolve_zerotts_codec_int8("auto", cpu_variant=edge_var) is True
        assert resolve_zerotts_codec_int8(None, cpu_variant=edge_var) is True
        assert is_zerotts_edge_variant(edge_var) is True

    # Auto resolution on desktop/server CPU variants (AVX2, AVX512, ARM_DOTPROD)
    for desktop_var in (ZeroTtsVariant.AVX2, ZeroTtsVariant.AVX512, ZeroTtsVariant.ARM_DOTPROD):
        assert resolve_zerotts_codec_int8("auto", cpu_variant=desktop_var) is False
        assert resolve_zerotts_codec_int8(None, cpu_variant=desktop_var) is False
        assert is_zerotts_edge_variant(desktop_var) is False


def test_zerotts_q4_model_spec_and_resolution() -> None:
    """Test the Q4 ZeroTTS model spec artifact and resolver."""
    assert ZEROTTS_Q4_MODEL.repo == "zeroweight-ai/ZeroTTS-GGUF"
    assert ZEROTTS_Q4_GGUF_ARTIFACT.remote_name == f"gguf/{ZeroTtsFile.DEFAULT_GGUF_MODEL}"
    assert ZEROTTS_Q4_GGUF_ARTIFACT.local_name == f"gguf/{ZeroTtsFile.Q4_HYBRID_GGUF_MODEL}"
    _assert_valid_sha256(ZEROTTS_Q4_GGUF_ARTIFACT.sha256)

    assert resolve_zerotts_model_spec("q8_0") is ZEROTTS_MODEL
    assert resolve_zerotts_model_spec("zerotts-q8_0.gguf") is ZEROTTS_MODEL
    assert resolve_zerotts_model_spec("q4_hybrid") is ZEROTTS_Q4_MODEL
    assert resolve_zerotts_model_spec("zerotts-q4-hybrid.gguf") is ZEROTTS_Q4_MODEL
    assert resolve_zerotts_model_spec("q4") is ZEROTTS_Q4_MODEL
    assert resolve_zerotts_model_spec(None, cpu_variant=ZeroTtsVariant.AVX2) is ZEROTTS_MODEL
    assert resolve_zerotts_model_spec(None, cpu_variant=ZeroTtsVariant.COMPAT) is ZEROTTS_Q4_MODEL
    assert resolve_zerotts_model_spec("auto", cpu_variant=ZeroTtsVariant.AVX2) is ZEROTTS_MODEL
    assert resolve_zerotts_model_spec("auto", cpu_variant=ZeroTtsVariant.AVX) is ZEROTTS_Q4_MODEL
    assert resolve_zerotts_model_spec("auto", cpu_variant=ZeroTtsVariant.COMPAT) is ZEROTTS_Q4_MODEL
    assert (
        frozenset({ZeroTtsVariant.AVX, ZeroTtsVariant.SSE4, ZeroTtsVariant.COMPAT})
        == ZEROTTS_EDGE_VARIANTS
    )
    assert resolve_zerotts_model_spec("/custom/path.gguf") is ZEROTTS_BASE_MODEL
    assert ZEROTTS_BASE_MODEL.gguf_model is None
    assert all("gguf" not in p for p in ZEROTTS_BASE_MODEL.allow_patterns)


def test_resolve_zerotts_model_centralized(tmp_path: Path) -> None:
    """Test resolve_zerotts_model resolves spec and local GGUF path across edge and desktop CPUs."""
    gguf_dir = tmp_path / ZeroTtsDirectory.GGUF
    gguf_dir.mkdir(parents=True, exist_ok=True)
    q8_path = gguf_dir / ZeroTtsFile.DEFAULT_GGUF_MODEL
    q4_path = gguf_dir / ZeroTtsFile.Q4_HYBRID_GGUF_MODEL

    # Explicit custom absolute path
    custom_path = tmp_path / "custom.gguf"
    res_custom = resolve_zerotts_model(tmp_path, custom_path)
    assert res_custom.spec is ZEROTTS_BASE_MODEL
    assert res_custom.gguf_path == custom_path

    # Explicit aliases
    res_q4 = resolve_zerotts_model(tmp_path, "q4_hybrid")
    assert res_q4.spec is ZEROTTS_Q4_MODEL
    assert res_q4.gguf_path == q4_path

    res_q8 = resolve_zerotts_model(tmp_path, "q8_0")
    assert res_q8.spec is ZEROTTS_MODEL
    assert res_q8.gguf_path == q8_path

    # Auto on edge CPU (SSE4 and AVX)
    res_auto_edge = resolve_zerotts_model(tmp_path, "auto", cpu_variant=ZeroTtsVariant.SSE4)
    assert res_auto_edge.spec is ZEROTTS_Q4_MODEL
    assert res_auto_edge.gguf_path == q4_path

    res_auto_avx = resolve_zerotts_model(tmp_path, "auto", cpu_variant=ZeroTtsVariant.AVX)
    assert res_auto_avx.spec is ZEROTTS_Q4_MODEL
    assert res_auto_avx.gguf_path == q4_path

    # Auto on AVX2 CPU
    res_auto_avx2 = resolve_zerotts_model(tmp_path, "auto", cpu_variant=ZeroTtsVariant.AVX2)
    assert res_auto_avx2.spec is ZEROTTS_MODEL
    assert res_auto_avx2.gguf_path == q8_path

    # Auto edge CPU with fallback when Q4 missing and Q8 present
    q8_path.write_bytes(b"q8_data")
    res_fallback_edge = resolve_zerotts_model(tmp_path, "auto", cpu_variant=ZeroTtsVariant.SSE4)
    assert res_fallback_edge.spec is ZEROTTS_MODEL
    assert res_fallback_edge.gguf_path == q8_path

    # Auto AVX2 CPU with fallback when Q8 missing and Q4 present
    q8_path.unlink()
    q4_path.write_bytes(b"q4_data")
    res_fallback_avx2 = resolve_zerotts_model(tmp_path, "auto", cpu_variant=ZeroTtsVariant.AVX2)
    assert res_fallback_avx2.spec is ZEROTTS_Q4_MODEL
    assert res_fallback_avx2.gguf_path == q4_path

    # Explicit selection does not fall back even if the chosen model file is missing
    res_explicit_q8_missing = resolve_zerotts_model(tmp_path, "q8_0")
    assert res_explicit_q8_missing.spec is ZEROTTS_MODEL
    assert res_explicit_q8_missing.gguf_path == q8_path

    q8_path.write_bytes(b"q8_data")
    q4_path.unlink()
    res_explicit_q4_missing = resolve_zerotts_model(tmp_path, "q4_hybrid")
    assert res_explicit_q4_missing.spec is ZEROTTS_Q4_MODEL
    assert res_explicit_q4_missing.gguf_path == q4_path


def test_normalize_zerotts_model_path_validation(tmp_path: Path) -> None:
    """Test normalize_zerotts_model validates absolute path extension and type."""
    from wyoming_vietnamese.tts_model import normalize_zerotts_model

    nonexistent_path = tmp_path / "nonexistent.gguf"
    assert normalize_zerotts_model(str(nonexistent_path)) == str(nonexistent_path)

    existing_path = tmp_path / "existing.gguf"
    existing_path.write_bytes(b"GGUF_TEST_DATA")
    assert normalize_zerotts_model(str(existing_path)) == str(existing_path)

    with pytest.raises(ValueError, match=r"must have a '\.gguf' file extension"):
        normalize_zerotts_model(str(tmp_path / "model.bin"))

    with pytest.raises(ValueError, match=r"must have a '\.gguf' file extension"):
        normalize_zerotts_model(str(tmp_path / "model"))

    dir_path = tmp_path / "model_dir.gguf"
    dir_path.mkdir()
    with pytest.raises(ValueError, match="cannot be a directory"):
        normalize_zerotts_model(str(dir_path))


def test_call_spec_resolver_does_not_swallow_internal_type_error() -> None:
    """Test call_spec_resolver propagates TypeError raised from within the resolver itself."""
    from wyoming_vietnamese.tts_model import call_spec_resolver

    def faulty_resolver(model_name: str | None, *, cpu_variant: object = None) -> ZeroTtsModelSpec:
        raise TypeError("internal type error within resolver body")

    with pytest.raises(TypeError, match="internal type error within resolver body"):
        call_spec_resolver(faulty_resolver, "test")


def test_resolve_zerotts_cpu_variant_memoization(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test resolve_zerotts_cpu_variant centralizes CPU detection and memoizes result."""
    reset_zerotts_cpu_variant()
    call_count = 0

    def mock_detect() -> str:
        nonlocal call_count
        call_count += 1
        return ZeroTtsVariant.AVX2

    monkeypatch.setattr("wyoming_vietnamese.tts_model.detect_cpu_variant", mock_detect)

    # First call detects host variant
    var1 = resolve_zerotts_cpu_variant()
    assert var1 == ZeroTtsVariant.AVX2
    assert call_count == 1

    # Second call returns memoized result without calling detect_cpu_variant again
    var2 = resolve_zerotts_cpu_variant()
    assert var2 == ZeroTtsVariant.AVX2
    assert call_count == 1

    # Explicit override uses provided variant without modifying memoized value
    override = resolve_zerotts_cpu_variant(ZeroTtsVariant.SSE4)
    assert override == ZeroTtsVariant.SSE4
    assert call_count == 1

    # Reset allows re-detection
    reset_zerotts_cpu_variant()
    var3 = resolve_zerotts_cpu_variant()
    assert var3 == ZeroTtsVariant.AVX2
    assert call_count == 2
    reset_zerotts_cpu_variant()


def test_normalize_zerotts_model_preserve_auto() -> None:
    """Test normalize_zerotts_model preserves auto and default when preserve_auto is True."""
    assert normalize_zerotts_model("auto", preserve_auto=True) == "auto"
    assert normalize_zerotts_model("default", preserve_auto=True) == "auto"
    assert normalize_zerotts_model("  AUTO  ", preserve_auto=True) == "auto"
    assert (
        normalize_zerotts_model("q4_hybrid", preserve_auto=True) == ZeroTtsFile.Q4_HYBRID_GGUF_MODEL
    )
    assert normalize_zerotts_model("q8_0", preserve_auto=True) == ZeroTtsFile.Q8_0_GGUF_MODEL
    assert normalize_zerotts_model(None, preserve_auto=True) is None


def test_resolve_quality_zerotts_model() -> None:
    """Test resolve_quality_zerotts_model maps quality tiers to model filenames."""
    assert resolve_quality_zerotts_model(ZeroTtsQuality.HIGH) == ZeroTtsFile.Q8_0_GGUF_MODEL
    assert resolve_quality_zerotts_model(ZeroTtsQuality.MEDIUM) == ZeroTtsFile.Q4_HYBRID_GGUF_MODEL
    assert resolve_quality_zerotts_model(ZeroTtsQuality.LOW) == ZeroTtsFile.Q4_HYBRID_GGUF_MODEL
    assert resolve_quality_zerotts_model(ZeroTtsQuality.AUTO) == DEFAULT_ZEROTTS_AUTO
    assert resolve_quality_zerotts_model() == DEFAULT_ZEROTTS_AUTO
    assert resolve_quality_zerotts_model("unknown") == DEFAULT_ZEROTTS_AUTO


def test_resolve_quality_codebooks() -> None:
    """Test resolve_quality_codebooks maps quality tiers and edge hardware to codebook counts."""
    # Explicit quality tiers are invariant to edge hardware
    assert resolve_quality_codebooks(ZeroTtsQuality.HIGH, is_edge=False) == ZeroTtsCodebook.HIGH
    assert resolve_quality_codebooks(ZeroTtsQuality.HIGH, is_edge=True) == ZeroTtsCodebook.HIGH
    assert resolve_quality_codebooks(ZeroTtsQuality.MEDIUM, is_edge=False) == ZeroTtsCodebook.MEDIUM
    assert resolve_quality_codebooks(ZeroTtsQuality.MEDIUM, is_edge=True) == ZeroTtsCodebook.MEDIUM
    assert resolve_quality_codebooks(ZeroTtsQuality.LOW, is_edge=False) == ZeroTtsCodebook.LOW
    assert resolve_quality_codebooks(ZeroTtsQuality.LOW, is_edge=True) == ZeroTtsCodebook.LOW

    # Auto and default depend on edge hardware
    assert resolve_quality_codebooks(ZeroTtsQuality.AUTO, is_edge=False) == ZeroTtsCodebook.DEFAULT
    assert resolve_quality_codebooks(ZeroTtsQuality.AUTO, is_edge=True) == ZeroTtsCodebook.MEDIUM
    assert resolve_quality_codebooks(is_edge=False) == ZeroTtsCodebook.DEFAULT
    assert resolve_quality_codebooks(is_edge=True) == ZeroTtsCodebook.MEDIUM
    assert resolve_quality_codebooks("unknown", is_edge=False) == ZeroTtsCodebook.DEFAULT
    assert resolve_quality_codebooks("unknown", is_edge=True) == ZeroTtsCodebook.MEDIUM
