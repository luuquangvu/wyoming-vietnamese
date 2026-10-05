"""Checksum-pinned Sherpa-ONNX speech-to-text model supported by the service.

The model is part of the service definition rather than a deployment setting, exactly
like the NghiTTS voice catalogue. Pinning the repository, its commit, and every file's
SHA-256 digest means an upstream replacement stops startup instead of silently changing
recognition quality or executing unreviewed model content.
"""

from __future__ import annotations

from dataclasses import dataclass

from .const import DEFAULT_STT_ENGINE, SttEngine


@dataclass(frozen=True, slots=True)
class SttArtifact:
    """Immutable identity and integrity metadata for one pinned repository file."""

    remote_name: str
    local_name: str
    sha256: str


@dataclass(frozen=True, slots=True)
class SttModelSpec:
    """Immutable identity of the speech-to-text model and every file it needs."""

    repo: str
    revision: str
    graphs: tuple[SttArtifact, ...]
    tokenizer: SttArtifact
    description: str = "Zipformer RNNT STT model"

    @property
    def artifacts(self) -> tuple[SttArtifact, ...]:
        """Return every pinned repository file, inference graphs first."""
        return (*self.graphs, self.tokenizer)

    @property
    def allow_patterns(self) -> tuple[str, ...]:
        """Return the exact repository files the service downloads."""
        return tuple(artifact.remote_name for artifact in self.artifacts)


# Verified against the repository at the pinned commit on 2026-08-11.
ZIPFORMER_MODEL = SttModelSpec(
    repo="hynt/Zipformer-30M-RNNT-6000h",
    revision="ad6f873b5f5b6ed083821253a88d81afed2505c4",
    graphs=(
        SttArtifact(
            "encoder-epoch-20-avg-10.int8.onnx",
            "encoder.onnx",
            "8ef5286dd427eb108055c2ddc1982aa31e544706072d5ea228729292dacade68",
        ),
        SttArtifact(
            "decoder-epoch-20-avg-10.int8.onnx",
            "decoder.onnx",
            "b491630c33e76146b7296ab68cee5ae3a8d572732d36a552ee231d4419e06d32",
        ),
        SttArtifact(
            "joiner-epoch-20-avg-10.int8.onnx",
            "joiner.onnx",
            "7311d2e17b810ecea515d79c71cc4668af8759256a06fa01d27047772320c821",
        ),
    ),
    # The repository ships no tokens.txt; it is generated from this SentencePiece model.
    tokenizer=SttArtifact(
        "bpe.model",
        "bpe.model",
        "002894e7a82d80ffa5e25008ec8c5496159db804005e2103de96b01b4c13d445",
    ),
    description="Zipformer RNNT STT model",
)

# Verified against the repository at the pinned commit on 2026-09-17.
GIPFORMER_MODEL = SttModelSpec(
    repo="g-group-ai-lab/gipformer1.5-68M-rnnt",
    revision="dd9227dcd8705c13f33bdbe59728d546ab94480f",
    graphs=(
        SttArtifact(
            "encoder.int8.onnx",
            "encoder.onnx",
            "b528768939c7711a889be81a718ea7f2ee50d0d2d384d53f399e15b44bd9408c",
        ),
        SttArtifact(
            "decoder.int8.onnx",
            "decoder.onnx",
            "e0a156b5454722a524230f9e35d5d928cfcdd3723437b418e5bd5e04f1c3a101",
        ),
        SttArtifact(
            "joiner.int8.onnx",
            "joiner.onnx",
            "12636559d135315f002a1e1b477077d415e888477378db0fd450aee5b21ac551",
        ),
    ),
    tokenizer=SttArtifact(
        "bpe.model",
        "bpe.model",
        "289dbb44527c13c419ae3a4d8ce6a349f01a97f8777e69934a77e3692d2f10db",
    ),
    description="Gipformer 1.5 68M RNNT STT model",
)

STT_MODELS: dict[str, SttModelSpec] = {
    SttEngine.ZIPFORMER: ZIPFORMER_MODEL,
    SttEngine.GIPFORMER: GIPFORMER_MODEL,
}

DEFAULT_STT_MODEL = ZIPFORMER_MODEL


def get_stt_model(engine: str = DEFAULT_STT_ENGINE) -> SttModelSpec:
    """Resolve one supported STT model specification or report the available choices."""
    try:
        return STT_MODELS[engine.strip().lower()]
    except KeyError as err:
        choices = ", ".join(STT_MODELS)
        raise ValueError(f"STT_ENGINE must be one of: {choices}") from err
