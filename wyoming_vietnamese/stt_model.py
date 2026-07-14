"""Checksum-pinned Sherpa-ONNX speech-to-text model supported by the service.

The model is part of the service definition rather than a deployment setting, exactly
like the NghiTTS voice catalogue. Pinning the repository, its commit, and every file's
SHA-256 digest means an upstream replacement stops startup instead of silently changing
recognition quality or executing unreviewed model content.
"""

from __future__ import annotations

from dataclasses import dataclass


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

    @property
    def artifacts(self) -> tuple[SttArtifact, ...]:
        """Return every pinned repository file, inference graphs first."""
        return (*self.graphs, self.tokenizer)

    @property
    def allow_patterns(self) -> tuple[str, ...]:
        """Return the exact repository files the service downloads."""
        return tuple(artifact.remote_name for artifact in self.artifacts)


# Verified against the repository at the pinned commit on 2026-08-11.
STT_MODEL = SttModelSpec(
    repo="hynt/Zipformer-30M-RNNT-6000h",
    revision="24ed30248e1c96bb690c81c24ab4e056f8cd9fce",
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
)
