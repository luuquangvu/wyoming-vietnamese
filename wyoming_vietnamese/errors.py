"""Typed synthesis limits shared by TTS handlers and engine adapters."""


class SynthesisTextLimitError(ValueError):
    """Raised when normalized text exceeds an engine's token capacity."""


class SynthesisOutputLimitError(RuntimeError):
    """Raised when an engine stops before reaching end-of-audio."""
