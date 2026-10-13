"""Offline developer tool to quantize MOSS Audio Tokenizer Codec step graph to INT8.

This tool produces ``moss_audio_tokenizer_decode_step.int8.onnx`` ahead of time
using dynamic INT8 weight quantization for matrix multiplication and convolution
operators. It includes an automated fidelity verification suite to detect
quantization drift against the FP32 baseline (cosine similarity >= 0.995,
NMSE <= 0.01, Mel-spectrogram L1 error <= 0.05).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    import onnxruntime as ort

import numpy as np

if __name__ == "__main__" and not __package__:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from wyoming_vietnamese.const import ZeroTtsFile

_LOGGER: Final[logging.Logger] = logging.getLogger("quantize_codec")

COSINE_SIMILARITY_THRESHOLD: Final[float] = 0.995
NMSE_THRESHOLD: Final[float] = 0.01
SPECTRAL_L1_THRESHOLD: Final[float] = 0.05


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D or flattened numeric arrays.

    Args:
        a: First numeric array.
        b: Second numeric array.

    Returns:
        Cosine similarity in range [-1.0, 1.0].
    """
    flat_a = a.astype(np.float64).reshape(-1)
    flat_b = b.astype(np.float64).reshape(-1)
    norm_a = float(np.linalg.norm(flat_a))
    norm_b = float(np.linalg.norm(flat_b))
    if math.isclose(norm_a, 0.0, abs_tol=1e-12) or math.isclose(norm_b, 0.0, abs_tol=1e-12):
        return 1.0 if math.isclose(norm_a, norm_b, abs_tol=1e-12) else 0.0
    dot = float(np.dot(flat_a, flat_b))
    return float(np.clip(dot / (norm_a * norm_b), -1.0, 1.0))


def compute_nmse(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Compute Normalized Mean Squared Error (NMSE) between reference and candidate.

    Args:
        reference: Ground-truth reference waveform array.
        candidate: Quantized candidate waveform array.

    Returns:
        Normalized mean squared error as a non-negative float.
    """
    ref = reference.astype(np.float64).reshape(-1)
    cand = candidate.astype(np.float64).reshape(-1)
    mse = float(np.mean((ref - cand) ** 2))
    var_ref = float(np.var(ref))
    if math.isclose(var_ref, 0.0, abs_tol=1e-12):
        return 0.0 if math.isclose(mse, 0.0, abs_tol=1e-12) else 1.0
    return float(mse / var_ref)


def compute_spectral_l1(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Compute mean normalized spectral L1 distance across Fourier magnitude bins.

    Args:
        reference: Ground-truth reference waveform array.
        candidate: Quantized candidate waveform array.

    Returns:
        Normalized spectral L1 magnitude difference.
    """
    ref = reference.astype(np.float64).reshape(-1)
    cand = candidate.astype(np.float64).reshape(-1)
    n_fft = 512
    hop = 128
    if len(ref) < n_fft:
        ref = np.pad(ref, (0, n_fft - len(ref)))
        cand = np.pad(cand, (0, n_fft - len(cand)))

    n_frames = (len(ref) - n_fft) // hop + 1
    if n_frames <= 0:
        return 0.0

    window = np.hanning(n_fft)
    spec_ref: list[np.ndarray] = []
    spec_cand: list[np.ndarray] = []
    for i in range(n_frames):
        start = i * hop
        end = start + n_fft
        chunk_ref = ref[start:end] * window
        chunk_cand = cand[start:end] * window
        spec_ref.append(np.abs(np.fft.rfft(chunk_ref)))
        spec_cand.append(np.abs(np.fft.rfft(chunk_cand)))

    mag_ref = np.array(spec_ref)
    mag_cand = np.array(spec_cand)
    ref_norm = float(np.mean(mag_ref))
    if math.isclose(ref_norm, 0.0, abs_tol=1e-12):
        return 0.0
    return float(np.mean(np.abs(mag_ref - mag_cand)) / ref_norm)


def _file_sha256(path: Path) -> str:
    """Calculate the hexadecimal SHA-256 digest of a file.

    Args:
        path: Path to the target file.

    Returns:
        64-character lowercase hex digest string.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def _build_synthetic_streaming_state(meta: Mapping[str, object]) -> dict[str, np.ndarray]:
    """Initialize empty input ring buffers matching codec_browser_onnx_meta.json.

    Args:
        meta: Loaded dictionary from codec_browser_onnx_meta.json.

    Returns:
        Dictionary of input buffer arrays keyed by input tensor name.
    """
    streaming_obj = meta.get("streaming_decode")
    streaming = streaming_obj if isinstance(streaming_obj, dict) else {}
    transformer_specs_obj = streaming.get("transformer_offsets")
    transformer_specs = (
        list(transformer_specs_obj) if isinstance(transformer_specs_obj, list) else []
    )
    attention_specs_obj = streaming.get("attention_caches")
    attention_specs = list(attention_specs_obj) if isinstance(attention_specs_obj, list) else []
    state: dict[str, np.ndarray] = {
        str(spec["input_name"]): np.zeros(tuple(spec["shape"]), dtype=np.int32)
        for spec in transformer_specs
        if isinstance(spec, dict) and "input_name" in spec and "shape" in spec
    }
    for spec in attention_specs:
        if isinstance(spec, dict):
            state[str(spec["offset_input_name"])] = np.zeros(
                tuple(spec["offset_shape"]), dtype=np.int32
            )
            state[str(spec["cached_keys_input_name"])] = np.zeros(
                tuple(spec["cache_shape"]), dtype=np.float32
            )
            state[str(spec["cached_values_input_name"])] = np.zeros(
                tuple(spec["cache_shape"]), dtype=np.float32
            )
            state[str(spec["cached_positions_input_name"])] = np.full(
                tuple(spec["positions_shape"]), -1, dtype=np.int32
            )
    return state


def _load_codec_fidelity_config(
    meta_path: Path,
    num_codebooks: int,
) -> tuple[dict[str, object], int] | None:
    """Load metadata configuration and resolve acoustic codebook count.

    Args:
        meta_path: Path to codec_browser_onnx_meta.json.
        num_codebooks: Default number of codebooks if omitted in metadata.

    Returns:
        Tuple of (meta dictionary, resolved num_codebooks), or None if file missing.
    """
    if not meta_path.is_file():
        _LOGGER.warning("Codec metadata file missing at %s", meta_path)
        return None
    raw_meta: object = json.loads(meta_path.read_text(encoding="utf-8"))
    meta: dict[str, object] = raw_meta if isinstance(raw_meta, dict) else {}
    cfg = meta.get("codec_config")
    resolved_k = (
        int(cfg["num_quantizers"])
        if isinstance(cfg, dict) and "num_quantizers" in cfg
        else num_codebooks
    )
    return meta, resolved_k


def _update_streaming_state(
    state: dict[str, np.ndarray],
    named_outputs: Mapping[str, np.ndarray],
    transformer_specs: Sequence[object],
    attention_specs: Sequence[object],
) -> None:
    """Update streaming ring buffers and caches from step inference outputs.

    Args:
        state: Mutable dictionary of input buffer arrays to update.
        named_outputs: Named output array mapping from the step model.
        transformer_specs: Sequence of transformer offset specifications.
        attention_specs: Sequence of attention cache specifications.
    """
    for spec in transformer_specs:
        if isinstance(spec, dict) and "input_name" in spec and "output_name" in spec:
            state[str(spec["input_name"])] = named_outputs[str(spec["output_name"])]
    for spec in attention_specs:
        if isinstance(spec, dict):
            for key in ("offset", "cached_keys", "cached_values", "cached_positions"):
                in_key = f"{key}_input_name"
                out_key = f"{key}_output_name"
                if in_key in spec and out_key in spec:
                    state[str(spec[in_key])] = named_outputs[str(spec[out_key])]


def _execute_codec_step(
    sess: ort.InferenceSession,
    output_names: Sequence[str],
    codes: np.ndarray,
    lengths: np.ndarray,
    state: dict[str, np.ndarray],
    transformer_specs: Sequence[object],
    attention_specs: Sequence[object],
) -> np.ndarray:
    """Execute a single streaming step, update ring buffer state, and return audio waveform.

    Args:
        sess: ONNX Runtime inference session.
        output_names: Ordered sequence of output tensor names.
        codes: Codebook input tensor of shape (1, 1, K).
        lengths: Code length tensor of shape (1,).
        state: Mutable streaming state ring buffers.
        transformer_specs: Sequence of transformer offset specifications.
        attention_specs: Sequence of attention cache specifications.

    Returns:
        Flattened 1D float32 audio waveform segment.
    """
    feeds = {"audio_codes": codes, "audio_code_lengths": lengths, **state}
    outputs = sess.run(None, feeds)
    named = {str(k): np.asarray(v) for k, v in zip(output_names, outputs, strict=False)}
    _update_streaming_state(state, named, transformer_specs, attention_specs)
    return named["audio"].reshape(-1)


def _run_codec_benchmark_simulation(
    fp32_sess: ort.InferenceSession,
    int8_sess: ort.InferenceSession,
    meta: Mapping[str, object],
    num_codebooks: int,
    benchmark_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Execute synthetic streaming inference across benchmark frames for FP32 and INT8 sessions.

    Args:
        fp32_sess: FP32 ONNX Runtime inference session.
        int8_sess: INT8 ONNX Runtime inference session.
        meta: Loaded dictionary from codec_browser_onnx_meta.json.
        num_codebooks: Number of acoustic codebooks (K).
        benchmark_frames: Number of progressive frames to simulate.

    Returns:
        Tuple of concatenated (full_fp32, full_int8) 1D audio waveforms.
    """
    state_fp32 = _build_synthetic_streaming_state(meta)
    state_int8 = _build_synthetic_streaming_state(meta)

    out_names_fp32 = [o.name for o in fp32_sess.get_outputs()]
    out_names_int8 = [o.name for o in int8_sess.get_outputs()]

    streaming_obj = meta.get("streaming_decode")
    streaming = streaming_obj if isinstance(streaming_obj, dict) else {}
    transformer_specs = list(streaming.get("transformer_offsets", []))
    attention_specs = list(streaming.get("attention_caches", []))

    rng = np.random.default_rng(2026)
    audio_fp32_chunks: list[np.ndarray] = []
    audio_int8_chunks: list[np.ndarray] = []

    for _ in range(benchmark_frames):
        codes = rng.integers(0, 1024, size=(1, 1, num_codebooks), dtype=np.int32)
        lengths = np.array([1], dtype=np.int32)

        audio_fp32 = _execute_codec_step(
            fp32_sess,
            out_names_fp32,
            codes,
            lengths,
            state_fp32,
            transformer_specs,
            attention_specs,
        )
        audio_int8 = _execute_codec_step(
            int8_sess,
            out_names_int8,
            codes,
            lengths,
            state_int8,
            transformer_specs,
            attention_specs,
        )
        audio_fp32_chunks.append(audio_fp32)
        audio_int8_chunks.append(audio_int8)

    return np.concatenate(audio_fp32_chunks), np.concatenate(audio_int8_chunks)


def _evaluate_codec_fidelity_metrics(full_fp32: np.ndarray, full_int8: np.ndarray) -> bool:
    """Compute error metrics against FP32 baseline, print summary, and verify thresholds.

    Args:
        full_fp32: Full 1D float32 reference audio waveform.
        full_int8: Full 1D float32 quantized audio waveform.

    Returns:
        True if all metrics satisfy their fidelity thresholds, False otherwise.
    """
    cos_sim = compute_cosine_similarity(full_fp32, full_int8)
    nmse = compute_nmse(full_fp32, full_int8)
    spec_l1 = compute_spectral_l1(full_fp32, full_int8)

    print("=" * 64)
    print("MOSS Codec INT8 Quantization Fidelity Verification")
    print("=" * 64)
    print(
        f"Cosine Similarity : {cos_sim:.5f} (Threshold >= {COSINE_SIMILARITY_THRESHOLD:.3f}) "
        f"{'PASS' if cos_sim >= COSINE_SIMILARITY_THRESHOLD else 'FAIL'}"
    )
    print(
        f"Normalized MSE    : {nmse:.5f} (Threshold <= {NMSE_THRESHOLD:.3f}) "
        f"{'PASS' if nmse <= NMSE_THRESHOLD else 'FAIL'}"
    )
    print(
        f"Spectral L1 Dist  : {spec_l1:.5f} (Threshold <= {SPECTRAL_L1_THRESHOLD:.3f}) "
        f"{'PASS' if spec_l1 <= SPECTRAL_L1_THRESHOLD else 'FAIL'}"
    )
    print("=" * 64)

    return (
        cos_sim >= COSINE_SIMILARITY_THRESHOLD
        and nmse <= NMSE_THRESHOLD
        and spec_l1 <= SPECTRAL_L1_THRESHOLD
    )


def verify_codec_fidelity(
    meta_path: Path,
    fp32_model_path: Path,
    int8_model_path: Path,
    *,
    num_codebooks: int = 16,
    benchmark_frames: int = 10,
) -> bool:
    """Execute reference and INT8 models to verify that quality metrics satisfy thresholds.

    Args:
        meta_path: Path to codec_browser_onnx_meta.json.
        fp32_model_path: Path to FP32 decode_step ONNX model.
        int8_model_path: Path to INT8 decode_step ONNX model.
        num_codebooks: Number of acoustic codebooks (K).
        benchmark_frames: Number of progressive frames to simulate.

    Returns:
        True if all mathematical error bounds are satisfied, False otherwise.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        _LOGGER.warning("onnxruntime unavailable for offline fidelity verification")
        return False

    config = _load_codec_fidelity_config(meta_path, num_codebooks)
    if config is None:
        return False
    meta, resolved_codebooks = config

    sess_opts = ort.SessionOptions()
    sess_opts.inter_op_num_threads = 1
    sess_opts.intra_op_num_threads = 2

    fp32_sess = ort.InferenceSession(
        str(fp32_model_path), sess_options=sess_opts, providers=["CPUExecutionProvider"]
    )
    int8_sess = ort.InferenceSession(
        str(int8_model_path), sess_options=sess_opts, providers=["CPUExecutionProvider"]
    )

    full_fp32, full_int8 = _run_codec_benchmark_simulation(
        fp32_sess, int8_sess, meta, resolved_codebooks, benchmark_frames
    )
    return _evaluate_codec_fidelity_metrics(full_fp32, full_int8)


def quantize_moss_codec_step(
    codec_dir: Path,
    output_path: Path | None = None,
    *,
    skip_verify: bool = False,
) -> Path | None:
    """Quantize the MOSS Codec decode_step ONNX model to INT8 precision.

    Args:
        codec_dir: Directory containing MOSS Codec model files.
        output_path: Optional explicit output path for the INT8 model.
        skip_verify: Whether to bypass mathematical drift verification.

    Returns:
        Path to the quantized INT8 model file if successful, None otherwise.
    """
    src_step = codec_dir / ZeroTtsFile.DECODE_STEP_ONNX
    if not src_step.is_file():
        _LOGGER.error("Source decode step model missing at %s", src_step)
        return None

    target_out = (
        output_path if output_path is not None else codec_dir / ZeroTtsFile.DECODE_STEP_INT8_ONNX
    )
    target_out.parent.mkdir(parents=True, exist_ok=True)

    stage_dir = target_out.parent / f".tmp_stage_{target_out.name}_{os.getpid()}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    try:
        try:
            import importlib

            importlib.import_module("onnx")
            ort_quant = importlib.import_module("onnxruntime.quantization")
            quant_type = ort_quant.QuantType
            quantize_dynamic_fn = ort_quant.quantize_dynamic
        except (ImportError, AttributeError) as err:
            _LOGGER.error(
                "Offline quantization requires 'onnx' and 'onnxruntime.quantization' (%s)",
                err,
            )
            return None

        src_shared = codec_dir / ZeroTtsFile.DECODE_SHARED_DATA
        if src_shared.is_file():
            shutil.copy2(src_shared, stage_dir / src_shared.name)
        shutil.copy2(src_step, stage_dir / src_step.name)

        stage_in = stage_dir / src_step.name
        stage_out = stage_dir / target_out.name

        quantize_dynamic_fn(
            model_input=str(stage_in),
            model_output=str(stage_out),
            weight_type=quant_type.QInt8,
        )

        if not stage_out.is_file():
            _LOGGER.error("Quantization failed to produce output at %s", stage_out)
            return None

        meta_path = codec_dir / "codec_browser_onnx_meta.json"
        if not skip_verify:
            if not meta_path.is_file():
                _LOGGER.error("Verification metadata missing at %s", meta_path)
                return None
            passed = verify_codec_fidelity(meta_path, src_step, stage_out)
            if not passed:
                _LOGGER.error("Quantized candidate model exceeded fidelity drift thresholds!")
                return None

        shutil.move(str(stage_out), str(target_out))
        return target_out
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)


def print_codec_quant_summary(
    codec_dir: Path,
    output_path: Path,
    success: bool,
) -> None:
    """Print architectural summary and quantization info for the MOSS Codec INT8 model.

    Args:
        codec_dir: Source codec directory path.
        output_path: Target INT8 ONNX output file path.
        success: Whether the quantization succeeded.
    """
    src_step = codec_dir / ZeroTtsFile.DECODE_STEP_ONNX
    src_data = codec_dir / ZeroTtsFile.DECODE_SHARED_DATA
    src_size = (src_step.stat().st_size if src_step.is_file() else 0) + (
        src_data.stat().st_size if src_data.is_file() else 0
    )
    dst_size = output_path.stat().st_size if output_path.is_file() else 0

    print("=" * 64)
    print("MOSS Audio Tokenizer INT8 Dynamic Quantization Summary")
    print("=" * 64)
    print(f"Source Codec Dir : {codec_dir}")
    print(f"Source Step ONNX : {src_step}")
    print(f"Output INT8 ONNX : {output_path}")
    print(f"Status           : {'SUCCESS' if success else 'FAILED'}")
    if success and dst_size > 0 and src_size > 0:
        sha = _file_sha256(output_path)
        print(f"Output SHA-256   : {sha}")
        print(f"Original Size    : {src_size / (1024 * 1024):.2f} MB")
        print(f"Quantized Size   : {dst_size / (1024 * 1024):.2f} MB")
        print(f"Reduction        : {((src_size - dst_size) / src_size) * 100:.1f}%")
    print("=" * 64)


def main() -> None:
    """CLI entry point for offline MOSS Codec INT8 dynamic quantization."""
    parser = argparse.ArgumentParser(
        description="Quantize MOSS Audio Tokenizer Codec step graph to INT8 dynamic precision."
    )
    parser.add_argument(
        "--codec-dir",
        type=Path,
        required=True,
        help="Path to directory containing MOSS codec assets (or ZeroTTS model directory)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit output path for moss_audio_tokenizer_decode_step.int8.onnx",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip mathematical drift verification against FP32 baseline",
    )
    args = parser.parse_args()

    codec_dir = args.codec_dir.resolve()
    if (codec_dir / "onnx" / "codec").is_dir():
        codec_dir = codec_dir / "onnx" / "codec"

    target_out = (
        args.output.resolve()
        if args.output is not None
        else codec_dir / ZeroTtsFile.DECODE_STEP_INT8_ONNX
    )

    result = quantize_moss_codec_step(codec_dir, target_out, skip_verify=args.skip_verify)
    success = result is not None and result.is_file()
    print_codec_quant_summary(codec_dir, target_out, success)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
