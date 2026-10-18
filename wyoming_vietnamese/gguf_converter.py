"""GGUF format parser, validator, and Q8_0 to Q4_0 hybrid converter for ZeroTTS."""

from __future__ import annotations

import os
import re
import struct
from pathlib import Path
from typing import Any, Final

import numpy as np

GGML_TYPE_F32: Final = 0
GGML_TYPE_F16: Final = 1
GGML_TYPE_Q4_0: Final = 2
GGML_TYPE_Q4_1: Final = 3
GGML_TYPE_Q5_0: Final = 6
GGML_TYPE_Q8_0: Final = 8

GGUF_MAGIC: Final = 0x46554747
GGUF_VERSION: Final = 3
GGUF_SUPPORTED_VERSIONS: Final = (2, 3)
GGUF_DEFAULT_ALIGNMENT: Final = 32

MAX_GGUF_TENSORS: Final = 1_000_000
MAX_GGUF_KV_COUNT: Final = 65_536
MAX_GGUF_STRING_LENGTH: Final = 65_536
MAX_GGUF_DIMENSIONS: Final = 8
MAX_GGUF_ALIGNMENT: Final = 65_536


_FFN_TOKEN_PATTERN: Final = re.compile(
    r"(?:^|[._])(?:ffn|mlp|feed_forward|ffn_up|ffn_down|ffn_gate|fc1|fc2|w1|w2|w3)(?:[._]|$)"
)
_ATTN_TOKEN_PATTERN: Final = re.compile(
    r"(?:^|[._])(?:"
    r"attn|attention|self_attn|self_attention|cross_attn|cross_attention|"
    r"self_q|self_k|self_v|self_o|cross_q|cross_k|cross_v|cross_o|"
    r"attn_q|attn_k|attn_v|attn_o|q_proj|k_proj|v_proj|o_proj|out_proj|"
    r"wq|wk|wv|wo"
    r")(?:[._]|$)"
)
_PINNED_Q8_TOKEN_PATTERN: Final = re.compile(
    r"(?:^|[._])(?:audio_head|audio_embed|text_embed|codebook|text_encoder|txt)(?:[._]|$)"
)
_ATTN_EXCLUDE_TOKEN_PATTERN: Final = re.compile(
    r"(?:^|[._])(?:norm|bias|emb|embed|embedding|head|codebook)(?:[._]|$)"
)
_FFN_EXCLUDE_TOKEN_PATTERN: Final = re.compile(
    r"(?:^|[._])(?:norm|bias|emb|embed|embedding|head|text_encoder|codebook)(?:[._]|$)"
)


def is_pinned_q8_tensor(tensor_name: str) -> bool:
    """Return True if tensor must remain in Q8_0 or higher precision to preserve audio fidelity.

    Audio codebook prediction heads (audio_head.*), audio embedding lookup tables (audio_embed.*),
    and text token embeddings (text_embed.*) are sensitive discrete probability distributions
    and embedding bases; quantizing them below Q8 causes metallic distortion, phonetic collapse,
    and chaotic codebook sampling.

    Args:
        tensor_name: Name of the GGUF tensor.

    Returns:
        True if the tensor must be pinned to Q8_0 or higher precision.
    """
    return bool(_PINNED_Q8_TOKEN_PATTERN.search(tensor_name.lower()))


def is_attention_tensor(tensor_name: str) -> bool:
    """Return True if tensor belongs to an Attention layer (Q, K, V, O projections).

    Args:
        tensor_name: Name of the GGUF tensor.

    Returns:
        True if the tensor represents Attention query/key/value/output projections.
    """
    if is_pinned_q8_tensor(tensor_name):
        return False
    name_lower = tensor_name.lower()
    if _ATTN_EXCLUDE_TOKEN_PATTERN.search(name_lower):
        return False
    return bool(_ATTN_TOKEN_PATTERN.search(name_lower))


def is_ffn_tensor(tensor_name: str) -> bool:
    """Return True if tensor belongs to a Feed-Forward Network suitable for Q4_0.

    Args:
        tensor_name: Name of the GGUF tensor.

    Returns:
        True if the tensor represents a Feed-Forward Network layer.
    """
    if is_pinned_q8_tensor(tensor_name):
        return False
    name_lower = tensor_name.lower()
    if _FFN_EXCLUDE_TOKEN_PATTERN.search(name_lower):
        return False
    return bool(_FFN_TOKEN_PATTERN.search(name_lower))


def quantize_row_q8_0_to_q4_0(q8_bytes: bytes, n_elements: int) -> bytes:
    """Quantize Q8_0 tensor bytes to Q4_0 format using vectorized NumPy.

    Args:
        q8_bytes: Contiguous Q8_0 block byte payload.
        n_elements: Total number of tensor float elements represented.

    Returns:
        Packed Q4_0 block byte payload.

    Raises:
        ValueError: If q8_bytes size does not match expected Q8_0 block count.
    """
    if n_elements <= 0 or n_elements % 32 != 0:
        raise ValueError(f"Invalid n_elements: {n_elements}; must be a positive multiple of 32")
    n_blocks = n_elements // 32
    if len(q8_bytes) != n_blocks * 34:
        raise ValueError(f"Unexpected Q8_0 size: expected {n_blocks * 34}, got {len(q8_bytes)}")

    q8_arr = np.frombuffer(q8_bytes, dtype=np.uint8).reshape(n_blocks, 34)
    deltas = q8_arr[:, :2].copy().view(dtype=np.float16).reshape(n_blocks, 1)
    qs = q8_arr[:, 2:].copy().view(dtype=np.int8)

    weights_f32 = qs.astype(np.float32) * deltas.astype(np.float32)
    max_abs = np.max(np.abs(weights_f32), axis=1, keepdims=True)
    d_q4 = np.where(max_abs > 0, max_abs / -8.0, 0.0).astype(np.float16)
    d_q4_f32 = d_q4.astype(np.float32)
    safe_d_q4 = np.where(d_q4_f32 != 0, d_q4_f32, 1.0)
    id_q4 = np.where(d_q4_f32 != 0, 1.0 / safe_d_q4, 0.0)

    q = np.clip(np.round(weights_f32 * id_q4) + 8, 0, 15).astype(np.uint8)
    low_nibble = q[:, :16]
    high_nibble = q[:, 16:]
    packed_bytes = (low_nibble & 0x0F) | ((high_nibble & 0x0F) << 4)

    out = np.empty((n_blocks, 18), dtype=np.uint8)
    out[:, :2] = d_q4.view(np.uint8).reshape(n_blocks, 2)
    out[:, 2:] = packed_bytes
    return out.tobytes()


def convert_zerotts_q8_to_q4_hybrid(
    src_path: Path,
    dst_path: Path,
) -> Path:
    """Convert a source ZeroTTS Q8_0 GGUF model into a Q4-Hybrid GGUF model.

    Preserves Audio Embeddings, Codebook Output Heads, Text Embeddings, and Normalization
    layers in native/Q8_0/FP32 precision. Quantizes Transformer Attention projections
    (Self-Attention and Cross-Attention Q, K, V, O) and FFN / MLP layers to Q4_0 for
    optimal CPU cache utilization, memory bandwidth reduction, and zero scalar fallback
    overhead on edge processors. Validates GGUF magic, versions, structure bounds,
    key-values, and tensor offsets.

    Args:
        src_path: Path to input Q8_0 GGUF model.
        dst_path: Path to target Q4-Hybrid GGUF model.

    Returns:
        Path to the generated GGUF file.

    Raises:
        FileNotFoundError: If src_path does not exist.
        ValueError: If src_path is not a valid GGUF file or contains unknown types.
    """
    if not src_path.is_file():
        raise FileNotFoundError(f"Source GGUF model file not found: {src_path}")
    file_size = src_path.stat().st_size
    if file_size < 24:
        raise ValueError(f"File too small to be a valid GGUF file ({file_size} bytes)")

    with open(src_path, "rb") as fin:
        magic, version, n_tensors, n_kv = struct.unpack("<IIQQ", fin.read(24))
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a valid GGUF file (magic={hex(magic)})")
        if version not in GGUF_SUPPORTED_VERSIONS:
            raise ValueError(f"Unsupported GGUF version: {version}; expected 2 or 3")
        if n_kv > MAX_GGUF_KV_COUNT:
            raise ValueError(f"Excessive GGUF key-value count: {n_kv}")
        if n_tensors > MAX_GGUF_TENSORS:
            raise ValueError(f"Excessive GGUF tensor count: {n_tensors}")

        alignment = GGUF_DEFAULT_ALIGNMENT
        kv_start = fin.tell()
        for _ in range(n_kv):
            if fin.tell() + 8 > file_size:
                raise ValueError("Truncated GGUF file reading KV header")
            k_len = struct.unpack("<Q", fin.read(8))[0]
            if k_len <= 0 or k_len > MAX_GGUF_STRING_LENGTH or fin.tell() + k_len > file_size:
                raise ValueError(f"Invalid GGUF key length: {k_len}")
            key_name = fin.read(k_len).decode("utf-8", errors="replace")
            if fin.tell() + 4 > file_size:
                raise ValueError("Truncated GGUF file reading KV type")
            vt = struct.unpack("<I", fin.read(4))[0]

            def skip_val(t: int) -> None:
                """Skip a GGUF KV value with bounds checking."""
                if t in {0, 1, 7}:
                    if fin.tell() + 1 > file_size:
                        raise ValueError("Truncated GGUF file reading scalar value")
                    fin.seek(1, 1)
                elif t in {2, 3}:
                    if fin.tell() + 2 > file_size:
                        raise ValueError("Truncated GGUF file reading scalar value")
                    fin.seek(2, 1)
                elif t in {4, 5, 6}:
                    if fin.tell() + 4 > file_size:
                        raise ValueError("Truncated GGUF file reading scalar value")
                    fin.seek(4, 1)
                elif t in {10, 11, 12}:
                    if fin.tell() + 8 > file_size:
                        raise ValueError("Truncated GGUF file reading scalar value")
                    fin.seek(8, 1)
                elif t == 8:
                    if fin.tell() + 8 > file_size:
                        raise ValueError("Truncated GGUF file reading string length")
                    s_len = struct.unpack("<Q", fin.read(8))[0]
                    if s_len < 0 or fin.tell() + s_len > file_size:
                        raise ValueError(f"Invalid GGUF string length: {s_len}")
                    fin.seek(s_len, 1)
                elif t == 9:
                    if fin.tell() + 12 > file_size:
                        raise ValueError("Truncated GGUF file reading array header")
                    ayt, cnt = struct.unpack("<IQ", fin.read(12))
                    if cnt < 0 or cnt > MAX_GGUF_KV_COUNT:
                        raise ValueError(f"Invalid GGUF array count: {cnt}")
                    for _ in range(cnt):
                        skip_val(ayt)
                else:
                    raise ValueError(f"Unknown GGUF KV type {t}")

            if key_name == "general.alignment":
                if vt in {4, 5}:
                    if fin.tell() + 4 > file_size:
                        raise ValueError("Truncated GGUF file reading alignment value")
                    fmt = "<I" if vt == 4 else "<i"
                    alignment_val = struct.unpack(fmt, fin.read(4))[0]
                elif vt == 10:
                    if fin.tell() + 8 > file_size:
                        raise ValueError("Truncated GGUF file reading alignment value")
                    alignment_val = struct.unpack("<Q", fin.read(8))[0]
                else:
                    raise ValueError(f"Unsupported type for general.alignment: {vt}")
                if (
                    alignment_val <= 0
                    or (alignment_val & (alignment_val - 1)) != 0
                    or alignment_val > MAX_GGUF_ALIGNMENT
                ):
                    raise ValueError(f"Invalid GGUF alignment: {alignment_val}; must be power of 2")
                alignment = alignment_val
            else:
                skip_val(vt)

        kv_end = fin.tell()
        fin.seek(kv_start)
        kv_bytes = fin.read(kv_end - kv_start)

        tensor_infos: list[dict[str, Any]] = []
        for _ in range(n_tensors):
            if fin.tell() + 8 > file_size:
                raise ValueError("Truncated GGUF file reading tensor name length")
            n_len = struct.unpack("<Q", fin.read(8))[0]
            if n_len <= 0 or n_len > MAX_GGUF_STRING_LENGTH or fin.tell() + n_len > file_size:
                raise ValueError(f"Invalid GGUF tensor name length: {n_len}")
            name = fin.read(n_len).decode("utf-8", errors="replace")
            if fin.tell() + 4 > file_size:
                raise ValueError(f"Truncated GGUF file reading dimensions for tensor {name!r}")
            n_dims = struct.unpack("<I", fin.read(4))[0]
            if n_dims <= 0 or n_dims > MAX_GGUF_DIMENSIONS:
                max_dims = MAX_GGUF_DIMENSIONS
                raise ValueError(
                    f"Invalid GGUF tensor dimension count: {n_dims}; must be 1..{max_dims}"
                )
            if fin.tell() + (n_dims * 8) + 12 > file_size:
                raise ValueError(f"Truncated GGUF file reading tensor metadata for {name!r}")
            dims = [struct.unpack("<Q", fin.read(8))[0] for _ in range(n_dims)]
            ttype, offset = struct.unpack("<IQ", fin.read(12))
            tensor_infos.append({"name": name, "dims": dims, "ttype": ttype, "old_offset": offset})

        header_end = fin.tell()
        pad = (alignment - (header_end % alignment)) % alignment
        src_data_start = header_end + pad
        for info in tensor_infos:
            n_elem = 1
            for d in info["dims"]:
                if d <= 0:
                    raise ValueError(f"Tensor {info['name']!r} has invalid dimension: {d}")
                n_elem *= d
            if n_elem <= 0:
                raise ValueError(f"Tensor {info['name']!r} has invalid element count: {n_elem}")
            info["n_elem"] = n_elem
            if (
                info["ttype"] in {GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1}
                and n_elem % 32 != 0
            ):
                raise ValueError(
                    f"Tensor {info['name']!r} elements ({n_elem}) must be a positive multiple of 32"
                )
            if info["ttype"] == GGML_TYPE_Q8_0:
                info["src_size"] = (n_elem // 32) * 34
            elif info["ttype"] == GGML_TYPE_F32:
                info["src_size"] = n_elem * 4
            elif info["ttype"] == GGML_TYPE_F16:
                info["src_size"] = n_elem * 2
            elif info["ttype"] == GGML_TYPE_Q4_0:
                info["src_size"] = (n_elem // 32) * 18
            elif info["ttype"] == GGML_TYPE_Q4_1:
                info["src_size"] = (n_elem // 32) * 20
            elif info["ttype"] == GGML_TYPE_Q5_0:
                info["src_size"] = (n_elem // 32) * 22
            else:
                raise ValueError(f"Unsupported tensor type {info['ttype']}")

            old_offset = info["old_offset"]
            src_size = info["src_size"]
            if (
                src_data_start > file_size
                or old_offset > file_size - src_data_start
                or src_size > file_size - src_data_start - old_offset
            ):
                raise ValueError(
                    f"Tensor {info['name']!r} payload extends beyond EOF "
                    f"(start={src_data_start}, offset={old_offset}, "
                    f"size={src_size}, file_size={file_size})"
                )

        new_offset = 0
        for info in tensor_infos:
            name = info["name"]
            should_q4 = (
                is_ffn_tensor(name) or is_attention_tensor(name)
            ) and not is_pinned_q8_tensor(name)
            if should_q4 and info["ttype"] == GGML_TYPE_Q8_0:
                info["new_ttype"] = GGML_TYPE_Q4_0
                info["dst_size"] = (info["n_elem"] // 32) * 18
            else:
                info["new_ttype"] = info["ttype"]
                info["dst_size"] = info["src_size"]
            info["new_offset"] = new_offset
            padded_size = info["dst_size"] + (
                (alignment - (info["dst_size"] % alignment)) % alignment
            )
            new_offset += padded_size

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_dst = dst_path.with_name(f".{dst_path.name}.{os.getpid()}.tmp")
        try:
            with open(tmp_dst, "wb") as fout:
                fout.write(struct.pack("<IIQQ", magic, version, n_tensors, n_kv))
                fout.write(kv_bytes)
                for info in tensor_infos:
                    raw_name = info["name"].encode("utf-8")
                    fout.write(struct.pack("<Q", len(raw_name)))
                    fout.write(raw_name)
                    fout.write(struct.pack("<I", len(info["dims"])))
                    for d in info["dims"]:
                        fout.write(struct.pack("<Q", d))
                    fout.write(struct.pack("<IQ", info["new_ttype"], info["new_offset"]))

                dst_header_end = fout.tell()
                dst_pad = (alignment - (dst_header_end % alignment)) % alignment
                fout.write(b"\x00" * dst_pad)

                for info in tensor_infos:
                    fin.seek(src_data_start + info["old_offset"])
                    raw_src = fin.read(info["src_size"])
                    if info["new_ttype"] == GGML_TYPE_Q4_0 and info["ttype"] == GGML_TYPE_Q8_0:
                        raw_dst = quantize_row_q8_0_to_q4_0(raw_src, info["n_elem"])
                    else:
                        raw_dst = raw_src
                    fout.write(raw_dst)
                    t_pad = (alignment - (len(raw_dst) % alignment)) % alignment
                    if t_pad > 0:
                        fout.write(b"\x00" * t_pad)

            tmp_dst.replace(dst_path)
        finally:
            tmp_dst.unlink(missing_ok=True)
    return dst_path
