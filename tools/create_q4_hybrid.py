"""Tool to create ZeroTTS Q4-Hybrid (Mixed Precision) GGUF models.

This tool implements the 'Q4-Hybrid' mixed-precision quantization plan
for ZeroTTS models targeting edge CPU processors (such as Intel Celeron N5105
with 4MB L3 cache and Raspberry Pi 4).

Architecture Strategy:
- Audio Embeddings & Codebook Heads: Keep Q8_0 (preserves acoustic discrete token fidelity).
- Attention Layers (Self-Attention, Cross-Attention): Keep Q8_0 (prevents phonetic token drift).
- Text Encoder: Keep Q8_0 or FP16 (lightweight, runs once per utterance).
- Transformer FFN (Feed-Forward Networks): Quantize to Q4_0 (reduces working memory footprint
  allowing the active local step to fit inside CPU cache and avoiding DRAM bus saturation).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __name__ == "__main__" and not __package__:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from wyoming_vietnamese.gguf_converter import (
    GGML_TYPE_Q4_1,
    GGML_TYPE_Q5_0,
    convert_zerotts_q8_to_q4_hybrid,
)


def print_q4_hybrid_summary(
    input_path: Path, output_path: Path, attention_quant: str = "q8"
) -> None:
    """Print architectural summary and validation info for the Q4-Hybrid model.

    Args:
        input_path: Source GGUF input file path.
        output_path: Target hybrid GGUF output file path.
        attention_quant: Selected quantization format for attention layers.
    """
    attn_label = {
        "q8": "[Q8_0 / Native]",
        "q5_0": "[Q5_0         ]",
        "q4_1": "[Q4_1         ]",
    }.get(attention_quant, "[Q8_0 / Native]")

    print("=" * 64)
    print("ZeroTTS Q4-Hybrid (Mixed Precision) Architecture Summary")
    print("=" * 64)
    print(f"Source Model : {input_path}")
    print(f"Target Hybrid: {output_path}")
    print("\nTensor Quantization Mapping:")
    print("  [Q8_0 / Native] Audio Embedding & Prediction Codebook Heads")
    print(f"  {attn_label} Self-Attention & Cross-Attention (Q, K, V, O)")
    print("  [Q8_0 / Native] Text Encoder & Layer Norms")
    print("  [Q4_0         ] Transformer Feed-Forward Networks (FFN / MLP)")
    print("\nBenefits:")
    print("  - Fits Local Transformer in 4MB L3 Cache (N5105 / J4125 / Pi 4)")
    print("  - Eliminates Memory Bus Saturation (RTF < 1.0 Real-time)")
    print("  - Prevents Audio Codebook Drift & Metallic Distortion")
    print("=" * 64)


def main() -> None:
    """CLI entry point for ZeroTTS Q4-Hybrid conversion."""
    parser = argparse.ArgumentParser(
        description="Convert ZeroTTS GGUF to Q4-Hybrid (Mixed Precision)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to source GGUF file (e.g. zerotts-q8_0.gguf)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to target hybrid GGUF file (default: <parent>/zerotts-q4-hybrid.gguf)",
    )
    parser.add_argument(
        "--attention-quant",
        choices=["q5_0", "q8", "q4_1"],
        default="q5_0",
        help=(
            "Quantization precision for Attention layers (Self-Attention & Cross-Attention): "
            "'q5_0' (default: Q5_0 mixed precision), 'q8' (keep native Q8_0), "
            "or 'q4_1'."
        ),
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = (
        args.output.resolve() if args.output else input_path.parent / "zerotts-q4-hybrid.gguf"
    )

    print_q4_hybrid_summary(input_path, output_path, attention_quant=args.attention_quant)
    if not input_path.is_file():
        print(f"Error: Source file {input_path} not found.")
        sys.exit(1)

    attn_map = {"q8": None, "q5_0": GGML_TYPE_Q5_0, "q4_1": GGML_TYPE_Q4_1}
    attention_type = attn_map[args.attention_quant]

    print(f"Converting {input_path} -> {output_path} (attention_quant={args.attention_quant})...")
    convert_zerotts_q8_to_q4_hybrid(input_path, output_path, attention_type=attention_type)
    print(f"Done! Hybrid model saved ({output_path.stat().st_size:,} bytes).")


if __name__ == "__main__":
    main()
