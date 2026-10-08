#!/usr/bin/env python3
"""Export the CASIMIR nnU-Net fold_all network to ONNX + OpenVINO IR."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from casimir_ov.ov_network import export_casimir_onnx_and_ir


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, default=None,
                   help="weights directory (default: CASIMIR_WEIGHTS_PATH or HF download)")
    p.add_argument("--outdir", type=Path, default=ROOT / "openvino_models")
    p.add_argument("--fp16-weights", action="store_true",
                   help="store IR weights as FP16 (half the size; CPU/GPU f32 results drift)")
    args = p.parse_args(argv)
    if args.weights is None:
        from casimir import config

        args.weights = config.ensure_weights()
    meta = export_casimir_onnx_and_ir(args.weights, args.outdir,
                                      compress_to_fp16=args.fp16_weights)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
