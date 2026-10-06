#!/usr/bin/env python3
"""CLI: casimir OpenVINO/ONNX Runtime inference with --device CPU|GPU|NPU."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from casimir_ov.infer_ov import infer_ov


def main(argv=None):
    p = argparse.ArgumentParser(description="CASIMIR OpenVINO / ONNX Runtime inference")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--outdir", default="segmentations_ov")
    p.add_argument("--device", default="CPU", choices=["CPU", "GPU", "NPU", "cpu", "gpu", "npu"])
    p.add_argument("--backend", default="openvino", choices=["openvino", "onnxruntime", "ort"])
    p.add_argument("--model-dir", default=None)
    p.add_argument("--ov-xml", default=str(ROOT / "openvino_models/casimir_2d.xml"))
    p.add_argument("--onnx", default=str(ROOT / "openvino_models/casimir_2d.onnx"))
    p.add_argument("--step-size", type=float, default=0.5)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    written = infer_ov(
        args.input,
        args.outdir,
        device=args.device.upper(),
        model_dir=args.model_dir,
        ov_model_xml=args.ov_xml,
        onnx_path=args.onnx,
        backend=args.backend,
        step_size=args.step_size,
        verbose=args.verbose,
    )
    print(f"wrote {len(written)} segmentation(s) to {args.outdir}")


if __name__ == "__main__":
    sys.exit(main())
