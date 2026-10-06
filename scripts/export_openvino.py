#!/usr/bin/env python3
"""Export CASIMIR nnU-Net fold_all network to ONNX + OpenVINO IR."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from casimir_ov.ov_network import export_casimir_onnx_and_ir


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, default=ROOT / "weights_local")
    p.add_argument("--outdir", type=Path, default=ROOT / "openvino_models")
    args = p.parse_args(argv)
    try:
        meta = export_casimir_onnx_and_ir(args.weights, args.outdir)
        print(meta)
    except Exception as e:
        import json
        args.outdir.mkdir(parents=True, exist_ok=True)
        (args.outdir / "export_failure.json").write_text(
            json.dumps({"status": "failed", "error": repr(e)}, indent=2)
        )
        print("EXPORT FAILED:", e, file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
