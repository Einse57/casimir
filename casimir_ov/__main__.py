"""python -m casimir_ov: CASIMIR inference on an Intel CPU, GPU or NPU with OpenVINO."""

import argparse
import importlib.util
import sys

from casimir import __version__


def build_parser():
    p = argparse.ArgumentParser(
        prog="python -m casimir_ov",
        description="Segment whole-body T1-weighted MRI with CASIMIR, the network run "
                    "by OpenVINO.")
    p.add_argument("-i", "--input", required=True, help="NIfTI file or directory")
    p.add_argument("-o", "--outdir", default="segmentations", help="output directory")
    p.add_argument("--device", default="CPU", type=str.upper, choices=["CPU", "GPU", "NPU"],
                   help="OpenVINO device")
    p.add_argument("--model-dir", help="local weights directory instead of downloading")
    p.add_argument("--ir", help="OpenVINO IR (.xml) instead of exporting one on first use")
    p.add_argument("--precision", default="auto",
                   help="INFERENCE_PRECISION_HINT: auto (f32 on CPU and GPU, FP16 on NPU), "
                        "default (plugin default), f32, f16 or bf16")
    p.add_argument("--no-fallback", action="store_true",
                   help="fail instead of falling back to CPU when the device is missing")
    p.add_argument("--step-size", type=float, default=0.5, help="sliding window step")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--version", action="version", version=f"casimir {__version__}")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if importlib.util.find_spec("openvino") is None:
        raise SystemExit('OpenVINO is not installed, install it with '
                         'pip install "casimir[openvino]"')
    from casimir_ov.inference import infer

    written = infer(args.input, args.outdir, device=args.device, model_dir=args.model_dir,
                    ir=args.ir, step_size=args.step_size, precision=args.precision,
                    allow_fallback=not args.no_fallback, verbose=args.verbose)
    print(f"wrote {len(written)} segmentation(s) to {args.outdir}")


if __name__ == "__main__":
    sys.exit(main())
