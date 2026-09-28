import argparse
import sys

from casimir import __version__, inference


def build_parser():
    p = argparse.ArgumentParser(
        prog="casimir",
        description="Segment six visceral organs and seven skeletal structures on "
                    "whole-body T1-weighted MRI.")
    p.add_argument("-i", "--input", required=True, help="NIfTI file or directory")
    p.add_argument("-o", "--outdir", default="segmentations", help="output directory")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    p.add_argument("--model-dir", help="local weights directory instead of downloading")
    p.add_argument("--step-size", type=float, default=0.5, help="sliding window step")
    p.add_argument("--non-deterministic", action="store_true",
                   help="leave cuDNN autotuning on")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--version", action="version", version=f"casimir {__version__}")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    written = inference.infer(args.input, args.outdir, device=args.device,
                              model_dir=args.model_dir, step_size=args.step_size,
                              deterministic=not args.non_deterministic,
                              verbose=args.verbose)
    print(f"wrote {len(written)} segmentation(s) to {args.outdir}")


if __name__ == "__main__":
    sys.exit(main())
