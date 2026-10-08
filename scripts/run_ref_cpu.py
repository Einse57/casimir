#!/usr/bin/env python3
"""Reference segmentation with the default CASIMIR PyTorch CPU path."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from casimir import inference


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input", type=Path, required=True)
    p.add_argument("-o", "--outdir", type=Path, default=Path("ref_out"))
    p.add_argument("--model-dir", default=None)
    args = p.parse_args(argv)

    t0 = time.perf_counter()
    outs = inference.infer(str(args.input), str(args.outdir), device="cpu",
                           model_dir=args.model_dir)
    elapsed = time.perf_counter() - t0
    data = np.asanyarray(nib.load(str(outs[0])).dataobj)
    labels, counts = np.unique(data, return_counts=True)
    print(json.dumps({
        "input": args.input.name,
        "elapsed_s": round(elapsed, 2),
        "shape": list(data.shape),
        "label_voxels": {int(k): int(v) for k, v in zip(labels, counts)},
    }, indent=2))


if __name__ == "__main__":
    main()
