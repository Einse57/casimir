#!/usr/bin/env python3
"""Create a spacing-matched synthetic whole-body-ish NIfTI for CASIMIR smoke tests."""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

# plans.json: original_median_spacing_after_transp / transpose_forward=[1,0,2]
AFTER_SPACING = np.array([3.125, 2.78, 3.125])
TRANSPOSE_BACKWARD = [1, 0, 2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-o", type=Path, default=Path("data/smoke/smoke_exam_0000.nii.gz"))
    p.add_argument("--shape-after", nargs=3, type=int, default=[64, 128, 64],
                   help="shape AFTER nnU-Net transpose_forward (z,y,x in plans space)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    S = np.array(args.shape_after)
    stored_shape = S[TRANSPOSE_BACKWARD]
    stored_spacing = AFTER_SPACING[TRANSPOSE_BACKWARD]
    rng = np.random.default_rng(args.seed)

    # MRI-like intensities near training foreground stats (mean~1667, std~1600)
    vol = rng.normal(1200, 400, size=tuple(stored_shape)).astype(np.float32)
    zz, yy, xx = np.indices(tuple(stored_shape))
    # soft tissue / organ-like blobs
    blobs = [
        ((0.35, 0.45, 0.5), 800, 900),   # liver-ish
        ((0.55, 0.55, 0.35), 600, 500),  # spleen-ish
        ((0.4, 0.5, 0.5), 500, 200),     # spine-ish elongated via anisotropy below
        ((0.3, 0.35, 0.55), 700, 400),   # heart-ish
        ((0.45, 0.6, 0.4), 650, 350),
        ((0.45, 0.6, 0.6), 650, 350),
    ]
    for (cz, cy, cx), amp, scale in blobs:
        vol += amp * np.exp(
            -(((zz - cz * stored_shape[0]) ** 2)
              + (yy - cy * stored_shape[1]) ** 2
              + (xx - cx * stored_shape[2]) ** 2) / scale
        )
    # bone-ish bright cortical shells along long axis
    vol += 300 * (np.sin(yy / 3.0) > 0.7).astype(np.float32)

    affine = np.diag([*stored_spacing.tolist(), 1.0])
    args.o.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(vol, affine), str(args.o))
    print(f"wrote {args.o} stored_shape={tuple(stored_shape)} spacing={stored_spacing.tolist()}")


if __name__ == "__main__":
    main()
