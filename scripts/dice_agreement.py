#!/usr/bin/env python3
"""Per-structure Dice and voxel agreement of one or more label maps vs a reference.

Usage:
  python scripts/dice_agreement.py --ref ref/case.nii.gz \
      --pred ov_cpu=ov_cpu/case.nii.gz --pred ov_npu=ov_npu/case.nii.gz -o agreement.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

LABELS = json.loads(
    (Path(__file__).resolve().parents[1] / "casimir" / "labels.json").read_text()
)["labels"]


def dice(a, b):
    sa, sb = np.count_nonzero(a), np.count_nonzero(b)
    if sa == 0 and sb == 0:
        return None  # structure absent from both
    return 2.0 * np.count_nonzero(a & b) / (sa + sb)


def compare(ref, pred):
    if ref.shape != pred.shape:
        raise SystemExit(f"shape mismatch {ref.shape} vs {pred.shape}")
    per = {}
    for name, lab in LABELS.items():
        if lab == 0:
            continue
        r, q = ref == lab, pred == lab
        per[name] = {
            "label": lab,
            "dice": dice(r, q),
            "ref_voxels": int(np.count_nonzero(r)),
            "pred_voxels": int(np.count_nonzero(q)),
        }
    present = [v["dice"] for v in per.values() if v["dice"] is not None]
    fg = (ref > 0) | (pred > 0)
    return {
        "voxel_agree": float((ref == pred).mean()),
        "voxels_differing": int(np.count_nonzero(ref != pred)),
        "foreground_voxel_agree": float((ref[fg] == pred[fg]).mean()) if fg.any() else None,
        "mean_dice_present": float(np.mean(present)) if present else None,
        "min_dice_present": float(np.min(present)) if present else None,
        "n_present": len(present),
        "per_label": per,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", type=Path, required=True)
    p.add_argument("--pred", action="append", required=True, metavar="NAME=PATH")
    p.add_argument("-o", "--json", type=Path, default=None)
    args = p.parse_args(argv)

    ref = np.asanyarray(nib.load(str(args.ref)).dataobj)
    out = {"ref": args.ref.name, "shape": list(ref.shape), "results": {}}
    for item in args.pred:
        name, _, path = item.partition("=")
        if not path:
            name, path = Path(item).parent.name, item
        pred = np.asanyarray(nib.load(path).dataobj)
        res = compare(ref, pred)
        out["results"][name] = res
        print(f"{name}: voxel_agree={res['voxel_agree']:.6f} "
              f"differing={res['voxels_differing']} "
              f"mean_dice={res['mean_dice_present']:.5f} min_dice={res['min_dice_present']:.5f}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
