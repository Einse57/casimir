#!/usr/bin/env python3
"""Per-label Dice between reference torch CPU mask and OpenVINO mask."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

LABELS = {
    0: "background",
    1: "fibulaLeft", 2: "fibulaRight",
    3: "femurLeft", 4: "femurRight",
    5: "tibiaLeft", 6: "tibiaRight",
    7: "spine", 8: "liver",
    9: "lungLeft", 10: "lungRight",
    11: "sacrum", 12: "spleen",
    13: "humerusLeft", 14: "humerusRight",
    15: "heart", 16: "urinaryBladder",
    17: "pelvisLeft", 18: "pelvisRight",
    19: "kidneyLeft", 20: "kidneyRight",
}


def dice(a, b):
    inter = np.count_nonzero((a) & (b))
    sa, sb = np.count_nonzero(a), np.count_nonzero(b)
    if sa == 0 and sb == 0:
        return 1.0
    if sa == 0 or sb == 0:
        return 0.0
    return 2.0 * inter / (sa + sb)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", type=Path, required=True)
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("-o", type=Path, default=Path("data/dice_agreement.json"))
    args = p.parse_args()

    ref = np.asanyarray(nib.load(str(args.ref)).dataobj)
    pred = np.asanyarray(nib.load(str(args.pred)).dataobj)
    if ref.shape != pred.shape:
        raise SystemExit(f"shape mismatch {ref.shape} vs {pred.shape}")

    per = {}
    for lab, name in LABELS.items():
        d = dice(ref == lab, pred == lab)
        per[name] = {
            "label": lab,
            "dice": d,
            "ref_voxels": int(np.count_nonzero(ref == lab)),
            "pred_voxels": int(np.count_nonzero(pred == lab)),
        }

    foreground = [v["dice"] for k, v in per.items() if k != "background"]
    present = [
        v["dice"]
        for k, v in per.items()
        if k != "background" and (v["ref_voxels"] + v["pred_voxels"]) > 0
    ]
    voxel_agree = float((ref == pred).mean())
    summary = {
        "voxel_agree": voxel_agree,
        "mean_dice_all_structures": float(np.mean(foreground)),
        "mean_dice_present_structures": float(np.mean(present)) if present else None,
        "n_present_structures": len(present),
        "ref_unique": [int(x) for x in np.unique(ref)],
        "pred_unique": [int(x) for x in np.unique(pred)],
        "per_label": per,
    }
    args.o.parent.mkdir(parents=True, exist_ok=True)
    args.o.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in summary if k != "per_label"}, indent=2))
    print(f"wrote {args.o}")


if __name__ == "__main__":
    main()
