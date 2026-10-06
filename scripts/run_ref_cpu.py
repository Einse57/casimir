#!/usr/bin/env python3
"""Reference CPU inference smoke for CASIMIR (box-only)."""
import json
import resource
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from casimir import inference

ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / "data/smoke/smoke_exam_0000.nii.gz"
OUT = ROOT / "data/ref_out"
WEIGHTS = ROOT / "weights_local"


def main():
    t0 = time.perf_counter()
    outs = inference.infer(
        str(INP),
        str(OUT),
        device="cpu",
        model_dir=str(WEIGHTS),
        verbose=True,
    )
    elapsed = time.perf_counter() - t0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    img = nib.load(str(outs[0]))
    data = np.asanyarray(img.dataobj)
    counts = {int(u): int((data == u).sum()) for u in np.unique(data)}
    stats = {
        "elapsed_sec": elapsed,
        "peak_rss_kb": peak,
        "shape": list(data.shape),
        "unique_labels": [int(x) for x in np.unique(data)],
        "label_counts": counts,
        "nonzero": int(np.count_nonzero(data)),
        "output": str(outs[0]),
    }
    (ROOT / "data/ref_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
