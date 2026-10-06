#!/usr/bin/env python3
"""One-command CASIMIR latency bench (box-only / non-target unless noted).

Prints per-exam latency, mean/p50/p90, peak RSS for torch CPU and OpenVINO.
Published baseline (README): ~4 min/exam on Apple M2 Pro CPU; ~1.5 min on RTX 4090 / M2 Pro GPU.

Usage:
  PYTHONPATH=. python scripts/bench_casimir.py --backend both --device CPU --runs 2
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def peak_rss_kb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def run_torch(inp, outdir, weights, step_size):
    from casimir import inference

    t0 = time.perf_counter()
    outs = inference.infer(
        str(inp), str(outdir), device="cpu", model_dir=str(weights), step_size=step_size
    )
    return time.perf_counter() - t0, outs


def run_ov(inp, outdir, weights, ov_xml, ov_device, step_size, backend="openvino"):
    from casimir_ov.infer_ov import infer_ov

    t0 = time.perf_counter()
    outs = infer_ov(
        str(inp),
        str(outdir),
        device=ov_device,
        model_dir=str(weights),
        ov_model_xml=str(ov_xml),
        backend=backend,
        step_size=step_size,
    )
    return time.perf_counter() - t0, outs


def summarize(times):
    times = sorted(times)

    def pct(p):
        if not times:
            return None
        k = (len(times) - 1) * p / 100.0
        f = int(k)
        c = min(f + 1, len(times) - 1)
        return times[f] + (times[c] - times[f]) * (k - f)

    return {
        "n": len(times),
        "mean_sec": statistics.mean(times) if times else None,
        "p50_sec": pct(50),
        "p90_sec": pct(90),
        "min_sec": times[0] if times else None,
        "max_sec": times[-1] if times else None,
        "times_sec": times,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input", type=Path, default=ROOT / "data/smoke/smoke_exam_0000.nii.gz")
    p.add_argument("--weights", type=Path, default=ROOT / "weights_local")
    p.add_argument("--ov-xml", type=Path, default=ROOT / "openvino_models/casimir_2d.xml")
    p.add_argument(
        "--backend",
        choices=["torch", "openvino", "onnxruntime", "both"],
        default="both",
        help="'both' = torch + openvino",
    )
    p.add_argument("--device", default="CPU", help="OpenVINO device: CPU|GPU|NPU")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--step-size", type=float, default=0.5)
    p.add_argument("-o", type=Path, default=ROOT / "data/bench_results.json")
    args = p.parse_args()

    os.environ.setdefault("CASIMIR_WEIGHTS_PATH", str(args.weights))
    results = {
        "label": "box-only / non-target (not Core/Xeon iron)",
        "published_baseline": {
            "m2_pro_cpu_sec": 240,
            "m2_pro_mps_or_rtx4090_sec": 90,
            "note": "CASIMIR README / HF model card whole-body exam latency",
        },
        "input": str(args.input),
        "backend_runs": {},
    }

    backends = []
    if args.backend == "both":
        backends = ["torch", "openvino"]
    else:
        backends = [args.backend]

    for be in backends:
        out_root = ROOT / "data" / f"bench_{be}"
        times = []
        outs = None
        for i in range(args.warmup + args.runs):
            outdir = out_root / f"run_{i}"
            if be == "torch":
                dt, outs = run_torch(args.input, outdir, args.weights, args.step_size)
                tag_dev = "cpu"
            else:
                dt, outs = run_ov(
                    args.input,
                    outdir,
                    args.weights,
                    args.ov_xml,
                    args.device,
                    args.step_size,
                    backend=be,
                )
                tag_dev = args.device if be == "openvino" else "cpu"
            tag = "warmup" if i < args.warmup else "timed"
            print(f"[{be}/{tag_dev}] {tag} run={i} {dt:.3f}s")
            if i >= args.warmup:
                times.append(dt)
        results["backend_runs"][be] = {
            **summarize(times),
            "peak_rss_kb": peak_rss_kb(),
            "device": tag_dev,
            "last_output": str(outs[0]) if outs else None,
        }

    args.o.parent.mkdir(parents=True, exist_ok=True)
    args.o.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"wrote {args.o}")


if __name__ == "__main__":
    main()
