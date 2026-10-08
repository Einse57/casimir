#!/usr/bin/env python3
"""Per-exam latency for CASIMIR: default PyTorch CPU path vs OpenVINO on one device.

One backend/device per invocation, so peak memory is per configuration. The model is
set up once (weights load, or OpenVINO compile), then the full nnU-Net pipeline
(preprocessing, sliding window with mirroring, resampling and NIfTI export) runs
``--warmup + --runs`` times on the same input. Reported separately:

* setup_s / compile_s: predictor setup, and OpenVINO compile_model alone
* first_run_s: the first exam after setup (warm-up, not in the statistics)
* runs: mean / p50 / min / max of the remaining exams
* sliding_window_s: the network part of each exam (nnU-Net predict_logits_*)
* peak_rss_mb: peak resident memory of the process tree, sampled every 100 ms

Usage:
  python scripts/bench_casimir.py -i case.nii.gz --backend torch --runs 5
  python scripts/bench_casimir.py -i case.nii.gz --backend openvino --device GPU --runs 5
"""
from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing
import os
import platform
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class PeakRSS:
    """Samples resident memory of this process plus its children."""

    def __init__(self, interval=0.1):
        import psutil

        self._proc = psutil.Process()
        self._interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self):
        import psutil

        total = 0
        with contextlib.suppress(psutil.Error):
            total = self._proc.memory_info().rss
            for child in self._proc.children(recursive=True):
                with contextlib.suppress(psutil.Error):
                    total += child.memory_info().rss
        self.peak = max(self.peak, total)

    def _run(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self._interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self._sample()


def cpu_name():
    if sys.platform == "win32":
        import winreg

        with contextlib.suppress(OSError):
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    if sys.platform.startswith("linux"):
        with contextlib.suppress(OSError):
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def build_torch_predictor(weights, step_size):
    """Same settings as casimir.inference.infer(device="cpu")."""
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    from casimir import config

    torch.set_num_threads(multiprocessing.cpu_count())
    predictor = nnUNetPredictor(
        tile_step_size=step_size,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device("cpu"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(weights), ("all",), checkpoint_name=config.CHECKPOINT
    )
    return predictor


def summarize(times):
    if not times:
        return {"n": 0}
    return {
        "n": len(times),
        "mean_s": round(statistics.mean(times), 3),
        "p50_s": round(statistics.median(times), 3),
        "min_s": round(min(times), 3),
        "max_s": round(max(times), 3),
        "stdev_s": round(statistics.stdev(times), 3) if len(times) > 1 else 0.0,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", type=Path, required=True, help="one NIfTI volume")
    p.add_argument("--weights", type=Path, default=None,
                   help="weights directory (default: CASIMIR_WEIGHTS_PATH or HF download)")
    p.add_argument("--backend", choices=["torch", "openvino"], default="openvino")
    p.add_argument("--device", default="CPU", type=str.upper, help="OpenVINO device")
    p.add_argument("--precision", default="auto",
                   help="OpenVINO INFERENCE_PRECISION_HINT (auto = f32 on CPU/GPU)")
    p.add_argument("--ov-xml", type=Path, default=ROOT / "openvino_models/casimir_2d.xml")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--step-size", type=float, default=0.5)
    p.add_argument("--outdir", type=Path, default=Path("bench_out"),
                   help="segmentations are written to <outdir>/<config>/run_<i>/")
    p.add_argument("--tag", default=None, help="label for this configuration")
    p.add_argument("-o", "--json", type=Path, default=None, help="write results JSON here")
    args = p.parse_args()

    import torch

    from casimir import config
    from casimir_ov.infer_ov import build_predictor, run_predictor

    weights = args.weights or config.ensure_weights()
    tag = args.tag or ("torch_cpu" if args.backend == "torch" else f"openvino_{args.device}")

    with PeakRSS() as mem:
        t0 = time.perf_counter()
        if args.backend == "torch":
            predictor = build_torch_predictor(weights, args.step_size)
            compile_s = None
            device_info = {"device": "cpu", "torch_threads": torch.get_num_threads()}
        else:
            predictor = build_predictor(
                device=args.device,
                model_dir=weights,
                ov_model_xml=args.ov_xml,
                step_size=args.step_size,
                precision=args.precision,
                allow_fallback=False,
            )
            compile_s = predictor.network.compile_s
            device_info = predictor.network.describe()
        setup_s = time.perf_counter() - t0

        # time the network part (sliding window + mirroring) of every exam
        sw_times = []
        inner = predictor.predict_logits_from_preprocessed_data

        def timed_predict(data):
            s = time.perf_counter()
            out = inner(data)
            sw_times.append(time.perf_counter() - s)
            return out

        predictor.predict_logits_from_preprocessed_data = timed_predict

        times = []
        for i in range(args.warmup + args.runs):
            s = time.perf_counter()
            run_predictor(predictor, args.input, args.outdir / tag / f"run_{i}")
            dt = time.perf_counter() - s
            times.append(dt)
            print(f"[{tag}] run {i} {'warmup' if i < args.warmup else 'timed'} {dt:.2f}s "
                  f"(sliding window {sw_times[-1]:.2f}s)", flush=True)

    try:
        import openvino as ov

        ov_version = ov.__version__
    except ImportError:
        ov_version = None

    result = {
        "config": tag,
        "backend": args.backend,
        "device_info": device_info,
        "input": args.input.name,
        "host": {
            "cpu": cpu_name(),
            "logical_cpus": os.cpu_count(),
            "os": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "openvino": ov_version,
            "nnUNet_def_n_proc": os.environ.get("nnUNet_def_n_proc"),
        },
        "setup_s": round(setup_s, 3),
        "compile_s": round(compile_s, 3) if compile_s is not None else None,
        "first_run_s": round(times[0], 3) if args.warmup else None,
        "runs": summarize(times[args.warmup:]),
        "sliding_window": summarize(sw_times[args.warmup:]),
        "all_runs_s": [round(t, 3) for t in times],
        "peak_rss_mb": round(mem.peak / 2**20, 1),
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text)


if __name__ == "__main__":
    main()
