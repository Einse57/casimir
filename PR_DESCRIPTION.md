## Optional OpenVINO inference path for CASIMIR (draft, fork only)

Adds an optional path that keeps the nnU-Net v2 pipeline and runs CASIMIR's 2D
`PlainConvUNet` with OpenVINO on an Intel CPU, GPU or NPU. The default PyTorch inference
and its dependencies are unchanged; the new dependencies are in the `openvino` extra.

### Licenses
- Code (`fjhorvath/casimir`) and weights (`huggingface.co/fjhorvath/casimir`, revision
  `d24ca411…`, tag v1.0.0): Apache-2.0.
- Test case: TotalSegmentator MRI dataset v3.0.0, case `s0987`, CC BY 4.0
  (doi:10.5281/zenodo.22688334). Not redistributed.

### Changes
- `casimir_ov/`: OpenVINO and ONNX Runtime drop-in networks; only the tile forward is
  swapped. `INFERENCE_PRECISION_HINT=f32` on CPU/GPU by default, FP16 on NPU; optional
  strict device selection (no CPU fallback).
- `scripts/export_openvino.py`: ONNX + IR export with FP32 weights; loads
  `list_of_parameters[0]` before export.
- `scripts/casimir_ov_cli.py`: `--device CPU|GPU|NPU`, `--precision`, `--no-fallback`.
- `scripts/bench_casimir.py`: per-exam timing (compile, first exam, mean/p50/min, network
  part, peak RSS of the process tree), cross-platform.
- `scripts/dice_agreement.py`, `scripts/run_ref_cpu.py`: agreement against the PyTorch
  CPU reference.
- `tests/test_openvino_agreement.py`: OpenVINO vs PyTorch on a small synthetic network;
  skipped when OpenVINO is not installed.
- `pyproject.toml`: `openvino` optional extra (`requirements-openvino.txt` mirrors it).
- `OPENVINO.md`: usage, tested hardware and results. README links to it.
- `data/openvino_results_s0987.json`: measured values behind the tables; `data/export_meta.json`
  from the FP32 export. The earlier synthetic smoke-run JSONs are removed.

### Tested hardware
Windows 11, OpenVINO 2026.4.1, PyTorch 2.14.1 CPU wheel, Python 3.12:
- Intel Core Ultra 9 285H: OpenVINO CPU, GPU (Arc 140T integrated GPU), NPU (AI Boost).
- Intel Core Ultra 7 265F: OpenVINO CPU, NPU.

### Agreement with the PyTorch CPU reference (whole-body case s0987)
| Host and device | Precision | Voxels differing (of 21.3 M) | Mean Dice, 18 present structures | Min Dice |
|---|---|---:|---:|---:|
| Core Ultra 9 285H CPU | f32 | 1 | 1.00000 | 0.99998 |
| Core Ultra 9 285H GPU | f32 | 0 | 1.00000 | 1.00000 |
| Core Ultra 9 285H NPU | fp16 | 169 | 0.99852 | 0.99485 |
| Core Ultra 7 265F CPU | f32 | 2 | 1.00000 | 0.99998 |
| Core Ultra 7 265F NPU | fp16 | 169 | 0.99852 | 0.99485 |

Per-structure Dice is in `OPENVINO.md`. The test case differs from CASIMIR's training data,
so these numbers compare backends only and do not measure segmentation accuracy.

### Per-exam time (full pipeline, 1 warm-up + 5 timed exams, desktop session)
| Host | Path | Precision | Compile (s) | First exam (s) | Mean (s) | p50 (s) | Min (s) | Network part, mean (s) | Peak RSS (GiB) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Core Ultra 9 285H | PyTorch CPU (CASIMIR default) | fp32 | – | 113.6 | 115.0 | 115.1 | 114.1 | 92.2 | 8.0 |
| Core Ultra 9 285H | PyTorch CPU, nnUNet_def_n_proc=all logical CPUs | fp32 | – | 123.0 | 122.7 | 120.4 | 118.6 | 98.8 | 8.1 |
| Core Ultra 9 285H | OpenVINO CPU | f32 | 0.26 | 76.7 | 76.9 | 76.9 | 76.5 | 54.6 | 8.4 |
| Core Ultra 9 285H | OpenVINO GPU (Intel Arc 140T iGPU) | f32 | 0.35 | 48.5 | 48.9 | 48.4 | 48.2 | 26.0 | 8.1 |
| Core Ultra 9 285H | OpenVINO NPU (Intel AI Boost) | fp16 | 0.20 (cold 1.94) | 55.9 | 58.2 | 57.6 | 55.7 | 34.0 | 8.7 |
| Core Ultra 7 265F | PyTorch CPU (CASIMIR default) | fp32 | – | 89.3 | 94.8 | 95.1 | 93.6 | 75.1 | 8.2 |
| Core Ultra 7 265F | PyTorch CPU, nnUNet_def_n_proc=all logical CPUs | fp32 | – | 74.7 | 75.3 | 75.3 | 75.0 | 55.4 | 8.3 |
| Core Ultra 7 265F | OpenVINO CPU | f32 | 0.14 | 84.9 | 85.4 | 85.4 | 84.8 | 65.4 | 8.5 |
| Core Ultra 7 265F | OpenVINO NPU (Intel AI Boost) | fp16 | 0.17 (cold 1.84) | 49.6 | 48.7 | 48.7 | 48.6 | 28.7 | 8.1 |

### Tests
- `pytest tests` with the `openvino` extra (Linux, Python 3.13, OpenVINO 2026.4.1): 8 passed.
- `pytest tests` without OpenVINO: 6 passed, 1 skipped (`test_openvino_agreement.py`).
- `ruff check` (ruff 0.16 defaults; the project has no lint config): clean.

### Not in this PR / notes
- `PR_DESCRIPTION.md` is a fork-only working file and is not meant for upstream.
