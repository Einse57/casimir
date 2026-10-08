# OpenVINO path for CASIMIR (experimental)

An optional inference path that keeps the nnU-Net v2 pipeline (preprocessing, sliding
window with mirroring, resampling and export) and replaces only the per-tile forward of
the `PlainConvUNet` with an OpenVINO model compiled for an Intel CPU, GPU or NPU. The
default PyTorch inference is unchanged, and nothing here is installed unless the
`openvino` extra is requested.

## Setup

```bash
pip install -e ".[openvino]"     # or: pip install -e . && pip install -r requirements-openvino.txt
```

The weights are downloaded on first use as usual, or read from `CASIMIR_WEIGHTS_PATH`.

## Export ONNX and OpenVINO IR

```bash
python scripts/export_openvino.py --outdir openvino_models
```

The IR keeps FP32 weights (about 132 MB); `--fp16-weights` halves the size. nnU-Net keeps
the fold weights in `list_of_parameters` and loads them only at predict time, so the export
script calls `load_state_dict` before `torch.onnx.export`; without that step the exported
network is uninitialized. The script also checks the ONNX and OpenVINO CPU outputs against
PyTorch on a random tile and writes `export_meta.json`.

## Inference

```bash
python scripts/casimir_ov_cli.py -i <nifti file or directory> -o <output directory> \
    --device CPU|GPU|NPU --no-fallback
```

`--precision auto` (the default) sets `INFERENCE_PRECISION_HINT=f32` on CPU and GPU, whose
plugin defaults can be bf16 or f16, and leaves the NPU at its native FP16. Without
`--no-fallback`, a missing GPU or NPU falls back to CPU with a message.
`--backend onnxruntime` runs the same ONNX file on ONNX Runtime's CPU provider.

## Agreement and timing scripts

```bash
python scripts/run_ref_cpu.py -i case.nii.gz -o ref_out           # default PyTorch CPU path
python scripts/dice_agreement.py --ref ref_out/case.nii.gz \
    --pred ov_cpu=ov_cpu/case.nii.gz --pred ov_npu=ov_npu/case.nii.gz -o agreement.json
python scripts/bench_casimir.py -i case.nii.gz --backend torch --runs 5
python scripts/bench_casimir.py -i case.nii.gz --backend openvino --device GPU --runs 5
```

`bench_casimir.py` runs one configuration per process. It sets the model up once, then runs
the full pipeline `--warmup + --runs` times and reports compile time, the first exam, the
statistics of the remaining exams, the network part of each exam, and the peak resident
memory of the process and its worker processes.

`tests/test_openvino_agreement.py` checks a small randomly initialized nnU-Net-style network
on OpenVINO CPU against PyTorch. It is skipped when OpenVINO is not installed.

## Tested hardware and results

Measured on 2026-10-07 with OpenVINO 2026.4.1, PyTorch 2.14.1 (CPU wheel), Python 3.12 and
Windows 11:

- Intel Core Ultra 9 285H (64 GB RAM): OpenVINO CPU, GPU (Intel Arc 140T integrated GPU)
  and NPU (Intel AI Boost).
- Intel Core Ultra 7 265F (64 GB RAM): OpenVINO CPU and NPU.

### Test case

Case `s0987` of the TotalSegmentator MRI dataset v3.0.0: a whole-body T1-weighted gradient
echo examination of a 4-year-old from University Hospital Basel (Siemens Skyra, 3 T;
256 × 208 × 400 voxels at 1.33 × 1.33 × 2.5 mm, head to feet). The dataset is released
under CC BY 4.0 (Wasserthal J, Akinci D'Antonoli T. TotalSegmentator MRI dataset, version
3.0.0. Zenodo, 2026. https://doi.org/10.5281/zenodo.22688334; Akinci D'Antonoli T, et al.
Radiology 2025;314(2):e241613). The image is not redistributed here.

The case was chosen for its license. Its acquisition differs from CASIMIR's training data,
so the numbers below compare inference backends with each other. They do not measure
segmentation accuracy.

### Agreement with the PyTorch CPU reference

Each OpenVINO output is compared with the label map from the default PyTorch CPU path on
the same host (`scripts/dice_agreement.py`). The PyTorch outputs on the two hosts are
identical voxel for voxel, and so are the two NPU outputs. "absent" means neither map
contains the structure.

| Structure | PyTorch voxels | Core Ultra 9 285H CPU f32 | Core Ultra 9 285H GPU f32 | Core Ultra 9 285H NPU fp16 | Core Ultra 7 265F CPU f32 | Core Ultra 7 265F NPU fp16 |
|---|---:|---:|---:|---:|---:|---:|
| fibulaLeft | 0 | absent | absent | absent | absent | absent |
| fibulaRight | 423 | 1.0000 | 1.0000 | 0.9976 | 1.0000 | 0.9976 |
| femurLeft | 517 | 1.0000 | 1.0000 | 0.9981 | 1.0000 | 0.9981 |
| femurRight | 195 | 1.0000 | 1.0000 | 0.9948 | 1.0000 | 0.9948 |
| tibiaLeft | 31 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| tibiaRight | 655 | 1.0000 | 1.0000 | 0.9992 | 1.0000 | 0.9992 |
| spine | 7922 | 1.0000 | 1.0000 | 0.9991 | 1.0000 | 0.9991 |
| liver | 61548 | 1.0000 | 1.0000 | 0.9995 | 1.0000 | 0.9995 |
| lungLeft | 29780 | 1.0000 | 1.0000 | 0.9997 | 1.0000 | 0.9997 |
| lungRight | 37427 | 1.0000 | 1.0000 | 0.9997 | 1.0000 | 0.9997 |
| sacrum | 866 | 1.0000 | 1.0000 | 0.9994 | 1.0000 | 0.9994 |
| spleen | 650 | 1.0000 | 1.0000 | 0.9961 | 1.0000 | 0.9961 |
| humerusLeft | 9 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| humerusRight | 0 | absent | absent | absent | absent | absent |
| heart | 8760 | 1.0000 | 1.0000 | 0.9992 | 1.0000 | 0.9992 |
| urinaryBladder | 4159 | 1.0000 | 1.0000 | 0.9989 | 1.0000 | 0.9989 |
| pelvisLeft | 1672 | 1.0000 | 1.0000 | 0.9997 | 1.0000 | 0.9997 |
| pelvisRight | 845 | 1.0000 | 1.0000 | 0.9988 | 1.0000 | 0.9988 |
| kidneyLeft | 1024 | 1.0000 | 1.0000 | 0.9976 | 1.0000 | 0.9976 |
| kidneyRight | 487 | 1.0000 | 1.0000 | 0.9959 | 1.0000 | 0.9959 |
| **Mean Dice (18 present)** | | 1.00000 | 1.00000 | 0.99852 | 1.00000 | 0.99852 |
| **Min Dice** | | 0.99998 | 1.00000 | 0.99485 | 0.99998 | 0.99485 |
| **Voxels differing** | | 1 | 0 | 169 | 2 | 169 |
| **Voxel agreement** | | 1.000000 | 1.000000 | 0.999992 | 1.000000 | 0.999992 |
| **Foreground voxel agreement** | | 0.99999 | 1.00000 | 0.99892 | 0.99999 | 0.99892 |

Precision: CPU and GPU ran with `INFERENCE_PRECISION_HINT=f32`; the NPU reports FP16
inference precision. On both CPUs the plugin default precision was already f32, and the
output was the same as with the explicit hint.

### Per-exam time

Full pipeline per exam, one process per row, 1 warm-up exam plus 5 timed exams, run from
the logged-in desktop session. "Compile" is `compile_model` for the IR in that process;
"cold" is the NPU compile with the driver cache bypassed (`NPU_BYPASS_UMD_CACHING`), the
minimum of 3 fresh processes. "Network part" is the sliding-window inference inside each
exam; the rest is preprocessing, resampling and NIfTI export. CASIMIR's default PyTorch CPU
path runs the network with nnU-Net's default of 8 PyTorch threads (`nnUNet_def_n_proc`);
the second PyTorch row raises that to the number of logical CPUs.

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

Peak RSS covers the main process and the nnU-Net preprocessing and export worker processes.
With the default LATENCY hint, OpenVINO CPU used 14 inference threads on the Core Ultra 9 285H
and 8 on the Core Ultra 7 265F (`INFERENCE_NUM_THREADS` of the compiled model).
Per-run values, device properties and the compile probe are in `data/openvino_results_s0987.json`.
