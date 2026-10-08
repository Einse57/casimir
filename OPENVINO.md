# OpenVINO inference (optional)

An optional inference path for Intel CPUs, GPUs and NPUs. It keeps the nnU-Net v2
pipeline of the default inference (preprocessing, sliding window with mirroring,
resampling and export) and runs only the 2D `PlainConvUNet` with
[OpenVINO](https://github.com/openvinotoolkit/openvino). The default `casimir` command,
its dependencies and its output are unchanged; nothing here is installed unless the
`openvino` extra is requested.

## Installation

From a clone of this repository:

```bash
pip install -e ".[openvino]"
```

## Inference

```bash
python -m casimir_ov -i <nifti file or directory> -o <output directory> --device CPU
```

`--device` is `CPU`, `GPU` or `NPU`. Inputs, outputs, `--model-dir` and
`CASIMIR_WEIGHTS_PATH` work as for `casimir`. `python -m casimir_ov --help` lists all
flags.

On first use the network is exported to OpenVINO IR (FP32 weights, about 132 MB) under
`~/.cache/casimir/openvino`, one subdirectory per checkpoint; set
`CASIMIR_OPENVINO_CACHE` to use another directory. To export by hand, run
`python -m casimir_ov.export --outdir <directory>` and pass the `.xml` with `--ir`.

`--precision auto`, the default, sets `INFERENCE_PRECISION_HINT=f32` on CPU and GPU,
whose plugin defaults can be bf16 or f16, and leaves the NPU at its native FP16. If the
requested device is missing, inference falls back to CPU with a message; `--no-fallback`
makes it an error instead.

From Python:

```python
from casimir_ov import inference

inference.infer("image_dir", "outdir", device="GPU")
```

## Checking agreement and timing

```bash
casimir -i case.nii.gz -o ref --device cpu
python -m casimir_ov -i case.nii.gz -o ov_npu --device NPU --no-fallback
python scripts/dice_agreement.py --ref ref/case.nii.gz --pred ov_npu=ov_npu/case.nii.gz
python scripts/bench_casimir.py -i case.nii.gz --backend torch --runs 5
python scripts/bench_casimir.py -i case.nii.gz --backend openvino --device GPU --runs 5
```

`dice_agreement.py` reports per-structure Dice and voxel agreement against a reference
label map. `bench_casimir.py` runs one configuration per process: it sets the model up
once, runs the full pipeline `--warmup` + `--runs` times on one volume, and reports the
compile time, the first exam, statistics of the remaining exams, the sliding-window part
of each exam and the peak resident memory of the process and its workers.

`tests/test_openvino_agreement.py` compares OpenVINO CPU with PyTorch on a small randomly
initialized nnU-Net network. It needs no download and is skipped when OpenVINO is not
installed.

## Tested hardware

Measured on 2026-10-07 with OpenVINO 2026.4.1, PyTorch 2.14.1 (CPU wheel), Python 3.12
and Windows 11, from the logged-in desktop session:

| Host | Devices tested | Precision |
|---|---|---|
| Intel Core Ultra 9 285H, 64 GB | CPU; GPU (Intel Arc 140T integrated GPU); NPU (Intel AI Boost) | CPU f32, GPU f32, NPU FP16 |
| Intel Core Ultra 7 265F, 64 GB | CPU; NPU (Intel AI Boost) | CPU f32, NPU FP16 |

### Test case

Case `s0987` of the TotalSegmentator MRI dataset v3.0.0, released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): Wasserthal J, Akinci
D'Antonoli T. TotalSegmentator MRI dataset, version 3.0.0. Zenodo, 2026.
https://doi.org/10.5281/zenodo.22688334. Described in Akinci D'Antonoli T, et al.
Radiology 2025;314(2):e241613. The image is not redistributed here.

It is a whole-body gradient-echo examination of a 4-year-old (3 T, 256 × 208 × 400
voxels at 1.33 × 1.33 × 2.5 mm, head to feet). It differs from CASIMIR's training data:
it is not fat-suppressed, and contrast enhancement is not documented. The numbers below
therefore show agreement between OpenVINO and PyTorch and their run time. They do not
measure segmentation accuracy. The volume is also smaller than a typical whole-body
examination of an older child or young adult, and run time grows with volume size.

### Agreement with PyTorch

Each OpenVINO output is compared with the label map from the default PyTorch CPU path on
the same host. The PyTorch outputs on the two hosts are identical voxel for voxel, and so
are the two NPU outputs. "absent" means neither map contains the structure.

| Structure | PyTorch voxels | 285H CPU f32 | 285H GPU f32 | 285H NPU FP16 | 265F CPU f32 | 265F NPU FP16 |
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
| **Voxels differing (of 21.3 M)** | | 1 | 0 | 169 | 2 | 169 |

CPU and GPU ran with `INFERENCE_PRECISION_HINT=f32`; the NPU reports FP16 inference
precision.

### Time per exam

Full pipeline per exam, one process per row, 1 warm-up exam followed by 5 timed exams on
the same volume. "Sliding window" is the network part of each exam (nnU-Net's sliding
window with mirroring); "pre/post" is the rest of the exam, preprocessing, resampling and
NIfTI export, which runs on the CPU in every row and takes 20–24 s on this case. The
PyTorch rows are the default `casimir --device cpu` path on the same host; it runs the
network with nnU-Net's default of 8 threads (`nnUNet_def_n_proc`), and the second PyTorch
row sets that to the number of logical CPUs. "Compile" is `compile_model` for the IR in
that process; "cold" is the NPU compile with the driver cache bypassed.

| Host | Path | Precision | Compile (s) | Mean (s) | Min (s) | Sliding window, mean (s) | Pre/post, mean (s) | Peak RSS (GiB) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Core Ultra 9 285H | PyTorch CPU (default) | fp32 | – | 115.0 | 114.1 | 92.2 | 22.8 | 8.0 |
| Core Ultra 9 285H | PyTorch CPU, `nnUNet_def_n_proc` = logical CPUs | fp32 | – | 122.7 | 118.6 | 98.8 | 23.9 | 8.1 |
| Core Ultra 9 285H | OpenVINO CPU | f32 | 0.26 | 76.9 | 76.5 | 54.6 | 22.3 | 8.4 |
| Core Ultra 9 285H | OpenVINO GPU (Arc 140T) | f32 | 0.35 | 48.9 | 48.2 | 26.0 | 22.9 | 8.1 |
| Core Ultra 9 285H | OpenVINO NPU | FP16 | 0.20 (cold 1.94) | 58.2 | 55.7 | 34.0 | 24.2 | 8.7 |
| Core Ultra 7 265F | PyTorch CPU (default) | fp32 | – | 94.8 | 93.6 | 75.1 | 19.7 | 8.2 |
| Core Ultra 7 265F | PyTorch CPU, `nnUNet_def_n_proc` = logical CPUs | fp32 | – | 75.3 | 75.0 | 55.4 | 19.9 | 8.3 |
| Core Ultra 7 265F | OpenVINO CPU | f32 | 0.14 | 85.4 | 84.8 | 65.4 | 20.0 | 8.5 |
| Core Ultra 7 265F | OpenVINO NPU | FP16 | 0.17 (cold 1.84) | 48.7 | 48.6 | 28.7 | 20.0 | 8.1 |

Peak RSS covers the main process and the nnU-Net preprocessing and export workers. With
the default LATENCY hint, OpenVINO CPU used 14 inference threads on the Core Ultra 9 285H
and 8 on the Core Ultra 7 265F (`INFERENCE_NUM_THREADS` of the compiled model).
