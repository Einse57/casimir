# OpenVINO / ONNX acceleration for CASIMIR

Experimental path that keeps nnU-Net v2 preprocessing, sliding-window, mirroring, and
export, but swaps the per-tile `PlainConvUNet` forward for OpenVINO (or ONNX Runtime).

## Setup

```bash
pip install -e .
pip install -r requirements-openvino.txt
# download weights once via the normal casimir path, or:
export CASIMIR_WEIGHTS_PATH=/path/to/local/weights   # contains dataset.json, plans.json, fold_all/weights.pth
```

## Export ONNX + OpenVINO IR

```bash
PYTHONPATH=. python scripts/export_openvino.py --weights "$CASIMIR_WEIGHTS_PATH" --outdir openvino_models
```

**Critical:** nnU-Net stores fold weights in `list_of_parameters` and only loads them at
predict time. The export script calls `load_state_dict` before `torch.onnx.export`.
Exporting without that step produces an uninitialized network.

Supports `--device CPU|GPU|NPU` on the inference CLI (GPU/NPU fall back to CPU when absent).

## Smoke sample

No clinical sample ships with the repo. Generate a spacing-matched synthetic volume:

```bash
PYTHONPATH=. python scripts/make_smoke_nifti.py -o data/smoke/smoke_exam_0000.nii.gz
```

## Reference CPU vs OpenVINO

```bash
PYTHONPATH=. python scripts/run_ref_cpu.py
PYTHONPATH=. python scripts/casimir_ov_cli.py -i data/smoke/smoke_exam_0000.nii.gz -o data/ov_out \
  --backend openvino --device CPU --model-dir "$CASIMIR_WEIGHTS_PATH"
PYTHONPATH=. python scripts/dice_agreement.py \
  --ref data/ref_out/smoke_exam_0000.nii.gz --pred data/ov_out/smoke_exam_0000.nii.gz
```

`--backend onnxruntime` is also available (matches torch logits more tightly; useful as a
sanity check). OpenVINO may show InstanceNorm logit drift but argmax / Dice on the smoke
sample matched the torch reference (Dice 1.0) after a correct weight-loaded export.

## Bench (box-only / non-target)

```bash
PYTHONPATH=. python scripts/bench_casimir.py --backend both --device CPU --runs 3
```

Published whole-body baseline from the CASIMIR README / HF card: **~4 min/exam on Apple M2
Pro CPU**, **~1.5 min on RTX 4090 or M2 Pro MPS**. Box numbers are not Core/Xeon target iron.
