## OpenVINO CPU path for CASIMIR (draft, fork-only)

Feasibility + box smoke for accelerating CASIMIR (nnU-Net v2, 2D, 13 structures / 20 labels)
with OpenVINO. **Draft PR on Einse57/casimir only — do not merge to upstream.**

### Licenses
- Code (`fjhorvath/casimir`): **Apache-2.0**
- Weights (`huggingface.co/fjhorvath/casimir`, rev `d24ca411…` / tag v1.0.0): **Apache-2.0**

### What landed
- `casimir_ov/`: OpenVINO and ONNX Runtime drop-in networks (same nnU-Net sliding-window /
  mirroring / export; only tile forward swapped)
- `scripts/export_openvino.py`: ONNX + IR export (**loads `list_of_parameters[0]` before export**)
- `scripts/casimir_ov_cli.py`: `--backend openvino|onnxruntime`, `--device CPU|GPU|NPU`
- `scripts/bench_casimir.py`: one-command latency (mean/p50/p90, peak RSS)
- `scripts/dice_agreement.py`, `make_smoke_nifti.py`, `run_ref_cpu.py`
- `OPENVINO.md`, `requirements-openvino.txt`

### Reference CPU (box smoke)
Synthetic spacing-matched NIfTI (`128×64×64` stored, plans spacing). Torch CPU nnU-Net path:
- ~39–41 s / volume on this box (8× Xeon vCPU) — **box-only / non-target**
- Output: labels `{0, 15}` (4 heart voxels); see `data/ref_stats.json`

### OpenVINO conversion
- OpenVINO **2026.4.1** via pip; full `PlainConvUNet` 2D fold_all → ONNX (opset 17) → IR
- Patch `[1,1,512,192]` → logits `[1,21,512,192]`
- After fixing weight load: ONNX↔torch argmax **1.0**; OV↔torch argmax **~0.9999** (padded tile **1.0**)
- See `data/export_meta.json`

### Agreement (smoke)
| Backend | voxel agree | mean Dice (present) | heart Dice |
|---------|-------------|---------------------|------------|
| onnxruntime | 1.0 | 1.0 | 1.0 |
| openvino CPU | 1.0 | 1.0 | 1.0 |

### Box bench (smoke volume — **not** whole-body exam size)

| Backend | mean | p50 | p90 | peak RSS |
|---------|------|-----|-----|----------|
| torch CPU | 40.2 s | 40.2 s | 40.9 s | ~1.9 GiB |
| OpenVINO CPU | 14.8 s | 14.8 s | 15.1 s | ~2.1 GiB |

**Published baseline (README / HF):** ~**4 min/exam** M2 Pro CPU; ~**1.5 min** RTX 4090 / M2 Pro MPS.
Goal for later Core/Xeon runs: beat 4 minutes. These box numbers are labeled **box-only / non-target**.

### How to try
```bash
pip install -e . && pip install -r requirements-openvino.txt
export CASIMIR_WEIGHTS_PATH=...   # or let casimir download
PYTHONPATH=. python scripts/export_openvino.py
PYTHONPATH=. python scripts/make_smoke_nifti.py
PYTHONPATH=. python scripts/bench_casimir.py --backend both --device CPU
```

### Blockers / notes
- No clinical sample in-repo; synthetic smoke only (tiny present-structure support).
- Box is not Core/Xeon target iron; re-bench on target for the 4‑minute claim.
- GPU/NPU flags accepted; this box only has OpenVINO `CPU`.
- Large ONNX/IR binaries gitignored — regenerate with the export script.
