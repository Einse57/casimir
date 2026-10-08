"""Torch-module-shaped wrapper around an OpenVINO-compiled CASIMIR 2D network."""
from __future__ import annotations

import contextlib
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

# Devices on which INFERENCE_PRECISION_HINT=f32 is honoured. The NPU plugin runs FP16.
_F32_CAPABLE = {"CPU", "GPU"}


def resolve_precision(device: str, precision: str | None) -> str | None:
    """Return the INFERENCE_PRECISION_HINT to set for ``device``, or None to keep the
    plugin default.

    ``"auto"`` (default) pins f32 on CPU and GPU, because the plugin defaults (bf16 on
    CPUs with AMX/AVX512-BF16, f16 on GPU) shift InstanceNorm logits enough to flip
    some argmax voxels. NPU keeps its default (FP16).
    """
    if precision is None or precision == "default":
        return None
    precision = precision.lower()
    if precision == "auto":
        return "f32" if device.split(".")[0] in _F32_CAPABLE else None
    return precision


class OpenVINONetwork(nn.Module):
    """Drop-in forward() replacement for the nnU-Net PlainConvUNet used by CASIMIR.

    Compiles ``casimir_2d.xml`` for an OpenVINO device (``CPU``, ``GPU``, ``NPU``).
    ``load_state_dict`` is a no-op so nnUNetPredictor's fold loop still works (the
    weights are baked into the IR).
    """

    def __init__(
        self,
        model_xml: str | Path,
        device: str = "CPU",
        precision: str | None = "auto",
        allow_fallback: bool = True,
        config: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        import openvino as ov

        device = device.upper()
        self.core = ov.Core()
        available = list(self.core.available_devices)
        self.fallback_note = None
        if device.split(".")[0] not in {d.split(".")[0] for d in available}:
            if allow_fallback and "CPU" in available:
                self.fallback_note = f"{device} unavailable; using CPU (have {available})"
                device = "CPU"
            else:
                raise RuntimeError(
                    f"OpenVINO device {device!r} not available; have {available}"
                )
        self.device_name = device
        cfg = dict(config or {})
        hint = resolve_precision(device, precision)
        if hint is not None:
            cfg.setdefault("INFERENCE_PRECISION_HINT", hint)
        self.config = cfg
        t0 = time.perf_counter()
        self.compiled = self.core.compile_model(str(model_xml), device, cfg)
        self.compile_s = time.perf_counter() - t0
        self._req = self.compiled.create_infer_request()

    def describe(self) -> dict:
        """Device name and the compiled-model properties that matter for results."""
        info = {"device": self.device_name, "config": dict(self.config)}
        with contextlib.suppress(RuntimeError):  # plugin dependent
            info["full_device_name"] = self.core.get_property(
                self.device_name, "FULL_DEVICE_NAME"
            )
        for key in (
            "INFERENCE_PRECISION_HINT",
            "PERFORMANCE_HINT",
            "NUM_STREAMS",
            "INFERENCE_NUM_THREADS",
            "EXECUTION_DEVICES",
        ):
            with contextlib.suppress(RuntimeError):
                info[key] = str(self.compiled.get_property(key))
        return info

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        return

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            arr = x.detach().cpu().numpy()
        else:
            arr = np.asarray(x)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        self._req.infer({0: arr})
        out = np.array(self._req.get_output_tensor(0).data)
        return torch.from_numpy(out)


def export_casimir_onnx_and_ir(
    weights_dir: str | Path,
    outdir: str | Path,
    opset: int = 17,
    compress_to_fp16: bool = False,
) -> dict:
    """Export the fold_all PlainConvUNet to ONNX + OpenVINO IR.

    IMPORTANT: nnUNetPredictor.initialize_from_trained_model_folder builds a random
    network and only stores weights in list_of_parameters; we must load_state_dict
    before export or the ONNX/IR will contain uninitialized weights.

    The IR keeps FP32 weights by default (``compress_to_fp16=False``), so CPU/GPU runs
    with INFERENCE_PRECISION_HINT=f32 compute with the original weights. NPU converts to
    FP16 at compile time either way.
    """
    import json

    import openvino as ov
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    weights_dir = Path(weights_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pred = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=False,
        device=torch.device("cpu"),
        verbose=False,
        allow_tqdm=False,
    )
    pred.initialize_from_trained_model_folder(
        str(weights_dir), ("all",), checkpoint_name="weights.pth"
    )
    # Critical: apply fold weights before export.
    pred.network.load_state_dict(pred.list_of_parameters[0])
    net = pred.network.eval()
    patch = list(pred.configuration_manager.patch_size)
    dummy = torch.from_numpy(
        np.random.default_rng(0).standard_normal((1, 1, *patch)).astype(np.float32)
    )

    onnx_path = outdir / "casimir_2d.onnx"
    with torch.no_grad():
        torch.onnx.export(
            net,
            dummy,
            str(onnx_path),
            input_names=["input"],
            output_names=["logits"],
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    # Remove stale external-data sidecars from prior dynamo exports.
    for stale in outdir.glob("*.onnx.data"):
        stale.unlink()

    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        torch_out = net(dummy).numpy()
    ort_out = sess.run(None, {"input": dummy.numpy()})[0]

    ov_model = ov.convert_model(str(onnx_path))
    xml_path = outdir / "casimir_2d.xml"
    ov.save_model(ov_model, str(xml_path), compress_to_fp16=compress_to_fp16)

    core = ov.Core()
    compiled = core.compile_model(str(xml_path), "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    req = compiled.create_infer_request()
    req.infer({0: dummy.numpy()})
    ov_out = np.array(req.get_output_tensor(0).data)

    meta = {
        "patch_size": patch,
        "num_classes": int(torch_out.shape[1]),
        "opset": opset,
        "ir_weights": "fp16" if compress_to_fp16 else "fp32",
        "onnx_mb": round(onnx_path.stat().st_size / 1e6, 1),
        "ir_bin_mb": round((outdir / "casimir_2d.bin").stat().st_size / 1e6, 1),
        "onnx_vs_torch_max_abs": float(np.max(np.abs(torch_out - ort_out))),
        "onnx_vs_torch_argmax_agree": float((torch_out.argmax(1) == ort_out.argmax(1)).mean()),
        "ov_cpu_f32_vs_torch_max_abs": float(np.max(np.abs(torch_out - ov_out))),
        "ov_cpu_f32_vs_torch_argmax_agree": float(
            (torch_out.argmax(1) == ov_out.argmax(1)).mean()
        ),
        "openvino_version": ov.__version__,
        "torch_version": torch.__version__,
    }
    (outdir / "export_meta.json").write_text(json.dumps(meta, indent=2))
    return meta
