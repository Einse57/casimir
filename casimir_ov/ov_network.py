"""Torch-module-shaped wrapper around an OpenVINO-compiled CASIMIR 2D network."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn


class OpenVINONetwork(nn.Module):
    """Drop-in forward() replacement for the nnU-Net PlainConvUNet used by CASIMIR.

    Compiles casimir_2d.xml for device in {CPU, GPU, NPU}. Box smoke uses CPU only;
    GPU/NPU are accepted so the same CLI works on Core Ultra / discrete targets later.

    load_state_dict is a no-op so nnUNetPredictor's fold loop still works (weights are
    already baked into the IR).
    """

    def __init__(self, model_xml: str | Path, device: str = "CPU"):
        super().__init__()
        import openvino as ov

        device = device.upper()
        self.device_name = device
        self.core = ov.Core()
        available = list(self.core.available_devices)
        if device not in available:
            if device in {"GPU", "NPU"} and "CPU" in available:
                self.fallback_note = f"{device} unavailable; using CPU (have {available})"
                device = "CPU"
            else:
                raise RuntimeError(
                    f"OpenVINO device {device!r} not available; have {available}"
                )
        else:
            self.fallback_note = None
        self.device_name = device
        self.compiled = self.core.compile_model(str(model_xml), device)
        self._req = self.compiled.create_infer_request()

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
) -> dict:
    """Export fold_all PlainConvUNet to ONNX + OpenVINO IR.

    IMPORTANT: nnUNetPredictor.initialize_from_trained_model_folder builds a random
    network and only stores weights in list_of_parameters; we must load_state_dict
    before export or the ONNX/IR will contain uninitialized weights.
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
    dummy = torch.randn(1, 1, *patch, dtype=torch.float32)

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
    onnx_max_abs = float(np.max(np.abs(torch_out - ort_out)))
    onnx_argmax = float((torch_out.argmax(1) == ort_out.argmax(1)).mean())

    ov_model = ov.convert_model(str(onnx_path))
    xml_path = outdir / "casimir_2d.xml"
    ov.save_model(ov_model, str(xml_path))

    core = ov.Core()
    compiled = core.compile_model(ov_model, "CPU")
    req = compiled.create_infer_request()
    req.infer({0: dummy.numpy()})
    ov_out = np.array(req.get_output_tensor(0).data)
    ov_max_abs = float(np.max(np.abs(torch_out - ov_out)))
    argmax_agree = float((torch_out.argmax(1) == ov_out.argmax(1)).mean())

    # padded-tile stress (nnU-Net pads to patch_size)
    pad = np.zeros((1, 1, *patch), dtype=np.float32)
    pad[0, 0, 50:122, 20:134] = np.random.default_rng(0).normal(0, 1, (72, 114)).astype(
        np.float32
    )
    with torch.no_grad():
        tpad = net(torch.from_numpy(pad)).numpy()
    opad = sess.run(None, {"input": pad})[0]
    req.infer({0: pad})
    ovpad = np.array(req.get_output_tensor(0).data)

    meta = {
        "patch_size": patch,
        "num_classes": int(torch_out.shape[1]),
        "onnx": str(onnx_path),
        "ir_xml": str(xml_path),
        "ir_bin": str(outdir / "casimir_2d.bin"),
        "onnx_mb": onnx_path.stat().st_size / 1e6,
        "onnx_vs_torch_max_abs": onnx_max_abs,
        "onnx_vs_torch_argmax_agree": onnx_argmax,
        "ov_vs_torch_max_abs": ov_max_abs,
        "ov_vs_torch_argmax_agree": argmax_agree,
        "padded_ort_argmax_agree": float((tpad.argmax(1) == opad.argmax(1)).mean()),
        "padded_ov_argmax_agree": float((tpad.argmax(1) == ovpad.argmax(1)).mean()),
        "openvino_version": ov.__version__,
        "devices": core.available_devices,
        "note": (
            "Weights loaded via list_of_parameters[0] before export. "
            "ONNX Runtime matches torch; OpenVINO may show InstanceNorm logit drift."
        ),
    }
    (outdir / "export_meta.json").write_text(json.dumps(meta, indent=2))
    return meta
