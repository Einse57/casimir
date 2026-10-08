"""Torch-module-shaped wrapper around the OpenVINO-compiled CASIMIR network."""

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
    CPUs with AMX or AVX512-BF16, f16 on GPU) shift InstanceNorm logits enough to flip
    some argmax voxels. The NPU keeps its default (FP16).
    """
    if precision is None or precision == "default":
        return None
    precision = precision.lower()
    if precision == "auto":
        return "f32" if device.split(".")[0] in _F32_CAPABLE else None
    return precision


class OpenVINONetwork(nn.Module):
    """Drop-in forward() replacement for the nnU-Net PlainConvUNet used by CASIMIR.

    Compiles the IR for an OpenVINO device (``CPU``, ``GPU``, ``NPU``).
    ``load_state_dict`` is a no-op so the fold loop of nnUNetPredictor still works; the
    weights are part of the IR.
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
                self.fallback_note = f"{device} unavailable, using CPU (have {available})"
                device = "CPU"
            else:
                raise RuntimeError(
                    f"OpenVINO device {device!r} not available, have {available}"
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
        # nnU-Net mirrors tiles with torch.flip, which can give non-contiguous arrays
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        self._req.infer({0: arr})
        out = np.array(self._req.get_output_tensor(0).data)
        return torch.from_numpy(out)
