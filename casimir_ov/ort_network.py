"""ONNX Runtime drop-in network (numerically matches torch; used for agreement)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort


class ONNXRuntimeNetwork(nn.Module):
    def __init__(self, onnx_path: str | Path, providers=None):
        super().__init__()
        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=providers or ["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        return

    def forward(self, x: torch.Tensor):
        arr = x.detach().cpu().numpy()
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        out = self.session.run(None, {self.input_name: arr})[0]
        return torch.from_numpy(out)
