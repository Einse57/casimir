"""Export the CASIMIR network to OpenVINO IR.

    python -m casimir_ov.export --outdir <directory>

The inference command exports on first use, so running this by hand is optional.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np

from casimir import config

IR_NAME = "casimir_2d.xml"


def export_ir(weights_dir: str | Path, outdir: str | Path, opset: int = 17,
              compress_to_fp16: bool = False) -> dict:
    """Export the fold_all PlainConvUNet through ONNX to OpenVINO IR in outdir.

    nnUNetPredictor.initialize_from_trained_model_folder builds the network with random
    weights and keeps the trained ones in list_of_parameters, so they are loaded here
    before export.

    The IR keeps FP32 weights by default, so CPU and GPU runs with
    INFERENCE_PRECISION_HINT=f32 compute with the original weights. The NPU converts to
    FP16 when it compiles the model either way.
    """
    import openvino as ov
    import torch
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
        str(weights_dir), ("all",), checkpoint_name=config.CHECKPOINT
    )
    pred.network.load_state_dict(pred.list_of_parameters[0])
    net = pred.network.eval()
    patch = list(pred.configuration_manager.patch_size)
    dummy = torch.from_numpy(
        np.random.default_rng(0).standard_normal((1, 1, *patch)).astype(np.float32)
    )

    xml_path = outdir / IR_NAME
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "casimir_2d.onnx"
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
            torch_out = net(dummy).numpy()
        ov.save_model(ov.convert_model(str(onnx_path)), str(xml_path),
                      compress_to_fp16=compress_to_fp16)

    compiled = ov.Core().compile_model(str(xml_path), "CPU",
                                       {"INFERENCE_PRECISION_HINT": "f32"})
    ov_out = compiled(dummy.numpy())[0]

    meta = {
        "patch_size": patch,
        "num_classes": int(torch_out.shape[1]),
        "opset": opset,
        "ir_weights": "fp16" if compress_to_fp16 else "fp32",
        "ir_bin_mb": round(xml_path.with_suffix(".bin").stat().st_size / 1e6, 1),
        "ov_cpu_f32_vs_torch_max_abs": float(np.max(np.abs(torch_out - ov_out))),
        "ov_cpu_f32_vs_torch_argmax_agree": float(
            (torch_out.argmax(1) == ov_out.argmax(1)).mean()
        ),
        "openvino_version": ov.__version__,
        "torch_version": torch.__version__,
    }
    (outdir / "export_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def ensure_ir(weights_dir: str | Path, outdir: str | Path) -> Path:
    """Return the IR in outdir, exporting it there first if it is missing."""
    outdir = Path(outdir)
    xml_path = outdir / IR_NAME
    if xml_path.is_file() and xml_path.with_suffix(".bin").is_file():
        return xml_path
    print(f"exporting the OpenVINO model to {outdir}, once per set of weights")
    outdir.parent.mkdir(parents=True, exist_ok=True)
    # export next to the target and move it in place, so an interrupted export
    # never leaves a partial model behind. Not tempfile.mkdtemp: its owner-only
    # permissions (an owner-only ACL on Windows) would carry over to the cache.
    tmp = outdir.parent / f".export-{uuid.uuid4().hex}"
    tmp.mkdir()
    try:
        export_ir(weights_dir, tmp)
        if outdir.exists():
            shutil.rmtree(outdir)
        tmp.rename(outdir)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return xml_path


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m casimir_ov.export",
                                description="Export the CASIMIR network to OpenVINO IR.")
    p.add_argument("--model-dir", help="local weights directory instead of downloading")
    p.add_argument("--outdir", required=True, help="directory for the IR")
    p.add_argument("--fp16-weights", action="store_true",
                   help="store the IR weights as FP16, half the size; CPU and GPU "
                        "output then differs slightly from PyTorch")
    args = p.parse_args(argv)
    weights = args.model_dir or config.ensure_weights()
    meta = export_ir(weights, args.outdir, compress_to_fp16=args.fp16_weights)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
