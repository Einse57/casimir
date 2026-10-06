"""CASIMIR inference with OpenVINO or ONNX Runtime network tiles.

Reuses nnU-Net v2 preprocessing / sliding-window / mirroring / export from the
reference path; only the per-tile forward is swapped.
"""
from __future__ import annotations

import multiprocessing
from pathlib import Path

import torch

from casimir import config
from casimir.inference import _inputs, _stem
from casimir_ov.ov_network import OpenVINONetwork
from casimir_ov.ort_network import ONNXRuntimeNetwork


def infer_ov(
    images,
    outdir,
    device: str = "CPU",
    model_dir=None,
    ov_model_xml=None,
    onnx_path=None,
    backend: str = "openvino",
    step_size: float = 0.5,
    verbose: bool = False,
):
    """Segment with OpenVINO or ONNX Runtime.

    device is OpenVINO device CPU|GPU|NPU (ignored for onnxruntime backend except logging).
    """
    files = _inputs(images)
    if not files:
        raise SystemExit(f"no NIfTI images in {images}")
    stems = [_stem(f) for f in files]
    if len(set(stems)) != len(stems):
        clashing = sorted({s for s in stems if stems.count(s) > 1})
        raise SystemExit(
            "inputs would overwrite each other, they differ only by NIfTI "
            f"extension: {', '.join(clashing)}"
        )

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(multiprocessing.cpu_count())

    predictor = nnUNetPredictor(
        tile_step_size=step_size,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=False,
        device=torch.device("cpu"),
        verbose=verbose,
        verbose_preprocessing=verbose,
        allow_tqdm=not verbose,
    )
    weights = Path(model_dir) if model_dir else config.ensure_weights()
    predictor.initialize_from_trained_model_folder(
        str(weights), ("all",), checkpoint_name=config.CHECKPOINT
    )

    backend = backend.lower()
    if backend in {"ort", "onnx", "onnxruntime"}:
        onnx = Path(onnx_path) if onnx_path else Path("openvino_models/casimir_2d.onnx")
        if not onnx.is_file():
            raise SystemExit(f"ONNX not found at {onnx}")
        predictor.network = ONNXRuntimeNetwork(onnx)
        print(f"backend=onnxruntime providers={predictor.network.session.get_providers()}")
    else:
        xml = Path(ov_model_xml) if ov_model_xml else Path("openvino_models/casimir_2d.xml")
        if not xml.is_file():
            raise SystemExit(
                f"OpenVINO IR not found at {xml}. Run scripts/export_openvino.py first."
            )
        ov_net = OpenVINONetwork(xml, device=device)
        if ov_net.fallback_note:
            print(ov_net.fallback_note)
        predictor.network = ov_net
        print(f"backend=openvino device={ov_net.device_name}")

    predictor.predict_from_files(
        [[str(f)] for f in files],
        [str(outdir / s) for s in stems],
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
    )
    return [outdir / f"{s}.nii.gz" for s in stems]
