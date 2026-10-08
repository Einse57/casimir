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


def build_predictor(
    device: str = "CPU",
    model_dir=None,
    ov_model_xml=None,
    onnx_path=None,
    backend: str = "openvino",
    step_size: float = 0.5,
    precision: str | None = "auto",
    allow_fallback: bool = True,
    verbose: bool = False,
):
    """Return an nnUNetPredictor whose network is an OpenVINO (or ONNX Runtime) model.

    ``device`` is the OpenVINO device (``CPU``, ``GPU``, ``NPU``); it is ignored by the
    onnxruntime backend. ``precision`` sets INFERENCE_PRECISION_HINT (``auto`` = f32 on
    CPU/GPU, plugin default on NPU; ``default`` = plugin default everywhere).
    """
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

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
        from casimir_ov.ort_network import ONNXRuntimeNetwork

        onnx = Path(onnx_path) if onnx_path else Path("openvino_models/casimir_2d.onnx")
        if not onnx.is_file():
            raise SystemExit(f"ONNX not found at {onnx}")
        predictor.network = ONNXRuntimeNetwork(onnx)
        print(f"backend=onnxruntime providers={predictor.network.session.get_providers()}")
    else:
        from casimir_ov.ov_network import OpenVINONetwork

        xml = Path(ov_model_xml) if ov_model_xml else Path("openvino_models/casimir_2d.xml")
        if not xml.is_file():
            raise SystemExit(
                f"OpenVINO IR not found at {xml}. Run scripts/export_openvino.py first."
            )
        ov_net = OpenVINONetwork(
            xml, device=device, precision=precision, allow_fallback=allow_fallback
        )
        if ov_net.fallback_note:
            print(ov_net.fallback_note)
        predictor.network = ov_net
        print(f"backend=openvino {ov_net.describe()}")
    return predictor


def run_predictor(predictor, images, outdir):
    """Segment images with a predictor from build_predictor; return the paths written."""
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
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    predictor.predict_from_files(
        [[str(f)] for f in files],
        [str(outdir / s) for s in stems],
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
    )
    return [outdir / f"{s}.nii.gz" for s in stems]


def infer_ov(
    images,
    outdir,
    device: str = "CPU",
    model_dir=None,
    ov_model_xml=None,
    onnx_path=None,
    backend: str = "openvino",
    step_size: float = 0.5,
    precision: str | None = "auto",
    allow_fallback: bool = True,
    verbose: bool = False,
):
    """Segment with OpenVINO or ONNX Runtime and return the paths written."""
    if not _inputs(images):
        raise SystemExit(f"no NIfTI images in {images}")
    predictor = build_predictor(
        device=device,
        model_dir=model_dir,
        ov_model_xml=ov_model_xml,
        onnx_path=onnx_path,
        backend=backend,
        step_size=step_size,
        precision=precision,
        allow_fallback=allow_fallback,
        verbose=verbose,
    )
    return run_predictor(predictor, images, outdir)
