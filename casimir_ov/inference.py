"""Inference with the CASIMIR network run by OpenVINO.

Preprocessing, the sliding window with mirroring, resampling and export are nnU-Net's,
set up as in casimir.inference. Only the forward pass of each tile is replaced.
"""

from __future__ import annotations

import hashlib
import multiprocessing
import os
from pathlib import Path

from casimir import config
from casimir.inference import _inputs, _stem


def _checkpoint_digest(weights_dir) -> str:
    h = hashlib.sha256()
    with open(Path(weights_dir) / config.FOLD_DIR / config.CHECKPOINT, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def ir_cache_dir(weights_dir) -> Path:
    """Where the exported IR for these weights is kept.

    ``CASIMIR_OPENVINO_CACHE`` if set, else ``~/.cache/casimir/openvino``, with one
    subdirectory per checkpoint, so other weights never reuse a stale IR.
    """
    root = os.environ.get("CASIMIR_OPENVINO_CACHE")
    base = Path(root).expanduser() if root else Path.home() / ".cache" / "casimir" / "openvino"
    return base / _checkpoint_digest(weights_dir)


def _check_inputs(images):
    files = _inputs(images)
    if not files:
        raise SystemExit(f"no NIfTI images in {images}")
    stems = [_stem(f) for f in files]
    if len(set(stems)) != len(stems):
        clashing = sorted({s for s in stems if stems.count(s) > 1})
        raise SystemExit("inputs would overwrite each other, they differ only by NIfTI "
                         f"extension: {', '.join(clashing)}")
    return files, stems


def build_predictor(device="CPU", model_dir=None, ir=None, step_size=0.5,
                    precision="auto", allow_fallback=True, verbose=False):
    """Return an nnUNetPredictor whose network is compiled with OpenVINO for device.

    ``ir`` is an OpenVINO IR ``.xml``; by default the IR is exported to
    ``ir_cache_dir`` on first use. ``precision`` sets INFERENCE_PRECISION_HINT:
    ``auto`` is f32 on CPU and GPU and the plugin default (FP16) on the NPU,
    ``default`` keeps every plugin default.
    """
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    from casimir_ov.export import ensure_ir
    from casimir_ov.network import OpenVINONetwork

    weights = Path(model_dir) if model_dir else config.ensure_weights()
    if ir is None:
        ir = ensure_ir(weights, ir_cache_dir(weights))
    elif not Path(ir).is_file():
        raise SystemExit(f"no OpenVINO IR at {ir}")

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
    predictor.initialize_from_trained_model_folder(
        str(weights), ("all",), checkpoint_name=config.CHECKPOINT
    )
    network = OpenVINONetwork(ir, device=device, precision=precision,
                              allow_fallback=allow_fallback)
    if network.fallback_note:
        print(network.fallback_note)
    if verbose:
        print(f"openvino {network.describe()}")
    predictor.network = network
    return predictor


def run_predictor(predictor, images, outdir):
    """Segment images with a predictor from build_predictor, return the paths written."""
    files, stems = _check_inputs(images)
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


def infer(images, outdir, device="CPU", model_dir=None, ir=None, step_size=0.5,
          precision="auto", allow_fallback=True, verbose=False):
    """Segment images into outdir with OpenVINO and return the paths written."""
    # Validate before importing torch and openvino, so a bad path fails in milliseconds.
    _check_inputs(images)
    predictor = build_predictor(device=device, model_dir=model_dir, ir=ir,
                                step_size=step_size, precision=precision,
                                allow_fallback=allow_fallback, verbose=verbose)
    return run_predictor(predictor, images, outdir)
