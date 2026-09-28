"""Inference."""

import multiprocessing
from pathlib import Path

from casimir import config

SUFFIXES = (".nii.gz", ".nii")


def _stem(path):
    """Input filename with the NIfTI extension removed and nothing else changed."""
    name = path.name
    for suffix in SUFFIXES:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def _inputs(images):
    if not isinstance(images, (str, Path)):
        return [Path(f) for f in images]
    path = Path(images)
    if not path.is_dir():
        return [path]
    return sorted(f for f in path.iterdir()
                  if f.name.endswith(SUFFIXES) and not f.name.startswith("."))


def infer(images, outdir, device="cuda", model_dir=None, step_size=0.5,
          deterministic=True, verbose=False):
    """Segment images into outdir and return the paths written."""
    # Validate before importing torch, so a bad path fails in milliseconds.
    files = _inputs(images)
    if not files:
        raise SystemExit(f"no NIfTI images in {images}")
    stems = [_stem(f) for f in files]
    if len(set(stems)) != len(stems):
        clashing = sorted({s for s in stems if stems.count(s) > 1})
        raise SystemExit("inputs would overwrite each other, they differ only by NIfTI "
                         f"extension: {', '.join(clashing)}")

    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if device == "cpu":
        torch.set_num_threads(multiprocessing.cpu_count())
    elif device == "cuda":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    predictor = nnUNetPredictor(
        tile_step_size=step_size,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device(device),
        verbose=verbose,
        verbose_preprocessing=verbose,
        allow_tqdm=not verbose,
    )
    # nnUNetPredictor's constructor switches cudnn.benchmark on, so this has to follow it
    if deterministic and device == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    predictor.initialize_from_trained_model_folder(
        str(model_dir or config.ensure_weights()), ("all",),
        checkpoint_name=config.CHECKPOINT,
    )
    predictor.predict_from_files(
        [[str(f)] for f in files],
        [str(outdir / s) for s in stems],
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
    )
    return [outdir / f"{s}.nii.gz" for s in stems]
