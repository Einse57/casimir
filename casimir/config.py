"""Weights download and cache."""

import os
import sys
from pathlib import Path

REPO_ID = "fjhorvath/casimir"
# Pinned to an immutable commit, so a given release always resolves the same weights.
REVISION = "d24ca411f0316bd6d96d08ba2379ea33ae772d45"  # tag v1.0.0

CHECKPOINT = "weights.pth"
# nnU-Net joins f"fold_{fold}" itself, so that subdirectory name is not ours to choose.
FOLD_DIR = "fold_all"


def weights_root():
    override = os.environ.get("CASIMIR_WEIGHTS_PATH")
    if override:
        return Path(override).expanduser()

    from huggingface_hub import snapshot_download

    return Path(snapshot_download(REPO_ID, revision=REVISION,
                                  allow_patterns=["*.json", f"{FOLD_DIR}/*"]))


def ensure_weights():
    root = weights_root()
    if not (root / FOLD_DIR / CHECKPOINT).is_file():
        sys.exit(f"no checkpoint under {root}")
    return root
