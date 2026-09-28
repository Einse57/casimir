import json
from pathlib import Path

import pytest

from casimir import config, inference, main
from casimir.inference import _inputs, _stem


def test_labels():
    labels = json.loads((Path(config.__file__).parent / "labels.json").read_text())["labels"]
    assert labels["background"] == 0
    assert sorted(labels.values()) == list(range(21))


def test_stem_removes_only_the_nifti_extension():
    assert _stem(Path("case.nii.gz")) == "case"
    assert _stem(Path("case.nii")) == "case"
    # Nothing else is stripped. The output name is the input name.
    assert _stem(Path("case_0000.nii.gz")) == "case_0000"
    assert _stem(Path("sub-01_T1w_0000.nii.gz")) == "sub-01_T1w_0000"
    assert _stem(Path("odd.name.with.dots.nii.gz")) == "odd.name.with.dots"


def test_inputs(tmp_path):
    a = tmp_path / "a.nii.gz"
    b = tmp_path / "b.nii"
    for f in (a, b):
        f.write_bytes(b"")
    (tmp_path / "notes.txt").write_text("ignored")
    assert _inputs(tmp_path) == [a, b]
    assert _inputs(a) == [a]


def test_weights_path_override(tmp_path, monkeypatch):
    monkeypatch.setenv("CASIMIR_WEIGHTS_PATH", str(tmp_path))
    assert config.weights_root() == tmp_path
    with pytest.raises(SystemExit):
        config.ensure_weights()


def test_parser():
    args = main.build_parser().parse_args(["-i", "x"])
    assert (args.outdir, args.device, args.step_size) == ("segmentations", "cuda", 0.5)
    with pytest.raises(SystemExit):
        main.build_parser().parse_args([])


def test_colliding_inputs_are_rejected(tmp_path):
    """case.nii and case.nii.gz would write the same output, so refuse."""
    (tmp_path / "case.nii").write_bytes(b"")
    (tmp_path / "case.nii.gz").write_bytes(b"")
    with pytest.raises(SystemExit):
        inference.infer(tmp_path, tmp_path / "out")
