"""OpenVINO path: a small nnU-Net-style network must segment like PyTorch.

Skipped when the optional OpenVINO dependency is not installed. No network access.
"""

import os

import numpy as np
import pytest
import torch

ov = pytest.importorskip("openvino")
unet = pytest.importorskip("dynamic_network_architectures.architectures.unet")

from casimir_ov.__main__ import build_parser
from casimir_ov.inference import ir_cache_dir
from casimir_ov.network import OpenVINONetwork, resolve_precision


def _small_unet(num_classes=21):
    torch.manual_seed(0)
    net = unet.PlainConvUNet(
        input_channels=1, n_stages=3, features_per_stage=(8, 16, 32),
        conv_op=torch.nn.Conv2d, kernel_sizes=3, strides=(1, 2, 2),
        n_conv_per_stage=2, num_classes=num_classes, n_conv_per_stage_decoder=2,
        conv_bias=True, norm_op=torch.nn.InstanceNorm2d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=torch.nn.LeakyReLU, nonlin_kwargs={"inplace": True},
    )
    return net.eval()


def test_precision_defaults():
    assert resolve_precision("CPU", "auto") == "f32"
    assert resolve_precision("GPU", "auto") == "f32"
    assert resolve_precision("GPU.0", "auto") == "f32"
    assert resolve_precision("NPU", "auto") is None
    assert resolve_precision("CPU", "default") is None
    assert resolve_precision("CPU", "bf16") == "bf16"


def test_parser():
    args = build_parser().parse_args(["-i", "x", "--device", "npu"])
    assert (args.outdir, args.device, args.precision, args.no_fallback) == (
        "segmentations", "NPU", "auto", False)
    with pytest.raises(SystemExit):
        build_parser().parse_args(["-i", "x", "--device", "mps"])


def test_ir_cache_is_per_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("CASIMIR_OPENVINO_CACHE", str(tmp_path / "cache"))
    dirs = []
    for content in (b"weights a", b"weights b"):
        weights = tmp_path / content.decode().replace(" ", "_")
        (weights / "fold_all").mkdir(parents=True)
        (weights / "fold_all" / "weights.pth").write_bytes(content)
        dirs.append(ir_cache_dir(weights))
    assert dirs[0].parent == dirs[1].parent == tmp_path / "cache"
    assert dirs[0] != dirs[1]


def test_ensure_ir_exports_once(tmp_path, monkeypatch):
    from casimir_ov import export

    calls = []

    def fake_export(weights_dir, outdir):
        calls.append(outdir)
        (outdir / export.IR_NAME).write_text("<net/>")
        (outdir / export.IR_NAME).with_suffix(".bin").write_bytes(b"")

    monkeypatch.setattr(export, "export_ir", fake_export)
    target = tmp_path / "cache" / "digest"
    xml = export.ensure_ir(tmp_path, target)
    assert xml == target / export.IR_NAME and xml.is_file()
    assert export.ensure_ir(tmp_path, target) == xml
    assert len(calls) == 1
    assert [p.name for p in target.parent.iterdir()] == ["digest"]
    if os.name == "posix":  # the cache keeps default permissions, not owner-only
        umask = os.umask(0)
        os.umask(umask)
        assert target.stat().st_mode & 0o777 == 0o777 & ~umask


def test_openvino_matches_torch_argmax(tmp_path):
    net = _small_unet()
    x = torch.from_numpy(
        np.random.default_rng(0).standard_normal((1, 1, 64, 48)).astype(np.float32))

    xml = tmp_path / "small.xml"
    ov.save_model(ov.convert_model(net, example_input=x), str(xml), compress_to_fp16=False)

    ov_net = OpenVINONetwork(xml, device="CPU", allow_fallback=False)
    assert ov_net.config["INFERENCE_PRECISION_HINT"] == "f32"
    # nnU-Net mirrors tiles, so the wrapper must accept non-contiguous tensors
    flipped = torch.flip(x, (2,))
    out = ov_net(flipped).numpy()
    with torch.no_grad():
        ref = net(flipped).numpy()

    assert out.shape == ref.shape
    assert (out.argmax(1) == ref.argmax(1)).mean() >= 0.999
    np.testing.assert_allclose(out, ref, atol=1e-3, rtol=1e-3)
