"""OpenVINO path: a small nnU-Net-style network must segment like PyTorch.

Skipped when the optional OpenVINO dependency is not installed.
"""
import numpy as np
import pytest
import torch

ov = pytest.importorskip("openvino")
unet = pytest.importorskip("dynamic_network_architectures.architectures.unet")

from casimir_ov.ov_network import OpenVINONetwork, resolve_precision


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


def test_openvino_matches_torch_argmax(tmp_path):
    net = _small_unet()
    x = torch.from_numpy(np.random.default_rng(0).standard_normal((1, 1, 64, 48)).astype(np.float32))
    with torch.no_grad():
        ref = net(x).numpy()

    xml = tmp_path / "small.xml"
    ov.save_model(ov.convert_model(net, example_input=x), str(xml), compress_to_fp16=False)

    ov_net = OpenVINONetwork(xml, device="CPU", allow_fallback=False)
    assert ov_net.config["INFERENCE_PRECISION_HINT"] == "f32"
    # nnU-Net mirrors tiles, so the wrapper must accept non-contiguous tensors
    out = ov_net(torch.flip(x, (2,))).numpy()
    with torch.no_grad():
        ref_flip = net(torch.flip(x, (2,))).numpy()

    assert out.shape == ref_flip.shape == ref.shape
    assert (out.argmax(1) == ref_flip.argmax(1)).mean() >= 0.999
    np.testing.assert_allclose(out, ref_flip, atol=1e-3, rtol=1e-3)
