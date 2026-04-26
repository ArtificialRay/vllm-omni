# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for FuseFP8AdaLayerNorm.

Compares the fused CUDA path (``forward()`` -> ``torch.ops.vllm.fuse_fp8_adalayernorm``
-> Triton kernel) against the PyTorch reference (``forward_native()``).

Both paths produce ``torch.float8_e4m3fn`` outputs. Differences come from:
  * fp32 vs bf16 round-trip in the affine step (kernel keeps fp32 throughout;
    native goes through bf16 after F.layer_norm).
  * single-pass naive var (kernel) vs F.layer_norm's internal reduction.
Either source can flip near-boundary fp8 elements by 1-2 levels.
"""

import pytest
import torch

pytestmark = [
    pytest.mark.diffusion,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA + Triton required"),
]


_FP8_DTYPE = torch.float8_e4m3fn


def _make_inputs(B: int, T: int, D: int, *, seed: int = 0, device: str = "cuda"):
    """Mimic Wan2.2 AdaLN inputs: bf16 hidden states with [B, 1, D] modulation."""
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(B, T, D, device=device, dtype=torch.bfloat16, generator=g)
    scale = torch.randn(B, 1, D, device=device, dtype=torch.bfloat16, generator=g) * 0.1
    shift = torch.randn(B, 1, D, device=device, dtype=torch.bfloat16, generator=g) * 0.1
    return x, scale, shift


def _assert_fp8_close(
    y_fused: torch.Tensor,
    y_native: torch.Tensor,
    *,
    max_mismatch_frac: float = 0.05,

) -> None:
    """Two fp8 outputs should agree on most elements; mismatches must stay
    within fp8 quantization noise (1-2 levels at bin boundaries).
    """
    y_kf, y_rf = y_fused.float(), y_native.float()
    diff_mask = y_kf != y_rf
    frac = diff_mask.float().mean().item()
    assert frac <= max_mismatch_frac, (
        f"{frac * 100:.3f}% of fp8 elements differ "
        f"(budget {max_mismatch_frac * 100:.1f}%); "
        f"max |Δ|={(y_kf - y_rf).abs().max().item()}"
    )

    if diff_mask.any():
        rel = (y_kf - y_rf).abs() / y_rf.abs().clamp_min(0.015625) # fp8 minimum subnormal: 0.015625
        rel_max = rel[diff_mask].max().item()
        assert rel_max < 0.251, (
            f"fp8 mismatches exceed quantization noise: max rel {rel_max:.4f} "
            "(budget 0.251)"
        )


@pytest.mark.parametrize(
    "B,T,D,input_scale",
    [
        (1, 128, 1024, 0.5),    # tiny
        (2, 256, 3072, 0.5),    # Wan2.2 TI2V-5B hidden dim
        (1, 64, 5120, 0.4),     # Wan2.2 I2V-14B hidden dim
        (4, 32, 3072, 0.6),     # batched
    ],
)
def test_forward_matches_native(B: int, T: int, D: int, input_scale: float):
    """forward() (Triton fused) vs forward_native() (fp32 reference)."""
    from vllm_omni.diffusion.layers.fuse_fp8_adalayernorm import FuseFP8AdaLayerNorm

    layer = FuseFP8AdaLayerNorm(
        hidden_size=D, eps=1e-6, elementwise_affine=False
    ).cuda()
    x, scale, shift = _make_inputs(B, T, D)
    input_scale_t = torch.tensor(input_scale, device=x.device, dtype=torch.float32)

    y_fused = layer.forward(x, scale, shift, input_scale_t)
    y_native = layer.forward_native(x, scale, shift, input_scale_t)

    assert y_fused.shape == (B, T, D)
    assert y_fused.dtype == _FP8_DTYPE
    assert y_native.shape == (B, T, D)
    assert y_native.dtype == _FP8_DTYPE

    _assert_fp8_close(y_fused, y_native)


def test_zero_input_is_finite():
    """All-zero input -> output must be finite (no inf/NaN from div-by-zero)."""
    from vllm_omni.diffusion.layers.fuse_fp8_adalayernorm import FuseFP8AdaLayerNorm

    D = 1024
    layer = FuseFP8AdaLayerNorm(
        hidden_size=D, eps=1e-6, elementwise_affine=False
    ).cuda()
    x = torch.zeros(1, 8, D, device="cuda", dtype=torch.bfloat16)
    scale = torch.zeros(1, 1, D, device="cuda", dtype=torch.bfloat16)
    shift = torch.zeros(1, 1, D, device="cuda", dtype=torch.bfloat16)
    input_scale = torch.tensor(0.5, device="cuda", dtype=torch.float32)

    y_fused = layer.forward(x, scale, shift, input_scale)
    y_native = layer.forward_native(x, scale, shift, input_scale)
    assert torch.isfinite(y_fused.float()).all()
    assert torch.isfinite(y_native.float()).all()


def test_torch_op_registered():
    """Sanity: the direct_register_custom_op side-effect ran at import time."""
    import vllm_omni.diffusion.layers.fuse_fp8_adalayernorm  # noqa: F401

    assert hasattr(torch.ops.vllm, "fuse_fp8_adalayernorm")