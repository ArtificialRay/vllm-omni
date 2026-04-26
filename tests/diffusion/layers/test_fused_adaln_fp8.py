# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the fused AdaLN + FP8 Triton kernels.

Compares the Triton kernels against their PyTorch references. Both sides use
the same eps-floor and clamp-before-cast semantics, so results should match
bit-for-bit.
"""

import pytest
import torch

pytestmark = [
    pytest.mark.diffusion,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


# Load the kernel module *by file path* — avoids going through
# vllm_omni/__init__.py which eagerly imports transformers etc.
# This keeps the unit test runnable in a torch+triton-only environment.
def _load():
    import importlib.util
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parents[3]
    kernel_path = repo_root / "vllm_omni" / "diffusion" / "layers" / "fused_adaln_fp8.py"
    spec = importlib.util.spec_from_file_location("fused_adaln_fp8", kernel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return (
        mod.fused_adaln_fp8,
        mod.fused_adaln_fp8_reference,
        mod.fused_adaln_fp8_per_group,
        mod.fused_adaln_fp8_per_group_reference,
    )


def _assert_fp8_tiles_close(y_k: torch.Tensor, y_r: torch.Tensor, *, max_mismatch_frac: float) -> None:
    """FP8 tiles are near-identical: differences come from 1-ULP scale drift pushing
    near-boundary values to the adjacent fp8 level. Permit a small mismatched fraction
    so long as each mismatch is at most one fp8 step (i.e. adjacent representable values).
    """
    y_kf, y_rf = y_k.float(), y_r.float()
    diff_mask = y_kf != y_rf
    frac = diff_mask.float().mean().item()
    if frac > max_mismatch_frac:
        raise AssertionError(
            f"{frac*100:.3f}% of fp8 elements differ (budget {max_mismatch_frac*100:.1f}%); "
            f"max |Δ|={ (y_kf - y_rf).abs().max().item() }"
        )
    # For mismatched elements, relative error must be small (~1 fp8 step at that magnitude).
    # fp8 e4m3 has ~8 representable levels per power-of-2 bin, so relative step ≈ 1/8 = 0.125.
    if diff_mask.any():
        rel = (y_kf - y_rf).abs() / y_rf.abs().clamp_min(1e-3)
        rel_mismatch = rel[diff_mask]
        assert rel_mismatch.max().item() < 0.2, (
            f"fp8 mismatches exceed 1 quantization step: max rel {rel_mismatch.max().item()}"
        )


def _make(N: int, D: int, *, seed: int = 0, device: str = "cuda"):
    """Realistic-ish inputs: bf16, scale ~ 1 + small noise, shift ~ small noise."""
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(N, D, device=device, dtype=torch.bfloat16, generator=g)
    scale = torch.randn(N, D, device=device, dtype=torch.bfloat16, generator=g) * 0.1 + 1.0
    shift = torch.randn(N, D, device=device, dtype=torch.bfloat16, generator=g) * 0.1
    return x, scale, shift


# ──────────────────────────────────────────────────────────────────────────
# Per-token
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "N,D,input_scale",
    [
        (1, 128,0.5),       # tiny
        (4, 3072,0.5),      # Wan 2.2 hidden dim, small batch
        (128, 3072,0.4),    # realistic token count
        (1024, 5120,0.4),   # larger hidden, many tokens
        (7, 130,0.6),       # non-power-of-2 D to exercise masking
    ],
)
def test_per_token_matches_reference(N: int, D: int, input_scale: float):
    fused_adaln_fp8, fused_adaln_fp8_reference, *_ = _load()
    x, s, b = _make(N, D)
    input_scale = torch.tensor(input_scale, device=x.device, dtype=torch.float32)
    y_k= fused_adaln_fp8(x, s, b, input_scale=input_scale)
    y_r= fused_adaln_fp8_reference(x, s, b, input_scale=input_scale)

    assert y_k.shape == (N, D) and y_k.dtype == torch.float8_e4m3fn
    # Quantized tiles may differ by at most 1 fp8 level on a small fraction
    # of near-boundary elements when the scale drifts by ~1 ULP.
    _assert_fp8_tiles_close(y_k, y_r, max_mismatch_frac=0.02)


def test_per_token_zero_row_eps_floor():
    """All-zero input -> reference absmax would be 0; eps-floor guards divide."""
    fused_adaln_fp8, *_ = _load()
    x = torch.zeros(2, 128, device="cuda", dtype=torch.bfloat16)
    s = torch.ones_like(x)
    b = torch.zeros_like(x)
    input_scale = torch.ones((1,),device=x.device,dtype=torch.float32)

    y = fused_adaln_fp8(x, s, b,input_scale=input_scale)
    assert torch.isfinite(y.float()).all(), "output should be finite (no inf from div-by-zero)"


def test_per_token_dequant_roundtrip():
    """dequant = x_fp8.float() * scale should be close to adaln(x) in fp32."""
    fused_adaln_fp8, fused_adaln_fp8_reference, *_ = _load()
    N, D = 32, 3072
    x, s, b = _make(N, D)
    input_scale = torch.rand((1,),device=x.device,dtype=torch.float32)
    y_k = fused_adaln_fp8(x, s, b,input_scale=input_scale)
    # Load _adaln_fp32 via the same importlib path as _load() to avoid
    # triggering vllm_omni package init.
    import importlib.util
    import pathlib
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "fused_adaln_fp8_mod",
        repo_root / "vllm_omni" / "diffusion" / "layers" / "fused_adaln_fp8.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    y_true = mod._adaln_fp32(x, s, b, eps=1e-6)            # [N, D] fp32
    y_dq = y_k.float() * input_scale.unsqueeze(-1)                # dequantized
    # FP8 e4m3 gives ~2-3 bits of mantissa; expect ~4% relative error on non-tiny values.
    rel_err = (y_dq - y_true).abs() / y_true.abs().clamp_min(1e-3)
    assert rel_err.median().item() < 0.08, f"median rel err {rel_err.median().item()}"


# ──────────────────────────────────────────────────────────────────────────
# Per-group
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "N,D,G",
    [
        (4, 3072, 128),      # Wan TI2V-5B hidden dim (non-pow2)
        (16, 2048, 128),     # pow2 baseline
        (64, 4096, 128),     # pow2 larger
        (8, 512, 64),        # pow2 smaller group size
        (4, 5120, 128),      # Wan I2V-14B hidden dim (non-pow2)
    ],
)
def test_per_group_matches_reference(N: int, D: int, G: int):
    *_, fused_adaln_fp8_per_group, fused_adaln_fp8_per_group_reference = _load()
    x, s, b = _make(N, D)

    y_k, sc_k = fused_adaln_fp8_per_group(x, s, b, group_size=G)
    y_r, sc_r = fused_adaln_fp8_per_group_reference(x, s, b, group_size=G)

    assert y_k.shape == (N, D) and y_k.dtype == torch.float8_e4m3fn
    assert sc_k.shape == (N, D // G) and sc_k.dtype == torch.float32
    torch.testing.assert_close(sc_k, sc_r, rtol=1e-5, atol=1e-7)
    _assert_fp8_tiles_close(y_k, y_r, max_mismatch_frac=0.02)


def test_per_group_rejects_bad_group_size():
    *_, fused_adaln_fp8_per_group, _ = _load()
    # group_size not dividing D
    x, s, b = _make(4, 2048)
    with pytest.raises(ValueError, match="must divide"):
        fused_adaln_fp8_per_group(x, s, b, group_size=96)

    # group_size divides D but is not a power of 2 (D = 96 * 32 = 3072)
    x, s, b = _make(4, 3072)
    with pytest.raises(ValueError, match="power of 2"):
        fused_adaln_fp8_per_group(x, s, b, group_size=96)
