# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused AdaLN + FP8 quantization kernels (Hopper, SM90).

Two variants, both producing ``float8_e4m3fn`` output:

* ``fused_adaln_fp8``           -- per-token (one fp32 scale per row).
* ``fused_adaln_fp8_per_group`` -- per-group (one fp32 scale per G-wide group within a row).

Per-token is the default activation scheme for vLLM's ``Fp8LinearMethod`` /
``cutlass_scaled_mm``. Per-group (typically G=128) matches DeepSeek-V3-style
block-wise W8A8 and vLLM's ``per_token_group_quant_fp8`` path.
"""

from __future__ import annotations
import logging
import torch
import triton
import triton.language as tl
logger = logging.getLogger(__name__)
# Resolve FP8 range from the target dtype (handles CUDA e4m3fn = 448.0 and
# ROCm e4m3fnuz = 224.0 uniformly).
_FP8_DTYPE = torch.float8_e4m3fn
_FP8_INFO = torch.finfo(_FP8_DTYPE)
FP8_MIN: tl.constexpr = _FP8_INFO.min
FP8_MAX: tl.constexpr = _FP8_INFO.max


# ──────────────────────────────────────────────────────────────────────────
# Per-token variant
# ──────────────────────────────────────────────────────────────────────────


@triton.jit
def _fused_adaln_fp8_per_token_kernel(
    X_ptr,                      # pointer to the input activations,              shape [N, D], bf16
    SCALE_ptr,                  # pointer to AdaLN scale (pre-incremented 1 + scale_msa), shape [N, D], bf16
    SHIFT_ptr,                  # pointer to AdaLN shift (shift_msa),            shape [N, D], bf16
    Y_ptr,                      # pointer to the quantized output,               shape [N, D], float8_e4m3fn,         shape [N],    fp32
    stride_x_n,                 # how much to increase X_ptr when moving by 1 row
    stride_x_d,                 # how much to increase X_ptr when moving by 1 column (typically 1)
    stride_scale_n,             # how much to increase SCALE_ptr when moving by 1 row
    stride_scale_d,             # how much to increase SCALE_ptr when moving by 1 column
    stride_shift_n,             # how much to increase SHIFT_ptr when moving by 1 row
    stride_shift_d,             # how much to increase SHIFT_ptr when moving by 1 column
    stride_y_n,                 # how much to increase Y_ptr when moving by 1 row
    stride_y_d,                 # how much to increase Y_ptr when moving by 1 column
    input_scale,                # input scale for pre-tensor quant
    D: tl.constexpr,            # number of columns in X (hidden dim) -- the reduction axis
    EPS: tl.constexpr,          # epsilon added to variance to avoid division by zero
    FP8_MIN: tl.constexpr,      # lower clamp bound (-448.0 for e4m3fn, -224.0 for e4m3fnuz)
    FP8_MAX: tl.constexpr,      # upper clamp bound (+448.0 for e4m3fn, +224.0 for e4m3fnuz)
    BLOCK_SIZE: tl.constexpr,   # tile size along D; streaming version uses BLOCK_SIZE <= D
):
    """One program per row (= one token).

    Steps (all reductions in fp32):
        1. ``pid = tl.program_id(0)``; compute row base pointers with
           ``pid.to(tl.int64) * stride`` -- int64 to avoid overflow on
           large activation tensors (B*T*D can exceed 2^31 for video).
        2. Load ``x, scale, shift`` rows with ``mask = cols < D``; cast to fp32.
        3. LayerNorm: ``mean, var, x_norm = (x - mean) * rsqrt(var + EPS)``.
        4. Affine: ``y = x_norm * scale + shift``. (scale is pre-incremented)
        5. Per-row absmax with eps-floor (branchless zero-guard):
               ``absmax = tl.maximum(tl.max(tl.abs(y)), EPS)``
               ``s = absmax / FP8_MAX``
        6. Clamp then cast (prevents ``inf`` on boundary values):
               ``y_q = tl.clamp(y / s, FP8_MIN, FP8_MAX).to(Y_ptr.dtype.element_ty)``
        7. Store ``y_q`` [D] with mask; store ``s`` [1].
    """
    # One program per row. BLOCK_SIZE = next_power_of_2(D), so the whole
    # row fits in a single tile -- no streaming loop needed.
    # DEBUG: if this fused kernel really runs
    if not hasattr(fused_adaln_fp8,"_fired"):
        logger.info("Fused AdaLN+FP8 kernel fired for the first time!")
        fused_adaln_fp8._fired = True
    row = tl.program_id(0).to(tl.int64)
    X_ptr += row * stride_x_n
    SCALE_ptr += row * stride_scale_n
    SHIFT_ptr += row * stride_shift_n
    Y_ptr += row * stride_y_n

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < D

    # 1. Load x row, cast to fp32 for stable reductions.
    x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # 2. LayerNorm (single-pass, whole row in SRAM).
    #    Masked positions carry 0 so they don't bias mean / variance.
    mean = tl.sum(x, axis=0) / D
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + EPS)
    x_norm = diff * rstd

    # 3. AdaLN affine. SCALE already holds (1 + scale_msa).
    s = tl.load(SCALE_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(SHIFT_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x_norm * s + b

    # # 4. Per-token absmax with eps-floor (branchless zero-guard).
    # absmax = tl.maximum(tl.max(tl.abs(y), axis=0), EPS)
    # y_scale = absmax * (1.0 / FP8_MAX)

    # 4. Clamp then cast -- clamp guards against values that round up past
    #    FP8_MAX and would otherwise become +inf in the fp8 cast.
    y_q = tl.clamp(y / input_scale, FP8_MIN, FP8_MAX).to(Y_ptr.dtype.element_ty)

    # 6. Store quantized row + per-token scale.
    tl.store(Y_ptr + cols, y_q, mask=mask)
    #tl.store(YSCALE_ptr + row, y_scale)


# ──────────────────────────────────────────────────────────────────────────
# Per-group variant (group_size along D)
# ──────────────────────────────────────────────────────────────────────────


@triton.jit
def _fused_adaln_fp8_per_group_kernel(
    X_ptr,                      # pointer to the input activations,              shape [N, D], bf16
    SCALE_ptr,                  # pointer to AdaLN scale (pre-incremented 1 + scale_msa), shape [N, D], bf16
    SHIFT_ptr,                  # pointer to AdaLN shift (shift_msa),            shape [N, D], bf16
    Y_ptr,                      # pointer to the quantized output,               shape [N, D], float8_e4m3fn
    YSCALE_ptr,                 # pointer to the per-group scale output,         shape [N, NUM_GROUPS], fp32 (row-major)
    stride_x_n,                 # how much to increase X_ptr when moving by 1 row
    stride_x_d,                 # how much to increase X_ptr when moving by 1 column (typically 1)
    stride_scale_n,             # how much to increase SCALE_ptr when moving by 1 row
    stride_scale_d,             # how much to increase SCALE_ptr when moving by 1 column
    stride_shift_n,             # how much to increase SHIFT_ptr when moving by 1 row
    stride_shift_d,             # how much to increase SHIFT_ptr when moving by 1 column
    stride_y_n,                 # how much to increase Y_ptr when moving by 1 row
    stride_y_d,                 # how much to increase Y_ptr when moving by 1 column
    stride_ys_n,                # how much to increase YSCALE_ptr when moving by 1 row
    stride_ys_g,                # how much to increase YSCALE_ptr when moving by 1 group (typically 1)
    D: tl.constexpr,                # number of columns in X (hidden dim) -- the reduction axis
    NUM_GROUPS: tl.constexpr,       # real number of groups along D (= D // GROUP_SIZE)
    NUM_GROUPS_PADDED: tl.constexpr,# = BLOCK_SIZE // GROUP_SIZE; may exceed NUM_GROUPS when D < BLOCK_SIZE
    GROUP_SIZE: tl.constexpr,       # width of each group in elements (power of 2, e.g. 128)
    EPS: tl.constexpr,              # epsilon added to variance to avoid division by zero
    FP8_MIN: tl.constexpr,          # lower clamp bound (-448.0 for e4m3fn, -224.0 for e4m3fnuz)
    FP8_MAX: tl.constexpr,          # upper clamp bound (+448.0 for e4m3fn, +224.0 for e4m3fnuz)
    BLOCK_SIZE: tl.constexpr,       # next_power_of_2(D); tail [D, BLOCK_SIZE) is padded with zeros
):
    """One program per row. LayerNorm + affine across the full row (fp32),
    then G-wide groups each get their own absmax/scale/cast.

    Implementation sketch:
        * Same LayerNorm + affine as the per-token kernel produces ``y`` of
          shape ``[BLOCK_SIZE]`` in fp32.
        * Reshape to 2D: ``y2d = tl.reshape(y, (NUM_GROUPS, GROUP_SIZE))``.
        * Per-group absmax with eps-floor:
              ``absmax = tl.maximum(tl.max(tl.abs(y2d), axis=1), EPS)``  # [NUM_GROUPS]
              ``s = absmax / FP8_MAX``
        * Clamp-then-cast with broadcast:
              ``y_q = tl.clamp(y2d / s[:, None], FP8_MIN, FP8_MAX).to(Y_ptr.dtype.element_ty)``
        * Store flattened ``y_q`` [D] and ``s`` [NUM_GROUPS].
        * Use ``pid.to(tl.int64)`` on strides.
    """
    # One program per row. BLOCK_SIZE = next_power_of_2(D), so the whole
    # row fits in a single tile -- no streaming loop needed.
    row = tl.program_id(0).to(tl.int64)
    X_ptr += row * stride_x_n
    SCALE_ptr += row * stride_scale_n
    SHIFT_ptr += row * stride_shift_n
    Y_ptr += row * stride_y_n

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < D

    # 1. Load x row, cast to fp32 for stable reductions.
    x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # 2. LayerNorm (single-pass, whole row in SRAM).
    #    Masked positions carry 0 so they don't bias mean / variance.
    mean = tl.sum(x, axis=0) / D
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + EPS)
    x_norm = diff * rstd

    # 3. AdaLN affine. SCALE already holds (1 + scale_msa).
    s = tl.load(SCALE_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(SHIFT_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x_norm * s + b

    # 4. Per-group quantization. When D is not a power of 2, BLOCK_SIZE > D and
    #    the tile has NUM_GROUPS_PADDED >= NUM_GROUPS logical groups after reshape;
    #    the tail groups [NUM_GROUPS, NUM_GROUPS_PADDED) sit over the masked-zero
    #    pad region and are discarded at store time.
    y2d = tl.reshape(y, (NUM_GROUPS_PADDED, GROUP_SIZE))  # [NUM_GROUPS_PADDED, GROUP_SIZE] fp32
    absmax = tl.max(tl.abs(y2d), axis=1)                  # [NUM_GROUPS_PADDED]             fp32
    absmax = tl.maximum(absmax, EPS)                      # [NUM_GROUPS_PADDED]             fp32
    group_scale = absmax * (1.0 / FP8_MAX)                # [NUM_GROUPS_PADDED]             fp32
    y_q_2d = tl.clamp(y2d / group_scale[:, None], FP8_MIN, FP8_MAX).to(Y_ptr.dtype.element_ty)

    # 5. Store quantized row (tail is masked out via `mask = cols < D`) and
    #    only the first NUM_GROUPS scales (tail groups are garbage from pad).
    y_q = tl.reshape(y_q_2d, (BLOCK_SIZE,))
    tl.store(Y_ptr + cols, y_q, mask=mask)
    gcols = tl.arange(0, NUM_GROUPS_PADDED)
    gmask = gcols < NUM_GROUPS
    tl.store(YSCALE_ptr + row * stride_ys_n + gcols * stride_ys_g, group_scale, mask=gmask)

# ──────────────────────────────────────────────────────────────────────────
# Public launchers
# ──────────────────────────────────────────────────────────────────────────


def fused_adaln_fp8(
    x: torch.Tensor,          # [N, D] BF16
    scale: torch.Tensor,      # [N, D] BF16  (this is (1 + scale_msa), NOT scale_msa)
    shift: torch.Tensor,      # [N, D] BF16
    eps: float = 1e-6,
    input_scale: torch.Tensor | None = None, # optional pre-tensor quant scale; if None, the kernel computes its own per-token scales
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused AdaLN + per-token FP8 quantization.

    Returns:
        x_fp8:   [N, D] float8_e4m3fn.
        x_scale: [N]    float32. Dequant: ``x_fp8.float() * x_scale[:, None]``.
    """
    _check_inputs(x, scale, shift)
    N, D = x.shape

    x_fp8 = torch.empty((N, D), dtype=_FP8_DTYPE, device=x.device)

    BLOCK_SIZE = triton.next_power_of_2(D)
    num_warps = _pick_num_warps(BLOCK_SIZE)

    _fused_adaln_fp8_per_token_kernel[(N,)](
        x, scale, shift, x_fp8,
        x.stride(0), x.stride(1),
        scale.stride(0), scale.stride(1),
        shift.stride(0), shift.stride(1),
        x_fp8.stride(0), x_fp8.stride(1),
        D=D, EPS=eps,
        FP8_MIN=FP8_MIN, FP8_MAX=FP8_MAX,
        BLOCK_SIZE=BLOCK_SIZE,
        input_scale=input_scale,
        num_warps=num_warps,
    )
    return x_fp8


def fused_adaln_fp8_per_group(
    x: torch.Tensor,          # [N, D] BF16
    scale: torch.Tensor,      # [N, D] BF16  (1 + scale_msa)
    shift: torch.Tensor,      # [N, D] BF16
    eps: float = 1e-6,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused AdaLN + per-group FP8 quantization along the D dimension.

    Args:
        group_size: D-axis group width. Must divide D. Typical: 128.

    Returns:
        x_fp8:   [N, D]              float8_e4m3fn.
        x_scale: [N, D // group_size] float32.
            Dequant: ``x_fp8.float() * x_scale.repeat_interleave(group_size, dim=-1)``.
    """
    _check_inputs(x, scale, shift)
    N, D = x.shape
    if D % group_size != 0:
        raise ValueError(f"group_size={group_size} must divide D={D}")
    if group_size & (group_size - 1) != 0:
        raise ValueError(f"group_size={group_size} must be a power of 2")

    num_groups = D // group_size
    BLOCK_SIZE = triton.next_power_of_2(D)
    # When D is not a power of 2, BLOCK_SIZE > D and the tile contains
    # NUM_GROUPS_PADDED logical groups; the kernel discards the tail groups.
    num_groups_padded = BLOCK_SIZE // group_size

    x_fp8 = torch.empty((N, D), dtype=_FP8_DTYPE, device=x.device)
    x_scale = torch.empty((N, num_groups), dtype=torch.float32, device=x.device)

    num_warps = _pick_num_warps(BLOCK_SIZE)

    _fused_adaln_fp8_per_group_kernel[(N,)](
        x, scale, shift, x_fp8, x_scale,
        x.stride(0), x.stride(1),
        scale.stride(0), scale.stride(1),
        shift.stride(0), shift.stride(1),
        x_fp8.stride(0), x_fp8.stride(1),
        x_scale.stride(0), x_scale.stride(1),
        D=D, NUM_GROUPS=num_groups, NUM_GROUPS_PADDED=num_groups_padded,
        GROUP_SIZE=group_size,
        EPS=eps,
        FP8_MIN=FP8_MIN, FP8_MAX=FP8_MAX,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return x_fp8, x_scale


# ──────────────────────────────────────────────────────────────────────────
# PyTorch reference implementations (for unit tests / debugging)
# ──────────────────────────────────────────────────────────────────────────


def fused_adaln_fp8_reference(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ground truth for ``fused_adaln_fp8`` (per-token).

    Uses the same eps-floor and clamp-before-cast semantics as the Triton kernel
    so the unit test can compare bit-for-bit at boundaries.
    """
    y = _adaln_fp32(x, scale, shift, eps)                       # [N, D] fp32
    absmax = torch.clamp(y.abs().amax(-1), min=eps)             # [N]    eps-floor
    s = absmax / FP8_MAX
    y_q = torch.clamp(y / s.unsqueeze(-1), FP8_MIN, FP8_MAX).to(_FP8_DTYPE)
    return y_q, s.float()


def fused_adaln_fp8_per_group_reference(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-6,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ground truth for ``fused_adaln_fp8_per_group``."""
    N, D = x.shape
    assert D % group_size == 0
    num_groups = D // group_size

    y = _adaln_fp32(x, scale, shift, eps)                       # [N, D]
    y_grouped = y.view(N, num_groups, group_size)               # [N, G, Gsize]
    absmax = torch.clamp(y_grouped.abs().amax(-1), min=eps)     # [N, G] eps-floor
    s = absmax / FP8_MAX
    y_q = torch.clamp(y_grouped / s.unsqueeze(-1), FP8_MIN, FP8_MAX).to(_FP8_DTYPE).view(N, D)
    return y_q, s.float()


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _adaln_fp32(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float
) -> torch.Tensor:
    """LayerNorm + affine in fp32. ``scale`` is already ``(1 + scale_msa)``."""
    x_f32 = x.float()
    mean = x_f32.mean(-1, keepdim=True)
    var = x_f32.var(-1, keepdim=True, unbiased=False)
    x_norm = (x_f32 - mean) * torch.rsqrt(var + eps)
    return x_norm * scale.float() + shift.float()


def _check_inputs(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> None:
    assert x.is_cuda and x.dtype == torch.bfloat16, "x must be CUDA bf16"
    assert scale.shape == x.shape and shift.shape == x.shape, "scale/shift must match x shape"
    assert scale.dtype == torch.bfloat16 and shift.dtype == torch.bfloat16
    assert x.is_contiguous() and scale.is_contiguous() and shift.is_contiguous()
    assert x.ndim == 2, "pass [N, D]; flatten [B, T, D] -> [B*T, D] before calling"


def _pick_num_warps(block_d: int) -> int:
    if block_d <= 1024:
        return 4
    if block_d <= 4096:
        return 8
    return 16
