# SPDX-License-Identifier: Apache-2.0
"""Fused AdaLN + FP8 quantization layer.

Two layers of registration:

* ``torch.ops.vllm.fuse_fp8_adalayernorm`` — registered via
  :func:`direct_register_custom_op`, opaque to ``torch._dynamo`` so the inner
  Triton kernel launch is hidden from ``torch.compile`` / CUDA-graph capture.
* :class:`FuseFP8AdaLayerNorm` — registered via
  ``@CustomOp.register("fuse_fp8_adalayernorm")`` so it joins vLLM's
  ``op_registry`` (enables OOT replacement).

CUDA fast path:
    out_fp8 = quantize(LayerNorm(x) * (1 + scale) + shift, input_scale)
"""

from __future__ import annotations

import torch
from vllm.model_executor.custom_op import CustomOp
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_omni.diffusion.layers.norm import LayerNorm

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_INFO = torch.finfo(_FP8_DTYPE)
_FP8_MIN = _FP8_INFO.min
_FP8_MAX = _FP8_INFO.max


def fuse_fp8_adalayernorm_impl(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    input_scale: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    from vllm_omni.diffusion.layers.fused_adaln_fp8 import fused_adaln_fp8

    orig_shape = x.shape
    D = orig_shape[-1]
    x = x.reshape(-1, D)
    # ``1 + scale`` happens on the unexpanded tensor (e.g. [B, 1, D]) to avoid
    # an extra B*T*D-sized add; expand then contiguous materializes the
    # broadcast view that the Triton kernel's ``is_contiguous()`` assert needs.
    scale = (1 + scale).expand(orig_shape).contiguous().view(-1, D)
    shift = shift.expand(orig_shape).contiguous().view(-1, D)
    out_2d = fused_adaln_fp8(x, scale, shift, eps, input_scale)
    return out_2d.view(*orig_shape[:-1], D)


def fuse_fp8_adalayernorm_fake(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    input_scale: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.empty_like(x, dtype=_FP8_DTYPE)


direct_register_custom_op(
    op_name="fuse_fp8_adalayernorm",
    op_func=fuse_fp8_adalayernorm_impl,
    mutates_args=[],
    fake_impl=fuse_fp8_adalayernorm_fake,
)


@CustomOp.register("fuse_fp8_adalayernorm")
class FuseFP8AdaLayerNorm(CustomOp):
    """Fused AdaLayerNorm + per-tensor static FP8 quantization.

    Output dtype is ``torch.float8_e4m3fn``; output shape matches ``x``.
    ``input_scale`` is a scalar fp32 tensor frozen during offline calibration.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.layernorm = LayerNorm(
            hidden_size, eps=eps, elementwise_affine=elementwise_affine
        )

    def forward(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
        input_scale: torch.Tensor,
    ) -> torch.Tensor:
        # Always go through the registered torch op so torch.compile / Dynamo
        # see an opaque boundary instead of the inner Triton kernel.
        return torch.ops.vllm.fuse_fp8_adalayernorm(
            x, scale, shift, input_scale, self.eps
        )

    def forward_cuda(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
        input_scale: torch.Tensor,
    ) -> torch.Tensor:
        return fuse_fp8_adalayernorm_impl(x, scale, shift, input_scale, self.eps)

    def forward_native(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
        input_scale: torch.Tensor,
    ) -> torch.Tensor:
        y = self.layernorm(x) * (1 + scale) + shift
        return (
            (y.float() / input_scale)
            .clamp(_FP8_MIN, _FP8_MAX)
            .to(_FP8_DTYPE)
        )

__all__ = ["FuseFP8AdaLayerNorm"]