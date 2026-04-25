Plan: Fused AdaLN + FP8 Quantization Triton Kernel (Level 2)
Context
In the Wan2.2 denoising loop, every transformer block runs two AdaLN → linear sequences before writing attention / FFN outputs:

# Self-attention path (norm1):
x_normed = AdaLN(x, scale_msa, shift_msa)   ← writes BF16 to HBM
qkv, _ = to_qkv(x_normed)                   ← reads BF16, re-quantizes to FP8, runs GEMM

# FFN path (norm3):
x_normed = AdaLN(x, c_scale_msa, c_shift_msa)  ← same HBM round-trip
h = ffn.net_0(x_normed)                         ← same re-quantization
Each round-trip stores/reloads the full BF16 residual (5120 × seq_len × 2 bytes), and the FP8 quantization inside the linear is a redundant second pass over data that just left SRAM.

A fused Triton kernel computes layernorm(x) * (1 + scale) + shift and FP8 per-token quantization in a single SRAM pass, writing only FP8 output to HBM. The vLLM kernel already has the correct hook: FP8ScaledMMLinearKernel.apply_weights() skips quant_fp8() when x.dtype == fp8_dtype (line 142 of ScaledMMLinearKernel.py). But it still needs As (activation scale) supplied externally — we bypass apply_weights() entirely and call apply_scaled_mm() directly with our pre-computed scale.

Fused Computation
BEFORE (3 kernels, 2 HBM round-trips per path):
  x[BF16] → HBM → [AdaLN kernel] → x_normed[BF16] → HBM → [quant_fp8] → x_fp8[FP8] → [FP8 GEMM]

AFTER (2 kernels, 1 HBM read per path):
  x[BF16] → HBM → [Fused AdaLN+FP8 kernel] → x_fp8[FP8]+scale → [FP8 GEMM]
The Triton kernel performs two reductions per token, all in SRAM:

LN pass: mean → variance → normalize
Quant pass: abs_max(x_adaln) → scale → quantize to float8_e4m3fn
For Wan2.2: hidden_dim = 5120 = 5 × 1024 tokens. One BF16 row = 10KB — fits comfortably in H100 SRAM (256KB).

Files to Create / Modify
1. NEW: vllm_omni/diffusion/kernels/fused_adaln_fp8.py
Triton kernel + Python wrapper. Signature:

def fused_adaln_fp8(
    x: torch.Tensor,          # [N, D] BF16
    scale: torch.Tensor,      # [N, D] BF16  (this is (1 + scale_msa), NOT scale_msa)
    shift: torch.Tensor,      # [N, D] BF16
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    # returns: x_fp8 [N, D] float8_e4m3fn, x_scale [N] float32
The @triton.jit kernel does per-row:

1. Load x row from HBM (BF16 → FP32 in registers)
2. Compute mean = sum(x) / D
3. Compute var = sum((x - mean)^2) / D
4. x_norm = (x - mean) / sqrt(var + eps)
5. x_adaln = x_norm * scale_row + shift_row   (scale = 1+scale_msa, shift = shift_msa)
6. abs_max = max(abs(x_adaln))
7. quant_scale = abs_max / 448.0              (448 = FP8 max value)
8. x_fp8 = x_adaln / quant_scale  →  clamp  →  cast to tl.float8e4nv
9. Write x_fp8 row + quant_scale scalar to HBM
BLOCK_D = 1024, loop over blocks for D=5120. Use tl.float32 throughout, tl.float8e4nv for output.

Also provide fused_adaln_bf16 (same kernel, skip FP8 quantization step, write BF16) for non-FP8 mode, as a cleaner replacement for existing forward_native.

2. MODIFY: vllm_omni/diffusion/layers/adalayernorm.py
Add return_fp8: bool = False to AdaLayerNorm.__init__.

Override forward_cuda:

def forward_cuda(self, x, scale, shift):
    if self.return_fp8:
        # scale here must be (1 + scale_msa), shift is shift_msa
        return fused_adaln_fp8(x, scale, shift, eps=self.eps)  # → (x_fp8, x_scale)
    return fused_adaln_bf16(x, scale, shift, eps=self.eps)       # → x_normed BF16
Important: Caller convention changes. Currently WanTransformerBlock.forward() passes scale_msa and adds 1 + inside the norm. The Triton kernel receives (1 + scale_msa) to match current behavior. Update callers accordingly (see step 3).

Return type when return_fp8=True: tuple[torch.Tensor, torch.Tensor] (x_fp8, x_scale).

3. MODIFY: vllm_omni/diffusion/models/wan2_2/wan2_2_transformer.py
WanTransformerBlock.__init__ — detect FP8 mode and enable fusion:

from vllm.model_executor.layers.quantization.fp8 import Fp8Config
...
def __init__(self, ..., quant_config=None):
    self._use_fp8_fusion = isinstance(quant_config, Fp8Config)
    self.norm1 = AdaLayerNorm(dim, elementwise_affine=False, eps=eps,
                               return_fp8=self._use_fp8_fusion)
    self.norm3 = AdaLayerNorm(dim, elementwise_affine=False, eps=eps,
                               return_fp8=self._use_fp8_fusion)
    self.attn1 = WanSelfAttention(dim=dim, ..., quant_config=quant_config)
    self.ffn = WanFeedForward(dim=dim, ..., quant_config=quant_config)
WanTransformerBlock.forward() — pass the tuple directly:

# 1. Self-attention — norm1 may return (x_fp8, x_scale) when fp8_fusion enabled
scale_in = 1 + scale_msa   # pre-apply (1+) so kernel receives correct scale
norm_hidden_states = self.norm1(hidden_states, scale_in, shift_msa)
# norm_hidden_states is BF16 tensor OR (fp8_tensor, fp32_scale) tuple
attn_output = self.attn1(norm_hidden_states, rotary_emb, hidden_states_mask)
hidden_states = (hidden_states + attn_output * gate_msa).type_as(hidden_states)

# 3. FFN — same pattern for norm3
scale_in = 1 + c_scale_msa
norm_hidden_states = self.norm3(hidden_states, scale_in, c_shift_msa)
ff_output = self.ffn(norm_hidden_states)
WanSelfAttention.forward() — handle pre-quantized input:

def forward(self, hidden_states, rotary_emb=None, attn_mask=None):
    if isinstance(hidden_states, tuple):
        # FP8 fusion path: bypass to_qkv.forward(), call apply_scaled_mm directly
        x_fp8, x_scale = hidden_states
        layer = self.to_qkv
        x_2d = x_fp8.view(-1, x_fp8.shape[-1])
        output_shape = [*x_fp8.shape[:-1], layer.weight.shape[1]]
        qkv = layer.quant_method.fp8_linear.apply_scaled_mm(
            A=x_2d, B=layer.weight, out_dtype=torch.bfloat16,
            As=x_scale, Bs=layer.weight_scale, bias=layer.bias,
            output_shape=output_shape,
        )
    else:
        qkv, _ = self.to_qkv(hidden_states)
    # ... rest unchanged ...
WanFeedForward.forward() — same pattern for net_0.proj:

def forward(self, hidden_states):
    if isinstance(hidden_states, tuple):
        x_fp8, x_scale = hidden_states
        layer = self.net_0.proj          # ColumnParallelLinear
        x_2d = x_fp8.view(-1, x_fp8.shape[-1])
        output_shape = [*x_fp8.shape[:-1], layer.weight.shape[1]]
        hidden_states = layer.quant_method.fp8_linear.apply_scaled_mm(
            A=x_2d, B=layer.weight, out_dtype=torch.bfloat16,
            As=x_scale, Bs=layer.weight_scale, bias=layer.bias,
            output_shape=output_shape,
        )
        hidden_states = F.gelu(hidden_states, approximate=self.net_0.approximate)
    else:
        hidden_states = self.net_0(hidden_states)
    hidden_states = self.net_1(hidden_states)
    hidden_states = self.net_2(hidden_states)
    return hidden_states
Also add quant_config threading (prerequisite, if not already done):

WanSelfAttention.__init__: quant_config → to_qkv, to_out
WanCrossAttention.__init__: quant_config → Q/K/V projections, to_out
WanFeedForward.__init__: quant_config → net_0.proj, net_2
WanTransformerBlock.__init__: propagate quant_config to all sub-modules
WanTransformer3DModel.__init__: accept + pass quant_config to all blocks
pipeline_wan2_2.py:create_transformer_from_config and _create_transformer: pass od_config.quantization_config
Key Implementation Notes
scale convention: The current AdaLayerNorm.forward_native computes layernorm(x) * (1 + scale) + shift. The Triton kernel receives scale_in = 1 + scale_msa from the caller to avoid an extra addition in the kernel. The caller in WanTransformerBlock.forward() must be updated to pre-apply 1 +.

apply_scaled_mm guard: The pre-quantized path bypasses to_qkv.forward() entirely, so it only works when to_qkv.quant_method is an Fp8LinearMethod with a non-None fp8_linear. Add a runtime guard:

if not self._use_fp8_fusion or not hasattr(self.to_qkv.quant_method, 'fp8_linear'):
    qkv, _ = self.to_qkv(hidden_states)
Fallback: On non-CUDA or non-FP8 paths, return_fp8=False → forward_cuda calls fused_adaln_bf16 → returns BF16 tensor → existing code path runs unchanged.

Triton availability guard: Wrap kernel import with try: import triton; HAS_TRITON = True — fall back to forward_native if unavailable.

Verification
# 1. Numerical correctness (unit test)
x = torch.randn(16, 5120, dtype=torch.bfloat16, device="cuda")
scale_msa = torch.randn(16, 5120, dtype=torch.bfloat16, device="cuda")
shift_msa = torch.randn(16, 5120, dtype=torch.bfloat16, device="cuda")

# Reference
x_ref = F.layer_norm(x, [5120]) * (1 + scale_msa) + shift_msa
x_fp8_ref, x_scale_ref = ops.scaled_fp8_quant(x_ref, scale=None)

# Fused
from vllm_omni.diffusion.kernels.fused_adaln_fp8 import fused_adaln_fp8
x_fp8, x_scale = fused_adaln_fp8(x, 1 + scale_msa, shift_msa)

# FP8 has ~0.4% max relative error, tolerance ~2× FP8 ULP
torch.testing.assert_close(x_scale, x_scale_ref, rtol=1e-3, atol=1e-3)
torch.testing.assert_close(x_fp8.float(), x_fp8_ref.float(), rtol=5e-2, atol=5e-2)

# 2. End-to-end: enable FP8 on a Wan2.2 T2V model, run one inference step,
#    compare output tensors vs BF16 baseline (LPIPS < 0.