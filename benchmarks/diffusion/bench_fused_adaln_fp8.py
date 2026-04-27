"""Benchmark fused AdaLN + FP8 kernel vs an unfused PyTorch baseline.

Baseline:  layer_norm(x) * (1+scale_msa) + shift_msa  -> bf16 intermediate
           -> per-token FP8 quant (absmax + cast)
Fused:     single kernel producing (x_fp8, x_scale)

Reports median ms per iter and effective HBM bandwidth (GB/s).
"""

import importlib.util
import pathlib

import torch
import triton

_HERE = pathlib.Path(__file__).resolve()
_KERNEL = _HERE.parents[2] / "vllm_omni" / "diffusion" / "layers" / "fused_adaln_fp8.py"
_spec = importlib.util.spec_from_file_location("fused_adaln_fp8", _KERNEL)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
fused_adaln_fp8 = _mod.fused_adaln_fp8
fused_adaln_fp8_per_group = _mod.fused_adaln_fp8_per_group

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def baseline_per_token(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor,input_scale:torch.Tensor,
                       eps: float = 1e-6):
    """Unfused reference: LayerNorm + affine in bf16, then FP8 quant."""
    D = x.shape[-1]
    y = torch.nn.functional.layer_norm(x, (D,), eps=eps) * scale + shift       # bf16
    y32 = y.float()                           # [N]
    s = (1/input_scale)
    y_q = (y32 * s.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return y_q, s.float()


def baseline_per_group(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor,
                       group_size: int = 128, eps: float = 1e-6):
    N, D = x.shape
    G = D // group_size
    y = torch.nn.functional.layer_norm(x, (D,), eps=eps) * scale + shift
    y32 = y.float().view(N, G, group_size)
    absmax = y32.abs().amax(-1).clamp_min(eps)
    s = absmax / FP8_MAX
    y_q = (y32 / s.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(N, D)
    return y_q, s.float()


def _hbm_bytes(N: int, D: int, num_groups: int) -> int:
    # Read: 3 * N*D * bf16(=2B). Write: N*D * fp8(=1B) + N*num_groups * fp32(=4B).
    return 3 * N * D * 2 + N * D * 1 + N * num_groups * 4


def bench_per_token(shapes):
    print("Per-token FP8")
    print(f"{'N':>7} {'D':>6} {'fused ms':>10} {'fused GB/s':>11} "
          f"{'base ms':>10} {'base GB/s':>11} {'speedup':>9}")
    for N, D,input_scale in shapes:
        x = torch.randn(N, D, device="cuda", dtype=torch.bfloat16)
        scale = torch.randn(N, D, device="cuda", dtype=torch.bfloat16) * 0.1 + 1.0
        shift = torch.randn(N, D, device="cuda", dtype=torch.bfloat16) * 0.1
        input_scale = torch.tensor(input_scale,device="cuda",dtype=torch.float32)
        t_f = triton.testing.do_bench(lambda: fused_adaln_fp8(x, scale, shift,input_scale=input_scale),
                                      warmup=25, rep=200)
        t_b = triton.testing.do_bench(lambda: baseline_per_token(x, scale, shift,input_scale=input_scale),
                                      warmup=25, rep=200)
        bytes_total = _hbm_bytes(N, D, num_groups=1)
        bw_f = bytes_total / (t_f * 1e-3) / 1e9
        bw_b = bytes_total / (t_b * 1e-3) / 1e9
        print(f"{N:>7} {D:>6} {t_f:>10.4f} {bw_f:>11.0f} "
              f"{t_b:>10.4f} {bw_b:>11.0f} {t_b / t_f:>8.2f}x")


def bench_per_group(shapes, group_size: int = 128):
    print(f"\nPer-group FP8 (group_size={group_size})")
    print(f"{'N':>7} {'D':>6} {'fused ms':>10} {'fused GB/s':>11} "
          f"{'base ms':>10} {'base GB/s':>11} {'speedup':>9}")
    for N, D in shapes:
        if D % group_size != 0:
            continue
        x = torch.randn(N, D, device="cuda", dtype=torch.bfloat16)
        scale = torch.randn(N, D, device="cuda", dtype=torch.bfloat16) * 0.1 + 1.0
        shift = torch.randn(N, D, device="cuda", dtype=torch.bfloat16) * 0.1

        t_f = triton.testing.do_bench(lambda: fused_adaln_fp8_per_group(x, scale, shift,
                                                                         group_size=group_size),
                                      warmup=25, rep=200)
        t_b = triton.testing.do_bench(lambda: baseline_per_group(x, scale, shift,
                                                                  group_size=group_size),
                                      warmup=25, rep=200)
        bytes_total = _hbm_bytes(N, D, num_groups=D // group_size)
        bw_f = bytes_total / (t_f * 1e-3) / 1e9
        bw_b = bytes_total / (t_b * 1e-3) / 1e9
        print(f"{N:>7} {D:>6} {t_f:>10.4f} {bw_f:>11.0f} "
              f"{t_b:>10.4f} {bw_b:>11.0f} {t_b / t_f:>8.2f}x")


if __name__ == "__main__":
    # Shapes cover Wan TI2V-5B (D=3072) and I2V-14B (D=5120) at several token counts.
    shapes = [
        (1024, 3072,0.5),     # small (few frames, low res)
        (8192, 3072,0.5),
        (65536, 3072,0.4),    # big video tile
        (150000, 3072,0.6),   # large video
        (1024, 5120,0.2),
        (65536, 5120,0.3),
    ]
    bench_per_token(shapes)
    #bench_per_group(shapes, group_size=128)
