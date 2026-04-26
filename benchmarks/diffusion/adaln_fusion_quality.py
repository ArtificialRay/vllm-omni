# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Compare Wan2.2 FP8 inference with and without ``--use-fp8-adaln-fusion``.

Pre-requisite: a Wan2.2 checkpoint that has already been offline-quantized
with ``examples/quantization/quantize_wan2_2_modelopt_fp8.py`` (the
config.json under ``transformer/`` carries ModelOpt FP8 metadata, so
``quantization="fp8"`` is auto-upgraded to the ModelOpt FP8 path at load
time).

For each prompt this script runs the offline-FP8 model twice, sharing seed
and sampling params, and reports:

* per-prompt and average wall-clock time
* peak GPU memory
* speedup (no-fusion / fusion)
* LPIPS perceptual distance between the two videos (sanity check that
  the fused kernel does not silently change outputs)

Example:
    python benchmarks/diffusion/wan22_fp8_adaln_fusion_compare.py \\
        --model ./wan22-ti2v-modelopt-fp8 \\
        --prompts \\
            "A serene lakeside sunrise with mist over the water." \\
            "A cat walking across a wooden bridge in autumn." \\
        --height 704 --width 1280 \\
        --num-frames 49 --num-inference-steps 30 --guidance-scale 5.0 \\
        --output-dir ./wan22_adaln_fusion_bench

Output directory layout:
    <output-dir>/
        no_fusion/      # videos generated with use_fp8_adaln_fusion=False
        fusion/         # videos generated with use_fp8_adaln_fusion=True
        results.md      # Markdown summary table
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import torch


def _build_omni_kwargs(args: argparse.Namespace, *, use_fusion: bool) -> dict:
    from vllm_omni.diffusion.data import DiffusionParallelConfig

    parallel_config = DiffusionParallelConfig(
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        cfg_parallel_size=args.cfg_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    kwargs = {
        "model": args.model,
        "parallel_config": parallel_config,
        "enforce_eager": args.enforce_eager,
        "quantization": args.quantization,
        "use_fp8_adaln_fusion": use_fusion,
    }
    if args.boundary_ratio is not None:
        kwargs["boundary_ratio"] = args.boundary_ratio
    if args.flow_shift is not None:
        kwargs["flow_shift"] = args.flow_shift
    return kwargs


def _generate_video(omni, args: argparse.Namespace, prompt: str, seed: int):
    """Run a single generation and return (frames[F,H,W,C] in [0,1], seconds, GiB)."""
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.outputs import OmniRequestOutput
    from vllm_omni.platforms import current_omni_platform

    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(seed)

    sampling_kwargs = dict(
        height=args.height,
        width=args.width,
        generator=generator,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        num_frames=args.num_frames,
    )
    if args.guidance_scale_high is not None:
        sampling_kwargs["guidance_scale_2"] = args.guidance_scale_high

    request = {"prompt": prompt}
    if args.negative_prompt:
        request["negative_prompt"] = args.negative_prompt

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = omni.generate(request, OmniDiffusionSamplingParams(**sampling_kwargs))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    first = outputs[0] if isinstance(outputs, list) else outputs
    peak_mem_gib = getattr(first, "peak_memory_mb", 0.0) / 1024.0

    # Unwrap nested OmniRequestOutput / images / list / tuple structures.
    frames = first
    if isinstance(frames, OmniRequestOutput):
        if frames.is_pipeline_output and frames.request_output is not None:
            frames = frames.request_output
        if isinstance(frames, OmniRequestOutput) and frames.images:
            frames = frames.images[0] if len(frames.images) == 1 else frames.images
    if isinstance(frames, list) and frames:
        frames = frames[0]
    if isinstance(frames, tuple) and len(frames) == 2:
        frames = frames[0]
    if isinstance(frames, dict):
        frames = frames.get("frames") or frames.get("video")

    if isinstance(frames, torch.Tensor):
        video = frames.detach().cpu()
        if video.dim() == 5:
            video = video[0].permute(1, 2, 3, 0) if video.shape[1] in (3, 4) else video[0]
        elif video.dim() == 4 and video.shape[0] in (3, 4):
            video = video.permute(1, 2, 3, 0)
        if video.is_floating_point():
            video = video.clamp(-1, 1) * 0.5 + 0.5
        frames_array = video.float().numpy()
    else:
        frames_array = np.asarray(frames)
        if frames_array.ndim == 5:
            frames_array = frames_array[0]
        if np.issubdtype(frames_array.dtype, np.integer):
            frames_array = frames_array.astype(np.float32) / 255.0

    return frames_array, elapsed, peak_mem_gib


def _compute_lpips_video(a: np.ndarray, b: np.ndarray, net: str = "alex") -> float:
    """Mean per-frame LPIPS between two (F, H, W, C) videos in [0, 1]."""
    import lpips

    loss_fn = lpips.LPIPS(net=net).eval()
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()

    num_frames = min(len(a), len(b))
    scores = []
    for i in range(num_frames):
        f_a = torch.from_numpy(a[i]).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
        f_b = torch.from_numpy(b[i]).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
        if torch.cuda.is_available():
            f_a, f_b = f_a.cuda(), f_b.cuda()
        with torch.no_grad():
            scores.append(loss_fn(f_a, f_b).item())
    return float(np.mean(scores))


def _save_video(frames: np.ndarray, path: Path, fps: int) -> None:
    try:
        from diffusers.utils import export_to_video

        frames_list = list(frames) if isinstance(frames, np.ndarray) and frames.ndim == 4 else frames
        export_to_video(frames_list, str(path), fps=fps)
    except ImportError:
        np.save(path.with_suffix(".npy"), frames)


def _unload(omni) -> None:
    del omni
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _run_config(args: argparse.Namespace, *, use_fusion: bool, label: str):
    from vllm_omni.entrypoints.omni import Omni

    print("\n" + "=" * 60)
    print(f"Running config: {label} (use_fp8_adaln_fusion={use_fusion})")
    print("=" * 60)

    omni = Omni(**_build_omni_kwargs(args, use_fusion=use_fusion))

    # Optional warmup pass on the first prompt to absorb compilation /
    # autotune costs so the timed runs reflect steady-state performance.
    if args.warmup and args.prompts:
        print(f"  [warmup] {args.prompts[0][:60]}...")
        _generate_video(omni, args, args.prompts[0], args.seed)

    per_prompt = []
    for i, prompt in enumerate(args.prompts):
        seed = args.seed + i
        print(f"  [{i + 1}/{len(args.prompts)}] {prompt[:60]}...  (seed={seed})")
        frames, elapsed, peak_mem = _generate_video(omni, args, prompt, seed)
        per_prompt.append({"prompt": prompt, "frames": frames, "time": elapsed, "mem": peak_mem})
        print(f"    -> {elapsed:.2f}s, peak {peak_mem:.2f} GiB")

    _unload(omni)
    return per_prompt


def run_benchmark(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    no_fusion_dir = output_dir / "no_fusion"
    fusion_dir = output_dir / "fusion"
    no_fusion_dir.mkdir(parents=True, exist_ok=True)
    fusion_dir.mkdir(parents=True, exist_ok=True)

    # Order is configurable so users can isolate any cold-cache / warmup
    # advantage that the first run may have on a shared GPU.
    if args.run_order == "fusion_first":
        first = _run_config(args, use_fusion=True, label="FP8 + AdaLN fusion")
        second = _run_config(args, use_fusion=False, label="FP8 (no fusion)")
        fusion_runs, no_fusion_runs = first, second
    else:
        first = _run_config(args, use_fusion=False, label="FP8 (no fusion)")
        second = _run_config(args, use_fusion=True, label="FP8 + AdaLN fusion")
        no_fusion_runs, fusion_runs = first, second

    # Save videos.
    for i, run in enumerate(no_fusion_runs):
        _save_video(run["frames"], no_fusion_dir / f"prompt_{i}.mp4", args.fps)
    for i, run in enumerate(fusion_runs):
        _save_video(run["frames"], fusion_dir / f"prompt_{i}.mp4", args.fps)

    # Per-prompt LPIPS (skip if the user disabled it - LPIPS pulls a small
    # network and can be noisy at low frame counts).
    lpips_scores: list[float | None] = []
    if args.compute_lpips:
        for nf, fu in zip(no_fusion_runs, fusion_runs):
            lpips_scores.append(_compute_lpips_video(nf["frames"], fu["frames"], net=args.lpips_net))
    else:
        lpips_scores = [None] * len(no_fusion_runs)

    nf_avg = float(np.mean([r["time"] for r in no_fusion_runs]))
    fu_avg = float(np.mean([r["time"] for r in fusion_runs]))
    nf_mem = max(r["mem"] for r in no_fusion_runs)
    fu_mem = max(r["mem"] for r in fusion_runs)
    speedup = nf_avg / fu_avg if fu_avg > 0 else float("inf")

    lines: list[str] = []
    lines.append(f"## Wan2.2 FP8 — AdaLN fusion vs no-fusion ({Path(args.model).name})")
    lines.append(
        f"Setup: {args.height}x{args.width}, {args.num_frames} frames, "
        f"{args.num_inference_steps} steps, guidance_scale={args.guidance_scale}, "
        f"seed={args.seed}, run_order={args.run_order}"
    )
    lines.append("")
    lines.append("### Summary")
    lines.append("")
    lines.append("| Config | Avg Time | Peak Mem (GiB) | Speedup |")
    lines.append("|--------|----------|----------------|---------|")
    lines.append(f"| FP8 (no fusion) | {nf_avg:.2f}s | {nf_mem:.2f} | 1.00x (ref) |")
    lines.append(f"| FP8 + AdaLN fusion | {fu_avg:.2f}s | {fu_mem:.2f} | {speedup:.2f}x |")
    lines.append("")

    lines.append("### Per-prompt detail")
    lines.append("")
    header = "| # | Prompt | No-fusion (s) | Fusion (s) | Speedup |"
    sep = "|---|--------|---------------|------------|---------|"
    if args.compute_lpips:
        header += " LPIPS |"
        sep += "-------|"
    lines.append(header)
    lines.append(sep)
    for i, (nf, fu) in enumerate(zip(no_fusion_runs, fusion_runs)):
        sp = nf["time"] / fu["time"] if fu["time"] > 0 else float("inf")
        short = nf["prompt"][:50] + ("..." if len(nf["prompt"]) > 50 else "")
        row = f"| {i} | {short} | {nf['time']:.2f} | {fu['time']:.2f} | {sp:.2f}x |"
        if args.compute_lpips:
            row += f" {lpips_scores[i]:.4f} |"
        lines.append(row)
    lines.append("")
    if args.compute_lpips:
        lines.append("> LPIPS < 0.01 = imperceptible, > 0.1 = clearly noticeable.")
        lines.append("> The fused kernel should be numerically close (low LPIPS); a large value")
        lines.append("> signals a bug in the fusion path.")
        lines.append("")

    md = "\n".join(lines)
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(md)

    results_path = output_dir / "results.md"
    results_path.write_text(md, encoding="utf-8")
    print(f"\nResults saved to {results_path}")
    print(f"  no-fusion videos: {no_fusion_dir}")
    print(f"  fusion videos:    {fusion_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare modelopt offline-FP8 inference with vs without --use-fp8-adaln-fusion.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to a modelopt offline checkpoint already produced by "
        "examples/quantization/quantize_wan2_2_modelopt_fp8.py.",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="fp8",
        choices=["fp8"],
        help="Quantization tag passed to Omni. The ModelOpt metadata in the checkpoint "
        "auto-upgrades 'fp8' to the ModelOpt FP8 adapter at runtime.",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=[
            "A serene lakeside sunrise with mist over the water.",
            "A cat walking across a wooden bridge in autumn.",
            "An astronaut riding a horse across the surface of Mars, cinematic wide shot.",
        ],
        help="One or more prompts to run through both configs.",
    )
    parser.add_argument("--negative-prompt", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=704, help="Wan2.2 TI2V-5B native height.")
    parser.add_argument("--width", type=int, default=1280, help="Wan2.2 TI2V-5B native width.")
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument(
        "--guidance-scale-high",
        type=float,
        default=None,
        help="Optional separate CFG for Wan2.2 high-noise stage.",
    )
    parser.add_argument("--boundary-ratio", type=float, default=None)
    parser.add_argument("--flow-shift", type=float, default=None)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--run-order",
        choices=["no_fusion_first", "fusion_first"],
        default="no_fusion_first",
        help="Which config to run first. Swap to check that ordering does not skew results.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run one untimed warmup pass per config before the timed prompts.",
    )
    parser.add_argument(
        "--compute-lpips",
        action="store_true",
        help="Also compute per-prompt LPIPS between the two videos. Requires `pip install lpips`.",
    )
    parser.add_argument("--lpips-net", choices=["alex", "vgg", "squeeze"], default="alex")
    parser.add_argument("--output-dir", type=str, default="./wan22_adaln_fusion_bench")
    parser.add_argument("--ulysses-degree", type=int, default=1)
    parser.add_argument("--ring-degree", type=int, default=1)
    parser.add_argument("--cfg-parallel-size", type=int, default=1, choices=[1, 2])
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
