# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Two-GPU parallel version of quantization_quality.py.

Runs the BF16 baseline on one GPU and the quantized model(s) on another GPU
concurrently, then computes LPIPS in the parent process on cuda:0 after both
workers exit. For the common case (BF16 vs a single quantization method) the
total wall time is roughly max(BF16, FP8) instead of BF16 + FP8.

Notes:
- Workers are spawned with start_method="spawn" and pin themselves to a GPU
  via CUDA_VISIBLE_DEVICES before any torch.cuda.* call.
- Outputs are written to disk by each worker; only metadata flows through the
  multiprocessing Queue. The parent reloads outputs from disk for LPIPS to
  avoid pickling large video tensors and to keep the LPIPS input lossless.
- Multiple quant methods are run sequentially on the quant GPU while the BF16
  GPU may sit idle after baseline finishes. This is intentional for
  simplicity; the primary use case is single-quant comparison.

Image example:
    python benchmarks/diffusion/quantization_quality_two_gpu.py \\
        --model Tongyi-MAI/Z-Image-Turbo --task t2i \\
        --quantization fp8 --prompts "a cup of coffee on the table" \\
        --height 1024 --width 1024 --num-inference-steps 50 --seed 42

Video example with offline quant:
    python benchmarks/diffusion/quantization_quality_two_gpu.py \\
        --use-offline-quant \\
        --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \\
        --model-quant-checkpoint vllm-omni/Wan2.2-T2V-A14B-Diffusers \\
        --task t2v --quantization fp8 \\
        --prompts "A serene lakeside sunrise with mist over the water" \\
        --height 720 --width 1280 \\
        --num-frames 81 --num-inference-steps 40 --seed 42
"""

import argparse
import gc
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch


# ----------------------------- LPIPS helpers ------------------------------
# Run only in the parent process, after both workers have exited.

def compute_lpips_images(
    baseline_images: list,
    quantized_images: list,
    net: str = "alex",
) -> list[float]:
    import lpips
    from torchvision import transforms

    loss_fn = lpips.LPIPS(net=net).eval()
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    scores = []
    for img_bl, img_qt in zip(baseline_images, quantized_images):
        t_bl = transform(img_bl.convert("RGB")).unsqueeze(0)
        t_qt = transform(img_qt.convert("RGB")).unsqueeze(0)
        if torch.cuda.is_available():
            t_bl, t_qt = t_bl.cuda(), t_qt.cuda()
        with torch.no_grad():
            score = loss_fn(t_bl, t_qt).item()
        scores.append(score)
    return scores


def compute_lpips_video(
    baseline_frames: np.ndarray,
    quantized_frames: np.ndarray,
    net: str = "alex",
) -> float:
    import lpips

    loss_fn = lpips.LPIPS(net=net).eval()
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()

    num_frames = min(len(baseline_frames), len(quantized_frames))
    scores = []
    for i in range(num_frames):
        f_bl = torch.from_numpy(baseline_frames[i]).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
        f_qt = torch.from_numpy(quantized_frames[i]).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
        if torch.cuda.is_available():
            f_bl, f_qt = f_bl.cuda(), f_qt.cuda()
        with torch.no_grad():
            score = loss_fn(f_bl, f_qt).item()
        scores.append(score)
    return float(np.mean(scores))


# --------------------------- Worker-side helpers --------------------------
# These are also imported by the parent module load, but only invoked
# inside worker processes (after CUDA_VISIBLE_DEVICES is pinned).

def _build_omni_kwargs(args, quantization=None):
    from vllm_omni.diffusion.data import DiffusionParallelConfig

    parallel_config = DiffusionParallelConfig(
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    if quantization:
        kwargs = {
            "model": args.model_quant_checkpoint if args.use_offline_quant else args.model,
            "parallel_config": parallel_config,
            "enforce_eager": args.enforce_eager,
            "quantization_config": quantization,
        }
    else:
        kwargs = {
            "model": args.model,
            "parallel_config": parallel_config,
            "enforce_eager": args.enforce_eager,
        }
    return kwargs


def _generate_image(omni, args, prompt, seed):
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.platforms import current_omni_platform

    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    outputs = omni.generate(
        {"prompt": prompt},
        OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            generator=generator,
            num_inference_steps=args.num_inference_steps,
        ),
    )
    elapsed = time.perf_counter() - start
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)

    first = outputs[0]
    req_out = first.request_output[0] if hasattr(first, "request_output") else first
    img = req_out.images[0]
    return img, elapsed, peak_mem


def _generate_video(omni, args, prompt, seed):
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.outputs import OmniRequestOutput
    from vllm_omni.platforms import current_omni_platform

    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    outputs = omni.generate(
        {"prompt": prompt, "negative_prompt": ""},
        OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            generator=generator,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            num_frames=args.num_frames,
        ),
    )
    elapsed = time.perf_counter() - start
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)

    first = outputs[0]
    if hasattr(first, "request_output") and isinstance(first.request_output, list):
        inner = first.request_output[0]
        if isinstance(inner, OmniRequestOutput) and hasattr(inner, "images"):
            frames = inner.images[0] if inner.images else None
        else:
            frames = inner
    elif hasattr(first, "images") and first.images:
        frames = first.images
    else:
        raise ValueError("Could not extract video frames from output.")

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
    return frames_array, elapsed, peak_mem


def _unload_omni(omni):
    del omni
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _save_output(out, is_video: bool, dst_dir: Path, prompt_idx: int, fps: int) -> Path:
    """Persist a single output. Returns the path that LPIPS should reload from."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    if is_video:
        # .npy is lossless and what LPIPS reads. Also export .mp4 for humans
        # when diffusers is available.
        npy_path = dst_dir / f"prompt_{prompt_idx}.npy"
        np.save(npy_path, out)
        try:
            from diffusers.utils import export_to_video

            frames_list = list(out) if isinstance(out, np.ndarray) and out.ndim == 4 else out
            export_to_video(frames_list, str(dst_dir / f"prompt_{prompt_idx}.mp4"), fps=fps)
        except ImportError:
            pass
        return npy_path
    png_path = dst_dir / f"prompt_{prompt_idx}.png"
    out.save(png_path)
    return png_path


def _worker_run_config(args, label, quantization, is_video, prompts, seed, output_dir, log_prefix):
    """Run all prompts for one config; save outputs; return metadata."""
    from vllm_omni.entrypoints.omni import Omni

    print(f"{log_prefix} building Omni  config={label}  quantization={quantization}", flush=True)
    omni = Omni(**_build_omni_kwargs(args, quantization=quantization))

    per_prompt = {}
    dst_dir = output_dir / label.replace(" ", "_")
    for i, prompt in enumerate(prompts):
        print(f"{log_prefix} [{label}] prompt {i}: {prompt[:60]}", flush=True)
        if is_video:
            out, t, mem = _generate_video(omni, args, prompt, seed)
        else:
            out, t, mem = _generate_image(omni, args, prompt, seed)
        path = _save_output(out, is_video, dst_dir, i, args.fps)
        per_prompt[i] = {"prompt": prompt, "path": str(path), "time": t, "mem": mem}

    avg_time = float(np.mean([v["time"] for v in per_prompt.values()]))
    # Use first prompt's peak-mem as representative, matching single-GPU script.
    peak_mem = per_prompt[0]["mem"]
    _unload_omni(omni)

    return {"label": label, "avg_time": avg_time, "peak_mem": peak_mem, "per_prompt": per_prompt}


def worker_main(gpu_id, role, args, configs, prompts, seed, output_dir, result_queue):
    """Entry point for a child process.

    role:    "baseline" or "quant" — used as the result key.
    configs: list of (label, quantization). For baseline pass [("baseline", None)].
             For quant pass [(method, method), ...] (sequenced on this GPU).
    """
    # Pin to one physical GPU before any torch.cuda.* call. `import torch` at
    # module load does not create a CUDA context, so this still takes effect.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_prefix = f"[GPU{gpu_id} {role}]"
    is_video = args.task == "t2v"

    try:
        results = []
        for label, quantization in configs:
            print(f"{log_prefix} ===== running {label} =====", flush=True)
            r = _worker_run_config(
                args, label, quantization, is_video, prompts, seed, output_dir, log_prefix
            )
            results.append(r)
        result_queue.put({"role": role, "ok": True, "results": results})
    except Exception as e:
        tb = traceback.format_exc()
        print(f"{log_prefix} FAILED: {e}\n{tb}", flush=True)
        result_queue.put({"role": role, "ok": False, "error": str(e), "traceback": tb})


# ----------------------------- Coordinator --------------------------------

def _load_image(path: str):
    from PIL import Image

    return Image.open(path).convert("RGB")


def _load_video(path: str) -> np.ndarray:
    return np.load(path)


def _render_markdown(args, is_video, bl_avg_time, bl_mem, all_results, wall):
    model = args.model_quant_checkpoint if args.use_offline_quant else args.model
    lines = []
    lines.append(
        f"## Quantization Quality Benchmark — {model.split('/')[-1]} (two-GPU parallel)"
    )
    lines.append(
        f"Setup: {args.height}x{args.width}, {args.num_inference_steps} steps, "
        f"seed={args.seed}, LPIPS ({args.lpips_net})"
    )
    if is_video:
        lines.append(f"Video: {args.num_frames} frames")
    lines.append(f"Wall time (parallel): {wall:.1f}s")
    lines.append("")
    lines.append("### Summary")
    lines.append("")
    lines.append("| Config | Avg Time | Speedup | Memory (GiB) | Mem Reduction | Mean LPIPS |")
    lines.append("|--------|----------|---------|--------------|---------------|------------|")
    lines.append(
        f"| BF16 baseline | {bl_avg_time:.2f}s | 1.00x | {bl_mem:.2f} | — | (ref) |"
    )
    for r in all_results:
        lines.append(
            f"| {r['config']} | {r['avg_time']:.2f}s | {r['speedup']:.2f}x "
            f"| {r['memory_gib']:.2f} | {r['mem_reduction_pct']:.0f}% "
            f"| {r['mean_lpips']:.4f} |"
        )
    lines.append("")
    lines.append("> LPIPS < 0.01 = imperceptible, > 0.1 = clearly noticeable.")
    lines.append("")

    if len(args.prompts) > 1:
        lines.append("### Per-Prompt LPIPS")
        lines.append("")
        header = "| Prompt |"
        sep = "|--------|"
        for r in all_results:
            header += f" {r['config']} |"
            sep += "--------|"
        lines.append(header)
        lines.append(sep)
        for i, prompt in enumerate(args.prompts):
            short = prompt[:50] + "..." if len(prompt) > 50 else prompt
            row = f"| {short} |"
            for r in all_results:
                row += f" {r['per_prompt'][i]['lpips']:.4f} |"
            lines.append(row)
        lines.append("")

    return "\n".join(lines)


def run_benchmark(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    is_video = args.task == "t2v"

    if torch.cuda.device_count() < 2:
        raise RuntimeError(
            f"two-GPU script requires >= 2 visible CUDA devices, found "
            f"{torch.cuda.device_count()}. Use quantization_quality.py for single-GPU runs."
        )
    if args.bf16_gpu == args.quant_gpu:
        raise ValueError("--bf16-gpu and --quant-gpu must differ.")

    bl_configs = [("baseline", None)]
    qt_configs = [(method, method) for method in args.quantization]

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    p_bl = ctx.Process(
        target=worker_main,
        args=(args.bf16_gpu, "baseline", args, bl_configs, args.prompts, args.seed,
              output_dir, result_queue),
    )
    p_qt = ctx.Process(
        target=worker_main,
        args=(args.quant_gpu, "quant", args, qt_configs, args.prompts, args.seed,
              output_dir, result_queue),
    )

    print("=" * 70)
    print(
        f"Launching workers: baseline -> GPU {args.bf16_gpu}, "
        f"quant ({len(qt_configs)} method(s)) -> GPU {args.quant_gpu}"
    )
    print("=" * 70, flush=True)

    t0 = time.perf_counter()
    p_bl.start()
    p_qt.start()

    # Drain queue first so workers don't block on the feeder, then join.
    received = {}
    for _ in range(2):
        msg = result_queue.get()
        received[msg["role"]] = msg

    p_bl.join()
    p_qt.join()
    wall = time.perf_counter() - t0
    print(f"\nBoth workers finished in {wall:.1f}s wall time.\n", flush=True)

    if not received["baseline"]["ok"]:
        raise RuntimeError(f"baseline worker failed:\n{received['baseline']['traceback']}")
    if not received["quant"]["ok"]:
        raise RuntimeError(f"quant worker failed:\n{received['quant']['traceback']}")

    bl = received["baseline"]["results"][0]
    qt_results_list = received["quant"]["results"]

    # ---- Compute LPIPS in the parent on cuda:0 ----
    print("=" * 70)
    print("Computing LPIPS on cuda:0 ...")
    print("=" * 70, flush=True)

    bl_loaded = {}
    for i in range(len(args.prompts)):
        path = bl["per_prompt"][i]["path"]
        bl_loaded[i] = _load_video(path) if is_video else _load_image(path)

    bl_avg_time = bl["avg_time"]
    bl_mem = bl["peak_mem"]

    all_results = []
    for r in qt_results_list:
        per_prompt = []
        for i, prompt in enumerate(args.prompts):
            qt_path = r["per_prompt"][i]["path"]
            qt_data = _load_video(qt_path) if is_video else _load_image(qt_path)
            if is_video:
                lpips_score = compute_lpips_video(bl_loaded[i], qt_data, net=args.lpips_net)
            else:
                lpips_score = compute_lpips_images(
                    [bl_loaded[i]], [qt_data], net=args.lpips_net
                )[0]
            per_prompt.append({"prompt": prompt, "lpips": lpips_score})

        mean_lpips = float(np.mean([p["lpips"] for p in per_prompt]))
        speedup = bl_avg_time / r["avg_time"] if r["avg_time"] > 0 else float("inf")
        mem_reduction = (bl_mem - r["peak_mem"]) / bl_mem * 100 if bl_mem > 0 else 0.0

        all_results.append(
            {
                "config": r["label"],
                "avg_time": r["avg_time"],
                "speedup": speedup,
                "memory_gib": r["peak_mem"],
                "mem_reduction_pct": mem_reduction,
                "mean_lpips": mean_lpips,
                "per_prompt": per_prompt,
            }
        )

    md = _render_markdown(args, is_video, bl_avg_time, bl_mem, all_results, wall)
    print("\n\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(md)

    results_path = output_dir / "results.md"
    results_path.write_text(md, encoding="utf-8")
    print(f"\nResults saved to {results_path}")
    print(f"Baseline outputs in {output_dir / 'baseline'}")
    for r in all_results:
        print(f"Quantized outputs in {output_dir / r['config'].replace(' ', '_')}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-GPU parallel quantization-quality benchmark for diffusion models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Model name or local path.")
    parser.add_argument("--model-quant-checkpoint", default=None,
                        help="Offline quantization model checkpoint.")
    parser.add_argument("--use-offline-quant", action="store_true",
                        help="Compare with an offline-quantized checkpoint.")
    parser.add_argument("--task", default="t2i", choices=["t2i", "t2v"])
    parser.add_argument("--quantization", nargs="+", required=True,
                        help="One or more quantization methods to benchmark (e.g. fp8 int8).")
    parser.add_argument("--prompts", nargs="+",
                        default=["a cup of coffee on the table"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--num-frames", type=int, default=81,
                        help="Number of video frames (t2v only).")
    parser.add_argument("--fps", type=int, default=24,
                        help="Video FPS for saving (t2v only).")
    parser.add_argument("--guidance-scale", type=float, default=4.0,
                        help="CFG scale (used for video).")
    parser.add_argument("--output-dir", type=str,
                        default="./quant_bench_output_two_gpu")
    parser.add_argument("--lpips-net", type=str, default="alex",
                        choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--ulysses-degree", type=int, default=1)
    parser.add_argument("--ring-degree", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--bf16-gpu", type=int, default=0,
                        help="Local CUDA index for the BF16 baseline worker.")
    parser.add_argument("--quant-gpu", type=int, default=1,
                        help="Local CUDA index for the quantized-model worker.")
    args = parser.parse_args()
    if args.use_offline_quant and not args.model_quant_checkpoint:
        parser.error("--use-offline-quant requires --model-quant-checkpoint")
    return args


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
