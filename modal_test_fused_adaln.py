"""Run the fused AdaLN + FP8 unit tests on a Modal H100.

Usage:
    modal run modal_test_fused_adaln.py

Mounts the local working tree directly — no git push required to iterate.
"""

import sys

import modal

app = modal.App("vllm-omni-fused-adaln-test")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.12")
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "UTC"})
    .apt_install("git", "curl")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({"PATH": "/root/.local/bin:$PATH"})
    # Kernel test only needs torch + triton. aenum is imported by
    # vllm_omni/patch.py at package import time.
    .run_commands(
        "uv pip install --system torch triton pytest aenum --torch-backend cu124"
    )
    .add_local_dir(".", remote_path="/workspace", ignore=[".git", "**/__pycache__"])
)


@app.function(image=image, gpu="H100", timeout=900)
def run_tests() -> int:
    import subprocess

    # --noconftest: skip the heavy tests/conftest.py (pulls in vllm/requests).
    # -v (not -xv): show all failures, don't stop on first.
    result = subprocess.run(
        ["python", "-m", "pytest", "-v", "--noconftest",
         "tests/diffusion/layers/test_fused_adaln_fp8.py"],
        cwd="/workspace",
        env={"PYTHONPATH": "/workspace", "PATH": "/usr/local/bin:/usr/bin:/bin"},
    )
    return result.returncode


@app.function(image=image, gpu="H100", timeout=900)
def run_benchmark() -> int:
    import subprocess

    result = subprocess.run(
        ["python", "benchmarks/bench_fused_adaln_fp8.py"],
        cwd="/workspace",
        env={"PYTHONPATH": "/workspace", "PATH": "/usr/local/bin:/usr/bin:/bin"},
    )
    return result.returncode


@app.local_entrypoint()
def main(mode: str = "test") -> None:
    """mode: 'test' (default) or 'bench'."""
    if mode == "bench":
        returncode = run_benchmark.remote()
    else:
        returncode = run_tests.remote()
    if returncode != 0:
        print(f"\nExit code {returncode}")
        sys.exit(returncode)
