import modal

app = modal.App("vllm-omni-example")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.12")
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "UTC"})
    .apt_install("wget", "git", "sox", "libsox-fmt-all", "jq", "curl")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({"PATH": "/root/.local/bin:$PATH"})
    .run_commands("uv pip install --system vllm==0.19.0 --torch-backend cu124")
    .run_commands("git clone https://github.com/ArtificialRay/vllm-omni.git /vllm-omni")
    .run_commands("cd /vllm-omni && uv pip install --system -e .")
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    volumes={"/root/.cache/huggingface": hf_cache_vol},
)
def f():
    import subprocess

    subprocess.run("cd /vllm-omni && git pull origin main", check=True, shell=True)
    subprocess.run("python /vllm-omni/wan_bf16_demo.py", check=True, shell=True)