import modal

app = modal.App("vllm-omni-example")

image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.19.0")
    .apt_install("git", "sox", "libsox-fmt-all", "jq")
    .run_commands("git clone https://github.com/vllm-project/vllm-omni.git /vllm-omni")
    .run_commands("cd /vllm-omni && uv pip install --system -e .")
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)


@app.function(image=image, gpu="H100", volumes={"/root/.cache/huggingface": hf_cache_vol})
def f():
    import subprocess

    subprocess.run(
        "python wan_bf16_demo.py",
        check=True,
        shell=True,
    )
