python benchmarks/diffusion/quantization_quality.py \
        --use-offline-quant \
        --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
        --model-quant-checkpoint /vllm-omni/wan22-i2v-modelopt-fp8\
        --task i2v \
        --quantization fp8 \
        --prompts \
            "An astronaut riding a horse across the surface of Mars, red dust swirling, cinematic wide shot." \
            "A skateboarder doing a kickflip in an urban plaza, slow motion, golden hour lighting." \
        --height 720 --width 1280 \
        --image /vllm-omni/images/ \
        --output-dir ./wan22-i2v-quant-bench-output \
        --num-frames 81 --num-inference-steps 40 --seed 42 --vae-use-tiling \


python benchmarks/diffusion/quantization_quality.py \
        --use-offline-quant \
        --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v \
        --model-quant-checkpoint /vllm-omni/hv15-720p-t2v-modelopt-fp8\
        --task t2v \
        --quantization fp8 \
        --prompts \
            "An astronaut riding a horse across the surface of Mars, red dust swirling, cinematic wide shot." \
            "A skateboarder doing a kickflip in an urban plaza, slow motion, golden hour lighting." \
        --height 720 --width 1280 \
        --image /vllm-omni/images/ \
        --output-dir ./hv15-720p-t2v-quant-bench-output \
        --num-frames 81 --num-inference-steps 40 --seed 42 --vae-use-tiling \

python benchmarks/diffusion/quantization_quality.py \
        --use-offline-quant \
        --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v \
        --model-quant-checkpoint /vllm-omni/hv15-480p-i2v-modelopt-fp8\
        --task i2v \
        --quantization fp8 \
        --prompts \
            "An astronaut riding a horse across the surface of Mars, red dust swirling, cinematic wide shot." \
            "A skateboarder doing a kickflip in an urban plaza, slow motion, golden hour lighting." \
        --height 720 --width 1280 \
        --image /vllm-omni/images/ \
        --output-dir ./hv15-480p-i2v-quant-bench-output \
        --num-frames 81 --num-inference-steps 40 --seed 42 --vae-use-tiling \

python benchmarks/diffusion/quantization_quality.py \
        --use-offline-quant \
        --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v \
        --model-quant-checkpoint /vllm-omni/hv15-720p-i2v-modelopt-fp8\
        --task i2v \
        --quantization fp8 \
        --prompts \
            "An astronaut riding a horse across the surface of Mars, red dust swirling, cinematic wide shot." \
            "A skateboarder doing a kickflip in an urban plaza, slow motion, golden hour lighting." \
        --height 720 --width 1280 \
        --image /vllm-omni/images/ \
        --output-dir ./hv15-720p-i2v-quant-bench-output \
        --num-frames 81 --num-inference-steps 40 --seed 42 --vae-use-tiling \