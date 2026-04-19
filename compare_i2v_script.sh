python benchmarks/diffusion/quantization_quality.py \
    --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
    --task i2v \
    --image input.jpg \
    --quantization fp8 \
    --prompts \
        "Cherry blossoms swaying gently in the breeze" \
    --height 480 --width 832 \
    --num-frames 81 --num-inference-steps 50 --seed 42