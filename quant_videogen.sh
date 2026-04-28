# 1 GPU task
# per-tensor:
python examples/quantization/quantize_hunyuanvideo_15_modelopt_fp8.py \
        --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v \
        --output ./hv15-720p-t2v-modelopt-fp8 \
        --overwrite \

python examples/quantization/quantize_hunyuanvideo_15_modelopt_fp8.py \
        --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v \
        --variant i2v \
        --reference-images /vllm-omni/reference-images \
        --output ./hv15-480p-i2v-modelopt-fp8 \
        --overwrite \

python examples/quantization/quantize_hunyuanvideo_15_modelopt_fp8.py \
        --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v \
        --variant i2v \
        --reference-images /vllm-omni/reference-images \
        --output ./hv15-720p-i2v-modelopt-fp8 \
        --overwrite \

python examples/quantization/quantize_wan2_2_vace_modelopt_fp8.py \
    --model Wan-AI/Wan2.1-VACE-1.3B-diffusers \
    --output ./wan21-vace-1.3b-fp8 \
    --overwrite \

python examples/quantization/quantize_wan2_2_vace_modelopt_fp8.py \
        --model Wan-AI/Wan2.1-VACE-14B-diffusers \
        --output ./wan21-vace-14b-fp8 \
        --reference-images /vllm-omni/reference-images \
        --overwrite \

# per-block: 
python examples/quantization/quantize_wan2_2_modelopt_fp8.py \
        --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
        --output ./wan22-ti2v-modelopt-fp8-per-block \
        --overwrite --weight-block-size '128,128'\

python examples/quantization/quantize_hunyuanvideo_15_modelopt_fp8.py \
        --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v \
        --output ./hv15-720p-t2v-modelopt-fp8-per-block \
        --overwrite --weight-block-size '128,128'\

python examples/quantization/quantize_hunyuanvideo_15_modelopt_fp8.py \
        --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v \
        --variant i2v \
        --reference-images /vllm-omni/reference_images \
        --output ./hv15-480p-i2v-modelopt-fp8-per-block \
        --overwrite --weight-block-size '128,128'\

python examples/quantization/quantize_hunyuanvideo_15_modelopt_fp8.py \
        --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v \
        --variant i2v \
        --reference-images /vllm-omni/reference_images \
        --output ./hv15-720p-i2v-modelopt-fp8-per-block \
        --overwrite --weight-block-size '128,128'\

python examples/quantization/quantize_wan2_2_vace_modelopt_fp8.py \
    --model Wan-AI/Wan2.1-VACE-1.3B-diffusers \
    --output ./wan21-vace-1.3b-fp8 \
    --overwrite --weight-block-size '128,128'\

python examples/quantization/quantize_wan2_2_vace_modelopt_fp8.py \
        --model Wan-AI/Wan2.1-VACE-14B-diffusers \
        --output ./wan21-vace-14b-fp8 \
        --reference-images /path/to/ref_images/ \
        --overwrite --weight-block-size '128,128'\

# 2 GPU task
# per-tensor:
python examples/quantization/quantize_wan2_2_modelopt_fp8.py \
            --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
            --output ./wan22-t2v-modelopt-fp8 \
            --calib-boundary-ratio 0.5 \
            --overwrite \

python examples/quantization/quantize_wan2_2_modelopt_fp8.py \
        --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
        --output ./wan22-i2v-modelopt-fp8 \
        --is-i2v --reference-images /path/to/ref_images/ \
        --calib-boundary-ratio 0.5 \
        --overwrite \

# per-block:
python examples/quantization/quantize_wan2_2_modelopt_fp8.py \
            --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
            --output ./wan22-t2v-modelopt-fp8-per-block \
            --calib-boundary-ratio 0.5 \
            --overwrite --weight-block-size '128,128'\

python examples/quantization/quantize_wan2_2_modelopt_fp8.py \
        --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
        --output ./wan22-i2v-modelopt-fp8-per-block \
        --is-i2v --reference-images /path/to/ref_images/ \
        --calib-boundary-ratio 0.5 \
        --overwrite --weight-block-size '128,128'\