
cd /vllm-omni/examples/offline_inference/image_to_video
wget https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg
# TI2V with Wan-AI/Wan2.2-TI2V-5B-Diffusers
python image_to_video.py \
  --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --image cherry_blossom.jpg \
  --prompt "Cherry blossoms swaying gently in the breeze, petals falling, smooth motion" \
  --negative-prompt "<optional quality filter>" \
  --height 480 \
  --width 832 \
  --num-frames 48 \
  --guidance-scale 4.0 \
  --num-inference-steps 40 \
  --flow-shift 12.0 \
  --fps 16 \
  --quantization "fp8" \
  --output i2v_output.mp4

# TI2V with Wan-AI/Wan2.2-TI2V-5B-Diffusers
# python image_to_video.py \
#   --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
#   --image test_image.jpg \
#   --prompt "图像生成视频示例" \
#   --negative-prompt "<optional quality filter>" \
#   --height 480 \
#   --width 832 \
#   --num-frames 48 \
#   --guidance-scale 4.0 \
#   --num-inference-steps 40 \
#   --flow-shift 12.0 \
#   --fps 16 \
#   --output i2v_output.mp4

# I2V with Wan-AI/Wan2.2-TI2V-5B-Diffusers-Video-Only
# python image_to_video.py \
#   --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
#   --image cherry_blossom.jpg \
#   --prompt "Cherry blossoms swaying gently in the breeze, petals falling, smooth motion" \
#   --negative-prompt "<optional quality filter>" \
#   --height 480 \
#   --width 832 \
#   --num-frames 48 \
#   --guidance-scale 5.0 \
#   --guidance-scale-high 6.0 \
#   --num-inference-steps 40 \
#   --boundary-ratio 0.875 \
#   --flow-shift 12.0 \
#   --fps 16 \
#   --output i2v_output.mp4