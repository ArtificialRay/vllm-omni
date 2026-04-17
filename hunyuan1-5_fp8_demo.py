from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(model='hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v',
            quantization="fp8",
            cache_backend="tea_cache",
            cache_config={"rel_l1_thresh":0.2},
            )
params = OmniDiffusionSamplingParams(num_inference_steps=5)  # 5 steps for fast test
out = omni.generate('a cat sitting on a table', params)
print('BF16 baseline: OK, output type =', type(out))
print(out)
