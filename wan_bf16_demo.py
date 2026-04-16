from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(model='Wan-AI/Wan2.2-TI2V-5B-Diffusers')
params = OmniDiffusionSamplingParams(num_inference_steps=5)  # 5 steps for fast test
out = omni.generate('a cat sitting on a table', params)
print('BF16 baseline: OK, output type =', type(out))
print(out)
