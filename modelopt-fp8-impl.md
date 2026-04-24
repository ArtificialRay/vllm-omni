Ready for review
Select text to add comments on the plan
分析：vllm-omni 如何使用 NVIDIA ModelOpt 对 Wan2.2 做离线 FP8 量化
Context
入口脚本 examples/quantization/quantize_wan2_2_modelopt_fp8.py 是 vllm-omni 中针对 Wan2.2 TI2V-5B 的离线静态 FP8 量化产物的生成器。它使用 NVIDIA ModelOpt 的 mtq.quantize 在视频扩散管线上做静态校准，然后把带有 FP8 张量 + scale 张量 + quantization_config 元数据的 diffusers 风格目录吐出来。运行时 vllm-omni 通过 vllm_omni/diffusion/model_loader/checkpoint_adapters/modelopt_fp8.py 这个 adapter 把这些产物消费掉，喂给已经在模型代码里完成 quant_config 穿透的 WanTransformer3DModel。

下面这份文档以入口脚本为主轴，把 producer 端（脚本）和 consumer 端（adapter + 模型代码）两条线索逐段串起来，并指出几处不显眼的「补丁/绕路」点。

1. 高层数据流总览
                         offline (this script)                                runtime (vllm-omni)
┌─────────────────────────────────────────────────────────────────┐    ┌────────────────────────────────────┐
│ 1. DiffusionPipeline.from_pretrained(BF16)                      │    │ DiffusersPipelineLoader.load_model │
│ 2. mtq.quantize(transformer, FP8_DEFAULT_CFG, forward_loop)     │    │   ├─ build OmniDiffusionConfig     │
│ 3. mtq.disable_quantizer(...)  # 排除 condition/proj/norm/MHA   │    │   ├─ ModelOptFp8Config             │
│ 4. _force_export_quantized_weights(...) # 真正写 FP8 权重         │    │   ├─ create_transformer_from_config│
│ 5. hide_quantizers_from_state_dict + save_pretrained             │───>│   │     (quant_config 穿到所有     │
│ 6. _patch_quant_config(transformer/config.json)                 │    │   │      ColumnParallelLinear /    │
│    => quant_method=modelopt, quant_algo=FP8, ignore=[...]       │    │   │      RowParallelLinear)         │
└─────────────────────────────────────────────────────────────────┘    │   └─ ModelOptFp8CheckpointAdapter   │
                                                                       │       .adapt(weights_iterator)      │
                                                                       │        ├─ resolve via WeightsMapper │
                                                                       │        │   (含 hf_to_vllm_mapper)   │
                                                                       │        ├─ scale 缓存/挂载             │
                                                                       │        └─ FP8→BF16 dequant 或透传    │
                                                                       └────────────────────────────────────┘
2. 入口脚本逐阶段拆解
参考 examples/quantization/quantize_wan2_2_modelopt_fp8.py。

2.1 输入/参数处理
必填：--model（原 BF16 diffusers 目录或 HF id）、--output（导出目录）
关键校准参数：--num-frames=49、--height=704、--width=1280、--calib-steps=10、--calib-size=8、--guidance-scale=5.0，跟 Wan2.2 TI2V-5B 推理基线对齐（49 帧 / 704×1280）。
默认值的取舍体现在 quantize_wan2_2_modelopt_fp8.py:60-74：注释里明说 17 帧是给显存紧的机器留的逃生口。
--quantize-mha 默认 关（quantize_wan2_2_modelopt_fp8.py:82-87）。注释提到 #2920 ablation 显示 Wan2.2 长 attention 序列上 FP8 的 K/V/softmax 量化会放大数值漂移，所以默认只量化 Linear，不动 attention 中间的 BMM。
--weight-block-size M,N 默认 None（per-tensor），脚本注释里同时提示 #2924 在追踪 adapter 的 block-wise dispatch 状态——这是个「能配但下游可能还消化不了」的实验旋钮。
2.2 加载管线 + 取出 backbone
quantize_wan2_2_modelopt_fp8.py:177-182：直接走 diffusers，丢到 cuda。注意整段都是用原始 diffusers 的 pipe.transformer（即 WanTransformer3DModel）作为量化对象——脚本不感知 vllm-omni 内部的 WanTransformer3DModel（那是运行时另一份实现）。这点很重要：离线校准跑的是 diffusers 命名空间的模型，所以导出来的 state-dict 里 FFN 是 ffn.net.0.proj.*；运行时 vllm-omni 里 FFN 是 ffn.net_0.proj.*，需要 mapper 修。（见 §5.2）

2.3 构造 ModelOpt 量化配置
quantize_wan2_2_modelopt_fp8.py:379-388：

quant_config = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
if weight_block_size is not None:
    quant_config["quant_cfg"]["*weight_quantizer"] = {
        "num_bits": (4, 3),  # E4M3
        "block_sizes": {-1: weight_block_size[1], -2: weight_block_size[0]},
    }
默认拿 ModelOpt 的 FP8_DEFAULT_CFG（per-tensor、E4M3、static）；只有指定 block size 时才覆写 *weight_quantizer。
block_sizes 的 -1/-2 指 Linear 权重最后两维（K、M）。
注意这里只调权重侧的 quantizer；*input_quantizer 仍走默认（per-tensor static）。
2.4 校准 forward loop
quantize_wan2_2_modelopt_fp8.py:185-217：

8 个英文 prompt（默认池），每个 10 步 latent-only 推理，固定 seed = args.seed + idx。
兼容性处理：pipe.guider.guidance_scale = ... 试着给新 diffusers API 用；调用 pipe(..., guidance_scale=...) 时如果接口不支持就吞掉 TypeError。这是 diffusers API 在 5.x 周期里漂移留下的痕迹，不是核心逻辑。
ModelOpt 的契约：mtq.quantize(model, cfg, forward_loop) 在内部会先把 quantizer 挂上，再调一次 forward_loop() 收 amax 统计，最后 freeze 出 static scale。脚本里 forward_loop 只跑视频生成本身，不返回任何东西，纯粹是「让模型再前向 N 次」。
2.5 mtq.quantize → 包装 + 校准
quantize_wan2_2_modelopt_fp8.py:391-394：

quantized = mtq.quantize(backbone, quant_config, forward_loop)
if quantized is not None:
    pipe.transformer = quantized
    backbone = quantized
ModelOpt 历史版本里 mtq.quantize 既可能 in-place 修改也可能返回新模块——脚本两手准备。返回后 pipe.transformer 已经是被「量化壳」包装过的 backbone，每个 Linear 层旁边多挂了 weight_quantizer / input_quantizer 两个伪算子，权重还在 BF16 上。

2.6 关掉「该全精度」的层
quantize_wan2_2_modelopt_fp8.py:152-174：

_filter_func_wan22(name)：用一个正则匹配出要保留全精度的层名：
proj_out.* | .*(condition_embedder|patch_embedding|norm_out|
                scale_shift_table|timestep_proj_prepare|
                output_scale_shift_prepare).*
这覆盖了：
condition_embedder：time + text + image 条件嵌入
patch_embedding：3D conv（不是 Linear，但带条 belt-and-suspenders 兜底）
scale_shift_table：AdaLN 调制参数（也不是 Linear）
norm_out / proj_out：最终归一化和输出投影
timestep_proj_prepare / output_scale_shift_prepare：sequence-parallel 辅助模块
_mha_filter_func(name)：匹配 q_bmm_quantizer/k_bmm_quantizer/v_bmm_quantizer/softmax_quantizer/bmm2_output_quantizer，默认调用 disable_quantizer 把 attention 的 BMM/softmax quantizer 关掉，保持 attention 计算在 BF16。
这里和后面 §2.8 的 ignore 列表是两套独立的「关掉量化」机制，必须双向一致：

这里 disable_quantizer 改的是内存里的量化壳（影响后面 _force_export_quantized_weights 还会不会把这些层导成 FP8）。
_wan22_quant_config_block 写到 config.json 的 ignore 数组（影响运行时是否把这些层当 FP8 装载）。
2.7 强制导出 FP8 权重（关键 workaround）
quantize_wan2_2_modelopt_fp8.py:240-272：

from modelopt.torch.export.quant_utils import (
    QUANTIZATION_NONE,
    get_quantization_format,
    quantizer_attr_names,
    weight_attr_names,
)
from modelopt.torch.export.unified_export_hf import _export_quantized_weight
ModelOpt 的「正经」导出 API export_hf_checkpoint 内置了一个 model-type registry，只有它认识的模型（HF transformers 那些 LLM）才会跑权重 FP8 序列化。Wan2.2 的 WanTransformer3DModel 不在表里，所以必须自己遍历模块，对每个量化过的 weight 调底层的 _export_quantized_weight(module, dtype, weight_name)，让权重从 BF16 真正转写成 torch.float8_e4m3fn 张量并把 weight_scale 挂上去。

跟 HunyuanVideo-1.5 / HunyuanImage-3 的同名 helper 是同一份代码（脚本的 docstring 明说参考自那两份，见 quantize_wan2_2_modelopt_fp8.py:243-247）。exported == 0 是错误兜底（quantize_wan2_2_modelopt_fp8.py:402-407）——如果一个权重都没导出来，要么是 disable 正则匹错了所有层，要么是 mtq.quantize 根本没挂上 quantizer。

2.8 落盘：复制源目录 + 单独 save 量化后的 transformer
quantize_wan2_2_modelopt_fp8.py:323-349：

shutil.copytree(src, output_dir, ignore=shutil.ignore_patterns("transformer", "transformer_2"))
with hide_quantizers_from_state_dict(pipe.transformer):
    pipe.transformer.save_pretrained(str(transformer_out), safe_serialization=True, max_shard_size="5GB")
保留 VAE、tokenizer、scheduler、model_index.json 等所有非 transformer 资产。
hide_quantizers_from_state_dict 是 ModelOpt 的上下文管理器，临时把 weight_quantizer/input_quantizer 子模块从 state_dict 里隐藏起来，保证 save 出来的 safetensors 只含真正的权重 + scale 张量，不要把 ModelOpt 的伪算子序列化进去。
TI2V-5B 单 transformer，所以只导一份；A14B（MoE）会有 transformer_2，本脚本明说不在范围内（quantize_wan2_2_modelopt_fp8.py:17-18）。
2.9 patch 元数据，让 vllm-omni adapter 能识别
quantize_wan2_2_modelopt_fp8.py:275-320 的 _wan22_quant_config_block + _patch_quant_config 往 transformer/config.json 注入：

{
  "quantization_config": {
    "quant_method": "modelopt",
    "quant_algo": "FP8",
    "producer": {"name": "modelopt"},
    "config_groups": {
      "group_0": {
        "input_activations": {"dynamic": false, "num_bits": 8, "type": "float"},
        "weights": {"dynamic": false, "num_bits": 8, "type": "float"},
        "targets": ["Linear"]
      }
    },
    "ignore": [
      "condition_embedder*", "norm_out*", "output_scale_shift_prepare*",
      "patch_embedding*", "proj_out*", "scale_shift_table*", "timestep_proj_prepare*"
    ]
  }
}
vllm-omni 的 ModelOptFp8Config 读这个块时关心的两个字段：

quant_method == "modelopt"：触发 adapter（modelopt_fp8.py:62-68）
is_checkpoint_fp8_serialized（由 vLLM 上游 ModelOptFp8Config 在解析 config_groups 时推导出来）：确认 weight 已经是 FP8 dtype，需要 dequant 流程
block-wise 时还会写 strategy: "block" + block_structure: "MxN"，由 adapter 的 _reshape_weight_scale 在 modelopt_fp8.py:143-153 消化。

3. ModelOpt 公开 API 用法清单
API	来源	在脚本中的用途
modelopt.torch.quantization.FP8_DEFAULT_CFG	mtq 包	per-tensor static FP8 模板
mtq.quantize(model, cfg, forward_loop)	mtq 包	挂 quantizer + 校准 + 冻结 scale
mtq.disable_quantizer(model, filter_fn)	mtq 包	关掉 condition/proj/norm/MHA 的量化壳
modelopt.torch.export.diffusers_utils.hide_quantizers_from_state_dict	export 子包	save_pretrained 时隐藏 quantizer 子模块
modelopt.torch.export.quant_utils.{get_quantization_format, quantizer_attr_names, weight_attr_names, QUANTIZATION_NONE}	export 子包内部工具	_force_export_quantized_weights 自己枚举量化层用的元数据
modelopt.torch.export.unified_export_hf._export_quantized_weight	私有 API	真正把 BF16 权重 cast 到 FP8 + 写 scale；正常路径 export_hf_checkpoint 才会调它
值得注意的是 _export_quantized_weight 带下划线，是 ModelOpt 的内部接口。脚本绕过 export_hf_checkpoint 直接调它，对 ModelOpt 版本敏感：未来 ModelOpt 重构 export 模块时这里会断。脚本里没有版本约束兜底，目前只能靠用户读 docstring 知道。

4. Adapter 端如何吃掉脚本的产物
源文件：vllm_omni/diffusion/model_loader/checkpoint_adapters/modelopt_fp8.py

4.1 触发条件
checkpoint_adapters/init.py:8-16 的 get_checkpoint_adapter 是分发入口，由 diffusers_loader.py:217-242 在每个 source（VAE/transformer/...）上调用一次。

is_compatible 三个 AND 条件（modelopt_fp8.py:46-68）：

use_safetensors=True
source 指向 transformer 子目录（subfolder == "transformer" 或 prefix 以 transformer. 开头）
quant_config 是 modelopt 且 is_checkpoint_fp8_serialized 为真
只要这三条都成立，weights iterator 就会被 adapter 包一层。

4.2 名称解析（含 hf_to_vllm_mapper 全树收集）
modelopt_fp8.py:90-124：

# Default packed-modules：把 to_q/to_k/to_v 三个分片名映射到 to_qkv
DEFAULT_PACKED_MODULES_MAPPING = {
    "to_qkv": ("to_q", "to_k", "to_v"),
    "add_kv_proj": ("add_q_proj", "add_k_proj", "add_v_proj"),
    "w13": ("w1", "w3"),
}
然后：

加上 .to_out.0. -> .to_out. 这条 diffusers attn 模块的标志性 substr remap。
遍历 model 全部子模块收集 hf_to_vllm_mapper 上的 orig_to_new_substr / orig_to_new_prefix。这一步是 commit 7e57b690 修复的：adapter 拿到的可能是整条 Pipeline，但 mapper 是定义在 WanTransformer3DModel 这种子模块上，必须 walk 才能收集到。
_resolve_target_name 的策略是先看是否能直接匹配 _loadable_tensors，匹配不上就过 mapper 的所有候选名，第一个能匹配到的就用。
4.3 Scale 张量与 FP8 权重的「重排序」处理
adapt(weights) 是个 generator，迭代 checkpoint 的 (name, tensor) 流（modelopt_fp8.py:258-278）：

for name, tensor in weights:
    target_name = self._resolve_target_name(name)
    if self._is_scale(name):                         # .input_scale / .weight_scale
        yield from self._handle_scale_tensor(...)    # 缓存 + flush pending
        continue
    target_dtype = self._target_dtype_for_dequantization(tensor, target_name)
    if target_dtype is not None:                     # checkpoint 是 FP8，model 期望 BF16
        tensor = self._maybe_dequantize_or_defer_weight(...)
        if tensor is None:
            continue                                 # 还没拿到对应 scale，先 defer
    yield name, tensor
为什么需要 defer：safetensors shard 的迭代顺序里，FP8 权重可能先于 weight_scale 出现。Adapter 用 pending_weights 队列把缺 scale 的权重挂起，等 scale 到达时（modelopt_fp8.py:172-179）一起 flush。

Dequant 公式（modelopt_fp8.py:156-170）：

weight = loaded_weight.to(torch.float32)
scale  = state.scale_tensors[scale_name].to(torch.float32, device=weight.device)
scale  = self._reshape_weight_scale(scale, loaded_weight.shape)
return (weight * scale).to(target_dtype)
如果 model 的 param 本身就是 FP8 dtype（_target_dtype_for_dequantization 在 modelopt_fp8.py:208-219 检测到这种情况），就 不 做 dequant，直接透传 FP8 权重，让运行时的 quantized Linear 自己用 scale。这种透传路径靠的是 quant_config 已经在模型构造期穿到 ColumnParallelLinear/RowParallelLinear，由它们的 weight loader 自己消费 weight_scale。

4.4 诊断日志（commit a5fb7890）
modelopt_fp8.py:189-203：

当 scale 张量名解析不到 target 时，前 3 条会以 WARNING 打印，并附上「同后缀的 loadable param 候选」做提示。这是给「checkpoint 里的命名跟 model param 命名对不上」这种 bug（典型例子：FFN 的 net.0 vs net_0，commit 4e6f5574 的根因）准备的诊断切入点。

如果有 weight 在迭代结束时还卡在 pending_weights 队列里（即整个 shard 流跑完都没等到对应 scale），_check_pending_weights 会直接 raise 报具体哪些 scale 缺（modelopt_fp8.py:239-245）——这是 hard failure，不会让模型带着错误权重启动。

5. 模型代码侧的 quant_config 穿透（commit 4047d1c0）
5.1 transformer 构造路径
WanTransformer3DModel.__init__ 在 vllm_omni/diffusion/models/wan2_2/wan2_2_transformer.py:858 接受 quant_config: "QuantizationConfig | None" = None。它向下传到：

WanTransformerBlock(...)（wan2_2_transformer.py:925-938），逐层都带 quant_config
block 内的 WanSelfAttention / WanCrossAttention / WanFeedForward（wan2_2_transformer.py:684-708）
各自再往下到 vLLM 的 QKVParallelLinear、ColumnParallelLinear、RowParallelLinear，并通过 prefix=... 把当前模块路径串好
显式不传 quant_config 的位置正是脚本里 _filter_func_wan22 对应的 7 个模块：

condition_embedder（wan2_2_transformer.py:915-922）
patch_embedding（wan2_2_transformer.py:907-912）
norm_out + proj_out（wan2_2_transformer.py:942-943）
scale_shift_table（在 block 内部，是 nn.Parameter，根本没法挂 quant）
timestep_proj_prepare + output_scale_shift_prepare（wan2_2_transformer.py:946-947）
这一致性是 三处保持同步 的产物：

模型代码里这些子模块构造时不接 quant_config
脚本的 _filter_func_wan22 关掉 ModelOpt 给它们挂的 quantizer
脚本的 _wan22_quant_config_block 把它们写进 ignore
任何一处漂移，要么 export 出 FP8 而 runtime 不接、要么 runtime 期望 FP8 而 export 没出，都会 hang 在 adapter 的 _check_pending_weights 或者 vLLM 的 weight loader 里。

5.2 FFN 命名约定与 mapper
WanFeedForward 用了非典型命名（wan2_2_transformer.py:153-167）：

self.net_0 = ColumnParallelGELU(...)
self.net_1 = nn.Identity()
self.net_2 = RowParallelLinear(...)
而 diffusers 原版 FFN 是 nn.Sequential，state-dict 里是 ffn.net.0.proj.*、ffn.net.2.*。所以 WanTransformer3DModel 在类级别声明（wan2_2_transformer.py:802-807）：

hf_to_vllm_mapper = WeightsMapper(
    orig_to_new_substr={
        ".ffn.net.0.": ".ffn.net_0.",
        ".ffn.net.2.": ".ffn.net_2.",
    },
)
这条 mapper 配合 §4.2 的 walk-submodules 收集逻辑，把 checkpoint 里的 blocks.N.ffn.net.0.proj.weight_scale 重映射到模型参数名 blocks.N.ffn.net_0.proj.weight_scale 上。脚本不需要感知这一点，它只管按 diffusers 原始命名 save；映射是 runtime 的责任。

历史教训（commit 4e6f5574）：在 mapper 引入前，约 120 个 FFN scale 张量都解析不到 target，fall through 给 _handle_scale_tensor 的 skip 路径，结果 FP8 推理是噪声但不报错。现在 §4.4 的诊断日志正是为了让这种漂移第一时间被看见。

5.3 Pipeline 层把 quant_config 接到 transformer
每个 Wan22 pipeline 都从 OmniDiffusionConfig.quantization_config 取 quant_config 并传给 create_transformer_from_config：

pipeline_wan2_2.py:376（T2V 基础）
pipeline_wan2_2_ti2v.py:205-207（这条是入口脚本默认目标 TI2V-5B 的运行路径）
pipeline_wan2_2_i2v.py:247-254（A14B MoE 还会建 transformer_2）
pipeline_wan2_2_vace.py:179（VACE 视频编辑）
脚本 docstring 里给的运行验证命令 quantize_wan2_2_modelopt_fp8.py:415-428 走的就是 Wan22TI2VPipeline 这条路，配合 --quantization fp8 由 vllm-omni 在解析 checkpoint 元数据时自动升格成 ModelOpt FP8。

6. 与同类脚本（HunyuanVideo-1.5）的对比
examples/quantization/ 下有三个相关文件：

quantize_wan2_2_modelopt_fp8.py（本文主角）
quantize_hunyuanvideo_15_modelopt_fp8.py（同模板的 HV-1.5 版本）
check_modelopt_fp8_export.py（验证脚本：检查 config.json 的 quantization_config、safetensors FP8 dtype、磁盘体积）
两个 quant 脚本是「模板共用，每模型只改 filter/ignore/校准超参」的关系：

共用	差异
_force_export_quantized_weights 实现完全一样	_filter_func_* 正则不同（HV 有 x_embedder/context_embedder/token_refiner/norm1.linear 等 13 项，Wan2.2 7 项含 SP helpers）
_mha_filter_func 正则相同，默认都关	校准帧数（HV 33 / Wan2.2 49）、guidance scale（HV 6.0 / Wan2.2 5.0）
hide_quantizers_from_state_dict + save_pretrained + _patch_quant_config 同一套	metadata 块的 ignore 列表不同
mtq.FP8_DEFAULT_CFG + *weight_quantizer block 覆写语法相同	入口模型类 / pipeline 类不同
memory 里那条「HunyuanVideo-1.5 transformer 没有 quant_config 穿透」的笔记是过时的：根据 explore agent 的二次确认，hunyuan_video_15_transformer.py:334 已经接 quant_config 并向下穿透到 attention/MLP/transformer block，时间点早于本文分析对象。建议本次任务结束后顺手清理这条 memory。

7. 关键设计决策与隐形约束（一句话清单）
离线脚本走原 diffusers 模型，不感知 vllm-omni 内部模型类——所以 state-dict 命名是 diffusers 风格，靠 runtime mapper 修。
静态 FP8 而非动态：input_activations.dynamic=false，校准期收的 amax 直接冻结成 scale。这就是「8 个 prompt × 10 步」必须真实代表生产分布的原因。
_force_export_quantized_weights 是绕开 ModelOpt 模型注册表的硬补丁，依赖私有 API _export_quantized_weight。脚本对 ModelOpt 版本敏感，没有 version pin 兜底。
三处 ignore/disable 必须严格一致（脚本 filter / 脚本 ignore JSON / 模型构造时是否传 quant_config），任何漂移都会让 adapter 加载阶段炸或者悄悄出噪声。
--quantize-mha 默认关：长视频序列上 K/V/softmax FP8 漂移大（#2920 ablation）。--weight-block-size 默认关：adapter 端 block-wise scale dispatch 状态待定（#2924）。
adapter 用 generator + pending queue 处理 scale 与 FP8 权重的乱序到达——不要假设 safetensors 流里 scale 一定先于权重出现。
Pipeline 整体作为 adapter 入参 + 全树 walk 收集 mapper（commit 7e57b690）是 adapter 拿不到「正确根」的兜底设计。
8. 验证清单（如何端到端跑通）
如果用户想亲手跑一遍验证整个链路：

跑离线脚本：

python examples/quantization/quantize_wan2_2_modelopt_fp8.py \
    --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --output ./wan22-ti2v-modelopt-fp8 \
    --overwrite
关注输出末尾的 Export summary，确认 quant_method=modelopt、quant_algo=FP8。

静态校验：

python examples/quantization/check_modelopt_fp8_export.py ./wan22-ti2v-modelopt-fp8
三项都过：metadata 块完整、safetensors 有 FP8 dtype、磁盘比 BF16 baseline 小。

runtime 推理（脚本末尾打印的命令）：

python examples/offline_inference/text_to_video/text_to_video.py \
    --model ./wan22-ti2v-modelopt-fp8 \
    --quantization fp8 \
    --prompt 'A dog running across a field of golden wheat.' \
    --height 704 --width 1280 --num-frames 49 \
    --num-inference-steps 30 --guidance-scale 5.0 --seed 42 \
    --output outputs/wan22_modelopt_fp8.mp4
关注 vllm-omni 启动日志：

应看到 Adapted ModelOpt FP8 ... weights: dequantized N full-precision weights, skipped 0 scale tensors
任何 skipping scale ... no target warning 都意味着有命名漂移，对照 §4.4 排查 mapper。
关键文件一览
角色	路径
入口脚本	examples/quantization/quantize_wan2_2_modelopt_fp8.py
同模板对照	examples/quantization/quantize_hunyuanvideo_15_modelopt_fp8.py
离线产物校验	examples/quantization/check_modelopt_fp8_export.py
Adapter 主体	vllm_omni/diffusion/model_loader/checkpoint_adapters/modelopt_fp8.py
Adapter 分发	vllm_omni/diffusion/model_loader/checkpoint_adapters/init.py
Loader 调用点	vllm_omni/diffusion/model_loader/diffusers_loader.py (≈L217-L242)
Wan2.2 transformer + mapper	vllm_omni/diffusion/models/wan2_2/wan2_2_transformer.py
TI2V-5B 默认 pipeline	vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2_ti2v.py
模型注册表	vllm_omni/diffusion/registry.py
FP8 文档	docs/user_guide/diffusion/quantization/fp8.md