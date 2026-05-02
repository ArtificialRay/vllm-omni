# FP8 Followups: Wan2.2 MoE/VACE + HunyuanVideo-1.5 Variants + Block-wise + HF Publish

## Context

vllm-omni 当前的离线 ModelOpt FP8 量化链路（producer 脚本 → ModelOptFp8CheckpointAdapter → vLLM ModelOptFp8Config）只在两个具体 case 上跑通了 per-tensor 静态 FP8：

- `Wan-AI/Wan2.2-TI2V-5B-Diffusers`（单 transformer, 704×1280 T2V）
- `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v`

[fp8.md](docs/user_guide/diffusion/quantization/fp8.md) 已经把 A14B、720p_t2v、480p_i2v、VACE 列在表里，但实际上量化脚本要么明说 out of scope（[quantize_wan2_2_modelopt_fp8.py:17-18](examples/quantization/quantize_wan2_2_modelopt_fp8.py#L17-L18) 关于 A14B），要么 forward_loop 只跑 T2V 路径（i2v 没 image 输入），要么注册表/adapter 端漏掉对应配置。block-wise 之前以为被上游阻塞，重新核实发现 vLLM 0.19 已经有 `ModelOptFp8PbWoLinearMethod`（硬编码 128×128），瓶颈实际在 producer 脚本写错了 `quant_algo`。最后这些已校准的产物分散在本地各处，需要一个上传 helper 把它们发布到 HF Hub `vllm-project-org/` 命名空间，方便用户直接 `Omni(model="vllm-project-org/Wan2.2-TI2V-5B-FP8")` 起 vLLM serving。

本计划把这五个 followups 串成五个独立可合并的阶段，每个阶段一份 PR，按下面顺序执行可让前一阶段的工具被后一阶段复用（A14B 的 dual-transformer save → VACE 复用；HV-1.5 i2v 的 image-conditioned forward_loop 引入；block-wise 的 quant_algo 修复落到所有脚本上；最后 publish helper 一次性把前四阶段的产物推上去）。

---

## Phase 1 — Wan2.2 T2V-A14B / I2V-A14B（dual-transformer MoE）

**目标**：让 [quantize_wan2_2_modelopt_fp8.py](examples/quantization/quantize_wan2_2_modelopt_fp8.py) 同时量化 `transformer` 和 `transformer_2`，导出标准 diffusers MoE 目录布局。

**关键事实**（已验证）：
- [pipeline_wan2_2.py:415-441](vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py#L415-L441) 已经按 `boundary_timestep` 在 `self.transformer` / `self.transformer_2` 之间切换 → 校准时只要 `pipe.transformer` 和 `pipe.transformer_2` 都在 GPU 上，单次 `pipe(...)` 就能让两个 backbone 都收到 amax 统计。不需要写两个 forward_loop。
- [quantize_wan2_2_modelopt_fp8.py:340](examples/quantization/quantize_wan2_2_modelopt_fp8.py#L340) 的 `shutil.copytree(... ignore_patterns("transformer", "transformer_2"))` 已经预留了 transformer_2 的目录排除，设计层已为 MoE 留口子。
- [modelopt_fp8.py:55-59](vllm_omni/diffusion/model_loader/checkpoint_adapters/modelopt_fp8.py#L55-L59) 的 `_is_transformer_source` 只匹配 `subfolder == "transformer"` → A14B 的 `transformer_2` 子目录会被 adapter 漏掉，必须扩。
- [diffusers_loader.py 第 281-353 行](vllm_omni/diffusion/model_loader/diffusers_loader.py) 已按 `boundary_ratio` 加载两个 `ComponentSource`（subfolder 分别 `transformer` 和 `transformer_2`），运行时不需要改。

**改动清单**：

1. **[examples/quantization/quantize_wan2_2_modelopt_fp8.py](examples/quantization/quantize_wan2_2_modelopt_fp8.py)**
   - docstring 删掉 `A14B ... out of scope here` 那句，改成支持矩阵
   - 新增 `--boundary-ratio` CLI 参数（A14B 默认 `0.875`，TI2V-5B 不传）
   - `main()` 在 `mtq.quantize` 调用前判断 `pipe.transformer_2 is not None`，如果有就把 `transformer_2` 也作为 backbone 跑一次 `mtq.quantize` + `_force_export_quantized_weights`（共享 quant_config 和 forward_loop，不需要分开校准——单次 `pipe(...)` 已经触达两个）
   - **重构**：把"quantize + force-export + save + patch_config 一个 transformer"的逻辑抽成 `_quantize_and_export_one(backbone, subfolder_name, output_dir, ...)`，然后对 `transformer` 和 `transformer_2`（如存在）各调一次
   - `_save_pipeline_with_fp8_transformer` 改名为 `_save_pipeline_with_fp8_transformers`，循环保存 `pipe.transformer` 和 `pipe.transformer_2`
   - `_patch_quant_config(output_dir, subfolder, ...)` 加 `subfolder` 参数（默认 `"transformer"`），对 A14B 调两次

2. **[vllm_omni/diffusion/model_loader/checkpoint_adapters/modelopt_fp8.py](vllm_omni/diffusion/model_loader/checkpoint_adapters/modelopt_fp8.py)**
   - 改 `_is_transformer_source` 接受 `transformer` 和 `transformer_2` 两种 subfolder，prefix 同样允许 `transformer_2.` 开头
   - 这是 adapter 主入口的最小改动，逻辑保持单实例（每个 source 独立调用一次）

3. **手动验证**（用户已确认有 2×H100）
   - `python quantize_wan2_2_modelopt_fp8.py --model Wan-AI/Wan2.2-T2V-A14B-Diffusers --output ./wan22-t2v-a14b-fp8 --boundary-ratio 0.875`
   - 校验输出目录有 `transformer/` 和 `transformer_2/`，两个都有 `config.json` 含 `quantization_config.quant_method=modelopt`
   - 用 `examples/quantization/check_modelopt_fp8_export.py` 对两个子目录分别跑
   - 跑 `examples/offline_inference/text_to_video/text_to_video.py --model ./wan22-t2v-a14b-fp8 --quantization fp8`，看启动日志两次 `Adapted ModelOpt FP8 weights` 各 0 个 skipped scale
   - I2V 同样 `--model Wan-AI/Wan2.2-I2V-A14B-Diffusers`，验证 `image_to_video.py` 跑通

---

## Phase 2 — Wan2.2 VACE

**目标**：相同脚本通过 `--variant vace` 或自动检测，量化 VACE 单 transformer，校准 prompt 池支持 `reference_images`。

**关键事实**（已验证）：
- [pipeline_wan2_2_vace.py](vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2_vace.py) 是单 transformer pipeline（无 `transformer_2`），所以 Phase 1 的 dual-transformer 抽象自然降级
- VACE 需要 `vace_context` / `vace_context_scale` 等额外条件输入；离线 diffusers `pipe(...)` 接受 `reference_images` 关键字（diffusers Wan2.2VACEPipeline 的标准 API）
- `_filter_func_wan22` 和 `_wan22_quant_config_block` 的 ignore 列表对 VACE transformer 同样适用（VACE transformer 复用 Wan22 backbone，子模块名一致）

**改动清单**：

1. **[examples/quantization/quantize_wan2_2_modelopt_fp8.py](examples/quantization/quantize_wan2_2_modelopt_fp8.py)**
   - 新增 `--variant {auto, ti2v, a14b, vace}` 参数（默认 `auto`，从 `model_index.json` 推断）
   - 新增 `--reference-images` 参数（接受图片路径 list 或一个目录），仅 VACE 启用
   - 新增 `VACE_DEFAULT_PROMPTS` —— 4 个文本 prompt + 4 个 (text, ref_image) pair，覆盖纯 T2V / R2V / V2V 三种 VACE 模式（V2V 暂跳过：需要源视频，校准成本太高）
   - `_build_forward_loop`：当 variant==vace 时，对每个校准样本传 `reference_images=[...]`
   - 校准用一组小尺寸（512×512、frames=33）的 ref images 即可，不必匹配生产分辨率（amax 统计与分辨率无关）

2. **校准 image 资产**
   - 在 [examples/quantization/calibration_assets/wan22_vace/](examples/quantization/calibration_assets/wan22_vace/) 新增 4 张 CC0 图片（建议从 Pexels/Unsplash 下，commit 进仓库 < 1MB 总）
   - 不要 commit 视频源（V2V 校准跳过）

3. **手动验证**
   - `python quantize_wan2_2_modelopt_fp8.py --model Wan-AI/Wan2.2-VACE-A14B-Diffusers --variant vace --output ./wan22-vace-fp8`
   - run inference 走 `pipeline_wan2_2_vace.py` 的 R2V 路径

---

## Phase 3 — HunyuanVideo-1.5 720p_t2v + 480p_i2v + 720p_i2v

**目标**：让 [quantize_hunyuanvideo_15_modelopt_fp8.py](examples/quantization/quantize_hunyuanvideo_15_modelopt_fp8.py) 支持三个新变体。

**关键事实**（已验证）：
- HV-1.5 transformer 的 quant_config 已经穿透到所有 ColumnParallelLinear/RowParallelLinear（[hunyuan_video_15_transformer.py:334-395, 505-521](vllm_omni/diffusion/models/hunyuan_video_15/hunyuan_video_15_transformer.py#L334-L521)）—— 之前 memory 里那条 "no quant_config threading" 的笔记**已经过时**，需要更新
- 720p_t2v 走相同脚本，只需把默认 `--height 480 --width 832` 改成命令行可覆盖（脚本本来就支持 CLI 覆盖）
- I2V pipeline ([pipeline_hunyuan_video_1_5_i2v.py:438-470](vllm_omni/diffusion/models/hunyuan_video_15/pipeline_hunyuan_video_1_5_i2v.py#L438-L470)) 必须传 `multi_modal_data={"image": <image>}` 才能激活 image_embedder 路径，T2V forward_loop 现在不会触达这条
- 注册表已经有 `HunyuanVideo15Pipeline` 和 `HunyuanVideo15ImageToVideoPipeline`，无需改 [registry.py](vllm_omni/diffusion/registry.py)

**改动清单**：

1. **[examples/quantization/quantize_hunyuanvideo_15_modelopt_fp8.py](examples/quantization/quantize_hunyuanvideo_15_modelopt_fp8.py)**
   - 新增 `--variant {auto, t2v, i2v}`（默认 auto，从 `pipe.__class__.__name__` 推断）
   - 新增 `--reference-images`（i2v 必填）
   - `_build_forward_loop` 在 variant==i2v 时对每个 prompt 配一张 ref image，通过 `image=...` kwarg 传给 `pipe(...)`（diffusers HV-1.5 i2v pipeline 的入参）
   - 把 `--height`/`--width`/`--num-frames` 默认值移到 docstring 推荐表里（不写死特定 spec），因为 720p 跑相同脚本只需 `--height 720 --width 1280`

2. **校准 image 资产**
   - [examples/quantization/calibration_assets/hv15_i2v/](examples/quantization/calibration_assets/hv15_i2v/) 放 8 张 CC0 图（与 VACE 那批可以复用）

3. **memory 更新**
   - 修改 [/home/rthu/.claude/projects/-home-rthu-vllm-omni/memory/project_fp8_implementation_gap.md](/home/rthu/.claude/projects/-home-rthu-vllm-omni/memory/project_fp8_implementation_gap.md)，把 "HunyuanVideo-1.5 transformer no quant_config threading" 的部分作废（保留历史观察的方法论价值，但标注 status: resolved 并指明本轮 PR 号）

4. **手动验证**
   - 720p_t2v: `python quantize_hunyuanvideo_15_modelopt_fp8.py --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v --height 720 --width 1280 --output ./hv15-720p-t2v-fp8`
   - 480p_i2v: `... --variant i2v --reference-images examples/quantization/calibration_assets/hv15_i2v --output ./hv15-480p-i2v-fp8`
   - 720p_i2v 同上加 `--height 720 --width 1280`
   - 三个 checkpoint 都跑 `check_modelopt_fp8_export.py`，再跑各自的 inference 入口

---

## Phase 4 — Block-wise 静态 FP8（128×128）

**目标**：让 `--weight-block-size 128,128` 真正工作，端到端验证 block-wise scale 能被上游 `ModelOptFp8PbWoLinearMethod` 正确派发。

**关键事实**（已重新核实，与之前 docstring 注释相反）：
- vLLM 0.19 的 `ModelOptFp8Config` ([upstream modelopt.py:366-399](vllm/model_executor/layers/quantization/modelopt.py)) 按 `quant_algo` 字段路由：`"FP8" → ModelOptFp8LinearMethod`（per-tensor），`"FP8_PB_WO" → ModelOptFp8PbWoLinearMethod`（block-wise，硬编码 128×128）—— **上游已 ready**
- producer 端 ([quantize_wan2_2_modelopt_fp8.py:299](examples/quantization/quantize_wan2_2_modelopt_fp8.py#L299)) `_wan22_quant_config_block` 在 block_size 非空时仍写 `quant_algo: "FP8"` —— **这是真正的瓶颈**
- adapter ([modelopt_fp8.py:136-154](vllm_omni/diffusion/model_loader/checkpoint_adapters/modelopt_fp8.py#L136-L154)) 的 `_reshape_weight_scale` 已经能处理 4D `[out_blk, 1, in_blk, 1]` block scale 形状 —— **不需要改 adapter**

**改动清单**：

1. **[examples/quantization/quantize_wan2_2_modelopt_fp8.py](examples/quantization/quantize_wan2_2_modelopt_fp8.py#L299)** + **[examples/quantization/quantize_hunyuanvideo_15_modelopt_fp8.py](examples/quantization/quantize_hunyuanvideo_15_modelopt_fp8.py)**（同名函数）
   - 当 `weight_block_size is not None` 时，把 `_*_quant_config_block` 返回的 `quant_algo` 从 `"FP8"` 改为 `"FP8_PB_WO"`
   - 加 assertion：上游 `ModelOptFp8PbWoLinearMethod` 硬编码 `(128, 128)`，所以脚本应在解析 `--weight-block-size` 时拒绝 `M != 128 or N != 128`，给出明确报错（"upstream only supports 128x128"）
   - docstring 关于 #2924 那条注释更新成 "supported via FP8_PB_WO when block_size=128,128"

2. **[examples/quantization/check_modelopt_fp8_export.py](examples/quantization/check_modelopt_fp8_export.py)**
   - 增加一个断言路径：当 metadata `strategy == "block"` 时，必须 `quant_algo == "FP8_PB_WO"`，否则失败（防止以后又写错）

3. **手动验证**
   - 跑 `quantize_wan2_2_modelopt_fp8.py --model Wan-AI/Wan2.2-TI2V-5B-Diffusers --weight-block-size 128,128 --output ./wan22-ti2v-fp8-block`
   - 静态校验：`check_modelopt_fp8_export.py ./wan22-ti2v-fp8-block` 看 quant_algo=FP8_PB_WO + block_structure=128x128
   - inference：启动日志应显示 `ModelOptFp8PbWoLinearMethod` 被实例化（在 vLLM Linear 层）；adapter 日志显示 `dequantized=0`（透传 FP8 路径）
   - 与 per-tensor 版本做生成质量对比（同 prompt + seed），记录 LPIPS/CLIP-score 差异

---

## Phase 5 — HF Hub Publish Helper

**目标**：写一个 `tools/publish_fp8_checkpoint.py`，把已生成的 ModelOpt FP8 diffusers 目录推到 HF Hub `vllm-project-org/<repo-name>`，并自动生成 README 卡片和 quantization metadata 摘要。

**关键事实**：
- 仓库目前没有任何 HF Hub 上传工具（grep `huggingface_hub.HfApi` 无结果）
- `huggingface_hub.snapshot_download` 已被脚本使用（[quantize_wan2_2_modelopt_fp8.py:334](examples/quantization/quantize_wan2_2_modelopt_fp8.py#L334)），所以 dependency 已在
- `vllm-project-org` 命名空间在仓库的 mkdocs / pyproject 里没有引用 —— 这是一个新发布目标，需要维护者准备好 HF org 并配 token

**改动清单**：

1. **[tools/publish_fp8_checkpoint.py](tools/publish_fp8_checkpoint.py)**（新文件，参考 [tools/configure_stage_memory.py](tools/configure_stage_memory.py) 风格）
   - 入参：`--checkpoint-dir`（本地 ModelOpt FP8 产物目录）、`--repo-name`（如 `Wan2.2-TI2V-5B-FP8`）、`--org`（默认 `vllm-project-org`）、`--private` flag、`--dry-run`
   - 行为：
     1. 校验本地目录跑过 `check_modelopt_fp8_export.py` 三项断言（直接 import 并调用，把它的检查函数模块化提取）
     2. 读 transformer/config.json 的 `quantization_config`，生成 README.md 模板（model 名、source HF id、quant_method、quant_algo、block 结构 if any、ignore list、校准 prompt 池来源、--num-frames/--height/--width 推理 baseline）
     3. 用 `huggingface_hub.HfApi.upload_folder(folder_path=..., repo_id=f"{org}/{repo_name}", repo_type="model")` 推上去
     4. `--dry-run` 模式只生成 README 并打印将要上传的文件清单，不真上传

2. **校验函数模块化**
   - 把 [check_modelopt_fp8_export.py](examples/quantization/check_modelopt_fp8_export.py) 的核心检查逻辑抽出成可 import 的函数（保留 CLI 入口），让 publish 脚本能复用
   - **不要**为这一步引入新 abstraction —— 抽取尺度限制在 publish 工具实际需要复用的部分

3. **README 模板**（写成 jinja-less 简单 f-string 即可）
   - 包含 `Omni(model="vllm-project-org/<repo>", quantization="fp8")` 这一行可拷贝命令
   - 与上游 [fp8.md](docs/user_guide/diffusion/quantization/fp8.md) 表格里的 ignored_layers 推荐保持一致

4. **本轮要发布的 7 个 repo**（用户已确认）
   - `vllm-project-org/Wan2.2-TI2V-5B-FP8`（per-tensor，已有产物）
   - `vllm-project-org/Wan2.2-T2V-A14B-FP8`（Phase 1 产物）
   - `vllm-project-org/Wan2.2-I2V-A14B-FP8`（Phase 1 产物）
   - `vllm-project-org/Wan2.2-VACE-A14B-FP8`（Phase 2 产物）
   - `vllm-project-org/HunyuanVideo-1.5-720p-t2v-FP8`（Phase 3 产物）
   - `vllm-project-org/HunyuanVideo-1.5-480p-i2v-FP8`（Phase 3 产物）
   - `vllm-project-org/HunyuanVideo-1.5-720p-i2v-FP8`（Phase 3 产物）
   - 已经能跑通的 `vllm-project-org/HunyuanVideo-1.5-480p-t2v-FP8` 也包含

5. **手动验证**
   - 先 `--dry-run` 跑全部 7 个，确认 README 正确、文件清单合理
   - 用 `--repo-name <test-repo> --org <user-personal-account> --private` 推一个测试仓库，从 HF 上 `Omni(model="<user>/<test-repo>", quantization="fp8")` 拉回来跑通
   - 全部验证后再正式推到 `vllm-project-org/`

---

## 跨阶段约定

- 每个 Phase 一个独立 PR；前 4 阶段必须 squash-merge 后才能跑 Phase 5 publish
- 每个 PR 的 description 里包括：本阶段改动的 hand-tested checkpoint id（HF 或本地路径），以及 inference 启动日志关键行（`Adapted ModelOpt FP8 weights ... skipped 0 scale tensors`）
- [docs/user_guide/diffusion/quantization/fp8.md](docs/user_guide/diffusion/quantization/fp8.md) 在 Phase 1/2/3 完成后同步更新支持矩阵（本轮新增的 model id 标记为 ✓ verified；block-wise 在 Phase 4 后单独加一行 recommendation note）

## 关键文件一览

| 角色 | 路径 |
|---|---|
| Wan2.2 量化脚本 | [examples/quantization/quantize_wan2_2_modelopt_fp8.py](examples/quantization/quantize_wan2_2_modelopt_fp8.py) |
| HV-1.5 量化脚本 | [examples/quantization/quantize_hunyuanvideo_15_modelopt_fp8.py](examples/quantization/quantize_hunyuanvideo_15_modelopt_fp8.py) |
| 静态校验工具 | [examples/quantization/check_modelopt_fp8_export.py](examples/quantization/check_modelopt_fp8_export.py) |
| Adapter | [vllm_omni/diffusion/model_loader/checkpoint_adapters/modelopt_fp8.py](vllm_omni/diffusion/model_loader/checkpoint_adapters/modelopt_fp8.py) |
| Wan2.2 base pipeline（含 boundary 调度） | [vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py](vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py) |
| Wan2.2 VACE pipeline | [vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2_vace.py](vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2_vace.py) |
| HV-1.5 i2v pipeline | [vllm_omni/diffusion/models/hunyuan_video_15/pipeline_hunyuan_video_1_5_i2v.py](vllm_omni/diffusion/models/hunyuan_video_15/pipeline_hunyuan_video_1_5_i2v.py) |
| Diffusers loader（boundary_ratio 解析） | [vllm_omni/diffusion/model_loader/diffusers_loader.py](vllm_omni/diffusion/model_loader/diffusers_loader.py) |
| HF publish 新工具 | tools/publish_fp8_checkpoint.py（新增） |
| FP8 用户文档 | [docs/user_guide/diffusion/quantization/fp8.md](docs/user_guide/diffusion/quantization/fp8.md) |
