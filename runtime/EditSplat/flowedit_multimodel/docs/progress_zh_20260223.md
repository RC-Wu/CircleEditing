# FlowEdit MultiModel 阶段进展（2026-02-23）

## 1. 环境与兼容性修复

已确认可用环境：
- 主环境：`/dev-vepfs/rc_wu/rc_wu/envs/editsplat_ttt3r_unified_copy`
- 备用环境：`/dev-vepfs/rc_wu/rc_wu/envs/editsplat_test`

关键修复：
- `diffusers` + `torch 2.4` 兼容补丁：
  - 文件：`.../site-packages/diffusers/models/attention_dispatch.py`
  - 修复点：`scaled_dot_product_attention(enable_gqa=...)` 在 torch 2.4 上签名不兼容，改为运行时探测并回退。
- FlowEdit 多模型适配器显存修复：
  - 文件：`flowedit_multimodel/src/flowedit_adapters.py`
  - 修复点：为 `Flux1/Flux2/SD3/Aura` 的 `edit()` 增加 `@torch.no_grad()`，避免 ODE 循环累积反向图导致 80GB OOM。

## 2. 已完成实验（flux1 基线）

实验目录：
- `EditSplat/output/flowedit_multimodel_exp/flux1_full_v1/`

测试集：
- `face_marble`
- `bear_marble`

评分函数（当前禁用 CLIP 时）：
- `score = -0.20 * L1(out, src)`（值越大越好）

### 2.1 最优超参（当前）
- `diffusion_steps=28`
- `n_avg=2`
- `src_guidance_scale=1.7`
- `tar_guidance_scale=7.0`
- `n_min=0`
- `n_max=24`

### 2.2 对比结果（按 avg_score）
1. `hp_03`: `avg_score=-0.01852`（当前最优）
2. `hp_02`: `avg_score=-0.02401`
3. `hp_01`: `avg_score=-0.04026`
4. `hp_00`: `avg_score=-0.04725`

说明：`n_avg=2` 对稳定性和目标风格迁移有明显正向作用，符合 FlowEdit 在 FLUX 上“速度差平均”可降低抖动的经验。

## 3. 模型可访问性状态

HF token：`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`

最新实测：
- `black-forest-labs/FLUX.2-dev`：`GatedRepoError 403`（仍不可直接下载）
- `stabilityai/stable-diffusion-3.5-large`：`GatedRepoError 403`
- `stabilityai/stable-diffusion-3.5-large-turbo`：`GatedRepoError 403`

结论：官方“最新 FLUX2 / SD3.5”代码路径已接入，但受账号门控限制，当前先用开放模型完成可运行闭环：
- `black-forest-labs/FLUX.2-klein-4B`
- `stabilityai/stable-diffusion-3-medium`
- `fal/AuraFlow-v0.3`

## 4. 下载策略与当前进度

已新增/更新：
- `scripts/hf_fast_download.py`（端点切换 + 断点续传 + 低并发稳定策略）
- `scripts/hf_sequential_retry_download.py`（单文件顺序重试，弱网更稳）
- `docs/hf_download_acceleration_zh.md`（加入 `HF_HUB_ENABLE_HF_TRANSFER=0` 的稳态方案）

当前进行中：
- `FLUX.2-klein-4B` 大文件断点续传（huggingface/hf-mirror 双端点自动切换）

## 5. 与 FlowEdit / 现有 `run_editing_flow.py` 的调参映射

### FLUX 系（含 FLUX2）
- 关注：`tar_guidance_scale`（过高易伪影）、`n_max`（过大带入高噪）、`n_avg`（>1 稳定但变慢）
- 经验区间：
  - `src_g`: `~1.0-1.8`
  - `tar_g`: `~4.0-7.5`
  - `steps`: `~22-34`

### SD3/SD3.5
- 关注：CFG 主导语义迁移，通常需更高 `tar_g`；`steps` 增加常带来细节收益
- 经验区间：
  - `src_g`: `~2.0-4.5`
  - `tar_g`: `~8.0-12.0`
  - `steps`: `~24-40`

### AuraFlow
- 关注：`tar_g` 过大容易纹理噪点；`n_max` 过大容易结构漂移
- 经验区间：
  - `src_g`: `~1.5-3.0`
  - `tar_g`: `~4.0-6.0`
  - `steps`: `~24-38`

## 6. 下一步执行顺序（正在进行）

1. 完成 `FLUX.2-klein-4B` 下载并跑 `2案例 x 多组超参`。
2. 继续 `sd3-medium` 与 `auraflow-v03` 下载、调参、生成结果。
3. 四卡并行跑最终组合（`run_tuning_4gpu.sh`）。
4. 汇总到 `EditSplat/output/flowedit_multimodel_exp/` 并形成最终报告。
