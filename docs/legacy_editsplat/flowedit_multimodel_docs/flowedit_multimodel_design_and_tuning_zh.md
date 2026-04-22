# FlowEdit 多模型接入设计与调参说明（含 FLUX2 / SD / AuraFlow）

## 1. 目录与目标
- 代码目录：`EditSplat/flowedit_multimodel/`
- 输出目录：`EditSplat/output/flowedit_multimodel_exp/`

目标：
- 保持你当前 `run_editing_flow.py` 的 FlowEdit 思路（速度差 ODE + 尾段 refinement）。
- 在新目录中提供可插拔多模型适配器，不破坏原训练脚本。
- 支持四卡并行调参运行。

## 2. 已接入模型接口

注册表在 `flowedit_multimodel/src/model_registry.py`：
- `flux1-dev`：当前基线（与你现有代码一致方向）。
- `flux2-dev`：官方最新 FLUX2（gated，代码支持）。
- `flux2-klein-4b`：开放可跑的 FLUX2 变体（默认实测用它）。
- `sd35-large` / `sd35-large-turbo`：官方最新 SD3.5（gated，代码支持）。
- `sd3-medium`：开放 fallback（官方 repo 单文件）。
- `auraflow-v03`：开放 flow-matching 参考模型。

## 3. FlowEdit 实现差异（与你当前代码对比）

### 3.1 共同主干
- 都采用：
  - 速度差段：`z_t <- z_t + (t_{i-1}-t_i) * (v_tar - v_src)`
  - 尾段：`n_min` 步使用目标速度单分支收敛。
  - 跳过早期高噪声：`n_max` 控制。

### 3.2 FLUX 系列（FLUX.1 / FLUX2）
- 特点：
  - token 化 latent（`[B, L, C]`），需要 `img_ids/txt_ids`。
  - timesteps 使用 flow sigma 轨迹 + 序列长度 shift（`mu`）。
- 调参重点：
  - `tar_guidance_scale`：过高容易局部伪影；通常 4~7（FLUX1）更稳。
  - `n_max`：过大易带入高噪误差，建议 16~24。
  - `n_avg`：>1 稳定但耗时线性增加。

### 3.3 SD3/SD3.5 系列
- 特点：
  - 空间 latent（`[B, C, H, W]`），显式 CFG（uncond/text 两分支）。
  - 通常需要更高 guidance 才有显著语义编辑。
- 调参重点：
  - `tar_guidance_scale` 常需要高于 FLUX（约 8~12）。
  - `src_guidance_scale` 不宜过高，避免 identity 过强导致编辑失败。
  - `steps` 对细节提升更敏感，24->32 常有增益。

### 3.4 AuraFlow
- 特点：
  - 也是 flow-matching，但无 FLUX2 那类显式图像条件分支。
  - 对文本驱动变化明显，结构保持依赖初始 latent 与 `n_max`。
- 调参重点：
  - `tar_guidance_scale` 过大容易纹理噪点，建议 4~6。
  - `n_max` 不宜过大（建议 18~22）。

## 4. 默认建议超参（首轮）
- `flux1-dev`：`steps=28, src_g=1.5, tar_g=6.5, n_min=0, n_max=24`
- `flux2-klein-4b`：`steps=24, src_g=1.2, tar_g=4.5, n_min=0, n_max=20`
- `sd3-medium`：`steps=32, src_g=3.5, tar_g=9.5, n_min=0, n_max=18`
- `auraflow-v03`：`steps=32, src_g=2.5, tar_g=5.0, n_min=0, n_max=20`

## 5. 调参与评估脚本
- 单模型调参：`flowedit_multimodel/scripts/tune_flowedit_model.py`
- 四卡并行启动：`flowedit_multimodel/scripts/run_tuning_4gpu.sh`
- 两例测试配置：`flowedit_multimodel/configs/test_cases_2examples.json`

指标（脚本内自动算）：
- `CLIP(tar_prompt)`：目标语义对齐
- `CLIP(src_prompt)`：源语义残留
- `LPIPS(out, src)`：与源图感知距离
- 综合分数：`tar_clip - 0.25*src_clip - 0.20*lpips`

## 6. Gated 模型切换说明
当你的 HF 账号开通权限后，可直接切换到：
- `flux2-dev`
- `sd35-large`
- `sd35-large-turbo`

不需要改算法代码，只需在命令里替换 `--model_key` 或 `--model_id`。
