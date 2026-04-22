# 全量真实实验 Round-1 进度记录（2026-02-15）

## 1. 目标
在 `run_editing_flow.py` 的真实设置（完整视角、完整 MFG/AGT/3DGS 优化）下，对比：
1. 原始 baseline（随机噪声）。
2. 3DGS 噪声初始化（不优化）。
3. 3DGS 噪声初始化 + 预优化（base）。
4. 3DGS 噪声初始化 + 预优化（balanced, hybrid anchor）。

## 2. 统一实验设置
- 数据：`/dev-vepfs/rc_wu/rc_wu/edit/EditSplat/dataset/dataset/fangzhou`
- 预训练 3DGS：`/dev-vepfs/rc_wu/rc_wu/edit/EditSplat/dataset/pretrained/fangzhou/chkpnt7000.pth`
- FlowEdit 文本：
  - `flow_src_prompt="a photo of a young man"`
  - `flow_tar_prompt="a photo of a young man made of white marble"`
- 采样参数：
  - `flow_steps=28`
  - `flow_n_avg=1`
  - `flow_n_min=0`
  - `flow_n_max=24`
  - `flow_src_guidance_scale=1.5`
  - `flow_tar_guidance_scale=5.5`
  - `flow_seed=10`
- 编辑管线参数：
  - `filtering_ratio=0.85`
  - `epoch=1`

## 3. 运行矩阵（并行）

| run | GPU | wrapper | 输出目录 | 日志 |
|---|---:|---|---|---|
| baseline | 0 | `run_editing_flow_baseline_wrapper.py` | `output/flowedit_3dnoise_exp/full_real_baseline_fangzhou_fixedrast_r1` | `exp_flowedit_3dnoise/logs/full_baseline_fangzhou_fixedrast_20260215_193322.log` |
| cov_init_noopt | 1 | `run_editing_flow_3dnoise_wrapper.py` | `output/flowedit_3dnoise_exp/full_real_cov_init_noopt_fangzhou_r1` | `exp_flowedit_3dnoise/logs/full_cov_init_noopt_fangzhou_20260215_200705.log` |
| cov_baseopt | 2 | `run_editing_flow_3dnoise_wrapper.py` | `output/flowedit_3dnoise_exp/full_real_cov_baseopt_fangzhou_r1` | `exp_flowedit_3dnoise/logs/full_cov_baseopt_fangzhou_20260215_200823.log` |
| hybrid_balanced | 3 | `run_editing_flow_3dnoise_wrapper.py` | `output/flowedit_3dnoise_exp/full_real_hybrid_balanced_fangzhou_r1` | `exp_flowedit_3dnoise/logs/full_hybrid_balanced_fangzhou_20260215_201118.log` |

## 4. 3D 噪声方案参数

### 4.1 cov_init_noopt
- `noise_anchor_mode=coverage_opacity`
- `noise_init_mode=coarse_smooth`
- `noise_max_anchors=40000`
- `noise_voxel_res=96`
- `noise_coarse_res=32`
- `noise_mix=0.55`
- `noise_opt_iters=0`

### 4.2 cov_baseopt
- 初始化同 `cov_init_noopt`。
- 预优化：
  - `noise_opt_iters=16`
  - `noise_opt_lr=0.02`
  - `noise_opt_mode=base`
  - `noise_opt_num_views=8`
  - `noise_opt_views_per_iter=2`
  - `noise_lambda_edit=0.9`
  - `noise_lambda_id=0.9`
  - `noise_lambda_smooth=0.25`
  - `noise_lambda_prior=0.08`
  - `noise_max_grad_norm=1.0`
- 预优化日志：`exp_flowedit_3dnoise/logs/full_noise_preopt_cov_baseopt_20260215_200855.jsonl`

### 4.3 hybrid_balanced
- `noise_anchor_mode=hybrid`
- `noise_hybrid_ratio=0.6`
- `noise_init_mode=coarse_smooth`
- `noise_max_anchors=40000`
- `noise_voxel_res=96`
- `noise_coarse_res=32`
- `noise_mix=0.45`
- 预优化：
  - `noise_opt_iters=16`
  - `noise_opt_lr=0.018`
  - `noise_opt_mode=balanced`
  - `noise_opt_num_views=8`
  - `noise_opt_views_per_iter=2`
  - `noise_lambda_edit=0.85`
  - `noise_lambda_id=0.95`
  - `noise_lambda_smooth=0.30`
  - `noise_lambda_prior=0.10`
  - `noise_lambda_view_var=0.25`
  - `noise_max_grad_norm=1.0`
- 预优化日志：`exp_flowedit_3dnoise/logs/full_noise_preopt_hybrid_balanced_20260215_201150.jsonl`

## 5. 关键运行状态（记录时刻）
- baseline：已完成（日志含 `Editing complete.`）。
- cov_init_noopt：已完成（日志含 `Editing complete.`）。
- cov_baseopt：已完成（日志含 `Editing complete.`）。
- hybrid_balanced：已完成（日志含 `Editing complete.`）。

## 6. 评测脚本（已准备）
- `exp_flowedit_3dnoise/scripts/eval_full_real_outputs.py`
- 功能：对齐 source 渲染与编辑后渲染，输出
  - `L1/PSNR/SSIM/LPIPS`
  - 高频比 `hf_ratio_vs_src`
  - 裁剪比例 `clip_ratio`
  - 多视角相对距离保持 `mv_rel_dist_mse`
  - ImageReward（可选，本地权重）
  - 点云顶点数变化 `vertex_ratio`

统一评测与面板已生成：
- `exp_flowedit_3dnoise/results/full_real_round1_eval_all_fangzhou.json`
- `exp_flowedit_3dnoise/results/full_real_round1_eval_all_fangzhou.md`
- `exp_flowedit_3dnoise/results/full_real_round1_all_methods_panel_fangzhou.png`

## 7. 结果表（已回填）

| method | views | L1↓ | PSNR↑ | SSIM↑ | LPIPS↓ | HF_ratio≈1 | Clip↓ | MV_rel_MSE↓ | IR_target↑ | IR_delta↑ | vertex_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 87 | 0.087716 | 18.1240 | 0.8717 | 0.2114 | 1.7250 | 0.025672 | 0.001839 | -1.3564 | -2.7597 | 1.0000 |
| cov_init_noopt | 87 | 0.081982 | 18.6201 | 0.8752 | 0.2195 | 1.7491 | 0.023945 | 0.001589 | -1.4082 | -2.8405 | 1.0000 |
| cov_baseopt | 87 | 0.080597 | 18.6047 | 0.8726 | 0.2217 | 1.7450 | 0.021429 | 0.001768 | -1.3650 | -2.8361 | 1.0000 |
| hybrid_balanced | 87 | 0.082401 | 18.3449 | 0.8677 | 0.2194 | 1.8179 | 0.022995 | 0.001869 | -1.3279 | -2.7973 | 1.0000 |

## 8. 快速结论（Round-1）

1. 相对 baseline，3D 噪声方法在 `L1/PSNR/clip_ratio` 上整体有改进，但 `LPIPS` 与 `ImageReward delta` 未提升。  
2. 从多视角关系看，`cov_init_noopt` 的 `mv_rel_dist_mse` 最优（0.001589）。  
3. 从伪影角度看，`hybrid_balanced` 的 `hf_ratio` 最差（1.8179），与可视化中的颗粒和色偏一致。  
4. 四组方法的 `IR_delta` 都是负值且幅度较大，说明“白色大理石面部”语义迁移在当前 prompt/损失配比下仍不足。  

## 9. 已完成的文档与文献整理动作

1. 中文方法与评测文档已扩展到“真实 setting + 论文对齐指标”版本。  
2. 文献综述已补齐 2025 年相关 3D 编辑与多视一致性方向条目。  
3. 修正了本地文献缓存中的错配文件：  
- `GaussianEditor.pdf` 已更正为 arXiv `2311.16037` 对应版本。  
- 原错误版本保留为 `GaussianEditor_wrong_2403.14213.pdf` 以追溯。  
- `InstructGS2GS` 标注为项目页来源，避免错误 arXiv 映射。  
4. 修复评测脚本导出细节：`eval_full_real_outputs.py` 的 Markdown 表格不再把负值 ImageReward 误显示为 `n/a`。  
