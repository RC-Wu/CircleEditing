# 全量真实实验 Round-1 报告（中文，2026-02-15）

## 1. 实验目标
在 `run_editing_flow.py` 真实 setting 下，对比原始随机噪声 baseline 与 3DGS 噪声初始化/优化方案，验证：
1. 是否能提升目标编辑语义达成度。  
2. 是否能在多视角一致性和结构保真上不劣于 baseline。  
3. 是否对后续 MFG/AGT/3DGS 优化阶段产生稳定正收益。  

## 2. 统一设置

1. 数据集：`dataset/dataset/fangzhou`。  
2. 预训练模型：`dataset/pretrained/fangzhou/chkpnt7000.pth`。  
3. Flow 文本：  
- `flow_src_prompt="a photo of a young man"`  
- `flow_tar_prompt="a photo of a young man made of white marble"`  
4. 采样参数：`flow_steps=28`, `flow_n_avg=1`, `flow_n_min=0`, `flow_n_max=24`, `src_gs=1.5`, `tar_gs=5.5`, `seed=10`。  
5. 编辑参数：`filtering_ratio=0.85`, `epoch=1`。  

## 3. 对比方法与配置

| method | 关键配置 | 输出目录 | 日志 |
|---|---|---|---|
| baseline | 原始随机噪声 | `output/flowedit_3dnoise_exp/full_real_baseline_fangzhou_fixedrast_r1` | `exp_flowedit_3dnoise/logs/full_baseline_fangzhou_fixedrast_20260215_193322.log` |
| cov_init_noopt | `coverage_opacity + coarse_smooth`, `noise_mix=0.55`, 无预优化 | `output/flowedit_3dnoise_exp/full_real_cov_init_noopt_fangzhou_r1` | `exp_flowedit_3dnoise/logs/full_cov_init_noopt_fangzhou_20260215_200705.log` |
| cov_baseopt | 在 `cov_init_noopt` 上加 `base` 预优化（16 iter） | `output/flowedit_3dnoise_exp/full_real_cov_baseopt_fangzhou_r1` | `exp_flowedit_3dnoise/logs/full_cov_baseopt_fangzhou_20260215_200823.log` |
| hybrid_balanced | `hybrid` 锚点 + `balanced` 预优化（16 iter） | `output/flowedit_3dnoise_exp/full_real_hybrid_balanced_fangzhou_r1` | `exp_flowedit_3dnoise/logs/full_hybrid_balanced_fangzhou_20260215_201118.log` |

预优化日志：
1. `exp_flowedit_3dnoise/logs/full_noise_preopt_cov_baseopt_20260215_200855.jsonl`  
2. `exp_flowedit_3dnoise/logs/full_noise_preopt_hybrid_balanced_20260215_201150.jsonl`  

## 4. 量化结果

结果文件（自动生成）：
1. `exp_flowedit_3dnoise/results/full_real_round1_eval_all_fangzhou.json`  
2. `exp_flowedit_3dnoise/results/full_real_round1_eval_all_fangzhou.md`  

| method | views | L1↓ | PSNR↑ | SSIM↑ | LPIPS↓ | HF_ratio≈1 | Clip↓ | MV_rel_MSE↓ | IR_target↑ | IR_delta↑ | vertex_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 87 | 0.087716 | 18.1240 | 0.8717 | 0.2114 | 1.7250 | 0.025672 | 0.001839 | -1.3564 | -2.7597 | 1.0000 |
| cov_init_noopt | 87 | 0.081982 | 18.6201 | 0.8752 | 0.2195 | 1.7491 | 0.023945 | 0.001589 | -1.4082 | -2.8405 | 1.0000 |
| cov_baseopt | 87 | 0.080597 | 18.6047 | 0.8726 | 0.2217 | 1.7450 | 0.021429 | 0.001768 | -1.3650 | -2.8361 | 1.0000 |
| hybrid_balanced | 87 | 0.082401 | 18.3449 | 0.8677 | 0.2194 | 1.8179 | 0.022995 | 0.001869 | -1.3279 | -2.7973 | 1.0000 |

相对 baseline 的关键变化：
1. `cov_baseopt` 在 `L1` 和 `clip_ratio` 最优，但 `LPIPS` 变差。  
2. `cov_init_noopt` 在 `PSNR/SSIM/mv_rel_dist_mse` 最优，但语义指标最差。  
3. `hybrid_balanced` 的 `IR_target` 略好于 baseline，但 `hf_ratio` 明显恶化，伪影风险最大。  

运行耗时（按日志文件名时间戳与文件修改时间估算）：
1. baseline：约 `1.38h`  
2. cov_init_noopt：约 `2.76h`  
3. cov_baseopt：约 `2.73h`  
4. hybrid_balanced：约 `2.75h`  

## 5. 可视化对比与现象分析

1. 多方法统一面板：`exp_flowedit_3dnoise/results/full_real_round1_all_methods_panel_fangzhou.png`  
2. baseline 面板：`exp_flowedit_3dnoise/results/full_real_round1_baseline_panel_fangzhou.png`  

建议从以下角度记录观察：
1. 前景编辑区域是否出现颜色偏移或颗粒化。  
2. 背景是否被误编辑。  
3. 跨视角是否出现闪烁、局部爆点或条纹结构。  

基于本轮面板观察的结论：
1. 四种方法都出现面部“偏红+饱和度上升”问题，说明当前文本条件下模型更倾向于颜色风格化，而不是稳定的“白色大理石材质化”。  
2. 3D 噪声方法相对 baseline 的改进主要体现为“失真控制”（更低 L1/clip），并未转化为更好的目标语义奖励。  
3. `hybrid_balanced` 在若干视角出现更显著高频颗粒，这与 `hf_ratio` 最高一致。  

补充统计（训练视角 ROI）：
1. 相比源图，baseline 的 `R-G` 偏移约 `+0.0949`，`R-B` 偏移约 `+0.1170`，`sat_delta≈+0.1192`。  
2. `cov_baseopt` 的对应值降至 `R-G≈+0.0738`, `R-B≈+0.0979`, `sat_delta≈+0.1005`，说明红偏和过饱和有一定缓解但未根治。  

## 6. 结论与下一轮动作

1. 相对 baseline 的“综合可用”候选：`cov_baseopt`。  
2. 主要收益：在不改原始核心管线的前提下，3D 噪声方案能降低整体失真与过曝裁剪比例。  
3. 主要失败模式：语义编辑不足（IR_delta 全负）+ 面部红偏/过饱和。  
4. 下一轮优先改进：
- `mask-aware anchors`：只在目标区域强化噪声优化。  
- `fg/bg split objective`：前景强化编辑、背景强化身份保持。  
- 降低背景噪声混合：`rho_bg < rho_fg`。  
- 文本策略微调：将目标 prompt 从“marble sculpture face”改为更强材质约束描述并加入“neutral skin tone preservation”约束语句。  

## 7. 复现实验命令（本轮）

评测命令：

```bash
python exp_flowedit_3dnoise/scripts/eval_full_real_outputs.py \
  --model_dirs \
    output/flowedit_3dnoise_exp/full_real_baseline_fangzhou_fixedrast_r1 \
    output/flowedit_3dnoise_exp/full_real_cov_init_noopt_fangzhou_r1 \
    output/flowedit_3dnoise_exp/full_real_cov_baseopt_fangzhou_r1 \
    output/flowedit_3dnoise_exp/full_real_hybrid_balanced_fangzhou_r1 \
  --source_pretrained_dir dataset/pretrained/fangzhou \
  --source_iter 30000 \
  --use_lpips \
  --use_imagereward \
  --target_prompt "a photo of a marble sculpture face" \
  --source_prompt "a photo of a young man" \
  --out_json exp_flowedit_3dnoise/results/full_real_round1_eval_all_fangzhou.json \
  --out_md exp_flowedit_3dnoise/results/full_real_round1_eval_all_fangzhou.md
```
