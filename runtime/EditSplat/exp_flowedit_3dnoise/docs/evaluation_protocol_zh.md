# 评测方案（真实 setting，离线本地，统一走 eval/）

## 1. 目标
对 `run_editing_flow.py` 真实流程输出做统一、可复现评估，覆盖：
1. 语义编辑达成
2. 源保持/失真
3. 多视角一致性
4. 几何与伪影健康度

## 2. 对比对象
至少包含：
1. 原始 baseline（随机噪声）
2. 3D 噪声初始化（不预优化）
3. 3D 噪声初始化 + 预优化

要求：同场景、同视角数、同 epoch、同 FlowEdit 关键超参。

## 3. 强制执行路径
- 评估只使用：`edit/EditSplat/eval`
- 主脚本：`edit/EditSplat/eval/scripts/run_full_eval.py`
- benchmark：`edit/EditSplat/eval/benchmark/*.json`

历史脚本 `exp_flowedit_3dnoise/scripts/eval_full_real_outputs.py` 仅保留兼容，不作为当前主线。

## 4. 指标集合（eval）

### 4.1 语义达成
- `clip_sim_mean`
- `clip_dir_mean`
- `clip_dir_consistency_mean`

### 4.2 源保持/失真
- `l1_to_src`
- `psnr_to_src`
- `ssim_to_src`
- `lpips_to_src`

### 4.3 多视角一致性
- `reproj_l1_mean`
- `reproj_lpips_mean`
- `reproj_visible_ratio_mean`
- `mv_rel_dist_mse`

### 4.4 伪影与几何
- `hf_ratio_vs_src`
- `clip_ratio`
- `vertex_ratio`

## 5. 红偏专项补充指标（本地后处理）
在 `eval` 指标外，额外计算颜色偏移：
- `red_idx_delta = mean(R - (G+B)/2)_edit - mean(R - (G+B)/2)_src`
- `rg_delta`, `rb_delta`

脚本：
- `exp_flowedit_3dnoise/scripts/analyze_redcast_metrics.py`

说明：该补充指标只用于解释“面部偏红是否缓解”，不替代主评估。

## 6. 推荐执行命令

```bash
python edit/EditSplat/eval/scripts/run_full_eval.py \
  --benchmark edit/EditSplat/eval/benchmark/<benchmark.json> \
  --cache_root edit/EditSplat/eval/cache/renders_<tag> \
  --metrics_root edit/EditSplat/eval/cache/metrics_<tag> \
  --summaries_root edit/EditSplat/eval/cache/summaries_<tag> \
  --pairs_per_sample 400 \
  --render_depth_source 1 \
  --compute_reproj 1 \
  --use_lpips 1 \
  --clip_backend transformers \
  --clip_model openai/clip-vit-base-patch32 \
  --open_clip_pretrained openai/clip-vit-base-patch32 \
  --device cuda
```

## 7. 判定规则
若方法满足以下条件，可认为对管线有价值：
1. `clip_dir_mean` 不低于 baseline，或有稳定提升。
2. `reproj_l1_mean` / `mv_rel_dist_mse` 不明显恶化。
3. `lpips_to_src`、`hf_ratio_vs_src` 在可控范围内。
4. 红偏专项指标（`red_idx_delta`）相对 baseline 下降。

## 8. 结果记录要求
每次实验需记录：
1. 完整命令与超参
2. 输出路径与日志路径
3. `summary.csv` / `by_method.csv` 对比表
4. 关键视角可视化与失败案例
5. 下一轮调参动作

