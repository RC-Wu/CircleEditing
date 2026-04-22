# 统一评估协议（强制使用 `edit/EditSplat/eval`）

## 1. 约束
- 仅使用 `edit/EditSplat/eval` 目录下脚本。
- 不使用在线评测服务。
- 评估权重与缓存均落地到 `/dev-vepfs/rc_wu/rc_wu/cache`。

## 2. 环境变量（建议）
```bash
export HF_HOME=/dev-vepfs/rc_wu/rc_wu/cache/hf_home
export HF_HUB_CACHE=/dev-vepfs/rc_wu/rc_wu/cache/hf_home/hub
export TORCH_HOME=/dev-vepfs/rc_wu/rc_wu/cache/torch_hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## 3. benchmark 组织
每个“场景 + 编辑任务 + 方法集合”一份 benchmark JSON，放在：
- `edit/EditSplat/eval/benchmark/`

推荐命名：
- `benchmark_<scene>_<task>_<date>.json`

必要字段：
- `scene_id`
- `edit_id`
- `method`
- `model_dir`
- `source_pretrained_dir`
- `source_checkpoint`
- `source_path`
- `split`
- `target_prompt`
- `source_caption`

## 4. 一键评估模板
```bash
python edit/EditSplat/eval/scripts/run_full_eval.py \
  --benchmark edit/EditSplat/eval/benchmark/<benchmark.json> \
  --cache_root edit/EditSplat/eval/cache/renders_<tag> \
  --metrics_root edit/EditSplat/eval/cache/metrics_<tag> \
  --summaries_root edit/EditSplat/eval/cache/summaries_<tag> \
  --pairs_per_sample 400 \
  --render_depth_source 1 \
  --render_depth_edit 0 \
  --compute_reproj 1 \
  --use_lpips 1 \
  --clip_backend transformers \
  --clip_model openai/clip-vit-base-patch32 \
  --open_clip_pretrained openai/clip-vit-base-patch32 \
  --device cuda
```

## 5. 指标集合（本任务建议）
- 语义：`clip_sim_mean`, `clip_dir_mean`, `clip_dir_consistency_mean`
- 一致性：`reproj_l1_mean`, `reproj_lpips_mean`, `reproj_visible_ratio_mean`
- 保真：`l1_to_src`, `psnr_to_src`, `ssim_to_src`, `lpips_to_src`
- 伪影：`hf_ratio_vs_src`, `clip_ratio`, `mv_rel_dist_mse`
- 几何保持：`vertex_ratio`

## 6. 红偏专项补充（离线本地）
建议在 eval 结果之外补一个颜色统计脚本（本地执行）：
- 对每个方法在 face ROI（或中心 ROI）统计 `mean(R-G)`, `mean(R-B)`；
- 与 source 同视角差分，得到 `red_cast_delta`；
- 输出 `csv` 与每方法均值/方差，辅助解释“偏红是否下降”。

