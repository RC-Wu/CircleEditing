# EditSplat Eval (No-GT, Step2+)

本目录提供一个独立评估流水线，尽量不改 `run_editing_flow.py` 主流程，面向论文级 3D 编辑评估（无 GT）。

## 1. 指标覆盖

- 语义达成
  - `clip_sim_mean` (CLIP image-target text similarity)
  - `clip_dir_mean` (directional: `(e_t - e_s)` vs `(img_edit - img_src)`)
  - `clip_dir_consistency_mean` (跨视角 `d_img` 一致性)
- 多视角一致性（无 GT）
  - `reproj_l1_mean`
  - `reproj_lpips_mean` (可选)
  - `reproj_visible_ratio_mean`
- 保真与伪影（复用已有方案）
  - `l1_to_src`, `psnr_to_src`, `ssim_to_src`, `lpips_to_src`
  - `hf_ratio_vs_src`, `clip_ratio`, `mv_rel_dist_mse`
  - `vertex_ratio`
- 效率
  - `runtime_sec`, `peak_mem_mib`, `flow_steps`, guidance 等

## 2. 目录结构

- `core/`: 指标与缓存核心实现
- `scripts/`: CLI 调度脚本
- `configs/`: 默认评测配置
- `benchmark/`: benchmark 清单
- `cache/`: 评测缓存/结果（renders, metrics, summaries）

## 3. 快速开始

从 `EditSplat/` 根目录执行。

建议先固定缓存路径（符合 `/dev-vepfs/rc_wu/rc_wu/cache` 路径政策）：

```bash
export HF_HOME=/dev-vepfs/rc_wu/rc_wu/cache/hf_home
export HF_HUB_CACHE=/dev-vepfs/rc_wu/rc_wu/cache/hf_home/hub
export TORCH_HOME=/dev-vepfs/rc_wu/rc_wu/cache/torch_hub
export PIP_CACHE_DIR=/dev-vepfs/rc_wu/rc_wu/cache/pip
export TMPDIR=/dev-vepfs/rc_wu/rc_wu/cache/tmp
```

### 3.1 自动生成 benchmark

```bash
python eval/scripts/build_benchmark.py \
  --search_roots output/flowedit_multimodel_exp/reconnect_round_20260224_3d_multicase_hk8_r1/fangzhou \
  --split train \
  --edit_id marble_face \
  --out eval/benchmark/benchmark_fangzhou.json
```

也可直接手写 benchmark（见 `benchmark/benchmark_example.json`）。

### 3.2 构建 render/depth cache

```bash
python eval/scripts/render_cache.py \
  --benchmark eval/benchmark/benchmark_fangzhou.json \
  --cache_root eval/cache/renders \
  --render_depth_source 1 \
  --render_depth_edit 0 \
  --overwrite 0 \
  --device cuda
```

### 3.3 计算指标

```bash
python eval/scripts/compute_metrics.py \
  --benchmark eval/benchmark/benchmark_fangzhou.json \
  --render_cache_root eval/cache/renders \
  --metrics_root eval/cache/metrics \
  --config eval/configs/eval_default.json \
  --pairs_per_sample 200 \
  --compute_reproj 1 \
  --use_lpips 1 \
  --device cuda
```

### 3.4 聚合输出

```bash
python eval/scripts/aggregate.py \
  --metrics_root eval/cache/metrics \
  --summaries_root eval/cache/summaries
```

输出：
- `eval/cache/summaries/summary.csv`
- `eval/cache/summaries/by_method.csv`
- `eval/cache/summaries/table_main.tex`
- `eval/cache/summaries/gallery.html`

### 3.5 一键执行

```bash
python eval/scripts/run_full_eval.py \
  --benchmark eval/benchmark/benchmark_fangzhou.json \
  --cache_root eval/cache/renders \
  --metrics_root eval/cache/metrics \
  --summaries_root eval/cache/summaries \
  --pairs_per_sample 200 \
  --render_depth_source 1 \
  --compute_reproj 1 \
  --use_lpips 1
```

## 4. 复现性与缓存策略

- 每个样本缓存目录写 `meta.json`，包含视角列表、迭代、git commit。
- `--overwrite 0` 时默认断点续跑（存在即跳过）。
- 每个样本指标写入一个 JSON，并同步 append 到 `results.jsonl`。

## 5. 环境建议

优先使用包含以下依赖的环境：
- `torch + cuda`
- `diff_gaussian_rasterization`（用于深度渲染）
- `clip` 或 `open_clip_torch`
- `lpips`（若启用 LPIPS 指标）

可用性探测：

```bash
python eval/scripts/probe_envs.py --out_json eval/cache/summaries/env_probe.json
```

`probe_envs.py` 会输出三类能力判定：
- `can_render_depth`: 能否执行深度缓存渲染（scene + gaussian_renderer + diff_gauss）
- `can_metrics_full`: 能否执行完整指标（CLIP + LPIPS）
- `can_full_eval`: 能否跑完整 Step2+ 评测链路
