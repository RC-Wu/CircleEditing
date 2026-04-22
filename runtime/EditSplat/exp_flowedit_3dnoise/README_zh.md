# FlowEdit 3D 噪声优化实验（中文）

该目录是独立实验区：
- 不修改原始 `run_editing_flow.py`；
- 代码、日志、文档、结果全部在本目录与 `output/flowedit_3dnoise_exp/*`。

## 主要能力

1. 基于 3DGS 的 3D 噪声初始化（多方案）
- `coverage_opacity`
- `opacity_topk`
- `hybrid`
- `harmonic_lowfreq`（低频谐波初始化）

2. 噪声优化策略（多方案）
- `base`
- `snr`
- `delta`
- `balanced`（跨视角均衡正则）

3. FlowEdit 采样替换
- 支持随机噪声基线与优化后 3D 噪声对比。

4. 自动化分析（当前主线）
- 统一使用 `EditSplat/eval` 子系统做离线评估与汇总。

## 关键脚本

- 全流程 baseline（不改原文件）：
  - `exp_flowedit_3dnoise/scripts/run_editing_flow_baseline_wrapper.py`

- 全流程 3D 噪声实验（真实 setting）：
  - `exp_flowedit_3dnoise/scripts/run_editing_flow_3dnoise_wrapper.py`

- 小规模噪声方法对比（快速迭代）：
  - `exp_flowedit_3dnoise/scripts/run_noise_opt_flowedit.py`

- 正式 FLUX 小规模并行套件：
  - `exp_flowedit_3dnoise/scripts/run_flux_dev_small_suite.sh`
  - `exp_flowedit_3dnoise/scripts/run_flux_dev_small_suite_bear.sh`
  - `exp_flowedit_3dnoise/scripts/run_flux_dev_small_suite_round2.sh`
  - `exp_flowedit_3dnoise/scripts/run_flux_dev_small_suite_bear_round2.sh`

- 权重下载（正式 FLUX，续传）：
  - `exp_flowedit_3dnoise/scripts/download_flux_dev_fast.sh`
  - `exp_flowedit_3dnoise/scripts/download_flux_dev_components_seq.py`

- 结果分析：
  - `exp_flowedit_3dnoise/scripts/analyze_suite_results.py`
  - `exp_flowedit_3dnoise/scripts/eval_full_real_outputs.py`
  - `exp_flowedit_3dnoise/scripts/run_full_redfix_round1.sh`

## 文档索引

- 扩展文献综述（中）：`docs/literature_review_zh.md`
- 方法设计（中）：`docs/method_design_zh.md`
- 评价协议（中）：`docs/evaluation_protocol_zh.md`
- 全量真实实验进度（中）：`docs/full_real_round1_progress_zh_20260215.md`
- 全量真实实验报告（中）：`docs/full_real_round1_report_zh_20260215.md`
- face 红偏修复（中）：`docs/full_real_face_redfix_r2_zh_20260303.md`

结果文件（Round-1，历史）：
- 指标 JSON：`results/full_real_round1_eval_all_fangzhou.json`
- 多方法对比图：`results/full_real_round1_all_methods_panel_fangzhou.png`

评估主路径（当前）：
- `edit/EditSplat/eval/README.md`
- `edit/EditSplat/eval/benchmark/*.json`
- `edit/EditSplat/eval/cache/summaries_*/summary.csv`

## 快速运行示例

```bash
cd /dev-vepfs/rc_wu/rc_wu/edit/EditSplat
bash exp_flowedit_3dnoise/scripts/run_flux_dev_small_suite.sh
```

分析结果：

```bash
/dev-vepfs/rc_wu/rc_wu/envs/editsplat_test/bin/python \
  exp_flowedit_3dnoise/scripts/analyze_suite_results.py \
  --run_dir /dev-vepfs/rc_wu/rc_wu/edit/EditSplat/output/flowedit_3dnoise_exp/flux_dev_small_suite/<run_id> \
  --tar_prompt "a photo of a toy dinosaur made of white marble" \
  --use_lpips --use_imagereward
```

## 注意

- 若正式权重下载未完成，`--use_local_files_only` 会加载失败。
- 下载与镜像配置均为命令级环境变量，不修改全局网络配置。
- `run_editing_flow.py` 主干默认不改，实验通过 wrapper 注入。
