# 3DGS 噪声实验时间线（中文）

## 2026-02-14 ~ 2026-02-15：最小系统与首轮 full real

### 已完成内容
1. 建立独立实验目录：`edit/EditSplat/exp_flowedit_3dnoise/`。
2. 保持 `run_editing_flow.py` 不改，通过 wrapper 注入 3D 噪声：
   - `scripts/run_editing_flow_baseline_wrapper.py`
   - `scripts/run_editing_flow_3dnoise_wrapper.py`
3. 首轮 full real（fangzhou）4 方法：
   - `full_real_baseline_fangzhou_fixedrast_r1`
   - `full_real_cov_init_noopt_fangzhou_r1`
   - `full_real_cov_baseopt_fangzhou_r1`
   - `full_real_hybrid_balanced_fangzhou_r1`
4. 统一评估已迁移到 `edit/EditSplat/eval/`，并产出汇总。

### 首轮（fangzhou）核心离线评估结果
来源：`edit/EditSplat/eval/cache/summaries_fangzhou_3dnoise_r1_20260303/by_method.csv`

| method | clip_dir_mean↑ | clip_dir_consistency↑ | l1_to_src↓ | psnr_to_src↑ | mv_rel_dist_mse↓ |
|---|---:|---:|---:|---:|---:|
| full_real_baseline_fangzhou_fixedrast_r1 | 0.02095 | 0.77307 | 0.08772 | 18.1240 | 0.001839 |
| full_real_cov_init_noopt_fangzhou_r1 | 0.03954 | 0.81858 | 0.08198 | 18.6201 | 0.001589 |
| full_real_cov_baseopt_fangzhou_r1 | 0.05019 | 0.78175 | 0.08060 | 18.6047 | 0.001768 |
| full_real_hybrid_balanced_fangzhou_r1 | 0.05970 | 0.82080 | 0.08240 | 18.3449 | 0.001869 |

观察：3D 噪声方法在语义方向指标（`clip_dir_mean`）上相对 baseline 有提升，但不同方法存在纹理噪声与局部伪影差异。

---

## 2026-03-03：恢复任务 + 正式 face 红偏定位实验（本轮）

### A. 评估链路纠偏（已完成）
1. 强制切换到 `eval/` 官方子系统评估。
2. 避免 CLIP 本地权重 checksum 循环下载：
   - 使用 `--clip_backend transformers`
   - 使用 `--clip_model openai/clip-vit-base-patch32`
3. 输出路径：
   - `edit/EditSplat/eval/cache/metrics_fangzhou_3dnoise_r1_20260303`
   - `edit/EditSplat/eval/cache/summaries_fangzhou_3dnoise_r1_20260303`

### B. 并行启动机制修复（已完成）
问题：在当前执行器环境里，普通 `nohup ... &` 子进程会随父进程结束被清理，导致日志空文件。

修复：将实验启动改为 `setsid -f` 完全脱离会话。
- 文件：`edit/EditSplat/exp_flowedit_3dnoise/scripts/run_full_redfix_round1.sh`

### C. 外部权重异常修复（已完成）
问题：`LangSAM` 会下载 `sam_vit_h_4b8939.pth` 到
`/dev-vepfs/rc_wu/rc_wu/cache/torch_hub/hub/checkpoints/`，网络中断导致失败。

定位：完整权重已存在于
`/dev-vepfs/rc_wu/rc_wu/cache/torch/hub/checkpoints/sam_vit_h_4b8939.pth`（约 2.4GB）。

修复动作：
- 删除失败的 `.partial`
- 建立软链接：
  - `cache/torch_hub/hub/checkpoints/sam_vit_h_4b8939.pth -> cache/torch/hub/checkpoints/sam_vit_h_4b8939.pth`

### D. face 数据集正式并行实验（进行中）
启动时间：`2026-03-03 08:19 UTC`

日志文件：
- `edit/EditSplat/exp_flowedit_3dnoise/logs/full_redfix_face_baseline_ref_20260303_081917.log`
- `edit/EditSplat/exp_flowedit_3dnoise/logs/full_redfix_face_baseline_neutral_20260303_081917.log`
- `edit/EditSplat/exp_flowedit_3dnoise/logs/full_redfix_face_cov_init_nomfg_20260303_081917.log`
- `edit/EditSplat/exp_flowedit_3dnoise/logs/full_redfix_face_cov_balanced_nomfg_20260303_081917.log`

输出目录：
- `edit/EditSplat/output/flowedit_3dnoise_exp/full_real_face_baseline_ref_r2`
- `edit/EditSplat/output/flowedit_3dnoise_exp/full_real_face_baseline_neutral_r2`
- `edit/EditSplat/output/flowedit_3dnoise_exp/full_real_face_cov_init_nomfg_r2`
- `edit/EditSplat/output/flowedit_3dnoise_exp/full_real_face_cov_balanced_nomfg_r2`

对照组设计：
1. `baseline_ref_r2`：原始 baseline 风格（高 tar guidance=5.5）。
2. `baseline_neutral_r2`：中性 marble 提示 + 低 guidance（3.8）+ 负面提示抑制红偏。
3. `cov_init_nomfg_r2`：3D 噪声初始化注入，关闭 MFG，保守混合比，无预优化。
4. `cov_balanced_nomfg_r2`：3D 噪声 + 轻量 balanced 预优化 + 关闭 MFG。

### E. 自动收尾评估（已部署，等待触发）
为避免会话中断导致链路断开，已部署 watcher：
- 脚本：`edit/EditSplat/exp_flowedit_3dnoise/scripts/watch_and_eval_face_redfix_r2.sh`
- 状态日志：
  - `edit/EditSplat/exp_flowedit_3dnoise/logs/watch_face_redfix_r2_20260303_090023.log`
  - `edit/EditSplat/exp_flowedit_3dnoise/logs/report_watch_face_redfix_r2_20260303_101650.log`（兜底报告生成 watcher）
- 功能：
  1. 轮询 4 个 `full_real_face_*_r2` 任务是否结束；
  2. 结束后自动执行 `run_eval_face_redfix_r2.sh`（`eval/` 一键评估）；
  3. 自动执行 `analyze_redcast_metrics.py` 生成红偏量化表；
  4. 自动调用 `make_full_real_panels.py` 生成对比图。
  5. 自动调用 `build_face_redfix_report.py` 生成 HTML 综合报告（指标 + panel + SOTA 参考）。

### F. 运行进度快照（2026-03-03 10:26 UTC）
- `baseline_neutral`：初始编辑 100%，reprojection `50/56`，剩余约 `14:19`，ETA `2026-03-03 10:40 UTC`
- `baseline_ref`：初始编辑 100%，reprojection `43/56`，剩余约 `32:25`，ETA `2026-03-03 10:58 UTC`
- `cov_init_nomfg`：初始编辑 100%，reprojection `51/56`，剩余约 `11:21`，ETA `2026-03-03 10:37 UTC`
- `cov_balanced_nomfg`：初始编辑 100%，reprojection `51/56`，剩余约 `11:34`，ETA `2026-03-03 10:37 UTC`
- 预计主实验全部结束：以慢任务 `baseline_ref` 为准，约 `2026-03-03 10:59 UTC`。
- 新增持续监控日志（每 60 秒采样）：
  - `edit/EditSplat/exp_flowedit_3dnoise/logs/progress_watch_face_redfix_r2_20260303_100136.log`

### G. 对比与可视化约束（本轮起强制）
1. 指标必须同时包含：`baseline(no-3d-noise)`、`3d-noise`、`SOTA 参考`（若同场景 SOTA 缺失，明确标注 gap）。
2. 可视化必须同时包含：
   - `eval/cache/summaries_*/gallery.html`（原始 src/edit 对照）
   - `face_redfix_r2_panel_20260303.png`（同视角多方法拼图）
   - `face_redfix_r2_comparison_report_20260303.html`（统一浏览入口）
3. 结论口径必须分层：
   - 同 benchmark 同场景结论（可直接比较）；
   - 跨 benchmark 的 SOTA 参考结论（仅上下文参考，不做严格胜负判定）。

---

## 下一步（待本轮跑完）
1. 等 watcher 自动触发 eval + 红偏量化 + panel + HTML 对比报告。
2. 产出后优先核对 `by_method.csv` 与 `redcast_metrics.csv` 是否一致支持结论。
3. 若同场景 SOTA 组仍缺失，按新流程登记 gap 并补跑对应基线/SOTA。
4. 回写本文件与 `redcast_diagnosis_and_fix_zh.md` 的“结果与结论”段落。

---

## 2026-03-03 10:42~10:58 UTC：训练收敛、评估修复、结果落盘

### H. 训练阶段最终状态（已完成）
1. `watch_face_redfix_r2_20260303_090023.log` 记录 4 个训练任务于 `2026-03-03T10:42:37Z` 全部结束。
2. 单方法墙钟约 `2h19m ~ 2h23m`（56 视角 full setting）。
3. 耗时主段是 `Multi-view reprojection progress`（占 tracked time `89.3% ~ 95.5%`）。

### I. 自动评估阻塞与修复（已完成）
问题：
- watcher 调起 eval 时失败，`watch_face_redfix_launcher_20260303_090023.log` 中报错：
  - `run_full_eval.py: error: unrecognized arguments: 0`

根因：
- `run_eval_face_redfix_r2.sh` 误传 `--skip_render_cache 0`，而该参数是无参布尔开关。

修复：
1. 删除非法参数传值。
2. `watch_and_eval_face_redfix_r2.sh` 改为将 stderr 也写入评估日志（`2>&1 | tee`）。

### J. 手动后处理链路实测（空闲卡执行，已完成）
日志：`exp_flowedit_3dnoise/logs/manual_post_face_redfix_r2_20260303_1102.log`

分段耗时：
1. eval 主链路（render_cache + compute_metrics + aggregate）：`82.016s`
2. 红偏统计：`5.152s`
3. panel 生成：`4.479s`
4. HTML 报告：`0.043s`

总计后处理：`91.690s`

### K. 本轮 face_redfix_r2 指标快照（同 benchmark）
来源：`eval/cache/summaries_face_redfix_r2_20260303/by_method.csv`

| method | clip_dir_mean_mean↑ | clip_dir_consistency_mean_mean↑ | l1_to_src_mean↓ | psnr_to_src_mean↑ | mv_rel_dist_mse_mean↓ |
|---|---:|---:|---:|---:|---:|
| baseline_ref_r2 | 0.03041 | 0.52700 | 0.03376 | 25.6563 | 0.00040985 |
| baseline_neutral_r2 | 0.01678 | 0.51920 | 0.02834 | 26.7481 | 0.00040842 |
| cov_init_nomfg_r2 | 0.01136 | 0.49775 | 0.02827 | 26.7847 | 0.00039726 |
| cov_balanced_nomfg_r2 | 0.02100 | 0.49961 | 0.02965 | 26.3829 | 0.00035508 |

红偏专项（`red_idx_delta_mean`，越低越好）：
- baseline_ref_r2: `0.05665`
- baseline_neutral_r2: `0.02382`
- cov_init_nomfg_r2: `0.02366`
- cov_balanced_nomfg_r2: `0.02647`

### L. 可视化交付（已生成）
1. `eval/cache/summaries_face_redfix_r2_20260303/gallery.html`
2. `exp_flowedit_3dnoise/results/face_redfix_r2_panel_20260303.png`
3. `exp_flowedit_3dnoise/results/face_redfix_r2_comparison_report_20260303.html`

### M. SOTA 对照口径（本轮）
1. 同 benchmark 严格对比：当前仅 baseline + 3d-noise 组可做严格比较。
2. 跨 benchmark SOTA 参考：来自 `eval/cache/summaries/by_method.csv`（`flux2-dev/sd35-large/qwen-image-edit/z-image`），仅作参考定位，不做严格胜负结论。
3. 下一轮需要补同场景 SOTA 才能完成“同任务 SOTA”闭环。
