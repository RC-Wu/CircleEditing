# face_redfix_r2 完整流程耗时与资源拆解（2026-03-03）

## 1) 关键结论
1. 你现在看到的超长时长，主因确实是 FlowEdit 多视图阶段（`Multi-view reprojection progress`）。
2. 在 full setting（56 train views）下，单方法总墙钟约 `2h19m ~ 2h23m`，其中 reprojection 阶段占 `89.3% ~ 95.5%`。
3. 3DGS 的 `EPOCH 0` 只要 `3~8s`，不是瓶颈。
4. 训练后处理（eval+红偏统计+panel+HTML）已实测跑通，总计约 `91.69s`。

## 2) 证据来源
- 训练日志（4 组 full setting）
  - `exp_flowedit_3dnoise/logs/full_redfix_face_baseline_ref_20260303_081917.log`
  - `exp_flowedit_3dnoise/logs/full_redfix_face_baseline_neutral_20260303_081917.log`
  - `exp_flowedit_3dnoise/logs/full_redfix_face_cov_init_nomfg_20260303_081917.log`
  - `exp_flowedit_3dnoise/logs/full_redfix_face_cov_balanced_nomfg_20260303_081917.log`
- 实测 profile（空闲卡）
  - `exp_flowedit_3dnoise/logs/profile_face_baseline_ref_v8_20260303_104510.log`
- 手动后处理链路耗时
  - `exp_flowedit_3dnoise/logs/manual_post_face_redfix_r2_20260303_1102.log`

## 3) 完整流程与阶段开销

| 阶段 | 主要计算 | 资源特征 | 耗时（本轮证据） |
|---|---|---|---|
| A. 模型加载/场景读取 | FLUX + ImageReward + GroundingDINO + 相机/点云加载 | 单卡，IO+GPU混合 | full run 中包含在墙钟与 tracked 差值，约 `3.7~7.6 min` |
| B. Initial editing | 每个 train view 做一次 `edit_image` | 单卡高负载 | `6~14.5 min`（56 views） |
| C. Depth processing | 深度预处理 | 轻量 | `0~2s` |
| D. Multi-view reprojection | 每目标视角：选源视角、reproject、LangSAM、`edit_image_MFG` | 单卡持续重负载，CPU/GPU混合 | `121~129 min`（56 views） |
| E. Attention + 3DGS epoch0 | 权重聚合 + 一轮 3DGS 优化 | 短时计算 | `3~8s` |
| F. 结果渲染 | train/test 渲染导出 | 单卡中等负载 | 约 `15~20s`（按日志渲染速率） |
| G. Eval 聚合链路 | render_cache + compute_metrics + aggregate + redcast + panel + report | 单卡+CPU混合 | `82.016 + 5.152 + 4.479 + 0.043 = 91.690s` |

## 4) full-setting（56 views）分方法耗时

| method | initial(s) | depth(s) | reproj(s) | epoch0(s) | tracked_total(s) | wall(s) | reproj占比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_neutral_r2 | 360 | 0 | 7742 | 4 | 8106 | 8367 | 95.5% |
| baseline_ref_r2 | 868 | 2 | 7265 | 3 | 8138 | 8593 | 89.3% |
| cov_init_nomfg_r2 | 363 | 0 | 7609 | 8 | 7980 | 8330 | 95.4% |
| cov_balanced_nomfg_r2 | 366 | 0 | 7606 | 7 | 7979 | 8353 | 95.3% |

注：`wall - tracked` 主要是初始化和末尾渲染等非 tqdm 统计段。

## 5) 空闲卡实测 profile（要求已执行）
- 运行：`baseline_ref` 配置，加入 `--max_train_views 8`（用于时长剖析）
- 日志：`exp_flowedit_3dnoise/logs/profile_face_baseline_ref_v8_20260303_104510.log`
- 结果：
  - Initial editing: `60s`
  - Depth: `0s`
  - Reprojection: `223s`
  - Epoch0: `~0s`
  - 总时长：`383.318s`
- 结论：即使缩到 8 视角，reprojection 仍是最大单段耗时（约 58% 总时长）。

## 6) 为什么慢（机制层）
1. `run_editing_flow.py` 在 `Multi-view reprojection` 中对每个目标视角执行一次完整编辑推理。
2. 每个目标视角还要处理多个源视角（当前配置约 5 个）和 mask/reprojection 逻辑。
3. 目前配置 `flow_steps=28`，每次推理成本不低；56 视角下会被线性放大。
4. 3DGS 优化轮数很少，因此后段几乎不贡献总时长。

## 7) 降耗方案（按优先级）

### P0：不改大框架先降时长
1. 降目标视角数（例如 56 -> 24/16 的关键视角子集）。
2. 降每目标的源视角数（例如 5 -> 3）。
3. 降 `flow_steps`（例如 28 -> 14/12）并验证质量退化边界。
4. 对低变化视角跳过 MFG 或直接复用邻近视角结果。

### P1：保持质量的工程优化
1. 对 LangSAM mask 做缓存（按视角与 prompt 复用）。
2. 将重投影与可并行的数据准备移到 CPU 异步队列，减少 GPU 等待。
3. 拆分为两阶段：先低分辨率粗编辑筛选，再对关键视角高质量重跑。

### P2：结构替换（你提到的 ttt3r 方向）
1. 用 ttt3r 提供跨视角一致性约束，目标是减少必须显式编辑的视角数量。
2. 若一致性模型可让有效编辑视角降到 `8~16`，理论上可把主耗时压到当前的 `1/3 ~ 1/6`。
3. 这一路线仍需保留 baseline（无 3d noise）和当前 FlowEdit 管线作为对照，防止只看到速度收益看不到质量回退。

## 8) 当前对比交付状态（已补齐）
- 指标：
  - `eval/cache/summaries_face_redfix_r2_20260303/summary.csv`
  - `eval/cache/summaries_face_redfix_r2_20260303/by_method.csv`
- 可视化：
  - `eval/cache/summaries_face_redfix_r2_20260303/gallery.html`
  - `exp_flowedit_3dnoise/results/face_redfix_r2_panel_20260303.png`
  - `exp_flowedit_3dnoise/results/face_redfix_r2_comparison_report_20260303.html`
- 专项红偏：
  - `exp_flowedit_3dnoise/results/face_redfix_r2_redcast_metrics_20260303.csv`
