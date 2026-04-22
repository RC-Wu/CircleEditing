# 3DGS Noise 实验文档索引（FlowEdit / EditSplat）

## 目标
该目录用于保存 3DGS 噪声初始化/优化相关实验的完整上下文，支持在丢失会话上下文后由新 Agent 快速接手。

## 文档列表
- `experiment_timeline_zh.md`：按时间记录所有关键实验、配置、异常、修复与结果路径。
- `runtime_cost_breakdown_20260303.md`：face_redfix_r2 完整流程分阶段耗时/资源消耗与优化建议。
- `redcast_diagnosis_and_fix_zh.md`：面部偏红问题的假设、排查路径、修复策略与对照设计。
- `eval_protocol_eval_folder_zh.md`：统一评估协议（强制使用 `edit/EditSplat/eval`）。
- `automation_maintenance_plan_zh.md`：自动化实验维护与目录分工方案（简化版）。
- `next_agent_handoff_prompt_zh.md`：直接给下一个 Agent 的可执行提示词模板。

## 代码与结果主路径
- 实验代码：`edit/EditSplat/exp_flowedit_3dnoise`
- 正式评估：`edit/EditSplat/eval`
- 输出结果：`edit/EditSplat/output/flowedit_3dnoise_exp`
- 日志：`edit/EditSplat/exp_flowedit_3dnoise/logs`

## 当前进行中的正式实验（2026-03-03）
- 批次名：`full_redfix_face_*_20260303_081917`
- 启动脚本：`edit/EditSplat/exp_flowedit_3dnoise/scripts/run_full_redfix_round1.sh`
- 数据集：`dataset/dataset/face`
- 目标：在真实 full setting 下定位/缓解面部红偏，并与 baseline 对比。

## 运行监控（持续）
- 单次快照：
  - `bash edit/EditSplat/exp_flowedit_3dnoise/scripts/status_face_redfix_r2.sh`
- 持续日志（每 60s 自动采样）：
  - `edit/EditSplat/exp_flowedit_3dnoise/logs/progress_watch_face_redfix_r2_*.log`
- 任务存活/自动收尾 watcher：
  - `edit/EditSplat/exp_flowedit_3dnoise/logs/watch_face_redfix_r2_*.log`

## 对比与可视化产物（本轮自动生成）
- Eval 指标汇总：
  - `edit/EditSplat/eval/cache/summaries_face_redfix_r2_20260303/summary.csv`
  - `edit/EditSplat/eval/cache/summaries_face_redfix_r2_20260303/by_method.csv`
- Eval 图库（HTML）：
  - `edit/EditSplat/eval/cache/summaries_face_redfix_r2_20260303/gallery.html`
- 红偏统计：
  - `edit/EditSplat/exp_flowedit_3dnoise/results/face_redfix_r2_redcast_metrics_20260303.csv`
- 同视角 panel：
  - `edit/EditSplat/exp_flowedit_3dnoise/results/face_redfix_r2_panel_20260303.png`
- 综合对比报告（HTML，含 baseline/3d-noise/SOTA 参考）：
  - `edit/EditSplat/exp_flowedit_3dnoise/results/face_redfix_r2_comparison_report_20260303.html`
