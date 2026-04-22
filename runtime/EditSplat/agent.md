# EditSplat Agent Entry

本文件是本仓库给自动化 Agent 的统一查询入口（重点覆盖 3DGS + FlowEdit + 3D 噪声实验）。

## 1) 先看哪里
- 项目 docs 根目录：`edit/EditSplat/docs/README.md`
- 总入口：`edit/EditSplat/docs_noise3d/README.md`
- 当前主线实验状态：`edit/EditSplat/docs_noise3d/experiment_timeline_zh.md`
- 运行耗时拆解：`edit/EditSplat/docs_noise3d/runtime_cost_breakdown_20260303.md`
- 红偏问题定位与方案：`edit/EditSplat/docs_noise3d/redcast_diagnosis_and_fix_zh.md`
- 评估协议（强制使用 eval 子系统）：`edit/EditSplat/docs_noise3d/eval_protocol_eval_folder_zh.md`
- 自动化维护方案：`edit/EditSplat/docs_noise3d/automation_maintenance_plan_zh.md`
- 下一个 Agent 直接可用提示词：`edit/EditSplat/docs_noise3d/next_agent_handoff_prompt_zh.md`
- 本次 sandbox：`edit/EditSplat/sandboxes/20260303_noise3d_monitoring_and_doc_scheme/`

## 2) 关键原则
- 原始主流程在 `edit/EditSplat/run_editing_flow.py`，尽量不改。
- 新实验代码放在 `edit/EditSplat/exp_flowedit_3dnoise/`。
- 评估统一走 `edit/EditSplat/eval/`，不再使用自定义临时评估脚本。
- 缓存与权重路径固定在 `/dev-vepfs/rc_wu/rc_wu/cache` 下。

## 3) 当前任务快捷命令
- 启动 face 红偏修复并行实验（4 卡）：
  - `bash edit/EditSplat/exp_flowedit_3dnoise/scripts/run_full_redfix_round1.sh`
- 看运行中任务：
  - `pgrep -af "run_editing_flow_(baseline|3dnoise)_wrapper.py .*full_real_face_"`
- 看单次进度：
  - `bash edit/EditSplat/exp_flowedit_3dnoise/scripts/status_face_redfix_r2.sh`
- 看日志：
  - `ls -lt edit/EditSplat/exp_flowedit_3dnoise/logs/full_redfix_face_*.log | head`
  - `tail -n 80 <log_path>`
- 看持续监控日志（每 60s）：
  - `tail -n 120 edit/EditSplat/exp_flowedit_3dnoise/logs/progress_watch_face_redfix_r2_*.log`

## 4) 评估入口（eval）
- benchmark 在：`edit/EditSplat/eval/benchmark/`
- 一键评估：`python edit/EditSplat/eval/scripts/run_full_eval.py ...`
- 汇总结果：`edit/EditSplat/eval/cache/summaries_*/summary.csv`
- 对比可视化（本轮自动产物）：
  - `edit/EditSplat/eval/cache/summaries_face_redfix_r2_20260303/gallery.html`
  - `edit/EditSplat/exp_flowedit_3dnoise/results/face_redfix_r2_panel_20260303.png`
  - `edit/EditSplat/exp_flowedit_3dnoise/results/face_redfix_r2_comparison_report_20260303.html`
