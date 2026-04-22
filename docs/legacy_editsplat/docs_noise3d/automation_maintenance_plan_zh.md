# 自动化实验维护方案（简化版）

## 1. 目录分工
- `exp_flowedit_3dnoise/scripts/`：实验启动、监控、离线分析脚本。
- `exp_flowedit_3dnoise/logs/`：运行日志（含 launcher/watch/eval）。
- `exp_flowedit_3dnoise/results/`：图表与补充统计结果。
- `eval/`：统一评估系统（benchmark、metrics、summaries）。
- `docs_noise3d/`：可交接文档（本目录）。

## 2. 最小闭环流程
1. 启动实验：`run_full_redfix_round1.sh`
2. 监控与自动收尾：`watch_and_eval_face_redfix_r2.sh`
3. 实时状态查看：`status_face_redfix_r2.sh`
4. 正式评估：`run_eval_face_redfix_r2.sh`
5. 专项统计：`analyze_redcast_metrics.py`
6. 结果回填：更新 `experiment_timeline_zh.md` 与 `agent_doc` 对应实验文档

## 3. 命名规范
- 输出目录：`full_real_<scene>_<method>_<round>`
- 日志文件：`<task>_<YYYYMMDD_HHMMSS>.log`
- benchmark：`benchmark_<scene>_<task>_<date>.json`

## 4. 变更控制
- 主干文件（`run_editing_flow.py`）默认不动；
- 优先通过 wrapper/脚本扩展；
- 每次实验新增文档记录“命令、超参、结果、失败、下一步”。

## 5. 下个 Agent 的工作顺序
1. 看 `agent.md` 与 `docs_noise3d/README.md`
2. 看 watcher 日志确认是否已出评估结果
3. 如果已完成，先读 `eval/cache/summaries_*` 与 `results/*.csv`
4. 再决定是否进入下一轮小幅调参
