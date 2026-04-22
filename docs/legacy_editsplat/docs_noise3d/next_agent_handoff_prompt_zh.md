# 下一个 Agent 交接提示词（可直接复制）

```text
[system prompt]
请把 /dev-vepfs/rc_wu/rc_wu/agent_doc/AGENT_DOCS.md 作为强制入口，先加载命中的 POLICIES 再执行；若任务属于 mesh_generation 或 3d_editing，必须进入 PROJECTS/<project>/PROJECT_DOCS.md；发现文档库问题可按 PB-TOOLS-DOC-LIBRARY-REFACTOR 改造并保留迁移链路。该会话中的所有任务和所有回答均需满足上述要求。

[user prompt]
请接手 EditSplat 的 3DGS noise 实验，先读：
1) /dev-vepfs/rc_wu/rc_wu/edit/EditSplat/agent.md
2) /dev-vepfs/rc_wu/rc_wu/edit/EditSplat/docs_noise3d/README.md
3) /dev-vepfs/rc_wu/rc_wu/edit/EditSplat/docs_noise3d/experiment_timeline_zh.md
4) /dev-vepfs/rc_wu/rc_wu/edit/EditSplat/docs_noise3d/redcast_diagnosis_and_fix_zh.md
5) /dev-vepfs/rc_wu/rc_wu/edit/EditSplat/docs_noise3d/eval_protocol_eval_folder_zh.md

目标：
- 跟进 full_redfix_face_*_20260303_081917 这批实验是否完成；
- 优先检查 watcher 状态：
  /dev-vepfs/rc_wu/rc_wu/edit/EditSplat/exp_flowedit_3dnoise/logs/watch_face_redfix_r2_20260303_090023.log
- 用 edit/EditSplat/eval 跑正式离线评估，生成 summary/by_method/table/gallery；
- 对比 baseline 与 3D noise 方法，重点分析面部偏红问题；
- 若结果不理想，只做小幅、稳健调参（避免复杂 trick）；
- 把本轮实验命令、配置、结果、图像分析更新到 docs_noise3d 与 agent_doc/PROJECTS/3d_editing。

约束：
- 不大改 run_editing_flow.py 主干；
- 新代码放在 exp_flowedit_3dnoise；
- 评估只用 eval/；
- 权重与缓存使用 /dev-vepfs/rc_wu/rc_wu/cache；
- 任何下载失败先定位本地已有缓存并优先复用。
```

