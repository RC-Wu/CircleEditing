# full real face 红偏修复实验（r2，2026-03-03）

## 1. 背景
在真实 setting 下，face 编辑存在明显红偏。该轮目标是用最小改动做稳健修复，并与原始 baseline 对照。

## 2. 实验约束
- 主流程不改：`run_editing_flow.py`
- 新逻辑仅在 wrapper/脚本：`exp_flowedit_3dnoise/`
- 评估统一：`eval/`

## 3. 本轮并行实验
启动脚本：`exp_flowedit_3dnoise/scripts/run_full_redfix_round1.sh`

方法列表：
1. `full_real_face_baseline_ref_r2`
2. `full_real_face_baseline_neutral_r2`
3. `full_real_face_cov_init_nomfg_r2`
4. `full_real_face_cov_balanced_nomfg_r2`

关键差异：
- baseline_ref：维持原有 prompt 与较高 tar guidance。
- baseline_neutral：中性 marble prompt + lower guidance + negative prompt。
- cov_init_nomfg：3D 噪声保守注入（no preopt, no MFG）。
- cov_balanced_nomfg：3D 噪声 + 轻量 balanced preopt（no MFG）。

## 4. 过程中遇到的问题与处理
### 4.1 后台任务被回收
- 现象：`nohup &` 启动后日志为空，进程消失。
- 根因：当前执行器会回收未彻底脱离会话的后台子进程。
- 处理：改为 `setsid -f` 启动。

### 4.2 SAM 权重下载失败
- 现象：`LangSAM` 下载 `sam_vit_h_4b8939.pth` 失败，流程中断。
- 根因：目标目录为 `cache/torch_hub/hub/checkpoints`，而完整权重存在于 `cache/torch/hub/checkpoints`。
- 处理：软链接复用已有完整权重，删除 partial。

## 5. 当前状态
- 本轮已成功进入正式编辑阶段（`Initial editing progress` 正在增长）。
- 结果图与评估将在 run 结束后补充到本文件。

## 6. 待补充结果模板
### 6.1 定量表（eval）
| method | clip_dir_mean | clip_dir_consistency | reproj_l1 | lpips_to_src | hf_ratio_vs_src |
|---|---:|---:|---:|---:|---:|
| baseline_ref_r2 | TBD | TBD | TBD | TBD | TBD |
| baseline_neutral_r2 | TBD | TBD | TBD | TBD | TBD |
| cov_init_nomfg_r2 | TBD | TBD | TBD | TBD | TBD |
| cov_balanced_nomfg_r2 | TBD | TBD | TBD | TBD | TBD |

### 6.2 视觉分析
- 红偏强度（主观）：TBD
- 局部伪影：TBD
- 多视角稳定性：TBD

