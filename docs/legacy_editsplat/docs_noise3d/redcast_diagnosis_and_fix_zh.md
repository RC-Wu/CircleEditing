# 面部偏红问题：假设、定位与修复方案（中文）

## 1. 问题定义
现象：编辑后的人脸区域出现明显红/橙偏色，且多视角下强度不一致，严重影响编辑可信度。

任务要求：
- 不大改主干；
- 在真实 full setting 下与原始 baseline 对照；
- 方法尽量简洁稳健。

## 2. 多假设（按优先级）

### H1. 文本目标与 guidance 过强导致色彩过驱动
- 触发条件：`flow_tar_guidance_scale` 高，且目标提示对材质/颜色约束不精确。
- 预期表现：语义更强但肤色向暖色偏移，饱和度上升。

### H2. 局部掩码/MFG 与跨视角不稳定耦合，导致区域性色偏
- 触发条件：编辑区域只在分割块内，MFG 若不稳定会放大局部颜色异常。
- 预期表现：局部色块、边界附近异常、视角间闪烁。

### H3. 3D 噪声注入过强或优化目标过激，放大颜色漂移
- 触发条件：`noise_mix` 较高、`lambda_edit` 偏大、`lambda_id` 偏小。
- 预期表现：编辑方向增强但出现伪色和纹理噪点。

### H4. 依赖权重加载失败导致流程异常（工程问题）
- 本轮已确认：`sam_vit_h_4b8939.pth` 下载失败会直接中断流程。
- 修复后不再是主要原因。

## 3. 本轮修复策略（简洁版）

### S1. baseline 文本与采样强度降温（对应 H1）
- 把目标提示改为“neutral gray-white marble, no red tint”。
- `flow_tar_guidance_scale` 从 `5.5` 降到 `3.8`。
- `flow_n_max` 从 `24` 降到 `20`，避免过强后期改写。
- 增加 `flow_negative_prompt` 抑制红肤色/暖色偏。

### S2. 3D 噪声保守化（对应 H2+H3）
- 关闭 MFG：`--noise_no_mfg`。
- 降低噪声混合比：`noise_mix=0.32~0.35`。
- 预优化使用 `balanced` 且增加 identity/smooth 约束：
  - `lambda_id=1.20`
  - `lambda_smooth=0.35`
  - `lambda_prior=0.12`
  - `lambda_edit=0.55`

### S3. 工程稳定性修复（对应 H4）
- `setsid -f` 解决后台进程被清理问题。
- SAM 权重改本地软链接，避免反复下载失败。

## 4. 本轮对照实验矩阵（face, full setting）
- `full_real_face_baseline_ref_r2`：原始 baseline。
- `full_real_face_baseline_neutral_r2`：S1。
- `full_real_face_cov_init_nomfg_r2`：S1 + S2（无预优化）。
- `full_real_face_cov_balanced_nomfg_r2`：S1 + S2（轻量预优化）。

## 5. 判据（先验）
优先目标：
1. 红偏显著下降（视觉与颜色统计一致）。
2. 多视角一致性不退化（`reproj_*`, `clip_dir_consistency` 不明显变差）。
3. 语义达成保持可接受（`clip_dir_mean` 不大幅下降）。

## 6. 结果回填位（实验结束后填写）
- 最优方法：TBD
- 红偏改善幅度：TBD
- 主要副作用：TBD
- 下一轮建议：TBD

## 7. 对比输出硬约束（本轮起）
1. 指标表必须同时覆盖：
   - 无 3D noise baseline：`baseline_ref` / `baseline_neutral`
   - 3D noise：`cov_init_nomfg` / `cov_balanced_nomfg`
   - SOTA 参考：`flux2-dev` / `sd35-large` / `qwen-image-edit` / `z-image`（可来自最近可用 benchmark，需标注是否同场景）
2. 可视化必须同时交付：
   - `eval` 自带 `gallery.html`（src/edit 直观对照）
   - 同视角 panel（`face_redfix_r2_panel_20260303.png`）
   - 综合 HTML 报告（`face_redfix_r2_comparison_report_20260303.html`）
3. 结论写法必须分开：
   - 同 benchmark 的严格对比结论；
   - 跨 benchmark 的 SOTA 参考性结论（不可直接下“胜负”）。
