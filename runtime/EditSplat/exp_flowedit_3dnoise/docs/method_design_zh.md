# 方法设计（面向 EditSplat 真实设置）

## 1. 设计原则

1. 不改原始 `run_editing_flow.py`，全部改动在 `exp_flowedit_3dnoise` 内实现。
2. 与 baseline 完全同设置对比：同数据、同视角数、同 FlowEdit 参数、同 MFG/AGT/3DGS 优化流程。
3. 重点针对 3DGS 本身：通过 3D 基元构造噪声，而非只做频谱技巧。
4. 针对 EditSplat 的任务特性：编辑区域是掩膜前景，背景应保持稳定。

## 2. 从 baseline 到 3D 噪声的改动位点

### 2.1 baseline 噪声
FlowEdit 每步使用 `torch.randn_like` 产生前向噪声，视角间相互独立。

### 2.2 本方法噪声
在不改原始代码的前提下，运行时 patch `edit_image/edit_image_MFG` 内部噪声采样：

`eps_mix = rho * eps_3d(camera) + sqrt(1-rho^2) * eps_rand`

其中：
- `eps_3d(camera)` 由共享 3DGS 锚点噪声场投影得到。
- `rho` 是混合系数。

## 3. 3D 噪声场参数化

### 3.1 稀疏锚点
从 3DGS 基元中采样锚点集合 `A`，每个锚点携带 `C` 维噪声向量（`C` 为 FLUX token 通道）。

### 3.2 锚点选择策略

1. `coverage_opacity`
- 体素覆盖采样 + opacity 重要性。
- 优点：稳定、泛化好。

2. `opacity_topk`
- 仅保留高 opacity 区域。
- 优点：快；缺点：易局部偏置。

3. `hybrid`
- coverage 与 top-k 混合。
- 优点：兼顾全局和局部。

### 3.3 初始化策略

1. `coarse_smooth`
- 随机初始化后在粗网格做平滑。

2. `harmonic_lowfreq`（保留为备选）
- 低频基函数初始化后再平滑。

说明：本任务主线不依赖频谱技巧，核心仍是“3D 几何锚点 + 可见性投影 + 约束优化”。

## 4. 3D 到视角 token 的映射

对锚点投影到当前相机，按可见性与深度衰减加权，双线性 splat 到 token 网格：

`eps_3d^v(p) = sum_k w_{k,p} * n_k / sum_k w_{k,p}`

`w_{k,p} = opacity_k * exp(-gamma * depth_k) * bilinear(k,p)`

该映射使噪声天然具备多视角一致性来源。

## 5. 噪声优化目标

预优化阶段冻结 FLUX transformer，仅优化锚点噪声参数，目标由以下项组成：

1. 编辑增益项 `L_edit`
- 最大化源/目标速度差带来的编辑驱动力。

2. 身份保持项 `L_id`
- 约束去噪重建接近源分布，防止结构崩坏。

3. 平滑项 `L_smooth`
- 限制邻域锚点噪声突变。

4. 先验项 `L_prior`
- 防止偏离初始化过大。

5. 视角均衡项 `L_view_var`（balanced 模式）
- 抑制多视角损失方差过大，降低单视角过拟合。

## 6. 当前实现的三类可比方法

1. `cov_init_noopt`
- `coverage_opacity + coarse_smooth`，不做预优化。

2. `cov_baseopt`
- 在 `cov_init_noopt` 基础上做 `base` 预优化。

3. `hybrid_balanced`
- `hybrid` 锚点 + `balanced` 预优化。

以上三类与 baseline 构成“初始化差异 + 优化差异”的最小闭环比较集。

## 7. 与原始 baseline 的公平对照设计

1. baseline 路径  
- 使用 `exp_flowedit_3dnoise/scripts/run_editing_flow_baseline_wrapper.py`。  
- 该 wrapper 只做必要兼容补丁，不引入 3D 噪声注入。  

2. 3D 噪声路径  
- 使用 `exp_flowedit_3dnoise/scripts/run_editing_flow_3dnoise_wrapper.py`。  
- 除噪声初始化/优化/注入外，其余流程与 baseline 一致。  

3. 对齐条件  
- 同数据、同视角数、同 prompt、同 `flow_steps/guidance/seed`、同 `epoch/filtering_ratio`。  
- 输出都在 `output/flowedit_3dnoise_exp/*`，且所有代码放在独立实验目录。  

## 8. 调参策略（真实 setting）

按“先稳再强”的顺序做：

1. 初始化强度  
- `noise_mix` 先在 `0.45~0.60` 扫描。  
- 经验：过高会导致纹理/色彩伪影，过低与 baseline 差异不明显。  

2. 预优化步数与学习率  
- `noise_opt_iters`: `8/16/24`。  
- `noise_opt_lr`: `0.01/0.018/0.02`。  
- 经验：`iters` 增大时需同步降低 `lr`，否则噪声场易不稳定。  

3. 损失权重平衡  
- 先固定 `lambda_id >= lambda_edit` 保结构，再逐步增加 `lambda_edit`。  
- `balanced` 模式下再增加 `lambda_view_var` 抑制单视角爆点。  

4. 锚点规模  
- `noise_max_anchors`: `2e4~4e4`。  
- 锚点过少会削弱一致性，过多会提高优化成本且可能引入高频噪纹。  

## 9. 掩膜区域导向的下一步增强（针对任务特性）

当前版本尚未把 LangSAM 掩膜直接纳入噪声参数更新；下一轮建议按以下顺序加入：

1. 掩膜可见度加权锚点选择
- 仅统计在目标区域可见频次高的锚点进入优化集。

2. 前景/背景双温度混合
- 前景 `rho_fg` 高，背景 `rho_bg` 低。

3. 区域分解损失
- 前景提升 `L_edit` 权重；背景提升 `L_id/L_smooth` 权重。

这三项将比频谱增强更贴合 EditSplat 的真实编辑机制。

## 10. 工程实现位置

- 主 wrapper：`exp_flowedit_3dnoise/scripts/run_editing_flow_3dnoise_wrapper.py`
- 噪声场：`exp_flowedit_3dnoise/src/noise_field.py`
- 预优化：`exp_flowedit_3dnoise/src/noise_optimizer.py`
- 评测（当前主线）：`eval/scripts/run_full_eval.py`（位于 `edit/EditSplat/eval/`）
- 历史脚本（仅做旧结果兼容）：`exp_flowedit_3dnoise/scripts/eval_full_real_outputs.py`

全部改动在独立实验目录内，未修改原始项目核心文件。
