# 文献综述（扩展版，面向 3DGS 噪声初始化与优化）

## 1. 检索目标与范围
目标是回答两个问题：
1. 3DGS 上“初始化什么样的噪声”更有利于后续编辑优化。
2. 如何在不破坏多视角一致性的前提下优化这组噪声，使其对 FlowEdit + EditSplat 全流程有增益。

检索范围覆盖三条主线：
1. 2D 流/扩散编辑中的噪声与轨迹优化。
2. 多视角扩散中的一致噪声或共享隐变量。
3. 3D 编辑（NeRF/3DGS）在一致性和可控性上的工程策略。

## 2. 2D 流编辑与噪声优化主线（可迁移理论）

1. `CycleDiffusion` (2023): https://arxiv.org/abs/2210.05559  
核心启发：源/目标共享噪声轨迹可显著减少编辑漂移。

2. `DDS` (2023): https://arxiv.org/abs/2303.15413  
核心启发：源/目标分数差可以作为稳定编辑方向，适合构造噪声优化目标。

3. `FlowEdit` (2024): https://arxiv.org/abs/2412.08629  
核心启发：无反演编辑仍依赖每步前向扰动，噪声不是“无关随机项”。

4. `FlowAlign` (2025): https://arxiv.org/abs/2505.23145  
核心启发：编辑质量更依赖轨迹对齐与稳定约束，而非单步强刺激。

5. `FlowCycle` (2025): https://arxiv.org/abs/2504.08019  
核心启发：目标感知与循环一致性可抑制过编辑和语义偏移。

6. `DNAEdit` (2025): https://arxiv.org/abs/2506.01430  
核心启发：直接优化噪声是有效控制杆，可作为 3D 噪声参数优化依据。

7. `RFEdit` (2025): https://arxiv.org/abs/2508.07502  
核心启发：整流流模型的“轨迹可控性”可以通过噪声/流匹配进一步增强。

结论：2D 文献已经形成共识，噪声与轨迹是可优化变量，且必须与稳定性约束联合建模。

## 3. 多视角一致噪声主线（3D 一致性直接证据）

1. `MVDream` (2023): https://arxiv.org/abs/2308.16512  
结论：多视角共享隐变量显著有利于几何一致性。

2. `SyncDreamer` (2023): https://arxiv.org/abs/2309.03453  
结论：同步扩散过程可减少跨视角几何抖动。

3. `SyncNoise` (2024): https://arxiv.org/abs/2406.17396  
结论：几何一致噪声预测能改善 3D 场景编辑稳定性。

4. `DGE` (2024): https://arxiv.org/abs/2401.16452  
结论：扩散特征/注意力在编辑时的几何约束是跨视角稳定的关键。

5. `3D-Consistent Multi-View Editing by Diffusion Guidance` (2025): https://arxiv.org/abs/2511.22228  
结论：即使不显式做 3D 噪声，也必须强制跨视图一致优化。

结论：共享噪声或共享约束不是可选项，而是 3D 编辑的必要条件。

## 4. 3D 编辑主线（与本任务 setting 直接相关）

1. `Instruct-NeRF2NeRF` (2023): https://arxiv.org/abs/2303.12789  
2. `Instruct 3D-to-3D` (2023): https://arxiv.org/abs/2303.15780  
3. `GaussianEditor` (2023): https://arxiv.org/abs/2311.16037  
4. `GSEdit` (2024): https://arxiv.org/abs/2403.05154  
5. `GaussCtrl` (2024): https://arxiv.org/abs/2403.08733  
6. `TIGER` (2024): https://arxiv.org/abs/2405.14455  
7. `GSEditPro` (2024): https://arxiv.org/abs/2411.10033  
8. `EditSplat` (2024): https://arxiv.org/abs/2412.11520  
9. `ICE-G` (2025): https://arxiv.org/abs/2507.18154  
10. `GaussEdit` (2025): https://arxiv.org/abs/2509.26055  
11. `Instruct-GS2GS` 项目页（非 arXiv 论文条目）：https://cvachha.github.io/instruct-gs2gs/

校正说明：
1. 之前的 `GaussianEditor` 链接写成了 `2403.14213`，该 arXiv 对应的是另一篇异常检测论文，已纠正为 `2311.16037`。  
2. `Instruct-GS2GS` 在当前检索中主要以项目页形式发布，不应强行映射到不对应的 arXiv ID。

## 5. 从文献归纳出的“3DGS 噪声初始化”可行族

## 5.1 覆盖优先（Coverage）
定义：按体素覆盖均匀采样锚点，保障全空间可见区域都可被噪声场解释。  
优点：不易漏掉局部。  
风险：对编辑区域聚焦不足。

## 5.2 重要性优先（Opacity / Saliency）
定义：按 opacity 或可见贡献加权抽样。  
优点：效率高、编辑刺激强。  
风险：易过拟合局部高权重区域，背景伪影更明显。

## 5.3 可见频次优先（Visibility Frequency）
定义：统计多视角可见次数，保留“跨视角公共锚点”。  
优点：天然偏向一致性。  
风险：可能牺牲难视角细节。

## 5.4 混合采样（Hybrid）
定义：`A = α A_cov + (1-α) A_imp`。  
优点：覆盖性与编辑强度折中。  
风险：需要额外调 `α`。

## 5.5 掩膜感知采样（Mask-Aware，任务特化）
定义：仅在编辑掩膜高可见区域提升采样密度。  
优点：最贴合 EditSplat 真实任务（只编辑分割区域）。  
风险：掩膜错误会导致噪声优化偏移。

## 6. 从文献归纳出的“3D 噪声优化”可行族

1. `base`：编辑增益 + 身份保持 + 平滑 + 先验。  
2. `balanced`：在 `base` 上加视角方差惩罚，抑制单视角过拟合。  
3. `snr`：按时序 SNR 重加权，避免不同流时间步梯度失衡。  
4. `mask-split`：前景/背景分解损失，前景偏编辑，背景偏保真。  
5. `mix-preserved`：优化时保留随机混合项，避免噪声坍塌到单一解。

对本任务最实用的组合是：
1. `coverage/hybrid` 初始化。  
2. `base/balanced` 轻量预优化。  
3. 采样期 `eps_mix = rho * eps_3d + sqrt(1-rho^2) * eps_rand`。

## 7. 文献对本实验的直接结论

1. 仅用每视角独立随机噪声通常对 3D 一致性不友好。  
2. 3D 一致噪声要和稳定正则同时优化，不能只追求编辑强度。  
3. EditSplat 任务里“编辑区域局部化”很强，掩膜感知噪声策略比纯频谱策略更有普适价值。  
4. 因此，“3DGS 锚点初始化 + 轻量预优化 + 混合注入 + 区域分解约束”是当前最可落地路线。
