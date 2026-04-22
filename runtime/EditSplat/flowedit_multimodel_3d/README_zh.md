# FlowEdit MultiModel 3D Wrapper（实验中）

## 1. 目的
在不修改 `run_editing_flow.py` 主体逻辑的情况下，把 2D 多模型 adapter（`flowedit_multimodel/src/flowedit_adapters.py`）接到 3D 编辑流程中。

当前实现方式：
1. 通过 wrapper 启动 `run_editing_flow.py`。
2. 运行前 monkeypatch `Editsplat_Pipeline.edit_image` 与 `edit_image_MFG`。
3. 两个方法内部统一调用 adapter 的 `edit(...)`。

## 2. 文件
- `scripts/run_editing_flow_multimodel_wrapper.py`

## 3. 使用示例
```bash
export HF_TOKEN="你的token"
CUDA_VISIBLE_DEVICES=0 \
/dev-vepfs/rc_wu/rc_wu/envs/editsplat_ttt3r_unified_copy/bin/python \
/dev-vepfs/rc_wu/rc_wu/edit/EditSplat/flowedit_multimodel_3d/scripts/run_editing_flow_multimodel_wrapper.py \
  --model_key flux2-dev \
  --hf_token "$HF_TOKEN" \
  --hf_home /dev-vepfs/rc_wu/rc_wu/cache/hf_home \
  --adapter_resize_side 512 \
  --source_path <your_scene_path> \
  --model_path <your_output_path> \
  --source_checkpoint <your_ckpt_path> \
  --flow_src_prompt "..." \
  --flow_tar_prompt "..." \
  --iter_train 1200
```

说明：
1. wrapper 参数后面直接接原始 `run_editing_flow.py` 参数。
2. `--model_id` 可选，用于覆盖仓库 ID（例如本地路径）。

## 4. 已知限制
1. 该 wrapper 当前属于“快速接入版”，`edit_image_MFG` 内部未保留原始 `lambda_M/lambda_S` 外力项细节，仅走 adapter FlowEdit 主流程。
2. 若模型权重未完整下载，会在 adapter 初始化时报错。
3. 3DNoise 预优化（`exp_flowedit_3dnoise`）与该 wrapper 尚未做深度融合，只能先作为多模型基线入口。

## 5. 指标评估建议
输出后可复用：
- `/dev-vepfs/rc_wu/rc_wu/edit/EditSplat/exp_flowedit_3dnoise/scripts/eval_full_real_outputs.py`

