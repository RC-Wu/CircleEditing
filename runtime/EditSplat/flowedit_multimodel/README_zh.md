# FlowEdit MultiModel（EditSplat 扩展）

## 1. 功能
- 在不改动主文件 `run_editing_flow.py` 的前提下，新增多模型 FlowEdit 适配。
- 支持：
  - FLUX 基线：`flux1-dev`
  - FLUX2：`flux2-dev`（gated）/ `flux2-klein-4b`（开放）
  - SD3.5：`sd35-large`（主实验）/ `sd35-large-turbo`（可选，需单独授权）/ `sd35-medium-turbo-open`（fallback）
  - Qwen：`qwen-image-edit`
  - Z-Image：`z-image`（FlowEdit 兼容性评估）
- 四卡并行调参脚本。
  - `run_tuning_4gpu.sh`：单环境四卡并行
  - `run_tuning_4gpu_mixed_env.sh`：混合环境四卡并行（`flux2*` 自动走 Python3.10 环境）

## 2. 目录
- `src/model_registry.py`：模型注册与默认超参
- `src/flowedit_adapters.py`：统一 FlowEdit 适配层
- `src/flux2_klein_pipeline_local.py`：本地 Flux2Klein pipeline（py3.9 兼容）
- `scripts/hf_fast_download.py`：加速下载脚本（镜像+断点续传）
- `scripts/hf_sequential_retry_download.py`：单文件顺序重试下载（弱网更稳）
- `scripts/tune_flowedit_model.py`：单模型调参
- `scripts/run_tuning_4gpu.sh`：四卡并行
- `scripts/run_tuning_4gpu_mixed_env.sh`：混合环境四卡并行
- `configs/test_cases_2examples.json`：两例测试
- `docs/`：详细设计与下载文档

## 3. 快速开始

### 3.1 下载
```bash
export HF_TOKEN="你的token"
/dev-vepfs/rc_wu/rc_wu/edit/EditSplat/flowedit_multimodel/scripts/hf_fast_download.sh
```

### 3.2 单模型调参（quick）
```bash
CUDA_VISIBLE_DEVICES=0 \
/dev-vepfs/rc_wu/rc_wu/envs/editsplat_ttt3r_unified_copy/bin/python \
/dev-vepfs/rc_wu/rc_wu/edit/EditSplat/flowedit_multimodel/scripts/tune_flowedit_model.py \
  --model_key flux2-dev \
  --cases_json /dev-vepfs/rc_wu/rc_wu/edit/EditSplat/flowedit_multimodel/configs/test_cases_2examples.json \
  --output_dir /dev-vepfs/rc_wu/rc_wu/edit/EditSplat/output/flowedit_multimodel_exp/smoke_flux2 \
  --gpu_id 0 \
  --quick \
  --no_clip
```

### 3.3 四卡并行
```bash
export HF_TOKEN="你的token"
/dev-vepfs/rc_wu/rc_wu/edit/EditSplat/flowedit_multimodel/scripts/run_tuning_4gpu.sh
```

若包含 `flux2-klein-4b`，推荐使用混合环境脚本（避免 Python3.9 + diffusers0.36 的兼容问题）：

```bash
export HF_TOKEN="你的token"
/dev-vepfs/rc_wu/rc_wu/edit/EditSplat/flowedit_multimodel/scripts/run_tuning_4gpu_mixed_env.sh
```

可调环境变量：
- `TUNE_MODE=quick|full`（默认 `quick`）
- `NO_CLIP=1|0`（默认 `1`）
- `MODELS="flux2-dev sd35-large qwen-image-edit"`

## 4. 输出位置
- 所有实验输出建议放：
  - `/dev-vepfs/rc_wu/rc_wu/edit/EditSplat/output/flowedit_multimodel_exp/`

## 5. 常见问题
- `GatedRepoError 403`：先去对应 HF 页面同意 license。
- `sd35-large` 无权限：改用 `sd35-medium-turbo-open` 继续实验。
- `flux2-klein-4b` 输出异常单一：优先使用 `run_tuning_4gpu_mixed_env.sh`（Python3.10 + diffusers main）。
- `Flux2KleinPipeline not found`：本扩展已内置本地类，使用 `flowedit_multimodel/src/flux2_klein_pipeline_local.py`。
- 下载反复中断：参考 `docs/hf_download_acceleration_zh.md`，优先 `HF_HUB_DISABLE_XET=1` + 低并发；若 `hf_transfer` 不稳定，设置 `HF_HUB_ENABLE_HF_TRANSFER=0`。
