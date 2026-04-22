# HuggingFace 权重加速下载（中国大陆可用）

本文档对应脚本：
- `flowedit_multimodel/scripts/hf_fast_download.sh`
- `flowedit_multimodel/scripts/hf_fast_download.py`
- `flowedit_multimodel/scripts/hf_sequential_retry_download.py`

## 1. 目标
在大模型权重下载阶段，尽量提升稳定性与速度，避免反复中断重下。方案包含：
- 自动端点切换：`huggingface.co` / `hf-mirror.com`
- `hf_transfer` 并行块下载
- `snapshot_download` 断点续传
- 可控并发线程数（`max_workers`）
- 单文件顺序重试（网络抖动时更稳）

## 1.1 2026-02-23 实测结论（本机）
在当前机器和网络环境下（中国大陆）：
- `HF_ENDPOINT=https://hf-mirror.com` + **关闭本地代理**（`unset HTTP_PROXY/HTTPS_PROXY/ALL_PROXY`）是最快且最稳定组合。
- 开启本地代理后，`hf-mirror` 与 `huggingface.co` 都会明显降速（从 MB/s 级降到 KB/s 级）。
- 直连 `huggingface.co`（关闭代理）经常不可达或超时。
- 当 `hf_hub_download` 出现“长时间低速”时，切换 `aria2` 分片下载可显著恢复速度。

建议默认环境：

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
```

## 2. 使用方式

先设置 token（建议只放环境变量，不写入命令历史）：

```bash
export HF_TOKEN="你的token"
```

下载默认模型（`FLUX.2-dev + SD3.5 large + SD3.5 large turbo`）：

```bash
/dev-vepfs/rc_wu/rc_wu/edit/EditSplat/flowedit_multimodel/scripts/hf_fast_download.sh
```

指定模型列表（逗号分隔）：

```bash
/dev-vepfs/rc_wu/rc_wu/edit/EditSplat/flowedit_multimodel/scripts/hf_fast_download.sh \
  "flux2-dev,sd35-large,sd35-large-turbo"
```

或直接调用 Python（可调并发）：

```bash
/dev-vepfs/rc_wu/rc_wu/envs/editsplat_ttt3r_unified_copy/bin/python \
  /dev-vepfs/rc_wu/rc_wu/edit/EditSplat/flowedit_multimodel/scripts/hf_fast_download.py \
  --models "flux2-dev,sd35-large" \
  --hf_home "/dev-vepfs/rc_wu/rc_wu/cache/hf_home" \
  --max_workers 24 \
  --prefer_mirror
```

网络不稳时，推荐“顺序重试下载”（更不容易整任务失败）：

```bash
HF_TOKEN="你的token" HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
/dev-vepfs/rc_wu/rc_wu/envs/editsplat_ttt3r_unified_copy/bin/python \
  /dev-vepfs/rc_wu/rc_wu/edit/EditSplat/flowedit_multimodel/scripts/hf_sequential_retry_download.py \
  --repo_id black-forest-labs/FLUX.2-klein-4B \
  --hf_home /dev-vepfs/rc_wu/rc_wu/cache/hf_home \
  --hf_token "$HF_TOKEN" \
  --max_retries 30
```

### 2.1 长时间低速时的 `aria2` 回退（推荐）
当单连接续传卡在低速（如 `<1 MB/s`）时，使用 `aria2` 分片并发续传：

```bash
# 先安装 aria2（建议关闭代理安装）
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  apt-get update -qq && \
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  apt-get install -y aria2
```

```bash
# 例：对单个大文件做分片断点续传
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
aria2c -c -x 16 -s 16 -k 1M --file-allocation=none \
  --header="Authorization: Bearer $HF_TOKEN" \
  --out="<目标文件名>" --dir "<目标目录>" \
  "https://hf-mirror.com/<repo>/resolve/main/<filename>"
```

实测（AuraFlow 大文件）中，`aria2` 可把低速续传恢复到约 `12~15 MiB/s`。

### 2.2 AuraFlow-v0.3 最小可用下载（fp16）
如果只需要 `variant=fp16` 推理，不建议全量拉取 fp32 文件。最小必需集合：
- `transformer/diffusion_pytorch_model.safetensors.fp16.index.json`
- `transformer/diffusion_pytorch_model-00001-of-00002.fp16.safetensors`
- `transformer/diffusion_pytorch_model-00002-of-00002.fp16.safetensors`
- `vae/config.json`
- `vae/diffusion_pytorch_model.fp16.safetensors`
- `text_encoder/model.fp16.safetensors`
- tokenizer/scheduler/model_index 对应配置文件

说明：
- `fal/AuraFlow-v0.3` 的 fp16 分片使用“旧命名格式”，已在 `flowedit_adapters.py` 里做兼容别名处理，可直接加载。
- 如需复现全量下载（fp16+fp32），再使用 `hf_sequential_retry_download.py --repo_id fal/AuraFlow-v0.3`。

## 3. 模型别名
- `flux2-dev` -> `black-forest-labs/FLUX.2-dev`
- `flux2-klein-4b` -> `black-forest-labs/FLUX.2-klein-4B`
- `sd35-large` -> `stabilityai/stable-diffusion-3.5-large`
- `sd35-large-turbo` -> `stabilityai/stable-diffusion-3.5-large-turbo`
- `sd3-medium` -> `stabilityai/stable-diffusion-3-medium`
- `auraflow-v03` -> `fal/AuraFlow-v0.3`

## 4. 关键环境变量
- `HF_TOKEN`: 访问受限仓库的访问令牌
- `HF_HOME`: 缓存根目录（默认 `/dev-vepfs/rc_wu/rc_wu/cache/hf_home`）
- `HF_HUB_CACHE`: 模型缓存目录
- `HF_HUB_ENABLE_HF_TRANSFER=1`: 启用 hf_transfer 加速
- `HF_HUB_ENABLE_HF_TRANSFER=0`: 在本机网络更稳（本次实测推荐）
- `HF_ENDPOINT`: 当前使用端点（脚本会自动设置）
- `HF_HUB_DISABLE_XET=1`: 禁用 xet（在部分网络环境更稳定）

## 5. 常见问题
- `403` / `gated repo`：先在模型页面同意 license，并确认 token 权限。
- `Connection reset`：脚本已自动切换端点并支持断点续传；可重试原命令。
- `hf_transfer` 报错：加上 `HF_HUB_ENABLE_HF_TRANSFER=0` 再试（更慢但更稳）。
- 单连接速度低：提高 `--max_workers`（例如 24~48），或配置可用代理后再运行。

## 6. 代理建议（可选）
若你有稳定代理，可在运行前设置：

```bash
export HTTPS_PROXY="http://127.0.0.1:7890"
export HTTP_PROXY="http://127.0.0.1:7890"
```

然后继续执行下载脚本即可。

但本机本次实测中，开启代理会显著降速；建议优先尝试“镜像+关闭代理”。
