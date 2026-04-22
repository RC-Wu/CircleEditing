#!/usr/bin/env bash
set -euo pipefail

ROOT="/dev-vepfs/rc_wu/rc_wu/edit/EditSplat"
PY="/dev-vepfs/rc_wu/rc_wu/envs/editsplat_test/bin/python"
TAG="face_redfix_r2_20260303"

export HF_HOME=/dev-vepfs/rc_wu/rc_wu/cache/hf_home
export HF_HUB_CACHE=/dev-vepfs/rc_wu/rc_wu/cache/hf_home/hub
export TORCH_HOME=/dev-vepfs/rc_wu/rc_wu/cache/torch_hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1

cd "$ROOT"

"$PY" eval/scripts/run_full_eval.py \
  --benchmark eval/benchmark/benchmark_face_redfix_r2_20260303.json \
  --cache_root "eval/cache/renders_${TAG}" \
  --metrics_root "eval/cache/metrics_${TAG}" \
  --summaries_root "eval/cache/summaries_${TAG}" \
  --pairs_per_sample 400 \
  --render_depth_source 1 \
  --render_depth_edit 0 \
  --compute_reproj 1 \
  --use_lpips 1 \
  --clip_backend transformers \
  --clip_model openai/clip-vit-base-patch32 \
  --open_clip_pretrained openai/clip-vit-base-patch32 \
  --overwrite 0 \
  --device cuda
