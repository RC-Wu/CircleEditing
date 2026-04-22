#!/usr/bin/env bash
set -euo pipefail

ROOT="/dev-vepfs/rc_wu/rc_wu/edit/EditSplat"
PY="/dev-vepfs/rc_wu/rc_wu/envs/editsplat_test/bin/python"
LOG_DIR="$ROOT/exp_flowedit_3dnoise/logs"
TS="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR"

launch_detached() {
  local gpu="$1"
  local log="$2"
  shift 2
  # In this tool runtime, plain nohup/background jobs may be reaped when parent exits.
  # setsid -f detaches the worker into a new session so long jobs survive.
  setsid -f env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    HOME=/dev-vepfs/rc_wu/rc_wu/cache/runtime_home \
    HF_HOME=/dev-vepfs/rc_wu/rc_wu/cache/hf_home \
    HF_HUB_CACHE=/dev-vepfs/rc_wu/rc_wu/cache/hf_home/hub \
    TORCH_HOME=/dev-vepfs/rc_wu/rc_wu/cache/torch_hub \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_XET=1 \
    HF_TOKEN=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 \
    PYTHONUNBUFFERED=1 \
    "$PY" "$@" >"$log" 2>&1
}

SRC="/dev-vepfs/rc_wu/rc_wu/edit/EditSplat/dataset/dataset/face"
CKPT="/dev-vepfs/rc_wu/rc_wu/edit/EditSplat/dataset/pretrained/face/chkpnt7000.pth"
OUT_BASE="/dev-vepfs/rc_wu/rc_wu/edit/EditSplat/output/flowedit_3dnoise_exp"

LOG1="$LOG_DIR/full_redfix_face_baseline_ref_${TS}.log"
LOG2="$LOG_DIR/full_redfix_face_baseline_neutral_${TS}.log"
LOG3="$LOG_DIR/full_redfix_face_cov_init_nomfg_${TS}.log"
LOG4="$LOG_DIR/full_redfix_face_cov_balanced_nomfg_${TS}.log"

launch_detached 0 "$LOG1" "$ROOT/exp_flowedit_3dnoise/scripts/run_editing_flow_baseline_wrapper.py" \
  -s "$SRC" \
  -m "$OUT_BASE/full_real_face_baseline_ref_r2" \
  --source_checkpoint "$CKPT" \
  --object_prompt "face" \
  --target_prompt "Make his face resemble a marble sculpture" \
  --sampling_prompt "a photo of a marble sculpture face" \
  --target_mask_prompt "face" \
  --flow_src_prompt "a photo of a young man" \
  --flow_tar_prompt "a photo of a young man made of white marble" \
  --flow_steps 28 --flow_n_avg 1 --flow_src_guidance_scale 1.5 --flow_tar_guidance_scale 5.5 \
  --flow_n_min 0 --flow_n_max 24 --flow_seed 10 --filtering_ratio 0.85 --epoch 1

launch_detached 1 "$LOG2" "$ROOT/exp_flowedit_3dnoise/scripts/run_editing_flow_baseline_wrapper.py" \
  -s "$SRC" \
  -m "$OUT_BASE/full_real_face_baseline_neutral_r2" \
  --source_checkpoint "$CKPT" \
  --object_prompt "face" \
  --target_prompt "Make his face look like neutral white marble without red tint" \
  --sampling_prompt "a portrait photo of a neutral gray-white marble statue face" \
  --target_mask_prompt "face" \
  --flow_src_prompt "a photo of a young man" \
  --flow_tar_prompt "a portrait photo of a neutral gray-white marble statue face, no red skin tint, no warm blush" \
  --flow_negative_prompt "red skin tone, warm blush, orange cast, oversaturated skin" \
  --flow_steps 28 --flow_n_avg 1 --flow_src_guidance_scale 1.5 --flow_tar_guidance_scale 3.8 \
  --flow_n_min 0 --flow_n_max 20 --flow_seed 10 --filtering_ratio 0.85 --epoch 1

launch_detached 2 "$LOG3" "$ROOT/exp_flowedit_3dnoise/scripts/run_editing_flow_3dnoise_wrapper.py" \
  -s "$SRC" \
  -m "$OUT_BASE/full_real_face_cov_init_nomfg_r2" \
  --source_checkpoint "$CKPT" \
  --object_prompt "face" \
  --target_prompt "Make his face look like neutral white marble without red tint" \
  --sampling_prompt "a portrait photo of a neutral gray-white marble statue face" \
  --target_mask_prompt "face" \
  --flow_src_prompt "a photo of a young man" \
  --flow_tar_prompt "a portrait photo of a neutral gray-white marble statue face, no red skin tint, no warm blush" \
  --flow_negative_prompt "red skin tone, warm blush, orange cast, oversaturated skin" \
  --flow_steps 28 --flow_n_avg 1 --flow_src_guidance_scale 1.5 --flow_tar_guidance_scale 3.8 \
  --flow_n_min 0 --flow_n_max 20 --flow_seed 10 --filtering_ratio 0.85 --epoch 1 \
  --noise_tag cov_init_nomfg_redfix --noise_anchor_mode coverage_opacity --noise_init_mode coarse_smooth \
  --noise_max_anchors 40000 --noise_voxel_res 96 --noise_coarse_res 32 --noise_mix 0.35 \
  --noise_no_mfg --noise_opt_iters 0

launch_detached 3 "$LOG4" "$ROOT/exp_flowedit_3dnoise/scripts/run_editing_flow_3dnoise_wrapper.py" \
  -s "$SRC" \
  -m "$OUT_BASE/full_real_face_cov_balanced_nomfg_r2" \
  --source_checkpoint "$CKPT" \
  --object_prompt "face" \
  --target_prompt "Make his face look like neutral white marble without red tint" \
  --sampling_prompt "a portrait photo of a neutral gray-white marble statue face" \
  --target_mask_prompt "face" \
  --flow_src_prompt "a photo of a young man" \
  --flow_tar_prompt "a portrait photo of a neutral gray-white marble statue face, no red skin tint, no warm blush" \
  --flow_negative_prompt "red skin tone, warm blush, orange cast, oversaturated skin" \
  --flow_steps 28 --flow_n_avg 1 --flow_src_guidance_scale 1.5 --flow_tar_guidance_scale 3.8 \
  --flow_n_min 0 --flow_n_max 20 --flow_seed 10 --filtering_ratio 0.85 --epoch 1 \
  --noise_tag cov_balanced_nomfg_redfix --noise_anchor_mode coverage_opacity --noise_init_mode coarse_smooth \
  --noise_max_anchors 40000 --noise_voxel_res 96 --noise_coarse_res 32 --noise_mix 0.32 \
  --noise_no_mfg --noise_opt_iters 12 --noise_opt_lr 0.015 --noise_opt_mode balanced \
  --noise_opt_num_views 8 --noise_opt_views_per_iter 2 \
  --noise_lambda_edit 0.55 --noise_lambda_id 1.20 --noise_lambda_smooth 0.35 --noise_lambda_prior 0.12 \
  --noise_lambda_view_var 0.30 --noise_max_grad_norm 1.0

echo "LOG1=$LOG1"
echo "LOG2=$LOG2"
echo "LOG3=$LOG3"
echo "LOG4=$LOG4"
