#!/usr/bin/env bash
set -euo pipefail

ROOT="/dev_vepfs/rc_wu/edit/CircleEditing"
RUNS_DIR="$ROOT/runs"
REPORT_DIR="$ROOT/reports/ralph_loop_20260328"
LOG_DIR="$REPORT_DIR/logs"
PYTHON_BIN="/dev_vepfs/rc_wu/envs/editsplat_multimodel_v2/bin/python"
WRAPPER="$ROOT/runtime/EditSplat_overlay_20260326/sandboxes/20260322_editsplat_ttt3r_flowedit_sam3_4567/scripts/run_sd35_ttt3r_sam3_wrapper.py"
DNA_OVERRIDE_IMAGE="$ROOT/runs/dnafix_offline_strong211_staticproxy_20260327_171928/debug_intermediates/mfg_edit/view000/mfg_output.png"
DEADLINE_EPOCH="$(date -u -d '2026-03-28 06:23:45Z' +%s)"

mkdir -p "$LOG_DIR"
QUEUE_LOG="$LOG_DIR/lead_queue_$(date -u +%Y%m%d_%H%M%S).log"

log() {
  local msg="$*"
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%d %H:%M:%SZ')" "$msg" | tee -a "$QUEUE_LOG" >&2
}

active_gpu_count() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F', *' 'BEGIN{c=0} {if (($2+0) >= 1024 || ($3+0) >= 5) c++} END{print c+0}'
}

preferred_idle_gpu() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F', *' '($1+0) >= 4 && ($1+0) <= 7 && ($2+0) < 1024 && ($3+0) < 5 {print $1; exit}'
}

preferred_idle_gpu_list() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F', *' '($1+0) >= 4 && ($1+0) <= 7 && ($2+0) < 1024 && ($3+0) < 5 {printf "%s ", $1}'
}

wait_for_safe_slot() {
  while true; do
    local now_epoch
    now_epoch="$(date -u +%s)"
    if [[ "$now_epoch" -ge "$DEADLINE_EPOCH" ]]; then
      log "deadline reached before a safe GPU slot opened"
      return 1
    fi

    local active_count
    active_count="$(active_gpu_count)"
    local idle_gpu
    idle_gpu="$(preferred_idle_gpu || true)"
    if [[ -n "${idle_gpu:-}" && "$active_count" -lt 4 ]]; then
      log "safe slot available: active_gpu_count=$active_count selected_gpu=$idle_gpu"
      printf '%s\n' "$idle_gpu"
      return 0
    fi

    local idle_list
    idle_list="$(preferred_idle_gpu_list || true)"
    log "waiting for safe slot: active_gpu_count=$active_count preferred_idle='${idle_list:-none}'"
    sleep 60
  done
}

run_grid3_job() {
  local run_name="$1"
  local epoch="$2"
  local gpu_index="$3"
  shift 3
  local -a extra_env=( "$@" )

  local run_dir="$RUNS_DIR/$run_name"
  local launcher_log="$run_dir/launcher.log"
  mkdir -p "$run_dir"

  local -a cmd=(
    "$PYTHON_BIN" "$WRAPPER"
    --model_key sd35-medium-turbo-open
    --hf_home /dev_vepfs/rc_wu/cache/hf_home_dev02
    --adapter_resize_side 512
    --adapter_gpu 0
    --base_gpu 0
    --head_k 6
    --depth_mode constant
    --skip_agt
    --aux_models_cpu
    --ttt3r_repo_root /dev_vepfs/rc_wu/edit/TTT3R
    --ttt3r_checkpoint /dev_vepfs/rc_wu/edit/TTT3R/src/cut3r_512_dpt_4_64.pth
    --ttt3r_support_views 2
    --ttt3r_support_stride 1
    --ttt3r_conf_power 1.0
    --ttt3r_conf_floor 0.0
    --ttt3r_geo_scale 1.0
    --ttt3r_edit_min_mass 0.0
    --ttt3r_preserve_min_mass 0.0
    --ttt3r_adaptive_max_scale 3.2
    --ttt3r_schedule_power 1.8
    --ttt3r_input_h 256
    --ttt3r_input_w 320
    --ttt3r_edit_mask_quantile 0.9
    --ttt3r_gpu -1
    --dump_intermediates
    --dump_max_per_stage 32
    --max_optimizer_steps 1
    --optimizer_lr_scale 0.6
    --fit_loss_mask_mode initial_edit
    --fit_loss_mask_quantile 0.75
    --fit_loss_mask_bg 0.05
    --fit_view_topk -1
    --source_path /dev_vepfs/rc_wu/edit/EditSplat/dataset/dataset/face
    --source_checkpoint /dev_vepfs/rc_wu/edit/EditSplat/sandboxes/20260322_editsplat_ttt3r_flowedit_sam3_4567/runtime/compat_pretrained_face/chkpnt7004.pth
    --eval
    --epoch "$epoch"
    --iterations 30000
    --debug
    --target_prompt "the same man in the same pose and camera framing, same background and clothes, with clear clown makeup: white face paint, a red clown nose, and colorful face paint"
    --sampling_prompt "the same man with clown makeup, same framing and identity"
    --object_prompt face
    --target_mask_prompt face
    --filtering_ratio 0.85
    --flow_src_prompt "a photo of a young man"
    --flow_tar_prompt "the same man in the same pose and camera framing, same background and clothes, with clear clown makeup: white face paint, a red clown nose, and colorful face paint"
    --flow_model_key sd35-medium-turbo-open
    --flow_hf_home /dev_vepfs/rc_wu/cache/hf_home_dev02
    --flow_adapter_resize_side 512
    --flow_adapter_gpu 0
    --flow_steps 24
    --flow_n_avg 1
    --flow_n_min 0
    --flow_n_max 10
    --flow_seed 211
    --enable
    --resize 512
    --timesteps 28
    --n_min 0
    --n_max 24
    --time_weight one
    --w_edit 1.1
    --w_id 0.15
    --src_guidance 1.4
    --tar_guidance 7.8
    --mask_bg 0.15
    --attn_thres 0.1
    --k_percent 0.15
    --text_guidance_scale 7.4
    --image_guidance_scale 1.6
    --source_guidance_scale 1.0
    --flow_src_guidance_scale 3.2
    --flow_tar_guidance_scale 7.8
    --ttt3r_prox_strength 0.06
    --ttt3r_preserve_strength 0.02
    --ttt3r_edit_boost 1.0
    --ttt3r_preserve_boost 1.0
    --model_path "$run_dir"
    --flow_method flowedit
    --ttt3r_mode velocity
  )

  printf 'CMD=' > "$launcher_log"
  printf '%q ' "${cmd[@]}" >> "$launcher_log"
  printf '\n' >> "$launcher_log"

  log "launching run_dir=$run_dir gpu=$gpu_index epoch=$epoch extra_env='${extra_env[*]:-none}'"

  local -a common_env=(
    CUDA_VISIBLE_DEVICES="$gpu_index"
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
    EDITSPLAT_MASK_BACKEND=sam3
    EDITSPLAT_MFG_MODE=frontier_seed1
    EDITSPLAT_MFG_BACKFILL=nearest
    EDITSPLAT_MFG_SOURCE_COUNT=5
    EDITSPLAT_MAX_TRAIN_VIEWS=3
    EDITSPLAT_MAX_GAUSSIANS=70000
    EDITSPLAT_SKIP_RENDER_SETS=1
    EDITSPLAT_SKIP_3DGS_BACKWARD_ON_ERROR=0
    EDITSPLAT_FRONTIER_BLACKFACE_FALLBACK=1
    EDITSPLAT_FRONTIER_BLACKFACE_MEAN_THR=0.08
    EDITSPLAT_FRONTIER_BLACKFACE_STD_THR=0.02
    EDITSPLAT_FRONTIER_BLACKFACE_MIN_COVERAGE=0.005
    EDITSPLAT_FRONTIER_FALLBACK_FEATHER=9
  )

  set +e
  env "${common_env[@]}" "${extra_env[@]}" "${cmd[@]}" >> "$launcher_log" 2>&1
  local rc=$?
  set -e

  if [[ "$rc" -ne 0 ]]; then
    log "run failed rc=$rc run_dir=$run_dir"
    tail -n 60 "$launcher_log" | sed 's/^/[launcher-tail] /' | tee -a "$QUEUE_LOG"
    return "$rc"
  fi

  log "run succeeded run_dir=$run_dir"
  log "expected review assets: $run_dir/debug_intermediates/mfg_edit/view000/mfg_output.png $run_dir/debug_intermediates/mfg_edit/view001/mfg_output.png $run_dir/debug_intermediates/mfg_edit/view002/mfg_output.png"
}

launch_epoch1_stability() {
  local gpu_index="$1"
  local stamp
  stamp="$(date -u +%Y%m%d_%H%M%S)"
  run_grid3_job "frontier_seed1_grid3_editstrong_211_epoch1_patchstable_${stamp}" 1 "$gpu_index"
}

launch_dna_face_on_source() {
  local gpu_index="$1"
  local stamp
  stamp="$(date -u +%Y%m%d_%H%M%S)"
  run_grid3_job \
    "frontier_seed1_grid3_editstrong_211_dnaface_source_${stamp}" \
    0 \
    "$gpu_index" \
    "EDITSPLAT_FRONTIER_ANCHOR_OVERRIDE_IMAGE=$DNA_OVERRIDE_IMAGE" \
    "EDITSPLAT_FRONTIER_ANCHOR_OVERRIDE_MODE=face_on_source" \
    "EDITSPLAT_FRONTIER_ANCHOR_OVERRIDE_FEATHER=9"
}

launch_dna_face_on_existing() {
  local gpu_index="$1"
  local stamp
  stamp="$(date -u +%Y%m%d_%H%M%S)"
  run_grid3_job \
    "frontier_seed1_grid3_editstrong_211_dnaface_existing_${stamp}" \
    0 \
    "$gpu_index" \
    "EDITSPLAT_FRONTIER_ANCHOR_OVERRIDE_IMAGE=$DNA_OVERRIDE_IMAGE" \
    "EDITSPLAT_FRONTIER_ANCHOR_OVERRIDE_MODE=face_on_existing" \
    "EDITSPLAT_FRONTIER_ANCHOR_OVERRIDE_FEATHER=9"
}

main() {
  log "lead queue started deadline_epoch=$DEADLINE_EPOCH"

  local gpu_index
  gpu_index="$(wait_for_safe_slot)" || exit 0
  launch_epoch1_stability "$gpu_index" || true

  gpu_index="$(wait_for_safe_slot)" || exit 0
  launch_dna_face_on_source "$gpu_index" || true

  gpu_index="$(wait_for_safe_slot)" || exit 0
  launch_dna_face_on_existing "$gpu_index" || true

  log "lead queue finished"
}

main "$@"
