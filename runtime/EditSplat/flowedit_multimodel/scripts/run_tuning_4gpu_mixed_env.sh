#!/usr/bin/env bash
set -euo pipefail

ROOT="/dev-vepfs/rc_wu/rc_wu/edit/EditSplat/flowedit_multimodel"
PY_DEFAULT="${PY_DEFAULT:-/dev-vepfs/rc_wu/rc_wu/envs/editsplat_ttt3r_unified_copy/bin/python}"
PY_FLUX2="${PY_FLUX2:-/dev-vepfs/rc_wu/rc_wu/envs/flowedit/bin/python}"
CASES_JSON="${1:-$ROOT/configs/test_cases_2examples.json}"
OUT_ROOT="${2:-/dev-vepfs/rc_wu/rc_wu/edit/EditSplat/output/flowedit_multimodel_exp/tune_4gpu_mixed}"
MODELS="${MODELS:-flux2-dev sd35-large qwen-image-edit}"
TUNE_MODE="${TUNE_MODE:-full}"   # quick | full
NO_CLIP="${NO_CLIP:-1}"          # 1 -> add --no_clip

mkdir -p "$OUT_ROOT/logs"

declare -a pids=()
declare -a names=()

choose_python() {
  local model_key="$1"
  case "$model_key" in
    flux2-klein* )
      echo "$PY_FLUX2"
      ;;
    * )
      echo "$PY_DEFAULT"
      ;;
  esac
}

launch_job() {
  local model_key="$1"
  local gpu_id="$2"
  local run_dir="$OUT_ROOT/$model_key"
  local log_file="$OUT_ROOT/logs/${model_key}.log"
  local py_bin
  py_bin="$(choose_python "$model_key")"

  echo "[LAUNCH] model=$model_key gpu=$gpu_id py=$py_bin -> $log_file"
  local mode_flag=""
  local clip_flag=""
  if [[ "$TUNE_MODE" == "quick" ]]; then
    mode_flag="--quick"
  fi
  if [[ "$NO_CLIP" == "1" ]]; then
    clip_flag="--no_clip"
  fi

  CUDA_VISIBLE_DEVICES="$gpu_id" \
  HF_TOKEN="${HF_TOKEN:-}" \
  "$py_bin" "$ROOT/scripts/tune_flowedit_model.py" \
    --model_key "$model_key" \
    --cases_json "$CASES_JSON" \
    --output_dir "$run_dir" \
    --gpu_id 0 \
    --hf_token "${HF_TOKEN:-}" \
    $mode_flag \
    $clip_flag \
    >"$log_file" 2>&1 &

  pids+=("$!")
  names+=("$model_key")
}

gpu_id=0
for model_key in $MODELS; do
  launch_job "$model_key" "$gpu_id"
  gpu_id=$(( (gpu_id + 1) % 4 ))
done

fails=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  name="${names[$i]}"
  if wait "$pid"; then
    echo "[DONE] $name"
  else
    echo "[FAIL] $name (check $OUT_ROOT/logs/${name}.log)"
    fails=$((fails + 1))
  fi
done

if [[ "$fails" -gt 0 ]]; then
  echo "[SUMMARY] $fails jobs failed."
  exit 1
fi

echo "[SUMMARY] All jobs finished successfully."
