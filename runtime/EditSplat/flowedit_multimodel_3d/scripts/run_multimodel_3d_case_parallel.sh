#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/dev-vepfs/rc_wu/rc_wu/edit/EditSplat"
PY="/dev-vepfs/rc_wu/rc_wu/envs/editsplat_multimodel_v2/bin/python"
WRAPPER="$ROOT_DIR/flowedit_multimodel_3d/scripts/run_editing_flow_multimodel_wrapper.py"
HF_HOME_DEFAULT="/dev-vepfs/rc_wu/rc_wu/cache/hf_home"

CASE_NAME="fangzhou"
OUT_ROOT=""
HEAD_K="0"
RESOLUTION="-1"
EPOCH="1"
HF_HOME="$HF_HOME_DEFAULT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --case) CASE_NAME="$2"; shift 2 ;;
    --out_root) OUT_ROOT="$2"; shift 2 ;;
    --head_k) HEAD_K="$2"; shift 2 ;;
    --resolution) RESOLUTION="$2"; shift 2 ;;
    --epoch) EPOCH="$2"; shift 2 ;;
    --hf_home) HF_HOME="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$OUT_ROOT" ]]; then
  echo "--out_root is required" >&2
  exit 2
fi

DATASET_DIR="$ROOT_DIR/dataset/dataset/$CASE_NAME"
CKPT="$ROOT_DIR/dataset/pretrained/$CASE_NAME/chkpnt7000.pth"
if [[ ! -d "$DATASET_DIR" ]]; then
  echo "Missing dataset dir: $DATASET_DIR" >&2
  exit 2
fi
if [[ ! -f "$CKPT" ]]; then
  echo "Missing checkpoint: $CKPT" >&2
  exit 2
fi

case "$CASE_NAME" in
  fangzhou)
    TARGET_PROMPT="Make his face resemble a marble sculpture"
    SAMPLING_PROMPT="a photo of a marble sculpture face"
    OBJECT_PROMPT="face"
    TARGET_MASK_PROMPT="face"
    FLOW_SRC_PROMPT="a photo of a young man"
    FLOW_TAR_PROMPT="a photo of a young man made of white marble"
    ;;
  dinosaur)
    TARGET_PROMPT="Turn the dinosaur statue into aged bronze with verdigris patina"
    SAMPLING_PROMPT="a photo of an aged bronze dinosaur statue with verdigris patina"
    OBJECT_PROMPT="dinosaur"
    TARGET_MASK_PROMPT="dinosaur"
    FLOW_SRC_PROMPT="a photo of a dinosaur statue in a park"
    FLOW_TAR_PROMPT="a photo of an aged bronze dinosaur statue with verdigris patina in a park"
    ;;
  *)
    echo "Unsupported case: $CASE_NAME" >&2
    exit 2
    ;;
esac

declare -a MODELS=("flux2-dev" "sd35-large" "qwen-image-edit" "z-image")
declare -A STEPS SRCG TARG NMAX SEED
STEPS["flux2-dev"]="32";      SRCG["flux2-dev"]="1.6"; TARG["flux2-dev"]="5.6"; NMAX["flux2-dev"]="26"; SEED["flux2-dev"]="1"
STEPS["sd35-large"]="24";     SRCG["sd35-large"]="2.5"; TARG["sd35-large"]="9.0"; NMAX["sd35-large"]="14"; SEED["sd35-large"]="2"
STEPS["qwen-image-edit"]="22";SRCG["qwen-image-edit"]="1.4"; TARG["qwen-image-edit"]="4.7"; NMAX["qwen-image-edit"]="17"; SEED["qwen-image-edit"]="2"
STEPS["z-image"]="22";        SRCG["z-image"]="1.3"; TARG["z-image"]="4.2"; NMAX["z-image"]="14"; SEED["z-image"]="2"

mkdir -p "$OUT_ROOT/logs"

export HF_HUB_OFFLINE=1
export FLOWEDIT_REAL_LPIPS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

declare -a PIDS=()
declare -A PID2MODEL
declare -A PID2START
declare -A PID2MON
declare -A PID2GPU
declare -A PID2TIMER

for i in "${!MODELS[@]}"; do
  model="${MODELS[$i]}"
  gpu="$i"
  out_dir="$OUT_ROOT/${model}_${CASE_NAME}"
  log_file="$OUT_ROOT/logs/${model}_${CASE_NAME}.log"
  mkdir -p "$out_dir"
  echo "[launch] model=$model gpu=$gpu out=$out_dir"
  CUDA_VISIBLE_DEVICES="$gpu" \
    "$PY" "$WRAPPER" \
    --model_key "$model" \
    --hf_home "$HF_HOME" \
    --adapter_gpu 0 \
    --base_gpu 0 \
    --head_k "$HEAD_K" \
    --source_path "$DATASET_DIR" \
    --model_path "$out_dir" \
    --source_checkpoint "$CKPT" \
    --eval \
    --resolution "$RESOLUTION" \
    --epoch "$EPOCH" \
    --target_prompt "$TARGET_PROMPT" \
    --sampling_prompt "$SAMPLING_PROMPT" \
    --object_prompt "$OBJECT_PROMPT" \
    --target_mask_prompt "$TARGET_MASK_PROMPT" \
    --flow_src_prompt "$FLOW_SRC_PROMPT" \
    --flow_tar_prompt "$FLOW_TAR_PROMPT" \
    --flow_steps "${STEPS[$model]}" \
    --flow_n_avg 1 \
    --flow_src_guidance_scale "${SRCG[$model]}" \
    --flow_tar_guidance_scale "${TARG[$model]}" \
    --flow_n_min 0 \
    --flow_n_max "${NMAX[$model]}" \
    --flow_seed "${SEED[$model]}" \
    >"$log_file" 2>&1 &
  pid=$!
  PIDS+=("$pid")
  PID2MODEL["$pid"]="$model"
  PID2START["$pid"]="$(date +%s)"
  PID2GPU["$pid"]="$gpu"

  # Per-process wall clock timer.
  (
    start_ts="${PID2START[$pid]}"
    while kill -0 "$pid" 2>/dev/null; do
      sleep 1
    done
    end_ts="$(date +%s)"
    echo "$(( end_ts - start_ts ))" > "$OUT_ROOT/logs/${model}_${CASE_NAME}.runtime_sec.txt"
  ) &
  PID2TIMER["$pid"]="$!"

  # Monitor peak GPU memory on the dedicated GPU index for this model.
  gpu_idx="$gpu"
  (
    peak=0
    while kill -0 "$pid" 2>/dev/null; do
      mem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sed -n "$(( gpu_idx + 1 ))p" | tr -d ' ')"
      mem="${mem:-0}"
      if [[ "$mem" =~ ^[0-9]+$ ]] && (( mem > peak )); then
        peak="$mem"
      fi
      sleep 2
    done
    echo "$peak" > "$OUT_ROOT/logs/${model}_${CASE_NAME}.peak_mem_mib.txt"
  ) &
  PID2MON["$pid"]="$!"
done

fail=0
summary_tsv="$OUT_ROOT/logs/run_summary.tsv"
echo -e "model\tstatus\truntime_sec\tpeak_mem_mib\tsteps\tsrc_g\ttar_g\tn_max\tseed" > "$summary_tsv"
for pid in "${PIDS[@]}"; do
  model="${PID2MODEL[$pid]}"
  if wait "$pid"; then
    echo "[done] $model"
    status="done"
  else
    echo "[fail] $model (pid=$pid), see logs/${model}_${CASE_NAME}.log" >&2
    fail=1
    status="fail"
  fi
  timer_pid="${PID2TIMER[$pid]:-}"
  if [[ -n "$timer_pid" ]]; then
    wait "$timer_pid" 2>/dev/null || true
  fi
  runtime_file="$OUT_ROOT/logs/${model}_${CASE_NAME}.runtime_sec.txt"
  if [[ -f "$runtime_file" ]]; then
    runtime="$(cat "$runtime_file")"
  else
    start_ts="${PID2START[$pid]}"
    end_ts="$(date +%s)"
    runtime=$(( end_ts - start_ts ))
  fi
  mon_pid="${PID2MON[$pid]}"
  if [[ -n "${mon_pid:-}" ]]; then
    wait "$mon_pid" 2>/dev/null || true
  fi
  peak_file="$OUT_ROOT/logs/${model}_${CASE_NAME}.peak_mem_mib.txt"
  if [[ -f "$peak_file" ]]; then
    peak_mem="$(cat "$peak_file")"
  else
    peak_mem="0"
  fi
  echo -e "${model}\t${status}\t${runtime}\t${peak_mem}\t${STEPS[$model]}\t${SRCG[$model]}\t${TARG[$model]}\t${NMAX[$model]}\t${SEED[$model]}" >> "$summary_tsv"
done

exit "$fail"
