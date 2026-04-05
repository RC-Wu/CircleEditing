#!/usr/bin/env bash
set -euo pipefail

ROOT=/dev_vepfs/rc_wu/edit/EditSplat/sandboxes/20260322_editsplat_ttt3r_flowedit_sam3_4567
PY=/dev_vepfs/rc_wu/envs/editsplat_multimodel_v2/bin/python
HELPER="$ROOT/scripts/build_fixed_gpu_overnight_queue.py"
LAUNCHER="$ROOT/scripts/launch_dev01_ttt3r_consistency_wave.py"
MANIFEST="$ROOT/scripts/manifests/20260406_wave16_semantic_bold_queue.json"
QUEUE_ROOT=/dev_vepfs/rc_wu/_codex_staging/ttt3r_overnight
WAVE_NAME=20260406_wave16_semantic_controller_queue
WAVE_ROOT="$QUEUE_ROOT/$WAVE_NAME"
LOG_DIR="$WAVE_ROOT/controller_launcher_logs"

if [[ -e "$WAVE_ROOT" ]]; then
  echo "[wave16-controller] wave root already exists: $WAVE_ROOT" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"
echo "[wave16-controller] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[wave16-controller] root=$ROOT"
echo "[wave16-controller] wave_root=$WAVE_ROOT"

cd "$ROOT"
"$PY" "$HELPER" emit \
  --manifest "$MANIFEST" \
  --queue-root "$QUEUE_ROOT" \
  --wave-name "$WAVE_NAME" \
  --launcher-module "$LAUNCHER" \
  --gpu-slots 4 5 6 7

declare -a pids=()
for gpu in 4 5 6 7; do
  script_path="$WAVE_ROOT/scripts/slot_gpu${gpu}.sh"
  log_path="$LOG_DIR/slot_gpu${gpu}.foreground.log"
  echo "[wave16-controller] launch gpu=$gpu script=$script_path log=$log_path"
  bash "$script_path" >"$log_path" 2>&1 &
  pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    rc=1
  fi
done

"$PY" - <<'PY'
from pathlib import Path
import sys

root = Path("/dev_vepfs/rc_wu/edit/EditSplat/sandboxes/20260322_editsplat_ttt3r_flowedit_sam3_4567")
sys.path.insert(0, str(root))
from scripts.watch_fixed_gpu_queue import build_review_panels

outputs = build_review_panels(
    queue_root=Path("/dev_vepfs/rc_wu/_codex_staging/ttt3r_overnight"),
    wave_name="20260406_wave16_semantic_controller_queue",
)
for path in outputs:
    print(path)
PY

echo "[wave16-controller] done $(date -u +%Y-%m-%dT%H:%M:%SZ) rc=$rc"
exit "$rc"
