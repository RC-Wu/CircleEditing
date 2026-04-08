#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/dev_vepfs/rc_wu/edit/EditSplat/sandboxes/20260322_editsplat_ttt3r_flowedit_sam3_4567}
PY=${PY:-/dev_vepfs/rc_wu/envs/editsplat_multimodel_v2/bin/python}
WATCHER=${WATCHER:-"$ROOT/scripts/watch_fixed_gpu_queue.py"}
HELPER=${HELPER:-"$ROOT/scripts/build_fixed_gpu_overnight_queue.py"}
LAUNCHER=${LAUNCHER:-"$ROOT/scripts/launch_dev01_ttt3r_consistency_wave.py"}
MANIFEST=${MANIFEST:-"$ROOT/scripts/manifests/20260408_wave18_bline_gpu123_queue.json"}
QUEUE_ROOT=${QUEUE_ROOT:-/dev_vepfs/rc_wu/_codex_staging/ttt3r_overnight}
WAVE_NAME=${WAVE_NAME:-20260408_wave18_bline_gpu123_queue}
LOG_MD=${LOG_MD:-"$QUEUE_ROOT/${WAVE_NAME}_launch_watch.md"}

cd "$ROOT"
"$PY" "$WATCHER" \
  --wave-name "$WAVE_NAME" \
  --manifest "$MANIFEST" \
  --gpu-slots 1 2 3 \
  --queue-root "$QUEUE_ROOT" \
  --helper "$HELPER" \
  --launcher-module "$LAUNCHER" \
  --wait-for-completion \
  --log-md "$LOG_MD"
