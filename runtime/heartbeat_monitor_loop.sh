#!/usr/bin/env bash
set -euo pipefail

ROOT=/dev_vepfs/rc_wu/edit/CircleEditing
NOTE="$ROOT/docs/2026-04-13-heartbeat-monitor.md"
DEV01=/dev_vepfs/rc_wu/edit/EditSplat/sandboxes/20260322_editsplat_ttt3r_flowedit_sam3_4567/runtime/queues/20260413_wave19_dev01_three_gpu
DEV02=/dev_vepfs/rc_wu/_codex_staging/ttt3r_overnight/20260413_wave18_gpu3_five_rounds_retry3
STOP_FILE="$ROOT/runtime/codex_logs/circleediting-heartbeat.stop"

mkdir -p "$ROOT/runtime/codex_logs"

while true; do
  if [ -f "$STOP_FILE" ]; then
    exit 0
  fi

  ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  manifest_jobs=$(grep -c '"name"' "$DEV01/manifest.json" 2>/dev/null || echo 0)
  dev01_status=$(find "$DEV01/status" -name '*.json' 2>/dev/null | wc -l)
  dev01_panels=$(find "$DEV01/results" -path '*/analysis/panel_final_grid.png' 2>/dev/null | wc -l)
  dev02_status=$(find "$DEV02/status" -name '*.json' 2>/dev/null | wc -l)
  dev02_panels=$(find "$DEV02/results" -path '*/analysis/panel_final_grid.png' 2>/dev/null | wc -l)
  dev01_slot0=$(tail -n 1 "$DEV01/slot_logs/gpu0.log" 2>/dev/null || echo "MISSING")
  dev01_slot1=$(tail -n 1 "$DEV01/slot_logs/gpu1.log" 2>/dev/null || echo "MISSING")
  local_pids=$(ps -eo pid,etimes,cmd | grep -E '20260413_wave19_dev01_three_gpu|20260413_wave18_gpu3_five_rounds_retry3|run_sd35_ttt3r_sam3_wrapper.py|slot_gpu[0-9]\.sh|build_fixed_gpu_overnight_queue.py run-one' | grep -v grep || true)

  if [ -z "$local_pids" ]; then
    local_pid_line="no matching local process"
  else
    local_pid_line="$local_pids"
  fi

  dev02_runner=$(python3 - <<'PY'
import json
from pathlib import Path

p = Path("/dev_vepfs/rc_wu/_codex_staging/ttt3r_overnight/20260413_wave18_gpu3_five_rounds_retry3/runner_state.json")
if not p.exists():
    print("runner_state missing")
else:
    obj = json.loads(p.read_text(encoding="utf-8"))
    history = obj.get("history", [])
    print(f"history={len(history)} updated_at={obj.get('updated_at', 'missing')}")
PY
)

  ssh01=$(ssh -o BatchMode=yes -o ConnectTimeout=5 dev-intern-01 \
    "hostname; date -u; nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits; echo ===== PROCS; ps -eo pid,etimes,cmd | grep -E '20260413_wave19_dev01_three_gpu|run_sd35_ttt3r_sam3_wrapper.py' | grep -v grep" \
    2>&1 || true)
  ssh02=$(ssh -o BatchMode=yes -o ConnectTimeout=5 dev-intern-02 \
    "hostname; date -u; nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits; echo ===== PROCS; ps -eo pid,etimes,cmd | grep -E '20260413_wave18_gpu3_five_rounds_retry3|run_sd35_ttt3r_sam3_wrapper.py' | grep -v grep" \
    2>&1 || true)

  {
    echo
    echo "## $ts"
    echo
    echo "- UTC time: \`$ts\`"
    echo "- Local time: \`$ts\` (Etc/UTC)"
    echo "- Current status summary:"
    echo "  - dev01 manifest jobs: $manifest_jobs; status JSONs: $dev01_status; panel_final_grid count: $dev01_panels"
    echo "  - dev01 gpu0 slot tail: $dev01_slot0"
    echo "  - dev01 gpu1 slot tail: $dev01_slot1"
    echo "  - dev02 status JSONs: $dev02_status; panel_final_grid count: $dev02_panels; $dev02_runner"
    echo "- Exact PIDs or liveness:"
    echo "  - local process check: $local_pid_line"
    echo "  - ssh dev-intern-01 result: $ssh01"
    echo "  - ssh dev-intern-02 result: $ssh02"
    echo "- Key file checks:"
    echo "  - $DEV01/status -> $dev01_status json files"
    echo "  - $DEV01/results/*/analysis/panel_final_grid.png -> $dev01_panels files"
    echo "  - $DEV02/status -> $dev02_status json files"
    echo "  - $DEV02/results/*/analysis/panel_final_grid.png -> $dev02_panels files"
    echo "- Intervention taken: none; this loop is monitor-only and will not restart any slot without successful remote PID and GPU verification."
    echo "- Next expected milestone or blocker: unblock host resolution for dev-intern-01 and dev-intern-02, then verify remote wrappers and GPU occupancy before any dev01 GPU0 or GPU1 relaunch. dev01 GPU2 remains forbidden."
  } >> "$NOTE"

  sleep 1200
done
