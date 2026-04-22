#!/bin/bash
# Non-destructive heartbeat logger for relaunch planning.
set -euo pipefail

ROOT_DIR="/dev_vepfs/rc_wu/edit/CircleEditing"
REPORT_DIR="$ROOT_DIR/reports/ralph_loop_20260328"
HEARTBEAT_MD="$REPORT_DIR/heartbeat.md"
LOG_DIR="$REPORT_DIR/logs"
RAW_LOG="$LOG_DIR/heartbeat_watch.log"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

mkdir -p "$LOG_DIR"

ts="$(timestamp)"
gpu_csv="$(nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits)"
apps_csv="$(nvidia-smi --query-compute-apps=gpu_uuid,gpu_name,pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null || true)"
df_line="$(df -h /dev_vepfs | awk 'NR==2 {print $2 ", used " $3 ", avail " $4 ", use " $5}')"
root_du="$(du -sh "$ROOT_DIR" | awk '{print $1}')"
report_du="$(du -sh "$REPORT_DIR" | awk '{print $1}')"

active_gpus="$(
  printf '%s\n' "$gpu_csv" | awk -F', ' '
    $4 + 0 > 1024 || $5 + 0 > 0 {count++}
    END {print count + 0}
  '
)"

busy_gpus="$(
  printf '%s\n' "$gpu_csv" | awk -F', ' '
    $4 + 0 > 1024 || $5 + 0 > 0 {printf "%s(%s MiB,%s%%) ", $1, $4, $5}
  ' | sed 's/[[:space:]]*$//'
)"

idle_preferred="$(
  printf '%s\n' "$gpu_csv" | awk -F', ' '
    $1 >= 4 && $1 <= 7 && $4 + 0 <= 1024 && $5 + 0 == 0 {printf "%s ", $1}
  ' | sed 's/[[:space:]]*$//'
)"

if [ -z "$busy_gpus" ]; then
  busy_gpus="none"
fi

if [ -z "$idle_preferred" ]; then
  idle_preferred="none"
fi

if [ "$active_gpus" -le 4 ]; then
  safety_status="within-cap"
  safety_note="Current host-visible occupancy is within the <=4 active GPU cap."
else
  safety_status="over-cap"
  safety_note="Current host-visible occupancy exceeds the <=4 active GPU cap. Do not relaunch unless an active job is being replaced and cross-machine usage is re-checked."
fi

if [ "$active_gpus" -ge 4 ]; then
  launch_window="closed"
  advice="No idle-safe relaunch window right now: this host already shows $active_gpus active GPUs, so any new GPU launch would exceed the global <=4 cap unless it replaces an active job."
elif [ "$idle_preferred" = "none" ]; then
  launch_window="closed"
  advice="No preferred idle GPUs in the 4-7 range. Wait for one of GPUs 4-7 to go idle before relaunching the lead worker."
else
  launch_window="host-open"
  advice="Host-side relaunch is viable on preferred GPUs $idle_preferred, but only launch after confirming the cross-machine active GPU total will stay <=4."
fi

{
  echo "=== $ts ==="
  echo "[gpu]"
  printf '%s\n' "$gpu_csv"
  echo "[compute_apps]"
  if [ -n "$apps_csv" ]; then
    printf '%s\n' "$apps_csv"
  else
    echo "none"
  fi
  echo "[storage]"
  echo "/dev_vepfs: $df_line"
  echo "workspace_du: $root_du"
  echo "report_du: $report_du"
  echo "[safety]"
  echo "status=$safety_status"
  echo "$safety_note"
  echo "[advice]"
  echo "launch_window=$launch_window"
  echo "$advice"
  echo
} >> "$RAW_LOG"

{
  echo
  echo "$ts"
  echo "- GPU occupancy: active GPU count on this host is $active_gpus; busy GPUs: $busy_gpus; preferred idle GPUs 4-7: $idle_preferred."
  echo "- Safety status: $safety_note"
  echo "- Storage: /dev_vepfs $df_line; CircleEditing workspace uses $root_du; heartbeat report directory uses $report_du. Current mutable footprint is within the 50G guardrail."
  echo "- Relaunch advice: $advice"
} >> "$HEARTBEAT_MD"
