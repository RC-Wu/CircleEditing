#!/usr/bin/env bash
set -euo pipefail

ROOT="/dev-vepfs/rc_wu/rc_wu/edit/EditSplat"
PY="/dev-vepfs/rc_wu/rc_wu/envs/editsplat_test/bin/python"
LOG_DIR="$ROOT/exp_flowedit_3dnoise/logs"
TS="$(date +%Y%m%d_%H%M%S)"
WAIT_LOG="$LOG_DIR/watch_face_redfix_r2_${TS}.log"
EVAL_LOG="$LOG_DIR/eval_face_redfix_r2_${TS}.log"

mkdir -p "$LOG_DIR"

echo "[watch] start $(date -Iseconds)" | tee -a "$WAIT_LOG"
while pgrep -f "run_editing_flow_(baseline|3dnoise)_wrapper.py .*full_real_face_" >/dev/null; do
  n=$(pgrep -f "run_editing_flow_(baseline|3dnoise)_wrapper.py .*full_real_face_" | wc -l)
  echo "[watch] $(date -Iseconds) running_jobs=$n" | tee -a "$WAIT_LOG"
  sleep 60
done

echo "[watch] all face jobs finished at $(date -Iseconds)" | tee -a "$WAIT_LOG"

cd "$ROOT"

bash "$ROOT/exp_flowedit_3dnoise/scripts/run_eval_face_redfix_r2.sh" 2>&1 | tee "$EVAL_LOG"

"$PY" "$ROOT/exp_flowedit_3dnoise/scripts/analyze_redcast_metrics.py" \
  --model_dirs \
    "$ROOT/output/flowedit_3dnoise_exp/full_real_face_baseline_ref_r2" \
    "$ROOT/output/flowedit_3dnoise_exp/full_real_face_baseline_neutral_r2" \
    "$ROOT/output/flowedit_3dnoise_exp/full_real_face_cov_init_nomfg_r2" \
    "$ROOT/output/flowedit_3dnoise_exp/full_real_face_cov_balanced_nomfg_r2" \
  --split train \
  --crop_ratio 0.6 \
  --out_json "$ROOT/exp_flowedit_3dnoise/results/face_redfix_r2_redcast_metrics_20260303.json" \
  --out_csv "$ROOT/exp_flowedit_3dnoise/results/face_redfix_r2_redcast_metrics_20260303.csv" 2>&1 | tee -a "$EVAL_LOG"

"$PY" "$ROOT/exp_flowedit_3dnoise/scripts/make_full_real_panels.py" \
  --source_pretrained_dir "$ROOT/dataset/pretrained/face" \
  --source_iter 30000 \
  --model_dirs \
    "$ROOT/output/flowedit_3dnoise_exp/full_real_face_baseline_ref_r2" \
    "$ROOT/output/flowedit_3dnoise_exp/full_real_face_baseline_neutral_r2" \
    "$ROOT/output/flowedit_3dnoise_exp/full_real_face_cov_init_nomfg_r2" \
    "$ROOT/output/flowedit_3dnoise_exp/full_real_face_cov_balanced_nomfg_r2" \
  --model_labels "baseline_ref,baseline_neutral,cov_init_nomfg,cov_balanced_nomfg" \
  --view_indices "0,8,16,24,32,40,48,55" \
  --out_path "$ROOT/exp_flowedit_3dnoise/results/face_redfix_r2_panel_20260303.png" 2>&1 | tee -a "$EVAL_LOG"

"$PY" "$ROOT/exp_flowedit_3dnoise/scripts/build_face_redfix_report.py" \
  --summary_csv "$ROOT/eval/cache/summaries_face_redfix_r2_20260303/by_method.csv" \
  --redcast_csv "$ROOT/exp_flowedit_3dnoise/results/face_redfix_r2_redcast_metrics_20260303.csv" \
  --panel_img "$ROOT/exp_flowedit_3dnoise/results/face_redfix_r2_panel_20260303.png" \
  --gallery_html "$ROOT/eval/cache/summaries_face_redfix_r2_20260303/gallery.html" \
  --sota_by_method_csv "$ROOT/eval/cache/summaries/by_method.csv" \
  --out_html "$ROOT/exp_flowedit_3dnoise/results/face_redfix_r2_comparison_report_20260303.html" 2>&1 | tee -a "$EVAL_LOG"

echo "[watch] post-eval done $(date -Iseconds)" | tee -a "$WAIT_LOG"
