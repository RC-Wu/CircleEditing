#!/usr/bin/env bash
set -euo pipefail

ROOT="/dev-vepfs/rc_wu/rc_wu/edit/EditSplat"
PY="/dev-vepfs/rc_wu/rc_wu/envs/editsplat_test/bin/python"

cd "$ROOT"

$PY exp_flowedit_3dnoise/scripts/run_noise_opt_flowedit.py \
  --source_path "$ROOT/dataset/dataset/dinosaur" \
  --source_checkpoint "$ROOT/dataset/pretrained/dinosaur/chkpnt7000.pth" \
  --model_id "black-forest-labs/FLUX.1-dev" \
  --use_local_files_only \
  --src_prompt "a photo of a dinosaur" \
  --tar_prompt "a photo of a toy dinosaur made of white marble" \
  --num_views 2 \
  --max_anchors 14000 \
  --opt_iters 5 \
  --views_per_iter 1 \
  --flow_steps 10 \
  --flow_n_max 8 \
  --noise_mix 0.9 \
  --suite_names "rand_baseline,cov_init_noopt,harmonic_init_noopt,cov_baseopt,hybrid_snropt,harmonic_balancedopt" \
  --run_suite \
  --output_dir "$ROOT/output/flowedit_3dnoise_exp/flux_dev_small_suite"
