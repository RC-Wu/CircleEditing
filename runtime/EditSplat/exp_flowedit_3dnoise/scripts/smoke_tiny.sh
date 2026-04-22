#!/usr/bin/env bash
set -euo pipefail

ROOT="/dev-vepfs/rc_wu/rc_wu/edit/EditSplat"
PY="/dev-vepfs/rc_wu/rc_wu/envs/editsplat_test/bin/python"

cd "$ROOT"

$PY exp_flowedit_3dnoise/scripts/run_noise_opt_flowedit.py \
  --source_path "$ROOT/dataset/dataset/dinosaur" \
  --source_checkpoint "$ROOT/dataset/pretrained/dinosaur/chkpnt7000.pth" \
  --model_id "yujiepan/FLUX.1-dev-tiny-random" \
  --src_prompt "a photo of a dinosaur" \
  --tar_prompt "a photo of a toy dinosaur made of marble" \
  --num_views 2 \
  --max_anchors 12000 \
  --opt_iters 4 \
  --flow_steps 8 \
  --flow_n_max 6 \
  --views_per_iter 1 \
  --noise_mix 0.9 \
  --output_dir "$ROOT/output/flowedit_3dnoise_exp/smoke_tiny"
