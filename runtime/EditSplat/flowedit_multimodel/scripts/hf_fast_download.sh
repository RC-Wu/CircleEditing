#!/usr/bin/env bash
set -euo pipefail

ROOT="/dev-vepfs/rc_wu/rc_wu/edit/EditSplat/flowedit_multimodel"
PY="/dev-vepfs/rc_wu/rc_wu/envs/editsplat_ttt3r_unified_copy/bin/python"
HF_HOME_DEFAULT="/dev-vepfs/rc_wu/rc_wu/cache/hf_home"

MODELS="${1:-flux2-dev,sd35-large,qwen-image-edit}"

if [[ -n "${2:-}" ]]; then
  export HF_TOKEN="$2"
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[ERR] HF_TOKEN is empty. Pass token as 2nd arg or export HF_TOKEN." >&2
  exit 2
fi

export HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p "$HF_HUB_CACHE"

exec "$PY" "$ROOT/scripts/hf_fast_download.py" \
  --models "$MODELS" \
  --hf_home "$HF_HOME" \
  --hf_token "$HF_TOKEN" \
  --prefer_mirror \
  --max_workers 24
