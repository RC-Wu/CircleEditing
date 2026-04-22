#!/usr/bin/env bash
set -euo pipefail

PY="/dev-vepfs/rc_wu/rc_wu/envs/editsplat_test/bin/python"
CACHE_ROOT="/dev-vepfs/rc_wu/rc_wu/cache/hf_home"
CACHE_HUB="$CACHE_ROOT/hub"

if [[ -n "${1:-}" ]]; then
  export HF_TOKEN="$1"
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is empty. Pass token as first arg or export HF_TOKEN." >&2
  exit 1
fi

export HF_HOME="$CACHE_ROOT"
export HF_HUB_CACHE="$CACHE_HUB"
export HF_HUB_ENABLE_HF_TRANSFER=1

# Full FLUX.1-dev download with resume and hf_transfer acceleration.
$PY -m huggingface_hub.commands.huggingface_cli download \
  black-forest-labs/FLUX.1-dev \
  --cache-dir "$CACHE_HUB" \
  --resume-download
