#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 PROMPT_FILE OUTPUT_FILE [extra codex args...]" >&2
  exit 2
fi

PROMPT_FILE="$1"
OUTPUT_FILE="$2"
shift 2

TMP_BIN="$(mktemp /tmp/codex_exec_XXXXXX)"
cleanup() {
  rm -f "$TMP_BIN"
}
trap cleanup EXIT

cp /dev_vepfs/rc_wu/bin/codex "$TMP_BIN"
chmod 755 "$TMP_BIN"

"$TMP_BIN" exec \
  --sandbox danger-full-access \
  --skip-git-repo-check \
  --cd /dev_vepfs/rc_wu/edit/CircleEditing \
  -o "$OUTPUT_FILE" \
  "$@" \
  - < "$PROMPT_FILE"
