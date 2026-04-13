#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${1:-circleediting-codex}"
PROMPT_FILE="${2:-}"
WORKDIR="${3:-/dev_vepfs/rc_wu/edit/CircleEditing}"

if [[ -n "${PROMPT_FILE}" && ! -f "${PROMPT_FILE}" ]]; then
  echo "prompt file not found: ${PROMPT_FILE}" >&2
  exit 1
fi

export HOME=/dev_vepfs/rc_wu
source /dev_vepfs/rc_wu/.codex/env.sh

CODEX_BIN="/dev_vepfs/rc_wu/.local/bin/codex"
if [[ ! -x "${CODEX_BIN}" ]]; then
  echo "codex binary wrapper missing: ${CODEX_BIN}" >&2
  exit 1
fi

mkdir -p "${WORKDIR}/runtime/codex_logs"
LOG_PATH="${WORKDIR}/runtime/codex_logs/${SESSION_NAME}.log"
STDERR_PATH="${WORKDIR}/runtime/codex_logs/${SESSION_NAME}.stderr.log"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}" >&2
  exit 1
fi

if [[ -n "${PROMPT_FILE}" ]]; then
  TMUX_CMD=$(cat <<EOF
cd "${WORKDIR}" && export HOME=/dev_vepfs/rc_wu && source /dev_vepfs/rc_wu/.codex/env.sh && ${CODEX_BIN} exec --sandbox danger-full-access --skip-git-repo-check --cd "${WORKDIR}" -o "${LOG_PATH}" - < "${PROMPT_FILE}" 2> "${STDERR_PATH}"
EOF
)
else
  TMUX_CMD=$(cat <<EOF
cd "${WORKDIR}" && export HOME=/dev_vepfs/rc_wu && source /dev_vepfs/rc_wu/.codex/env.sh && ${CODEX_BIN} exec --sandbox danger-full-access --skip-git-repo-check --cd "${WORKDIR}" > "${LOG_PATH}" 2> "${STDERR_PATH}"
EOF
)
fi

tmux new-session -d -s "${SESSION_NAME}" "${TMUX_CMD}"
echo "session=${SESSION_NAME}"
echo "log=${LOG_PATH}"
