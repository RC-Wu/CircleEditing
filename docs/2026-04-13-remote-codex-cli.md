# Remote Codex CLI Startup

Date: 2026-04-13

Scope:
- Standardize detached Codex CLI startup inside `/dev_vepfs/rc_wu/edit/CircleEditing`.
- Reuse the shared `/dev_vepfs/rc_wu/.codex` home so the custom relay endpoint stays active.

Requirements:
- `HOME` must be `/dev_vepfs/rc_wu`.
- Launch through `/dev_vepfs/rc_wu/.local/bin/codex`, not a random binary copy.
- Source `/dev_vepfs/rc_wu/.codex/env.sh` before `codex exec`.
- Keep logs under `runtime/codex_logs/`.

Operational entrypoint:
- `runtime/tools/run_remote_codex_tmux.sh`

Example:
- `bash runtime/tools/run_remote_codex_tmux.sh circleediting-doc-sync /tmp/prompt.md /dev_vepfs/rc_wu/edit/CircleEditing`
