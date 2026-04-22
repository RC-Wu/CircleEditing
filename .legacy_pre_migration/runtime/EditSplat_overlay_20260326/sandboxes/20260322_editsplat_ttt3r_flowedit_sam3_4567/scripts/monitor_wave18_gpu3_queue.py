#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_QUEUE_ROOT = Path("/dev_vepfs/rc_wu/_codex_staging/ttt3r_overnight")
DEFAULT_WAVE_NAME = "20260413_wave18_gpu3_five_rounds"
DEFAULT_AGENTDOC_NOTE = Path(
    "/dev_vepfs/rc_wu/AgentDoc/PROJECTS/3d_editing/experiments/exp_wave18_gpu3_five_rounds_20260413.md"
)
DEFAULT_PROJECT_NOTE = Path(
    "/dev_vepfs/rc_wu/edit/EditSplat/sandboxes/20260322_editsplat_ttt3r_flowedit_sam3_4567/docs/2026-04-13-dev02-gpu3-five-round-status.md"
)
DEFAULT_OUTBOX = Path("/dev_vepfs/rc_wu/AgentDoc/outbox/runtime_doc_checkpoints.md")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, fallback):
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def latest_visual_paths(model_path: Path) -> list[str]:
    patterns = [
        "analysis/*.png",
        "analysis/*.html",
        "debug_intermediates/mfg_edit/view*/stats.json",
        "debug_intermediates/semantic_guidance/gaussian_mask_stats.json",
    ]
    hits: list[str] = []
    for pattern in patterns:
        for path in sorted(model_path.glob(pattern)):
            hits.append(str(path))
            if len(hits) >= 6:
                return hits
    return hits


def running_process_lines(wave_name: str) -> list[str]:
    cmd = (
        "ps -eo pid,etime,cmd | grep -E "
        f"'{wave_name}|run_sd35_ttt3r_sam3_wrapper.py' | grep -v grep || true"
    )
    out = subprocess.check_output(["bash", "-lc", cmd], text=True)
    return [line for line in out.splitlines() if line.strip()]


def queue_snapshot(queue_wave_root: Path) -> dict:
    manifest = read_json(queue_wave_root / "manifest.json", [])
    statuses = []
    status_dir = queue_wave_root / "status"
    if status_dir.exists():
        for path in sorted(status_dir.glob("*.json")):
            payload = read_json(path, {})
            if isinstance(payload, dict):
                payload["_status_path"] = str(path)
                statuses.append(payload)
    history = read_json(queue_wave_root / "runner_state.json", {}).get("history", [])
    completed = sum(1 for item in statuses if int(item.get("returncode", 1)) == 0)
    failed = sum(1 for item in statuses if int(item.get("returncode", 1)) != 0)
    return {
        "manifest": manifest,
        "statuses": statuses,
        "history": history,
        "completed": completed,
        "failed": failed,
        "pending": max(len(manifest) - len(statuses), 0),
        "running": running_process_lines(queue_wave_root.name),
    }


def render_markdown(queue_wave_root: Path, snapshot: dict) -> str:
    lines = [
        "---",
        f'title: dev02 GPU3 five-round carrier queue monitor {queue_wave_root.name}',
        "updated_at: " + utc_now(),
        "tags:",
        "  - 3d_editing",
        "  - EditSplat",
        "  - wave18",
        "  - gpu3",
        "  - carrier",
        "status: active",
        "---",
        "",
        f"# {queue_wave_root.name}",
        "",
        "## Objective",
        "- Keep `dev-intern-02` GPU `3` busy with a single-card five-round carrier queue.",
        "- Follow the current research order from human docs: `A-baseline` first, then `B-lite`, while retaining visual evidence.",
        "- Produce render/panel artifacts for direct visual inspection, not metrics-only triage.",
        "",
        "## Queue Summary",
        f"- total_jobs: `{len(snapshot['manifest'])}`",
        f"- completed: `{snapshot['completed']}`",
        f"- failed: `{snapshot['failed']}`",
        f"- pending: `{snapshot['pending']}`",
        "",
        "## Running Processes",
    ]
    if snapshot["running"]:
        lines.extend(f"- `{line}`" for line in snapshot["running"])
    else:
        lines.append("- none")
    lines.extend(["", "## Job Ledger"])
    if not snapshot["statuses"]:
        lines.append("- no completed status files yet")
    else:
        for status in snapshot["statuses"]:
            model_path = Path(str(status.get("model_path", "")))
            visuals = latest_visual_paths(model_path) if model_path.exists() else []
            lines.append(
                f"- `{status.get('run_name')}`: returncode=`{status.get('returncode')}`, "
                f"log=`{status.get('log_path')}`, model=`{status.get('model_path')}`"
            )
            if visuals:
                lines.append(f"  visuals: {', '.join(visuals[:4])}")
    lines.extend(
        [
            "",
            "## Research Rationale",
            "- `wave17` human analysis says the blocker is full-face collapse; semantic sweeps alone are no longer the main lever.",
            "- The immediate priority is carrier probing: `A-baseline` sanity check, then `B-lite` canonical edit-field probe.",
            "- This queue intentionally stays on one prompt family long enough to inspect visual separation before expanding wider.",
        ]
    )
    return "\n".join(lines) + "\n"


def append_checkpoint(outbox_path: Path, queue_wave_root: Path, snapshot: dict) -> None:
    outbox_path.parent.mkdir(parents=True, exist_ok=True)
    with outbox_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"- {utc_now()} wave=`{queue_wave_root.name}` completed={snapshot['completed']} "
            f"failed={snapshot['failed']} pending={snapshot['pending']}\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor the single-GPU five-round carrier queue.")
    parser.add_argument("--queue-root", type=Path, default=DEFAULT_QUEUE_ROOT)
    parser.add_argument("--wave-name", type=str, default=DEFAULT_WAVE_NAME)
    parser.add_argument("--agentdoc-note", type=Path, default=DEFAULT_AGENTDOC_NOTE)
    parser.add_argument("--project-note", type=Path, default=DEFAULT_PROJECT_NOTE)
    parser.add_argument("--outbox", type=Path, default=DEFAULT_OUTBOX)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    queue_wave_root = args.queue_root / args.wave_name
    state_path = queue_wave_root / "monitor_state.json"
    previous = read_json(state_path, {})

    while True:
        snapshot = queue_snapshot(queue_wave_root)
        text = render_markdown(queue_wave_root, snapshot)
        for note_path in (args.agentdoc_note, args.project_note):
            note_path.parent.mkdir(parents=True, exist_ok=True)
            note_path.write_text(text, encoding="utf-8")
        current_key = {
            "completed": snapshot["completed"],
            "failed": snapshot["failed"],
            "pending": snapshot["pending"],
        }
        if current_key != previous:
            append_checkpoint(args.outbox, queue_wave_root, snapshot)
            state_path.write_text(json.dumps(current_key, indent=2), encoding="utf-8")
            previous = current_key
        if snapshot["pending"] == 0 and not snapshot["running"]:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
