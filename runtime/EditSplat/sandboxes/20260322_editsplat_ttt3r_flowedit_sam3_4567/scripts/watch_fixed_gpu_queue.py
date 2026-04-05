#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_QUEUE_ROOT = Path("/dev_vepfs/rc_wu/_codex_staging/ttt3r_overnight")
DEFAULT_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_HELPER = DEFAULT_SCRIPT_DIR / "build_fixed_gpu_overnight_queue.py"
DEFAULT_LAUNCHER = DEFAULT_SCRIPT_DIR / "launch_dev01_ttt3r_consistency_wave.py"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_md(log_md: Path, line: str) -> None:
    log_md.parent.mkdir(parents=True, exist_ok=True)
    if not log_md.exists():
        log_md.write_text("# Fixed GPU Queue Watch\n\n", encoding="utf-8")
    with log_md.open("a", encoding="utf-8") as handle:
        handle.write(f"- {utc_stamp()} {line}\n")


def load_manifest_jobs(manifest: Path) -> list[dict]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"manifest must contain a list: {manifest}")
    return payload


def query_gpu_state(gpus: Iterable[int]) -> dict[int, dict[str, int]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    wanted = {int(x) for x in gpus}
    state: dict[int, dict[str, int]] = {}
    for raw in output.strip().splitlines():
        idx_s, mem_s, util_s = [chunk.strip() for chunk in raw.split(",")]
        idx = int(idx_s)
        if idx not in wanted:
            continue
        state[idx] = {"memory_used_mib": int(mem_s), "util_gpu": int(util_s)}
    return state


def query_dev_vepfs_free_gb() -> int:
    output = subprocess.check_output(["df", "-BG", "/dev_vepfs/rc_wu"], text=True)
    line = output.strip().splitlines()[-1]
    return int(line.split()[3].rstrip("G"))


def gpu_slots_are_idle(
    gpu_state: dict[int, dict[str, int]],
    gpu_slots: list[int],
    max_memory_used_mib: int,
    max_util_gpu: int,
) -> bool:
    for gpu in gpu_slots:
        stats = gpu_state.get(gpu)
        if stats is None:
            return False
        if stats["memory_used_mib"] > max_memory_used_mib:
            return False
        if stats["util_gpu"] > max_util_gpu:
            return False
    return True


def wave_root(queue_root: Path, wave_name: str) -> Path:
    return queue_root / wave_name


def active_wave_process_count(wave_name: str) -> int:
    output = subprocess.check_output(["ps", "-eo", "cmd"], text=True)
    needles = ("run-one", "run_sd35_ttt3r_sam3_wrapper.py", f"/{wave_name}/scripts/slot_")
    count = 0
    for line in output.splitlines():
        if wave_name not in line:
            continue
        if any(needle in line for needle in needles):
            count += 1
    return count


def status_count(queue_root: Path, wave_name: str) -> int:
    status_dir = wave_root(queue_root, wave_name) / "status"
    if not status_dir.exists():
        return 0
    return len(list(status_dir.glob("*.json")))


def build_sheet_for_relpath(
    runs: list[Path],
    relpath: str,
    out_path: Path,
    strip_prefix: str,
) -> bool:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return False

    entries = []
    for run in runs:
        panel = run / relpath
        if panel.exists():
            entries.append((run.name.replace(strip_prefix, ""), panel))
    if not entries:
        return False

    font = ImageFont.load_default()
    target_w = 420
    label_h = 42
    pad = 12
    cols = 2
    rendered = []
    for name, panel_path in entries:
        img = Image.open(panel_path).convert("RGB")
        target_h = max(1, int(img.height * target_w / img.width))
        rendered.append((name, img.resize((target_w, target_h))))

    rows = (len(rendered) + cols - 1) // cols
    cell_w = target_w + pad * 2
    cell_h = max(img.height for _, img in rendered) + label_h + pad * 2
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)

    for idx, (name, img) in enumerate(rendered):
        row = idx // cols
        col = idx % cols
        x = col * cell_w + pad
        y = row * cell_h + pad
        draw.rectangle([x, y, x + target_w, y + label_h - 4], fill=(245, 245, 245))
        draw.text((x + 6, y + 10), name, fill="black", font=font)
        sheet.paste(img, (x, y + label_h))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return True


def build_review_panels(queue_root: Path, wave_name: str) -> list[Path]:
    root = wave_root(queue_root, wave_name)
    results_root = root / "results"
    runs = sorted([path for path in results_root.iterdir() if path.is_dir()]) if results_root.exists() else []
    strip_prefix = f"{wave_name}_"
    outputs: list[Path] = []
    for relpath, out_name in (
        ("analysis/panel_final_grid.png", f"{wave_name}_final_sheet.png"),
        ("analysis/panel_teacher_fit_focus.png", f"{wave_name}_teacher_fit_sheet.png"),
        ("analysis/panel_pipeline.png", f"{wave_name}_pipeline_sheet.png"),
    ):
        out_path = root / "review_panels" / out_name
        if build_sheet_for_relpath(runs=runs, relpath=relpath, out_path=out_path, strip_prefix=strip_prefix):
            outputs.append(out_path)
    return outputs


def launch_queue(
    helper: Path,
    launcher: Path,
    manifest: Path,
    queue_root: Path,
    wave_name: str,
    gpu_slots: list[int],
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(helper),
        "launch",
        "--manifest",
        str(manifest),
        "--queue-root",
        str(queue_root),
        "--wave-name",
        wave_name,
        "--launcher-module",
        str(launcher),
        "--gpu-slots",
        *[str(gpu) for gpu in gpu_slots],
    ]
    return subprocess.run(cmd, text=True, capture_output=True, check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Wait for fixed GPUs to go idle, then launch a queue wave.")
    parser.add_argument("--wave-name", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpu-slots", type=int, nargs="+", required=True)
    parser.add_argument("--queue-root", type=Path, default=DEFAULT_QUEUE_ROOT)
    parser.add_argument("--helper", type=Path, default=DEFAULT_HELPER)
    parser.add_argument("--launcher-module", type=Path, default=DEFAULT_LAUNCHER)
    parser.add_argument("--poll-seconds", type=int, default=180)
    parser.add_argument("--max-memory-used-mib", type=int, default=2000)
    parser.add_argument("--max-util-gpu", type=int, default=10)
    parser.add_argument("--min-free-dev-vepfs-gb", type=int, default=80)
    parser.add_argument("--build-review-panels", action="store_true")
    parser.add_argument("--wait-for-completion", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-md", type=Path, default=None)
    args = parser.parse_args()

    gpu_slots = [int(gpu) for gpu in args.gpu_slots]
    log_md = args.log_md or (args.queue_root / f"{args.wave_name}_launch_watch.md")
    jobs = load_manifest_jobs(args.manifest)
    expected_jobs = len(jobs)
    target_root = wave_root(args.queue_root, args.wave_name)

    append_md(
        log_md,
        (
            f"watch_started wave={args.wave_name} manifest={args.manifest} "
            f"gpu_slots={gpu_slots} expected_jobs={expected_jobs}"
        ),
    )

    if target_root.exists():
        existing = sum(1 for _ in target_root.rglob("*"))
        append_md(log_md, f"wave_root_already_exists path={target_root} existing_entries={existing}; stopping")
        raise SystemExit(0)

    launched = False
    while not launched:
        free_gb = query_dev_vepfs_free_gb()
        gpu_state = query_gpu_state(gpu_slots)
        flat_state = " ".join(
            f"gpu{gpu}:mem={gpu_state.get(gpu, {}).get('memory_used_mib', -1)}MiB"
            f"/util={gpu_state.get(gpu, {}).get('util_gpu', -1)}"
            for gpu in gpu_slots
        )
        append_md(log_md, f"poll free_dev_vepfs={free_gb}G {flat_state}")

        if free_gb < int(args.min_free_dev_vepfs_gb):
            append_md(
                log_md,
                (
                    f"free_dev_vepfs_below_threshold free={free_gb}G "
                    f"threshold={args.min_free_dev_vepfs_gb}G; stopping"
                ),
            )
            raise SystemExit(0)

        if gpu_slots_are_idle(
            gpu_state=gpu_state,
            gpu_slots=gpu_slots,
            max_memory_used_mib=int(args.max_memory_used_mib),
            max_util_gpu=int(args.max_util_gpu),
        ):
            if args.dry_run:
                append_md(log_md, "dry_run_idle_gate_satisfied")
                return
            proc = launch_queue(
                helper=args.helper,
                launcher=args.launcher_module,
                manifest=args.manifest,
                queue_root=args.queue_root,
                wave_name=args.wave_name,
                gpu_slots=gpu_slots,
            )
            append_md(log_md, f"launch_returncode={proc.returncode}")
            stdout = proc.stdout.strip()
            stderr = proc.stderr.strip()
            if stdout:
                append_md(log_md, f"launch_stdout={stdout}")
            if stderr:
                append_md(log_md, f"launch_stderr={stderr}")
            if proc.returncode != 0:
                raise SystemExit(proc.returncode)
            launched = True
            break

        time.sleep(max(10, int(args.poll_seconds)))

    if not args.wait_for_completion:
        append_md(log_md, "launch_complete watcher_exit_without_completion_wait")
        return

    append_md(log_md, "completion_watch_started")
    while True:
        current_status = status_count(args.queue_root, args.wave_name)
        active = active_wave_process_count(args.wave_name)
        append_md(log_md, f"completion_poll status={current_status}/{expected_jobs} active={active}")
        if current_status >= expected_jobs and active == 0:
            outputs: list[Path] = []
            if args.build_review_panels:
                outputs = build_review_panels(args.queue_root, args.wave_name)
            if outputs:
                append_md(log_md, "review_panels_ready " + " ".join(str(path) for path in outputs))
            append_md(log_md, "wave_complete watcher_exit")
            return
        time.sleep(max(30, int(args.poll_seconds)))


if __name__ == "__main__":
    main()
