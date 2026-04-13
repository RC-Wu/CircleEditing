#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

import build_fixed_gpu_overnight_queue as queue_builder


DEFAULT_QUEUE_ROOT = Path("/dev_vepfs/rc_wu/_codex_staging/ttt3r_overnight")
DEFAULT_WAVE_NAME = "20260413_wave18_gpu3_five_rounds"
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "manifests" / f"{DEFAULT_WAVE_NAME}.json"
DEFAULT_LAUNCHER_MODULE = Path(__file__).resolve().parent / "launch_dev01_ttt3r_consistency_wave.py"
DEFAULT_SLOT_GPU = 3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_runner_state(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run five sequential carrier rounds on one GPU.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--queue-root", type=Path, default=DEFAULT_QUEUE_ROOT)
    parser.add_argument("--wave-name", type=str, default=DEFAULT_WAVE_NAME)
    parser.add_argument("--launcher-module", type=Path, default=DEFAULT_LAUNCHER_MODULE)
    parser.add_argument("--slot-gpu", type=int, default=DEFAULT_SLOT_GPU)
    args = parser.parse_args()

    jobs = queue_builder.load_jobs(args.manifest)
    slots = [queue_builder.QueueSlot(gpu=args.slot_gpu)]
    queue_builder.write_queue_files(
        jobs=jobs,
        slots=slots,
        queue_root=args.queue_root,
        wave_name=args.wave_name,
        launcher_module_path=args.launcher_module,
    )

    queue_wave_root = args.queue_root / args.wave_name
    runner_state_path = queue_wave_root / "runner_state.json"
    history: list[dict] = []
    write_runner_state(
        runner_state_path,
        {
            "wave_name": args.wave_name,
            "slot_gpu": args.slot_gpu,
            "manifest": str(args.manifest),
            "started_at": utc_now(),
            "history": history,
        },
    )

    for job in jobs:
        job_json = queue_wave_root / "jobs" / f"{job.name}.json"
        row = {
            "job_name": job.name,
            "started_at": utc_now(),
            "job_json": str(job_json),
        }
        try:
            returncode = queue_builder.run_one_job(
                launcher_module_path=args.launcher_module,
                queue_root=args.queue_root,
                wave_name=args.wave_name,
                slot_gpu=args.slot_gpu,
                job_json=job_json,
            )
            row["returncode"] = returncode
        except Exception as exc:  # pragma: no cover - live protection path
            row["returncode"] = -999
            row["exception"] = repr(exc)
            row["traceback"] = traceback.format_exc()
        try:
            queue_builder.postprocess_job(
                launcher_module_path=args.launcher_module,
                queue_root=args.queue_root,
                wave_name=args.wave_name,
                slot_gpu=args.slot_gpu,
                job_json=job_json,
            )
        except Exception as exc:  # pragma: no cover
            row["postprocess_exception"] = repr(exc)
        try:
            queue_builder.gc_job(
                queue_root=args.queue_root,
                wave_name=args.wave_name,
                job_json=job_json,
            )
        except Exception as exc:  # pragma: no cover
            row["gc_exception"] = repr(exc)
        row["finished_at"] = utc_now()
        history.append(row)
        write_runner_state(
            runner_state_path,
            {
                "wave_name": args.wave_name,
                "slot_gpu": args.slot_gpu,
                "manifest": str(args.manifest),
                "started_at": json.loads(runner_state_path.read_text(encoding="utf-8")).get("started_at"),
                "updated_at": utc_now(),
                "history": history,
            },
        )


if __name__ == "__main__":
    main()
