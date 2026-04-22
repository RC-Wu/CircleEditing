#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LAUNCHER_MODULE = SCRIPT_DIR / "launch_dev01_ttt3r_consistency_wave.py"
DEFAULT_QUEUE_ROOT = Path("/dev_vepfs/rc_wu/_codex_staging/ttt3r_overnight")
DEFAULT_RENDER_PROJECT_ROOT = Path("/dev_vepfs/rc_wu/edit/EditSplat")
DEFAULT_RENDER_SCRIPT = DEFAULT_RENDER_PROJECT_ROOT / "render.py"
DEFAULT_PANEL_SCRIPT = (
    DEFAULT_RENDER_PROJECT_ROOT
    / "sandboxes"
    / "20260319_aris_ttt3r_flowedit_45"
    / "scripts"
    / "build_run_panels.py"
)


@dataclass(frozen=True)
class QueueSlot:
    gpu: int
    label: str | None = None

    @property
    def key(self) -> str:
        return self.label or f"gpu{self.gpu}"


@dataclass
class QueueJob:
    name: str
    exp_kwargs: Dict[str, object] = field(default_factory=dict)
    extra_env: Dict[str, str] = field(default_factory=dict)


def assign_jobs_round_robin(
    jobs: List[QueueJob],
    slots: List[QueueSlot],
) -> Dict[str, List[QueueJob]]:
    if not slots:
        raise ValueError("at least one queue slot is required")
    assigned: Dict[str, List[QueueJob]] = {slot.key: [] for slot in slots}
    for idx, job in enumerate(jobs):
        assigned[slots[idx % len(slots)].key].append(job)
    return assigned


def load_launcher_module(launcher_module_path: Path):
    spec = importlib.util.spec_from_file_location("ttt3r_wave_launcher", launcher_module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load launcher module from {launcher_module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def render_slot_script(
    slot: QueueSlot,
    jobs: List[QueueJob],
    launcher_module_path: Path,
    queue_root: Path,
    wave_name: str,
    queue_script_path: Path | None = None,
) -> str:
    queue_wave_root = queue_root / wave_name
    job_dir = queue_wave_root / "jobs"
    queue_script = queue_script_path or Path(__file__).resolve()
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"export CUDA_VISIBLE_DEVICES={slot.gpu}",
        f"QUEUE_ROOT={shlex.quote(str(queue_root))}",
        f"WAVE_NAME={shlex.quote(wave_name)}",
        "",
    ]
    for job in jobs:
        lines.append(f"# {job.name}")
        for key, value in sorted(job.extra_env.items()):
            lines.append(f"export {key}={shlex.quote(str(value))}")
        lines.append(
            "python3 "
            + shlex.quote(str(queue_script))
            + " run-one "
            + f"--launcher-module {shlex.quote(str(launcher_module_path))} "
            + f"--queue-root {shlex.quote(str(queue_root))} "
            + f"--wave-name {shlex.quote(wave_name)} "
            + f"--slot-gpu {slot.gpu} "
            + f"--job-json {shlex.quote(str(job_dir / (job.name + '.json')))}"
        )
        lines.append(
            "python3 "
            + shlex.quote(str(queue_script))
            + " postprocess-job "
            + f"--launcher-module {shlex.quote(str(launcher_module_path))} "
            + f"--queue-root {shlex.quote(str(queue_root))} "
            + f"--wave-name {shlex.quote(wave_name)} "
            + f"--slot-gpu {slot.gpu} "
            + f"--job-json {shlex.quote(str(job_dir / (job.name + '.json')))}"
        )
        lines.append(
            "# GC transient artifacts: point_cloud/*.ply, point_cloud/**/chkpnt*.pth, "
            "debug_intermediates/*_step*, duplicate train/debug PNGs, while keeping gaussian_mask_stats.json "
            "and light review outputs"
        )
        lines.append(
            "python3 "
            + shlex.quote(str(queue_script))
            + " gc-job "
            + f"--launcher-module {shlex.quote(str(launcher_module_path))} "
            + f"--queue-root {shlex.quote(str(queue_root))} "
            + f"--wave-name {shlex.quote(wave_name)} "
            + f"--job-json {shlex.quote(str(job_dir / (job.name + '.json')))}"
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def build_detached_launch_command(script_path: Path, log_path: Path) -> str:
    return f"nohup bash {shlex.quote(str(script_path))} > {shlex.quote(str(log_path))} 2>&1 < /dev/null &"


def build_default_jobs() -> List[QueueJob]:
    return []


def load_jobs(manifest_path: Path | None) -> List[QueueJob]:
    if manifest_path is None:
        return build_default_jobs()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [QueueJob(**item) for item in payload]


def write_queue_files(
    jobs: List[QueueJob],
    slots: List[QueueSlot],
    queue_root: Path,
    wave_name: str,
    launcher_module_path: Path,
) -> Dict[str, Path]:
    queue_wave_root = queue_root / wave_name
    jobs_dir = queue_wave_root / "jobs"
    scripts_dir = queue_wave_root / "scripts"
    logs_dir = queue_wave_root / "slot_logs"
    for path in (jobs_dir, scripts_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    assignments = assign_jobs_round_robin(jobs=jobs, slots=slots)
    (queue_wave_root / "manifest.json").write_text(
        json.dumps([asdict(job) for job in jobs], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (queue_wave_root / "queue_config.json").write_text(
        json.dumps(
            {
                "launcher_module": str(launcher_module_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    for job in jobs:
        (jobs_dir / f"{job.name}.json").write_text(
            json.dumps(asdict(job), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    script_paths: Dict[str, Path] = {}
    for slot in slots:
        script_path = scripts_dir / f"slot_{slot.key}.sh"
        script_path.write_text(
            render_slot_script(
                slot=slot,
                jobs=assignments[slot.key],
                launcher_module_path=launcher_module_path,
                queue_root=queue_root,
                wave_name=wave_name,
                queue_script_path=Path(__file__).resolve(),
            ),
            encoding="utf-8",
        )
        script_path.chmod(0o755)
        script_paths[slot.key] = script_path
    return script_paths


def run_one_job(
    launcher_module_path: Path,
    queue_root: Path,
    wave_name: str,
    slot_gpu: int,
    job_json: Path,
) -> int:
    launcher = load_launcher_module(launcher_module_path)
    queue_wave_root = queue_root / wave_name
    launcher.LOG_DIR = queue_wave_root / "logs"
    launcher.RESULTS_DIR = queue_wave_root / "results"
    launcher.SUMMARY_DIR = queue_wave_root / "summaries"
    launcher.ensure_layout()

    job = QueueJob(**json.loads(job_json.read_text(encoding="utf-8")))
    exp_kwargs = dict(job.exp_kwargs)
    exp_kwargs["name"] = job.name
    exp_kwargs["gpu"] = slot_gpu
    exp = launcher.Experiment(**exp_kwargs)
    run_name = launcher.build_run_name(exp, wave_name)
    model_path = launcher.RESULTS_DIR / run_name
    source_path = launcher.dataset_for_case(exp.case_name)
    launcher.ensure_cfg_args(model_path=model_path, source_path=source_path)

    if hasattr(launcher, "build_launch_env"):
        env = launcher.build_launch_env(exp)
    else:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(slot_gpu)
        env["HF_HOME"] = str(launcher.HF_HOME)
        env["HF_HUB_CACHE"] = str(launcher.HF_HOME / "hub")
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["EDITSPLAT_HF_HOME"] = str(launcher.HF_HOME)
        env["EDITSPLAT_HF_TOKEN_FILE"] = str(launcher.HF_TOKEN)
        env["EDITSPLAT_SAM3_CHECKPOINT_PATH"] = str(launcher.SAM3_PT)
        env["EDITSPLAT_MASK_BACKEND"] = str(exp.mask_backend)
        env["EDITSPLAT_SAM3_DEVICE"] = "cpu"
        env["EDITSPLAT_EXTERNAL_BACKEND_ONLY"] = "1"
        env["EDITSPLAT_GAUSSIAN_MASK_MODE"] = "projection"
        env["EDITSPLAT_SKIP_3DGS_BACKWARD_ON_ERROR"] = "1"
        env["EDITSPLAT_SKIP_RENDER_SETS"] = "1"
        env["EDITSPLAT_MAX_TRAIN_VIEWS"] = str(exp.max_train_views)
        env["EDITSPLAT_MAX_GAUSSIANS"] = str(exp.max_gaussians)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        if exp.dump_intermediates:
            env["EDITSPLAT_DUMP_INTERMEDIATES"] = "1"
    for key, value in job.extra_env.items():
        env[key] = str(value)

    log_path = launcher.LOG_DIR / f"{run_name}.log"
    cmd = launcher.build_command(exp=exp, wave_name=wave_name)
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            cmd,
            cwd=str(launcher.ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )

    status_path = queue_wave_root / "status"
    status_path.mkdir(parents=True, exist_ok=True)
    (status_path / f"{run_name}.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "returncode": proc.returncode,
                "log_path": str(log_path),
                "model_path": str(model_path),
                "extra_env": job.extra_env,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    wave_summary_path = launcher.SUMMARY_DIR / f"{wave_name}_summary.json"
    existing_rows = []
    if wave_summary_path.exists():
        try:
            loaded = json.loads(wave_summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                existing_rows = [row for row in loaded if isinstance(row, dict) and row.get("run_name") != run_name]
        except Exception:
            existing_rows = []

    summary_path = launcher.collect_summary(
        runs=[
            {
                "run_name": run_name,
                "gpu": slot_gpu,
                "log_path": str(log_path),
                "model_path": str(model_path),
            }
        ],
        wave_name=wave_name,
    )
    try:
        fresh_rows = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    except Exception:
        fresh_rows = []
    if not isinstance(fresh_rows, list):
        fresh_rows = []
    merged_rows = list(existing_rows)
    merged_rows.extend(row for row in fresh_rows if isinstance(row, dict))
    wave_summary_path.write_text(
        json.dumps(merged_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return proc.returncode


def resolve_gc_launcher_module(
    queue_wave_root: Path,
    launcher_module_path: Path | None,
) -> Path:
    if launcher_module_path is not None:
        return launcher_module_path

    queue_config_path = queue_wave_root / "queue_config.json"
    if queue_config_path.exists():
        try:
            payload = json.loads(queue_config_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        launcher_value = payload.get("launcher_module") if isinstance(payload, dict) else None
        if launcher_value:
            return Path(str(launcher_value))

    scripts_dir = queue_wave_root / "scripts"
    for script_path in sorted(scripts_dir.glob("slot_*.sh")):
        try:
            text = script_path.read_text(encoding="utf-8")
        except OSError:
            continue
        marker = "--launcher-module "
        idx = text.find(marker)
        if idx == -1:
            continue
        remainder = text[idx + len(marker) :]
        candidate = remainder.split()[0].strip()
        if candidate:
            return Path(candidate)

    return DEFAULT_LAUNCHER_MODULE


def gc_job(
    queue_root: Path,
    wave_name: str,
    job_json: Path,
    launcher_module_path: Path | None = None,
) -> None:
    queue_wave_root = queue_root / wave_name
    launcher = load_launcher_module(resolve_gc_launcher_module(queue_wave_root, launcher_module_path))
    job = QueueJob(**json.loads(job_json.read_text(encoding="utf-8")))
    exp_kwargs = dict(job.exp_kwargs)
    exp_kwargs["name"] = job.name
    exp_kwargs["gpu"] = 0
    exp = launcher.Experiment(**exp_kwargs)
    run_name = launcher.build_run_name(exp, wave_name)
    model_path = queue_wave_root / "results" / run_name
    if not model_path.exists():
        return

    for pattern in ("point_cloud/**/chkpnt*.pth", "point_cloud/**/point_cloud.ply"):
        for path in model_path.glob(pattern):
            if path.is_file():
                path.unlink(missing_ok=True)

    debug_root = model_path / "debug_intermediates"
    if debug_root.exists():
        for path in debug_root.glob("**/*_step*"):
            if path.is_dir():
                for sub in sorted(path.rglob("*"), reverse=True):
                    if sub.is_file():
                        sub.unlink(missing_ok=True)
                    elif sub.is_dir():
                        try:
                            sub.rmdir()
                        except OSError:
                            pass
                try:
                    path.rmdir()
                except OSError:
                    pass
        for pattern in ("initial_edit/*/input.png", "mfg_edit/*/gt.png"):
            for path in debug_root.glob(pattern):
                if path.is_file():
                    path.unlink(missing_ok=True)

    for pattern in ("train/ours_*/gt/*", "train/ours_*/renders/*"):
        for path in model_path.glob(pattern):
            if path.is_file():
                path.unlink(missing_ok=True)

    for path in sorted(model_path.glob("point_cloud/**"), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass


def postprocess_job(
    launcher_module_path: Path,
    queue_root: Path,
    wave_name: str,
    slot_gpu: int,
    job_json: Path,
) -> None:
    launcher = load_launcher_module(launcher_module_path)
    queue_wave_root = queue_root / wave_name
    job = QueueJob(**json.loads(job_json.read_text(encoding="utf-8")))
    exp_kwargs = dict(job.exp_kwargs)
    exp_kwargs["name"] = job.name
    exp_kwargs["gpu"] = slot_gpu
    exp = launcher.Experiment(**exp_kwargs)
    run_name = launcher.build_run_name(exp, wave_name)
    model_path = queue_wave_root / "results" / run_name
    source_path = launcher.dataset_for_case(exp.case_name)
    point_cloud_root = model_path / "point_cloud"
    if not point_cloud_root.exists():
        return

    iteration_dirs = sorted(point_cloud_root.glob("iteration_*"))
    if not iteration_dirs:
        return
    latest_iter = max(
        int(path.name.split("iteration_", 1)[1])
        for path in iteration_dirs
        if path.name.startswith("iteration_")
    )

    render_resolution = getattr(exp, "resolution", 384)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(slot_gpu)
    try:
        subprocess.run(
            [
                str(launcher.PYTHON),
                str(DEFAULT_RENDER_SCRIPT),
                "-s",
                str(source_path),
                "-m",
                str(model_path),
                "--resolution",
                str(render_resolution),
                "--iteration",
                str(latest_iter),
                "--skip_test",
                "--quiet",
            ],
            cwd=str(DEFAULT_RENDER_PROJECT_ROOT),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        subprocess.run(
            [
                str(launcher.PYTHON),
                str(DEFAULT_PANEL_SCRIPT),
                "--run-dir",
                str(model_path),
                "--out-dir",
                str(model_path / "analysis"),
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        return


def launch_slot_scripts(script_paths: Dict[str, Path], queue_root: Path, wave_name: str) -> None:
    queue_wave_root = queue_root / wave_name
    slot_logs = queue_wave_root / "slot_logs"
    for slot_key, script_path in script_paths.items():
        log_path = slot_logs / f"{slot_key}.log"
        subprocess.run(
            ["bash", "-lc", build_detached_launch_command(script_path=script_path, log_path=log_path)],
            check=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and run a fixed-GPU overnight TTT3R queue")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    for name in ("emit", "launch"):
        sub = subparsers.add_parser(name, help=f"{name} queue files")
        sub.add_argument("--manifest", type=Path, default=None)
        sub.add_argument("--queue-root", type=Path, default=DEFAULT_QUEUE_ROOT)
        sub.add_argument("--wave-name", type=str, required=True)
        sub.add_argument("--launcher-module", type=Path, default=DEFAULT_LAUNCHER_MODULE)
        sub.add_argument("--gpu-slots", type=int, nargs="+", default=[1, 2])

    run_one = subparsers.add_parser("run-one", help="internal: run a single queued job")
    run_one.add_argument("--launcher-module", type=Path, required=True)
    run_one.add_argument("--queue-root", type=Path, required=True)
    run_one.add_argument("--wave-name", type=str, required=True)
    run_one.add_argument("--slot-gpu", type=int, required=True)
    run_one.add_argument("--job-json", type=Path, required=True)

    post = subparsers.add_parser("postprocess-job", help="render and build panels for one job")
    post.add_argument("--launcher-module", type=Path, required=True)
    post.add_argument("--queue-root", type=Path, required=True)
    post.add_argument("--wave-name", type=str, required=True)
    post.add_argument("--slot-gpu", type=int, required=True)
    post.add_argument("--job-json", type=Path, required=True)

    gc = subparsers.add_parser("gc-job", help="clean transient artifacts for one job")
    gc.add_argument("--launcher-module", type=Path, default=None)
    gc.add_argument("--queue-root", type=Path, required=True)
    gc.add_argument("--wave-name", type=str, required=True)
    gc.add_argument("--job-json", type=Path, required=True)

    args = parser.parse_args()

    if args.cmd in {"emit", "launch"}:
        jobs = load_jobs(args.manifest)
        slots = [QueueSlot(gpu=gpu) for gpu in args.gpu_slots]
        script_paths = write_queue_files(
            jobs=jobs,
            slots=slots,
            queue_root=args.queue_root,
            wave_name=args.wave_name,
            launcher_module_path=args.launcher_module,
        )
        print(json.dumps({key: str(value) for key, value in script_paths.items()}, indent=2))
        if args.cmd == "launch":
            launch_slot_scripts(script_paths=script_paths, queue_root=args.queue_root, wave_name=args.wave_name)
        return

    if args.cmd == "run-one":
        raise SystemExit(
            run_one_job(
                launcher_module_path=args.launcher_module,
                queue_root=args.queue_root,
                wave_name=args.wave_name,
                slot_gpu=args.slot_gpu,
                job_json=args.job_json,
            )
        )
    if args.cmd == "postprocess-job":
        postprocess_job(
            launcher_module_path=args.launcher_module,
            queue_root=args.queue_root,
            wave_name=args.wave_name,
            slot_gpu=args.slot_gpu,
            job_json=args.job_json,
        )
        return
    gc_job(
        queue_root=args.queue_root,
        wave_name=args.wave_name,
        job_json=args.job_json,
        launcher_module_path=args.launcher_module,
    )


if __name__ == "__main__":
    main()
