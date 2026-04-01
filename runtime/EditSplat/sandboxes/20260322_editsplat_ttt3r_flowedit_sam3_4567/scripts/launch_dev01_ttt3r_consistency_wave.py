#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


ROOT = Path(
    "/dev_vepfs/rc_wu/edit/EditSplat/sandboxes/20260322_editsplat_ttt3r_flowedit_sam3_4567"
).resolve()
WRAPPER = ROOT / "scripts" / "run_sd35_ttt3r_sam3_wrapper.py"
PYTHON = Path("/dev_vepfs/rc_wu/envs/editsplat_multimodel_v2/bin/python").resolve()
DATASET_FACE = Path("/dev_vepfs/rc_wu/_codex_staging/20260401_a_casebank_dev01/dataset/dataset/face").resolve()
SOURCE_CKPT = (ROOT / "runtime" / "compat_pretrained_face" / "chkpnt7004.pth").resolve()
HF_HOME = Path("/dev_vepfs/rc_wu/cache/hf_home_dev02").resolve()
HF_TOKEN = Path("/dev_vepfs/rc_wu/.huggingface/token").resolve()
SAM3_PT = (
    HF_HOME
    / "hub"
    / "models--facebook--sam3"
    / "snapshots"
    / "3c879f39826c281e95690f02c7821c4de09afae7"
    / "sam3.pt"
).resolve()
LOG_DIR = ROOT / "logs"
RESULTS_DIR = ROOT / "results"
SUMMARY_DIR = ROOT / "results" / "summaries"


@dataclass
class Experiment:
    name: str
    gpu: int
    ttt3r_mode: str
    conf_power: float
    conf_floor: float
    prox_strength: float
    preserve_strength: float
    edit_boost: float
    preserve_boost: float
    adaptive_max_scale: float
    schedule_power: float
    support_views: int = 2
    support_stride: int = 1
    include_gt_view: bool = True
    fit_loss_mask_mode: str = "initial_edit"
    fit_loss_mask_bg: float = 0.03
    optimizer_lr_scale: float = 0.48
    max_optimizer_steps: int = 150
    disable_densify: bool = True
    freeze_geometry: bool = True
    freeze_opacity: bool = False
    head_k: int = 6
    depth_mode: str = "constant"
    max_train_views: int = 2
    max_gaussians: int = 60000
    flow_steps: int = 24
    flow_src_guidance_scale: float = 3.6
    flow_tar_guidance_scale: float = 6.9
    flow_n_max: int = 11
    flow_seed: int = 211
    text_guidance_scale: float = 6.6
    image_guidance_scale: float = 1.7
    source_guidance_scale: float = 1.5
    mask_bg: float = 0.18
    case_name: str = "clown"
    flow_src_prompt: str = "a photo of a young man"
    target_prompt: str = (
        "the same man in the same pose and camera framing, same background and clothes, "
        "with clear clown makeup: white face paint, a red clown nose, and colorful face paint"
    )
    sampling_prompt: str = "the same man with clown makeup, same framing and identity"
    object_prompt: str = "face"
    target_mask_prompt: str = "face"
    mask_backend: str = "sam3"
    dump_intermediates: bool = True
    notes: str = ""


def build_wave() -> List[Experiment]:
    return [
        Experiment(
            name="support_anchor",
            gpu=4,
            ttt3r_mode="velocity",
            conf_power=1.0,
            conf_floor=0.0,
            prox_strength=0.0,
            preserve_strength=0.0,
            edit_boost=1.0,
            preserve_boost=1.0,
            adaptive_max_scale=3.0,
            schedule_power=2.0,
            notes="support-only anchor; no TTT3R correction inside solver",
        ),
        Experiment(
            name="confidence_solver_velocity",
            gpu=5,
            ttt3r_mode="velocity",
            conf_power=1.12,
            conf_floor=0.08,
            prox_strength=0.32,
            preserve_strength=0.08,
            edit_boost=1.10,
            preserve_boost=1.04,
            adaptive_max_scale=2.5,
            schedule_power=1.7,
            notes="confidence-guided solver reweighting; mainline candidate",
        ),
        Experiment(
            name="static_proxy_baseline",
            gpu=6,
            ttt3r_mode="static_proxy",
            conf_power=1.08,
            conf_floor=0.05,
            prox_strength=0.0,
            preserve_strength=0.0,
            edit_boost=1.0,
            preserve_boost=1.0,
            adaptive_max_scale=2.5,
            schedule_power=1.7,
            notes="proxy-image baseline approximating warp/fuse teacher without solver coupling",
        ),
    ]


def ensure_layout() -> None:
    for path in (LOG_DIR, RESULTS_DIR, SUMMARY_DIR):
        path.mkdir(parents=True, exist_ok=True)


def build_run_name(exp: Experiment, wave_name: str) -> str:
    return f"{wave_name}_{exp.case_name}_{exp.name}"


def build_command(exp: Experiment, wave_name: str) -> List[str]:
    run_name = build_run_name(exp, wave_name)
    model_path = RESULTS_DIR / run_name
    cmd = [
        str(PYTHON),
        str(WRAPPER),
        "--model_key",
        "sd35-medium-turbo-open",
        "--hf_home",
        str(HF_HOME),
        "--adapter_gpu",
        "0",
        "--base_gpu",
        "0",
        "--head_k",
        str(exp.head_k),
        "--depth_mode",
        str(exp.depth_mode),
        "--skip_agt",
        "--ttt3r_repo_root",
        "/dev_vepfs/rc_wu/edit/TTT3R",
        "--ttt3r_checkpoint",
        "/dev_vepfs/rc_wu/edit/TTT3R/src/cut3r_512_dpt_4_64.pth",
        "--ttt3r_support_views",
        str(exp.support_views),
        "--ttt3r_support_stride",
        str(exp.support_stride),
        "--ttt3r_conf_power",
        str(exp.conf_power),
        "--ttt3r_conf_floor",
        str(exp.conf_floor),
        "--ttt3r_geo_scale",
        "1.0",
        "--ttt3r_prox_strength",
        str(exp.prox_strength),
        "--ttt3r_preserve_strength",
        str(exp.preserve_strength),
        "--ttt3r_edit_boost",
        str(exp.edit_boost),
        "--ttt3r_preserve_boost",
        str(exp.preserve_boost),
        "--ttt3r_edit_min_mass",
        "0.02",
        "--ttt3r_preserve_min_mass",
        "0.02",
        "--ttt3r_adaptive_max_scale",
        str(exp.adaptive_max_scale),
        "--ttt3r_schedule_power",
        str(exp.schedule_power),
        "--ttt3r_mode",
        str(exp.ttt3r_mode),
        "--ttt3r_gpu",
        "-1",
        "--optimizer_lr_scale",
        str(exp.optimizer_lr_scale),
        "--max_optimizer_steps",
        str(exp.max_optimizer_steps),
        "--fit_loss_mask_mode",
        str(exp.fit_loss_mask_mode),
        "--fit_loss_mask_bg",
        str(exp.fit_loss_mask_bg),
        "--dump_intermediates",
        "--disable_densify",
        "--freeze_geometry",
        "-s",
        str(DATASET_FACE),
        "-m",
        str(model_path),
        "--source_checkpoint",
        str(SOURCE_CKPT),
        "--resolution",
        "8",
        "--epoch",
        "2",
        "--flow_model_key",
        "sd35-medium-turbo-open",
        "--flow_hf_home",
        str(HF_HOME),
        "--flow_method",
        "flowedit",
        "--object_prompt",
        str(exp.object_prompt),
        "--target_mask_prompt",
        str(exp.target_mask_prompt),
        "--target_prompt",
        str(exp.target_prompt),
        "--sampling_prompt",
        str(exp.sampling_prompt),
        "--flow_src_prompt",
        str(exp.flow_src_prompt),
        "--flow_tar_prompt",
        str(exp.target_prompt),
        "--flow_steps",
        str(exp.flow_steps),
        "--flow_n_avg",
        "1",
        "--flow_src_guidance_scale",
        str(exp.flow_src_guidance_scale),
        "--flow_tar_guidance_scale",
        str(exp.flow_tar_guidance_scale),
        "--flow_n_min",
        "0",
        "--flow_n_max",
        str(exp.flow_n_max),
        "--flow_seed",
        str(exp.flow_seed),
        "--text_guidance_scale",
        str(exp.text_guidance_scale),
        "--image_guidance_scale",
        str(exp.image_guidance_scale),
        "--source_guidance_scale",
        str(exp.source_guidance_scale),
        "--filtering_ratio",
        "0.85",
        "--mask_bg",
        str(exp.mask_bg),
    ]
    if exp.include_gt_view:
        cmd.append("--ttt3r_include_gt_view")
    else:
        cmd.append("--ttt3r_no_include_gt_view")
    return cmd


def launch_one(exp: Experiment, wave_name: str) -> Dict[str, object]:
    run_name = build_run_name(exp, wave_name)
    log_path = LOG_DIR / f"{run_name}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(exp.gpu)
    env["HF_HOME"] = str(HF_HOME)
    env["HF_HUB_CACHE"] = str(HF_HOME / "hub")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["EDITSPLAT_HF_HOME"] = str(HF_HOME)
    env["EDITSPLAT_HF_TOKEN_FILE"] = str(HF_TOKEN)
    env["EDITSPLAT_SAM3_CHECKPOINT_PATH"] = str(SAM3_PT)
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

    cmd = build_command(exp=exp, wave_name=wave_name)
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    return {
        "run_name": run_name,
        "gpu": exp.gpu,
        "pid": proc.pid,
        "log_path": str(log_path),
        "model_path": str(RESULTS_DIR / run_name),
        "experiment": asdict(exp),
    }


def collect_summary(runs: List[Dict[str, object]], wave_name: str) -> Path:
    rows: List[Dict[str, object]] = []
    for item in runs:
        model_path = Path(str(item["model_path"]))
        meta_path = model_path / "ttt3r_proximal_wrapper_meta.json"
        mask_meta_path = model_path / "mask_backend_info.json"
        mfg_stats_path = model_path / "debug_intermediates" / "mfg_edit" / "view000" / "stats.json"
        row: Dict[str, object] = {
            "run_name": item["run_name"],
            "gpu": item["gpu"],
            "log_path": item["log_path"],
            "model_path": item["model_path"],
            "meta_exists": meta_path.exists(),
            "mask_meta_exists": mask_meta_path.exists(),
            "mfg_stats_exists": mfg_stats_path.exists(),
        }
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            ttt3r_cfg = meta.get("ttt3r_cfg", {}) if isinstance(meta, dict) else {}
            for key in (
                "mode",
                "conf_power",
                "conf_floor",
                "prox_strength",
                "preserve_strength",
                "edit_boost",
                "preserve_boost",
                "adaptive_max_scale",
                "schedule_power",
            ):
                row[key] = ttt3r_cfg.get(key)
            for key in ("head_k", "fit_loss_mask_mode", "fit_loss_mask_bg"):
                row[key] = meta.get(key)
        if mask_meta_path.exists():
            mask_meta = json.loads(mask_meta_path.read_text(encoding="utf-8"))
            if isinstance(mask_meta, dict):
                row["mask_backend_requested"] = mask_meta.get("requested")
                row["mask_backend_effective"] = mask_meta.get("effective")
                row["mask_backend_detail"] = mask_meta.get("detail")
        if mfg_stats_path.exists():
            mfg_stats = json.loads(mfg_stats_path.read_text(encoding="utf-8"))
            for key in ("proxy_rgb", "geo_weight", "edit_weight", "preserve_weight"):
                if key in mfg_stats and isinstance(mfg_stats[key], dict):
                    row[f"{key}_mean"] = mfg_stats[key].get("mean")
        rows.append(row)

    out_path = SUMMARY_DIR / f"{wave_name}_summary.json"
    out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a dev01 TTT3R consistency wave")
    parser.add_argument("--wave-name", type=str, default=f"20260401_ttt3r_wave_{utc_stamp()}")
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--sleep-before-collect", type=int, default=0)
    args = parser.parse_args()

    ensure_layout()
    experiments = build_wave()
    if args.print_only:
        payload = []
        for exp in experiments:
            payload.append(
                {
                    "run_name": build_run_name(exp, args.wave_name),
                    "gpu": exp.gpu,
                    "command": build_command(exp, args.wave_name),
                    "experiment": asdict(exp),
                }
            )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    runs: List[Dict[str, object]] = []
    if not args.collect_only:
        for exp in experiments:
            launched = launch_one(exp=exp, wave_name=args.wave_name)
            runs.append(launched)
        launch_path = SUMMARY_DIR / f"{args.wave_name}_launch.json"
        launch_path.write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")
        print(launch_path)
    else:
        launch_path = SUMMARY_DIR / f"{args.wave_name}_launch.json"
        runs = json.loads(launch_path.read_text(encoding="utf-8"))

    if args.sleep_before_collect > 0:
        time.sleep(int(args.sleep_before_collect))
    summary_path = collect_summary(runs=runs, wave_name=args.wave_name)
    print(summary_path)


if __name__ == "__main__":
    main()
