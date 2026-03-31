#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


SANDBOX_ROOT = Path(
    "/dev_vepfs/rc_wu/edit/EditSplat/sandboxes/20260322_editsplat_ttt3r_flowedit_sam3_4567"
).resolve()
PROJECT_ROOT = Path("/dev_vepfs/rc_wu/edit/EditSplat").resolve()
WRAPPER = SANDBOX_ROOT / "scripts" / "run_sd35_ttt3r_sam3_wrapper.py"
PYTHON = Path("/dev_vepfs/rc_wu/envs/editsplat_multimodel_v2/bin/python")
STATUS_PATH = SANDBOX_ROOT / "status" / "aris_dev02_0123_status.json"
LOG_DIR = SANDBOX_ROOT / "logs"
RESULTS_DIR = SANDBOX_ROOT / "results"
RUNTIME_DIR = SANDBOX_ROOT / "runtime"

SHARED_CKPT = Path("/dev_vepfs/rc_wu/cache/models/ttt3r/cut3r_512_dpt_4_64.pth").resolve()
EXPECTED_CKPT_LINK = Path("/dev_vepfs/rc_wu/edit/TTT3R/src/cut3r_512_dpt_4_64.pth").resolve()
CKPT_MIN_BYTES = 3_100_000_000

DATASET_SOURCE = Path("/dev_vepfs/rc_wu/edit/EditSplat/dataset/dataset/face").resolve()
# Workspace compatibility: legacy dataset/pretrained/face checkpoint path is absent here.
# Use a known-good local baseline checkpoint that has matching point_cloud assets.
SOURCE_CKPT = Path(
    "/dev_vepfs/rc_wu/edit/EditSplat/sandboxes/20260322_editsplat_ttt3r_flowedit_sam3_4567/"
    "runtime/compat_pretrained_face/chkpnt7004.pth"
)
HF_HOME = Path("/dev_vepfs/rc_wu/cache/hf_home_dev02").resolve()
TORCH_HOME = Path("/dev_vepfs/rc_wu/cache/torch").resolve()
XDG_CACHE_HOME = Path("/dev_vepfs/rc_wu/cache/xdg").resolve()
BASE_MODEL_ID = "cocktailpeanut/xulf-s"
BASE_MODEL_CACHE_ROOT = HF_HOME / "hub" / "models--cocktailpeanut--xulf-s"

# Official sd35-large is gated in this environment. Keep SD3.5 family with open teacher.
TEACHER_MODEL_KEY = "sd35-medium-turbo-open"
TEACHER_MODEL_ID = "tensorart/stable-diffusion-3.5-medium-turbo"
TEACHER_MODEL_CACHE_ROOT = HF_HOME / "hub" / "models--tensorart--stable-diffusion-3.5-medium-turbo"

ALLOWED_GPUS = (0, 1, 2, 3)
RUN_STAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
# Small deterministic schedule knob so each keepalive cycle explores a nearby
# hyperparameter variant instead of repeating an identical plateau.
RUN_VARIANT = int(RUN_STAMP[-2:]) % 3
# dev-intern-02 currently has heavy resident allocations on GPUs 0-3 from other workloads.
# Keep SD3.5 teacher on-GPU but force TTT3R runtime to CPU to avoid repeated adapter OOM.
TTT3R_GPU_OVERRIDE = -1

FLOW_SRC_PROMPT = "a photo of a young man"
CASE_PROMPTS = {
    "clown": {
        "target": (
            "the same man in the same pose and camera framing, same background and clothes, "
            "with clear clown makeup: white face paint, a red clown nose, and colorful face paint"
        ),
        "sampling": "the same man with clown makeup, same framing and identity",
    },
    "beard": {
        "target": (
            "the same man in the same pose and framing, with a clearly visible short dark beard and moustache, "
            "while preserving identity, hair, clothes, and background"
        ),
        "sampling": "the same man with a short dark beard and moustache, same framing",
    },
    "glasses": {
        "target": (
            "the same man in the same pose and framing, clearly wearing black-rimmed eyeglasses, "
            "while preserving identity, face shape, clothes, and background"
        ),
        "sampling": "the same man wearing black-rimmed eyeglasses, same framing",
    },
}

DIFFGS_PROBE_FLAG = SANDBOX_ROOT / "runtime" / "enable_diffgs_probe"
COMPUTE_SANITIZER_BIN = Path("/usr/local/cuda-12.4/bin/compute-sanitizer")


@dataclass
class Experiment:
    name: str
    gpu: int
    case: str
    mfg_mode: str
    mask_backend: str
    hf_offline: str
    ttt3r_mode: str
    conf_power: float
    conf_floor: float
    prox_strength: float
    preserve_strength: float
    optimizer_lr_scale: float
    max_optimizer_steps: int
    epoch: int
    k_percent: float
    mask_bg: float
    flow_steps: int = 24
    flow_src_guidance_scale: float = 3.2
    flow_tar_guidance_scale: float = 7.0
    flow_n_max: int = 10
    flow_seed: int = 2
    flow_adapter_resize_side: int = 512
    text_guidance_scale: float = 6.5
    image_guidance_scale: float = 1.8
    source_guidance_scale: float = 1.2
    fit_loss_mask_mode: str = "initial_edit"
    fit_loss_mask_bg: float = 0.20
    fit_view_topk: int = -1
    ttt3r_edit_boost: float = 1.0
    ttt3r_preserve_boost: float = 1.0
    ttt3r_edit_min_mass: float = 0.0
    ttt3r_preserve_min_mass: float = 0.0
    ttt3r_adaptive_max_scale: float = 3.0
    ttt3r_schedule_power: float = 2.0
    elite_conf_correction: bool = False
    elite_support_alpha: float = 0.35
    elite_edit_alpha: float = 0.35
    elite_confidence_alpha: float = 0.45
    elite_scale_min: float = 0.0
    elite_scale_max: float = 1.0
    blite_canonical_prior: bool = False
    blite_canonical_dump: bool = True
    disable_densify: bool = False
    freeze_geometry: bool = False
    freeze_opacity: bool = False
    max_train_views: int = 2
    max_gaussians: int = 120000
    resolution: int = -1
    cuda_launch_blocking: bool = False
    real_lpips: bool = True
    skip_3dgs_guard: bool = True


EXPERIMENTS_BY_GPU: Dict[int, List[Experiment]] = {
    # Support-side queue: keep a conservative anchor plus a stronger update-law-only probe.
    0: [
        Experiment(
            name="clown_support_prior_strong",
            gpu=0,
            case="clown",
            mfg_mode="initial_only",
            mask_backend="langsam",
            hf_offline="1",
            ttt3r_mode="proximal",
            conf_power=1.0,
            conf_floor=0.0,
            prox_strength=0.0,
            preserve_strength=0.0,
            optimizer_lr_scale=1.0,
            max_optimizer_steps=-1,
            epoch=2,
            k_percent=0.0,
            mask_bg=0.24,
            flow_steps=24,
            flow_src_guidance_scale=3.6,
            flow_tar_guidance_scale=7.4,
            flow_n_max=11,
            text_guidance_scale=6.8,
            image_guidance_scale=1.95,
            source_guidance_scale=1.45,
            flow_seed=73,
            fit_loss_mask_mode="none",
            # Keep this branch stable first; L1 proxy avoids frequent LPIPS-driven CUDA faults.
            real_lpips=False,
        ),
        Experiment(
            name="clown_support_consistency_gsopt",
            gpu=0,
            case="clown",
            mfg_mode="full",
            mask_backend="langsam",
            hf_offline="1",
            # Keep confidence correction off to isolate optimization/update-law effects.
            ttt3r_mode="velocity",
            conf_power=1.0,
            conf_floor=0.0,
            prox_strength=0.0,
            preserve_strength=0.0,
            optimizer_lr_scale=0.48,
            max_optimizer_steps=150,
            epoch=3,
            k_percent=0.0,
            mask_bg=0.18,
            flow_steps=24,
            flow_src_guidance_scale=3.6,
            flow_tar_guidance_scale=6.75,
            flow_n_max=11,
            text_guidance_scale=6.60,
            image_guidance_scale=1.68,
            source_guidance_scale=1.52,
            flow_seed=113,
            fit_loss_mask_mode="initial_edit",
            fit_loss_mask_bg=0.025,
            disable_densify=True,
            freeze_geometry=True,
            freeze_opacity=False,
            max_train_views=2,
            max_gaussians=60000,
            cuda_launch_blocking=False,
            real_lpips=False,
        ),
    ],
    # Reliability prior branch: confidence as precision signal; keep this branch as the benchmark for TTT3R usefulness.
    1: [
        Experiment(
            name="clown_reliability_precision_mild",
            gpu=1,
            case="clown",
            mfg_mode="full",
            mask_backend="langsam",
            hf_offline="1",
            ttt3r_mode="proximal",
            conf_power=1.08,
            conf_floor=0.06,
            prox_strength=0.28,
            preserve_strength=0.06,
            optimizer_lr_scale=1.0,
            max_optimizer_steps=-1,
            epoch=2,
            k_percent=0.0,
            mask_bg=0.24,
            flow_steps=24,
            flow_src_guidance_scale=3.6,
            flow_tar_guidance_scale=7.35,
            flow_n_max=11,
            text_guidance_scale=7.0,
            image_guidance_scale=1.90,
            source_guidance_scale=1.45,
            flow_seed=79,
            fit_loss_mask_bg=0.06,
            ttt3r_edit_boost=1.06,
            ttt3r_preserve_boost=1.02,
            ttt3r_edit_min_mass=0.02,
            ttt3r_preserve_min_mass=0.02,
            ttt3r_adaptive_max_scale=2.6,
            ttt3r_schedule_power=1.8,
            real_lpips=False,
        ),
    ],
    # 3DGS optimization/update-law branch: no TTT3R confidence correction, tuned for stronger-but-stable support-only edits.
    2: [
        Experiment(
            name="clown_support_app_balanced_gsopt",
            gpu=2,
            case="clown",
            mfg_mode="full",
            mask_backend="langsam",
            hf_offline="1",
            # Keep confidence correction off to test pure optimization/update-law gains.
            ttt3r_mode="velocity",
            conf_power=1.0,
            conf_floor=0.0,
            prox_strength=0.0,
            preserve_strength=0.0,
            optimizer_lr_scale=0.50,
            max_optimizer_steps=150,
            epoch=3,
            k_percent=0.0,
            mask_bg=0.18,
            flow_steps=24,
            flow_src_guidance_scale=3.6,
            flow_tar_guidance_scale=6.85,
            flow_n_max=11,
            text_guidance_scale=6.60,
            image_guidance_scale=1.70,
            source_guidance_scale=1.52,
            flow_seed=97,
            fit_loss_mask_mode="initial_edit",
            fit_loss_mask_bg=0.025,
            ttt3r_edit_boost=1.0,
            ttt3r_preserve_boost=1.0,
            ttt3r_edit_min_mass=0.0,
            ttt3r_preserve_min_mass=0.0,
            ttt3r_adaptive_max_scale=3.0,
            ttt3r_schedule_power=2.0,
            disable_densify=True,
            freeze_geometry=True,
            freeze_opacity=False,
            max_train_views=2,
            max_gaussians=60000,
            cuda_launch_blocking=False,
            # Avoid LPIPS CUDA-state instability in this branch while we tune update-law behavior.
            real_lpips=False,
        ),
    ],
    # Dedicated SAM repair branch mirrors reliability settings for clean backend-comparison evidence.
    3: [
        Experiment(
            name="clown_sam3_repair_precision_mild",
            gpu=3,
            case="clown",
            mfg_mode="full",
            mask_backend="sam3",
            hf_offline="0",
            ttt3r_mode="proximal",
            conf_power=1.08,
            conf_floor=0.06,
            prox_strength=0.28,
            preserve_strength=0.06,
            optimizer_lr_scale=1.0,
            max_optimizer_steps=-1,
            epoch=2,
            k_percent=0.0,
            mask_bg=0.24,
            flow_steps=24,
            flow_src_guidance_scale=3.6,
            flow_tar_guidance_scale=7.35,
            flow_n_max=11,
            text_guidance_scale=7.0,
            image_guidance_scale=1.90,
            source_guidance_scale=1.45,
            flow_seed=79,
            fit_loss_mask_bg=0.06,
            ttt3r_edit_boost=1.06,
            ttt3r_preserve_boost=1.02,
            ttt3r_edit_min_mass=0.02,
            ttt3r_preserve_min_mass=0.02,
            ttt3r_adaptive_max_scale=2.6,
            ttt3r_schedule_power=1.8,
            real_lpips=False,
        ),
    ],
}

if DIFFGS_PROBE_FLAG.exists():
    print("[DIAG] diff_gaussian probe enabled for GPU2 queue.")
    EXPERIMENTS_BY_GPU.setdefault(2, [])
    EXPERIMENTS_BY_GPU[2].insert(
        0,
        Experiment(
            name="probe_diffgs_illegal_mem",
            gpu=2,
            case="clown",
            mfg_mode="initial_only",
            mask_backend="langsam",
            hf_offline="1",
            ttt3r_mode="velocity",
            conf_power=1.0,
            conf_floor=0.0,
            prox_strength=0.0,
            preserve_strength=0.0,
            optimizer_lr_scale=0.35,
            max_optimizer_steps=30,
            epoch=1,
            k_percent=0.0,
            mask_bg=0.28,
            flow_steps=6,
            flow_src_guidance_scale=3.0,
            flow_tar_guidance_scale=6.0,
            flow_n_max=6,
            text_guidance_scale=6.0,
            image_guidance_scale=1.5,
            source_guidance_scale=1.2,
            flow_seed=17,
            flow_adapter_resize_side=384,
            fit_loss_mask_mode="initial_edit",
            fit_loss_mask_bg=0.2,
            disable_densify=True,
            freeze_geometry=True,
            freeze_opacity=True,
            max_train_views=1,
            max_gaussians=4000,
            resolution=384,
            cuda_launch_blocking=True,
            real_lpips=False,
            skip_3dgs_guard=False,
        ),
    )
    try:
        DIFFGS_PROBE_FLAG.unlink()
    except FileNotFoundError:
        pass


def _align_sam3_repair_mirror() -> None:
    reliability = None
    sam_repair = None
    for exp in EXPERIMENTS_BY_GPU.get(1, []):
        if exp.name == "clown_reliability_precision_mild":
            reliability = exp
            break
    for exp in EXPERIMENTS_BY_GPU.get(3, []):
        if exp.name == "clown_sam3_repair_precision_mild":
            sam_repair = exp
            break
    if reliability is None or sam_repair is None:
        return

    # Keep backend/offline controls separate, but mirror optimization knobs for fair backend comparisons.
    mirror_fields = (
        "mfg_mode",
        "ttt3r_mode",
        "conf_power",
        "conf_floor",
        "prox_strength",
        "preserve_strength",
        "optimizer_lr_scale",
        "max_optimizer_steps",
        "epoch",
        "k_percent",
        "mask_bg",
        "flow_steps",
        "flow_src_guidance_scale",
        "flow_tar_guidance_scale",
        "flow_n_max",
        "flow_seed",
        "text_guidance_scale",
        "image_guidance_scale",
        "source_guidance_scale",
        "fit_loss_mask_mode",
        "fit_loss_mask_bg",
        "fit_view_topk",
        "ttt3r_edit_boost",
        "ttt3r_preserve_boost",
        "ttt3r_edit_min_mass",
        "ttt3r_preserve_min_mass",
        "ttt3r_adaptive_max_scale",
        "ttt3r_schedule_power",
        "disable_densify",
        "freeze_geometry",
        "freeze_opacity",
        "max_train_views",
        "max_gaussians",
        "cuda_launch_blocking",
        "real_lpips",
    )
    for field in mirror_fields:
        setattr(sam_repair, field, getattr(reliability, field))


def _apply_cycle_variant() -> None:
    seed_bump = RUN_VARIANT * 17
    for queue in EXPERIMENTS_BY_GPU.values():
        for exp in queue:
            exp.flow_seed += seed_bump
            if "reliability_precision" in exp.name or "sam3_repair_precision" in exp.name:
                exp.conf_floor = max(0.04, exp.conf_floor - 0.005 * RUN_VARIANT)
                exp.prox_strength = min(0.40, exp.prox_strength + 0.01 * RUN_VARIANT)
                exp.flow_tar_guidance_scale += 0.06 * RUN_VARIANT
                exp.text_guidance_scale += 0.04 * RUN_VARIANT
            if "reliability_velocity_gsopt" in exp.name:
                exp.prox_strength = min(0.45, exp.prox_strength + 0.02 * RUN_VARIANT)
                exp.ttt3r_edit_boost = min(1.26, exp.ttt3r_edit_boost + 0.02 * RUN_VARIANT)
                exp.optimizer_lr_scale = max(0.58, exp.optimizer_lr_scale - 0.02 * RUN_VARIANT)
            if "support_app_balanced_gsopt" in exp.name:
                exp.flow_tar_guidance_scale += 0.02 * RUN_VARIANT
                exp.text_guidance_scale += 0.015 * RUN_VARIANT
                exp.optimizer_lr_scale = max(0.46, exp.optimizer_lr_scale - 0.01 * RUN_VARIANT)
            if "support_consistency_gsopt" in exp.name:
                exp.flow_tar_guidance_scale += 0.015 * RUN_VARIANT
                exp.text_guidance_scale += 0.01 * RUN_VARIANT
                exp.optimizer_lr_scale = max(0.44, exp.optimizer_lr_scale - 0.008 * RUN_VARIANT)
                exp.max_gaussians = max(52000, exp.max_gaussians - 3000 * RUN_VARIANT)


def _inject_elite_blite_optin_experiments() -> None:
    if _env_flag("EDITSPLAT_ENABLE_ELITE_EXPERIMENTS", False):
        base = next((exp for exp in EXPERIMENTS_BY_GPU.get(1, []) if exp.name == "clown_reliability_precision_mild"), None)
        if base is not None:
            EXPERIMENTS_BY_GPU.setdefault(1, []).append(
                replace(
                    base,
                    name=f"{base.name}_elite_optin",
                    elite_conf_correction=True,
                    elite_support_alpha=0.45,
                    elite_edit_alpha=0.35,
                    elite_confidence_alpha=0.55,
                    elite_scale_min=0.10,
                    elite_scale_max=1.00,
                )
            )

    if _env_flag("EDITSPLAT_ENABLE_BLITE_EXPERIMENTS", False):
        base = next((exp for exp in EXPERIMENTS_BY_GPU.get(2, []) if exp.name == "clown_support_app_balanced_gsopt"), None)
        if base is not None:
            EXPERIMENTS_BY_GPU.setdefault(2, []).append(
                replace(
                    base,
                    name=f"{base.name}_blite_prior_optin",
                    blite_canonical_prior=True,
                    blite_canonical_dump=True,
                )
            )


_align_sam3_repair_mirror()
_apply_cycle_variant()
_inject_elite_blite_optin_experiments()


class State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.payload: Dict[str, object] = {
            "line": "EditSplat-TTT3R-dev02-0123",
            "sandbox": str(SANDBOX_ROOT),
            "allowed_gpus": list(ALLOWED_GPUS),
            "run_stamp": RUN_STAMP,
            "phase": "init",
            "updated_at_utc": utc_now(),
            "checkpoint": {
                "path": str(SHARED_CKPT),
                "expected_link": str(EXPECTED_CKPT_LINK),
                "min_bytes": CKPT_MIN_BYTES,
                "ready": False,
                "size_bytes": 0,
            },
            "base_model": {
                "id": BASE_MODEL_ID,
                "cache_root": str(BASE_MODEL_CACHE_ROOT),
                "ready": False,
                "incomplete_shards": -1,
            },
            "teacher_model": {
                "key": TEACHER_MODEL_KEY,
                "id": TEACHER_MODEL_ID,
                "cache_root": str(TEACHER_MODEL_CACHE_ROOT),
                "ready": False,
                "incomplete_shards": -1,
            },
            "workers": {},
            "notes": [
                "2026-03-24 22:05 UTC: keepalive/supervisor relaunched inside tmux session dev02_0123_keepalive so GPUs 0-3 keep running even when the Codex shell disconnects.",
                "SAM3 mask fusion patch is live: the wrapper caches every SAM/LangSAM mask per view and the patched _fit_mask_from_runtime merges that support into the fit_loss mask; expect new runs to log sam_support_meta entries under debug_intermediates/fit_masks.",
                "diff_gaussian_rasterization rebuild with TORCH_USE_CUDA_DSA=1 failed because nvidia-cuda-nvcc-cu11 installs nvcc as a symlink to /root/.../nvcc (permission denied); need a readable CUDA 11.8 toolkit path before rerunning the DSA probe.",
                "Active line uses GPUs [0,1,2,3] only.",
                "Current frontier is clown-only and hard-case-first; beard/glasses are temporarily paused to speed hard-case iteration.",
                "GPU0 runs a two-stage support queue: conservative support-prior baseline + support-only GS consistency probe.",
                "GPU1 keeps TTT3R only as a mild reliability / precision prior; if it cannot beat support-only visually, it should not stay on the main path.",
                "GPU2 runs a support-only balanced GS optimization branch (no TTT3R confidence correction) to prioritize stable visual quality gains.",
                "GPU3 is the dedicated SAM repair branch and is auto-mirrored to GPU1 parameters for clean backend-only comparison.",
                "SAM repair branch keeps requesting sam3; when unavailable it records fallback:langsam:cpu evidence.",
                "Keepalive/supervisor PIDs refreshed (2356979/836355); only the 20260322 sandbox is executing and the older 20260319 line remains idle.",
                f"Cycle variant={RUN_VARIANT} applies small seed/hyperparameter offsets to avoid deterministic plateaus.",
                "Legacy wrapper now forces FLOWEDIT_REAL_LANGSAM=1 and keeps HF/TORCH/XDG caches under /dev_vepfs/rc_wu.",
                "Known blocker: diff_gaussian_rasterization backward can hit illegal memory access.",
                "3DGS illegal-memory guard is enabled via EDITSPLAT_SKIP_3DGS_BACKWARD_ON_ERROR=1 to keep runs alive.",
            "Current frontier prioritizes visual identity-preserving edits on hard cases over scalar-only wins.",
            "Inspect logs for LangSAM predict fallback-to-stub events; mask_backend_info alone can miss this runtime degradation.",
            "Wrapper now attempts LangSAM prompt fallbacks (person/head/portrait) before full-image stub; check mask_backend_info detail for langsam_prompt_fallback.",
                "Latest judged panel is 20260324_200102: support_consistency_gsopt remains the cleanest branch, support_app_balanced_gsopt keeps a cyan jaw leak, and the SAM repair branch is still tied to reliability even though SAM3 is active.",
                "SAM instrumentation now logs per-call `sam3_mask_stats`; the next fix is `_predict_langsam_mask` / mask-fusion so SAM-only runs can diverge from TTT3R.",
                "High-priority diag: drop runtime/enable_diffgs_probe before the next restart so GPU2 runs probe_diffgs_illegal_mem (CUDA_LAUNCH_BLOCKING=1, ≤20k Gaussians, densify off) and captures the raw illegal-memory stack trace.",
                "2026-03-25 00:12 UTC: diff_gaussian_rasterization rebuilt with TORCH_USE_CUDA_DSA=1 (pip --no-binary) and runtime/enable_diffgs_probe touched so the next GPU2 cycle reruns the illegal-memory sentinel with the new wheel.",
                "2026-03-25 00:15 UTC: panel 20260324_235309 judged via chafa/panel_digest; support_consistency_gsopt remains the cleanest view000 branch, support_app_balanced_gsopt still leaks cyan on view001, and SAM vs reliability now diverge only in mask stats (geo_weight ≈0.30 vs 0.0003) while render sets stay disabled.",
                "2026-03-25 00:35 UTC: staged a sandbox-local diff_gaussian_rasterization DSA build under runtime/diff_gaussian_rasterization_dsa, updated probe_diffgs_illegal_* to prepend that path to PYTHONPATH, and logged panel 20260325_001315 (chafa + panel_digest) showing support_consistency_gsopt still cleanest while SAM vs reliability diverge only in mask stats (effective=sam3, geo_weight ≈0.30 vs 3e-4).",
                "2026-03-25 02:26 UTC: diffGS probe 20260325_020829 still hit compute-sanitizer memory exhaustion and ended with the usual 'Compile with TORCH_USE_CUDA_DSA' footer, so the probe queue now runs at 384px adapter resize / 384 render resolution with 4k gaussians and runtime/enable_diffgs_probe was re-armed for the next GPU2 idle cycle.",
                "2026-03-25 02:27 UTC: panel 20260325_020829 (panel_digest + chafa) keeps support_consistency_gsopt as the cleanest branch despite support_app_balanced_gsopt winning view000 MAD; reliability_precision_mild and sam3_repair_branch remain pixel-identical because render sets stay disabled, but mask_backend_info now shows effective=sam3 with geo_weight≈0.30 while reliability is still ≈3e-4.",
            ],
        }

    def update(self, **kwargs: object) -> None:
        with self.lock:
            self.payload.update(kwargs)
            self.payload["updated_at_utc"] = utc_now()

    def patch_worker(self, worker_key: str, patch: Dict[str, object]) -> None:
        with self.lock:
            workers = self.payload.setdefault("workers", {})
            entry = workers.setdefault(worker_key, {})
            entry.update(patch)
            self.payload["updated_at_utc"] = utc_now()

    def write(self) -> None:
        with self.lock:
            body = json.dumps(self.payload, indent=2, ensure_ascii=False) + "\n"
            tmp = STATUS_PATH.with_suffix(".json.tmp")
            tmp.write_text(body, encoding="utf-8")
            tmp.replace(STATUS_PATH)


STATE = State()


def checkpoint_ready() -> bool:
    return SHARED_CKPT.is_file() and SHARED_CKPT.stat().st_size >= CKPT_MIN_BYTES


def model_cache_ready(cache_root: Path) -> bool:
    if not cache_root.is_dir():
        return False
    snapshots_root = cache_root / "snapshots"
    snapshots = list(snapshots_root.glob("*")) if snapshots_root.is_dir() else []
    if not snapshots:
        return False
    if not any((snap / "model_index.json").is_file() for snap in snapshots):
        return False
    blobs_root = cache_root / "blobs"
    if not blobs_root.is_dir():
        return False
    return len(list(blobs_root.glob("*.incomplete"))) == 0


def model_incomplete_count(cache_root: Path) -> int:
    blobs_root = cache_root / "blobs"
    if not blobs_root.is_dir():
        return -1
    return len(list(blobs_root.glob("*.incomplete")))


def ensure_checkpoint_link() -> None:
    EXPECTED_CKPT_LINK.parent.mkdir(parents=True, exist_ok=True)
    if EXPECTED_CKPT_LINK.is_symlink() or EXPECTED_CKPT_LINK.exists():
        try:
            if EXPECTED_CKPT_LINK.resolve() == SHARED_CKPT:
                return
        except Exception:
            pass
        EXPECTED_CKPT_LINK.unlink()
    EXPECTED_CKPT_LINK.symlink_to(SHARED_CKPT)


def build_command(exp: Experiment, run_name: str) -> List[str]:
    case = CASE_PROMPTS[exp.case]
    model_path = RESULTS_DIR / run_name
    # Each worker gets a single visible GPU (remapped as cuda:0 inside the process).
    local_gpu = 0
    return [
        str(PYTHON),
        str(WRAPPER),
        "--model_key",
        TEACHER_MODEL_KEY,
        "--hf_home",
        str(HF_HOME),
        "--base_gpu",
        str(local_gpu),
        "--adapter_gpu",
        str(local_gpu),
        "--head_k",
        "6",
        "--skip_agt",
        "--ttt3r_repo_root",
        "/dev_vepfs/rc_wu/edit/TTT3R",
        "--ttt3r_checkpoint",
        str(EXPECTED_CKPT_LINK),
        "--ttt3r_model_update_type",
        "ttt3r",
        "--ttt3r_support_views",
        "2",
        "--ttt3r_support_stride",
        "1",
        "--ttt3r_input_h",
        "256",
        "--ttt3r_input_w",
        "320",
        "--ttt3r_mode",
        exp.ttt3r_mode,
        "--ttt3r_gpu",
        str(TTT3R_GPU_OVERRIDE),
        "--ttt3r_conf_power",
        str(exp.conf_power),
        "--ttt3r_conf_floor",
        str(exp.conf_floor),
        "--ttt3r_prox_strength",
        str(exp.prox_strength),
        "--ttt3r_preserve_strength",
        str(exp.preserve_strength),
        "--ttt3r_edit_boost",
        str(exp.ttt3r_edit_boost),
        "--ttt3r_preserve_boost",
        str(exp.ttt3r_preserve_boost),
        "--ttt3r_edit_min_mass",
        str(exp.ttt3r_edit_min_mass),
        "--ttt3r_preserve_min_mass",
        str(exp.ttt3r_preserve_min_mass),
        "--ttt3r_adaptive_max_scale",
        str(exp.ttt3r_adaptive_max_scale),
        "--ttt3r_schedule_power",
        str(exp.ttt3r_schedule_power),
        "--ttt3r_edit_mask_quantile",
        "0.90",
        "--dump_intermediates",
        "--optimizer_lr_scale",
        str(exp.optimizer_lr_scale),
        "--max_optimizer_steps",
        str(exp.max_optimizer_steps),
        "--fit_loss_mask_mode",
        exp.fit_loss_mask_mode,
        "--fit_loss_mask_quantile",
        "0.75",
        "--fit_loss_mask_bg",
        str(exp.fit_loss_mask_bg),
        "--fit_view_topk",
        str(exp.fit_view_topk),
        "--depth_mode",
        "constant",
        "-s",
        str(DATASET_SOURCE),
        "-m",
        str(model_path),
        "--source_checkpoint",
        str(SOURCE_CKPT),
        "--flow_model_key",
        TEACHER_MODEL_KEY,
        "--flow_method",
        "flowedit",
        "--flow_hf_home",
        str(HF_HOME),
        "--flow_adapter_resize_side",
        str(exp.flow_adapter_resize_side),
        "--flow_adapter_gpu",
        str(local_gpu),
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
        "--target_prompt",
        case["target"],
        "--sampling_prompt",
        case["sampling"],
        "--object_prompt",
        "face",
        "--target_mask_prompt",
        "face",
        "--text_guidance_scale",
        str(exp.text_guidance_scale),
        "--image_guidance_scale",
        str(exp.image_guidance_scale),
        "--source_guidance_scale",
        str(exp.source_guidance_scale),
        "--filtering_ratio",
        "0.85",
        "--flow_src_prompt",
        FLOW_SRC_PROMPT,
        "--flow_tar_prompt",
        case["target"],
        "--epoch",
        str(exp.epoch),
        "--k_percent",
        str(exp.k_percent),
        "--mask_bg",
        str(exp.mask_bg),
        "--resolution",
        str(exp.resolution),
    ] + (["--disable_densify"] if exp.disable_densify else []) + (["--freeze_geometry"] if exp.freeze_geometry else []) + (["--freeze_opacity"] if exp.freeze_opacity else [])


def worker_loop(gpu: int, queue: List[Experiment]) -> None:
    worker_key = f"gpu{gpu}"
    for exp in queue:
        run_name = f"dev02_0123_{exp.name}_{RUN_STAMP}"
        log_path = LOG_DIR / f"{run_name}.log"
        model_path = RESULTS_DIR / run_name
        cmd = build_command(exp, run_name)
        if exp.name.startswith("probe_diffgs_illegal") and COMPUTE_SANITIZER_BIN.is_file():
            cmd = [
                str(COMPUTE_SANITIZER_BIN),
                "--tool",
                "memcheck",
                "--leak-check",
                "full",
                "--error-exitcode",
                "86",
            ] + cmd
        env = os.environ.copy()
        skip_guard_flag = "1" if exp.skip_3dgs_guard else "0"
        env.update(
            {
                "PYTHONUNBUFFERED": "1",
                "CUDA_VISIBLE_DEVICES": str(exp.gpu),
                "HF_HOME": str(HF_HOME),
                "HUGGINGFACE_HUB_CACHE": str(HF_HOME / "hub"),
                "TORCH_HOME": str(TORCH_HOME),
                "XDG_CACHE_HOME": str(XDG_CACHE_HOME),
                "EDITSPLAT_HF_HOME": str(HF_HOME),
                "EDITSPLAT_TORCH_HOME": str(TORCH_HOME),
                "EDITSPLAT_MFG_MODE": exp.mfg_mode,
                "EDITSPLAT_MFG_BACKFILL": "nearest",
                "EDITSPLAT_MFG_SOURCE_COUNT": "5",
                # Do not rely on ImageReward ranking in the active line.
                "EDITSPLAT_FILTER_MODE": "all",
                # Keep long-run alive when rasterizer backward hits illegal memory.
                "EDITSPLAT_SKIP_3DGS_BACKWARD_ON_ERROR": skip_guard_flag,
                "EDITSPLAT_MASK_BACKEND": exp.mask_backend,
                "EDITSPLAT_LANGSAM_DEVICE": "cpu",
                # Force real LangSAM path; avoid stub injection in legacy wrapper.
                "FLOWEDIT_REAL_LANGSAM": "1",
                "EDITSPLAT_BASE_MODEL_ID": BASE_MODEL_ID,
                "HF_HUB_OFFLINE": exp.hf_offline,
                "TRANSFORMERS_OFFLINE": exp.hf_offline,
                # xet transport has frequent resets on this node; prefer plain hub transfer.
                "HF_HUB_DISABLE_XET": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:128",
                # Stabilization track: stay on known-good view subset while fixing rasterizer crashes.
                "EDITSPLAT_MAX_TRAIN_VIEWS": str(exp.max_train_views),
                "EDITSPLAT_MAX_GAUSSIANS": str(exp.max_gaussians),
                "EDITSPLAT_SKIP_RENDER_SETS": "1",
                # Use real LPIPS (available in this env) instead of L1 stub fallback.
                "FLOWEDIT_REAL_LPIPS": "1" if exp.real_lpips else "0",
                "CUDA_LAUNCH_BLOCKING": "1" if exp.cuda_launch_blocking else "0",
                "EDITSPLAT_ACTIVE_MODEL_PATH": str(model_path),
                "EDITSPLAT_SAM3_DEBUG_ROOT": str(model_path),
                "EDITSPLAT_SAM3_DEBUG_LIMIT": "4",
                "EDITSPLAT_ELITE_CONF_CORRECTION": "1" if exp.elite_conf_correction else "0",
                "EDITSPLAT_ELITE_SUPPORT_ALPHA": str(exp.elite_support_alpha),
                "EDITSPLAT_ELITE_EDIT_ALPHA": str(exp.elite_edit_alpha),
                "EDITSPLAT_ELITE_CONFIDENCE_ALPHA": str(exp.elite_confidence_alpha),
                "EDITSPLAT_ELITE_SCALE_MIN": str(exp.elite_scale_min),
                "EDITSPLAT_ELITE_SCALE_MAX": str(exp.elite_scale_max),
                "EDITSPLAT_BLITE_CANONICAL_PRIOR": "1" if exp.blite_canonical_prior else "0",
                "EDITSPLAT_BLITE_CANONICAL_DUMP": "1" if exp.blite_canonical_dump else "0",
            }
        )
        if exp.name.startswith("probe_diffgs_illegal"):
            # Capture full device-side stack traces during the illegal-memory repro runs.
            env.setdefault("TORCH_USE_CUDA_DSA", "1")
            dsa_site = SANDBOX_ROOT / "runtime" / "diff_gaussian_rasterization_dsa"
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                f"{dsa_site}:{existing}" if existing else str(dsa_site)
            )
        if exp.mask_backend == "sam3":
            # Keep SAM3 off the saturated training GPUs; run it on CPU to avoid OOM-induced fallback.
            env.setdefault("EDITSPLAT_SAM3_DEVICE", "cpu")

        STATE.patch_worker(
            worker_key,
            {
                "state": "starting",
                "experiment": exp.name,
                "case": exp.case,
                "run_name": run_name,
                "gpu": gpu,
                "model_path": str(model_path),
                "log_path": str(log_path),
                "started_at_utc": utc_now(),
                "mask_backend_requested": exp.mask_backend,
                "mfg_mode": exp.mfg_mode,
                "max_train_views": exp.max_train_views,
                "max_gaussians": exp.max_gaussians,
                "cuda_launch_blocking": exp.cuda_launch_blocking,
                "elite_conf_correction": exp.elite_conf_correction,
                "blite_canonical_prior": exp.blite_canonical_prior,
                "return_code": None,
                "ended_at_utc": None,
                "mask_backend_info": None,
            },
        )
        STATE.write()

        log_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as fout:
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=fout,
                stderr=subprocess.STDOUT,
            )
            STATE.patch_worker(worker_key, {"state": "running", "pid": proc.pid})
            STATE.write()

            while True:
                rc = proc.poll()
                if rc is not None:
                    break
                STATE.patch_worker(
                    worker_key,
                    {
                        "state": "running",
                        "pid": proc.pid,
                        "heartbeat_utc": utc_now(),
                        "log_size_bytes": log_path.stat().st_size if log_path.exists() else 0,
                    },
                )
                STATE.write()
                time.sleep(30)

        mask_meta_path = model_path / "mask_backend_info.json"
        mask_meta: Optional[Dict[str, object]] = None
        if mask_meta_path.is_file():
            try:
                mask_meta = json.loads(mask_meta_path.read_text(encoding="utf-8"))
            except Exception:
                mask_meta = None

        STATE.patch_worker(
            worker_key,
            {
                "state": "completed" if rc == 0 else "failed",
                "return_code": rc,
                "ended_at_utc": utc_now(),
                "mask_backend_info": mask_meta,
            },
        )
        STATE.write()

    STATE.patch_worker(worker_key, {"state": "idle", "finished_queue_at_utc": utc_now()})
    STATE.write()


def main() -> None:
    if not PYTHON.is_file():
        raise FileNotFoundError(f"Missing python env: {PYTHON}")
    if not WRAPPER.is_file():
        raise FileNotFoundError(f"Missing wrapper: {WRAPPER}")

    HF_HOME.mkdir(parents=True, exist_ok=True)
    (HF_HOME / "hub").mkdir(parents=True, exist_ok=True)
    TORCH_HOME.mkdir(parents=True, exist_ok=True)
    XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)

    for gpu in EXPERIMENTS_BY_GPU:
        if gpu not in ALLOWED_GPUS:
            raise ValueError(f"Invalid GPU assignment {gpu}; only {ALLOWED_GPUS} allowed.")

    STATE.update(phase="waiting_checkpoint")
    STATE.write()
    while not checkpoint_ready():
        size = SHARED_CKPT.stat().st_size if SHARED_CKPT.is_file() else 0
        STATE.update(
            checkpoint={
                "path": str(SHARED_CKPT),
                "expected_link": str(EXPECTED_CKPT_LINK),
                "min_bytes": CKPT_MIN_BYTES,
                "ready": False,
                "size_bytes": size,
            }
        )
        STATE.write()
        time.sleep(30)

    STATE.update(
        phase="waiting_base_model_cache",
        checkpoint={
            "path": str(SHARED_CKPT),
            "expected_link": str(EXPECTED_CKPT_LINK),
            "min_bytes": CKPT_MIN_BYTES,
            "ready": True,
            "size_bytes": SHARED_CKPT.stat().st_size,
        },
    )
    STATE.write()
    while not model_cache_ready(BASE_MODEL_CACHE_ROOT):
        STATE.update(
            base_model={
                "id": BASE_MODEL_ID,
                "cache_root": str(BASE_MODEL_CACHE_ROOT),
                "ready": False,
                "incomplete_shards": model_incomplete_count(BASE_MODEL_CACHE_ROOT),
            }
        )
        STATE.write()
        time.sleep(30)

    STATE.update(
        phase="waiting_teacher_model_cache",
        base_model={
            "id": BASE_MODEL_ID,
            "cache_root": str(BASE_MODEL_CACHE_ROOT),
            "ready": True,
            "incomplete_shards": 0,
        },
    )
    STATE.write()
    while not model_cache_ready(TEACHER_MODEL_CACHE_ROOT):
        STATE.update(
            teacher_model={
                "key": TEACHER_MODEL_KEY,
                "id": TEACHER_MODEL_ID,
                "cache_root": str(TEACHER_MODEL_CACHE_ROOT),
                "ready": False,
                "incomplete_shards": model_incomplete_count(TEACHER_MODEL_CACHE_ROOT),
            }
        )
        STATE.write()
        time.sleep(30)

    ensure_checkpoint_link()
    STATE.update(
        phase="running",
        checkpoint={
            "path": str(SHARED_CKPT),
            "expected_link": str(EXPECTED_CKPT_LINK),
            "min_bytes": CKPT_MIN_BYTES,
            "ready": True,
            "size_bytes": SHARED_CKPT.stat().st_size,
        },
        teacher_model={
            "key": TEACHER_MODEL_KEY,
            "id": TEACHER_MODEL_ID,
            "cache_root": str(TEACHER_MODEL_CACHE_ROOT),
            "ready": True,
            "incomplete_shards": 0,
        },
    )
    STATE.write()

    threads: List[threading.Thread] = []
    for gpu, queue in EXPERIMENTS_BY_GPU.items():
        t = threading.Thread(target=worker_loop, args=(gpu, queue), daemon=True, name=f"gpu{gpu}-worker")
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    STATE.update(phase="all_workers_finished")
    STATE.write()


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    HF_HOME.mkdir(parents=True, exist_ok=True)
    main()
