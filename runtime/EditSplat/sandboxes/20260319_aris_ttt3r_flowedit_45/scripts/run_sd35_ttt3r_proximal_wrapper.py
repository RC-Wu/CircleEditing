#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps


ROOT = Path(__file__).resolve().parents[3]
MULTI2D = ROOT / "flowedit_multimodel"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(MULTI2D) not in sys.path:
    sys.path.insert(0, str(MULTI2D))


def _ensure_imagereward_stub() -> None:
    try:
        import ImageReward  # noqa: F401
        return
    except Exception:
        pass

    class _DummyRewardModel:
        def inference_rank(self, prompt, images):
            del prompt
            n = len(images)
            return list(range(1, n + 1)), [0.0] * n

    stub = types.ModuleType("ImageReward")

    def _load(name="ImageReward-v1.0"):
        del name
        print("[WARN] ImageReward import failed; using dummy reward ranker.")
        return _DummyRewardModel()

    stub.load = _load  # type: ignore[attr-defined]
    sys.modules["ImageReward"] = stub


def _ensure_lpips_stub() -> None:
    if os.environ.get("FLOWEDIT_REAL_LPIPS", "0") == "1":
        return
    try:
        import lpips  # noqa: F401
    except Exception:
        pass

    stub = types.ModuleType("lpips")

    class _LPIPS:
        def __init__(self, net="vgg"):
            self.net = net

        def to(self, device):
            del device
            return self

        def requires_grad_(self, flag):
            del flag
            return self

        def __call__(self, x, y):
            if x.ndim == 3:
                x = x.unsqueeze(0)
            if y.ndim == 3:
                y = y.unsqueeze(0)
            return (x.float() - y.float()).abs().mean(dim=(1, 2, 3), keepdim=True)

    stub.LPIPS = _LPIPS  # type: ignore[attr-defined]
    print("[WARN] Using LPIPS stub (L1 fallback). Set FLOWEDIT_REAL_LPIPS=1 to disable.")
    sys.modules["lpips"] = stub


def _ensure_langsam_stub() -> None:
    if os.environ.get("FLOWEDIT_REAL_LANGSAM", "0") == "1":
        return
    stub = types.ModuleType("lang_sam")

    class _LangSAM:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            print("[WARN] Using LangSAM stub (full-image mask). Set FLOWEDIT_REAL_LANGSAM=1 to disable.")

        def predict(self, image_pil, text_prompt):
            del text_prompt
            w, h = image_pil.size
            mask = torch.ones((1, h, w), dtype=torch.float32)
            return mask, None, None, None

    stub.LangSAM = _LangSAM  # type: ignore[attr-defined]
    sys.modules["lang_sam"] = stub


_ensure_imagereward_stub()
_ensure_lpips_stub()
_ensure_langsam_stub()

import run_editing_flow as ref
from run_editing_flow import Editsplat_Pipeline
from scene.gaussian_model import GaussianModel
from src.core_backend import FlowBackendConfig, FlowEditCoreBackend
from src.flowedit_adapters import FlowEditParams, SD3Adapter, create_adapter


def _tensor_to_pil01(img: torch.Tensor) -> Image.Image:
    if img.ndim == 3:
        img = img.unsqueeze(0)
    x = img[0].detach().float()
    if x.min() < 0.0:
        x = (x + 1.0) * 0.5
    x = x.clamp(0.0, 1.0).cpu()
    arr = (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def _pil_to_tensor01(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _to_negative_prompt(negative_prompt) -> str:
    if isinstance(negative_prompt, str):
        return negative_prompt
    return ""


def _to_01_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        x = x.unsqueeze(0)
    y = x.detach().float()
    if y.min() < 0:
        y = (y + 1.0) * 0.5
    return y.clamp(0.0, 1.0)


def _to_m11_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        x = x.unsqueeze(0)
    return x.detach().float().clamp(0.0, 1.0) * 2.0 - 1.0


def _pack_weight_map(x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    w = F.interpolate(x.float(), size=target_hw, mode="bilinear", align_corners=False)
    return w.clamp(0.0, 1.0)


@dataclass
class TTT3RConfig:
    enabled: bool
    repo_root: str
    checkpoint: str
    model_update_type: str
    support_views: int
    support_stride: int
    include_gt_view: bool
    conf_power: float
    conf_floor: float
    geo_scale: float
    input_h: int
    input_w: int
    prox_strength: float
    preserve_strength: float
    edit_boost: float
    preserve_boost: float
    edit_min_mass: float
    preserve_min_mass: float
    adaptive_max_scale: float
    schedule_power: float
    dump_intermediates: bool
    dump_max_per_stage: int
    edit_mask_quantile: float
    mode: str
    ttt3r_gpu: int


class TTT3RRuntime:
    def __init__(self, cfg: TTT3RConfig):
        self.cfg = cfg
        self.model = None
        self.inference_fn = None
        self.device = torch.device(f"cuda:{cfg.ttt3r_gpu}") if torch.cuda.is_available() else torch.device("cpu")
        self.camera_list = None
        self.current_idx = 0
        self.dump_root: Optional[Path] = None
        self.stage_counts: Dict[str, int] = {"initial_edit": 0, "mfg_edit": 0}
        self.initial_edit_cache: Dict[int, torch.Tensor] = {}
        self.source_image_cache: Dict[int, torch.Tensor] = {}
        self.fit_mask_cache: Dict[int, torch.Tensor] = {}
        self.fit_mask_dumped: set[int] = set()
        self.fit_view_score_cache: Dict[int, float] = {}
        self.fit_view_topk_cache: Dict[Tuple[int, float], List[int]] = {}
        self.fit_view_dumped_keys: set[Tuple[int, float]] = set()

    def set_dump_root(self, model_path: Path) -> None:
        if not self.cfg.dump_intermediates:
            return
        self.dump_root = model_path / "debug_intermediates"
        (self.dump_root / "initial_edit").mkdir(parents=True, exist_ok=True)
        (self.dump_root / "mfg_edit").mkdir(parents=True, exist_ok=True)
        meta = {
            "mode": self.cfg.mode,
            "support_views": self.cfg.support_views,
            "support_stride": self.cfg.support_stride,
            "conf_power": self.cfg.conf_power,
            "conf_floor": self.cfg.conf_floor,
            "geo_scale": self.cfg.geo_scale,
            "prox_strength": self.cfg.prox_strength,
            "preserve_strength": self.cfg.preserve_strength,
            "edit_boost": self.cfg.edit_boost,
            "preserve_boost": self.cfg.preserve_boost,
            "edit_min_mass": self.cfg.edit_min_mass,
            "preserve_min_mass": self.cfg.preserve_min_mass,
            "adaptive_max_scale": self.cfg.adaptive_max_scale,
            "schedule_power": self.cfg.schedule_power,
            "edit_mask_quantile": self.cfg.edit_mask_quantile,
        }
        (self.dump_root / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    def init_model(self) -> None:
        if not self.cfg.enabled:
            return
        repo_root = Path(self.cfg.repo_root).resolve()
        src_root = repo_root / "src"
        for candidate in (src_root, repo_root):
            if candidate.exists() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
        try:
            from dust3r.inference import inference_recurrent_lighter
            from dust3r.model import ARCroco3DStereo
        except ModuleNotFoundError:
            from src.dust3r.inference import inference_recurrent_lighter
            from src.dust3r.model import ARCroco3DStereo

        ckpt = Path(self.cfg.checkpoint).resolve()
        self.model = ARCroco3DStereo.from_pretrained(str(ckpt)).to(self.device)
        self.model.config.model_update_type = str(self.cfg.model_update_type)
        self.model.eval()
        self.inference_fn = inference_recurrent_lighter
        print(f"[TTT3R] loaded checkpoint={ckpt} update_type={self.cfg.model_update_type} device={self.device}")

    def _save_tensor_png(self, x: torch.Tensor, out: Path) -> None:
        x01 = _to_01_bchw(x).cpu()
        arr = (x01[0].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 1:
            Image.fromarray(arr[..., 0], mode="L").save(out)
            return
        if arr.ndim == 3 and arr.shape[2] not in (3, 4):
            arr = np.repeat(arr[..., :1], 3, axis=2)
        Image.fromarray(arr).save(out)

    def dump_stage(self, stage: str, payload: Dict[str, torch.Tensor | float | int | str | None]) -> None:
        if not self.cfg.dump_intermediates or self.dump_root is None:
            return
        max_keep = int(self.cfg.dump_max_per_stage)
        if max_keep > 0 and self.stage_counts.get(stage, 0) >= max_keep:
            return
        step_idx = self.stage_counts.get(stage, 0)
        stage_dir = self.dump_root / stage / f"view{int(self.current_idx):03d}_step{step_idx:04d}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        stats: Dict[str, object] = {}
        for key, value in payload.items():
            if isinstance(value, torch.Tensor):
                t = value.detach().float()
                stats[key] = {
                    "shape": list(t.shape),
                    "min": float(t.min().item()),
                    "max": float(t.max().item()),
                    "mean": float(t.mean().item()),
                }
                if t.ndim >= 3:
                    self._save_tensor_png(t, stage_dir / f"{key}.png")
            elif value is not None:
                stats[key] = value
        (stage_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
        self.stage_counts[stage] = step_idx + 1

    def preprocess_for_ttt3r(self, x: torch.Tensor) -> torch.Tensor:
        x01 = _to_01_bchw(x).cpu()
        return F.interpolate(x01, size=(int(self.cfg.input_h), int(self.cfg.input_w)), mode="bilinear", align_corners=False)

    def _build_view(self, x01: torch.Tensor, idx: int):
        h, w = x01.shape[-2], x01.shape[-1]
        eye = torch.eye(4, dtype=torch.float32).unsqueeze(0)
        return {
            "img": (x01 * 2.0 - 1.0).to(torch.float32),
            "ray_map": torch.full((1, 6, h, w), torch.nan, dtype=torch.float32),
            "true_shape": torch.tensor([[h, w]], dtype=torch.int32),
            "idx": int(idx),
            "instance": str(idx),
            "camera_pose": eye,
            "img_mask": torch.tensor([True]),
            "ray_mask": torch.tensor([False]),
            "update": torch.tensor([True]),
            "reset": torch.tensor([False]),
        }

    def _support_indices(self, cur: int, total: int) -> List[int]:
        picks: List[int] = []
        stride = max(1, int(self.cfg.support_stride))
        need = max(1, int(self.cfg.support_views))
        for s in range(1, total + 1):
            for sign in (-1, 1):
                j = cur + sign * s * stride
                if j < 0 or j >= total or j == cur or j in picks:
                    continue
                picks.append(j)
                if len(picks) >= need:
                    return picks
        return picks[:need]

    def proxy_and_weights(self, image: torch.Tensor, mf_image_cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.model is None or self.inference_fn is None:
            raise RuntimeError("TTT3R model not initialized")
        if self.camera_list is None:
            raise RuntimeError("camera_list unavailable")

        cur = int(self.current_idx)
        total = len(self.camera_list)
        supports = self._support_indices(cur, total)
        views = []
        vid = 0
        for j in supports:
            cam = self.camera_list[int(j)]
            gt = getattr(cam, "gt_image", None)
            if gt is None:
                continue
            views.append(self._build_view(self.preprocess_for_ttt3r(gt.unsqueeze(0)), vid))
            vid += 1
        if bool(self.cfg.include_gt_view):
            views.append(self._build_view(self.preprocess_for_ttt3r(image), vid))
            vid += 1
        x_mf = self.preprocess_for_ttt3r(mf_image_cond)
        views.append(self._build_view(x_mf, vid))

        with torch.no_grad():
            outputs, _state = self.inference_fn(views, self.model, self.device, verbose=False)

        pred_last = outputs["pred"][-1]
        rgb_m11 = pred_last["rgb"].permute(0, 3, 1, 2).detach().float()
        conf = pred_last["conf"].detach().float().unsqueeze(1)
        rgb01 = ((rgb_m11 + 1.0) * 0.5).clamp(0.0, 1.0)
        cmin = conf.amin(dim=(-2, -1), keepdim=True)
        cmax = conf.amax(dim=(-2, -1), keepdim=True)
        weight = (conf - cmin) / (cmax - cmin + 1e-6)
        weight = weight.clamp(0.0, 1.0).pow(float(self.cfg.conf_power))
        if float(self.cfg.conf_floor) > 0.0:
            floor = float(self.cfg.conf_floor)
            weight = ((weight - floor) / max(1e-6, 1.0 - floor)).clamp(0.0, 1.0)
        weight = (weight * float(self.cfg.geo_scale)).clamp(0.0, 1.0)
        proxy = rgb01
        return proxy, weight


def patch_aux_models_to_cpu() -> None:
    try:
        orig_rm_load = ref.RM.load

        def _rm_load_cpu(name="ImageReward-v1.0", device="cpu", *args, **kwargs):
            kwargs["device"] = "cpu"
            try:
                return orig_rm_load(name, *args, **kwargs)
            except TypeError:
                kwargs.pop("device", None)
                return orig_rm_load(name, *args, **kwargs)

        ref.RM.load = _rm_load_cpu
    except Exception:
        pass

    try:
        if hasattr(ref.LangSAM, "build_groundingdino") and hasattr(ref.LangSAM, "build_sam"):
            class _LangSAMCPU(ref.LangSAM):
                def __init__(self, sam_type="vit_h", ckpt_path=None):
                    self.sam_type = sam_type
                    self.device = torch.device("cpu")
                    self.build_groundingdino()
                    self.build_sam(ckpt_path)

            ref.LangSAM = _LangSAMCPU
    except Exception:
        pass


def patch_scene_load_from_checkpoint() -> None:
    base_scene_cls = ref.Scene

    class _SceneLoadFromCheckpoint(base_scene_cls):
        def __init__(self, args, gaussians, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
            if load_iteration is None and getattr(args, "source_checkpoint", ""):
                load_iteration = parse_iter_from_checkpoint(args.source_checkpoint)
            super().__init__(
                args=args,
                gaussians=gaussians,
                load_iteration=load_iteration,
                shuffle=shuffle,
                resolution_scales=resolution_scales,
            )

    ref.Scene = _SceneLoadFromCheckpoint


def patch_camera_dataset(runtime: TTT3RRuntime, head_k: int) -> None:
    base_dataset_cls = ref.CameraDataset
    orig_init = base_dataset_cls.__init__
    orig_getitem = base_dataset_cls.__getitem__

    def _init(self, scene):
        orig_init(self, scene)
        if int(head_k) > 0 and hasattr(self, "camera_list") and isinstance(self.camera_list, list):
            self.camera_list = self.camera_list[: int(head_k)]
            print(f"[SPARSE] CameraDataset truncated to {len(self.camera_list)} views")
        runtime.camera_list = self.camera_list

    def _getitem(self, idx):
        runtime.current_idx = int(idx)
        return orig_getitem(self, idx)

    base_dataset_cls.__init__ = _init
    base_dataset_cls.__getitem__ = _getitem


def patch_densify_control(disable_densify: bool) -> None:
    if not disable_densify:
        return

    def _skip_densify(self, *args, **kwargs):
        del args, kwargs
        if not getattr(self, "_editsplat_skip_densify_warned", False):
            print("[WARN] disable_densify=1: skipping Gaussian densify_and_prune() in sandbox wrapper.")
            self._editsplat_skip_densify_warned = True
        return

    def _skip_densify_stats(self, viewspace_point_tensor, update_filter):
        del viewspace_point_tensor, update_filter
        return

    GaussianModel.densify_and_prune = _skip_densify
    GaussianModel.add_densification_stats = _skip_densify_stats


def patch_geometry_freeze_control(freeze_geometry: bool, freeze_opacity: bool) -> None:
    if not freeze_geometry and not freeze_opacity:
        return

    orig_training_setup = GaussianModel.training_setup
    orig_update_learning_rate = GaussianModel.update_learning_rate

    def _apply_freeze(self) -> None:
        field_names = []
        group_names = []
        if freeze_geometry:
            field_names.extend(("_xyz", "_scaling", "_rotation"))
            group_names.extend(("xyz", "scaling", "rotation"))
        if freeze_opacity:
            field_names.append("_opacity")
            group_names.append("opacity")
        for field in field_names:
            tensor = getattr(self, field, None)
            if tensor is not None:
                tensor.requires_grad_(False)
        if getattr(self, "optimizer", None) is not None:
            for group in self.optimizer.param_groups:
                if group.get("name") in set(group_names):
                    group["lr"] = 0.0

    def _training_setup(self, training_args):
        orig_training_setup(self, training_args)
        _apply_freeze(self)
        if not getattr(self, "_editsplat_freeze_geometry_warned", False):
            frozen_parts = []
            if freeze_geometry:
                frozen_parts.append("xyz/scaling/rotation")
            if freeze_opacity:
                frozen_parts.append("opacity")
            frozen_desc = " + ".join(frozen_parts) if frozen_parts else "nothing"
            print(f"[WARN] sandbox wrapper freezing {frozen_desc} updates.")
            self._editsplat_freeze_geometry_warned = True

    def _update_learning_rate(self, iteration):
        orig_update_learning_rate(self, iteration)
        _apply_freeze(self)

    GaussianModel.training_setup = _training_setup
    GaussianModel.update_learning_rate = _update_learning_rate


def patch_optimizer_step_control(max_optimizer_steps: int) -> None:
    limit = int(max_optimizer_steps)
    if limit < 0:
        return

    orig_training_setup = GaussianModel.training_setup

    def _training_setup(self, training_args):
        orig_training_setup(self, training_args)
        optimizer = getattr(self, "optimizer", None)
        if optimizer is None:
            return
        orig_step = optimizer.step
        counter = {"count": 0}

        def _limited_step(*args, **kwargs):
            if counter["count"] >= limit:
                if not getattr(self, "_editsplat_max_optimizer_steps_warned", False):
                    print(
                        "[WARN] sandbox wrapper reached "
                        f"max_optimizer_steps={limit}; skipping later optimizer.step() calls."
                    )
                    self._editsplat_max_optimizer_steps_warned = True
                return None
            counter["count"] += 1
            return orig_step(*args, **kwargs)

        optimizer.step = _limited_step

    GaussianModel.training_setup = _training_setup


def patch_optimizer_lr_scale(optimizer_lr_scale: float) -> None:
    scale = float(optimizer_lr_scale)
    if abs(scale - 1.0) < 1e-8:
        return

    orig_training_setup = GaussianModel.training_setup
    orig_update_learning_rate = GaussianModel.update_learning_rate

    def _apply_scale(self) -> None:
        optimizer = getattr(self, "optimizer", None)
        if optimizer is None:
            return
        for group in optimizer.param_groups:
            base_lr = group.get("_editsplat_base_lr")
            if base_lr is None:
                base_lr = float(group.get("lr", 0.0))
                group["_editsplat_base_lr"] = base_lr
            group["lr"] = float(base_lr) * scale

    def _training_setup(self, training_args):
        orig_training_setup(self, training_args)
        _apply_scale(self)
        if not getattr(self, "_editsplat_optimizer_lr_scale_warned", False):
            print(f"[WARN] sandbox wrapper scaling optimizer learning rates by {scale:.4f}.")
            self._editsplat_optimizer_lr_scale_warned = True

    def _update_learning_rate(self, iteration):
        orig_update_learning_rate(self, iteration)
        _apply_scale(self)

    GaussianModel.training_setup = _training_setup
    GaussianModel.update_learning_rate = _update_learning_rate


def parse_iter_from_checkpoint(path: str) -> int:
    m = re.search(r"chkpnt(\\d+)\\.pth$", str(path))
    return int(m.group(1)) if m else 7000


def ensure_point_cloud_link(model_path: str, source_checkpoint: str) -> None:
    if not source_checkpoint:
        return
    model_dir = Path(model_path)
    src_ckpt = Path(source_checkpoint).resolve()
    src_pretrained_dir = src_ckpt.parent
    src_pc = src_pretrained_dir / "point_cloud"
    dst_pc = model_dir / "point_cloud"
    load_iter = parse_iter_from_checkpoint(str(src_ckpt))
    src_iter_dir = src_pc / f"iteration_{load_iter}"
    src_ply = src_iter_dir / "point_cloud.ply"
    dst_iter_dir = dst_pc / f"iteration_{load_iter}"
    dst_ply = dst_iter_dir / "point_cloud.ply"
    if dst_ply.exists():
        return
    if not src_ply.exists():
        raise FileNotFoundError(f"Missing source point_cloud ply: {src_ply}")
    model_dir.mkdir(parents=True, exist_ok=True)
    if dst_pc.is_symlink():
        dst_pc.unlink()
    elif dst_pc.exists() and not dst_pc.is_dir():
        raise RuntimeError(f"Unexpected point_cloud path type: {dst_pc}")
    dst_iter_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_ply, dst_ply)


def _make_edit_mask(src_image: torch.Tensor, ref_image: torch.Tensor, quantile: float) -> torch.Tensor:
    diff = (_to_01_bchw(ref_image) - _to_01_bchw(src_image)).abs().mean(dim=1, keepdim=True)
    q = float(torch.quantile(diff.flatten(), torch.tensor([max(0.5, min(0.999, quantile))], device=diff.device)).item())
    scale = max(q, 1e-4)
    mask = (diff / scale).clamp(0.0, 1.0)
    return mask


def _schedule_weight(step_idx: int, total_steps: int, base_strength: float, power: float) -> float:
    if total_steps <= 1 or base_strength <= 0:
        return 0.0
    phase = float(step_idx + 1) / float(total_steps)
    window = math.sin(math.pi * phase) ** max(1e-6, float(power))
    return float(base_strength) * float(window)


def _adaptive_min_mass_scale(weight: torch.Tensor, min_mass: float, max_scale: float) -> Tuple[float, float]:
    mass = float(weight.detach().float().mean().item())
    if min_mass <= 0.0:
        return 1.0, mass
    if mass >= min_mass:
        return 1.0, mass
    if mass <= 1e-6:
        return float(max_scale), mass
    return min(float(max_scale), float(min_mass) / mass), mass


def _fit_mask_from_runtime(
    runtime: TTT3RRuntime,
    mode: str,
    quantile: float,
    bg: float,
) -> Optional[torch.Tensor]:
    if mode == "none":
        return None
    idx = int(runtime.current_idx)
    cached = runtime.fit_mask_cache.get(idx)
    if cached is not None:
        return cached

    src = runtime.source_image_cache.get(idx)
    init_ref = runtime.initial_edit_cache.get(idx)
    if src is None or init_ref is None:
        return None

    mask = _make_edit_mask(src.to(torch.float32), init_ref.to(torch.float32), quantile).detach().cpu()
    if float(bg) > 0.0:
        mask = mask + (1.0 - mask) * float(bg)
        mask = mask.clamp(0.0, 1.0)
    runtime.fit_mask_cache[idx] = mask

    if runtime.dump_root is not None and idx not in runtime.fit_mask_dumped:
        out_dir = runtime.dump_root / "fit_masks"
        out_dir.mkdir(parents=True, exist_ok=True)
        runtime._save_tensor_png(mask.to(torch.float32), out_dir / f"view{idx:03d}.png")
        stats = {
            "mode": mode,
            "quantile": float(quantile),
            "bg": float(bg),
            "mean": float(mask.float().mean().item()),
            "min": float(mask.float().min().item()),
            "max": float(mask.float().max().item()),
        }
        (out_dir / f"view{idx:03d}.json").write_text(
            json.dumps(stats, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        runtime.fit_mask_dumped.add(idx)
    return mask


def _fit_view_score(runtime: TTT3RRuntime, idx: int, quantile: float) -> float:
    cached = runtime.fit_view_score_cache.get(idx)
    if cached is not None:
        return cached

    src = runtime.source_image_cache.get(idx)
    init_ref = runtime.initial_edit_cache.get(idx)
    if src is None or init_ref is None:
        runtime.fit_view_score_cache[idx] = 0.0
        return 0.0

    mask = _make_edit_mask(src.to(torch.float32), init_ref.to(torch.float32), quantile).detach().cpu()
    score = float(mask.float().mean().item())
    runtime.fit_view_score_cache[idx] = score
    return score


def _fit_view_topk_indices(runtime: TTT3RRuntime, topk: int, quantile: float) -> List[int]:
    if topk < 1:
        return sorted(runtime.initial_edit_cache.keys())

    cache_key = (int(topk), round(float(quantile), 6))
    cached = runtime.fit_view_topk_cache.get(cache_key)
    if cached is not None:
        return cached

    candidates = sorted(
        {int(k) for k in runtime.initial_edit_cache.keys() if k in runtime.source_image_cache}
    )
    ranked = sorted(
        ((idx, _fit_view_score(runtime, idx, quantile)) for idx in candidates),
        key=lambda item: (-item[1], item[0]),
    )
    selected = [idx for idx, _score in ranked[:topk]]
    runtime.fit_view_topk_cache[cache_key] = selected

    if runtime.dump_root is not None and cache_key not in runtime.fit_view_dumped_keys:
        out_path = runtime.dump_root / "fit_view_selection.json"
        payload = {
            "topk": int(topk),
            "quantile": float(quantile),
            "selected_indices": selected,
            "scores": [{"idx": int(idx), "score": float(score)} for idx, score in ranked],
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        runtime.fit_view_dumped_keys.add(cache_key)

    return selected


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


def _runtime_support_mask(runtime: TTT3RRuntime, role: str, target_size: Tuple[int, int]) -> Optional[torch.Tensor]:
    cache = getattr(runtime, "support_mask_cache", None)
    if not isinstance(cache, dict):
        return None
    entry = cache.get(int(runtime.current_idx))
    if isinstance(entry, torch.Tensor):
        mask = entry.detach().float().cpu()
    elif isinstance(entry, dict):
        payload = None
        roles = entry.get("roles")
        if isinstance(roles, dict):
            payload = roles.get(role) or roles.get("gt_view") or roles.get("reproject") or entry.get("last")
        if not isinstance(payload, dict):
            payload = entry.get("last") if isinstance(entry.get("last"), dict) else None
        if not isinstance(payload, dict):
            return None
        mask = payload.get("soft_mask")
        if not isinstance(mask, torch.Tensor):
            mask = payload.get("mask")
        if not isinstance(mask, torch.Tensor):
            return None
        mask = mask.detach().float().cpu()
    else:
        return None

    if mask.ndim == 3:
        mask = mask.unsqueeze(0)
    if tuple(mask.shape[-2:]) != tuple(target_size):
        mask = F.interpolate(mask, size=target_size, mode="bilinear", align_corners=False)
    return mask.clamp(0.0, 1.0)


def patch_fit_loss_control(
    runtime: TTT3RRuntime,
    fit_mask_mode: str,
    fit_mask_quantile: float,
    fit_mask_bg: float,
    fit_view_topk: int,
) -> None:
    mode = str(fit_mask_mode).strip().lower()
    topk = int(fit_view_topk)
    if mode == "none" and topk < 1:
        return

    orig_l1_loss = ref.l1_loss
    orig_build_lpips_loss = ref._build_lpips_loss

    def _view_is_selected() -> bool:
        if topk < 1:
            return True
        selected = _fit_view_topk_indices(
            runtime=runtime,
            topk=topk,
            quantile=float(fit_mask_quantile),
        )
        return int(runtime.current_idx) in selected

    def _masked_l1_loss(network_output, gt):
        if not _view_is_selected():
            return (network_output.float() - gt.float()).sum() * 0.0
        mask = _fit_mask_from_runtime(
            runtime=runtime,
            mode=mode,
            quantile=float(fit_mask_quantile),
            bg=float(fit_mask_bg),
        )
        if mask is None:
            return orig_l1_loss(network_output, gt)
        mask_dev = mask.to(device=network_output.device, dtype=network_output.dtype)
        if mask_dev.ndim == 3:
            mask_dev = mask_dev.unsqueeze(0)
        if network_output.ndim == 3:
            network_output = network_output.unsqueeze(0)
        if gt.ndim == 3:
            gt = gt.unsqueeze(0)
        target_device = gt.device
        gt = gt.to(device=network_output.device, dtype=network_output.dtype)
        diff = (network_output - gt).abs()
        denom = mask_dev.sum() * diff.shape[1] + 1e-6
        return ((diff * mask_dev).sum() / denom).to(target_device)

    def _build_masked_lpips_loss(device: torch.device):
        base_loss = orig_build_lpips_loss(device)

        class _MaskedLPIPS:
            def to(self, target_device):
                if hasattr(base_loss, "to"):
                    base_loss.to(target_device)
                return self

            def requires_grad_(self, flag):
                if hasattr(base_loss, "requires_grad_"):
                    base_loss.requires_grad_(flag)
                return self

            def __call__(self, x, y):
                if not _view_is_selected():
                    return (x.float() - y.float()).sum() * 0.0
                target_device = x.device
                mask = _fit_mask_from_runtime(
                    runtime=runtime,
                    mode=mode,
                    quantile=float(fit_mask_quantile),
                    bg=float(fit_mask_bg),
                )
                if mask is None:
                    if hasattr(base_loss, "to"):
                        base_loss.to(target_device)
                    try:
                        return base_loss(x.to(target_device), y.to(target_device)).to(target_device)
                    except RuntimeError as exc:
                        msg = str(exc)
                        if "same device" not in msg and "cuda" not in msg:
                            raise
                        print(f"[WARN] LPIPS runtime failed, fallback to L1 proxy. exc={exc}")
                        x_ref = x.to(target_device).float()
                        y_ref = y.to(target_device).float()
                        return (x_ref - y_ref).abs().mean(dim=(1, 2, 3), keepdim=True).to(target_device)
                if x.ndim == 3:
                    x = x.unsqueeze(0)
                if y.ndim == 3:
                    y = y.unsqueeze(0)
                x = x.to(device=target_device, dtype=x.dtype)
                y = y.to(device=target_device, dtype=x.dtype)
                mask_dev = mask.to(device=target_device, dtype=x.dtype)
                if mask_dev.ndim == 3:
                    mask_dev = mask_dev.unsqueeze(0)
                x_masked = x * mask_dev + y.detach() * (1.0 - mask_dev)
                y_masked = y
                if hasattr(base_loss, "to"):
                    base_loss.to(target_device)
                try:
                    return base_loss(x_masked, y_masked).to(target_device)
                except RuntimeError as exc:
                    msg = str(exc)
                    if "same device" not in msg and "cuda" not in msg:
                        raise
                    print(f"[WARN] LPIPS runtime failed, fallback to L1 proxy. exc={exc}")
                    return (x_masked.float() - y_masked.float()).abs().mean(dim=(1, 2, 3), keepdim=True).to(target_device)

        return _MaskedLPIPS()

    ref.l1_loss = _masked_l1_loss
    ref._build_lpips_loss = _build_masked_lpips_loss


@torch.no_grad()
def _sd3_edit_with_proximal(
    adapter: SD3Adapter,
    image_pil: Image.Image,
    proxy_pil: Image.Image,
    src_prompt: str,
    tar_prompt: str,
    params: FlowEditParams,
    edit_weight_px: torch.Tensor,
    preserve_weight_px: torch.Tensor,
    runtime: TTT3RRuntime,
) -> torch.Tensor:
    x_src, shift, scale = adapter._encode_src(image_pil, params.resize_side)
    x_proxy, _shift_proxy, _scale_proxy = adapter._encode_src(proxy_pil, params.resize_side)
    scheduler = adapter.pipe.scheduler
    timesteps, _ = retrieve_timesteps(scheduler, params.diffusion_steps, adapter.device)
    use_velocity_guidance = runtime.cfg.mode == "velocity"

    src = adapter.pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=params.negative_prompt if params.negative_prompt else "",
        do_classifier_free_guidance=True,
        device=adapter.device,
    )
    tar = adapter.pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=params.negative_prompt if params.negative_prompt else "",
        do_classifier_free_guidance=True,
        device=adapter.device,
    )
    src_pos, src_neg, src_pool_pos, src_pool_neg = [x.to(adapter.model_dtype) for x in src]
    tar_pos, tar_neg, tar_pool_pos, tar_pool_neg = [x.to(adapter.model_dtype) for x in tar]

    def _velocity(latents, t, pos_emb, neg_emb, pos_pool, neg_pool, guidance):
        latent_in = torch.cat([latents, latents], dim=0)
        timestep = t.expand(latent_in.shape[0])
        enc = torch.cat([neg_emb, pos_emb], dim=0)
        pool = torch.cat([neg_pool, pos_pool], dim=0)
        pred = adapter.pipe.transformer(
            hidden_states=latent_in,
            timestep=timestep,
            encoder_hidden_states=enc,
            pooled_projections=pool,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
        p_u, p_c = pred.chunk(2)
        return p_u + guidance * (p_c - p_u)

    edit_lat = _pack_weight_map(edit_weight_px, x_src.shape[-2:]).to(device=x_src.device, dtype=torch.float32)
    preserve_lat = _pack_weight_map(preserve_weight_px, x_src.shape[-2:]).to(device=x_src.device, dtype=torch.float32)
    edit_scale, edit_mass = _adaptive_min_mass_scale(
        edit_lat,
        runtime.cfg.edit_min_mass,
        runtime.cfg.adaptive_max_scale,
    )
    preserve_scale, preserve_mass = _adaptive_min_mass_scale(
        preserve_lat,
        runtime.cfg.preserve_min_mass,
        runtime.cfg.adaptive_max_scale,
    )

    zt_edit = x_src.clone()
    diffusion_steps = len(timesteps)
    for i, t in enumerate(timesteps):
        if diffusion_steps - i > params.n_max:
            continue
        scheduler._init_step_index(t)
        t_i = scheduler.sigmas[scheduler.step_index]
        t_im1 = scheduler.sigmas[scheduler.step_index + 1] if i < len(timesteps) - 1 else t_i

        if diffusion_steps - i > params.n_min:
            v_delta_avg = torch.zeros_like(x_src)
            prox_delta = torch.zeros_like(x_src, dtype=torch.float32)
            preserve_delta = torch.zeros_like(x_src, dtype=torch.float32)
            vel_edit_delta = torch.zeros_like(x_src, dtype=torch.float32)
            vel_preserve_delta = torch.zeros_like(x_src, dtype=torch.float32)

            for _ in range(params.n_avg):
                fwd_noise = torch.randn_like(x_src)
                zt_src = (1.0 - t_i) * x_src + t_i * fwd_noise
                zt_proxy = (1.0 - t_i) * x_proxy + t_i * fwd_noise
                zt_tar_in = zt_edit + zt_src - x_src
                v_src = _velocity(zt_src, t, src_pos, src_neg, src_pool_pos, src_pool_neg, params.src_guidance_scale)
                v_tar = _velocity(zt_tar_in, t, tar_pos, tar_neg, tar_pool_pos, tar_pool_neg, params.tar_guidance_scale)
                v_delta_avg = v_delta_avg + (v_tar - v_src) / float(params.n_avg)
                if use_velocity_guidance:
                    v_proxy = _velocity(zt_proxy, t, tar_pos, tar_neg, tar_pool_pos, tar_pool_neg, params.tar_guidance_scale)
                    vel_edit_delta = vel_edit_delta + (edit_lat * (v_proxy.float() - v_tar.float())) / float(params.n_avg)
                    vel_preserve_delta = vel_preserve_delta + (preserve_lat * (v_src.float() - v_tar.float())) / float(params.n_avg)
                else:
                    prox_delta = prox_delta + (edit_lat * (zt_proxy.float() - zt_edit.float())) / float(params.n_avg)
                    preserve_delta = preserve_delta + (preserve_lat * (zt_src.float() - zt_edit.float())) / float(params.n_avg)

            zt_edit = zt_edit.to(torch.float32)
            eta_prox = _schedule_weight(i, diffusion_steps, runtime.cfg.prox_strength, runtime.cfg.schedule_power)
            eta_pres = _schedule_weight(i, diffusion_steps, runtime.cfg.preserve_strength, runtime.cfg.schedule_power)
            eta_prox *= edit_scale
            eta_pres *= preserve_scale
            if use_velocity_guidance:
                if eta_prox > 0:
                    v_delta_avg = v_delta_avg + eta_prox * vel_edit_delta.to(v_delta_avg.dtype)
                if eta_pres > 0:
                    v_delta_avg = v_delta_avg + eta_pres * vel_preserve_delta.to(v_delta_avg.dtype)
                zt_edit = zt_edit + (t_im1 - t_i) * v_delta_avg.to(torch.float32)
            else:
                zt_edit = zt_edit + (t_im1 - t_i) * v_delta_avg.to(torch.float32)
                if eta_prox > 0:
                    zt_edit = zt_edit + eta_prox * prox_delta
                if eta_pres > 0:
                    zt_edit = zt_edit + eta_pres * preserve_delta
            zt_edit = zt_edit.to(adapter.model_dtype)
        else:
            zt_edit = zt_edit.to(adapter.model_dtype)

    out_denorm = (zt_edit / scale) + shift
    image = adapter.pipe.vae.decode(out_denorm.to(adapter.pipe.vae.dtype), return_dict=False)[0]
    runtime.dump_stage(
        "mfg_edit",
        {
            "proxy": _pil_to_tensor01(proxy_pil),
            "edit_weight": edit_weight_px,
            "preserve_weight": preserve_weight_px,
            "latent_edit_weight": edit_lat.mean(dim=1, keepdim=True),
            "latent_preserve_weight": preserve_lat.mean(dim=1, keepdim=True),
            "latent_edit_mass": torch.full((1, 1, 1, 1), float(edit_mass), dtype=torch.float32),
            "latent_preserve_mass": torch.full((1, 1, 1, 1), float(preserve_mass), dtype=torch.float32),
            "adaptive_edit_scale": torch.full((1, 1, 1, 1), float(edit_scale), dtype=torch.float32),
            "adaptive_preserve_scale": torch.full((1, 1, 1, 1), float(preserve_scale), dtype=torch.float32),
            "mode_velocity_guidance": float(use_velocity_guidance),
        },
    )
    return image


class _DNAEditStaticProxyAdapter:
    def __init__(self, backend: FlowEditCoreBackend, device: torch.device, resize_side: int):
        self.backend = backend
        self.device = device
        self.resize_side = int(resize_side)

    def edit(
        self,
        image_pil: Image.Image,
        src_prompt: str,
        tar_prompt: str,
        params: FlowEditParams,
    ) -> torch.Tensor:
        image = _pil_to_tensor01(image_pil).to(self.device, dtype=torch.float32)
        return self.backend.edit(
            image=image,
            src_prompt=src_prompt,
            tar_prompt=tar_prompt,
            negative_prompt=_to_negative_prompt(params.negative_prompt),
            diffusion_steps=int(params.diffusion_steps),
            n_avg=int(params.n_avg),
            src_guidance_scale=float(params.src_guidance_scale),
            tar_guidance_scale=float(params.tar_guidance_scale),
            n_min=int(params.n_min),
            n_max=int(params.n_max),
            seed=int(params.seed),
        ).to(self.device, dtype=torch.float32)


def patch_edit_methods(adapter, runtime: TTT3RRuntime) -> None:
    orig_edit_image = Editsplat_Pipeline.edit_image

    def _edit_image(
        self,
        image: torch.Tensor,
        src_prompt: str,
        tar_prompt: str,
        negative_prompt: Optional[str] = None,
        diffusion_steps: int = 28,
        n_avg: int = 1,
        src_guidance_scale: float = 1.5,
        tar_guidance_scale: float = 5.5,
        n_min: int = 0,
        n_max: int = 24,
        seed: int = 10,
        lambda_S: float = 0.0,
        mask_S: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del self, lambda_S, mask_S
        out = adapter.edit(
            _tensor_to_pil01(image),
            src_prompt,
            tar_prompt,
            FlowEditParams(
                diffusion_steps=int(diffusion_steps),
                n_avg=int(n_avg),
                src_guidance_scale=float(src_guidance_scale),
                tar_guidance_scale=float(tar_guidance_scale),
                n_min=int(n_min),
                n_max=int(n_max),
                seed=int(seed),
                negative_prompt=_to_negative_prompt(negative_prompt),
                resize_side=int(runtime.cfg.input_w),
            ),
        ).to(image.device, dtype=torch.float32)
        runtime.source_image_cache[int(runtime.current_idx)] = _to_01_bchw(image).detach().cpu()
        runtime.initial_edit_cache[int(runtime.current_idx)] = _to_01_bchw(out).detach().cpu()
        runtime.dump_stage("initial_edit", {"input": image, "initial_edit": out})
        return out

    def _edit_image_mfg(
        self,
        image: torch.Tensor,
        MF_image_cond: torch.Tensor,
        src_prompt: str,
        tar_prompt: str,
        negative_prompt: Optional[str] = None,
        diffusion_steps: int = 28,
        n_avg: int = 1,
        src_guidance_scale: float = 1.5,
        tar_guidance_scale: float = 5.5,
        n_min: int = 0,
        n_max: int = 24,
        seed: int = 10,
        lambda_S: float = 0.0,
        lambda_M: float = 0.0,
        mask_S: Optional[torch.Tensor] = None,
        mask_M: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del self, lambda_S, lambda_M, mask_S, mask_M
        if not runtime.cfg.enabled:
            return adapter.edit(
                _tensor_to_pil01(MF_image_cond),
                src_prompt,
                tar_prompt,
                FlowEditParams(
                    diffusion_steps=int(diffusion_steps),
                    n_avg=int(n_avg),
                    src_guidance_scale=float(src_guidance_scale),
                    tar_guidance_scale=float(tar_guidance_scale),
                    n_min=int(n_min),
                    n_max=int(n_max),
                    seed=int(seed),
                    negative_prompt=_to_negative_prompt(negative_prompt),
                    resize_side=int(runtime.cfg.input_w),
                ),
            ).to(image.device, dtype=torch.float32)

        proxy, geo_weight = runtime.proxy_and_weights(image=image, mf_image_cond=MF_image_cond)
        proxy = F.interpolate(proxy, size=image.shape[-2:], mode="bilinear", align_corners=False).to(image.device, dtype=torch.float32)
        geo_weight = F.interpolate(geo_weight, size=image.shape[-2:], mode="bilinear", align_corners=False).to(image.device, dtype=torch.float32)

        init_ref = runtime.initial_edit_cache.get(int(runtime.current_idx))
        if init_ref is None:
            init_ref = _to_01_bchw(proxy).detach().cpu()
        init_ref = init_ref.to(image.device, dtype=torch.float32)
        edit_mask = _make_edit_mask(image, init_ref, runtime.cfg.edit_mask_quantile)
        edit_ratio = (edit_mask * float(runtime.cfg.edit_boost)).clamp(0.0, 1.0)
        preserve_ratio = ((1.0 - edit_mask) * float(runtime.cfg.preserve_boost)).clamp(0.0, 1.0)
        support_role = os.environ.get("EDITSPLAT_SAM3_MFG_ROLE", "gt_view").strip().lower() or "gt_view"
        support_mask = _runtime_support_mask(runtime, role=support_role, target_size=tuple(image.shape[-2:]))
        if isinstance(support_mask, torch.Tensor):
            support_mask = support_mask.to(image.device, dtype=torch.float32)
            edit_alpha = max(0.0, min(1.0, _env_float("EDITSPLAT_SAM3_EDIT_WEIGHT_ALPHA", 0.85)))
            preserve_alpha = max(0.0, min(1.0, _env_float("EDITSPLAT_SAM3_PRESERVE_WEIGHT_ALPHA", 0.35)))
            edit_ratio = (edit_ratio * ((1.0 - edit_alpha) + edit_alpha * support_mask)).clamp(0.0, 1.0)
            preserve_ratio = (preserve_ratio * (1.0 - preserve_alpha * support_mask)).clamp(0.0, 1.0)
        edit_weight = (geo_weight * edit_ratio).clamp(0.0, 1.0)
        preserve_weight = (geo_weight * preserve_ratio).clamp(0.0, 1.0)

        runtime.dump_stage(
            "mfg_edit",
            {
                "source": image,
                "mf_cond": MF_image_cond,
                "proxy_rgb": proxy,
                "geo_weight": geo_weight,
                "edit_mask": edit_mask,
                "support_mask": support_mask if isinstance(support_mask, torch.Tensor) else torch.zeros_like(edit_mask),
                "edit_ratio": edit_ratio,
                "preserve_ratio": preserve_ratio,
                "edit_weight": edit_weight,
                "preserve_weight": preserve_weight,
            },
        )

        params = FlowEditParams(
            diffusion_steps=int(diffusion_steps),
            n_avg=int(n_avg),
            src_guidance_scale=float(src_guidance_scale),
            tar_guidance_scale=float(tar_guidance_scale),
            n_min=int(n_min),
            n_max=int(n_max),
            seed=int(seed),
            negative_prompt=_to_negative_prompt(negative_prompt),
            resize_side=int(runtime.cfg.input_w),
        )

        if runtime.cfg.mode == "static_proxy":
            return adapter.edit(_tensor_to_pil01(proxy), src_prompt, tar_prompt, params).to(image.device, dtype=torch.float32)

        return _sd3_edit_with_proximal(
            adapter=adapter,
            image_pil=_tensor_to_pil01(image),
            proxy_pil=_tensor_to_pil01(proxy),
            src_prompt=src_prompt,
            tar_prompt=tar_prompt,
            params=params,
            edit_weight_px=edit_weight,
            preserve_weight_px=preserve_weight,
            runtime=runtime,
        ).to(image.device, dtype=torch.float32)

    Editsplat_Pipeline.edit_image = _edit_image
    Editsplat_Pipeline.edit_image_MFG = _edit_image_mfg


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model_key", type=str, default="sd35-large")
    parser.add_argument("--model_id", type=str, default="")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--hf_home", type=str, default="/dev_vepfs/rc_wu/cache/hf_home")
    parser.add_argument("--adapter_resize_side", type=int, default=512)
    parser.add_argument("--adapter_gpu", type=int, default=1)
    parser.add_argument("--base_gpu", type=int, default=0)
    parser.add_argument("--head_k", type=int, default=6)
    parser.add_argument("--depth_mode", type=str, default="render", choices=["render", "constant"])
    parser.add_argument("--skip_agt", action="store_true")
    parser.add_argument("--aux_models_cpu", action="store_true", default=True)
    parser.add_argument("--no_aux_models_cpu", action="store_false", dest="aux_models_cpu")
    parser.add_argument("--ttt3r_repo_root", type=str, default="/dev_vepfs/rc_wu/edit/TTT3R")
    parser.add_argument("--ttt3r_checkpoint", type=str, default="/dev_vepfs/rc_wu/edit/TTT3R/src/cut3r_512_dpt_4_64.pth")
    parser.add_argument("--ttt3r_model_update_type", type=str, default="ttt3r", choices=["cut3r", "ttt3r"])
    parser.add_argument("--ttt3r_support_views", type=int, default=2)
    parser.add_argument("--ttt3r_support_stride", type=int, default=1)
    parser.add_argument("--ttt3r_include_gt_view", action="store_true", default=True)
    parser.add_argument("--ttt3r_no_include_gt_view", action="store_false", dest="ttt3r_include_gt_view")
    parser.add_argument("--ttt3r_conf_power", type=float, default=1.0)
    parser.add_argument("--ttt3r_conf_floor", type=float, default=0.0)
    parser.add_argument("--ttt3r_geo_scale", type=float, default=1.0)
    parser.add_argument("--ttt3r_prox_strength", type=float, default=0.55)
    parser.add_argument("--ttt3r_preserve_strength", type=float, default=0.18)
    parser.add_argument("--ttt3r_edit_boost", type=float, default=1.0)
    parser.add_argument("--ttt3r_preserve_boost", type=float, default=1.0)
    parser.add_argument("--ttt3r_edit_min_mass", type=float, default=0.0)
    parser.add_argument("--ttt3r_preserve_min_mass", type=float, default=0.0)
    parser.add_argument("--ttt3r_adaptive_max_scale", type=float, default=3.0)
    parser.add_argument("--ttt3r_schedule_power", type=float, default=1.5)
    parser.add_argument("--ttt3r_input_h", type=int, default=384)
    parser.add_argument("--ttt3r_input_w", type=int, default=512)
    parser.add_argument("--ttt3r_edit_mask_quantile", type=float, default=0.95)
    parser.add_argument("--ttt3r_mode", type=str, default="proximal", choices=["proximal", "static_proxy", "velocity"])
    parser.add_argument("--ttt3r_gpu", type=int, default=0)
    parser.add_argument("--dump_intermediates", action="store_true")
    parser.add_argument("--dump_max_per_stage", type=int, default=32)
    parser.add_argument("--disable_densify", action="store_true")
    parser.add_argument("--freeze_geometry", action="store_true")
    parser.add_argument("--freeze_opacity", action="store_true")
    parser.add_argument("--max_optimizer_steps", type=int, default=-1)
    parser.add_argument("--optimizer_lr_scale", type=float, default=1.0)
    parser.add_argument("--fit_loss_mask_mode", type=str, default="none", choices=["none", "initial_edit"])
    parser.add_argument("--fit_loss_mask_quantile", type=float, default=0.80)
    parser.add_argument("--fit_loss_mask_bg", type=float, default=0.0)
    parser.add_argument("--fit_view_topk", type=int, default=-1)
    wargs, remaining = parser.parse_known_args()

    os.environ["HF_HOME"] = wargs.hf_home
    os.environ["HF_HUB_CACHE"] = str(Path(wargs.hf_home) / "hub")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ["EDITSPLAT_DEPTH_MODE"] = str(wargs.depth_mode)
    os.environ["EDITSPLAT_SKIP_AGT"] = "1" if bool(wargs.skip_agt) else "0"

    if torch.cuda.is_available():
        base_device = torch.device(f"cuda:{int(wargs.base_gpu)}")
        adapter_device = torch.device(f"cuda:{int(wargs.adapter_gpu)}")
        torch.cuda.set_device(base_device)
    else:
        base_device = torch.device("cpu")
        adapter_device = torch.device("cpu")

    patch_scene_load_from_checkpoint()
    if bool(wargs.aux_models_cpu):
        patch_aux_models_to_cpu()

    runtime = TTT3RRuntime(
        TTT3RConfig(
            enabled=True,
            repo_root=str(wargs.ttt3r_repo_root),
            checkpoint=str(wargs.ttt3r_checkpoint),
            model_update_type=str(wargs.ttt3r_model_update_type),
            support_views=int(wargs.ttt3r_support_views),
            support_stride=int(wargs.ttt3r_support_stride),
            include_gt_view=bool(wargs.ttt3r_include_gt_view),
            conf_power=float(wargs.ttt3r_conf_power),
            conf_floor=float(wargs.ttt3r_conf_floor),
            geo_scale=float(wargs.ttt3r_geo_scale),
            input_h=int(wargs.ttt3r_input_h),
            input_w=int(wargs.ttt3r_input_w),
            prox_strength=float(wargs.ttt3r_prox_strength),
            preserve_strength=float(wargs.ttt3r_preserve_strength),
            edit_boost=float(wargs.ttt3r_edit_boost),
            preserve_boost=float(wargs.ttt3r_preserve_boost),
            edit_min_mass=float(wargs.ttt3r_edit_min_mass),
            preserve_min_mass=float(wargs.ttt3r_preserve_min_mass),
            adaptive_max_scale=float(wargs.ttt3r_adaptive_max_scale),
            schedule_power=float(wargs.ttt3r_schedule_power),
            dump_intermediates=bool(wargs.dump_intermediates),
            dump_max_per_stage=int(wargs.dump_max_per_stage),
            edit_mask_quantile=float(wargs.ttt3r_edit_mask_quantile),
            mode=str(wargs.ttt3r_mode),
            ttt3r_gpu=int(wargs.ttt3r_gpu),
        )
    )
    patch_camera_dataset(runtime=runtime, head_k=int(wargs.head_k))
    patch_densify_control(disable_densify=bool(wargs.disable_densify))
    patch_geometry_freeze_control(
        freeze_geometry=bool(wargs.freeze_geometry),
        freeze_opacity=bool(wargs.freeze_opacity),
    )
    patch_optimizer_step_control(max_optimizer_steps=int(wargs.max_optimizer_steps))
    patch_optimizer_lr_scale(optimizer_lr_scale=float(wargs.optimizer_lr_scale))
    patch_fit_loss_control(
        runtime=runtime,
        fit_mask_mode=str(wargs.fit_loss_mask_mode),
        fit_mask_quantile=float(wargs.fit_loss_mask_quantile),
        fit_mask_bg=float(wargs.fit_loss_mask_bg),
        fit_view_topk=int(wargs.fit_view_topk),
    )
    runtime.init_model()

    run_parser = argparse.ArgumentParser(description="Editing Training script parameters")
    lp = ref.ModelParams(run_parser)
    op = ref.OptimizationParams(run_parser)
    pp = ref.PipelineParams(run_parser)
    ed = ref.EditingParams(run_parser)
    sdp_param = ref.ScoreDistillParams(run_parser)
    run_args = run_parser.parse_args(remaining)

    if hasattr(ref, "set_seed"):
        ref.set_seed(0)

    run_dtype = torch.bfloat16 if (base_device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    base_model_id = os.environ.get("EDITSPLAT_BASE_MODEL_ID", "black-forest-labs/FLUX.1-dev")
    pipeline = Editsplat_Pipeline.from_pretrained(
        base_model_id,
        torch_dtype=run_dtype,
        use_safetensors=True,
        token=os.environ.get("HF_TOKEN", None),
        cache_dir=str(Path(wargs.hf_home) / "hub"),
    )
    for name in ("transformer", "text_encoder", "text_encoder_2"):
        mod = getattr(pipeline, name, None)
        if mod is not None:
            try:
                mod.to("cpu")
            except Exception:
                pass
    vae = getattr(pipeline, "vae", None)
    if vae is not None:
        try:
            vae.to(str(base_device))
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dataset = lp.extract(run_args)
    opt = op.extract(run_args)
    pipe = pp.extract(run_args)
    edp = ed.extract(run_args)
    sdp_cfg = sdp_param.extract(run_args)

    method = str(getattr(edp, "flow_method", "flowedit")).strip().lower()
    model_key = str(getattr(edp, "flow_model_key", wargs.model_key)).strip()
    if method not in {"flowedit", "dnaedit"}:
        raise ValueError(
            f"run_sd35_ttt3r_proximal_wrapper.py currently supports flowedit/dnaedit only, got {method}"
        )

    if method == "flowedit":
        adapter = create_adapter(
            model_key=model_key,
            device=adapter_device,
            dtype=torch.float16,
            hf_token=wargs.hf_token if wargs.hf_token else None,
            cache_dir=str(Path(wargs.hf_home) / "hub"),
            override_model_id=wargs.model_id if wargs.model_id else None,
        )
        if not isinstance(adapter, SD3Adapter):
            raise TypeError(f"Expected SD3Adapter for SD3.5 branch, got {type(adapter).__name__}")
        backend_summary = lambda: {
            "backend": "wrapper_preloaded_sd3_adapter",
            "device": str(adapter.device),
            "mode": runtime.cfg.mode,
            "flow_method": method,
        }
    else:
        if runtime.cfg.mode != "static_proxy":
            raise ValueError(
                "dnaedit control is currently supported only with --ttt3r_mode static_proxy; "
                f"got {runtime.cfg.mode}"
            )
        backend = FlowEditCoreBackend(
            config=FlowBackendConfig(
                model_key=model_key,
                model_id=wargs.model_id if wargs.model_id else "",
                method=method,
                hf_home=str(wargs.hf_home),
                adapter_resize_side=int(wargs.adapter_resize_side),
                adapter_gpu=int(wargs.adapter_gpu),
                hf_token=wargs.hf_token if wargs.hf_token else "",
                dna_steps=int(getattr(edp, "flow_dna_steps", 40)),
                dna_src_guidance_scale=float(getattr(edp, "flow_dna_src_guidance_scale", 1.0)),
                dna_tar_guidance_scale=float(getattr(edp, "flow_dna_tar_guidance_scale", 3.5)),
                dna_t_start=int(getattr(edp, "flow_dna_t_start", 13)),
                dna_mvg=float(getattr(edp, "flow_dna_mvg", 0.8)),
            ),
            project_root=str(ROOT),
        )
        adapter = _DNAEditStaticProxyAdapter(
            backend=backend,
            device=adapter_device,
            resize_side=int(wargs.adapter_resize_side),
        )
        backend_summary = lambda: {
            "backend": "wrapper_core_backend_dnaedit_static_proxy",
            "device": str(adapter.device),
            "mode": runtime.cfg.mode,
            "flow_method": method,
            **backend.summarize(),
        }

    patch_edit_methods(adapter=adapter, runtime=runtime)
    pipeline._external_edit_backend = types.SimpleNamespace(
        device=adapter.device,
        summarize=backend_summary,
    )

    os.makedirs(dataset.model_path, exist_ok=True)
    ensure_point_cloud_link(dataset.model_path, getattr(dataset, "source_checkpoint", ""))
    runtime.set_dump_root(Path(dataset.model_path))

    with open(os.path.join(dataset.model_path, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(run_args), f, indent=2, ensure_ascii=False)
    meta = {
        "wrapper": "run_sd35_ttt3r_proximal_wrapper.py",
        "model_key": str(wargs.model_key),
        "adapter_gpu": int(wargs.adapter_gpu),
        "base_gpu": int(wargs.base_gpu),
        "head_k": int(wargs.head_k),
        "depth_mode": str(wargs.depth_mode),
        "skip_agt": bool(wargs.skip_agt),
        "aux_models_cpu": bool(wargs.aux_models_cpu),
        "disable_densify": bool(wargs.disable_densify),
        "freeze_geometry": bool(wargs.freeze_geometry),
        "freeze_opacity": bool(wargs.freeze_opacity),
        "max_optimizer_steps": int(wargs.max_optimizer_steps),
        "optimizer_lr_scale": float(wargs.optimizer_lr_scale),
        "fit_loss_mask_mode": str(wargs.fit_loss_mask_mode),
        "fit_loss_mask_quantile": float(wargs.fit_loss_mask_quantile),
        "fit_loss_mask_bg": float(wargs.fit_loss_mask_bg),
        "fit_view_topk": int(wargs.fit_view_topk),
        "ttt3r_cfg": vars(runtime.cfg),
    }
    with open(os.path.join(dataset.model_path, "ttt3r_proximal_wrapper_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    try:
        shutil.copyfile(ref.__file__, os.path.join(dataset.model_path, "train_frozen.py"))
    except Exception:
        pass

    if hasattr(pipeline, "set_sds_params"):
        try:
            pipeline.set_sds_params(sdp_cfg)
        except Exception:
            pass

    _ = pipeline(
        dataset=dataset,
        opt=opt,
        pipe=pipe,
        ed=edp,
        sdp=sdp_cfg,
    )


if __name__ == "__main__":
    main()
