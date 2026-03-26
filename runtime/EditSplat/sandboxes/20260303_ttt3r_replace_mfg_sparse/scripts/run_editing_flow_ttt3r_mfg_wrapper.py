#!/usr/bin/env python3
"""EditSplat wrapper: replace MFG target generation with TTT3R-assisted proxy.

Design goals:
1. Keep `run_editing_flow.py` unchanged.
2. Replace `Editsplat_Pipeline.edit_image_MFG` behavior via monkeypatch.
3. Favor sparse-view setups (`--max_train_views`) to reduce cost.
4. Preserve a fallback path to original MFG when TTT3R fails.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import sys
import types
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _ensure_imagereward_stub() -> None:
    """ImageReward import may fail in mixed envs; provide a deterministic stub."""
    try:
        import ImageReward  # noqa: F401
        return
    except Exception:
        pass

    class _DummyRewardModel:
        def inference_rank(self, prompt, images):
            del prompt
            n = len(images)
            return list(range(n)), [0.0] * n

    stub = types.ModuleType("ImageReward")

    def _load(name="ImageReward-v1.0", device="cpu", *args, **kwargs):
        del name, device, args, kwargs
        print("[WARN] ImageReward import failed; using dummy reward ranker.")
        return _DummyRewardModel()

    stub.load = _load  # type: ignore[attr-defined]
    sys.modules["ImageReward"] = stub


def _ensure_lpips_stub() -> None:
    if os.environ.get("EDITSPLAT_REAL_LPIPS", "0") == "1":
        return
    try:
        import lpips  # noqa: F401
        return
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
    print("[WARN] Using LPIPS stub (L1 fallback). Set EDITSPLAT_REAL_LPIPS=1 to disable.")
    sys.modules["lpips"] = stub


def _ensure_langsam_stub() -> None:
    if os.environ.get("EDITSPLAT_REAL_LANGSAM", "0") == "1":
        return
    try:
        import lang_sam  # noqa: F401
        return
    except Exception:
        pass

    stub = types.ModuleType("lang_sam")

    class _LangSAM:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            print("[WARN] Using LangSAM stub (full-image mask). Set EDITSPLAT_REAL_LANGSAM=1 to disable.")

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


ROOT = Path(__file__).resolve().parents[3]  # EditSplat/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arguments import EditingParams, ModelParams, OptimizationParams, PipelineParams, ScoreDistillParams
import run_editing_flow as ref
from run_editing_flow import Editsplat_Pipeline


@dataclass
class TTT3RConfig:
    enabled: bool
    repo_root: str
    checkpoint: str
    model_update_type: str
    strategy: str
    support_views: int
    support_stride: int
    include_gt_view: bool
    blend_strength: float
    conf_power: float
    flowedit_blend: float
    input_h: int
    input_w: int
    flow_output_scale_mode: str
    dump_flux_intermediates: bool
    dump_max_per_stage: int
    semantic_keep_weight: float
    semantic_keep_tar_guidance_scale: float
    semantic_keep_steps: int
    semantic_keep_n_max: int
    semantic_keep_seed_offset: int
    front_anchor_enable: bool
    front_anchor_view: int
    front_anchor_weight: float
    front_anchor_candidates: int
    front_anchor_tar_guidance_scale: float
    front_anchor_steps: int
    front_anchor_seed_offset: int


class TTT3RRuntime:
    def __init__(self, cfg: TTT3RConfig):
        self.cfg = cfg
        self.camera_list = None
        self.current_idx = 0
        self.model = None
        self.inference_fn = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dump_root: Optional[Path] = None
        self.reward_model = None
        self.front_anchor_01: Optional[torch.Tensor] = None
        self.stage_counts = {
            "initial_edit": 0,
            "mfg_edit": 0,
        }

    def set_dump_root(self, model_path: Path) -> None:
        if not bool(self.cfg.dump_flux_intermediates):
            return
        self.dump_root = model_path / "flux_intermediates"
        (self.dump_root / "initial_edit").mkdir(parents=True, exist_ok=True)
        (self.dump_root / "mfg_edit").mkdir(parents=True, exist_ok=True)
        meta = {
            "flow_output_scale_mode": str(self.cfg.flow_output_scale_mode),
            "dump_max_per_stage": int(self.cfg.dump_max_per_stage),
            "enabled_ttt3r": bool(self.cfg.enabled),
            "strategy": str(self.cfg.strategy),
            "model_update_type": str(self.cfg.model_update_type),
            "semantic_keep_weight": float(self.cfg.semantic_keep_weight),
            "front_anchor_enable": bool(self.cfg.front_anchor_enable),
            "front_anchor_view": int(self.cfg.front_anchor_view),
            "front_anchor_weight": float(self.cfg.front_anchor_weight),
            "front_anchor_candidates": int(self.cfg.front_anchor_candidates),
        }
        (self.dump_root / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    def init_model(self) -> None:
        if not self.cfg.enabled:
            return
        repo_root = Path(self.cfg.repo_root).resolve()
        if not repo_root.exists():
            raise FileNotFoundError(f"TTT3R repo not found: {repo_root}")
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from src.dust3r.model import ARCroco3DStereo
        from src.dust3r.inference import inference_recurrent_lighter

        ckpt = Path(self.cfg.checkpoint)
        if not ckpt.exists():
            raise FileNotFoundError(f"TTT3R checkpoint not found: {ckpt}")

        print(f"[TTT3R] loading model: {ckpt}")
        self.model = ARCroco3DStereo.from_pretrained(str(ckpt)).to(self.device)
        self.model.config.model_update_type = str(self.cfg.model_update_type)
        self.model.eval()
        self.inference_fn = inference_recurrent_lighter
        print(f"[TTT3R] ready. update_type={self.model.config.model_update_type}, device={self.device}")
        if int(self.cfg.front_anchor_candidates) > 1:
            try:
                self.reward_model = ref.RM.load("ImageReward-v1.0")
                print("[TTT3R] ImageReward loaded for front-anchor candidate ranking.")
            except Exception as exc:
                self.reward_model = None
                print(f"[TTT3R][WARN] ImageReward unavailable for candidate ranking: {exc}")

    @staticmethod
    def _to_01_bchw(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(0)
        y = x.detach().float()
        if y.min() < 0:
            y = (y + 1.0) * 0.5
        return y.clamp(0.0, 1.0)

    def _convert_flow_output(self, x: torch.Tensor) -> torch.Tensor:
        mode = str(self.cfg.flow_output_scale_mode)
        if mode == "raw_m11":
            return x.detach().float()
        if mode == "to_01":
            return self._to_01_bchw(x).detach().float()
        raise ValueError(f"Unknown flow_output_scale_mode: {mode}")

    @staticmethod
    def _to_m11_bchw(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(0)
        y = x.detach().float()
        return y.clamp(0.0, 1.0) * 2.0 - 1.0

    def _blend_outputs(self, base: torch.Tensor, ref: torch.Tensor, weight: float) -> torch.Tensor:
        w = float(max(0.0, min(1.0, weight)))
        out = (1.0 - w) * base + w * ref
        if str(self.cfg.flow_output_scale_mode) == "to_01":
            out = out.clamp(0.0, 1.0)
        return out

    def _tensor_to_pil(self, x: torch.Tensor) -> Image.Image:
        x01 = self._to_01_bchw(x).cpu()
        arr = (x01[0].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(arr)

    def _save_tensor_png(self, x: torch.Tensor, out: Path) -> None:
        x01 = self._to_01_bchw(x).cpu()
        y = x01[0]
        arr = (y.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        Image.fromarray(arr).save(out)

    def _dump_images(self, stage: str, payload: dict) -> None:
        if not bool(self.cfg.dump_flux_intermediates):
            return
        if self.dump_root is None:
            return
        max_keep = int(self.cfg.dump_max_per_stage)
        if max_keep > 0 and self.stage_counts.get(stage, 0) >= max_keep:
            return
        idx = int(self.current_idx)
        step_idx = self.stage_counts.get(stage, 0)
        stage_dir = self.dump_root / stage / f"view{idx:03d}_step{step_idx:04d}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        stats = {}
        for name, tensor in payload.items():
            if tensor is None or not isinstance(tensor, torch.Tensor):
                continue
            t = tensor.detach().float()
            stats[name] = {
                "min": float(t.min().item()),
                "max": float(t.max().item()),
                "mean": float(t.mean().item()),
                "shape": list(t.shape),
            }
            # Only dump image tensors as png; keep non-image tensors in stats.json.
            if t.ndim >= 3:
                self._save_tensor_png(tensor, stage_dir / f"{name}.png")
        (stage_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
        self.stage_counts[stage] = step_idx + 1

    def _preprocess_for_ttt3r(self, x: torch.Tensor) -> torch.Tensor:
        x01 = self._to_01_bchw(x).cpu()
        x01 = F.interpolate(
            x01,
            size=(int(self.cfg.input_h), int(self.cfg.input_w)),
            mode="bilinear",
            align_corners=False,
        )
        return x01

    def _build_view(self, x01: torch.Tensor, idx: int, update: bool = True, reset: bool = False):
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
            "update": torch.tensor([bool(update)]),
            "reset": torch.tensor([bool(reset)]),
        }

    def _support_indices(self, cur: int, total: int) -> List[int]:
        if total <= 1:
            return []
        stride = max(1, int(self.cfg.support_stride))
        max_k = max(1, int(self.cfg.support_views))
        picks: List[int] = []

        for s in range(1, total + 1):
            for sign in (-1, 1):
                j = cur + sign * s * stride
                if j < 0 or j >= total or j == cur:
                    continue
                if j not in picks:
                    picks.append(j)
                if len(picks) >= max_k:
                    return picks

        if len(picks) < max_k:
            for j in np.linspace(0, total - 1, num=min(total, max_k * 2), dtype=int).tolist():
                if j == cur:
                    continue
                if j not in picks:
                    picks.append(j)
                if len(picks) >= max_k:
                    break

        return picks[:max_k]

    def _ttt3r_proxy(self, image: torch.Tensor, mf_image_cond: torch.Tensor) -> torch.Tensor:
        if self.model is None or self.inference_fn is None:
            raise RuntimeError("TTT3R model is not initialized")
        if self.camera_list is None:
            raise RuntimeError("camera_list is not ready in runtime")

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
            x01 = self._preprocess_for_ttt3r(gt.unsqueeze(0))
            views.append(self._build_view(x01=x01, idx=vid, update=True, reset=False))
            vid += 1

        if bool(self.cfg.include_gt_view):
            x01 = self._preprocess_for_ttt3r(image)
            views.append(self._build_view(x01=x01, idx=vid, update=True, reset=False))
            vid += 1

        x_mf = self._preprocess_for_ttt3r(mf_image_cond)
        views.append(self._build_view(x01=x_mf, idx=vid, update=True, reset=False))

        with torch.no_grad():
            outputs, _state = self.inference_fn(views, self.model, self.device, verbose=False)

        pred_last = outputs["pred"][-1]
        rgb_m11 = pred_last["rgb"].permute(0, 3, 1, 2).detach().float()  # [1,3,H,W], [-1,1]
        conf = pred_last["conf"].detach().float().unsqueeze(1)  # [1,1,H,W]

        rgb01 = ((rgb_m11 + 1.0) * 0.5).clamp(0.0, 1.0)
        mf01 = x_mf

        cmin = conf.amin(dim=(-2, -1), keepdim=True)
        cmax = conf.amax(dim=(-2, -1), keepdim=True)
        conf_n = (conf - cmin) / (cmax - cmin + 1e-6)
        conf_n = conf_n.clamp(0.0, 1.0).pow(float(self.cfg.conf_power))
        conf_n = conf_n * float(self.cfg.blend_strength)
        conf_n = conf_n.clamp(0.0, 1.0)

        proxy = conf_n * rgb01 + (1.0 - conf_n) * mf01
        proxy = proxy.clamp(0.0, 1.0)
        if bool(self.cfg.front_anchor_enable) and self.front_anchor_01 is not None and int(self.current_idx) != int(self.cfg.front_anchor_view):
            wa = float(max(0.0, min(1.0, float(self.cfg.front_anchor_weight))))
            if wa > 0.0:
                anchor = self.front_anchor_01.to(proxy.device, dtype=proxy.dtype)
                if anchor.shape[-2:] != proxy.shape[-2:]:
                    anchor = F.interpolate(anchor, size=proxy.shape[-2:], mode="bilinear", align_corners=False)
                proxy = ((1.0 - wa) * proxy + wa * anchor).clamp(0.0, 1.0)
        return proxy


def patch_aux_models_to_cpu() -> None:
    orig_rm_load = ref.RM.load

    class _DummyRewardModel:
        def inference_rank(self, prompt, images):
            del prompt
            return list(range(len(images))), [0.0] * len(images)

    def _rm_load_cpu(name="ImageReward-v1.0", device="cpu", *args, **kwargs):
        kwargs["device"] = "cpu"
        try:
            return orig_rm_load(name, *args, **kwargs)
        except TypeError:
            kwargs.pop("device", None)
            try:
                return orig_rm_load(name, *args, **kwargs)
            except Exception as exc:
                print(f"[WARN] ImageReward load failed; using dummy reward ranker. exc={exc}")
                return _DummyRewardModel()
        except Exception as exc:
            print(f"[WARN] ImageReward load failed; using dummy reward ranker. exc={exc}")
            return _DummyRewardModel()

    use_real_langsam = os.environ.get("EDITSPLAT_REAL_LANGSAM", "0") == "1"
    if use_real_langsam:
        class _LangSAMCPU(ref.LangSAM):
            def __init__(self, sam_type="vit_h", ckpt_path=None):
                self.sam_type = sam_type
                self.device = torch.device("cpu")
                self.build_groundingdino()
                self.build_sam(ckpt_path)
    else:
        class _LangSAMCPU:
            def __init__(self, *args, **kwargs):
                del args, kwargs
                self.device = torch.device("cpu")
                print("[WARN] Using LangSAM CPU stub (full-image mask). Set EDITSPLAT_REAL_LANGSAM=1 to use real LangSAM.")

            def predict(self, image_pil, text_prompt):
                del text_prompt
                w, h = image_pil.size
                mask = torch.ones((1, h, w), dtype=torch.float32)
                return mask, None, None, None

    ref.RM.load = _rm_load_cpu
    ref.LangSAM = _LangSAMCPU


def parse_iter_from_checkpoint(path: str) -> int:
    m = re.search(r"chkpnt(\d+)\.pth$", str(path))
    return int(m.group(1)) if m else 7000


def ensure_point_cloud_link(model_path: str, source_checkpoint: str) -> None:
    model_dir = Path(model_path)
    src_ckpt = Path(source_checkpoint).resolve()
    src_pretrained_dir = src_ckpt.parent
    src_pc = src_pretrained_dir / "point_cloud"
    dst_pc = model_dir / "point_cloud"
    if dst_pc.exists():
        return
    if not src_pc.exists():
        raise FileNotFoundError(f"Missing source point_cloud dir: {src_pc}")
    model_dir.mkdir(parents=True, exist_ok=True)
    os.symlink(src_pc, dst_pc)


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


def patch_camera_dataset(runtime: TTT3RRuntime, max_train_views: int) -> None:
    base_dataset_cls = ref.CameraDataset
    orig_init = base_dataset_cls.__init__
    orig_getitem = base_dataset_cls.__getitem__

    def _init(self, scene):
        orig_init(self, scene)
        n0 = len(self.camera_list)
        if int(max_train_views) > 0:
            self.camera_list = self.camera_list[: int(max_train_views)]
            print(f"[SPARSE] CameraDataset truncated: {n0} -> {len(self.camera_list)} views")
        runtime.camera_list = self.camera_list

    def _getitem(self, idx):
        runtime.current_idx = int(idx)
        return orig_getitem(self, idx)

    base_dataset_cls.__init__ = _init
    base_dataset_cls.__getitem__ = _getitem


def patch_edit_image_mfg(runtime: TTT3RRuntime) -> None:
    orig_edit_image = Editsplat_Pipeline.edit_image
    orig_edit_image_mfg = Editsplat_Pipeline.edit_image_MFG

    def _semantic_ref_from_src(
        self,
        image: torch.Tensor,
        src_prompt: str,
        tar_prompt: str,
        negative_prompt: Optional[str],
        diffusion_steps: int,
        n_avg: int,
        src_guidance_scale: float,
        tar_guidance_scale: float,
        n_min: int,
        n_max: int,
        seed: int,
        mask_S: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sem_steps = int(runtime.cfg.semantic_keep_steps) if int(runtime.cfg.semantic_keep_steps) > 0 else int(diffusion_steps)
        sem_nmax = int(runtime.cfg.semantic_keep_n_max) if int(runtime.cfg.semantic_keep_n_max) >= 0 else int(n_max)
        sem_tar = (
            float(runtime.cfg.semantic_keep_tar_guidance_scale)
            if float(runtime.cfg.semantic_keep_tar_guidance_scale) > 0
            else float(tar_guidance_scale)
        )
        sem_seed = int(seed) + int(runtime.cfg.semantic_keep_seed_offset)
        sem_raw = orig_edit_image(
            self,
            image=image,
            src_prompt=src_prompt,
            tar_prompt=tar_prompt,
            negative_prompt=negative_prompt,
            diffusion_steps=sem_steps,
            n_avg=n_avg,
            src_guidance_scale=src_guidance_scale,
            tar_guidance_scale=sem_tar,
            n_min=n_min,
            n_max=sem_nmax,
            seed=sem_seed,
            lambda_S=0.0,
            mask_S=mask_S,
        )
        sem_final = runtime._convert_flow_output(sem_raw).to(image.device, dtype=torch.float32)
        return sem_raw, sem_final

    def _select_front_anchor(
        self,
        image: torch.Tensor,
        src_prompt: str,
        tar_prompt: str,
        negative_prompt: Optional[str],
        diffusion_steps: int,
        n_avg: int,
        src_guidance_scale: float,
        tar_guidance_scale: float,
        n_min: int,
        n_max: int,
        seed: int,
        mask_S: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[float]]:
        k = max(1, int(runtime.cfg.front_anchor_candidates))
        front_steps = int(runtime.cfg.front_anchor_steps) if int(runtime.cfg.front_anchor_steps) > 0 else int(diffusion_steps)
        front_tar = (
            float(runtime.cfg.front_anchor_tar_guidance_scale)
            if float(runtime.cfg.front_anchor_tar_guidance_scale) > 0
            else float(tar_guidance_scale)
        )

        cand_raw: List[torch.Tensor] = []
        cand_final: List[torch.Tensor] = []
        for i in range(k):
            seed_i = int(seed) + int(runtime.cfg.front_anchor_seed_offset) + i
            raw = orig_edit_image(
                self,
                image=image,
                src_prompt=src_prompt,
                tar_prompt=tar_prompt,
                negative_prompt=negative_prompt,
                diffusion_steps=front_steps,
                n_avg=n_avg,
                src_guidance_scale=src_guidance_scale,
                tar_guidance_scale=front_tar,
                n_min=n_min,
                n_max=n_max,
                seed=seed_i,
                lambda_S=0.0,
                mask_S=mask_S,
            )
            final = runtime._convert_flow_output(raw).to(image.device, dtype=torch.float32)
            cand_raw.append(raw)
            cand_final.append(final)

        if len(cand_final) == 1:
            return cand_final[0], cand_raw[0], None

        best_idx = 0
        best_score = None
        try:
            if runtime.reward_model is not None:
                pil_list = [runtime._tensor_to_pil(x) for x in cand_final]
                ranking, rewards = runtime.reward_model.inference_rank(tar_prompt, pil_list)
                best_idx = int(ranking[0]) - 1 if ranking else 0
                best_idx = max(0, min(len(cand_final) - 1, best_idx))
                if isinstance(rewards, (list, tuple)) and len(rewards) >= len(cand_final):
                    best_score = float(rewards[best_idx])
            else:
                src01 = runtime._to_01_bchw(image)
                scores = [float((runtime._to_01_bchw(x) - src01).abs().mean().item()) for x in cand_final]
                best_idx = int(np.argmax(np.array(scores)))
                best_score = float(scores[best_idx])
        except Exception as exc:
            print(f"[SEM][WARN] front-anchor ranking failed, fallback to first candidate: {exc}")
            best_idx = 0

        return cand_final[best_idx], cand_raw[best_idx], best_score

    def _apply_semantic_and_anchor(
        self,
        base_out: torch.Tensor,
        image: torch.Tensor,
        src_prompt: str,
        tar_prompt: str,
        negative_prompt: Optional[str],
        diffusion_steps: int,
        n_avg: int,
        src_guidance_scale: float,
        tar_guidance_scale: float,
        n_min: int,
        n_max: int,
        seed: int,
        mask_S: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        sem_raw = None
        sem_final = None
        anchor_ref = None
        out = base_out

        if float(runtime.cfg.semantic_keep_weight) > 0.0:
            sem_raw, sem_final = _semantic_ref_from_src(
                self=self,
                image=image,
                src_prompt=src_prompt,
                tar_prompt=tar_prompt,
                negative_prompt=negative_prompt,
                diffusion_steps=diffusion_steps,
                n_avg=n_avg,
                src_guidance_scale=src_guidance_scale,
                tar_guidance_scale=tar_guidance_scale,
                n_min=n_min,
                n_max=n_max,
                seed=seed,
                mask_S=mask_S,
            )
            out = runtime._blend_outputs(out, sem_final, float(runtime.cfg.semantic_keep_weight))

        if bool(runtime.cfg.front_anchor_enable) and runtime.front_anchor_01 is not None and int(runtime.current_idx) != int(runtime.cfg.front_anchor_view):
            wa = float(max(0.0, min(1.0, float(runtime.cfg.front_anchor_weight))))
            if wa > 0.0:
                anchor_ref = runtime.front_anchor_01.to(out.device, dtype=torch.float32)
                if anchor_ref.shape[-2:] != out.shape[-2:]:
                    anchor_ref = F.interpolate(anchor_ref, size=out.shape[-2:], mode="bilinear", align_corners=False)
                if str(runtime.cfg.flow_output_scale_mode) == "raw_m11":
                    anchor_ref = runtime._to_m11_bchw(anchor_ref)
                out = runtime._blend_outputs(out, anchor_ref, wa)

        return out, sem_raw, sem_final, anchor_ref

    def _patched_edit_image(
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
        out_raw = orig_edit_image(
            self,
            image=image,
            src_prompt=src_prompt,
            tar_prompt=tar_prompt,
            negative_prompt=negative_prompt,
            diffusion_steps=diffusion_steps,
            n_avg=n_avg,
            src_guidance_scale=src_guidance_scale,
            tar_guidance_scale=tar_guidance_scale,
            n_min=n_min,
            n_max=n_max,
            seed=seed,
            lambda_S=lambda_S,
            mask_S=mask_S,
        )
        out_final = runtime._convert_flow_output(out_raw).to(image.device, dtype=torch.float32)

        front_anchor_raw = None
        front_anchor_score = None
        if bool(runtime.cfg.front_anchor_enable) and int(runtime.current_idx) == int(runtime.cfg.front_anchor_view):
            try:
                front_anchor_final, front_anchor_raw, front_anchor_score = _select_front_anchor(
                    self=self,
                    image=image,
                    src_prompt=src_prompt,
                    tar_prompt=tar_prompt,
                    negative_prompt=negative_prompt,
                    diffusion_steps=diffusion_steps,
                    n_avg=n_avg,
                    src_guidance_scale=src_guidance_scale,
                    tar_guidance_scale=tar_guidance_scale,
                    n_min=n_min,
                    n_max=n_max,
                    seed=seed,
                    mask_S=mask_S,
                )
                out_final = front_anchor_final.to(image.device, dtype=torch.float32)
                runtime.front_anchor_01 = runtime._to_01_bchw(out_final).to(image.device, dtype=torch.float32)
            except Exception as exc:
                print(f"[SEM][WARN] front-anchor selection failed, using default edit: {exc}")
                runtime.front_anchor_01 = runtime._to_01_bchw(out_final).to(image.device, dtype=torch.float32)

        runtime._dump_images(
            stage="initial_edit",
            payload={
                "input": image,
                "flow_raw": out_raw,
                "flow_final": out_final,
                "front_anchor_raw": front_anchor_raw,
                "front_anchor_score": torch.tensor([front_anchor_score], dtype=torch.float32, device=out_final.device)
                if front_anchor_score is not None
                else None,
            },
        )
        return out_final

    def _patched_edit_image_mfg(
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
        stage_payload = {
            "input": image,
            "mf_cond": MF_image_cond,
        }
        if not runtime.cfg.enabled:
            out_raw = orig_edit_image_mfg(
                self,
                image=image,
                MF_image_cond=MF_image_cond,
                src_prompt=src_prompt,
                tar_prompt=tar_prompt,
                negative_prompt=negative_prompt,
                diffusion_steps=diffusion_steps,
                n_avg=n_avg,
                src_guidance_scale=src_guidance_scale,
                tar_guidance_scale=tar_guidance_scale,
                n_min=n_min,
                n_max=n_max,
                seed=seed,
                lambda_S=lambda_S,
                lambda_M=lambda_M,
                mask_S=mask_S,
                mask_M=mask_M,
            )
            base_out = runtime._convert_flow_output(out_raw).to(image.device, dtype=torch.float32)
            stage_payload.update({"mfg_raw": out_raw, "base_out": base_out})
            out_final, sem_raw, sem_final, anchor_ref = _apply_semantic_and_anchor(
                self=self,
                base_out=base_out,
                image=image,
                src_prompt=src_prompt,
                tar_prompt=tar_prompt,
                negative_prompt=negative_prompt,
                diffusion_steps=diffusion_steps,
                n_avg=n_avg,
                src_guidance_scale=src_guidance_scale,
                tar_guidance_scale=tar_guidance_scale,
                n_min=n_min,
                n_max=n_max,
                seed=seed,
                mask_S=mask_S,
            )
            stage_payload.update({"semantic_raw": sem_raw, "semantic_ref": sem_final, "anchor_ref": anchor_ref, "final": out_final})
            runtime._dump_images(stage="mfg_edit", payload=stage_payload)
            return out_final

        try:
            proxy = runtime._ttt3r_proxy(image=image, mf_image_cond=MF_image_cond)
            proxy = F.interpolate(
                proxy,
                size=(image.shape[-2], image.shape[-1]),
                mode="bilinear",
                align_corners=False,
            ).to(image.device, dtype=torch.float32)
            stage_payload["proxy"] = proxy

            strategy = str(runtime.cfg.strategy)
            if strategy == "proxy_only":
                base_out = runtime._convert_flow_output(proxy).to(image.device, dtype=torch.float32)
                stage_payload["base_out"] = base_out
            else:
                flow_raw = orig_edit_image(
                    self,
                    image=proxy,
                    src_prompt=src_prompt,
                    tar_prompt=tar_prompt,
                    negative_prompt=negative_prompt,
                    diffusion_steps=diffusion_steps,
                    n_avg=n_avg,
                    src_guidance_scale=src_guidance_scale,
                    tar_guidance_scale=tar_guidance_scale,
                    n_min=n_min,
                    n_max=n_max,
                    seed=seed,
                    lambda_S=0.0,
                    mask_S=None,
                )
                flow_final = runtime._convert_flow_output(flow_raw).to(proxy.device, dtype=torch.float32)
                stage_payload.update({"flow_raw": flow_raw, "flow_final": flow_final})

                if strategy == "proxy_flowedit_blend":
                    a = float(max(0.0, min(1.0, float(runtime.cfg.flowedit_blend))))
                    base_out = runtime._blend_outputs(flow_final, runtime._convert_flow_output(proxy).to(proxy.device), 1.0 - a)
                else:
                    base_out = flow_final
                stage_payload["base_out"] = base_out

            out_final, sem_raw, sem_final, anchor_ref = _apply_semantic_and_anchor(
                self=self,
                base_out=stage_payload["base_out"],
                image=image,
                src_prompt=src_prompt,
                tar_prompt=tar_prompt,
                negative_prompt=negative_prompt,
                diffusion_steps=diffusion_steps,
                n_avg=n_avg,
                src_guidance_scale=src_guidance_scale,
                tar_guidance_scale=tar_guidance_scale,
                n_min=n_min,
                n_max=n_max,
                seed=seed,
                mask_S=mask_S,
            )
            stage_payload.update({"semantic_raw": sem_raw, "semantic_ref": sem_final, "anchor_ref": anchor_ref, "final": out_final})
            runtime._dump_images(stage="mfg_edit", payload=stage_payload)
            return out_final
        except Exception as exc:
            print(f"[TTT3R][WARN] MFG replacement failed, fallback to original MFG: {exc}")
            out_raw = orig_edit_image_mfg(
                self,
                image=image,
                MF_image_cond=MF_image_cond,
                src_prompt=src_prompt,
                tar_prompt=tar_prompt,
                negative_prompt=negative_prompt,
                diffusion_steps=diffusion_steps,
                n_avg=n_avg,
                src_guidance_scale=src_guidance_scale,
                tar_guidance_scale=tar_guidance_scale,
                n_min=n_min,
                n_max=n_max,
                seed=seed,
                lambda_S=lambda_S,
                lambda_M=lambda_M,
                mask_S=mask_S,
                mask_M=mask_M,
            )
            base_out = runtime._convert_flow_output(out_raw).to(image.device, dtype=torch.float32)
            stage_payload.update({"fallback_raw": out_raw, "base_out": base_out})
            out_final, sem_raw, sem_final, anchor_ref = _apply_semantic_and_anchor(
                self=self,
                base_out=base_out,
                image=image,
                src_prompt=src_prompt,
                tar_prompt=tar_prompt,
                negative_prompt=negative_prompt,
                diffusion_steps=diffusion_steps,
                n_avg=n_avg,
                src_guidance_scale=src_guidance_scale,
                tar_guidance_scale=tar_guidance_scale,
                n_min=n_min,
                n_max=n_max,
                seed=seed,
                mask_S=mask_S,
            )
            stage_payload.update({"semantic_raw": sem_raw, "semantic_ref": sem_final, "anchor_ref": anchor_ref, "final": out_final})
            runtime._dump_images(stage="mfg_edit", payload=stage_payload)
            return out_final

    Editsplat_Pipeline.edit_image = _patched_edit_image
    Editsplat_Pipeline.edit_image_MFG = _patched_edit_image_mfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    parser = ArgumentParser(description="EditSplat wrapper with TTT3R-based MFG replacement")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    ed = EditingParams(parser)
    _ = ScoreDistillParams(parser)

    parser.add_argument("--aux_models_cpu", action="store_true", default=True)
    parser.add_argument("--no_aux_models_cpu", action="store_false", dest="aux_models_cpu")
    parser.add_argument("--max_train_views", type=int, default=8)

    parser.add_argument("--disable_ttt3r_mfg", action="store_true")
    parser.add_argument("--ttt3r_repo_root", type=str, default="/dev-vepfs/rc_wu/rc_wu/edit/TTT3R")
    parser.add_argument("--ttt3r_checkpoint", type=str, default="/dev-vepfs/rc_wu/rc_wu/edit/TTT3R/src/cut3r_512_dpt_4_64.pth")
    parser.add_argument("--ttt3r_model_update_type", type=str, default="ttt3r", choices=["cut3r", "ttt3r"])
    parser.add_argument("--ttt3r_strategy", type=str, default="proxy_flowedit", choices=["proxy_only", "proxy_flowedit", "proxy_flowedit_blend"])
    parser.add_argument("--ttt3r_support_views", type=int, default=3)
    parser.add_argument("--ttt3r_support_stride", type=int, default=2)
    parser.add_argument("--ttt3r_include_gt_view", action="store_true", default=True)
    parser.add_argument("--ttt3r_no_include_gt_view", action="store_false", dest="ttt3r_include_gt_view")
    parser.add_argument("--ttt3r_blend_strength", type=float, default=0.65)
    parser.add_argument("--ttt3r_conf_power", type=float, default=1.0)
    parser.add_argument("--ttt3r_flowedit_blend", type=float, default=0.6)
    parser.add_argument("--ttt3r_input_h", type=int, default=384)
    parser.add_argument("--ttt3r_input_w", type=int, default=512)
    parser.add_argument("--flow_output_scale_mode", type=str, default="raw_m11", choices=["raw_m11", "to_01"])
    parser.add_argument("--dump_flux_intermediates", action="store_true", default=False)
    parser.add_argument("--no_dump_flux_intermediates", action="store_false", dest="dump_flux_intermediates")
    parser.add_argument("--dump_max_per_stage", type=int, default=200)
    parser.add_argument("--semantic_keep_weight", type=float, default=0.0)
    parser.add_argument("--semantic_keep_tar_guidance_scale", type=float, default=-1.0)
    parser.add_argument("--semantic_keep_steps", type=int, default=0)
    parser.add_argument("--semantic_keep_n_max", type=int, default=-1)
    parser.add_argument("--semantic_keep_seed_offset", type=int, default=20000)
    parser.add_argument("--front_anchor_enable", action="store_true", default=False)
    parser.add_argument("--front_anchor_view", type=int, default=0)
    parser.add_argument("--front_anchor_weight", type=float, default=0.0)
    parser.add_argument("--front_anchor_candidates", type=int, default=1)
    parser.add_argument("--front_anchor_tar_guidance_scale", type=float, default=-1.0)
    parser.add_argument("--front_anchor_steps", type=int, default=0)
    parser.add_argument("--front_anchor_seed_offset", type=int, default=30000)
    parser.add_argument("--skip_final_render_sets", action="store_true", default=False)

    args = parser.parse_args(sys.argv[1:])

    set_seed(0)
    patch_scene_load_from_checkpoint()
    if bool(args.aux_models_cpu):
        patch_aux_models_to_cpu()

    tcfg = TTT3RConfig(
        enabled=not bool(args.disable_ttt3r_mfg),
        repo_root=str(args.ttt3r_repo_root),
        checkpoint=str(args.ttt3r_checkpoint),
        model_update_type=str(args.ttt3r_model_update_type),
        strategy=str(args.ttt3r_strategy),
        support_views=int(args.ttt3r_support_views),
        support_stride=int(args.ttt3r_support_stride),
        include_gt_view=bool(args.ttt3r_include_gt_view),
        blend_strength=float(args.ttt3r_blend_strength),
        conf_power=float(args.ttt3r_conf_power),
        flowedit_blend=float(args.ttt3r_flowedit_blend),
        input_h=int(args.ttt3r_input_h),
        input_w=int(args.ttt3r_input_w),
        flow_output_scale_mode=str(args.flow_output_scale_mode),
        dump_flux_intermediates=bool(args.dump_flux_intermediates),
        dump_max_per_stage=int(args.dump_max_per_stage),
        semantic_keep_weight=float(args.semantic_keep_weight),
        semantic_keep_tar_guidance_scale=float(args.semantic_keep_tar_guidance_scale),
        semantic_keep_steps=int(args.semantic_keep_steps),
        semantic_keep_n_max=int(args.semantic_keep_n_max),
        semantic_keep_seed_offset=int(args.semantic_keep_seed_offset),
        front_anchor_enable=bool(args.front_anchor_enable),
        front_anchor_view=int(args.front_anchor_view),
        front_anchor_weight=float(args.front_anchor_weight),
        front_anchor_candidates=int(args.front_anchor_candidates),
        front_anchor_tar_guidance_scale=float(args.front_anchor_tar_guidance_scale),
        front_anchor_steps=int(args.front_anchor_steps),
        front_anchor_seed_offset=int(args.front_anchor_seed_offset),
    )
    runtime = TTT3RRuntime(cfg=tcfg)
    patch_camera_dataset(runtime=runtime, max_train_views=int(args.max_train_views))
    patch_edit_image_mfg(runtime=runtime)
    if bool(args.skip_final_render_sets):
        def _skip_render_sets(dataset, iteration, pipe, skip_train, skip_test, video):
            del dataset, iteration, pipe, skip_train, skip_test, video
            print("[WRAPPER] skip_final_render_sets=1; skipping in-process render_sets.")

        ref.render_sets = _skip_render_sets
    runtime.init_model()

    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model_id = os.environ.get("EDITSPLAT_BASE_MODEL_ID", "black-forest-labs/FLUX.1-dev")
    pipeline = Editsplat_Pipeline.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        token=os.environ.get("HF_TOKEN", None),
    ).to(device)

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)
    edp = ed.extract(args)

    ensure_point_cloud_link(dataset.model_path, dataset.source_checkpoint)
    out_model_dir = Path(dataset.model_path)
    out_model_dir.mkdir(parents=True, exist_ok=True)
    runtime.set_dump_root(out_model_dir)

    with open(out_model_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    meta = {
        "max_train_views": int(args.max_train_views),
        "ttt3r_cfg": vars(tcfg),
        "aux_models_cpu": bool(args.aux_models_cpu),
    }
    with open(out_model_dir / "ttt3r_mfg_wrapper_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    shutil.copyfile(__file__, out_model_dir / "train_frozen.py")

    _ = pipeline(
        dataset=dataset,
        opt=opt,
        pipe=pipe,
        ed=edp,
    )
    print("\nEditing complete.")


if __name__ == "__main__":
    main()
