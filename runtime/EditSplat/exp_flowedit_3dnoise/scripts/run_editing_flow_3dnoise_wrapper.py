#!/usr/bin/env python3
"""Full-pipeline launcher with 3DGS-anchored noise for FlowEdit sampling.

This wrapper keeps `run_editing_flow.py` untouched and applies runtime monkeypatches:
1) Preserve baseline compatibility patches (`sdp` call mismatch avoidance via wrapper entry,
   Scene checkpoint loading, optional CPU auxiliary models).
2) Record current camera index through patched `CameraDataset`.
3) Inject 3D-consistent noise by patching `torch.randn_like` only inside
   `edit_image` / `edit_image_MFG` execution.
4) Optional pre-optimization of sparse 3D noise field before the full pipeline.
"""

import json
import math
import os
import random
import re
import shutil
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]  # EditSplat/
EXP_ROOT = Path(__file__).resolve().parents[1]  # exp_flowedit_3dnoise/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from arguments import EditingParams, ModelParams, OptimizationParams, PipelineParams, ScoreDistillParams
import run_editing_flow as ref
from run_editing_flow import Editsplat_Pipeline
from scene import GaussianModel
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos

from src.flowedit_3dnoise_core import PackedView, pack_image_latents
from src.noise_field import SparseGaussianNoiseField, SparseNoiseFieldConfig
from src.noise_optimizer import NoiseOptimizeConfig, optimize_noise_field


@dataclass
class NoiseWrapperConfig:
    enabled: bool = True
    anchor_mode: str = "coverage_opacity"
    init_mode: str = "coarse_smooth"
    max_anchors: int = 60000
    voxel_res: int = 96
    coarse_res: int = 32
    hybrid_ratio: float = 0.5
    harmonic_mix: float = 0.7
    seed: int = 0
    mix_ratio: float = 0.9
    apply_on_mfg: bool = True


class NoiseRuntime:
    def __init__(self, cfg: NoiseWrapperConfig):
        self.cfg = cfg
        self.camera_list = None
        self.current_idx: Optional[int] = None
        self.field: Optional[SparseGaussianNoiseField] = None
        self.xyz: Optional[torch.Tensor] = None
        self.opacity: Optional[torch.Tensor] = None
        self.warn_count = 0

    def set_gaussians(self, xyz: torch.Tensor, opacity: torch.Tensor) -> None:
        self.xyz = xyz.detach()
        self.opacity = opacity.detach()

    def _build_field(self, channels: int, device: torch.device) -> None:
        if self.xyz is None or self.opacity is None:
            raise RuntimeError("NoiseRuntime gaussians are not initialized")

        field_cfg = SparseNoiseFieldConfig(
            max_anchors=int(self.cfg.max_anchors),
            voxel_res=int(self.cfg.voxel_res),
            coarse_res=int(self.cfg.coarse_res),
            anchor_mode=str(self.cfg.anchor_mode),
            init_mode=str(self.cfg.init_mode),
            hybrid_ratio=float(self.cfg.hybrid_ratio),
            harmonic_mix=float(self.cfg.harmonic_mix),
            seed=int(self.cfg.seed),
        )
        self.field = SparseGaussianNoiseField.from_gaussians(
            xyz=self.xyz.to(device),
            opacity=self.opacity.to(device),
            channels=int(channels),
            cfg=field_cfg,
        ).to(device)
        print(
            f"[3DNOISE] field initialized: anchors={self.field.num_anchors}, "
            f"channels={self.field.channels}, anchor_mode={field_cfg.anchor_mode}, init_mode={field_cfg.init_mode}"
        )

    def ensure_field(self, channels: int, device: torch.device) -> SparseGaussianNoiseField:
        if self.field is None:
            self._build_field(channels=channels, device=device)
        elif self.field.channels != int(channels):
            # Rebuild if token channel changes.
            print(f"[3DNOISE] channel changed ({self.field.channels}->{channels}), rebuilding field")
            self._build_field(channels=channels, device=device)
        elif self.field.noise.device != device:
            self.field = self.field.to(device)
        return self.field

    @staticmethod
    def _factor_hw(num_tokens: int) -> Tuple[int, int]:
        h = int(round(math.sqrt(float(num_tokens))))
        h = max(1, h)
        while h > 1 and num_tokens % h != 0:
            h -= 1
        w = max(1, num_tokens // h)
        return int(h), int(w)

    def render_noise_like(self, ref_noise: torch.Tensor, orig_randn_like: Callable) -> torch.Tensor:
        rnd = orig_randn_like(ref_noise)

        if not self.cfg.enabled:
            return rnd
        if self.camera_list is None or self.current_idx is None:
            return rnd
        if ref_noise.ndim != 3 or ref_noise.shape[0] != 1:
            return rnd

        try:
            field = self.ensure_field(channels=int(ref_noise.shape[-1]), device=ref_noise.device)
            cam = self.camera_list[int(self.current_idx)]
            token_h, token_w = self._factor_hw(int(ref_noise.shape[1]))
            n3d, _ = field.render_to_tokens(camera=cam, token_h=token_h, token_w=token_w, normalize=True)
            n3d = n3d.to(device=ref_noise.device, dtype=ref_noise.dtype)
            if n3d.shape != ref_noise.shape:
                return rnd
            m = float(self.cfg.mix_ratio)
            m = max(0.0, min(1.0, m))
            return m * n3d + math.sqrt(max(1e-6, 1.0 - m * m)) * rnd
        except Exception as exc:  # pragma: no cover - runtime fallback
            if self.warn_count < 10:
                print(f"[3DNOISE][WARN] fallback to random noise: {exc}")
                self.warn_count += 1
            return rnd

    def call_with_noise_patch(self, fn: Callable):
        if not self.cfg.enabled:
            return fn()

        orig_randn_like = torch.randn_like

        def _patched_randn_like(input, *args, **kwargs):
            if args or kwargs:
                rnd = orig_randn_like(input, *args, **kwargs)
                if not self.cfg.enabled or input.ndim != 3 or input.shape[0] != 1:
                    return rnd
                try:
                    field = self.ensure_field(channels=int(input.shape[-1]), device=input.device)
                    cam = self.camera_list[int(self.current_idx)] if (self.camera_list is not None and self.current_idx is not None) else None
                    if cam is None:
                        return rnd
                    token_h, token_w = self._factor_hw(int(input.shape[1]))
                    n3d, _ = field.render_to_tokens(camera=cam, token_h=token_h, token_w=token_w, normalize=True)
                    n3d = n3d.to(device=input.device, dtype=input.dtype)
                    if n3d.shape != input.shape:
                        return rnd
                    m = float(self.cfg.mix_ratio)
                    m = max(0.0, min(1.0, m))
                    return m * n3d + math.sqrt(max(1e-6, 1.0 - m * m)) * rnd
                except Exception as exc:
                    if self.warn_count < 10:
                        print(f"[3DNOISE][WARN] fallback to random noise: {exc}")
                        self.warn_count += 1
                    return rnd
            return self.render_noise_like(ref_noise=input, orig_randn_like=orig_randn_like)

        torch.randn_like = _patched_randn_like
        try:
            return fn()
        finally:
            torch.randn_like = orig_randn_like


def patch_aux_models_to_cpu() -> None:
    """Keep FlowEdit core on GPU while moving auxiliary models to CPU."""
    orig_rm_load = ref.RM.load

    def _rm_load_cpu(name="ImageReward-v1.0", device="cpu", *args, **kwargs):
        kwargs["device"] = "cpu"
        return orig_rm_load(name, *args, **kwargs)

    class _LangSAMCPU(ref.LangSAM):
        def __init__(self, sam_type="vit_h", ckpt_path=None):
            self.sam_type = sam_type
            self.device = torch.device("cpu")
            self.build_groundingdino()
            self.build_sam(ckpt_path)

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


def patch_camera_dataset_runtime(runtime: NoiseRuntime) -> None:
    base_dataset_cls = ref.CameraDataset
    orig_init = base_dataset_cls.__init__
    orig_getitem = base_dataset_cls.__getitem__

    def _init(self, scene):
        orig_init(self, scene)
        runtime.camera_list = self.camera_list

    def _getitem(self, idx):
        runtime.current_idx = int(idx)
        return orig_getitem(self, idx)

    base_dataset_cls.__init__ = _init
    base_dataset_cls.__getitem__ = _getitem


def patch_edit_methods_with_3dnoise(runtime: NoiseRuntime) -> None:
    orig_edit_image = Editsplat_Pipeline.edit_image
    orig_edit_image_mfg = Editsplat_Pipeline.edit_image_MFG

    def _edit_image(self, *args, **kwargs):
        return runtime.call_with_noise_patch(lambda: orig_edit_image(self, *args, **kwargs))

    def _edit_image_mfg(self, *args, **kwargs):
        if runtime.cfg.apply_on_mfg:
            return runtime.call_with_noise_patch(lambda: orig_edit_image_mfg(self, *args, **kwargs))
        return orig_edit_image_mfg(self, *args, **kwargs)

    Editsplat_Pipeline.edit_image = _edit_image
    Editsplat_Pipeline.edit_image_MFG = _edit_image_mfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dataset_cfg(source_path: str, model_path: str, data_device: str):
    parser = ArgumentParser(add_help=False)
    mp = ModelParams(parser)
    args = parser.parse_args(
        [
            "--source_path",
            source_path,
            "--model_path",
            model_path,
            "--data_device",
            data_device,
            "--images",
            "images",
            "--resolution",
            "1",
        ]
    )
    return mp.extract(args)


def build_opt_cfg_for_restore():
    parser = ArgumentParser(add_help=False)
    op = OptimizationParams(parser)
    args = parser.parse_args([])
    return op.extract(args)


def build_packed_views(pipe, train_cameras, view_indices: List[int], device: torch.device) -> List[PackedView]:
    packed_views: List[PackedView] = []
    for idx in view_indices:
        cam = train_cameras[int(idx)]
        src_img = cam.gt_image.unsqueeze(0).to(device)
        x_src_packed, img_ids, oh, ow, th, tw = pack_image_latents(pipe, src_img, device=device)
        packed_views.append(
            PackedView(
                camera=cam,
                image_src_01=src_img,
                x_src_packed=x_src_packed,
                latent_image_ids=img_ids,
                orig_height=oh,
                orig_width=ow,
                token_h=th,
                token_w=tw,
            )
        )
    return packed_views


def maybe_run_noise_preopt(
    args,
    pipe,
    runtime: NoiseRuntime,
    dataset,
    edp,
    out_model_dir: Path,
) -> Optional[Path]:
    if int(args.noise_opt_iters) <= 0:
        return None
    if not runtime.cfg.enabled:
        return None

    # Build train cameras in the same order as training split.
    temp_model_dir = str(EXP_ROOT / "results" / "tmp_scene_model")
    Path(temp_model_dir).mkdir(parents=True, exist_ok=True)
    dataset_cfg = build_dataset_cfg(dataset.source_path, temp_model_dir, dataset.data_device)
    scene_info = sceneLoadTypeCallbacks["Colmap"](
        dataset_cfg.source_path,
        dataset_cfg.images,
        dataset_cfg.eval,
    )
    train_cameras = cameraList_from_camInfos(scene_info.train_cameras, 1.0, dataset_cfg)

    total_views = len(train_cameras)
    n_views = max(1, min(int(args.noise_opt_num_views), total_views))
    view_indices = np.linspace(0, total_views - 1, num=n_views, dtype=int).tolist()

    device = pipe._execution_device if hasattr(pipe, "_execution_device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    packed_views = build_packed_views(pipe, train_cameras, view_indices, device)

    channels = int(packed_views[0].x_src_packed.shape[-1])
    runtime.ensure_field(channels=channels, device=device)

    opt_cfg = NoiseOptimizeConfig(
        iterations=int(args.noise_opt_iters),
        lr=float(args.noise_opt_lr),
        views_per_iter=int(args.noise_opt_views_per_iter),
        objective_mode=str(args.noise_opt_mode),
        diffusion_steps=int(edp.flow_steps),
        n_min=int(edp.flow_n_min),
        n_max=int(edp.flow_n_max),
        src_guidance_scale=float(edp.flow_src_guidance_scale),
        tar_guidance_scale=float(edp.flow_tar_guidance_scale),
        noise_mix_ratio=float(runtime.cfg.mix_ratio),
        lambda_edit=float(args.noise_lambda_edit),
        lambda_id=float(args.noise_lambda_id),
        lambda_smooth=float(args.noise_lambda_smooth),
        lambda_prior=float(args.noise_lambda_prior),
        lambda_delta=float(args.noise_lambda_delta),
        lambda_view_var=float(args.noise_lambda_view_var),
        snr_gamma=float(args.noise_snr_gamma),
        max_grad_norm=float(args.noise_max_grad_norm),
        seed=int(args.noise_seed),
    )

    log_dir = EXP_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    preopt_log = log_dir / f"full_noise_preopt_{args.noise_tag}_{ts}.jsonl"

    print(
        f"[3DNOISE] pre-opt start: iters={opt_cfg.iterations}, views={n_views}, "
        f"mode={opt_cfg.objective_mode}, log={preopt_log}"
    )
    history = optimize_noise_field(
        pipe=pipe,
        field=runtime.field,
        views=packed_views,
        src_prompt=edp.flow_src_prompt,
        tar_prompt=edp.flow_tar_prompt,
        cfg=opt_cfg,
        log_jsonl=str(preopt_log),
    )
    print(f"[3DNOISE] pre-opt done: last={history[-1] if history else None}")

    noise_dump = {
        "field_cfg": vars(runtime.field.cfg),
        "preopt_cfg": vars(opt_cfg),
        "history_last": history[-1] if history else None,
        "num_opt_views": n_views,
        "view_indices": view_indices,
    }
    out_model_dir.mkdir(parents=True, exist_ok=True)
    with open(out_model_dir / "noise_preopt_summary.json", "w", encoding="utf-8") as f:
        json.dump(noise_dump, f, indent=2, ensure_ascii=False)

    torch.save(
        {
            "noise": runtime.field.noise.detach().cpu(),
            "noise_init": runtime.field.noise_init.detach().cpu(),
            "anchor_xyz": runtime.field.anchor_xyz.detach().cpu(),
            "anchor_opacity": runtime.field.anchor_opacity.detach().cpu(),
            "field_cfg": vars(runtime.field.cfg),
            "preopt_cfg": vars(opt_cfg),
            "history_last": history[-1] if history else None,
        },
        out_model_dir / "noise_field_preopt.pt",
    )
    return preopt_log


def main() -> None:
    parser = ArgumentParser(description="Editing Training script parameters (3D noise wrapper)")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    ed = EditingParams(parser)
    _ = ScoreDistillParams(parser)

    parser.add_argument("--aux_models_cpu", action="store_true", default=True)

    parser.add_argument("--disable_3dnoise", action="store_true", help="Disable 3D noise injection.")
    parser.add_argument("--noise_tag", type=str, default="cov_init_noopt")
    parser.add_argument("--noise_anchor_mode", type=str, default="coverage_opacity", choices=["coverage_opacity", "opacity_topk", "hybrid"])
    parser.add_argument("--noise_init_mode", type=str, default="coarse_smooth", choices=["coarse_smooth", "harmonic_lowfreq"])
    parser.add_argument("--noise_max_anchors", type=int, default=60000)
    parser.add_argument("--noise_voxel_res", type=int, default=96)
    parser.add_argument("--noise_coarse_res", type=int, default=32)
    parser.add_argument("--noise_hybrid_ratio", type=float, default=0.5)
    parser.add_argument("--noise_harmonic_mix", type=float, default=0.7)
    parser.add_argument("--noise_seed", type=int, default=10)
    parser.add_argument("--noise_mix", type=float, default=0.9)
    parser.add_argument("--noise_no_mfg", action="store_true", help="Do not apply 3D noise on edit_image_MFG.")

    parser.add_argument("--noise_opt_iters", type=int, default=0)
    parser.add_argument("--noise_opt_lr", type=float, default=0.03)
    parser.add_argument("--noise_opt_mode", type=str, default="base", choices=["base", "snr", "delta", "balanced"])
    parser.add_argument("--noise_opt_num_views", type=int, default=8)
    parser.add_argument("--noise_opt_views_per_iter", type=int, default=2)

    parser.add_argument("--noise_lambda_edit", type=float, default=1.0)
    parser.add_argument("--noise_lambda_id", type=float, default=0.5)
    parser.add_argument("--noise_lambda_smooth", type=float, default=0.2)
    parser.add_argument("--noise_lambda_prior", type=float, default=0.05)
    parser.add_argument("--noise_lambda_delta", type=float, default=0.2)
    parser.add_argument("--noise_lambda_view_var", type=float, default=0.2)
    parser.add_argument("--noise_snr_gamma", type=float, default=2.0)
    parser.add_argument("--noise_max_grad_norm", type=float, default=1.0)

    args = parser.parse_args(sys.argv[1:])

    set_seed(0)
    if args.aux_models_cpu:
        patch_aux_models_to_cpu()
    patch_scene_load_from_checkpoint()

    noise_cfg = NoiseWrapperConfig(
        enabled=not bool(args.disable_3dnoise),
        anchor_mode=str(args.noise_anchor_mode),
        init_mode=str(args.noise_init_mode),
        max_anchors=int(args.noise_max_anchors),
        voxel_res=int(args.noise_voxel_res),
        coarse_res=int(args.noise_coarse_res),
        hybrid_ratio=float(args.noise_hybrid_ratio),
        harmonic_mix=float(args.noise_harmonic_mix),
        seed=int(args.noise_seed),
        mix_ratio=float(args.noise_mix),
        apply_on_mfg=not bool(args.noise_no_mfg),
    )
    runtime = NoiseRuntime(cfg=noise_cfg)
    patch_camera_dataset_runtime(runtime)
    patch_edit_methods_with_3dnoise(runtime)

    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline = Editsplat_Pipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=dtype,
        use_safetensors=True,
        token=os.environ.get("HF_TOKEN", None),
    ).to(device)

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)
    edp = ed.extract(args)

    ensure_point_cloud_link(dataset.model_path, dataset.source_checkpoint)

    # Prepare gaussian stats for 3D noise anchors.
    if runtime.cfg.enabled:
        opt_restore = build_opt_cfg_for_restore()
        gaussians_for_noise = GaussianModel(dataset.sh_degree)
        ckpt = torch.load(dataset.source_checkpoint, map_location=device)
        model_params, first_iter = ckpt
        gaussians_for_noise.restore(model_params, opt_restore)
        runtime.set_gaussians(
            xyz=gaussians_for_noise.get_xyz.detach(),
            opacity=gaussians_for_noise.get_opacity.detach(),
        )
        print(
            f"[3DNOISE] checkpoint restored for noise anchors: "
            f"iter={first_iter}, gaussians={gaussians_for_noise.get_xyz.shape[0]}"
        )

    out_model_dir = Path(dataset.model_path)
    out_model_dir.mkdir(parents=True, exist_ok=True)

    preopt_log = maybe_run_noise_preopt(
        args=args,
        pipe=pipeline,
        runtime=runtime,
        dataset=dataset,
        edp=edp,
        out_model_dir=out_model_dir,
    )

    with open(out_model_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    meta = {
        "noise_enabled": runtime.cfg.enabled,
        "noise_cfg": vars(runtime.cfg),
        "preopt_log": str(preopt_log) if preopt_log is not None else None,
    }
    with open(out_model_dir / "noise_wrapper_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    shutil.copyfile(__file__, out_model_dir / "train_frozen.py")

    _ = pipeline(
        dataset=dataset,
        opt=opt,
        pipe=pipe,
        ed=edp,
    )

    # Save final field state if enabled.
    if runtime.cfg.enabled and runtime.field is not None:
        torch.save(
            {
                "noise": runtime.field.noise.detach().cpu(),
                "noise_init": runtime.field.noise_init.detach().cpu(),
                "anchor_xyz": runtime.field.anchor_xyz.detach().cpu(),
                "anchor_opacity": runtime.field.anchor_opacity.detach().cpu(),
                "field_cfg": vars(runtime.field.cfg),
                "state_summary": runtime.field.state_summary(),
            },
            out_model_dir / "noise_field_final.pt",
        )

    print("\nEditing complete.")


if __name__ == "__main__":
    main()
