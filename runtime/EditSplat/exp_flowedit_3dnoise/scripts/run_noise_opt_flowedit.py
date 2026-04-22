#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from diffusers import FluxPipeline
from torchvision.transforms import ToPILImage

ROOT = Path(__file__).resolve().parents[2]  # EditSplat
EXP_ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from arguments import ModelParams, OptimizationParams
from scene import GaussianModel
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos

from src.flowedit_3dnoise_core import PackedView, pack_image_latents, run_flowedit_with_noise
from src.noise_field import SparseGaussianNoiseField, SparseNoiseFieldConfig
from src.noise_optimizer import NoiseOptimizeConfig, optimize_noise_field


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset_cfg(source_path: str, model_path: str, data_device: str):
    parser = argparse.ArgumentParser(add_help=False)
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


def build_opt_cfg():
    parser = argparse.ArgumentParser(add_help=False)
    op = OptimizationParams(parser)
    args = parser.parse_args([])
    return op.extract(args)


def tensor_to_pil(img_bchw: torch.Tensor):
    img_01 = ((img_bchw[0].detach().float().cpu().clamp(-1, 1) + 1.0) * 0.5).clamp(0, 1)
    return ToPILImage()(img_01)


def build_packed_views(pipe, train_cameras, view_indices: List[int], device: torch.device) -> List[PackedView]:
    packed_views: List[PackedView] = []
    for idx in view_indices:
        cam = train_cameras[idx]
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


def default_variants(args) -> List[Dict]:
    if not args.run_suite:
        return [
            {
                "name": "single",
                "use_field": True,
                "anchor_mode": args.anchor_mode,
                "init_mode": args.init_mode,
                "do_opt": args.opt_iters > 0,
                "opt_mode": args.opt_mode,
                "views_per_iter": args.views_per_iter,
            }
        ]

    # Parallel suite for fast ablations.
    variants = [
        {
            "name": "rand_baseline",
            "use_field": False,
            "anchor_mode": "none",
            "init_mode": "coarse_smooth",
            "do_opt": False,
            "opt_mode": "base",
            "views_per_iter": args.views_per_iter,
        },
        {
            "name": "cov_init_noopt",
            "use_field": True,
            "anchor_mode": "coverage_opacity",
            "init_mode": "coarse_smooth",
            "do_opt": False,
            "opt_mode": "base",
            "views_per_iter": args.views_per_iter,
        },
        {
            "name": "topk_init_noopt",
            "use_field": True,
            "anchor_mode": "opacity_topk",
            "init_mode": "coarse_smooth",
            "do_opt": False,
            "opt_mode": "base",
            "views_per_iter": args.views_per_iter,
        },
        {
            "name": "harmonic_init_noopt",
            "use_field": True,
            "anchor_mode": "coverage_opacity",
            "init_mode": "harmonic_lowfreq",
            "do_opt": False,
            "opt_mode": "base",
            "views_per_iter": args.views_per_iter,
        },
        {
            "name": "cov_baseopt",
            "use_field": True,
            "anchor_mode": "coverage_opacity",
            "init_mode": "coarse_smooth",
            "do_opt": True,
            "opt_mode": "base",
            "views_per_iter": args.views_per_iter,
        },
        {
            "name": "hybrid_snropt",
            "use_field": True,
            "anchor_mode": "hybrid",
            "init_mode": "coarse_smooth",
            "do_opt": True,
            "opt_mode": "snr",
            "views_per_iter": args.views_per_iter,
        },
        {
            "name": "cov_deltaopt",
            "use_field": True,
            "anchor_mode": "coverage_opacity",
            "init_mode": "coarse_smooth",
            "do_opt": True,
            "opt_mode": "delta",
            "views_per_iter": args.views_per_iter,
        },
        {
            "name": "harmonic_balancedopt",
            "use_field": True,
            "anchor_mode": "coverage_opacity",
            "init_mode": "harmonic_lowfreq",
            "do_opt": True,
            "opt_mode": "balanced",
            "views_per_iter": max(2, args.views_per_iter),
        },
    ]

    if args.suite_names.strip():
        allow = {x.strip() for x in args.suite_names.split(",") if x.strip()}
        variants = [v for v in variants if v["name"] in allow]

    return variants


def run_variant(
    pipe,
    gaussians,
    packed_views: List[PackedView],
    variant: Dict,
    args,
    run_dir: Path,
    run_id: str,
):
    vdir = run_dir / variant["name"]
    vdir.mkdir(parents=True, exist_ok=True)

    field = None
    history = []
    opt_log_path = None

    if variant["use_field"]:
        field_cfg = SparseNoiseFieldConfig(
            max_anchors=args.max_anchors,
            voxel_res=args.voxel_res,
            coarse_res=args.coarse_res,
            anchor_mode=variant["anchor_mode"],
            init_mode=variant.get("init_mode", args.init_mode),
            hybrid_ratio=args.hybrid_ratio,
            harmonic_mix=args.harmonic_mix,
            seed=args.seed,
        )
        field = SparseGaussianNoiseField.from_gaussians(
            xyz=gaussians.get_xyz.detach(),
            opacity=gaussians.get_opacity.detach(),
            channels=packed_views[0].x_src_packed.shape[-1],
            cfg=field_cfg,
        ).to(packed_views[0].x_src_packed.device)

        if variant["do_opt"]:
            log_dir = EXP_ROOT / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            opt_log_path = log_dir / f"noise_opt_{run_id}_{variant['name']}.jsonl"

            opt_cfg = NoiseOptimizeConfig(
                iterations=args.opt_iters,
                lr=args.opt_lr,
                views_per_iter=variant.get("views_per_iter", args.views_per_iter),
                objective_mode=variant["opt_mode"],
                diffusion_steps=args.flow_steps,
                n_min=args.flow_n_min,
                n_max=args.flow_n_max,
                src_guidance_scale=args.src_guidance,
                tar_guidance_scale=args.tar_guidance,
                noise_mix_ratio=args.noise_mix,
                lambda_edit=args.lambda_edit,
                lambda_id=args.lambda_id,
                lambda_smooth=args.lambda_smooth,
                lambda_prior=args.lambda_prior,
                lambda_delta=args.lambda_delta,
                lambda_view_var=args.lambda_view_var,
                snr_gamma=args.snr_gamma,
                max_grad_norm=args.max_grad_norm,
                seed=args.seed,
            )
            history = optimize_noise_field(
                pipe=pipe,
                field=field,
                views=packed_views,
                src_prompt=args.src_prompt,
                tar_prompt=args.tar_prompt,
                cfg=opt_cfg,
                log_jsonl=str(opt_log_path),
            )

    for i, pv in enumerate(packed_views):
        with torch.no_grad():
            out_rand = run_flowedit_with_noise(
                pipe=pipe,
                packed_view=pv,
                src_prompt=args.src_prompt,
                tar_prompt=args.tar_prompt,
                diffusion_steps=args.flow_steps,
                n_avg=args.flow_n_avg,
                n_min=args.flow_n_min,
                n_max=args.flow_n_max,
                src_guidance_scale=args.src_guidance,
                tar_guidance_scale=args.tar_guidance,
                seed=args.seed,
                noise_provider=None,
                noise_mix_ratio=1.0,
            )

            if field is None:
                out_var = out_rand
            else:

                def provider():
                    noise_map, _ = field.render_to_tokens(pv.camera, pv.token_h, pv.token_w, normalize=True)
                    return noise_map

                out_var = run_flowedit_with_noise(
                    pipe=pipe,
                    packed_view=pv,
                    src_prompt=args.src_prompt,
                    tar_prompt=args.tar_prompt,
                    diffusion_steps=args.flow_steps,
                    n_avg=args.flow_n_avg,
                    n_min=args.flow_n_min,
                    n_max=args.flow_n_max,
                    src_guidance_scale=args.src_guidance,
                    tar_guidance_scale=args.tar_guidance,
                    seed=args.seed,
                    noise_provider=provider,
                    noise_mix_ratio=args.noise_mix,
                )

        src_pil = ToPILImage()(pv.image_src_01[0].detach().cpu().clamp(0, 1))
        src_pil.save(vdir / f"view_{i:03d}_src.png")
        tensor_to_pil(out_rand).save(vdir / f"view_{i:03d}_rand.png")
        tensor_to_pil(out_var).save(vdir / f"view_{i:03d}_variant.png")

    summary = {
        "variant": variant,
        "opt_log": str(opt_log_path) if opt_log_path is not None else None,
        "history_last": history[-1] if history else None,
        "field_summary": field.state_summary() if field is not None else None,
    }

    if field is not None:
        torch.save(
            {
                "noise": field.noise.detach().cpu(),
                "noise_init": field.noise_init.detach().cpu(),
                "anchor_xyz": field.anchor_xyz.detach().cpu(),
                "anchor_opacity": field.anchor_opacity.detach().cpu(),
                "field_cfg": vars(field.cfg),
                "history_last": summary["history_last"],
            },
            vdir / "noise_field.pt",
        )

    with open(vdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Avoid accumulation when running suite.
    del field
    torch.cuda.empty_cache()

    return summary


def main():
    ap = argparse.ArgumentParser("3DGS-anchored noise optimization for FlowEdit")
    ap.add_argument("--source_path", type=str, required=True)
    ap.add_argument("--source_checkpoint", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default=str(ROOT / "output" / "flowedit_3dnoise_exp" / "smoke"))
    ap.add_argument("--temp_model_dir", type=str, default=str(EXP_ROOT / "results" / "tmp_scene_model"))

    ap.add_argument("--model_id", type=str, default="yujiepan/FLUX.1-dev-tiny-random")
    ap.add_argument("--use_local_files_only", action="store_true")
    ap.add_argument("--hf_home", type=str, default="/dev-vepfs/rc_wu/rc_wu/cache/hf_home")
    ap.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""))

    ap.add_argument("--src_prompt", type=str, required=True)
    ap.add_argument("--tar_prompt", type=str, required=True)

    ap.add_argument("--num_views", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--max_anchors", type=int, default=20000)
    ap.add_argument("--voxel_res", type=int, default=72)
    ap.add_argument("--coarse_res", type=int, default=24)
    ap.add_argument("--anchor_mode", type=str, default="coverage_opacity", choices=["coverage_opacity", "opacity_topk", "hybrid"])
    ap.add_argument("--init_mode", type=str, default="coarse_smooth", choices=["coarse_smooth", "harmonic_lowfreq"])
    ap.add_argument("--hybrid_ratio", type=float, default=0.5)
    ap.add_argument("--harmonic_mix", type=float, default=0.7)

    ap.add_argument("--opt_iters", type=int, default=8)
    ap.add_argument("--opt_lr", type=float, default=0.03)
    ap.add_argument("--views_per_iter", type=int, default=1)
    ap.add_argument("--opt_mode", type=str, default="base", choices=["base", "snr", "delta", "balanced"])
    ap.add_argument("--lambda_edit", type=float, default=1.0)
    ap.add_argument("--lambda_id", type=float, default=0.5)
    ap.add_argument("--lambda_smooth", type=float, default=0.2)
    ap.add_argument("--lambda_prior", type=float, default=0.05)
    ap.add_argument("--lambda_delta", type=float, default=0.2)
    ap.add_argument("--lambda_view_var", type=float, default=0.2)
    ap.add_argument("--snr_gamma", type=float, default=2.0)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    ap.add_argument("--flow_steps", type=int, default=10)
    ap.add_argument("--flow_n_min", type=int, default=0)
    ap.add_argument("--flow_n_max", type=int, default=8)
    ap.add_argument("--flow_n_avg", type=int, default=1)
    ap.add_argument("--src_guidance", type=float, default=1.5)
    ap.add_argument("--tar_guidance", type=float, default=5.5)
    ap.add_argument("--noise_mix", type=float, default=0.9)

    ap.add_argument("--run_suite", action="store_true", help="Run multiple initialization/optimization variants.")
    ap.add_argument(
        "--suite_names",
        type=str,
        default="",
        help="Comma-separated suite variant names to run. Empty means all defaults.",
    )

    args = ap.parse_args()

    set_seed(args.seed)

    os.environ["HF_HOME"] = args.hf_home
    os.environ["HF_HUB_CACHE"] = str(Path(args.hf_home) / "hub")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    print(f"[INFO] Loading FLUX model: {args.model_id}")
    pipe = FluxPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        token=(args.hf_token if args.hf_token else None),
        local_files_only=bool(args.use_local_files_only),
        cache_dir=str(Path(args.hf_home) / "hub"),
    ).to(device)
    pipe.transformer.eval()
    pipe.transformer.requires_grad_(False)
    pipe.vae.eval()

    print("[INFO] Loading scene metadata and 3DGS")
    Path(args.temp_model_dir).mkdir(parents=True, exist_ok=True)
    dataset_cfg = build_dataset_cfg(args.source_path, args.temp_model_dir, "cuda")
    opt_cfg_scene = build_opt_cfg()

    scene_info = sceneLoadTypeCallbacks["Colmap"](
        dataset_cfg.source_path,
        dataset_cfg.images,
        dataset_cfg.eval,
    )
    train_cameras = cameraList_from_camInfos(scene_info.train_cameras, 1.0, dataset_cfg)

    gaussians = GaussianModel(dataset_cfg.sh_degree)
    ckpt = torch.load(args.source_checkpoint, map_location=device)
    model_params, first_iter = ckpt
    gaussians.restore(model_params, opt_cfg_scene)
    print(f"[INFO] Restored checkpoint iter={first_iter}, gaussians={gaussians.get_xyz.shape[0]}")

    total_views = len(train_cameras)
    view_indices = np.linspace(0, total_views - 1, num=max(1, args.num_views), dtype=int).tolist()
    packed_views = build_packed_views(pipe, train_cameras, view_indices, device)

    variants = default_variants(args)
    all_summaries = {}
    for v in variants:
        print(f"[INFO] Running variant: {v['name']}")
        all_summaries[v["name"]] = run_variant(
            pipe=pipe,
            gaussians=gaussians,
            packed_views=packed_views,
            variant=v,
            args=args,
            run_dir=run_dir,
            run_id=run_id,
        )

    final_summary = {
        "run_id": run_id,
        "output_dir": str(run_dir),
        "model_id": args.model_id,
        "source_path": args.source_path,
        "source_checkpoint": args.source_checkpoint,
        "num_views": len(packed_views),
        "view_indices": view_indices,
        "variants": all_summaries,
    }

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    print("[DONE] Summary:")
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
