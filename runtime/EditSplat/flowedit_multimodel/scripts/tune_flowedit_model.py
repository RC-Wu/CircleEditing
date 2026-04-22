#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]  # EditSplat
PKG = Path(__file__).resolve().parents[1]   # flowedit_multimodel
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

try:
    import clip  # type: ignore
except Exception:  # noqa: BLE001
    clip = None

try:
    import lpips  # type: ignore
except Exception:  # noqa: BLE001
    lpips = None
from PIL import Image

from src.flowedit_adapters import FlowEditParams, create_adapter, tensor_m11_to_pil
from src.model_registry import MODEL_SPECS, get_model_spec


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Tune FlowEdit hyperparameters for one model")
    ap.add_argument("--model_key", type=str, required=True, choices=sorted(MODEL_SPECS.keys()))
    ap.add_argument("--model_id", type=str, default="", help="Optional override model id")
    ap.add_argument("--cases_json", type=str, required=True, help="List of test cases")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--hf_home", type=str, default="/dev-vepfs/rc_wu/rc_wu/cache/hf_home")
    ap.add_argument("--quick", action="store_true", help="Smaller hyperparameter set")
    ap.add_argument("--no_clip", action="store_true", help="Disable CLIP metric if CLIP weights are unstable.")
    ap.add_argument("--use_lpips", action="store_true", help="Enable LPIPS (will download VGG weights).")
    return ap.parse_args()


def build_param_grid(spec_key: str, quick: bool) -> List[FlowEditParams]:
    spec = get_model_spec(spec_key)
    base = FlowEditParams(
        diffusion_steps=spec.default_steps,
        n_avg=1,
        src_guidance_scale=spec.default_src_guidance,
        tar_guidance_scale=spec.default_tar_guidance,
        n_min=spec.default_n_min,
        n_max=spec.default_n_max,
        seed=0,
        negative_prompt="",
        resize_side=512,
    )

    def _params(
        *,
        steps: int,
        src_g: float,
        tar_g: float,
        n_max: int,
        seed: int,
        n_avg: int = 1,
    ) -> FlowEditParams:
        return FlowEditParams(
            diffusion_steps=steps,
            n_avg=n_avg,
            src_guidance_scale=src_g,
            tar_guidance_scale=tar_g,
            n_min=base.n_min,
            n_max=n_max,
            seed=seed,
            negative_prompt="",
            resize_side=512,
        )

    # FLUX family: guidance should stay moderate, n_avg may stabilize details.
    if spec_key.startswith("flux1"):
        if quick:
            return [
                base,
                _params(
                    steps=max(20, base.diffusion_steps - 4),
                    src_g=max(1.0, base.src_guidance_scale - 0.3),
                    tar_g=base.tar_guidance_scale + 1.0,
                    n_max=max(base.n_min + 8, base.n_max - 4),
                    seed=1,
                ),
            ]
        return [
            base,
            _params(
                steps=base.diffusion_steps + 6,
                src_g=base.src_guidance_scale,
                tar_g=base.tar_guidance_scale + 1.0,
                n_max=min(base.diffusion_steps + 4, base.n_max + 4),
                seed=1,
            ),
            _params(
                steps=max(20, base.diffusion_steps - 6),
                src_g=max(1.0, base.src_guidance_scale - 0.5),
                tar_g=max(4.0, base.tar_guidance_scale - 1.0),
                n_max=max(base.n_min + 8, base.n_max - 6),
                seed=2,
            ),
            _params(
                steps=base.diffusion_steps,
                src_g=base.src_guidance_scale + 0.2,
                tar_g=base.tar_guidance_scale + 0.5,
                n_max=base.n_max,
                seed=3,
                n_avg=2,
            ),
        ]

    if spec_key.startswith("flux2"):
        if quick:
            return [
                base,
                _params(
                    steps=max(18, base.diffusion_steps - 4),
                    src_g=max(0.8, base.src_guidance_scale - 0.2),
                    tar_g=min(6.0, base.tar_guidance_scale + 0.8),
                    n_max=max(base.n_min + 8, base.n_max - 2),
                    seed=1,
                ),
            ]
        return [
            base,
            _params(
                steps=base.diffusion_steps + 4,
                src_g=base.src_guidance_scale + 0.1,
                tar_g=min(6.0, base.tar_guidance_scale + 0.6),
                n_max=min(base.diffusion_steps + 2, base.n_max + 2),
                seed=1,
            ),
            _params(
                steps=max(18, base.diffusion_steps - 4),
                src_g=max(0.8, base.src_guidance_scale - 0.3),
                tar_g=max(3.5, base.tar_guidance_scale - 0.5),
                n_max=max(base.n_min + 6, base.n_max - 4),
                seed=2,
            ),
            _params(
                steps=base.diffusion_steps,
                src_g=base.src_guidance_scale,
                tar_g=base.tar_guidance_scale + 0.8,
                n_max=base.n_max,
                seed=3,
                n_avg=2,
            ),
        ]

    # SD3/SD3.5 family: stronger CFG and more steps usually help semantic shift.
    if spec_key.startswith("sd3") or spec_key.startswith("sd35"):
        if quick:
            return [
                base,
                _params(
                    steps=max(24, base.diffusion_steps - 8),
                    src_g=max(2.0, base.src_guidance_scale - 0.8),
                    tar_g=base.tar_guidance_scale + 1.5,
                    n_max=max(base.n_min + 8, base.n_max - 4),
                    seed=1,
                ),
            ]
        return [
            base,
            _params(
                steps=base.diffusion_steps + 8,
                src_g=base.src_guidance_scale,
                tar_g=base.tar_guidance_scale + 1.5,
                n_max=min(base.diffusion_steps, base.n_max + 4),
                seed=1,
            ),
            _params(
                steps=max(24, base.diffusion_steps - 8),
                src_g=max(2.0, base.src_guidance_scale - 1.0),
                tar_g=max(7.0, base.tar_guidance_scale - 1.5),
                n_max=max(base.n_min + 8, base.n_max - 6),
                seed=2,
            ),
            _params(
                steps=base.diffusion_steps,
                src_g=base.src_guidance_scale + 0.5,
                tar_g=base.tar_guidance_scale + 0.5,
                n_max=base.n_max,
                seed=3,
            ),
        ]

    # Qwen image edit: explicit image conditioning + true CFG branch.
    if spec_key.startswith("qwen"):
        if quick:
            return [
                base,
                _params(
                    steps=max(20, base.diffusion_steps - 6),
                    src_g=max(1.2, base.src_guidance_scale - 0.5),
                    tar_g=min(7.0, base.tar_guidance_scale + 1.0),
                    n_max=max(base.n_min + 8, base.n_max - 3),
                    seed=1,
                ),
            ]
        return [
            base,
            _params(
                steps=base.diffusion_steps + 6,
                src_g=base.src_guidance_scale,
                tar_g=min(7.0, base.tar_guidance_scale + 1.2),
                n_max=min(base.diffusion_steps + 2, base.n_max + 2),
                seed=1,
            ),
            _params(
                steps=max(20, base.diffusion_steps - 6),
                src_g=max(1.2, base.src_guidance_scale - 0.6),
                tar_g=max(3.8, base.tar_guidance_scale - 0.8),
                n_max=max(base.n_min + 8, base.n_max - 5),
                seed=2,
            ),
            _params(
                steps=base.diffusion_steps,
                src_g=base.src_guidance_scale + 0.3,
                tar_g=base.tar_guidance_scale + 0.6,
                n_max=base.n_max,
                seed=3,
                n_avg=2,
            ),
        ]

    # Z-Image: no explicit image branch, rely on latent-space FlowEdit and moderate CFG.
    if quick:
        return [
            base,
            _params(
                steps=max(20, base.diffusion_steps - 6),
                src_g=max(1.3, base.src_guidance_scale - 0.5),
                tar_g=min(6.5, base.tar_guidance_scale + 1.0),
                n_max=max(base.n_min + 8, base.n_max - 4),
                seed=1,
            ),
        ]
    return [
        base,
        _params(
            steps=base.diffusion_steps + 4,
            src_g=base.src_guidance_scale,
            tar_g=min(6.5, base.tar_guidance_scale + 1.0),
            n_max=min(base.diffusion_steps, base.n_max + 2),
            seed=1,
        ),
        _params(
            steps=max(20, base.diffusion_steps - 6),
            src_g=max(1.2, base.src_guidance_scale - 0.7),
            tar_g=max(3.6, base.tar_guidance_scale - 0.8),
            n_max=max(base.n_min + 6, base.n_max - 6),
            seed=2,
        ),
        _params(
            steps=base.diffusion_steps,
            src_g=base.src_guidance_scale + 0.2,
            tar_g=base.tar_guidance_scale + 0.6,
            n_max=base.n_max,
            seed=3,
        ),
    ]


@torch.no_grad()
def compute_clip_scores(clip_model, clip_preprocess, image_pil: Image.Image, src_prompt: str, tar_prompt: str, device):
    if clip_model is None:
        return 0.0, 0.0
    img = clip_preprocess(image_pil).unsqueeze(0).to(device)
    txt = clip.tokenize([src_prompt, tar_prompt]).to(device)
    img_feat = clip_model.encode_image(img)
    txt_feat = clip_model.encode_text(txt)
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
    src_sim = float((img_feat @ txt_feat[0:1].T).squeeze().item())
    tar_sim = float((img_feat @ txt_feat[1:2].T).squeeze().item())
    return src_sim, tar_sim


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ["HF_HOME"] = args.hf_home
    os.environ["HF_HUB_CACHE"] = str(Path(args.hf_home) / "hub")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    model_id = args.model_id if args.model_id else None
    adapter = create_adapter(
        model_key=args.model_key,
        device=device,
        dtype=dtype,
        hf_token=args.hf_token if args.hf_token else None,
        cache_dir=str(Path(args.hf_home) / "hub"),
        override_model_id=model_id,
    )

    clip_model = None
    clip_preprocess = None
    if not args.no_clip:
        if clip is None:
            print("[WARN] CLIP disabled because 'clip' package is not installed.")
        else:
            try:
                clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
                clip_model.eval()
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] CLIP disabled due to load failure: {type(e).__name__}: {e}")
                clip_model = None
                clip_preprocess = None
    lpips_fn = None
    if args.use_lpips:
        if lpips is None:
            raise RuntimeError("LPIPS requested but 'lpips' package is not installed.")
        lpips_fn = lpips.LPIPS(net="vgg").to(device)
        lpips_fn.eval()

    with open(args.cases_json, "r", encoding="utf-8") as f:
        cases = json.load(f)

    grid = build_param_grid(args.model_key, args.quick)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for idx, hp in enumerate(grid):
        hp_dir = out_dir / f"hp_{idx:02d}"
        hp_dir.mkdir(parents=True, exist_ok=True)
        case_rows = []
        t0 = time.time()
        for case in cases:
            image_path = Path(case["image"])
            src_prompt = case["src_prompt"]
            tar_prompt = case["tar_prompt"]
            name = case["name"]
            src_pil = Image.open(image_path).convert("RGB")

            out_m11 = adapter.edit(src_pil, src_prompt, tar_prompt, hp)
            out_m11 = torch.nan_to_num(out_m11.float(), nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
            out_pil = tensor_m11_to_pil(out_m11)
            out_path = hp_dir / f"{name}_edit.png"
            out_pil.save(out_path)

            src_sim, tar_sim = compute_clip_scores(clip_model, clip_preprocess, out_pil, src_prompt, tar_prompt, device)
            if not math.isfinite(src_sim):
                src_sim = 0.0
            if not math.isfinite(tar_sim):
                tar_sim = 0.0
            src_ref = (
                torch.from_numpy(np.asarray(src_pil.resize(out_pil.size), dtype=np.float32) / 127.5 - 1.0)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device)
            )
            if lpips_fn is not None:
                dist_src = float(lpips_fn(out_m11.float(), src_ref.float()).mean().item())
            else:
                dist_src = float((out_m11.float() - src_ref.float()).abs().mean().item())
            if not math.isfinite(dist_src):
                dist_src = 1.0

            # Higher is better.
            score = (tar_sim - 0.25 * src_sim - 0.20 * dist_src) if clip_model is not None else (-0.20 * dist_src)
            row = {
                "case": name,
                "image": str(image_path),
                "output": str(out_path),
                "src_clip": src_sim,
                "tar_clip": tar_sim,
                "dist_to_src": dist_src,
                "dist_metric": "lpips" if lpips_fn is not None else "l1",
                "score": score,
            }
            case_rows.append(row)

        agg = float(np.mean([r["score"] for r in case_rows]))
        result = {
            "hp_index": idx,
            "hp": vars(hp),
            "duration_sec": round(time.time() - t0, 2),
            "avg_score": agg,
            "cases": case_rows,
        }
        with open(hp_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        results.append(result)

    results_sorted = sorted(results, key=lambda x: x["avg_score"], reverse=True)
    summary = {
        "model_key": args.model_key,
        "model_id": model_id if model_id else get_model_spec(args.model_key).model_id,
        "gpu_id": args.gpu_id,
        "top1": results_sorted[0],
        "all": results_sorted,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary["top1"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
