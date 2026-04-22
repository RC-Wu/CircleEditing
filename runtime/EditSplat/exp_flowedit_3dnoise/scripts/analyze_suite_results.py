#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def load_img(path: Path) -> torch.Tensor:
    arr = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return ten


def edge_mag(x: torch.Tensor) -> torch.Tensor:
    gx = x[..., :, 1:] - x[..., :, :-1]
    gy = x[..., 1:, :] - x[..., :-1, :]
    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))
    return (gx.pow(2) + gy.pow(2) + 1e-8).sqrt()


def mean_list(xs: List[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def collect_variant_metrics(vdir: Path, lpips_fn=None, reward_model=None, tar_prompt: str = "") -> Dict:
    srcs = sorted(vdir.glob("view_*_src.png"))
    vars_ = sorted(vdir.glob("view_*_variant.png"))
    rands = sorted(vdir.glob("view_*_rand.png"))

    n = min(len(srcs), len(vars_), len(rands))
    l1_src = []
    l1_rand = []
    edge_src = []
    edge_rand = []
    lpips_src = []
    ir_tar = []

    pil_imgs = []

    for i in range(n):
        s = load_img(srcs[i])
        v = load_img(vars_[i])
        r = load_img(rands[i])
        if v.shape[-2:] != s.shape[-2:]:
            v = F.interpolate(v, size=s.shape[-2:], mode="bilinear", align_corners=False)
        if r.shape[-2:] != s.shape[-2:]:
            r = F.interpolate(r, size=s.shape[-2:], mode="bilinear", align_corners=False)

        l1_src.append(float((v - s).abs().mean().item()))
        l1_rand.append(float((v - r).abs().mean().item()))

        es = edge_mag(s)
        ev = edge_mag(v)
        er = edge_mag(r)
        edge_src.append(float((ev - es).abs().mean().item()))
        edge_rand.append(float((ev - er).abs().mean().item()))

        if lpips_fn is not None:
            vv = v * 2.0 - 1.0
            ss = s * 2.0 - 1.0
            lp = lpips_fn(vv.cuda(), ss.cuda())
            lpips_src.append(float(lp.item()))

        if reward_model is not None:
            pil_imgs.append(Image.open(vars_[i]).convert("RGB"))

    if reward_model is not None and pil_imgs:
        _, rewards = reward_model.inference_rank(tar_prompt, pil_imgs)
        ir_tar = [float(x) for x in rewards]

    return {
        "num_views": n,
        "l1_to_src": mean_list(l1_src),
        "l1_to_rand": mean_list(l1_rand),
        "edge_diff_to_src": mean_list(edge_src),
        "edge_diff_to_rand": mean_list(edge_rand),
        "lpips_to_src": mean_list(lpips_src),
        "imagereward_tar": mean_list(ir_tar),
    }


def main():
    ap = argparse.ArgumentParser("Analyze variant image outputs")
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--tar_prompt", type=str, default="")
    ap.add_argument("--use_lpips", action="store_true")
    ap.add_argument("--use_imagereward", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    assert run_dir.exists(), f"run_dir not found: {run_dir}"

    lpips_fn = None
    if args.use_lpips:
        import lpips

        lpips_fn = lpips.LPIPS(net="vgg").cuda().eval()
        for p in lpips_fn.parameters():
            p.requires_grad_(False)

    reward_model = None
    if args.use_imagereward:
        import ImageReward as RM

        reward_model = RM.load("ImageReward-v1.0")

    variants = [x for x in run_dir.iterdir() if x.is_dir() and (x / "summary.json").exists()]
    metrics = {}
    for v in sorted(variants):
        metrics[v.name] = collect_variant_metrics(
            vdir=v,
            lpips_fn=lpips_fn,
            reward_model=reward_model,
            tar_prompt=args.tar_prompt,
        )

    out_json = run_dir / "metrics_analysis.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"[DONE] {out_json}")


if __name__ == "__main__":
    main()
