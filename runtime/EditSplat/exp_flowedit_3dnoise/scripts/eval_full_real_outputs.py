#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def load_img_01(path: Path) -> torch.Tensor:
    arr = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def gaussian_kernel(size: int = 11, sigma: float = 1.5, device: torch.device = torch.device("cpu")):
    ax = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(ax * ax) / (2 * sigma * sigma))
    g = g / g.sum()
    k = torch.outer(g, g)
    return k


def ssim_torch(x: torch.Tensor, y: torch.Tensor, ksize: int = 11, sigma: float = 1.5) -> float:
    # x,y: [1,3,H,W], [0,1]
    device = x.device
    k = gaussian_kernel(ksize, sigma, device=device)
    w = k.view(1, 1, ksize, ksize).repeat(3, 1, 1, 1)

    mu_x = F.conv2d(x, w, padding=ksize // 2, groups=3)
    mu_y = F.conv2d(y, w, padding=ksize // 2, groups=3)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, w, padding=ksize // 2, groups=3) - mu_x2
    sigma_y2 = F.conv2d(y * y, w, padding=ksize // 2, groups=3) - mu_y2
    sigma_xy = F.conv2d(x * y, w, padding=ksize // 2, groups=3) - mu_xy

    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)

    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = num / (den + 1e-8)
    return float(ssim_map.mean().item())


def psnr_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = torch.mean((x - y) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * math.log10(1.0 / mse))


def hf_energy(x: torch.Tensor) -> float:
    gx = x[..., :, 1:] - x[..., :, :-1]
    gy = x[..., 1:, :] - x[..., :-1, :]
    return float((gx.pow(2).mean() + gy.pow(2).mean()).item())


def clip_ratio(x: torch.Tensor, lo: float = 0.01, hi: float = 0.99) -> float:
    c = ((x <= lo) | (x >= hi)).float().mean().item()
    return float(c)


def parse_iter_from_name(p: Path) -> int:
    m = re.search(r"ours_(\d+)$", p.name)
    return int(m.group(1)) if m else -1


def find_latest_render_dir(model_dir: Path, split: str = "train") -> Path:
    split_dir = model_dir / split
    cands = [d for d in split_dir.glob("ours_*") if d.is_dir() and (d / "renders").exists()]
    if not cands:
        raise FileNotFoundError(f"No render dirs in {split_dir}")
    latest = sorted(cands, key=parse_iter_from_name)[-1]
    return latest / "renders"


def find_latest_point_cloud(model_dir: Path) -> Path:
    pc_root = model_dir / "point_cloud"
    cands = [d for d in pc_root.glob("iteration_*") if d.is_dir() and (d / "point_cloud.ply").exists()]
    if not cands:
        raise FileNotFoundError(f"No point_cloud in {pc_root}")
    latest = sorted(cands, key=lambda p: int(p.name.split("_")[-1]))[-1]
    return latest / "point_cloud.ply"


def count_vertices_from_ply_header(ply_path: Path) -> int:
    with open(ply_path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in range(200):
            line = f.readline()
            if not line:
                break
            if line.startswith("element vertex"):
                parts = line.strip().split()
                return int(parts[-1])
            if line.strip() == "end_header":
                break
    return -1


def pairwise_distance_matrix(imgs: List[torch.Tensor], side: int = 64) -> torch.Tensor:
    xs = []
    for im in imgs:
        x = F.interpolate(im, size=(side, side), mode="bilinear", align_corners=False)
        x = x.mean(dim=1, keepdim=True)  # gray
        xs.append(x.flatten())
    mat = torch.stack(xs, dim=0)  # [N, D]
    # Euclidean distances
    d2 = torch.cdist(mat, mat, p=2)
    return d2


def mv_distance_preservation(src_imgs: List[torch.Tensor], edit_imgs: List[torch.Tensor], max_views: int = 24) -> float:
    n = min(len(src_imgs), len(edit_imgs))
    if n < 3:
        return 0.0
    if n > max_views:
        idx = np.linspace(0, n - 1, num=max_views, dtype=int).tolist()
    else:
        idx = list(range(n))

    src_sub = [src_imgs[i] for i in idx]
    edit_sub = [edit_imgs[i] for i in idx]

    ds = pairwise_distance_matrix(src_sub)
    de = pairwise_distance_matrix(edit_sub)

    ds = ds / (ds.mean() + 1e-8)
    de = de / (de.mean() + 1e-8)

    tri = torch.triu(torch.ones_like(ds, dtype=torch.bool), diagonal=1)
    err = (de[tri] - ds[tri]).pow(2).mean().item()
    return float(err)


def eval_one_model(
    model_dir: Path,
    src_render_dir: Path,
    src_point_cloud: Path,
    use_lpips: bool,
    use_imagereward: bool,
    target_prompt: str,
    source_prompt: str,
    max_pair_views: int,
) -> Dict[str, float]:
    edit_render_dir = find_latest_render_dir(model_dir, split="train")

    src_files = sorted(src_render_dir.glob("*.png"))
    edit_files = sorted(edit_render_dir.glob("*.png"))
    src_map = {p.name: p for p in src_files}
    edit_map = {p.name: p for p in edit_files}
    common = sorted(set(src_map.keys()) & set(edit_map.keys()))

    if not common:
        raise RuntimeError(f"No common render files between {src_render_dir} and {edit_render_dir}")

    lpips_fn = None
    if use_lpips:
        import lpips

        lpips_fn = lpips.LPIPS(net="vgg").cuda().eval()
        for p in lpips_fn.parameters():
            p.requires_grad_(False)

    src_imgs = []
    edit_imgs = []

    l1s, psnrs, ssims, lpv = [], [], [], []
    hf_ratio_list, clip_list = [], []

    for name in common:
        s = load_img_01(src_map[name])
        e = load_img_01(edit_map[name])
        if e.shape[-2:] != s.shape[-2:]:
            e = F.interpolate(e, size=s.shape[-2:], mode="bilinear", align_corners=False)

        src_imgs.append(s)
        edit_imgs.append(e)

        l1s.append(float((e - s).abs().mean().item()))
        psnrs.append(psnr_torch(e, s))
        ssims.append(ssim_torch(e, s))

        if lpips_fn is not None:
            ev = e.cuda() * 2.0 - 1.0
            sv = s.cuda() * 2.0 - 1.0
            lpv.append(float(lpips_fn(ev, sv).item()))

        hf_s = hf_energy(s)
        hf_e = hf_energy(e)
        hf_ratio_list.append(float(hf_e / (hf_s + 1e-8)))
        clip_list.append(clip_ratio(e))

    ir_target = []
    ir_source = []
    if use_imagereward:
        import ImageReward as RM

        rm = RM.load("ImageReward-v1.0", device="cpu")
        pil_imgs = [Image.open(edit_map[n]).convert("RGB") for n in common]
        if target_prompt.strip():
            _, r_t = rm.inference_rank(target_prompt, pil_imgs)
            ir_target = [float(x) for x in r_t]
        if source_prompt.strip():
            _, r_s = rm.inference_rank(source_prompt, pil_imgs)
            ir_source = [float(x) for x in r_s]

    mv_rel = mv_distance_preservation(src_imgs, edit_imgs, max_views=max_pair_views)

    src_vertices = count_vertices_from_ply_header(src_point_cloud)
    edit_vertices = -1
    try:
        edit_pc = find_latest_point_cloud(model_dir)
        edit_vertices = count_vertices_from_ply_header(edit_pc)
    except Exception:
        edit_pc = None

    out = {
        "num_views": float(len(common)),
        "l1_to_src": float(np.mean(l1s)),
        "psnr_to_src": float(np.mean(psnrs)),
        "ssim_to_src": float(np.mean(ssims)),
        "lpips_to_src": float(np.mean(lpv)) if lpv else None,
        "hf_ratio_vs_src": float(np.mean(hf_ratio_list)),
        "clip_ratio": float(np.mean(clip_list)),
        "mv_rel_dist_mse": float(mv_rel),
        "imagereward_target": float(np.mean(ir_target)) if ir_target else None,
        "imagereward_source": float(np.mean(ir_source)) if ir_source else None,
        "imagereward_delta_t_minus_s": float(np.mean(ir_target) - np.mean(ir_source)) if (ir_target and ir_source) else None,
        "src_vertices": float(src_vertices),
        "edit_vertices": float(edit_vertices),
        "vertex_ratio": float(edit_vertices / (src_vertices + 1e-8)) if (src_vertices > 0 and edit_vertices > 0) else None,
        "edit_render_dir": str(edit_render_dir),
        "src_render_dir": str(src_render_dir),
        "edit_point_cloud": str(edit_pc) if edit_vertices > 0 else "",
        "src_point_cloud": str(src_point_cloud),
    }
    return out


def markdown_table(rows: List[Tuple[str, Dict[str, float]]]) -> str:
    def fmt_opt(v: float, fmt: str) -> str:
        if v is None:
            return "n/a"
        try:
            return fmt.format(float(v))
        except Exception:
            return "n/a"

    lines = []
    lines.append("| model | views | L1↓ | PSNR↑ | SSIM↑ | LPIPS↓ | HF_ratio≈1 | Clip↓ | MV_rel_MSE↓ | IR_tgt↑ | IR_delta↑ | Vert_ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, m in rows:
        lines.append(
            "| {} | {} | {:.6f} | {:.4f} | {:.4f} | {} | {:.4f} | {:.6f} | {:.6f} | {} | {} | {} |".format(
                name,
                int(m["num_views"]),
                m["l1_to_src"],
                m["psnr_to_src"],
                m["ssim_to_src"],
                fmt_opt(m.get("lpips_to_src"), "{:.4f}"),
                m["hf_ratio_vs_src"],
                m["clip_ratio"],
                m["mv_rel_dist_mse"],
                fmt_opt(m.get("imagereward_target"), "{:.4f}"),
                fmt_opt(m.get("imagereward_delta_t_minus_s"), "{:.4f}"),
                fmt_opt(m.get("vertex_ratio"), "{:.4f}"),
            )
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser("Evaluate full-pipeline EditSplat runs against source renders")
    ap.add_argument("--model_dirs", type=str, nargs="+", required=True)
    ap.add_argument("--source_pretrained_dir", type=str, required=True)
    ap.add_argument("--source_iter", type=int, default=30000)
    ap.add_argument("--use_lpips", action="store_true")
    ap.add_argument("--use_imagereward", action="store_true")
    ap.add_argument("--target_prompt", type=str, default="")
    ap.add_argument("--source_prompt", type=str, default="")
    ap.add_argument("--max_pair_views", type=int, default=24)
    ap.add_argument("--out_json", type=str, default="")
    ap.add_argument("--out_md", type=str, default="")
    args = ap.parse_args()

    src_dir = Path(args.source_pretrained_dir)
    src_render_dir = src_dir / "train" / f"ours_{int(args.source_iter)}" / "renders"
    if not src_render_dir.exists():
        raise FileNotFoundError(f"Missing source render dir: {src_render_dir}")

    src_pc = src_dir / "point_cloud" / f"iteration_{int(args.source_iter)}" / "point_cloud.ply"
    if not src_pc.exists():
        # fallback to checkpoint iter folder if 30000 does not exist
        pcs = sorted((src_dir / "point_cloud").glob("iteration_*"), key=lambda p: int(p.name.split("_")[-1]))
        if not pcs:
            raise FileNotFoundError(f"Missing source point cloud under {src_dir / 'point_cloud'}")
        src_pc = pcs[-1] / "point_cloud.ply"

    results = {}
    rows = []

    for md in args.model_dirs:
        model_dir = Path(md)
        name = model_dir.name
        metrics = eval_one_model(
            model_dir=model_dir,
            src_render_dir=src_render_dir,
            src_point_cloud=src_pc,
            use_lpips=bool(args.use_lpips),
            use_imagereward=bool(args.use_imagereward),
            target_prompt=str(args.target_prompt),
            source_prompt=str(args.source_prompt),
            max_pair_views=int(args.max_pair_views),
        )
        results[name] = metrics
        rows.append((name, metrics))

    out = {
        "source_render_dir": str(src_render_dir),
        "source_point_cloud": str(src_pc),
        "results": results,
    }

    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("\n# Markdown")
    md = markdown_table(rows)
    print(md)

    if args.out_json:
        p = Path(args.out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

    if args.out_md:
        p = Path(args.out_md)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(md + "\n")


if __name__ == "__main__":
    main()
