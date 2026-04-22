from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .io_utils import find_latest_iteration


def load_img_01(path: Path) -> torch.Tensor:
    arr = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def gaussian_kernel(size: int = 11, sigma: float = 1.5, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    ax = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(ax * ax) / (2 * sigma * sigma))
    g = g / g.sum()
    return torch.outer(g, g)


def ssim_torch(x: torch.Tensor, y: torch.Tensor, ksize: int = 11, sigma: float = 1.5) -> float:
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

    c1 = 0.01**2
    c2 = 0.03**2

    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return float((num / (den + 1e-8)).mean().item())


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
    return float(((x <= lo) | (x >= hi)).float().mean().item())


def pairwise_distance_matrix(imgs: Sequence[torch.Tensor], side: int = 64) -> torch.Tensor:
    xs = []
    for im in imgs:
        x = F.interpolate(im, size=(side, side), mode="bilinear", align_corners=False)
        x = x.mean(dim=1, keepdim=True)
        xs.append(x.flatten())
    mat = torch.stack(xs, dim=0)
    return torch.cdist(mat, mat, p=2)


def mv_distance_preservation(src_imgs: Sequence[torch.Tensor], edit_imgs: Sequence[torch.Tensor], max_views: int = 24) -> float:
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
    return float((de[tri] - ds[tri]).pow(2).mean().item())


def count_vertices_from_ply_header(ply_path: Path) -> int:
    if not ply_path.exists():
        return -1
    with open(ply_path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in range(200):
            line = f.readline()
            if not line:
                break
            if line.startswith("element vertex"):
                parts = line.strip().split()
                try:
                    return int(parts[-1])
                except Exception:
                    return -1
            if line.strip() == "end_header":
                break
    return -1


def find_point_cloud_file(model_dir: Path, iteration: int = -1) -> Path:
    pc_root = model_dir / "point_cloud"
    if not pc_root.exists():
        return Path("")

    if iteration < 0:
        iteration = find_latest_iteration(model_dir)
    if iteration < 0:
        return Path("")

    p = pc_root / f"iteration_{iteration}" / "point_cloud.ply"
    return p if p.exists() else Path("")


def _build_lpips(use_lpips: bool, device: torch.device):
    if not use_lpips:
        return None
    try:
        import lpips

        fn = lpips.LPIPS(net="vgg").to(device).eval()
        for p in fn.parameters():
            p.requires_grad_(False)
        return fn
    except Exception:
        return None


def compute_proxy_metrics(
    src_paths: Sequence[Path],
    edit_paths: Sequence[Path],
    *,
    use_lpips: bool,
    device: str,
    source_model_dir: Optional[Path] = None,
    edit_model_dir: Optional[Path] = None,
    source_iter: int = -1,
    edit_iter: int = -1,
) -> Dict[str, object]:
    src_map = {p.name: p for p in src_paths}
    edit_map = {p.name: p for p in edit_paths}
    common = sorted(set(src_map.keys()) & set(edit_map.keys()))
    if not common:
        raise RuntimeError("No common view files between source and edited renders.")

    dev = torch.device(device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    lpips_fn = _build_lpips(use_lpips=use_lpips, device=dev)

    src_imgs: List[torch.Tensor] = []
    edit_imgs: List[torch.Tensor] = []

    l1s: List[float] = []
    psnrs: List[float] = []
    ssims: List[float] = []
    lpv: List[float] = []
    hf_ratio_list: List[float] = []
    clip_list: List[float] = []

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
            with torch.no_grad():
                ev = e.to(dev) * 2.0 - 1.0
                sv = s.to(dev) * 2.0 - 1.0
                lpv.append(float(lpips_fn(ev, sv).item()))

        hf_s = hf_energy(s)
        hf_e = hf_energy(e)
        hf_ratio_list.append(float(hf_e / (hf_s + 1e-8)))
        clip_list.append(clip_ratio(e))

    mv_rel = mv_distance_preservation(src_imgs, edit_imgs, max_views=24)

    src_vertices = -1
    edit_vertices = -1
    src_pc = Path("")
    edit_pc = Path("")

    if source_model_dir is not None:
        src_pc = find_point_cloud_file(source_model_dir, iteration=source_iter)
        if src_pc.exists():
            src_vertices = count_vertices_from_ply_header(src_pc)

    if edit_model_dir is not None:
        edit_pc = find_point_cloud_file(edit_model_dir, iteration=edit_iter)
        if edit_pc.exists():
            edit_vertices = count_vertices_from_ply_header(edit_pc)

    out: Dict[str, object] = {
        "num_views": len(common),
        "l1_to_src": float(np.mean(l1s)),
        "psnr_to_src": float(np.mean(psnrs)),
        "ssim_to_src": float(np.mean(ssims)),
        "lpips_to_src": float(np.mean(lpv)) if lpv else None,
        "hf_ratio_vs_src": float(np.mean(hf_ratio_list)),
        "clip_ratio": float(np.mean(clip_list)),
        "mv_rel_dist_mse": float(mv_rel),
        "src_vertices": float(src_vertices) if src_vertices > 0 else None,
        "edit_vertices": float(edit_vertices) if edit_vertices > 0 else None,
        "vertex_ratio": float(edit_vertices / (src_vertices + 1e-8)) if (src_vertices > 0 and edit_vertices > 0) else None,
        "src_point_cloud": str(src_pc) if src_pc else "",
        "edit_point_cloud": str(edit_pc) if edit_pc else "",
        "per_view": {
            "l1_to_src": l1s,
            "psnr_to_src": psnrs,
            "ssim_to_src": ssims,
            "lpips_to_src": lpv if lpv else None,
            "hf_ratio_vs_src": hf_ratio_list,
            "clip_ratio": clip_list,
        },
    }
    return out
