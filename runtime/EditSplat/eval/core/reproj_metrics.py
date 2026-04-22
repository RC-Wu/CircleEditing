from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class CameraParam:
    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    R: torch.Tensor
    T: torch.Tensor
    C: torch.Tensor


def _fov_to_focal(fov: float, image_size: int) -> float:
    return float((image_size / 2.0) / np.tan(float(fov) / 2.0))


def build_camera_params(camera_list: Sequence[object], view_names: Sequence[str], device: torch.device) -> Dict[str, CameraParam]:
    out: Dict[str, CameraParam] = {}
    for name in view_names:
        idx = int(Path(name).stem)
        cam = camera_list[idx]
        w = int(cam.image_width)
        h = int(cam.image_height)
        fx = _fov_to_focal(float(cam.FoVx), w)
        fy = _fov_to_focal(float(cam.FoVy), h)
        cx = float(w / 2.0)
        cy = float(h / 2.0)
        R = torch.tensor(cam.R, dtype=torch.float32, device=device)
        T = torch.tensor(cam.T, dtype=torch.float32, device=device)
        # Camera center: C = -R * T
        C = (-R @ T.view(3, 1)).view(3)
        out[name] = CameraParam(
            name=name,
            width=w,
            height=h,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            R=R,
            T=T,
            C=C,
        )
    return out


def _load_img(path: Path, device: torch.device) -> torch.Tensor:
    arr = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t


def _load_depth(path: Path, device: torch.device) -> torch.Tensor:
    dep = np.load(path).astype(np.float32)
    return torch.from_numpy(dep).to(device)


def _project_points(cam_i: CameraParam, cam_j: CameraParam, depth_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns projected coordinates from i->j:
    - u_j: [N]
    - v_j: [N]
    - z_j: [N]
    """

    h, w = depth_i.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=depth_i.device, dtype=torch.float32),
        torch.arange(w, device=depth_i.device, dtype=torch.float32),
        indexing="ij",
    )
    z = depth_i.reshape(-1)

    x_ci = (xx.reshape(-1) - cam_i.cx) * z / cam_i.fx
    y_ci = (yy.reshape(-1) - cam_i.cy) * z / cam_i.fy
    pts_ci = torch.stack([x_ci, y_ci, z], dim=1)

    # camera i -> world : X_w = R_i * X_ci + C_i
    pts_w = (cam_i.R @ pts_ci.T).T + cam_i.C.view(1, 3)

    # world -> camera j : X_cj = R_j^T * (X_w - C_j)
    pts_cj = (cam_j.R.T @ (pts_w - cam_j.C.view(1, 3)).T).T

    z_j = pts_cj[:, 2]
    u_j = cam_j.fx * (pts_cj[:, 0] / z_j.clamp_min(1e-8)) + cam_j.cx
    v_j = cam_j.fy * (pts_cj[:, 1] / z_j.clamp_min(1e-8)) + cam_j.cy
    return u_j, v_j, z_j


def _sample_image_at_uv(img_bchw: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Sample image at floating pixel coords; return [N, 3]."""

    _, _, h, w = img_bchw.shape
    x_norm = (u / max(w - 1, 1)) * 2.0 - 1.0
    y_norm = (v / max(h - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).view(1, 1, -1, 2)
    samp = F.grid_sample(img_bchw, grid, mode="bilinear", align_corners=True)
    return samp.view(3, -1).T


def _sample_depth_at_uv(depth_hw: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    x = depth_hw.unsqueeze(0).unsqueeze(0)
    _, _, h, w = x.shape
    x_norm = (u / max(w - 1, 1)) * 2.0 - 1.0
    y_norm = (v / max(h - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).view(1, 1, -1, 2)
    samp = F.grid_sample(x, grid, mode="bilinear", align_corners=True)
    return samp.view(-1)


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


def _pair_metric(
    *,
    img_i: torch.Tensor,
    img_j: torch.Tensor,
    depth_i: torch.Tensor,
    depth_j: torch.Tensor,
    cam_i: CameraParam,
    cam_j: CameraParam,
    occ_abs: float,
    occ_rel: float,
    lpips_fn,
) -> Tuple[float, float, float]:
    """Return (L1, LPIPS, visible_ratio) for i->j."""

    u_j, v_j, z_j = _project_points(cam_i, cam_j, depth_i)

    h, w = cam_j.height, cam_j.width
    valid = (
        (z_j > 1e-6)
        & (u_j >= 0.0)
        & (u_j <= (w - 1))
        & (v_j >= 0.0)
        & (v_j <= (h - 1))
    )
    if valid.sum().item() < 16:
        return 0.0, 0.0, 0.0

    u_v = u_j[valid]
    v_v = v_j[valid]
    z_v = z_j[valid]

    depth_j_proj = _sample_depth_at_uv(depth_j, u_v, v_v)
    depth_ok = depth_j_proj > 1e-6
    depth_tol = occ_abs + occ_rel * depth_j_proj.abs()
    occ_ok = (z_v - depth_j_proj).abs() <= depth_tol
    valid2 = depth_ok & occ_ok
    if valid2.sum().item() < 16:
        return 0.0, 0.0, 0.0

    u_f = u_v[valid2]
    v_f = v_v[valid2]

    # src colors from image i at integer grid (flattened order aligns with projection order)
    src_flat = img_i.view(3, -1).T
    src_colors = src_flat[valid][valid2]
    dst_colors = _sample_image_at_uv(img_j, u_f, v_f)

    l1 = float((src_colors - dst_colors).abs().mean().item())

    vis_ratio = float(valid2.float().mean().item())

    lp = 0.0
    if lpips_fn is not None and src_colors.shape[0] >= 64:
        # Approximate masked LPIPS by splatting sparse correspondences to target plane.
        yy = torch.round(v_f).long().clamp(0, h - 1)
        xx = torch.round(u_f).long().clamp(0, w - 1)
        lin = yy * w + xx

        src_canvas = torch.zeros((1, 3, h, w), dtype=img_i.dtype, device=img_i.device)
        dst_canvas = torch.zeros((1, 3, h, w), dtype=img_i.dtype, device=img_i.device)
        cnt = torch.zeros((1, 1, h, w), dtype=img_i.dtype, device=img_i.device)

        src_flat_canvas = src_canvas.view(3, -1)
        dst_flat_canvas = dst_canvas.view(3, -1)
        cnt_flat = cnt.view(-1)

        src_flat_canvas.index_add_(1, lin, src_colors.T)
        dst_flat_canvas.index_add_(1, lin, dst_colors.T)
        ones = torch.ones_like(lin, dtype=img_i.dtype)
        cnt_flat.index_add_(0, lin, ones)

        cnt_safe = cnt_flat.clamp_min(1.0)
        src_flat_canvas = src_flat_canvas / cnt_safe.unsqueeze(0)
        dst_flat_canvas = dst_flat_canvas / cnt_safe.unsqueeze(0)

        mask = (cnt_flat > 0).view(1, 1, h, w).float()
        src_canvas = src_flat_canvas.view(1, 3, h, w) * mask
        dst_canvas = dst_flat_canvas.view(1, 3, h, w) * mask

        with torch.no_grad():
            lp_val = lpips_fn(src_canvas * 2.0 - 1.0, dst_canvas * 2.0 - 1.0)
        lp = float(lp_val.item()) / max(vis_ratio, 1e-6)

    return l1, lp, vis_ratio


def compute_reprojection_consistency(
    *,
    edit_rgb_dir: Path,
    src_depth_dir: Path,
    view_names: Sequence[str],
    camera_list: Sequence[object],
    pairs_per_sample: int,
    seed: int,
    occ_abs: float,
    occ_rel: float,
    use_lpips: bool,
    device: str,
) -> Dict[str, object]:
    dev = torch.device(device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")

    cam_params = build_camera_params(camera_list=camera_list, view_names=view_names, device=dev)
    lpips_fn = _build_lpips(use_lpips=use_lpips, device=dev)

    images: Dict[str, torch.Tensor] = {}
    depths: Dict[str, torch.Tensor] = {}
    valid_names: List[str] = []

    for name in view_names:
        p_img = edit_rgb_dir / name
        p_dep = src_depth_dir / f"{Path(name).stem}.npy"
        if (not p_img.exists()) or (not p_dep.exists()):
            continue
        images[name] = _load_img(p_img, dev)
        depths[name] = _load_depth(p_dep, dev)
        valid_names.append(name)

    n = len(valid_names)
    if n < 2:
        return {
            "reproj_l1_mean": None,
            "reproj_lpips_mean": None,
            "reproj_visible_ratio_mean": None,
            "reproj_num_pairs": 0,
            "reproj_pair_l1": [],
            "reproj_pair_lpips": [],
            "reproj_pair_visible_ratio": [],
        }

    all_pairs: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            all_pairs.append((i, j))

    rng = random.Random(seed)
    if len(all_pairs) > pairs_per_sample > 0:
        idx = rng.sample(range(len(all_pairs)), pairs_per_sample)
        pairs = [all_pairs[i] for i in idx]
    else:
        pairs = all_pairs

    l1_vals: List[float] = []
    lpips_vals: List[float] = []
    vis_vals: List[float] = []

    # Bidirectional pair metric.
    for i, j in pairs:
        ni = valid_names[i]
        nj = valid_names[j]
        l1_ij, lp_ij, vis_ij = _pair_metric(
            img_i=images[ni],
            img_j=images[nj],
            depth_i=depths[ni],
            depth_j=depths[nj],
            cam_i=cam_params[ni],
            cam_j=cam_params[nj],
            occ_abs=occ_abs,
            occ_rel=occ_rel,
            lpips_fn=lpips_fn,
        )
        l1_ji, lp_ji, vis_ji = _pair_metric(
            img_i=images[nj],
            img_j=images[ni],
            depth_i=depths[nj],
            depth_j=depths[ni],
            cam_i=cam_params[nj],
            cam_j=cam_params[ni],
            occ_abs=occ_abs,
            occ_rel=occ_rel,
            lpips_fn=lpips_fn,
        )

        if vis_ij > 0:
            l1_vals.append(l1_ij)
            vis_vals.append(vis_ij)
            if lpips_fn is not None:
                lpips_vals.append(lp_ij)
        if vis_ji > 0:
            l1_vals.append(l1_ji)
            vis_vals.append(vis_ji)
            if lpips_fn is not None:
                lpips_vals.append(lp_ji)

    return {
        "reproj_l1_mean": float(np.mean(l1_vals)) if l1_vals else None,
        "reproj_lpips_mean": float(np.mean(lpips_vals)) if lpips_vals else None,
        "reproj_visible_ratio_mean": float(np.mean(vis_vals)) if vis_vals else None,
        "reproj_num_pairs": len(l1_vals),
        "reproj_pair_l1": l1_vals,
        "reproj_pair_lpips": lpips_vals if lpips_vals else None,
        "reproj_pair_visible_ratio": vis_vals,
    }
