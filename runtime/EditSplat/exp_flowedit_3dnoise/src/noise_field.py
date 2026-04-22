import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn


@dataclass
class SparseNoiseFieldConfig:
    max_anchors: int = 60000
    voxel_res: int = 96
    coarse_res: int = 32
    anchor_mode: str = "coverage_opacity"
    init_mode: str = "coarse_smooth"
    hybrid_ratio: float = 0.5
    harmonic_mix: float = 0.7
    init_std: float = 1.0
    smooth_ratio: float = 0.65
    depth_gamma: float = 1.0
    max_visible: int = 120000
    seed: int = 0


def _bbox_normalize(xyz: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xyz_min = xyz.min(dim=0).values
    xyz_max = xyz.max(dim=0).values
    span = (xyz_max - xyz_min).clamp_min(eps)
    xyz01 = (xyz - xyz_min) / span
    return xyz01, xyz_min, xyz_max


def _voxel_keys(xyz01: torch.Tensor, res: int) -> torch.Tensor:
    xyz_i = torch.clamp((xyz01 * float(res)).floor().long(), min=0, max=res - 1)
    keys = xyz_i[:, 0] + res * (xyz_i[:, 1] + res * xyz_i[:, 2])
    return keys


def _select_sparse_anchors(
    xyz: torch.Tensor,
    opacity: torch.Tensor,
    cfg: SparseNoiseFieldConfig,
) -> torch.Tensor:
    device = xyz.device
    n = xyz.shape[0]
    xyz01, _, _ = _bbox_normalize(xyz)
    keys = _voxel_keys(xyz01, cfg.voxel_res)

    # Keep one representative per voxel to preserve coverage.
    g = torch.Generator(device=device)
    g.manual_seed(int(cfg.seed))
    perm = torch.randperm(n, generator=g, device=device)
    keys_perm = keys[perm]

    order = torch.argsort(keys_perm)
    sorted_idx = perm[order]
    sorted_keys = keys[sorted_idx]

    keep = torch.ones_like(sorted_keys, dtype=torch.bool)
    keep[1:] = sorted_keys[1:] != sorted_keys[:-1]
    anchors = sorted_idx[keep]

    if anchors.numel() > cfg.max_anchors:
        op = opacity[anchors].squeeze(-1).clamp_min(1e-8)
        probs = op / op.sum()
        sel = torch.multinomial(
            probs,
            num_samples=cfg.max_anchors,
            replacement=False,
            generator=g,
        )
        anchors = anchors[sel]

    return anchors


def _init_noise_coarse_smooth(
    anchor_xyz: torch.Tensor,
    channels: int,
    cfg: SparseNoiseFieldConfig,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    device = anchor_xyz.device
    dtype = anchor_xyz.dtype
    k = anchor_xyz.shape[0]

    raw = torch.randn((k, channels), generator=generator, device=device, dtype=dtype)
    anchor_xyz01, _, _ = _bbox_normalize(anchor_xyz)
    coarse_keys = _voxel_keys(anchor_xyz01, cfg.coarse_res)
    uniq, inverse = torch.unique(coarse_keys, sorted=True, return_inverse=True)
    n_cells = int(uniq.numel())

    sum_buf = torch.zeros((n_cells, channels), device=device, dtype=dtype)
    sum_buf.index_add_(0, inverse, raw)
    cnt = torch.bincount(inverse, minlength=n_cells).to(dtype).unsqueeze(-1).clamp_min(1.0)
    cell_mean = sum_buf / cnt

    smooth = cfg.smooth_ratio * cell_mean[inverse] + (1.0 - cfg.smooth_ratio) * raw
    return smooth, inverse, n_cells


def _init_noise_harmonic_lowfreq(
    anchor_xyz: torch.Tensor,
    channels: int,
    cfg: SparseNoiseFieldConfig,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    device = anchor_xyz.device
    dtype = anchor_xyz.dtype
    k = anchor_xyz.shape[0]
    anchor_xyz01, _, _ = _bbox_normalize(anchor_xyz)
    xyz = anchor_xyz01 * 2.0 - 1.0

    freqs = [1.0, 2.0, 4.0]
    basis = []
    for f in freqs:
        w = 2.0 * math.pi * f
        basis.append(torch.sin(w * xyz[:, 0]))
        basis.append(torch.cos(w * xyz[:, 0]))
        basis.append(torch.sin(w * xyz[:, 1]))
        basis.append(torch.cos(w * xyz[:, 1]))
        basis.append(torch.sin(w * xyz[:, 2]))
        basis.append(torch.cos(w * xyz[:, 2]))
    b = torch.stack(basis, dim=-1)  # [K, B]
    b = b - b.mean(dim=0, keepdim=True)
    b = b / b.std(dim=0, keepdim=True).clamp_min(1e-4)

    w = torch.randn((b.shape[1], channels), generator=generator, device=device, dtype=dtype)
    harmonic = (b @ w) / math.sqrt(float(b.shape[1]))
    raw = torch.randn((k, channels), generator=generator, device=device, dtype=dtype)

    mix = float(cfg.harmonic_mix)
    base = mix * harmonic + (1.0 - mix) * raw

    coarse_keys = _voxel_keys(anchor_xyz01, cfg.coarse_res)
    uniq, inverse = torch.unique(coarse_keys, sorted=True, return_inverse=True)
    n_cells = int(uniq.numel())

    sum_buf = torch.zeros((n_cells, channels), device=device, dtype=dtype)
    sum_buf.index_add_(0, inverse, base)
    cnt = torch.bincount(inverse, minlength=n_cells).to(dtype).unsqueeze(-1).clamp_min(1.0)
    cell_mean = sum_buf / cnt
    smooth = cfg.smooth_ratio * cell_mean[inverse] + (1.0 - cfg.smooth_ratio) * base
    return smooth, inverse, n_cells


def init_noise_from_anchors(
    anchor_xyz: torch.Tensor,
    anchor_opacity: torch.Tensor,
    channels: int,
    cfg: SparseNoiseFieldConfig,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    mode = str(cfg.init_mode).lower()
    if mode == "coarse_smooth":
        smooth, inverse, n_cells = _init_noise_coarse_smooth(anchor_xyz, channels, cfg, generator)
    elif mode == "harmonic_lowfreq":
        smooth, inverse, n_cells = _init_noise_harmonic_lowfreq(anchor_xyz, channels, cfg, generator)
    else:
        raise ValueError(f"Unknown init_mode={cfg.init_mode}")

    amp = anchor_opacity.squeeze(-1).clamp_min(1e-4).sqrt().unsqueeze(-1)
    noise_init = cfg.init_std * smooth * amp
    return noise_init, inverse, n_cells


def _select_top_opacity_anchors(opacity: torch.Tensor, cfg: SparseNoiseFieldConfig) -> torch.Tensor:
    k = min(int(cfg.max_anchors), int(opacity.shape[0]))
    scores = opacity.squeeze(-1)
    return torch.topk(scores, k=k, largest=True, sorted=False).indices


def _select_hybrid_anchors(
    xyz: torch.Tensor,
    opacity: torch.Tensor,
    cfg: SparseNoiseFieldConfig,
) -> torch.Tensor:
    n_cov = max(1, int(round(float(cfg.max_anchors) * float(cfg.hybrid_ratio))))
    n_top = max(1, int(cfg.max_anchors - n_cov))

    cfg_cov = SparseNoiseFieldConfig(**{**cfg.__dict__, "max_anchors": n_cov})
    cov = _select_sparse_anchors(xyz, opacity, cfg_cov)

    cfg_top = SparseNoiseFieldConfig(**{**cfg.__dict__, "max_anchors": n_top})
    top = _select_top_opacity_anchors(opacity, cfg_top)

    all_idx = torch.cat([cov, top], dim=0)
    uniq = torch.unique(all_idx, sorted=False)

    if uniq.numel() > cfg.max_anchors:
        g = torch.Generator(device=uniq.device)
        g.manual_seed(int(cfg.seed) + 123)
        perm = torch.randperm(uniq.numel(), generator=g, device=uniq.device)[: int(cfg.max_anchors)]
        uniq = uniq[perm]

    return uniq


def select_anchor_indices(
    xyz: torch.Tensor,
    opacity: torch.Tensor,
    cfg: SparseNoiseFieldConfig,
) -> torch.Tensor:
    mode = str(cfg.anchor_mode).lower()
    if mode == "coverage_opacity":
        return _select_sparse_anchors(xyz, opacity, cfg)
    if mode == "opacity_topk":
        return _select_top_opacity_anchors(opacity, cfg)
    if mode == "hybrid":
        return _select_hybrid_anchors(xyz, opacity, cfg)
    raise ValueError(f"Unknown anchor_mode={cfg.anchor_mode}")


def _project_ndc(camera, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    ones = torch.ones((xyz.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
    xyz_h = torch.cat([xyz, ones], dim=-1)

    proj = camera.full_proj_transform
    if proj.device != xyz.device:
        proj = proj.to(xyz.device)
    proj = proj.to(xyz.dtype)

    clip = xyz_h @ proj
    w = clip[:, 3]
    ndc = clip[:, :3] / (w.unsqueeze(-1).clamp_min(1e-6))
    return ndc, w


class SparseGaussianNoiseField(nn.Module):
    """
    3D-consistent sparse noise field anchored on a subset of 3DGS primitives.

    Noise is optimized in anchor space and rendered to FLUX token grids by
    projected bilinear splatting.
    """

    def __init__(
        self,
        anchor_xyz: torch.Tensor,
        anchor_opacity: torch.Tensor,
        noise_init: torch.Tensor,
        coarse_inverse: torch.Tensor,
        num_coarse_cells: int,
        cfg: SparseNoiseFieldConfig,
    ):
        super().__init__()
        self.cfg = cfg

        self.register_buffer("anchor_xyz", anchor_xyz)
        self.register_buffer("anchor_opacity", anchor_opacity)
        self.register_buffer("coarse_inverse", coarse_inverse)
        self.num_coarse_cells = int(num_coarse_cells)

        self.noise = nn.Parameter(noise_init.clone())
        self.register_buffer("noise_init", noise_init.clone())

    @property
    def channels(self) -> int:
        return int(self.noise.shape[1])

    @property
    def num_anchors(self) -> int:
        return int(self.noise.shape[0])

    @classmethod
    def from_gaussians(
        cls,
        xyz: torch.Tensor,
        opacity: torch.Tensor,
        channels: int,
        cfg: Optional[SparseNoiseFieldConfig] = None,
    ) -> "SparseGaussianNoiseField":
        if cfg is None:
            cfg = SparseNoiseFieldConfig()

        anchor_idx = select_anchor_indices(xyz, opacity, cfg)
        anchor_xyz = xyz[anchor_idx].detach()
        anchor_opacity = opacity[anchor_idx].detach()

        device = anchor_xyz.device

        g = torch.Generator(device=device)
        g.manual_seed(int(cfg.seed) + 17)
        noise_init, inverse, n_cells = init_noise_from_anchors(
            anchor_xyz=anchor_xyz,
            anchor_opacity=anchor_opacity,
            channels=channels,
            cfg=cfg,
            generator=g,
        )

        return cls(
            anchor_xyz=anchor_xyz,
            anchor_opacity=anchor_opacity,
            noise_init=noise_init,
            coarse_inverse=inverse,
            num_coarse_cells=n_cells,
            cfg=cfg,
        )

    def smoothness_loss(self) -> torch.Tensor:
        sum_buf = self.noise.new_zeros((self.num_coarse_cells, self.channels))
        sum_buf.index_add_(0, self.coarse_inverse, self.noise)
        cnt = torch.bincount(
            self.coarse_inverse,
            minlength=self.num_coarse_cells,
        ).to(self.noise.dtype).unsqueeze(-1).clamp_min(1.0)
        cell_mean = sum_buf / cnt
        return ((self.noise - cell_mean[self.coarse_inverse]) ** 2).mean()

    def prior_loss(self) -> torch.Tensor:
        return ((self.noise - self.noise_init) ** 2).mean()

    def render_to_tokens(
        self,
        camera,
        token_h: int,
        token_w: int,
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        ndc, w = _project_ndc(camera, self.anchor_xyz)

        vis = (
            (w > 0.0)
            & (ndc[:, 0] >= -1.05)
            & (ndc[:, 0] <= 1.05)
            & (ndc[:, 1] >= -1.05)
            & (ndc[:, 1] <= 1.05)
            & (ndc[:, 2] >= -0.2)
            & (ndc[:, 2] <= 1.2)
        )

        if not torch.any(vis):
            out = torch.zeros(
                (1, token_h * token_w, self.channels),
                device=self.noise.device,
                dtype=self.noise.dtype,
            )
            meta = {"visible": 0.0, "used": 0.0, "std": 0.0}
            return out, meta

        uv = ndc[vis, :2]
        depth = ndc[vis, 2].clamp_min(0.0)
        n = self.noise[vis]
        op = self.anchor_opacity[vis].squeeze(-1)

        u = (uv[:, 0] * 0.5 + 0.5) * float(token_w - 1)
        v = (1.0 - (uv[:, 1] * 0.5 + 0.5)) * float(token_h - 1)
        u = u.clamp(0.0, float(token_w - 1) - 1e-4)
        v = v.clamp(0.0, float(token_h - 1) - 1e-4)

        weight = op * torch.exp(-self.cfg.depth_gamma * depth)

        m = weight.shape[0]
        if m > self.cfg.max_visible:
            topk = torch.topk(weight, k=self.cfg.max_visible, sorted=False).indices
            u = u[topk]
            v = v[topk]
            n = n[topk]
            weight = weight[topk]

        x0 = torch.floor(u).long()
        y0 = torch.floor(v).long()
        x1 = (x0 + 1).clamp(max=token_w - 1)
        y1 = (y0 + 1).clamp(max=token_h - 1)

        wx = (u - x0.to(u.dtype)).clamp(0.0, 1.0)
        wy = (v - y0.to(v.dtype)).clamp(0.0, 1.0)

        slots = token_h * token_w
        acc = torch.zeros((slots, self.channels), device=n.device, dtype=n.dtype)
        den = torch.zeros((slots, 1), device=n.device, dtype=n.dtype)

        def splat(ix: torch.Tensor, iy: torch.Tensor, w_local: torch.Tensor):
            idx = iy * token_w + ix
            contrib = n * (weight * w_local).unsqueeze(-1)
            acc.index_add_(0, idx, contrib)
            den.index_add_(0, idx, (weight * w_local).unsqueeze(-1))

        splat(x0, y0, (1.0 - wx) * (1.0 - wy))
        splat(x1, y0, wx * (1.0 - wy))
        splat(x0, y1, (1.0 - wx) * wy)
        splat(x1, y1, wx * wy)

        out = acc / den.clamp_min(1e-6)

        if normalize:
            out = out - out.mean(dim=0, keepdim=True)
            out = out / out.std(dim=0, keepdim=True).clamp_min(1e-4)

        out = out.unsqueeze(0)  # [1, L, C]
        std_val = float(out.std().detach().cpu())
        meta = {
            "visible": float(vis.sum().detach().cpu()),
            "used": float(weight.shape[0]),
            "std": std_val,
        }
        return out, meta

    def state_summary(self) -> Dict[str, float]:
        with torch.no_grad():
            return {
                "num_anchors": float(self.num_anchors),
                "channels": float(self.channels),
                "anchor_mode": str(self.cfg.anchor_mode),
                "init_mode": str(self.cfg.init_mode),
                "noise_mean": float(self.noise.mean().cpu()),
                "noise_std": float(self.noise.std().cpu()),
                "prior_l2": float(self.prior_loss().cpu()),
                "smooth_l2": float(self.smoothness_loss().cpu()),
            }
