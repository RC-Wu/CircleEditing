import torch
import torch.nn.functional as F


def _ensure_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:
        if x.shape[0] in (1, 3):
            x = x.unsqueeze(0)
        else:
            x = x.unsqueeze(1)
    return x.detach().float()


def _ensure_mask(mask: torch.Tensor, size) -> torch.Tensor:
    mask = _ensure_bchw(mask)
    if mask.shape[1] != 1:
        mask = mask.mean(dim=1, keepdim=True)
    if tuple(mask.shape[-2:]) != tuple(size):
        mask = F.interpolate(mask, size=size, mode="bilinear", align_corners=False)
    return mask.clamp(0.0, 1.0)


def _mask_bbox(mask_2d: torch.Tensor):
    ys, xs = torch.where(mask_2d > 0.5)
    if ys.numel() == 0:
        return None
    return int(ys.min().item()), int(ys.max().item()) + 1, int(xs.min().item()), int(xs.max().item()) + 1


def _proxy_region_stats(proxy: torch.Tensor, mask: torch.Tensor):
    proxy = _ensure_bchw(proxy)
    mask = _ensure_mask(mask, proxy.shape[-2:]).to(device=proxy.device, dtype=proxy.dtype)
    region = mask > 0.5
    coverage = float(region.float().mean().item())
    if not bool(region.any()):
        return {"mean": 1.0, "std": 1.0, "coverage": coverage, "pixel_count": 0}
    luminance = proxy.mean(dim=1, keepdim=True)
    values = luminance[region]
    std = float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0
    return {
        "mean": float(values.mean().item()),
        "std": std,
        "coverage": coverage,
        "pixel_count": int(values.numel()),
    }


def _should_use_frontier_fallback(stats, mean_threshold: float = 0.08, std_threshold: float = 0.02, min_coverage: float = 0.005) -> bool:
    coverage = float(stats.get("coverage", 0.0))
    mean = float(stats.get("mean", 1.0))
    std = float(stats.get("std", 1.0))
    if coverage < float(min_coverage):
        return False
    if mean < float(mean_threshold):
        return True
    return mean < max(float(mean_threshold) * 2.0, 0.2) and std < float(std_threshold)


def _soften_mask(mask: torch.Tensor, feather_radius: int) -> torch.Tensor:
    feather_radius = max(0, int(feather_radius))
    if feather_radius <= 1:
        return mask.clamp(0.0, 1.0)
    kernel = feather_radius if feather_radius % 2 == 1 else feather_radius + 1
    return F.avg_pool2d(mask, kernel_size=kernel, stride=1, padding=kernel // 2).clamp(0.0, 1.0)


def _compose_anchor_face_fallback(
    proxy: torch.Tensor,
    gt: torch.Tensor,
    anchor: torch.Tensor,
    anchor_mask: torch.Tensor,
    target_mask: torch.Tensor,
    feather_radius: int = 9,
) -> torch.Tensor:
    proxy = _ensure_bchw(proxy)
    gt = _ensure_bchw(gt).to(device=proxy.device, dtype=proxy.dtype)
    anchor = _ensure_bchw(anchor).to(device=proxy.device, dtype=proxy.dtype)
    if tuple(gt.shape[-2:]) != tuple(proxy.shape[-2:]):
        gt = F.interpolate(gt, size=proxy.shape[-2:], mode="bilinear", align_corners=False)
    if tuple(anchor.shape[-2:]) != tuple(proxy.shape[-2:]):
        anchor = F.interpolate(anchor, size=proxy.shape[-2:], mode="bilinear", align_corners=False)

    anchor_mask = _ensure_mask(anchor_mask, proxy.shape[-2:]).to(device=proxy.device, dtype=proxy.dtype)
    target_mask = _ensure_mask(target_mask, proxy.shape[-2:]).to(device=proxy.device, dtype=proxy.dtype)

    anchor_box = _mask_bbox(anchor_mask[0, 0])
    target_box = _mask_bbox(target_mask[0, 0])
    if anchor_box is None or target_box is None:
        return proxy.clone()

    ay0, ay1, ax0, ax1 = anchor_box
    ty0, ty1, tx0, tx1 = target_box
    target_h = max(1, ty1 - ty0)
    target_w = max(1, tx1 - tx0)

    anchor_crop = anchor[:, :, ay0:ay1, ax0:ax1]
    anchor_mask_crop = anchor_mask[:, :, ay0:ay1, ax0:ax1]
    target_crop_mask = target_mask[:, :, ty0:ty1, tx0:tx1]
    if anchor_crop.numel() == 0 or target_crop_mask.numel() == 0:
        return proxy.clone()

    anchor_crop = F.interpolate(anchor_crop, size=(target_h, target_w), mode="bilinear", align_corners=False)
    anchor_mask_crop = F.interpolate(anchor_mask_crop, size=(target_h, target_w), mode="bilinear", align_corners=False)
    soft_mask = _soften_mask(anchor_mask_crop * target_crop_mask, feather_radius=feather_radius) * target_crop_mask

    out = proxy.clone()
    proxy_crop = out[:, :, ty0:ty1, tx0:tx1]
    gt_crop = gt[:, :, ty0:ty1, tx0:tx1]
    blended = proxy_crop * (1.0 - soft_mask) + anchor_crop * soft_mask
    blended = gt_crop * (1.0 - target_crop_mask) + blended * target_crop_mask
    out[:, :, ty0:ty1, tx0:tx1] = blended
    return out


def _blend_face_override(
    base: torch.Tensor,
    override: torch.Tensor,
    mask: torch.Tensor,
    feather_radius: int = 9,
) -> torch.Tensor:
    base = _ensure_bchw(base)
    override = _ensure_bchw(override).to(device=base.device, dtype=base.dtype)
    if tuple(override.shape[-2:]) != tuple(base.shape[-2:]):
        override = F.interpolate(override, size=base.shape[-2:], mode="bilinear", align_corners=False)

    mask = _ensure_mask(mask, base.shape[-2:]).to(device=base.device, dtype=base.dtype)
    soft_mask = _soften_mask(mask, feather_radius=feather_radius) * mask
    return base * (1.0 - soft_mask) + override * soft_mask
