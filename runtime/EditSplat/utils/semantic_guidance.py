from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional


try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - optional in local test runner
    torch = None
    F = None


@dataclass
class SemanticGuidance:
    mask: Any
    color_scale: float
    position_scale: float
    used_support: bool


def _is_torch_tensor(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _clone_like(value: Any) -> Any:
    if _is_torch_tensor(value):
        return value.detach().clone().to(dtype=torch.float32)
    return [float(x) for x in value]


def _mask_values_list(mask: Any) -> Any:
    if _is_torch_tensor(mask):
        return mask.detach().to(dtype=torch.float32).reshape(-1)
    return [float(x) for x in mask]


def _threshold_key(threshold: float) -> str:
    return f"ge_{threshold:.2f}_ratio".replace(".", "_")


def _blend_masks(selected_mask: Any, support_mask: Any, support_weight: float) -> Any:
    support_weight = max(0.0, min(1.0, float(support_weight)))
    if _is_torch_tensor(selected_mask):
        selected = selected_mask.detach().clone().to(dtype=torch.float32)
        support = support_mask.detach().clone().to(dtype=torch.float32)
        blended = selected * (1.0 - support_weight) + support * support_weight
        return torch.maximum(selected, blended)

    selected = [float(x) for x in selected_mask]
    support = [float(x) for x in support_mask]
    blended = []
    for sel, sup in zip(selected, support):
        mix = sel * (1.0 - support_weight) + sup * support_weight
        blended.append(max(sel, mix))
    return blended


def build_semantic_guidance(
    selected_mask: Any,
    support_mask: Optional[Any],
    enabled: bool,
    support_weight: float,
    color_scale: float,
    position_scale: float,
    freeze_geometry: bool,
    mask_power: float = 1.0,
    label_threshold: float = 0.0,
    background_floor: float = 0.0,
) -> SemanticGuidance:
    base_mask = _clone_like(selected_mask)
    if not enabled or support_mask is None:
        return SemanticGuidance(
            mask=base_mask,
            color_scale=float(color_scale),
            position_scale=float(position_scale),
            used_support=False,
        )

    merged_mask = _blend_masks(selected_mask=base_mask, support_mask=support_mask, support_weight=support_weight)
    return SemanticGuidance(
        mask=refine_semantic_guidance_mask(
            mask=merged_mask,
            power=mask_power,
            threshold=label_threshold,
            background_floor=background_floor,
        ),
        color_scale=float(color_scale),
        position_scale=0.0 if freeze_geometry else float(position_scale),
        used_support=True,
    )


def normalize_gaussian_support_mask(weight_sum: Any, weight_count: Any, eps: float = 1e-7) -> Any:
    if _is_torch_tensor(weight_sum):
        weight_sum_t = weight_sum.detach().clone().to(dtype=torch.float32)
        weight_count_t = weight_count.detach().clone().to(dtype=torch.float32)
        normalized = weight_sum_t / (weight_count_t + float(eps))
        if normalized.ndim == 2 and normalized.shape[1] == 1:
            return normalized[:, 0]
        return normalized

    out = []
    for cur_sum, cur_count in zip(weight_sum, weight_count):
        count = float(cur_count)
        out.append(float(cur_sum) / (count + float(eps)) if count > 0.0 else 0.0)
    return out


def _mask_to_float_tensor(mask: Any) -> Any:
    if torch is None:
        raise RuntimeError("mask summarization requires torch")
    if _is_torch_tensor(mask):
        return mask.detach().to(dtype=torch.float32)
    return torch.as_tensor(mask, dtype=torch.float32)


def _quantile_linear(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    q = max(0.0, min(1.0, float(q)))
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - float(lo)
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def summarize_mask(mask: Any, thresholds: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9)) -> dict[str, Any]:
    values = _mask_values_list(mask)
    if _is_torch_tensor(values):
        flat = values.reshape(-1)
        if flat.numel() == 0:
            return {
                "count": 0,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "nonzero_ratio": 0.0,
                "quantiles": {},
                "mass_above": {},
            }

        quantile_points = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], dtype=torch.float32, device=flat.device)
        quantile_values = torch.quantile(flat, quantile_points).detach().cpu().tolist()
        summary = {
            "count": int(flat.numel()),
            "min": float(flat.min().item()),
            "max": float(flat.max().item()),
            "mean": float(flat.mean().item()),
            "std": float(flat.std(unbiased=False).item()),
            "nonzero_ratio": float((flat > 0.0).float().mean().item()),
            "quantiles": {
                "q05": float(quantile_values[0]),
                "q25": float(quantile_values[1]),
                "q50": float(quantile_values[2]),
                "q75": float(quantile_values[3]),
                "q95": float(quantile_values[4]),
            },
            "mass_above": {},
        }
        for threshold in thresholds:
            key = f"{float(threshold):.2f}"
            summary["mass_above"][key] = float((flat >= float(threshold)).float().mean().item())
        return summary

    flat = [float(x) for x in values]
    count = len(flat)
    if count == 0:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "nonzero_ratio": 0.0,
            "quantiles": {},
            "mass_above": {},
        }

    mean = sum(flat) / float(count)
    variance = sum((cur - mean) ** 2 for cur in flat) / float(count)
    sorted_values = sorted(flat)
    summary = {
        "count": count,
        "min": float(sorted_values[0]),
        "max": float(sorted_values[-1]),
        "mean": float(mean),
        "std": float(math.sqrt(max(variance, 0.0))),
        "nonzero_ratio": float(sum(1 for cur in flat if cur > 0.0) / float(count)),
        "quantiles": {
            "q05": _quantile_linear(sorted_values, 0.05),
            "q25": _quantile_linear(sorted_values, 0.25),
            "q50": _quantile_linear(sorted_values, 0.50),
            "q75": _quantile_linear(sorted_values, 0.75),
            "q95": _quantile_linear(sorted_values, 0.95),
        },
        "mass_above": {},
    }
    for threshold in thresholds:
        key = f"{float(threshold):.2f}"
        summary["mass_above"][key] = float(sum(1 for cur in flat if cur >= float(threshold)) / float(count))
    return summary


def summarize_mask_overlap(selected_mask: Any, support_mask: Any, threshold: float = 0.5) -> dict[str, Any]:
    selected_values = _mask_values_list(selected_mask)
    support_values = _mask_values_list(support_mask)
    if _is_torch_tensor(selected_values) and _is_torch_tensor(support_values):
        selected = selected_values.reshape(-1)
        support = support_values.reshape(-1)
        if selected.numel() != support.numel():
            raise ValueError("selected_mask and support_mask must have the same flattened size")
        if selected.numel() == 0:
            return {
                "threshold": float(threshold),
                "intersection": 0,
                "union": 0,
                "iou": 0.0,
                "selected_covered_by_support": 0.0,
                "support_covered_by_selected": 0.0,
                "support_outside_selected": 0.0,
                "selected_outside_support": 0.0,
                "mean_absolute_gap": 0.0,
            }

        selected_bin = selected >= float(threshold)
        support_bin = support >= float(threshold)
        intersection = int((selected_bin & support_bin).sum().item())
        union = int((selected_bin | support_bin).sum().item())
        selected_count = int(selected_bin.sum().item())
        support_count = int(support_bin.sum().item())
        support_only = int((support_bin & (~selected_bin)).sum().item())
        selected_only = int((selected_bin & (~support_bin)).sum().item())

        return {
            "threshold": float(threshold),
            "intersection": intersection,
            "union": union,
            "iou": float(intersection / union) if union > 0 else 0.0,
            "selected_covered_by_support": float(intersection / selected_count) if selected_count > 0 else 0.0,
            "support_covered_by_selected": float(intersection / support_count) if support_count > 0 else 0.0,
            "support_outside_selected": float(support_only / support_count) if support_count > 0 else 0.0,
            "selected_outside_support": float(selected_only / selected_count) if selected_count > 0 else 0.0,
            "mean_absolute_gap": float((selected - support).abs().mean().item()),
        }

    selected = [float(x) for x in selected_values]
    support = [float(x) for x in support_values]
    if len(selected) != len(support):
        raise ValueError("selected_mask and support_mask must have the same flattened size")
    if not selected:
        return {
            "threshold": float(threshold),
            "intersection": 0,
            "union": 0,
            "iou": 0.0,
            "selected_covered_by_support": 0.0,
            "support_covered_by_selected": 0.0,
            "support_outside_selected": 0.0,
            "selected_outside_support": 0.0,
            "mean_absolute_gap": 0.0,
        }

    selected_bin = [cur >= float(threshold) for cur in selected]
    support_bin = [cur >= float(threshold) for cur in support]
    intersection = sum(1 for a, b in zip(selected_bin, support_bin) if a and b)
    union = sum(1 for a, b in zip(selected_bin, support_bin) if a or b)
    selected_count = sum(1 for cur in selected_bin if cur)
    support_count = sum(1 for cur in support_bin if cur)
    support_only = sum(1 for a, b in zip(selected_bin, support_bin) if b and not a)
    selected_only = sum(1 for a, b in zip(selected_bin, support_bin) if a and not b)
    mean_absolute_gap = sum(abs(a - b) for a, b in zip(selected, support)) / float(len(selected))

    return {
        "threshold": float(threshold),
        "intersection": int(intersection),
        "union": int(union),
        "iou": float(intersection / union) if union > 0 else 0.0,
        "selected_covered_by_support": float(intersection / selected_count) if selected_count > 0 else 0.0,
        "support_covered_by_selected": float(intersection / support_count) if support_count > 0 else 0.0,
        "support_outside_selected": float(support_only / support_count) if support_count > 0 else 0.0,
        "selected_outside_support": float(selected_only / selected_count) if selected_count > 0 else 0.0,
        "mean_absolute_gap": float(mean_absolute_gap),
    }


def summarize_mask_distribution(
    mask: Any,
    thresholds: Any = (0.10, 0.25, 0.50, 0.75, 0.90),
) -> dict:
    values = _mask_values_list(mask)
    if _is_torch_tensor(values):
        numel = int(values.numel())
        if numel <= 0:
            base = {"numel": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "nonzero_ratio": 0.0}
        else:
            base = {
                "numel": numel,
                "mean": float(values.mean().item()),
                "std": float(values.std(unbiased=False).item()),
                "min": float(values.min().item()),
                "max": float(values.max().item()),
                "nonzero_ratio": float((values > 0).to(dtype=torch.float32).mean().item()),
            }
            for threshold in thresholds:
                base[_threshold_key(float(threshold))] = float(
                    (values >= float(threshold)).to(dtype=torch.float32).mean().item()
                )
        return base

    numel = len(values)
    if numel <= 0:
        base = {"numel": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "nonzero_ratio": 0.0}
    else:
        mean = sum(values) / float(numel)
        variance = sum((cur - mean) ** 2 for cur in values) / float(numel)
        base = {
            "numel": numel,
            "mean": float(mean),
            "std": float(math.sqrt(max(variance, 0.0))),
            "min": float(min(values)),
            "max": float(max(values)),
            "nonzero_ratio": float(sum(1.0 for cur in values if cur > 0.0) / float(numel)),
        }
        for threshold in thresholds:
            th = float(threshold)
            base[_threshold_key(th)] = float(sum(1.0 for cur in values if cur >= th) / float(numel))
    return base


def refine_semantic_guidance_mask(
    mask: Any,
    power: float = 1.0,
    threshold: float = 0.0,
    background_floor: float = 0.0,
) -> Any:
    power = max(float(power), 1e-6)
    threshold = max(0.0, min(1.0, float(threshold)))
    background_floor = max(0.0, min(1.0, float(background_floor)))

    if _is_torch_tensor(mask):
        mask_t = mask.detach().clone().to(dtype=torch.float32)
        refined = torch.where(mask_t >= threshold, mask_t.pow(power), torch.full_like(mask_t, background_floor))
        return refined.clamp(0.0, 1.0)

    refined = []
    for value in mask:
        cur = float(value)
        refined.append(cur ** power if cur >= threshold else background_floor)
    return [float(round(max(0.0, min(1.0, float(x))), 6)) for x in refined]


def summarize_gaussian_mask(mask: Any, label_threshold: float = 0.5) -> dict:
    values = _mask_values_list(mask)
    threshold = max(0.0, min(1.0, float(label_threshold)))
    if _is_torch_tensor(values):
        numel = int(values.numel())
        if numel <= 0:
            return {
                "count": 0,
                "active_count": 0,
                "foreground_count": 0,
                "foreground_ratio": 0.0,
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
        active_count = int((values > 0).sum().item())
        foreground_count = int((values >= threshold).sum().item())
        return {
            "count": numel,
            "active_count": active_count,
            "foreground_count": foreground_count,
            "foreground_ratio": float(foreground_count / float(numel)),
            "mean": float(values.mean().item()),
            "min": float(values.min().item()),
            "max": float(values.max().item()),
        }

    numel = len(values)
    if numel <= 0:
        return {
            "count": 0,
            "active_count": 0,
            "foreground_count": 0,
            "foreground_ratio": 0.0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    active_count = sum(1 for value in values if value > 0.0)
    foreground_count = sum(1 for value in values if value >= threshold)
    return {
        "count": numel,
        "active_count": int(active_count),
        "foreground_count": int(foreground_count),
        "foreground_ratio": float(foreground_count / float(numel)),
        "mean": float(sum(values) / float(numel)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _fov_to_focal(fov: Any, image_size: int, device: Any, dtype: Any) -> Any:
    half_fov = torch.as_tensor(float(fov) * 0.5, dtype=dtype, device=device)
    return (float(image_size) * 0.5) / torch.tan(half_fov)


def _prepare_image_mask(mask: Any, device: Any, dtype: Any) -> Any:
    if not _is_torch_tensor(mask):
        mask = torch.as_tensor(mask, dtype=dtype, device=device)
    else:
        mask = mask.to(device=device, dtype=dtype)

    if mask.ndim == 4:
        if mask.shape[0] == 1:
            mask = mask[0]
        else:
            mask = mask[:, 0]
    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask[0]
        else:
            mask = mask.amax(dim=0)
    if mask.ndim != 2:
        raise ValueError(f"Expected mask to reduce to [H, W], got shape={tuple(mask.shape)}")
    return mask.contiguous()


def _camera_center(camera: Any, device: Any, dtype: Any) -> Any:
    if hasattr(camera, "camera_center"):
        center = getattr(camera, "camera_center")
        if center is not None:
            return torch.as_tensor(center, dtype=dtype, device=device).reshape(3)

    rotation = torch.as_tensor(camera.R, dtype=dtype, device=device)
    translation = torch.as_tensor(camera.T, dtype=dtype, device=device).reshape(3)
    world_to_cam = torch.zeros((4, 4), dtype=dtype, device=device)
    world_to_cam[:3, :3] = rotation.transpose(0, 1)
    world_to_cam[:3, 3] = translation
    world_to_cam[3, 3] = 1.0
    return torch.linalg.inv(world_to_cam)[:3, 3]


def _sample_mask_values(mask_hw: Any, pixel_x: Any, pixel_y: Any) -> Any:
    height, width = mask_hw.shape
    if width <= 1 or height <= 1:
        return torch.zeros_like(pixel_x, dtype=torch.float32)

    grid_x = (pixel_x / float(width - 1)) * 2.0 - 1.0
    grid_y = (pixel_y / float(height - 1)) * 2.0 - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        mask_hw.view(1, 1, height, width),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.view(-1)


def _visibility_filter(pixel_x: Any, pixel_y: Any, depth: Any, width: int, height: int, depth_tolerance: float) -> Any:
    if float(depth_tolerance) <= 0.0:
        return torch.ones_like(depth, dtype=torch.bool)
    if not hasattr(torch.Tensor, "scatter_reduce_"):
        return torch.ones_like(depth, dtype=torch.bool)

    pixel_x_int = pixel_x.round().long().clamp(0, width - 1)
    pixel_y_int = pixel_y.round().long().clamp(0, height - 1)
    linear_idx = pixel_y_int * width + pixel_x_int
    pixel_count = width * height
    if pixel_count <= 0:
        return torch.ones_like(depth, dtype=torch.bool)

    min_depth = torch.full((pixel_count,), torch.inf, dtype=depth.dtype, device=depth.device)
    min_depth.scatter_reduce_(0, linear_idx, depth, reduce="amin", include_self=True)
    return depth <= (min_depth[linear_idx] + float(depth_tolerance))


def accumulate_projected_gaussian_mask(
    gaussian_xyz: Any,
    camera_list: Any,
    image_masks: Any,
    chunk_size: int = 65536,
    depth_tolerance: float = 0.0,
) -> Any:
    if torch is None or F is None:
        raise RuntimeError("accumulate_projected_gaussian_mask requires torch")

    if not _is_torch_tensor(gaussian_xyz):
        gaussian_xyz = torch.as_tensor(gaussian_xyz, dtype=torch.float32)
    gaussian_xyz = gaussian_xyz.detach().to(dtype=torch.float32)
    if gaussian_xyz.ndim != 2 or gaussian_xyz.shape[1] != 3:
        raise ValueError(f"Expected gaussian_xyz to have shape [N, 3], got {tuple(gaussian_xyz.shape)}")

    if not isinstance(camera_list, (list, tuple)):
        camera_list = [camera_list]
    if not isinstance(image_masks, (list, tuple)):
        image_masks = [image_masks]
    if len(camera_list) != len(image_masks):
        raise ValueError("camera_list and image_masks must have the same length")

    device = gaussian_xyz.device
    weight_sum = torch.zeros((gaussian_xyz.shape[0],), dtype=torch.float32, device=device)
    weight_count = torch.zeros((gaussian_xyz.shape[0],), dtype=torch.float32, device=device)
    step = max(1, int(chunk_size))

    for camera, image_mask in zip(camera_list, image_masks):
        mask_hw = _prepare_image_mask(image_mask, device=device, dtype=torch.float32)
        height, width = mask_hw.shape
        rotation = torch.as_tensor(camera.R, dtype=torch.float32, device=device)
        camera_center = _camera_center(camera, device=device, dtype=torch.float32)
        focal_x = _fov_to_focal(camera.FoVx, width, device=device, dtype=torch.float32)
        focal_y = _fov_to_focal(camera.FoVy, height, device=device, dtype=torch.float32)

        for start in range(0, gaussian_xyz.shape[0], step):
            end = min(start + step, gaussian_xyz.shape[0])
            xyz_chunk = gaussian_xyz[start:end]
            points_cam = (rotation.transpose(0, 1) @ (xyz_chunk - camera_center).transpose(0, 1)).transpose(0, 1)
            depth = points_cam[:, 2]
            valid = depth > 1e-6
            if not bool(valid.any()):
                continue

            safe_depth = depth.clamp_min(1e-6)
            pixel_x = focal_x * (points_cam[:, 0] / safe_depth) + (float(width) * 0.5)
            pixel_y = focal_y * (points_cam[:, 1] / safe_depth) + (float(height) * 0.5)

            valid = valid & (pixel_x >= 0.0) & (pixel_x <= float(width - 1)) & (pixel_y >= 0.0) & (pixel_y <= float(height - 1))
            if not bool(valid.any()):
                continue

            valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
            visible = _visibility_filter(
                pixel_x=pixel_x[valid],
                pixel_y=pixel_y[valid],
                depth=depth[valid],
                width=width,
                height=height,
                depth_tolerance=depth_tolerance,
            )
            if not bool(visible.any()):
                continue

            valid_idx = valid_idx[visible]
            sampled = _sample_mask_values(
                mask_hw=mask_hw,
                pixel_x=pixel_x[valid][visible],
                pixel_y=pixel_y[valid][visible],
            )
            weight_sum_chunk = weight_sum[start:end]
            weight_count_chunk = weight_count[start:end]
            weight_sum_chunk[valid_idx] += sampled.to(dtype=torch.float32)
            weight_count_chunk[valid_idx] += 1.0
            weight_sum[start:end] = weight_sum_chunk
            weight_count[start:end] = weight_count_chunk

    return weight_sum, weight_count


def expand_loss_guidance_mask(mask: Any, background_weight: float) -> Any:
    background_weight = max(0.0, min(1.0, float(background_weight)))
    if _is_torch_tensor(mask):
        mask_t = mask.detach().clone().to(dtype=torch.float32)
        return mask_t + (1.0 - mask_t) * background_weight

    return [
        float(value) + (1.0 - float(value)) * background_weight
        for value in mask
    ]
