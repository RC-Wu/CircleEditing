from __future__ import annotations

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
        mask=merged_mask,
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
