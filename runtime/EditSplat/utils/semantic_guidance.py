from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


try:
    import torch
except Exception:  # pragma: no cover - optional in local test runner
    torch = None


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


def expand_loss_guidance_mask(mask: Any, background_weight: float) -> Any:
    background_weight = max(0.0, min(1.0, float(background_weight)))
    if _is_torch_tensor(mask):
        mask_t = mask.detach().clone().to(dtype=torch.float32)
        return mask_t + (1.0 - mask_t) * background_weight

    return [
        float(value) + (1.0 - float(value)) * background_weight
        for value in mask
    ]
