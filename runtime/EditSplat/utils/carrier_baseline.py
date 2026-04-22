from __future__ import annotations

from typing import Dict, Optional

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("torch is required for carrier baseline runtime")


def _as_weight_mask(mask: Optional["torch.Tensor"], reference: "torch.Tensor") -> "torch.Tensor":
    _require_torch()
    if mask is None:
        return torch.zeros((reference.shape[0], 1, reference.shape[-2], reference.shape[-1]), dtype=reference.dtype, device=reference.device)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[1] != 1:
        mask = mask.mean(dim=1, keepdim=True)
    return mask.to(device=reference.device, dtype=reference.dtype).clamp(0.0, 1.0)


def coerce_tensor_like(value: "torch.Tensor", reference: "torch.Tensor") -> "torch.Tensor":
    _require_torch()
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def build_a_baseline_carrier(
    source: "torch.Tensor",
    initial_edit: "torch.Tensor",
    mf_cond: "torch.Tensor",
    proxy: "torch.Tensor",
    geo_weight: Optional["torch.Tensor"] = None,
    support_mask: Optional["torch.Tensor"] = None,
    support_mix: float = 0.5,
    proxy_mix: float = 0.5,
    mask_floor: float = 0.0,
) -> Dict[str, "torch.Tensor"]:
    _require_torch()
    source = source.to(dtype=torch.float32)
    initial_edit = initial_edit.to(device=source.device, dtype=torch.float32)
    mf_cond = mf_cond.to(device=source.device, dtype=torch.float32)
    proxy = proxy.to(device=source.device, dtype=torch.float32)

    geo_mask = _as_weight_mask(geo_weight, source)
    support_weight = _as_weight_mask(support_mask, source)
    carrier_mask = torch.maximum(geo_mask, support_weight)
    if mask_floor > 0.0:
        carrier_mask = torch.maximum(carrier_mask, torch.full_like(carrier_mask, float(mask_floor)))
    carrier_mask = carrier_mask.clamp(0.0, 1.0)

    teacher_residual = initial_edit - source
    support_residual = mf_cond - source
    support_mix = float(max(0.0, min(1.0, support_mix)))
    proxy_mix = float(max(0.0, min(1.0, proxy_mix)))
    carrier_residual = (1.0 - support_mix) * teacher_residual + support_mix * support_residual
    carrier_target = (source + carrier_mask * carrier_residual).clamp(0.0, 1.0)
    carrier_proxy = ((1.0 - proxy_mix) * proxy + proxy_mix * carrier_target).clamp(0.0, 1.0)

    return {
        "carrier_mask": carrier_mask,
        "teacher_residual": teacher_residual,
        "support_residual": support_residual,
        "carrier_residual": carrier_residual,
        "carrier_target": carrier_target,
        "carrier_proxy": carrier_proxy,
    }


def prompt_separation_retention_ratio(teacher_mad: float, final_mad: float, eps: float = 1e-6) -> float:
    teacher = float(teacher_mad)
    final = float(final_mad)
    denom = teacher + float(eps)
    if denom <= 0.0:
        return 0.0 if final <= 0.0 else float("inf")
    return final / denom
