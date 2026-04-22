from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _is_torch(x: Any) -> bool:
    return bool(torch is not None and isinstance(x, torch.Tensor))


def _to_float(x: Any) -> Any:
    if _is_torch(x):
        return x.to(dtype=torch.float32)
    return np.asarray(x, dtype=np.float32)


def _clip01(x: Any) -> Any:
    if _is_torch(x):
        return x.clamp(0.0, 1.0)
    return np.clip(x, 0.0, 1.0)


def _clip_signed(x: Any, limit: Optional[float]) -> Any:
    if limit is None:
        return x
    bound = float(max(0.0, limit))
    if _is_torch(x):
        return x.clamp(-bound, bound)
    return np.clip(x, -bound, bound)


def _maximum(a: Any, b: Any) -> Any:
    if _is_torch(a) or _is_torch(b):
        return torch.maximum(_to_float(a), _to_float(b))
    return np.maximum(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32))


def build_canonical_target(
    gt_view: Any,
    reprojected_edit: Any,
    reprojected_source: Any,
    residual_clamp: Optional[float] = None,
) -> Tuple[Any, Any]:
    gt = _to_float(gt_view)
    edit = _to_float(reprojected_edit)
    source = _to_float(reprojected_source)
    residual = edit - source
    residual = _clip_signed(residual, residual_clamp)
    target = _clip01(gt + residual)
    return target, residual


def blend_targets(
    flow_target: Any,
    canonical_target: Any,
    alpha: float,
) -> Any:
    mix = float(np.clip(alpha, 0.0, 1.0))
    flow = _to_float(flow_target)
    canonical = _to_float(canonical_target)
    return _clip01((1.0 - mix) * flow + mix * canonical)


def build_frozen_canonical_carrier(
    source_view: Any,
    teacher_edit: Any,
    flow_proxy: Any,
    edit_mask: Any,
    confidence_weight: Any,
    support_mask: Optional[Any] = None,
    prior_mask: Optional[Any] = None,
    residual_clamp: Optional[float] = None,
    teacher_residual_weight: float = 0.72,
    blend_alpha: float = 0.60,
) -> Dict[str, Any]:
    source = _to_float(source_view)
    teacher = _to_float(teacher_edit)
    proxy = _to_float(flow_proxy)
    mask = _clip01(_to_float(edit_mask))
    confidence = _clip01(_to_float(confidence_weight))

    if support_mask is not None:
        mask = _maximum(mask, _clip01(_to_float(support_mask)))
    if prior_mask is not None:
        mask = _maximum(mask, _clip01(_to_float(prior_mask)))
    carrier_mask = _clip01(mask * confidence)

    teacher_residual = _clip_signed(teacher - source, residual_clamp)
    flow_residual = _clip_signed(proxy - source, residual_clamp)
    teacher_mix = float(np.clip(teacher_residual_weight, 0.0, 1.0))
    carrier_residual = teacher_mix * teacher_residual + (1.0 - teacher_mix) * flow_residual
    carrier_target = _clip01(source + carrier_mask * carrier_residual)
    carrier_proxy = blend_targets(flow_target=proxy, canonical_target=carrier_target, alpha=blend_alpha)

    return {
        'source': source,
        'teacher_edit': teacher,
        'flow_proxy': proxy,
        'teacher_residual': teacher_residual,
        'flow_residual': flow_residual,
        'carrier_mask': carrier_mask,
        'carrier_residual': carrier_residual,
        'carrier_target': carrier_target,
        'carrier_proxy': carrier_proxy,
        'prior_mask': _clip01(_to_float(prior_mask)) if prior_mask is not None else None,
    }
