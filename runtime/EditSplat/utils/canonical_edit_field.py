from __future__ import annotations

from typing import Any, Optional, Tuple

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
