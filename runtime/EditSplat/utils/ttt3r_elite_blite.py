from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple

import numpy as np

try:  # pragma: no cover - optional in lightweight smoke environments
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _is_torch_tensor(x: Any) -> bool:
    return bool(torch is not None and isinstance(x, torch.Tensor))


def _to_work_tensor(x: Any) -> Any:
    if _is_torch_tensor(x):
        return x.to(dtype=torch.float32)
    return np.asarray(x, dtype=np.float32)


def _clip01(x: Any) -> Any:
    if _is_torch_tensor(x):
        return x.clamp(0.0, 1.0)
    return np.clip(x, 0.0, 1.0)


def _ones_like(x: Any) -> Any:
    if _is_torch_tensor(x):
        return torch.ones_like(x, dtype=torch.float32)
    return np.ones_like(np.asarray(x, dtype=np.float32), dtype=np.float32)


def _mul(a: Any, b: Any) -> Any:
    return a * b


def _to_scalar(x: Any, op: str) -> float:
    if _is_torch_tensor(x):
        fn = getattr(x.float(), op)
        return float(fn().item())
    arr = np.asarray(x, dtype=np.float32)
    return float(getattr(np, op)(arr))


def _safe_shape2d(x: Any) -> Tuple[int, int]:
    shape = tuple(getattr(x, "shape", ()))
    if len(shape) < 2:
        return (1, 1)
    return int(shape[-2]), int(shape[-1])


def _prepare_like(x: Optional[Any], reference: Any) -> Optional[Any]:
    if x is None:
        return None
    ref = _to_work_tensor(reference)
    out = _to_work_tensor(x)
    ref_h, ref_w = _safe_shape2d(ref)
    out_h, out_w = _safe_shape2d(out)
    if (out_h, out_w) == (ref_h, ref_w):
        return _clip01(out)
    if _is_torch_tensor(out) and _is_torch_tensor(ref):
        import torch.nn.functional as F  # local import keeps numpy-only smoke lightweight

        if out.ndim == 2:
            out = out.unsqueeze(0).unsqueeze(0)
        elif out.ndim == 3:
            out = out.unsqueeze(0)
        out = F.interpolate(out.float(), size=(ref_h, ref_w), mode="bilinear", align_corners=False)
        if out.ndim == 4 and out.shape[0] == 1:
            out = out[0]
        return _clip01(out)
    # numpy fallback (nearest-like resize via repeat/crop) for smoke tests.
    out_np = np.asarray(out, dtype=np.float32)
    y_repeat = max(1, int(np.ceil(ref_h / max(1, out_h))))
    x_repeat = max(1, int(np.ceil(ref_w / max(1, out_w))))
    out_np = np.repeat(np.repeat(out_np, y_repeat, axis=-2), x_repeat, axis=-1)
    out_np = out_np[..., :ref_h, :ref_w]
    return _clip01(out_np)


@dataclass
class ELiteCorrectionConfig:
    enabled: bool = False
    support_alpha: float = 0.0
    edit_alpha: float = 0.0
    confidence_alpha: float = 0.0
    scale_min: float = 0.0
    scale_max: float = 1.0


def apply_elite_correction_weights(
    edit_weight: Any,
    preserve_weight: Any,
    support_weight: Optional[Any],
    edit_mask: Optional[Any],
    confidence_weight: Optional[Any],
    cfg: ELiteCorrectionConfig,
) -> Tuple[Any, Any, Any]:
    if not bool(cfg.enabled):
        return edit_weight, preserve_weight, _ones_like(edit_weight)

    edit_w = _clip01(_to_work_tensor(edit_weight))
    preserve_w = _clip01(_to_work_tensor(preserve_weight))
    combo = _ones_like(edit_w)

    support = _prepare_like(support_weight, edit_w)
    edit = _prepare_like(edit_mask, edit_w)
    confidence = _prepare_like(confidence_weight, edit_w)

    if support is not None and float(cfg.support_alpha) > 0.0:
        alpha = float(np.clip(cfg.support_alpha, 0.0, 1.0))
        combo = _mul(combo, (1.0 - alpha) + alpha * _clip01(support))
    if edit is not None and float(cfg.edit_alpha) > 0.0:
        alpha = float(np.clip(cfg.edit_alpha, 0.0, 1.0))
        combo = _mul(combo, (1.0 - alpha) + alpha * _clip01(edit))
    if confidence is not None and float(cfg.confidence_alpha) > 0.0:
        alpha = float(np.clip(cfg.confidence_alpha, 0.0, 1.0))
        combo = _mul(combo, (1.0 - alpha) + alpha * _clip01(confidence))

    min_scale = float(max(0.0, cfg.scale_min))
    max_scale = float(max(min_scale, cfg.scale_max))
    if _is_torch_tensor(combo):
        combo = combo.clamp(min_scale, max_scale)
    else:
        combo = np.clip(combo, min_scale, max_scale)
    return _clip01(_mul(edit_w, combo)), _clip01(_mul(preserve_w, combo)), combo


def _tensor_stats(tensor: Any) -> Dict[str, float]:
    w = _clip01(_to_work_tensor(tensor))
    return {
        "mean": _to_scalar(w, "mean"),
        "min": _to_scalar(w, "min"),
        "max": _to_scalar(w, "max"),
        "mass": _to_scalar(w, "sum"),
    }


@dataclass
class SourceCanonicalPrior:
    schema: str = "source_canonical_prior_v0"
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    updated_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    total_updates: int = 0
    views: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_serializable(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "total_updates": int(self.total_updates),
            "num_views": int(len(self.views)),
            "views": self.views,
        }


def update_source_canonical_prior(
    prior: SourceCanonicalPrior,
    view_idx: int,
    edit_weight: Any,
    preserve_weight: Any,
    confidence_weight: Any,
    support_weight: Optional[Any] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> MutableMapping[str, Any]:
    key = str(int(view_idx))
    entry = dict(prior.views.get(key, {}))
    old_count = int(entry.get("count", 0))
    new_count = old_count + 1

    edit_stats = _tensor_stats(edit_weight)
    preserve_stats = _tensor_stats(preserve_weight)
    confidence_stats = _tensor_stats(confidence_weight)
    support_stats = _tensor_stats(support_weight) if support_weight is not None else None

    def _running_mean(prev: float, curr: float) -> float:
        if old_count <= 0:
            return float(curr)
        return float((prev * old_count + curr) / new_count)

    running = dict(entry.get("running_mean", {}))
    running["edit_mean"] = _running_mean(float(running.get("edit_mean", 0.0)), edit_stats["mean"])
    running["preserve_mean"] = _running_mean(float(running.get("preserve_mean", 0.0)), preserve_stats["mean"])
    running["confidence_mean"] = _running_mean(float(running.get("confidence_mean", 0.0)), confidence_stats["mean"])
    if support_stats is not None:
        running["support_mean"] = _running_mean(float(running.get("support_mean", 0.0)), support_stats["mean"])

    snapshot = {
        "edit": edit_stats,
        "preserve": preserve_stats,
        "confidence": confidence_stats,
        "support": support_stats,
        "metadata": dict(metadata or {}),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    entry.update(
        {
            "count": new_count,
            "running_mean": running,
            "last": snapshot,
        }
    )
    prior.views[key] = entry
    prior.total_updates += 1
    prior.updated_at_utc = snapshot["updated_at_utc"]
    return entry
