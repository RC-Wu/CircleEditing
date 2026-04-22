from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .io_utils import load_tsv, maybe_read_json
from .types import BenchmarkEntry


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _find_row_from_tsv(tsv: Path, model_key: str) -> Dict[str, Any]:
    _, rows = load_tsv(tsv)
    if not rows:
        return {}

    model_key = (model_key or "").strip()
    if model_key:
        for row in rows:
            if row.get("model", "") == model_key:
                return row

    # fallback: return first done row
    for row in rows:
        if row.get("status", "").strip().lower() == "done":
            return row
    return rows[0]


def collect_efficiency(entry: BenchmarkEntry) -> Dict[str, Any]:
    """Collect runtime/memory metadata from benchmark fields and known files."""

    out: Dict[str, Any] = {
        "runtime_sec": entry.runtime_sec,
        "peak_mem_mib": entry.peak_mem_mib,
        "flow_steps": None,
        "flow_src_guidance": None,
        "flow_tar_guidance": None,
        "flow_n_max": None,
        "flow_seed": None,
        "efficiency_source": "",
    }

    args = maybe_read_json(Path(entry.model_dir) / "args.json")
    if args:
        out["flow_steps"] = _to_float(args.get("flow_steps", None))
        out["flow_src_guidance"] = _to_float(args.get("flow_src_guidance_scale", None))
        out["flow_tar_guidance"] = _to_float(args.get("flow_tar_guidance_scale", None))
        out["flow_n_max"] = _to_float(args.get("flow_n_max", None))
        out["flow_seed"] = _to_float(args.get("flow_seed", None))

    model_key = entry.efficiency_model_key
    wrapper_meta = maybe_read_json(Path(entry.model_dir) / "multimodel_wrapper_meta.json")
    if not model_key and wrapper_meta:
        model_key = str(wrapper_meta.get("model_key", "")).strip()

    # Priority 1: explicit tsv path in benchmark.
    if entry.efficiency_tsv:
        tsv = Path(entry.efficiency_tsv)
        if tsv.exists():
            row = _find_row_from_tsv(tsv, model_key)
            if row:
                out["runtime_sec"] = _to_float(row.get("runtime_sec", out["runtime_sec"]))
                out["peak_mem_mib"] = _to_float(row.get("peak_mem_mib", out["peak_mem_mib"]))
                out["efficiency_source"] = str(tsv)

    # Priority 2: sibling logs/run_summary.tsv.
    if not out["efficiency_source"]:
        auto_tsv = Path(entry.model_dir).parent / "logs" / "run_summary.tsv"
        if auto_tsv.exists():
            row = _find_row_from_tsv(auto_tsv, model_key)
            if row:
                out["runtime_sec"] = _to_float(row.get("runtime_sec", out["runtime_sec"]))
                out["peak_mem_mib"] = _to_float(row.get("peak_mem_mib", out["peak_mem_mib"]))
                out["efficiency_source"] = str(auto_tsv)

    # Priority 3: model-local run_meta json (optional future extension).
    run_meta = maybe_read_json(Path(entry.model_dir) / "run_meta.json")
    if run_meta:
        if out["runtime_sec"] is None:
            out["runtime_sec"] = _to_float(run_meta.get("runtime_sec", None))
        if out["peak_mem_mib"] is None:
            out["peak_mem_mib"] = _to_float(run_meta.get("peak_mem_mib", None))
        if not out["efficiency_source"]:
            out["efficiency_source"] = str(Path(entry.model_dir) / "run_meta.json")

    # Basic quality guard: treat invalid zeros as missing (historical broken logs).
    if out["peak_mem_mib"] is not None and out["peak_mem_mib"] <= 0:
        out["peak_mem_mib"] = None
    if out["runtime_sec"] is not None and out["runtime_sec"] <= 0:
        out["runtime_sec"] = None

    return out
