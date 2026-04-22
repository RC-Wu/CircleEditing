from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .io_utils import ensure_dir, load_json, save_json


def _iter_metric_files(metrics_root: Path) -> Iterable[Path]:
    if not metrics_root.exists():
        return []
    skip = {"schema.json", "metrics_summary.json", "aggregate_meta.json"}
    return sorted([p for p in metrics_root.rglob("*.json") if p.is_file() and p.name not in skip])


def _num(v: Any) -> float:
    try:
        if v is None:
            return math.nan
        return float(v)
    except Exception:
        return math.nan


def _mean_std(vals: Sequence[float]) -> Tuple[float, float]:
    clean = [float(v) for v in vals if not math.isnan(float(v))]
    if not clean:
        return math.nan, math.nan
    if len(clean) == 1:
        return clean[0], 0.0
    return mean(clean), pstdev(clean)


def _flatten_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "scene_id": rec.get("scene_id", ""),
        "edit_id": rec.get("edit_id", ""),
        "method": rec.get("method", ""),
        "split": rec.get("split", ""),
        "n_views": rec.get("n_views", 0),
        "cache_dir": rec.get("cache_dir", ""),
        "metrics_file": rec.get("metrics_file", ""),
        "target_prompt": rec.get("target_prompt", ""),
        "source_caption": rec.get("source_caption", ""),
    }
    metrics = rec.get("metrics", {})
    for k, v in metrics.items():
        if isinstance(v, (int, float)) or v is None:
            row[k] = v
    return row


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return

    keys: List[str] = []
    key_set = set()
    for r in rows:
        for k in r.keys():
            if k not in key_set:
                key_set.add(k)
                keys.append(k)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _build_method_stats(rows: Sequence[Dict[str, Any]], metric_keys: Sequence[str]) -> List[Dict[str, Any]]:
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_method.setdefault(str(r.get("method", "unknown")), []).append(r)

    out: List[Dict[str, Any]] = []
    for method in sorted(by_method.keys()):
        rs = by_method[method]
        item: Dict[str, Any] = {
            "method": method,
            "num_samples": len(rs),
        }
        for k in metric_keys:
            vals = [_num(r.get(k, None)) for r in rs]
            m, s = _mean_std(vals)
            item[f"{k}_mean"] = None if math.isnan(m) else m
            item[f"{k}_std"] = None if math.isnan(s) else s
        out.append(item)
    return out


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "n/a"
    try:
        fv = float(v)
    except Exception:
        return "n/a"
    if math.isnan(fv):
        return "n/a"
    return f"{fv:.{nd}f}"


def _build_latex_table(by_method: Sequence[Dict[str, Any]], out_path: Path) -> None:
    cols = [
        ("clip_dir_mean", "CLIPdir$\\uparrow$"),
        ("clip_sim_mean", "CLIPsim$\\uparrow$"),
        ("clip_dir_consistency_mean", "DirCons$\\uparrow$"),
        ("reproj_l1_mean", "Reproj-L1$\\downarrow$"),
        ("reproj_lpips_mean", "Reproj-LPIPS$\\downarrow$"),
        ("runtime_sec", "Time(s)$\\downarrow$"),
        ("peak_mem_mib", "VRAM(MiB)$\\downarrow$"),
    ]

    lines = []
    lines.append("\\begin{tabular}{l" + "c" * len(cols) + "}")
    lines.append("\\toprule")
    lines.append("Method & " + " & ".join([c[1] for c in cols]) + " \\\\")
    lines.append("\\midrule")
    for r in by_method:
        vals = []
        for key, _ in cols:
            m = r.get(f"{key}_mean", None)
            s = r.get(f"{key}_std", None)
            if m is None:
                vals.append("n/a")
            elif s is None:
                vals.append(_fmt(m))
            else:
                vals.append(f"{_fmt(m)}$\\pm${_fmt(s)}")
        lines.append(f"{r.get('method','')} & " + " & ".join(vals) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    ensure_dir(out_path.parent)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_gallery_html(rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    cards = []
    for r in rows:
        cache_dir = Path(str(r.get("cache_dir", "")))
        n_views = int(r.get("n_views", 0) or 0)
        first_name = "00000.png"
        if n_views > 0:
            first_name = "00000.png"
        src = cache_dir / "src_rgb" / first_name
        edt = cache_dir / "edit_rgb" / first_name
        src_rel = src.as_posix()
        edt_rel = edt.as_posix()
        cards.append(
            f"""
            <div class=\"card\">
              <div class=\"meta\"><b>{r.get('method','')}</b> | {r.get('scene_id','')} / {r.get('edit_id','')}</div>
              <div class=\"imgs\">
                <figure><img src=\"{src_rel}\" alt=\"src\"><figcaption>src</figcaption></figure>
                <figure><img src=\"{edt_rel}\" alt=\"edit\"><figcaption>edit</figcaption></figure>
              </div>
            </div>
            """.strip()
        )

    html = f"""
<!doctype html>
<html>
<head>
<meta charset=\"utf-8\" />
<title>EditSplat Eval Gallery</title>
<style>
body {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; background: #f7f7f7; margin: 24px; }}
.card {{ background: #fff; border: 1px solid #ddd; border-radius: 10px; margin-bottom: 12px; padding: 10px; }}
.meta {{ font-size: 13px; margin-bottom: 8px; }}
.imgs {{ display: flex; gap: 10px; flex-wrap: wrap; }}
figure {{ margin: 0; }}
img {{ width: 280px; border: 1px solid #ddd; border-radius: 6px; }}
figcaption {{ font-size: 12px; color: #555; }}
</style>
</head>
<body>
<h2>EditSplat Eval Gallery</h2>
{''.join(cards)}
</body>
</html>
"""
    ensure_dir(out_path.parent)
    out_path.write_text(html, encoding="utf-8")


def aggregate_metrics(metrics_root: Path, summaries_root: Path) -> Dict[str, Any]:
    ensure_dir(summaries_root)

    records: List[Dict[str, Any]] = []
    for p in _iter_metric_files(metrics_root):
        try:
            rec = load_json(p)
            rec["metrics_file"] = str(p)
            records.append(rec)
        except Exception:
            continue

    flat_rows = [_flatten_record(r) for r in records]
    summary_csv = summaries_root / "summary.csv"
    _write_csv(summary_csv, flat_rows)

    metric_keys = [
        "clip_sim_mean",
        "clip_src_sim_mean",
        "clip_dir_mean",
        "clip_dir_consistency_mean",
        "reproj_l1_mean",
        "reproj_lpips_mean",
        "reproj_visible_ratio_mean",
        "l1_to_src",
        "psnr_to_src",
        "ssim_to_src",
        "lpips_to_src",
        "mv_rel_dist_mse",
        "hf_ratio_vs_src",
        "clip_ratio",
        "vertex_ratio",
        "runtime_sec",
        "peak_mem_mib",
    ]
    by_method = _build_method_stats(flat_rows, metric_keys)
    by_method_csv = summaries_root / "by_method.csv"
    _write_csv(by_method_csv, by_method)

    table_tex = summaries_root / "table_main.tex"
    _build_latex_table(by_method, table_tex)

    gallery = summaries_root / "gallery.html"
    _build_gallery_html(flat_rows, gallery)

    out = {
        "num_samples": len(records),
        "summary_csv": str(summary_csv),
        "by_method_csv": str(by_method_csv),
        "table_tex": str(table_tex),
        "gallery_html": str(gallery),
        "metric_keys": metric_keys,
    }
    save_json(summaries_root / "aggregate_meta.json", out)
    return out
