#!/usr/bin/env python3
"""Build a single HTML report for face redfix comparisons.

The report merges:
1) same-benchmark eval metrics (baseline vs 3d-noise methods),
2) red-cast color statistics,
3) panel image and gallery link,
4) optional SOTA reference table from another benchmark.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Dict, List, Optional


PRIMARY_COLUMNS = [
    ("clip_dir_mean_mean", "clip_dir_mean"),
    ("clip_dir_consistency_mean_mean", "clip_dir_consistency"),
    ("reproj_lpips_mean_mean", "reproj_lpips"),
    ("l1_to_src_mean", "l1_to_src"),
    ("psnr_to_src_mean", "psnr_to_src"),
    ("mv_rel_dist_mse_mean", "mv_rel_dist_mse"),
    ("red_idx_delta_mean", "red_idx_delta"),
]

SOTA_COLUMNS = [
    ("clip_dir_mean_mean", "clip_dir_mean"),
    ("clip_dir_consistency_mean_mean", "clip_dir_consistency"),
    ("reproj_lpips_mean_mean", "reproj_lpips"),
    ("l1_to_src_mean", "l1_to_src"),
    ("psnr_to_src_mean", "psnr_to_src"),
]

SOTA_HINTS = ("flux", "sd35", "qwen", "z-image", "instruct", "gaussian", "dream")


def load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(v: Optional[str]) -> Optional[float]:
    if v is None:
        return None
    t = str(v).strip()
    if not t:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def fmt_num(v: Optional[float], digits: int = 5) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def classify_method(method: str) -> str:
    m = method.lower()
    if "baseline" in m:
        return "baseline(no-3d-noise)"
    if "cov_" in m or "3dnoise" in m or "noise" in m or "hybrid" in m:
        return "3d-noise"
    return "other"


def sort_by_metric(rows: List[Dict[str, str]], key: str, reverse: bool = True) -> List[Dict[str, str]]:
    def _rank(r: Dict[str, str]) -> float:
        v = to_float(r.get(key))
        if v is None:
            return float("-inf") if reverse else float("inf")
        return v

    return sorted(rows, key=_rank, reverse=reverse)


def build_primary_table(rows: List[Dict[str, str]]) -> str:
    head = ["method", "group"] + [label for _, label in PRIMARY_COLUMNS]
    lines = [
        "<table>",
        "<thead><tr>" + "".join(f"<th>{escape(h)}</th>" for h in head) + "</tr></thead>",
        "<tbody>",
    ]

    for r in sort_by_metric(rows, "clip_dir_mean_mean", reverse=True):
        method = r.get("method", "")
        group = classify_method(method)
        cells = [escape(method), escape(group)]
        for col, _ in PRIMARY_COLUMNS:
            cells.append(fmt_num(to_float(r.get(col))))
        lines.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")

    lines += ["</tbody>", "</table>"]
    return "\n".join(lines)


def build_sota_table(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return "<p class='warn'>No SOTA reference csv found. Fill --sota_by_method_csv after available.</p>"

    head = ["method"] + [label for _, label in SOTA_COLUMNS]
    lines = [
        "<table>",
        "<thead><tr>" + "".join(f"<th>{escape(h)}</th>" for h in head) + "</tr></thead>",
        "<tbody>",
    ]
    for r in sort_by_metric(rows, "clip_dir_mean_mean", reverse=True):
        method = r.get("method", "")
        cells = [escape(method)]
        for col, _ in SOTA_COLUMNS:
            cells.append(fmt_num(to_float(r.get(col))))
        lines.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    lines += ["</tbody>", "</table>"]
    return "\n".join(lines)


def merge_redcast(primary_rows: List[Dict[str, str]], redcast_rows: List[Dict[str, str]]) -> None:
    idx = {r.get("method", ""): r for r in redcast_rows}
    for r in primary_rows:
        red = idx.get(r.get("method", ""))
        if red is None:
            continue
        r["red_idx_delta_mean"] = red.get("red_idx_delta_mean", "")
        r["red_idx_delta_std"] = red.get("red_idx_delta_std", "")


def coverage_summary(primary_rows: List[Dict[str, str]], sota_rows: List[Dict[str, str]]) -> str:
    methods = [r.get("method", "").lower() for r in primary_rows]
    has_baseline = any("baseline" in m for m in methods)
    has_noise = any(("cov_" in m or "noise" in m or "hybrid" in m) for m in methods)
    has_sota_ref = len(sota_rows) > 0
    hint_in_primary = any(any(h in m for h in SOTA_HINTS) for m in methods)

    lines = [
        "<ul>",
        f"<li>baseline(no-3d-noise) included: <b>{has_baseline}</b></li>",
        f"<li>3d-noise variants included: <b>{has_noise}</b></li>",
        f"<li>SOTA reference table available: <b>{has_sota_ref}</b></li>",
        f"<li>same-benchmark SOTA present: <b>{hint_in_primary}</b> (recommended target: true)</li>",
        "</ul>",
    ]
    if not has_baseline:
        lines.append("<p class='warn'>Missing no-3d-noise baseline. Re-run baseline before final conclusions.</p>")
    if not has_noise:
        lines.append("<p class='warn'>Missing 3d-noise arm. Cannot claim improvement from 3d-noise.</p>")
    if not hint_in_primary:
        lines.append(
            "<p class='warn'>No explicit SOTA method in this benchmark. Keep conclusions as task-local and add same-scene SOTA run when resources free.</p>"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--redcast_csv", required=True)
    ap.add_argument("--panel_img", required=True)
    ap.add_argument("--gallery_html", required=True)
    ap.add_argument("--sota_by_method_csv", default="")
    ap.add_argument("--out_html", required=True)
    args = ap.parse_args()

    summary_csv = Path(args.summary_csv)
    redcast_csv = Path(args.redcast_csv)
    panel_img = Path(args.panel_img)
    gallery_html = Path(args.gallery_html)
    sota_csv = Path(args.sota_by_method_csv) if args.sota_by_method_csv else None
    out_html = Path(args.out_html)

    primary_rows = load_csv(summary_csv)
    if not primary_rows:
        raise FileNotFoundError(f"Empty or missing summary csv: {summary_csv}")
    redcast_rows = load_csv(redcast_csv)
    merge_redcast(primary_rows, redcast_rows)
    sota_rows = load_csv(sota_csv) if sota_csv else []

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    panel_tag = (
        f"<img src=\"{panel_img.as_posix()}\" alt=\"panel\" />"
        if panel_img.exists()
        else "<p class='warn'>Panel image not found yet.</p>"
    )
    gallery_tag = (
        f"<a href=\"{gallery_html.as_posix()}\">{escape(gallery_html.as_posix())}</a>"
        if gallery_html.exists()
        else "<span class='warn'>gallery.html not found yet.</span>"
    )

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Face Redfix R2 Comparison Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 20px; color: #111; background: #f7f8fa; }}
h1, h2 {{ margin: 8px 0; }}
.card {{ background: #fff; border: 1px solid #d9dde3; border-radius: 10px; padding: 14px; margin-bottom: 14px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border: 1px solid #dde2e8; padding: 6px 8px; text-align: left; }}
th {{ background: #eef2f6; }}
.muted {{ color: #5a6472; font-size: 13px; }}
.warn {{ color: #a15b00; }}
img {{ width: min(100%, 1800px); border: 1px solid #dfe5ec; border-radius: 8px; }}
</style>
</head>
<body>
  <h1>Face Redfix R2 Comparison Report</h1>
  <p class="muted">generated_at: {escape(ts)}</p>

  <div class="card">
    <h2>1) Same-Benchmark Comparison (baseline vs 3d-noise)</h2>
    {build_primary_table(primary_rows)}
  </div>

  <div class="card">
    <h2>2) Visualization</h2>
    <p class="muted">panel image</p>
    {panel_tag}
    <p class="muted">eval gallery: {gallery_tag}</p>
  </div>

  <div class="card">
    <h2>3) SOTA Reference (nearest available benchmark)</h2>
    <p class="muted">This section is contextual; not always same-scene/same-prompt.</p>
    {build_sota_table(sota_rows)}
  </div>

  <div class="card">
    <h2>4) Coverage Checks</h2>
    {coverage_summary(primary_rows, sota_rows)}
  </div>

  <div class="card">
    <h2>5) Source Files</h2>
    <ul>
      <li>summary_csv: {escape(summary_csv.as_posix())}</li>
      <li>redcast_csv: {escape(redcast_csv.as_posix())}</li>
      <li>panel_img: {escape(panel_img.as_posix())}</li>
      <li>gallery_html: {escape(gallery_html.as_posix())}</li>
      <li>sota_by_method_csv: {escape(sota_csv.as_posix()) if sota_csv else "n/a"}</li>
    </ul>
  </div>
</body>
</html>
"""

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    print(f"saved: {out_html}")


if __name__ == "__main__":
    main()
