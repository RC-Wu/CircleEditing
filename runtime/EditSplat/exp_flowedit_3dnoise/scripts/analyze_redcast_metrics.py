#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb01(path: str) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return arr


def center_crop(arr: np.ndarray, ratio: float) -> np.ndarray:
    if ratio >= 0.999:
        return arr
    h, w = arr.shape[:2]
    ch = max(1, int(round(h * ratio)))
    cw = max(1, int(round(w * ratio)))
    y0 = max(0, (h - ch) // 2)
    x0 = max(0, (w - cw) // 2)
    return arr[y0 : y0 + ch, x0 : x0 + cw, :]


def metric_red(arr: np.ndarray) -> float:
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    return float(np.mean(r - 0.5 * (g + b)))


def metric_rg(arr: np.ndarray) -> float:
    r, g = arr[..., 0], arr[..., 1]
    return float(np.mean(r - g))


def metric_rb(arr: np.ndarray) -> float:
    r, b = arr[..., 0], arr[..., 2]
    return float(np.mean(r - b))


def find_latest_split_dir(model_dir: str, split: str) -> Path:
    root = Path(model_dir) / split
    cands = sorted(root.glob("ours_*"))
    if not cands:
        raise FileNotFoundError(f"No {split}/ours_* in {model_dir}")
    return cands[-1]


def analyze_model(model_dir: str, split: str, crop_ratio: float):
    split_dir = find_latest_split_dir(model_dir, split)
    render_dir = split_dir / "renders"
    gt_dir = split_dir / "gt"
    if not render_dir.exists() or not gt_dir.exists():
        raise FileNotFoundError(f"Missing renders/gt under {split_dir}")

    render_files = sorted(glob.glob(str(render_dir / "*.png")))
    rows = []
    for rp in render_files:
        name = os.path.basename(rp)
        gp = str(gt_dir / name)
        if not os.path.exists(gp):
            continue

        r_img = center_crop(load_rgb01(rp), crop_ratio)
        g_img = center_crop(load_rgb01(gp), crop_ratio)

        rec = {
            "view": name,
            "red_idx_render": metric_red(r_img),
            "red_idx_gt": metric_red(g_img),
            "red_idx_delta": metric_red(r_img) - metric_red(g_img),
            "rg_delta": metric_rg(r_img) - metric_rg(g_img),
            "rb_delta": metric_rb(r_img) - metric_rb(g_img),
        }
        rows.append(rec)

    if not rows:
        raise RuntimeError(f"No matched render/gt images in {split_dir}")

    def mean_std(key):
        vals = np.array([r[key] for r in rows], dtype=np.float32)
        return float(vals.mean()), float(vals.std())

    out = {
        "model_dir": model_dir,
        "split_dir": str(split_dir),
        "num_views": len(rows),
        "crop_ratio": crop_ratio,
    }
    for k in ["red_idx_delta", "rg_delta", "rb_delta", "red_idx_render", "red_idx_gt"]:
        m, s = mean_std(k)
        out[f"{k}_mean"] = m
        out[f"{k}_std"] = s

    return out, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dirs", nargs="+", required=True)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--crop_ratio", type=float, default=0.6)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    summary = []
    all_rows = []
    for md in args.model_dirs:
        stat, rows = analyze_model(md, args.split, args.crop_ratio)
        stat["method"] = os.path.basename(md.rstrip("/"))
        summary.append(stat)
        for r in rows:
            rr = dict(r)
            rr["method"] = stat["method"]
            all_rows.append(rr)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": all_rows}, f, indent=2, ensure_ascii=False)

    fieldnames = [
        "method",
        "num_views",
        "crop_ratio",
        "red_idx_delta_mean",
        "red_idx_delta_std",
        "rg_delta_mean",
        "rg_delta_std",
        "rb_delta_mean",
        "rb_delta_std",
        "red_idx_render_mean",
        "red_idx_gt_mean",
    ]
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in summary:
            w.writerow({k: s.get(k, "") for k in fieldnames})

    print(f"saved: {args.out_json}")
    print(f"saved: {args.out_csv}")


if __name__ == "__main__":
    main()
