#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def _run(cmd):
    print("[CMD]", " ".join(shlex.quote(str(x)) for x in cmd))
    subprocess.check_call(cmd)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("One-shot EditSplat evaluation runner")
    ap.add_argument("--benchmark", type=str, required=True)
    ap.add_argument("--cache_root", type=str, default="eval/cache/renders")
    ap.add_argument("--metrics_root", type=str, default="eval/cache/metrics")
    ap.add_argument("--summaries_root", type=str, default="eval/cache/summaries")

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--clip_backend", type=str, default="auto")
    ap.add_argument("--clip_model", type=str, default="ViT-B/32")
    ap.add_argument("--open_clip_pretrained", type=str, default="laion2b_s34b_b79k")
    ap.add_argument("--pairs_per_sample", type=int, default=200)

    ap.add_argument("--render_depth_source", type=int, default=1)
    ap.add_argument("--render_depth_edit", type=int, default=0)
    ap.add_argument("--compute_reproj", type=int, default=1)
    ap.add_argument("--use_lpips", type=int, default=1)

    ap.add_argument("--overwrite", type=int, default=0)
    ap.add_argument("--skip_render_cache", action="store_true")
    ap.add_argument("--skip_compute", action="store_true")
    ap.add_argument("--skip_aggregate", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    py = sys.executable
    root = Path(__file__).resolve().parents[2]

    if not args.skip_render_cache:
        _run(
            [
                py,
                str(root / "eval/scripts/render_cache.py"),
                "--benchmark",
                args.benchmark,
                "--cache_root",
                args.cache_root,
                "--render_depth_source",
                str(args.render_depth_source),
                "--render_depth_edit",
                str(args.render_depth_edit),
                "--overwrite",
                str(args.overwrite),
                "--device",
                args.device,
            ]
        )

    if not args.skip_compute:
        _run(
            [
                py,
                str(root / "eval/scripts/compute_metrics.py"),
                "--benchmark",
                args.benchmark,
                "--render_cache_root",
                args.cache_root,
                "--metrics_root",
                args.metrics_root,
                "--device",
                args.device,
                "--clip_backend",
                args.clip_backend,
                "--clip_model",
                args.clip_model,
                "--open_clip_pretrained",
                args.open_clip_pretrained,
                "--pairs_per_sample",
                str(args.pairs_per_sample),
                "--compute_reproj",
                str(args.compute_reproj),
                "--use_lpips",
                str(args.use_lpips),
                "--overwrite",
                str(args.overwrite),
            ]
        )

    if not args.skip_aggregate:
        _run(
            [
                py,
                str(root / "eval/scripts/aggregate.py"),
                "--metrics_root",
                args.metrics_root,
                "--summaries_root",
                args.summaries_root,
            ]
        )


if __name__ == "__main__":
    main()
