#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.core.aggregate import aggregate_metrics


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Aggregate EditSplat evaluation results")
    ap.add_argument("--metrics_root", type=str, default="eval/cache/metrics")
    ap.add_argument("--summaries_root", type=str, default="eval/cache/summaries")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out = aggregate_metrics(
        metrics_root=Path(args.metrics_root).resolve(),
        summaries_root=Path(args.summaries_root).resolve(),
    )
    print("[DONE]", out)


if __name__ == "__main__":
    main()
