#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_metrics(run_dir: Path) -> Dict[str, Dict[str, float]]:
    p = run_dir / "metrics_analysis.json"
    if not p.exists():
        raise FileNotFoundError(f"missing metrics file: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser("Aggregate metrics across multiple case runs")
    ap.add_argument("--run_dirs", type=str, nargs="+", required=True)
    ap.add_argument("--out_json", type=str, default="")
    args = ap.parse_args()

    cases = []
    all_variants = set()
    for rd in args.run_dirs:
        run_dir = Path(rd)
        metrics = load_metrics(run_dir)
        case_name = run_dir.parent.name + "/" + run_dir.name
        cases.append({"case": case_name, "metrics": metrics})
        all_variants.update(metrics.keys())

    all_variants = sorted(all_variants)
    key_metrics = ["l1_to_src", "l1_to_rand", "edge_diff_to_src", "edge_diff_to_rand"]

    agg = {}
    for v in all_variants:
        agg[v] = {"num_cases": 0}
        for k in key_metrics:
            vals = []
            for c in cases:
                if v in c["metrics"] and k in c["metrics"][v]:
                    vals.append(float(c["metrics"][v][k]))
            if vals:
                agg[v][k] = float(np.mean(vals))
                agg[v][k + "_std"] = float(np.std(vals))
                agg[v]["num_cases"] = len(vals)

    out = {
        "num_cases": len(cases),
        "cases": [c["case"] for c in cases],
        "aggregated": agg,
    }

    print(json.dumps(out, indent=2, ensure_ascii=False))

    print("\n# Markdown Table")
    print("| variant | cases | l1_to_src | l1_to_rand | edge_diff_to_src | edge_diff_to_rand |")
    print("|---|---:|---:|---:|---:|---:|")
    for v in all_variants:
        d = agg[v]
        print(
            f"| {v} | {d.get('num_cases',0)} | "
            f"{d.get('l1_to_src', float('nan')):.6f} | "
            f"{d.get('l1_to_rand', float('nan')):.6f} | "
            f"{d.get('edge_diff_to_src', float('nan')):.6f} | "
            f"{d.get('edge_diff_to_rand', float('nan')):.6f} |"
        )

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
