#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List


def _probe_python(py: Path, timeout_sec: int, repo_root: Path) -> Dict[str, object]:
    script = r'''
import json, sys
vals = {
  "python": sys.version.split()[0],
  "torch": False,
  "cuda": False,
  "diff_gauss": False,
  "scene": False,
  "gaussian_renderer": False,
  "eval_import": False,
  "open_clip": False,
  "clip": False,
  "transformers": False,
  "lpips": False,
  "ImageReward": False,
  "cv2": False,
  "pandas": False,
  "numpy": False,
}
try:
  import numpy
  vals["numpy"] = True
except Exception:
  pass
try:
  import torch
  vals["torch"] = True
  vals["cuda"] = bool(torch.cuda.is_available())
except Exception:
  pass
for k, mod in [
  ("diff_gauss", "diff_gaussian_rasterization"),
  ("scene", "scene"),
  ("gaussian_renderer", "gaussian_renderer"),
  ("transformers", "transformers"),
  ("open_clip", "open_clip"),
  ("clip", "clip"),
  ("lpips", "lpips"),
  ("ImageReward", "ImageReward"),
  ("cv2", "cv2"),
  ("pandas", "pandas"),
]:
  try:
    __import__(mod)
    vals[k] = True
  except Exception:
    vals[k] = False

try:
  import eval.core.render_cache  # noqa: F401
  import eval.core.reproj_metrics  # noqa: F401
  vals["eval_import"] = True
except Exception:
  vals["eval_import"] = False

clip_candidates = []
for k in ("open_clip", "clip", "transformers"):
  if vals.get(k, False):
    clip_candidates.append(k)
vals["clip_backend_candidates"] = clip_candidates

vals["can_render_depth"] = bool(
  vals.get("torch", False)
  and vals.get("cuda", False)
  and vals.get("diff_gauss", False)
  and vals.get("scene", False)
  and vals.get("gaussian_renderer", False)
)
vals["can_metrics_basic"] = bool(
  vals.get("torch", False)
  and vals.get("cuda", False)
  and len(clip_candidates) > 0
)
vals["can_metrics_full"] = bool(
  vals.get("can_metrics_basic", False)
  and vals.get("lpips", False)
)
vals["can_full_eval"] = bool(
  vals.get("can_render_depth", False)
  and vals.get("can_metrics_full", False)
  and vals.get("eval_import", False)
)
print(json.dumps(vals))
'''
    try:
        proc = subprocess.run(
            [str(py), "-c", script],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        line = (proc.stdout or "").strip().splitlines()[-1]
        data = json.loads(line)
        data["ok"] = True
        return data
    except Exception as exc:
        return {
            "python": "error",
            "torch": False,
            "cuda": False,
            "diff_gauss": False,
            "scene": False,
            "gaussian_renderer": False,
            "eval_import": False,
            "open_clip": False,
            "clip": False,
            "transformers": False,
            "lpips": False,
            "ImageReward": False,
            "cv2": False,
            "pandas": False,
            "numpy": False,
            "clip_backend_candidates": [],
            "can_render_depth": False,
            "can_metrics_basic": False,
            "can_metrics_full": False,
            "can_full_eval": False,
            "ok": False,
            "error": str(exc),
        }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Probe capabilities of envs under /dev-vepfs/rc_wu/rc_wu/envs")
    ap.add_argument("--env_root", type=str, default="/dev-vepfs/rc_wu/rc_wu/envs")
    ap.add_argument("--timeout_sec", type=int, default=120)
    ap.add_argument("--out_json", type=str, default="")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    env_root = Path(args.env_root)
    repo_root = Path(__file__).resolve().parents[2]

    rows: List[Dict[str, object]] = []
    for env in sorted([p for p in env_root.iterdir() if p.is_dir()]):
        py = env / "bin" / "python"
        if not py.exists():
            rows.append({"env": env.name, "ok": False, "python": "missing"})
            continue
        res = _probe_python(py, timeout_sec=args.timeout_sec, repo_root=repo_root)
        res["env"] = env.name
        rows.append(res)

    print(
        "env\tpython\ttorch\tcuda\tdiff_gauss\tscene\tgaussian_renderer\t"
        "eval_import\topen_clip\tclip\ttransformers\tlpips\tImageReward\t"
        "can_render_depth\tcan_metrics_full\tcan_full_eval\tclip_backends"
    )
    for r in rows:
        backends = ",".join([str(x) for x in r.get("clip_backend_candidates", [])])
        print(
            "\t".join(
                [
                    str(r.get("env", "")),
                    str(r.get("python", "")),
                    str(r.get("torch", False)),
                    str(r.get("cuda", False)),
                    str(r.get("diff_gauss", False)),
                    str(r.get("scene", False)),
                    str(r.get("gaussian_renderer", False)),
                    str(r.get("eval_import", False)),
                    str(r.get("open_clip", False)),
                    str(r.get("clip", False)),
                    str(r.get("transformers", False)),
                    str(r.get("lpips", False)),
                    str(r.get("ImageReward", False)),
                    str(r.get("can_render_depth", False)),
                    str(r.get("can_metrics_full", False)),
                    str(r.get("can_full_eval", False)),
                    backends,
                ]
            )
        )

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
