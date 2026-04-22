#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.core.io_utils import append_jsonl, ensure_dir, load_benchmark, save_json
from eval.core.render_cache import build_cache_for_entry


def _want(entry, scene: str, method: str) -> bool:
    if scene and entry.scene_id != scene:
        return False
    if method and entry.method != method:
        return False
    return True


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Build render/depth cache for EditSplat evaluation")
    ap.add_argument("--benchmark", type=str, required=True)
    ap.add_argument("--cache_root", type=str, default="eval/cache/renders")
    ap.add_argument("--render_depth_source", type=int, default=1)
    ap.add_argument("--render_depth_edit", type=int, default=0)
    ap.add_argument("--overwrite", type=int, default=0)
    ap.add_argument("--link_mode", type=str, default="symlink", choices=["symlink", "copy"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--scene_id", type=str, default="")
    ap.add_argument("--method", type=str, default="")
    ap.add_argument("--max_entries", type=int, default=-1)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    benchmark_path = Path(args.benchmark).resolve()
    cache_root = Path(args.cache_root).resolve()
    repo_root = Path(__file__).resolve().parents[2]

    entries = load_benchmark(benchmark_path)
    entries = [e for e in entries if _want(e, args.scene_id, args.method)]
    if args.max_entries > 0:
        entries = entries[: args.max_entries]

    ensure_dir(cache_root)
    index_jsonl = cache_root / "render_index.jsonl"
    if args.overwrite and index_jsonl.exists():
        index_jsonl.unlink()

    ok = 0
    failed = 0
    errors: List[dict] = []

    for i, entry in enumerate(entries):
        try:
            meta = build_cache_for_entry(
                entry=entry,
                cache_root=cache_root,
                render_depth_source=bool(args.render_depth_source),
                render_depth_edit=bool(args.render_depth_edit),
                overwrite=bool(args.overwrite),
                link_mode=args.link_mode,
                device=args.device,
                repo_root=repo_root,
            )
            append_jsonl(index_jsonl, meta)
            ok += 1
            print(f"[{i+1}/{len(entries)}] OK  {entry.uid}")
        except Exception as exc:
            failed += 1
            err = {"uid": entry.uid, "error": str(exc)}
            errors.append(err)
            print(f"[{i+1}/{len(entries)}] ERR {entry.uid}: {exc}")

    summary = {
        "benchmark": str(benchmark_path),
        "cache_root": str(cache_root),
        "num_entries": len(entries),
        "ok": ok,
        "failed": failed,
        "errors": errors,
        "render_index_jsonl": str(index_jsonl),
    }
    save_json(cache_root / "render_cache_summary.json", summary)
    print(f"[DONE] ok={ok}, failed={failed}, summary={cache_root / 'render_cache_summary.json'}")


if __name__ == "__main__":
    main()
