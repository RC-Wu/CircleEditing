#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.core.io_utils import (
    find_latest_render_dir,
    parse_iter_from_checkpoint,
    parse_iter_from_name,
    sanitize_slug,
    save_json,
)
from eval.core.types import BenchmarkEntry


def _is_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "args.json").exists():
        return False
    if find_latest_render_dir(path, split="train") is not None:
        return True
    if find_latest_render_dir(path, split="test") is not None:
        return True
    return False


def _discover_model_dirs(search_roots: List[Path]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for root in search_roots:
        if not root.exists():
            continue
        for p in root.rglob("args.json"):
            model_dir = p.parent
            if model_dir in seen:
                continue
            if _is_model_dir(model_dir):
                out.append(model_dir)
                seen.add(model_dir)
    return sorted(out)


def _read_json(path: Path) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj
        return {}
    except Exception:
        return {}


def _infer_method(model_dir: Path, wrapper_meta: Dict, scene_id: str) -> str:
    mk = str(wrapper_meta.get("model_key", "")).strip()
    if mk:
        return mk
    name = model_dir.name
    suffix = f"_{scene_id}"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return name


def _infer_edit_id(args: Dict, default: str = "edit") -> str:
    tar = str(args.get("flow_tar_prompt", "") or args.get("target_prompt", "")).strip()
    if not tar:
        return default
    words = tar.split()
    short = "_".join(words[:6])
    return sanitize_slug(short, default=default)


def _infer_source_iter(source_pretrained_dir: Path, split: str, source_ckpt: str) -> int:
    rd = find_latest_render_dir(source_pretrained_dir, split=split)
    if rd is not None:
        x = parse_iter_from_name(rd.parent.name)
        if x >= 0:
            return x
    x = parse_iter_from_checkpoint(source_ckpt)
    return x if x >= 0 else -1


def build_entries(model_dirs: List[Path], split: str, override_edit_id: str = "") -> List[BenchmarkEntry]:
    entries: List[BenchmarkEntry] = []
    for md in model_dirs:
        args = _read_json(md / "args.json")
        if not args:
            continue
        wrapper_meta = _read_json(md / "multimodel_wrapper_meta.json")

        source_path = str(args.get("source_path", "")).strip()
        if not source_path:
            continue
        scene_id = Path(source_path).name

        source_ckpt = str(args.get("source_checkpoint", "")).strip()
        source_pretrained = str(Path(source_ckpt).resolve().parent) if source_ckpt else ""
        source_pretrained_dir = Path(source_pretrained) if source_pretrained else Path("")
        if source_pretrained_dir and not source_pretrained_dir.exists():
            source_pretrained_dir = Path("")

        method = _infer_method(md, wrapper_meta, scene_id)
        edit_id = sanitize_slug(override_edit_id, default="") if override_edit_id else _infer_edit_id(args, default="edit")

        target_prompt = str(args.get("flow_tar_prompt", "") or args.get("target_prompt", "")).strip()
        source_prompt = str(args.get("flow_src_prompt", "") or args.get("source_prompt", "")).strip()
        source_caption = source_prompt if source_prompt else "source scene"

        source_iter = -1
        if source_pretrained_dir:
            source_iter = _infer_source_iter(source_pretrained_dir, split=split, source_ckpt=source_ckpt)

        eff_tsv = md.parent / "logs" / "run_summary.tsv"
        entry = BenchmarkEntry(
            scene_id=scene_id,
            edit_id=edit_id,
            method=method,
            model_dir=str(md.resolve()),
            source_pretrained_dir=str(source_pretrained_dir.resolve()) if source_pretrained_dir else "",
            source_checkpoint=source_ckpt,
            source_path=source_path,
            source_iter=source_iter,
            split=split,
            target_prompt=target_prompt,
            source_prompt=source_prompt,
            source_caption=source_caption,
            efficiency_tsv=str(eff_tsv.resolve()) if eff_tsv.exists() else "",
            efficiency_model_key=method,
        )
        entries.append(entry)

    entries.sort(key=lambda e: (e.scene_id, e.edit_id, e.method))
    return entries


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Build benchmark JSON from existing EditSplat output model dirs")
    ap.add_argument("--model_dirs", type=str, nargs="*", default=[])
    ap.add_argument("--search_roots", type=str, nargs="*", default=[])
    ap.add_argument("--split", type=str, default="train", choices=["train", "test"])
    ap.add_argument("--edit_id", type=str, default="", help="Optional fixed edit_id for all entries")
    ap.add_argument("--out", type=str, required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    model_dirs: List[Path] = []
    for p in args.model_dirs:
        pp = Path(p)
        if _is_model_dir(pp):
            model_dirs.append(pp)

    if args.search_roots:
        roots = [Path(p) for p in args.search_roots]
        model_dirs.extend(_discover_model_dirs(roots))

    uniq = []
    seen = set()
    for p in model_dirs:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(rp)

    entries = build_entries(uniq, split=args.split, override_edit_id=args.edit_id)

    out = {
        "version": "editsplat_eval_benchmark_v1",
        "entries": [e.to_dict() for e in entries],
    }
    save_json(Path(args.out), out)
    print(f"[DONE] benchmark entries={len(entries)} -> {args.out}")


if __name__ == "__main__":
    main()
