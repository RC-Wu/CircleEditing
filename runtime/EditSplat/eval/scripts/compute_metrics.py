#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.core.camera_loader import load_cameras
from eval.core.clip_metrics import ClipEncoder, compute_clip_metrics, compute_direction_consistency
from eval.core.efficiency import collect_efficiency
from eval.core.io_utils import (
    append_jsonl,
    load_benchmark,
    load_json,
    maybe_read_json,
    save_json,
    sanitize_slug,
    sorted_pngs,
)
from eval.core.proxy_metrics import compute_proxy_metrics
from eval.core.reproj_metrics import compute_reprojection_consistency
from eval.core.types import EvalConfig


def _load_cfg(path: str) -> Dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return load_json(p)


def _pick_view_names(meta: Dict, src_dir: Path, edit_dir: Path) -> List[str]:
    if isinstance(meta.get("view_names", None), list) and meta["view_names"]:
        return [str(x) for x in meta["view_names"]]
    src = {p.name for p in sorted_pngs(src_dir)}
    edt = {p.name for p in sorted_pngs(edit_dir)}
    return sorted(src & edt)


def _want(entry, scene: str, method: str) -> bool:
    if scene and entry.scene_id != scene:
        return False
    if method and entry.method != method:
        return False
    return True


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Compute no-GT evaluation metrics for EditSplat outputs")
    ap.add_argument("--benchmark", type=str, required=True)
    ap.add_argument("--render_cache_root", type=str, default="eval/cache/renders")
    ap.add_argument("--metrics_root", type=str, default="eval/cache/metrics")
    ap.add_argument("--config", type=str, default="")

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--clip_backend", type=str, default="auto", choices=["auto", "open_clip", "clip", "transformers"])
    ap.add_argument("--clip_model", type=str, default="ViT-B/32")
    ap.add_argument("--open_clip_pretrained", type=str, default="laion2b_s34b_b79k")
    ap.add_argument("--pairs_per_sample", type=int, default=200)

    ap.add_argument("--compute_reproj", type=int, default=1)
    ap.add_argument("--reproj_occlusion_abs", type=float, default=0.02)
    ap.add_argument("--reproj_occlusion_rel", type=float, default=0.05)
    ap.add_argument("--use_lpips", type=int, default=1)

    ap.add_argument("--overwrite", type=int, default=0)
    ap.add_argument("--scene_id", type=str, default="")
    ap.add_argument("--method", type=str, default="")
    ap.add_argument("--max_entries", type=int, default=-1)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    cfg = EvalConfig.from_dict(_load_cfg(args.config))
    # CLI override
    cfg.device = args.device or cfg.device
    cfg.clip_backend = args.clip_backend or cfg.clip_backend
    cfg.clip_model = args.clip_model or cfg.clip_model
    cfg.open_clip_pretrained = args.open_clip_pretrained or cfg.open_clip_pretrained
    cfg.pairs_per_sample = int(args.pairs_per_sample)
    cfg.compute_reproj = bool(args.compute_reproj)
    cfg.reproj_occlusion_abs = float(args.reproj_occlusion_abs)
    cfg.reproj_occlusion_rel = float(args.reproj_occlusion_rel)
    cfg.compute_lpips = bool(args.use_lpips)

    benchmark = load_benchmark(Path(args.benchmark).resolve())
    benchmark = [e for e in benchmark if _want(e, args.scene_id, args.method)]
    if args.max_entries > 0:
        benchmark = benchmark[: args.max_entries]

    render_cache_root = Path(args.render_cache_root).resolve()
    metrics_root = Path(args.metrics_root).resolve()
    metrics_root.mkdir(parents=True, exist_ok=True)

    results_jsonl = metrics_root / "results.jsonl"
    if args.overwrite and results_jsonl.exists():
        results_jsonl.unlink()

    clip_encoder = ClipEncoder(
        clip_backend=cfg.clip_backend,
        clip_model=cfg.clip_model,
        open_clip_pretrained=cfg.open_clip_pretrained,
        device=cfg.device,
    )

    camera_cache: Dict[str, Sequence[object]] = {}

    ok = 0
    failed = 0
    errors: List[dict] = []

    for i, entry in enumerate(benchmark):
        scene_slug = sanitize_slug(entry.scene_id)
        edit_slug = sanitize_slug(entry.edit_id)
        method_slug = sanitize_slug(entry.method)
        cache_dir = render_cache_root / scene_slug / edit_slug / method_slug
        meta_path = cache_dir / "meta.json"

        out_path = metrics_root / scene_slug / edit_slug / f"{method_slug}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not bool(args.overwrite):
            print(f"[{i+1}/{len(benchmark)}] SKIP {entry.uid} (exists)")
            continue

        try:
            meta = maybe_read_json(meta_path)
            src_dir = cache_dir / "src_rgb"
            edit_dir = cache_dir / "edit_rgb"
            src_depth_dir = cache_dir / "src_depth"

            view_names = _pick_view_names(meta, src_dir, edit_dir)
            src_paths = [src_dir / n for n in view_names if (src_dir / n).exists()]
            edit_paths = [edit_dir / n for n in view_names if (edit_dir / n).exists()]

            if not src_paths or not edit_paths:
                raise RuntimeError(f"Missing cached rgb views under {cache_dir}")

            source_caption = entry.source_caption or entry.source_prompt or "source scene"
            target_prompt = entry.target_prompt
            if not target_prompt:
                model_args = maybe_read_json(Path(entry.model_dir) / "args.json")
                target_prompt = str(model_args.get("flow_tar_prompt", "") or model_args.get("target_prompt", ""))

            clip_out = compute_clip_metrics(
                encoder=clip_encoder,
                src_paths=src_paths,
                edit_paths=edit_paths,
                target_prompt=target_prompt,
                source_caption=source_caption,
            )
            cons_out = compute_direction_consistency(
                clip_out.get("d_img_vectors", []),
                pairs_per_sample=cfg.pairs_per_sample,
                seed=int(entry.pair_seed),
            )

            model_args = maybe_read_json(Path(entry.model_dir) / "args.json")

            source_model_dir = Path(entry.source_pretrained_dir) if entry.source_pretrained_dir else None
            if source_model_dir is None or not source_model_dir.exists():
                if entry.source_checkpoint:
                    source_model_dir = Path(entry.source_checkpoint).resolve().parent
                else:
                    source_model_dir = None

            proxy_out = compute_proxy_metrics(
                src_paths=src_paths,
                edit_paths=edit_paths,
                use_lpips=cfg.compute_lpips,
                device=cfg.device,
                source_model_dir=source_model_dir,
                edit_model_dir=Path(entry.model_dir),
                source_iter=int(meta.get("source_iter", entry.source_iter)),
                edit_iter=int(meta.get("edited_iter", entry.edited_iter)),
            )

            reproj_out = {
                "reproj_l1_mean": None,
                "reproj_lpips_mean": None,
                "reproj_visible_ratio_mean": None,
                "reproj_num_pairs": 0,
            }
            if cfg.compute_reproj:
                source_path = entry.source_path or str(model_args.get("source_path", ""))
                src_model_dir_for_cam = str(source_model_dir) if source_model_dir is not None else ""
                source_iter = int(meta.get("source_iter", entry.source_iter))
                cam_key = "::".join([source_path, src_model_dir_for_cam, str(source_iter), entry.split, cfg.device])
                if cam_key not in camera_cache:
                    if not source_path:
                        raise RuntimeError(f"Missing source_path for reprojection: {entry.uid}")
                    if not src_model_dir_for_cam:
                        raise RuntimeError(f"Missing source model dir for reprojection: {entry.uid}")
                    camera_cache[cam_key] = load_cameras(
                        source_path=source_path,
                        model_path=src_model_dir_for_cam,
                        iteration=source_iter,
                        base_args=model_args,
                        split=entry.split,
                        data_device=cfg.device,
                    )

                reproj_out = compute_reprojection_consistency(
                    edit_rgb_dir=edit_dir,
                    src_depth_dir=src_depth_dir,
                    view_names=view_names,
                    camera_list=camera_cache[cam_key],
                    pairs_per_sample=cfg.pairs_per_sample,
                    seed=int(entry.pair_seed),
                    occ_abs=cfg.reproj_occlusion_abs,
                    occ_rel=cfg.reproj_occlusion_rel,
                    use_lpips=cfg.compute_lpips,
                    device=cfg.device,
                )

            eff = collect_efficiency(entry)

            metrics = {
                "clip_sim_mean": clip_out.get("clip_sim_mean"),
                "clip_src_sim_mean": clip_out.get("clip_src_sim_mean"),
                "clip_dir_mean": clip_out.get("clip_dir_mean"),
                "clip_dir_consistency_mean": cons_out.get("clip_dir_consistency_mean"),
                "reproj_l1_mean": reproj_out.get("reproj_l1_mean"),
                "reproj_lpips_mean": reproj_out.get("reproj_lpips_mean"),
                "reproj_visible_ratio_mean": reproj_out.get("reproj_visible_ratio_mean"),
                "reproj_num_pairs": reproj_out.get("reproj_num_pairs"),
                "l1_to_src": proxy_out.get("l1_to_src"),
                "psnr_to_src": proxy_out.get("psnr_to_src"),
                "ssim_to_src": proxy_out.get("ssim_to_src"),
                "lpips_to_src": proxy_out.get("lpips_to_src"),
                "hf_ratio_vs_src": proxy_out.get("hf_ratio_vs_src"),
                "clip_ratio": proxy_out.get("clip_ratio"),
                "mv_rel_dist_mse": proxy_out.get("mv_rel_dist_mse"),
                "vertex_ratio": proxy_out.get("vertex_ratio"),
                "runtime_sec": eff.get("runtime_sec"),
                "peak_mem_mib": eff.get("peak_mem_mib"),
                "flow_steps": eff.get("flow_steps"),
                "flow_src_guidance": eff.get("flow_src_guidance"),
                "flow_tar_guidance": eff.get("flow_tar_guidance"),
                "flow_n_max": eff.get("flow_n_max"),
                "flow_seed": eff.get("flow_seed"),
            }

            row = {
                "scene_id": entry.scene_id,
                "edit_id": entry.edit_id,
                "method": entry.method,
                "split": entry.split,
                "target_prompt": target_prompt,
                "source_caption": source_caption,
                "n_views": len(view_names),
                "cache_dir": str(cache_dir),
                "model_dir": entry.model_dir,
                "metrics": metrics,
                "per_view": {
                    "view_names": view_names,
                    "clip_sim": clip_out.get("clip_sim_per_view"),
                    "clip_src_sim": clip_out.get("clip_src_sim_per_view"),
                    "clip_dir": clip_out.get("clip_dir_per_view"),
                    "proxy": proxy_out.get("per_view", {}),
                    "reproj": {
                        "pair_l1": reproj_out.get("reproj_pair_l1"),
                        "pair_lpips": reproj_out.get("reproj_pair_lpips"),
                        "pair_visible_ratio": reproj_out.get("reproj_pair_visible_ratio"),
                    },
                },
                "artifacts": {
                    "src_rgb_dir": str(src_dir),
                    "edit_rgb_dir": str(edit_dir),
                    "src_depth_dir": str(src_depth_dir),
                    "source_render_dir": meta.get("source_render_dir", ""),
                    "edit_render_dir": meta.get("edit_render_dir", ""),
                    "efficiency_source": eff.get("efficiency_source", ""),
                },
                "clip_info": {
                    "backend": clip_out.get("clip_backend"),
                    "model": clip_out.get("clip_model"),
                    "pretrained": clip_out.get("open_clip_pretrained"),
                },
            }

            save_json(out_path, row)
            append_jsonl(results_jsonl, row)
            ok += 1
            print(f"[{i+1}/{len(benchmark)}] OK  {entry.uid}")
        except Exception as exc:
            failed += 1
            err = {"uid": entry.uid, "error": str(exc)}
            errors.append(err)
            print(f"[{i+1}/{len(benchmark)}] ERR {entry.uid}: {exc}")

    summary = {
        "benchmark": str(Path(args.benchmark).resolve()),
        "render_cache_root": str(render_cache_root),
        "metrics_root": str(metrics_root),
        "num_entries": len(benchmark),
        "ok": ok,
        "failed": failed,
        "errors": errors,
        "results_jsonl": str(results_jsonl),
        "clip_backend": clip_encoder.spec.backend,
        "clip_model": clip_encoder.spec.model_name,
        "open_clip_pretrained": clip_encoder.spec.pretrained,
    }
    save_json(metrics_root / "metrics_summary.json", summary)
    print(f"[DONE] ok={ok}, failed={failed}, summary={metrics_root / 'metrics_summary.json'}")


if __name__ == "__main__":
    main()
