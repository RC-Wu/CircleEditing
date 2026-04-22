from __future__ import annotations

import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .io_utils import (
    ensure_dir,
    find_latest_iteration,
    find_latest_render_dir,
    get_git_commit,
    intersect_filenames,
    maybe_read_json,
    now_iso,
    parse_iter_from_checkpoint,
    parse_iter_from_name,
    sanitize_slug,
    save_json,
    sorted_pngs,
)
from .types import BenchmarkEntry


class _PipeArgs:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False


def _copy_or_link(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    ensure_dir(dst.parent)

    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    else:
        shutil.copy2(src, dst)


def _resolve_source_render_dir(entry: BenchmarkEntry, model_args: Dict[str, object]) -> Path:
    if entry.source_render_dir and Path(entry.source_render_dir).exists():
        return Path(entry.source_render_dir)

    src_pretrained = Path(entry.source_pretrained_dir) if entry.source_pretrained_dir else None
    if src_pretrained is None or not src_pretrained.exists():
        # fallback from source checkpoint
        if entry.source_checkpoint:
            src_pretrained = Path(entry.source_checkpoint).resolve().parent

    if src_pretrained is None or not src_pretrained.exists():
        raise FileNotFoundError(f"Cannot resolve source_pretrained_dir for entry: {entry.uid}")

    split = entry.split
    if entry.source_iter >= 0:
        p = src_pretrained / split / f"ours_{entry.source_iter}" / "renders"
        if p.exists():
            return p

    latest = find_latest_render_dir(src_pretrained, split=split)
    if latest is not None:
        return latest

    raise FileNotFoundError(f"No source render dir under {src_pretrained}/{split}")


def _resolve_edit_render_dir(entry: BenchmarkEntry) -> Path:
    if entry.edit_render_dir and Path(entry.edit_render_dir).exists():
        return Path(entry.edit_render_dir)

    p = find_latest_render_dir(Path(entry.model_dir), split=entry.split)
    if p is None:
        raise FileNotFoundError(f"No edited render dir found for model_dir={entry.model_dir}, split={entry.split}")
    return p


def _extract_iter_from_render_dir(render_dir: Path) -> int:
    # .../<split>/ours_XXXX/renders
    parent = render_dir.parent
    return parse_iter_from_name(parent.name)


def _build_scene_args(
    *,
    source_path: str,
    model_path: str,
    base_args: Dict[str, object],
    data_device: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        sh_degree=int(base_args.get("sh_degree", 3)),
        source_path=str(source_path),
        model_path=str(model_path),
        source_checkpoint="",
        images=str(base_args.get("images", "images")),
        resolution=int(base_args.get("resolution", -1)),
        white_background=bool(base_args.get("white_background", False)),
        data_device=str(data_device),
        eval=bool(base_args.get("eval", True)),
        render_items=list(base_args.get("render_items", ["RGB", "Depth"])),
        view_shuffling=False,
    )


def _load_scene_and_cameras(
    *,
    source_path: str,
    model_path: str,
    iteration: int,
    base_args: Dict[str, object],
    split: str,
    data_device: str,
):
    # Lazy import to avoid forcing render deps for non-depth workflows.
    from scene import Scene
    from scene.gaussian_model import GaussianModel

    args = _build_scene_args(
        source_path=source_path,
        model_path=model_path,
        base_args=base_args,
        data_device=data_device,
    )
    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(args, gaussians, load_iteration=iteration, shuffle=False)
    if split == "test":
        cameras = scene.getTestCameras()
    else:
        cameras = scene.getTrainCameras()
    return scene, gaussians, cameras


def _render_depth_to_cache(
    *,
    source_path: str,
    model_path: str,
    iteration: int,
    base_args: Dict[str, object],
    split: str,
    view_names: Sequence[str],
    out_dir: Path,
    overwrite: bool,
    device: str,
) -> Dict[str, object]:
    from gaussian_renderer import render

    scene, gaussians, cameras = _load_scene_and_cameras(
        source_path=source_path,
        model_path=model_path,
        iteration=iteration,
        base_args=base_args,
        split=split,
        data_device=device,
    )

    render_device = gaussians.get_xyz.device
    if not isinstance(render_device, torch.device):
        render_device = torch.device(render_device)
    if render_device.type == "cpu" and torch.cuda.is_available():
        render_device = torch.device("cuda")

    bg_color = [1, 1, 1] if bool(base_args.get("white_background", False)) else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=render_device)

    ensure_dir(out_dir)

    ok = 0
    failed = 0

    with torch.no_grad():
        for name in view_names:
            idx = int(Path(name).stem)
            out_path = out_dir / f"{idx:05d}.npy"
            if out_path.exists() and not overwrite:
                ok += 1
                continue
            try:
                cam = cameras[idx]
                pkg = render(cam, gaussians, _PipeArgs(), background)
                depth = pkg["depth_3dgs"].detach().squeeze().float().cpu().numpy()
                np.save(out_path, depth.astype(np.float32))
                ok += 1
            except Exception:
                failed += 1

    return {
        "scene_loaded_iter": int(getattr(scene, "loaded_iter", iteration)),
        "depth_saved": int(ok),
        "depth_failed": int(failed),
    }


def build_cache_for_entry(
    *,
    entry: BenchmarkEntry,
    cache_root: Path,
    render_depth_source: bool,
    render_depth_edit: bool,
    overwrite: bool,
    link_mode: str,
    device: str,
    repo_root: Path,
) -> Dict[str, object]:
    model_args = maybe_read_json(Path(entry.model_dir) / "args.json")

    source_render_dir = _resolve_source_render_dir(entry, model_args)
    edit_render_dir = _resolve_edit_render_dir(entry)

    src_pngs = sorted_pngs(source_render_dir)
    edit_pngs = sorted_pngs(edit_render_dir)
    common = intersect_filenames(src_pngs, edit_pngs)
    if not common:
        raise RuntimeError(f"No common render files for entry={entry.uid}")

    if entry.max_views > 0:
        common = common[: entry.max_views]

    scene_slug = sanitize_slug(entry.scene_id)
    edit_slug = sanitize_slug(entry.edit_id)
    method_slug = sanitize_slug(entry.method)
    base = ensure_dir(cache_root / scene_slug / edit_slug / method_slug)

    src_rgb_dir = ensure_dir(base / "src_rgb")
    edit_rgb_dir = ensure_dir(base / "edit_rgb")
    src_depth_dir = ensure_dir(base / "src_depth")
    edit_depth_dir = ensure_dir(base / "edit_depth")

    src_map = {p.name: p for p in src_pngs}
    edit_map = {p.name: p for p in edit_pngs}

    for name in common:
        _copy_or_link(src_map[name], src_rgb_dir / name, mode=link_mode, overwrite=overwrite)
        _copy_or_link(edit_map[name], edit_rgb_dir / name, mode=link_mode, overwrite=overwrite)

    source_iter = int(entry.source_iter)
    if source_iter < 0:
        source_iter = _extract_iter_from_render_dir(source_render_dir)
    if source_iter < 0:
        source_iter = parse_iter_from_checkpoint(entry.source_checkpoint)
    if source_iter < 0:
        source_iter = find_latest_iteration(Path(entry.source_pretrained_dir))

    edited_iter = int(entry.edited_iter)
    if edited_iter < 0:
        edited_iter = _extract_iter_from_render_dir(edit_render_dir)
    if edited_iter < 0:
        edited_iter = find_latest_iteration(Path(entry.model_dir))

    depth_source_meta: Dict[str, object] = {}
    depth_edit_meta: Dict[str, object] = {}

    source_path = entry.source_path or str(model_args.get("source_path", ""))
    if not source_path:
        raise RuntimeError(f"source_path is missing for entry={entry.uid}")

    if render_depth_source:
        src_model_dir = entry.source_pretrained_dir
        if not src_model_dir and entry.source_checkpoint:
            src_model_dir = str(Path(entry.source_checkpoint).resolve().parent)
        if not src_model_dir:
            raise RuntimeError(f"source_pretrained_dir is missing for entry={entry.uid}")
        depth_source_meta = _render_depth_to_cache(
            source_path=source_path,
            model_path=src_model_dir,
            iteration=source_iter,
            base_args=model_args,
            split=entry.split,
            view_names=common,
            out_dir=src_depth_dir,
            overwrite=overwrite,
            device=device,
        )

    if render_depth_edit:
        depth_edit_meta = _render_depth_to_cache(
            source_path=source_path,
            model_path=entry.model_dir,
            iteration=edited_iter,
            base_args=model_args,
            split=entry.split,
            view_names=common,
            out_dir=edit_depth_dir,
            overwrite=overwrite,
            device=device,
        )

    meta = {
        "entry": entry.to_dict(),
        "cache_uid": entry.uid,
        "cache_dir": str(base),
        "scene_id": entry.scene_id,
        "edit_id": entry.edit_id,
        "method": entry.method,
        "split": entry.split,
        "n_views": len(common),
        "view_names": common,
        "source_render_dir": str(source_render_dir),
        "edit_render_dir": str(edit_render_dir),
        "source_iter": int(source_iter),
        "edited_iter": int(edited_iter),
        "depth_source_meta": depth_source_meta,
        "depth_edit_meta": depth_edit_meta,
        "link_mode": link_mode,
        "render_depth_source": bool(render_depth_source),
        "render_depth_edit": bool(render_depth_edit),
        "overwrite": bool(overwrite),
        "created_at": now_iso(),
        "git_commit": get_git_commit(repo_root),
    }
    save_json(base / "meta.json", meta)
    return meta
