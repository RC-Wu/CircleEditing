#!/usr/bin/env python3
"""Real-anchor 2.5D bootstrap for CircleEditing visual failure diagnosis.

This script uses the committed frontier visual artifacts rather than procedural
objects. It extracts the anchor edit, builds a persistent textured face patch
plus source background plane, and renders training/held-out yaw views from that
single representation. It is a dirty feed-forward reconstruction bootstrap, not
a final DGE/EditSplat substitute.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_rgb(path: Path, size: int | None = None) -> np.ndarray:
    img = Image.open(path).convert('RGB')
    if size is not None:
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(img).astype(np.float32) / 255.0


def save_rgb(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8), mode='RGB').save(path)


def annotate(img: np.ndarray, text: str) -> np.ndarray:
    pil = Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8), mode='RGB')
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 18)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle((0, 0, pil.width, 29), fill=(0, 0, 0))
    draw.text((6, 4), text, fill=(255, 255, 255), font=font)
    return np.asarray(pil).astype(np.float32) / 255.0


def contact(rows: List[Tuple[str, List[np.ndarray]]], out: Path, tile: int = 210) -> None:
    row_imgs = []
    for label, imgs in rows:
        tiles = [annotate(cv2.resize(im, (tile, tile), interpolation=cv2.INTER_AREA), f'{label} {i:02d}') for i, im in enumerate(imgs)]
        row_imgs.append(np.concatenate(tiles, axis=1))
    save_rgb(out, np.concatenate(row_imgs, axis=0))


def make_video(paths: List[Path], out: Path, fps: int = 8) -> None:
    frames = [cv2.imread(str(p)) for p in paths]
    frames = [f for f in frames if f is not None]
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for fr in frames + frames[::-1]:
        writer.write(fr)
    writer.release()


def estimate_face_mask(anchor_edit: np.ndarray, anchor_src: np.ndarray) -> np.ndarray:
    # Combine anchor source/edit difference with a conservative skin/red-edit prior.
    diff = np.linalg.norm(anchor_edit - anchor_src, axis=2)
    hsv = cv2.cvtColor((anchor_edit * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[..., 1] / 255.0
    val = hsv[..., 2] / 255.0
    y, x = np.mgrid[0:anchor_edit.shape[0], 0:anchor_edit.shape[1]]
    cx, cy = anchor_edit.shape[1] * 0.50, anchor_edit.shape[0] * 0.38
    oval = (((x - cx) / (anchor_edit.shape[1] * 0.22)) ** 2 + ((y - cy) / (anchor_edit.shape[0] * 0.30)) ** 2) < 1.0
    not_black = val > 0.16
    upper_face = y < anchor_edit.shape[0] * 0.67
    mask = ((diff > 0.055) | ((sat > 0.20) & (val > 0.25))) & oval & not_black & upper_face
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8)
    mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 3.0)
    return np.clip(mask, 0, 1)


def bbox_from_mask(mask: np.ndarray, pad: int = 8) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0.2)
    if len(xs) == 0:
        h, w = mask.shape
        return w // 3, h // 4, 2 * w // 3, 3 * h // 4
    x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad + 1)
    y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad + 1)
    return int(x0), int(y0), int(x1), int(y1)


def extract_patch(img: np.ndarray, mask: np.ndarray, out_size: int = 384) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    x0, y0, x1, y1 = bbox_from_mask(mask, 12)
    patch = img[y0:y1, x0:x1]
    m = mask[y0:y1, x0:x1]
    patch = cv2.resize(patch, (out_size, out_size), interpolation=cv2.INTER_CUBIC)
    m = cv2.resize(m, (out_size, out_size), interpolation=cv2.INTER_AREA)
    # Fill transparent/low-mask areas by nearest blur so side yaw has no black holes.
    blur = cv2.GaussianBlur(patch, (0, 0), 8)
    # Do not let old black/corrupt pixels become part of the persistent patch.
    valid = (patch.mean(axis=2) > 0.08).astype(np.float32)
    m = np.clip(m * valid, 0, 1)
    blur = cv2.GaussianBlur(np.where(valid[..., None] > 0, patch, blur), (0, 0), 8)
    patch = patch * m[..., None] + blur * (1 - m[..., None])
    return np.clip(patch, 0, 1), np.clip(m, 0, 1), (x0, y0, x1, y1)


def warp_patch_to_yaw(patch: np.ndarray, mask: np.ndarray, yaw: float, out_size: int = 512) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Render an elliptical 2.5D face surface by horizontally compressing/rolling the anchor texture.
    h, w = out_size, out_size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w * 0.50, h * 0.42
    rx = w * 0.19 * max(0.42, np.cos(np.deg2rad(abs(yaw)) * 0.45))
    ry = h * 0.30
    xnorm = (xx - cx) / rx
    ynorm = (yy - cy) / ry
    face = (xnorm**2 + ynorm**2) <= 1.0
    side_shift = np.sin(np.deg2rad(yaw)) * 0.18
    u = np.clip(0.5 + xnorm * 0.42 + side_shift, 0, 1)
    v = np.clip(0.5 + ynorm * 0.50, 0, 1)
    px = np.clip((u * (patch.shape[1] - 1)).astype(np.int32), 0, patch.shape[1] - 1)
    py = np.clip((v * (patch.shape[0] - 1)).astype(np.int32), 0, patch.shape[0] - 1)
    tex = patch[py, px]
    alpha = mask[py, px] * face.astype(np.float32)
    # Avoid hard holes from anchor mask by adding a low-confidence filled face support.
    support = cv2.GaussianBlur(face.astype(np.float32), (0, 0), 1.5)
    alpha = np.clip(np.maximum(alpha, support * 0.62), 0, 1)
    shade = (0.75 + 0.25 * np.sqrt(np.clip(1 - xnorm**2, 0, 1)))[..., None]
    tex = np.clip(tex * shade, 0, 1)
    return tex, alpha, face.astype(np.float32)


def compose_view(background: np.ndarray, patch: np.ndarray, patch_mask: np.ndarray, yaw: float, target_hint: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
    out_size = background.shape[0]
    tex, alpha, support = warp_patch_to_yaw(patch, patch_mask, yaw, out_size)
    bg = background.copy()
    # For side views, blend with the real target input to preserve hair/clothes/background if provided.
    if target_hint is not None:
        bg = 0.72 * target_hint + 0.28 * bg
    comp = bg * (1 - alpha[..., None]) + tex * alpha[..., None]
    # Feather only the face area; keep outside unchanged.
    return np.clip(comp, 0, 1), np.repeat(alpha[..., None], 3, axis=2)


def write_ply(path: Path, patch: np.ndarray, mask: np.ndarray, n: int = 24000) -> None:
    rng = np.random.default_rng(9)
    us = rng.uniform(0, 1, n)
    vs = rng.uniform(0, 1, n)
    px = np.clip((us * (patch.shape[1] - 1)).astype(np.int32), 0, patch.shape[1] - 1)
    py = np.clip((vs * (patch.shape[0] - 1)).astype(np.int32), 0, patch.shape[0] - 1)
    keep = mask[py, px] > rng.uniform(0.0, 0.8, n)
    us, vs, px, py = us[keep], vs[keep], px[keep], py[keep]
    x = (us - 0.5) * 0.9
    y = -(vs - 0.5) * 1.25
    z = 0.10 * (1 - ((us - 0.5) / 0.5) ** 2 - ((vs - 0.5) / 0.5) ** 2)
    colors = np.clip(patch[py, px] * 255, 0, 255).astype(np.uint8)
    pts = np.stack([x, y, z], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {pts.shape[0]}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for p, c in zip(pts, colors):
            f.write(f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo-root', type=Path, default=Path('.'))
    ap.add_argument('--out-root', type=Path, default=Path('runs/paper_standard_harness'))
    ap.add_argument('--tag', default='r5_real_anchor_bootstrap')
    ap.add_argument('--asset-dir', type=Path, default=Path('assets/review/frontier_seed1_constdepth_dev01_20260326_122905'))
    ap.add_argument('--size', type=int, default=512)
    args = ap.parse_args()
    repo = args.repo_root.resolve()
    asset = args.asset_dir if args.asset_dir.is_absolute() else repo / args.asset_dir
    if not (asset / 'view000_input.png').is_file():
        fallback = repo / 'assets/review/frontier_seed1_constdepth_dev01_20260326_122905'
        if (fallback / 'view000_input.png').is_file():
            asset = fallback
    out = args.out_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    src0_path = asset / 'view000_input.png'
    if (asset / 'thumbs/view000_input.jpg').is_file():
        src0_path = asset / 'thumbs/view000_input.jpg'
    src0 = load_rgb(src0_path, args.size)
    edit_path = asset / 'view000_initial_edit.png'
    if (asset / 'thumbs/view000_initial_edit.jpg').is_file():
        edit_path = asset / 'thumbs/view000_initial_edit.jpg'
    elif not edit_path.is_file():
        edit_path = asset / 'thumbs/view000_initial_edit.jpg'
    edit0_raw = load_rgb(edit_path)
    edit0 = cv2.resize(edit0_raw, (args.size, args.size), interpolation=cv2.INTER_CUBIC)
    v1_path = asset / 'view001_input.png'
    if not v1_path.is_file():
        v1_path = asset / 'thumbs/view001_input.jpg'
    v1 = load_rgb(v1_path, args.size)
    # v2 source is absent in this compact asset; use proxy/input if available.
    v2_path = asset / 'view002_proxy_rgb.png'
    if (asset / 'thumbs/view002_proxy_rgb.jpg').is_file():
        v2_path = asset / 'thumbs/view002_proxy_rgb.jpg'
    v2 = load_rgb(v2_path, args.size) if v2_path.is_file() else src0
    m = estimate_face_mask(edit0, src0)
    patch, patch_mask, bbox = extract_patch(edit0, m)
    save_rgb(out / 'anchor_source.png', src0)
    save_rgb(out / 'anchor_edit.png', edit0)
    save_rgb(out / 'anchor_face_mask.png', np.repeat(m[..., None], 3, axis=2))
    save_rgb(out / 'texture_patch.png', patch)
    save_rgb(out / 'texture_patch_mask.png', np.repeat(patch_mask[..., None], 3, axis=2))
    write_ply(out / 'bootstrap_face_patch.ply', patch, patch_mask)
    np.savez_compressed(out / 'bootstrap_representation.npz', texture_patch=patch, texture_patch_mask=patch_mask, anchor_mask=m, bbox=np.asarray(bbox), note='2.5D textured face patch from real anchor edit')

    train_defs = [(-34, src0, 'anchor-left'), (0, src0, 'anchor'), (28, v1, 'v1-target'), (52, v2, 'v2-target')]
    heldout_yaws = [-58, -44, -30, -16, 0, 16, 30, 44, 58]
    source_imgs = [src0, src0, v1, v2]
    edited_imgs = []
    masks = []
    for i, (yaw, bg, label) in enumerate(train_defs):
        comp, alpha = compose_view(src0, patch, patch_mask, yaw, target_hint=bg)
        edited_imgs.append(comp); masks.append(alpha)
        save_rgb(out / 'edited_orbit' / f'view{i:03d}_{label}.png', comp)
        save_rgb(out / 'masks' / f'view{i:03d}_{label}.png', alpha)
    heldout_imgs = []
    for i, yaw in enumerate(heldout_yaws):
        comp, _ = compose_view(src0, patch, patch_mask, yaw, target_hint=None)
        heldout_imgs.append(comp)
        save_rgb(out / 'heldout_orbit' / f'view{i:03d}_yaw{yaw:+03d}.png', comp)
    contact([('source', source_imgs), ('mask', masks), ('edited', edited_imgs)], out / 'train_views_contact_sheet.jpg')
    contact([('heldout', heldout_imgs)], out / 'heldout_contact_sheet.jpg')
    make_video(sorted((out / 'heldout_orbit').glob('*.png')), out / 'heldout_turntable.mp4')
    summary = {
        'out_dir': str(out),
        'representation_gate': 'PASS: bootstrap_representation.npz and bootstrap_face_patch.ply saved',
        'real_input_gate': 'PARTIAL: uses real committed anchor source/edit assets, but geometry is heuristic 2.5D, not learned/reconstructed 3DGS',
        'novel_view_gate': 'PARTIAL: held-out yaw views rendered from one persistent patch representation',
        'known_failure_risk': 'Likely identity/side geometry distortions; this is meant to expose whether black-face collapse can be avoided before full DGE/3DGS.',
        'bbox': list(map(int, bbox)),
        'train_views': len(train_defs),
        'heldout_views': len(heldout_yaws),
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
