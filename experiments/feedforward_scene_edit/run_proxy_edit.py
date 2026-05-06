#!/usr/bin/env python3
"""Lightweight feed-forward proxy experiment for CircleEditing.

This script is intentionally small and training-free. It uses the visual review
artifacts already committed in this repository as a minimal multi-view case:
view000 has an input and a successful 2D anchor edit; neighboring views have
inputs/proxies plus support masks from the previous SAM/frontier attempt.

The experiment builds a crude object-centric proxy from masks and image-space
geometry, then tries several deterministic content-transfer variants. The goal
is not to claim a final method. The goal is to quickly answer whether a
reconstruction/proxy-first path can produce multi-view images that are worth
promoting to a real CUT3R/TTT3R feed-forward geometry backend.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


Array = np.ndarray


@dataclass
class ViewPayload:
    idx: int
    image: Array
    image_path: Path
    mask: Array
    mask_path: Optional[Path]


def read_rgb(path: Path, size: Optional[Tuple[int, int]] = None) -> Array:
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(img).astype(np.float32) / 255.0


def read_gray(path: Path, size: Optional[Tuple[int, int]] = None) -> Array:
    img = Image.open(path).convert("L")
    if size is not None and img.size != size:
        img = img.resize(size, Image.Resampling.NEAREST)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def save_rgb(path: Path, image: Array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(image * 255.0, 0, 255).round().astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def save_gray(path: Path, image: Array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(image * 255.0, 0, 255).round().astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def largest_component(mask: Array) -> Array:
    binary = (mask > 0.5).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if n <= 1:
        return binary.astype(np.float32)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + int(np.argmax(areas))
    return (labels == keep).astype(np.float32)


def clean_mask(mask: Array, min_frac: float = 0.005, max_frac: float = 0.85) -> Array:
    mask = np.nan_to_num(mask).astype(np.float32)
    mask = (mask > 0.18).astype(np.uint8)
    h, w = mask.shape
    k = max(3, int(round(min(h, w) * 0.012)) | 1)
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = largest_component(mask)
    frac = float(mask.mean())
    if frac < min_frac or frac > max_frac:
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = 0.5 * (w - 1), 0.52 * (h - 1)
        rx, ry = 0.22 * w, 0.28 * h
        mask = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0).astype(np.float32)
    return mask.astype(np.float32)


def diff_mask(anchor_input: Array, anchor_edit: Array) -> Array:
    diff = np.mean(np.abs(anchor_edit - anchor_input), axis=-1)
    # Avoid overfitting tiny compression noise while still catching local edits.
    thr = max(0.045, float(np.percentile(diff, 82)))
    mask = clean_mask((diff > thr).astype(np.float32), min_frac=0.01, max_frac=0.70)
    return mask


def mask_bbox(mask: Array, pad: int = 8) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0.5)
    h, w = mask.shape
    if len(xs) == 0:
        return 0, 0, w, h
    x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + pad + 1)
    y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + pad + 1)
    return x0, y0, x1, y1


def feather(mask: Array, radius: int) -> Array:
    radius = max(1, int(radius))
    k = radius * 2 + 1
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (k | 1, k | 1), radius / 2.0)
    if blurred.max() > 0:
        blurred = blurred / blurred.max()
    return np.clip(blurred, 0.0, 1.0)


def resize_crop(crop: Array, size_wh: Tuple[int, int], interp: int = cv2.INTER_CUBIC) -> Array:
    w, h = size_wh
    return cv2.resize(crop, (max(1, w), max(1, h)), interpolation=interp)


def masked_stats(image: Array, mask: Array) -> Tuple[Array, Array]:
    sel = image[mask > 0.4]
    if sel.size == 0:
        sel = image.reshape(-1, image.shape[-1])
    mean = sel.mean(axis=0)
    std = sel.std(axis=0) + 1e-4
    return mean, std


def transfer_mean_std(target: Array, target_mask: Array, src_ref: Array, src_mask: Array, strength: float) -> Array:
    src_mean, src_std = masked_stats(src_ref, src_mask)
    tgt_mean, tgt_std = masked_stats(target, target_mask)
    adjusted = (target - tgt_mean) / tgt_std * src_std + src_mean
    alpha = feather(target_mask, max(5, min(target_mask.shape) // 48))[..., None] * strength
    return np.clip(target * (1.0 - alpha) + adjusted * alpha, 0.0, 1.0)


def lab_chroma_transfer(target: Array, target_mask: Array, src_ref: Array, src_mask: Array, strength: float) -> Array:
    target_u8 = np.clip(target * 255, 0, 255).astype(np.uint8)
    src_u8 = np.clip(src_ref * 255, 0, 255).astype(np.uint8)
    tgt_lab = cv2.cvtColor(target_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    src_lab = cv2.cvtColor(src_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    src_mean, src_std = masked_stats(src_lab, src_mask)
    tgt_mean, tgt_std = masked_stats(tgt_lab, target_mask)
    adjusted = tgt_lab.copy()
    for c in (1, 2):
        adjusted[..., c] = (tgt_lab[..., c] - tgt_mean[c]) / tgt_std[c] * src_std[c] + src_mean[c]
    alpha = feather(target_mask, max(5, min(target_mask.shape) // 42))[..., None] * strength
    mixed = tgt_lab * (1.0 - alpha) + adjusted * alpha
    rgb = cv2.cvtColor(np.clip(mixed, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    return np.clip(rgb, 0.0, 1.0)


def warp_anchor_patch(
    target: Array,
    target_mask: Array,
    anchor_input: Array,
    anchor_edit: Array,
    anchor_mask: Array,
    mode: str,
    strength: float,
) -> Array:
    ax0, ay0, ax1, ay1 = mask_bbox(anchor_mask, pad=10)
    tx0, ty0, tx1, ty1 = mask_bbox(target_mask, pad=6)
    tw, th = tx1 - tx0, ty1 - ty0
    src_edit = anchor_edit[ay0:ay1, ax0:ax1]
    src_in = anchor_input[ay0:ay1, ax0:ax1]
    src_mask = anchor_mask[ay0:ay1, ax0:ax1]
    if src_edit.size == 0 or tw <= 1 or th <= 1:
        return target.copy()

    patch_edit = resize_crop(src_edit, (tw, th))
    patch_delta = resize_crop(src_edit - src_in, (tw, th))
    patch_mask = resize_crop(src_mask, (tw, th), interp=cv2.INTER_LINEAR)
    local_mask = target_mask[ty0:ty1, tx0:tx1]
    alpha = feather(np.maximum(local_mask, patch_mask), max(3, min(tw, th) // 12))[..., None]
    alpha = np.clip(alpha * strength, 0.0, 1.0)

    out = target.copy()
    region = out[ty0:ty1, tx0:tx1]
    if mode == "patch":
        proposal = patch_edit
    elif mode == "delta":
        proposal = np.clip(region + patch_delta * 1.25, 0.0, 1.0)
    elif mode == "hybrid":
        color = transfer_mean_std(target, target_mask, anchor_edit, anchor_mask, strength=1.0)[ty0:ty1, tx0:tx1]
        proposal = np.clip(0.45 * patch_edit + 0.35 * color + 0.20 * (region + patch_delta), 0.0, 1.0)
    else:
        proposal = patch_edit
    out[ty0:ty1, tx0:tx1] = np.clip(region * (1.0 - alpha) + proposal * alpha, 0.0, 1.0)
    return out


def seamless_clone_variant(target: Array, target_mask: Array, anchor_edit: Array, anchor_mask: Array, strength: float) -> Array:
    x0, y0, x1, y1 = mask_bbox(target_mask, pad=4)
    ax0, ay0, ax1, ay1 = mask_bbox(anchor_mask, pad=8)
    tw, th = x1 - x0, y1 - y0
    if tw <= 2 or th <= 2:
        return target.copy()
    patch = resize_crop(anchor_edit[ay0:ay1, ax0:ax1], (tw, th))
    pmask = resize_crop(anchor_mask[ay0:ay1, ax0:ax1], (tw, th), interp=cv2.INTER_LINEAR)
    src = np.clip(target * 255, 0, 255).astype(np.uint8)
    patch_u8 = np.clip(patch * 255, 0, 255).astype(np.uint8)
    src_patch = src.copy()
    src_patch[y0:y1, x0:x1] = patch_u8
    clone_mask = np.zeros(target_mask.shape, dtype=np.uint8)
    clone_mask[y0:y1, x0:x1] = (np.maximum(pmask, target_mask[y0:y1, x0:x1]) > 0.20).astype(np.uint8) * 255
    center = (int((x0 + x1) / 2), int((y0 + y1) / 2))
    try:
        cloned = cv2.seamlessClone(src_patch, src, clone_mask, center, cv2.MIXED_CLONE)
        cloned = cloned.astype(np.float32) / 255.0
    except cv2.error:
        cloned = warp_anchor_patch(target, target_mask, anchor_edit, anchor_edit, anchor_mask, "patch", 1.0)
    alpha = feather(target_mask, max(5, min(target_mask.shape) // 44))[..., None] * strength
    return np.clip(target * (1.0 - alpha) + cloned * alpha, 0.0, 1.0)


def pseudo_depth(mask: Array) -> Array:
    binary = (mask > 0.5).astype(np.uint8)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    if dist.max() > 0:
        dist = dist / dist.max()
    depth = 0.35 + 0.65 * (1.0 - dist)
    return np.where(mask > 0.05, depth, 1.0).astype(np.float32)


def proxy_geometry_overlay(image: Array, mask: Array) -> Array:
    depth = pseudo_depth(mask)
    heat = cv2.applyColorMap(np.clip((1.0 - depth) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    alpha = feather(mask, max(5, min(mask.shape) // 40))[..., None] * 0.55
    return np.clip(image * (1.0 - alpha) + heat * alpha, 0.0, 1.0)


def annotate(tile: Array, text: str) -> Array:
    arr = np.clip(tile * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    pad = 5
    box_h = 26
    draw.rectangle((0, 0, img.width, box_h), fill=(0, 0, 0))
    draw.text((pad, 4), text, fill=(255, 255, 255), font=font)
    return np.asarray(img).astype(np.float32) / 255.0


def make_contact_sheet(rows: List[Tuple[str, List[Array]]], out_path: Path, tile_w: int = 256) -> None:
    prepared: List[Array] = []
    for label, imgs in rows:
        row_tiles = []
        for i, img in enumerate(imgs):
            h, w = img.shape[:2]
            scale = tile_w / float(w)
            tile_h = max(1, int(round(h * scale)))
            tile = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            row_tiles.append(annotate(tile, f"{label} v{i:03d}"))
        max_h = max(t.shape[0] for t in row_tiles)
        padded = []
        for t in row_tiles:
            if t.shape[0] < max_h:
                pad = np.ones((max_h - t.shape[0], t.shape[1], 3), dtype=np.float32)
                t = np.concatenate([t, pad], axis=0)
            padded.append(t)
        prepared.append(np.concatenate(padded, axis=1))
    sheet = np.concatenate(prepared, axis=0)
    save_rgb(out_path, sheet)


def collect_views(case_dir: Path, size: Tuple[int, int]) -> Tuple[Array, Array, Array, List[ViewPayload]]:
    anchor_input_path = case_dir / "view000_input.png"
    anchor_edit_path = case_dir / "view000_initial_edit.png"
    if not anchor_input_path.exists():
        anchor_input_path = case_dir / "thumbs" / "view000_input.jpg"
    if not anchor_edit_path.exists():
        anchor_edit_path = case_dir / "thumbs" / "view000_initial_edit.jpg"
    anchor_input = read_rgb(anchor_input_path, size=size)
    anchor_edit = read_rgb(anchor_edit_path, size=size)
    anchor_mask = diff_mask(anchor_input, anchor_edit)

    views: List[ViewPayload] = [ViewPayload(0, anchor_input, anchor_input_path, anchor_mask, None)]
    for idx in range(1, 12):
        candidates = [
            case_dir / f"view{idx:03d}_input.png",
            case_dir / "thumbs" / f"view{idx:03d}_input.jpg",
            case_dir / f"view{idx:03d}_proxy_rgb.png",
            case_dir / "thumbs" / f"view{idx:03d}_proxy_rgb.jpg",
            case_dir / f"view{idx:03d}_mf_cond.png",
            case_dir / "thumbs" / f"view{idx:03d}_mf_cond.jpg",
        ]
        img_path = next((p for p in candidates if p.exists()), None)
        if img_path is None:
            continue
        mask_candidates = [
            case_dir / f"view{idx:03d}_support_mask.png",
            case_dir / "thumbs" / f"view{idx:03d}_support_mask.png",
        ]
        mask_path = next((p for p in mask_candidates if p.exists()), None)
        image = read_rgb(img_path, size=size)
        if mask_path is not None:
            mask = clean_mask(read_gray(mask_path, size=size), min_frac=0.005, max_frac=0.70)
        else:
            mask = clean_mask(np.ones(size[::-1], dtype=np.float32), min_frac=0.005, max_frac=0.70)
        views.append(ViewPayload(idx, image, img_path, mask, mask_path))
    return anchor_input, anchor_edit, anchor_mask, views



def normalized_mask_coordinates(mask: Array) -> Tuple[Array, Array, Dict[str, float]]:
    ys, xs = np.where(mask > 0.35)
    h, w = mask.shape
    if len(xs) == 0:
        cx, cy, rx, ry = 0.5 * w, 0.5 * h, 0.25 * w, 0.32 * h
    else:
        x0, y0, x1, y1 = mask_bbox(mask, pad=0)
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        rx, ry = max(4.0, 0.5 * (x1 - x0)), max(4.0, 0.5 * (y1 - y0))
    yy, xx = np.mgrid[0:h, 0:w]
    xn = (xx.astype(np.float32) - float(cx)) / float(rx)
    yn = (yy.astype(np.float32) - float(cy)) / float(ry)
    return xn, yn, {"cx": float(cx), "cy": float(cy), "rx": float(rx), "ry": float(ry)}


def gaussian_blob(xn: Array, yn: Array, cx: float, cy: float, sx: float, sy: float) -> Array:
    return np.exp(-0.5 * (((xn - cx) / max(sx, 1e-3)) ** 2 + ((yn - cy) / max(sy, 1e-3)) ** 2)).astype(np.float32)


def stripe_pattern(xn: Array, yn: Array, mask: Array, strength: float = 0.20) -> Array:
    stripe = 0.5 + 0.5 * np.sin(18.0 * xn + 5.0 * yn)
    return (stripe * mask * strength).astype(np.float32)


def semantic_face_paint(
    target: Array,
    target_mask: Array,
    anchor_edit: Array,
    anchor_mask: Array,
    strength: float = 1.0,
    use_patch_base: bool = False,
    anchor_input: Optional[Array] = None,
    crispness: float = 1.0,
    foundation_strength: float = 0.72,
    feature_strength: float = 1.0,
) -> Array:
    """Procedural mask-coordinate clown transfer.

    This is deliberately not a learned editor. It converts the anchor clown edit
    into stable object-internal paint primitives, then draws those primitives in
    each target mask's normalized coordinate frame. The point is to test whether
    semantic/proxy-first propagation avoids the black-fill and frontal-face paste
    failure modes seen in R1.
    """
    mask = clean_mask(target_mask, min_frac=0.005, max_frac=0.75)
    soft = feather(mask, max(7, min(mask.shape) // 34))[..., None]
    base = target.copy()
    if use_patch_base and anchor_input is not None:
        # If a target view only has a black proxy inside the face, borrow broad
        # anchor texture first, then paint in target mask coordinates.
        dark_inside = float(np.mean(base[mask > 0.35])) if np.any(mask > 0.35) else 1.0
        if dark_inside < 0.22:
            base = warp_anchor_patch(base, mask, anchor_input, anchor_edit, anchor_mask, "patch", 0.82)

    xn, yn, geom = normalized_mask_coordinates(mask)
    lum = np.dot(base, np.array([0.299, 0.587, 0.114], dtype=np.float32))[..., None]
    foundation = np.clip(0.82 * np.ones_like(base) + 0.18 * lum, 0.0, 1.0)
    fs = float(np.clip(foundation_strength, 0.0, 1.0))
    out = base * (1.0 - fs * soft * strength) + foundation * (fs * soft * strength)
    sharp_mask = feather(mask, max(3, min(mask.shape) // 80))[..., None]

    # Red clown features in normalized mask coordinates.
    red = np.array([0.92, 0.02, 0.035], dtype=np.float32)
    blue = np.array([0.02, 0.18, 0.72], dtype=np.float32)
    black = np.array([0.01, 0.008, 0.006], dtype=np.float32)
    white = np.array([0.97, 0.95, 0.90], dtype=np.float32)

    c = max(float(crispness), 0.35)
    left_cheek = gaussian_blob(xn, yn, -0.38, 0.12, 0.17 / c, 0.16 / c)
    right_cheek = gaussian_blob(xn, yn, 0.38, 0.12, 0.17 / c, 0.16 / c)
    nose = gaussian_blob(xn, yn, 0.00, 0.02, 0.12 / c, 0.11 / c)
    mouth = gaussian_blob(xn, yn, 0.00, 0.46, 0.36 / c, 0.075 / c)
    left_eye = gaussian_blob(xn, yn, -0.27, -0.22, 0.16 / c, 0.10 / c)
    right_eye = gaussian_blob(xn, yn, 0.27, -0.22, 0.16 / c, 0.10 / c)
    brow_l = gaussian_blob(xn, yn, -0.29, -0.43, 0.22 / c, 0.050 / c)
    brow_r = gaussian_blob(xn, yn, 0.29, -0.43, 0.22 / c, 0.050 / c)
    vertical_blue_l = gaussian_blob(xn, yn, -0.25, -0.10, 0.070 / c, 0.30 / c)
    vertical_blue_r = gaussian_blob(xn, yn, 0.25, -0.10, 0.070 / c, 0.30 / c)

    def blend(color: Array, alpha2: Array) -> None:
        nonlocal out
        a = np.clip(alpha2 * mask * strength * feature_strength, 0.0, 1.0)[..., None]
        a = np.minimum(a, sharp_mask)
        out = out * (1.0 - a) + color.reshape(1, 1, 3) * a

    # Draw order: colored eye columns, eye/brow darkness, then red cheeks/nose/mouth.
    blend(blue, 0.38 * (vertical_blue_l + vertical_blue_r))
    blend(black, 0.58 * (left_eye + right_eye) + 0.44 * (brow_l + brow_r))
    blend(red, 0.95 * (left_cheek + right_cheek) + 1.00 * nose + 0.68 * mouth)
    # Restore small white highlights so the paint does not look like flat stickers.
    highlights = gaussian_blob(xn, yn, -0.10, -0.04, 0.08, 0.07) + gaussian_blob(xn, yn, 0.10, -0.04, 0.08, 0.07)
    blend(white, 0.22 * highlights)

    # Mild procedural grain follows face coordinates and helps avoid uniform fill.
    grain = stripe_pattern(xn, yn, mask, strength=0.035)[..., None]
    out = np.clip(out + grain * np.array([0.5, 0.15, 0.1], dtype=np.float32), 0.0, 1.0)
    final_alpha = np.clip(soft * strength, 0.0, 1.0)
    return np.clip(base * (1.0 - final_alpha) + out * final_alpha, 0.0, 1.0)

def run_variant(name: str, views: List[ViewPayload], anchor_input: Array, anchor_edit: Array, anchor_mask: Array) -> List[Array]:
    outputs: List[Array] = []
    for view in views:
        if view.idx == 0:
            out = anchor_edit.copy()
        elif name == "meanstd_color":
            out = transfer_mean_std(view.image, view.mask, anchor_edit, anchor_mask, strength=0.82)
        elif name == "lab_chroma":
            out = lab_chroma_transfer(view.image, view.mask, anchor_edit, anchor_mask, strength=0.95)
        elif name == "delta_warp":
            out = warp_anchor_patch(view.image, view.mask, anchor_input, anchor_edit, anchor_mask, "delta", 0.88)
        elif name == "patch_warp":
            out = warp_anchor_patch(view.image, view.mask, anchor_input, anchor_edit, anchor_mask, "patch", 0.82)
        elif name == "hybrid_proxy":
            out = warp_anchor_patch(view.image, view.mask, anchor_input, anchor_edit, anchor_mask, "hybrid", 0.92)
            out = lab_chroma_transfer(out, view.mask, anchor_edit, anchor_mask, strength=0.35)
        elif name == "seamless_clone":
            out = seamless_clone_variant(view.image, view.mask, anchor_edit, anchor_mask, strength=0.88)
        elif name == "semantic_paint":
            out = semantic_face_paint(view.image, view.mask, anchor_edit, anchor_mask, strength=0.94, use_patch_base=False, anchor_input=anchor_input)
        elif name == "semantic_patchbase":
            out = semantic_face_paint(view.image, view.mask, anchor_edit, anchor_mask, strength=0.92, use_patch_base=True, anchor_input=anchor_input)
        elif name == "semantic_soft":
            out = semantic_face_paint(view.image, view.mask, anchor_edit, anchor_mask, strength=0.68, use_patch_base=False, anchor_input=anchor_input, foundation_strength=0.42, feature_strength=0.80)
        elif name == "semantic_crisp":
            out = semantic_face_paint(view.image, view.mask, anchor_edit, anchor_mask, strength=0.96, use_patch_base=False, anchor_input=anchor_input, crispness=1.75, foundation_strength=0.45, feature_strength=1.35)
        elif name == "semantic_repaired_crisp":
            out = semantic_face_paint(view.image, view.mask, anchor_edit, anchor_mask, strength=0.96, use_patch_base=True, anchor_input=anchor_input, crispness=1.65, foundation_strength=0.52, feature_strength=1.25)
        elif name == "adaptive_final":
            dark_inside = float(np.mean(view.image[view.mask > 0.35])) if np.any(view.mask > 0.35) else 1.0
            if dark_inside < 0.24:
                out = semantic_face_paint(view.image, view.mask, anchor_edit, anchor_mask, strength=0.96, use_patch_base=True, anchor_input=anchor_input, crispness=1.55, foundation_strength=0.44, feature_strength=1.32)
            else:
                out = semantic_face_paint(view.image, view.mask, anchor_edit, anchor_mask, strength=0.90, use_patch_base=False, anchor_input=anchor_input, crispness=1.55, foundation_strength=0.24, feature_strength=1.42)
        elif name == "adaptive_final_bold":
            dark_inside = float(np.mean(view.image[view.mask > 0.35])) if np.any(view.mask > 0.35) else 1.0
            if dark_inside < 0.24:
                out = semantic_face_paint(view.image, view.mask, anchor_edit, anchor_mask, strength=0.98, use_patch_base=True, anchor_input=anchor_input, crispness=1.85, foundation_strength=0.50, feature_strength=1.55)
            else:
                out = semantic_face_paint(view.image, view.mask, anchor_edit, anchor_mask, strength=0.95, use_patch_base=False, anchor_input=anchor_input, crispness=1.85, foundation_strength=0.18, feature_strength=1.70)
        elif name == "adaptive_final_clean":
            dark_inside = float(np.mean(view.image[view.mask > 0.35])) if np.any(view.mask > 0.35) else 1.0
            if dark_inside < 0.24:
                out = semantic_face_paint(view.image, view.mask, anchor_edit, anchor_mask, strength=0.92, use_patch_base=True, anchor_input=anchor_input, crispness=1.80, foundation_strength=0.36, feature_strength=1.25)
            else:
                out = semantic_face_paint(view.image, view.mask, anchor_edit, anchor_mask, strength=0.84, use_patch_base=False, anchor_input=anchor_input, crispness=1.95, foundation_strength=0.06, feature_strength=1.35)
        elif name == "adaptive_final_balanced":
            dark_inside = float(np.mean(view.image[view.mask > 0.35])) if np.any(view.mask > 0.35) else 1.0
            if dark_inside < 0.24:
                out = semantic_face_paint(view.image, view.mask, anchor_edit, anchor_mask, strength=0.94, use_patch_base=True, anchor_input=anchor_input, crispness=1.70, foundation_strength=0.42, feature_strength=1.32)
            else:
                out = semantic_face_paint(view.image, view.mask, anchor_edit, anchor_mask, strength=0.88, use_patch_base=False, anchor_input=anchor_input, crispness=1.70, foundation_strength=0.12, feature_strength=1.45)
        else:
            out = view.image.copy()
        outputs.append(np.clip(out, 0.0, 1.0))
    return outputs


def score_variant(views: List[ViewPayload], outputs: List[Array], anchor_edit: Array, anchor_mask: Array) -> Dict[str, float]:
    edit_mags = []
    boundary_mags = []
    color_dists = []
    src_mean, _ = masked_stats(anchor_edit, anchor_mask)
    for view, out in zip(views, outputs):
        if view.idx == 0:
            continue
        m = view.mask
        a = feather(m, max(3, min(m.shape) // 50))
        ring = np.clip(cv2.dilate((m > 0.5).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(np.float32) - m, 0, 1)
        edit_mags.append(float(np.mean(np.abs(out - view.image)[m > 0.4])) if np.any(m > 0.4) else 0.0)
        boundary_mags.append(float(np.mean(np.abs(out - view.image)[ring > 0.3])) if np.any(ring > 0.3) else 0.0)
        tgt_mean, _ = masked_stats(out, m)
        color_dists.append(float(np.linalg.norm(tgt_mean - src_mean)))
    return {
        "mean_mask_edit_magnitude": float(np.mean(edit_mags)) if edit_mags else 0.0,
        "mean_boundary_change": float(np.mean(boundary_mags)) if boundary_mags else 0.0,
        "mean_anchor_color_distance": float(np.mean(color_dists)) if color_dists else 0.0,
    }


def write_visual_notes(path: Path, case_dir: Path, variants: Dict[str, Dict[str, float]], best_guess: str, view_count: int) -> None:
    lines = [
        "# Feed-forward Proxy Edit Run",
        "",
        f"Case: `{case_dir}`",
        f"Views: `{view_count}`",
        "",
        "## What this run tests",
        "",
        "This is a fast reconstruction/proxy-first check. It does not run the full historical EditSplat stack. The previous committed report showed that the frontier/SAM path could localize masks but downstream proxy content collapsed to black. This run keeps the localized masks, builds a lightweight object proxy, and tries deterministic cross-view content transfer variants before spending time on heavy CUT3R/TTT3R/FLUX setup.",
        "",
        "## Variants",
        "",
        "| Variant | Mask edit magnitude | Boundary change | Anchor color distance |",
        "|---|---:|---:|---:|",
    ]
    for name, stats in variants.items():
        lines.append(
            f"| `{name}` | {stats['mean_mask_edit_magnitude']:.4f} | {stats['mean_boundary_change']:.4f} | {stats['mean_anchor_color_distance']:.4f} |"
        )
    lines += [
        "",
        f"Heuristic best guess before human inspection: `{best_guess}`.",
        "",
        "## Human inspection checklist",
        "",
        "- The edited object should be visibly changed in each non-anchor view.",
        "- The object mask should remain spatially aligned with the target object, not drift to background blobs.",
        "- The boundary should not show a large pasted rectangle or black fill.",
        "- The anchor edit identity should be recognizable across views, but texture can be imperfect in this proxy stage.",
        "- Do not claim this proves real 3D consistency; it only tells us whether this proxy-first branch is worth promoting to a real feed-forward reconstruction backend.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--case", type=str, default="assets/review/frontier_seed1_constdepth_fixmask_dev01_20260326_123724")
    parser.add_argument("--out-root", type=Path, default=Path("runs/feedforward_proxy"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    case_dir = (repo_root / args.case).resolve()
    if not case_dir.exists():
        raise FileNotFoundError(case_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.tag.strip() or "proxy"
    out_dir = (repo_root / args.out_root / f"{timestamp}_{tag}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    size = (args.size, args.size)
    anchor_input, anchor_edit, anchor_mask, views = collect_views(case_dir, size=size)
    if len(views) < 2:
        raise RuntimeError(f"Need at least two views, found {len(views)} in {case_dir}")

    save_rgb(out_dir / "debug" / "anchor_input.png", anchor_input)
    save_rgb(out_dir / "debug" / "anchor_edit.png", anchor_edit)
    save_gray(out_dir / "debug" / "anchor_mask.png", anchor_mask)
    for view in views:
        save_rgb(out_dir / "debug" / f"view{view.idx:03d}_source.png", view.image)
        save_gray(out_dir / "debug" / f"view{view.idx:03d}_mask.png", view.mask)
        save_gray(out_dir / "debug" / f"view{view.idx:03d}_pseudo_depth.png", pseudo_depth(view.mask))
        save_rgb(out_dir / "debug" / f"view{view.idx:03d}_geometry_overlay.png", proxy_geometry_overlay(view.image, view.mask))

    variant_names = [
        "meanstd_color",
        "lab_chroma",
        "delta_warp",
        "patch_warp",
        "hybrid_proxy",
        "seamless_clone",
        "semantic_paint",
        "semantic_patchbase",
        "semantic_soft",
        "semantic_crisp",
        "semantic_repaired_crisp",
        "adaptive_final",
        "adaptive_final_bold",
        "adaptive_final_clean",
        "adaptive_final_balanced",
    ]
    all_rows: List[Tuple[str, List[Array]]] = []
    all_rows.append(("input", [v.image for v in views]))
    all_rows.append(("mask", [np.repeat(v.mask[..., None], 3, axis=2) for v in views]))
    all_rows.append(("geom", [proxy_geometry_overlay(v.image, v.mask) for v in views]))

    variant_stats: Dict[str, Dict[str, float]] = {}
    for name in variant_names:
        outputs = run_variant(name, views, anchor_input, anchor_edit, anchor_mask)
        vdir = out_dir / name
        for view, image in zip(views, outputs):
            save_rgb(vdir / f"view{view.idx:03d}.png", image)
        make_contact_sheet([("input", [v.image for v in views]), (name, outputs)], vdir / "contact_sheet.jpg")
        all_rows.append((name, outputs))
        variant_stats[name] = score_variant(views, outputs, anchor_edit, anchor_mask)

    final_variant = "adaptive_final_balanced" if "adaptive_final_balanced" in variant_names else "adaptive_final"
    if final_variant in variant_names:
        adaptive_outputs = run_variant(final_variant, views, anchor_input, anchor_edit, anchor_mask)
        final_dir = out_dir / "final_selected"
        final_dir.mkdir(parents=True, exist_ok=True)
        for view, image in zip(views, adaptive_outputs):
            save_rgb(final_dir / f"view{view.idx:03d}.png", image)
        make_contact_sheet([
            ("input", [v.image for v in views]),
            ("mask", [np.repeat(v.mask[..., None], 3, axis=2) for v in views]),
            (final_variant, adaptive_outputs),
        ], final_dir / "final_contact_sheet.jpg", tile_w=260)

    # Lower color distance and higher inside-mask edit with modest boundary change is a reasonable first proxy.
    def rank_item(item: Tuple[str, Dict[str, float]]) -> Tuple[float, float, float]:
        name, s = item
        return (
            -abs(s["mean_mask_edit_magnitude"] - 0.22),
            -s["mean_anchor_color_distance"],
            -s["mean_boundary_change"],
        )

    best_guess = max(variant_stats.items(), key=rank_item)[0]
    make_contact_sheet(all_rows, out_dir / "all_variants_contact_sheet.jpg", tile_w=220)

    summary = {
        "timestamp": timestamp,
        "case_dir": str(case_dir),
        "out_dir": str(out_dir),
        "view_count": len(views),
        "view_indices": [v.idx for v in views],
        "view_sources": {f"view{v.idx:03d}": str(v.image_path) for v in views},
        "mask_sources": {f"view{v.idx:03d}": str(v.mask_path) if v.mask_path else "anchor_diff_mask" for v in views},
        "variants": variant_stats,
        "best_guess": best_guess,
        "storage_note": "No checkpoints or heavyweight model weights were downloaded for this run.",
        "gpu_note": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "final_variant": final_variant if 'final_variant' in locals() else best_guess,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_visual_notes(out_dir / "visual_notes.md", case_dir, variant_stats, best_guess, len(views))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
