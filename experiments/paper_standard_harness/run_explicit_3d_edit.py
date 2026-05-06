#!/usr/bin/env python3
"""Explicit 3D representation harness for paper-standard edit contracts.

This is a deliberately lightweight renderer based on colored 3D surfels. Unlike
R1-R5, it does not claim success from view-wise PNG compositing: the final views
are rendered from an edited persistent 3D representation saved to disk.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SURFEL_SCHEMA = "x y z nx ny nz r g b radius mask"


@dataclass
class Camera:
    azimuth: float
    elevation: float
    radius: float = 3.0
    fov_deg: float = 42.0


def save_rgb(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8), mode="RGB").save(path)


def look_at(eye: np.ndarray, target: np.ndarray = np.zeros(3), up: np.ndarray = np.array([0, 1, 0], dtype=np.float32)) -> np.ndarray:
    z = eye - target
    z = z / (np.linalg.norm(z) + 1e-8)
    x = np.cross(up, z)
    x = x / (np.linalg.norm(x) + 1e-8)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=0).astype(np.float32)


def camera_pose(cam: Camera) -> Tuple[np.ndarray, np.ndarray]:
    az = np.deg2rad(cam.azimuth)
    el = np.deg2rad(cam.elevation)
    eye = np.array([
        cam.radius * np.sin(az) * np.cos(el),
        cam.radius * np.sin(el),
        cam.radius * np.cos(az) * np.cos(el),
    ], dtype=np.float32)
    rot = look_at(eye)
    return eye, rot


def make_face_surfel_model(n: int = 95000, seed: int = 4) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    # Ellipsoid face shell facing +z, with mild asymmetry and hair/neck/clothes context.
    u = rng.uniform(-1.0, 1.0, n)
    theta = rng.uniform(0, 2 * np.pi, n)
    # bias samples to visible front hemisphere while still preserving sides.
    phi = np.arccos(u)
    x = 0.72 * np.sin(phi) * np.cos(theta)
    y = 0.95 * np.cos(phi)
    z = 0.48 * np.sin(phi) * np.sin(theta) + 0.18
    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    # Keep a head-like oval and front/side shell.
    keep = (pts[:, 1] > -0.95) & (pts[:, 1] < 0.98) & (pts[:, 2] > -0.36)
    pts = pts[keep]
    # Add neck and torso context to test localization preservation.
    neck_n = 6500
    neck = np.stack([
        rng.normal(0, 0.18, neck_n),
        rng.uniform(-1.35, -0.82, neck_n),
        rng.normal(0.10, 0.10, neck_n),
    ], axis=1).astype(np.float32)
    torso_n = 16000
    torso = np.stack([
        rng.uniform(-0.95, 0.95, torso_n),
        rng.uniform(-2.05, -1.20, torso_n),
        rng.normal(0.02, 0.09, torso_n),
    ], axis=1).astype(np.float32)
    pts = np.concatenate([pts, neck, torso], axis=0)

    colors = np.zeros_like(pts)
    face = pts[:, 1] > -0.88
    neck_mask = (pts[:, 1] <= -0.82) & (np.abs(pts[:, 0]) < 0.32)
    torso_mask = pts[:, 1] < -1.15
    skin = np.array([0.78, 0.58, 0.46], dtype=np.float32)
    colors[face] = skin + rng.normal(0, 0.035, (face.sum(), 3))
    colors[neck_mask] = skin * 0.95 + rng.normal(0, 0.03, (neck_mask.sum(), 3))
    colors[torso_mask] = np.array([0.52, 0.55, 0.55], dtype=np.float32) + rng.normal(0, 0.025, (torso_mask.sum(), 3))
    # Hair cap and side hair, unedited context.
    hair = face & ((pts[:, 1] > 0.40) | ((np.abs(pts[:, 0]) > 0.52) & (pts[:, 1] > -0.25)))
    colors[hair] = np.array([0.23, 0.13, 0.06], dtype=np.float32) + rng.normal(0, 0.025, (hair.sum(), 3))
    # Object-space edit mask: face interior only, excluding hair/torso.
    edit_mask = face & (~hair) & (pts[:, 1] > -0.65) & (np.abs(pts[:, 0]) < 0.58) & (pts[:, 2] > -0.18)
    radii = np.full((pts.shape[0],), 3.8, dtype=np.float32)
    return {"points": pts.astype(np.float32), "colors": np.clip(colors, 0, 1).astype(np.float32), "edit_mask": edit_mask.astype(np.bool_), "radii": radii}


def object_space_clown_edit(model: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    pts = model["points"].copy()
    colors = model["colors"].copy()
    edit_mask = model["edit_mask"].copy()
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    face = edit_mask
    # White makeup foundation is applied in object space, not image space.
    foundation = face
    colors[foundation] = 0.45 * colors[foundation] + 0.55 * np.array([0.98, 0.95, 0.88], dtype=np.float32)

    def blob(cx: float, cy: float, sx: float, sy: float) -> np.ndarray:
        return np.exp(-0.5 * (((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))

    red = np.array([0.95, 0.03, 0.03], dtype=np.float32)
    blue = np.array([0.02, 0.10, 0.70], dtype=np.float32)
    black = np.array([0.02, 0.01, 0.01], dtype=np.float32)
    cheeks = (blob(-0.30, -0.10, 0.16, 0.14) + blob(0.30, -0.10, 0.16, 0.14) + blob(0.00, 0.02, 0.11, 0.10))
    eyes = blob(-0.20, 0.22, 0.14, 0.08) + blob(0.20, 0.22, 0.14, 0.08)
    brows = blob(-0.22, 0.38, 0.18, 0.04) + blob(0.22, 0.38, 0.18, 0.04)
    mouth = blob(0.0, -0.42, 0.34, 0.075)
    blue_stripes = blob(-0.20, 0.08, 0.075, 0.33) + blob(0.20, 0.08, 0.075, 0.33)
    for col, alpha in [(blue, 0.42 * blue_stripes), (black, 0.62 * eyes + 0.42 * brows), (red, 1.00 * cheeks + 0.75 * mouth)]:
        a = np.clip(alpha, 0, 1)[:, None] * face[:, None]
        colors = colors * (1 - a) + col.reshape(1, 3) * a
    out = {k: v.copy() for k, v in model.items()}
    out["colors"] = np.clip(colors, 0, 1).astype(np.float32)
    return out


def render(model: Dict[str, np.ndarray], cam: Camera, size: int = 640, bg=(0.72, 0.68, 0.60)) -> Tuple[np.ndarray, np.ndarray]:
    pts = model["points"]
    colors = model["colors"]
    edit_mask = model.get("edit_mask", np.zeros((pts.shape[0],), dtype=bool))
    radii = model.get("radii", np.full((pts.shape[0],), 2.0, dtype=np.float32))
    eye, rot = camera_pose(cam)
    pc = (pts - eye[None, :]) @ rot.T
    # Camera looks down -z in this convention after look_at rows; use negative depth.
    depth = -pc[:, 2]
    valid = depth > 0.2
    f = 0.5 * size / np.tan(np.deg2rad(cam.fov_deg) / 2)
    uv = np.empty((pts.shape[0], 2), dtype=np.float32)
    uv[:, 0] = f * (pc[:, 0] / np.maximum(depth, 1e-6)) + size / 2
    uv[:, 1] = -f * (pc[:, 1] / np.maximum(depth, 1e-6)) + size / 2
    valid &= (uv[:, 0] >= -5) & (uv[:, 0] < size + 5) & (uv[:, 1] >= -5) & (uv[:, 1] < size + 5)
    order = np.argsort(depth[valid])[::-1]  # far to near
    idxs = np.where(valid)[0][order]
    img = np.ones((size, size, 3), dtype=np.float32) * np.array(bg, dtype=np.float32).reshape(1, 1, 3)
    mask_img = np.zeros((size, size), dtype=np.float32)
    for idx in idxs:
        x, y = int(round(uv[idx, 0])), int(round(uv[idx, 1]))
        if x < 0 or y < 0 or x >= size or y >= size:
            continue
        r = max(1, int(round(radii[idx] * f / max(depth[idx], 1e-6) / 260)))
        color = tuple(float(c) for c in colors[idx])
        cv2.circle(img, (x, y), r, color, thickness=-1, lineType=cv2.LINE_AA)
        if bool(edit_mask[idx]):
            cv2.circle(mask_img, (x, y), r, 1.0, thickness=-1, lineType=cv2.LINE_AA)
    return np.clip(img, 0, 1), np.clip(mask_img, 0, 1)


def annotate(img: np.ndarray, text: str) -> np.ndarray:
    pil = Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle((0, 0, pil.width, 28), fill=(0, 0, 0))
    draw.text((5, 4), text, fill=(255, 255, 255), font=font)
    return np.asarray(pil).astype(np.float32) / 255.0


def contact(rows: List[Tuple[str, List[np.ndarray]]], out: Path, tile: int = 210) -> None:
    row_imgs = []
    for label, imgs in rows:
        tiles = []
        for i, im in enumerate(imgs):
            t = cv2.resize(im, (tile, tile), interpolation=cv2.INTER_AREA)
            tiles.append(annotate(t, f"{label} {i:02d}"))
        row_imgs.append(np.concatenate(tiles, axis=1))
    save_rgb(out, np.concatenate(row_imgs, axis=0))


def save_npz(path: Path, model: Dict[str, np.ndarray], meta: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **model, meta=json.dumps(meta, indent=2))


def save_ply(path: Path, model: Dict[str, np.ndarray]) -> None:
    """Save a lightweight colored surfel PLY for external 3D inspection."""
    pts = model["points"]
    colors = np.clip(model["colors"] * 255, 0, 255).astype(np.uint8)
    radii = model.get("radii", np.full((pts.shape[0],), 2.0, dtype=np.float32))
    mask = model.get("edit_mask", np.zeros((pts.shape[0],), dtype=bool)).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        for name in ["x", "y", "z"]:
            f.write(f"property float {name}\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property float radius\nproperty uchar edit_mask\nend_header\n")
        for p, c, r, m in zip(pts, colors, radii, mask):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])} {float(r):.4f} {int(m)}\n")


def make_turntable_video(image_paths: List[Path], out_path: Path, fps: int = 8) -> None:
    frames = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in image_paths]
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fr in frames + frames[::-1]:
        writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    writer.release()


def make_paper_gate_scene(n: int = 180000, seed: int = 17) -> Dict[str, np.ndarray]:
    """A deterministic explicit 3D scene with clean local edit semantics.

    The object is intentionally simple: a ceramic mug with a handle and a front
    logo patch. It gives us clear geometry, a localized editable region, and
    strong held-out view consistency without pretending to be a diffusion result.
    """
    rng = np.random.default_rng(seed)
    body_n = int(n * 0.58)
    theta = rng.uniform(0, 2 * np.pi, body_n)
    y = rng.uniform(-0.72, 0.72, body_n)
    rad = 0.62 + rng.normal(0, 0.006, body_n)
    pts = [np.stack([rad * np.sin(theta), y, rad * np.cos(theta)], axis=1)]

    cap_n = int(n * 0.10)
    for yy in [-0.74, 0.74]:
        rr = 0.62 * np.sqrt(rng.uniform(0, 1, cap_n // 2))
        tt = rng.uniform(0, 2 * np.pi, cap_n // 2)
        pts.append(np.stack([rr * np.sin(tt), np.full_like(rr, yy), rr * np.cos(tt)], axis=1))

    handle_n = int(n * 0.18)
    t = rng.uniform(-1.05, 1.05, handle_n)
    tube = rng.uniform(0, 2 * np.pi, handle_n)
    cx = 0.74 + 0.23 * np.cos(t)
    cy = 0.57 * np.sin(t)
    cz = -0.02 + rng.normal(0, 0.018, handle_n)
    pts.append(np.stack([cx + 0.045 * np.cos(tube), cy + 0.070 * np.sin(tube), cz], axis=1))

    table_n = int(n * 0.14)
    tx = rng.uniform(-1.35, 1.35, table_n)
    tz = rng.uniform(-1.10, 1.10, table_n)
    ty = np.full(table_n, -0.82) + rng.normal(0, 0.004, table_n)
    pts.append(np.stack([tx, ty, tz], axis=1))

    p = np.concatenate(pts, axis=0).astype(np.float32)
    colors = np.zeros_like(p)
    body = p[:, 1] > -0.80
    table = p[:, 1] <= -0.80
    base = np.array([0.86, 0.82, 0.72], dtype=np.float32)
    colors[body] = base + rng.normal(0, 0.018, (body.sum(), 3))
    colors[table] = np.array([0.45, 0.34, 0.23], dtype=np.float32) + rng.normal(0, 0.018, (table.sum(), 3))

    front = (p[:, 2] > 0.42) & body
    logo = front & (np.abs(p[:, 0]) < 0.30) & (p[:, 1] > -0.20) & (p[:, 1] < 0.33)
    # Source has a subtle low-contrast rectangular label so the edit target is visible.
    colors[logo] = 0.82 * colors[logo] + 0.18 * np.array([0.95, 0.93, 0.86], dtype=np.float32)
    radii = np.full((p.shape[0],), 4.7, dtype=np.float32)
    return {"points": p, "colors": np.clip(colors, 0, 1).astype(np.float32), "edit_mask": logo.astype(np.bool_), "radii": radii}


def object_space_strawberry_logo_edit(model: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    pts = model["points"].copy()
    colors = model["colors"].copy()
    edit_mask = model["edit_mask"].copy()
    x, y = pts[:, 0], pts[:, 1]
    # White patch background with crisp red strawberry and green leaf, all in object coordinates.
    colors[edit_mask] = 0.35 * colors[edit_mask] + 0.65 * np.array([0.98, 0.96, 0.89], dtype=np.float32)
    def e(cx, cy, sx, sy):
        return np.exp(-0.5 * (((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))
    red_body = np.maximum(e(-0.055, 0.02, 0.13, 0.17), e(0.055, 0.02, 0.13, 0.17)) * e(0.0, -0.03, 0.23, 0.24)
    leaf = np.maximum(e(-0.045, 0.195, 0.065, 0.045), e(0.045, 0.195, 0.065, 0.045))
    border = ((np.abs(x) > 0.265) | (np.abs(y - 0.065) > 0.235)) & edit_mask
    red = np.array([0.92, 0.02, 0.04], dtype=np.float32)
    green = np.array([0.05, 0.48, 0.12], dtype=np.float32)
    dark = np.array([0.13, 0.05, 0.03], dtype=np.float32)
    for col, alpha in [(red, 0.96 * red_body), (green, 0.88 * leaf)]:
        a = np.clip(alpha, 0, 1)[:, None] * edit_mask[:, None]
        colors = colors * (1 - a) + col.reshape(1, 3) * a
    colors[border] = 0.20 * colors[border] + 0.80 * dark
    # Deterministic yellow seeds on the same 3D patch.
    seed_centers = [(-0.10, 0.05), (0.00, 0.06), (0.10, 0.05), (-0.06, -0.05), (0.06, -0.05), (0.00, -0.14)]
    yellow = np.array([1.0, 0.82, 0.20], dtype=np.float32)
    for cx, cy in seed_centers:
        a = (e(cx, cy, 0.018, 0.026) * 0.95)[:, None] * edit_mask[:, None]
        colors = colors * (1 - a) + yellow.reshape(1, 3) * a
    out = {k: v.copy() for k, v in model.items()}
    out["colors"] = np.clip(colors, 0, 1).astype(np.float32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=Path("runs/paper_standard_harness"))
    ap.add_argument("--tag", default="explicit3d")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--scene", choices=["face", "paper_gate_mug"], default="paper_gate_mug")
    ap.add_argument("--points", type=int, default=180000)
    args = ap.parse_args()
    out_dir = args.out_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.scene == "face":
        source = make_face_surfel_model(max(args.points, 95000))
        edited = object_space_clown_edit(source)
        edit_desc = "object-space clown makeup"
        cams = [Camera(a, 0) for a in np.linspace(-55, 55, 7)]
        heldout = [Camera(a, 8) for a in np.linspace(-70, 70, 9)]
    else:
        source = make_paper_gate_scene(args.points)
        edited = object_space_strawberry_logo_edit(source)
        edit_desc = "localized object-space strawberry logo on mug"
        cams = [Camera(a, 3, radius=3.2, fov_deg=38) for a in np.linspace(-62, 62, 9)]
        heldout = [Camera(a, 9, radius=3.2, fov_deg=38) for a in np.linspace(-82, 82, 13)]
    source_imgs, guidance_imgs, edited_imgs, heldout_imgs, masks = [], [], [], [], []
    for i, cam in enumerate(cams):
        src, m = render(source, cam, args.size)
        edt, _ = render(edited, cam, args.size)
        source_imgs.append(src); guidance_imgs.append(edt); edited_imgs.append(edt); masks.append(np.repeat(m[..., None], 3, axis=2))
        save_rgb(out_dir / "source_orbit" / f"view{i:03d}.png", src)
        save_rgb(out_dir / "guidance_orbit" / f"view{i:03d}.png", edt)
        save_rgb(out_dir / "edited_orbit" / f"view{i:03d}.png", edt)
        save_rgb(out_dir / "masks" / f"view{i:03d}.png", np.repeat(m[..., None], 3, axis=2))
    for i, cam in enumerate(heldout):
        img, _ = render(edited, cam, args.size)
        heldout_imgs.append(img)
        save_rgb(out_dir / "heldout_orbit" / f"view{i:03d}.png", img)
    contact([("source", source_imgs), ("mask", masks), ("guidance", guidance_imgs), ("edited", edited_imgs)], out_dir / "train_views_contact_sheet.jpg")
    contact([("heldout", heldout_imgs)], out_dir / "heldout_contact_sheet.jpg")
    save_npz(out_dir / "source_representation.npz", source, {"type": "colored surfels", "camera_count": len(cams), "scene": args.scene})
    save_npz(out_dir / "edited_representation.npz", edited, {"type": "colored surfels", "edit": edit_desc, "scene": args.scene})
    save_ply(out_dir / "source_representation.ply", source)
    save_ply(out_dir / "edited_representation.ply", edited)
    make_turntable_video(sorted((out_dir / "edited_orbit").glob("*.png")), out_dir / "edited_train_turntable.mp4")
    make_turntable_video(sorted((out_dir / "heldout_orbit").glob("*.png")), out_dir / "edited_heldout_turntable.mp4")
    summary = {
        "out_dir": str(out_dir),
        "representation_gate": "PASS: source_representation.npz and edited_representation.npz saved",
        "novel_view_gate": "PASS: heldout_orbit rendered from edited representation",
        "multi_view_guidance_gate": "PASS for harness: guidance generated from shared object-space edit, not independent 2D paints",
        "ply_gate": "PASS: source_representation.ply and edited_representation.ply saved",
        "visual_target": "Paper-gate harness should be semantically legible and view-consistent, but still not claimed as final diffusion/3DGS quality.",
        "limitation": "This is a lightweight surfel harness, not yet 3DGS/DGE quality. It is a contract test for future DGE/EditSplat integration.",
        "scene": args.scene,
        "edit": edit_desc,
        "source_points": int(source["points"].shape[0]),
        "edited_points": int(edited["points"].shape[0]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
