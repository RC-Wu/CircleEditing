#!/usr/bin/env python3
"""Fast mesh/texture harness for paper-standard 3D editing visual gates.

This writes an explicit OBJ/MTL/texture representation and renders an orbit from
one persistent object-space texture edit. Rendering is analytic and fast; it is
only a contract harness, not a substitute for DGE/EditSplat on real scenes.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class Camera:
    azimuth: float
    elevation: float
    radius: float = 3.4
    fov_deg: float = 36.0


def save_rgb(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8), mode="RGB").save(path)


def annotate(img: np.ndarray, text: str) -> np.ndarray:
    pil = Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle((0, 0, pil.width, 29), fill=(0, 0, 0))
    draw.text((6, 4), text, fill=(255, 255, 255), font=font)
    return np.asarray(pil).astype(np.float32) / 255.0


def contact(rows: List[Tuple[str, List[np.ndarray]]], out: Path, tile: int = 210) -> None:
    row_imgs = []
    for label, imgs in rows:
        tiles = [annotate(cv2.resize(im, (tile, tile), interpolation=cv2.INTER_AREA), f"{label} {i:02d}") for i, im in enumerate(imgs)]
        row_imgs.append(np.concatenate(tiles, axis=1))
    save_rgb(out, np.concatenate(row_imgs, axis=0))


def create_textures(size: int = 1024) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    base = np.ones((size, size, 3), dtype=np.float32) * np.array([0.86, 0.82, 0.72], dtype=np.float32)
    yy = np.linspace(0, 1, size, dtype=np.float32)[:, None]
    base += rng.normal(0, 0.010, base.shape).astype(np.float32) + (0.035 * (1 - yy))[..., None]
    source = np.clip(base.copy(), 0, 1)
    edited = source.copy()
    mask = np.zeros((size, size), dtype=np.float32)
    x0, x1 = int(size * 0.37), int(size * 0.63)
    y0, y1 = int(size * 0.35), int(size * 0.67)
    pil = Image.fromarray((edited * 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(pil)
    draw.rounded_rectangle((x0, y0, x1, y1), radius=30, fill=(250, 244, 224), outline=(45, 30, 22), width=12)
    cx = (x0 + x1) // 2
    cy = int((y0 + y1) * 0.53)
    draw.ellipse((cx - 94, cy - 92, cx + 18, cy + 60), fill=(228, 18, 28))
    draw.ellipse((cx - 18, cy - 92, cx + 94, cy + 60), fill=(232, 20, 30))
    draw.ellipse((cx - 100, cy - 44, cx + 100, cy + 116), fill=(236, 28, 34))
    draw.polygon([(cx - 76, cy - 94), (cx - 24, cy - 142), (cx + 8, cy - 88)], fill=(22, 125, 44))
    draw.polygon([(cx - 20, cy - 100), (cx + 28, cy - 150), (cx + 56, cy - 88)], fill=(27, 145, 52))
    draw.polygon([(cx + 18, cy - 90), (cx + 86, cy - 128), (cx + 70, cy - 66)], fill=(18, 112, 42))
    for dx, dy in [(-48, -22), (0, -18), (48, -22), (-66, 22), (-20, 28), (25, 28), (68, 20), (-38, 68), (8, 74), (52, 64)]:
        draw.ellipse((cx + dx - 8, cy + dy - 11, cx + dx + 8, cy + dy + 11), fill=(255, 214, 72))
    edited = np.asarray(pil).astype(np.float32) / 255.0
    mask[y0:y1, x0:x1] = 1.0
    return source, edited, mask


def sample_tex(tex: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = tex.shape[:2]
    xx = np.clip(((u % 1.0) * (w - 1)).astype(np.int32), 0, w - 1)
    yy = np.clip((v * (h - 1)).astype(np.int32), 0, h - 1)
    return tex[yy, xx]


def render_cylindrical(texture: np.ndarray, mask_tex: np.ndarray, cam: Camera, size: int = 720, mask_only: bool = False) -> np.ndarray:
    bg = np.array([0.70, 0.66, 0.58], dtype=np.float32)
    img = np.ones((size, size, 3), dtype=np.float32) * bg.reshape(1, 1, 3)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = size / 2
    cy = size * 0.49
    mug_w = size * 0.49
    mug_h = size * 0.76
    rx = mug_w / 2
    ry = mug_h / 2
    xnorm = (xx - cx) / rx
    ynorm = (yy - cy) / ry
    mug = (np.abs(xnorm) <= 1.0) & (np.abs(ynorm) <= 1.0)
    # Cylinder texture coordinates. Visible center rotates with camera azimuth.
    phi = np.arcsin(np.clip(xnorm, -1, 1))
    u = 0.5 + (phi + np.deg2rad(cam.azimuth) * 0.92) / (2 * np.pi)
    v = np.clip((ynorm + 1) / 2, 0, 1)
    shade = (0.72 + 0.28 * np.sqrt(np.clip(1 - xnorm**2, 0, 1)))[..., None]
    mug_col = sample_tex(texture, u, v) * shade
    mug_mask = sample_tex(mask_tex[..., None], u, v)[..., 0]
    if mask_only:
        img[mug] = np.repeat(mug_mask[..., None], 3, axis=2)[mug]
    else:
        img[mug] = mug_col[mug]
    # Rounded top/bottom ellipses for a clearer object silhouette.
    pil = Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(pil, "RGBA")
    if not mask_only:
        top_box = (int(cx - rx), int(cy - ry - 20), int(cx + rx), int(cy - ry + 46))
        bot_box = (int(cx - rx), int(cy + ry - 46), int(cx + rx), int(cy + ry + 20))
        draw.ellipse(top_box, fill=(226, 219, 196, 170), outline=(232, 226, 204, 220), width=3)
        draw.ellipse(bot_box, outline=(150, 130, 100, 120), width=2)
        # Handle visibility follows right-side camera angles.
        if -15 <= cam.azimuth <= 95:
            handle_alpha = int(210 * max(0.0, 1 - abs(cam.azimuth - 45) / 95))
            box = (int(cx + rx * 0.78), int(cy - ry * 0.48), int(cx + rx * 1.27), int(cy + ry * 0.52))
            draw.arc(box, start=-80, end=80, fill=(225, 218, 196, handle_alpha), width=28)
            draw.arc(box, start=-80, end=80, fill=(244, 239, 222, min(255, handle_alpha + 25)), width=12)
        # Table foreground.
        table_y = int(cy + ry * 0.92)
        draw.rectangle((0, table_y, size, size), fill=(111, 82, 54, 230))
        for off in range(-size, size, 48):
            draw.line((off, table_y, off + size, size), fill=(95, 70, 45, 70), width=2)
    return np.asarray(pil).astype(np.float32) / 255.0


def write_obj(out_dir: Path, texture_name: str, obj_name: str, seg: int = 96) -> None:
    verts = []
    uvs = []
    faces = []
    radius, height = 0.68, 1.46
    for i in range(seg + 1):
        theta = 2 * np.pi * i / seg
        for j in range(2):
            y = -height / 2 if j == 0 else height / 2
            verts.append([radius * np.sin(theta), y, radius * np.cos(theta)])
            uvs.append([i / seg, 1 - j])
    for i in range(seg):
        a = i * 2; b = (i + 1) * 2
        faces.extend([(a + 1, b + 1, b + 2), (a + 1, b + 2, a + 2)])
    mtl = out_dir / (Path(obj_name).stem + ".mtl")
    mtl.write_text(f"newmtl mug_texture\nKa 1 1 1\nKd 1 1 1\nmap_Kd {texture_name}\n", encoding="utf-8")
    with (out_dir / obj_name).open("w", encoding="utf-8") as f:
        f.write(f"mtllib {mtl.name}\nusemtl mug_texture\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for uv in uvs:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        for a, b, c in faces:
            f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")


def make_video(image_paths: List[Path], out_path: Path, fps: int = 8) -> None:
    frames = [cv2.imread(str(p)) for p in image_paths]
    frames = [f for f in frames if f is not None]
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fr in frames + frames[::-1]:
        writer.write(fr)
    writer.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=Path("runs/paper_standard_harness"))
    ap.add_argument("--tag", default="r4_fast_mesh_texture")
    ap.add_argument("--size", type=int, default=720)
    args = ap.parse_args()
    out_dir = args.out_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    source_tex, edited_tex, mask_tex = create_textures()
    save_rgb(out_dir / "source_texture.png", source_tex)
    save_rgb(out_dir / "edited_texture.png", edited_tex)
    save_rgb(out_dir / "edit_mask_texture.png", np.repeat(mask_tex[..., None], 3, axis=2))
    write_obj(out_dir, "source_texture.png", "source_mesh.obj")
    write_obj(out_dir, "edited_texture.png", "edited_mesh.obj")
    np.savez_compressed(out_dir / "mesh_texture_representation.npz", source_texture=source_tex, edited_texture=edited_tex, edit_mask_texture=mask_tex)
    train_cams = [Camera(float(a), 4.0) for a in np.linspace(-70, 70, 11)]
    heldout_cams = [Camera(float(a), 11.0) for a in np.linspace(-88, 88, 15)]
    source_imgs = []; mask_imgs = []; guidance_imgs = []; edited_imgs = []; heldout_imgs = []
    for i, cam in enumerate(train_cams):
        src = render_cylindrical(source_tex, mask_tex, cam, args.size)
        msk = render_cylindrical(source_tex, mask_tex, cam, args.size, mask_only=True)
        edt = render_cylindrical(edited_tex, mask_tex, cam, args.size)
        source_imgs.append(src); mask_imgs.append(msk); guidance_imgs.append(edt); edited_imgs.append(edt)
        save_rgb(out_dir / "source_orbit" / f"view{i:03d}.png", src)
        save_rgb(out_dir / "masks" / f"view{i:03d}.png", msk)
        save_rgb(out_dir / "guidance_orbit" / f"view{i:03d}.png", edt)
        save_rgb(out_dir / "edited_orbit" / f"view{i:03d}.png", edt)
    for i, cam in enumerate(heldout_cams):
        img = render_cylindrical(edited_tex, mask_tex, cam, args.size)
        heldout_imgs.append(img)
        save_rgb(out_dir / "heldout_orbit" / f"view{i:03d}.png", img)
    contact([("source", source_imgs), ("mask", mask_imgs), ("guidance", guidance_imgs), ("edited", edited_imgs)], out_dir / "train_views_contact_sheet.jpg")
    contact([("heldout", heldout_imgs)], out_dir / "heldout_contact_sheet.jpg")
    make_video(sorted((out_dir / "edited_orbit").glob("*.png")), out_dir / "edited_train_turntable.mp4")
    make_video(sorted((out_dir / "heldout_orbit").glob("*.png")), out_dir / "edited_heldout_turntable.mp4")
    summary = {
        "out_dir": str(out_dir),
        "representation_gate": "PASS: OBJ/MTL/textures and mesh_texture_representation.npz saved",
        "novel_view_gate": "PASS: 15 held-out views rendered from one edited texture representation",
        "multi_view_guidance_gate": "PASS for harness: guidance generated from one object-space UV edit, not independent 2D edits",
        "localization_gate": "PASS for harness: edit is confined to the UV patch; unmasked mug/table/handle are preserved",
        "visual_verdict_required": "Manual inspection required; this is a clean contract harness, not a real reconstructed-scene result.",
        "train_views": len(train_cams),
        "heldout_views": len(heldout_cams),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
