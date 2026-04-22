#!/usr/bin/env python3
import argparse
import math
import re
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw


def parse_iter(name: str) -> int:
    m = re.search(r"ours_(\d+)$", name)
    return int(m.group(1)) if m else -1


def latest_render_dir(model_dir: Path, split: str = "train") -> Path:
    cands = [p for p in (model_dir / split).glob("ours_*") if p.is_dir() and (p / "renders").exists()]
    if not cands:
        raise FileNotFoundError(f"No render dir in {model_dir / split}")
    cands = sorted(cands, key=lambda p: parse_iter(p.name))
    return cands[-1] / "renders"


def parse_indices(s: str) -> List[int]:
    if not s.strip():
        return []
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    return out


def load_image_or_blank(path: Path, size=(512, 512)) -> Image.Image:
    if path.exists():
        return Image.open(path).convert("RGB")
    return Image.new("RGB", size, (20, 20, 20))


def main():
    ap = argparse.ArgumentParser("Create comparison panels for full real outputs")
    ap.add_argument("--source_pretrained_dir", type=str, required=True)
    ap.add_argument("--source_iter", type=int, default=30000)
    ap.add_argument("--model_dirs", type=str, nargs="+", required=True)
    ap.add_argument("--model_labels", type=str, default="")
    ap.add_argument("--view_indices", type=str, default="0,10,20,30")
    ap.add_argument("--out_path", type=str, required=True)
    ap.add_argument("--tile_w", type=int, default=512)
    ap.add_argument("--tile_h", type=int, default=512)
    ap.add_argument("--pad", type=int, default=8)
    args = ap.parse_args()

    src_render = Path(args.source_pretrained_dir) / "train" / f"ours_{int(args.source_iter)}" / "renders"
    if not src_render.exists():
        raise FileNotFoundError(src_render)

    model_dirs = [Path(x) for x in args.model_dirs]
    if args.model_labels.strip():
        labels = [x.strip() for x in args.model_labels.split(",") if x.strip()]
        if len(labels) != len(model_dirs):
            raise ValueError("model_labels count must match model_dirs")
    else:
        labels = [p.name for p in model_dirs]

    renders = [latest_render_dir(md, split="train") for md in model_dirs]

    idxs = parse_indices(args.view_indices)
    if not idxs:
        idxs = [0, 10, 20, 30]

    cols = 1 + len(model_dirs)
    rows = len(idxs)

    tw = int(args.tile_w)
    th = int(args.tile_h)
    pad = int(args.pad)
    title_h = 28
    row_h = th + title_h

    W = cols * tw + (cols + 1) * pad
    H = rows * row_h + (rows + 1) * pad
    canvas = Image.new("RGB", (W, H), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    head_labels = ["src"] + labels

    for r, idx in enumerate(idxs):
        y0 = pad + r * row_h + r * pad

        # column 0: source
        name = f"{idx:05d}.png"
        src_im = load_image_or_blank(src_render / name, size=(tw, th)).resize((tw, th), Image.BILINEAR)
        x0 = pad
        canvas.paste(src_im, (x0, y0))
        draw.text((x0 + 6, y0 + th + 4), f"src view {idx}", fill=(255, 255, 255))

        for c, rd in enumerate(renders, start=1):
            x = pad + c * tw + c * pad
            im = load_image_or_blank(rd / name, size=(tw, th)).resize((tw, th), Image.BILINEAR)
            canvas.paste(im, (x, y0))
            draw.text((x + 6, y0 + th + 4), f"{head_labels[c]}", fill=(255, 255, 255))

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"[DONE] {out_path}")


if __name__ == "__main__":
    main()
