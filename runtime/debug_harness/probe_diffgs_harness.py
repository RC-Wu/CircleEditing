#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Iterable, List

torch = None
F = None


def _default_source_root() -> Path:
    env_root = os.environ.get("DIFFGS_SOURCE_ROOT") or os.environ.get("EDITSPLAT_ROOT")
    if env_root:
        return Path(env_root).expanduser()
    return Path(__file__).resolve().parents[2]


def _candidate_code_roots(source_root: Path) -> Iterable[Path]:
    yield source_root
    yield source_root / "runtime" / "EditSplat"
    yield source_root / "EditSplat"


def _discover_code_root(source_root: Path) -> Path:
    markers = (
        "scene",
        "run_editing_flow.py",
        "gaussian_renderer.py",
    )
    for candidate in _candidate_code_roots(source_root):
        if not candidate.is_dir():
            continue
        if any((candidate / marker).exists() for marker in markers):
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not find an EditSplat code root under {source_root}. "
        "Expected either the original checkout root or the CircleEditing overlay root."
    )


def _prepend_sys_paths(paths: Iterable[Path]) -> None:
    for path in reversed([p.resolve() for p in paths]):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _perspective_matrix(fov_y_deg: float, aspect: float, z_near: float, z_far: float, device: torch.device) -> torch.Tensor:
    tan_half_y = math.tan(math.radians(fov_y_deg) * 0.5)
    tan_half_x = tan_half_y * aspect

    proj = torch.zeros((4, 4), dtype=torch.float32, device=device)
    proj[0, 0] = 1.0 / tan_half_x
    proj[1, 1] = 1.0 / tan_half_y
    proj[2, 2] = z_far / (z_far - z_near)
    proj[2, 3] = -(z_far * z_near) / (z_far - z_near)
    proj[3, 2] = 1.0
    return proj


def _make_target(height: int, width: int, device: torch.device) -> torch.Tensor:
    ys = torch.linspace(0.0, 1.0, height, device=device)
    xs = torch.linspace(0.0, 1.0, width, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    target = torch.stack(
        [
            xx,
            yy,
            0.5 * torch.ones_like(xx),
        ],
        dim=0,
    )
    return target.contiguous()


def _make_gaussians(n: int, device: torch.device, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    xyz = torch.empty((n, 3), dtype=torch.float32)
    xyz.uniform_(-1.0, 1.0, generator=generator)
    xyz[:, 0:2] *= 0.22
    xyz[:, 2] = 1.8 + xyz[:, 2] * 0.35

    colors_raw = torch.empty((n, 3), dtype=torch.float32)
    colors_raw.uniform_(-1.5, 1.5, generator=generator)
    colors = torch.sigmoid(colors_raw)

    opacities_raw = torch.empty((n, 1), dtype=torch.float32)
    opacities_raw.uniform_(-0.5, 0.5, generator=generator)
    opacities = torch.sigmoid(opacities_raw) * 0.85 + 0.1

    scales_raw = torch.empty((n, 3), dtype=torch.float32)
    scales_raw.uniform_(-0.2, 0.2, generator=generator)
    scales = torch.exp(scales_raw) * 0.04

    rotations_raw = torch.zeros((n, 4), dtype=torch.float32)
    rotations_raw[:, 0] = 1.0
    rotations_raw[:, 1:].uniform_(-0.05, 0.05, generator=generator)

    xyz = xyz.to(device=device).contiguous().requires_grad_(True)
    colors = colors.to(device=device).contiguous().requires_grad_(True)
    opacities = opacities.to(device=device).contiguous().requires_grad_(True)
    scales = scales.to(device=device).contiguous().requires_grad_(True)
    rotations_raw = rotations_raw.to(device=device).contiguous().requires_grad_(True)

    return {
        "xyz": xyz,
        "colors": colors,
        "opacities": opacities,
        "scales": scales,
        "rotations_raw": rotations_raw,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal diffGS forward/backward harness.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=_default_source_root(),
        help="Path to either the CircleEditing repo root or the original EditSplat root.",
    )
    parser.add_argument(
        "--extra-path",
        action="append",
        default=[],
        type=Path,
        help="Extra import path to prepend before loading the rasterizer module. Repeatable.",
    )
    parser.add_argument("--gaussians", type=int, default=64, help="Number of synthetic Gaussians to render.")
    parser.add_argument("--height", type=int, default=64, help="Render height.")
    parser.add_argument("--width", type=int, default=64, help="Render width.")
    parser.add_argument("--seed", type=int, default=1, help="Deterministic seed for the synthetic batch.")
    parser.add_argument("--fov-deg", type=float, default=60.0, help="Vertical field of view in degrees.")
    parser.add_argument("--z-near", type=float, default=0.1, help="Near plane for the synthetic camera.")
    parser.add_argument("--z-far", type=float, default=10.0, help="Far plane for the synthetic camera.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device, usually cuda.")
    parser.add_argument(
        "--launch-blocking",
        action="store_true",
        help="Force CUDA_LAUNCH_BLOCKING=1 inside the harness for a cleaner traceback.",
    )
    parser.add_argument(
        "--debug-rasterizer",
        action="store_true",
        help="Enable the rasterizer debug flag in GaussianRasterizationSettings.",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    global torch, F
    args = _build_parser().parse_args(argv)

    if args.launch_blocking:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    try:
        import torch as _torch
        import torch.nn.functional as _F
    except Exception as exc:
        print("[ERROR] PyTorch is not available in this Python environment.", file=sys.stderr)
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    torch = _torch
    F = _F

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[ERROR] CUDA is not available in this environment.", file=sys.stderr)
        return 2

    source_root = Path(args.source_root).expanduser().resolve()
    code_root = _discover_code_root(source_root)
    import_paths = [*args.extra_path, code_root]
    _prepend_sys_paths(import_paths)

    print(f"[INFO] source_root={source_root}")
    print(f"[INFO] code_root={code_root}")
    if args.extra_path:
        print("[INFO] extra_paths=" + ", ".join(str(Path(p).expanduser().resolve()) for p in args.extra_path))
    print(f"[INFO] CUDA_LAUNCH_BLOCKING={os.environ.get('CUDA_LAUNCH_BLOCKING', '') or '<unset>'}")

    try:
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    except Exception as exc:
        print("[ERROR] Failed to import diff_gaussian_rasterization.", file=sys.stderr)
        print(f"[ERROR] {exc}", file=sys.stderr)
        print("[ERROR] Imported paths were:", file=sys.stderr)
        for entry in import_paths:
            print(f"  - {entry}", file=sys.stderr)
        return 3

    device = torch.device(args.device)
    n = int(args.gaussians)
    height = int(args.height)
    width = int(args.width)

    try:
        gaussians = _make_gaussians(n=n, device=device, seed=args.seed)
        means2d = torch.zeros_like(gaussians["xyz"], requires_grad=True)
        background = torch.zeros(3, dtype=torch.float32, device=device)
        target = _make_target(height=height, width=width, device=device)
        rotations = F.normalize(gaussians["rotations_raw"], dim=-1)

        tanfovy = math.tan(math.radians(args.fov_deg) * 0.5)
        tanfovx = tanfovy * (width / height)
        viewmatrix = torch.eye(4, dtype=torch.float32, device=device)
        projmatrix = _perspective_matrix(
            fov_y_deg=args.fov_deg,
            aspect=width / height,
            z_near=args.z_near,
            z_far=args.z_far,
            device=device,
        )

        settings = GaussianRasterizationSettings(
            image_height=height,
            image_width=width,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=background,
            scale_modifier=1.0,
            viewmatrix=viewmatrix,
            projmatrix=projmatrix,
            sh_degree=0,
            campos=torch.zeros(3, dtype=torch.float32, device=device),
            prefiltered=False,
            debug=bool(args.debug_rasterizer),
        )

        rasterizer = GaussianRasterizer(raster_settings=settings)
        forward_out = rasterizer(
            means3D=gaussians["xyz"],
            means2D=means2d,
            shs=None,
            colors_precomp=gaussians["colors"],
            opacities=gaussians["opacities"],
            scales=gaussians["scales"],
            rotations=rotations,
            cov3D_precomp=None,
        )

        if not isinstance(forward_out, tuple) or len(forward_out) < 2:
            raise RuntimeError(f"Unexpected rasterizer output: {type(forward_out)!r}")

        rendered = forward_out[0]
        radii = forward_out[1]
        if rendered.ndim == 3 and rendered.shape[0] not in (1, 3) and rendered.shape[-1] in (1, 3):
            rendered = rendered.permute(2, 0, 1).contiguous()

        if rendered.shape != target.shape:
            raise RuntimeError(
                f"Rendered image shape {tuple(rendered.shape)} does not match target {tuple(target.shape)}."
            )

        if device.type == "cuda":
            torch.cuda.synchronize()

        loss = F.mse_loss(rendered, target)
        loss.backward()

        if device.type == "cuda":
            torch.cuda.synchronize()

        visible = int((radii > 0).sum().item())
        print(f"[OK] loss={loss.item():.6f}")
        print(f"[OK] visible_gaussians={visible}/{n}")
        print(f"[OK] rendered_shape={tuple(rendered.shape)}")

        grad_report = {
            "xyz": gaussians["xyz"].grad,
            "colors": gaussians["colors"].grad,
            "opacities": gaussians["opacities"].grad,
            "scales": gaussians["scales"].grad,
            "rotations_raw": gaussians["rotations_raw"].grad,
            "means2D": means2d.grad,
        }
        for name, grad in grad_report.items():
            norm = float("nan") if grad is None else float(grad.norm().item())
            print(f"[GRAD] {name} norm={norm:.6e}")

        return 0
    except Exception as exc:
        print("[ERROR] Harness failed.", file=sys.stderr)
        print(f"[ERROR] {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
