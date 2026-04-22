#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import json
import os
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn
from . import _C


_DUMP_COUNTERS = {"forward": 0, "backward": 0}


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [
        item.cpu().clone() if isinstance(item, torch.Tensor) else item
        for item in input_tuple
    ]
    return tuple(copied_tensors)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _tensor_debug_summary(tensor: torch.Tensor):
    summary = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "contiguous": bool(tensor.is_contiguous()),
        "stride": list(tensor.stride()),
        "storage_offset": int(tensor.storage_offset()),
        "numel": int(tensor.numel()),
        "data_ptr": int(tensor.data_ptr()),
    }
    if tensor.numel() > 0 and torch.is_floating_point(tensor):
        finite = torch.isfinite(tensor.float())
        summary["finite"] = bool(finite.all().item())
        if summary["finite"]:
            tensor32 = tensor.float()
            summary["min"] = float(tensor32.min().item())
            summary["max"] = float(tensor32.max().item())
    return summary


def _tensor_name_map(stage: str):
    if stage == "forward":
        return [
            "bg",
            "means3D",
            "colors_precomp",
            "opacities",
            "scales",
            "rotations",
            "scale_modifier",
            "cov3Ds_precomp",
            "viewmatrix",
            "projmatrix",
            "tanfovx",
            "tanfovy",
            "image_height",
            "image_width",
            "sh",
            "sh_degree",
            "campos",
            "prefiltered",
            "debug",
        ]
    return [
        "bg",
        "means3D",
        "radii",
        "colors_precomp",
        "scales",
        "rotations",
        "scale_modifier",
        "cov3Ds_precomp",
        "viewmatrix",
        "projmatrix",
        "tanfovx",
        "tanfovy",
        "grad_out_color",
        "sh",
        "sh_degree",
        "campos",
        "geomBuffer",
        "num_rendered",
        "binningBuffer",
        "imgBuffer",
        "debug",
    ]


def _debug_dump_dir():
    root = (
        os.environ.get("EDITSPLAT_RASTER_DEBUG_DIR")
        or os.environ.get("EDITSPLAT_ACTIVE_MODEL_PATH")
        or os.getcwd()
    )
    out_dir = Path(root) / "diffgs_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _dump_arg_metadata(stage: str, args, extra=None):
    if not _env_flag("EDITSPLAT_RASTER_DEBUG", False):
        return
    _DUMP_COUNTERS[stage] += 1
    payload = {"stage": stage, "args": {}, "extra": extra or {}}
    for name, value in zip(_tensor_name_map(stage), args):
        if isinstance(value, torch.Tensor):
            payload["args"][name] = _tensor_debug_summary(value)
        else:
            payload["args"][name] = value
    out_path = _debug_dump_dir() / f"{stage}_{_DUMP_COUNTERS[stage]:03d}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _prepare_ext_tensor(name: str, tensor: torch.Tensor, *, clone_buffer: bool = False):
    if tensor is None:
        return None
    out = tensor
    if not out.is_contiguous():
        out = out.contiguous()
    if torch.is_floating_point(out) and not torch.isfinite(out.float()).all():
        raise RuntimeError(f"non-finite diffgs tensor: {name}")
    if clone_buffer and out.numel() > 0:
        out = out.clone()
    return out


def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):
        # Restructure arguments the way that the C++ lib expects them
        means3D = _prepare_ext_tensor("means3D", means3D)
        means2D = _prepare_ext_tensor("means2D", means2D)
        sh = _prepare_ext_tensor("sh", sh)
        colors_precomp = _prepare_ext_tensor("colors_precomp", colors_precomp)
        opacities = _prepare_ext_tensor("opacities", opacities)
        scales = _prepare_ext_tensor("scales", scales)
        rotations = _prepare_ext_tensor("rotations", rotations)
        cov3Ds_precomp = _prepare_ext_tensor("cov3Ds_precomp", cov3Ds_precomp)
        bg = _prepare_ext_tensor("bg", raster_settings.bg)
        viewmatrix = _prepare_ext_tensor("viewmatrix", raster_settings.viewmatrix)
        projmatrix = _prepare_ext_tensor("projmatrix", raster_settings.projmatrix.float())
        campos = _prepare_ext_tensor("campos", raster_settings.campos)
        args = (
            bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            viewmatrix,
            projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            campos,
            raster_settings.prefiltered,
            raster_settings.debug,
        )
        _dump_arg_metadata("forward", args, extra={"c_ext": getattr(_C, "__file__", "unknown")})

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(
                args
            )  # Copy them before they can be corrupted
            try:
                (
                    num_rendered,
                    color,
                    depth,
                    radii,
                    geomBuffer,
                    binningBuffer,
                    imgBuffer,
                ) = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print(
                    "\nAn error occured in forward. Please forward snapshot_fw.dump for debugging."
                )
                raise ex
        else:
            (
                num_rendered,
                color,
                depth,
                radii,
                geomBuffer,
                binningBuffer,
                imgBuffer,
            ) = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        clone_buffers = _env_flag("EDITSPLAT_RASTER_CLONE_BUFFERS", False)
        radii = _prepare_ext_tensor("radii", radii, clone_buffer=clone_buffers)
        geomBuffer = _prepare_ext_tensor("geomBuffer", geomBuffer, clone_buffer=clone_buffers)
        binningBuffer = _prepare_ext_tensor("binningBuffer", binningBuffer, clone_buffer=clone_buffers)
        imgBuffer = _prepare_ext_tensor("imgBuffer", imgBuffer, clone_buffer=clone_buffers)
        ctx.raster_settings = raster_settings._replace(
            bg=bg,
            viewmatrix=viewmatrix,
            projmatrix=projmatrix,
            campos=campos,
        )
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            colors_precomp,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        )
        return color, radii, depth

    @staticmethod
    def backward(ctx, grad_out_color, grad_radii, grad_depth):
        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        (
            colors_precomp,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        ) = ctx.saved_tensors

        # Restructure args as C++ method expects them
        grad_out_color = _prepare_ext_tensor("grad_out_color", grad_out_color)
        radii = _prepare_ext_tensor("radii", radii)
        colors_precomp = _prepare_ext_tensor("colors_precomp", colors_precomp)
        means3D = _prepare_ext_tensor("means3D", means3D)
        scales = _prepare_ext_tensor("scales", scales)
        rotations = _prepare_ext_tensor("rotations", rotations)
        cov3Ds_precomp = _prepare_ext_tensor("cov3Ds_precomp", cov3Ds_precomp)
        sh = _prepare_ext_tensor("sh", sh)
        geomBuffer = _prepare_ext_tensor("geomBuffer", geomBuffer)
        binningBuffer = _prepare_ext_tensor("binningBuffer", binningBuffer)
        imgBuffer = _prepare_ext_tensor("imgBuffer", imgBuffer)
        bg = _prepare_ext_tensor("bg", raster_settings.bg)
        viewmatrix = _prepare_ext_tensor("viewmatrix", raster_settings.viewmatrix)
        projmatrix = _prepare_ext_tensor("projmatrix", raster_settings.projmatrix.float())
        campos = _prepare_ext_tensor("campos", raster_settings.campos)
        args = (
            bg,
            means3D,
            radii,
            colors_precomp,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            viewmatrix,
            projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            grad_out_color,
            sh,
            raster_settings.sh_degree,
            campos,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            raster_settings.debug,
        )
        _dump_arg_metadata("backward", args, extra={"num_rendered": int(num_rendered)})

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(
                args
            )  # Copy them before they can be corrupted
            try:
                (
                    grad_means2D,
                    grad_colors_precomp,
                    grad_opacities,
                    grad_means3D,
                    grad_cov3Ds_precomp,
                    grad_sh,
                    grad_scales,
                    grad_rotations,
                ) = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print(
                    "\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n"
                )
                raise ex
        else:
            (
                grad_means2D,
                grad_colors_precomp,
                grad_opacities,
                grad_means3D,
                grad_cov3Ds_precomp,
                grad_sh,
                grad_scales,
                grad_rotations,
            ) = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions, raster_settings.viewmatrix, raster_settings.projmatrix.float()
            )

        return visible

    def forward(
        self,
        means3D,
        means2D,
        opacities,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
    ):
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (
            shs is not None and colors_precomp is not None
        ):
            raise Exception(
                "Please provide excatly one of either SHs or precomputed colors!"
            )

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!"
            )

        if shs is None:
            shs = torch.Tensor([]).to(torch.float32).to("cuda")
        if colors_precomp is None:
            colors_precomp = torch.Tensor([]).to(torch.float32).to("cuda")

        if scales is None:
            scales = torch.Tensor([]).to(torch.float32).to("cuda")
        if rotations is None:
            rotations = torch.Tensor([]).to(torch.float32).to("cuda")
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([]).to(torch.float32).to("cuda")


        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
        )

    def apply_weights(
        self,
        means3D,
        means2D,
        opacities,
        shs=None,
        weights=None,
        scales=None,
        rotations=None,
        cov3Ds_precomp=None,
        cnt=None,
        image_weights=None,
    ):
        assert weights is not None
        assert cnt is not None
        assert image_weights is not None

        raster_settings = self.raster_settings
        means2D = torch.zeros_like(means3D)
        if shs is None:
            shs = torch.Tensor([]).to(torch.float32).to("cuda")

        if scales is None:
            scales = torch.Tensor([]).to(torch.float32).to("cuda")
        if rotations is None:
            rotations = torch.Tensor([]).to(torch.float32).to("cuda")
        if cov3Ds_precomp is None:
            cov3Ds_precomp = torch.Tensor([]).to(torch.float32).to("cuda")

        args = (
            raster_settings.bg,
            means3D,
            weights,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix.float(),
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            shs,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            image_weights,
            cnt,
            raster_settings.debug,
        )

        _C.apply_weights(*args)

