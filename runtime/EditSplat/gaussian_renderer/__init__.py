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

import math
import os

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from utils.sh_utils import eval_sh
from scene.gaussian_model import GaussianModel


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _raster_debug_enabled(pipe=None) -> bool:
    pipe_debug = bool(getattr(pipe, "debug", False)) if pipe is not None else False
    return _env_flag("EDITSPLAT_RASTER_DEBUG", pipe_debug)


def _prepare_raster_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    device: torch.device = None,
    dtype: torch.dtype = None,
    require_finite: bool = False,
):
    if tensor is None:
        return None

    out = tensor
    if device is not None and out.device != device:
        out = out.to(device=device)
    if dtype is not None and out.dtype != dtype:
        out = out.to(dtype=dtype)
    if not out.is_contiguous():
        out = out.contiguous()
    if require_finite and not torch.isfinite(out.float()).all():
        raise RuntimeError(f"non-finite raster tensor: {name}")
    return out

def camera2rasterizer(viewpoint_camera, bg_color: torch.Tensor, sh_degree: int = 0):
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    device = bg_color.device
    bg = _prepare_raster_tensor("bg", bg_color, device=device, dtype=torch.float32, require_finite=True)
    viewmatrix = _prepare_raster_tensor(
        "viewmatrix",
        viewpoint_camera.world_view_transform,
        device=device,
        dtype=torch.float32,
        require_finite=True,
    )
    projmatrix = _prepare_raster_tensor(
        "projmatrix",
        viewpoint_camera.full_proj_transform,
        device=device,
        dtype=torch.float32,
        require_finite=True,
    )
    campos = _prepare_raster_tensor(
        "campos",
        viewpoint_camera.camera_center,
        device=device,
        dtype=torch.float32,
        require_finite=True,
    )

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg,
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        sh_degree=sh_degree,
        campos=campos,
        prefiltered=False,
        debug=_raster_debug_enabled(),
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    return rasterizer


def render(
    viewpoint_camera,
    pc: GaussianModel,
    # pc,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    device = pc.get_xyz.device
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz,
            dtype=pc.get_xyz.dtype,
            requires_grad=True,
            device=device,
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    bg = _prepare_raster_tensor("bg", bg_color, device=device, dtype=torch.float32, require_finite=True)
    viewmatrix = _prepare_raster_tensor(
        "viewmatrix",
        viewpoint_camera.world_view_transform,
        device=device,
        dtype=torch.float32,
        require_finite=True,
    )
    projmatrix = _prepare_raster_tensor(
        "projmatrix",
        viewpoint_camera.full_proj_transform,
        device=device,
        dtype=torch.float32,
        require_finite=True,
    )
    campos = _prepare_raster_tensor(
        "campos",
        viewpoint_camera.camera_center,
        device=device,
        dtype=torch.float32,
        require_finite=True,
    )

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg,
        scale_modifier=scaling_modifier,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        sh_degree=pc.active_sh_degree,
        campos=campos,
        prefiltered=False,
        debug=_raster_debug_enabled(pipe),
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = _prepare_raster_tensor("means3D", pc.get_xyz, device=device, dtype=torch.float32, require_finite=True)
    means2D = _prepare_raster_tensor("means2D", screenspace_points, device=device, dtype=torch.float32)
    opacity = _prepare_raster_tensor("opacity", pc.get_opacity, device=device, dtype=torch.float32, require_finite=True)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = _prepare_raster_tensor(
            "cov3D_precomp",
            pc.get_covariance(scaling_modifier),
            device=device,
            dtype=torch.float32,
            require_finite=True,
        )
    else:
        scales = _prepare_raster_tensor(
            "scales",
            pc.get_scaling,
            device=device,
            dtype=torch.float32,
            require_finite=True,
        )
        rotations = _prepare_raster_tensor(
            "rotations",
            pc.get_rotation,
            device=device,
            dtype=torch.float32,
            require_finite=True,
        )

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            camera_center = viewpoint_camera.camera_center.to(pc.get_xyz.device)
            dir_pp = pc.get_xyz - camera_center.repeat(
                pc.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = _prepare_raster_tensor(
                "shs",
                pc.get_features,
                device=device,
                dtype=torch.float32,
                require_finite=True,
            )

        if colors_precomp is not None:
            colors_precomp = _prepare_raster_tensor(
                "colors_precomp",
                colors_precomp,
                device=device,
                dtype=torch.float32,
                require_finite=True,
            )
    else:
        colors_precomp = _prepare_raster_tensor(
            "override_color",
            override_color,
            device=device,
            dtype=torch.float32,
            require_finite=True,
        )

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # import pdb; pdb.set_trace()
    rendered_image, radii, depth = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth_3dgs": depth,
    }


# from gaussiansplatting.scene.gaussian_model import GaussianModel


def point_cloud_render(
    viewpoint_camera,
    xyz,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
):
    screenspace_points = (
        torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device=xyz.device) + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    bg = _prepare_raster_tensor("bg", bg_color, device=xyz.device, dtype=torch.float32, require_finite=True)
    viewmatrix = _prepare_raster_tensor(
        "viewmatrix",
        viewpoint_camera.world_view_transform,
        device=xyz.device,
        dtype=torch.float32,
        require_finite=True,
    )
    projmatrix = _prepare_raster_tensor(
        "projmatrix",
        viewpoint_camera.full_proj_transform,
        device=xyz.device,
        dtype=torch.float32,
        require_finite=True,
    )
    campos = _prepare_raster_tensor(
        "campos",
        viewpoint_camera.camera_center,
        device=xyz.device,
        dtype=torch.float32,
        require_finite=True,
    )

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg,
        scale_modifier=scaling_modifier,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        sh_degree=0,
        campos=campos,
        prefiltered=False,
        debug=_raster_debug_enabled(pipe),
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = _prepare_raster_tensor("means3D", xyz, device=xyz.device, dtype=torch.float32, require_finite=True)
    means2D = _prepare_raster_tensor("means2D", screenspace_points, device=xyz.device, dtype=torch.float32)
    opacity = _prepare_raster_tensor(
        "opacity",
        torch.ones_like(xyz[..., 0:1]),
        device=xyz.device,
        dtype=torch.float32,
        require_finite=True,
    )

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    scales = _prepare_raster_tensor(
        "scales",
        torch.ones_like(xyz) * 0.005,
        device=xyz.device,
        dtype=torch.float32,
        require_finite=True,
    )
    rotations = torch.zeros([xyz.shape[0], 4], dtype=xyz.dtype, device=xyz.device)
    rotations[..., 0] = 1.0
    rotations = _prepare_raster_tensor(
        "rotations",
        rotations,
        device=xyz.device,
        dtype=torch.float32,
        require_finite=True,
    )

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    # if override_color is None:
    #     if pipe.convert_SHs_python:
    #         shs_view = pc.get_features.transpose(1, 2).view(
    #             -1, 3, (pc.max_sh_degree + 1) ** 2
    #         )
    #         dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
    #             pc.get_features.shape[0], 1
    #         )
    #         dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    #         sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    #         colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    #     else:
    #         shs = pc.get_features

    #     shs = shs.float()
    # else:
    #     colors_precomp = override_color
    colors_precomp = _prepare_raster_tensor(
        "colors_precomp",
        torch.ones_like(xyz[..., 0:1]).repeat(1, 3),
        device=xyz.device,
        dtype=torch.float32,
        require_finite=True,
    )

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # import pdb; pdb.set_trace()
    rendered_image, radii, depth = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth_3dgs": depth,
    }

