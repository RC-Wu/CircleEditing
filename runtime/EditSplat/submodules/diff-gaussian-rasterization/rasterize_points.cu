/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <cstdio>
#include <cuda_runtime_api.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <math.h>
#include <memory>
#include <sstream>
#include <stdio.h>
#include <string>
#include <torch/extension.h>
#include <tuple>

std::function<char *(size_t N)> resizeFunctional(torch::Tensor &t) {
  auto lambda = [&t](size_t N) {
    t.resize_({(long long)N});
    return reinterpret_cast<char *>(t.contiguous().data_ptr());
  };
  return lambda;
}

template <typename T>
const T *tensor_data_or_null(const torch::Tensor &t) {
  if (!t.defined() || t.numel() == 0) {
    return nullptr;
  }
  return t.data_ptr<T>();
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
    const torch::Tensor &background, const torch::Tensor &means3D,
    const torch::Tensor &colors, const torch::Tensor &opacity,
    const torch::Tensor &scales, const torch::Tensor &rotations,
    const float scale_modifier, const torch::Tensor &cov3D_precomp,
    const torch::Tensor &viewmatrix, const torch::Tensor &projmatrix,
    const float tan_fovx, const float tan_fovy, const int image_height,
    const int image_width, const torch::Tensor &sh, const int degree,
    const torch::Tensor &campos, const bool prefiltered, const bool debug) {
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }

  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);
  auto background_c = background.contiguous();
  auto means3D_c = means3D.contiguous();
  auto colors_c = colors.contiguous();
  auto opacity_c = opacity.contiguous();
  auto scales_c = scales.contiguous();
  auto rotations_c = rotations.contiguous();
  auto cov3D_precomp_c = cov3D_precomp.contiguous();
  auto viewmatrix_c = viewmatrix.contiguous();
  auto projmatrix_c = projmatrix.contiguous();
  auto sh_c = sh.contiguous();
  auto campos_c = campos.contiguous();

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor radii =
      torch::full({P}, 0, means3D.options().dtype(torch::kInt32));

  torch::Device device = means3D.device();
  torch::TensorOptions options = torch::TensorOptions().dtype(torch::kByte).device(device);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char *(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char *(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char *(size_t)> imgFunc = resizeFunctional(imgBuffer);

  int rendered = 0;
  if (P != 0) {
    int M = 0;
    if (sh.size(0) != 0) {
      M = sh.size(1);
    }

    rendered = CudaRasterizer::Rasterizer::forward(
        geomFunc, binningFunc, imgFunc, P, degree, M,
        background_c.data_ptr<float>(), W, H,
        means3D_c.data_ptr<float>(), tensor_data_or_null<float>(sh_c),
        tensor_data_or_null<float>(colors_c), opacity_c.data_ptr<float>(),
        tensor_data_or_null<float>(scales_c), scale_modifier,
        tensor_data_or_null<float>(rotations_c),
        tensor_data_or_null<float>(cov3D_precomp_c),
        viewmatrix_c.data_ptr<float>(),
        projmatrix_c.data_ptr<float>(),
        campos_c.data_ptr<float>(), tan_fovx, tan_fovy, prefiltered,
        out_color.data_ptr<float>(),
        out_depth.data_ptr<float>(), radii.data_ptr<int>(),
        debug);
  }
  return std::make_tuple(rendered, out_color, out_depth, radii, geomBuffer,
                         binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
    const torch::Tensor &background, const torch::Tensor &means3D,
    const torch::Tensor &radii, const torch::Tensor &colors,
    const torch::Tensor &scales, const torch::Tensor &rotations,
    const float scale_modifier, const torch::Tensor &cov3D_precomp,
    const torch::Tensor &viewmatrix, const torch::Tensor &projmatrix,
    const float tan_fovx, const float tan_fovy,
    const torch::Tensor &dL_dout_color, const torch::Tensor &sh,
    const int degree, const torch::Tensor &campos,
    const torch::Tensor &geomBuffer, const int R,
    const torch::Tensor &binningBuffer, const torch::Tensor &imageBuffer,
    const bool debug) {
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);

  int M = 0;
  if (sh.size(0) != 0) {
    M = sh.size(1);
  }
  auto background_c = background.contiguous();
  auto means3D_c = means3D.contiguous();
  auto radii_c = radii.contiguous();
  auto colors_c = colors.contiguous();
  auto scales_c = scales.contiguous();
  auto rotations_c = rotations.contiguous();
  auto cov3D_precomp_c = cov3D_precomp.contiguous();
  auto viewmatrix_c = viewmatrix.contiguous();
  auto projmatrix_c = projmatrix.contiguous();
  auto dL_dout_color_c = dL_dout_color.contiguous();
  auto sh_c = sh.contiguous();
  auto campos_c = campos.contiguous();
  auto geomBuffer_c = geomBuffer.contiguous();
  auto binningBuffer_c = binningBuffer.contiguous();
  auto imageBuffer_c = imageBuffer.contiguous();

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());

  if (P != 0) {
    CudaRasterizer::Rasterizer::backward(
        P, degree, M, R, background_c.data_ptr<float>(), W, H,
        means3D_c.data_ptr<float>(), tensor_data_or_null<float>(sh_c),
        tensor_data_or_null<float>(colors_c), tensor_data_or_null<float>(scales_c),
        scale_modifier, tensor_data_or_null<float>(rotations_c),
        tensor_data_or_null<float>(cov3D_precomp_c),
        viewmatrix_c.data_ptr<float>(),
        projmatrix_c.data_ptr<float>(),
        campos_c.data_ptr<float>(), tan_fovx, tan_fovy,
        radii_c.data_ptr<int>(),
        reinterpret_cast<char *>(geomBuffer_c.data_ptr()),
        reinterpret_cast<char *>(binningBuffer_c.data_ptr()),
        reinterpret_cast<char *>(imageBuffer_c.data_ptr()),
        dL_dout_color_c.data_ptr<float>(),
        dL_dmeans2D.data_ptr<float>(),
        dL_dconic.data_ptr<float>(),
        dL_dopacity.data_ptr<float>(),
        dL_dcolors.data_ptr<float>(),
        dL_dmeans3D.data_ptr<float>(),
        dL_dcov3D.data_ptr<float>(), dL_dsh.data_ptr<float>(),
        dL_dscales.data_ptr<float>(),
        dL_drotations.data_ptr<float>(), debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D,
                         dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(torch::Tensor &means3D, torch::Tensor &viewmatrix,
                          torch::Tensor &projmatrix) {
  const int P = means3D.size(0);

  torch::Tensor present =
      torch::full({P}, false, means3D.options().dtype(at::kBool));

  if (P != 0) {
    CudaRasterizer::Rasterizer::markVisible(
        P, means3D.contiguous().data<float>(),
        viewmatrix.contiguous().data<float>(),
        projmatrix.contiguous().data<float>(),
        present.contiguous().data<bool>());
  }

  return present;
}

void applyWeightsGaussiansCUDA(
    const torch::Tensor &background, const torch::Tensor &means3D,
    const torch::Tensor &weights, const torch::Tensor &opacity,
    const torch::Tensor &scales, const torch::Tensor &rotations,
    const float scale_modifier, torch::Tensor &cov3D_precomp,
    const torch::Tensor &viewmatrix, const torch::Tensor &projmatrix,
    const float tan_fovx, const float tan_fovy, const int image_height,
    const int image_width, const torch::Tensor &sh, const int degree,
    const torch::Tensor &campos, const bool prefiltered,
    const torch::Tensor &image_weights, torch::Tensor &cnt, const bool debug) {
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  const int num_channels = image_weights.size(0);
  // printf("num_channels %d\n", num_channels);

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);
  auto background_c = background.contiguous();
  auto means3D_c = means3D.contiguous();
  auto weights_c = weights.contiguous();
  auto opacity_c = opacity.contiguous();
  auto scales_c = scales.contiguous();
  auto rotations_c = rotations.contiguous();
  auto cov3D_precomp_c = cov3D_precomp.contiguous();
  auto viewmatrix_c = viewmatrix.contiguous();
  auto projmatrix_c = projmatrix.contiguous();
  auto sh_c = sh.contiguous();
  auto campos_c = campos.contiguous();
  auto image_weights_c = image_weights.contiguous();
  auto cnt_c = cnt.contiguous();

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor radii =
      torch::full({P}, 0, means3D.options().dtype(torch::kInt32));

  torch::Device device = means3D.device();
  torch::TensorOptions options = torch::TensorOptions().dtype(torch::kByte).device(device);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char *(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char *(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char *(size_t)> imgFunc = resizeFunctional(imgBuffer);

  if (P != 0) {
    int M = 0;
    if (sh.size(0) != 0) {
      M = sh.size(1);
    }

    CudaRasterizer::Rasterizer::apply_weights(
        geomFunc, binningFunc, imgFunc, P, degree, M,
        background_c.data_ptr<float>(), W, H,
        means3D_c.data_ptr<float>(), tensor_data_or_null<float>(sh_c),
        weights_c.data_ptr<float>(), opacity_c.data_ptr<float>(),
        tensor_data_or_null<float>(scales_c), scale_modifier,
        tensor_data_or_null<float>(rotations_c),
        tensor_data_or_null<float>(cov3D_precomp_c),
        viewmatrix_c.data_ptr<float>(),
        projmatrix_c.data_ptr<float>(),
        campos_c.data_ptr<float>(), tan_fovx, tan_fovy, prefiltered,
        image_weights_c.data_ptr<float>(),
        radii.data_ptr<int>(), cnt_c.data_ptr<int>(),
        num_channels, debug);
    if (!cnt.is_contiguous()) {
      cnt.copy_(cnt_c);
    }
  }
}
