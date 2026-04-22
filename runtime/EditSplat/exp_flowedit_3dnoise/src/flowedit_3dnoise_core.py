import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

from utils.flow_utils import calculate_shift, scale_noise


@dataclass
class PackedView:
    camera: object
    image_src_01: torch.Tensor
    x_src_packed: torch.Tensor
    latent_image_ids: torch.Tensor
    orig_height: int
    orig_width: int
    token_h: int
    token_w: int


def preprocess_like_flowedit(pipe, image_bchw_01: torch.Tensor, device: torch.device) -> torch.Tensor:
    assert image_bchw_01.ndim == 4 and image_bchw_01.shape[0] == 1

    if image_bchw_01.shape[-2:] != (512, 512):
        image_bchw_01 = F.interpolate(image_bchw_01, size=(512, 512), mode="bilinear", align_corners=True)

    img_pil = Image.fromarray(
        (image_bchw_01[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
    )

    w, h = img_pil.size
    wc = w - (w % 16)
    hc = h - (h % 16)
    if wc != w or hc != h:
        img_pil = img_pil.crop((0, 0, wc, hc))

    image_src = pipe.image_processor.preprocess(img_pil)
    image_src = image_src.to(device=device, dtype=pipe.vae.dtype)

    shift = float(getattr(pipe.vae.config, "shift_factor", 0.0))
    scale = float(getattr(pipe.vae.config, "scaling_factor", 1.0))

    with torch.autocast(device_type="cuda"), torch.inference_mode():
        x_denorm = pipe.vae.encode(image_src).latent_dist.mode()

    x_src = (x_denorm - shift) * scale
    return x_src


def pack_image_latents(pipe, image_bchw_01: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, int, int, int, int]:
    x_src_lat = preprocess_like_flowedit(pipe, image_bchw_01, device=device)

    orig_height = x_src_lat.shape[2] * pipe.vae_scale_factor // 2
    orig_width = x_src_lat.shape[3] * pipe.vae_scale_factor // 2

    pipe.check_inputs(
        prompt="(cache)",
        prompt_2=None,
        height=orig_height,
        width=orig_width,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=512,
    )

    num_channels_latents = pipe.transformer.config.in_channels // 4
    x_src_lat, latent_image_ids = pipe.prepare_latents(
        batch_size=x_src_lat.shape[0],
        num_channels_latents=num_channels_latents,
        height=orig_height,
        width=orig_width,
        dtype=x_src_lat.dtype,
        device=x_src_lat.device,
        generator=None,
        latents=x_src_lat,
    )

    x_src_packed = pipe._pack_latents(
        x_src_lat,
        x_src_lat.shape[0],
        num_channels_latents,
        x_src_lat.shape[2],
        x_src_lat.shape[3],
    )

    # FLUX _pack_latents packs 2x2 neighborhoods into one token.
    token_h = int(x_src_lat.shape[2] // 2)
    token_w = int(x_src_lat.shape[3] // 2)
    return x_src_packed, latent_image_ids, int(orig_height), int(orig_width), token_h, token_w


def unpack_and_decode(pipe, packed_latents: torch.Tensor, orig_height: int, orig_width: int) -> torch.Tensor:
    shift = float(getattr(pipe.vae.config, "shift_factor", 0.0))
    scale = float(getattr(pipe.vae.config, "scaling_factor", 1.0))

    latents = pipe._unpack_latents(packed_latents, orig_height, orig_width, pipe.vae_scale_factor)
    latents = (latents / scale) + shift
    image = pipe.vae.decode(latents.to(pipe.vae.dtype), return_dict=False)[0]
    return image


def encode_prompts(pipe, src_prompt: str, tar_prompt: str, src_guidance_scale: float, tar_guidance_scale: float, device: torch.device) -> Dict[str, torch.Tensor]:
    src_prompt_embeds, src_pooled, src_text_ids = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        device=device,
    )
    tar_prompt_embeds, tar_pooled, tar_text_ids = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        device=device,
    )

    if getattr(pipe.transformer.config, "guidance_embeds", False):
        src_guidance = torch.tensor([src_guidance_scale], device=device).expand(1)
        tar_guidance = torch.tensor([tar_guidance_scale], device=device).expand(1)
    else:
        src_guidance = None
        tar_guidance = None

    return {
        "src_prompt_embeds": src_prompt_embeds,
        "src_pooled": src_pooled,
        "src_text_ids": src_text_ids,
        "tar_prompt_embeds": tar_prompt_embeds,
        "tar_pooled": tar_pooled,
        "tar_text_ids": tar_text_ids,
        "src_guidance": src_guidance,
        "tar_guidance": tar_guidance,
    }


def flux_velocity(
    pipe,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    guidance: Optional[torch.Tensor],
    text_ids: torch.Tensor,
    latent_image_ids: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    timestep = t.expand(latents.shape[0])
    return pipe.transformer(
        hidden_states=latents,
        timestep=timestep / 1000,
        guidance=guidance,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=latent_image_ids,
        pooled_projections=pooled_prompt_embeds,
        joint_attention_kwargs=None,
        return_dict=False,
    )[0]


def build_flux_timesteps(pipe, seq_len: int, diffusion_steps: int, device: torch.device):
    scheduler = pipe.scheduler
    mu = calculate_shift(
        seq_len,
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )
    sigmas = np.linspace(1.0, 1.0 / diffusion_steps, diffusion_steps)
    timesteps, diffusion_steps_eff = retrieve_timesteps(
        scheduler,
        diffusion_steps,
        device,
        timesteps=None,
        sigmas=sigmas,
        mu=mu,
    )
    return scheduler, timesteps, int(diffusion_steps_eff)


@torch.no_grad()
def run_flowedit_with_noise(
    pipe,
    packed_view: PackedView,
    src_prompt: str,
    tar_prompt: str,
    diffusion_steps: int,
    n_avg: int,
    n_min: int,
    n_max: int,
    src_guidance_scale: float,
    tar_guidance_scale: float,
    seed: int,
    noise_provider: Optional[Callable[[], torch.Tensor]] = None,
    noise_mix_ratio: float = 1.0,
) -> torch.Tensor:
    x_src_packed = packed_view.x_src_packed
    latent_image_ids = packed_view.latent_image_ids
    device = x_src_packed.device

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    prompt_cache = encode_prompts(
        pipe,
        src_prompt=src_prompt,
        tar_prompt=tar_prompt,
        src_guidance_scale=src_guidance_scale,
        tar_guidance_scale=tar_guidance_scale,
        device=device,
    )

    model_dtype = next(pipe.transformer.parameters()).dtype

    def _to_dtype(x):
        return x.to(model_dtype) if (x is not None and torch.is_floating_point(x)) else x

    x_src_packed = _to_dtype(x_src_packed)
    zt_edit = x_src_packed.clone()

    scheduler, timesteps, diffusion_steps_eff = build_flux_timesteps(
        pipe,
        seq_len=x_src_packed.shape[1],
        diffusion_steps=diffusion_steps,
        device=device,
    )

    src_prompt_embeds = _to_dtype(prompt_cache["src_prompt_embeds"])
    src_pooled = _to_dtype(prompt_cache["src_pooled"])
    src_guidance = _to_dtype(prompt_cache["src_guidance"])

    tar_prompt_embeds = _to_dtype(prompt_cache["tar_prompt_embeds"])
    tar_pooled = _to_dtype(prompt_cache["tar_pooled"])
    tar_guidance = _to_dtype(prompt_cache["tar_guidance"])

    src_text_ids = prompt_cache["src_text_ids"]
    tar_text_ids = prompt_cache["tar_text_ids"]

    for i, t in enumerate(timesteps):
        if diffusion_steps_eff - i > n_max:
            continue

        scheduler._init_step_index(t)
        t_i = scheduler.sigmas[scheduler.step_index]
        t_im1 = scheduler.sigmas[scheduler.step_index + 1] if i < len(timesteps) - 1 else t_i

        if diffusion_steps_eff - i > n_min:
            v_delta_avg = torch.zeros_like(x_src_packed)
            for _ in range(n_avg):
                if noise_provider is None:
                    fwd_noise = torch.randn_like(x_src_packed)
                else:
                    n3d = noise_provider().to(x_src_packed.dtype)
                    rnd = torch.randn_like(x_src_packed)
                    mix = float(noise_mix_ratio)
                    fwd_noise = mix * n3d + math.sqrt(max(1e-6, 1.0 - mix * mix)) * rnd

                zt_src = (1.0 - t_i) * x_src_packed + t_i * fwd_noise
                zt_tar = zt_edit + zt_src - x_src_packed

                v_src = flux_velocity(
                    pipe,
                    latents=zt_src,
                    prompt_embeds=src_prompt_embeds,
                    pooled_prompt_embeds=src_pooled,
                    guidance=src_guidance,
                    text_ids=src_text_ids,
                    latent_image_ids=latent_image_ids,
                    t=t,
                )
                v_tar = flux_velocity(
                    pipe,
                    latents=zt_tar,
                    prompt_embeds=tar_prompt_embeds,
                    pooled_prompt_embeds=tar_pooled,
                    guidance=tar_guidance,
                    text_ids=tar_text_ids,
                    latent_image_ids=latent_image_ids,
                    t=t,
                )
                v_delta_avg = v_delta_avg + (v_tar - v_src) / float(n_avg)

            zt_edit = zt_edit.to(torch.float32)
            zt_edit = zt_edit + (t_im1 - t_i) * v_delta_avg.to(torch.float32)
            zt_edit = zt_edit.to(v_delta_avg.dtype)

        else:
            if i == diffusion_steps_eff - n_min:
                if noise_provider is None:
                    fwd_noise = torch.randn_like(x_src_packed)
                else:
                    n3d = noise_provider().to(x_src_packed.dtype)
                    rnd = torch.randn_like(x_src_packed)
                    mix = float(noise_mix_ratio)
                    fwd_noise = mix * n3d + math.sqrt(max(1e-6, 1.0 - mix * mix)) * rnd

                xt_src = scale_noise(scheduler, x_src_packed, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src_packed

            v_tar = flux_velocity(
                pipe,
                latents=xt_tar,
                prompt_embeds=tar_prompt_embeds,
                pooled_prompt_embeds=tar_pooled,
                guidance=tar_guidance,
                text_ids=tar_text_ids,
                latent_image_ids=latent_image_ids,
                t=t,
            )
            xt_tar = xt_tar.to(torch.float32)
            xt_tar = xt_tar + (t_im1 - t_i) * v_tar.to(torch.float32)
            xt_tar = xt_tar.to(v_tar.dtype)

    out_packed = zt_edit if n_min == 0 else xt_tar
    image = unpack_and_decode(pipe, out_packed, packed_view.orig_height, packed_view.orig_width)
    return image
