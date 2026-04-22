import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from .flowedit_3dnoise_core import (
    PackedView,
    build_flux_timesteps,
    encode_prompts,
    flux_velocity,
)
from .noise_field import SparseGaussianNoiseField


@dataclass
class NoiseOptimizeConfig:
    iterations: int = 60
    lr: float = 3e-2
    views_per_iter: int = 1
    objective_mode: str = "base"
    diffusion_steps: int = 28
    n_min: int = 0
    n_max: int = 24
    src_guidance_scale: float = 1.5
    tar_guidance_scale: float = 5.5
    noise_mix_ratio: float = 0.9
    lambda_edit: float = 1.0
    lambda_id: float = 0.5
    lambda_smooth: float = 0.2
    lambda_prior: float = 0.05
    lambda_delta: float = 0.2
    lambda_view_var: float = 0.2
    snr_gamma: float = 2.0
    max_grad_norm: float = 1.0
    seed: int = 0


def _set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def optimize_noise_field(
    pipe,
    field: SparseGaussianNoiseField,
    views: Sequence[PackedView],
    src_prompt: str,
    tar_prompt: str,
    cfg: NoiseOptimizeConfig,
    log_jsonl: str,
) -> List[Dict[str, float]]:
    assert len(views) > 0

    _set_seed(int(cfg.seed))

    device = views[0].x_src_packed.device
    model_dtype = next(pipe.transformer.parameters()).dtype

    pipe.transformer.eval()
    for p in pipe.transformer.parameters():
        p.requires_grad_(False)

    prompt_cache = encode_prompts(
        pipe,
        src_prompt=src_prompt,
        tar_prompt=tar_prompt,
        src_guidance_scale=cfg.src_guidance_scale,
        tar_guidance_scale=cfg.tar_guidance_scale,
        device=device,
    )

    def _to_dtype(x):
        return x.to(model_dtype) if (x is not None and torch.is_floating_point(x)) else x

    src_prompt_embeds = _to_dtype(prompt_cache["src_prompt_embeds"]).detach()
    src_pooled = _to_dtype(prompt_cache["src_pooled"]).detach()
    src_guidance = _to_dtype(prompt_cache["src_guidance"])
    if src_guidance is not None:
        src_guidance = src_guidance.detach()

    tar_prompt_embeds = _to_dtype(prompt_cache["tar_prompt_embeds"]).detach()
    tar_pooled = _to_dtype(prompt_cache["tar_pooled"]).detach()
    tar_guidance = _to_dtype(prompt_cache["tar_guidance"])
    if tar_guidance is not None:
        tar_guidance = tar_guidance.detach()

    src_text_ids = prompt_cache["src_text_ids"]
    tar_text_ids = prompt_cache["tar_text_ids"]

    scheduler, timesteps, t_steps_eff = build_flux_timesteps(
        pipe,
        seq_len=views[0].x_src_packed.shape[1],
        diffusion_steps=cfg.diffusion_steps,
        device=device,
    )

    valid_lo = max(0, t_steps_eff - int(cfg.n_max))
    valid_hi = max(1, t_steps_eff - int(cfg.n_min))
    valid_indices = list(range(valid_lo, valid_hi))
    if len(valid_indices) == 0:
        valid_indices = [max(0, t_steps_eff - 1)]

    os.makedirs(os.path.dirname(log_jsonl), exist_ok=True)
    with open(log_jsonl, "w", encoding="utf-8") as f:
        f.write(json.dumps({"config": asdict(cfg)}) + "\n")

    opt = torch.optim.Adam([field.noise], lr=float(cfg.lr))

    history: List[Dict[str, float]] = []

    for it in range(int(cfg.iterations)):
        opt.zero_grad(set_to_none=True)

        tidx = random.choice(valid_indices)
        t = timesteps[tidx]
        scheduler._init_step_index(t)
        sigma_t = scheduler.sigmas[scheduler.step_index]

        if int(cfg.views_per_iter) >= len(views):
            batch_views = list(views)
        else:
            batch_views = random.sample(list(views), int(cfg.views_per_iter))

        loss_edit_items = []
        loss_id_items = []

        for v in batch_views:
            x_src = v.x_src_packed.to(model_dtype)

            noise_3d, meta = field.render_to_tokens(
                camera=v.camera,
                token_h=v.token_h,
                token_w=v.token_w,
                normalize=True,
            )
            noise_3d = noise_3d.to(model_dtype)

            rnd = torch.randn_like(noise_3d)
            mix = float(cfg.noise_mix_ratio)
            noise_mix = mix * noise_3d + (max(1e-6, 1.0 - mix * mix) ** 0.5) * rnd

            xt = (1.0 - sigma_t) * x_src + sigma_t * noise_mix

            v_src = flux_velocity(
                pipe,
                latents=xt,
                prompt_embeds=src_prompt_embeds,
                pooled_prompt_embeds=src_pooled,
                guidance=src_guidance,
                text_ids=src_text_ids,
                latent_image_ids=v.latent_image_ids,
                t=t,
            )
            v_tar = flux_velocity(
                pipe,
                latents=xt,
                prompt_embeds=tar_prompt_embeds,
                pooled_prompt_embeds=tar_pooled,
                guidance=tar_guidance,
                text_ids=tar_text_ids,
                latent_image_ids=v.latent_image_ids,
                t=t,
            )

            delta = v_tar - v_src
            x0_src_hat = xt - sigma_t * v_src

            loss_edit_items.append(delta.float().pow(2).mean())
            loss_id_items.append(F.l1_loss(x0_src_hat.float(), x_src.float()))

        loss_edit_tensor = torch.stack(loss_edit_items)
        loss_id_tensor = torch.stack(loss_id_items)
        loss_edit = loss_edit_tensor.mean()
        loss_id = loss_id_tensor.mean()
        if loss_edit_tensor.numel() > 1:
            loss_view_var = loss_edit_tensor.var(unbiased=False) + loss_id_tensor.var(unbiased=False)
        else:
            loss_view_var = torch.zeros([], device=device, dtype=torch.float32)

        loss_smooth = field.smoothness_loss().float()
        loss_prior = field.prior_loss().float()

        mode = str(cfg.objective_mode).lower()
        sigma = sigma_t.float()
        snr = ((1.0 - sigma).clamp_min(1e-6) / sigma.clamp_min(1e-6)) ** 2
        w_snr = snr.pow(float(cfg.snr_gamma)).clamp(0.2, 5.0)

        if mode == "snr":
            total = (
                float(cfg.lambda_id) * w_snr * loss_id
                + float(cfg.lambda_smooth) * loss_smooth
                + float(cfg.lambda_prior) * loss_prior
                - float(cfg.lambda_edit) * (1.0 / w_snr) * loss_edit
            )
            loss_delta = torch.zeros([], device=device, dtype=torch.float32)
        elif mode == "delta":
            # Two-time delta objective (lightweight surrogate of interval-based flow optimization).
            t_prev_idx = min(tidx + 1, len(timesteps) - 1)
            t_prev = timesteps[t_prev_idx]
            scheduler._init_step_index(t_prev)
            sigma_prev = scheduler.sigmas[scheduler.step_index]

            loss_delta = torch.zeros([], device=device, dtype=torch.float32)
            for v in batch_views:
                x_src = v.x_src_packed.to(model_dtype)
                noise_3d, _ = field.render_to_tokens(v.camera, v.token_h, v.token_w, normalize=True)
                noise_3d = noise_3d.to(model_dtype)
                rnd = torch.randn_like(noise_3d)
                mix = float(cfg.noise_mix_ratio)
                noise_mix = mix * noise_3d + (max(1e-6, 1.0 - mix * mix) ** 0.5) * rnd

                xt = (1.0 - sigma) * x_src + sigma * noise_mix
                xt_prev = (1.0 - sigma_prev) * x_src + sigma_prev * noise_mix

                v_src_t = flux_velocity(pipe, xt, src_prompt_embeds, src_pooled, src_guidance, src_text_ids, v.latent_image_ids, t)
                v_tar_t = flux_velocity(pipe, xt, tar_prompt_embeds, tar_pooled, tar_guidance, tar_text_ids, v.latent_image_ids, t)
                v_src_p = flux_velocity(pipe, xt_prev, src_prompt_embeds, src_pooled, src_guidance, src_text_ids, v.latent_image_ids, t_prev)
                v_tar_p = flux_velocity(pipe, xt_prev, tar_prompt_embeds, tar_pooled, tar_guidance, tar_text_ids, v.latent_image_ids, t_prev)

                delta_t = (v_tar_t - v_src_t).float()
                delta_p = (v_tar_p - v_src_p).float()
                loss_delta = loss_delta + (delta_t - delta_p).pow(2).mean()

            loss_delta = loss_delta * (1.0 / float(len(batch_views)))
            total = (
                float(cfg.lambda_id) * loss_id
                + float(cfg.lambda_smooth) * loss_smooth
                + float(cfg.lambda_prior) * loss_prior
                - float(cfg.lambda_edit) * loss_edit
                - float(cfg.lambda_delta) * loss_delta
            )
        elif mode == "balanced":
            loss_delta = torch.zeros([], device=device, dtype=torch.float32)
            total = (
                float(cfg.lambda_id) * loss_id
                + float(cfg.lambda_smooth) * loss_smooth
                + float(cfg.lambda_prior) * loss_prior
                - float(cfg.lambda_edit) * loss_edit
                + float(cfg.lambda_view_var) * loss_view_var
            )
        else:
            loss_delta = torch.zeros([], device=device, dtype=torch.float32)
            total = (
                float(cfg.lambda_id) * loss_id
                + float(cfg.lambda_smooth) * loss_smooth
                + float(cfg.lambda_prior) * loss_prior
                - float(cfg.lambda_edit) * loss_edit
            )

        total.backward()
        if float(cfg.max_grad_norm) > 0:
            torch.nn.utils.clip_grad_norm_([field.noise], float(cfg.max_grad_norm))
        opt.step()

        rec = {
            "iter": float(it),
            "t_index": float(tidx),
            "sigma": float(sigma_t.detach().cpu()),
            "loss_total": float(total.detach().cpu()),
            "loss_edit_gain": float(loss_edit.detach().cpu()),
            "loss_id": float(loss_id.detach().cpu()),
            "loss_smooth": float(loss_smooth.detach().cpu()),
            "loss_prior": float(loss_prior.detach().cpu()),
            "loss_delta": float(loss_delta.detach().cpu()),
            "loss_view_var": float(loss_view_var.detach().cpu()),
            "noise_std": float(field.noise.detach().std().cpu()),
        }
        history.append(rec)

        with open(log_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    return history
