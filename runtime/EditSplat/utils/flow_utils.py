from typing import Optional, Tuple, Union
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from tqdm import tqdm
import numpy as np

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


def scale_noise(
    scheduler,
    sample: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    noise: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """
    Foward process in flow-matching

    Args:
        sample (`torch.FloatTensor`):
            The input sample.
        timestep (`int`, *optional*):
            The current timestep in the diffusion chain.

    Returns:
        `torch.FloatTensor`:
            A scaled input sample.
    """
    # if scheduler.step_index is None:
    scheduler._init_step_index(timestep)

    sigma = scheduler.sigmas[scheduler.step_index]
    sample = sigma * noise + (1.0 - sigma) * sample

    return sample


# for flux
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def calc_v_sd3(pipe, src_tar_latent_model_input, src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, src_guidance_scale, tar_guidance_scale, t):
    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timestep = t.expand(src_tar_latent_model_input.shape[0])
    # joint_attention_kwargs = {}
    # # add timestep to joint_attention_kwargs
    # joint_attention_kwargs["timestep"] = timestep[0]
    # joint_attention_kwargs["timestep_idx"] = i


    with torch.no_grad():
        # # predict the noise for the source prompt
        noise_pred_src_tar = pipe.transformer(
            hidden_states=src_tar_latent_model_input,
            timestep=timestep,
            encoder_hidden_states=src_tar_prompt_embeds,
            pooled_projections=src_tar_pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        # perform guidance source
        if pipe.do_classifier_free_guidance:
            src_noise_pred_uncond, src_noise_pred_text, tar_noise_pred_uncond, tar_noise_pred_text = noise_pred_src_tar.chunk(4)
            noise_pred_src = src_noise_pred_uncond + src_guidance_scale * (src_noise_pred_text - src_noise_pred_uncond)
            noise_pred_tar = tar_noise_pred_uncond + tar_guidance_scale * (tar_noise_pred_text - tar_noise_pred_uncond)

    return noise_pred_src, noise_pred_tar

def calc_v_flux(pipe, latents, prompt_embeds, pooled_prompt_embeds, guidance, text_ids, latent_image_ids, t):
    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timestep = t.expand(latents.shape[0])
    # joint_attention_kwargs = {}
    # # add timestep to joint_attention_kwargs
    # joint_attention_kwargs["timestep"] = timestep[0]
    # joint_attention_kwargs["timestep_idx"] = i
    # 对齐 dtype/device & batch 维


    with torch.no_grad():
        # # predict the noise for the source prompt
        noise_pred = pipe.transformer(
            hidden_states=latents, # torch.Size([1, 4096, 64])
            timestep=timestep / 1000, # tensor([949.8726], device='cuda:0')
            guidance=guidance,
            encoder_hidden_states=prompt_embeds, # torch.Size([1, 512, 4096])
            txt_ids=text_ids,  # torch.Size([512, 3])
            img_ids=latent_image_ids, # torch.Size([1024, 3])
            pooled_projections=pooled_prompt_embeds, # torch.Size([1, 768])
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

    return noise_pred



@torch.no_grad()
def FlowEditSD3(pipe,
    scheduler,
    x_src,
    src_prompt,
    tar_prompt,
    negative_prompt,
    T_steps: int = 50,
    n_avg: int = 1,
    src_guidance_scale: float = 3.5,
    tar_guidance_scale: float = 13.5,
    n_min: int = 0,
    n_max: int = 15,):
    
    device = x_src.device

    timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None)

    num_warmup_steps = max(len(timesteps) - T_steps * scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)
    pipe._guidance_scale = src_guidance_scale
    
    # src prompts
    (
        src_prompt_embeds,
        src_negative_prompt_embeds,
        src_pooled_prompt_embeds,
        src_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    # tar prompts
    pipe._guidance_scale = tar_guidance_scale
    (
        tar_prompt_embeds,
        tar_negative_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )
 
    # CFG prep
    src_tar_prompt_embeds = torch.cat([src_negative_prompt_embeds, src_prompt_embeds, tar_negative_prompt_embeds, tar_prompt_embeds], dim=0)
    src_tar_pooled_prompt_embeds = torch.cat([src_negative_pooled_prompt_embeds, src_pooled_prompt_embeds, tar_negative_pooled_prompt_embeds, tar_pooled_prompt_embeds], dim=0)
    
    # initialize our ODE Zt_edit_1=x_src
    zt_edit = x_src.clone()

    for i, t in tqdm(enumerate(timesteps)):
        
        if T_steps - i > n_max:
            continue
        
        t_i = t/1000
        if i+1 < len(timesteps): 
            t_im1 = (timesteps[i+1])/1000
        else:
            t_im1 = torch.zeros_like(t_i).to(t_i.device)
        
        if T_steps - i > n_min:

            # Calculate the average of the V predictions
            V_delta_avg = torch.zeros_like(x_src)
            for k in range(n_avg):

                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                
                zt_src = (1-t_i)*x_src + (t_i)*fwd_noise

                zt_tar = zt_edit + zt_src - x_src

                src_tar_latent_model_input = torch.cat([zt_src, zt_src, zt_tar, zt_tar]) if pipe.do_classifier_free_guidance else (zt_src, zt_tar) 

                Vt_src, Vt_tar = calc_v_sd3(pipe, src_tar_latent_model_input,src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, src_guidance_scale, tar_guidance_scale, t)

                V_delta_avg += (1/n_avg) * (Vt_tar - Vt_src) # - (hfg-1)*( x_src))

            # propagate direct ODE
            zt_edit = zt_edit.to(torch.float32)

            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg
            
            zt_edit = zt_edit.to(V_delta_avg.dtype)

        else: # i >= T_steps-n_min # regular sampling for last n_min steps

            if i == T_steps-n_min:
                # initialize SDEDIT-style generation phase
                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                xt_src = scale_noise(scheduler, x_src, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src
                
            src_tar_latent_model_input = torch.cat([xt_tar, xt_tar, xt_tar, xt_tar]) if pipe.do_classifier_free_guidance else (xt_src, xt_tar)

            _, Vt_tar = calc_v_sd3(pipe, src_tar_latent_model_input,src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, src_guidance_scale, tar_guidance_scale, t)

            xt_tar = xt_tar.to(torch.float32)

            prev_sample = xt_tar + (t_im1 - t_i) * (Vt_tar)

            prev_sample = prev_sample.to(noise_pred_tar.dtype)

            xt_tar = prev_sample
        
    return zt_edit if n_min == 0 else xt_tar


@torch.no_grad()
def FlowEditFLUX(pipe,
    scheduler,
    x_src,
    src_prompt,
    tar_prompt,
    negative_prompt,
    T_steps: int = 28,
    n_avg: int = 1,
    src_guidance_scale: float = 1.5,
    tar_guidance_scale: float = 5.5,
    n_min: int = 0,
    n_max: int = 24,):
    # TODO 可能是encode这部分的问题
    # FlowEdit原始实现：
    '''
    image = Image.open(image_src_path)
                # print the shape of the image
                print(f"Loaded image {image_src_path} with size {image.size}") 
                # crop image to have both dimensions divisibe by 16 - avoids issues with resizing
                image = image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))
                print(f"Cropped image to size {image.size}")
                image_src = pipe.image_processor.preprocess(image)
                # cast image to half precision
                image_src = image_src.to(device).half()
                with torch.autocast("cuda"), torch.inference_mode():
                    x0_src_denorm = pipe.vae.encode(image_src).latent_dist.mode()
                x0_src = (x0_src_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
                # send to cuda
                x0_src = x0_src.to(device)
                print(f"target prompts: {tar_prompts}")
                
                for tar_num, tar_prompt in enumerate(tar_prompts):

                    if model_type == 'SD3':
                        x0_tar = FlowEditSD3(pipe,
                                                                scheduler,
                                                                x0_src,
                                                                src_prompt,
                                                                tar_prompt,
                                                                negative_prompt,
                                                                T_steps,
                                                                n_avg,
                                                                src_guidance_scale,
                                                                tar_guidance_scale,
                                                                n_min,
                                                                n_max,)
                        
                    elif model_type == 'FLUX':
                        x0_tar = FlowEditFLUX(pipe,
                                                                scheduler,
                                                                x0_src,
                                                                src_prompt,
                                                                tar_prompt,
                                                                negative_prompt,
                                                                T_steps,
                                                                n_avg,
                                                                src_guidance_scale,
                                                                tar_guidance_scale,
                                                                n_min,
                                                                n_max,)
                    else:
                        raise NotImplementedError(f"Sampler type {model_type} not implemented")


                    x0_tar_denorm = (x0_tar / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
                    # print the shape of x0_tar_denorm
                    print(f"x0_tar_denorm shape: {x0_tar_denorm.shape}")
                    with torch.autocast("cuda"), torch.inference_mode():
                        image_tar = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
                    print(f"image_tar shape: {image_tar.shape}")
                    print(f"image_tar shape: {image_tar.shape}")
                    image_tar = pipe.image_processor.postprocess(image_tar)

                    src_prompt_txt = data_dict["input_img"].split("/")[-1].split(".")[0]

                    tar_prompt_txt = str(tar_num)
                    
                    # make sure to create the directories before saving
                    save_dir = f"outputs/{exp_name}/{model_type}/src_{src_prompt_txt}/tar_{tar_prompt_txt}"
                    os.makedirs(save_dir, exist_ok=True)
                    print(f"Saving to {save_dir}")
                    
                    image_tar[0].save(f"{save_dir}/output_T_steps_{T_steps}_n_avg_{n_avg}_cfg_enc_{src_guidance_scale}_cfg_dec{tar_guidance_scale}_n_min_{n_min}_n_max_{n_max}_seed{seed}.png")
                    # also save source and target prompt in txt file
                    with open(f"{save_dir}/prompts.txt", "w") as f:
                        f.write(f"Source prompt: {src_prompt}\n")
                        f.write(f"Target prompt: {tar_prompt}\n")
                        f.write(f"Seed: {seed}\n")
                        f.write(f"Sampler type: {model_type}\n")
        print("Done")
    '''
    device = x_src.device
    orig_height, orig_width = x_src.shape[2]*pipe.vae_scale_factor//2, x_src.shape[3]*pipe.vae_scale_factor//2
    num_channels_latents = pipe.transformer.config.in_channels // 4

    pipe.check_inputs(
        prompt=src_prompt,
        prompt_2=None,
        height=orig_height,
        width=orig_width,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=512,
    )

    x_src, latent_src_image_ids = pipe.prepare_latents(batch_size= x_src.shape[0], num_channels_latents=num_channels_latents, height=orig_height, width=orig_width, dtype=x_src.dtype, device=x_src.device, generator=None,latents=x_src)
    x_src_packed = pipe._pack_latents(x_src, x_src.shape[0], num_channels_latents, x_src.shape[2], x_src.shape[3])
    latent_tar_image_ids = latent_src_image_ids

    # 5. Prepare timesteps
    sigmas = np.linspace(1.0, 1 / T_steps, T_steps)
    image_seq_len = x_src_packed.shape[1]
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )
    timesteps, T_steps = retrieve_timesteps(
        scheduler,
        T_steps,
        device,
        timesteps=None,
        sigmas=sigmas,
        mu=mu,
        )
    
    num_warmup_steps = max(len(timesteps) - T_steps * pipe.scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)

    
    # src prompts
    (
        src_prompt_embeds,
        src_pooled_prompt_embeds,
        src_text_ids,

    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        device=device,
    )

    # tar prompts
    pipe._guidance_scale = tar_guidance_scale
    (
        tar_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_text_ids,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        device=device,
    )

    # handle guidance
    if pipe.transformer.config.guidance_embeds:
        src_guidance = torch.tensor([src_guidance_scale], device=device)
        src_guidance = src_guidance.expand(x_src_packed.shape[0])
        tar_guidance = torch.tensor([tar_guidance_scale], device=device)
        tar_guidance = tar_guidance.expand(x_src_packed.shape[0])
    else:
        src_guidance = None
        tar_guidance = None

    # initialize our ODE Zt_edit_1=x_src
    zt_edit = x_src_packed.clone()

    for i, t in tqdm(enumerate(timesteps)):
        
        if T_steps - i > n_max:
            continue
        
        scheduler._init_step_index(t)
        t_i = scheduler.sigmas[scheduler.step_index]
        if i < len(timesteps):
            t_im1 = scheduler.sigmas[scheduler.step_index + 1]
        else:
            t_im1 = t_i
        
        if T_steps - i > n_min:

            # Calculate the average of the V predictions
            V_delta_avg = torch.zeros_like(x_src_packed)

            for k in range(n_avg):
                                    

                fwd_noise = torch.randn_like(x_src_packed).to(x_src_packed.device)
                
                zt_src = (1-t_i)*x_src_packed + (t_i)*fwd_noise

                zt_tar = zt_edit + zt_src - x_src_packed

                # Merge in the future to avoid double computation
                Vt_src = calc_v_flux(pipe,
                                                    latents=zt_src,
                                                    prompt_embeds=src_prompt_embeds, 
                                                    pooled_prompt_embeds=src_pooled_prompt_embeds, 
                                                    guidance=src_guidance,
                                                    text_ids=src_text_ids, 
                                                    latent_image_ids=latent_src_image_ids, 
                                                    t=t)
                
                Vt_tar = calc_v_flux(pipe,
                                                    latents=zt_tar,
                                                    prompt_embeds=tar_prompt_embeds, 
                                                    pooled_prompt_embeds=tar_pooled_prompt_embeds, 
                                                    guidance=tar_guidance,
                                                    text_ids=tar_text_ids, 
                                                    latent_image_ids=latent_tar_image_ids, 
                                                    t=t)

                V_delta_avg += (1/n_avg) * (Vt_tar - Vt_src) # - (hfg-1)*( x_src))

            # propagate direct ODE
            zt_edit = zt_edit.to(torch.float32)

            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg

            zt_edit = zt_edit.to(V_delta_avg.dtype)

        else: # i >= T_steps-n_min # regular sampling last n_min steps

            if i == T_steps-n_min:
                # initialize SDEDIT-style generation phase
                fwd_noise = torch.randn_like(x_src_packed).to(x_src_packed.device)
                xt_src = scale_noise(scheduler, x_src_packed, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src_packed
                
            Vt_tar = calc_v_flux(pipe,
                                    latents=xt_tar,
                                    prompt_embeds=tar_prompt_embeds, 
                                    pooled_prompt_embeds=tar_pooled_prompt_embeds, 
                                    guidance=tar_guidance,
                                    text_ids=tar_text_ids, 
                                    latent_image_ids=latent_tar_image_ids, 
                                    t=t)


            xt_tar = xt_tar.to(torch.float32)

            prev_sample = xt_tar + (t_im1 - t_i) * (Vt_tar)

            prev_sample = prev_sample.to(Vt_tar.dtype)
            xt_tar = prev_sample
    out = zt_edit if n_min == 0 else xt_tar
    unpacked_out = pipe._unpack_latents(out, orig_height, orig_width, pipe.vae_scale_factor)
    return unpacked_out

# -------------------------
# 工具：Flux 速度场（v_t）计算（teacher，无梯度）
# -------------------------
def _flux_velocity(
    pipe,
    latents_packed: torch.Tensor,
    t: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    latent_image_ids: torch.Tensor,
    guidance: Optional[torch.Tensor],   # shape: [B]，当 config.guidance_embeds=True 时必须给
):
    with torch.no_grad():
        timestep = t.expand(latents_packed.shape[0])

        # 若模型需要 guidance 嵌入，则确保 guidance 是 [B] 的张量；否则传 None
        if getattr(pipe.transformer.config, "guidance_embeds", False):
            if guidance is None:
                # 给一个“中性”的 guidance（例如 1.0 或对应 src/tgt 的 scale）
                guidance = torch.ones(latents_packed.shape[0], device=latents_packed.device, dtype=latents_packed.dtype)
        else:
            guidance = None

        v = pipe.transformer(
            hidden_states=latents_packed,
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            pooled_projections=pooled_prompt_embeds,  # 注意：diffusers 使用复数名 pooled_projections
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
    return v


# -------------------------
# 工具：Flux 的 CFG 只用于 p_t (附录里明确说明)
# v_theta(p_t,c) = v(p_t, c_src_null) + w * [ v(p_t, c_tgt) - v(p_t, c_src_null) ]
# v_theta(q_t,c) = v(q_t, c_src)              (不做 CFG)
# -------------------------
def _flux_velocity_cfg_for_pt(
    pipe,
    latents_packed_pt: torch.Tensor,
    t: torch.Tensor,
    embeds_src_null: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    embeds_tgt: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    latent_image_ids: torch.Tensor,
    cfg_scale: float,
    tar_guidance: Optional[torch.Tensor],  # 新增
):
    (src_null_prompt_embeds, src_null_pooled, src_null_text_ids) = embeds_src_null
    (tgt_prompt_embeds, tgt_pooled, tgt_text_ids) = embeds_tgt

    v_uncond = _flux_velocity(
        pipe, latents_packed_pt, t,
        src_null_prompt_embeds, src_null_pooled, src_null_text_ids,
        latent_image_ids,
        guidance=tar_guidance,    # 传 tar 的 guidance
    )
    v_cond = _flux_velocity(
        pipe, latents_packed_pt, t,
        tgt_prompt_embeds, tgt_pooled, tgt_text_ids,
        latent_image_ids,
        guidance=tar_guidance,
    )
    return v_uncond + cfg_scale * (v_cond - v_uncond)


# -------------------------
# 3D FlowAlign 的“teacher 梯度”（对 x_t 的梯度）  —— 对应式(46)/(48)
# ∇_{x_t} L_FA := v_t(p_t,c_tgt) - v_t(q_t,c_src) + γ ( E[p0|p_t] - E[q0|q_t] )
# 其中 E[p0|p_t] = p_t - t * v_t(p_t), E[q0|q_t] = q_t - t * v_t(q_t)
# Jacobian 近似为 I（论文假设），所以只需要把上式当作对 x_t 的梯度向量。
# -------------------------
def flowalign_flux3d_teacher_grad(
    pipe,
    scheduler,
    x_src_packed: torch.Tensor,
    x_t_packed: torch.Tensor,
    t_scalar: torch.Tensor,
    embeds_src: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    embeds_src_null: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    embeds_tgt: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    latent_image_ids: torch.Tensor,
    gamma: float = 1.0,
    cfg_scale_for_pt: float = 7.5,
    src_guidance_scale: float = 3.5,   # 新增：给 q_t 用
    tar_guidance_scale: float = 13.5,  # 新增：给 p_t 用
):
    device = x_src_packed.device
    scheduler._init_step_index(t_scalar)
    sigma_t = scheduler.sigmas[scheduler.step_index]
    noise = torch.randn_like(x_t_packed)
    # 这部分很可能是错的，gpt给了一个修正版 TODO
    # q_t = (1.0 - sigma_t) * x_src_packed + sigma_t * noise
    # p_t = q_t + x_t_packed - x_src_packed
    q_t = (1.0 - sigma_t) * x_src_packed + sigma_t * noise
    p_t = (1.0 - sigma_t) * x_t_packed   + sigma_t * noise

    B = p_t.shape[0]
    # gpt版 TODO 已经替换成了flowedit版
    # src_guidance = None
    # tar_guidance = None
    # if getattr(pipe.transformer.config, "guidance_embeds", False):
    #     src_guidance = torch.full((B,), float(src_guidance_scale), device=p_t.device, dtype=p_t.dtype)
    #     tar_guidance = torch.full((B,), float(tar_guidance_scale), device=p_t.device, dtype=p_t.dtype)
    # TODO 下面的部分还没有对照改完，记得对照着继续改（https://chatgpt.com/share/68e2ed6f-6460-800e-8b42-1efc72ffb2bb）
    # FlowEdit版 handle guidance
    if pipe.transformer.config.guidance_embeds:
        src_guidance = torch.tensor([src_guidance_scale], device=device)
        src_guidance = src_guidance.expand(x_src_packed.shape[0])
        tar_guidance = torch.tensor([tar_guidance_scale], device=device)
        tar_guidance = tar_guidance.expand(x_src_packed.shape[0])
    else:
        src_guidance = None
        tar_guidance = None
    # print("src_gd:", src_guidance)
    # print("tar_gd:", tar_guidance)

    (src_prompt_embeds, src_pooled, src_text_ids) = embeds_src

    # v(q_t, c_src)：不做 CFG，但仍需传 guidance（若模型需要）
    v_q = _flux_velocity(
        pipe, q_t, t_scalar,
        src_prompt_embeds, 
        src_pooled, 
        src_text_ids,
        latent_image_ids,
        guidance=src_guidance,
    )

    # v(p_t, ·)：只对 p_t 做 CFG
    v_p = _flux_velocity_cfg_for_pt(
        pipe, p_t, t_scalar,
        embeds_src_null=embeds_src_null,
        embeds_tgt=embeds_tgt,
        latent_image_ids=latent_image_ids,
        cfg_scale=cfg_scale_for_pt,
        tar_guidance=tar_guidance,
    )

    p0_hat = p_t - sigma_t * v_p
    q0_hat = q_t - sigma_t * v_q
    grad_xt = (v_p - v_q) + gamma * (p0_hat - q0_hat)
    return grad_xt, p_t, q_t, v_p, v_q


# =========================
# LucidDreamer: interval (t, t-τ) score distillation in latent space
# =========================

def _lucid_sample_pair_from_xt(
    xt_packed: torch.Tensor,
    scheduler,
    t_scalar: torch.Tensor,
    tau_steps: int = 1,
):
    """
    从当前学生 latent x_t 构造 (x_t_noisy, x_tm_noisy) 两个噪声等级的样本，噪声共享。
    公式：x_σ = (1-σ)*xt + σ*ε  （共享 ε；与 Flow 家族的线性噪声分解一致）
    返回：x_t_noisy, x_tm_noisy, sigma_t, sigma_tm, t_prev_scalar
    """
    scheduler._init_step_index(t_scalar)
    i = int(scheduler.step_index)
    j = max(i - int(tau_steps), 0)

    sigma_t  = scheduler.sigmas[i]
    sigma_tm = scheduler.sigmas[j]

    # 共享噪声，用 fp32 采样后 cast，数值更稳
    eps = torch.randn_like(xt_packed, dtype=torch.float32).to(xt_packed.dtype)

    x_t_noisy  = (1.0 - sigma_t ) * xt_packed + sigma_t  * eps
    x_tm_noisy = (1.0 - sigma_tm) * xt_packed + sigma_tm * eps

    # 与 transformer 前向对应的离散时间标量
    t_prev_scalar = scheduler.timesteps[j].to(t_scalar.device, dtype=t_scalar.dtype)
    return x_t_noisy, x_tm_noisy, sigma_t, sigma_tm, t_prev_scalar


def lucid_flux3d_teacher_grad(
    pipe,
    scheduler,
    x_t_packed: torch.Tensor,                          # 学生当前 latent（来自渲染+VAE）
    t_scalar: torch.Tensor,
    embeds_src_null: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],  # 仅用于 uncond 分支
    embeds_tgt: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    latent_image_ids: torch.Tensor,
    cfg_scale_for_pt: float = 50.0,    # Lucid 通常对两端都用 target CFG（可与 FlowAlign 保持一致）
    tar_guidance_scale: float = 50.0,
    tau_steps: int = 1,
    tweedie_weight: float = 1.0,       # 差分 Tweedie 的权重（Lucid 的核心项）
):
    """
    LucidDreamer-style interval SDS：
      - 以 xt 作为“干净信号”的替身，构造同噪声的 (x_t, x_{t-τ})
      - 两端都用 target 的 CFG 估计速度 v，并用 Tweedie：x0_hat = x - σ v
      - 以 ΔTweedie = (x0_hat_t - x0_hat_{t-τ}) 作为对 xt 的 teacher 梯度（忽略雅可比，和 FlowAlign/SDS 同假设）
    """
    # 采对 (t, t-τ)
    x_t_noisy, x_tm_noisy, sigma_t, sigma_tm, t_prev = _lucid_sample_pair_from_xt(
        x_t_packed, scheduler, t_scalar, tau_steps=tau_steps
    )

    B = x_t_packed.shape[0]
    tar_guidance = None
    if getattr(pipe.transformer.config, "guidance_embeds", False):
        tar_guidance = torch.full((B,), float(tar_guidance_scale),
                                  device=x_t_packed.device, dtype=x_t_packed.dtype)

    # 两端都做 target-CFG（uncond: null-src；cond: tgt）
    v_t   = _flux_velocity_cfg_for_pt(
        pipe, x_t_noisy,  t_scalar,
        embeds_src_null=embeds_src_null,
        embeds_tgt=embeds_tgt,
        latent_image_ids=latent_image_ids,
        cfg_scale=cfg_scale_for_pt,
        tar_guidance=tar_guidance,
    )
    v_tm  = _flux_velocity_cfg_for_pt(
        pipe, x_tm_noisy, t_prev,
        embeds_src_null=embeds_src_null,
        embeds_tgt=embeds_tgt,
        latent_image_ids=latent_image_ids,
        cfg_scale=cfg_scale_for_pt,
        tar_guidance=tar_guidance,
    )

    # Tweedie posterior means
    p0_hat_t  = x_t_noisy  - sigma_t  * v_t
    p0_hat_tm = x_tm_noisy - sigma_tm * v_tm

    # 教师梯度（忽略 d x_noisy / d xt 的系数，遵循 SDS/FlowAlign 的 I 雅可比近似）
    grad_xt = tweedie_weight * (p0_hat_t - p0_hat_tm)

    return grad_xt, {
        "x_t_noisy": x_t_noisy, "x_tm_noisy": x_tm_noisy,
        "v_t": v_t, "v_tm": v_tm,
        "p0_hat_t": p0_hat_t, "p0_hat_tm": p0_hat_tm,
        "sigma_t": sigma_t, "sigma_tm": sigma_tm,
    }


def _flux_velocity_cfg(
    pipe,
    latents_packed: torch.Tensor,
    t: torch.Tensor,
    embeds_uncond: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    embeds_cond: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    latent_image_ids: torch.Tensor,
    cfg_scale: float,
    guidance: Optional[torch.Tensor],
):
    uncond_prompt_embeds, uncond_pooled, uncond_text_ids = embeds_uncond
    cond_prompt_embeds,   cond_pooled,   cond_text_ids   = embeds_cond

    v_uncond = _flux_velocity(
        pipe, latents_packed, t,
        uncond_prompt_embeds, uncond_pooled, uncond_text_ids,
        latent_image_ids,
        guidance=guidance,
    )
    v_cond = _flux_velocity(
        pipe, latents_packed, t,
        cond_prompt_embeds, cond_pooled, cond_text_ids,
        latent_image_ids,
        guidance=guidance,
    )
    return v_uncond + cfg_scale * (v_cond - v_uncond)


# =========================
# 采样 (t, t-τ) 成对噪声（近似 Lucid：共享噪声，不做真实 DDIM 反演）
# =========================
def _lucid_sample_pair_from_xt(
    xt_packed: torch.Tensor,
    scheduler,
    t_scalar: torch.Tensor,
    tau_steps: int = 1,
):
    """
    用共享噪声 ε 构造 (x_t_noisy, x_tm_noisy)，两者分别对应当前 step i 与 i-τ 的 σ。
    x_σ = (1-σ)*xt + σ*ε  （Flow/EDM 风格线性噪声分解；ε fp32 再 cast，数值更稳）
    """
    scheduler._init_step_index(t_scalar)
    i = int(scheduler.step_index)
    j = max(i - int(tau_steps), 0)

    sigma_t  = scheduler.sigmas[i]
    sigma_tm = scheduler.sigmas[j]

    eps = torch.randn_like(xt_packed, dtype=torch.float32).to(xt_packed.dtype)
    x_t_noisy  = (1.0 - sigma_t ) * xt_packed + sigma_t  * eps
    x_tm_noisy = (1.0 - sigma_tm) * xt_packed + sigma_tm * eps

    t_prev_scalar = scheduler.timesteps[j].to(t_scalar.device, dtype=t_scalar.dtype)
    return x_t_noisy, x_tm_noisy, sigma_t, sigma_tm, t_prev_scalar


# =========================
# LucidDreamer 的“编辑版”：差的差
# Δ_tgt = Tweedie(t, tgt) - Tweedie(t-τ, tgt)
# Δ_src = Tweedie(t, src) - Tweedie(t-τ, src)
# grad = ω * (Δ_tgt - Δ_src)
# =========================
def lucid_edit_flux3d_teacher_grad(
    pipe,
    scheduler,
    x_t_packed: torch.Tensor,                                    # 学生当前 latent
    t_scalar: torch.Tensor,
    embeds_src: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], # 源条件
    embeds_src_null: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], # null 作为 uncond
    embeds_tgt: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], # 目标条件
    latent_image_ids: torch.Tensor,
    cfg_scale_for_pt: float = 7.5,        # 建议小 CFG（Lucid 推荐量级）
    tar_guidance_scale: float = 7.5,      # 数值 guidance（FLUX）
    src_guidance_scale: float = 7.5,
    tau_steps: int = 2,                   # t 与 t-τ 的间隔（以 scheduler 步为单位）
    tweedie_weight: float = 1.0,          # ω：Tweedie 差的权重
):
    # 1) 采 (t, t-τ) 成对样本（共享噪声）
    x_t_noisy, x_tm_noisy, sigma_t, sigma_tm, t_prev = _lucid_sample_pair_from_xt(
        x_t_packed, scheduler, t_scalar, tau_steps=tau_steps
    )

    B = x_t_packed.shape[0]
    tar_guidance = src_guidance = None
    if getattr(pipe.transformer.config, "guidance_embeds", False):
        tar_guidance = torch.full((B,), float(tar_guidance_scale),
                                  device=x_t_packed.device, dtype=x_t_packed.dtype)
        src_guidance = torch.full((B,), float(src_guidance_scale),
                                  device=x_t_packed.device, dtype=x_t_packed.dtype)

    # 2) 目标域：两时刻的 v 与 Tweedie
    v_tgt_t  = _flux_velocity_cfg(
        pipe, x_t_noisy,  t_scalar,
        embeds_uncond=embeds_src_null, embeds_cond=embeds_tgt,
        latent_image_ids=latent_image_ids,
        cfg_scale=cfg_scale_for_pt, guidance=tar_guidance,
    )
    v_tgt_s  = _flux_velocity_cfg(
        pipe, x_tm_noisy, t_prev,
        embeds_uncond=embeds_src_null, embeds_cond=embeds_tgt,
        latent_image_ids=latent_image_ids,
        cfg_scale=cfg_scale_for_pt, guidance=tar_guidance,
    )
    Tweedie_tgt_t = x_t_noisy  - sigma_t  * v_tgt_t
    Tweedie_tgt_s = x_tm_noisy - sigma_tm * v_tgt_s
    delta_tgt = Tweedie_tgt_t - Tweedie_tgt_s

    # 3) 源域：两时刻的 v 与 Tweedie
    v_src_t  = _flux_velocity_cfg(
        pipe, x_t_noisy,  t_scalar,
        embeds_uncond=embeds_src_null, embeds_cond=embeds_src,
        latent_image_ids=latent_image_ids,
        cfg_scale=cfg_scale_for_pt, guidance=src_guidance,
    )
    v_src_s  = _flux_velocity_cfg(
        pipe, x_tm_noisy, t_prev,
        embeds_uncond=embeds_src_null, embeds_cond=embeds_src,
        latent_image_ids=latent_image_ids,
        cfg_scale=cfg_scale_for_pt, guidance=src_guidance,
    )
    Tweedie_src_t = x_t_noisy  - sigma_t  * v_src_t
    Tweedie_src_s = x_tm_noisy - sigma_tm * v_src_s
    delta_src = Tweedie_src_t - Tweedie_src_s

    # 4) 差的差：teacher 梯度（忽略雅可比）
    grad_xt = tweedie_weight * (delta_tgt - delta_src)

    extras = {
        "x_t_noisy": x_t_noisy, "x_tm_noisy": x_tm_noisy,
        "v_tgt_t": v_tgt_t, "v_tgt_s": v_tgt_s,
        "v_src_t": v_src_t, "v_src_s": v_src_s,
        "Tweedie_tgt_t": Tweedie_tgt_t, "Tweedie_tgt_s": Tweedie_tgt_s,
        "Tweedie_src_t": Tweedie_src_t, "Tweedie_src_s": Tweedie_src_s,
        "sigma_t": sigma_t, "sigma_tm": sigma_tm,
    }
    return grad_xt, extras


# =========================
# 统一入口：增加 method="lucid_edit"
# =========================
def flowalign_flux3d_step(
    pipe,
    scheduler,
    xt_packed: torch.Tensor,
    xsrc_packed: torch.Tensor,       # 仍保留以兼容 flowalign 分支；lucid_edit 不需要它
    t_scalar: torch.Tensor,
    embeds_src, embeds_src_null, embeds_tgt,
    latent_image_ids: torch.Tensor,
    gamma: float,
    cfg_scale_for_pt: float,
    src_guidance_scale: float,
    tar_guidance_scale: float,
    # 新增参数（保持向后兼容）
    method: str = "flowalign",       # "flowalign" | "lucid" | "lucid_edit"
    lucid_tau_steps: int = 2,
    lucid_tweedie_weight: float = 1.0,
):
    if method.lower() == "flowalign":
        grad_xt, p_t, q_t, v_p, v_q = flowalign_flux3d_teacher_grad(
            pipe, 
            scheduler,
            x_src_packed=xsrc_packed,
            x_t_packed=xt_packed,
            t_scalar=t_scalar,
            embeds_src=embeds_src,
            embeds_src_null=embeds_src_null,
            embeds_tgt=embeds_tgt,
            latent_image_ids=latent_image_ids,
            gamma=gamma,
            cfg_scale_for_pt=cfg_scale_for_pt,
            src_guidance_scale=src_guidance_scale,
            tar_guidance_scale=tar_guidance_scale,
        )
        loss = (xt_packed * grad_xt.detach()).mean()
        return loss, {"method": "flowalign", "p_t": p_t, "q_t": q_t, "v_p": v_p, "v_q": v_q, "grad_xt": grad_xt}

    elif method.lower() == "lucid":
        grad_xt, extra = lucid_flux3d_teacher_grad(
            pipe, scheduler,
            x_t_packed=xt_packed,
            t_scalar=t_scalar,
            embeds_src_null=embeds_src_null,
            embeds_tgt=embeds_tgt,
            latent_image_ids=latent_image_ids,
            cfg_scale_for_pt=cfg_scale_for_pt,
            tar_guidance_scale=tar_guidance_scale,
            tau_steps=lucid_tau_steps,
            tweedie_weight=lucid_tweedie_weight,
        )
        loss = (xt_packed * grad_xt.detach()).mean()
        extra.update({"method": "lucid", "grad_xt": grad_xt})
        return loss, extra

    elif method.lower() == "lucid_edit":
        # —— 新增：编辑版 Lucid（差的差）
        grad_xt, extra = lucid_edit_flux3d_teacher_grad(
            pipe, scheduler,
            x_t_packed=xt_packed,
            t_scalar=t_scalar,
            embeds_src=embeds_src,
            embeds_src_null=embeds_src_null,
            embeds_tgt=embeds_tgt,
            latent_image_ids=latent_image_ids,
            cfg_scale_for_pt=cfg_scale_for_pt,
            tar_guidance_scale=tar_guidance_scale,
            src_guidance_scale=src_guidance_scale,
            tau_steps=lucid_tau_steps,
            tweedie_weight=lucid_tweedie_weight,
        )
        loss = (xt_packed * grad_xt.detach()).mean()
        extra.update({"method": "lucid_edit", "grad_xt": grad_xt})
        return loss, extra

    else:
        raise ValueError(f"Unknown method='{method}'.")


def flux_encode_text_triplet(pipe, prompt: str, device: torch.device):
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=prompt, prompt_2=None, device=device
    )
    return prompt_embeds, pooled_prompt_embeds, text_ids


if __name__ == "__main__":
    pass
