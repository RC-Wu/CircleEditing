
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os
from typing import List
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import DiffusionPipeline
import attention as AM  # your module with UNet cross-attn utilities

def save_heatmap_and_masks(heat: np.ndarray, save_dir: str, prefix: str, fixed_thresholds: List[float], top_pcts: List[float]):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(5,5)); plt.imshow(heat, cmap="viridis", interpolation="nearest"); plt.axis("off"); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}heatmap.png"), dpi=200); plt.close()
    for thr in fixed_thresholds:
        m=(heat>=thr).astype(np.uint8)*255; plt.figure(figsize=(5,5)); plt.imshow(m, cmap="gray", vmin=0, vmax=255); plt.axis("off"); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}mask_thr_{thr:.2f}.png"), dpi=200); plt.close()
    flat=heat.flatten()
    for p in top_pcts:
        p=float(p)
        cut=np.quantile(flat, 1.0 - p/100.0)
        m=(heat>=cut).astype(np.uint8)*255
        plt.figure(figsize=(5,5)); plt.imshow(m, cmap="gray", vmin=0, vmax=255); plt.axis("off"); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}mask_top_{int(round(p))}p.png"), dpi=200); plt.close()

def run(model_id:str, prompt:str, object_word:str, steps:int, height:int, width:int, device:str, dtype:str, save_dir:str, fixed_thresholds:List[float], top_pcts:List[float], image_path:str=None):
    dt_map={"bf16":torch.bfloat16,"fp16":torch.float16,"fp32":torch.float32}
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dt_map.get(dtype, torch.float16)).to(device)
    pipe.set_progress_bar_config(disable=True)

    # enable attention capture on cross-attn
    AM.prep_unet(pipe.unet, is_cross_attn=True)
    pipe.unet.eval(); pipe.unet.requires_grad_(False)

    cls_name = pipe.__class__.__name__
    is_ip2p = "InstructPix2Pix" in cls_name or "StableDiffusionInstructPix2PixPipeline" in cls_name or "pix2pix" in model_id.lower()

    if is_ip2p:
        # InstructPix2Pix REQUIRES an input image. If not provided, create a neutral blank image.
        if image_path and os.path.exists(image_path):
            init_image = Image.open(image_path).convert("RGB")
            if init_image.size != (width, height):
                init_image = init_image.resize((width, height), Image.BICUBIC)
        else:
            init_image = Image.new("RGB", (width, height), (128,128,128))
        out = pipe(prompt=prompt, image=init_image, num_inference_steps=steps, guidance_scale=1.5)
    else:
        out = pipe(prompt=prompt, num_inference_steps=steps, height=height, width=width, guidance_scale=7.5)

    os.makedirs(save_dir, exist_ok=True)
    try: out.images[0].save(os.path.join(save_dir,"ip2p_sample.png"))
    except Exception: pass

    # Collect attention maps
    attn_maps = AM.get_all_attention_maps(pipe.unet)
    tokenizer = getattr(pipe, "tokenizer", None)
    if tokenizer is None: raise RuntimeError("Pipeline has no tokenizer attribute.")
    token_maps = AM.seperate_attention_maps_by_tokens(pipe.unet, attn_maps, tokenizer, prompt)

    # Save heatmap for object_word (uses AM.save_attention_maps)
    obj_map, obj_map_512 = AM.save_attention_maps(
        token_maps, attn_maps, object_word=object_word, output_dir=None, image_height=height, image_width=width
    )

    heat = obj_map_512
    if heat is None:
        # fallback: average of all tokens' heatmaps if object_word wasn't matched
        # Re-run save_attention_maps without object filtering to get average per-token maps and then average them.
        vals = []
        for token, lst in token_maps.items():
            for v in lst:
                vals.append(v)
        if vals:
            # v are already resized to 64x64 in AM.save_attention_maps path; we resize to (height,width)
            from skimage.transform import resize
            avg = np.mean([resize(v.reshape(64,64), (height,width)) if v.ndim==2 else v for v in vals], axis=0)
            heat = avg
        else:
            raise RuntimeError("No attention maps available to form a heatmap.")
    if torch.is_tensor(heat): heat = heat.detach().cpu().numpy()
    heat = heat.astype(np.float32)
    vmin, vmax = float(np.min(heat)), float(np.max(heat))
    heat = (heat - vmin) / (vmax - vmin) if vmax>vmin else np.zeros_like(heat, dtype=np.float32)

    save_heatmap_and_masks(heat, save_dir, "ip2p_", fixed_thresholds, top_pcts)
    AM.reset_attention_maps(pipe.unet)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--prompt", type=str, default="a photo of a panda on a bench, city background")
    p.add_argument("--object_word", type=str, default="panda")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="fp16", choices=["bf16","fp16","fp32"])
    p.add_argument("--save_dir", type=str, default="/root/autodl-tmp/EditSplat/utils/atten_test")
    p.add_argument("--fixed_thresholds", type=float, nargs="+", default=[0.30,0.50,0.70])
    p.add_argument("--top_pcts", type=float, nargs="+", default=[5,10,20])
    p.add_argument("--image", type=str, default=None, help="Path to an input image when using InstructPix2Pix.")
    a=p.parse_args()
    run(a.model_id,a.prompt,a.object_word,a.steps,a.height,a.width,a.device,a.dtype,a.save_dir,list(a.fixed_thresholds),list(a.top_pcts),image_path=a.image)

if __name__=="__main__":
    main()

'''
python /root/autodl-tmp/EditSplat/utils/ip2p_attention_test.py \
  --model_id timbrooks/instruct-pix2pix \
  --prompt "a photo of a stone bear statue in the forest" \
  --object_word bear \
  --image /root/autodl-tmp/EditSplat/dataset/dataset/bear/images_2/frame_00003.jpg \
  --steps 30 --height 512 --width 512 \
  --dtype fp16 \
  --save_dir /root/autodl-tmp/EditSplat/utils/atten_test \
  --fixed_thresholds 0.30 0.50 0.70 \
  --top_pcts 5 10 20

'''