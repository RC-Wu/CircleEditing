#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, math, os, sys, inspect
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

try:
    import diffusers
    from diffusers import FluxPipeline
    from diffusers.models.attention import Attention
except Exception as e:
    print("[E] Failed to import diffusers/FluxPipeline/Attention:", e, file=sys.stderr)
    raise


# ===================== helpers =====================
def _hr(s: str = "", char: str = "-") -> None:
    print((char * 10 + f" {s} " + char * 10) if s else (char * 40))


def version_check(min_required: str = "0.32.0") -> None:
    cur = diffusers.__version__
    def vt(v: str) -> Tuple[int, int, int]:
        p = (v.split(".") + ["0", "0", "0"])[:3]
        return tuple(int(x.split("+")[0]) for x in p)
    if vt(cur) < vt(min_required):
        print(f"[!] diffusers=={cur} (< {min_required}). Using proxy-capture w/ RoPE on 0.30.x")
    else:
        print(f"[OK] diffusers=={cur} (>= {min_required}).")


def best_factor_pair(n: int) -> Tuple[int, int]:
    r = int(math.sqrt(n))
    best, best_gap = (1, n), n
    for a in range(1, r + 1):
        if n % a == 0:
            b = n // a
            g = abs(a - b)
            if g < best_gap:
                best_gap, best = g, (a, b)
    return best


def robust_grid_from_img_ids(img_ids_1st: np.ndarray, n_img: int) -> Tuple[int, int]:
    uniq = []
    for ax in range(min(img_ids_1st.shape[-1], 3)):
        uniq.append(np.unique(np.round(img_ids_1st[:, ax], 6)).size)
    cand = [u for u in uniq if u > 1]
    if len(cand) >= 2:
        cand.sort()
        Hs, Ws = cand[-2], cand[-1]
        if Hs * Ws == n_img:
            return (Hs, Ws)
    return best_factor_pair(n_img)


def upsample_heatmap(heat: np.ndarray, height: int, width: int, smooth: int = 0) -> np.ndarray:
    t = torch.from_numpy(heat).float().unsqueeze(0).unsqueeze(0)  # 1x1xH'xW'
    if smooth and min(heat.shape) >= 3:
        t = F.avg_pool2d(t, kernel_size=3, stride=1, padding=1)
    t = F.interpolate(t, size=(height, width), mode="bilinear", align_corners=False)
    return t.squeeze(0).squeeze(0).clamp(0, 1).numpy()


def build_flux_pipeline(model_id: str, device: str, dtype: torch.dtype) -> FluxPipeline:
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)
    if device:
        pipe = pipe.to(device)
    try:
        if hasattr(pipe, "transformer") and hasattr(pipe.transformer, "unfuse_qkv_projections"):
            pipe.transformer.unfuse_qkv_projections()
            print("[i] transformer.unfuse_qkv_projections() called.")
    except Exception as e:
        print("[w] Could not unfuse QKV projections:", e)
    return pipe


def show_tokenization(pipe: FluxPipeline, prompt: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if hasattr(pipe, "tokenizer"):
        tok = pipe.tokenizer
        enc = tok(prompt, padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt")
        ids = enc.input_ids[0].tolist()
        toks = tok.convert_ids_to_tokens(ids)
        mask = enc.attention_mask[0].tolist() if "attention_mask" in enc else [1] * len(toks)
        valid = [t for t, m in zip(toks, mask) if m == 1]
        out["clip"] = {"tokens_valid": valid}
        print("\n[Tokenizer: CLIP] valid tokens:", valid)

    if hasattr(pipe, "tokenizer_2"):
        tok2 = pipe.tokenizer_2
        enc2 = tok2(prompt, padding="max_length", max_length=tok2.model_max_length, truncation=True, return_tensors="pt")
        ids2 = enc2.input_ids[0].tolist()
        toks2 = tok2.convert_ids_to_tokens(ids2)
        mask2 = enc2.attention_mask[0].tolist() if "attention_mask" in enc2 else [1] * len(toks2)
        valid2 = [t for t, m in zip(toks2, mask2) if m == 1]
        out["t5"] = {"tokens_valid": valid2}
        print("\n[Tokenizer: T5]   valid tokens:", valid2)
    return out


def find_object_subtokens(token_list: List[str], object_word: str) -> List[int]:
    if not object_word:
        return []
    key = object_word.lower()
    return [i for i, t in enumerate(token_list) if key in t.lower()]


# ===================== probe hook =====================
class FluxProbeStore:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.first_img_ids: Optional[np.ndarray] = None  # (N_img, 3)
        self.first_txt_ids: Optional[np.ndarray] = None  # (N_txt, 3)
        self.derived: Optional[Dict[str, int]] = None

    def add(self, info: Dict[str, Any]) -> None:
        self.calls.append(info)


def attach_probe_hook(pipe: FluxPipeline) -> FluxProbeStore:
    store = FluxProbeStore()
    module = pipe.transformer

    def hook_fn(mod, args, kwargs):
        H = kwargs.get("hidden_states", None)
        E = kwargs.get("encoder_hidden_states", None)
        if store.derived is None and isinstance(H, torch.Tensor) and isinstance(E, torch.Tensor):
            B, N_all = H.shape[:2]
            N_txt = int(E.shape[1])
            N_img = int(N_all - N_txt)
            store.derived = {"B": int(B), "N_all": int(N_all), "N_txt": int(N_txt), "N_img": int(N_img)}

        img_ids = kwargs.get("img_ids", None)
        if isinstance(img_ids, torch.Tensor) and store.first_img_ids is None:
            arr = img_ids.detach().to(torch.float32).cpu().numpy()
            if arr.ndim == 3:
                store.first_img_ids = arr[0]
        txt_ids = kwargs.get("txt_ids", None)
        if isinstance(txt_ids, torch.Tensor) and store.first_txt_ids is None:
            arr = txt_ids.detach().to(torch.float32).cpu().numpy()
            if arr.ndim == 3:
                store.first_txt_ids = arr[0]
        return None

    module.register_forward_pre_hook(hook_fn, with_kwargs=True)
    print("[OK] Installed probe hook on pipe.transformer (pre-forward).")
    return store


# ===================== RoPE (flexible, FLUX-friendly) =====================
def _as_cpu(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to(torch.float32).cpu()


def _split_even_odd(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # x: (..., D), D even
    return x[..., ::2], x[..., 1::2]


def _merge_even_odd(even: torch.Tensor, odd: torch.Tensor) -> torch.Tensor:
    # interleave last dim
    s = even.shape
    out = torch.empty((*s[:-1], s[-1] * 2), dtype=even.dtype, device=even.device)
    out[..., ::2] = even
    out[..., 1::2] = odd
    return out


def _apply_rope_slice(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, start: int, length: int) -> torch.Tensor:
    """
    在 x[..., start:start+length] 上应用 1D RoPE。length 必须为偶数（两两偶奇配对）。
    cos/sin 形状允许：(N, length/2) 或 (1,N,length/2) 或 (B,N,length/2)。自动广播到 (B*H, N, length/2)。
    """
    if length <= 0 or length % 2 == 1:
        return x
    seg = x[..., start:start + length]                      # (B*H, N, L)
    even, odd = _split_even_odd(seg)                        # (B*H, N, L/2) each

    # 广播 cos/sin 到 (B*H, N, L/2)
    def _prep(z):
        if z.dim() == 2:            # (N, L/2)
            z = z.unsqueeze(0)      # (1,N,L/2)
        elif z.dim() == 3:          # (B,N,L/2)
            pass
        else:
            raise RuntimeError("Unexpected cos/sin rank.")
        return z.to(x.device, dtype=x.dtype)

    cos = _prep(cos); sin = _prep(sin)
    # 若 B 与 B*H 不等，靠广播一维到多维（我们不分 head，这里 head 已并到 batch）
    while cos.dim() < seg.dim():  # want (B*H, N, L/2)
        cos = cos.expand(-1, -1, -1)
        sin = sin.expand(-1, -1, -1)

    rot_even = even * cos - odd * sin
    rot_odd  = even * sin + odd * cos
    rot = _merge_even_odd(rot_even, rot_odd)
    x[..., start:start + length] = rot
    return x


def _normalize_rope_emb(image_rotary_emb) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    将 FLUX 的 image_rotary_emb 统一归一化为 [(cos_x, sin_x), (cos_y, sin_y)] 的列表。
    每个 cos/sin 形状 accept:
      - (N, D/2)
      - (B, N, D/2)
    """
    axes: List[Tuple[torch.Tensor, torch.Tensor]] = []
    if image_rotary_emb is None:
        return axes

    # 常见：list/tuple of axes; 每个 axis 是 (cos, sin)
    if isinstance(image_rotary_emb, (list, tuple)):
        for ax in image_rotary_emb:
            if isinstance(ax, (list, tuple)) and len(ax) == 2:
                cos, sin = ax
                if isinstance(cos, torch.Tensor) and isinstance(sin, torch.Tensor):
                    axes.append((cos, sin))
    # 兜底：若直接给了 (cos, sin)
    elif isinstance(image_rotary_emb, dict):
        if "cos" in image_rotary_emb and "sin" in image_rotary_emb:
            axes.append((image_rotary_emb["cos"], image_rotary_emb["sin"]))
    return axes


def apply_flux_rope_to_q(q_bh: torch.Tensor, image_rotary_emb, max_rotary_fraction: float = 1.0) -> torch.Tensor:
    """
    在旁路的 head-batched q 上应用 2D RoPE。
      q_bh: (B*H, N_img, Dh)
      image_rotary_emb: [(cos_x, sin_x), (cos_y, sin_y)] or [(cos_x,sin_x)] etc.
      max_rotary_fraction: 仅对前 max_rotary_fraction*Dh 的维度应用（默认 100%）
    """
    axes = _normalize_rope_emb(image_rotary_emb)
    if not axes:
        return q_bh

    Dh = q_bh.shape[-1]
    rotary_budget = int(Dh * float(max_rotary_fraction))
    rotary_budget -= (rotary_budget % 2)  # 偶数

    # 逐轴分配旋转维度：优先均分；每轴最多 2 * cos.size(-1)
    remains = rotary_budget
    start = 0
    for (cos, sin) in axes:
        per_axis_max = int(cos.shape[-1]) * 2
        use = min(per_axis_max, remains)
        use -= (use % 2)
        if use > 0:
            q_bh = _apply_rope_slice(q_bh, cos, sin, start=start, length=use)
            start += use
            remains -= use
        if remains <= 0:
            break
    return q_bh


# ===================== proxy processor =====================
def _layer_depth_from_name(name: str) -> int:
    for token in name.replace("/", ".").split("."):
        if token.isdigit():
            return int(token)
    return 0


class ProxyCaptureProcessor:
    """
    - 原样调用原处理器（返回形态不变）
    - 仅在 cross-attn（有 encoder_hidden_states）旁路计算 image->text：
        h_img = hidden_states[:, :N_img]
        q = to_q(h_img)         -> head_to_batch_dim -> **RoPE(q, image_rotary_emb)**
        k = to_k(encoder_hidden_states)  (默认不对 k 做 RoPE，贴近 FLUX cross-attn)
        probs = get_attention_scores(q, k, ...)
    - 每层保存一条 (B*H, N_img, N_txt) 概率矩阵
    """
    def __init__(self, layer_name: str, original_processor, store: Dict[str, List[torch.Tensor]],
                 capture_limit: int = 1, derived_getter=None, rope_fraction: float = 1.0):
        self.layer_name = layer_name
        self.orig = original_processor
        self.store = store
        self.capture_limit = capture_limit
        self._saved = 0
        self.derived_getter = derived_getter
        self.rope_fraction = rope_fraction

        try:
            self._orig_call = self.orig.__call__ if hasattr(self.orig, "__call__") else self.orig
            self._sig = inspect.signature(self._orig_call)
            self._accepts = set(self._sig.parameters.keys())
        except Exception:
            self._orig_call = self.orig
            self._sig = None
            self._accepts = set()

    def __call__(self,
                 attn: Attention,
                 hidden_states,
                 encoder_hidden_states=None,
                 attention_mask=None,
                 image_rotary_emb=None,
                 cross_attention_kwargs=None,
                 **kwargs):
        # 1) 原样调用原处理器（签名自适应+回退）
        call_kwargs = dict(
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
        if isinstance(cross_attention_kwargs, dict):
            call_kwargs.update(cross_attention_kwargs)
        call_kwargs.update(kwargs)

        if self._sig is not None:
            filtered = {k: v for k, v in call_kwargs.items() if k in self._accepts}
        else:
            filtered = call_kwargs

        try:
            out = self._orig_call(attn, hidden_states, **filtered)
        except TypeError:
            try:
                filtered2 = {k: v for k, v in filtered.items() if k not in ("image_rotary_emb", "cross_attention_kwargs")}
                out = self._orig_call(attn, hidden_states, **filtered2)
            except TypeError:
                base = {}
                if "encoder_hidden_states" in self._accepts:
                    base["encoder_hidden_states"] = encoder_hidden_states
                if "attention_mask" in self._accepts:
                    base["attention_mask"] = attention_mask
                try:
                    out = self._orig_call(attn, hidden_states, **base)
                except TypeError:
                    out = self._orig_call(attn, hidden_states)

        # 2) 旁路 capture（仅 cross-attn）
        try:
            if self._saved < self.capture_limit and encoder_hidden_states is not None:
                B, N_all, _ = hidden_states.shape
                N_txt = int(encoder_hidden_states.shape[1])
                N_img = int(N_all - N_txt)
                if N_img <= 0 or N_txt <= 0:
                    return out

                h_img = hidden_states[:, :N_img, :]
                q = attn.to_q(h_img)                 # (B, N_img, D)
                k = attn.to_k(encoder_hidden_states) # (B, N_txt, D)

                # head-batch
                q = attn.head_to_batch_dim(q)        # (B*H, N_img, Dh)
                k = attn.head_to_batch_dim(k)        # (B*H, N_txt, Dh)

                # >>> RoPE 对齐：仅对 q 应用 2D RoPE <<<
                q = apply_flux_rope_to_q(q, image_rotary_emb, max_rotary_fraction=self.rope_fraction)

                am = attn.prepare_attention_mask(attention_mask, N_img, B)
                probs = attn.get_attention_scores(q, k, am)  # (B*H, N_img, N_txt)
                self.store.setdefault(self.layer_name, []).append(probs.detach().to(torch.float32).cpu())
                self._saved += 1
        except Exception:
            pass

        return out


def attach_proxy_processors(pipe, probe_store, capture_limit_per_layer: int = 1, rope_fraction: float = 1.0) -> Dict[str, List[torch.Tensor]]:
    store: Dict[str, List[torch.Tensor]] = {}
    def getter():
        return probe_store.derived

    cnt = 0
    for name, module in pipe.transformer.named_modules():
        if isinstance(module, Attention) and hasattr(module, "processor"):
            try:
                orig = module.processor
                module.set_processor(
                    ProxyCaptureProcessor(name, orig, store,
                                          capture_limit=capture_limit_per_layer,
                                          derived_getter=getter,
                                          rope_fraction=rope_fraction)
                )
                cnt += 1
            except Exception:
                pass
    print(f"[OK] Installed ProxyCaptureProcessor on {cnt} Attention modules.")
    return store


# ===================== aggregate & save =====================
def _layer_depth_from_name(name: str) -> int:
    # already defined above; keep unique
    for token in name.replace("/", ".").split("."):
        if token.isdigit():
            return int(token)
    return 0


def pick_last_layer_names(attn_store: Dict[str, List[torch.Tensor]], last_n: int) -> List[str]:
    names = sorted(attn_store.keys(), key=_layer_depth_from_name)
    if last_n <= 0 or last_n >= len(names):
        return names
    return names[-last_n:]


def aggregate_flux_attn_to_heatmap(
    attn_store: Dict[str, List[torch.Tensor]],
    n_img: int,
    n_txt: int,
    img_grid: Tuple[int, int],
    use_names: Optional[List[str]] = None,
    assume_order_img_then_txt: bool = True,  # 仅在个别层返回 (N_all,N_all) 时备用
    object_token_indices: Optional[List[int]] = None,
) -> np.ndarray:
    Hs, Ws = img_grid
    cols: List[np.ndarray] = []
    names_iter = (use_names if use_names is not None else attn_store.keys())

    for lname in names_iter:
        for t in attn_store.get(lname, []):
            bh, Lq, Lk = t.shape
            Tmean = t.mean(dim=0)  # (Lq, Lk)

            if Lk == n_txt:
                sub = Tmean[:n_img, :n_txt]    # 主路径：我们旁路重算得到的 image->text
            elif Lk == (n_img + n_txt):
                if assume_order_img_then_txt:
                    sub = Tmean[:n_img, n_img:n_img + n_txt]
                else:
                    sub = Tmean[n_txt:n_txt + n_img, :n_txt]
            else:
                continue

            if sub.numel() == 0:
                continue

            if object_token_indices:
                sel = [sub[:, j] for j in object_token_indices if 0 <= j < sub.shape[-1]]
                vec = torch.stack(sel, dim=-1).mean(dim=-1) if sel else sub.mean(dim=-1)
            else:
                vec = sub.mean(dim=-1)

            cols.append(vec.cpu().numpy())

    if not cols:
        raise RuntimeError("No valid cross-attention slices collected (after RoPE).")

    vec = np.mean(np.stack(cols, axis=0), axis=0)  # (N_img,)
    vmin, vmax = float(np.min(vec)), float(np.max(vec))
    vec = (vec - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(vec)
    return vec.reshape(Hs, Ws)


def save_heatmap_and_masks(heat_lr: np.ndarray, save_dir: str, prefix: str,
                           fixed_thresholds: List[float], top_pcts: List[float],
                           height: Optional[int] = None, width: Optional[int] = None,
                           smooth: int = 0):
    os.makedirs(save_dir, exist_ok=True)

    # low-res
    plt.figure(figsize=(5, 5))
    plt.imshow(heat_lr, cmap="viridis", interpolation="nearest")
    plt.axis("off"); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}heatmap_lowres.png"), dpi=220)
    plt.close()

    if height is not None and width is not None:
        heat = upsample_heatmap(heat_lr, height, width, smooth=smooth)
        plt.figure(figsize=(5, 5))
        plt.imshow(heat, cmap="viridis", interpolation="nearest")
        plt.axis("off"); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}heatmap.png"), dpi=220)
        plt.close()
    else:
        heat = heat_lr

    for thr in fixed_thresholds:
        m = (heat >= thr).astype(np.uint8) * 255
        plt.figure(figsize=(5, 5))
        plt.imshow(m, cmap="gray", vmin=0, vmax=255)
        plt.axis("off"); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}mask_thr_{thr:.2f}.png"), dpi=220)
        plt.close()

    flat = heat.flatten()
    for p in top_pcts:
        cut = np.quantile(flat, 1.0 - float(p) / 100.0)
        m = (heat >= cut).astype(np.uint8) * 255
        plt.figure(figsize=(5, 5))
        plt.imshow(m, cmap="gray", vmin=0, vmax=255)
        plt.axis("off"); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}mask_top_{int(p)}p.png"), dpi=220)
        plt.close()


# ===================== runner =====================
def run_flux_attention_test(
    model_id: str,
    prompt: str,
    object_word: str,
    steps: int,
    height: int,
    width: int,
    device: str,
    dtype: str,
    save_dir: str,
    fixed_thresholds: List[float],
    top_pcts: List[float],
    capture_limit_per_layer: int = 1,
    last_n: int = 4,
    upsample_to_input: bool = True,
    smooth: int = 0,
    rope_fraction: float = 1.0,
    seed: Optional[int] = 42,
) -> None:
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))
    else:
        g = None

    dt_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dt_map.get(dtype, torch.bfloat16)

    version_check("0.32.0")
    pipe = build_flux_pipeline(model_id=model_id, device=device, dtype=torch_dtype)

    tok_info = show_tokenization(pipe, prompt)
    object_idxs: List[int] = []
    if tok_info.get("clip"):
        object_idxs = find_object_subtokens(tok_info["clip"]["tokens_valid"], object_word)
        print(f"[clip] indices containing '{object_word}': {object_idxs}")
    elif tok_info.get("t5"):
        object_idxs = find_object_subtokens(tok_info["t5"]["tokens_valid"], object_word)
        print(f"[t5]   indices containing '{object_word}': {object_idxs}")

    probe_store = attach_probe_hook(pipe)
    attn_store = attach_proxy_processors(
        pipe, probe_store,
        capture_limit_per_layer=capture_limit_per_layer,
        rope_fraction=rope_fraction
    )

    # one pass
    gen_kwargs = dict(prompt=[prompt], num_inference_steps=steps,
                      height=height, width=width, guidance_scale=3.5,
                      generator=g)
    if "schnell" in model_id.lower():
        gen_kwargs["guidance_scale"] = 0.0
        gen_kwargs["max_sequence_length"] = 256

    _hr("Generate (one pass)")
    out = pipe(**gen_kwargs)
    os.makedirs(save_dir, exist_ok=True)
    try:
        out.images[0].save(os.path.join(save_dir, "flux_sample.png"))
    except Exception as e:
        print("[w] Could not save generated image:", e)

    if not probe_store.derived:
        raise RuntimeError("Could not derive (N_img, N_txt).")
    n_img = int(probe_store.derived["N_img"]); n_txt = int(probe_store.derived["N_txt"])
    grid = robust_grid_from_img_ids(probe_store.first_img_ids, n_img) if probe_store.first_img_ids is not None else best_factor_pair(n_img)
    print(f"[i] Using grid H'xW' = {grid}  (n_img={n_img}, n_txt={n_txt})")

    names = pick_last_layer_names(attn_store, last_n=last_n)
    print(f"[i] Aggregating last {min(last_n, len(names))} layers:", names[-min(last_n, len(names)):])

    for order_tag, assume_img_txt in [("imgtxt", True), ("txtimg", False)]:
        try:
            heat_lr = aggregate_flux_attn_to_heatmap(
                attn_store=attn_store,
                n_img=n_img, n_txt=n_txt,
                img_grid=grid,
                use_names=names,
                assume_order_img_then_txt=assume_img_txt,
                object_token_indices=object_idxs,
            )
            save_heatmap_and_masks(
                heat_lr=heat_lr,
                save_dir=save_dir,
                prefix=f"flux_{order_tag}_",
                fixed_thresholds=fixed_thresholds,
                top_pcts=top_pcts,
                height=(height if upsample_to_input else None),
                width=(width  if upsample_to_input else None),
                smooth=smooth,
            )
        except Exception as e:
            print(f"[w] {order_tag} aggregation failed:", e)

    import json
    meta = dict(model_id=model_id, prompt=prompt, object_word=object_word,
                n_img=n_img, n_txt=n_txt, grid=grid, last_n=last_n,
                object_token_indices=object_idxs, fixed_thresholds=fixed_thresholds,
                top_pcts=top_pcts, upsample_to_input=upsample_to_input, smooth=smooth,
                rope_fraction=rope_fraction, seed=seed)
    with open(os.path.join(save_dir, "flux_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved heatmap/masks/meta to: {save_dir}")


def main():
    p = argparse.ArgumentParser(description="FLUX attention test (RoPE-aligned): capture & save heatmap/masks")
    p.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--prompt", type=str, default="a photo of a panda on a bench, city background")
    p.add_argument("--object_word", type=str, default="panda")
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--save_dir", type=str, default="/root/autodl-tmp/EditSplat/utils/atten_test")
    p.add_argument("--fixed_thresholds", type=float, nargs="+", default=[0.30, 0.50, 0.70])
    p.add_argument("--top_pcts", type=float, nargs="+", default=[5, 10, 20])
    p.add_argument("--last_n", type=int, default=4, help="use last N layers when aggregating attention")
    p.add_argument("--upsample_to_input", action="store_true")
    p.add_argument("--smooth", type=int, default=0, help="0=off,1=avgpool 3x3 before upsample")
    p.add_argument("--rope_fraction", type=float, default=1.0, help="0~1: portion of head_dim to apply RoPE to q")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    run_flux_attention_test(
        a.model_id, a.prompt, a.object_word, a.steps, a.height, a.width,
        a.device, a.dtype, a.save_dir, list(a.fixed_thresholds), list(a.top_pcts),
        capture_limit_per_layer=1, last_n=a.last_n,
        upsample_to_input=a.upsample_to_input, smooth=a.smooth,
        rope_fraction=a.rope_fraction, seed=a.seed
    )


if __name__ == "__main__":
    main()
