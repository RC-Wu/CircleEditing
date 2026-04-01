#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import importlib
import importlib.util
import json
import os
import re
import shutil
import sys
import traceback
import types
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
LEGACY_WRAPPER = (
    ROOT / "sandboxes" / "20260319_aris_ttt3r_flowedit_45" / "scripts" / "run_sd35_ttt3r_proximal_wrapper.py"
)

CACHE_ROOT = Path("/dev_vepfs/rc_wu/cache")
DEFAULT_HF_HOME = CACHE_ROOT / "hf_home_dev02"
DEFAULT_TORCH_HOME = CACHE_ROOT / "torch"
DEFAULT_XDG_CACHE_HOME = CACHE_ROOT / "xdg"

# Keep all mutable caches under /dev_vepfs/rc_wu and force real LangSAM path.
os.environ.setdefault("FLOWEDIT_REAL_LANGSAM", "1")
os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
os.environ.setdefault("TORCH_HOME", str(DEFAULT_TORCH_HOME))
os.environ.setdefault("XDG_CACHE_HOME", str(DEFAULT_XDG_CACHE_HOME))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# Force SAM3 to prefer CUDA only when the caller has not requested a specific device.
os.environ.setdefault("EDITSPLAT_SAM3_DEVICE", "cuda")
for _cache_dir in (
    Path(os.environ["HF_HOME"]),
    Path(os.environ["HUGGINGFACE_HUB_CACHE"]),
    Path(os.environ["TORCH_HOME"]),
    Path(os.environ["XDG_CACHE_HOME"]),
):
    _cache_dir.mkdir(parents=True, exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.sam3_support import resolve_sam3_backend_request


_MASK_BACKEND_INFO = {
    "requested": None,
    "effective": None,
    "detail": None,
    "sam3_mask_stats": [],
}
_SAM3_DEBUG_COUNTER = 0
_CURRENT_RUNTIME = None


def _patch_sam3_decoder_cache_device() -> None:
    """Ensure SAM3 decoder caches always match the reference box device (CPU-safe)."""

    try:
        from sam3.model.decoder import TransformerDecoder  # type: ignore
    except Exception as exc:  # pragma: no cover - defensive import guard
        print(f"[SAM3] decoder cache patch skipped: import error: {exc}")
        return

    if getattr(TransformerDecoder, "_editsplat_cache_device_patch", False):
        return

    orig_get_rpb_matrix = TransformerDecoder._get_rpb_matrix

    def _move_cache_pair(pair, device):
        if pair is None:
            return None
        coords_h, coords_w = pair
        need_move = coords_h.device != device or coords_w.device != device
        if need_move:
            coords_h = coords_h.to(device)
            coords_w = coords_w.to(device)
        return coords_h, coords_w

    def _patched_get_rpb_matrix(self, reference_boxes, feat_size):
        device = reference_boxes.device
        if getattr(self, "compilable_cord_cache", None) is not None:
            moved = _move_cache_pair(self.compilable_cord_cache, device)
            if moved is not None:
                self.compilable_cord_cache = moved

        coord_cache = getattr(self, "coord_cache", None)
        if isinstance(coord_cache, dict) and feat_size in coord_cache:
            moved = _move_cache_pair(coord_cache.get(feat_size), device)
            if moved is not None:
                coord_cache[feat_size] = moved

        return orig_get_rpb_matrix(self, reference_boxes, feat_size)

    TransformerDecoder._get_rpb_matrix = _patched_get_rpb_matrix  # type: ignore[attr-defined]
    TransformerDecoder._editsplat_cache_device_patch = True  # type: ignore[attr-defined]
    print("[SAM3] patched TransformerDecoder._get_rpb_matrix for cache/device consistency.")


def _cuda_autocast_disabled():
    if not torch.cuda.is_available():
        return contextlib.nullcontext()
    try:
        from torch.cuda.amp import autocast as cuda_autocast
    except Exception:
        return contextlib.nullcontext()
    return cuda_autocast(enabled=False)


def _get_sam3_device_candidates() -> List[str]:
    """Return preferred SAM3 device order with a safe CPU fallback."""

    raw = os.environ.get("EDITSPLAT_SAM3_DEVICE") or os.environ.get("SAM3_DEVICE")
    devices: List[str] = []
    if raw:
        for chunk in raw.split(","):
            token = chunk.strip().lower()
            if not token or token in {"auto", "default"}:
                continue
            if token in {"cuda", "cpu"} and token not in devices:
                devices.append(token)
        if devices:
            if "cpu" not in devices:
                devices.append("cpu")
            return devices

    fallback: List[str] = []
    if torch.cuda.is_available():
        fallback.append("cuda")
    fallback.append("cpu")
    return fallback


def _seed_huggingface_auth_token() -> Optional[str]:
    """Ensure Hugging Face auth survives the sandbox-specific HF_HOME override."""

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token

    candidate_paths = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidate_paths.append(Path(hf_home) / "token")
    home = Path.home()
    candidate_paths.extend(
        [
            home / ".huggingface" / "token",
            home / ".cache" / "huggingface" / "token",
        ]
    )

    for token_path in candidate_paths:
        try:
            raw = token_path.read_text().strip()
        except FileNotFoundError:
            continue
        except OSError as exc:
            print(f"[SAM3] unable to read {token_path}: {exc}")
            continue
        if not raw:
            continue
        first_line = raw.splitlines()[0].strip()
        if "=" in first_line and not first_line.startswith("hf_"):
            first_line = first_line.split("=", 1)[1].strip()
        first_line = first_line.strip('\"').strip("'")
        if not first_line:
            continue
        os.environ.setdefault("HF_TOKEN", first_line)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", first_line)
        print(f"[SAM3] seeded HuggingFace token from {token_path}")
        return first_line
    return None


def _stabilize_ttt3r_attention() -> None:
    if not torch.cuda.is_available():
        return
    toggles = []
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        toggles.append("flash=off")
    except Exception:
        pass
    try:
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        toggles.append("mem_efficient=off")
    except Exception:
        pass
    try:
        torch.backends.cuda.enable_math_sdp(True)
        toggles.append("math=on")
    except Exception:
        pass
    if toggles:
        print(f"[TTT3R] SDPA stability mode: {' '.join(toggles)}")


def _patch_ttt3r_runtime_device() -> None:
    runtime_cls = getattr(_LEGACY_WRAPPER, "TTT3RRuntime", None)
    if runtime_cls is None:
        return
    orig_init_model = getattr(runtime_cls, "init_model", None)

    def _patch_cpu_rope_dtype_once() -> None:
        def _apply_rope(module, tensor, pos):
            if module.rope is None or pos is None:
                return tensor
            tensor_type = tensor.dtype
            if tensor.device.type == "cpu":
                work = tensor.float()
                with torch.autocast(device_type="cpu", enabled=False):
                    work = module.rope(work, pos)
            else:
                work = tensor.to(torch.float16)
                with torch.autocast(device_type="cuda", enabled=False):
                    work = module.rope(work, pos)
            return work.to(tensor_type)

        def _patch_attention_class(module_name: str) -> bool:
            try:
                mod = importlib.import_module(module_name)
            except Exception:
                return False
            attn_cls = getattr(mod, "Attention", None)
            if attn_cls is None or getattr(attn_cls, "_editsplat_cpu_rope_patch", False):
                return False

            def _forward(self, x, xpos, return_attn=False):
                B, N, C = x.shape
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)
                q, k, v = [qkv[:, :, i] for i in range(3)]
                q = _apply_rope(self, q, xpos)
                k = _apply_rope(self, k, xpos)

                if return_attn:
                    attn = (q @ k.transpose(-2, -1)) * self.scale
                    attn_before_softmax = attn.detach().clone()
                    attn = attn.softmax(dim=-1)
                    attn = self.attn_drop(attn)
                    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                    x = self.proj(x)
                    x = self.proj_drop(x)
                    return x, attn_before_softmax

                x = (
                    torch.nn.functional.scaled_dot_product_attention(
                        query=q, key=k, value=v, dropout_p=self.attn_drop.p, scale=self.scale
                    )
                    .transpose(1, 2)
                    .reshape(B, N, C)
                )
                x = self.proj(x)
                x = self.proj_drop(x)
                return x

            attn_cls.forward = _forward
            attn_cls._editsplat_cpu_rope_patch = True
            print(f"[TTT3R] patched {module_name}.Attention.forward for CPU-safe RoPE with return_attn support.")
            return True

        def _patch_cross_attention_class(module_name: str) -> bool:
            try:
                mod = importlib.import_module(module_name)
            except Exception:
                return False
            attn_cls = getattr(mod, "CrossAttention", None)
            if attn_cls is None or getattr(attn_cls, "_editsplat_cpu_rope_patch", False):
                return False

            def _forward(self, query, key, value, qpos, kpos, return_attn=False):
                B, Nq, C = query.shape
                Nk = key.shape[1]
                Nv = value.shape[1]
                q = self.projq(query).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
                k = self.projk(key).reshape(B, Nk, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
                v = self.projv(value).reshape(B, Nv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
                q = _apply_rope(self, q, qpos)
                k = _apply_rope(self, k, kpos)

                if return_attn:
                    attn = (q @ k.transpose(-2, -1)) * self.scale
                    attn_before_softmax = attn.detach().clone()
                    attn = attn.softmax(dim=-1)
                    attn = self.attn_drop(attn)
                    x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
                    x = self.proj(x)
                    x = self.proj_drop(x)
                    return x, attn_before_softmax

                x = (
                    torch.nn.functional.scaled_dot_product_attention(
                        query=q, key=k, value=v, dropout_p=self.attn_drop.p, scale=self.scale
                    )
                    .transpose(1, 2)
                    .reshape(B, Nq, C)
                )
                x = self.proj(x)
                x = self.proj_drop(x)
                return x

            attn_cls.forward = _forward
            attn_cls._editsplat_cpu_rope_patch = True
            print(f"[TTT3R] patched {module_name}.CrossAttention.forward for CPU-safe RoPE.")
            return True

        patched_any = False
        for module_name in ("dust3r.blocks", "croco.models.blocks", "models.blocks"):
            patched_any = _patch_attention_class(module_name) or patched_any
            patched_any = _patch_cross_attention_class(module_name) or patched_any
        if not patched_any:
            print("[WARN] TTT3R CPU RoPE patch did not find a target Attention/CrossAttention class.")

    def _runtime_init(self, cfg):
        self.cfg = cfg
        self.model = None
        self.inference_fn = None
        force_cpu = os.environ.get("EDITSPLAT_TTT3R_FORCE_CPU", "0") == "1" or int(getattr(cfg, "ttt3r_gpu", 0)) < 0
        if force_cpu or not torch.cuda.is_available():
            self.device = torch.device("cpu")
            print(
                f"[TTT3R] forcing CPU runtime "
                f"(gpu_flag={getattr(cfg, 'ttt3r_gpu', None)} force_cpu_env={os.environ.get('EDITSPLAT_TTT3R_FORCE_CPU', '0')})."
            )
        else:
            self.device = torch.device(f"cuda:{cfg.ttt3r_gpu}")
        self.camera_list = None
        self.current_idx = 0
        self.dump_root = None
        self.stage_counts = {"initial_edit": 0, "mfg_edit": 0}
        self.initial_edit_cache = {}
        self.source_image_cache = {}
        self.fit_mask_cache = {}
        self.fit_mask_dumped = set()
        self.fit_view_score_cache = {}
        self.fit_view_topk_cache = {}
        self.fit_view_dumped_keys = set()
        self.support_mask_cache = {}
        self.support_mask_meta = {}

        global _CURRENT_RUNTIME
        _CURRENT_RUNTIME = self

    runtime_cls.__init__ = _runtime_init

    if orig_init_model is not None:
        def _init_model(self):
            orig_init_model(self)
            _patch_cpu_rope_dtype_once()
            if getattr(self, "model", None) is not None and getattr(self, "device", torch.device("cpu")).type == "cpu":
                self.model = self.model.float()
                print("[TTT3R] cast CPU runtime model to float32.")

        runtime_cls.init_model = _init_model


def _set_mask_backend_info(requested: str, effective: str, detail: Optional[str] = None) -> None:
    _MASK_BACKEND_INFO["requested"] = requested
    _MASK_BACKEND_INFO["effective"] = effective
    _MASK_BACKEND_INFO["detail"] = detail
    os.environ["EDITSPLAT_MASK_BACKEND_REQUESTED"] = requested
    os.environ["EDITSPLAT_MASK_BACKEND_EFFECTIVE"] = effective
    if detail:
        os.environ["EDITSPLAT_MASK_BACKEND_DETAIL"] = detail


def _append_mask_backend_detail(extra: str) -> None:
    requested = str(_MASK_BACKEND_INFO.get("requested") or "langsam")
    effective = str(_MASK_BACKEND_INFO.get("effective") or "langsam:cpu")
    prefix = _MASK_BACKEND_INFO.get("detail")
    detail = f"{prefix}; {extra}" if prefix else extra
    _set_mask_backend_info(requested=requested, effective=effective, detail=detail)


def _record_sam3_mask_stats(mask_t: torch.Tensor, prompt: str, threshold: float) -> None:
    stats_list = _MASK_BACKEND_INFO.setdefault("sam3_mask_stats", [])
    mask_cpu = mask_t.detach().float().cpu()
    mean_val = float(mask_cpu.mean().item())
    min_val = float(mask_cpu.min().item())
    max_val = float(mask_cpu.max().item())
    entries = [
        {
            "prompt": prompt,
            "threshold": float(threshold),
            "mean": mean_val,
            "min": min_val,
            "max": max_val,
        }
    ]
    stats_list.extend(entries)
    print(
        "[SAM3] mask stats "
        f"prompt={prompt or 'unknown'} "
        f"thr={threshold:.3f} mean={mean_val:.3f} "
        f"min={min_val:.3f} max={max_val:.3f}"
    )


def _extract_flag_value(flag: str) -> Optional[str]:
    argv = sys.argv[1:]
    for idx, arg in enumerate(argv):
        if arg == flag and idx + 1 < len(argv):
            return argv[idx + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    if flag == "--model_path":
        for idx, arg in enumerate(argv):
            if arg == "-m" and idx + 1 < len(argv):
                return argv[idx + 1]
    return None


def _write_mask_backend_metadata(model_path_arg: Optional[str]) -> None:
    if not model_path_arg:
        return
    model_path = Path(model_path_arg)
    model_path.mkdir(parents=True, exist_ok=True)
    info_path = model_path / "mask_backend_info.json"
    info_path.write_text(json.dumps(_MASK_BACKEND_INFO, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    meta_path = model_path / "ttt3r_proximal_wrapper_meta.json"
    if meta_path.is_file():
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        payload["wrapper"] = "run_sd35_ttt3r_sam3_wrapper.py"
        payload["mask_backend"] = dict(_MASK_BACKEND_INFO)
        meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _tensor_to_pil01(img: torch.Tensor) -> Image.Image:
    if img.ndim == 3:
        img = img.unsqueeze(0)
    x = img[0].detach().float()
    if x.min() < 0.0:
        x = (x + 1.0) * 0.5
    x = x.clamp(0.0, 1.0).cpu()
    arr = (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def _pil_to_tensor01(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _normalize_mask_tensor(mask: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    if mask.ndim == 0:
        mask = mask.view(1, 1, 1)
    if mask.ndim == 4:
        if mask.shape[1] == 1:
            mask = mask[:, 0]
        elif mask.shape[0] == 1:
            mask = mask[0]
        else:
            mask = mask[:, 0]
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 3 and mask.shape[0] != 1:
        mask = mask.amax(dim=0, keepdim=True)
    if tuple(mask.shape[-2:]) != image_size:
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0).float(),
            size=image_size,
            mode="nearest",
        ).squeeze(0)
    mask = mask.float().clamp(0.0, 1.0)
    if os.environ.get("EDITSPLAT_BINARIZE_SUPPORT_MASK", "1").strip().lower() in {"1", "true", "yes", "on"}:
        threshold = float(os.environ.get("EDITSPLAT_SUPPORT_MASK_THRESHOLD", "0.0"))
        return (mask > threshold).float()
    return mask


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _normalize_support_role(role: Optional[str]) -> str:
    token = str(role or os.environ.get("EDITSPLAT_MASK_ROLE", "default")).strip().lower()
    return token or "default"


def _get_support_payload(runtime, idx: int, role: Optional[str]) -> Optional[dict]:
    cache = getattr(runtime, "support_mask_cache", None)
    if not isinstance(cache, dict):
        return None
    payload = cache.get(int(idx))
    if isinstance(payload, torch.Tensor):
        return {"mask": payload.detach().float().cpu(), "soft_mask": payload.detach().float().cpu(), "role": "legacy"}
    if not isinstance(payload, dict):
        return None
    roles = payload.get("roles")
    if isinstance(roles, dict):
        normalized_role = _normalize_support_role(role)
        chosen = roles.get(normalized_role)
        if isinstance(chosen, dict):
            return chosen
        for fallback in ("gt_view", "reproject", "default", "last"):
            chosen = roles.get(fallback)
            if isinstance(chosen, dict):
                return chosen
    last_payload = payload.get("last")
    if isinstance(last_payload, dict):
        return last_payload
    return None


def _payload_mask(payload: Optional[dict], image_size: Optional[Tuple[int, int]] = None) -> Optional[torch.Tensor]:
    if not isinstance(payload, dict):
        return None
    mask = payload.get("soft_mask")
    if not isinstance(mask, torch.Tensor):
        mask = payload.get("mask")
    if not isinstance(mask, torch.Tensor):
        return None
    mask = mask.detach().float().cpu()
    if image_size is not None:
        mask = _normalize_mask_tensor(mask, image_size=image_size)
    return mask


class _FullImageMaskStub:
    backend_name = "stub"

    def predict(self, image_pil, text_prompt):
        del text_prompt
        w, h = image_pil.size
        mask = torch.ones((1, h, w), dtype=torch.float32)
        return mask, None, None, None


class _Sam3MaskAdapter:
    backend_name = "sam3"

    def __init__(self):
        print(f"[SAM3] adapter init env_device={os.environ.get('EDITSPLAT_SAM3_DEVICE')}")
        self._processor = None
        self._model = None
        self._init_error: Optional[Exception] = None
        self._base_confidence_threshold: float = 0.18
        self._last_used_threshold: float = self._base_confidence_threshold
        fallback_raw = os.environ.get("EDITSPLAT_SAM3_CONFIDENCE_FALLBACKS", "0.12,0.08,0.04,0.0")
        fallback_values: List[float] = []
        for chunk in fallback_raw.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                value = float(chunk)
            except ValueError:
                continue
            fallback_values.append(value)
        # Deduplicate while preserving descending order so we drop gradually.
        seen = set()
        filtered_fallbacks: List[float] = []
        for value in sorted(fallback_values, reverse=True):
            if value in seen:
                continue
            seen.add(value)
            filtered_fallbacks.append(value)
        self._confidence_fallbacks = filtered_fallbacks
        if self._confidence_fallbacks:
            print(f"[SAM3] confidence fallback ladder={self._confidence_fallbacks}")

        token = _seed_huggingface_auth_token()
        if token is None:
            print("[SAM3] no HuggingFace token detected; attempting unauthenticated download.")

        default_tensor = torch.tensor(0.0)
        default_is_cuda = default_tensor.is_cuda
        del default_tensor
        restored_global_default = False
        if default_is_cuda:
            torch.set_default_tensor_type(torch.FloatTensor)
            restored_global_default = True

        # Try to import SAM3 - if it doesn't exist, we'll handle the error gracefully
        try:
            # First check if sam3 module exists
            import sam3
            print("[SAM3] Found SAM3 module, attempting to load...")
            _patch_sam3_decoder_cache_device()

            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model

            device_candidates = _get_sam3_device_candidates()
            print(f"[SAM3] device candidates={device_candidates}")
            last_error: Optional[Exception] = None
            for device in device_candidates:
                restore_after_device = False
                if device.startswith("cuda"):
                    if restored_global_default:
                        torch.set_default_tensor_type(torch.cuda.FloatTensor)
                        restore_after_device = True
                elif default_is_cuda:
                    # Ensure CPU loads never inherit a stale CUDA default.
                    torch.set_default_tensor_type(torch.FloatTensor)

                try:
                    print(f"[SAM3] loading official backend on device={device} ...")
                    self._model = build_sam3_image_model(device=device)
                    self._processor = Sam3Processor(self._model, device=device)
                    _set_mask_backend_info(
                        requested="sam3",
                        effective="sam3",
                        detail=f"sam3_device={device}",
                    )
                    print(f"[SAM3] loaded official image segmentation backend on {device}.")
                    last_error = None
                    break
                except Exception as model_load_exc:
                    last_error = model_load_exc
                    print(f"[WARN] SAM3 model loading failed on device={device}: {model_load_exc}")
                    if device != device_candidates[-1]:
                        print("[SAM3] attempting next device candidate...")
                finally:
                    if restore_after_device and default_is_cuda:
                        torch.set_default_tensor_type(torch.FloatTensor)

            if self._processor is None:
                if last_error is not None:
                    raise last_error
                raise RuntimeError("SAM3 backend unavailable after trying all devices.")
            conf_threshold = float(os.environ.get("EDITSPLAT_SAM3_CONFIDENCE", "0.18"))
            self._base_confidence_threshold = conf_threshold
            self._processor.set_confidence_threshold(conf_threshold)
            print(f"[SAM3] confidence threshold set to {conf_threshold:.3f}")

        except ImportError as exc:
            # SAM3 is not installed, provide clear error and fallback
            self._init_error = exc
            print(f"[INFO] SAM3 not available in environment: {exc}")
            print("[INFO] This is expected in environments without SAM3 installed.")
            raise exc  # This will trigger the fallback logic
        except Exception as exc:
            # Any other error during SAM3 initialization
            self._init_error = exc
            print(f"[WARN] SAM3 init failed with error: {exc}")
            raise exc  # This will trigger the fallback logic
        finally:
            if restored_global_default and default_is_cuda:
                torch.set_default_tensor_type(torch.cuda.FloatTensor)

    def _extract_mask_from_state(self, state: dict, image_size: Tuple[int, int]):
        masks = state.get("masks")
        if masks is None:
            return None
        mask_t = torch.as_tensor(masks).detach().float().cpu()
        if mask_t.numel() == 0:
            return None
        boxes = state.get("boxes")
        scores = state.get("scores")
        score_t: Optional[torch.Tensor] = None
        if scores is not None:
            score_t = torch.as_tensor(scores).detach().float().cpu().flatten()
            if score_t.numel() > 0:
                best_idx = int(torch.argmax(score_t).item())
                if mask_t.ndim >= 4:
                    mask_t = mask_t[best_idx]
                elif mask_t.ndim == 3 and mask_t.shape[0] > 1:
                    mask_t = mask_t[best_idx : best_idx + 1]
                score_t = score_t[best_idx : best_idx + 1]
        mask_t = _normalize_mask_tensor(mask_t, image_size=image_size)
        return mask_t, boxes, score_t

    def predict(self, image_pil, text_prompt):
        if self._processor is None:
            raise RuntimeError(f"SAM3 backend unavailable: {self._init_error}")
        default_tensor = torch.tensor(0.0)
        default_tensor_device = default_tensor.device
        default_is_cuda = default_tensor.is_cuda
        print(
            f"[SAM3] predict start prompt={text_prompt} "
            f"default_device={default_tensor_device} "
            f"autocast_cuda_enabled={torch.is_autocast_enabled()}"
        )
        if default_is_cuda:
            torch.set_default_tensor_type(torch.FloatTensor)
        with contextlib.ExitStack() as stack:
            stack.enter_context(torch.inference_mode())
            stack.enter_context(_cuda_autocast_disabled())
            try:
                state = self._processor.set_image(image_pil)
                output = self._processor.set_text_prompt(state=state, prompt=text_prompt)
            finally:
                if default_is_cuda:
                    torch.set_default_tensor_type(torch.cuda.FloatTensor)
        image_size = image_pil.size[::-1]
        mask_pack = self._extract_mask_from_state(output, image_size=image_size)
        used_threshold = self._base_confidence_threshold
        if mask_pack is None and self._confidence_fallbacks:
            for fallback in sorted(self._confidence_fallbacks, reverse=True):
                if fallback >= used_threshold:
                    continue
                print(f"[SAM3] lowering confidence threshold to {fallback:.3f}")
                output = self._processor.set_confidence_threshold(fallback, state=output)
                mask_pack = self._extract_mask_from_state(output, image_size=image_size)
                if mask_pack is not None:
                    used_threshold = fallback
                    _append_mask_backend_detail(f"sam3_confidence_fallback={fallback:.3f}")
                    break
        if mask_pack is None:
            raise RuntimeError("SAM3 returned empty mask tensor")
        if used_threshold != self._base_confidence_threshold:
            self._processor.set_confidence_threshold(self._base_confidence_threshold)
        mask_t, boxes, scores = mask_pack
        self._last_used_threshold = used_threshold
        _record_sam3_mask_stats(mask_t, prompt=text_prompt, threshold=used_threshold)
        return mask_t, boxes, scores, output


def _load_legacy_wrapper():
    spec = importlib.util.spec_from_file_location("editsplat_legacy_ttt3r_wrapper", LEGACY_WRAPPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load legacy wrapper from {LEGACY_WRAPPER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_LEGACY_WRAPPER = _load_legacy_wrapper()
_LEGACY_LANGSAM_LOADER = _LEGACY_WRAPPER.ref._load_langsam  # type: ignore[attr-defined]
_patch_ttt3r_runtime_device()


def _patch_lpips_device_guard() -> None:
    build_lpips = getattr(_LEGACY_WRAPPER.ref, "_build_lpips_loss", None)
    if build_lpips is None or getattr(build_lpips, "_editsplat_lpips_device_guard_patch", False):
        return

    def _infer_module_device(module, fallback: torch.device) -> torch.device:
        for getter_name in ("parameters", "buffers"):
            getter = getattr(module, getter_name, None)
            if not callable(getter):
                continue
            try:
                for tensor in getter():
                    if isinstance(tensor, torch.Tensor):
                        return tensor.device
            except Exception:
                continue
        return fallback

    def _safe_build_lpips(device: torch.device):
        inner = build_lpips(device)

        class _SafeLPIPS:
            def __init__(self, wrapped):
                self._wrapped = wrapped
                self._warned_fallback = False

            def _fallback_l1(self, x_src: torch.Tensor, y_src: torch.Tensor, reason: str):
                if not self._warned_fallback:
                    print(
                        "[WARN] LPIPS unstable in sandbox wrapper; "
                        f"using per-step L1 fallback. reason={reason}"
                    )
                    self._warned_fallback = True
                x_l1 = x_src.float()
                y_l1 = y_src.float()
                if x_l1.ndim == 3:
                    x_l1 = x_l1.unsqueeze(0)
                if y_l1.ndim == 3:
                    y_l1 = y_l1.unsqueeze(0)
                if y_l1.device != x_l1.device:
                    try:
                        y_l1 = y_l1.to(x_l1.device)
                    except Exception:
                        x_l1 = x_l1.cpu()
                        y_l1 = y_l1.cpu()
                return (x_l1 - y_l1).abs().mean(dim=(1, 2, 3), keepdim=True)

            def to(self, dev):
                to_fn = getattr(self._wrapped, "to", None)
                if callable(to_fn):
                    try:
                        self._wrapped = to_fn(dev)
                    except RuntimeError as exc:
                        msg = str(exc).lower()
                        if (
                            "cuda error" in msg
                            or "illegal memory access" in msg
                            or "device-side assert" in msg
                        ):
                            if not self._warned_fallback:
                                print(
                                    "[WARN] LPIPS module .to() failed in sandbox wrapper; "
                                    f"keeping current device and using fallback path if needed. exc={exc}"
                                )
                                self._warned_fallback = True
                        else:
                            raise
                return self

            def requires_grad_(self, flag):
                req_fn = getattr(self._wrapped, "requires_grad_", None)
                if callable(req_fn):
                    req_fn(flag)
                return self

            def __call__(self, x, y):
                x_src = x
                y_src = y
                if x_src.ndim == 3:
                    x_src = x_src.unsqueeze(0)
                if y_src.ndim == 3:
                    y_src = y_src.unsqueeze(0)

                run_device = x_src.device
                if y_src.device != run_device:
                    try:
                        y_src = y_src.to(run_device)
                    except RuntimeError as exc:
                        return self._fallback_l1(x_src, y_src, reason=f"align_input_device:{exc}")

                target_device = _infer_module_device(self._wrapped, fallback=run_device)
                if target_device != run_device:
                    to_fn = getattr(self._wrapped, "to", None)
                    if callable(to_fn):
                        try:
                            self._wrapped = to_fn(run_device)
                            target_device = run_device
                        except RuntimeError as exc:
                            return self._fallback_l1(x_src, y_src, reason=f"move_lpips_module:{exc}")

                try:
                    out = self._wrapped(x_src, y_src)
                except RuntimeError as exc:
                    msg = str(exc).lower()
                    retryable = (
                        "expected all tensors to be on the same device" in msg
                        or "illegal memory access" in msg
                        or "cuda error" in msg
                        or "device-side assert" in msg
                    )
                    if not retryable:
                        raise
                    return self._fallback_l1(x_src, y_src, reason=f"lpips_forward:{exc}")

                if isinstance(out, torch.Tensor) and out.device != x_src.device:
                    try:
                        out = out.to(x_src.device)
                    except RuntimeError as exc:
                        return self._fallback_l1(x_src, y_src, reason=f"move_loss_output:{exc}")
                return out

        return _SafeLPIPS(inner)

    _safe_build_lpips._editsplat_lpips_device_guard_patch = True  # type: ignore[attr-defined]
    _LEGACY_WRAPPER.ref._build_lpips_loss = _safe_build_lpips
    print("[TTT3R] patched run_editing_flow LPIPS builder with cross-device guard.")


_patch_lpips_device_guard()


def _patch_legacy_parse_iter() -> None:
    parse_iter = getattr(_LEGACY_WRAPPER, "parse_iter_from_checkpoint", None)
    if parse_iter is None:
        return
    if getattr(parse_iter, "_editsplat_regex_fix", False):
        return

    def _fixed_parse_iter_from_checkpoint(path: str) -> int:
        m = re.search(r"chkpnt(\d+)\.pth$", str(path))
        return int(m.group(1)) if m else 7000

    _fixed_parse_iter_from_checkpoint._editsplat_regex_fix = True  # type: ignore[attr-defined]
    _LEGACY_WRAPPER.parse_iter_from_checkpoint = _fixed_parse_iter_from_checkpoint
    print("[TTT3R] patched legacy parse_iter_from_checkpoint regex for numeric checkpoint extraction.")


_patch_legacy_parse_iter()


def _patch_legacy_edit_mask_alignment() -> None:
    orig_make_edit_mask = getattr(_LEGACY_WRAPPER, "_make_edit_mask", None)
    if orig_make_edit_mask is None or getattr(orig_make_edit_mask, "_editsplat_resolution_patch", False):
        return

    def _make_edit_mask(src_image: torch.Tensor, ref_image: torch.Tensor, quantile: float) -> torch.Tensor:
        src = _LEGACY_WRAPPER._to_01_bchw(src_image)
        ref = _LEGACY_WRAPPER._to_01_bchw(ref_image)
        if tuple(ref.shape[-2:]) != tuple(src.shape[-2:]):
            ref = torch.nn.functional.interpolate(ref.float(), size=src.shape[-2:], mode="bilinear", align_corners=False)
        diff = (ref - src).abs().mean(dim=1, keepdim=True)
        q = float(
            torch.quantile(
                diff.flatten(),
                torch.tensor([max(0.5, min(0.999, quantile))], device=diff.device),
            ).item()
        )
        scale = max(q, 1e-4)
        return (diff / scale).clamp(0.0, 1.0)

    _make_edit_mask._editsplat_resolution_patch = True  # type: ignore[attr-defined]
    _LEGACY_WRAPPER._make_edit_mask = _make_edit_mask
    print("[TTT3R] patched legacy _make_edit_mask to align ref/source resolution before edit-mask construction.")


_patch_legacy_edit_mask_alignment()


def _patch_gaussian_grad_mask_for_frozen_fields() -> None:
    gaussian_cls = getattr(_LEGACY_WRAPPER, "GaussianModel", None)
    if gaussian_cls is None:
        return
    orig_apply_grad_mask = getattr(gaussian_cls, "apply_grad_mask", None)
    if orig_apply_grad_mask is None or getattr(orig_apply_grad_mask, "_editsplat_frozen_field_patch", False):
        return

    def apply_grad_mask(self, mask, l_color=1.0, l_position=1.0):
        assert self.mask.shape[0] == self._xyz.shape[0]
        self.set_mask(mask)

        def position_hook(grad):
            final_grad = l_position * grad * (self.mask[:, None] if grad.ndim == 2 else self.mask[:, None, None])
            return final_grad

        def color_hook(grad):
            final_grad = l_color * grad * (self.mask[:, None] if grad.ndim == 2 else self.mask[:, None, None])
            return final_grad

        fields = ["_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"]
        hooks = []
        skipped = []
        if hasattr(self, "hooks"):
            for hook in getattr(self, "hooks", []):
                try:
                    hook.remove()
                except Exception:
                    pass

        for field in fields:
            this_field = getattr(self, field)
            if not getattr(this_field, "is_leaf", False) or not bool(getattr(this_field, "requires_grad", False)):
                skipped.append(field)
                continue

            if field in {"_features_dc", "_features_rest", "_opacity"}:
                hooks.append(this_field.register_hook(color_hook))
            else:
                hooks.append(this_field.register_hook(position_hook))

        self.hooks = hooks
        if skipped and not getattr(self, "_editsplat_grad_mask_skip_warned", False):
            print(f"[WARN] sandbox wrapper apply_grad_mask skipping frozen/non-leaf fields: {', '.join(skipped)}")
            self._editsplat_grad_mask_skip_warned = True

    apply_grad_mask._editsplat_frozen_field_patch = True  # type: ignore[attr-defined]
    gaussian_cls.apply_grad_mask = apply_grad_mask
    print("[TTT3R] patched GaussianModel.apply_grad_mask to tolerate frozen geometry/opacity fields.")


_patch_gaussian_grad_mask_for_frozen_fields()


def _resolve_source_point_cloud_ply(src_ckpt: Path, load_iter: int) -> Optional[Path]:
    candidates = []
    seen = set()

    def _push(path: Path) -> None:
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        candidates.append(path)

    resolved = src_ckpt.resolve() if src_ckpt.exists() else src_ckpt
    for ckpt in (src_ckpt, resolved):
        parent = ckpt.parent
        _push(parent / "point_cloud" / f"iteration_{load_iter}" / "point_cloud.ply")
        _push(parent / f"iteration_{load_iter}" / "point_cloud.ply")
        _push(parent / "point_cloud.ply")
        if parent.name.startswith("iteration_"):
            _push(parent / "point_cloud.ply")
            if parent.parent.name == "point_cloud":
                _push(parent.parent / f"iteration_{load_iter}" / "point_cloud.ply")

    for path in candidates:
        if path.exists():
            return path

    pc_roots = []
    pc_seen = set()

    def _push_pc_root(path: Path) -> None:
        key = str(path)
        if key in pc_seen:
            return
        pc_seen.add(key)
        pc_roots.append(path)

    for ckpt in (src_ckpt, resolved):
        parent = ckpt.parent
        if (parent / "point_cloud").is_dir():
            _push_pc_root(parent / "point_cloud")
        if parent.name == "point_cloud":
            _push_pc_root(parent)
        if parent.name.startswith("iteration_") and parent.parent.name == "point_cloud":
            _push_pc_root(parent.parent)

    def _iter_key(path: Path) -> int:
        name = path.name
        if name.startswith("iteration_"):
            try:
                return int(name.split("_", 1)[1])
            except Exception:
                return -1
        return -1

    for pc_root in pc_roots:
        iter_dirs = sorted(
            [p for p in pc_root.iterdir() if p.is_dir() and p.name.startswith("iteration_")],
            key=_iter_key,
            reverse=True,
        )
        for iter_dir in iter_dirs:
            ply = iter_dir / "point_cloud.ply"
            if ply.exists():
                return ply

    return None


def _patch_legacy_point_cloud_link_compat() -> None:
    orig_ensure_point_cloud_link = getattr(_LEGACY_WRAPPER, "ensure_point_cloud_link", None)
    if orig_ensure_point_cloud_link is None:
        return
    if getattr(orig_ensure_point_cloud_link, "_editsplat_source_layout_patch", False):
        return
    parse_iter = getattr(_LEGACY_WRAPPER, "parse_iter_from_checkpoint", None)

    def ensure_point_cloud_link(model_path: str, source_checkpoint: str) -> None:
        if not source_checkpoint:
            return

        model_dir = Path(model_path)
        src_ckpt = Path(source_checkpoint)
        load_iter = int(parse_iter(str(src_ckpt))) if parse_iter is not None else 7000
        src_ply = _resolve_source_point_cloud_ply(src_ckpt, load_iter)
        if src_ply is None:
            # Fall back to original logic to preserve original error message details.
            return orig_ensure_point_cloud_link(model_path, source_checkpoint)

        dst_iter = load_iter
        if src_ply.parent.name.startswith("iteration_"):
            try:
                dst_iter = int(src_ply.parent.name.split("_", 1)[1])
            except Exception:
                dst_iter = load_iter

        dst_ply = model_dir / "point_cloud" / f"iteration_{dst_iter}" / "point_cloud.ply"
        if dst_ply.exists():
            return
        model_dir.mkdir(parents=True, exist_ok=True)
        dst_ply.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_ply, dst_ply)
        print(f"[TTT3R] copied source point cloud from {src_ply} -> {dst_ply}")

    ensure_point_cloud_link._editsplat_source_layout_patch = True  # type: ignore[attr-defined]
    _LEGACY_WRAPPER.ensure_point_cloud_link = ensure_point_cloud_link
    print("[TTT3R] patched legacy ensure_point_cloud_link for checkpoint layout compatibility.")


_patch_legacy_point_cloud_link_compat()


def _normalize_langsam_effective(adapter) -> Tuple[str, Optional[str]]:
    name = str(getattr(adapter, "backend_name", "legacy")).strip().lower()
    if name in {"langsam", "langsam:cpu"}:
        return "langsam:cpu", None
    if name in {"stub", "full-image", "full_image", "full-image-stub"}:
        return "langsam:cpu", "langsam_unavailable_full_image_stub"
    return "langsam:cpu", f"legacy_backend={name or 'unknown'}"


def _load_legacy_langsam_adapter() -> object:
    prev_backend = os.environ.get("EDITSPLAT_MASK_BACKEND")
    os.environ["EDITSPLAT_MASK_BACKEND"] = "langsam"
    try:
        return _LEGACY_LANGSAM_LOADER()
    finally:
        if prev_backend is None:
            os.environ.pop("EDITSPLAT_MASK_BACKEND", None)
        else:
            os.environ["EDITSPLAT_MASK_BACKEND"] = prev_backend


def _load_mask_backend():
    backend = resolve_sam3_backend_request(os.environ.get("EDITSPLAT_MASK_BACKEND", "sam3"))
    if backend == "stub":
        _set_mask_backend_info(requested="stub", effective="stub")
        return _FullImageMaskStub()
    try:
        adapter = _Sam3MaskAdapter()
        if adapter._processor is None:
            raise RuntimeError("SAM3 adapter initialized without a live processor")
        _set_mask_backend_info(requested="sam3", effective="sam3")
        return adapter
    except Exception as exc:
        detail = f"{type(exc).__name__}: {str(exc).strip().replace(chr(10), ' ')}"[:300]
        _set_mask_backend_info(requested="sam3", effective="error:sam3", detail=detail)
        raise RuntimeError(f"SAM3-only wrapper failed to initialize SAM3 backend: {detail}") from exc


_LEGACY_WRAPPER.ref._load_langsam = _load_mask_backend  # type: ignore[attr-defined]


def _register_support_mask(
    mask: torch.Tensor,
    prompt: str,
    backend: str,
    detail: Optional[str] = None,
    role: Optional[str] = None,
    soft_mask: Optional[torch.Tensor] = None,
    boxes=None,
    scores=None,
) -> None:
    runtime = _CURRENT_RUNTIME
    if runtime is None:
        return
    try:
        idx = int(getattr(runtime, "current_idx", -1))
    except Exception:
        idx = -1
    if idx < 0:
        return
    cache = getattr(runtime, "support_mask_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        runtime.support_mask_cache = cache
    meta_cache = getattr(runtime, "support_mask_meta", None)
    if not isinstance(meta_cache, dict):
        meta_cache = {}
        runtime.support_mask_meta = meta_cache
    normalized_role = _normalize_support_role(role)
    mask_cpu = mask.detach().float().cpu()
    soft_cpu = soft_mask.detach().float().cpu() if isinstance(soft_mask, torch.Tensor) else mask_cpu
    payload = {
        "mask": mask_cpu,
        "soft_mask": soft_cpu,
        "prompt": prompt,
        "backend": backend,
        "detail": detail,
        "role": normalized_role,
    }
    if boxes is not None:
        payload["boxes"] = boxes.detach().cpu().tolist() if isinstance(boxes, torch.Tensor) else boxes
    if scores is not None:
        payload["scores"] = scores.detach().cpu().tolist() if isinstance(scores, torch.Tensor) else scores
    entry = cache.get(idx)
    if not isinstance(entry, dict):
        entry = {"roles": {}}
    roles = entry.get("roles")
    if not isinstance(roles, dict):
        roles = {}
        entry["roles"] = roles
    roles[normalized_role] = payload
    entry["last"] = payload
    cache[idx] = entry
    meta_cache[idx] = {
        "prompt": prompt,
        "backend": backend,
        "detail": detail,
        "role": normalized_role,
        "available_roles": sorted(str(k) for k in roles.keys()),
    }
    fit_cache = getattr(runtime, "fit_mask_cache", None)
    if isinstance(fit_cache, dict):
        fit_cache.pop(idx, None)


def _patch_langsam_predict_tracking() -> None:
    predict_fn = getattr(_LEGACY_WRAPPER.ref, "_predict_langsam_mask", None)
    normalize_fn = getattr(_LEGACY_WRAPPER.ref, "_normalize_langsam_mask", None)
    full_mask_fn = getattr(_LEGACY_WRAPPER.ref, "_full_image_mask", None)
    if predict_fn is None or normalize_fn is None or full_mask_fn is None:
        return
    if getattr(predict_fn, "_editsplat_mask_tracking_patch", False):
        return

    def _iter_langsam_prompts(primary: str):
        base = (primary or "").strip()
        seen = set()
        if base:
            seen.add(base)
            yield base
        for alt in ("person", "person face", "human face", "head", "portrait"):
            if alt and alt not in seen:
                seen.add(alt)
                yield alt

    def _predict_sam3_mask(sam3_adapter, image_pil, text_prompt, image_height, image_width, mask_role: Optional[str] = None):
        last_exc: Optional[Exception] = None
        last_empty_prompt: Optional[str] = None
        for prompt in _iter_langsam_prompts(text_prompt):
            try:
                mask, boxes, scores, _ = sam3_adapter.predict(image_pil, prompt)
                mask = normalize_fn(mask, image_height=image_height, image_width=image_width)
            except Exception as exc:
                last_exc = exc
                continue
            if mask is None or mask.numel() == 0 or float(mask.max().item()) <= 0.0:
                last_empty_prompt = prompt
                _append_mask_backend_detail(f"sam3_empty_mask prompt={prompt}")
                continue
            if prompt != text_prompt:
                _append_mask_backend_detail(f"sam3_prompt_fallback={prompt}")
                print(f"[WARN] SAM3 prompt fallback succeeded: {text_prompt} -> {prompt}")
            used_threshold = float(
                getattr(
                    sam3_adapter,
                    "_last_used_threshold",
                    getattr(sam3_adapter, "_base_confidence_threshold", 0.0),
                )
            )
            _register_support_mask(
                mask,
                prompt,
                backend="sam3",
                detail=f"thr={used_threshold:.3f}",
                role=mask_role,
                soft_mask=mask,
                boxes=boxes,
                scores=scores,
            )
            return mask

        detail_bits = []
        if last_empty_prompt:
            detail_bits.append(f"last_empty_prompt={last_empty_prompt}")
        if last_exc is not None:
            detail_bits.append(f"last_exc={type(last_exc).__name__}")
        extra = "sam3_predict_full_image_stub"
        if detail_bits:
            extra = f"{extra}({', '.join(detail_bits)})"
        _append_mask_backend_detail(extra)
        print(
            "[WARN] SAM3 predict failed after trying all prompts; "
            f"prompt={text_prompt} last_exc={last_exc}"
        )
        _maybe_dump_sam3_failure(image_pil, text_prompt, extra)
        return full_mask_fn(image_height, image_width)

    def _predict_langsam_mask(lang_sam, image_pil, text_prompt, image_height, image_width, mask_role: Optional[str] = None):
        backend_name = str(getattr(lang_sam, "backend_name", "")).strip().lower()
        if backend_name == "sam3":
            return _predict_sam3_mask(lang_sam, image_pil, text_prompt, image_height, image_width, mask_role=mask_role)
        if text_prompt == "no_mask":
            return full_mask_fn(image_height, image_width)
        last_exc = None
        for prompt in _iter_langsam_prompts(text_prompt):
            try:
                mask, _, _, _ = lang_sam.predict(image_pil, prompt)
                mask = normalize_fn(mask, image_height=image_height, image_width=image_width)
                if mask is None or mask.numel() == 0 or float(mask.max().item()) <= 0.0:
                    raise RuntimeError("LangSAM returned an empty mask")
                if prompt != text_prompt:
                    _append_mask_backend_detail(f"langsam_prompt_fallback={prompt}")
                    print(f"[WARN] LangSAM prompt fallback succeeded: {text_prompt} -> {prompt}")
                _register_support_mask(mask, prompt, backend=backend_name or "langsam", role=mask_role, soft_mask=mask)
                return mask
            except Exception as exc:
                last_exc = exc
                continue
        _append_mask_backend_detail("langsam_predict_full_image_stub")
        print(
            "[WARN] LangSAM predict failed, using full-image mask stub. "
            f"prompt={text_prompt} exc={last_exc}"
        )
        return full_mask_fn(image_height, image_width)

    _predict_langsam_mask._editsplat_mask_tracking_patch = True  # type: ignore[attr-defined]
    _LEGACY_WRAPPER.ref._predict_langsam_mask = _predict_langsam_mask


_patch_langsam_predict_tracking()


def _patch_fit_mask_fusion() -> None:
    orig_make_edit_mask = getattr(_LEGACY_WRAPPER, "_make_edit_mask", None)
    base_runtime_cls = getattr(_LEGACY_WRAPPER, "TTT3RRuntime", None)
    if orig_make_edit_mask is None or base_runtime_cls is None:
        return

    def _fit_mask_from_runtime(runtime, mode, quantile, bg):
        if str(mode).strip().lower() == "none":
            return None
        idx = int(getattr(runtime, "current_idx", -1))
        cached = runtime.fit_mask_cache.get(idx)
        if cached is not None:
            return cached
        src = runtime.source_image_cache.get(idx)
        init_ref = runtime.initial_edit_cache.get(idx)
        if src is None or init_ref is None:
            return None
        mask = orig_make_edit_mask(src.to(torch.float32), init_ref.to(torch.float32), float(quantile))
        mask = mask.detach().cpu()
        used_support = False
        support_role = os.environ.get("EDITSPLAT_SAM3_FIT_ROLE", "gt_view")
        support_alpha = max(0.0, min(1.0, _env_float("EDITSPLAT_SAM3_FIT_ALPHA", 1.0)))
        support_payload = _get_support_payload(runtime, idx, role=support_role)
        sam_mask = _payload_mask(support_payload, image_size=tuple(mask.shape[-2:]))
        if isinstance(sam_mask, torch.Tensor):
            if support_alpha >= 1.0:
                fused_support = sam_mask.to(mask.dtype)
            else:
                fused_support = ((1.0 - support_alpha) * mask + support_alpha * sam_mask.to(mask.dtype)).clamp(0.0, 1.0)
            mask = torch.max(mask, fused_support)
            used_support = True
        if float(bg) > 0.0:
            mask = mask + (1.0 - mask) * float(bg)
            mask = mask.clamp(0.0, 1.0)
        runtime.fit_mask_cache[idx] = mask
        if runtime.dump_root is not None and idx not in runtime.fit_mask_dumped:
            out_dir = runtime.dump_root / "fit_masks"
            out_dir.mkdir(parents=True, exist_ok=True)
            runtime._save_tensor_png(mask.to(torch.float32), out_dir / f"view{idx:03d}.png")
            stats = {
                "mode": mode,
                "quantile": float(quantile),
                "bg": float(bg),
                "mean": float(mask.float().mean().item()),
                "min": float(mask.float().min().item()),
                "max": float(mask.float().max().item()),
            }
            if used_support:
                stats["sam_support"] = True
                stats["sam_support_role"] = support_role
                stats["sam_support_alpha"] = support_alpha
                if isinstance(support_payload, dict):
                    stats["sam_support_meta"] = {
                        "prompt": support_payload.get("prompt"),
                        "backend": support_payload.get("backend"),
                        "detail": support_payload.get("detail"),
                        "role": support_payload.get("role"),
                    }
            (out_dir / f"view{idx:03d}.json").write_text(
                json.dumps(stats, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            runtime.fit_mask_dumped.add(idx)
        return mask

    _LEGACY_WRAPPER._fit_mask_from_runtime = _fit_mask_from_runtime


_patch_fit_mask_fusion()


def _maybe_dump_sam3_failure(image_pil: Image.Image, prompt: str, reason: str) -> None:
    """Persist a debug snapshot when SAM3 returns empty masks to speed diagnosis."""

    global _SAM3_DEBUG_COUNTER
    debug_root = os.environ.get("EDITSPLAT_SAM3_DEBUG_ROOT") or os.environ.get("EDITSPLAT_ACTIVE_MODEL_PATH")
    limit = int(os.environ.get("EDITSPLAT_SAM3_DEBUG_LIMIT", "4"))
    if not debug_root or _SAM3_DEBUG_COUNTER >= limit:
        return
    try:
        root = Path(debug_root) / "sam3_debug"
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        png_path = root / f"failure_{stamp}.png"
        meta_path = root / f"failure_{stamp}.json"
        arr = np.asarray(image_pil.convert("RGB"), dtype=np.uint8)
        image_pil.save(png_path)
        payload = {
            "prompt": prompt,
            "reason": reason,
            "size": {"width": image_pil.width, "height": image_pil.height},
            "stats": {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": int(arr.min()),
                "max": int(arr.max()),
            },
        }
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _SAM3_DEBUG_COUNTER += 1
        print(f"[SAM3][debug] wrote failure dump to {png_path} reason={reason}")
    except Exception as exc:  # pragma: no cover - best-effort diagnostics
        print(f"[WARN] failed to dump SAM3 debug snapshot: {exc}")


def main() -> None:
    _stabilize_ttt3r_attention()
    model_path = _extract_flag_value("--model_path")
    if model_path:
        os.environ["EDITSPLAT_ACTIVE_MODEL_PATH"] = model_path
    skip_guard_env = os.environ.get("EDITSPLAT_SKIP_3DGS_BACKWARD_ON_ERROR", "1")
    skip_guard_enabled = skip_guard_env.strip().lower() not in {"0", "false", "off", "no"}
    try:
        _LEGACY_WRAPPER.main()
    except (RuntimeError, torch.OutOfMemoryError) as exc:
        msg = str(exc).lower()
        if skip_guard_enabled and ("illegal memory access" in msg or "out of memory" in msg):
            traceback.print_exc()
            print(
                "[WARN] sandbox wrapper caught CUDA runtime memory fault; "
                "preserving partial artifacts and continuing keepalive loop."
            )
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        else:
            raise
    finally:
        _write_mask_backend_metadata(model_path)


if __name__ == "__main__":
    main()


def _patch_advanced_gaussian_optimization() -> None:
    """Apply advanced optimization techniques to Gaussian model for better editing results."""
    gaussian_cls = getattr(_LEGACY_WRAPPER, "GaussianModel", None)
    if gaussian_cls is None:
        return
    
    # Import the necessary functions from the original code
    try:
        from utils.general_utils import get_expon_lr_func
    except ImportError:
        # Define basic exponential learning rate function if not available
        def get_expon_lr_func(lr_init, lr_final, lr_delay_mult, max_steps):
            def func(step):
                if step < 0:
                    return lr_init
                scale = lr_final / lr_init
                progress = (step - 1) / (max_steps - 1)
                progress = min(progress, 1.0)
                return lr_init * (scale ** (lr_delay_mult + (1 - lr_delay_mult) * progress))
            return func

    # Patch the densification method to add more sophisticated densification logic
    orig_densify_and_prune = getattr(gaussian_cls, "densify_and_prune", None)
    if orig_densify_and_prune is None:
        return
    
    def densify_and_prune_with_adaptive_strategy(
        self, 
        max_grad, 
        min_opacity, 
        extent, 
        max_screen_size, 
        percent_dense=0.01, 
        voxel_size=0.01,
        is_first_densification=False,
        k_percent=0.15,
        attn_thres=0.1
    ):
        """
        Enhanced densification with adaptive strategies for better editing results.
        """
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # Apply spatially adaptive densification based on gradient magnitude
        n = torch.sum(grads.detach(), dim=-1) > max_grad
        
        # Additional condition: densify only in regions with significant changes
        if hasattr(self, '_editing_region_mask') and self._editing_region_mask is not None:
            # Only densify in the editing region
            editing_mask = self._editing_region_mask.to(grads.device)
            n = n & (editing_mask > 0.5)  # Only densify where editing is happening
        
        # Get positions that require densification
        big_points_ws = self.get_xyz[n]
        
        # Check if we're at a valid densification step
        if big_points_ws.shape[0] > 0:
            # Calculate new positions for split operations
            big_points_ind = torch.where(n)[0]
            new_xyz = self.get_xyz[big_points_ind]
            
            # Determine which points to clone vs split based on scale
            scales = self.get_scaling[big_points_ind]
            split_mask = torch.max(scales, dim=1).values > 0.05  # Adjust threshold as needed
            clone_mask = ~split_mask
            
            # Split points that are too large
            split_indices = big_points_ind[split_mask]
            if len(split_indices) > 0:
                # Get the properties to be duplicated
                split_xyz = self.get_xyz[split_indices]
                split_features_dc = self._features_dc[split_indices]
                split_features_rest = self._features_rest[split_indices]
                split_opacities = self._opacity[split_indices]
                split_scales = self._scaling[split_indices]
                split_rotations = self._rotation[split_indices]
                
                # Halve the scales for the split points
                split_scales = torch.log(torch.exp(split_scales) * 0.5)
                
                # Duplicate the points with modified properties
                new_xyz_split = split_xyz
                new_features_dc_split = split_features_dc
                new_features_rest_split = split_features_rest
                new_opacities_split = split_opacities
                new_scales_split = split_scales
                new_rotations_split = split_rotations
                
                # Add the new points
                self.densification_postfix(
                    new_xyz_split, 
                    new_features_dc_split, 
                    new_features_rest_split, 
                    new_opacities_split, 
                    new_scales_split, 
                    new_rotations_split
                )

            # Clone points that are small but have high gradients
            clone_indices = big_points_ind[clone_mask]
            if len(clone_indices) > 0:
                clone_xyz = self.get_xyz[clone_indices]
                clone_features_dc = self._features_dc[clone_indices]
                clone_features_rest = self._features_rest[clone_indices]
                clone_opacities = self._opacity[clone_indices]
                clone_scales = self._scaling[clone_indices]
                clone_rotations = self._rotation[clone_indices]
                
                self.densification_postfix(
                    clone_xyz, 
                    clone_features_dc, 
                    clone_features_rest, 
                    clone_opacities, 
                    clone_scales, 
                    clone_rotations
                )

        # Prune operation remains the same
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if hasattr(self, 'max_gaussians') and self.max_gaussians > 0:
            # Additional constraint: if we have too many Gaussians, be more aggressive in pruning
            n_gaussians = self.get_xyz.shape[0]
            if n_gaussians > int(self.max_gaussians * 0.9):  # Start being more aggressive at 90% capacity
                # Prune more aggressively based on opacity and spatial distribution
                prune_mask = prune_mask | (self.get_opacity < (min_opacity * 2)).squeeze()
        
        # Perform pruning
        self.prune_points(prune_mask)

        # Reset accumulators
        torch.cuda.empty_cache()
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda", dtype=torch.float32)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda", dtype=torch.float32)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda", dtype=torch.float32)

    densify_and_prune_with_adaptive_strategy._editsplat_advanced_densify_patch = True
    gaussian_cls.densify_and_prune = densify_and_prune_with_adaptive_strategy

    # Also patch the training setup to enable adaptive learning rates
    orig_training_setup = getattr(gaussian_cls, "training_setup", None)
    if orig_training_setup is not None:
        def training_setup_with_adaptive_lr(self, training_args):
            """Enhanced training setup with adaptive learning rate strategies."""
            # Original optimizer setup
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init, "name": "xyz"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
            ]

            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
            
            # Scheduler setup - make it adaptive to editing stage
            self.xyz_scheduler_args = get_expon_lr_func(
                lr_init=training_args.position_lr_init,
                lr_final=training_args.position_lr_final,
                lr_delay_mult=training_args.position_lr_delay_mult,
                max_steps=training_args.position_lr_max_steps)

            # Store the training args for access in other methods
            self.training_args = training_args

        training_setup_with_adaptive_lr._editsplat_adaptive_lr_patch = True
        gaussian_cls.training_setup = training_setup_with_adaptive_lr

    print("[TTT3R] patched GaussianModel with advanced optimization techniques.")


def _patch_gaussian_dynamic_properties() -> None:
    """Add dynamic property adjustment during editing."""
    gaussian_cls = getattr(_LEGACY_WRAPPER, "GaussianModel", None)
    if gaussian_cls is None:
        return
    
    # Add a method to dynamically adjust properties based on edit progress
    def adjust_learning_rates_during_training(self, iteration, total_iterations, editing_phase='early'):
        """Dynamically adjust learning rates based on training progress and editing phase."""
        # Calculate progress ratio
        progress = iteration / total_iterations
        
        # Adjust learning rates based on phase and progress
        if editing_phase == 'early':
            # Early phase: higher learning rates for major adjustments
            xyz_lr = self.xyz_scheduler_args(iteration) * 1.0
            feature_lr_factor = 1.0
        elif editing_phase == 'mid':
            # Mid phase: moderate learning rates for refinement
            xyz_lr = self.xyz_scheduler_args(iteration) * 0.7
            feature_lr_factor = 0.8
        else:  # late
            # Late phase: lower learning rates for fine-tuning
            xyz_lr = self.xyz_scheduler_args(iteration) * 0.3
            feature_lr_factor = 0.5
        
        # Update optimizer parameter groups
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                param_group['lr'] = xyz_lr
            elif param_group["name"] in ["f_dc", "f_rest"]:
                param_group['lr'] = param_group['lr'] * feature_lr_factor

    gaussian_cls.adjust_learning_rates_during_training = adjust_learning_rates_during_training


_patch_advanced_gaussian_optimization()
_patch_gaussian_dynamic_properties()
