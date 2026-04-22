import sys
import os
import random
from argparse import ArgumentParser
import json
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List
from jaxtyping import Float

from PIL import Image
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Subset, Dataset
from torchvision.transforms import ToPILImage
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers import FluxPipeline, DDIMScheduler
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

try:
    import lpips
except Exception:
    lpips = None

try:
    import ImageReward as RM
except Exception:
    RM = None

if "EDITSPLAT_HF_HOME" in os.environ:
    os.environ["HF_HOME"] = os.environ["EDITSPLAT_HF_HOME"]
else:
    os.environ.setdefault("HF_HOME", "/dev_vepfs/rc_wu/cache/hf_home")

if "EDITSPLAT_TORCH_HOME" in os.environ:
    os.environ["TORCH_HOME"] = os.environ["EDITSPLAT_TORCH_HOME"]
else:
    os.environ.setdefault("TORCH_HOME", "/dev_vepfs/rc_wu/cache/torch")

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

try:
    from lang_sam import LangSAM
except Exception:
    LangSAM = None

from scene import Scene, GaussianModel
from gaussian_renderer import render
from render import render_sets
from arguments import ModelParams, PipelineParams, OptimizationParams, EditingParams, ScoreDistillParams
from scene.dataloader import CameraDataset

from utils.attention import prep_unet, get_all_attention_maps, reset_attention_maps, seperate_attention_maps_by_tokens, save_attention_maps
from utils.loss_utils import l1_loss
from utils.rgbd_warping import reproject_rgbd, reprojected2img
from utils.camera_proximity_utils import find_nearby_camera
from utils.flow_utils import scale_noise, calculate_shift, calc_v_flux
from utils.frontier_fallback import (
    _blend_face_override,
    _compose_anchor_face_fallback,
    _proxy_region_stats,
    _should_use_frontier_fallback,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_lpips_loss(device: torch.device):
    if lpips is not None:
        try:
            loss_fn = lpips.LPIPS(net='vgg').to(device)
            loss_fn.requires_grad_(False)
            return loss_fn
        except Exception as exc:
            print(f"[WARN] LPIPS unavailable, fallback to L1 proxy. exc={exc}")

    class _LPIPSStub:
        def to(self, device):
            del device
            return self

        def requires_grad_(self, flag):
            del flag
            return self

        def __call__(self, x, y):
            if x.ndim == 3:
                x = x.unsqueeze(0)
            if y.ndim == 3:
                y = y.unsqueeze(0)
            return (x.float() - y.float()).abs().mean(dim=(1, 2, 3), keepdim=True)

    return _LPIPSStub()


def _load_reward_model():
    if RM is None:
        print("[WARN] ImageReward unavailable, using dummy reward ranker.")

        class _DummyRewardModel:
            def inference_rank(self, prompt, images):
                del prompt
                return list(range(1, len(images) + 1)), [0.0] * len(images)

        return _DummyRewardModel()
    try:
        return RM.load("ImageReward-v1.0")
    except Exception as exc:
        print(f"[WARN] ImageReward load failed, using dummy reward ranker. exc={exc}")

        class _DummyRewardModel:
            def inference_rank(self, prompt, images):
                del prompt
                return list(range(1, len(images) + 1)), [0.0] * len(images)

        return _DummyRewardModel()


def _load_langsam():
    class _LangSAMStub:
        backend_name = "stub"

        def predict(self, image_pil, text_prompt):
            del text_prompt
            w, h = image_pil.size
            mask = torch.ones((1, h, w), dtype=torch.float32)
            return mask, None, None, None

    backend = os.environ.get("EDITSPLAT_MASK_BACKEND", "langsam").strip().lower()
    if backend in {"stub", "full", "full-image", "full_image"}:
        print(f"[WARN] EDITSPLAT_MASK_BACKEND={backend}: using full-image mask stub.")
        return _LangSAMStub()

    if backend not in {"langsam", "auto", ""}:
        print(f"[WARN] Unknown EDITSPLAT_MASK_BACKEND={backend}; falling back to LangSAM.")

    if LangSAM is None:
        print("[WARN] LangSAM unavailable, using full-image mask stub.")
        return _LangSAMStub()

    hf_home = os.environ.get("EDITSPLAT_HF_HOME") or os.environ.get("HF_HOME") or "/dev_vepfs/rc_wu/cache/hf_home"
    torch_home = os.environ.get("EDITSPLAT_TORCH_HOME") or os.environ.get("TORCH_HOME") or "/dev_vepfs/rc_wu/cache/torch"
    sam_ckpt = os.environ.get("EDITSPLAT_LANGSAM_SAM_CKPT") or os.path.join(torch_home, "hub", "checkpoints", "sam_vit_h_4b8939.pth")
    langsam_device = os.environ.get("EDITSPLAT_LANGSAM_DEVICE", "cpu").strip().lower() or "cpu"
    if langsam_device not in {"auto", "cpu", "cuda"}:
        print(f"[WARN] Unknown EDITSPLAT_LANGSAM_DEVICE={langsam_device}; falling back to cpu.")
        langsam_device = "cpu"

    os.environ["HF_HOME"] = hf_home
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ["TORCH_HOME"] = torch_home

    repo_root = Path(hf_home) / "hub" / "models--ShilongLiu--GroundingDINO"
    snapshot_id = os.environ.get("EDITSPLAT_LANGSAM_DINO_SNAPSHOT_ID", "").strip()
    if not snapshot_id:
        ref_main = repo_root / "refs" / "main"
        if ref_main.is_file():
            snapshot_id = ref_main.read_text(encoding="utf-8").strip()
    default_snapshot = repo_root / "snapshots" / snapshot_id if snapshot_id else repo_root / "snapshots"
    dino_snapshot = Path(os.environ.get("EDITSPLAT_LANGSAM_DINO_SNAPSHOT", str(default_snapshot)))
    dino_cfg = dino_snapshot / "GroundingDINO_SwinB.cfg.py"
    dino_ckpt = dino_snapshot / "groundingdino_swinb_cogcoor.pth"

    try:
        from lang_sam import lang_sam as lang_sam_module

        original_load_model_hf = getattr(lang_sam_module, "load_model_hf", None)

        def _load_model_hf_local(repo_id, filename, ckpt_config_filename, device='cpu'):
            cfg_path = dino_snapshot / ckpt_config_filename
            ckpt_path = dino_snapshot / filename
            if cfg_path.is_file() and ckpt_path.is_file():
                args = lang_sam_module.SLConfig.fromfile(str(cfg_path))
                model = lang_sam_module.build_model(args)
                args.device = device
                checkpoint = torch.load(str(ckpt_path), map_location='cpu')
                log = model.load_state_dict(lang_sam_module.clean_state_dict(checkpoint['model']), strict=False)
                print(
                    f"[INFO] LangSAM local GroundingDINO load: repo_id={repo_id} cfg={cfg_path} ckpt={ckpt_path} => {log}"
                )
                model.eval()
                return model
            if callable(original_load_model_hf):
                return original_load_model_hf(repo_id, filename, ckpt_config_filename, device=device)
            raise FileNotFoundError(
                f"GroundingDINO local snapshot incomplete: cfg={cfg_path} ckpt={ckpt_path}"
            )

        lang_sam_module.load_model_hf = _load_model_hf_local

        original_cuda_is_available = torch.cuda.is_available
        if os.path.isfile(sam_ckpt):
            print(
                f"[INFO] LangSAM local init: HF_HOME={hf_home} TORCH_HOME={torch_home} "
                f"DINO_SNAPSHOT={dino_snapshot} SAM_CKPT={sam_ckpt} DEVICE={langsam_device}"
            )
            try:
                if langsam_device == "cpu":
                    torch.cuda.is_available = lambda: False
                model = LangSAM(ckpt_path=sam_ckpt)
            finally:
                torch.cuda.is_available = original_cuda_is_available
        else:
            print(f"[WARN] LangSAM SAM checkpoint missing at {sam_ckpt}; trying default LangSAM() path.")
            try:
                if langsam_device == "cpu":
                    torch.cuda.is_available = lambda: False
                model = LangSAM()
            finally:
                torch.cuda.is_available = original_cuda_is_available
        if model is None or not hasattr(model, "predict"):
            raise RuntimeError("LangSAM() returned an invalid object")
        resolved_device = str(getattr(model, "device", langsam_device))
        setattr(model, "backend_name", f"langsam:{resolved_device}")
        return model
    except Exception as exc:
        print(
            "[WARN] LangSAM init failed, using full-image mask stub. "
            f"backend={backend} hf_home={hf_home} torch_home={torch_home} "
            f"dino_snapshot={dino_snapshot} sam_ckpt={sam_ckpt} exc={exc}"
        )
        return _LangSAMStub()


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return int(default)
    try:
        return int(value)
    except ValueError:
        return int(default)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


def _env_choice(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower()


def _load_anchor_override_image(path_str: str, image_height: int, image_width: int) -> Optional[torch.Tensor]:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_file():
        print(f"[WARN] Anchor override image missing: {path}")
        return None
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    if tuple(tensor.shape[-2:]) != (image_height, image_width):
        tensor = F.interpolate(tensor, size=(image_height, image_width), mode="bilinear", align_corners=False)
    return tensor.to(torch.float32)


def _full_image_mask(image_height: int, image_width: int) -> torch.Tensor:
    return torch.ones((1, image_height, image_width), dtype=torch.float32)


def _normalize_langsam_mask(
    mask: Optional[torch.Tensor],
    image_height: int,
    image_width: int,
) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask)
    if mask.numel() == 0:
        return None

    mask = mask.detach().float().cpu()
    if mask.ndim == 4:
        if mask.shape[1] == 1:
            mask = mask[:, 0]
        elif mask.shape[0] == 1:
            mask = mask[0]
        else:
            mask = mask[:, 0]

    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    elif mask.ndim == 3:
        if mask.shape[0] != 1:
            mask = mask.amax(dim=0, keepdim=True)
    else:
        return None

    if tuple(mask.shape[-2:]) != (image_height, image_width):
        mask = F.interpolate(mask.unsqueeze(0), size=(image_height, image_width), mode="nearest").squeeze(0)
    mask = mask.float().clamp(0.0, 1.0)
    if _env_flag("EDITSPLAT_BINARIZE_SUPPORT_MASK", True):
        threshold = float(os.environ.get("EDITSPLAT_SUPPORT_MASK_THRESHOLD", "0.0"))
        return (mask > threshold).float()
    return mask


def _predict_langsam_mask(
    lang_sam,
    image_pil: Image.Image,
    text_prompt: str,
    image_height: int,
    image_width: int,
    mask_role: Optional[str] = None,
) -> torch.Tensor:
    del mask_role
    if text_prompt == "no_mask":
        return _full_image_mask(image_height, image_width)
    try:
        mask, _, _, _ = lang_sam.predict(image_pil, text_prompt)
        mask = _normalize_langsam_mask(mask, image_height=image_height, image_width=image_width)
        if mask is None or mask.numel() == 0 or float(mask.max().item()) <= 0.0:
            raise RuntimeError("LangSAM returned an empty mask")
        return mask
    except Exception as exc:
        print(f"[WARN] LangSAM predict failed, using full-image mask stub. prompt={text_prompt} exc={exc}")
        return _full_image_mask(image_height, image_width)


def _to_01_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    y = x.detach().float()
    if y.numel() and y.min() < 0:
        y = (y + 1.0) * 0.5
    return y.clamp(0.0, 1.0)


def _save_debug_tensor(x: torch.Tensor, out_path: Path) -> None:
    x01 = _to_01_bchw(x).cpu()
    if x01.shape[1] == 1:
        arr = (x01[0, 0].numpy() * 255.0).round().astype(np.uint8)
        Image.fromarray(arr, mode="L").save(out_path)
        return
    arr = (x01[0].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(out_path)


def _initial_edit_diff_score(edited_image: torch.Tensor, gt_image: torch.Tensor) -> float:
    edited = _to_01_bchw(edited_image)
    gt = _to_01_bchw(gt_image)
    if tuple(edited.shape[-2:]) != tuple(gt.shape[-2:]):
        gt = F.interpolate(gt, size=edited.shape[-2:], mode="bilinear", align_corners=False)
    return float((edited - gt).abs().mean().item())


def _pick_frontier_anchor(initial_edit_scores: List[float], candidate_indices: List[int]) -> int:
    if not candidate_indices:
        candidate_indices = list(range(len(initial_edit_scores)))
    center = 0.5 * max(0, len(initial_edit_scores) - 1)
    return max(
        candidate_indices,
        key=lambda idx: (float(initial_edit_scores[idx]), -abs(float(idx) - center)),
    )


def _should_abort_3dgs_backward_error(msg: str) -> bool:
    lowered = str(msg).lower()
    return any(token in lowered for token in (
        'illegal memory access',
        'invalid argument',
    ))


def _prepare_debug_root(model_path: str) -> Optional[Path]:
    if not _env_flag("EDITSPLAT_DUMP_INTERMEDIATES", False):
        return None
    root = Path(model_path) / "debug_intermediates"
    for stage in ("initial_edit", "selection", "mfg_edit"):
        (root / stage).mkdir(parents=True, exist_ok=True)
    meta = {
        "filter_mode": _env_choice("EDITSPLAT_FILTER_MODE", "keep_ratio"),
        "filter_keep_count": _env_int("EDITSPLAT_FILTER_KEEP_COUNT", -1),
        "mfg_mode": _env_choice("EDITSPLAT_MFG_MODE", "full"),
        "mfg_backfill": _env_choice("EDITSPLAT_MFG_BACKFILL", "nearest"),
        "mfg_source_count": _env_int("EDITSPLAT_MFG_SOURCE_COUNT", 5),
        "skip_agt": _env_flag("EDITSPLAT_SKIP_AGT", False),
    }
    (root / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return root


def _dump_stage_payload(
    debug_root: Optional[Path],
    stage: str,
    view_idx: int,
    payload: Dict[str, torch.Tensor],
    meta: Optional[Dict[str, object]] = None,
) -> None:
    if debug_root is None:
        return
    stage_dir = debug_root / stage / f"view{view_idx:03d}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    stats: Dict[str, object] = {}
    for name, tensor in payload.items():
        if tensor is None or not isinstance(tensor, torch.Tensor):
            continue
        t = tensor.detach().float()
        stats[name] = {
            "shape": list(t.shape),
            "min": float(t.min().item()),
            "max": float(t.max().item()),
            "mean": float(t.mean().item()),
        }
        if t.ndim >= 2:
            _save_debug_tensor(tensor, stage_dir / f"{name}.png")
    if meta:
        stats["meta"] = meta
    (stage_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")


def _normalize_reward_ranks(ranking, num_items: int) -> List[int]:
    if not isinstance(ranking, (list, tuple)) or len(ranking) != num_items:
        return list(range(1, num_items + 1))
    ranks = [int(x) for x in ranking]
    if num_items > 0 and min(ranks) == 0:
        ranks = [x + 1 for x in ranks]
    return ranks


def _resolve_reward_selection(ranking, rewards, filtering_ratio: float, num_items: int):
    ranks = _normalize_reward_ranks(ranking, num_items)
    if isinstance(rewards, np.ndarray):
        reward_list = rewards.tolist()
    elif isinstance(rewards, (list, tuple)):
        reward_list = list(rewards)
    else:
        reward_list = [None] * num_items
    if len(reward_list) != num_items:
        reward_list = [None] * num_items

    mode = _env_choice("EDITSPLAT_FILTER_MODE", "keep_ratio")
    keep_count = _env_int("EDITSPLAT_FILTER_KEEP_COUNT", -1)

    if keep_count < 0:
        if mode == "all":
            keep_count = num_items
        elif mode == "none":
            keep_count = 0
        elif mode == "legacy":
            keep_ratio = max(0.0, min(1.0, 1.0 - float(filtering_ratio)))
            keep_count = int(num_items * keep_ratio)
        else:
            keep_ratio = max(0.0, min(1.0, float(filtering_ratio)))
            keep_count = int(np.ceil(num_items * keep_ratio)) if num_items > 0 else 0

    keep_count = max(0, min(num_items, int(keep_count)))
    if num_items > 0 and mode != "none":
        keep_count = max(1, keep_count)

    selected = [rank <= keep_count for rank in ranks]
    if num_items > 0 and keep_count > 0 and not any(selected):
        best_idx = int(np.argmin(np.asarray(ranks)))
        selected[best_idx] = True

    meta = {
        "filter_mode": mode,
        "filtering_ratio": float(filtering_ratio),
        "keep_count": int(keep_count),
        "ranks": ranks,
        "rewards": [float(x) if x is not None else None for x in reward_list],
        "selected": selected,
    }
    return selected, meta
    try:
        return LangSAM()
    except Exception as exc:
        print(f"[WARN] LangSAM init failed, using full-image mask stub. exc={exc}")

        class _LangSAMStub:
            def predict(self, image_pil, text_prompt):
                del text_prompt
                w, h = image_pil.size
                mask = torch.ones((1, h, w), dtype=torch.float32)
                return mask, None, None, None

        return _LangSAMStub()

class HeadCameraDataset(Dataset):
    def __init__(self, base_dataset, k: int):
        # 保留 camera_list 属性，供 pipeline 使用
        self.camera_list = base_dataset.camera_list[:k]

    def __len__(self):
        return len(self.camera_list)

    def __getitem__(self, idx):
        cam = self.camera_list[idx]
        return {
            "idx": idx,                  # 若要保留原始全局 idx，可在这里换成别的映射
            "gt_image": cam.gt_image,
        }

def _lowpass_like(x: torch.Tensor, pack_shape: Tuple[int, int, int, int]) -> torch.Tensor:
    """
    对 packed 的 [B, L, C] token 进行低频化（通过 unpack->blur->pack）；
    pack_shape: (B, C, H, W) 对应的 reshape 信息
    """
    B, C, H, W = pack_shape
    x_img = x.view(B, H * W, C).transpose(1, 2).contiguous().view(B, C, H, W)  # [B,C,H,W]
    # 轻量低通：双线性下采样再上采样
    x_low = F.interpolate(F.interpolate(x_img, scale_factor=0.5, mode="bilinear", align_corners=False),
                          size=(H, W), mode="bilinear", align_corners=False)
    x_out = x_low.view(B, C, H * W).transpose(1, 2).contiguous()  # 回到 [B,L,C]
    return x_out


def _load_flowedit_core_backend_symbols():
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from flowedit_multimodel.src.core_backend import FlowBackendConfig, FlowEditCoreBackend

    return FlowBackendConfig, FlowEditCoreBackend

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

class Editsplat_Pipeline(FluxPipeline):
    def configure_edit_backend(self, ed) -> None:
        method = str(getattr(ed, "flow_method", "flowedit")).strip().lower()
        model_key = str(getattr(ed, "flow_model_key", "flux1-dev")).strip()

        self._external_edit_backend = None
        if method in ("native", "native_flowedit"):
            print("[INFO] Using native FLUX FlowEdit path.")
            return
        if method not in ("flowedit", "dnaedit"):
            print(
                f"[WARN] flow_method={method} is not wired into the core multimodel backend yet. "
                "Falling back to the native FLUX path."
            )
            return
        if method == "flowedit" and model_key == "flux1-dev":
            print("[INFO] Using native FLUX FlowEdit path for flux1-dev.")
            return

        FlowBackendConfig, FlowEditCoreBackend = _load_flowedit_core_backend_symbols()
        cfg = FlowBackendConfig(
            model_key=model_key,
            model_id=str(getattr(ed, "flow_model_id", "")).strip(),
            method=method,
            hf_home=str(getattr(ed, "flow_hf_home", "/dev-vepfs/rc_wu/rc_wu/cache/hf_home")).strip(),
            adapter_resize_side=int(getattr(ed, "flow_adapter_resize_side", 512)),
            adapter_gpu=int(getattr(ed, "flow_adapter_gpu", -1)),
            hf_token=os.environ.get("HF_TOKEN", ""),
            dna_steps=int(getattr(ed, "flow_dna_steps", 40)),
            dna_src_guidance_scale=float(getattr(ed, "flow_dna_src_guidance_scale", 1.0)),
            dna_tar_guidance_scale=float(getattr(ed, "flow_dna_tar_guidance_scale", 3.5)),
            dna_t_start=int(getattr(ed, "flow_dna_t_start", 13)),
            dna_mvg=float(getattr(ed, "flow_dna_mvg", 0.8)),
        )
        self._external_edit_backend = FlowEditCoreBackend(config=cfg, project_root=str(Path(__file__).resolve().parent))
        print(
            f"[INFO] External edit backend enabled: method={method}, model_key={model_key}, "
            f"adapter_device={self._external_edit_backend.device}"
        )

    @torch.no_grad()
    def preprocess_like_flowedit(
        self,
        image: Union[Image.Image, torch.Tensor],
        device: torch.device,
        use_autocast: bool = True,
    ) -> torch.Tensor:
        """
        输入:
          - image: 单张 PIL.Image 或 Tensor[B,3,H,W] (0..1)
        输出:
          - x0_src: FlowEdit/FLUX 规范下的“模型用 latent”，形状 [B, C_lat, H_lat, W_lat]
                    计算方式: x0_src = (vae.encode(preprocess(image)).latent_dist.mode() - shift) * scaling
        """
        # A. 保证是单张图（FlowEdit 的脚本逐张处理）
        if isinstance(image, torch.Tensor):
            # 仅支持 B=1，若你有批处理可按需扩展
            assert image.ndim == 4 and image.shape[0] == 1, "请先按 FlowEdit 脚本逐张处理 (B=1)。"
            # 转回 PIL（FlowEdit 用的是 PIL + image_processor）
            # 确保类型在 CPU/float32，避免 bfloat16 -> PIL 报错
            img_pil = Image.fromarray(
                (image[0].detach().cpu().clamp(0,1).permute(1,2,0).numpy() * 255).astype("uint8")
            )
        else:
            img_pil = image

        # B. 裁剪到 16 的整除（FlowEdit 逐句照搬）
        W, H = img_pil.size
        Wc, Hc = W - (W % 16), H - (H % 16)
        if (Wc != W) or (Hc != H):
            img_pil = img_pil.crop((0, 0, Wc, Hc))

        # C. image_processor.preprocess -> Tensor[B,3,H,W]
        #    （它会做 to_tensor/归一化/尺寸对齐等，行为与 FlowEdit 保持）
        image_src = self.image_processor.preprocess(img_pil)   # [1,3,H,W], float32
        image_src = image_src.to(device, dtype=self.vae.dtype) # ✅ 对齐到 VAE 的 dtype（fp16 或 bf16）

        # D. VAE 编码 + “denorm -> model latent” 变换
        #    FlowEdit: mode() + ( - shift_factor ) * scaling_factor
        shift = getattr(self.vae.config, "shift_factor", 0.0)
        scale = getattr(self.vae.config, "scaling_factor", 1.0)

        if use_autocast:
            autocast_ctx = torch.autocast(device_type=str(device).split(":")[0])
        else:
            # 空上下文
            from contextlib import nullcontext
            autocast_ctx = nullcontext()

        with autocast_ctx, torch.inference_mode():
            x0_src_denorm = self.vae.encode(image_src).latent_dist.mode()

        x0_src = (x0_src_denorm - shift) * scale   # ⭐ 关键：与 FlowEdit 完全一致
        return x0_src  # [1, C_lat, H_lat, W_lat]

    @torch.no_grad()
    def postprocess_like_flowedit(
        self,
        x0_tar: torch.Tensor,   # unpack 后的模型 latent（和 FlowEdit 返回的 x0_tar 语义一致）
        device: torch.device,
    ) -> list:
        # 取出 VAE 的 shift/scale
        shift = float(getattr(self.vae.config, "shift_factor", 0.0))
        scale = float(getattr(self.vae.config, "scaling_factor", 1.0))

        # 1) 先在 float32 里做“反缩放”
        x = x0_tar.detach()
        if x.dtype != torch.float32:
            x = x.float()
        x = (x / scale) + shift

        # 2) 解码前把 dtype 对齐到 VAE 权重的 dtype（通常是 fp16 或 bf16）
        x = x.to(self.vae.dtype)
        image_tar = self.vae.decode(x, return_dict=False)[0]    # Tensor[B,3,H,W] ∈ [-1,1]
        images = self.image_processor.postprocess(image_tar)    # List[PIL.Image]
        return images, image_tar

    def edit_image(self,
                image: torch.Tensor,
                src_prompt: str,
                tar_prompt: str,
                negative_prompt: Optional[str] = None,
                diffusion_steps: int = 28,
                n_avg: int = 1,
                src_guidance_scale: float = 1.5,
                tar_guidance_scale: float = 5.5,
                n_min: int = 0,
                n_max: int = 24,
                seed: int = 10,
                lambda_S: float = 0.0,
                mask_S: Optional[torch.Tensor] = None) -> torch.Tensor:
        backend = getattr(self, "_external_edit_backend", None)
        if backend is not None:
            return backend.edit(
                image=image,
                src_prompt=src_prompt,
                tar_prompt=tar_prompt,
                negative_prompt=negative_prompt or "",
                diffusion_steps=diffusion_steps,
                n_avg=n_avg,
                src_guidance_scale=src_guidance_scale,
                tar_guidance_scale=tar_guidance_scale,
                n_min=n_min,
                n_max=n_max,
                seed=seed,
            )

        print("Shape of input image:", image.shape)   # [B,3,H_img,W_img], 0..1
        device = image.device
        torch.manual_seed(seed)

        # === FlowEdit 风格：先把像素图编码到 latent（注意：这一步只做 encode，不做我们自写的 pack/ids）===
        # 你已有 encode_image(self, img, is_sample=False)，复用它
        x_src_lat = self.preprocess_like_flowedit(image, device=device, use_autocast=True)  # [1,C,H',W']          # [B, C_lat, H_lat, W_lat]

        # === 用 latent 尺度推导“orig_height/width”，后面 prepare_latents/_unpack_latents 都用这个语义 ===
        # FlowEdit 的做法：orig = H_lat * vae_scale_factor // 2
        # （在当前 diffusers 的 FLUX 实现里，vae_scale_factor=1 → orig = H_lat//2 = H_tokens）
        orig_height = x_src_lat.shape[2] * self.vae_scale_factor // 2
        orig_width  = x_src_lat.shape[3] * self.vae_scale_factor // 2

        # === FlowEdit 一致：check_inputs 用 orig_height/width（注意：这里的 orig 是基于 latent 推导的）===
        self.check_inputs(
            prompt=src_prompt,
            prompt_2=None,
            height=orig_height,
            width=orig_width,
            callback_on_step_end_tensor_inputs=None,
            max_sequence_length=512,
        )

        # === 关键：完全复用 pipeline 自带的 prepare_latents（把我们“已编码的 latent”作为 latents 传入）===
        num_channels_latents = self.transformer.config.in_channels // 4
        

        x_src_lat, latent_src_image_ids = self.prepare_latents(
            batch_size=x_src_lat.shape[0],
            num_channels_latents=num_channels_latents,
            height=orig_height,
            width=orig_width,
            dtype=x_src_lat.dtype,
            device=x_src_lat.device,
            generator=None,
            latents=x_src_lat,     # 🔴 把“已编码的 latent”交给 pipeline，内部会生成配套的 img_ids
        )


        # === 同样复用 pipeline 的 _pack_latents（以 latent H/W 打包，不要用像素 H/W）===
        x_src_packed = self._pack_latents(
            x_src_lat,
            x_src_lat.shape[0],
            num_channels_latents,
            x_src_lat.shape[2],
            x_src_lat.shape[3],
        )  # -> [B, N, C_tok]，其中 N 应等于 orig_height*orig_width（当前实现下）
        
        # 2) 生成时序（含 seq_len shift）
        scheduler = self.scheduler
        image_seq_len = x_src_packed.shape[1]
        mu = calculate_shift(
            image_seq_len,
            scheduler.config.base_image_seq_len,
            scheduler.config.max_image_seq_len,
            scheduler.config.base_shift,
            scheduler.config.max_shift,
        )
        sigmas = np.linspace(1.0, 1.0 / diffusion_steps, diffusion_steps)              # [1 -> 1/T]
        timesteps, diffusion_steps = retrieve_timesteps(
            scheduler, diffusion_steps, device, timesteps=None, sigmas=sigmas, mu=mu
        )
        self._num_timesteps = len(timesteps)
        pack_shape = (x_src_lat.shape[0],
                      self.transformer.config.in_channels // 4,
                      x_src_lat.shape[2], x_src_lat.shape[3])  # (B,C,H',W')

        # 3) 文本编码 & guidance
        self._guidance_scale = tar_guidance_scale  # 兼容 diffusers 的 “_guidance_scale” 行为
        src_prompt_embeds, src_pooled_prompt_embeds, src_text_ids = self.encode_prompt(
            prompt=src_prompt, prompt_2=None, device=device
        )
        tar_prompt_embeds, tar_pooled_prompt_embeds, tar_text_ids = self.encode_prompt(
            prompt=tar_prompt, prompt_2=None, device=device
        )
        if negative_prompt is not None:
            # 这里按需要可把 negative 编到 embeds 并在 calc_v_flux 内组合；留空=无负提示
            pass

        if self.transformer.config.guidance_embeds:
            src_guidance = torch.tensor([src_guidance_scale], device=device).expand(x_src_packed.shape[0])
            tar_guidance = torch.tensor([tar_guidance_scale], device=device).expand(x_src_packed.shape[0])
        else:
            src_guidance = None
            tar_guidance = None

        # 4) 初始化 ODE 状态
        zt_edit = x_src_packed.clone()  # z_t 的当前估计
        try:
            model_dtype = next(self.transformer.parameters()).dtype
        except StopIteration:
            # 保险：极少数场景下 module 里没参数（基本不会发生）
            model_dtype = torch.bfloat16

        def _to_model_dtype(x):
            return x.to(model_dtype) if (x is not None and torch.is_floating_point(x)) else x

        # 2) 将所有会送入 transformer 的浮点张量，统一到 model_dtype
        #    注意：text_ids / latent_image_ids 是整数，把它们保留为 long 不要改
        x_src_packed = _to_model_dtype(x_src_packed)
        zt_edit      = _to_model_dtype(zt_edit)

        src_prompt_embeds       = _to_model_dtype(src_prompt_embeds)
        src_pooled_prompt_embeds= _to_model_dtype(src_pooled_prompt_embeds)
        tar_prompt_embeds       = _to_model_dtype(tar_prompt_embeds)
        tar_pooled_prompt_embeds= _to_model_dtype(tar_pooled_prompt_embeds)

        if src_guidance is not None:
            src_guidance = src_guidance.to(dtype=model_dtype)
        if tar_guidance is not None:
            tar_guidance = tar_guidance.to(dtype=model_dtype)

        # 5) 主循环：速度差 ODE（前段），SDEdit 风格收尾（后段）
        for i, t in enumerate(timesteps):

            # —— 跳过最早的高噪声步以稳态（可选；对齐 FlowEdit 的 n_max 逻辑）——
            if diffusion_steps - i > n_max:
                continue

            scheduler._init_step_index(t)
            t_i = scheduler.sigmas[scheduler.step_index]
            t_im1 = scheduler.sigmas[scheduler.step_index + 1] if i < len(timesteps) - 1 else t_i

            # (A) ODE 段：仅速度差
            if diffusion_steps - i > n_min:

                V_delta_avg = torch.zeros_like(x_src_packed)
                for _ in range(n_avg):
                    # 源分布前向点/目标对齐点
                    fwd_noise = torch.randn_like(x_src_packed)
                    zt_src = (1.0 - t_i) * x_src_packed + t_i * fwd_noise
                    zt_tar = zt_edit + zt_src - x_src_packed

                    # 速度预测
                    Vt_src = calc_v_flux(self,
                                         latents=zt_src,
                                         prompt_embeds=src_prompt_embeds,
                                         pooled_prompt_embeds=src_pooled_prompt_embeds,
                                         guidance=src_guidance,
                                         text_ids=src_text_ids,
                                         latent_image_ids=latent_src_image_ids,
                                         t=t)
                    Vt_tar = calc_v_flux(self,
                                         latents=zt_tar,
                                         prompt_embeds=tar_prompt_embeds,
                                         pooled_prompt_embeds=tar_pooled_prompt_embeds,
                                         guidance=tar_guidance,
                                         text_ids=tar_text_ids,
                                         latent_image_ids=latent_src_image_ids,
                                         t=t)

                    V_delta_avg = V_delta_avg + (Vt_tar - Vt_src) / float(n_avg)

                # —— 可选：Source fidelity 外力（低频拉回源结构），默认关闭（lambda_S=0）——
                if lambda_S > 0.0:
                    F_S = _lowpass_like(x_src_packed - zt_edit, pack_shape)
                    if mask_S is not None:
                        # mask_S: [B,1,H,W] -> 展平到 [B,L,1] 做门控
                        B, _, H, W = mask_S.shape
                        m = F.interpolate(mask_S, size=(pack_shape[2], pack_shape[3]),
                                          mode="nearest")  # [B,1,H',W']
                        m = m.view(B, 1, pack_shape[2] * pack_shape[3]).transpose(1, 2).contiguous()  # [B,L,1]
                        F_S = F_S * m
                    V_delta_avg = V_delta_avg + lambda_S * F_S

                # Euler 步进
                zt_edit = zt_edit.to(torch.float32)
                zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg.to(torch.float32)
                zt_edit = zt_edit.to(V_delta_avg.dtype)

            # (B) 收尾：仅目标速度（SDEdit-like）
            else:
                if i == diffusion_steps - n_min:
                    # 初始化收尾相位的 x_t
                    fwd_noise = torch.randn_like(x_src_packed)
                    xt_src = scale_noise(scheduler, x_src_packed, t, noise=fwd_noise)
                    xt_tar = zt_edit + xt_src - x_src_packed  # 对齐

                Vt_tar = calc_v_flux(self,
                                     latents=xt_tar,
                                     prompt_embeds=tar_prompt_embeds,
                                     pooled_prompt_embeds=tar_pooled_prompt_embeds,
                                     guidance=tar_guidance,
                                     text_ids=tar_text_ids,
                                     latent_image_ids=latent_src_image_ids,
                                     t=t)
                xt_tar = xt_tar.to(torch.float32)
                xt_tar = xt_tar + (t_im1 - t_i) * Vt_tar.to(torch.float32)
                xt_tar = xt_tar.to(Vt_tar.dtype)

        # 6) 输出（packed -> image latent -> 像素）
        out_packed = zt_edit if n_min == 0 else xt_tar

        # 关键：_unpack_latents 的 height/width 是 token 网格大小（H_tokens/W_tokens），不是像素
        out_latents = self._unpack_latents(out_packed, orig_height, orig_width, self.vae_scale_factor)

        images, image_tar = self.postprocess_like_flowedit(x0_tar=out_latents, device=device)
        return image_tar

    # ------------------ (C) FlowEdit + MFG 一致性 ------------------

    def edit_image_MFG(self,
                    image: torch.Tensor,                 # [B,3,H,W] 源图
                    MF_image_cond: torch.Tensor,         # [B,3,H,W] 融合视图图像
                    src_prompt: str,
                    tar_prompt: str,
                    negative_prompt: Optional[torch.Tensor] = None,
                    diffusion_steps: int = 28,
                    n_avg: int = 1,
                    src_guidance_scale: float = 1.5,
                    tar_guidance_scale: float = 5.5,
                    n_min: int = 0,
                    n_max: int = 24,
                    seed: int = 10,
                    # 一致性外力
                    lambda_S: float = 0.0,
                    lambda_M: float = 0.0,
                    mask_S: Optional[torch.Tensor] = None,
                    mask_M: Optional[torch.Tensor] = None) -> torch.Tensor:
        backend = getattr(self, "_external_edit_backend", None)
        if backend is not None:
            base = MF_image_cond if MF_image_cond is not None else image
            return backend.edit(
                image=base,
                src_prompt=src_prompt,
                tar_prompt=tar_prompt,
                negative_prompt=negative_prompt or "",
                diffusion_steps=diffusion_steps,
                n_avg=n_avg,
                src_guidance_scale=src_guidance_scale,
                tar_guidance_scale=tar_guidance_scale,
                n_min=n_min,
                n_max=n_max,
                seed=seed,
            )
        import torch.nn.functional as F

        device = image.device
        torch.manual_seed(seed)

        # === 源图：FlowEdit 路线的预处理/编码 ===
        x_src_lat = self.preprocess_like_flowedit(image, device=device, use_autocast=True)  # [B,C_lat,H',W']
        orig_height = x_src_lat.shape[2] * self.vae_scale_factor // 2
        orig_width  = x_src_lat.shape[3] * self.vae_scale_factor // 2

        self.check_inputs(
            prompt=src_prompt,
            prompt_2=None,
            height=orig_height,
            width=orig_width,
            callback_on_step_end_tensor_inputs=None,
            max_sequence_length=512,
        )

        num_channels_latents = self.transformer.config.in_channels // 4

        # 将“我们已编码好的 latent”交给 pipeline，让其生成匹配的 image_ids
        x_src_lat, latent_src_image_ids = self.prepare_latents(
            batch_size=x_src_lat.shape[0],
            num_channels_latents=num_channels_latents,
            height=orig_height,
            width=orig_width,
            dtype=x_src_lat.dtype,
            device=x_src_lat.device,
            generator=None,
            latents=x_src_lat,
        )
        x_src_packed = self._pack_latents(
            x_src_lat,
            x_src_lat.shape[0],
            num_channels_latents,
            x_src_lat.shape[2],
            x_src_lat.shape[3],
        )  # [B, L, C_tok]，L ≈ H_tokens*W_tokens

        # === 融合视图：同样路线（但前传时仍复用“源图 image_ids”）===
        x_mf_lat = self.preprocess_like_flowedit(MF_image_cond, device=device, use_autocast=True)
        x_mf_lat, _latent_mf_image_ids = self.prepare_latents(
            batch_size=x_mf_lat.shape[0],
            num_channels_latents=num_channels_latents,
            height=orig_height,
            width=orig_width,
            dtype=x_mf_lat.dtype,
            device=x_mf_lat.device,
            generator=None,
            latents=x_mf_lat,
        )
        x_mf_packed = self._pack_latents(
            x_mf_lat,
            x_mf_lat.shape[0],
            num_channels_latents,
            x_mf_lat.shape[2],
            x_mf_lat.shape[3],
        )

        # === 时间步（含序列长度 shift）===
        scheduler = self.scheduler
        image_seq_len = x_src_packed.shape[1]
        mu = calculate_shift(
            image_seq_len,
            scheduler.config.base_image_seq_len,
            scheduler.config.max_image_seq_len,
            scheduler.config.base_shift,
            scheduler.config.max_shift,
        )
        sigmas = np.linspace(1.0, 1.0 / diffusion_steps, diffusion_steps)
        timesteps, diffusion_steps = retrieve_timesteps(
            scheduler, diffusion_steps, device, timesteps=None, sigmas=sigmas, mu=mu
        )
        self._num_timesteps = len(timesteps)
        pack_shape = (x_src_lat.shape[0],
                    self.transformer.config.in_channels // 4,
                    x_src_lat.shape[2], x_src_lat.shape[3])  # (B,C,H_tokens,W_tokens)

        # === 文本编码 & guidance ===
        self._guidance_scale = tar_guidance_scale  # 与 diffusers 约定保持一致
        src_prompt_embeds, src_pooled_prompt_embeds, src_text_ids = self.encode_prompt(
            prompt=src_prompt, prompt_2=None, device=device
        )
        tar_prompt_embeds, tar_pooled_prompt_embeds, tar_text_ids = self.encode_prompt(
            prompt=tar_prompt, prompt_2=None, device=device
        )
        if self.transformer.config.guidance_embeds:
            src_guidance = torch.tensor([src_guidance_scale], device=device).expand(x_src_packed.shape[0])
            tar_guidance = torch.tensor([tar_guidance_scale], device=device).expand(x_src_packed.shape[0])
        else:
            src_guidance = None
            tar_guidance = None

        # === dtype 统一：与 edit_image 同步 ===
        try:
            model_dtype = next(self.transformer.parameters()).dtype
        except StopIteration:
            model_dtype = torch.bfloat16

        def _to_model_dtype(x):
            return x.to(model_dtype) if (x is not None and torch.is_floating_point(x)) else x

        x_src_packed = _to_model_dtype(x_src_packed)
        x_mf_packed  = _to_model_dtype(x_mf_packed)
        zt_edit      = _to_model_dtype(x_src_packed.clone())

        src_prompt_embeds        = _to_model_dtype(src_prompt_embeds)
        src_pooled_prompt_embeds = _to_model_dtype(src_pooled_prompt_embeds)
        tar_prompt_embeds        = _to_model_dtype(tar_prompt_embeds)
        tar_pooled_prompt_embeds = _to_model_dtype(tar_pooled_prompt_embeds)

        if src_guidance is not None:
            src_guidance = src_guidance.to(dtype=model_dtype)
        if tar_guidance is not None:
            tar_guidance = tar_guidance.to(dtype=model_dtype)

        # === 主循环 ===
        xt_tar = None  # 若进入尾段会被赋值
        for i, t in enumerate(timesteps):

            # 跳过最早的高噪声步（n_max 逻辑）
            if diffusion_steps - i > n_max:
                continue

            scheduler._init_step_index(t)
            t_i = scheduler.sigmas[scheduler.step_index]
            t_im1 = scheduler.sigmas[scheduler.step_index + 1] if i < len(timesteps) - 1 else t_i

            # (A) ODE 段：速度差 + 可选外力
            if diffusion_steps - i > n_min:

                V_delta_avg = torch.zeros_like(x_src_packed)
                for _ in range(n_avg):
                    fwd_noise = torch.randn_like(x_src_packed)
                    zt_src = (1.0 - t_i) * x_src_packed + t_i * fwd_noise
                    zt_tar = zt_edit + zt_src - x_src_packed

                    Vt_src = calc_v_flux(
                        self,
                        latents=zt_src,
                        prompt_embeds=src_prompt_embeds,
                        pooled_prompt_embeds=src_pooled_prompt_embeds,
                        guidance=src_guidance,
                        text_ids=src_text_ids,
                        latent_image_ids=latent_src_image_ids,
                        t=t,
                    )
                    Vt_tar = calc_v_flux(
                        self,
                        latents=zt_tar,
                        prompt_embeds=tar_prompt_embeds,
                        pooled_prompt_embeds=tar_pooled_prompt_embeds,
                        guidance=tar_guidance,
                        text_ids=tar_text_ids,
                        latent_image_ids=latent_src_image_ids,  # 复用源图的 ids（FlowEdit 做法）
                        t=t,
                    )
                    V_delta_avg = V_delta_avg + (Vt_tar - Vt_src) / float(n_avg)

                # 源外观/结构保持（可选）
                if lambda_S > 0.0:
                    F_S = _lowpass_like(x_src_packed - zt_edit, pack_shape)
                    if mask_S is not None:
                        B, _, H, W = mask_S.shape
                        m = F.interpolate(mask_S, size=(pack_shape[2], pack_shape[3]), mode="nearest")
                        m = m.view(B, 1, pack_shape[2] * pack_shape[3]).transpose(1, 2).contiguous()
                        F_S = F_S * m
                    V_delta_avg = V_delta_avg + lambda_S * F_S

                # MFG 多视图一致性（可选）
                if lambda_M > 0.0:
                    F_M = _lowpass_like(x_mf_packed - zt_edit, pack_shape)
                    if mask_M is not None:
                        B, _, H, W = mask_M.shape
                        m = F.interpolate(mask_M, size=(pack_shape[2], pack_shape[3]), mode="nearest")
                        m = m.view(B, 1, pack_shape[2] * pack_shape[3]).transpose(1, 2).contiguous()
                        F_M = F_M * m
                    V_delta_avg = V_delta_avg + lambda_M * F_M

                # Euler 更新：float32 计算后再转回，避免数值不稳
                zt_edit = zt_edit.to(torch.float32)
                zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg.to(torch.float32)
                zt_edit = zt_edit.to(V_delta_avg.dtype)

            # (B) 尾段：仅用目标速度作 SDEdit 式细化
            else:
                if i == diffusion_steps - n_min:
                    fwd_noise = torch.randn_like(x_src_packed)
                    xt_src = scale_noise(scheduler, x_src_packed, t, noise=fwd_noise)
                    xt_tar = zt_edit + xt_src - x_src_packed

                Vt_tar = calc_v_flux(
                    self,
                    latents=xt_tar,
                    prompt_embeds=tar_prompt_embeds,
                    pooled_prompt_embeds=tar_pooled_prompt_embeds,
                    guidance=tar_guidance,
                    text_ids=tar_text_ids,
                    latent_image_ids=latent_src_image_ids,  # 仍然用源 ids
                    t=t,
                )
                xt_tar = xt_tar.to(torch.float32)
                xt_tar = xt_tar + (t_im1 - t_i) * Vt_tar.to(torch.float32)
                xt_tar = xt_tar.to(Vt_tar.dtype)

        out_packed = zt_edit if n_min == 0 else xt_tar

        # 解包：这里的 H/W 必须是 token 网格尺寸（不是像素）
        out_latents = self._unpack_latents(out_packed, orig_height, orig_width, self.vae_scale_factor)

        # 后处理：与 FlowEdit 路径一致（已验证更稳、更不易黑图）
        images, image_tar = self.postprocess_like_flowedit(x0_tar=out_latents, device=device)
        return image_tar
    
    ############ Score Distillation Sampling 相关 ############
    def set_sds_params(self, sdp):
        """
        把解析好的 ScoreDistillParams 实例/字典塞进 pipeline，便于在训练循环里直接用。
        """
        self._sds_cfg = sdp
        self._sds_cache = None  # 会在第一次调用时构建

    @torch.no_grad()
    def _build_sds_prompt_cache(self, src_prompt: str, tar_prompt: str, device: torch.device):
        """
        仅在首次调用时编码一次文本，把需要的 embedding/guidance 常量缓存起来。
        - 注意：FLUX.1-dev 是 guidance-distilled，diffusers<=0.30 下 transformer.config.guidance_embeds 为 True。
        """
        src_embeds, src_pooled, src_ids = self.encode_prompt(prompt=src_prompt, prompt_2=None, device=device)
        tar_embeds, tar_pooled, tar_ids = self.encode_prompt(prompt=tar_prompt, prompt_2=None, device=device)

        if self.transformer.config.guidance_embeds:
            src_guid = torch.tensor([self._sds_cfg.src_guidance], device=device, dtype=src_embeds.dtype)
            tar_guid = torch.tensor([self._sds_cfg.tar_guidance], device=device, dtype=tar_embeds.dtype)
        else:
            src_guid = None
            tar_guid = None

        self._sds_cache = dict(
            src_embeds=src_embeds, src_pooled=src_pooled, src_ids=src_ids, src_guid=src_guid,
            tar_embeds=tar_embeds, tar_pooled=tar_pooled, tar_ids=tar_ids, tar_guid=tar_guid
        )

    def _resize_for_flux(self, img_bchw: torch.Tensor, side: int) -> torch.Tensor:
        """
        把 [-1,1] 或 [0,1] 的张量统一到 [0,1]，再 resize 到 side×side（保持可微插值），B=1。
        """
        assert img_bchw.ndim == 4 and img_bchw.shape[0] == 1
        x = img_bchw
        if x.min() < 0.0:     # 从渲染器回来通常在 [-1,1]
            x = (x + 1.0) * 0.5
        if (x.shape[-2] != side) or (x.shape[-1] != side):
            x = F.interpolate(x, size=(side, side), mode="bilinear", align_corners=True)
        return x.clamp(0, 1)

    def _img_to_packed_latents_and_ids(self, image_bchw_01: torch.Tensor, device: torch.device):
        """
        复用你已有的 FlowEdit 前处理 → VAE 编码 → prepare_latents → _pack_latents
        返回：
        x_lat      : [1, C, H', W'] （模型 latent）
        img_ids    : [L] long，FLUX 需要的 image token ids
        x_packed   : [1, L, C_tok] 打包后的 token
        H_tokens,W_tokens: token 网格（等于 orig_height/width）
        timesteps_builder: 一个闭包，给你 image_seq_len 后能生成 (mu, timesteps)
        """
        device = image_bchw_01.device
        # VAE latent
        x_lat = self.preprocess_like_flowedit(image_bchw_01, device=device, use_autocast=True)  # [1,C,H',W']

        # 由 latent 尺度反推 token 尺度（和你 edit_image 里一致）
        orig_h = x_lat.shape[2] * self.vae_scale_factor // 2
        orig_w = x_lat.shape[3] * self.vae_scale_factor // 2

        # 走标准的 check_inputs / prepare_latents（把我们自己的 latent 作为 latents 传入）
        self.check_inputs(
            prompt="(cache only)", prompt_2=None, height=orig_h, width=orig_w,
            callback_on_step_end_tensor_inputs=None, max_sequence_length=512,
        )
        num_channels_latents = self.transformer.config.in_channels // 4
        x_lat, img_ids = self.prepare_latents(
            batch_size=1, num_channels_latents=num_channels_latents,
            height=orig_h, width=orig_w, dtype=x_lat.dtype, device=device,
            generator=None, latents=x_lat,
        )
        # 打包
        x_packed = self._pack_latents(
            x_lat, x_lat.shape[0], num_channels_latents, x_lat.shape[2], x_lat.shape[3]
        )
        H_tokens, W_tokens = orig_h, orig_w

        # 构造和你 edit_image 完全一致的时序生成器（含 seq_len shift）
        def build_timesteps(image_seq_len: int, diffusion_steps: int):
            scheduler = self.scheduler
            mu = calculate_shift(
                image_seq_len,
                scheduler.config.base_image_seq_len,
                scheduler.config.max_image_seq_len,
                scheduler.config.base_shift,
                scheduler.config.max_shift,
            )
            sigmas = np.linspace(1.0, 1.0 / diffusion_steps, diffusion_steps)  # [1 .. 1/T]
            timesteps, diffusion_steps_eff = retrieve_timesteps(
                scheduler, diffusion_steps, device, timesteps=None, sigmas=sigmas, mu=mu
            )
            return scheduler, mu, timesteps, diffusion_steps_eff

        return x_lat, img_ids, x_packed, H_tokens, W_tokens, build_timesteps

    # ========= 2) 基于 FlowEdit / FlowAlign / PDS 的 3DGS 用“分数蒸馏”损失 =========

    def compute_flow_sds_loss(
        self,
        rendered_bchw: torch.Tensor,      # [1,3,H,W]，来自 3DGS 的当前渲染（需要反传）
        src_bchw: torch.Tensor,           # [1,3,H,W]，同视角 GT（身份保持用）
        src_prompt: str,
        tar_prompt: str,
        sdp,
        mask_b1hw: Optional[torch.Tensor] = None,  # LangSAM mask，可为 [H,W]/[1,H,W]/[1,1,H,W]
    ):
        device = rendered_bchw.device
        # 组态 & 缓存
        self._sds_cfg = sdp if getattr(self, "_sds_cfg", None) is None else self._sds_cfg
        if self._sds_cache is None:
            self._build_sds_prompt_cache(src_prompt, tar_prompt, device)

        # 统一分辨率（仅用于进入 VAE/FLUX 的路径；不改变原渲染图 tensor）
        side = int(getattr(self._sds_cfg, "resize", 512))
        img_tar_01 = self._resize_for_flux(rendered_bchw, side)
        img_src_01 = self._resize_for_flux(src_bchw, side)

        # 编码 → 打包 → 得到 token 级特征与 ids
        x_lat_tar, img_ids_tar, x_tar_packed, Htok, Wtok, build_ts = self._img_to_packed_latents_and_ids(img_tar_01, device)
        x_lat_src, _img_ids_src, x_src_packed, _, _, _ = self._img_to_packed_latents_and_ids(img_src_01, device)
        latent_image_ids = img_ids_tar  # 对应 token 的 ids

        # 构造时间步（保持与 FlowEdit 的 shift / sigmas 逻辑一致）
        scheduler, mu, timesteps, T = build_ts(x_tar_packed.shape[1], int(self._sds_cfg.timesteps))
        n_min = int(self._sds_cfg.n_min); n_max = int(self._sds_cfg.n_max)
        valid_idx = list(range(max(0, T - n_max), max(1, T - n_min)))
        t_idx = random.choice(valid_idx)
        t = timesteps[t_idx]
        scheduler._init_step_index(t)
        t_i = scheduler.sigmas[scheduler.step_index]
        t_ip1 = scheduler.sigmas[scheduler.step_index + 1] if t_idx < len(timesteps) - 1 else t_i

        # 共享噪声（PDS/DDS）
        fwd_noise = torch.randn_like(x_tar_packed)
        xt_tar = (1.0 - t_i) * x_tar_packed + t_i * fwd_noise
        xt_src = (1.0 - t_i) * x_src_packed + t_i * fwd_noise

        cache = self._sds_cache
        model_dtype = next(self.transformer.parameters()).dtype
        def _to(x): 
            return x.to(model_dtype) if (x is not None and torch.is_floating_point(x)) else x

        V_src = self.calc_v_flux(
            latents=_to(xt_src),
            prompt_embeds=_to(cache["src_embeds"]),
            pooled_prompt_embeds=_to(cache["src_pooled"]),
            guidance=_to(cache["src_guid"]),
            text_ids=cache["src_ids"],
            latent_image_ids=latent_image_ids,
            t=t,
        )
        V_tar = self.calc_v_flux(
            latents=_to(xt_tar),
            prompt_embeds=_to(cache["tar_embeds"]),
            pooled_prompt_embeds=_to(cache["tar_pooled"]),
            guidance=_to(cache["tar_guid"]),
            text_ids=cache["tar_ids"],
            latent_image_ids=latent_image_ids,
            t=t,
        )
        dV = (V_tar - V_src)

        # 时间权重
        if str(getattr(self._sds_cfg, "time_weight", "one")) == "poly":
            wt = float((1.0 - float(t_i)) ** 2)
        else:
            wt = 1.0
        wt = torch.tensor(wt, device=device, dtype=xt_tar.dtype)

        # ---- LangSAM 掩码：仅在损失里做 gating（不改变前向）----
        m_tok = None
        if mask_b1hw is not None:
            m = mask_b1hw
            # 归一形状 -> [1,1,H,W]
            if m.ndim == 2:
                m = m.unsqueeze(0).unsqueeze(0)
            elif m.ndim == 3:
                m = m.unsqueeze(0) if m.shape[0] != 1 else m.unsqueeze(1)  # [1,H,W] 或 [C,H,W]→尽量变 [1,1,H,W]
                if m.ndim == 3:
                    m = m.unsqueeze(1)
            m = m.to(device=device, dtype=xt_tar.dtype)

            # 可选软背景权
            bg = float(getattr(self._sds_cfg, "mask_bg", 0.0))
            if bg > 0.0:
                hard = (m > 0.5).to(m.dtype)
                m = hard + bg * (1.0 - hard)

            # 映射到 token 网格并展平为 [1, L, 1]
            m = F.interpolate(m, size=(Htok, Wtok), mode="nearest")
            m_tok = m.view(1, 1, Htok * Wtok).transpose(1, 2).contiguous()

        # --- 蒸馏项（SDS/DDS 的 stop-grad 在线性化到 flow 速度差）---
        if m_tok is None:
            L_edit = wt * (dV.detach() * xt_tar).mean()
        else:
            L_edit = wt * ( (dV.detach() * xt_tar) * m_tok ).sum() / (m_tok.sum() * xt_tar.shape[-1] + 1e-6)

        # --- Tweedie 身份保持（FlowAlign F.2）：x0_hat = x_t - t * V(·) ---
        x0_tar = xt_tar - t_i * V_tar
        x0_src = xt_src - t_i * V_src
        if m_tok is None:
            L_id = F.l1_loss(x0_tar, x0_src.detach())
        else:
            L_id = ( (x0_tar - x0_src.detach()).abs() * m_tok ).sum() / (m_tok.sum() * x0_tar.shape[-1] + 1e-6)

        loss = float(self._sds_cfg.w_edit) * L_edit + float(self._sds_cfg.w_id) * L_id
        return loss
    
    
    def __call__(
        self,
        dataset = None,
        opt = None,
        pipe = None,
        ed = None,
        sdp = None,
    ):
        if getattr(self, "_external_edit_backend", None) is None and ed is not None:
            self.configure_edit_backend(ed)

        # set scheduler TODO
        # self.scheduler = DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler", torch_dtype=torch.bfloat16)
        # self.num_train_timesteps = 1000
        # self.alphas = self.scheduler.alphas_cumprod.to(self._execution_device)

        # set unet to save cross-attention map TODO
        # ip2p_pipe = StableDiffusionPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.bfloat16).to(device)
        # ip2p_pipe.unet = prep_unet(ip2p_pipe.unet)
        # ip2p_pipe.unet.eval()
        # ip2p_pipe.unet.requires_grad_(False)

        # set weights dtype to bfloat16 TODO
        # self.weights_dtype=torch.bfloat16
        # ip2p_pipe.unet = ip2p_pipe.unet.to(self.vae.dtype)

        # encode target prompt 这部分不用encode，直接输入 TODO
        # trg_prompt_embeds = self._encode_prompt(
        #     ed.target_prompt, device=self._execution_device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=""
        # )
        self.transformer = self.transformer.to(torch.bfloat16)
        self.transformer.eval()
        self.transformer.requires_grad_(False)
        # load ImageReward
        reward_model = _load_reward_model()

        # load Lang-SAM
        lang_sam = _load_langsam()

        # load 3D Gaussian Splatting
        gaussians = GaussianModel(dataset.sh_degree)

        scene = Scene(dataset, gaussians)

        gaussians.training_setup(opt)

        if dataset.source_checkpoint:
            # Avoid cross-GPU restore/OOM from serialized CUDA device ids in checkpoint.
            (model_params, first_iter) = torch.load(dataset.source_checkpoint, map_location="cpu")
            restore_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            gaussians.restore(model_params, opt, device=restore_device)
            start_iteration = first_iter

        max_gaussians = _env_int("EDITSPLAT_MAX_GAUSSIANS", 0)
        if max_gaussians > 0:
            xyz = gaussians.get_xyz
            total_gaussians = int(xyz.shape[0])
            if total_gaussians > max_gaussians:
                # Deterministic stride subsampling for stability / memory control.
                device_idx = xyz.device
                keep = torch.zeros(total_gaussians, dtype=torch.bool, device=device_idx)
                stride = max(1, total_gaussians // max_gaussians)
                keep_idx = torch.arange(0, total_gaussians, stride, device=device_idx)[:max_gaussians]
                keep[keep_idx] = True
                gaussians.prune_points(~keep)
                print(
                    f"[SPARSE] Gaussian count pruned: {total_gaussians} -> {gaussians.get_xyz.shape[0]} "
                    f"(EDITSPLAT_MAX_GAUSSIANS={max_gaussians})"
                )

        # In multimodel wrappers, pipeline._execution_device can become CPU when
        # some components are offloaded. 3DGS ops must stay on the Gaussian device.
        render_device = gaussians.get_xyz.device
        if not isinstance(render_device, torch.device):
            render_device = torch.device(render_device)
        if render_device.type == "cpu" and torch.cuda.is_available():
            render_device = torch.device("cuda")

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=render_device)

        # for multi-view attention weighting
        attn_list = []

        # utility setting
        topilimage = ToPILImage()

        # LPIPS Loss
        lpips_loss_fn = _build_lpips_loss(render_device)
        debug_root = _prepare_debug_root(dataset.model_path)
        mfg_mode = _env_choice("EDITSPLAT_MFG_MODE", "full")
        mfg_backfill = _env_choice("EDITSPLAT_MFG_BACKFILL", "nearest")
        mfg_source_count = max(1, _env_int("EDITSPLAT_MFG_SOURCE_COUNT", 5))

        # Get the training dataset
        train_dataset = CameraDataset(scene)
        max_train_views = _env_int("EDITSPLAT_MAX_TRAIN_VIEWS", 0)
        if max_train_views > 0 and hasattr(train_dataset, "camera_list"):
            n0 = len(train_dataset.camera_list)
            train_dataset.camera_list = train_dataset.camera_list[:max_train_views]
            print(f"[SPARSE] CameraDataset truncated: {n0} -> {len(train_dataset.camera_list)} views")
        # train_dataset = HeadCameraDataset(train_dataset, 2)

        # DataLoaders creation:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=dataset.view_shuffling, num_workers=0
        )

        # train_dataloader = train_dataloader[:1]

        # Get Camera distance matrix
        camera_list = train_dataset.camera_list
        camera_dist_order, _ = find_nearby_camera(camera_list)

        image_height = camera_list[0].image_height
        image_width = camera_list[0].image_width

        # Initially edit all images
        with torch.no_grad():
            
            edited_image_list = []
            edited_image_pil_list_RM = []
            rendered_depth_list = []
            is_top_selection = []
            initial_edit_scores = []

            for step, batch in enumerate(tqdm(train_dataloader, desc="Initial editing progress")):
                
                gt_image = batch['gt_image'].to(render_device)
                idx = batch['idx'].item()

                # reset_attention_maps(ip2p_pipe.unet) #  TODO

                if gt_image.shape[2] != 512 or gt_image.shape[3] != 512:
                    gt_image = F.interpolate(gt_image, size=(512, 512), mode='bilinear', align_corners=True)
                edited_image = self.edit_image( # torch.Size([1, 3, 512, 512]) TODO 这部分输入改了很多，注意这个
                image=gt_image,
                src_prompt=ed.flow_src_prompt,
                tar_prompt=ed.flow_tar_prompt,
                diffusion_steps=ed.flow_steps,
                n_avg=ed.flow_n_avg,
                src_guidance_scale=ed.flow_src_guidance_scale,
                tar_guidance_scale=ed.flow_tar_guidance_scale,
                negative_prompt=ed.flow_negative_prompt,
                n_min=ed.flow_n_min,
                n_max=ed.flow_n_max,
                seed=ed.flow_seed,
                lambda_S=0,
                mask_S=None
                )

                # Save edited image to list
                edited_image = F.interpolate(edited_image, size=(image_height, image_width), mode='bilinear', align_corners=True).to(torch.float32)

                edited_image_list.append(edited_image.squeeze(0).detach().cpu().clone())
                initial_edit_scores.append(_initial_edit_diff_score(edited_image, gt_image))

                # save pil image for imagereward sampling
                edited_pil = topilimage(edited_image.squeeze(0))
                edited_image_pil_list_RM.append(edited_pil)
                _dump_stage_payload(
                    debug_root=debug_root,
                    stage="initial_edit",
                    view_idx=idx,
                    payload={
                        "input": gt_image,
                        "edited": edited_image,
                    },
                )
            
            depth_mode = os.environ.get("EDITSPLAT_DEPTH_MODE", "render").strip().lower()
            # Depth stage only needs view indices; avoid another dataloader collate pass on tensors.
            for idx in tqdm(range(len(camera_list)), desc="Depth processing progress"):
                if depth_mode == "constant":
                    rendered_depth_list.append(torch.ones((image_height, image_width), dtype=torch.float32))
                    continue

                # for k in ['xyz','scales','rotations','opacities','features_dc','features_rest']:
                #     gaussians[k] = gaussians[k].to(self._execution_device, dtype=torch.float32).contiguous()
                # background = background.to(self._execution_device, dtype=torch.float32).contiguous()
                try:
                    # render depth map from pretrained 3dgs
                    render_pkg = render(camera_list[idx], gaussians, pipe, background)
                    depth_3d = render_pkg["depth_3dgs"]
                    rendered_depth_list.append(depth_3d.detach().squeeze().cpu().clone())
                except Exception as e:
                    print(f"[WARN] Depth rendering failed at idx={idx}: {e}. Using constant depth fallback.")
                    rendered_depth_list.append(torch.ones((image_height, image_width), dtype=torch.float32))
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.empty_cache()
                        except Exception as cache_exc:
                            print(f"[WARN] torch.cuda.empty_cache() failed after depth fallback: {cache_exc}")
            
            # Filtering the edited images using ImageReward
            # get ranking and rewards
            with torch.cuda.amp.autocast(dtype=torch.float32):
                ranking, rewards = reward_model.inference_rank(ed.sampling_prompt, edited_image_pil_list_RM)

            is_top_selection, selection_meta = _resolve_reward_selection(
                ranking=ranking,
                rewards=rewards,
                filtering_ratio=ed.filtering_ratio,
                num_items=len(edited_image_pil_list_RM),
            )
            selection_meta["sampling_prompt"] = ed.sampling_prompt
            selection_meta["selected_indices"] = [idx for idx, flag in enumerate(is_top_selection) if flag]
            print(
                f"[IR] mode={selection_meta['filter_mode']} keep_count={selection_meta['keep_count']} "
                f"selected={selection_meta['selected_indices']}"
            )
            if debug_root is not None:
                (debug_root / "selection" / "selection.json").write_text(
                    json.dumps(selection_meta, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

        frontier_anchor_idx = None
        frontier_neighbor_set = set()
        frontier_anchor_mask = None
        frontier_fallback_enabled = _env_flag("EDITSPLAT_FRONTIER_BLACKFACE_FALLBACK", False)
        frontier_fallback_mean_thr = _env_float("EDITSPLAT_FRONTIER_BLACKFACE_MEAN_THR", 0.08)
        frontier_fallback_std_thr = _env_float("EDITSPLAT_FRONTIER_BLACKFACE_STD_THR", 0.02)
        frontier_fallback_min_coverage = _env_float("EDITSPLAT_FRONTIER_BLACKFACE_MIN_COVERAGE", 0.005)
        frontier_fallback_feather = _env_int("EDITSPLAT_FRONTIER_FALLBACK_FEATHER", 9)
        frontier_anchor_override_path = os.environ.get("EDITSPLAT_FRONTIER_ANCHOR_OVERRIDE_IMAGE", "").strip()
        frontier_anchor_override_mode = _env_choice("EDITSPLAT_FRONTIER_ANCHOR_OVERRIDE_MODE", "replace")
        frontier_anchor_override_feather = _env_int(
            "EDITSPLAT_FRONTIER_ANCHOR_OVERRIDE_FEATHER",
            frontier_fallback_feather,
        )
        frontier_anchor_override_image = _load_anchor_override_image(
            frontier_anchor_override_path,
            image_height=image_height,
            image_width=image_width,
        )
        if mfg_mode == "frontier_seed1":
            frontier_candidates = [idx for idx, flag in enumerate(is_top_selection) if flag]
            frontier_anchor_idx = _pick_frontier_anchor(initial_edit_scores, frontier_candidates)
            frontier_neighbor_set = {
                int(x)
                for x in camera_dist_order[frontier_anchor_idx][1 : 1 + max(1, mfg_source_count)]
            }
            frontier_meta = {
                "anchor_idx": int(frontier_anchor_idx),
                "anchor_score": float(initial_edit_scores[frontier_anchor_idx]),
                "neighbor_indices": sorted(int(x) for x in frontier_neighbor_set),
                "candidate_indices": [int(x) for x in frontier_candidates],
                "anchor_override_image": frontier_anchor_override_path or None,
                "anchor_override_mode": frontier_anchor_override_mode if frontier_anchor_override_image is not None else None,
            }
            print(
                "[FRONTIER] "
                f"anchor={frontier_meta['anchor_idx']} "
                f"score={frontier_meta['anchor_score']:.6f} "
                f"neighbors={frontier_meta['neighbor_indices']}"
            )
            if debug_root is not None:
                (debug_root / "selection" / "frontier_seed1.json").write_text(
                    json.dumps(frontier_meta, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

            if frontier_anchor_override_image is not None:
                anchor_batch = train_dataset[frontier_anchor_idx]
                anchor_gt_image = anchor_batch["gt_image"]
                if anchor_gt_image.ndim == 3:
                    anchor_gt_image = anchor_gt_image.unsqueeze(0)
                anchor_gt_image = anchor_gt_image.to(render_device, dtype=torch.float32)
                if anchor_gt_image.shape[2] != image_height or anchor_gt_image.shape[3] != image_width:
                    anchor_gt_image = F.interpolate(
                        anchor_gt_image,
                        size=(image_height, image_width),
                        mode="bilinear",
                        align_corners=True,
                    )
                anchor_gt_image_np = (
                    anchor_gt_image.detach().cpu().numpy().squeeze(0).transpose(1, 2, 0).clip(0, 1)
                )
                anchor_gt_image_pil = Image.fromarray((anchor_gt_image_np * 255).astype(np.uint8))
                frontier_anchor_mask = _predict_langsam_mask(
                    lang_sam=lang_sam,
                    image_pil=anchor_gt_image_pil,
                    text_prompt=ed.target_mask_prompt,
                    image_height=image_height,
                    image_width=image_width,
                    mask_role="gt_view",
                ).detach().cpu().clone()
                existing_anchor = edited_image_list[frontier_anchor_idx].unsqueeze(0).to(
                    device=render_device,
                    dtype=torch.float32,
                )
                override_anchor = frontier_anchor_override_image.to(device=render_device, dtype=torch.float32)
                if frontier_anchor_override_mode in {"replace", "raw", "full"}:
                    anchor_image = override_anchor
                elif frontier_anchor_override_mode in {"face_on_source", "source"}:
                    anchor_image = _blend_face_override(
                        anchor_gt_image,
                        override_anchor,
                        frontier_anchor_mask.to(render_device),
                        feather_radius=frontier_anchor_override_feather,
                    )
                elif frontier_anchor_override_mode in {"face_on_existing", "existing", "flow"}:
                    anchor_image = _blend_face_override(
                        existing_anchor,
                        override_anchor,
                        frontier_anchor_mask.to(render_device),
                        feather_radius=frontier_anchor_override_feather,
                    )
                else:
                    print(
                        "[WARN] Unknown EDITSPLAT_FRONTIER_ANCHOR_OVERRIDE_MODE="
                        f"{frontier_anchor_override_mode}; keeping existing frontier anchor."
                    )
                    anchor_image = existing_anchor
                edited_image_list[frontier_anchor_idx] = anchor_image.squeeze(0).detach().cpu().clone()


        """Multi-View Fusion Guidance (MFG)"""
        edited_image_MFG_list = []

        for step, batch in enumerate(tqdm(train_dataloader, desc="Multi-view reprojection progress")):

            gt_image = batch['gt_image'].to(render_device) # [1, 3, 512,512]
            idx = batch['idx'].item() # current camera index

            gt_image = F.interpolate(gt_image, size=(image_height, image_width), mode='bilinear', align_corners=True) 

            if mfg_mode == "frontier_seed1" and frontier_anchor_idx is not None:
                is_anchor_view = idx == frontier_anchor_idx
                is_frontier_neighbor = idx in frontier_neighbor_set
                if is_anchor_view or not is_frontier_neighbor:
                    if is_anchor_view and frontier_anchor_mask is not None:
                        gt_mask = frontier_anchor_mask.clone()
                    else:
                        gt_image_np = gt_image.detach().cpu().numpy().squeeze(0).transpose(1, 2, 0).clip(0, 1)
                        gt_image_pil = Image.fromarray((gt_image_np * 255).astype(np.uint8))
                        gt_mask = _predict_langsam_mask(
                            lang_sam=lang_sam,
                            image_pil=gt_image_pil,
                            text_prompt=ed.target_mask_prompt,
                            image_height=image_height,
                            image_width=image_width,
                            mask_role="gt_view",
                        )
                    edited_image_MFG = edited_image_list[idx].unsqueeze(0).to(torch.float32)
                    if is_anchor_view:
                        frontier_anchor_mask = gt_mask.detach().cpu().clone()
                    attn_list.append(gt_mask.to(render_device))
                    edited_image_MFG_list.append(edited_image_MFG.squeeze(0).detach().cpu().clone())
                    _dump_stage_payload(
                        debug_root=debug_root,
                        stage="mfg_edit",
                        view_idx=idx,
                        payload={
                            "gt": gt_image,
                            "initial_edit": edited_image_MFG,
                            "gt_mask": gt_mask,
                            "mfg_output": edited_image_MFG,
                        },
                        meta={
                            "mfg_mode": mfg_mode,
                            "frontier_role": "anchor" if is_anchor_view else "context_passthrough",
                            "frontier_anchor_idx": int(frontier_anchor_idx),
                            "anchor_override_image": frontier_anchor_override_path or None,
                            "anchor_override_mode": frontier_anchor_override_mode if is_anchor_view else None,
                            "source_indices": [int(idx)],
                        },
                    )
                    continue

            if mfg_mode == "initial_only":
                full_mask = torch.ones((1, image_height, image_width), dtype=torch.float32, device=render_device)
                edited_image_MFG = edited_image_list[idx].unsqueeze(0).to(torch.float32)
                attn_list.append(full_mask)
                edited_image_MFG_list.append(edited_image_MFG.squeeze(0).detach().cpu().clone())
                _dump_stage_payload(
                    debug_root=debug_root,
                    stage="mfg_edit",
                    view_idx=idx,
                    payload={
                        "gt": gt_image,
                        "initial_edit": edited_image_MFG,
                        "mfg_output": edited_image_MFG,
                    },
                    meta={
                        "mfg_mode": mfg_mode,
                        "source_indices": [int(idx)],
                    },
                )
                continue

            # reprojecting
            with torch.cuda.amp.autocast(dtype=torch.float32):
                
                src_cam_idx_list = []
                dst_cam_idx = idx

                if mfg_mode == "frontier_seed1" and frontier_anchor_idx is not None and idx in frontier_neighbor_set:
                    src_cam_idx_list = [int(frontier_anchor_idx)]
                else:
                    # Prefer top-ranked source views; then backfill from nearest views.
                    # This avoids dead loops when very few views are marked as top.
                    for camera_idx in camera_dist_order[idx][1:]:
                        if is_top_selection[camera_idx]:
                            src_cam_idx_list.append(camera_idx)
                        if len(src_cam_idx_list) >= mfg_source_count:
                            break

                    if len(src_cam_idx_list) < mfg_source_count and mfg_backfill != "selected_only":
                        for camera_idx in camera_dist_order[idx][1:]:
                            if camera_idx not in src_cam_idx_list:
                                src_cam_idx_list.append(camera_idx)
                            if len(src_cam_idx_list) >= mfg_source_count:
                                break

                if len(src_cam_idx_list) == 0:
                    src_cam_idx_list = [dst_cam_idx]

                while len(src_cam_idx_list) < mfg_source_count:
                    src_cam_idx_list.append(src_cam_idx_list[-1])
                
                dst_camera = camera_list[dst_cam_idx]
                reprejected_pixels_list = []
                reprejected_colors_list = []

                for camera_idx in src_cam_idx_list:
                    camera = camera_list[camera_idx]

                    color = edited_image_list[camera_idx].detach()
                    depth = rendered_depth_list[camera_idx].squeeze()

                    reprejected_points, reprejected_colors = reproject_rgbd(
                        camera,
                        dst_camera,
                        color.to(render_device),
                        depth.to(render_device),
                    )

                    reprejected_pixels_list.append(reprejected_points)
                    reprejected_colors_list.append(reprejected_colors)

                # reprojected image
                dst_image, _ = reprojected2img(
                    reprejected_pixels_list,
                    reprejected_colors_list,
                    dst_camera,
                    alpha_blend=True,
                )
                
                dst_image_np = dst_image.detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1)
                dst_image_pil = Image.fromarray((dst_image_np * 255).astype(np.uint8))
                gt_image_np = gt_image.detach().cpu().numpy().squeeze(0).transpose(1, 2, 0).clip(0, 1)
                gt_image_pil = Image.fromarray((gt_image_np * 255).astype(np.uint8))
                
                reprejected_image = dst_image.unsqueeze(0)
                frontier_target_mask = None
                if mfg_mode == "frontier_seed1" and frontier_anchor_idx is not None and idx in frontier_neighbor_set:
                    frontier_target_mask = _predict_langsam_mask(
                        lang_sam=lang_sam,
                        image_pil=gt_image_pil,
                        text_prompt=ed.target_mask_prompt,
                        image_height=image_height,
                        image_width=image_width,
                        mask_role="gt_view",
                    )
                    mask = frontier_target_mask
                else:
                    mask = _predict_langsam_mask(
                        lang_sam=lang_sam,
                        image_pil=dst_image_pil,
                        text_prompt=ed.target_mask_prompt,
                        image_height=image_height,
                        image_width=image_width,
                        mask_role="reproject",
                    )

                # background replacement
                MF_image = reprejected_image * mask.to(reprejected_image.device)

                mask_bool = mask.bool().to(render_device)
                MF_image = MF_image + (gt_image * ~mask_bool) # (3, 512, 512)

                frontier_proxy_stats = None
                frontier_fallback_triggered = False
                if (
                    frontier_fallback_enabled
                    and mfg_mode == "frontier_seed1"
                    and frontier_anchor_idx is not None
                    and idx in frontier_neighbor_set
                    and frontier_target_mask is not None
                    and frontier_anchor_mask is not None
                ):
                    frontier_proxy_stats = _proxy_region_stats(reprejected_image, frontier_target_mask.to(render_device))
                    frontier_fallback_triggered = _should_use_frontier_fallback(
                        frontier_proxy_stats,
                        mean_threshold=frontier_fallback_mean_thr,
                        std_threshold=frontier_fallback_std_thr,
                        min_coverage=frontier_fallback_min_coverage,
                    )
                    if frontier_fallback_triggered:
                        anchor_image = edited_image_list[frontier_anchor_idx].unsqueeze(0).to(
                            device=render_device,
                            dtype=MF_image.dtype,
                        )
                        MF_image = _compose_anchor_face_fallback(
                            proxy=MF_image,
                            gt=gt_image,
                            anchor=anchor_image,
                            anchor_mask=frontier_anchor_mask.to(render_device),
                            target_mask=frontier_target_mask.to(render_device),
                            feather_radius=frontier_fallback_feather,
                        )

            # reset_attention_maps(ip2p_pipe.unet)

            if MF_image.shape[2] != 512 or MF_image.shape[3] != 512:
                MF_image = F.interpolate(MF_image, size=(512, 512), mode='bilinear', align_corners=True)

            if gt_image.shape[2] != 512 or gt_image.shape[3] != 512:
                gt_image = F.interpolate(gt_image, size=(512, 512), mode='bilinear', align_corners=True)

            
            
            # MFG (Multi-View Fusion Guidance)
            edited_image_MFG = self.edit_image_MFG( # edited_image_MFG -> torch.Size([1, 3, 512, 512]) TODO 输入改了很多
                image=gt_image,
                MF_image_cond=MF_image,
                src_prompt=ed.flow_src_prompt,
                tar_prompt=ed.flow_tar_prompt,
                diffusion_steps=ed.flow_steps,
                n_avg=ed.flow_n_avg,
                src_guidance_scale=ed.flow_src_guidance_scale,
                tar_guidance_scale=ed.flow_tar_guidance_scale,
                negative_prompt=ed.flow_negative_prompt,
                n_min=ed.flow_n_min,
                n_max=ed.flow_n_max,
                seed=ed.flow_seed,
                lambda_S=0,
                lambda_M=0,
                mask_S=None,
                mask_M=None
                )
            
            edited_image_MFG = F.interpolate(edited_image_MFG, size=(image_height, image_width), mode='bilinear', align_corners=True).to(torch.float32)
            # ------------------ save attention map ------------------
            # attention map的调用要改，下面都要仔细看一看 TODO
            # save mfg edited images attention map
            # trg_attention_map = get_all_attention_maps(ip2p_pipe.unet)
            # trg_attention_map_by_tokens = seperate_attention_maps_by_tokens(ip2p_pipe.unet, trg_attention_map, ip2p_pipe.tokenizer, ed.target_prompt)

            vis_path_trg = None

            # get target object prompt attention map
            # trg_object_average_attention_map, trg_object_average_attention_map_512 = save_attention_maps(
            #     trg_attention_map_by_tokens, trg_attention_map, ed.object_prompt, output_dir=vis_path_trg,
            #     image_height=image_height, image_width=image_width
            # )
            
            # TODO !! LangSAM替代 
            # gt_img (1,3,h,w) -> (3,h,w)
            alter_np_gt_img = gt_image.detach().cpu().numpy().squeeze(0).transpose(1, 2, 0).clip(0, 1)
            alter_np_gt_img_pil = Image.fromarray((alter_np_gt_img * 255).astype(np.uint8))
            
            # reprejected_image = dst_image.unsqueeze(0)
            if mfg_mode == "frontier_seed1" and frontier_target_mask is not None:
                alter_gt_mask = frontier_target_mask
            else:
                alter_gt_mask = _predict_langsam_mask(
                    lang_sam=lang_sam,
                    image_pil=alter_np_gt_img_pil,
                    text_prompt=ed.target_mask_prompt,
                    image_height=image_height,
                    image_width=image_width,
                    mask_role="gt_view",
                )

            # trg_object_average_attention_map_512 = torch.tensor(trg_object_average_attention_map_512)

            # # Min-Max Normalization: [0, 1]
            # min_val = trg_object_average_attention_map_512.min()
            # max_val = trg_object_average_attention_map_512.max()
            # trg_object_average_attention_map_512 = (trg_object_average_attention_map_512 - min_val) / (max_val - min_val)

            attn_list.append(alter_gt_mask.to(render_device))
            
            # save gaussian target image
            edited_image_MFG_list.append(edited_image_MFG.squeeze(0).detach().cpu().clone())
            _dump_stage_payload(
                debug_root=debug_root,
                stage="mfg_edit",
                view_idx=idx,
                payload={
                    "gt": gt_image,
                    "reprojected": reprejected_image,
                    "proxy_rgb": reprejected_image,
                    "mask": mask,
                    "mf_cond": MF_image,
                    "mfg_output": edited_image_MFG,
                },
                meta={
                    "mfg_mode": mfg_mode,
                    "mfg_backfill": mfg_backfill,
                    "frontier_anchor_idx": int(frontier_anchor_idx) if frontier_anchor_idx is not None else None,
                    "source_indices": [int(x) for x in src_cam_idx_list],
                    "selected_source_indices": [int(i) for i, flag in enumerate(is_top_selection) if flag],
                    "frontier_fallback_enabled": frontier_fallback_enabled,
                    "frontier_fallback_triggered": frontier_fallback_triggered,
                    "frontier_proxy_mean": None if frontier_proxy_stats is None else float(frontier_proxy_stats["mean"]),
                    "frontier_proxy_std": None if frontier_proxy_stats is None else float(frontier_proxy_stats["std"]),
                    "frontier_proxy_coverage": None if frontier_proxy_stats is None else float(frontier_proxy_stats["coverage"]),
                },
            )
        
        # clean GPU Resources
        # del self.unet
        torch.cuda.empty_cache()

        '''Attention-Guided Trimming (AGT)'''
        skip_agt = os.environ.get("EDITSPLAT_SKIP_AGT", "0").strip().lower() in ("1", "true", "yes")
        if skip_agt:
            print("[WARN] EDITSPLAT_SKIP_AGT=1: skip attention-guided trimming, use full mask.")
            selected_mask = torch.ones_like(gaussians._opacity[:, 0], dtype=torch.float32)
        else:
            # attention Weighting
            attn_weights = torch.zeros_like(gaussians._opacity)
            attn_weights_cnt = torch.zeros_like(gaussians._opacity, dtype=torch.int32)

            for step, batch in enumerate(tqdm(train_dataloader, desc="Attention Weighting")):
                idx = batch['idx'].item()
                camera = camera_list[idx]

                attn_mask = attn_list[step].to(render_device).float()
                
                temp_binary = attn_mask > 0.5
                attn_mask = attn_mask * temp_binary + 0.2 * attn_mask * (~temp_binary)
                attn_mask = attn_mask.unsqueeze(0)

                gaussians.apply_weights(camera, attn_weights, attn_weights_cnt, attn_mask)

            attn_weights /= attn_weights_cnt + 1e-7
            selected_mask = attn_weights[:, 0]

        gaussians.set_mask(selected_mask)
        gaussians.apply_grad_mask(selected_mask)

        iteration = start_iteration
        skip_backward_on_error = _env_flag("EDITSPLAT_SKIP_3DGS_BACKWARD_ON_ERROR", False)
        abort_optim = False
        last_epoch = -1
        for epoch in range(opt.epoch):
            last_epoch = epoch
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"EPOCH {epoch}: optimizing 3D Gaussian Splatting")):
                if iteration % 1000 == 0:
                    gaussians.oneupSHdegree()
                
                total_loss = 0.0

                idx = batch['idx'].item()

                viewpoint_cam = camera_list[idx]
                gaussians.update_learning_rate(iteration)

                viewspace_point_list = []
                
                render_pkg = render(viewpoint_cam, gaussians, pipe, background)

                rendered_image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

                if rendered_image.shape[1] != image_height or rendered_image.shape[2] != image_width:
                    rendered_image = F.interpolate(rendered_image, size=(image_height, image_width), mode='bilinear', align_corners=True)

                viewspace_point_list.append(viewspace_point_tensor)

                edited_image_MFG_for_3dgs = edited_image_MFG_list[idx].to(
                    device=rendered_image.device,
                    dtype=rendered_image.dtype,
                )

                if edited_image_MFG_for_3dgs.shape[1] != image_height or edited_image_MFG_for_3dgs.shape[2] != image_width:
                    edited_image_MFG_for_3dgs = F.interpolate(edited_image_MFG_for_3dgs, size=(image_height, image_width), mode='bilinear', align_corners=True)

                # calculate loss
                Ll1 = l1_loss(rendered_image, edited_image_MFG_for_3dgs)
                p_loss = lpips_loss_fn(torch.clamp(edited_image_MFG_for_3dgs, -1, 1), torch.clamp(rendered_image, -1, 1))
                
                total_loss = Ll1 + p_loss 

                try:
                    total_loss.backward()
                except RuntimeError as exc:
                    msg = str(exc)
                    if skip_backward_on_error and _should_abort_3dgs_backward_error(msg):
                        print(
                            "[WARN] EDITSPLAT_SKIP_3DGS_BACKWARD_ON_ERROR=1: "
                            f"{msg.strip()} during 3DGS backward; aborting optimization loop and keeping partial artifacts."
                        )
                        abort_optim = True
                        break
                    raise

                # optimization Step
                with torch.no_grad():
                    viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_list[0])  
                    for idex in range(len(viewspace_point_list)):
                        viewspace_point_tensor_grad = (
                            viewspace_point_tensor_grad
                            + viewspace_point_list[idex].grad
                        )

                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter],
                        radii[visibility_filter],
                    )
                    gaussians.add_densification_stats(
                        viewspace_point_tensor_grad, visibility_filter
                        )

                    if iteration == start_iteration:
                        # Densification
                        gaussians.densify_and_prune(
                            0.001, 0.005, scene.cameras_extent, 5, is_first_densification=True, k_percent=opt.k_percent, attn_thres=opt.attn_thres
                        )
                    elif iteration % opt.densification_interval == 0:
                        # Densification
                        gaussians.densify_and_prune(
                            opt.densify_grad_threshold, 0.005, scene.cameras_extent, 5, is_first_densification=False, k_percent=opt.k_percent, attn_thres=opt.attn_thres
                        )

                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                
                iteration = iteration + 1

            if abort_optim:
                break

        # save point_cloud
        save_epoch = last_epoch + 1 if last_epoch >= 0 else 0
        print(f"\n[EPOCH {save_epoch}] Saving Gaussians")
        scene.save(iteration)

        # save checkpoint
        print(f"\n[EPOCH {save_epoch}] Saving Checkpoint\n")
        torch.save((gaussians.capture(), iteration), scene.model_path + f"/point_cloud/iteration_{iteration}" + f"/chkpnt{iteration}.pth")

        # save rendering result
        if _env_flag("EDITSPLAT_SKIP_RENDER_SETS", False):
            print("[WARN] EDITSPLAT_SKIP_RENDER_SETS=1: skip final render_sets export.")
        else:
            render_sets(dataset, iteration, pipe, False, False, False)

        backend = getattr(self, "_external_edit_backend", None)
        if backend is not None:
            print(f"[INFO] External backend runtime summary: {backend.summarize()}")
        
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = ArgumentParser(description="Editing Training script parameters")

    # 组装参数组（注意：这里仅实例化，真正的值来自命令行）
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    ed = EditingParams(parser)
    sdp = ScoreDistillParams(parser)

    args = parser.parse_args(sys.argv[1:])

    set_seed(0)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline = Editsplat_Pipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=dtype,
        use_safetensors=True,
        token=os.environ.get("HF_TOKEN", None),
    ).to(device)

    # 取出每个参数组
    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)
    edp = ed.extract(args)
    sdp = sdp.extract(args)
    pipeline.configure_edit_backend(edp)

    os.makedirs(dataset.model_path, exist_ok=True)
    with open(os.path.join(dataset.model_path, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    shutil.copyfile(__file__, os.path.join(dataset.model_path, 'train_frozen.py'))

    _ = pipeline(
        dataset=dataset,
        opt=opt,
        pipe=pipe,
        ed=edp,
        sdp=sdp,          # 新增
    )

    print("\nEditing complete.")
'''
python run_editing_flow.py \
    -s ./dataset/dataset/face \
    -m output/face_to_hulk \
    --source_checkpoint ./dataset/pretrained/face/chkpnt30000.pth \
    --flow_model_key sd35-large \
    --flow_method flowedit \
    --object_prompt "face" \
    --target_prompt "Make his face resemble that of a marble sculpture" \
    --sampling_prompt "a photo of a joker" \
    --target_mask_prompt "face" \
    --flow_src_prompt "a photo of a young man with wavy light-brown hair, wearing a gray zip sweater." \
    --flow_tar_prompt "a photo of a Hulk with red hair, wearing a gray zip sweater." \
    --flow_steps 28 \
    --flow_n_avg 1 \
    --flow_src_guidance_scale 1.5 \
    --flow_tar_guidance_scale 10.5 \
    --flow_n_min 0 \
    --flow_n_max 18 \
    --flow_seed 10 \
    --filtering_ratio 0.65 
'''

