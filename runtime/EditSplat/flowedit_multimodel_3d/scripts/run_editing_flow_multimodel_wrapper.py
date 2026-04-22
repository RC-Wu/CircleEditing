#!/usr/bin/env python3
"""Run EditSplat 3D pipeline with pluggable FlowEdit adapters (FLUX2 / SD3.5 / Qwen / Z).

This wrapper avoids touching `run_editing_flow.py` by monkeypatching
`Editsplat_Pipeline.edit_image` and `edit_image_MFG` to call the unified 2D adapter API.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import shutil
import sys
import types
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]  # EditSplat/
MULTI2D = ROOT / "flowedit_multimodel"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(MULTI2D) not in sys.path:
    sys.path.insert(0, str(MULTI2D))


def _ensure_imagereward_stub() -> None:
    try:
        import ImageReward  # noqa: F401
        return
    except Exception:
        pass

    class _DummyRewardModel:
        def inference_rank(self, prompt, images):
            del prompt
            n = len(images)
            return list(range(1, n + 1)), [0.0] * n

    stub = types.ModuleType("ImageReward")

    def _load(name="ImageReward-v1.0"):
        del name
        print("[WARN] ImageReward import failed; using dummy reward ranker.")
        return _DummyRewardModel()

    stub.load = _load  # type: ignore[attr-defined]
    sys.modules["ImageReward"] = stub


def _ensure_lpips_stub() -> None:
    # Set FLOWEDIT_REAL_LPIPS=1 to force real lpips import.
    if os.environ.get("FLOWEDIT_REAL_LPIPS", "0") == "1":
        return
    try:
        import lpips  # noqa: F401
        # Replace regardless to avoid downloading VGG weights in constrained networks.
    except Exception:
        pass

    stub = types.ModuleType("lpips")

    class _LPIPS:
        def __init__(self, net="vgg"):
            self.net = net

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

    stub.LPIPS = _LPIPS  # type: ignore[attr-defined]
    print("[WARN] Using LPIPS stub (L1 fallback). Set FLOWEDIT_REAL_LPIPS=1 to disable.")
    sys.modules["lpips"] = stub


def _ensure_langsam_stub() -> None:
    # Set FLOWEDIT_REAL_LANGSAM=1 to force real LangSAM.
    if os.environ.get("FLOWEDIT_REAL_LANGSAM", "0") == "1":
        return

    stub = types.ModuleType("lang_sam")

    class _LangSAM:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            print("[WARN] Using LangSAM stub (full-image mask). Set FLOWEDIT_REAL_LANGSAM=1 to disable.")

        def predict(self, image_pil, text_prompt):
            del text_prompt
            w, h = image_pil.size
            mask = torch.ones((1, h, w), dtype=torch.float32)
            return mask, None, None, None

    stub.LangSAM = _LangSAM  # type: ignore[attr-defined]
    sys.modules["lang_sam"] = stub


_ensure_imagereward_stub()
_ensure_lpips_stub()
_ensure_langsam_stub()

import run_editing_flow as ref
from run_editing_flow import Editsplat_Pipeline
from src.flowedit_adapters import FlowEditParams, create_adapter


def _tensor_to_pil01(img: torch.Tensor) -> Image.Image:
    if img.ndim == 3:
        img = img.unsqueeze(0)
    x = img[0].detach().float()
    if x.min() < 0.0:
        x = (x + 1.0) * 0.5
    x = x.clamp(0.0, 1.0).cpu()
    arr = (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def _to_negative_prompt(negative_prompt) -> str:
    if isinstance(negative_prompt, str):
        return negative_prompt
    return ""


def patch_edit_methods(adapter=None, backend=None, resize_side: int = 512) -> None:
    def _edit_image(
        self,
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
        mask_S: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del self, lambda_S, mask_S
        if backend is not None:
            out = backend.edit(
                image=image,
                src_prompt=src_prompt,
                tar_prompt=tar_prompt,
                negative_prompt=_to_negative_prompt(negative_prompt),
                diffusion_steps=int(diffusion_steps),
                n_avg=int(n_avg),
                src_guidance_scale=float(src_guidance_scale),
                tar_guidance_scale=float(tar_guidance_scale),
                n_min=int(n_min),
                n_max=int(n_max),
                seed=int(seed),
            )
            return out.to(image.device, dtype=torch.float32)

        src_pil = _tensor_to_pil01(image)
        params = FlowEditParams(
            diffusion_steps=int(diffusion_steps),
            n_avg=int(n_avg),
            src_guidance_scale=float(src_guidance_scale),
            tar_guidance_scale=float(tar_guidance_scale),
            n_min=int(n_min),
            n_max=int(n_max),
            seed=int(seed),
            negative_prompt=_to_negative_prompt(negative_prompt),
            resize_side=int(resize_side),
        )
        out = adapter.edit(src_pil, src_prompt, tar_prompt, params)
        return out.to(image.device, dtype=torch.float32)

    def _edit_image_mfg(
        self,
        image: torch.Tensor,
        MF_image_cond: torch.Tensor,
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
        lambda_M: float = 0.0,
        mask_S: Optional[torch.Tensor] = None,
        mask_M: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del self, lambda_S, lambda_M, mask_S, mask_M
        base = MF_image_cond if MF_image_cond is not None else image
        if backend is not None:
            out = backend.edit(
                image=base,
                src_prompt=src_prompt,
                tar_prompt=tar_prompt,
                negative_prompt=_to_negative_prompt(negative_prompt),
                diffusion_steps=int(diffusion_steps),
                n_avg=int(n_avg),
                src_guidance_scale=float(src_guidance_scale),
                tar_guidance_scale=float(tar_guidance_scale),
                n_min=int(n_min),
                n_max=int(n_max),
                seed=int(seed),
            )
            return out.to(image.device, dtype=torch.float32)

        src_pil = _tensor_to_pil01(base)
        params = FlowEditParams(
            diffusion_steps=int(diffusion_steps),
            n_avg=int(n_avg),
            src_guidance_scale=float(src_guidance_scale),
            tar_guidance_scale=float(tar_guidance_scale),
            n_min=int(n_min),
            n_max=int(n_max),
            seed=int(seed),
            negative_prompt=_to_negative_prompt(negative_prompt),
            resize_side=int(resize_side),
        )
        out = adapter.edit(src_pil, src_prompt, tar_prompt, params)
        return out.to(image.device, dtype=torch.float32)

    Editsplat_Pipeline.edit_image = _edit_image
    Editsplat_Pipeline.edit_image_MFG = _edit_image_mfg


def patch_aux_models_to_cpu() -> None:
    # Keep memory-heavy auxiliary models on CPU to avoid competing with 3DGS.
    try:
        orig_rm_load = ref.RM.load

        def _rm_load_cpu(name="ImageReward-v1.0", device="cpu", *args, **kwargs):
            kwargs["device"] = "cpu"
            try:
                return orig_rm_load(name, *args, **kwargs)
            except TypeError:
                # Compatibility with local stub that does not accept "device".
                kwargs.pop("device", None)
                return orig_rm_load(name, *args, **kwargs)

        ref.RM.load = _rm_load_cpu
    except Exception:
        pass

    try:
        if not (hasattr(ref.LangSAM, "build_groundingdino") and hasattr(ref.LangSAM, "build_sam")):
            return

        class _LangSAMCPU(ref.LangSAM):
            def __init__(self, sam_type="vit_h", ckpt_path=None):
                self.sam_type = sam_type
                self.device = torch.device("cpu")
                self.build_groundingdino()
                self.build_sam(ckpt_path)

        ref.LangSAM = _LangSAMCPU
    except Exception:
        pass


def patch_head_camera_dataset(head_k: int) -> None:
    if int(head_k) <= 0:
        return

    base_cls = ref.CameraDataset
    orig_init = base_cls.__init__

    def _init(self, scene):
        orig_init(self, scene)
        if hasattr(self, "camera_list") and isinstance(self.camera_list, list):
            self.camera_list = self.camera_list[: int(head_k)]

    base_cls.__init__ = _init


def parse_iter_from_checkpoint(path: str) -> int:
    m = re.search(r"chkpnt(\d+)\.pth$", str(path))
    return int(m.group(1)) if m else 7000


def ensure_point_cloud_link(model_path: str, source_checkpoint: str) -> None:
    if not source_checkpoint:
        return
    model_dir = Path(model_path)
    src_ckpt = Path(source_checkpoint).resolve()
    src_pretrained_dir = src_ckpt.parent
    src_pc = src_pretrained_dir / "point_cloud"
    dst_pc = model_dir / "point_cloud"
    load_iter = parse_iter_from_checkpoint(str(src_ckpt))
    src_iter_dir = src_pc / f"iteration_{load_iter}"
    src_ply = src_iter_dir / "point_cloud.ply"
    dst_iter_dir = dst_pc / f"iteration_{load_iter}"
    dst_ply = dst_iter_dir / "point_cloud.ply"

    if dst_ply.exists():
        return
    if not src_ply.exists():
        raise FileNotFoundError(f"Missing source point_cloud ply: {src_ply}")

    model_dir.mkdir(parents=True, exist_ok=True)
    if dst_pc.is_symlink():
        dst_pc.unlink()
    elif dst_pc.exists() and not dst_pc.is_dir():
        raise RuntimeError(f"Unexpected point_cloud path type: {dst_pc}")

    dst_iter_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_ply, dst_ply)


def patch_scene_load_from_checkpoint() -> None:
    base_scene_cls = ref.Scene

    class _SceneLoadFromCheckpoint(base_scene_cls):
        def __init__(self, args, gaussians, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
            if load_iteration is None and getattr(args, "source_checkpoint", ""):
                load_iteration = parse_iter_from_checkpoint(args.source_checkpoint)
            super().__init__(
                args=args,
                gaussians=gaussians,
                load_iteration=load_iteration,
                shuffle=shuffle,
                resolution_scales=resolution_scales,
            )

    ref.Scene = _SceneLoadFromCheckpoint


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Run 3D EditSplat with FlowEdit multimodel adapter", add_help=False)
    ap.add_argument("--model_key", type=str, default="flux2-dev")
    ap.add_argument("--model_id", type=str, default="", help="Optional explicit repo/path override.")
    ap.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--hf_home", type=str, default="/dev-vepfs/rc_wu/rc_wu/cache/hf_home")
    ap.add_argument("--adapter_resize_side", type=int, default=512)
    ap.add_argument("--adapter_gpu", type=int, default=0, help="CUDA index for multimodel adapter.")
    ap.add_argument("--base_gpu", type=int, default=0, help="CUDA index for EditSplat base pipeline.")
    ap.add_argument("--keep_base_on_gpu", action="store_true", help="Keep base Flux modules on GPU.")
    ap.add_argument("--head_k", type=int, default=0, help="Use only first K training views (0=all).")
    ap.add_argument("--depth_mode", type=str, default="render", choices=["render", "constant"])
    ap.add_argument("--skip_agt", action="store_true", help="Skip attention-guided trimming stage.")
    ap.add_argument("--aux_models_cpu", action="store_true", default=True, help="Force ImageReward/LangSAM on CPU.")
    ap.add_argument("--no_aux_models_cpu", action="store_false", dest="aux_models_cpu")
    return ap.parse_known_args()[0]


def main():
    # Parse wrapper-only args, keep the original run_editing_flow args in sys.argv.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model_key", type=str, default="flux2-dev")
    parser.add_argument("--model_id", type=str, default="")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--hf_home", type=str, default="/dev-vepfs/rc_wu/rc_wu/cache/hf_home")
    parser.add_argument("--adapter_resize_side", type=int, default=512)
    parser.add_argument("--adapter_gpu", type=int, default=0)
    parser.add_argument("--base_gpu", type=int, default=0)
    parser.add_argument("--keep_base_on_gpu", action="store_true")
    parser.add_argument("--head_k", type=int, default=0)
    parser.add_argument("--depth_mode", type=str, default="render", choices=["render", "constant"])
    parser.add_argument("--skip_agt", action="store_true")
    parser.add_argument("--aux_models_cpu", action="store_true", default=True)
    parser.add_argument("--no_aux_models_cpu", action="store_false", dest="aux_models_cpu")
    wargs, remaining = parser.parse_known_args()

    os.environ["HF_HOME"] = wargs.hf_home
    os.environ["HF_HUB_CACHE"] = str(Path(wargs.hf_home) / "hub")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ["EDITSPLAT_DEPTH_MODE"] = str(wargs.depth_mode)
    os.environ["EDITSPLAT_SKIP_AGT"] = "1" if bool(wargs.skip_agt) else "0"

    if torch.cuda.is_available():
        adapter_device = torch.device(f"cuda:{int(wargs.adapter_gpu)}")
        base_device = torch.device(f"cuda:{int(wargs.base_gpu)}")
        torch.cuda.set_device(base_device)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        # FLUX.2-dev may OOM in bf16 on a single card and trigger device_map fallback,
        # which breaks manual transformer calls in our FlowEdit loop.
        if str(wargs.model_key).startswith("flux2"):
            dtype = torch.float16
    else:
        adapter_device = torch.device("cpu")
        base_device = torch.device("cpu")
        dtype = torch.float16

    patch_head_camera_dataset(head_k=int(wargs.head_k))
    if bool(wargs.aux_models_cpu):
        patch_aux_models_to_cpu()
    patch_scene_load_from_checkpoint()

    # Mirror run_editing_flow.py's __main__ logic (the module has no ref.main()).
    run_parser = argparse.ArgumentParser(description="Editing Training script parameters")
    lp = ref.ModelParams(run_parser)
    op = ref.OptimizationParams(run_parser)
    pp = ref.PipelineParams(run_parser)
    ed = ref.EditingParams(run_parser)
    sdp_param = ref.ScoreDistillParams(run_parser)
    run_args = run_parser.parse_args(remaining)

    if hasattr(ref, "set_seed"):
        ref.set_seed(0)

    run_dtype = torch.bfloat16 if (base_device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    run_device = str(base_device)
    base_model_id = os.environ.get("EDITSPLAT_BASE_MODEL_ID", "black-forest-labs/FLUX.1-dev")
    pipeline = Editsplat_Pipeline.from_pretrained(
        base_model_id,
        torch_dtype=run_dtype,
        use_safetensors=True,
        token=os.environ.get("HF_TOKEN", None),
        cache_dir=str(Path(wargs.hf_home) / "hub"),
    )

    # In multimodel mode, edit steps are delegated to external adapters.
    # Keep heavy Flux.1 blocks on CPU, and only move VAE to base GPU so
    # pipeline._execution_device stays on CUDA for 3DGS tensors.
    if bool(wargs.keep_base_on_gpu):
        pipeline = pipeline.to(run_device)
    else:
        for name in ("transformer", "text_encoder", "text_encoder_2"):
            mod = getattr(pipeline, name, None)
            if mod is not None:
                try:
                    mod.to("cpu")
                except Exception:
                    pass
        vae = getattr(pipeline, "vae", None)
        if vae is not None:
            try:
                vae.to(run_device)
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    dataset = lp.extract(run_args)
    opt = op.extract(run_args)
    pipe = pp.extract(run_args)
    edp = ed.extract(run_args)
    sdp_cfg = sdp_param.extract(run_args)

    method = str(getattr(edp, "flow_method", "flowedit")).strip().lower()
    adapter = None
    backend = None
    if method == "dnaedit":
        FlowBackendConfig, FlowEditCoreBackend = ref._load_flowedit_core_backend_symbols()
        cfg = FlowBackendConfig(
            model_key=str(getattr(edp, "flow_model_key", "sd35-large")).strip(),
            model_id=str(getattr(edp, "flow_model_id", "")).strip(),
            method=method,
            hf_home=str(getattr(edp, "flow_hf_home", wargs.hf_home)).strip(),
            adapter_resize_side=int(getattr(edp, "flow_adapter_resize_side", wargs.adapter_resize_side)),
            adapter_gpu=int(getattr(edp, "flow_adapter_gpu", wargs.adapter_gpu)),
            hf_token=os.environ.get("HF_TOKEN", ""),
            dna_steps=int(getattr(edp, "flow_dna_steps", 40)),
            dna_src_guidance_scale=float(getattr(edp, "flow_dna_src_guidance_scale", 1.0)),
            dna_tar_guidance_scale=float(getattr(edp, "flow_dna_tar_guidance_scale", 3.5)),
            dna_t_start=int(getattr(edp, "flow_dna_t_start", 13)),
            dna_mvg=float(getattr(edp, "flow_dna_mvg", 0.8)),
        )
        backend = FlowEditCoreBackend(config=cfg, project_root=str(ROOT))
    else:
        adapter = create_adapter(
            model_key=wargs.model_key,
            device=adapter_device,
            dtype=dtype,
            hf_token=wargs.hf_token if wargs.hf_token else None,
            cache_dir=str(Path(wargs.hf_home) / "hub"),
            override_model_id=wargs.model_id if wargs.model_id else None,
        )
    patch_edit_methods(adapter=adapter, backend=backend, resize_side=int(wargs.adapter_resize_side))

    os.makedirs(dataset.model_path, exist_ok=True)
    with open(os.path.join(dataset.model_path, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(run_args), f, indent=2, ensure_ascii=False)
    wrapper_meta = {
        "model_key": wargs.model_key,
        "model_id": wargs.model_id if wargs.model_id else "",
        "hf_home": wargs.hf_home,
        "adapter_resize_side": int(wargs.adapter_resize_side),
        "adapter_gpu": int(wargs.adapter_gpu),
        "base_gpu": int(wargs.base_gpu),
        "keep_base_on_gpu": bool(wargs.keep_base_on_gpu),
        "head_k": int(wargs.head_k),
        "depth_mode": str(wargs.depth_mode),
        "skip_agt": bool(wargs.skip_agt),
        "aux_models_cpu": bool(wargs.aux_models_cpu),
    }
    with open(os.path.join(dataset.model_path, "multimodel_wrapper_meta.json"), "w", encoding="utf-8") as f:
        json.dump(wrapper_meta, f, indent=2, ensure_ascii=False)
    try:
        ensure_point_cloud_link(dataset.model_path, getattr(dataset, "source_checkpoint", ""))
    except Exception:
        pass

    try:
        shutil.copyfile(ref.__file__, os.path.join(dataset.model_path, "train_frozen.py"))
    except Exception:
        pass

    if hasattr(pipeline, "set_sds_params"):
        try:
            pipeline.set_sds_params(sdp_cfg)
        except Exception:
            pass

    call_kwargs = {"dataset": dataset, "opt": opt, "pipe": pipe, "ed": edp}
    if "sdp" in inspect.signature(pipeline.__call__).parameters:
        call_kwargs["sdp"] = sdp_cfg

    _ = pipeline(**call_kwargs)
    if backend is not None and hasattr(backend, "summarize"):
        try:
            print(f"[INFO] External backend runtime summary: {backend.summarize()}")
        except Exception:
            pass
    print("\nEditing complete.")


if __name__ == "__main__":
    main()
