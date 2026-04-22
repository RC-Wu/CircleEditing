#!/usr/bin/env python3
"""Baseline launcher for run_editing_flow.py without modifying the original file.

The upstream script currently passes `sdp` into pipeline.__call__(), while the
call signature in this workspace does not accept it. This wrapper preserves the
original pipeline behavior and arguments, but drops that incompatible call arg.
"""

import json
import os
import random
import re
import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]  # EditSplat/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arguments import EditingParams, ModelParams, OptimizationParams, PipelineParams, ScoreDistillParams
import run_editing_flow as ref
from run_editing_flow import Editsplat_Pipeline


def patch_aux_models_to_cpu() -> None:
    """Keep FlowEdit core on GPU while moving auxiliary models to CPU."""
    orig_rm_load = ref.RM.load

    def _rm_load_cpu(name="ImageReward-v1.0", device="cpu", *args, **kwargs):
        kwargs["device"] = "cpu"
        return orig_rm_load(name, *args, **kwargs)

    class _LangSAMCPU(ref.LangSAM):
        def __init__(self, sam_type="vit_h", ckpt_path=None):
            self.sam_type = sam_type
            self.device = torch.device("cpu")
            self.build_groundingdino()
            self.build_sam(ckpt_path)

    ref.RM.load = _rm_load_cpu
    ref.LangSAM = _LangSAMCPU


def parse_iter_from_checkpoint(path: str) -> int:
    m = re.search(r"chkpnt(\d+)\.pth$", str(path))
    return int(m.group(1)) if m else 7000


def ensure_point_cloud_link(model_path: str, source_checkpoint: str) -> None:
    model_dir = Path(model_path)
    src_ckpt = Path(source_checkpoint).resolve()
    src_pretrained_dir = src_ckpt.parent
    src_pc = src_pretrained_dir / "point_cloud"
    dst_pc = model_dir / "point_cloud"
    if dst_pc.exists():
        return
    if not src_pc.exists():
        raise FileNotFoundError(f"Missing source point_cloud dir: {src_pc}")
    model_dir.mkdir(parents=True, exist_ok=True)
    os.symlink(src_pc, dst_pc)


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


def patch_camera_dataset_max_views(max_views: int) -> None:
    """Optionally truncate train views for runtime profiling experiments."""
    if int(max_views) <= 0:
        return

    base_dataset_cls = ref.CameraDataset
    orig_init = base_dataset_cls.__init__

    def _init(self, scene):
        orig_init(self, scene)
        n0 = len(self.camera_list)
        self.camera_list = self.camera_list[: int(max_views)]
        print(f"[PROFILE] CameraDataset truncated: {n0} -> {len(self.camera_list)} views")

    base_dataset_cls.__init__ = _init


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    parser = ArgumentParser(description="Editing Training script parameters (baseline wrapper)")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    ed = EditingParams(parser)
    _ = ScoreDistillParams(parser)  # Keep CLI compatibility with original script.
    parser.add_argument(
        "--aux_models_cpu",
        action="store_true",
        default=True,
        help="Force ImageReward and LangSAM on CPU to reduce GPU memory pressure.",
    )
    parser.add_argument(
        "--max_train_views",
        type=int,
        default=0,
        help="If >0, truncate train views to the first N cameras (for profiling only).",
    )

    args = parser.parse_args(sys.argv[1:])

    set_seed(0)
    if args.aux_models_cpu:
        patch_aux_models_to_cpu()
    patch_scene_load_from_checkpoint()
    patch_camera_dataset_max_views(args.max_train_views)

    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline = Editsplat_Pipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=dtype,
        use_safetensors=True,
        token=os.environ.get("HF_TOKEN", None),
    ).to(device)

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)
    edp = ed.extract(args)

    ensure_point_cloud_link(dataset.model_path, dataset.source_checkpoint)

    os.makedirs(dataset.model_path, exist_ok=True)
    with open(os.path.join(dataset.model_path, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # Keep a copy of the launcher used for reproducibility.
    shutil.copyfile(__file__, os.path.join(dataset.model_path, "train_frozen.py"))

    _ = pipeline(
        dataset=dataset,
        opt=opt,
        pipe=pipe,
        ed=edp,
    )

    print("\nEditing complete.")


if __name__ == "__main__":
    main()
