from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List, Sequence


def _build_scene_args(source_path: str, model_path: str, base_args: Dict[str, object], data_device: str) -> SimpleNamespace:
    return SimpleNamespace(
        sh_degree=int(base_args.get("sh_degree", 3)),
        source_path=str(source_path),
        model_path=str(model_path),
        source_checkpoint="",
        images=str(base_args.get("images", "images")),
        resolution=int(base_args.get("resolution", -1)),
        white_background=bool(base_args.get("white_background", False)),
        data_device=str(data_device),
        eval=bool(base_args.get("eval", True)),
        render_items=list(base_args.get("render_items", ["RGB", "Depth"])),
        view_shuffling=False,
    )


def load_cameras(
    *,
    source_path: str,
    model_path: str,
    iteration: int,
    base_args: Dict[str, object],
    split: str,
    data_device: str,
) -> Sequence[object]:
    from scene import Scene
    from scene.gaussian_model import GaussianModel

    args = _build_scene_args(source_path, model_path, base_args, data_device)
    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(args, gaussians, load_iteration=iteration, shuffle=False)
    if split == "test":
        return scene.getTestCameras()
    return scene.getTrainCameras()
