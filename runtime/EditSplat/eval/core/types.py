from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class BenchmarkEntry:
    """Single evaluation unit for one edited model output."""

    scene_id: str
    edit_id: str
    method: str
    model_dir: str

    source_pretrained_dir: str = ""
    source_checkpoint: str = ""
    source_path: str = ""
    source_iter: int = -1
    edited_iter: int = -1
    split: str = "train"

    target_prompt: str = ""
    source_prompt: str = ""
    source_caption: str = ""

    source_render_dir: str = ""
    edit_render_dir: str = ""

    efficiency_tsv: str = ""
    efficiency_model_key: str = ""
    runtime_sec: Optional[float] = None
    peak_mem_mib: Optional[float] = None

    max_views: int = -1
    pair_seed: int = 0

    tags: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def uid(self) -> str:
        return f"{self.scene_id}__{self.edit_id}__{self.method}"

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "BenchmarkEntry":
        required = ["scene_id", "edit_id", "method", "model_dir"]
        missing = [k for k in required if k not in d or str(d[k]).strip() == ""]
        if missing:
            raise ValueError(f"Benchmark entry missing fields: {missing}")

        known = {
            "scene_id",
            "edit_id",
            "method",
            "model_dir",
            "source_pretrained_dir",
            "source_checkpoint",
            "source_path",
            "source_iter",
            "edited_iter",
            "split",
            "target_prompt",
            "source_prompt",
            "source_caption",
            "source_render_dir",
            "edit_render_dir",
            "efficiency_tsv",
            "efficiency_model_key",
            "runtime_sec",
            "peak_mem_mib",
            "max_views",
            "pair_seed",
            "tags",
        }
        extra = {k: v for k, v in d.items() if k not in known}

        return BenchmarkEntry(
            scene_id=str(d["scene_id"]),
            edit_id=str(d["edit_id"]),
            method=str(d["method"]),
            model_dir=str(d["model_dir"]),
            source_pretrained_dir=str(d.get("source_pretrained_dir", "")),
            source_checkpoint=str(d.get("source_checkpoint", "")),
            source_path=str(d.get("source_path", "")),
            source_iter=int(d.get("source_iter", -1)),
            edited_iter=int(d.get("edited_iter", -1)),
            split=str(d.get("split", "train")),
            target_prompt=str(d.get("target_prompt", "")),
            source_prompt=str(d.get("source_prompt", "")),
            source_caption=str(d.get("source_caption", "")),
            source_render_dir=str(d.get("source_render_dir", "")),
            edit_render_dir=str(d.get("edit_render_dir", "")),
            efficiency_tsv=str(d.get("efficiency_tsv", "")),
            efficiency_model_key=str(d.get("efficiency_model_key", "")),
            runtime_sec=float(d["runtime_sec"]) if d.get("runtime_sec", None) is not None else None,
            peak_mem_mib=float(d["peak_mem_mib"]) if d.get("peak_mem_mib", None) is not None else None,
            max_views=int(d.get("max_views", -1)),
            pair_seed=int(d.get("pair_seed", 0)),
            tags=list(d.get("tags", [])),
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "scene_id": self.scene_id,
            "edit_id": self.edit_id,
            "method": self.method,
            "model_dir": self.model_dir,
            "source_pretrained_dir": self.source_pretrained_dir,
            "source_checkpoint": self.source_checkpoint,
            "source_path": self.source_path,
            "source_iter": int(self.source_iter),
            "edited_iter": int(self.edited_iter),
            "split": self.split,
            "target_prompt": self.target_prompt,
            "source_prompt": self.source_prompt,
            "source_caption": self.source_caption,
            "source_render_dir": self.source_render_dir,
            "edit_render_dir": self.edit_render_dir,
            "efficiency_tsv": self.efficiency_tsv,
            "efficiency_model_key": self.efficiency_model_key,
            "runtime_sec": self.runtime_sec,
            "peak_mem_mib": self.peak_mem_mib,
            "max_views": int(self.max_views),
            "pair_seed": int(self.pair_seed),
            "tags": list(self.tags),
        }
        out.update(self.extra)
        return out

    def resolve_paths(self, root: Path) -> None:
        """Resolve entry-local relative paths against a root."""

        for key in [
            "model_dir",
            "source_pretrained_dir",
            "source_checkpoint",
            "source_path",
            "source_render_dir",
            "edit_render_dir",
            "efficiency_tsv",
        ]:
            value = getattr(self, key)
            if not value:
                continue
            p = Path(value)
            if not p.is_absolute():
                setattr(self, key, str((root / p).resolve()))


@dataclass
class EvalConfig:
    """Runtime configuration for metric scripts."""

    clip_model: str = "ViT-B/32"
    open_clip_pretrained: str = "laion2b_s34b_b79k"
    clip_backend: str = "auto"
    pairs_per_sample: int = 200
    reproj_occlusion_abs: float = 0.02
    reproj_occlusion_rel: float = 0.05
    compute_reproj: bool = True
    compute_lpips: bool = True
    device: str = "cuda"

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "EvalConfig":
        return EvalConfig(
            clip_model=str(d.get("clip_model", "ViT-B/32")),
            open_clip_pretrained=str(d.get("open_clip_pretrained", "laion2b_s34b_b79k")),
            clip_backend=str(d.get("clip_backend", "auto")),
            pairs_per_sample=int(d.get("pairs_per_sample", 200)),
            reproj_occlusion_abs=float(d.get("reproj_occlusion_abs", 0.02)),
            reproj_occlusion_rel=float(d.get("reproj_occlusion_rel", 0.05)),
            compute_reproj=bool(d.get("compute_reproj", True)),
            compute_lpips=bool(d.get("compute_lpips", True)),
            device=str(d.get("device", "cuda")),
        )
