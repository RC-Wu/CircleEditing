from __future__ import annotations

import os
import random
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


@dataclass
class ClipModelSpec:
    backend: str
    model_name: str
    pretrained: str
    device: str


class ClipEncoder:
    """Unified CLIP encoder supporting open_clip, OpenAI clip, and transformers CLIP."""

    def __init__(
        self,
        *,
        clip_backend: str = "auto",
        clip_model: str = "ViT-B/32",
        open_clip_pretrained: str = "laion2b_s34b_b79k",
        device: str = "cuda",
    ):
        self.device = torch.device(device if (str(device).startswith("cuda") and torch.cuda.is_available()) else "cpu")
        self.backend = ""
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.processor = None
        self.spec = ClipModelSpec(
            backend="",
            model_name=clip_model,
            pretrained=open_clip_pretrained,
            device=str(self.device),
        )

        backend_req = (clip_backend or "auto").strip().lower()
        if backend_req not in {"auto", "open_clip", "clip", "transformers"}:
            raise ValueError(f"Unsupported clip backend: {clip_backend}")

        # 1) open_clip
        if backend_req in {"auto", "open_clip"}:
            try:
                import open_clip

                model, _, preprocess = open_clip.create_model_and_transforms(
                    clip_model,
                    pretrained=open_clip_pretrained,
                    device=self.device,
                )
                model.eval()
                self.backend = "open_clip"
                self.model = model
                self.preprocess = preprocess
                self.tokenizer = open_clip.get_tokenizer(clip_model)
                self.spec.backend = self.backend
                return
            except Exception:
                if backend_req == "open_clip":
                    raise

        # 2) OpenAI clip
        if backend_req in {"auto", "clip"}:
            try:
                import clip

                clip_cache = os.environ.get("CLIP_CACHE_DIR", "/dev-vepfs/rc_wu/rc_wu/cache/model_weights/clip")
                Path(clip_cache).mkdir(parents=True, exist_ok=True)
                model, preprocess = self._load_openai_clip_with_retry(
                    clip_pkg=clip,
                    clip_model=clip_model,
                    clip_cache=clip_cache,
                )
                model.eval()
                self.backend = "clip"
                self.model = model
                self.preprocess = preprocess
                self.tokenizer = clip.tokenize
                self.spec.backend = self.backend
                return
            except Exception as exc:
                if backend_req == "clip":
                    raise RuntimeError(
                        "Failed to initialize OpenAI clip backend. "
                        "Use --clip_backend transformers or ensure CLIP checkpoint download is reachable."
                    ) from exc

        # 3) transformers CLIP
        if backend_req in {"auto", "transformers"}:
            try:
                from transformers import AutoProcessor, CLIPModel

                model_id = self._resolve_transformers_model_id(clip_model)
                model, processor = self._load_transformers_clip_with_retry(
                    model_id=model_id,
                    CLIPModel=CLIPModel,
                    AutoProcessor=AutoProcessor,
                )
                model = model.to(self.device)
                model.eval()

                self.backend = "transformers"
                self.model = model
                self.processor = processor
                self.spec.backend = self.backend
                self.spec.model_name = model_id
                self.spec.pretrained = model_id
                return
            except Exception as exc:
                raise RuntimeError(
                    "No CLIP backend is available. Tried open_clip, clip, and transformers CLIP."
                ) from exc

        raise RuntimeError("No CLIP backend is available.")

    def _load_openai_clip_with_retry(self, clip_pkg, clip_model: str, clip_cache: str):
        """Recover from partial/corrupted OpenAI clip downloads by cleaning once and retrying."""
        try:
            return clip_pkg.load(clip_model, device=self.device, download_root=clip_cache)
        except RuntimeError as e:
            msg = str(e)
            if "checksum" not in msg.lower():
                raise

            # Best-effort cleanup of broken target file.
            try:
                url = clip_pkg._MODELS.get(clip_model, "")  # type: ignore[attr-defined]
                fname = Path(urllib.parse.urlparse(url).path).name
                if fname:
                    bad_file = Path(clip_cache) / fname
                    if bad_file.exists():
                        bad_file.unlink()
            except Exception:
                pass

            # Remove temporary partial files.
            try:
                for p in Path(clip_cache).glob("*.pt.tmp*"):
                    if p.is_file():
                        p.unlink()
            except Exception:
                pass

            return clip_pkg.load(clip_model, device=self.device, download_root=clip_cache)

    def _load_transformers_clip_with_retry(self, *, model_id: str, CLIPModel, AutoProcessor):
        """Retry transformers CLIP loading by cleaning partial files and force re-download."""
        last_exc = None
        for attempt in range(3):
            force = attempt > 0
            try:
                model = CLIPModel.from_pretrained(model_id, force_download=force)
                processor = AutoProcessor.from_pretrained(model_id, force_download=force)
                return model, processor
            except Exception as exc:
                last_exc = exc
                self._cleanup_hf_partial(model_id)
                continue
        raise RuntimeError(f"Failed to load transformers CLIP model after retries: {model_id}") from last_exc

    @staticmethod
    def _cleanup_hf_partial(model_id: str) -> None:
        """Best effort cleanup for corrupted/incomplete HF hub artifacts."""
        cache_root = os.environ.get("HF_HUB_CACHE", "").strip()
        if not cache_root:
            hf_home = os.environ.get("HF_HOME", "").strip()
            if hf_home:
                cache_root = str(Path(hf_home) / "hub")
        if not cache_root:
            cache_root = str(Path.home() / ".cache" / "huggingface" / "hub")

        root = Path(cache_root)
        model_dir = root / f"models--{model_id.replace('/', '--')}"
        if model_dir.exists():
            for p in model_dir.rglob("*.incomplete"):
                try:
                    p.unlink()
                except Exception:
                    pass

        lock_dir = root / ".locks" / f"models--{model_id.replace('/', '--')}"
        if lock_dir.exists():
            for p in lock_dir.glob("*.lock"):
                try:
                    p.unlink()
                except Exception:
                    pass

    @staticmethod
    def _resolve_transformers_model_id(clip_model: str) -> str:
        name = (clip_model or "").strip()
        mapping = {
            "vit-b/32": "openai/clip-vit-base-patch32",
            "vit-l/14": "openai/clip-vit-large-patch14",
            "vit-b-32": "openai/clip-vit-base-patch32",
            "vit-l-14": "openai/clip-vit-large-patch14",
        }
        key = name.lower()
        if key in mapping:
            return mapping[key]
        if "/" in name:
            return name
        return "openai/clip-vit-base-patch32"

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        texts = [t if t is not None else "" for t in texts]
        if self.backend == "open_clip":
            tokens = self.tokenizer(texts).to(self.device)
            feats = self.model.encode_text(tokens)
        elif self.backend == "transformers":
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            feats = self.model.get_text_features(**inputs)
        else:
            tokens = self.tokenizer(texts, truncate=True).to(self.device)
            feats = self.model.encode_text(tokens)
        feats = feats.float()
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return feats

    @torch.no_grad()
    def encode_images(self, image_paths: Sequence[Path], batch_size: int = 16) -> torch.Tensor:
        feats: List[torch.Tensor] = []
        for i in range(0, len(image_paths), max(1, batch_size)):
            chunk = image_paths[i : i + batch_size]
            if self.backend == "transformers":
                pil_imgs = [Image.open(p).convert("RGB") for p in chunk]
                inputs = self.processor(images=pil_imgs, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device)
                f = self.model.get_image_features(pixel_values=pixel_values).float()
            else:
                ims = []
                for p in chunk:
                    img = Image.open(p).convert("RGB")
                    ims.append(self.preprocess(img))
                x = torch.stack(ims, dim=0).to(self.device)
                f = self.model.encode_image(x).float()
            f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            feats.append(f.cpu())
        return torch.cat(feats, dim=0)


def _cosine_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return (a * b).sum(dim=-1)


def compute_clip_metrics(
    encoder: ClipEncoder,
    src_paths: Sequence[Path],
    edit_paths: Sequence[Path],
    *,
    target_prompt: str,
    source_caption: str,
) -> Dict[str, object]:
    src_map = {p.name: p for p in src_paths}
    edit_map = {p.name: p for p in edit_paths}
    common = sorted(set(src_map.keys()) & set(edit_map.keys()))
    if not common:
        raise RuntimeError("No common views for CLIP metrics.")

    src_common = [src_map[n] for n in common]
    edit_common = [edit_map[n] for n in common]

    e_src = encoder.encode_images(src_common)
    e_edit = encoder.encode_images(edit_common)

    txt_tar = encoder.encode_texts([target_prompt])[0].cpu()
    txt_src = encoder.encode_texts([source_caption])[0].cpu()

    clip_sim = _cosine_rows(e_edit, txt_tar.unsqueeze(0).repeat(e_edit.shape[0], 1))
    clip_src_sim = _cosine_rows(e_edit, txt_src.unsqueeze(0).repeat(e_edit.shape[0], 1))

    d_text = (txt_tar - txt_src).unsqueeze(0)
    d_text = d_text / d_text.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    d_img = e_edit - e_src
    d_img = d_img / d_img.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    clip_dir = _cosine_rows(d_img, d_text.repeat(d_img.shape[0], 1))

    return {
        "view_names": common,
        "clip_sim_per_view": clip_sim.tolist(),
        "clip_sim_mean": float(clip_sim.mean().item()),
        "clip_src_sim_per_view": clip_src_sim.tolist(),
        "clip_src_sim_mean": float(clip_src_sim.mean().item()),
        "clip_dir_per_view": clip_dir.tolist(),
        "clip_dir_mean": float(clip_dir.mean().item()),
        "d_img_vectors": d_img.numpy().tolist(),
        "clip_backend": encoder.spec.backend,
        "clip_model": encoder.spec.model_name,
        "open_clip_pretrained": encoder.spec.pretrained,
    }


def compute_direction_consistency(
    d_img_vectors: Sequence[Sequence[float]],
    *,
    pairs_per_sample: int,
    seed: int,
) -> Dict[str, object]:
    if not d_img_vectors:
        return {
            "clip_dir_consistency_mean": 0.0,
            "clip_dir_consistency_num_pairs": 0,
            "clip_dir_consistency_pairs": [],
        }

    x = torch.tensor(np.asarray(d_img_vectors), dtype=torch.float32)
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    n = x.shape[0]

    all_pairs: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            all_pairs.append((i, j))

    if not all_pairs:
        return {
            "clip_dir_consistency_mean": 0.0,
            "clip_dir_consistency_num_pairs": 0,
            "clip_dir_consistency_pairs": [],
        }

    rng = random.Random(seed)
    if len(all_pairs) > pairs_per_sample > 0:
        pair_idx = rng.sample(range(len(all_pairs)), pairs_per_sample)
        pairs = [all_pairs[i] for i in pair_idx]
    else:
        pairs = all_pairs

    vals = []
    for i, j in pairs:
        vals.append(float(torch.dot(x[i], x[j]).item()))

    return {
        "clip_dir_consistency_mean": float(np.mean(vals)) if vals else 0.0,
        "clip_dir_consistency_num_pairs": len(vals),
        "clip_dir_consistency_pairs": vals,
    }
