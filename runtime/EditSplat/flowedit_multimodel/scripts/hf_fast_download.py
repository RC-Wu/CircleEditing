#!/usr/bin/env python3
"""Fast + resumable Hugging Face downloader for large diffusion checkpoints.

Key features:
- Endpoint fallback (`huggingface.co` <-> `hf-mirror.com`)
- Resume downloads (`snapshot_download`)
- Multi-worker file downloads
- Optional minimal-file mode to skip training-only artifacts
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional


ALIASES = {
    "flux2-dev": "black-forest-labs/FLUX.2-dev",
    "flux2-klein-4b": "black-forest-labs/FLUX.2-klein-4B",
    "sd35-large": "stabilityai/stable-diffusion-3.5-large",
    "sd35-large-turbo": "stabilityai/stable-diffusion-3.5-large-turbo",
    "sd35-medium-turbo-open": "tensorart/stable-diffusion-3.5-medium-turbo",
    "qwen-image-edit": "Qwen/Qwen-Image-Edit",
    "z-image": "Tongyi-MAI/Z-Image",
}

DEFAULT_ENDPOINTS = [
    "https://huggingface.co",
    "https://hf-mirror.com",
]

DEFAULT_IGNORE_PATTERNS = [
    "*.msgpack",
    "*.h5",
    "*.ot",
    "*.onnx",
    "*.ckpt",
    "*.pt",
    "*.pth",
    "*.gguf",
    "*.tflite",
    "*.tar",
    "*.zip",
    "*.webp",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*lora*",
    "lora*",
    "*_ad.*",
    "contrast_*.json",
    "*ckpt*",
    "sd3.5m_turbo.safetensors",
]

REPO_EXTRA_IGNORE: Dict[str, List[str]] = {
    # Keep only diffusers runtime components from this community SD3.5 repo.
    "tensorart/stable-diffusion-3.5-medium-turbo": [
        "lora*",
        "*lora*",
        "*.gguf",
        "*.webp",
        "*ckpt*",
        "contrast_*.json",
        "sd3.5m_turbo.safetensors",
    ],
    "stabilityai/stable-diffusion-3.5-large": [
        "sd3.5_large.safetensors",
        "sd3.5_large_turbo.safetensors",
        "SD3.5L_example_workflow.json",
        "text_encoders/*",
    ],
    "stabilityai/stable-diffusion-3.5-large-turbo": [
        "sd3.5_large.safetensors",
        "sd3.5_large_turbo.safetensors",
        "SD3.5L_example_workflow.json",
        "text_encoders/*",
    ],
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Fast HF model downloader for FlowEdit multimodel experiments")
    ap.add_argument(
        "--models",
        type=str,
        required=True,
        help=(
            "Comma-separated model ids or aliases. "
            "Aliases: flux2-dev, flux2-klein-4b, sd35-large, sd35-large-turbo, sd35-medium-turbo-open, qwen-image-edit, z-image"
        ),
    )
    ap.add_argument("--hf_home", type=str, default="/dev-vepfs/rc_wu/rc_wu/cache/hf_home")
    ap.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--max_workers", type=int, default=16)
    ap.add_argument("--prefer_mirror", action="store_true", help="Try hf-mirror.com before huggingface.co")
    ap.add_argument("--local_dir", type=str, default="", help="Optional explicit local model directory root")
    ap.add_argument(
        "--full_snapshot",
        action="store_true",
        help="Download full repo snapshot (default skips common training/export artifacts).",
    )
    return ap.parse_args()


def resolve_models(spec: str) -> List[str]:
    out: List[str] = []
    for raw in [x.strip() for x in spec.split(",") if x.strip()]:
        out.append(ALIASES.get(raw, raw))
    # Keep insertion order and deduplicate.
    return list(dict.fromkeys(out))


def endpoint_candidates(prefer_mirror: bool) -> List[str]:
    if prefer_mirror:
        return [DEFAULT_ENDPOINTS[1], DEFAULT_ENDPOINTS[0]]
    return list(DEFAULT_ENDPOINTS)


def endpoint_ready(repo_id: str, token: str, endpoint: str) -> bool:
    try:
        from huggingface_hub import HfApi

        api = HfApi(endpoint=endpoint, token=token or None)
        _ = api.model_info(repo_id=repo_id, timeout=10)
        return True
    except Exception:  # noqa: BLE001
        return False


def build_ignore_patterns(repo_id: str, full_snapshot: bool) -> Optional[List[str]]:
    if full_snapshot:
        return None
    out = list(DEFAULT_IGNORE_PATTERNS)
    out.extend(REPO_EXTRA_IGNORE.get(repo_id, []))
    # Keep order stable while deduplicating.
    return list(dict.fromkeys(out))


def main() -> int:
    args = parse_args()
    models = resolve_models(args.models)
    if not models:
        raise ValueError("No models resolved from --models.")

    if not args.hf_token:
        raise ValueError("HF token is empty. Pass --hf_token or export HF_TOKEN.")

    hf_home = Path(args.hf_home).expanduser().resolve()
    cache_dir = hf_home / "hub"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Throughput-related knobs. Set before importing huggingface_hub so backends pick them up.
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(cache_dir)
    # hf_transfer is fast on some links but can be fragile behind proxies/mirrors.
    # Default to disabled unless explicitly opted in from the caller environment.
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0")
    # Xet backend may be unstable on some links; fallback to regular hub download.
    os.environ["HF_HUB_DISABLE_XET"] = os.environ.get("HF_HUB_DISABLE_XET", "1")
    os.environ["HF_XET_HIGH_PERFORMANCE"] = os.environ.get("HF_XET_HIGH_PERFORMANCE", "0")
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "120"

    from huggingface_hub import HfApi, snapshot_download

    local_root = Path(args.local_dir).expanduser().resolve() if args.local_dir else None
    if local_root is not None:
        local_root.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Dict] = {}
    start_all = time.time()
    candidates = endpoint_candidates(args.prefer_mirror)

    for repo_id in models:
        start = time.time()
        local_dir = None
        if local_root is not None:
            local_dir = str(local_root / repo_id.replace("/", "__"))

        print(f"[INFO] repo={repo_id}")
        print(f"[INFO] cache_dir={cache_dir}")

        last_err = None
        path = None
        endpoint = None
        for ep in candidates:
            if not endpoint_ready(repo_id, args.hf_token, ep):
                continue
            try:
                os.environ["HF_ENDPOINT"] = ep
                print(f"[INFO] try endpoint={ep}")
                path = snapshot_download(
                    repo_id=repo_id,
                    token=args.hf_token,
                    cache_dir=str(cache_dir),
                    resume_download=True,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False,
                    max_workers=int(args.max_workers),
                    ignore_patterns=build_ignore_patterns(repo_id, args.full_snapshot),
                )
                endpoint = ep
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                print(f"[WARN] endpoint={ep} failed: {type(e).__name__}: {e}")

        if path is None or endpoint is None:
            raise RuntimeError(f"All endpoints failed for {repo_id}: {last_err}")

        dur = time.time() - start
        repo_info = HfApi(endpoint=endpoint, token=args.hf_token).model_info(repo_id=repo_id)
        summary[repo_id] = {
            "endpoint": endpoint,
            "local_path": path,
            "last_modified": str(repo_info.last_modified),
            "sha": repo_info.sha,
            "downloads": repo_info.downloads,
            "duration_sec": round(dur, 2),
        }
        print(f"[DONE] {repo_id} in {dur:.1f}s -> {path}")

    out = {
        "models": models,
        "hf_home": str(hf_home),
        "cache_dir": str(cache_dir),
        "total_duration_sec": round(time.time() - start_all, 2),
        "summary": summary,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
