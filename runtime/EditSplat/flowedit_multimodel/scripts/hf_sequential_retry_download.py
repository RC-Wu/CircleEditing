#!/usr/bin/env python3
"""Robust sequential HF downloader for unstable networks.

Downloads one file at a time with resume + retries, so transient failures
won't restart the whole snapshot task.
"""

import argparse
import fnmatch
import json
import os
import time
from pathlib import Path
from typing import Iterable, List


DEFAULT_ENDPOINTS = ["https://huggingface.co", "https://hf-mirror.com"]
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
    "flux2-dev.safetensors",
]

REPO_EXTRA_IGNORE = {
    # Keep modular diffusers components; skip monolithic checkpoint.
    "black-forest-labs/FLUX.2-dev": [
        "flux2-dev.safetensors",
        "teaser_*.png",
    ],
    "tensorart/stable-diffusion-3.5-medium-turbo": [
        "lora*",
        "*lora*",
        "*.gguf",
        "*.webp",
        "*ckpt*",
        "contrast_*.json",
        "sd3.5m_turbo.safetensors",
    ],
    # Keep diffusers modular components, skip monolithic checkpoint / demo assets.
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
    ap = argparse.ArgumentParser("Sequential HF downloader with retries")
    ap.add_argument("--repo_id", type=str, required=True)
    ap.add_argument("--hf_home", type=str, default="/dev-vepfs/rc_wu/rc_wu/cache/hf_home")
    ap.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--max_retries", type=int, default=20)
    ap.add_argument("--retry_sleep", type=float, default=5.0)
    ap.add_argument("--prefer_mirror", action="store_true")
    ap.add_argument("--mirror_only", action="store_true")
    ap.add_argument("--official_only", action="store_true")
    ap.add_argument("--precheck_timeout", type=int, default=12)
    ap.add_argument("--full_snapshot", action="store_true")
    ap.add_argument("--use_hf_transfer", action="store_true")
    return ap.parse_args()


def keep_file(repo_id: str, filename: str, full_snapshot: bool) -> bool:
    if full_snapshot:
        return True
    pats = list(DEFAULT_IGNORE_PATTERNS) + list(REPO_EXTRA_IGNORE.get(repo_id, []))
    for pat in pats:
        if fnmatch.fnmatch(filename, pat):
            return False
    return True


def endpoint_order(prefer_mirror: bool, mirror_only: bool, official_only: bool) -> List[str]:
    if mirror_only and official_only:
        raise ValueError("--mirror_only and --official_only cannot both be set.")
    if mirror_only:
        return [DEFAULT_ENDPOINTS[1]]
    if official_only:
        return [DEFAULT_ENDPOINTS[0]]
    return [DEFAULT_ENDPOINTS[1], DEFAULT_ENDPOINTS[0]] if prefer_mirror else list(DEFAULT_ENDPOINTS)


def main() -> int:
    args = parse_args()
    if not args.hf_token:
        raise ValueError("HF token is empty. Pass --hf_token or export HF_TOKEN.")

    hf_home = Path(args.hf_home).expanduser().resolve()
    cache_dir = hf_home / "hub"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Must be set before importing huggingface_hub internals.
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(cache_dir)
    os.environ["HF_HUB_DISABLE_XET"] = os.environ.get("HF_HUB_DISABLE_XET", "1")
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1" if args.use_hf_transfer else "0"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "120"

    from huggingface_hub import HfApi, hf_hub_download

    candidate_endpoints = endpoint_order(args.prefer_mirror, args.mirror_only, args.official_only)
    files = None
    model_sha = None

    usable_endpoints = []
    for ep in candidate_endpoints:
        try:
            api = HfApi(endpoint=ep, token=args.hf_token)
            _ = api.model_info(repo_id=args.repo_id, timeout=int(args.precheck_timeout))
            usable_endpoints.append(ep)
            print(f"[INFO] endpoint ready: {ep}")
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] endpoint precheck failed: {ep}: {type(e).__name__}: {e}")

    if not usable_endpoints:
        print("[WARN] no endpoint passed precheck; falling back to configured endpoint list.")
        usable_endpoints = list(candidate_endpoints)

    for ep in usable_endpoints:
        try:
            api = HfApi(endpoint=ep, token=args.hf_token)
            info = api.model_info(repo_id=args.repo_id)
            model_sha = info.sha
            files = [s.rfilename for s in info.siblings if keep_file(args.repo_id, s.rfilename, args.full_snapshot)]
            files.sort()
            print(f"[INFO] list files via {ep}: {len(files)} files")
            break
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] list files failed on {ep}: {type(e).__name__}: {e}")

    if files is None:
        raise RuntimeError(f"Cannot list files for repo={args.repo_id}")

    results = []
    t0_all = time.time()

    for i, filename in enumerate(files, start=1):
        done = False
        last_err = None
        t0 = time.time()

        for attempt in range(1, args.max_retries + 1):
            for ep in usable_endpoints:
                os.environ["HF_ENDPOINT"] = ep
                try:
                    local_path = hf_hub_download(
                        repo_id=args.repo_id,
                        filename=filename,
                        token=args.hf_token,
                        endpoint=ep,
                        cache_dir=str(cache_dir),
                        resume_download=True,
                    )
                    dt = round(time.time() - t0, 2)
                    print(f"[OK] ({i}/{len(files)}) {filename} via {ep} in {dt}s")
                    results.append({
                        "file": filename,
                        "endpoint": ep,
                        "duration_sec": dt,
                        "path": local_path,
                    })
                    done = True
                    break
                except KeyboardInterrupt:
                    raise
                except Exception as e:  # noqa: BLE001
                    last_err = f"{type(e).__name__}: {e}"
                    print(f"[WARN] ({i}/{len(files)}) {filename} attempt={attempt} endpoint={ep} failed: {last_err}")
            if done:
                break
            time.sleep(args.retry_sleep)

        if not done:
            raise RuntimeError(f"Failed file after retries: {filename}; last_err={last_err}")

    summary = {
        "repo_id": args.repo_id,
        "sha": model_sha,
        "files": len(files),
        "total_duration_sec": round(time.time() - t0_all, 2),
        "hf_transfer": os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0"),
        "results": results,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
