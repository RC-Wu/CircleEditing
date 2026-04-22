#!/usr/bin/env python3
"""Download only component files required by Flux2KleinPipeline.from_pretrained.

This skips single-file checkpoints like `flux-2-klein-4b.safetensors` which are
not needed for component-based loading and can waste many hours on slow links.
"""

import argparse
import os
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_DEFAULT = "black-forest-labs/FLUX.2-klein-4B"

# Minimal component set for from_pretrained-style loading.
FILES = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/merges.txt",
    "tokenizer/vocab.json",
    "tokenizer/tokenizer_config.json",
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00002.safetensors",
    "text_encoder/model-00002-of-00002.safetensors",
    "text_encoder/model.safetensors.index.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Download Flux2 Klein component files sequentially")
    ap.add_argument("--repo_id", type=str, default=REPO_DEFAULT)
    ap.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--hf_home", type=str, default="/dev-vepfs/rc_wu/rc_wu/cache/hf_home")
    ap.add_argument("--max_retries", type=int, default=30)
    ap.add_argument("--prefer_mirror", action="store_true")
    ap.add_argument("--retry_sleep", type=float, default=4.0)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if not args.hf_token:
        raise ValueError("HF token is empty. Pass --hf_token or export HF_TOKEN.")

    hf_home = Path(args.hf_home).expanduser().resolve()
    cache_dir = hf_home / "hub"
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(cache_dir)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    endpoints = ["https://huggingface.co", "https://hf-mirror.com"]
    if args.prefer_mirror:
        endpoints = [endpoints[1], endpoints[0]]

    print(f"[INFO] repo={args.repo_id}")
    print(f"[INFO] cache_dir={cache_dir}")

    for i, fn in enumerate(FILES, 1):
        ok = False
        last_err = None
        for r in range(1, args.max_retries + 1):
            for ep in endpoints:
                os.environ["HF_ENDPOINT"] = ep
                try:
                    p = hf_hub_download(
                        repo_id=args.repo_id,
                        filename=fn,
                        token=args.hf_token,
                        cache_dir=str(cache_dir),
                        resume_download=True,
                    )
                    size = Path(p).stat().st_size
                    print(f"[OK] {i:02d}/{len(FILES)} {fn} via {ep} -> {size} bytes")
                    ok = True
                    break
                except Exception as e:  # noqa: BLE001
                    last_err = f"{type(e).__name__}: {e}"
                    print(f"[WARN] {fn} retry={r}/{args.max_retries} ep={ep}: {last_err}")
            if ok:
                break
            time.sleep(args.retry_sleep)

        if not ok:
            raise RuntimeError(f"Failed file: {fn}; last_err={last_err}")

    print("[DONE] Flux2 Klein component files downloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
