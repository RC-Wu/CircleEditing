#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "black-forest-labs/FLUX.1-dev"
FILES = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/merges.txt",
    "tokenizer/vocab.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer_2/special_tokens_map.json",
    "tokenizer_2/spiece.model",
    "tokenizer_2/tokenizer_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model-00001-of-00002.safetensors",
    "text_encoder_2/model-00002-of-00002.safetensors",
    "text_encoder_2/model.safetensors.index.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]


def main() -> int:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("[ERR] HF_TOKEN is empty")
        return 2

    cache_dir = os.environ.get("HF_HUB_CACHE", "/dev-vepfs/rc_wu/rc_wu/cache/hf_home/hub")
    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ["HF_ENDPOINT"] = endpoint

    print(f"[INFO] repo={REPO}")
    print(f"[INFO] cache_dir={cache_dir}")
    print(f"[INFO] endpoint={endpoint}")

    for i, fn in enumerate(FILES, 1):
        ok = False
        for k in range(5):
            try:
                print(f"[DL] {i}/{len(FILES)} {fn} (try {k+1}/5)")
                p = hf_hub_download(
                    repo_id=REPO,
                    filename=fn,
                    token=token,
                    cache_dir=cache_dir,
                    resume_download=True,
                )
                size = Path(p).stat().st_size
                print(f"[OK] {fn} -> {p} ({size} bytes)")
                ok = True
                break
            except Exception as e:
                print(f"[WARN] {fn} try {k+1} failed: {type(e).__name__}: {e}")
                time.sleep(3 * (k + 1))
        if not ok:
            print(f"[ERR] failed file: {fn}")
            return 1

    print("[DONE] all required FLUX.1-dev component files downloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
