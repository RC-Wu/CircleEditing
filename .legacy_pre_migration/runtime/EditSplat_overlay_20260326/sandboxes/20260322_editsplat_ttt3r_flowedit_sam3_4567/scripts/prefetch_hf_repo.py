#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Callable, List

from storage_guardrails import enforce_storage_guardrails


DEFAULT_CACHE_DIR = Path("/dev_vepfs/rc_wu/cache/hf_home_dev02/hub")
DEFAULT_REPO_ID = "cocktailpeanut/xulf-s"


def repo_cache_dir(cache_dir: Path, repo_id: str) -> Path:
    namespace, name = repo_id.split("/", 1)
    return cache_dir / f"models--{namespace}--{name}"


def incomplete_blobs(cache_dir: Path, repo_id: str) -> List[Path]:
    blobs_dir = repo_cache_dir(cache_dir=cache_dir, repo_id=repo_id) / "blobs"
    if not blobs_dir.exists():
        return []
    return sorted(path for path in blobs_dir.glob("*.incomplete") if path.is_file())


def prefetch_repo(
    *,
    repo_id: str,
    cache_dir: Path,
    retries: int,
    retry_sleep: float,
    max_workers: int,
    token: str | None,
    downloader=None,
) -> Path:
    enforce_storage_guardrails([repo_cache_dir(cache_dir=cache_dir, repo_id=repo_id)])
    if downloader is None:
        try:
            from huggingface_hub import snapshot_download as downloader
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime env
            raise RuntimeError("huggingface_hub is required to prefetch model weights") from exc
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            snapshot_path = Path(
                downloader(
                    repo_id=repo_id,
                    cache_dir=str(cache_dir),
                    token=token,
                    resume_download=True,
                    max_workers=max_workers,
                )
            )
            enforce_storage_guardrails([repo_cache_dir(cache_dir=cache_dir, repo_id=repo_id)])
            pending = incomplete_blobs(cache_dir=cache_dir, repo_id=repo_id)
            if not pending:
                return snapshot_path
            last_error = RuntimeError(
                f"incomplete blobs remain after snapshot_download: {[path.name for path in pending]}"
            )
        except Exception as exc:  # pragma: no cover - exercised in live retry path
            last_error = exc
        if attempt < retries:
            time.sleep(retry_sleep)
    if last_error is None:  # pragma: no cover
        raise RuntimeError(f"prefetch failed for {repo_id} without a captured exception")
    raise last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serial, resumable HF repo prefetch with retry.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--retry-sleep", type=float, default=15.0)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--token-file", type=Path, default=None)
    return parser.parse_args()


def load_token(token_file: Path | None) -> str | None:
    if token_file is not None and token_file.exists():
        text = token_file.read_text(encoding="utf-8").strip()
        if text:
            return text
    return os.environ.get("HF_TOKEN") or None


def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    token = load_token(args.token_file)
    snapshot_path = prefetch_repo(
        repo_id=str(args.repo_id),
        cache_dir=args.cache_dir,
        retries=int(args.retries),
        retry_sleep=float(args.retry_sleep),
        max_workers=int(args.max_workers),
        token=token,
    )
    print(snapshot_path)


if __name__ == "__main__":
    main()
