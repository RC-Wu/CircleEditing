#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence


LOCAL_PROJECT_LIMIT_BYTES = 20 * 1024 * 1024
VEPFS_PROJECT_LIMIT_BYTES = 50 * 1024 * 1024 * 1024
VEPFS_PREFIXES = (Path("/dev_vepfs"),)


def _resolve_path(path: Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _is_under_prefix(path: Path, prefixes: Sequence[Path]) -> bool:
    for prefix in prefixes:
        resolved_prefix = _resolve_path(prefix)
        if path == resolved_prefix or resolved_prefix in path.parents:
            return True
    return False


def path_size_bytes(path: Path) -> int:
    target = _resolve_path(path)
    if not target.exists():
        return 0
    if target.is_file():
        return target.stat().st_size

    total = 0
    for root, _, files in os.walk(target):
        root_path = Path(root)
        try:
            total += root_path.stat().st_size
        except OSError:
            continue
        for name in files:
            file_path = root_path / name
            try:
                total += file_path.stat().st_size
            except OSError:
                continue
    return total


def _format_bytes(num_bytes: int) -> str:
    gib = 1024 * 1024 * 1024
    mib = 1024 * 1024
    if num_bytes >= gib:
        return f"{num_bytes / gib:.2f} GiB"
    return f"{num_bytes / mib:.2f} MiB"


def enforce_storage_guardrails(
    paths: Iterable[Path],
    *,
    local_limit_bytes: int = LOCAL_PROJECT_LIMIT_BYTES,
    vepfs_limit_bytes: int = VEPFS_PROJECT_LIMIT_BYTES,
    vepfs_prefixes: Sequence[Path] = VEPFS_PREFIXES,
) -> None:
    totals = {"dev-root": 0, "vepfs": 0}
    details = {"dev-root": [], "vepfs": []}
    seen: set[Path] = set()

    for path in paths:
        resolved = _resolve_path(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        bucket = "vepfs" if _is_under_prefix(resolved, vepfs_prefixes) else "dev-root"
        size_bytes = path_size_bytes(resolved)
        totals[bucket] += size_bytes
        details[bucket].append((resolved, size_bytes))

    limits = {"dev-root": local_limit_bytes, "vepfs": vepfs_limit_bytes}
    for bucket, total_bytes in totals.items():
        limit_bytes = limits[bucket]
        if total_bytes <= limit_bytes:
            continue
        entries = ", ".join(
            f"{path}={_format_bytes(size_bytes)}" for path, size_bytes in details[bucket]
        )
        raise RuntimeError(
            f"storage guardrail tripped for {bucket}: {_format_bytes(total_bytes)} exceeds "
            f"{_format_bytes(limit_bytes)}. Tracked paths: {entries}"
        )
