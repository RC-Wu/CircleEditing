from __future__ import annotations


def parse_hf_token_line(raw: str) -> str:
    token = str(raw or "").strip()
    if "=" in token and not token.startswith("hf_"):
        token = token.split("=", 1)[1].strip()
    return token.strip('"').strip("'")


def iter_mask_prompts(primary: str):
    base = str(primary or "").strip()
    seen = set()
    if base:
        seen.add(base)
        yield base
    for alt in ("person", "person face", "human face", "head", "portrait"):
        if alt not in seen:
            seen.add(alt)
            yield alt


def resolve_sam3_backend_request(raw: str) -> str:
    backend = str(raw or "").strip().lower()
    if backend in {"", "sam3", "auto"}:
        return "sam3"
    if backend in {"stub", "full", "full-image", "full_image"}:
        return "stub"
    raise ValueError(
        f"Unsupported mask backend for SAM3-only pipeline: {backend or '<empty>'}. "
        "Only sam3 or explicit full-image stub are allowed."
    )
