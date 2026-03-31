from __future__ import annotations


def should_bootstrap_external_backend_only(
    flow_method: str,
    flow_model_key: str,
    env_enabled: bool,
) -> bool:
    method = str(flow_method or "").strip().lower()
    model_key = str(flow_model_key or "").strip().lower()
    if not bool(env_enabled):
        return False
    if method not in {"flowedit", "dnaedit"}:
        return False
    return model_key not in {"", "flux1-dev"}
