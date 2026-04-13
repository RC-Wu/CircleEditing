#!/usr/bin/env python3
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


OUT_PATH = Path(__file__).resolve().parent / "manifests" / "20260413_wave19_dev01_three_gpu.json"
MIN_VALIDATION_RESOLUTION = 256


BASE_EXP = {
    "case_name": "face",
    "ttt3r_mode": "velocity",
    "conf_power": 1.12,
    "conf_floor": 0.08,
    "prox_strength": 0.62,
    "preserve_strength": 0.04,
    "edit_boost": 1.38,
    "preserve_boost": 0.86,
    "adaptive_max_scale": 3.0,
    "schedule_power": 1.28,
    "support_views": 8,
    "include_gt_view": False,
    "optimizer_lr_scale": 1.05,
    "max_optimizer_steps": 540,
    "max_train_views": 6,
    "max_gaussians": 120000,
    "disable_densify": False,
    "freeze_geometry": False,
    "resolution": 384,
    "epoch": 2,
}


BASE_ENV = {
    "EDITSPLAT_SAM3_FIT_ROLE": "reproject",
    "EDITSPLAT_SAM3_FIT_ALPHA": "1.0",
    "EDITSPLAT_SAM3_MFG_ROLE": "reproject",
    "EDITSPLAT_SAM3_CONFIDENCE": "0.10",
    "EDITSPLAT_SAM3_CONFIDENCE_FALLBACKS": "0.16,0.12,0.08,0.04,0.0",
    "EDITSPLAT_BINARIZE_SUPPORT_MASK": "0",
    "EDITSPLAT_ELITE_CONF_CORRECTION": "1",
    "EDITSPLAT_ELITE_SUPPORT_ALPHA": "0.96",
    "EDITSPLAT_ELITE_EDIT_ALPHA": "0.18",
    "EDITSPLAT_ELITE_CONFIDENCE_ALPHA": "0.99",
    "EDITSPLAT_ELITE_SCALE_MIN": "0.95",
    "EDITSPLAT_ELITE_SCALE_MAX": "2.06",
    "EDITSPLAT_ENABLE_SEMANTIC_GS_GUIDANCE": "1",
    "EDITSPLAT_ENABLE_SEMANTIC_LOSS_MASK": "1",
    "EDITSPLAT_SEMANTIC_SUPPORT_WEIGHT": "0.82",
    "EDITSPLAT_SEMANTIC_COLOR_SCALE": "1.38",
    "EDITSPLAT_SEMANTIC_POSITION_SCALE": "1.18",
    "EDITSPLAT_SEMANTIC_MASK_POWER": "1.70",
    "EDITSPLAT_SEMANTIC_LABEL_THRESHOLD": "0.22",
    "EDITSPLAT_SEMANTIC_LABEL_BG_FLOOR": "0.00",
    "EDITSPLAT_SEMANTIC_BG_WEIGHT": "0.06",
    "EDITSPLAT_ENABLE_CANONICAL_CARRIER": "0",
}


PROMPTS = {
    "bandage_wrap": {
        "flow_tar_guidance_scale": 6.75,
        "target_prompt": (
            "the same man in the same pose and camera framing, same background and clothes, "
            "with his face partially wrapped in layered off-white medical bandages around the cheeks and forehead"
        ),
        "sampling_prompt": "the same man with layered bandages wrapped on the face, same framing and identity",
        "object_prompt": "face bandage wrap",
        "target_mask_prompt": "bandages on face",
        "notes": "full-face structured soft wrap",
    },
    "goldmask_structured": {
        "flow_tar_guidance_scale": 6.75,
        "target_prompt": (
            "the same man in the same pose and camera framing, same background and clothes, "
            "now wearing a rigid ornate reflective gold face mask with crisp metallic edges and visible structure"
        ),
        "sampling_prompt": "the same man wearing a rigid reflective gold face mask, same framing and identity",
        "object_prompt": "gold face mask",
        "target_mask_prompt": "rigid face mask on face",
        "notes": "full-face structured metal cover",
    },
    "cyborg_visor": {
        "flow_tar_guidance_scale": 6.75,
        "target_prompt": (
            "the same man in the same pose and camera framing, same background and clothes, "
            "wearing a sleek futuristic cyborg visor wrapping across the eyes and upper face with clear hard edges"
        ),
        "sampling_prompt": "the same man wearing a futuristic hard-edged cyborg visor, same framing and identity",
        "object_prompt": "cyborg visor",
        "target_mask_prompt": "futuristic visor on upper face",
        "notes": "upper-face hard structured cover",
    },
}


METHODS = {
    "open_semboost_core": {
        "exp": {},
        "env": {},
        "notes": "anchor regime",
    },
    "open_semboost_core_tightmask": {
        "exp": {},
        "env": {
            "EDITSPLAT_SAM3_CONFIDENCE": "0.16",
            "EDITSPLAT_SAM3_CONFIDENCE_FALLBACKS": "0.20,0.16,0.12,0.08,0.04",
            "EDITSPLAT_SEMANTIC_LABEL_THRESHOLD": "0.30",
            "EDITSPLAT_SEMANTIC_BG_WEIGHT": "0.03",
        },
        "notes": "tighter mask variant of anchor regime",
    },
    "open_semboost_core_gsrelax": {
        "exp": {},
        "env": {
            "EDITSPLAT_SEMANTIC_SUPPORT_WEIGHT": "0.72",
            "EDITSPLAT_SEMANTIC_COLOR_SCALE": "1.18",
            "EDITSPLAT_SEMANTIC_POSITION_SCALE": "0.90",
            "EDITSPLAT_SEMANTIC_BG_WEIGHT": "0.10",
        },
        "notes": "gaussian-guidance relaxed anchor regime",
    },
    "open_semboost_core_a_baseline": {
        "exp": {},
        "env": {
            "EDITSPLAT_CARRIER_MODE": "a_baseline",
        },
        "notes": "A-baseline carrier",
    },
    "open_semboost_core_blite": {
        "exp": {},
        "env": {
            "EDITSPLAT_ENABLE_CANONICAL_CARRIER": "1",
            "EDITSPLAT_BLITE_CANONICAL_PRIOR": "1",
            "EDITSPLAT_BLITE_CANONICAL_DUMP": "1",
            "EDITSPLAT_CANONICAL_BLEND_ALPHA": "0.65",
            "EDITSPLAT_CANONICAL_RESIDUAL_CLAMP": "0.28",
        },
        "notes": "B-lite canonical carrier",
    },
}


JOB_SPECS = [
    # GPU0: prompt-separation line
    ("bandage_wrap", "open_semboost_core", {"resolution": 384}),
    # GPU1: resolution sweep line
    ("bandage_wrap", "open_semboost_core_a_baseline", {"resolution": 320, "max_train_views": 6}),
    # GPU2: carrier stability line
    ("bandage_wrap", "open_semboost_core_blite", {"resolution": 384, "max_train_views": 6, "max_gaussians": 120000}),

    ("goldmask_structured", "open_semboost_core", {"resolution": 384}),
    ("bandage_wrap", "open_semboost_core_a_baseline", {"resolution": 256, "max_train_views": 6}),
    ("goldmask_structured", "open_semboost_core_blite", {"resolution": 384, "max_train_views": 6, "max_gaussians": 120000}),

    ("cyborg_visor", "open_semboost_core", {"resolution": 384}),
    ("bandage_wrap", "open_semboost_core_a_baseline", {"resolution": 384, "max_train_views": 6}),
    ("bandage_wrap", "open_semboost_core_blite", {"resolution": 512, "max_train_views": 8, "max_gaussians": 160000, "max_optimizer_steps": 720}),

    ("bandage_wrap", "open_semboost_core_tightmask", {"resolution": 384}),
    ("bandage_wrap", "open_semboost_core_a_baseline", {"resolution": 512, "max_train_views": 6}),
    ("goldmask_structured", "open_semboost_core_a_baseline", {"resolution": 384, "max_train_views": 8, "max_gaussians": 140000, "max_optimizer_steps": 720}),

    ("goldmask_structured", "open_semboost_core_gsrelax", {"resolution": 384}),
    ("bandage_wrap", "open_semboost_core_a_baseline", {"resolution": -1, "max_train_views": 6}),
    ("bandage_wrap", "open_semboost_core_blite", {"resolution": -1, "max_train_views": 8, "max_gaussians": 160000, "max_optimizer_steps": 720}),
]


def build_manifest() -> list[dict]:
    rows = []
    for prompt_key, method_key, exp_overrides in JOB_SPECS:
        prompt_cfg = PROMPTS[prompt_key]
        method_cfg = METHODS[method_key]
        exp = deepcopy(BASE_EXP)
        exp.update(prompt_cfg)
        exp.update(method_cfg["exp"])
        exp.update(exp_overrides)
        resolution = int(exp["resolution"])
        if resolution != -1 and resolution < MIN_VALIDATION_RESOLUTION:
            raise ValueError(
                f"wave19 validation runs require resolution >= {MIN_VALIDATION_RESOLUTION} or -1, got {resolution}"
            )
        env = deepcopy(BASE_ENV)
        env.update(method_cfg["env"])
        rows.append(
            {
                "name": f"{prompt_key}_{method_key}_r{exp['resolution']}",
                "exp_kwargs": {
                    **exp,
                    "notes": (
                        "wave19 dev01 three-gpu queue: "
                        f"{method_cfg['notes']} for {prompt_cfg['notes']} at resolution {exp['resolution']}"
                    ),
                },
                "extra_env": env,
            }
        )
    return rows


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    OUT_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(OUT_PATH)
    print(f"jobs={len(manifest)}")


if __name__ == "__main__":
    main()
