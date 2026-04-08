#!/usr/bin/env python3
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


OUT_PATH = Path(__file__).resolve().parent / "manifests" / "20260408_wave18_bline_gpu123_queue.json"


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
    "EDITSPLAT_BLITE_CANONICAL_PRIOR": "1",
    "EDITSPLAT_BLITE_CANONICAL_DUMP": "1",
    "EDITSPLAT_ENABLE_SEMANTIC_GS_GUIDANCE": "1",
    "EDITSPLAT_ENABLE_SEMANTIC_LOSS_MASK": "1",
    "EDITSPLAT_SEMANTIC_SUPPORT_WEIGHT": "0.82",
    "EDITSPLAT_SEMANTIC_COLOR_SCALE": "1.38",
    "EDITSPLAT_SEMANTIC_POSITION_SCALE": "1.18",
    "EDITSPLAT_SEMANTIC_MASK_POWER": "1.70",
    "EDITSPLAT_SEMANTIC_LABEL_THRESHOLD": "0.22",
    "EDITSPLAT_SEMANTIC_LABEL_BG_FLOOR": "0.00",
    "EDITSPLAT_SEMANTIC_BG_WEIGHT": "0.06",
}

PROMPTS = {
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
    "cyborg_visor": {
        "flow_tar_guidance_scale": 6.78,
        "target_prompt": (
            "the same man in the same pose and camera framing, same background and clothes, "
            "wearing a rigid dark metallic cybernetic visor and segmented faceplate with hard mechanical edges"
        ),
        "sampling_prompt": "the same man with a metallic cybernetic visor and face plate, same framing and identity",
        "object_prompt": "metal face visor",
        "target_mask_prompt": "visor and face plate",
        "notes": "full-face structured hard visor",
    },
    "glasses": {
        "target_prompt": (
            "the same man in the same pose and camera framing, same background and clothes, "
            "now wearing clear rectangular eyeglasses"
        ),
        "sampling_prompt": "the same man wearing clear eyeglasses, same framing and identity",
        "object_prompt": "eyeglasses",
        "target_mask_prompt": "eyeglasses",
        "notes": "local safeguard anchor",
    },
}

METHODS = {
    "open_semboost_core": {
        "exp": {},
        "env": {
            "EDITSPLAT_ENABLE_CANONICAL_CARRIER": "0",
        },
        "notes": "wave18 anchor core",
    },
    "open_semboost_core_blite": {
        "exp": {},
        "env": {
            "EDITSPLAT_ENABLE_CANONICAL_CARRIER": "1",
            "EDITSPLAT_CANONICAL_BLEND_ALPHA": "0.65",
            "EDITSPLAT_CANONICAL_RESIDUAL_CLAMP": "0.28",
        },
        "notes": "wave18 B-lite canonical carrier",
    },
}

JOB_SPECS = [
    ("bandage_wrap", "open_semboost_core"),
    ("bandage_wrap", "open_semboost_core_blite"),
    ("goldmask_structured", "open_semboost_core"),
    ("goldmask_structured", "open_semboost_core_blite"),
    ("cyborg_visor", "open_semboost_core"),
    ("cyborg_visor", "open_semboost_core_blite"),
    ("glasses", "open_semboost_core"),
    ("glasses", "open_semboost_core_blite"),
]


def build_manifest() -> list[dict]:
    rows = []
    for prompt_key, method_key in JOB_SPECS:
        prompt_cfg = PROMPTS[prompt_key]
        method_cfg = METHODS[method_key]
        exp = deepcopy(BASE_EXP)
        exp.update(prompt_cfg)
        exp.update(method_cfg["exp"])
        env = deepcopy(BASE_ENV)
        env.update(method_cfg["env"])
        rows.append(
            {
                "name": f"{prompt_key}_{method_key}",
                "exp_kwargs": {
                    **exp,
                    "notes": f"wave18 {method_cfg['notes']} for {prompt_cfg['notes']}",
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
