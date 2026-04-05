#!/usr/bin/env python3
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


OUT_PATH = Path(__file__).resolve().parent / "manifests" / "20260406_wave16_semantic_bold_queue.json"


BASE_EXP = {
    "case_name": "face",
    "ttt3r_mode": "velocity",
    "conf_power": 1.12,
    "conf_floor": 0.08,
    "prox_strength": 0.58,
    "preserve_strength": 0.02,
    "edit_boost": 1.32,
    "preserve_boost": 0.82,
    "adaptive_max_scale": 3.0,
    "schedule_power": 1.35,
    "support_views": 8,
    "include_gt_view": False,
    "optimizer_lr_scale": 1.00,
    "max_optimizer_steps": 420,
    "max_train_views": 6,
    "max_gaussians": 110000,
    "disable_densify": True,
    "freeze_geometry": True,
}

BASE_ENV = {
    "EDITSPLAT_SAM3_FIT_ROLE": "reproject",
    "EDITSPLAT_SAM3_FIT_ALPHA": "1.0",
    "EDITSPLAT_SAM3_MFG_ROLE": "reproject",
    "EDITSPLAT_SAM3_CONFIDENCE": "0.08",
    "EDITSPLAT_SAM3_CONFIDENCE_FALLBACKS": "0.12,0.08,0.04,0.0",
    "EDITSPLAT_BINARIZE_SUPPORT_MASK": "0",
    "EDITSPLAT_ELITE_CONF_CORRECTION": "1",
    "EDITSPLAT_ELITE_SUPPORT_ALPHA": "0.95",
    "EDITSPLAT_ELITE_EDIT_ALPHA": "0.15",
    "EDITSPLAT_ELITE_CONFIDENCE_ALPHA": "0.98",
    "EDITSPLAT_ELITE_SCALE_MIN": "0.93",
    "EDITSPLAT_ELITE_SCALE_MAX": "2.00",
    "EDITSPLAT_BLITE_CANONICAL_PRIOR": "1",
    "EDITSPLAT_BLITE_CANONICAL_DUMP": "1",
}

PROMPTS = {
    "clown": {
        "target_prompt": "the same man in the same pose and camera framing, same background and clothes, with clear clown makeup: white face paint, a red clown nose, and colorful face paint",
        "sampling_prompt": "the same man with clown makeup, same framing and identity",
        "object_prompt": "face",
        "target_mask_prompt": "face",
        "notes": "full-face recolor edit",
    },
    "glasses": {
        "target_prompt": "the same man in the same pose and camera framing, same background and clothes, now wearing clear eyeglasses",
        "sampling_prompt": "the same man wearing eyeglasses, same framing and identity",
        "object_prompt": "eyeglasses",
        "target_mask_prompt": "eyeglasses",
        "notes": "small local edit",
    },
    "beard": {
        "target_prompt": "the same man in the same pose and camera framing, same background and clothes, with a clear beard added",
        "sampling_prompt": "the same man with a beard, same framing and identity",
        "object_prompt": "beard",
        "target_mask_prompt": "beard",
        "notes": "medium local edit",
    },
    "marble": {
        "target_prompt": "the same man in the same pose and camera framing, same background and clothes, transformed into a marble sculpture face",
        "sampling_prompt": "the same man as a marble sculpture, same framing and identity",
        "object_prompt": "face",
        "target_mask_prompt": "face",
        "notes": "global style edit",
    },
    "zombie": {
        "target_prompt": "the same man in the same pose and camera framing, same background and clothes, transformed into a zombie face with pale skin, dark eye makeup, and decayed facial details",
        "sampling_prompt": "the same man as a zombie, same framing and identity",
        "object_prompt": "face",
        "target_mask_prompt": "face",
        "notes": "aggressive face-style edit",
    },
    "goldmask": {
        "target_prompt": "the same man in the same pose and camera framing, same background and clothes, now wearing an ornate reflective gold face mask",
        "sampling_prompt": "the same man wearing a reflective gold face mask, same framing and identity",
        "object_prompt": "face",
        "target_mask_prompt": "face",
        "notes": "structured face-covering edit",
    },
}

REGIMES = {
    "locked_base": {
        "exp": {},
        "env": {},
        "notes": "baseline anchor",
    },
    "locked_semtight": {
        "exp": {},
        "env": {
            "EDITSPLAT_ENABLE_SEMANTIC_GS_GUIDANCE": "1",
            "EDITSPLAT_ENABLE_SEMANTIC_LOSS_MASK": "1",
            "EDITSPLAT_SEMANTIC_SUPPORT_WEIGHT": "0.75",
            "EDITSPLAT_SEMANTIC_COLOR_SCALE": "1.45",
            "EDITSPLAT_SEMANTIC_POSITION_SCALE": "1.00",
            "EDITSPLAT_SEMANTIC_MASK_POWER": "1.80",
            "EDITSPLAT_SEMANTIC_LABEL_THRESHOLD": "0.25",
            "EDITSPLAT_SEMANTIC_LABEL_BG_FLOOR": "0.02",
            "EDITSPLAT_SEMANTIC_BG_WEIGHT": "0.08",
            "EDITSPLAT_SEMANTIC_FREEZE_GEOMETRY": "1",
        },
        "notes": "semantic locked branch",
    },
    "open_semboost": {
        "exp": {
            "prox_strength": 0.62,
            "preserve_strength": 0.04,
            "edit_boost": 1.36,
            "preserve_boost": 0.86,
            "schedule_power": 1.30,
            "optimizer_lr_scale": 1.05,
            "max_optimizer_steps": 520,
            "max_train_views": 6,
            "max_gaussians": 120000,
            "disable_densify": False,
            "freeze_geometry": False,
        },
        "env": {
            "EDITSPLAT_ELITE_SUPPORT_ALPHA": "0.96",
            "EDITSPLAT_ELITE_EDIT_ALPHA": "0.18",
            "EDITSPLAT_ELITE_CONFIDENCE_ALPHA": "0.99",
            "EDITSPLAT_ELITE_SCALE_MIN": "0.95",
            "EDITSPLAT_ELITE_SCALE_MAX": "2.05",
            "EDITSPLAT_ENABLE_SEMANTIC_GS_GUIDANCE": "1",
            "EDITSPLAT_ENABLE_SEMANTIC_LOSS_MASK": "1",
            "EDITSPLAT_SEMANTIC_SUPPORT_WEIGHT": "0.80",
            "EDITSPLAT_SEMANTIC_COLOR_SCALE": "1.35",
            "EDITSPLAT_SEMANTIC_POSITION_SCALE": "1.15",
            "EDITSPLAT_SEMANTIC_MASK_POWER": "1.50",
            "EDITSPLAT_SEMANTIC_LABEL_THRESHOLD": "0.18",
            "EDITSPLAT_SEMANTIC_LABEL_BG_FLOOR": "0.01",
            "EDITSPLAT_SEMANTIC_BG_WEIGHT": "0.08",
        },
        "notes": "semantic open branch",
    },
    "open_semhammer": {
        "exp": {
            "conf_power": 1.16,
            "conf_floor": 0.06,
            "prox_strength": 0.68,
            "preserve_strength": 0.04,
            "edit_boost": 1.44,
            "preserve_boost": 0.88,
            "adaptive_max_scale": 3.2,
            "schedule_power": 1.20,
            "optimizer_lr_scale": 1.18,
            "max_optimizer_steps": 620,
            "max_train_views": 8,
            "max_gaussians": 140000,
            "disable_densify": False,
            "freeze_geometry": False,
        },
        "env": {
            "EDITSPLAT_ELITE_SUPPORT_ALPHA": "0.98",
            "EDITSPLAT_ELITE_EDIT_ALPHA": "0.12",
            "EDITSPLAT_ELITE_CONFIDENCE_ALPHA": "1.00",
            "EDITSPLAT_ELITE_SCALE_MIN": "0.98",
            "EDITSPLAT_ELITE_SCALE_MAX": "2.15",
            "EDITSPLAT_ENABLE_SEMANTIC_GS_GUIDANCE": "1",
            "EDITSPLAT_ENABLE_SEMANTIC_LOSS_MASK": "1",
            "EDITSPLAT_SEMANTIC_SUPPORT_WEIGHT": "0.85",
            "EDITSPLAT_SEMANTIC_COLOR_SCALE": "1.55",
            "EDITSPLAT_SEMANTIC_POSITION_SCALE": "1.25",
            "EDITSPLAT_SEMANTIC_MASK_POWER": "1.95",
            "EDITSPLAT_SEMANTIC_LABEL_THRESHOLD": "0.22",
            "EDITSPLAT_SEMANTIC_LABEL_BG_FLOOR": "0.00",
            "EDITSPLAT_SEMANTIC_BG_WEIGHT": "0.06",
        },
        "notes": "semantic hammer branch",
    },
}

PROMPT_ENV_OVERRIDES = {
    "glasses": {
        "locked_semtight": {
            "EDITSPLAT_SEMANTIC_SUPPORT_WEIGHT": "0.80",
            "EDITSPLAT_SEMANTIC_COLOR_SCALE": "1.55",
            "EDITSPLAT_SEMANTIC_MASK_POWER": "2.20",
            "EDITSPLAT_SEMANTIC_LABEL_THRESHOLD": "0.30",
            "EDITSPLAT_SEMANTIC_LABEL_BG_FLOOR": "0.00",
            "EDITSPLAT_SEMANTIC_BG_WEIGHT": "0.04",
        },
        "open_semboost": {
            "EDITSPLAT_SEMANTIC_SUPPORT_WEIGHT": "0.85",
            "EDITSPLAT_SEMANTIC_COLOR_SCALE": "1.45",
            "EDITSPLAT_SEMANTIC_POSITION_SCALE": "1.25",
            "EDITSPLAT_SEMANTIC_MASK_POWER": "1.90",
            "EDITSPLAT_SEMANTIC_LABEL_THRESHOLD": "0.24",
            "EDITSPLAT_SEMANTIC_LABEL_BG_FLOOR": "0.00",
            "EDITSPLAT_SEMANTIC_BG_WEIGHT": "0.05",
        },
        "open_semhammer": {
            "EDITSPLAT_SEMANTIC_SUPPORT_WEIGHT": "0.90",
            "EDITSPLAT_SEMANTIC_COLOR_SCALE": "1.65",
            "EDITSPLAT_SEMANTIC_POSITION_SCALE": "1.35",
            "EDITSPLAT_SEMANTIC_MASK_POWER": "2.30",
            "EDITSPLAT_SEMANTIC_LABEL_THRESHOLD": "0.28",
            "EDITSPLAT_SEMANTIC_BG_WEIGHT": "0.04",
        },
    },
    "beard": {
        "locked_semtight": {
            "EDITSPLAT_SEMANTIC_SUPPORT_WEIGHT": "0.80",
            "EDITSPLAT_SEMANTIC_COLOR_SCALE": "1.50",
            "EDITSPLAT_SEMANTIC_MASK_POWER": "2.00",
            "EDITSPLAT_SEMANTIC_LABEL_THRESHOLD": "0.26",
            "EDITSPLAT_SEMANTIC_LABEL_BG_FLOOR": "0.00",
            "EDITSPLAT_SEMANTIC_BG_WEIGHT": "0.05",
        },
        "open_semboost": {
            "EDITSPLAT_SEMANTIC_SUPPORT_WEIGHT": "0.85",
            "EDITSPLAT_SEMANTIC_COLOR_SCALE": "1.45",
            "EDITSPLAT_SEMANTIC_POSITION_SCALE": "1.25",
            "EDITSPLAT_SEMANTIC_MASK_POWER": "1.85",
            "EDITSPLAT_SEMANTIC_LABEL_THRESHOLD": "0.22",
            "EDITSPLAT_SEMANTIC_LABEL_BG_FLOOR": "0.00",
            "EDITSPLAT_SEMANTIC_BG_WEIGHT": "0.05",
        },
        "open_semhammer": {
            "EDITSPLAT_SEMANTIC_SUPPORT_WEIGHT": "0.90",
            "EDITSPLAT_SEMANTIC_COLOR_SCALE": "1.65",
            "EDITSPLAT_SEMANTIC_POSITION_SCALE": "1.35",
            "EDITSPLAT_SEMANTIC_MASK_POWER": "2.10",
            "EDITSPLAT_SEMANTIC_LABEL_THRESHOLD": "0.26",
            "EDITSPLAT_SEMANTIC_BG_WEIGHT": "0.04",
        },
    },
    "marble": {
        "locked_semtight": {
            "EDITSPLAT_SEMANTIC_COLOR_SCALE": "1.40",
            "EDITSPLAT_SEMANTIC_MASK_POWER": "1.75",
            "EDITSPLAT_SEMANTIC_LABEL_THRESHOLD": "0.24",
        },
        "open_semboost": {
            "EDITSPLAT_SEMANTIC_LABEL_BG_FLOOR": "0.02",
            "EDITSPLAT_SEMANTIC_BG_WEIGHT": "0.07",
        },
        "open_semhammer": {
            "EDITSPLAT_SEMANTIC_LABEL_BG_FLOOR": "0.01",
        },
    },
}


def build_manifest() -> list[dict]:
    rows = []
    for prompt_key, prompt_cfg in PROMPTS.items():
        for regime_key, regime_cfg in REGIMES.items():
            exp = deepcopy(BASE_EXP)
            exp.update(prompt_cfg)
            exp.update(regime_cfg["exp"])
            env = deepcopy(BASE_ENV)
            env.update(regime_cfg["env"])
            env.update(PROMPT_ENV_OVERRIDES.get(prompt_key, {}).get(regime_key, {}))
            rows.append(
                {
                    "name": f"{prompt_key}_{regime_key}",
                    "exp_kwargs": {
                        **exp,
                        "notes": f"wave16 {regime_cfg['notes']} for {prompt_cfg['notes']}",
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
