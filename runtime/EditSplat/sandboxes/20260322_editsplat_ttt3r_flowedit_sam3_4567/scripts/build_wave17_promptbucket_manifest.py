#!/usr/bin/env python3
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


OUT_PATH = Path(__file__).resolve().parent / "manifests" / "20260406_wave17_promptbucket_queue.json"


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
    "EDITSPLAT_SAM3_CONFIDENCE": "0.10",
    "EDITSPLAT_SAM3_CONFIDENCE_FALLBACKS": "0.16,0.12,0.08,0.04,0.0",
    "EDITSPLAT_BINARIZE_SUPPORT_MASK": "0",
    "EDITSPLAT_ELITE_CONF_CORRECTION": "1",
    "EDITSPLAT_ELITE_SUPPORT_ALPHA": "0.95",
    "EDITSPLAT_ELITE_EDIT_ALPHA": "0.16",
    "EDITSPLAT_ELITE_CONFIDENCE_ALPHA": "0.98",
    "EDITSPLAT_ELITE_SCALE_MIN": "0.93",
    "EDITSPLAT_ELITE_SCALE_MAX": "2.00",
    "EDITSPLAT_BLITE_CANONICAL_PRIOR": "1",
    "EDITSPLAT_BLITE_CANONICAL_DUMP": "1",
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
        "notes": "structured full-face metal cover",
        "bucket": "fullface_main",
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
        "notes": "structured full-face soft wrap",
        "bucket": "fullface_main",
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
        "notes": "structured full-face hard visor",
        "bucket": "fullface_main",
    },
    "marble_bust": {
        "flow_tar_guidance_scale": 6.72,
        "target_prompt": (
            "the same man in the same pose and camera framing, same background and clothes, "
            "with his face surface transformed into carved white marble with stone texture and sculpted features"
        ),
        "sampling_prompt": "the same man with a carved marble face surface, same framing and identity",
        "object_prompt": "marble face surface",
        "target_mask_prompt": "stone face surface",
        "notes": "full-face material replacement",
        "bucket": "fullface_main",
    },
    "glasses": {
        "target_prompt": (
            "the same man in the same pose and camera framing, same background and clothes, "
            "now wearing clear rectangular eyeglasses"
        ),
        "sampling_prompt": "the same man wearing clear eyeglasses, same framing and identity",
        "object_prompt": "eyeglasses",
        "target_mask_prompt": "eyeglasses",
        "notes": "small local anchor",
        "bucket": "local_anchor",
    },
    "beard": {
        "target_prompt": (
            "the same man in the same pose and camera framing, same background and clothes, "
            "with a clearly added beard and mustache"
        ),
        "sampling_prompt": "the same man with a beard and mustache, same framing and identity",
        "object_prompt": "beard",
        "target_mask_prompt": "beard",
        "notes": "medium local anchor",
        "bucket": "local_anchor",
    },
}

REGIMES = {
    "locked_semtight_ctrl": {
        "exp": {},
        "env": {
            "EDITSPLAT_ENABLE_SEMANTIC_GS_GUIDANCE": "1",
            "EDITSPLAT_ENABLE_SEMANTIC_LOSS_MASK": "1",
            "EDITSPLAT_SEMANTIC_SUPPORT_WEIGHT": "0.80",
            "EDITSPLAT_SEMANTIC_COLOR_SCALE": "1.42",
            "EDITSPLAT_SEMANTIC_POSITION_SCALE": "1.05",
            "EDITSPLAT_SEMANTIC_MASK_POWER": "2.05",
            "EDITSPLAT_SEMANTIC_LABEL_THRESHOLD": "0.28",
            "EDITSPLAT_SEMANTIC_LABEL_BG_FLOOR": "0.00",
            "EDITSPLAT_SEMANTIC_BG_WEIGHT": "0.04",
            "EDITSPLAT_SEMANTIC_FREEZE_GEOMETRY": "1",
        },
        "notes": "locked semantic control",
    },
    "open_semboost_core": {
        "exp": {
            "prox_strength": 0.62,
            "preserve_strength": 0.04,
            "edit_boost": 1.38,
            "preserve_boost": 0.86,
            "schedule_power": 1.28,
            "optimizer_lr_scale": 1.05,
            "max_optimizer_steps": 540,
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
        },
        "notes": "main semantic open branch",
    },
    "open_semboost_tightmask": {
        "exp": {
            "prox_strength": 0.64,
            "preserve_strength": 0.04,
            "edit_boost": 1.40,
            "preserve_boost": 0.85,
            "schedule_power": 1.26,
            "optimizer_lr_scale": 1.06,
            "max_optimizer_steps": 560,
            "max_train_views": 6,
            "max_gaussians": 122000,
            "disable_densify": False,
            "freeze_geometry": False,
        },
        "env": {
            "EDITSPLAT_ELITE_SUPPORT_ALPHA": "0.97",
            "EDITSPLAT_ELITE_EDIT_ALPHA": "0.20",
            "EDITSPLAT_ELITE_CONFIDENCE_ALPHA": "0.99",
            "EDITSPLAT_ELITE_SCALE_MIN": "0.96",
            "EDITSPLAT_ELITE_SCALE_MAX": "2.08",
            "EDITSPLAT_ENABLE_SEMANTIC_GS_GUIDANCE": "1",
            "EDITSPLAT_ENABLE_SEMANTIC_LOSS_MASK": "1",
            "EDITSPLAT_SEMANTIC_SUPPORT_WEIGHT": "0.88",
            "EDITSPLAT_SEMANTIC_COLOR_SCALE": "1.42",
            "EDITSPLAT_SEMANTIC_POSITION_SCALE": "1.28",
            "EDITSPLAT_SEMANTIC_MASK_POWER": "2.15",
            "EDITSPLAT_SEMANTIC_LABEL_THRESHOLD": "0.30",
            "EDITSPLAT_SEMANTIC_LABEL_BG_FLOOR": "0.00",
            "EDITSPLAT_SEMANTIC_BG_WEIGHT": "0.03",
        },
        "notes": "mask-selective semantic open branch",
    },
    "open_semboost_focusmask": {
        "exp": {
            "prox_strength": 0.65,
            "preserve_strength": 0.04,
            "edit_boost": 1.41,
            "preserve_boost": 0.85,
            "schedule_power": 1.24,
            "optimizer_lr_scale": 1.07,
            "max_optimizer_steps": 580,
            "max_train_views": 6,
            "max_gaussians": 124000,
            "disable_densify": False,
            "freeze_geometry": False,
        },
        "env": {
            "EDITSPLAT_SAM3_CONFIDENCE": "0.12",
            "EDITSPLAT_SAM3_CONFIDENCE_FALLBACKS": "0.18,0.14,0.10,0.06,0.0",
            "EDITSPLAT_ELITE_SUPPORT_ALPHA": "0.97",
            "EDITSPLAT_ELITE_EDIT_ALPHA": "0.20",
            "EDITSPLAT_ELITE_CONFIDENCE_ALPHA": "1.00",
            "EDITSPLAT_ELITE_SCALE_MIN": "0.96",
            "EDITSPLAT_ELITE_SCALE_MAX": "2.08",
            "EDITSPLAT_ENABLE_SEMANTIC_GS_GUIDANCE": "1",
            "EDITSPLAT_ENABLE_SEMANTIC_LOSS_MASK": "1",
            "EDITSPLAT_SEMANTIC_SUPPORT_WEIGHT": "0.90",
            "EDITSPLAT_SEMANTIC_COLOR_SCALE": "1.40",
            "EDITSPLAT_SEMANTIC_POSITION_SCALE": "1.34",
            "EDITSPLAT_SEMANTIC_MASK_POWER": "2.25",
            "EDITSPLAT_SEMANTIC_LABEL_THRESHOLD": "0.34",
            "EDITSPLAT_SEMANTIC_LABEL_BG_FLOOR": "0.00",
            "EDITSPLAT_SEMANTIC_BG_WEIGHT": "0.02",
        },
        "notes": "prompt-focused semantic open branch",
    },
    "open_semboost_cleanbg": {
        "exp": {
            "prox_strength": 0.58,
            "preserve_strength": 0.06,
            "edit_boost": 1.32,
            "preserve_boost": 0.92,
            "schedule_power": 1.28,
            "optimizer_lr_scale": 1.02,
            "max_optimizer_steps": 560,
            "max_train_views": 6,
            "max_gaussians": 118000,
            "disable_densify": False,
            "freeze_geometry": False,
        },
        "env": {
            "EDITSPLAT_ELITE_SUPPORT_ALPHA": "0.99",
            "EDITSPLAT_ELITE_EDIT_ALPHA": "0.12",
            "EDITSPLAT_ELITE_CONFIDENCE_ALPHA": "1.00",
            "EDITSPLAT_ELITE_SCALE_MIN": "0.97",
            "EDITSPLAT_ELITE_SCALE_MAX": "2.04",
            "EDITSPLAT_ENABLE_SEMANTIC_GS_GUIDANCE": "1",
            "EDITSPLAT_ENABLE_SEMANTIC_LOSS_MASK": "1",
            "EDITSPLAT_SEMANTIC_SUPPORT_WEIGHT": "0.87",
            "EDITSPLAT_SEMANTIC_COLOR_SCALE": "1.34",
            "EDITSPLAT_SEMANTIC_POSITION_SCALE": "1.22",
            "EDITSPLAT_SEMANTIC_MASK_POWER": "2.00",
            "EDITSPLAT_SEMANTIC_LABEL_THRESHOLD": "0.28",
            "EDITSPLAT_SEMANTIC_LABEL_BG_FLOOR": "0.00",
            "EDITSPLAT_SEMANTIC_BG_WEIGHT": "0.02",
        },
        "notes": "background-clean local precision branch",
    },
    "open_semboost_gsrelax": {
        "exp": {
            "conf_power": 1.14,
            "conf_floor": 0.06,
            "prox_strength": 0.68,
            "preserve_strength": 0.03,
            "edit_boost": 1.46,
            "preserve_boost": 0.84,
            "adaptive_max_scale": 3.15,
            "schedule_power": 1.18,
            "optimizer_lr_scale": 1.10,
            "max_optimizer_steps": 660,
            "max_train_views": 8,
            "max_gaussians": 136000,
            "disable_densify": False,
            "freeze_geometry": False,
        },
        "env": {
            "EDITSPLAT_ELITE_SUPPORT_ALPHA": "0.98",
            "EDITSPLAT_ELITE_EDIT_ALPHA": "0.22",
            "EDITSPLAT_ELITE_CONFIDENCE_ALPHA": "1.00",
            "EDITSPLAT_ELITE_SCALE_MIN": "0.97",
            "EDITSPLAT_ELITE_SCALE_MAX": "2.10",
            "EDITSPLAT_ENABLE_SEMANTIC_GS_GUIDANCE": "1",
            "EDITSPLAT_ENABLE_SEMANTIC_LOSS_MASK": "1",
            "EDITSPLAT_SEMANTIC_SUPPORT_WEIGHT": "0.89",
            "EDITSPLAT_SEMANTIC_COLOR_SCALE": "1.44",
            "EDITSPLAT_SEMANTIC_POSITION_SCALE": "1.30",
            "EDITSPLAT_SEMANTIC_MASK_POWER": "2.20",
            "EDITSPLAT_SEMANTIC_LABEL_THRESHOLD": "0.30",
            "EDITSPLAT_SEMANTIC_LABEL_BG_FLOOR": "0.00",
            "EDITSPLAT_SEMANTIC_BG_WEIGHT": "0.03",
        },
        "notes": "propagation-barrier open branch",
    },
    "open_semboost_softfit055": {
        "exp": {
            "prox_strength": 0.61,
            "preserve_strength": 0.05,
            "edit_boost": 1.36,
            "preserve_boost": 0.88,
            "schedule_power": 1.26,
            "optimizer_lr_scale": 1.04,
            "max_optimizer_steps": 560,
            "max_train_views": 6,
            "max_gaussians": 120000,
            "disable_densify": False,
            "freeze_geometry": False,
        },
        "env": {
            "EDITSPLAT_SAM3_FIT_ALPHA": "0.55",
            "EDITSPLAT_SAM3_CONFIDENCE": "0.08",
            "EDITSPLAT_SAM3_CONFIDENCE_FALLBACKS": "0.12,0.08,0.04,0.0",
            "EDITSPLAT_BINARIZE_SUPPORT_MASK": "0",
            "EDITSPLAT_ELITE_SUPPORT_ALPHA": "0.96",
            "EDITSPLAT_ELITE_EDIT_ALPHA": "0.16",
            "EDITSPLAT_ELITE_CONFIDENCE_ALPHA": "0.99",
            "EDITSPLAT_ELITE_SCALE_MIN": "0.95",
            "EDITSPLAT_ELITE_SCALE_MAX": "2.04",
            "EDITSPLAT_ENABLE_SEMANTIC_GS_GUIDANCE": "1",
            "EDITSPLAT_ENABLE_SEMANTIC_LOSS_MASK": "1",
            "EDITSPLAT_SEMANTIC_SUPPORT_WEIGHT": "0.84",
            "EDITSPLAT_SEMANTIC_COLOR_SCALE": "1.38",
            "EDITSPLAT_SEMANTIC_POSITION_SCALE": "1.18",
            "EDITSPLAT_SEMANTIC_MASK_POWER": "1.82",
            "EDITSPLAT_SEMANTIC_LABEL_THRESHOLD": "0.22",
            "EDITSPLAT_SEMANTIC_LABEL_BG_FLOOR": "0.00",
            "EDITSPLAT_SEMANTIC_BG_WEIGHT": "0.04",
        },
        "notes": "soft-fit support-gating probe",
    },
    "open_semhammer_probe": {
        "exp": {
            "conf_power": 1.18,
            "conf_floor": 0.04,
            "prox_strength": 0.74,
            "preserve_strength": 0.03,
            "edit_boost": 1.52,
            "preserve_boost": 0.84,
            "adaptive_max_scale": 3.30,
            "schedule_power": 1.10,
            "optimizer_lr_scale": 1.16,
            "max_optimizer_steps": 720,
            "max_train_views": 8,
            "max_gaussians": 150000,
            "disable_densify": False,
            "freeze_geometry": False,
        },
        "env": {
            "EDITSPLAT_ELITE_SUPPORT_ALPHA": "0.99",
            "EDITSPLAT_ELITE_EDIT_ALPHA": "0.20",
            "EDITSPLAT_ELITE_CONFIDENCE_ALPHA": "1.00",
            "EDITSPLAT_ELITE_SCALE_MIN": "0.98",
            "EDITSPLAT_ELITE_SCALE_MAX": "2.14",
            "EDITSPLAT_ENABLE_SEMANTIC_GS_GUIDANCE": "1",
            "EDITSPLAT_ENABLE_SEMANTIC_LOSS_MASK": "1",
            "EDITSPLAT_SEMANTIC_SUPPORT_WEIGHT": "0.92",
            "EDITSPLAT_SEMANTIC_COLOR_SCALE": "1.50",
            "EDITSPLAT_SEMANTIC_POSITION_SCALE": "1.36",
            "EDITSPLAT_SEMANTIC_MASK_POWER": "2.35",
            "EDITSPLAT_SEMANTIC_LABEL_THRESHOLD": "0.32",
            "EDITSPLAT_SEMANTIC_LABEL_BG_FLOOR": "0.00",
            "EDITSPLAT_SEMANTIC_BG_WEIGHT": "0.02",
        },
        "notes": "upper-bound hammer probe",
    },
}

PROMPT_ENV_OVERRIDES = {
    "goldmask_structured": {
        "locked_semtight_ctrl": {
            "EDITSPLAT_SEMANTIC_POSITION_SCALE": "1.12",
            "EDITSPLAT_SEMANTIC_MASK_POWER": "2.15",
            "EDITSPLAT_SEMANTIC_LABEL_THRESHOLD": "0.30",
        },
        "open_semboost_tightmask": {
            "EDITSPLAT_SEMANTIC_POSITION_SCALE": "1.32",
            "EDITSPLAT_SEMANTIC_MASK_POWER": "2.25",
            "EDITSPLAT_SEMANTIC_LABEL_THRESHOLD": "0.32",
        },
        "open_semboost_focusmask": {
            "EDITSPLAT_SEMANTIC_POSITION_SCALE": "1.36",
            "EDITSPLAT_SEMANTIC_MASK_POWER": "2.30",
            "EDITSPLAT_SEMANTIC_LABEL_THRESHOLD": "0.35",
        },
        "open_semboost_gsrelax": {
            "EDITSPLAT_SEMANTIC_POSITION_SCALE": "1.34",
            "EDITSPLAT_SEMANTIC_MASK_POWER": "2.24",
            "EDITSPLAT_SEMANTIC_LABEL_THRESHOLD": "0.31",
        },
        "open_semhammer_probe": {
            "EDITSPLAT_SEMANTIC_POSITION_SCALE": "1.40",
            "EDITSPLAT_SEMANTIC_LABEL_THRESHOLD": "0.33",
        },
    },
    "bandage_wrap": {
        "locked_semtight_ctrl": {
            "EDITSPLAT_SEMANTIC_MASK_POWER": "2.20",
            "EDITSPLAT_SEMANTIC_LABEL_THRESHOLD": "0.28",
            "EDITSPLAT_SEMANTIC_BG_WEIGHT": "0.03",
        },
        "open_semboost_tightmask": {
            "EDITSPLAT_SEMANTIC_SUPPORT_WEIGHT": "0.90",
            "EDITSPLAT_SEMANTIC_MASK_POWER": "2.30",
            "EDITSPLAT_SEMANTIC_LABEL_THRESHOLD": "0.28",
            "EDITSPLAT_SEMANTIC_BG_WEIGHT": "0.02",
        },
        "open_semboost_focusmask": {
            "EDITSPLAT_SAM3_CONFIDENCE": "0.12",
            "EDITSPLAT_SEMANTIC_SUPPORT_WEIGHT": "0.91",
            "EDITSPLAT_SEMANTIC_MASK_POWER": "2.34",
            "EDITSPLAT_SEMANTIC_LABEL_THRESHOLD": "0.29",
            "EDITSPLAT_SEMANTIC_BG_WEIGHT": "0.02",
        },
        "open_semboost_gsrelax": {
            "EDITSPLAT_SEMANTIC_SUPPORT_WEIGHT": "0.91",
            "EDITSPLAT_SEMANTIC_MASK_POWER": "2.28",
            "EDITSPLAT_SEMANTIC_LABEL_THRESHOLD": "0.28",
            "EDITSPLAT_SEMANTIC_BG_WEIGHT": "0.02",
        },
    },
    "cyborg_visor": {
        "open_semboost_tightmask": {
            "EDITSPLAT_SEMANTIC_POSITION_SCALE": "1.35",
            "EDITSPLAT_SEMANTIC_MASK_POWER": "2.28",
            "EDITSPLAT_SEMANTIC_LABEL_THRESHOLD": "0.31",
            "EDITSPLAT_SEMANTIC_BG_WEIGHT": "0.02",
        },
        "open_semboost_gsrelax": {
            "EDITSPLAT_SEMANTIC_POSITION_SCALE": "1.36",
            "EDITSPLAT_SEMANTIC_LABEL_THRESHOLD": "0.31",
            "EDITSPLAT_SEMANTIC_BG_WEIGHT": "0.02",
        },
        "open_semhammer_probe": {
            "EDITSPLAT_SEMANTIC_POSITION_SCALE": "1.42",
            "EDITSPLAT_SEMANTIC_LABEL_THRESHOLD": "0.34",
        },
    },
    "marble_bust": {
        "open_semboost_core": {
            "EDITSPLAT_SEMANTIC_COLOR_SCALE": "1.48",
            "EDITSPLAT_SEMANTIC_POSITION_SCALE": "1.10",
            "EDITSPLAT_SEMANTIC_LABEL_BG_FLOOR": "0.02",
            "EDITSPLAT_SEMANTIC_BG_WEIGHT": "0.07",
        },
        "open_semboost_tightmask": {
            "EDITSPLAT_SEMANTIC_COLOR_SCALE": "1.50",
            "EDITSPLAT_SEMANTIC_POSITION_SCALE": "1.16",
            "EDITSPLAT_SEMANTIC_MASK_POWER": "2.00",
            "EDITSPLAT_SEMANTIC_LABEL_THRESHOLD": "0.27",
            "EDITSPLAT_SEMANTIC_LABEL_BG_FLOOR": "0.02",
            "EDITSPLAT_SEMANTIC_BG_WEIGHT": "0.05",
        },
        "open_semboost_gsrelax": {
            "EDITSPLAT_SEMANTIC_COLOR_SCALE": "1.52",
            "EDITSPLAT_SEMANTIC_POSITION_SCALE": "1.18",
            "EDITSPLAT_SEMANTIC_LABEL_BG_FLOOR": "0.02",
            "EDITSPLAT_SEMANTIC_BG_WEIGHT": "0.05",
        },
    },
    "glasses": {
        "open_semboost_tightmask": {
            "EDITSPLAT_SEMANTIC_SUPPORT_WEIGHT": "0.86",
            "EDITSPLAT_SEMANTIC_MASK_POWER": "2.05",
            "EDITSPLAT_SEMANTIC_LABEL_THRESHOLD": "0.26",
            "EDITSPLAT_SEMANTIC_BG_WEIGHT": "0.04",
        },
        "open_semboost_cleanbg": {
            "EDITSPLAT_SEMANTIC_SUPPORT_WEIGHT": "0.88",
            "EDITSPLAT_SEMANTIC_MASK_POWER": "1.95",
            "EDITSPLAT_SEMANTIC_LABEL_THRESHOLD": "0.27",
        },
        "open_semboost_gsrelax": {
            "EDITSPLAT_SEMANTIC_SUPPORT_WEIGHT": "0.87",
            "EDITSPLAT_SEMANTIC_POSITION_SCALE": "1.26",
            "EDITSPLAT_SEMANTIC_MASK_POWER": "2.10",
            "EDITSPLAT_SEMANTIC_LABEL_THRESHOLD": "0.27",
            "EDITSPLAT_SEMANTIC_BG_WEIGHT": "0.03",
        },
    },
    "beard": {
        "open_semboost_tightmask": {
            "EDITSPLAT_SEMANTIC_SUPPORT_WEIGHT": "0.86",
            "EDITSPLAT_SEMANTIC_POSITION_SCALE": "1.26",
            "EDITSPLAT_SEMANTIC_MASK_POWER": "2.05",
            "EDITSPLAT_SEMANTIC_LABEL_THRESHOLD": "0.25",
            "EDITSPLAT_SEMANTIC_BG_WEIGHT": "0.04",
        },
        "open_semboost_cleanbg": {
            "EDITSPLAT_SEMANTIC_SUPPORT_WEIGHT": "0.88",
            "EDITSPLAT_SEMANTIC_POSITION_SCALE": "1.24",
            "EDITSPLAT_SEMANTIC_MASK_POWER": "1.98",
            "EDITSPLAT_SEMANTIC_LABEL_THRESHOLD": "0.26",
        },
        "open_semboost_gsrelax": {
            "EDITSPLAT_SEMANTIC_SUPPORT_WEIGHT": "0.88",
            "EDITSPLAT_SEMANTIC_POSITION_SCALE": "1.28",
            "EDITSPLAT_SEMANTIC_MASK_POWER": "2.12",
            "EDITSPLAT_SEMANTIC_LABEL_THRESHOLD": "0.27",
            "EDITSPLAT_SEMANTIC_BG_WEIGHT": "0.03",
        },
    },
}

JOB_SPECS = [
    ("goldmask_structured", "open_semboost_core"),
    ("goldmask_structured", "open_semboost_tightmask"),
    ("goldmask_structured", "open_semboost_focusmask"),
    ("bandage_wrap", "open_semboost_core"),
    ("bandage_wrap", "open_semboost_tightmask"),
    ("bandage_wrap", "open_semboost_focusmask"),
    ("cyborg_visor", "open_semboost_core"),
    ("cyborg_visor", "open_semboost_tightmask"),
    ("marble_bust", "open_semboost_core"),
    ("marble_bust", "open_semboost_tightmask"),
    ("glasses", "open_semboost_core"),
    ("glasses", "open_semboost_tightmask"),
    ("glasses", "open_semboost_cleanbg"),
    ("glasses", "open_semboost_gsrelax"),
    ("beard", "open_semboost_core"),
    ("beard", "open_semboost_tightmask"),
    ("beard", "open_semboost_cleanbg"),
    ("beard", "open_semboost_gsrelax"),
    ("goldmask_structured", "locked_semtight_ctrl"),
    ("goldmask_structured", "open_semboost_gsrelax"),
    ("goldmask_structured", "open_semboost_softfit055"),
    ("bandage_wrap", "locked_semtight_ctrl"),
    ("bandage_wrap", "open_semboost_gsrelax"),
    ("bandage_wrap", "open_semboost_softfit055"),
]


def build_manifest() -> list[dict]:
    rows = []
    for prompt_key, regime_key in JOB_SPECS:
        prompt_cfg = PROMPTS[prompt_key]
        regime_cfg = REGIMES[regime_key]
        exp = deepcopy(BASE_EXP)
        exp.update({key: value for key, value in prompt_cfg.items() if key != "bucket"})
        exp.update(regime_cfg["exp"])
        env = deepcopy(BASE_ENV)
        env.update(regime_cfg["env"])
        env.update(PROMPT_ENV_OVERRIDES.get(prompt_key, {}).get(regime_key, {}))
        rows.append(
            {
                "name": f"{prompt_key}_{regime_key}",
                "exp_kwargs": {
                    **exp,
                    "notes": (
                        f"wave17 {prompt_cfg['bucket']} {regime_cfg['notes']} "
                        f"for {prompt_cfg['notes']}"
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
