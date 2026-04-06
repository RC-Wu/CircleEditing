from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "build_wave17_promptbucket_manifest.py"
SPEC = importlib.util.spec_from_file_location("wave17_promptbucket_manifest", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load manifest module from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Wave17PromptBucketManifestTests(unittest.TestCase):
    def test_manifest_has_curated_24_job_budget(self) -> None:
        manifest = MODULE.build_manifest()
        self.assertEqual(len(manifest), 24)
        names = [item["name"] for item in manifest]
        self.assertEqual(len(names), len(set(names)))

    def test_budget_is_split_across_three_directions(self) -> None:
        manifest = MODULE.build_manifest()
        names = [item["name"] for item in manifest]

        self.assertEqual(sum("locked_semtight_ctrl" in name for name in names), 2)
        self.assertEqual(sum("open_semboost_softfit055" in name for name in names), 2)

        fullface_main = {
            "goldmask_structured_open_semboost_core",
            "goldmask_structured_open_semboost_tightmask",
            "goldmask_structured_open_semboost_focusmask",
            "bandage_wrap_open_semboost_core",
            "bandage_wrap_open_semboost_tightmask",
            "bandage_wrap_open_semboost_focusmask",
            "cyborg_visor_open_semboost_core",
            "cyborg_visor_open_semboost_tightmask",
            "marble_bust_open_semboost_core",
            "marble_bust_open_semboost_tightmask",
        }
        self.assertTrue(fullface_main.issubset(set(names)))

        local_precision = {
            "glasses_open_semboost_core",
            "glasses_open_semboost_tightmask",
            "glasses_open_semboost_cleanbg",
            "glasses_open_semboost_gsrelax",
            "beard_open_semboost_core",
            "beard_open_semboost_tightmask",
            "beard_open_semboost_cleanbg",
            "beard_open_semboost_gsrelax",
        }
        self.assertTrue(local_precision.issubset(set(names)))

        barrier_probe = {
            "goldmask_structured_locked_semtight_ctrl",
            "goldmask_structured_open_semboost_gsrelax",
            "goldmask_structured_open_semboost_softfit055",
            "bandage_wrap_locked_semtight_ctrl",
            "bandage_wrap_open_semboost_gsrelax",
            "bandage_wrap_open_semboost_softfit055",
        }
        self.assertTrue(barrier_probe.issubset(set(names)))


if __name__ == "__main__":
    unittest.main()
