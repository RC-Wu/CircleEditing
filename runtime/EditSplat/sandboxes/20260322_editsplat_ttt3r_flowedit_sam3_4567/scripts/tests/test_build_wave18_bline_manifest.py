from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "build_wave18_bline_manifest.py"
SPEC = importlib.util.spec_from_file_location("wave18_bline_manifest", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load manifest module from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Wave18BLineManifestTests(unittest.TestCase):
    def test_manifest_has_exact_eight_jobs(self) -> None:
        manifest = MODULE.build_manifest()
        self.assertEqual(len(manifest), 8)
        names = [item["name"] for item in manifest]
        self.assertEqual(len(names), len(set(names)))

    def test_manifest_covers_fullface_triplet_and_local_guard(self) -> None:
        manifest = MODULE.build_manifest()
        names = {item["name"] for item in manifest}
        expected = {
            "bandage_wrap_open_semboost_core",
            "bandage_wrap_open_semboost_core_blite",
            "goldmask_structured_open_semboost_core",
            "goldmask_structured_open_semboost_core_blite",
            "cyborg_visor_open_semboost_core",
            "cyborg_visor_open_semboost_core_blite",
            "glasses_open_semboost_core",
            "glasses_open_semboost_core_blite",
        }
        self.assertEqual(names, expected)

    def test_blite_branch_enables_canonical_carrier(self) -> None:
        manifest = MODULE.build_manifest()
        by_name = {item["name"]: item for item in manifest}
        blite = by_name["bandage_wrap_open_semboost_core_blite"]
        env = blite["extra_env"]
        self.assertEqual(env["EDITSPLAT_ENABLE_CANONICAL_CARRIER"], "1")
        self.assertIn("EDITSPLAT_CANONICAL_BLEND_ALPHA", env)
        self.assertIn("EDITSPLAT_CANONICAL_RESIDUAL_CLAMP", env)


if __name__ == "__main__":
    unittest.main()
