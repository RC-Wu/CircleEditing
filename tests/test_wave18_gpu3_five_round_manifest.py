import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPO_ROOT
    / "runtime"
    / "EditSplat"
    / "sandboxes"
    / "20260322_editsplat_ttt3r_flowedit_sam3_4567"
    / "scripts"
    / "build_wave18_gpu3_five_round_manifest.py"
)
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location("build_wave18_gpu3_five_round_manifest", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Unable to import manifest builder from {SCRIPT_PATH}")
manifest_builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = manifest_builder
SPEC.loader.exec_module(manifest_builder)


class Wave18Gpu3ManifestTests(unittest.TestCase):
    def test_build_manifest_contains_five_jobs_in_research_order(self):
        manifest = manifest_builder.build_manifest()
        self.assertEqual(len(manifest), 5)
        self.assertEqual(
            [item["name"] for item in manifest],
            [
                "bandage_wrap_open_semboost_core",
                "bandage_wrap_open_semboost_core_a_baseline",
                "bandage_wrap_open_semboost_core_blite",
                "goldmask_structured_open_semboost_core_a_baseline",
                "goldmask_structured_open_semboost_core_blite",
            ],
        )
        self.assertEqual(manifest[1]["extra_env"]["EDITSPLAT_CARRIER_MODE"], "a_baseline")
        self.assertEqual(manifest[2]["extra_env"]["EDITSPLAT_ENABLE_CANONICAL_CARRIER"], "1")


if __name__ == "__main__":
    unittest.main()
