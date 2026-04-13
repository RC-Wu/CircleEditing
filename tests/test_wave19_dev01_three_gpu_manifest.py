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
    / "build_wave19_dev01_three_gpu_manifest.py"
)
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location("build_wave19_dev01_three_gpu_manifest", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Unable to import manifest builder from {SCRIPT_PATH}")
manifest_builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = manifest_builder
SPEC.loader.exec_module(manifest_builder)


class Wave19Dev01ThreeGpuManifestTests(unittest.TestCase):
    def test_resolution_sweep_uses_only_sane_validation_widths(self):
        manifest = manifest_builder.build_manifest()
        resolution_line = [
            item["exp_kwargs"]["resolution"]
            for item in manifest
            if item["name"].startswith("bandage_wrap_open_semboost_core_a_baseline_r")
        ]

        self.assertEqual(resolution_line, [320, 256, 384, 512, -1])
        self.assertTrue(all(res == -1 or res >= 256 for res in resolution_line))


if __name__ == "__main__":
    unittest.main()
