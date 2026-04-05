from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "launch_dev01_ttt3r_consistency_wave.py"
SPEC = importlib.util.spec_from_file_location("editsplat_wave_launcher", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load launcher module from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class LauncherCommandTests(unittest.TestCase):
    def test_build_command_respects_geometry_and_densify_flags(self) -> None:
        exp = MODULE.Experiment(
            name="unit",
            gpu=0,
            ttt3r_mode="velocity",
            conf_power=1.0,
            conf_floor=0.0,
            prox_strength=0.0,
            preserve_strength=0.0,
            edit_boost=1.0,
            preserve_boost=1.0,
            adaptive_max_scale=2.0,
            schedule_power=1.0,
            disable_densify=False,
            freeze_geometry=False,
            freeze_opacity=True,
        )
        cmd = MODULE.build_command(exp, "unit_wave")
        self.assertNotIn("--disable_densify", cmd)
        self.assertNotIn("--freeze_geometry", cmd)
        self.assertIn("--freeze_opacity", cmd)

    def test_build_command_includes_flags_when_enabled(self) -> None:
        exp = MODULE.Experiment(
            name="unit",
            gpu=0,
            ttt3r_mode="velocity",
            conf_power=1.0,
            conf_floor=0.0,
            prox_strength=0.0,
            preserve_strength=0.0,
            edit_boost=1.0,
            preserve_boost=1.0,
            adaptive_max_scale=2.0,
            schedule_power=1.0,
            disable_densify=True,
            freeze_geometry=True,
            freeze_opacity=False,
        )
        cmd = MODULE.build_command(exp, "unit_wave")
        self.assertIn("--disable_densify", cmd)
        self.assertIn("--freeze_geometry", cmd)
        self.assertNotIn("--freeze_opacity", cmd)


if __name__ == "__main__":
    unittest.main()
