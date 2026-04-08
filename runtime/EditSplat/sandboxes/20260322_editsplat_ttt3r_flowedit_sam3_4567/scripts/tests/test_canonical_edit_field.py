from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[4] / "utils" / "canonical_edit_field.py"
)
SPEC = importlib.util.spec_from_file_location("canonical_edit_field", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load canonical carrier module from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CanonicalEditFieldTests(unittest.TestCase):
    def test_build_canonical_target_uses_reprojected_residual(self) -> None:
        gt_view = np.full((1, 3, 2, 2), 0.20, dtype=np.float32)
        reprojected_edit = np.full((1, 3, 2, 2), 0.90, dtype=np.float32)
        reprojected_source = np.full((1, 3, 2, 2), 0.40, dtype=np.float32)

        target, residual = MODULE.build_canonical_target(
            gt_view=gt_view,
            reprojected_edit=reprojected_edit,
            reprojected_source=reprojected_source,
            residual_clamp=None,
        )

        self.assertTrue(np.allclose(residual, np.full_like(residual, 0.50)))
        self.assertTrue(np.allclose(target, np.full_like(target, 0.70)))

    def test_build_canonical_target_supports_residual_clamp(self) -> None:
        gt_view = np.full((1, 3, 1, 1), 0.20, dtype=np.float32)
        reprojected_edit = np.full((1, 3, 1, 1), 0.90, dtype=np.float32)
        reprojected_source = np.full((1, 3, 1, 1), 0.40, dtype=np.float32)

        target, residual = MODULE.build_canonical_target(
            gt_view=gt_view,
            reprojected_edit=reprojected_edit,
            reprojected_source=reprojected_source,
            residual_clamp=0.10,
        )

        self.assertTrue(np.allclose(residual, np.full_like(residual, 0.10)))
        self.assertTrue(np.allclose(target, np.full_like(target, 0.30)))

    def test_blend_targets_interpolates_between_flow_and_canonical(self) -> None:
        flow_target = np.full((1, 3, 1, 1), 0.40, dtype=np.float32)
        canonical_target = np.full((1, 3, 1, 1), 0.80, dtype=np.float32)

        blended = MODULE.blend_targets(
            flow_target=flow_target,
            canonical_target=canonical_target,
            alpha=0.25,
        )

        self.assertTrue(np.allclose(blended, np.full_like(blended, 0.50)))


if __name__ == "__main__":
    unittest.main()
