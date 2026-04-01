import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EDITSPLAT_ROOT = REPO_ROOT / "runtime" / "EditSplat"
if str(EDITSPLAT_ROOT) not in sys.path:
    sys.path.insert(0, str(EDITSPLAT_ROOT))

from utils.semantic_guidance import (  # noqa: E402
    build_semantic_guidance,
    expand_loss_guidance_mask,
    normalize_gaussian_support_mask,
)


class SemanticGuidanceTests(unittest.TestCase):
    def test_disabled_mode_is_a_no_op(self):
        selected_mask = [0.2, 0.8, 0.4]
        support_mask = [0.9, 0.1, 0.0]

        result = build_semantic_guidance(
            selected_mask=selected_mask,
            support_mask=support_mask,
            enabled=False,
            support_weight=0.75,
            color_scale=1.0,
            position_scale=0.5,
            freeze_geometry=False,
        )

        self.assertEqual(result.mask, selected_mask)
        self.assertEqual(result.color_scale, 1.0)
        self.assertEqual(result.position_scale, 0.5)
        self.assertFalse(result.used_support)

    def test_enabled_mode_raises_color_guidance_and_freezes_position(self):
        selected_mask = [0.1, 0.3, 0.0]
        support_mask = [0.0, 0.8, 0.6]

        result = build_semantic_guidance(
            selected_mask=selected_mask,
            support_mask=support_mask,
            enabled=True,
            support_weight=0.5,
            color_scale=1.25,
            position_scale=0.8,
            freeze_geometry=True,
        )

        expected_mask = [0.1, 0.55, 0.3]
        self.assertEqual(result.mask, expected_mask)
        self.assertEqual(result.color_scale, 1.25)
        self.assertEqual(result.position_scale, 0.0)
        self.assertTrue(result.used_support)

    def test_loss_guidance_mask_keeps_background_floor(self):
        result = expand_loss_guidance_mask(mask=[0.0, 1.0, 0.5], background_weight=0.1)
        self.assertEqual(result, [0.1, 1.0, 0.55])

    def test_normalize_gaussian_support_mask_handles_zero_counts(self):
        result = normalize_gaussian_support_mask(
            weight_sum=[0.6, 0.0, 0.2],
            weight_count=[2.0, 0.0, 1.0],
        )
        self.assertAlmostEqual(result[0], 0.3, places=6)
        self.assertAlmostEqual(result[1], 0.0, places=6)
        self.assertAlmostEqual(result[2], 0.2, places=6)


if __name__ == "__main__":
    unittest.main()
