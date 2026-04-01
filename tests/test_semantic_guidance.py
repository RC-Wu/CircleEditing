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
    refine_semantic_guidance_mask,
    summarize_gaussian_mask,
    summarize_mask,
    summarize_mask_distribution,
    summarize_mask_overlap,
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

    def test_enabled_mode_can_refine_mask_into_harder_semantic_labels(self):
        result = build_semantic_guidance(
            selected_mask=[0.2, 0.7],
            support_mask=[0.2, 1.0],
            enabled=True,
            support_weight=0.5,
            color_scale=1.0,
            position_scale=0.8,
            freeze_geometry=False,
            mask_power=2.0,
            label_threshold=0.4,
            background_floor=0.05,
        )

        self.assertEqual(result.mask, [0.05, 0.7225])
        self.assertEqual(result.position_scale, 0.8)

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

    def test_summarize_mask_distribution_reports_threshold_coverage(self):
        stats = summarize_mask_distribution([0.0, 0.25, 0.5, 1.0])
        self.assertEqual(stats["numel"], 4)
        self.assertAlmostEqual(stats["mean"], 0.4375, places=6)
        self.assertAlmostEqual(stats["min"], 0.0, places=6)
        self.assertAlmostEqual(stats["max"], 1.0, places=6)
        self.assertAlmostEqual(stats["nonzero_ratio"], 0.75, places=6)
        self.assertAlmostEqual(stats["ge_0_25_ratio"], 0.75, places=6)
        self.assertAlmostEqual(stats["ge_0_50_ratio"], 0.5, places=6)
        self.assertAlmostEqual(stats["ge_0_90_ratio"], 0.25, places=6)

    def test_refine_semantic_guidance_mask_can_harden_foreground_and_floor_background(self):
        result = refine_semantic_guidance_mask(
            mask=[0.2, 0.7, 1.0],
            power=2.0,
            threshold=0.4,
            background_floor=0.05,
        )
        self.assertEqual(result, [0.05, 0.49, 1.0])

    def test_summarize_gaussian_mask_reports_foreground_ratio(self):
        stats = summarize_gaussian_mask(
            mask=[0.05, 0.49, 1.0, 0.0],
            label_threshold=0.4,
        )
        self.assertEqual(stats["count"], 4)
        self.assertEqual(stats["active_count"], 3)
        self.assertEqual(stats["foreground_count"], 2)
        self.assertAlmostEqual(stats["foreground_ratio"], 0.5, places=6)
        self.assertAlmostEqual(stats["mean"], 0.385, places=6)
        self.assertAlmostEqual(stats["min"], 0.0, places=6)
        self.assertAlmostEqual(stats["max"], 1.0, places=6)

    def test_summarize_mask_reports_quantiles_and_mass_above(self):
        stats = summarize_mask([0.0, 0.1, 0.6, 1.0])
        self.assertEqual(stats["count"], 4)
        self.assertAlmostEqual(stats["mean"], 0.425, places=6)
        self.assertAlmostEqual(stats["nonzero_ratio"], 0.75, places=6)
        self.assertAlmostEqual(stats["mass_above"]["0.50"], 0.5, places=6)
        self.assertAlmostEqual(stats["quantiles"]["q50"], 0.35, places=6)

    def test_summarize_mask_overlap_reports_support_leakage(self):
        stats = summarize_mask_overlap(
            selected_mask=[1.0, 1.0, 0.0, 0.0],
            support_mask=[1.0, 0.0, 1.0, 0.0],
        )
        self.assertEqual(stats["intersection"], 1)
        self.assertEqual(stats["union"], 3)
        self.assertAlmostEqual(stats["iou"], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(stats["selected_covered_by_support"], 0.5, places=6)
        self.assertAlmostEqual(stats["support_covered_by_selected"], 0.5, places=6)
        self.assertAlmostEqual(stats["support_outside_selected"], 0.5, places=6)
        self.assertAlmostEqual(stats["selected_outside_support"], 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
