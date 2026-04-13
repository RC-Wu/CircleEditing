import sys
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

REPO_ROOT = Path(__file__).resolve().parents[1]
EDITSPLAT_ROOT = REPO_ROOT / "runtime" / "EditSplat"
if str(EDITSPLAT_ROOT) not in sys.path:
    sys.path.insert(0, str(EDITSPLAT_ROOT))

from utils.carrier_baseline import (  # noqa: E402
    build_a_baseline_carrier,
    coerce_tensor_like,
    prompt_separation_retention_ratio,
)


@unittest.skipIf(torch is None, "torch is required for carrier baseline tests")
class CarrierBaselineTests(unittest.TestCase):
    def test_build_a_baseline_carrier_blends_teacher_and_support_residuals_under_combined_mask(self):
        source = torch.tensor([[[[0.2]], [[0.4]], [[0.6]]]], dtype=torch.float32)
        initial_edit = torch.tensor([[[[0.6]], [[0.5]], [[0.2]]]], dtype=torch.float32)
        mf_cond = torch.tensor([[[[0.4]], [[0.9]], [[0.4]]]], dtype=torch.float32)
        proxy = torch.tensor([[[[0.1]], [[0.1]], [[0.1]]]], dtype=torch.float32)
        geo_weight = torch.tensor([[[[0.25]]]], dtype=torch.float32)
        support_mask = torch.tensor([[[[0.75]]]], dtype=torch.float32)

        result = build_a_baseline_carrier(
            source=source,
            initial_edit=initial_edit,
            mf_cond=mf_cond,
            proxy=proxy,
            geo_weight=geo_weight,
            support_mask=support_mask,
            support_mix=0.25,
            proxy_mix=0.5,
            mask_floor=0.1,
        )

        expected_mask = torch.tensor([[[[0.75]]]], dtype=torch.float32)
        expected_residual = torch.tensor([[[[0.35]], [[0.2]], [[-0.35]]]], dtype=torch.float32)
        expected_target = torch.tensor([[[[0.4625]], [[0.55]], [[0.3375]]]], dtype=torch.float32)
        expected_proxy = torch.tensor([[[[0.28125]], [[0.325]], [[0.21875]]]], dtype=torch.float32)

        self.assertTrue(torch.allclose(result["carrier_mask"], expected_mask, atol=1e-6))
        self.assertTrue(torch.allclose(result["carrier_residual"], expected_residual, atol=1e-6))
        self.assertTrue(torch.allclose(result["carrier_target"], expected_target, atol=1e-6))
        self.assertTrue(torch.allclose(result["carrier_proxy"], expected_proxy, atol=1e-6))

    def test_psrr_is_small_when_final_separation_collapses(self):
        self.assertAlmostEqual(
            prompt_separation_retention_ratio(teacher_mad=2.93, final_mad=0.000237),
            0.000080887,
            places=9,
        )

    def test_psrr_uses_epsilon_for_zero_teacher_gap(self):
        self.assertEqual(prompt_separation_retention_ratio(teacher_mad=0.0, final_mad=0.0), 0.0)
        self.assertGreater(prompt_separation_retention_ratio(teacher_mad=0.0, final_mad=0.1), 1000.0)

    def test_coerce_tensor_like_matches_reference_dtype_and_device(self):
        reference = torch.zeros((1, 3, 4, 4), dtype=torch.float16)
        value = torch.ones((1, 3, 4, 4), dtype=torch.float32)

        result = coerce_tensor_like(value, reference)

        self.assertEqual(result.dtype, reference.dtype)
        self.assertEqual(result.device, reference.device)
        self.assertTrue(torch.allclose(result, torch.ones_like(reference)))


if __name__ == "__main__":
    unittest.main()
