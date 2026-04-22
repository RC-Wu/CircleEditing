import sys
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

REPO_ROOT = Path(__file__).resolve().parents[1]
EDITSPLAT_ROOT = REPO_ROOT / 'runtime' / 'EditSplat'
if str(EDITSPLAT_ROOT) not in sys.path:
    sys.path.insert(0, str(EDITSPLAT_ROOT))

from utils.canonical_edit_field import build_frozen_canonical_carrier  # noqa: E402
from utils.ttt3r_elite_blite import (  # noqa: E402
    SourceCanonicalPrior,
    build_source_canonical_prior_mask,
    update_source_canonical_prior,
)


@unittest.skipIf(torch is None, 'torch is required for canonical carrier tests')
class CanonicalEditFieldTests(unittest.TestCase):
    def test_frozen_canonical_carrier_uses_teacher_residual_and_combined_mask(self):
        source = torch.tensor([[[[0.2]], [[0.4]], [[0.6]]]], dtype=torch.float32)
        teacher = torch.tensor([[[[0.8]], [[0.5]], [[0.1]]]], dtype=torch.float32)
        proxy = torch.tensor([[[[0.4]], [[0.9]], [[0.3]]]], dtype=torch.float32)
        edit_mask = torch.tensor([[[[0.5]]]], dtype=torch.float32)
        confidence = torch.tensor([[[[0.25]]]], dtype=torch.float32)
        support = torch.tensor([[[[0.8]]]], dtype=torch.float32)
        prior = torch.tensor([[[[0.6]]]], dtype=torch.float32)

        result = build_frozen_canonical_carrier(
            source_view=source,
            teacher_edit=teacher,
            flow_proxy=proxy,
            edit_mask=edit_mask,
            confidence_weight=confidence,
            support_mask=support,
            prior_mask=prior,
            residual_clamp=None,
            teacher_residual_weight=0.75,
            blend_alpha=0.60,
        )

        self.assertTrue(torch.allclose(result['carrier_mask'], torch.tensor([[[[0.2]]]]), atol=1e-6))
        self.assertTrue(
            torch.allclose(
                result['carrier_residual'],
                torch.tensor([[[[0.5]], [[0.2]], [[-0.45]]]], dtype=torch.float32),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                result['carrier_target'],
                torch.tensor([[[[0.3]], [[0.44]], [[0.51]]]], dtype=torch.float32),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                result['carrier_proxy'],
                torch.tensor([[[[0.34]], [[0.624]], [[0.426]]]], dtype=torch.float32),
                atol=1e-6,
            )
        )

    def test_source_canonical_prior_mask_restores_tensor_state(self):
        prior = SourceCanonicalPrior()
        ref = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
        update_source_canonical_prior(
            prior=prior,
            view_idx=3,
            edit_weight=torch.tensor([[[[0.2]]]], dtype=torch.float32),
            preserve_weight=torch.tensor([[[[0.8]]]], dtype=torch.float32),
            confidence_weight=torch.tensor([[[[0.5]]]], dtype=torch.float32),
            support_weight=torch.tensor([[[[0.6]]]], dtype=torch.float32),
        )
        update_source_canonical_prior(
            prior=prior,
            view_idx=3,
            edit_weight=torch.tensor([[[[0.6]]]], dtype=torch.float32),
            preserve_weight=torch.tensor([[[[0.4]]]], dtype=torch.float32),
            confidence_weight=torch.tensor([[[[0.75]]]], dtype=torch.float32),
            support_weight=torch.tensor([[[[0.2]]]], dtype=torch.float32),
        )

        mask = build_source_canonical_prior_mask(
            prior=prior,
            view_idx=3,
            reference=ref,
            support_floor=0.45,
            confidence_floor=0.50,
        )

        self.assertTrue(torch.allclose(mask, torch.tensor([[[[0.28125]]]], dtype=torch.float32), atol=1e-4))
        self.assertEqual(prior.to_serializable()['num_tensor_views'], 1)


if __name__ == '__main__':
    unittest.main()
