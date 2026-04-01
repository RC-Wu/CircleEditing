import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - local PC env may not have torch
    torch = None

REPO_ROOT = Path(__file__).resolve().parents[1]
EDITSPLAT_ROOT = REPO_ROOT / "runtime" / "EditSplat"
if str(EDITSPLAT_ROOT) not in sys.path:
    sys.path.insert(0, str(EDITSPLAT_ROOT))

from utils.semantic_guidance import accumulate_projected_gaussian_mask  # noqa: E402


def _make_camera():
    if torch is None:
        raise RuntimeError("torch is required for projection tests")
    return SimpleNamespace(
        FoVx=float(torch.pi / 2),
        FoVy=float(torch.pi / 2),
        image_width=8,
        image_height=8,
        R=torch.eye(3, dtype=torch.float32).numpy(),
        T=torch.zeros(3, dtype=torch.float32).numpy(),
    )


@unittest.skipIf(torch is None, "torch is required for projection tests")
class GaussianSupportProjectionTests(unittest.TestCase):
    def test_projects_visible_points_and_ignores_points_behind_camera(self):
        camera = _make_camera()
        gaussian_xyz = torch.tensor(
            [
                [0.0, 0.0, 2.0],
                [1.0, 0.0, 2.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=torch.float32,
        )
        image_mask = torch.zeros((1, 8, 8), dtype=torch.float32)
        image_mask[0, 4, 4] = 1.0
        image_mask[0, 4, 6] = 0.5

        weight_sum, weight_count = accumulate_projected_gaussian_mask(
            gaussian_xyz=gaussian_xyz,
            camera_list=[camera],
            image_masks=[image_mask],
            chunk_size=1,
        )

        self.assertTrue(torch.allclose(weight_sum, torch.tensor([1.0, 0.5, 0.0])))
        self.assertTrue(torch.equal(weight_count, torch.tensor([1.0, 1.0, 0.0])))

    def test_accumulates_multiple_views(self):
        camera = _make_camera()
        gaussian_xyz = torch.tensor(
            [
                [0.0, 0.0, 2.0],
                [1.0, 0.0, 2.0],
            ],
            dtype=torch.float32,
        )
        mask_a = torch.zeros((1, 8, 8), dtype=torch.float32)
        mask_a[0, 4, 4] = 1.0
        mask_a[0, 4, 6] = 0.5
        mask_b = torch.zeros((1, 8, 8), dtype=torch.float32)
        mask_b[0, 4, 4] = 0.25
        mask_b[0, 4, 6] = 0.75

        weight_sum, weight_count = accumulate_projected_gaussian_mask(
            gaussian_xyz=gaussian_xyz,
            camera_list=[camera, camera],
            image_masks=[mask_a, mask_b],
            chunk_size=8,
        )

        self.assertTrue(torch.allclose(weight_sum, torch.tensor([1.25, 1.25])))
        self.assertTrue(torch.equal(weight_count, torch.tensor([2.0, 2.0])))


if __name__ == "__main__":
    unittest.main()
