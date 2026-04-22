import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
DIFFGS_ROOT = ROOT.parents[1] / "diffgs_patch_20260326"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(DIFFGS_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFGS_ROOT))

from gaussian_renderer import _prepare_raster_tensor, _raster_debug_enabled
from diff_gaussian_rasterization import _tensor_debug_summary


class _Pipe:
    def __init__(self, debug: bool):
        self.debug = debug


class RasterSafetyTests(unittest.TestCase):
    def test_raster_debug_enabled_respects_env_override(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("EDITSPLAT_RASTER_DEBUG", None)
            self.assertFalse(_raster_debug_enabled(_Pipe(False)))
            self.assertTrue(_raster_debug_enabled(_Pipe(True)))

        with mock.patch.dict(os.environ, {"EDITSPLAT_RASTER_DEBUG": "0"}, clear=False):
            self.assertFalse(_raster_debug_enabled(_Pipe(True)))

        with mock.patch.dict(os.environ, {"EDITSPLAT_RASTER_DEBUG": "1"}, clear=False):
            self.assertTrue(_raster_debug_enabled(_Pipe(False)))

    def test_prepare_raster_tensor_makes_noncontiguous_tensor_contiguous(self):
        src = torch.arange(12, dtype=torch.float32).view(3, 4).transpose(0, 1)
        self.assertFalse(src.is_contiguous())

        out = _prepare_raster_tensor("src", src, device=src.device, dtype=torch.float32)

        self.assertTrue(out.is_contiguous())
        self.assertTrue(torch.allclose(out, src.contiguous()))

    def test_prepare_raster_tensor_rejects_nonfinite_values(self):
        bad = torch.tensor([1.0, float("nan")], dtype=torch.float32)

        with self.assertRaisesRegex(RuntimeError, "non-finite raster tensor"):
            _prepare_raster_tensor("bad", bad, device=bad.device, dtype=torch.float32, require_finite=True)

    def test_tensor_debug_summary_reports_shape_and_contiguity(self):
        src = torch.arange(6, dtype=torch.float32).view(2, 3).transpose(0, 1)

        summary = _tensor_debug_summary(src)

        self.assertEqual(summary["shape"], [3, 2])
        self.assertFalse(summary["contiguous"])
        self.assertTrue(summary["finite"])


if __name__ == "__main__":
    unittest.main()

