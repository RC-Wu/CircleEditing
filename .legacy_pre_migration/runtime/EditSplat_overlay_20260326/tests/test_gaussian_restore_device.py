import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene.gaussian_model import _as_parameter_on_device, _move_optimizer_state_to_device


class GaussianRestoreDeviceTests(unittest.TestCase):
    def assertSameDevice(self, lhs: torch.device, rhs: torch.device):
        self.assertEqual(lhs.type, rhs.type)
        if rhs.type == "cuda" and rhs.index is not None:
            self.assertEqual(lhs.index, rhs.index)

    def test_as_parameter_on_device_moves_tensor(self):
        target = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        param = _as_parameter_on_device(torch.ones(3), target)
        self.assertIsInstance(param, torch.nn.Parameter)
        self.assertSameDevice(param.device, target)
        self.assertTrue(param.requires_grad)

    def test_move_optimizer_state_to_device_moves_tensor_slots(self):
        target = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        param = torch.nn.Parameter(torch.ones(3, device=target))
        optimizer = torch.optim.Adam([param], lr=0.1)
        loss = (param ** 2).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        _move_optimizer_state_to_device(optimizer, target)

        for state in optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    self.assertSameDevice(value.device, target)


if __name__ == "__main__":
    unittest.main()

