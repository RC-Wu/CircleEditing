import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = (
    REPO_ROOT
    / "runtime"
    / "EditSplat"
    / "sandboxes"
    / "20260322_editsplat_ttt3r_flowedit_sam3_4567"
    / "scripts"
)


def load_module(name: str, filename: str):
    path = SCRIPT_DIR / filename
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"Unable to import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


launch_dev01 = load_module("launch_dev01_ttt3r_consistency_wave", "launch_dev01_ttt3r_consistency_wave.py")
launch_carrier = load_module("launch_carrier_probe_wave", "launch_carrier_probe_wave.py")


class WaveLauncherDatasetRootTests(unittest.TestCase):
    def test_default_casebank_root_points_to_live_dataset_tree(self):
        self.assertEqual(
            launch_dev01.default_casebank_root(),
            Path("/dev_vepfs/rc_wu/edit/EditSplat/dataset/dataset"),
        )
        self.assertEqual(
            launch_carrier.default_casebank_root(),
            Path("/dev_vepfs/rc_wu/edit/EditSplat/dataset/dataset"),
        )

    def test_dataset_for_case_falls_back_to_face_dataset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            face = root / "face"
            face.mkdir()
            missing_case = "bandage_wrap"
            with mock.patch.object(launch_dev01, "CASEBANK_ROOT", root), mock.patch.object(
                launch_dev01, "DATASET_FACE", face
            ):
                self.assertEqual(launch_dev01.dataset_for_case(missing_case), face)


if __name__ == "__main__":
    unittest.main()
