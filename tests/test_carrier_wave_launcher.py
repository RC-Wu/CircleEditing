import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = (
    REPO_ROOT
    / "runtime"
    / "EditSplat"
    / "sandboxes"
    / "20260322_editsplat_ttt3r_flowedit_sam3_4567"
    / "scripts"
    / "launch_carrier_probe_wave.py"
)
if str(LAUNCHER_PATH.parent) not in sys.path:
    sys.path.insert(0, str(LAUNCHER_PATH.parent))
SPEC = importlib.util.spec_from_file_location("carrier_wave_launcher", LAUNCHER_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Unable to import launcher module from {LAUNCHER_PATH}")
carrier_wave_launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = carrier_wave_launcher
SPEC.loader.exec_module(carrier_wave_launcher)


class CarrierWaveLauncherEnvTests(unittest.TestCase):
    def test_build_launch_env_defaults_to_cached_xulf_s_in_offline_mode(self):
        exp = carrier_wave_launcher.build_wave()[0]
        with mock.patch.dict(os.environ, {}, clear=True):
            env = carrier_wave_launcher.build_launch_env(exp)

        self.assertEqual(env["EDITSPLAT_BASE_MODEL_ID"], carrier_wave_launcher.BASE_MODEL_ID)
        self.assertEqual(env["HF_HUB_OFFLINE"], carrier_wave_launcher.DEFAULT_HF_OFFLINE)
        self.assertEqual(env["TRANSFORMERS_OFFLINE"], carrier_wave_launcher.DEFAULT_HF_OFFLINE)

    def test_build_launch_env_respects_online_prefetch_override(self):
        exp = carrier_wave_launcher.build_wave()[0]
        with mock.patch.dict(
            os.environ,
            {
                "EDITSPLAT_HF_OFFLINE": "0",
                "EDITSPLAT_BASE_MODEL_ID": "custom/model",
            },
            clear=True,
        ):
            env = carrier_wave_launcher.build_launch_env(exp)

        self.assertEqual(env["HF_HUB_OFFLINE"], "0")
        self.assertEqual(env["TRANSFORMERS_OFFLINE"], "0")
        self.assertEqual(env["EDITSPLAT_BASE_MODEL_ID"], "custom/model")

    def test_launch_one_blocks_when_storage_guardrail_trips(self):
        exp = carrier_wave_launcher.build_wave()[0]
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with mock.patch.object(carrier_wave_launcher, "LOG_DIR", tmp_path / "logs"), mock.patch.object(
                carrier_wave_launcher, "RESULTS_DIR", tmp_path / "results"
            ), mock.patch.object(
                carrier_wave_launcher, "dataset_for_case", return_value=tmp_path / "dataset"
            ), mock.patch.object(
                carrier_wave_launcher, "ensure_cfg_args"
            ), mock.patch.object(
                carrier_wave_launcher, "build_launch_env", return_value={}
            ), mock.patch.object(
                carrier_wave_launcher, "build_command", return_value=["python", "task.py"]
            ), mock.patch.object(
                carrier_wave_launcher, "enforce_storage_guardrails",
                side_effect=RuntimeError("storage guardrail tripped"),
            ), mock.patch.object(
                carrier_wave_launcher.subprocess, "Popen"
            ) as mocked_popen:
                with self.assertRaisesRegex(RuntimeError, "storage guardrail tripped"):
                    carrier_wave_launcher.launch_one(exp=exp, wave_name="wave")
        mocked_popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
