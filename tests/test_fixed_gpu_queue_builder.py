import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPO_ROOT
    / "runtime"
    / "EditSplat"
    / "sandboxes"
    / "20260322_editsplat_ttt3r_flowedit_sam3_4567"
    / "scripts"
    / "build_fixed_gpu_overnight_queue.py"
)
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location("build_fixed_gpu_overnight_queue", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Unable to import queue builder from {SCRIPT_PATH}")
build_fixed_gpu_overnight_queue = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = build_fixed_gpu_overnight_queue
SPEC.loader.exec_module(build_fixed_gpu_overnight_queue)


class BuildFixedGpuQueueTests(unittest.TestCase):
    def test_gc_job_recovers_launcher_from_existing_slot_script(self):
        tmp_path = Path(tempfile.mkdtemp(prefix="queue_builder_test_"))
        self.addCleanup(lambda: shutil.rmtree(tmp_path, ignore_errors=True))

        launcher_module = tmp_path / "wave19_launcher.py"
        launcher_module.write_text(
            "\n".join(
                [
                    "from dataclasses import dataclass",
                    "",
                    "@dataclass",
                    "class Experiment:",
                    "    name: str",
                    "    gpu: int",
                    '    case_name: str = "face"',
                    "    resolution: int = 384",
                    "    epoch: int = 2",
                    "",
                    "def build_run_name(exp: Experiment, wave_name: str) -> str:",
                    '    return f\"{wave_name}_{exp.case_name}_{exp.name}_r{exp.resolution}\"',
                    "",
                ]
            ),
            encoding="utf-8",
        )

        queue_root = tmp_path / "queues"
        wave_name = "wave19"
        queue_wave_root = queue_root / wave_name
        (queue_wave_root / "scripts").mkdir(parents=True)
        (queue_wave_root / "scripts" / "slot_gpu0.sh").write_text(
            (
                "python3 build_fixed_gpu_overnight_queue.py run-one "
                f"--launcher-module {launcher_module} "
                "--queue-root /tmp/queues --wave-name wave19 --slot-gpu 0 --job-json /tmp/job.json\n"
            ),
            encoding="utf-8",
        )

        job_json = queue_wave_root / "job.json"
        job_json.write_text(
            json.dumps(
                {
                    "name": "carrier_probe",
                    "exp_kwargs": {
                        "case_name": "face",
                        "resolution": 384,
                        "epoch": 2,
                    },
                    "extra_env": {},
                }
            ),
            encoding="utf-8",
        )

        run_root = queue_wave_root / "results" / "wave19_face_carrier_probe_r384"
        checkpoint = run_root / "point_cloud" / "iteration_7" / "chkpnt7000.pth"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_text("stub", encoding="utf-8")

        build_fixed_gpu_overnight_queue.gc_job(
            queue_root=queue_root,
            wave_name=wave_name,
            job_json=job_json,
        )

        self.assertFalse(checkpoint.exists())

    def test_render_slot_script_passes_launcher_to_gc_job(self):
        slot = build_fixed_gpu_overnight_queue.QueueSlot(gpu=1)
        jobs = [build_fixed_gpu_overnight_queue.QueueJob(name="carrier_probe")]
        launcher_module_path = Path("/tmp/launch_carrier_probe_wave.py")

        script_text = build_fixed_gpu_overnight_queue.render_slot_script(
            slot=slot,
            jobs=jobs,
            launcher_module_path=launcher_module_path,
            queue_root=Path("/tmp/queues"),
            wave_name="wave19",
            queue_script_path=Path("/tmp/build_fixed_gpu_overnight_queue.py"),
        )

        self.assertIn("gc-job", script_text)
        self.assertIn(f"--launcher-module {launcher_module_path}", script_text)

    def test_postprocess_job_uses_experiment_resolution_for_render(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            queue_root = tmp_path / "queue"
            wave_name = "wave"
            job_json = tmp_path / "job.json"
            job_json.write_text(
                json.dumps(
                    {
                        "name": "bandage_wrap_open_semboost_core",
                        "exp_kwargs": {
                            "case_name": "face",
                            "ttt3r_mode": "velocity",
                            "conf_power": 1.0,
                            "conf_floor": 0.0,
                            "prox_strength": 0.0,
                            "preserve_strength": 0.0,
                            "edit_boost": 1.0,
                            "preserve_boost": 1.0,
                            "adaptive_max_scale": 3.0,
                            "schedule_power": 2.0,
                            "resolution": 384,
                        },
                        "extra_env": {},
                    }
                ),
                encoding="utf-8",
            )

            model_path = queue_root / wave_name / "results" / f"{wave_name}_bandage_wrap_open_semboost_core"
            point_cloud_dir = model_path / "point_cloud" / "iteration_7004"
            point_cloud_dir.mkdir(parents=True, exist_ok=True)

            class DummyExperiment:
                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)
                    self.resolution = kwargs.get("resolution", 384)
                    self.case_name = kwargs.get("case_name", "face")

            class DummyLauncher:
                PYTHON = Path("/usr/bin/python3")
                Experiment = DummyExperiment

                @staticmethod
                def build_run_name(exp, wave_name):
                    return f"{wave_name}_{exp.name}"

                @staticmethod
                def dataset_for_case(_case_name):
                    return tmp_path / "dataset"

            with mock.patch.object(
                build_fixed_gpu_overnight_queue,
                "load_launcher_module",
                return_value=DummyLauncher,
            ), mock.patch.object(
                build_fixed_gpu_overnight_queue.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=0),
            ) as mocked_run:
                build_fixed_gpu_overnight_queue.postprocess_job(
                    launcher_module_path=SCRIPT_PATH,
                    queue_root=queue_root,
                    wave_name=wave_name,
                    slot_gpu=0,
                    job_json=job_json,
                )

        render_cmd = mocked_run.call_args_list[0].args[0]
        resolution_index = render_cmd.index("--resolution")
        self.assertEqual(render_cmd[resolution_index + 1], "384")

    def test_run_one_job_uses_launcher_build_launch_env(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            queue_root = tmp_path / "queue"
            job_json = tmp_path / "job.json"
            job_json.write_text(
                json.dumps(
                    {
                        "name": "bandage_wrap_open_semboost_core",
                        "exp_kwargs": {
                            "case_name": "face",
                            "ttt3r_mode": "velocity",
                            "conf_power": 1.0,
                            "conf_floor": 0.0,
                            "prox_strength": 0.0,
                            "preserve_strength": 0.0,
                            "edit_boost": 1.0,
                            "preserve_boost": 1.0,
                            "adaptive_max_scale": 3.0,
                            "schedule_power": 2.0,
                        },
                        "extra_env": {
                            "EDITSPLAT_CARRIER_MODE": "a_baseline",
                        },
                    }
                ),
                encoding="utf-8",
            )

            class DummyExperiment:
                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)
                    self.dump_intermediates = True
                    self.mask_backend = "sam3"
                    self.max_train_views = 6
                    self.max_gaussians = 120000

            class DummyLauncher:
                HF_HOME = tmp_path / "hf_home"
                HF_TOKEN = tmp_path / "token"
                SAM3_PT = tmp_path / "sam3.pt"
                PYTHON = Path("/usr/bin/python3")
                ROOT = tmp_path
                LOG_DIR = tmp_path / "logs"
                RESULTS_DIR = tmp_path / "results"
                SUMMARY_DIR = tmp_path / "summaries"
                Experiment = DummyExperiment

                @staticmethod
                def ensure_layout():
                    for path in (
                        DummyLauncher.LOG_DIR,
                        DummyLauncher.RESULTS_DIR,
                        DummyLauncher.SUMMARY_DIR,
                    ):
                        path.mkdir(parents=True, exist_ok=True)

                @staticmethod
                def dataset_for_case(_case_name):
                    source = tmp_path / "dataset"
                    source.mkdir(exist_ok=True)
                    return source

                @staticmethod
                def ensure_cfg_args(model_path, source_path):
                    model_path.mkdir(parents=True, exist_ok=True)
                    cfg_path = model_path / "cfg_args"
                    cfg_path.write_text(
                        f"source_path={source_path}",
                        encoding="utf-8",
                    )
                    return cfg_path

                @staticmethod
                def build_launch_env(exp):
                    return {
                        "CUDA_VISIBLE_DEVICES": str(exp.gpu),
                        "HF_HOME": str(DummyLauncher.HF_HOME),
                        "HF_HUB_CACHE": str(DummyLauncher.HF_HOME / "hub"),
                        "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1",
                        "EDITSPLAT_BASE_MODEL_ID": "cocktailpeanut/xulf-s",
                    }

                @staticmethod
                def build_command(exp, wave_name):
                    return ["python3", "run.py", exp.name, wave_name]

                @staticmethod
                def build_run_name(exp, wave_name):
                    return f"{wave_name}_{exp.name}"

                @staticmethod
                def collect_summary(runs, wave_name):
                    summary_path = DummyLauncher.SUMMARY_DIR / f"{wave_name}_summary.json"
                    summary_path.parent.mkdir(parents=True, exist_ok=True)
                    summary_path.write_text(json.dumps(runs), encoding="utf-8")
                    return summary_path

            with mock.patch.object(
                build_fixed_gpu_overnight_queue,
                "load_launcher_module",
                return_value=DummyLauncher,
            ), mock.patch.object(
                build_fixed_gpu_overnight_queue.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=0),
            ) as mocked_run:
                rc = build_fixed_gpu_overnight_queue.run_one_job(
                    launcher_module_path=SCRIPT_PATH,
                    queue_root=queue_root,
                    wave_name="wave",
                    slot_gpu=3,
                    job_json=job_json,
                )

        self.assertEqual(rc, 0)
        run_env = mocked_run.call_args.kwargs["env"]
        self.assertEqual(run_env["EDITSPLAT_BASE_MODEL_ID"], "cocktailpeanut/xulf-s")
        self.assertEqual(run_env["EDITSPLAT_CARRIER_MODE"], "a_baseline")


if __name__ == "__main__":
    unittest.main()
