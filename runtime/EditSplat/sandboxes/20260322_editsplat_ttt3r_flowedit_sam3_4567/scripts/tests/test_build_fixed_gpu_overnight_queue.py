from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "build_fixed_gpu_overnight_queue.py"
)
SPEC = importlib.util.spec_from_file_location("ttt3r_overnight_queue", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load queue module from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class OvernightQueueTests(unittest.TestCase):
    def test_round_robin_assignments_preserve_order_per_slot(self) -> None:
        jobs = [
            MODULE.QueueJob(name="job_a"),
            MODULE.QueueJob(name="job_b"),
            MODULE.QueueJob(name="job_c"),
            MODULE.QueueJob(name="job_d"),
            MODULE.QueueJob(name="job_e"),
        ]
        slots = [MODULE.QueueSlot(gpu=1), MODULE.QueueSlot(gpu=2)]

        assignments = MODULE.assign_jobs_round_robin(jobs=jobs, slots=slots)

        self.assertEqual(
            [job.name for job in assignments["gpu1"]],
            ["job_a", "job_c", "job_e"],
        )
        self.assertEqual(
            [job.name for job in assignments["gpu2"]],
            ["job_b", "job_d"],
        )

    def test_render_slot_script_includes_gc_and_fixed_gpu_binding(self) -> None:
        slot = MODULE.QueueSlot(gpu=3)
        job = MODULE.QueueJob(
            name="job_gc",
            exp_kwargs={"case_name": "clown", "support_views": 4, "include_gt_view": False},
            extra_env={"EDITSPLAT_ELITE_CONF_CORRECTION": "1"},
        )

        script = MODULE.render_slot_script(
            slot=slot,
            jobs=[job],
            launcher_module_path=Path("/remote/launch.py"),
            queue_root=Path("/dev_vepfs/rc_wu/_codex_staging/ttt3r_overnight/wave_x"),
            wave_name="wave_x",
            queue_script_path=Path("/remote/build_fixed_gpu_overnight_queue.py"),
        )

        self.assertIn("CUDA_VISIBLE_DEVICES=3", script)
        self.assertIn("EDITSPLAT_ELITE_CONF_CORRECTION=1", script)
        self.assertIn("build_fixed_gpu_overnight_queue.py", script)
        self.assertIn("postprocess-job", script)
        self.assertIn("point_cloud", script)
        self.assertIn("chkpnt*.pth", script)
        self.assertIn("debug_intermediates", script)
        self.assertIn("duplicate train/debug PNGs", script)
        self.assertIn("gaussian_mask_stats.json", script)
        self.assertIn("job_gc", script)

    def test_build_detached_launch_command_uses_nohup_and_background(self) -> None:
        command = MODULE.build_detached_launch_command(
            script_path=Path("/tmp/slot_gpu1.sh"),
            log_path=Path("/tmp/slot_gpu1.log"),
        )

        self.assertIn("nohup bash", command)
        self.assertIn("slot_gpu1.sh", command)
        self.assertIn("slot_gpu1.log", command)
        self.assertIn("< /dev/null &", command)


if __name__ == "__main__":
    unittest.main()
