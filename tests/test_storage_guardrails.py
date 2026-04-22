import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT
    / "runtime"
    / "EditSplat"
    / "sandboxes"
    / "20260322_editsplat_ttt3r_flowedit_sam3_4567"
    / "scripts"
    / "storage_guardrails.py"
)
if str(MODULE_PATH.parent) not in sys.path:
    sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("storage_guardrails", MODULE_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Unable to import storage guardrails module from {MODULE_PATH}")
storage_guardrails = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = storage_guardrails
SPEC.loader.exec_module(storage_guardrails)


class StorageGuardrailsTests(unittest.TestCase):
    def test_enforce_storage_guardrails_rejects_local_project_over_limit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            payload = tmp_path / "payload.bin"
            payload.write_bytes(b"x" * 32)
            with self.assertRaisesRegex(RuntimeError, "dev-root"):
                storage_guardrails.enforce_storage_guardrails(
                    [tmp_path],
                    local_limit_bytes=16,
                    vepfs_limit_bytes=1024,
                )

    def test_enforce_storage_guardrails_rejects_vepfs_project_over_limit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            vepfs_root = Path(tmp_dir) / "dev_vepfs"
            project_root = vepfs_root / "rc_wu" / "project"
            project_root.mkdir(parents=True)
            (project_root / "payload.bin").write_bytes(b"x" * 64)
            with self.assertRaisesRegex(RuntimeError, "vepfs"):
                storage_guardrails.enforce_storage_guardrails(
                    [project_root],
                    local_limit_bytes=1024,
                    vepfs_limit_bytes=32,
                    vepfs_prefixes=(vepfs_root,),
                )
