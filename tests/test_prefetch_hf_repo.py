import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPO_ROOT
    / "runtime"
    / "EditSplat"
    / "sandboxes"
    / "20260322_editsplat_ttt3r_flowedit_sam3_4567"
    / "scripts"
    / "prefetch_hf_repo.py"
)
SPEC = importlib.util.spec_from_file_location("prefetch_hf_repo", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Unable to import script from {SCRIPT_PATH}")
prefetch_hf_repo = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prefetch_hf_repo
SPEC.loader.exec_module(prefetch_hf_repo)


class PrefetchHfRepoTests(unittest.TestCase):
    def test_repo_cache_dir_uses_hf_model_layout(self):
        cache_dir = Path("/tmp/hf")
        self.assertEqual(
            prefetch_hf_repo.repo_cache_dir(cache_dir, "cocktailpeanut/xulf-s"),
            cache_dir / "models--cocktailpeanut--xulf-s",
        )

    def test_incomplete_blobs_detects_partial_downloads(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            blobs = cache_dir / "models--cocktailpeanut--xulf-s" / "blobs"
            blobs.mkdir(parents=True)
            (blobs / "a.incomplete").write_text("", encoding="utf-8")
            (blobs / "b").write_text("", encoding="utf-8")
            found = prefetch_hf_repo.incomplete_blobs(cache_dir, "cocktailpeanut/xulf-s")
            self.assertEqual([path.name for path in found], ["a.incomplete"])

    def test_prefetch_repo_uses_serial_resume_download_and_waits_for_clean_cache(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            snapshot_path = cache_dir / "models--cocktailpeanut--xulf-s" / "snapshots" / "abc"
            snapshot_path.mkdir(parents=True)
            blobs = cache_dir / "models--cocktailpeanut--xulf-s" / "blobs"
            blobs.mkdir(parents=True)
            incomplete = blobs / "x.incomplete"
            incomplete.write_text("", encoding="utf-8")

            def fake_snapshot_download(**kwargs):
                incomplete.unlink(missing_ok=True)
                return str(snapshot_path)

            mocked = mock.Mock(side_effect=fake_snapshot_download)
            result = prefetch_hf_repo.prefetch_repo(
                repo_id="cocktailpeanut/xulf-s",
                cache_dir=cache_dir,
                retries=2,
                retry_sleep=0.0,
                max_workers=1,
                token=None,
                downloader=mocked,
            )

            self.assertEqual(result, snapshot_path)
            mocked.assert_called_once()
            self.assertEqual(mocked.call_args.kwargs["repo_id"], "cocktailpeanut/xulf-s")
            self.assertEqual(mocked.call_args.kwargs["cache_dir"], str(cache_dir))
            self.assertEqual(mocked.call_args.kwargs["max_workers"], 1)
            self.assertTrue(mocked.call_args.kwargs["resume_download"])


if __name__ == "__main__":
    unittest.main()
