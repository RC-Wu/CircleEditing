import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "runtime" / "tools" / "run_remote_codex_tmux.sh"


class RemoteCodexTmuxScriptTests(unittest.TestCase):
    def test_script_exists_and_uses_shared_codex_home(self):
        self.assertTrue(SCRIPT_PATH.exists(), f"missing script: {SCRIPT_PATH}")
        text = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn("export HOME=/dev_vepfs/rc_wu", text)
        self.assertIn("/dev_vepfs/rc_wu/.local/bin/codex", text)
        self.assertIn("/dev_vepfs/rc_wu/.codex/env.sh", text)
        self.assertIn("/dev_vepfs/rc_wu/edit/CircleEditing", text)
        self.assertIn("tmux new-session -d", text)
        self.assertIn("-o", text)


if __name__ == "__main__":
    unittest.main()
