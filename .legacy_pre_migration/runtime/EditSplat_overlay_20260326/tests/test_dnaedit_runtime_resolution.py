import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowedit_multimodel.src.core_backend import _resolve_dnaedit_runtime_root


class DNAEditRuntimeResolutionTests(unittest.TestCase):
    def test_env_override_wins_when_utils_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            utils_path = root / "scripts" / "DNAEdit_utils.py"
            utils_path.parent.mkdir(parents=True, exist_ok=True)
            utils_path.write_text("# stub\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"EDITSPLAT_DNAEDIT_RUNTIME_ROOT": str(root)}, clear=False):
                resolved = _resolve_dnaedit_runtime_root(Path("/nonexistent/project"))
            self.assertEqual(resolved, root)


if __name__ == "__main__":
    unittest.main()

