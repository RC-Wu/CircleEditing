import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EDITSPLAT_ROOT = REPO_ROOT / "runtime" / "EditSplat"
if str(EDITSPLAT_ROOT) not in sys.path:
    sys.path.insert(0, str(EDITSPLAT_ROOT))

from utils.runtime_bootstrap import should_bootstrap_external_backend_only  # noqa: E402


class RuntimeBootstrapTests(unittest.TestCase):
    def test_requires_explicit_env_toggle(self):
        self.assertFalse(
            should_bootstrap_external_backend_only(
                flow_method="flowedit",
                flow_model_key="sd35-medium-turbo-open",
                env_enabled=False,
            )
        )

    def test_allows_flowedit_sd35_when_enabled(self):
        self.assertTrue(
            should_bootstrap_external_backend_only(
                flow_method="flowedit",
                flow_model_key="sd35-medium-turbo-open",
                env_enabled=True,
            )
        )

    def test_allows_dnaedit_sd35_when_enabled(self):
        self.assertTrue(
            should_bootstrap_external_backend_only(
                flow_method="dnaedit",
                flow_model_key="sd35-large",
                env_enabled=True,
            )
        )

    def test_rejects_native_flux_path(self):
        self.assertFalse(
            should_bootstrap_external_backend_only(
                flow_method="flowedit",
                flow_model_key="flux1-dev",
                env_enabled=True,
            )
        )

    def test_rejects_non_external_methods(self):
        self.assertFalse(
            should_bootstrap_external_backend_only(
                flow_method="native",
                flow_model_key="sd35-medium-turbo-open",
                env_enabled=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
