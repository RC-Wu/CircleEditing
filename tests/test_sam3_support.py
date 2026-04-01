import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EDITSPLAT_ROOT = REPO_ROOT / "runtime" / "EditSplat"
if str(EDITSPLAT_ROOT) not in sys.path:
    sys.path.insert(0, str(EDITSPLAT_ROOT))

from utils.sam3_support import iter_mask_prompts, parse_hf_token_line  # noqa: E402


class Sam3SupportTests(unittest.TestCase):
    def test_parse_hf_token_line_handles_key_value(self):
        self.assertEqual(parse_hf_token_line("HF_TOKEN='hf_abc123XYZ'"), "hf_abc123XYZ")

    def test_parse_hf_token_line_handles_plain_token(self):
        self.assertEqual(parse_hf_token_line('"hf_plainToken987"'), "hf_plainToken987")

    def test_iter_mask_prompts_keeps_primary_then_fallbacks(self):
        prompts = list(iter_mask_prompts("face"))
        self.assertEqual(prompts[0], "face")
        self.assertIn("person", prompts)
        self.assertIn("portrait", prompts)
        self.assertEqual(len(prompts), len(set(prompts)))


if __name__ == "__main__":
    unittest.main()
