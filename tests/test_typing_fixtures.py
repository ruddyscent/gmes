"""Verify that checker-only negative samples remain rejected."""

import subprocess
import sys
import unittest
from pathlib import Path


class TypingFixtureTest(unittest.TestCase):
    """Exercise misuse cases separately from the passing canonical fixture."""

    def test_invalid_public_api_sample_is_rejected(self):
        project_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "mypy",
                "--no-pretty",
                "tests/typing/invalid_api.py",
            ],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, output)
        self.assertEqual(output.count("[arg-type]"), 3, output)


if __name__ == "__main__":
    unittest.main()
