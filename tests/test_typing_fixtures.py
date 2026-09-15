"""Verify that checker-only negative samples remain rejected."""

import subprocess
import sys
from pathlib import Path


class TestTypingFixture:
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
        assert result.returncode == 1, output
        errors = [line for line in output.splitlines() if ": error:" in line]
        fixture_errors = [
            line for line in errors if line.startswith("tests/typing/invalid_api.py:")
        ]
        assert len(fixture_errors) == 3, output
        for line, argument in zip(fixture_errors, ("device", "location", "target")):
            assert f'Argument "{argument}"' in line and "[arg-type]" in line, output
        assert errors == fixture_errors, output
