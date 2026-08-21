import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path, PurePosixPath


class SourceDistributionTest(unittest.TestCase):
    def test_includes_example_verification_document(self):
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as output_directory:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "build",
                    "--sdist",
                    "--no-isolation",
                    "--outdir",
                    output_directory,
                ],
                cwd=project_root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            archives = list(Path(output_directory).glob("*.tar.gz"))
            self.assertEqual(len(archives), 1)
            with tarfile.open(archives[0], "r:gz") as archive:
                packaged_paths = {
                    PurePosixPath(*PurePosixPath(name).parts[1:])
                    for name in archive.getnames()
                }

        self.assertIn(PurePosixPath("examples/VERIFICATION.md"), packaged_paths)


if __name__ == "__main__":
    unittest.main()
