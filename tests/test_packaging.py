import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path, PurePosixPath

from utils.release import verify_sdist

EXTERNAL_CPU_BASELINES = {
    PurePosixPath("benchmarks/evidence/issue-123/torch-cpu-baseline-one.json"),
    PurePosixPath("benchmarks/evidence/issue-123/torch-cpu-baseline-physical.json"),
}


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
            verify_sdist(archives[0], "0.10.0")
            with tarfile.open(archives[0], "r:gz") as archive:
                packaged_members = {
                    PurePosixPath(*PurePosixPath(member.name).parts[1:]): member
                    for member in archive.getmembers()
                }
                packaged_paths = set(packaged_members)

        self.assertIn(PurePosixPath("examples/VERIFICATION.md"), packaged_paths)
        self.assertIn(PurePosixPath("gmes/py.typed"), packaged_paths)
        self.assertIn(PurePosixPath("gmes/constant.pyi"), packaged_paths)
        self.assertIn(PurePosixPath("gmes/pw_material.pyi"), packaged_paths)
        for path in EXTERNAL_CPU_BASELINES:
            with self.subTest(path=path):
                self.assertNotIn(path, packaged_paths)


if __name__ == "__main__":
    unittest.main()
