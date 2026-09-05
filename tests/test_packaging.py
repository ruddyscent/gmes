import hashlib
import io
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
import venv
from pathlib import Path, PurePosixPath

from tests.test_historical_fixture_integrity import COMPLETE_CLOSURE
from utils.release import REQUIRED_SDIST_PATHS, verify_distribution_set, verify_sdist

EXTERNAL_CPU_BASELINES = {
    PurePosixPath("benchmarks/evidence/issue-123/torch-cpu-baseline-one.json"),
    PurePosixPath("benchmarks/evidence/issue-123/torch-cpu-baseline-physical.json"),
}
FIXTURE_ROOT = Path("tests/fixtures/issue124")


class SourceDistributionTest(unittest.TestCase):
    def test_fixture_includes_required_pure_package_members(self):
        with tempfile.TemporaryDirectory() as output_directory:
            archive_path = Path(output_directory) / "gmes-0.10.0.tar.gz"
            root = "gmes-0.10.0"
            paths = REQUIRED_SDIST_PATHS | {
                "examples/VERIFICATION.md",
                "examples/example.py",
                "tests/test_example.py",
            }
            with tarfile.open(archive_path, "w:gz") as archive:
                for name in sorted(paths):
                    content = b""
                    member = tarfile.TarInfo(f"{root}/{name}")
                    member.size = len(content)
                    archive.addfile(member, io.BytesIO(content))
                metadata = b"Name: gmes\nVersion: 0.10.0\n# GMES\n"
                member = tarfile.TarInfo(f"{root}/PKG-INFO")
                member.size = len(metadata)
                archive.addfile(member, io.BytesIO(metadata))

            verify_sdist(archive_path, "0.10.0")
            with tarfile.open(archive_path, "r:gz") as archive:
                packaged_members = {
                    PurePosixPath(*PurePosixPath(member.name).parts[1:]): member
                    for member in archive.getmembers()
                }
                packaged_paths = set(packaged_members)

        self.assertIn(PurePosixPath("examples/VERIFICATION.md"), packaged_paths)
        self.assertIn(PurePosixPath("gmes/py.typed"), packaged_paths)
        self.assertIn(PurePosixPath("gmes/constant.pyi"), packaged_paths)
        self.assertIn(PurePosixPath("gmes/constant.py"), packaged_paths)
        self.assertNotIn(PurePosixPath("gmes/pw_material.pyi"), packaged_paths)
        for path in EXTERNAL_CPU_BASELINES:
            with self.subTest(path=path):
                self.assertNotIn(path, packaged_paths)

    def test_pep517_archives_install_and_run_from_clean_project_path(self):
        """Final candidates must prove the built, installed artifact is runnable."""
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as output_directory:
            output_directory = Path(output_directory)
            dist_directory = output_directory / "dist"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "build",
                    "--wheel",
                    "--sdist",
                    "--outdir",
                    str(dist_directory),
                ],
                cwd=project_root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            verify_distribution_set(dist_directory, "0.10.0")
            wheel = next(dist_directory.glob("*.whl"))
            sdist = next(dist_directory.glob("*.tar.gz"))

            expected_fixture_digests = {
                relative: hashlib.sha256(
                    (FIXTURE_ROOT / relative).read_bytes()
                ).hexdigest()
                for relative in COMPLETE_CLOSURE
            }
            with tarfile.open(sdist, "r:gz") as archive:
                fixture_prefix = "gmes-0.10.0/tests/fixtures/issue124/"
                archived_fixture_digests = {}
                for member in archive.getmembers():
                    if not member.isfile() or not member.name.startswith(
                        fixture_prefix
                    ):
                        continue
                    contents = archive.extractfile(member)
                    self.assertIsNotNone(contents, member.name)
                    archived_fixture_digests[
                        member.name.removeprefix(fixture_prefix)
                    ] = hashlib.sha256(contents.read()).hexdigest()
            self.assertEqual(archived_fixture_digests, expected_fixture_digests)

            for archive in (wheel, sdist):
                environment = output_directory / archive.stem
                venv.EnvBuilder(with_pip=True).create(environment)
                python = environment / (
                    "Scripts/python.exe" if os.name == "nt" else "bin/python"
                )
                environment_variables = {
                    key: value
                    for key, value in os.environ.items()
                    if key != "PYTHONPATH"
                } | {"UV_PROJECT_ENVIRONMENT": str(environment)}
                sync = subprocess.run(
                    [
                        "uv",
                        "sync",
                        "--locked",
                        "--no-install-project",
                        "--no-dev",
                        "--extra",
                        "torch-cpu",
                        "--extra",
                        "hdf5",
                    ],
                    cwd=project_root,
                    check=False,
                    capture_output=True,
                    text=True,
                    env=environment_variables,
                )
                self.assertEqual(sync.returncode, 0, sync.stdout + sync.stderr)
                bootstrap = subprocess.run(
                    [str(python), "-m", "ensurepip"],
                    cwd=output_directory,
                    check=False,
                    capture_output=True,
                    text=True,
                    env=environment_variables,
                )
                self.assertEqual(
                    bootstrap.returncode, 0, bootstrap.stdout + bootstrap.stderr
                )
                build_tools = subprocess.run(
                    [
                        str(python),
                        "-m",
                        "pip",
                        "install",
                        "--no-deps",
                        "--constraint",
                        str(project_root / "build-constraints.txt"),
                        "setuptools==84.0.0",
                        "wheel==0.48.0",
                    ],
                    cwd=output_directory,
                    check=False,
                    capture_output=True,
                    text=True,
                    env=environment_variables,
                )
                self.assertEqual(
                    build_tools.returncode, 0, build_tools.stdout + build_tools.stderr
                )
                archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
                installer = [
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--no-index",
                    "--force-reinstall",
                ]
                if archive.suffix == ".gz":
                    installer.append("--no-build-isolation")
                install = subprocess.run(
                    installer + [f"gmes @ {archive.as_uri()}#sha256={archive_digest}"],
                    cwd=output_directory,
                    check=False,
                    capture_output=True,
                    text=True,
                    env=environment_variables,
                )
                self.assertEqual(install.returncode, 0, install.stdout + install.stderr)
                smoke = subprocess.run(
                    [
                        str(python),
                        "-I",
                        "-c",
                        "from pathlib import Path\n"
                        "import gmes\n"
                        "from gmes.geometry import Cartesian, DefaultMedium\n"
                        "from gmes.material import Dielectric\n"
                        "from gmes.torch_fdtd import TorchRuntimeConfig, TorchSimulation\n"
                        "assert Path(gmes.__file__).resolve().is_relative_to(Path(r'"
                        + str(environment)
                        + "').resolve())\n"
                        "simulation = TorchSimulation(\n"
                        "    space=Cartesian((2, 2, 2), 1),\n"
                        "    geometry=(DefaultMedium(Dielectric()),),\n"
                        "    runtime=TorchRuntimeConfig(device='cpu', cpu_threads=1),\n"
                        ")\n"
                        "simulation.step()\n",
                    ],
                    cwd=output_directory,
                    check=False,
                    capture_output=True,
                    text=True,
                    env=environment_variables,
                )
                self.assertEqual(smoke.returncode, 0, smoke.stdout + smoke.stderr)


if __name__ == "__main__":
    unittest.main()
