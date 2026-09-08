import tomllib
from pathlib import Path

import pytest


@pytest.fixture(scope="class", autouse=True)
def project_files(request):
    project_root = Path(__file__).resolve().parents[1]
    request.cls.project_root = project_root
    with (project_root / "pyproject.toml").open("rb") as stream:
        request.cls.project_configuration = tomllib.load(stream)
    request.cls.uv_configuration = request.cls.project_configuration["tool"]["uv"]
    request.cls.cache_keys = request.cls.uv_configuration["cache-keys"]
    with (project_root / "uv.lock").open("rb") as stream:
        request.cls.lockfile = tomllib.load(stream)


class TestUvCacheKey:

    def test_preserves_pure_package_build_inputs(self):
        file_keys = {entry["file"] for entry in self.cache_keys if "file" in entry}
        assert (file_keys) >= (
            {
                "pyproject.toml",
                "setup.py",
                "VERSION",
                "MANIFEST.in",
                "build-constraints.txt",
                "README.md",
            }
        )

    def test_excludes_native_build_inputs_and_environment(self):
        file_keys = {entry["file"] for entry in self.cache_keys if "file" in entry}
        environment_keys = {entry["env"] for entry in self.cache_keys if "env" in entry}
        assert not (any(key.startswith("src/") for key in file_keys))
        assert (environment_keys) == (set())

    def test_pins_uv_and_pure_build_dependencies(self):
        assert (self.uv_configuration["required-version"]) == ("==0.12.5")
        constraints = set(self.uv_configuration["build-constraint-dependencies"])
        assert (constraints) == (
            {
                "setuptools==84.0.0",
                "wheel==0.48.0",
            }
        )
        pip_constraints = {
            line.strip()
            for line in (self.project_root / "build-constraints.txt")
            .read_text()
            .splitlines()
            if line.strip()
        }
        assert (pip_constraints) == (constraints)

    def test_numpy_remains_a_runtime_dependency_only(self):
        numpy_package = next(
            package
            for package in self.lockfile["package"]
            if package["name"] == "numpy"
        )
        assert (numpy_package["version"]) >= ("2.3")
        assert ("numpy>=2.3") in (self.project_configuration["project"]["dependencies"])

    def test_locks_explicit_pytorch_213_accelerator_variants(self):
        project = self.project_configuration["project"]
        assert ("torch>=2.13,<2.14") in (project["dependencies"])
        extras = project["optional-dependencies"]
        for extra in ("torch-cpu", "torch-cu126", "torch-cu130"):
            assert (extras[extra]) == (["torch>=2.13,<2.14"])

        expected_indexes = {
            "pytorch-cpu": "https://download.pytorch.org/whl/cpu",
            "pytorch-cu126": "https://download.pytorch.org/whl/cu126",
            "pytorch-cu130": "https://download.pytorch.org/whl/cu130",
        }
        indexes = {item["name"]: item for item in self.uv_configuration["index"]}
        assert ({name: indexes[name]["url"] for name in expected_indexes}) == (
            expected_indexes
        )
        assert all(indexes[name]["explicit"] for name in expected_indexes)
        sources = self.uv_configuration["sources"]["torch"]
        assert ({(source["extra"], source["index"]) for source in sources}) == (
            {
                ("torch-cpu", "pytorch-cpu"),
                ("torch-cu126", "pytorch-cu126"),
                ("torch-cu130", "pytorch-cu130"),
            }
        )

        torch_packages = {
            (package["version"], package["source"]["registry"])
            for package in self.lockfile["package"]
            if package["name"] == "torch"
        }
        assert (torch_packages) >= (
            {
                ("2.13.0+cpu", expected_indexes["pytorch-cpu"]),
                ("2.13.0+cu126", expected_indexes["pytorch-cu126"]),
                ("2.13.0+cu130", expected_indexes["pytorch-cu130"]),
            }
        )

    def test_locks_strict_python_314_typing_and_pep561_data(self):
        assert ("mypy>=1.18.2") in (
            self.project_configuration["dependency-groups"]["dev"]
        )
        assert ("pytest>=8.4") in (
            self.project_configuration["dependency-groups"]["dev"]
        )
        assert self.project_configuration["tool"]["pytest"]["ini_options"][
            "norecursedirs"
        ] == ["tests/fixtures"]
        mypy = self.project_configuration["tool"]["mypy"]
        assert (mypy["python_version"]) == ("3.14")
        assert mypy["strict"]
        assert mypy["warn_unreachable"]
        assert mypy["warn_unused_configs"]
        assert (
            self.project_configuration["tool"]["setuptools"]["package-data"]["gmes"]
        ) == (["py.typed", "constant.pyi"])
        locked_names = {package["name"] for package in self.lockfile["package"]}
        assert ("mypy") in (locked_names)
        assert ("mpi4py") not in (locked_names)
        assert ("mpi") not in (
            self.project_configuration["project"]["optional-dependencies"]
        )
