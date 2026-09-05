import tomllib
import unittest
from pathlib import Path


class UvCacheKeyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        project_root = Path(__file__).resolve().parents[1]
        cls.project_root = project_root
        project_file = project_root / "pyproject.toml"
        with project_file.open("rb") as stream:
            cls.project_configuration = tomllib.load(stream)
        cls.uv_configuration = cls.project_configuration["tool"]["uv"]
        cls.cache_keys = cls.uv_configuration["cache-keys"]
        with (project_root / "uv.lock").open("rb") as stream:
            cls.lockfile = tomllib.load(stream)

    def test_preserves_pure_package_build_inputs(self):
        file_keys = {entry["file"] for entry in self.cache_keys if "file" in entry}
        self.assertGreaterEqual(
            file_keys,
            {
                "pyproject.toml",
                "setup.py",
                "VERSION",
                "MANIFEST.in",
                "build-constraints.txt",
                "README.md",
            },
        )

    def test_excludes_native_build_inputs_and_environment(self):
        file_keys = {entry["file"] for entry in self.cache_keys if "file" in entry}
        environment_keys = {entry["env"] for entry in self.cache_keys if "env" in entry}
        self.assertFalse(any(key.startswith("src/") for key in file_keys))
        self.assertEqual(environment_keys, set())

    def test_pins_uv_and_pure_build_dependencies(self):
        self.assertEqual(self.uv_configuration["required-version"], "==0.12.5")
        constraints = set(self.uv_configuration["build-constraint-dependencies"])
        self.assertEqual(
            constraints,
            {
                "setuptools==84.0.0",
                "wheel==0.48.0",
            },
        )
        pip_constraints = {
            line.strip()
            for line in (self.project_root / "build-constraints.txt")
            .read_text()
            .splitlines()
            if line.strip()
        }
        self.assertEqual(pip_constraints, constraints)

    def test_numpy_remains_a_runtime_dependency_only(self):
        numpy_package = next(
            package
            for package in self.lockfile["package"]
            if package["name"] == "numpy"
        )
        self.assertGreaterEqual(numpy_package["version"], "2.3")
        self.assertIn(
            "numpy>=2.3", self.project_configuration["project"]["dependencies"]
        )

    def test_locks_explicit_pytorch_213_accelerator_variants(self):
        project = self.project_configuration["project"]
        self.assertIn("torch>=2.13,<2.14", project["dependencies"])
        extras = project["optional-dependencies"]
        for extra in ("torch-cpu", "torch-cu126", "torch-cu130"):
            self.assertEqual(extras[extra], ["torch>=2.13,<2.14"])

        expected_indexes = {
            "pytorch-cpu": "https://download.pytorch.org/whl/cpu",
            "pytorch-cu126": "https://download.pytorch.org/whl/cu126",
            "pytorch-cu130": "https://download.pytorch.org/whl/cu130",
        }
        indexes = {item["name"]: item for item in self.uv_configuration["index"]}
        self.assertEqual(
            {name: indexes[name]["url"] for name in expected_indexes},
            expected_indexes,
        )
        self.assertTrue(all(indexes[name]["explicit"] for name in expected_indexes))
        sources = self.uv_configuration["sources"]["torch"]
        self.assertEqual(
            {(source["extra"], source["index"]) for source in sources},
            {
                ("torch-cpu", "pytorch-cpu"),
                ("torch-cu126", "pytorch-cu126"),
                ("torch-cu130", "pytorch-cu130"),
            },
        )

        torch_packages = {
            (package["version"], package["source"]["registry"])
            for package in self.lockfile["package"]
            if package["name"] == "torch"
        }
        self.assertGreaterEqual(
            torch_packages,
            {
                ("2.13.0+cpu", expected_indexes["pytorch-cpu"]),
                ("2.13.0+cu126", expected_indexes["pytorch-cu126"]),
                ("2.13.0+cu130", expected_indexes["pytorch-cu130"]),
            },
        )

    def test_locks_strict_python_314_typing_and_pep561_data(self):
        self.assertIn(
            "mypy>=1.18.2", self.project_configuration["dependency-groups"]["dev"]
        )
        mypy = self.project_configuration["tool"]["mypy"]
        self.assertEqual(mypy["python_version"], "3.14")
        self.assertTrue(mypy["strict"])
        self.assertTrue(mypy["warn_unreachable"])
        self.assertTrue(mypy["warn_unused_configs"])
        self.assertEqual(
            self.project_configuration["tool"]["setuptools"]["package-data"]["gmes"],
            ["py.typed", "constant.pyi"],
        )
        locked_names = {package["name"] for package in self.lockfile["package"]}
        self.assertIn("mypy", locked_names)
        self.assertNotIn("mpi4py", locked_names)
        self.assertNotIn(
            "mpi", self.project_configuration["project"]["optional-dependencies"]
        )


if __name__ == "__main__":
    unittest.main()
