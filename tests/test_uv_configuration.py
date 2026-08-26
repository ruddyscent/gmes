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

    def test_preserves_default_and_dynamic_build_inputs(self):
        file_keys = {entry["file"] for entry in self.cache_keys if "file" in entry}
        self.assertGreaterEqual(
            file_keys,
            {
                "pyproject.toml",
                "setup.py",
                "VERSION",
                "MANIFEST.in",
                "README.md",
                "utils/macos_build.py",
                "utils/openmp_build.py",
            },
        )

    def test_tracks_all_native_source_types(self):
        file_keys = {entry["file"] for entry in self.cache_keys if "file" in entry}
        self.assertGreaterEqual(
            file_keys,
            {"src/*.cc", "src/*.hh", "src/*.i"},
        )
        self.assertNotIn("src/*.pyx", file_keys)
        self.assertFalse(any(key.startswith("gmes/") for key in file_keys))

    def test_tracks_macos_deployment_target(self):
        environment_keys = {entry["env"] for entry in self.cache_keys if "env" in entry}
        self.assertGreaterEqual(
            environment_keys,
            {
                "GMES_ENABLE_OPENMP",
                "GMES_OPENMP_PREFIX",
                "MACOSX_DEPLOYMENT_TARGET",
            },
        )

    def test_pins_uv_and_native_build_dependencies(self):
        self.assertEqual(self.uv_configuration["required-version"], "==0.12.5")
        constraints = set(self.uv_configuration["build-constraint-dependencies"])
        self.assertEqual(
            constraints,
            {
                "setuptools==84.0.0",
                "wheel==0.48.0",
                "numpy==2.5.2",
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

    def test_build_and_runtime_numpy_versions_match(self):
        numpy_package = next(
            package
            for package in self.lockfile["package"]
            if package["name"] == "numpy"
        )
        constraints = self.uv_configuration["build-constraint-dependencies"]
        self.assertIn(f"numpy=={numpy_package['version']}", constraints)

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


if __name__ == "__main__":
    unittest.main()
