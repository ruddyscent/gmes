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
            cls.uv_configuration = tomllib.load(stream)["tool"]["uv"]
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
            },
        )

    def test_tracks_all_native_source_types(self):
        file_keys = {entry["file"] for entry in self.cache_keys if "file" in entry}
        self.assertGreaterEqual(
            file_keys,
            {"src/*.cc", "src/*.hh", "src/*.i", "src/*.pyx"},
        )
        self.assertFalse(any(key.startswith("gmes/") for key in file_keys))

    def test_tracks_macos_deployment_target(self):
        environment_keys = {entry["env"] for entry in self.cache_keys if "env" in entry}
        self.assertIn("MACOSX_DEPLOYMENT_TARGET", environment_keys)

    def test_pins_uv_and_native_build_dependencies(self):
        self.assertEqual(self.uv_configuration["required-version"], "==0.12.5")
        constraints = set(self.uv_configuration["build-constraint-dependencies"])
        self.assertEqual(
            constraints,
            {
                "setuptools==84.0.0",
                "wheel==0.48.0",
                "Cython==3.2.9",
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


if __name__ == "__main__":
    unittest.main()
