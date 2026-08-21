import tomllib
import unittest
from pathlib import Path


class UvCacheKeyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        project_file = Path(__file__).resolve().parents[1] / "pyproject.toml"
        with project_file.open("rb") as stream:
            cls.cache_keys = tomllib.load(stream)["tool"]["uv"]["cache-keys"]

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


if __name__ == "__main__":
    unittest.main()
