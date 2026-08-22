import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from utils.macos_build import (
    extract_macos_targets,
    select_macos_openmp_prefix,
    verify_extension_targets,
    verify_macos_dependency_target,
    verify_wheel_platform_tag,
)


class MacosBuildTest(unittest.TestCase):
    def test_wheel_platform_tag_matches_target(self):
        verify_wheel_platform_tag("macosx_11_0_arm64", "11.0")
        verify_wheel_platform_tag("macosx_11_0_x86_64.macosx_11_0_arm64", "11.0")

    def test_wheel_platform_tag_rejects_host_version(self):
        with self.assertRaisesRegex(RuntimeError, "does not target 11.0"):
            verify_wheel_platform_tag("macosx_26_0_arm64", "11.0")

    def test_extracts_current_and_legacy_load_commands(self):
        output = """
          cmd LC_BUILD_VERSION
        minos 11.0
          cmd LC_VERSION_MIN_MACOSX
      version 11.0
        """
        self.assertEqual(extract_macos_targets(output), ["11.0", "11.0"])

    @patch("utils.macos_build.subprocess.run")
    def test_extension_target_uses_otool_metadata(self, run):
        run.return_value.stdout = "cmd LC_BUILD_VERSION\n  minos 11.0\n"

        verify_extension_targets(["build/gmes/material.so"], "11.0")

        run.assert_called_once_with(
            ["otool", "-l", "build/gmes/material.so"],
            check=True,
            capture_output=True,
            text=True,
        )

    @patch("utils.macos_build.subprocess.run")
    def test_extension_target_rejects_newer_minos(self, run):
        run.return_value.stdout = "cmd LC_BUILD_VERSION\n  minos 26.0\n"

        with self.assertRaisesRegex(RuntimeError, "expected 11.0"):
            verify_extension_targets(["build/gmes/material.so"], "11.0")

    @patch("utils.macos_build.subprocess.run")
    def test_dependency_target_accepts_older_minos(self, run):
        run.return_value.stdout = "cmd LC_BUILD_VERSION\n  minos 10.15\n"

        verify_macos_dependency_target("/opt/lib/libomp.dylib", "11.0")

    @patch("utils.macos_build.subprocess.run")
    def test_dependency_target_rejects_newer_minos(self, run):
        run.return_value.stdout = "cmd LC_BUILD_VERSION\n  minos 26.0\n"

        with self.assertRaisesRegex(RuntimeError, "newer than requested 11.0"):
            verify_macos_dependency_target("/opt/lib/libomp.dylib", "11.0")

    @patch("utils.macos_build.subprocess.run")
    def test_openmp_prefix_selection_skips_incompatible_runtime(self, run):
        run.side_effect = [
            Mock(stdout="cmd LC_BUILD_VERSION\n minos 26.0\n"),
            Mock(stdout="cmd LC_BUILD_VERSION\n minos 11.0\n"),
        ]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prefixes = [root / "newer", root / "compatible"]
            for prefix in prefixes:
                (prefix / "include").mkdir(parents=True)
                (prefix / "include" / "omp.h").touch()
                (prefix / "lib").mkdir()
                (prefix / "lib" / "libomp.dylib").touch()

            selected, incompatibilities = select_macos_openmp_prefix(prefixes, "11.0")

        self.assertEqual(selected, prefixes[1])
        self.assertEqual(len(incompatibilities), 1)
        self.assertIn("newer than requested 11.0", incompatibilities[0])


if __name__ == "__main__":
    unittest.main()
