import unittest
from unittest.mock import patch

from utils.macos_build import (
    extract_macos_targets,
    verify_extension_targets,
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


if __name__ == "__main__":
    unittest.main()
