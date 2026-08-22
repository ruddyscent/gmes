"""Regression tests for every native OpenMP build path."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from utils.openmp_build import openmp_options


class OpenMPBuildModeTest(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.prefix = Path(self.directory.name)
        (self.prefix / "include").mkdir()
        (self.prefix / "include" / "omp.h").touch()
        (self.prefix / "lib").mkdir()
        (self.prefix / "lib" / "libomp.dylib").touch()

    def environment(self, mode):
        return {
            "GMES_ENABLE_OPENMP": mode,
            "GMES_OPENMP_PREFIX": str(self.prefix),
            "MACOSX_DEPLOYMENT_TARGET": "11.0",
        }

    @patch("utils.macos_build.subprocess.run")
    def test_auto_with_compatible_runtime_enables_openmp(self, run):
        run.return_value = Mock(stdout="cmd LC_BUILD_VERSION\n  minos 11.0\n")

        setting, options, diagnostic = openmp_options(
            self.environment("auto"), "darwin"
        )

        self.assertEqual(setting, "auto")
        self.assertIn("-fopenmp", options[0])
        self.assertIn("-lomp", options[1])
        self.assertIsNone(diagnostic)

    @patch("utils.macos_build.subprocess.run")
    def test_auto_with_incompatible_runtime_uses_serial_fallback(self, run):
        run.return_value = Mock(stdout="cmd LC_BUILD_VERSION\n  minos 26.0\n")

        setting, options, diagnostic = openmp_options(
            self.environment("auto"), "darwin"
        )

        self.assertEqual(setting, "auto")
        self.assertIsNone(options)
        self.assertIn("newer than requested 11.0", diagnostic)

    @patch("utils.macos_build.subprocess.run")
    def test_required_with_incompatible_runtime_fails(self, run):
        run.return_value = Mock(stdout="cmd LC_BUILD_VERSION\n  minos 26.0\n")

        with self.assertRaisesRegex(RuntimeError, "OpenMP was requested"):
            openmp_options(self.environment("1"), "darwin")

    @patch("utils.macos_build.subprocess.run")
    def test_required_with_compatible_runtime_enables_openmp(self, run):
        run.return_value = Mock(stdout="cmd LC_BUILD_VERSION\n  minos 11.0\n")

        setting, options, diagnostic = openmp_options(self.environment("1"), "darwin")

        self.assertEqual(setting, "required")
        self.assertIn("-fopenmp", options[0])
        self.assertIn("-lomp", options[1])
        self.assertIsNone(diagnostic)

    @patch("utils.macos_build.subprocess.run")
    def test_disabled_mode_uses_serial_without_inspecting_runtime(self, run):
        setting, options, diagnostic = openmp_options(self.environment("0"), "darwin")

        self.assertEqual(setting, "disabled")
        self.assertIsNone(options)
        self.assertIsNone(diagnostic)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
