#!/usr/bin/env python3
"""Tests for the native OpenMP build configuration helpers."""

import os
import unittest

from gmes import pw_material


class OpenMPConfigurationTest(unittest.TestCase):
    """Check configuration reporting for serial and OpenMP builds."""

    def test_configuration_values_are_valid(self):
        self.assertIsInstance(pw_material.openmp_enabled(), bool)
        self.assertGreaterEqual(pw_material.openmp_max_threads(), 1)
        self.assertGreaterEqual(pw_material.openmp_cell_threshold(), 0)

    def test_required_openmp_build_is_enabled(self):
        setting = os.environ.get("GMES_ENABLE_OPENMP", "").lower()
        if setting in {"1", "true", "yes", "on"}:
            self.assertTrue(pw_material.openmp_enabled())


if __name__ == "__main__":
    unittest.main()
