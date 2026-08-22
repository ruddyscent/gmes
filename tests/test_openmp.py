"""Regression tests for native OpenMP configuration and execution."""

import os
import subprocess
import sys
import unittest

import numpy as np

from gmes import pw_material


class OpenMPConfigurationTest(unittest.TestCase):
    """Verify build reporting, threshold handling, and update execution."""

    def test_configuration_values_are_valid(self):
        self.assertIsInstance(pw_material.openmp_enabled(), bool)
        self.assertGreaterEqual(pw_material.openmp_max_threads(), 1)
        self.assertGreaterEqual(pw_material.openmp_cell_threshold(), 0)

    def test_build_setting_matches_runtime_introspection(self):
        setting = os.environ.get("GMES_ENABLE_OPENMP", "").strip().lower()
        if setting in {"1", "true", "yes", "on"}:
            self.assertTrue(pw_material.openmp_enabled())
        elif setting in {"0", "false", "no", "off"}:
            self.assertFalse(pw_material.openmp_enabled())
            self.assertEqual(pw_material.openmp_max_threads(), 1)

    def test_runtime_threshold_matches_valid_environment_value(self):
        value = os.environ.get("GMES_OPENMP_THRESHOLD")
        if value is not None and value.isdecimal():
            self.assertEqual(pw_material.openmp_cell_threshold(), int(value))

    def test_invalid_runtime_threshold_uses_default(self):
        environment = os.environ.copy()
        environment["GMES_OPENMP_THRESHOLD"] = "invalid"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from gmes import pw_material; "
                "print(pw_material.openmp_cell_threshold())",
            ],
            check=False,
            capture_output=True,
            env=environment,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "8192")

    def test_native_update_produces_expected_values(self):
        shape = (16, 16, 4)
        material = pw_material.ConstExReal()
        parameter = pw_material.ConstElectricParamReal()
        parameter.eps_inf = 2.5
        parameter.value = 0.75
        for index in np.ndindex(shape):
            material.attach(index, parameter)

        target, input_one, input_two = [np.zeros(shape) for _ in range(3)]
        material.update_all(target, input_one, input_two, 1, 1, 1, 0)

        np.testing.assert_array_equal(target, np.full(shape, parameter.value))


if __name__ == "__main__":
    unittest.main()
