import unittest

import numpy as np

from gmes.pw_material import (
    DcpAdeElectricParamCmplx,
    DcpAdeElectricParamReal,
    DcpPlrcElectricParamCmplx,
    DcpPlrcElectricParamReal,
    DrudeElectricParamCmplx,
    DrudeElectricParamReal,
    LorentzElectricParamCmplx,
    LorentzElectricParamReal,
)


class FixedCoefficientShapeTest(unittest.TestCase):
    def assert_rejects_widths(self, param_class, valid_args, arg_index, widths):
        for width in widths:
            with self.subTest(
                parameter=param_class.__name__, argument=arg_index, width=width
            ):
                args = list(valid_args)
                dtype = args[arg_index].dtype
                args[arg_index] = np.zeros((1, width), dtype=dtype)
                with self.assertRaises(ValueError):
                    param_class().set(*args)

    def assert_rejects_lengths(self, param_class, valid_args, arg_index, lengths):
        for length in lengths:
            with self.subTest(
                parameter=param_class.__name__, argument=arg_index, length=length
            ):
                args = list(valid_args)
                args[arg_index] = np.zeros(length, dtype=args[arg_index].dtype)
                with self.assertRaises(ValueError):
                    param_class().set(*args)

    def test_drude_and_lorentz_require_exact_coefficient_shapes(self):
        valid_args = (np.zeros((1, 3)), np.zeros(3))

        for param_class in (
            DrudeElectricParamReal,
            DrudeElectricParamCmplx,
            LorentzElectricParamReal,
            LorentzElectricParamCmplx,
        ):
            param_class().set(*valid_args)
            self.assert_rejects_widths(param_class, valid_args, 0, (2, 4))
            self.assert_rejects_lengths(param_class, valid_args, 1, (2, 4))

    def test_dcp_ade_requires_exact_coefficient_shapes(self):
        valid_args = (np.zeros((1, 3)), np.zeros((1, 5)), np.zeros(4))

        for param_class in (DcpAdeElectricParamReal, DcpAdeElectricParamCmplx):
            param_class().set(*valid_args)
            self.assert_rejects_widths(param_class, valid_args, 0, (2, 4))
            self.assert_rejects_widths(param_class, valid_args, 1, (4, 6))
            self.assert_rejects_lengths(param_class, valid_args, 2, (3, 5))

    def test_dcp_plrc_requires_exact_coefficient_shapes(self):
        valid_args = (
            np.zeros((1, 3)),
            np.zeros((1, 3), dtype=np.complex128),
            np.zeros(3),
        )

        for param_class in (DcpPlrcElectricParamReal, DcpPlrcElectricParamCmplx):
            param_class().set(*valid_args)
            self.assert_rejects_widths(param_class, valid_args, 0, (2, 4))
            self.assert_rejects_widths(param_class, valid_args, 1, (2, 4))
            self.assert_rejects_lengths(param_class, valid_args, 2, (2, 4))


if __name__ == "__main__":
    unittest.main()
