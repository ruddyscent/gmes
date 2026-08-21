#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle
import unittest

import numpy as np

from gmes.geometry import Cartesian
from gmes.material import Upml
from gmes.pw_material import (
    UpmlElectricParamReal,
    UpmlExReal,
    UpmlEyReal,
    UpmlEzReal,
    UpmlHxReal,
    UpmlHyReal,
    UpmlHzReal,
    UpmlMagneticParamReal,
)


class TestSequence(unittest.TestCase):
    def setUp(self):
        self.idx = (1, 1, 1)

        self.spc = Cartesian((0, 0, 0))
        self.spc.dt = 1

        self.upml = Upml()
        self.upml.init(self.spc, ((0, 0, 0), (1, 1, 1), 0.5))

    def test_initialized_pickle_round_trip(self):
        restored = pickle.loads(pickle.dumps(self.upml))

        for name in (
            "eps_inf",
            "mu_inf",
            "initialized",
            "d",
            "dt",
            "m",
            "kappa_max",
            "sigma_max_ratio",
        ):
            self.assertEqual(getattr(restored, name), getattr(self.upml, name))
        for name in ("center", "half_size", "dw", "sigma_max"):
            np.testing.assert_array_equal(
                getattr(restored, name), getattr(self.upml, name)
            )
            self.assertIsNot(getattr(restored, name), getattr(self.upml, name))

    def test_auxiliary_state_persists_across_updates(self):
        cases = (
            (UpmlExReal, UpmlElectricParamReal, (2, 2, 1), 1),
            (UpmlEyReal, UpmlElectricParamReal, (1, 2, 2), 1),
            (UpmlEzReal, UpmlElectricParamReal, (2, 1, 2), 1),
            (UpmlHxReal, UpmlMagneticParamReal, (1, 1, 0), -1),
            (UpmlHyReal, UpmlMagneticParamReal, (0, 1, 1), -1),
            (UpmlHzReal, UpmlMagneticParamReal, (1, 0, 1), -1),
        )

        for material_type, parameter_type, curl_idx, sign in cases:
            with self.subTest(material=material_type.__name__):
                parameter = parameter_type()
                if isinstance(parameter, UpmlElectricParamReal):
                    parameter.eps_inf = 1
                    parameter.d = 0
                else:
                    parameter.mu_inf = 1
                    parameter.b = 0
                parameter.c1 = 0.5
                parameter.c2 = 1
                parameter.c3 = 0
                parameter.c4 = 1
                parameter.c5 = 1
                parameter.c6 = 0

                material = material_type()
                material.attach(self.idx, parameter)
                field, in_field1, in_field2 = [np.zeros((3, 3, 3)) for _ in range(3)]
                in_field1[curl_idx] = 1

                values = []
                for step in range(3):
                    material.update_all(field, in_field1, in_field2, 1, 1, 1, step)
                    values.append(field[self.idx])

                np.testing.assert_allclose(values, sign * np.array((1, 1.5, 1.75)))

    def testExReal(self):
        sample = self.upml.get_pw_material_ex(self.idx, (0, 0, 0))

        for idx in np.ndindex(3, 3, 3):
            if idx == self.idx:
                self.assertEqual(sample.get_eps_inf(idx), self.upml.eps_inf)
            else:
                self.assertEqual(sample.get_eps_inf(idx), 0)

        ex, hz, hy = [np.zeros((3, 3, 3)) for _ in range(3)]
        dy = dz = dt = self.spc.dt
        n = 0
        sample.update_all(ex, hz, hy, dy, dz, dt, n)
        for idx in np.ndindex(3, 3, 3):
            self.assertEqual(ex[idx], 0)

    def testEyReal(self):
        sample = self.upml.get_pw_material_ey(self.idx, (0, 0, 0))

        for idx in np.ndindex(3, 3, 3):
            if idx == self.idx:
                self.assertEqual(sample.get_eps_inf(idx), self.upml.eps_inf)
            else:
                self.assertEqual(sample.get_eps_inf(idx), 0)

        ey, hx, hz = [np.zeros((3, 3, 3)) for _ in range(3)]
        dz = dx = dt = 1
        n = 0
        sample.update_all(ey, hx, hz, dz, dx, dt, n)
        for idx in np.ndindex(3, 3, 3):
            self.assertEqual(ey[idx], 0)

    def testEzReal(self):
        sample = self.upml.get_pw_material_ez(self.idx, (0, 0, 0))

        for idx in np.ndindex(3, 3, 3):
            if idx == self.idx:
                self.assertEqual(sample.get_eps_inf(idx), self.upml.eps_inf)
            else:
                self.assertEqual(sample.get_eps_inf(idx), 0)

        ez, hy, hx = [np.zeros((3, 3, 3)) for _ in range(3)]
        dx = dy = dt = 1
        n = 0
        sample.update_all(ez, hy, hx, dx, dy, dt, n)
        for idx in np.ndindex(3, 3, 3):
            self.assertEqual(ez[idx], 0)

    def testHxReal(self):
        sample = self.upml.get_pw_material_hx(self.idx, (0, 0, 0))

        for idx in np.ndindex(3, 3, 3):
            if idx == self.idx:
                self.assertEqual(sample.get_mu_inf(idx), self.upml.mu_inf)
            else:
                self.assertEqual(sample.get_mu_inf(idx), 0)

        hx, ez, ey = [np.zeros((3, 3, 3)) for _ in range(3)]
        dy = dz = dt = 1
        n = 0
        sample.update_all(hx, ez, ey, dy, dz, dt, n)
        for idx in np.ndindex(3, 3, 3):
            self.assertEqual(hx[idx], 0)

    def testHyReal(self):
        sample = self.upml.get_pw_material_hy(self.idx, (0, 0, 0))

        for idx in np.ndindex(3, 3, 3):
            if idx == self.idx:
                self.assertEqual(sample.get_mu_inf(idx), self.upml.mu_inf)
            else:
                self.assertEqual(sample.get_mu_inf(idx), 0)

        hy, ex, ez = [np.zeros((3, 3, 3)) for _ in range(3)]
        dz = dx = dt = 1
        n = 0
        sample.update_all(hy, ex, ez, dz, dx, dt, n)
        for idx in np.ndindex(3, 3, 3):
            self.assertEqual(hy[idx], 0)

    def testHzReal(self):
        sample = self.upml.get_pw_material_hz(self.idx, (0, 0, 0))

        for idx in np.ndindex(3, 3, 3):
            if idx == self.idx:
                self.assertEqual(sample.get_mu_inf(idx), self.upml.mu_inf)
            else:
                self.assertEqual(sample.get_mu_inf(idx), 0)

        hz, ey, ex = [np.zeros((3, 3, 3)) for _ in range(3)]
        dx = dy = dt = 1
        n = 0
        sample.update_all(hz, ey, ex, dx, dy, dt, n)
        for idx in np.ndindex(3, 3, 3):
            self.assertEqual(hz[idx], 0)

    def testExCmplx(self):
        sample = self.upml.get_pw_material_ex(self.idx, (0, 0, 0), cmplx=True)

        for idx in np.ndindex(3, 3, 3):
            if idx == self.idx:
                self.assertEqual(sample.get_eps_inf(idx), self.upml.eps_inf)
            else:
                self.assertEqual(sample.get_eps_inf(idx), 0)

        ex, hz, hy = [np.zeros((3, 3, 3), complex) for _ in range(3)]
        dy = dz = dt = 1
        n = 0
        sample.update_all(ex, hz, hy, dy, dz, dt, n)
        for idx in np.ndindex(3, 3, 3):
            self.assertEqual(ex[idx], 0j)

    def testEyCmplx(self):
        sample = self.upml.get_pw_material_ey(self.idx, (0, 0, 0), cmplx=True)

        for idx in np.ndindex(3, 3, 3):
            if idx == self.idx:
                self.assertEqual(sample.get_eps_inf(idx), self.upml.eps_inf)
            else:
                self.assertEqual(sample.get_eps_inf(idx), 0)

        ey, hx, hz = [np.zeros((3, 3, 3), complex) for _ in range(3)]
        dz = dx = dt = 1
        n = 0
        sample.update_all(ey, hx, hz, dz, dx, dt, n)
        for idx in np.ndindex(3, 3, 3):
            self.assertEqual(ey[idx], 0j)

    def testEzCmplx(self):
        sample = self.upml.get_pw_material_ez(self.idx, (0, 0, 0), cmplx=True)

        for idx in np.ndindex(3, 3, 3):
            if idx == self.idx:
                self.assertEqual(sample.get_eps_inf(idx), self.upml.eps_inf)
            else:
                self.assertEqual(sample.get_eps_inf(idx), 0)

        ez, hy, hx = [np.zeros((3, 3, 3), complex) for _ in range(3)]
        dx = dy = dt = 1
        n = 0
        sample.update_all(ez, hy, hx, dx, dy, dt, n)
        for idx in np.ndindex(3, 3, 3):
            self.assertEqual(ez[idx], 0j)

    def testHxCmplx(self):
        sample = self.upml.get_pw_material_hx(self.idx, (0, 0, 0), cmplx=True)

        for idx in np.ndindex(3, 3, 3):
            if idx == self.idx:
                self.assertEqual(sample.get_mu_inf(idx), self.upml.mu_inf)
            else:
                self.assertEqual(sample.get_mu_inf(idx), 0)

        hx, ez, ey = [np.zeros((3, 3, 3), complex) for _ in range(3)]
        dy = dz = dt = 1
        n = 0
        sample.update_all(hx, ez, ey, dy, dz, dt, n)
        for idx in np.ndindex(3, 3, 3):
            self.assertEqual(hx[idx], 0j)

    def testHyCmplx(self):
        sample = self.upml.get_pw_material_hy(self.idx, (0, 0, 0), cmplx=True)

        for idx in np.ndindex(3, 3, 3):
            if idx == self.idx:
                self.assertEqual(sample.get_mu_inf(idx), self.upml.mu_inf)
            else:
                self.assertEqual(sample.get_mu_inf(idx), 0)

        hy, ex, ez = [np.zeros((3, 3, 3), complex) for _ in range(3)]
        dz = dx = dt = 1
        n = 0
        sample.update_all(hy, ex, ez, dz, dx, dt, n)
        for idx in np.ndindex(3, 3, 3):
            self.assertEqual(hy[idx], 0j)

    def testHzCmplx(self):
        sample = self.upml.get_pw_material_hz(self.idx, (0, 0, 0), cmplx=True)

        for idx in np.ndindex(3, 3, 3):
            if idx == self.idx:
                self.assertEqual(sample.get_mu_inf(idx), self.upml.mu_inf)
            else:
                self.assertEqual(sample.get_mu_inf(idx), 0)

        hz, ey, ex = [np.zeros((3, 3, 3), complex) for _ in range(3)]
        dx = dy = dt = 1
        n = 0
        sample.update_all(hz, ey, ex, dx, dy, dt, n)
        for idx in np.ndindex(3, 3, 3):
            self.assertEqual(hz[idx], 0j)


if __name__ == "__main__":
    unittest.main(argv=("", "-v"))
