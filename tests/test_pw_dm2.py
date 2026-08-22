import pickle
import unittest
from types import SimpleNamespace

import numpy as np

from gmes.material import Dm2
from gmes.pw_material import Dm2ElectricParamReal, _dm2_relative_error


class Dm2Test(unittest.TestCase):
    def test_initial_bloch_drive_does_not_gain_an_inverse_t1_factor(self):
        """Appendix Eq. (A3e) contains a dimensionally inconsistent /T1."""

        dt = 1e-7
        expected = 2 * 0.25 / 2 * 4 * -1
        for component in ("ex", "ey", "ez"):
            with self.subTest(component=component):
                slopes = []
                for t1 in (0.5, 2.0):
                    material = Dm2(
                        omega=(0,),
                        n_atom=(0,),
                        rho30=-1,
                        gamma=0.25,
                        t1=t1,
                        t2=3,
                        hbar=2,
                        rtol=1e-12,
                    )
                    pointwise = getattr(material, f"get_pw_material_{component}")(
                        (1, 1, 1), (0, 0, 0)
                    )
                    field = np.zeros((3, 3, 3))
                    field[1, 1, 1] = 4
                    magnetic1, magnetic2 = [np.zeros((3, 3, 3)) for _ in range(2)]

                    pointwise.update_all(field, magnetic1, magnetic2, 1, 1, dt, 0)
                    rho2 = pointwise.get_rho((1, 1, 1), 0, 1, dt)
                    slopes.append(rho2 / dt)

                for slope in slopes:
                    self.assertAlmostEqual(slope, expected, places=5)
                self.assertAlmostEqual(slopes[0], slopes[1], places=5)

    def test_lossless_bloch_sphere_invariant(self):
        material = Dm2(
            omega=(1.25,),
            n_atom=(0,),
            rho30=-1,
            gamma=0.2,
            t1=np.inf,
            t2=np.inf,
            hbar=1,
            rtol=1e-10,
        )
        pointwise = material.get_pw_material_ex((1, 1, 1), (0, 0, 0))
        field = np.zeros((3, 3, 3))
        field[1, 1, 1] = 0.75
        magnetic1, magnetic2 = [np.zeros((3, 3, 3)) for _ in range(2)]
        dt = 0.01

        for step in range(500):
            pointwise.update_all(field, magnetic1, magnetic2, 1, 1, dt, float(step))

        rho = np.array(
            [pointwise.get_rho((1, 1, 1), 0, index, 500 * dt) for index in range(3)]
        )
        self.assertAlmostEqual(float(rho @ rho), 1, places=8)

    def test_diagnostic_getters_validate_transition_bounds(self):
        pointwise = Dm2(omega=(1, 2), n_atom=(3, 4)).get_pw_material_ex(
            (1, 1, 1), (0, 0, 0)
        )

        self.assertEqual(pointwise.get_rho((1, 1, 1), 0, 0, 0), 0)
        self.assertEqual(pointwise.get_rho((1, 1, 1), 1, 0, 0), 0)
        for bin_index in (-1, 2):
            with self.subTest(bin=bin_index):
                with self.assertRaisesRegex(IndexError, "transition bin"):
                    pointwise.get_rho((1, 1, 1), bin_index, 0, 0)

        for name in ("get_u", "get_v", "get_w"):
            with self.subTest(getter=name):
                values = getattr(pointwise, name)((1, 1, 1), 0, 2)
                self.assertEqual(values.shape, (2,))
                with self.assertRaisesRegex(IndexError, "exceeds transition count"):
                    getattr(pointwise, name)((1, 1, 1), 0, 3)

    def test_parameter_setup_validates_transition_lengths(self):
        for omega, n_atom in (
            (np.array([], dtype=float), np.array([], dtype=float)),
            (np.array((1.0, 2.0)), np.array((3.0, 4.0))),
        ):
            with self.subTest(length=len(omega)):
                Dm2ElectricParamReal().set(omega, n_atom)

        for omega, n_atom in (
            (np.array((1.0, 2.0)), np.array((3.0,))),
            (np.array((1.0,)), np.array((2.0, 3.0))),
        ):
            with self.subTest(omega=len(omega), n_atom=len(n_atom)):
                with self.assertRaisesRegex(ValueError, "must have equal lengths"):
                    Dm2ElectricParamReal().set(omega, n_atom)

    def test_pickle_round_trip_before_and_after_init(self):
        for initialized in (False, True):
            with self.subTest(initialized=initialized):
                material = Dm2(
                    eps_inf=2,
                    mu_inf=3,
                    omega=(4, 5),
                    n_atom=(6, 7),
                    rho30=-0.5,
                    gamma=0.25,
                    t1=8,
                    t2=9,
                    hbar=10,
                    rtol=1e-6,
                )
                if initialized:
                    material.init(SimpleNamespace(dt=0.125))

                restored = pickle.loads(pickle.dumps(material))

                for name in (
                    "eps_inf",
                    "mu_inf",
                    "omega",
                    "n_atom",
                    "rho30",
                    "gamma",
                    "t1",
                    "t2",
                    "hbar",
                    "rtol",
                    "initialized",
                ):
                    self.assertEqual(getattr(restored, name), getattr(material, name))
                if initialized:
                    self.assertEqual(restored.dt, material.dt)

    def test_relative_error_includes_fields_and_atomic_state(self):
        zero = np.zeros((1, 3))
        one = np.array(((1.0, 0, 0),))
        two = np.array(((2.0, 0, 0),))

        self.assertEqual(_dm2_relative_error(0, zero, 0, zero), 0)
        self.assertTrue(np.isinf(_dm2_relative_error(1, zero, 0, zero)))
        self.assertEqual(_dm2_relative_error(2, zero, 1, zero), 1)
        self.assertAlmostEqual(_dm2_relative_error(1, two, 1, one), 1 / np.sqrt(2))
        self.assertEqual(_dm2_relative_error(2, two, 1, one), 1)

    def test_corrector_has_iteration_limit(self):
        material = Dm2(rtol=-1)
        pointwise = material.get_pw_material_ex((1, 1, 1), (0, 0, 0))
        field, in_field1, in_field2 = [np.zeros((3, 3, 3)) for _ in range(3)]

        with self.assertRaisesRegex(RuntimeError, "failed to converge"):
            pointwise.update_all(field, in_field1, in_field2, 1, 1, 0.1, 0)

    def test_corrector_can_start_from_zero_reference(self):
        material = Dm2()
        pointwise = material.get_pw_material_ex((1, 1, 1), (0, 0, 0))
        field, in_field1, in_field2 = [np.zeros((3, 3, 3)) for _ in range(3)]
        in_field2[2, 1, 2] = 1

        pointwise.update_all(field, in_field1, in_field2, 1, 1, 0.1, 0)

        self.assertTrue(np.isfinite(field).all())
        self.assertNotEqual(field[1, 1, 1], 0)

    def test_electric_components_construct_and_update(self):
        material = Dm2(eps_inf=2, omega=(1, 2), n_atom=(3, 4))

        for component in ("ex", "ey", "ez"):
            with self.subTest(component=component):
                pointwise = getattr(material, f"get_pw_material_{component}")(
                    (1, 1, 1), (0, 0, 0)
                )
                field, in_field1, in_field2 = [np.zeros((3, 3, 3)) for _ in range(3)]

                pointwise.update_all(field, in_field1, in_field2, 1, 1, 0.1, 0)

                self.assertEqual(pointwise.get_eps_inf((1, 1, 1)), 2)
                self.assertTrue(np.isfinite(field).all())


if __name__ == "__main__":
    unittest.main()
