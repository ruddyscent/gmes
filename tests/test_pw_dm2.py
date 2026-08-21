import unittest

import numpy as np

from gmes.material import Dm2
from gmes.pw_material import _dm2_relative_error


class Dm2Test(unittest.TestCase):
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
