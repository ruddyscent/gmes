import unittest

import numpy as np

from gmes.material import Dm2


class Dm2Test(unittest.TestCase):
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
