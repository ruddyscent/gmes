import unittest

import numpy as np

from examples.slab_waveguide import make_simulation


class SlabWaveguideExampleTest(unittest.TestCase):
    def test_finite_slab_covers_the_intended_waveguide(self):
        simulation = make_simulation(verbose=False)
        slab = simulation.geom_list[1]

        self.assertTrue(np.isfinite(slab.size).all())
        for point in ((-6, 0, 0), (0, 0, 0), (6, 0, 0)):
            with self.subTest(point=point):
                material, _ = simulation.geom_tree.material_of_point(point)
                self.assertEqual(material.eps_inf, 12)

        for point in ((-6, 0.6, 0), (0, 0.6, 0), (6, 0.6, 0)):
            with self.subTest(point=point):
                material, _ = simulation.geom_tree.material_of_point(point)
                self.assertEqual(material.eps_inf, 1)


if __name__ == "__main__":
    unittest.main()
