import unittest

import numpy as np

from gmes import (
    Cartesian,
    Continuous,
    Cpml,
    DefaultMedium,
    Dielectric,
    Ez,
    PointSource,
    Shell,
    TMzFDTD,
)


class FDTDSmokeTest(unittest.TestCase):
    def test_tmz_point_source_regression(self):
        space = Cartesian(size=(2, 2, 0), resolution=5)
        geometry = [
            DefaultMedium(material=Dielectric()),
            Shell(material=Cpml()),
        ]
        sources = [
            PointSource(
                src_time=Continuous(freq=0.8, width=0.5),
                center=(0, 0, 0),
                component=Ez,
            ),
        ]
        simulation = TMzFDTD(space, geometry, sources, verbose=False)

        simulation.init()
        for _ in range(5):
            simulation.step()

        self.assertEqual(simulation.time_step.n, 5.0)
        self.assertAlmostEqual(simulation.time_step.t, 0.7000357133746822)
        self.assertEqual(simulation.ez.shape, (11, 11, 1))
        self.assertTrue(np.isfinite(simulation.ez).all())
        self.assertAlmostEqual(simulation.ez[5, 5, 0], -0.9996801161298625)
        self.assertAlmostEqual(np.sum(np.abs(simulation.ez) ** 2),
                               1.5780099423691636)


if __name__ == "__main__":
    unittest.main()
