import unittest
from copy import deepcopy

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
        self.assertAlmostEqual(np.sum(np.abs(simulation.ez) ** 2), 1.5780099423691636)

    def test_initialized_real_and_bloch_simulations_deepcopy(self):
        for bloch in (None, (0.1, 0.2, 0)):
            with self.subTest(bloch=bloch):
                space = Cartesian(size=(2, 2, 0), resolution=3)
                geometry = [DefaultMedium(material=Dielectric())]
                sources = [
                    PointSource(
                        src_time=Continuous(freq=0.8, width=0.5),
                        center=(0, 0, 0),
                        component=Ez,
                    )
                ]
                simulation = TMzFDTD(
                    space, geometry, sources, bloch=bloch, verbose=False
                )
                simulation.init()
                simulation.step()

                copied = deepcopy(simulation)

                self.assertIsNot(copied.space, simulation.space)
                self.assertIsNot(copied.geom_list[0], simulation.geom_list[0])
                self.assertIsNot(copied.src_list[0], simulation.src_list[0])
                self.assertEqual(copied.time_step.n, simulation.time_step.n)
                self.assertEqual(copied.cmplx, simulation.cmplx)
                if bloch is not None:
                    np.testing.assert_allclose(copied.bloch, simulation.bloch)
                for component in ("ex", "ey", "ez", "hx", "hy", "hz"):
                    original_field = getattr(simulation, component)
                    copied_field = getattr(copied, component)
                    self.assertIsNot(copied_field, original_field)
                    np.testing.assert_allclose(copied_field, original_field)

                simulation.step()
                self.assertEqual(simulation.time_step.n, 2)
                self.assertEqual(copied.time_step.n, 1)
                copied.step()
                for component in ("ex", "ey", "ez", "hx", "hy", "hz"):
                    np.testing.assert_allclose(
                        getattr(copied, component), getattr(simulation, component)
                    )


if __name__ == "__main__":
    unittest.main()
