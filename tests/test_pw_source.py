import unittest
from types import SimpleNamespace

import numpy as np

from gmes.constant import (
    Electric,
    ElectricCurrent,
    Jx,
    Jy,
    Jz,
    Magnetic,
    MagneticCurrent,
    Mx,
    My,
    Mz,
)
from gmes.pw_source import (
    PointSourceEx,
    PointSourceEy,
    PointSourceEz,
    PointSourceHx,
    PointSourceHy,
    PointSourceHz,
    PointSourceParam,
)


class PointSourceTest(unittest.TestCase):
    def test_current_component_hierarchy(self):
        for component in (Jx, Jy, Jz):
            self.assertTrue(issubclass(component, ElectricCurrent))
            self.assertFalse(issubclass(component, Electric))

        for component in (Mx, My, Mz):
            self.assertTrue(issubclass(component, MagneticCurrent))
            self.assertFalse(issubclass(component, Magnetic))

    def test_point_current_sources_apply_material_scaled_updates(self):
        source_time = SimpleNamespace(oscillator=lambda _time: 2.0)
        cases = (
            (PointSourceEx, Jx),
            (PointSourceEy, Jy),
            (PointSourceEz, Jz),
            (PointSourceHx, Mx),
            (PointSourceHy, My),
            (PointSourceHz, Mz),
        )

        for source_type, component in cases:
            with self.subTest(component=component.str()):
                field = np.zeros((1, 1, 1))
                source = source_type()
                source.attach(
                    (0, 0, 0),
                    PointSourceParam(
                        src_time=source_time,
                        comp=component,
                        eps_inf=4,
                        mu_inf=4,
                    ),
                )

                source.update_all(field, field, field, 1, 1, 0.5, 0)

                self.assertEqual(field[0, 0, 0], -0.25)


if __name__ == "__main__":
    unittest.main()
