import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from gmes.constant import (
    Electric,
    ElectricCurrent,
    Ex,
    Hx,
    Jx,
    Jy,
    Jz,
    Magnetic,
    MagneticCurrent,
    Mx,
    My,
    Mz,
)
from gmes.geometry import Cartesian
from gmes.pw_source import (
    PointSourceEx,
    PointSourceEy,
    PointSourceEz,
    PointSourceHx,
    PointSourceHy,
    PointSourceHz,
    PointSourceParam,
)
from gmes.source import PointSource


class InterfaceGeometry:
    def __init__(self, boundary, dense_on_high_side):
        self.boundary = boundary
        self.dense_on_high_side = dense_on_high_side
        self.queries = []

    def material_of_point(self, point):
        self.queries.append(point)
        on_high_side = point[0] >= self.boundary
        value = 4 if on_high_side == self.dense_on_high_side else 1
        return SimpleNamespace(eps_inf=value, mu_inf=value), None


class PointSourceTest(unittest.TestCase):
    def test_real_and_complex_source_recordings(self):
        cases = ((PointSourceEx, Ex), (PointSourceHx, Hx))
        values = ((2.0, (0.5, 2.0)), (2 + 3j, (0.5, 2.0, 3.0)))

        for source_type, component in cases:
            for value, expected in values:
                with self.subTest(source=source_type.__name__, value=value):
                    with TemporaryDirectory() as directory:
                        output = Path(directory, "source.dat")
                        parameter = PointSourceParam(
                            src_time=SimpleNamespace(oscillator=lambda _time: value),
                            comp=component,
                            filename=output,
                        )
                        source = source_type()
                        source.attach((0, 0, 0), parameter)
                        field = np.zeros(
                            (1, 1, 1),
                            dtype=complex if np.iscomplexobj(value) else float,
                        )

                        source.update_all(field, field, field, 1, 1, 0.25, 2)
                        parameter.f.close()

                        columns = tuple(
                            float(item) for item in output.read_text().split()
                        )
                        self.assertEqual(columns, expected)
                        self.assertEqual(field[0, 0, 0], value)

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

    def test_current_sources_use_material_at_snapped_yee_point(self):
        source_time = SimpleNamespace(oscillator=lambda _time: 2.0)
        space = Cartesian((2, 2, 2), resolution=2)
        field = np.zeros((5, 5, 5))
        cases = (
            (Jx, "ex", (-0.01, 0, 0), -0.1, False),
            (Jx, "ex", (0.01, 0, 0), 0.1, True),
            (Mx, "hx", (-0.01, 0, 0), -0.005, True),
            (Mx, "hx", (0.01, 0, 0), 0.005, False),
        )

        for component, name, center, boundary, dense_on_high_side in cases:
            with self.subTest(component=component.str(), center=center):
                geometry = InterfaceGeometry(boundary, dense_on_high_side)
                source = PointSource(source_time, center, component)
                index = getattr(space, f"space_to_{name}_index")(*center)
                snapped = getattr(space, f"{name}_index_to_space")(*index)
                get_source = getattr(source, f"get_pw_source_{name}")
                pw_source = get_source(field, space, geometry)

                request_material, _ = geometry.material_of_point(center)
                self.assertEqual(request_material.eps_inf, 1)
                np.testing.assert_allclose(geometry.queries[0], snapped)

                updated = np.zeros_like(field)
                pw_source.update_all(updated, updated, updated, 1, 1, 0.5, 0)
                self.assertEqual(updated[index], -0.25)


if __name__ == "__main__":
    unittest.main()
