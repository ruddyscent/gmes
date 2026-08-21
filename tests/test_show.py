import unittest

import numpy as np

from gmes.constant import Ex, Ey, Ez, Hx, Hy, Hz, X, Y, Z
from gmes.fdtd import FDTD
from gmes.geometry import Cartesian, DefaultMedium
from gmes.material import Dielectric

try:
    from gmes.show import ShowPlane, Snapshot
except ModuleNotFoundError as error:
    if error.name != "matplotlib":
        raise
    Snapshot = None
    ShowPlane = None


@unittest.skipIf(Snapshot is None, "matplotlib is not installed")
class PlaneDisplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fdtd = FDTD(
            space=Cartesian(size=(2, 2, 2), resolution=2),
            geom_list=[DefaultMedium(Dielectric())],
            src_list=[],
            verbose=False,
        )
        cls.fdtd.init()

    def test_snapshot_constructs_for_every_component_and_axis(self):
        for component in (Ex, Ey, Ez, Hx, Hy, Hz):
            for axis in (X, Y, Z):
                with self.subTest(component=component.__name__, axis=axis.__name__):
                    snapshot = Snapshot(
                        self.fdtd,
                        component,
                        axis,
                        cut=0,
                        vrange=(0, 2),
                        title="",
                        fig_id=0,
                    )

                    self.assertEqual(snapshot.data.shape, (4, 4))

    def test_show_plane_includes_every_boundary_value(self):
        start_indices = {
            Ex: (0, 0, 0),
            Ey: (0, 0, 0),
            Ez: (0, 0, 0),
            Hx: (0, 1, 1),
            Hy: (1, 0, 1),
            Hz: (1, 1, 0),
        }
        end_indices = {
            Ex: (3, 3, 3),
            Ey: (3, 3, 3),
            Ez: (3, 3, 3),
            Hx: (3, 4, 4),
            Hy: (4, 3, 4),
            Hz: (4, 4, 3),
        }
        axis_indices = {X: 0, Y: 1, Z: 2}
        space_to_index = {
            Ex: self.fdtd.space.space_to_ex_index,
            Ey: self.fdtd.space.space_to_ey_index,
            Ez: self.fdtd.space.space_to_ez_index,
            Hx: self.fdtd.space.space_to_hx_index,
            Hy: self.fdtd.space.space_to_hy_index,
            Hz: self.fdtd.space.space_to_hz_index,
        }
        index_to_space = {
            Ex: self.fdtd.space.ex_index_to_space,
            Ey: self.fdtd.space.ey_index_to_space,
            Ez: self.fdtd.space.ez_index_to_space,
            Hx: self.fdtd.space.hx_index_to_space,
            Hy: self.fdtd.space.hy_index_to_space,
            Hz: self.fdtd.space.hz_index_to_space,
        }

        for component in (Ex, Ey, Ez, Hx, Hy, Hz):
            field = self.fdtd.field[component]
            field[:] = np.arange(field.size).reshape(field.shape)
            start = start_indices[component]
            end = end_indices[component]
            for axis in (X, Y, Z):
                with self.subTest(component=component.__name__, axis=axis.__name__):
                    axis_index = axis_indices[axis]
                    cut_space = list(index_to_space[component](*end))
                    cut_space[axis_index] = 0
                    cut_index = space_to_index[component](*cut_space)[axis_index]
                    slices = [
                        slice(start[dimension], end[dimension] + 1)
                        for dimension in range(3)
                    ]
                    slices[axis_index] = cut_index
                    expected = field[tuple(slices)]

                    plane = ShowPlane(
                        self.fdtd,
                        component,
                        axis,
                        cut=0,
                        vrange=(0, field.size),
                        interval=100,
                        title="",
                        fig_id=0,
                    )

                    np.testing.assert_array_equal(plane.data, expected)
                    self.assertEqual(plane.data[-1, -1], expected[-1, -1])


if __name__ == "__main__":
    unittest.main()
