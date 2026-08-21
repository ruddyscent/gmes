import unittest

from gmes.constant import Ex, Ey, Ez, Hx, Hy, Hz, X, Y, Z
from gmes.fdtd import FDTD
from gmes.geometry import Cartesian, DefaultMedium
from gmes.material import Dielectric
from gmes.show import Snapshot


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


if __name__ == "__main__":
    unittest.main()
