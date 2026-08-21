import unittest

import numpy as np

from gmes import Cone, Cylinder, DefaultMedium, Dielectric
from gmes.pygeom import GeomBoxTree


class ConeBoundsTest(unittest.TestCase):
    def test_rotated_end_caps_define_cartesian_bounds(self):
        center = np.array((0.25, -0.5, 0.75))
        axes = ((1, 1, 0), (1, -2, 3), (-3, 1, 2))

        for shape_class, radius, radius2 in (
            (Cylinder, 1.2, 1.2),
            (Cone, 1.2, 0.4),
        ):
            for axis in axes:
                with self.subTest(shape=shape_class.__name__, axis=axis):
                    if shape_class is Cylinder:
                        shape = shape_class(
                            Dielectric(),
                            center=center,
                            axis=axis,
                            radius=radius,
                            height=1.5,
                        )
                    else:
                        shape = shape_class(
                            Dielectric(),
                            center=center,
                            axis=axis,
                            radius=radius,
                            radius2=radius2,
                            height=1.5,
                        )

                    box = shape.geom_box()
                    radial = np.sqrt(1 - shape.axis * shape.axis)
                    low_cap = center - 0.75 * shape.axis
                    high_cap = center + 0.75 * shape.axis
                    expected_low = np.minimum(
                        low_cap - radius * radial,
                        high_cap - radius2 * radial,
                    )
                    expected_high = np.maximum(
                        low_cap + radius * radial,
                        high_cap + radius2 * radial,
                    )

                    np.testing.assert_allclose(box.low, expected_low)
                    np.testing.assert_allclose(box.high, expected_high)

                    perpendicular = np.cross(shape.axis, (0, 0, 1))
                    if np.linalg.norm(perpendicular) < 1e-12:
                        perpendicular = np.cross(shape.axis, (0, 1, 0))
                    perpendicular /= np.linalg.norm(perpendicular)
                    point = center - 0.75 * shape.axis + 0.9 * radius * perpendicular
                    self.assertTrue(shape.in_object(tuple(point)))
                    self.assertTrue(box.in_box(tuple(point)))

    def test_tree_keeps_rotated_cylinder_interior_point(self):
        default = DefaultMedium(Dielectric(1))
        cylinder = Cylinder(Dielectric(4), axis=(1, 1, 0), radius=1, height=0.1)
        default.init(None)
        cylinder.init(None)
        tree = GeomBoxTree((default, cylinder))
        point = (0.9 / np.sqrt(2), -0.9 / np.sqrt(2), 0)

        shape, _ = tree.object_of_point(point)

        self.assertIs(shape, cylinder)


if __name__ == "__main__":
    unittest.main()
