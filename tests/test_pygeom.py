import unittest

import numpy as np

from gmes import (
    Block,
    Cartesian,
    Cone,
    Cylinder,
    DefaultMedium,
    Dielectric,
    Ellipsoid,
    Shell,
)
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
                    point = center - 0.7 * shape.axis + 0.9 * radius * perpendicular
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


class ShellBoundsTest(unittest.TestCase):
    def test_shell_bounds_and_tree_follow_center(self):
        space = Cartesian(size=(12, 4, 4), resolution=10)

        for center in ((0, 0, 0), (5, 0, 0)):
            with self.subTest(center=center):
                default = DefaultMedium(Dielectric(1))
                shell = Shell(
                    Dielectric(4),
                    center=center,
                    size=(2, 2, 2),
                    thickness=0.2,
                )
                default.init(space)
                shell.init(space)
                tree = GeomBoxTree((default, shell))
                point = (center[0] + 0.9, center[1], center[2])

                np.testing.assert_allclose(shell.box.low, np.array(center) - 1)
                np.testing.assert_allclose(shell.box.high, np.array(center) + 1)
                self.assertTrue(shell.in_object(point))
                self.assertTrue(shell.box.in_box(point))
                shape, _ = tree.object_of_point(point)
                self.assertIs(shape, shell)


class SkewBasisTest(unittest.TestCase):
    def test_block_uses_inverse_basis_coordinates(self):
        block = Block(
            Dielectric(),
            center=(0.25, -0.5, 0.75),
            e1=(1, 0, 0),
            e2=(1, 1, 0),
            e3=(0, 0, 1),
            size=(2, 2, 2),
        )

        inside = block.center + 0.9 * block.e1 + 0.9 * block.e2
        outside = block.center + 1.1 * block.e1

        self.assertTrue(block.in_object(tuple(inside)))
        self.assertFalse(block.in_object(tuple(outside)))

    def test_ellipsoid_uses_inverse_basis_coordinates(self):
        ellipsoid = Ellipsoid(
            Dielectric(),
            center=(-0.25, 0.5, -0.75),
            e1=(1, 0, 0),
            e2=(1, 1, 0),
            e3=(0, 0, 1),
            size=(2, 2, 2),
        )

        inside = ellipsoid.center + 0.6 * ellipsoid.e1 + 0.6 * ellipsoid.e2
        outside = ellipsoid.center + 0.8 * ellipsoid.e1 + 0.8 * ellipsoid.e2

        self.assertTrue(ellipsoid.in_object(tuple(inside)))
        self.assertFalse(ellipsoid.in_object(tuple(outside)))


if __name__ == "__main__":
    unittest.main()
