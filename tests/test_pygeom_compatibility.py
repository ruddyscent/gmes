import base64
import pickle
import unittest
from copy import deepcopy

import numpy as np

from gmes import (
    Block,
    Cartesian,
    Cone,
    Cpml,
    Cylinder,
    DefaultMedium,
    Dielectric,
    Ellipsoid,
    Shell,
    Sphere,
)
from gmes.pygeom import Compound, GeomBoxTree, GeometryMap, Material


class CompoundTestMaterial(Material, Compound):
    def init(self, space, param=None):
        pass


# Protocol-4 payloads produced by the Cython gmes.pygeom module at f3f0627.
LEGACY_CYTHON_SPHERE_PICKLES = {
    "before_init": (
        "gASVPQEAAAAAAACMC2dtZXMucHlnZW9tlIwGU3BoZXJllJOUToWUUpR9lCiMCG1h"
        "dGVyaWFslIwNZ21lcy5tYXRlcmlhbJSMCkRpZWxlY3RyaWOUk5QpUpR9lCiMB2Vw"
        "c19pbmaUR0AEAAAAAAAAjAZtdV9pbmaURz/wAAAAAAAAdWKMA2JveJROjAZyYWRp"
        "dXOURz/ZmZmZmZmajAZjZW50ZXKUjBZudW1weS5fY29yZS5tdWx0aWFycmF5lIwM"
        "X3JlY29uc3RydWN0lJOUjAVudW1weZSMB25kYXJyYXmUk5RLAIWUQwFilIeUUpQo"
        "SwFLA4WUaBSMBWR0eXBllJOUjAJmOJSJiIeUUpQoSwOMATyUTk5OSv////9K////"
        "/0sAdJRiiUMYAAAAAAAA0D8AAAAAAADgvwAAAAAAAOg/lHSUYnViLg=="
    ),
    "after_init": (
        "gASVywEAAAAAAACMC2dtZXMucHlnZW9tlIwGU3BoZXJllJOUToWUUpR9lCiMCG1h"
        "dGVyaWFslIwNZ21lcy5tYXRlcmlhbJSMCkRpZWxlY3RyaWOUk5QpUpR9lCiMB2Vw"
        "c19pbmaUR0AEAAAAAAAAjAZtdV9pbmaURz/wAAAAAAAAdWKMA2JveJRoAIwHR2Vv"
        "bUJveJSTlClSlH2UKIwDbG93lIwWbnVtcHkuX2NvcmUubXVsdGlhcnJheZSMDF9y"
        "ZWNvbnN0cnVjdJSTlIwFbnVtcHmUjAduZGFycmF5lJOUSwCFlEMBYpSHlFKUKEsB"
        "SwOFlGgXjAVkdHlwZZSTlIwCZjiUiYiHlFKUKEsDjAE8lE5OTkr/////Sv////9L"
        "AHSUYolDGDQzMzMzM8O/zczMzMzM7L9mZmZmZmbWP5R0lGKMBGhpZ2iUaBZoGUsA"
        "hZRoG4eUUpQoSwFLA4WUaCOJQxjNzMzMzMzkP5iZmZmZmbm/ZmZmZmZm8j+UdJRi"
        "dWKMBnJhZGl1c5RHP9mZmZmZmZqMBmNlbnRlcpRoFmgZSwCFlGgbh5RSlChLAUsD"
        "hZRoI4lDGAAAAAAAANA/AAAAAAAA4L8AAAAAAADoP5R0lGJ1Yi4="
    ),
}


class GeometryMapCompatibilityTest(unittest.TestCase):
    def setUp(self):
        self.space = Cartesian(size=(4, 4, 4), resolution=4)
        self.space.dt = 0.1

    def initialized_tree(self, geometry):
        for obj in geometry:
            obj.init(self.space)
        return GeomBoxTree(geometry)

    def test_region_ids_preserve_overlap_and_immediate_underlying_region(self):
        geometry = (
            DefaultMedium(Dielectric(1)),
            Sphere(Dielectric(2), radius=1),
            Sphere(CompoundTestMaterial(), radius=0.75),
            Sphere(CompoundTestMaterial(), radius=0.5),
            Sphere(Dielectric(5), radius=0.25),
        )
        tree = self.initialized_tree(geometry)
        axes = (
            np.array((0.8, 0.6, 0.4, 0.0), dtype=np.double),
            np.array((0.0,), dtype=np.double),
            np.array((0.0,), dtype=np.double),
        )

        geometry_map = tree.lower_grid(*axes, component="Ez")

        np.testing.assert_array_equal(geometry_map.material_ids, (1, 2, 3, 4))
        np.testing.assert_array_equal(geometry_map.underlying_ids, (-1, 1, 2, -1))
        self.assertEqual(
            geometry_map.materials, tuple(obj.material for obj in geometry)
        )
        self.assertEqual(geometry_map.shape, (4, 1, 1))
        self.assertEqual((geometry_map.start, geometry_map.stop), (0, 4))
        self.assertEqual(geometry_map.component, "Ez")

    def test_boundary_and_adjacent_points_match_scalar_lookup(self):
        cases = (
            (Sphere(Dielectric(2), radius=1), (1.0, 0.0, 0.0)),
            (Block(Dielectric(2), size=(2, 2, 2)), (1.0, 0.0, 0.0)),
            (Ellipsoid(Dielectric(2), size=(2, 2, 2)), (1.0, 0.0, 0.0)),
            (
                Cylinder(Dielectric(2), axis=(0, 0, 1), radius=1, height=2),
                (1.0, 0.0, 0.0),
            ),
            (
                Cone(Dielectric(2), axis=(0, 0, 1), radius=1, height=2),
                (1.0, 0.0, -1.0),
            ),
            (
                Shell(Dielectric(2), size=(4, 4, 4), thickness=0.5),
                (1.5, 0.0, 0.0),
            ),
        )
        for geometry, boundary in cases:
            with self.subTest(geometry=type(geometry).__name__):
                tree = self.initialized_tree((DefaultMedium(Dielectric(1)), geometry))
                points = (
                    boundary,
                    tuple(np.nextafter(value, -np.inf) for value in boundary),
                    tuple(np.nextafter(value, np.inf) for value in boundary),
                )
                for point in points:
                    axes = tuple(
                        np.array((coordinate,), dtype=np.double) for coordinate in point
                    )
                    geometry_map = tree.lower_grid(*axes)
                    mapped = geometry_map.geometries[geometry_map.material_ids[0]]
                    scalar, _ = tree.object_of_point(point)
                    self.assertIs(mapped, scalar)

    def test_custom_vectorized_protocol_is_explicit_and_inherited(self):
        class PointwiseSphere(Sphere):
            calls = 0

            def in_object(self, point):
                type(self).calls += 1
                return super().in_object(point)

        class VectorizedSphere(Sphere):
            _gmes_vectorized_geometry = True

            def in_object(self, point):
                raise AssertionError("the vectorized protocol was not used")

        class InheritedVectorizedSphere(VectorizedSphere):
            pass

        axes = (
            np.array((-0.5, 0.0, 0.5), dtype=np.double),
            np.array((0.0,), dtype=np.double),
            np.array((0.0,), dtype=np.double),
        )
        pointwise = PointwiseSphere(Dielectric(2), radius=0.25)
        self.initialized_tree((DefaultMedium(Dielectric(1)), pointwise)).lower_grid(
            *axes
        )
        self.assertGreater(PointwiseSphere.calls, 0)

        vectorized = InheritedVectorizedSphere(Dielectric(2), radius=0.25)
        geometry_map = self.initialized_tree(
            (DefaultMedium(Dielectric(1)), vectorized)
        ).lower_grid(*axes)
        np.testing.assert_array_equal(geometry_map.material_ids, (0, 1, 0))

    def test_geometry_map_rejects_invalid_region_ids(self):
        geometry = (DefaultMedium(Dielectric()),)
        with self.assertRaisesRegex(ValueError, "region ID"):
            GeometryMap(
                np.array((1,), dtype=np.int32),
                np.array((-1,), dtype=np.int32),
                geometry,
                (1, 1, 1),
                0,
                1,
            )


class GeometryPickleCompatibilityTest(unittest.TestCase):
    def test_loads_cython_sphere_pickles_before_and_after_initialization(self):
        for state, payload in LEGACY_CYTHON_SPHERE_PICKLES.items():
            with self.subTest(state=state):
                sphere = pickle.loads(base64.b64decode(payload))
                self.assertIsInstance(sphere, Sphere)
                self.assertEqual(sphere.material.eps_inf, 2.5)
                self.assertEqual(sphere.radius, 0.4)
                np.testing.assert_array_equal(sphere.center, (0.25, -0.5, 0.75))
                if state == "before_init":
                    self.assertIsNone(sphere.box)
                else:
                    np.testing.assert_allclose(sphere.box.low, (-0.15, -0.9, 0.35))
                    np.testing.assert_allclose(sphere.box.high, (0.65, -0.1, 1.15))

    def test_current_geometry_pickle_and_deepcopy_preserve_point_queries(self):
        space = Cartesian(size=(4, 4, 4), resolution=2)
        space.dt = 0.1
        geometry = (
            DefaultMedium(Dielectric(1)),
            Cylinder(Dielectric(2), axis=(1, 1, 0), radius=0.5, height=2),
            Shell(Cpml(), center=(0.5, 0, 0), size=(3, 3, 3), thickness=0.25),
        )
        for obj in geometry:
            obj.init(space)
        tree = GeomBoxTree(geometry)
        points = ((0, 0, 0), (0.7, -0.7, 0), (1.9, 0, 0))

        for restored in (pickle.loads(pickle.dumps(tree)), deepcopy(tree)):
            self.assertEqual(
                [type(restored.material_of_point(point)[0]) for point in points],
                [type(tree.material_of_point(point)[0]) for point in points],
            )

    def test_material_assignment_retains_cython_type_requirement(self):
        sphere = Sphere(Dielectric())
        with self.assertRaisesRegex(TypeError, "Material instance"):
            sphere.material = object()


if __name__ == "__main__":
    unittest.main()
