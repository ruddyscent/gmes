"""Retained geometry lowering coverage for the Torch material planner."""

import unittest

import numpy as np

import gmes
from gmes.pygeom import GeomBoxTree

_COMPONENT_TYPES = {
    "Ex": gmes.Ex,
    "Ey": gmes.Ey,
    "Ez": gmes.Ez,
    "Hx": gmes.Hx,
    "Hy": gmes.Hy,
    "Hz": gmes.Hz,
}


def _runtime():
    """Return the explicit eager CPU configuration used by mapping regressions."""
    return gmes.TorchRuntimeConfig(
        device="cpu", precision="float64", execution_policy="dense", cpu_threads=1
    )


def _geometry(sphere_type=gmes.Sphere):
    """Build nested dielectric regions clipped by a CPML shell."""
    return [
        gmes.DefaultMedium(gmes.Dielectric(1)),
        sphere_type(gmes.Dielectric(2), radius=0.6),
        sphere_type(gmes.Dielectric(3), radius=0.3),
        gmes.Shell(gmes.Cpml()),
    ]


def _simulation(size, sphere_type=gmes.Sphere, *, source=False):
    """Create one supported eager geometry/planner realization."""
    sources = ()
    if source:
        sources = (
            gmes.PointSource(
                gmes.Continuous(freq=0.8, width=0.5),
                center=(0, 0, 0),
                component=gmes.Ez,
            ),
        )
    return gmes.TorchSimulation(
        space=gmes.Cartesian(size=size, resolution=3),
        geometry=_geometry(sphere_type),
        sources=sources,
        runtime=_runtime(),
    )


class MaterialMappingFastPathTest(unittest.TestCase):
    """Keep vectorized lowering equivalent to pointwise geometry semantics."""

    def assert_planner_maps_match_pointwise_geometry(self, simulation):
        tree = GeomBoxTree(simulation.geometry)
        geometries = tree.root.geom_list
        geometry_ids = {
            id(geometry): index for index, geometry in enumerate(geometries)
        }
        for name, component in _COMPONENT_TYPES.items():
            with self.subTest(component=name):
                shape = simulation.plan.shapes[name]
                axes = simulation.space.component_coordinate_axes(component, shape)
                expected_material = []
                expected_underlying = []
                for x in axes[0]:
                    for y in axes[1]:
                        for z in axes[2]:
                            top, underneath = tree.object_of_point((x, y, z))
                            expected_material.append(geometry_ids[id(top)])
                            expected_underlying.append(
                                -1
                                if underneath is None
                                else geometry_ids[id(underneath)]
                            )
                plan = simulation.plan.components[name]
                np.testing.assert_array_equal(
                    plan.material_ids,
                    np.asarray(expected_material, dtype=np.int32).reshape(shape),
                )
                np.testing.assert_array_equal(
                    plan.underlying_ids,
                    np.asarray(expected_underlying, dtype=np.int32).reshape(shape),
                )

    def test_batched_mapping_matches_pointwise_for_all_components_and_clipping(self):
        for size in ((1, 1, 1), (2, 0, 0)):
            with self.subTest(size=size):
                simulation = _simulation(size)
                tree = GeomBoxTree(simulation.geometry)
                self.assertTrue(tree.supports_bulk_lowering())
                top, underneath = tree.object_of_point((0, 0, 0))
                self.assertIs(top, simulation.geometry[3])
                self.assertIs(underneath, simulation.geometry[2])
                self.assert_planner_maps_match_pointwise_geometry(simulation)
                self.assertTrue(
                    np.any(simulation.plan.components["Ex"].underlying_ids == 2)
                )

    def test_pointwise_and_vectorized_geometry_have_equal_multistep_fields(self):
        class PointwiseSphere(gmes.Sphere):
            calls = 0

            def in_object(self, point):
                type(self).calls += 1
                return super().in_object(point)

        vectorized = _simulation((1, 1, 1), source=True)
        pointwise = _simulation((1, 1, 1), PointwiseSphere, source=True)
        self.assertFalse(GeomBoxTree(pointwise.geometry).supports_bulk_lowering())
        self.assertGreater(PointwiseSphere.calls, 0)
        for _ in range(3):
            vectorized.step()
            pointwise.step()
        for name, expected in vectorized.host_snapshot().items():
            np.testing.assert_array_equal(pointwise.host_snapshot()[name], expected)

    def test_opted_in_custom_geometry_uses_vectorized_lowering(self):
        class VectorizedSphere(gmes.Sphere):
            _gmes_vectorized_geometry = True

            def in_object(self, point):
                raise AssertionError("pointwise fallback must not be used")

        baseline = _simulation((1, 1, 1))
        opted_in = _simulation((1, 1, 1), VectorizedSphere)
        self.assertTrue(GeomBoxTree(opted_in.geometry).supports_bulk_lowering())
        for name, expected in baseline.plan.components.items():
            np.testing.assert_array_equal(
                opted_in.plan.components[name].material_ids, expected.material_ids
            )

    def test_custom_material_subclasses_fail_closed_before_runtime_execution(self):
        class CustomDielectric(gmes.Dielectric):
            pass

        with self.assertRaisesRegex(NotImplementedError, "does not support material"):
            gmes.TorchSimulation(
                space=gmes.Cartesian((1, 1, 1), 2),
                geometry=[
                    gmes.DefaultMedium(gmes.Dielectric()),
                    gmes.Block(CustomDielectric(), size=(1, 1, 1)),
                ],
                runtime=_runtime(),
            )


if __name__ == "__main__":
    unittest.main()
