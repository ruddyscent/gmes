"""Tests for occupancy-aware Torch material planning."""

import unittest

import numpy as np
import torch

import gmes
from gmes.geometry import GeomBoxTree
from gmes.torch_fdtd import _field_shapes
from gmes.torch_plan import COMPONENT_TYPES, TorchExecutionPlanner


def _host_plans(
    geometry,
    *,
    policy="auto",
    size=(4, 4, 4),
    resolution=2,
    tile_size=16,
    cpml_sparse_residual=False,
    precision="float64",
):
    space = gmes.Cartesian(size, resolution)
    space.dt = 0.05
    for geometric_object in geometry:
        geometric_object.init(space)
    tree = GeomBoxTree(tuple(geometry))
    shapes = _field_shapes(space)
    plans = TorchExecutionPlanner(
        geom_tree=tree,
        space=space,
        shapes=shapes,
        precision=precision,
        device_type="cpu",
        policy=policy,
        material_tile_size=31,
        execution_tile_size=tile_size,
        cpml_sparse_residual=cpml_sparse_residual,
    ).build()
    return space, tree, {plan.name: plan for plan in plans}


def _dielectric_regions(count=2):
    geometry = [gmes.DefaultMedium(gmes.Dielectric(eps_inf=1.7, mu_inf=1.05))]
    for index in range(count):
        geometry.append(
            gmes.Block(
                material=gmes.Dielectric(eps_inf=3.4, mu_inf=1.2),
                center=(-1.5 + index * 3 / max(1, count - 1), 0, 0),
                size=(0.8, 1.1, 1.1),
            )
        )
    return geometry


class ComponentPlanTest(unittest.TestCase):
    def test_exact_maps_unique_ownership_and_coefficient_sharing(self):
        geometry = _dielectric_regions(count=2)
        space, tree, plans = _host_plans(geometry)
        expected_active = int(np.prod(space.my_field_size))
        for name, plan in plans.items():
            with self.subTest(component=name):
                axes = space.component_coordinate_axes(
                    COMPONENT_TYPES[name], plan.shape
                )
                expected = tree.lower_grid(
                    *axes,
                    0,
                    int(np.prod(plan.shape)),
                    component=COMPONENT_TYPES[name],
                )
                np.testing.assert_array_equal(
                    plan.material_ids.reshape(-1), expected.material_ids
                )
                np.testing.assert_array_equal(
                    plan.underlying_ids.reshape(-1), expected.underlying_ids
                )
                self.assertEqual(plan.active_count, expected_active)
                self.assertEqual(np.count_nonzero(plan.ownership >= 0), expected_active)
                self.assertEqual(plan.launch_count, 1)
                self.assertEqual(len(plan.buckets), 1)
                bucket = plan.buckets[0]
                self.assertEqual(bucket.signature.model, "dielectric")
                self.assertEqual(len(bucket.coefficient_table), 2)
                self.assertEqual(
                    len(bucket.region_keys), len(np.unique(expected.material_ids))
                )
                self.assertFalse(plan.material_ids.flags.writeable)
                self.assertNotEqual(plan.material_ids.dtype, np.dtype(object))

    def test_auto_decision_records_static_cost_evidence(self):
        _, _, plans = _host_plans(_dielectric_regions(count=6))
        for plan in plans.values():
            record = plan.decision_record()
            self.assertEqual(record["requested_policy"], "auto")
            self.assertEqual(record["active_cells"], plan.active_count)
            for bucket, bucket_record in zip(plan.buckets, record["buckets"]):
                costs = dict(bucket.estimated_costs)
                self.assertEqual(bucket.selected_policy, min(costs, key=costs.get))
                self.assertIn("occupancy=", bucket.decision)
                self.assertGreater(bucket.estimated_bytes, 0)
                self.assertEqual(
                    bucket_record["selected_policy"], bucket.selected_policy
                )

    def test_state_width_buckets_and_magnetic_normalization(self):
        drude_one_a = gmes.Drude(
            eps_inf=1.2,
            dps=(gmes.DrudePole(omega=0.6, gamma=0.03),),
        )
        drude_one_b = gmes.Drude(
            eps_inf=1.4,
            dps=(gmes.DrudePole(omega=0.8, gamma=0.04),),
        )
        drude_four = gmes.Drude(
            eps_inf=1.3,
            dps=tuple(
                gmes.DrudePole(omega=0.5 + index * 0.1, gamma=0.03)
                for index in range(4)
            ),
        )
        geometry = [
            gmes.DefaultMedium(gmes.Dielectric(eps_inf=1.7, mu_inf=1.05)),
            gmes.Block(drude_one_a, center=(-1.2, 0, 0), size=(0.8, 2, 2)),
            gmes.Block(drude_one_b, center=(0, 0, 0), size=(0.8, 2, 2)),
            gmes.Block(drude_four, center=(1.2, 0, 0), size=(0.8, 2, 2)),
        ]
        _, _, plans = _host_plans(geometry, policy="tiled", tile_size=8)
        electric = plans["Ex"]
        drude = [
            bucket for bucket in electric.buckets if bucket.signature.model == "drude"
        ]
        self.assertEqual(
            {bucket.signature.state_shape for bucket in drude}, {(1,), (4,)}
        )
        width_one = next(
            bucket for bucket in drude if bucket.signature.state_shape == (1,)
        )
        self.assertGreaterEqual(len(width_one.region_keys), 2)
        self.assertEqual(width_one.state_width, 2)
        self.assertEqual(width_one.padded_state_width, 8)
        self.assertGreater(width_one.padding_elements_avoided, 0)
        self.assertIn("bounded max-width merge", width_one.width_decision)
        self.assertTrue(
            all(bucket.selected_policy == "tiled" for bucket in electric.buckets)
        )
        self.assertTrue(all(len(bucket.tile_origins) for bucket in electric.buckets))
        magnetic = plans["Hx"]
        self.assertEqual(
            {bucket.signature.model for bucket in magnetic.buckets},
            {"dielectric"},
        )
        self.assertEqual(len(magnetic.buckets), 1)

    def test_compound_underlying_ids_survive_bucket_indirection(self):
        geometry = [
            gmes.DefaultMedium(gmes.Dielectric(eps_inf=2.5, mu_inf=1.2)),
            gmes.Shell(material=gmes.Cpml(), thickness=0.5),
        ]
        _, _, plans = _host_plans(geometry, policy="compact")
        for plan in plans.values():
            cpml = next(
                bucket for bucket in plan.buckets if bucket.signature.model == "cpml"
            )
            self.assertTrue(np.any(cpml.region_keys[:, 1] >= 0))
            keys = np.column_stack(
                (
                    plan.material_ids.reshape(-1)[cpml.targets],
                    plan.underlying_ids.reshape(-1)[cpml.targets],
                )
            )
            np.testing.assert_array_equal(
                keys, cpml.region_keys[cpml.target_region_indices]
            )
            self.assertEqual(cpml.selected_policy, "compact")
            self.assertEqual(len(cpml.tile_origins), 0)

    def test_cpml_sparse_residual_maps_dense_base_and_active_axes(self):
        geometry = [
            gmes.DefaultMedium(gmes.Dielectric(eps_inf=2.5, mu_inf=1.2)),
            gmes.Shell(material=gmes.Cpml(kappa_max=3.0), thickness=0.5),
        ]
        _, _, plans = _host_plans(
            geometry, policy="compact", cpml_sparse_residual=True
        )
        for component_name, plan in plans.items():
            with self.subTest(component=component_name):
                bucket = next(
                    bucket
                    for bucket in plan.buckets
                    if bucket.signature.model == "cpml"
                )
                self.assertEqual(len(bucket.cpml_residual_axes), 2)
                np.testing.assert_array_equal(
                    plan.dense_inverse.reshape(-1)[bucket.targets],
                    bucket.cell_coefficients[:, 0],
                )
                active_states = 0
                for axis, residual in enumerate(bucket.cpml_residual_axes):
                    b_column, c_column, kappa_column = (
                        (1, 2, 3) if axis == 0 else (4, 5, 6)
                    )
                    expected_positions = np.flatnonzero(
                        np.logical_or(
                            bucket.cell_coefficients[:, c_column] != 0.0,
                            bucket.cell_coefficients[:, kappa_column] != 1.0,
                        )
                    )
                    np.testing.assert_array_equal(
                        residual.positions, expected_positions
                    )
                    np.testing.assert_array_equal(
                        residual.targets, bucket.targets[expected_positions]
                    )
                    np.testing.assert_array_equal(
                        residual.stencil_indices,
                        bucket.stencil_indices[
                            expected_positions, 2 * axis : 2 * axis + 2
                        ],
                    )
                    np.testing.assert_allclose(
                        residual.parameters,
                        np.column_stack(
                            (
                                bucket.cell_coefficients[expected_positions, 0],
                                bucket.cell_coefficients[
                                    expected_positions, b_column
                                ],
                                bucket.cell_coefficients[
                                    expected_positions, c_column
                                ],
                                1.0
                                / bucket.cell_coefficients[
                                    expected_positions, kappa_column
                                ]
                                - 1.0,
                            )
                        ),
                        rtol=0.0,
                        atol=0.0,
                    )
                    active_states += len(residual.targets)
                self.assertLess(active_states, 2 * bucket.target_count)
                self.assertEqual(
                    bucket.launch_count,
                    sum(bool(len(axis.targets)) for axis in bucket.cpml_residual_axes),
                )

    def test_float32_cpml_falls_back_when_residual_cancellation_is_unstable(self):
        geometry = [
            gmes.DefaultMedium(gmes.Dielectric(eps_inf=2.5, mu_inf=1.2)),
            gmes.Shell(material=gmes.Cpml(kappa_max=1e8), thickness=0.5),
        ]
        _, _, plans = _host_plans(
            geometry,
            policy="compact",
            cpml_sparse_residual=True,
            precision="float32",
        )
        for component_name, plan in plans.items():
            with self.subTest(component=component_name):
                bucket = next(
                    bucket
                    for bucket in plan.buckets
                    if bucket.signature.model == "cpml"
                )
                self.assertEqual(bucket.cpml_residual_axes, ())
                np.testing.assert_array_equal(
                    plan.dense_inverse.reshape(-1)[bucket.targets],
                    0.0,
                )
                self.assertEqual(bucket.launch_count, 1)


class ExecutionPolicyTest(unittest.TestCase):
    def test_forced_policies_produce_identical_complete_fields(self):
        rng = np.random.default_rng(117)
        fields = None
        results = {}
        for policy in ("dense", "compact", "tiled"):
            simulation = gmes.TorchSimulation(
                space=gmes.Cartesian((2, 2, 2), 2),
                geometry=_dielectric_regions(),
                runtime=gmes.TorchRuntimeConfig(
                    device="cpu",
                    cpu_threads=2,
                    execution_policy=policy,
                    planner_tile_size=8,
                ),
            )
            if fields is None:
                fields = {
                    name: rng.normal(size=tuple(value.shape)) * 1e-3
                    for name, value in simulation.state.fields().items()
                }
            simulation.load_host_fields(fields)
            simulation.advance(3)
            results[policy] = simulation.state.host_snapshot()
            for component in simulation.plan.components.values():
                self.assertTrue(
                    all(
                        bucket.selected_policy == policy for bucket in component.buckets
                    )
                )
        for name in gmes.torch_plan.COMPONENTS:
            np.testing.assert_array_equal(
                results["dense"][name], results["compact"][name]
            )
            np.testing.assert_array_equal(
                results["dense"][name], results["tiled"][name]
            )

    def test_const_and_dummy_paths_match_native_from_nonzero_fields(self):
        geometry = [
            gmes.DefaultMedium(gmes.Dielectric(eps_inf=1.7, mu_inf=1.05)),
            gmes.Block(
                gmes.Const(value=0.25, eps_inf=1.1, mu_inf=1.2),
                center=(-0.5, 0, 0),
                size=(0.7, 1.5, 1.5),
            ),
            gmes.Block(
                gmes.Dummy(eps_inf=1.3, mu_inf=1.1),
                center=(0.5, 0, 0),
                size=(0.7, 1.5, 1.5),
            ),
        ]
        native = gmes.FDTD(gmes.Cartesian((2, 2, 2), 3), geometry, verbose=False)
        native.init()
        simulation = gmes.TorchSimulation(
            space=gmes.Cartesian((2, 2, 2), 3),
            geometry=geometry,
            runtime=gmes.TorchRuntimeConfig(device="cpu", cpu_threads=2),
        )
        rng = np.random.default_rng(1717)
        fields = {}
        for component, native_field in native.field.items():
            values = rng.normal(size=native_field.shape) * 1e-3
            native_field[...] = values
            fields[component.__name__] = values.copy()
        simulation.load_host_fields(fields)
        native.step()
        simulation.step()
        actual = simulation.state.host_snapshot()
        for component, native_field in native.field.items():
            np.testing.assert_allclose(
                actual[component.__name__],
                native_field,
                rtol=1e-13,
                atol=1e-15,
                err_msg=component.__name__,
            )
        addresses = simulation.buffer_addresses()
        simulation.advance(4)
        self.assertEqual(addresses, simulation.buffer_addresses())

    def test_policy_buffers_are_finalized_once_and_non_trainable(self):
        simulation = gmes.TorchSimulation(
            space=gmes.Cartesian((2, 2, 2), 2),
            geometry=_dielectric_regions(count=6),
            runtime=gmes.TorchRuntimeConfig(
                device="cpu",
                cpu_threads=2,
                execution_policy="tiled",
                planner_tile_size=8,
            ),
        )
        self.assertEqual(list(simulation.plan.parameters()), [])
        self.assertTrue(
            any(
                "tile_region_indices" in name
                for name, _ in simulation.plan.named_buffers()
            )
        )
        for _, value in simulation.plan.named_buffers():
            self.assertFalse(value.requires_grad)
            self.assertEqual(value.device.type, "cpu")
        before = {
            name: value.data_ptr() for name, value in simulation.plan.named_buffers()
        }
        simulation.advance(5)
        after = {
            name: value.data_ptr() for name, value in simulation.plan.named_buffers()
        }
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
