"""Tests for the deliberately breaking Torch-native execution path."""

import hashlib
import json
import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import gmes
from gmes.torch_fdtd import (
    BOUNDARY_SYNC_REPRESENTATION,
    DEFAULT_VIEW_MUTATION_REPRESENTATION,
    DIRECT_VIEW_MUTATION_REPRESENTATION,
    EXTERNAL_SOURCE_REPRESENTATION,
    FUNCTIONAL_DM2_REPRESENTATION,
    FUSED_SOURCE_REPRESENTATION,
    LOCAL_COMPILED_REGION_TOPOLOGY,
    PACKED_DM2_REPRESENTATION,
    TORCH_SOLVER_ABI,
    DistributedLaunch,
    TorchConfigurationError,
    TorchRuntimeConfig,
    TorchSimulation,
    _boundary_plane,
    _field_region,
    torch_runtime_diagnostics,
)

_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
_MANIFEST = json.loads(
    (
        Path(__file__).parents[1] / "benchmarks" / "native_oracle_workloads.json"
    ).read_text()
)
_TOLERANCES = _MANIFEST["tolerances"]["torch"]["dielectric"]


def _geometry():
    return [gmes.DefaultMedium(gmes.Dielectric(eps_inf=1.7, mu_inf=1.05))]


def _simulation(
    *,
    size=(2, 2, 2),
    resolution=2,
    device="cpu",
    precision="float64",
    compile_policy="eager",
    bloch=None,
):
    return TorchSimulation(
        space=gmes.Cartesian(size, resolution),
        geometry=_geometry(),
        runtime=TorchRuntimeConfig(
            device=device,
            precision=precision,
            compile_policy=compile_policy,
            cpu_threads=2,
        ),
        bloch=bloch,
    )


def _native_and_fields(*, size=(2, 2, 2), resolution=2, bloch=None):
    native = gmes.FDTD(
        gmes.Cartesian(size, resolution),
        _geometry(),
        bloch=bloch,
        verbose=False,
    )
    native.init()
    rng = np.random.default_rng(1729)
    for field in native.field.values():
        values = rng.normal(size=field.shape) * 1e-3
        if bloch is not None:
            values = values + 1j * rng.normal(size=field.shape) * 1e-3
        field[...] = values
        assert np.all(field != 0)
    fields = {
        component.__name__: field.copy() for component, field in native.field.items()
    }
    return native, fields


def _assert_matches_native(test, simulation, native, *, steps, precision):
    simulation.advance(steps)
    for _ in range(steps):
        native.step()
    actual = simulation.state.host_snapshot()
    if native.cmplx:
        tolerance_name = "complex128" if precision == "float64" else "complex64"
    else:
        tolerance_name = precision
    tolerance = _TOLERANCES[tolerance_name]
    for name in _COMPONENTS:
        np.testing.assert_allclose(
            actual[name],
            native.field[getattr(gmes, name)],
            rtol=tolerance["rtol"],
            atol=tolerance["atol"],
            err_msg=name,
        )


class TorchRuntimeConfigTest(unittest.TestCase):
    def test_public_api_exports_diagnostics(self):
        self.assertIn("torch_runtime_diagnostics", gmes.__all__)
        self.assertIs(gmes.torch_runtime_diagnostics, torch_runtime_diagnostics)

    def test_diagnostics_are_focused_and_report_requested_precision(self):
        config = TorchRuntimeConfig(device="cpu", precision="float32", cpu_threads=1)
        diagnostics = torch_runtime_diagnostics(config)
        self.assertEqual(diagnostics["requested_device"], "cpu")
        self.assertEqual(diagnostics["requested_precision"], "float32")
        self.assertIn("torch", diagnostics)
        self.assertIn("cuda_available", diagnostics)
        self.assertIn("nccl_available", diagnostics)
        self.assertFalse(diagnostics["experimental_dispersive_grouping"])
        self.assertNotIn("environment", diagnostics)

    def test_invalid_requests_are_rejected(self):
        invalid = (
            TorchRuntimeConfig(device="cpu", precision="float16"),
            TorchRuntimeConfig(device="mps"),
            TorchRuntimeConfig(device="cpu", autograd=True),
            TorchRuntimeConfig(device="cpu", compile_mode="reduce-overhead"),
            TorchRuntimeConfig(
                device="cuda:0",
                compile_policy="eager",
                compile_mode="max-autotune",
            ),
            TorchRuntimeConfig(device="cpu", cpu_interop_threads=0),
            TorchRuntimeConfig(
                device="cpu",
                experimental_dispersive_grouping=1,
            ),
            TorchRuntimeConfig(
                device="cpu",
                experimental_dispersive_grouping_scope="unknown",
            ),
            TorchRuntimeConfig(
                device="cpu",
                launch=DistributedLaunch(world_size=2, local_world_size=2),
            ),
        )
        for config in invalid:
            with (
                self.subTest(config=config),
                self.assertRaises((TorchConfigurationError, ValueError)),
            ):
                _ = TorchSimulation(
                    space=gmes.Cartesian((1, 1, 1), 1),
                    geometry=_geometry(),
                    runtime=config,
                )

    def test_oversubscribed_process_metadata_is_rejected(self):
        processors = os.cpu_count() or 1
        config = TorchRuntimeConfig(
            device="cpu",
            cpu_threads=processors,
            launch=DistributedLaunch(local_world_size=2),
        )
        with self.assertRaisesRegex(TorchConfigurationError, "oversubscribe"):
            TorchSimulation(
                space=gmes.Cartesian((1, 1, 1), 1),
                geometry=_geometry(),
                runtime=config,
            )

    def test_compile_cache_key_tracks_execution_specialization(self):
        common = {
            "space": gmes.Cartesian((2, 2, 0), 2),
            "geometry": _geometry(),
        }
        first = TorchSimulation(
            **common,
            runtime=TorchRuntimeConfig(
                device="cpu",
                compile_policy="compile",
                execution_policy="auto",
                cpu_threads=1,
            ),
        )
        second = TorchSimulation(
            **common,
            runtime=TorchRuntimeConfig(
                device="cpu",
                compile_policy="compile",
                execution_policy="dense",
                cpu_threads=1,
            ),
        )

        self.assertEqual(len(first.compile_cache_key), 64)
        self.assertEqual(
            hashlib.sha256(
                repr(first._compile_cache_key_preimage).encode()
            ).hexdigest(),
            first.compile_cache_key,
        )
        self.assertEqual(first.compile_cache_key, second.compile_cache_key)
        self.assertEqual(
            first.diagnostics()["material_execution_representation"],
            "dense-base+compact-indexed-materials-v1",
        )
        self.assertEqual(first.diagnostics()["phase_specialization"], "z-collapsed-v1")
        self.assertEqual(
            first.diagnostics()["compile_cache_key"], first.compile_cache_key
        )
        self.assertEqual(first.diagnostics()["compile_solver_abi"], TORCH_SOLVER_ABI)
        self.assertEqual(first._compile_cache_key_preimage[0], TORCH_SOLVER_ABI)
        self.assertEqual(
            first._compile_cache_key_preimage[6], LOCAL_COMPILED_REGION_TOPOLOGY
        )
        self.assertEqual(
            first.diagnostics()["view_mutation_representation"],
            DIRECT_VIEW_MUTATION_REPRESENTATION,
        )
        self.assertEqual(
            first.diagnostics()["dm2_execution_representation"],
            PACKED_DM2_REPRESENTATION,
        )
        self.assertEqual(
            first.diagnostics()["sources"]["execution_representation"],
            FUSED_SOURCE_REPRESENTATION,
        )
        self.assertEqual(
            first.diagnostics()["boundaries"]["execution_representation"],
            BOUNDARY_SYNC_REPRESENTATION,
        )

        three_dimensional = {
            "space": gmes.Cartesian((2, 2, 2), 1),
            "geometry": _geometry(),
        }
        compiled_3d = TorchSimulation(
            **three_dimensional,
            runtime=TorchRuntimeConfig(
                device="cpu", compile_policy="compile", cpu_threads=1
            ),
        )
        eager_3d = TorchSimulation(
            **three_dimensional,
            runtime=TorchRuntimeConfig(
                device="cpu", compile_policy="eager", cpu_threads=1
            ),
        )
        self.assertNotEqual(compiled_3d.compile_cache_key, eager_3d.compile_cache_key)
        self.assertEqual(
            eager_3d.diagnostics()["view_mutation_representation"],
            DEFAULT_VIEW_MUTATION_REPRESENTATION,
        )
        self.assertEqual(
            eager_3d.diagnostics()["dm2_execution_representation"],
            FUNCTIONAL_DM2_REPRESENTATION,
        )
        self.assertEqual(
            eager_3d.diagnostics()["sources"]["execution_representation"],
            EXTERNAL_SOURCE_REPRESENTATION,
        )

        bloch_a = TorchSimulation(
            **common,
            bloch=(0.07, 0.11, 0.0),
            runtime=TorchRuntimeConfig(
                device="cpu", compile_policy="compile", cpu_threads=1
            ),
        )
        bloch_b = TorchSimulation(
            **common,
            bloch=(0.08, 0.11, 0.0),
            runtime=TorchRuntimeConfig(
                device="cpu", compile_policy="compile", cpu_threads=1
            ),
        )
        self.assertNotEqual(first.compile_cache_key, bloch_a.compile_cache_key)
        self.assertNotEqual(bloch_a.compile_cache_key, bloch_b.compile_cache_key)
        self.assertEqual(bloch_a.diagnostics()["phase_specialization"], "three-axis-v1")

        pole = gmes.DrudePole(omega=0.6, gamma=0.03)
        material = gmes.Drude(eps_inf=1.2, dps=(pole,))
        material_runtime = TorchRuntimeConfig(
            device="cpu", compile_policy="compile", cpu_threads=1
        )
        sparse = TorchSimulation(
            space=common["space"],
            geometry=[
                *_geometry(),
                gmes.Block(material, center=(0, 0, 0), size=(0.5, 0.5, 1)),
            ],
            runtime=material_runtime,
        )
        sparse_forced = TorchSimulation(
            space=common["space"],
            geometry=[
                *_geometry(),
                gmes.Block(material, center=(0, 0, 0), size=(0.5, 0.5, 1)),
            ],
            runtime=TorchRuntimeConfig(
                device="cpu",
                compile_policy="compile",
                execution_policy="dense",
                cpu_threads=1,
            ),
        )
        broad = TorchSimulation(
            space=common["space"],
            geometry=[
                *_geometry(),
                gmes.Block(material, center=(0, 0, 0), size=(1.5, 1.5, 1)),
            ],
            runtime=material_runtime,
        )
        self.assertNotEqual(sparse.compile_cache_key, sparse_forced.compile_cache_key)
        self.assertNotEqual(
            sparse.diagnostics()["dispersive"]["execution_representation"],
            sparse_forced.diagnostics()["dispersive"]["execution_representation"],
        )
        drude_buckets = [
            bucket
            for component in sparse.plan.components.values()
            for bucket in component.buckets
            if bucket.signature.model == "drude"
        ]
        self.assertTrue(drude_buckets)
        self.assertTrue(
            all(bucket.selected_policy == "compact" for bucket in drude_buckets)
        )
        self.assertNotEqual(sparse.compile_cache_key, broad.compile_cache_key)

    def test_missing_cuda_has_actionable_error_and_no_fallback(self):
        config = TorchRuntimeConfig(device="cuda:0")
        with (
            mock.patch("torch.cuda.is_available", return_value=False),
            self.assertRaisesRegex(
                TorchConfigurationError,
                "CUDA 12.6/13.0.*device='cpu'",
            ),
        ):
            TorchSimulation(
                space=gmes.Cartesian((1, 1, 1), 1),
                geometry=_geometry(),
                runtime=config,
            )


class TorchStateTest(unittest.TestCase):
    def test_direct_mutation_views_match_the_solver_slices(self):
        field = torch.arange(5 * 6 * 7 * 2, dtype=torch.float64).reshape(5, 6, 7, 2)
        regions = (
            ((0, 0, 0), (0, 1, 1), field[:, :-1, :-1]),
            ((0, 0, 0), (1, 0, 1), field[:-1, :, :-1]),
            ((0, 0, 0), (1, 1, 0), field[:-1, :-1, :]),
            ((0, 1, 1), (0, 0, 0), field[:, 1:, 1:]),
            ((1, 0, 1), (0, 0, 0), field[1:, :, 1:]),
            ((1, 1, 0), (0, 0, 0), field[1:, 1:, :]),
        )
        for starts, trims, expected in regions:
            with self.subTest(starts=starts, trims=trims):
                actual = _field_region(field, starts, trims)
                self.assertTrue(torch.equal(actual, expected))
                self.assertEqual(actual.storage_offset(), expected.storage_offset())
                self.assertEqual(actual.stride(), expected.stride())

        for axis in range(3):
            for index in (0, -1):
                with self.subTest(axis=axis, index=index):
                    expected = field.select(axis, index)
                    actual = _boundary_plane(field, axis, index)
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertEqual(actual.storage_offset(), expected.storage_offset())
                    self.assertEqual(actual.stride(), expected.stride())

    def test_fixed_state_rejects_buffer_replacement_or_conversion(self):
        simulation = _simulation(bloch=(0.07, 0.11, 0.13))
        addresses = {
            name: value.data_ptr() for name, value in simulation.state.named_buffers()
        }
        with self.assertRaisesRegex(ValueError, "assign=True.*fixed"):
            simulation.state.load_state_dict(simulation.state.state_dict(), assign=True)
        with self.assertRaisesRegex(TorchConfigurationError, "fixed device and dtype"):
            simulation.state.to(dtype=torch.float32)
        self.assertEqual(
            addresses,
            {
                name: value.data_ptr()
                for name, value in simulation.state.named_buffers()
            },
        )

    def test_batched_boundary_sync_preserves_order_and_skips_collapsed_axes(self):
        bloch = (0.07, 0.11, 0.13)
        rng = np.random.default_rng(37)
        families = (
            (("Ex", "Ey", "Ez"), True, "_sync_electric_boundaries"),
            (("Hx", "Hy", "Hz"), False, "_sync_magnetic_boundaries"),
        )

        cases = (
            ((3, 2, 2), None),
            ((3, 2, 0), None),
            ((3, 2, 2), 0),
            ((3, 2, 2), 1),
            ((3, 2, 2), 2),
        )
        for precision in ("float32", "float64"):
            for active_bloch in (None, bloch):
                for size, skip_axis in cases:
                    with self.subTest(
                        precision=precision,
                        bloch=active_bloch is not None,
                        size=size,
                        skip_axis=skip_axis,
                    ):
                        self._assert_boundary_sync_matches_scalar_reference(
                            rng=rng,
                            families=families,
                            precision=precision,
                            bloch=active_bloch,
                            size=size,
                            skip_axis=skip_axis,
                        )

    def _assert_boundary_sync_matches_scalar_reference(
        self, *, rng, families, precision, bloch, size, skip_axis
    ):
        simulation = _simulation(
            size=size,
            resolution=2,
            precision=precision,
            bloch=bloch,
        )
        values = {
            name: (
                rng.normal(size=simulation.plan.shapes[name])
                if bloch is None
                else rng.normal(size=simulation.plan.shapes[name])
                + 1j * rng.normal(size=simulation.plan.shapes[name])
            )
            for name in _COMPONENTS
        }
        simulation.load_host_fields(values)
        expected = {
            name: field.clone() for name, field in simulation.state.fields().items()
        }

        for names, high_from_low, method_name in families:
            for name in names:
                component_axis = ("x", "y", "z").index(name[1].lower())
                field = expected[name]
                for axis in range(3):
                    if (
                        axis == component_axis
                        or axis == skip_axis
                        or simulation.plan.shapes[name][axis] <= 1
                    ):
                        continue
                    destination_index = -1 if high_from_low else 0
                    source_index = 0 if high_from_low else -1
                    direction = 1 if high_from_low else -1
                    destination = field.select(axis, destination_index)
                    source = field.select(axis, source_index).clone()
                    if bloch is None:
                        destination.copy_(source)
                    else:
                        length = (
                            simulation.plan.shapes[name][axis] - 1
                        ) * simulation.plan.dr[axis]
                        angle = direction * bloch[axis] * length
                        cosine = float(np.cos(angle))
                        sine = float(np.sin(angle))
                        destination[..., 0].copy_(source[..., 0]).mul_(cosine)
                        destination[..., 0].add_(source[..., 1], alpha=-sine)
                        destination[..., 1].copy_(source[..., 0]).mul_(sine)
                        destination[..., 1].add_(source[..., 1], alpha=cosine)

            getattr(simulation, method_name)(skip_axis=skip_axis)
            stages = simulation._boundary_sync_stages(
                names,
                high_from_low=high_from_low,
                skip_axis=skip_axis,
            )
            self.assertIs(
                stages,
                simulation._boundary_sync_stages(
                    names,
                    high_from_low=high_from_low,
                    skip_axis=skip_axis,
                ),
            )
            expected_operations = sum(
                simulation.state.field(name).numel() > 0
                and axis != ("x", "y", "z").index(name[1].lower())
                and axis != skip_axis
                and simulation.plan.shapes[name][axis] > 1
                for name in names
                for axis in range(3)
            )
            self.assertEqual(
                sum(len(stage[0]) for stage in stages), expected_operations
            )
            for destinations, sources, phases in stages:
                if bloch is None:
                    self.assertIsNone(phases)
                else:
                    self.assertIsNotNone(phases)
                for destination, source in zip(destinations, sources):
                    self.assertNotEqual(destination.data_ptr(), source.data_ptr())

        for name, field in simulation.state.fields().items():
            torch.testing.assert_close(field, expected[name])

    def test_yee_shapes_cover_collapsed_1d_2d_3d(self):
        for size in ((8, 0, 0), (8, 6, 0), (6, 5, 4), (0, 0, 0)):
            with self.subTest(size=size):
                simulation = _simulation(size=size, resolution=2)
                nx, ny, nz = (int(value) for value in simulation.space.my_field_size)
                expected = {
                    "Ex": (nx, ny + 1, nz + 1),
                    "Ey": (nx + 1, ny, nz + 1),
                    "Ez": (nx + 1, ny + 1, nz),
                    "Hx": (nx, ny + 1, nz + 1),
                    "Hy": (nx + 1, ny, nz + 1),
                    "Hz": (nx + 1, ny + 1, nz),
                }
                self.assertEqual(
                    {
                        name: tuple(value.shape)
                        for name, value in simulation.state.fields().items()
                    },
                    expected,
                )

    def test_every_buffer_is_non_trainable_on_requested_device_and_dtype(self):
        simulation = _simulation(precision="float32", bloch=(0.07, 0.11, 0.13))
        self.assertEqual(list(simulation.state.parameters()), [])
        for name, value in simulation.state.named_buffers():
            self.assertEqual(value.device.type, "cpu", name)
            if name == "_dm2_status":
                expected_dtype = torch.int8
            elif name == "_dm2_iterations":
                expected_dtype = torch.int32
            elif (
                name in {"step_count", "_step_increment"}
                or "targets" in name
                or "tile_origins" in name
            ):
                expected_dtype = torch.int64
            elif any(
                marker in name
                for marker in (
                    "material_ids",
                    "underlying_ids",
                    "target_region_indices",
                    "region_keys",
                    "region_coefficient_indices",
                    "tile_region_indices",
                )
            ):
                expected_dtype = torch.int32
            elif "ownership" in name:
                expected_dtype = torch.int16
            else:
                expected_dtype = torch.float32
            self.assertEqual(value.dtype, expected_dtype, name)
            self.assertFalse(value.requires_grad, name)
        for field in simulation.state.fields().values():
            self.assertEqual(field.shape[-1], 2)

    def test_snapshot_checkpoint_aliasing_and_fixed_addresses(self):
        simulation = _simulation()
        rng = np.random.default_rng(13)
        values = {
            name: rng.normal(size=tuple(field.shape))
            for name, field in simulation.state.fields().items()
        }
        simulation.load_host_fields(values)
        addresses = simulation.buffer_addresses()
        snapshot = simulation.state.snapshot()
        checkpoint = simulation.state.checkpoint()
        self.assertNotIn("_step_increment", simulation.state.state_dict())
        self.assertIn("state._step_increment", addresses)
        simulation.advance(4)
        self.assertEqual(addresses, simulation.buffer_addresses())
        self.assertFalse(torch.equal(snapshot["Ex"], simulation.state.ex))
        simulation.state.load_checkpoint(checkpoint)
        self.assertEqual(addresses, simulation.buffer_addresses())
        torch.testing.assert_close(
            simulation.state.ex,
            torch.as_tensor(values["Ex"], dtype=torch.float64),
        )

    def test_plan_buffers_cannot_be_replaced(self):
        simulation = _simulation()
        with self.assertRaises(AttributeError):
            simulation.plan.inv_eps_ex = simulation.plan.inv_eps_ex.clone()
        with self.assertRaises(AttributeError):
            simulation.plan.dt = simulation.plan.dt

    def test_heterogeneous_dielectric_geometry_is_lowered_without_cell_objects(self):
        geometry = _geometry() + [
            gmes.Block(
                material=gmes.Dielectric(eps_inf=3.4, mu_inf=1.2),
                center=(0, 0, 0),
                size=(1, 1, 1),
            )
        ]
        simulation = TorchSimulation(
            space=gmes.Cartesian((2, 2, 2), 3),
            geometry=geometry,
            runtime=TorchRuntimeConfig(device="cpu", cpu_threads=2),
        )
        values = torch.unique(simulation.plan.inv_eps_ex)
        self.assertTrue(
            torch.any(torch.isclose(values, torch.tensor(1 / 3.4, dtype=values.dtype)))
        )
        self.assertTrue(
            torch.any(torch.isclose(values, torch.tensor(1 / 1.7, dtype=values.dtype)))
        )
        self.assertEqual(simulation.plan.material_ids_ex.dtype, torch.int32)


class TorchOracleTest(unittest.TestCase):
    def _compare(
        self,
        *,
        device,
        precision,
        compile_policy,
        bloch=None,
        size=(2, 2, 2),
        resolution=2,
    ):
        native, fields = _native_and_fields(
            size=size, resolution=resolution, bloch=bloch
        )
        simulation = _simulation(
            size=size,
            resolution=resolution,
            device=device,
            precision=precision,
            compile_policy=compile_policy,
            bloch=bloch,
        )
        simulation.load_host_fields(fields)
        _assert_matches_native(
            self,
            simulation,
            native,
            steps=3,
            precision=precision,
        )
        return simulation

    def test_cpu_eager_float64_matches_frozen_native_contract(self):
        self._compare(device="cpu", precision="float64", compile_policy="eager")

    def test_cpu_eager_float32_uses_separate_tolerance(self):
        self._compare(device="cpu", precision="float32", compile_policy="eager")

    def test_cpu_fullgraph_collapsed_z_matches_native(self):
        self._compare(
            device="cpu",
            precision="float64",
            compile_policy="compile",
            size=(4, 4, 0),
            resolution=4,
        )

    def test_cpu_fullgraph_paired_real_matches_complex_native(self):
        torch._dynamo.reset()
        simulation = self._compare(
            device="cpu",
            precision="float64",
            compile_policy="compile",
            bloch=(0.07, 0.11, 0.13),
        )
        graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        addresses = simulation.buffer_addresses()
        simulation.advance(4)
        self.assertEqual(
            graphs,
            torch._dynamo.utils.counters["stats"]["unique_graphs"],
        )
        self.assertEqual(addresses, simulation.buffer_addresses())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_fullgraph_collapsed_z_matches_native(self):
        self._compare(
            device="cuda:0",
            precision="float32",
            compile_policy="compile",
            size=(4, 4, 0),
            resolution=4,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_eager_paired_real_matches_complex_native(self):
        self._compare(
            device="cuda:0",
            precision="float64",
            compile_policy="eager",
            bloch=(0.07, 0.11, 0.13),
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_fullgraph_float32_has_stable_storage_and_allocation(self):
        simulation = self._compare(
            device="cuda:0",
            precision="float32",
            compile_policy="compile",
            bloch=(0.07, 0.11, 0.13),
        )
        torch.cuda.synchronize(simulation.device)
        addresses = simulation.buffer_addresses()
        allocated = torch.cuda.memory_allocated(simulation.device)
        simulation.advance(8)
        torch.cuda.synchronize(simulation.device)
        self.assertEqual(addresses, simulation.buffer_addresses())
        self.assertEqual(
            allocated,
            torch.cuda.memory_allocated(simulation.device),
        )


if __name__ == "__main__":
    unittest.main()
