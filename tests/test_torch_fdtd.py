"""Tests for the deliberately breaking Torch-only execution path."""

import hashlib
import json
import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import gmes
from benchmarks import issue123_completion
from gmes.torch_dm2 import (
    DM2_ITERATIONS_PER_CHUNK,
    DM2_MAX_ITERATIONS,
    DM2_PACKED_ITERATIONS_PER_CONDITION,
)
from gmes.torch_fdtd import (
    BOUNDARY_SYNC_REPRESENTATION,
    CUDA_GRAPH_EXECUTION_REPRESENTATION,
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
    _compile_fullgraph,
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
    execution_policy="auto",
    bloch=None,
    geometry=None,
):
    return TorchSimulation(
        space=gmes.Cartesian(size, resolution),
        geometry=_geometry() if geometry is None else geometry,
        runtime=TorchRuntimeConfig(
            device=device,
            precision=precision,
            compile_policy=compile_policy,
            execution_policy=execution_policy,
            cpu_threads=2,
        ),
        bloch=bloch,
    )


def _seeded_fields(simulation, *, complex_fields):
    rng = np.random.default_rng(1729)
    fields = {}
    for name, field in simulation.state.host_snapshot().items():
        values = rng.normal(size=field.shape) * 1e-3
        if complex_fields:
            values = values + 1j * rng.normal(size=field.shape) * 1e-3
        assert np.all(values != 0)
        fields[name] = values
    return fields


def _assert_matches_reference(test, simulation, reference, *, steps, precision):
    simulation.advance(steps)
    reference.advance(steps)
    actual = simulation.state.host_snapshot()
    expected = reference.state.host_snapshot()
    if np.iscomplexobj(expected["Ex"]):
        tolerance_name = "complex128" if precision == "float64" else "complex64"
    else:
        tolerance_name = precision
    tolerance = _TOLERANCES[tolerance_name]
    for name in _COMPONENTS:
        np.testing.assert_allclose(
            actual[name],
            expected[name],
            rtol=tolerance["rtol"],
            atol=tolerance["atol"],
            err_msg=name,
        )


def _sync_numpy_boundaries(fields, names, *, high_from_low, dr, bloch):
    """Apply the periodic/Bloch plane relation without Torch planner helpers."""
    for name in names:
        component_axis = "xyz".index(name[1].lower())
        field = fields[name]
        for axis in range(3):
            if axis == component_axis or field.shape[axis] <= 1:
                continue
            destination = -1 if high_from_low else 0
            source = 0 if high_from_low else -1
            direction = 1 if high_from_low else -1
            destination_slice = [slice(None)] * field.ndim
            source_slice = [slice(None)] * field.ndim
            destination_slice[axis] = destination
            source_slice[axis] = source
            phase = 1.0
            if bloch is not None:
                length = (field.shape[axis] - 1) * dr[axis]
                phase = np.exp(1j * direction * bloch[axis] * length)
            field[tuple(destination_slice)] = field[tuple(source_slice)].copy() * phase


def _numpy_dielectric_step(fields, *, resolution, bloch):
    """Advance one homogeneous Yee step from the published scalar equations."""
    eps_inf = 1.7
    mu_inf = 1.05
    courant_ratio = 0.99
    dr = (1.0 / resolution,) * 3
    dt_limit = np.sqrt(eps_inf * mu_inf) / np.sqrt(sum(spacing**-2 for spacing in dr))
    dt = courant_ratio * dt_limit
    dx, dy, dz = dr
    expected = {name: value.copy() for name, value in fields.items()}
    ex, ey, ez = (expected[name] for name in ("Ex", "Ey", "Ez"))
    hx, hy, hz = (expected[name] for name in ("Hx", "Hy", "Hz"))

    _sync_numpy_boundaries(
        expected,
        ("Hx", "Hy", "Hz"),
        high_from_low=False,
        dr=dr,
        bloch=bloch,
    )
    ex[:, :-1, :-1] += (dt / eps_inf) * (
        (hz[1:, 1:, :] - hz[1:, :-1, :]) / dy - (hy[1:, :, 1:] - hy[1:, :, :-1]) / dz
    )
    ey[:-1, :, :-1] += (dt / eps_inf) * (
        (hx[:, 1:, 1:] - hx[:, 1:, :-1]) / dz - (hz[1:, 1:, :] - hz[:-1, 1:, :]) / dx
    )
    ez[:-1, :-1, :] += (dt / eps_inf) * (
        (hy[1:, :, 1:] - hy[:-1, :, 1:]) / dx - (hx[:, 1:, 1:] - hx[:, :-1, 1:]) / dy
    )

    _sync_numpy_boundaries(
        expected,
        ("Ex", "Ey", "Ez"),
        high_from_low=True,
        dr=dr,
        bloch=bloch,
    )
    hx[:, 1:, 1:] += (dt / mu_inf) * (
        (ey[:-1, :, 1:] - ey[:-1, :, :-1]) / dz
        - (ez[:-1, 1:, :] - ez[:-1, :-1, :]) / dy
    )
    hy[1:, :, 1:] += (dt / mu_inf) * (
        (ez[1:, :-1, :] - ez[:-1, :-1, :]) / dx
        - (ex[:, :-1, 1:] - ex[:, :-1, :-1]) / dz
    )
    hz[1:, 1:, :] += (dt / mu_inf) * (
        (ex[:, 1:, :-1] - ex[:, :-1, :-1]) / dy
        - (ey[1:, :, :-1] - ey[:-1, :, :-1]) / dx
    )
    return expected, dt


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

    def test_cuda_graph_compile_explicitly_disables_inner_cudagraphs(self):
        cases = (
            ("default", {"triton.cudagraphs": False}),
            ("reduce-overhead", {"triton.cudagraphs": False}),
            (
                "max-autotune",
                {
                    "triton.cudagraphs": False,
                    "max_autotune": True,
                    "coordinate_descent_tuning": True,
                },
            ),
        )
        for compile_mode, expected_options in cases:
            with (
                self.subTest(compile_mode=compile_mode),
                mock.patch(
                    "gmes.torch_fdtd.torch.compile",
                    return_value=mock.sentinel.compiled,
                ) as compile_function,
            ):
                function = mock.sentinel.function
                result = _compile_fullgraph(
                    function,
                    TorchRuntimeConfig(
                        device="cuda:0",
                        compile_policy="compile",
                        compile_mode=compile_mode,
                        cpu_threads=1,
                    ),
                    torch.device("cuda:0"),
                    dynamic=False,
                    disable_cuda_graphs=True,
                )

            self.assertIs(result, mock.sentinel.compiled)
            compile_function.assert_called_once_with(
                function,
                fullgraph=True,
                dynamic=False,
                options=expected_options,
            )

        runtime = TorchRuntimeConfig(
            device="cuda:0",
            compile_policy="compile",
            compile_mode="reduce-overhead",
            cpu_threads=1,
        )
        with mock.patch(
            "gmes.torch_fdtd.torch.compile",
            return_value=mock.sentinel.compiled,
        ) as compile_function:
            _compile_fullgraph(
                mock.sentinel.function,
                runtime,
                torch.device("cuda:0"),
                dynamic=False,
            )
        compile_function.assert_called_once_with(
            mock.sentinel.function,
            fullgraph=True,
            dynamic=False,
            mode="reduce-overhead",
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
        self.assertEqual(TORCH_SOLVER_ABI, "torch-fdtd-regions-v15")
        self.assertEqual(issue123_completion.TORCH_SOLVER_ABI, TORCH_SOLVER_ABI)
        self.assertEqual(PACKED_DM2_REPRESENTATION, "single-carry-packed-loop-v2")
        self.assertEqual(
            CUDA_GRAPH_EXECUTION_REPRESENTATION,
            "external-no-inner-cudagraph-regions+dm2-raw-fixed-masked-v1",
        )
        self.assertEqual(DM2_PACKED_ITERATIONS_PER_CONDITION, 3)
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
        self.assertEqual(len(first._compile_cache_key_preimage), 31)
        self.assertEqual(
            first._compile_cache_key_preimage[6], LOCAL_COMPILED_REGION_TOPOLOGY
        )
        self.assertEqual(
            first._compile_cache_key_preimage[21][1:4],
            (
                DM2_MAX_ITERATIONS,
                DM2_ITERATIONS_PER_CHUNK,
                DM2_PACKED_ITERATIONS_PER_CONDITION,
            ),
        )
        self.assertEqual(
            first._compile_cache_key_preimage[21][-1],
            CUDA_GRAPH_EXECUTION_REPRESENTATION,
        )
        self.assertEqual(
            first.diagnostics()["cuda_graph_execution_representation"],
            CUDA_GRAPH_EXECUTION_REPRESENTATION,
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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_reduce_overhead_cuda_graph_uses_non_nested_half_steps(self):
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        poles = tuple(
            gmes.DrudePole(
                omega=0.6 + 0.1 * index,
                gamma=0.03 + 0.01 * index,
            )
            for index in range(4)
        )
        simulation = TorchSimulation(
            space=gmes.Cartesian((6, 5, 4), 3),
            geometry=[
                gmes.DefaultMedium(gmes.Dielectric(eps_inf=1.2)),
                gmes.Block(
                    gmes.Drude(eps_inf=1.2, dps=poles),
                    center=(0, 0, 0),
                    size=(6, 5, 4),
                ),
            ],
            runtime=TorchRuntimeConfig(
                device="cuda:0",
                precision="float32",
                compile_policy="compile",
                compile_mode="reduce-overhead",
                cpu_threads=1,
            ),
            bloch=(0.07, 0.11, 0.13),
        )
        rng = np.random.default_rng(123)
        simulation.load_host_fields(
            {
                name: rng.normal(size=tuple(field.shape)).astype(np.float32) * 1e-3
                for name, field in simulation.state.fields().items()
            }
        )
        simulation.advance(1)
        checkpoint = simulation.checkpoint()
        addresses = simulation.buffer_addresses()

        simulation.capture_cuda_graphs()
        self.assertEqual(
            sorted(simulation._cuda_graphs), ["electric_half", "magnetic_half"]
        )
        self.assertIsNot(
            simulation._electric_cuda_graph_half, simulation._electric_half
        )
        self.assertIsNot(
            simulation._magnetic_cuda_graph_half, simulation._magnetic_half
        )
        self.assertEqual(addresses, simulation.buffer_addresses())
        restored = simulation.checkpoint()
        for name, value in checkpoint["state"].items():
            self.assertTrue(torch.equal(restored["state"][name], value), name)

        simulation.advance(2)
        captured = simulation.state.checkpoint()
        simulation._cuda_graphs.clear()
        torch.cuda.synchronize(simulation.device)
        simulation.load_checkpoint(checkpoint).advance(2)
        normal = simulation.state.checkpoint()
        self.assertEqual(addresses, simulation.buffer_addresses())
        for name, value in normal.items():
            torch.testing.assert_close(captured[name], value, msg=name)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_graph_capture_failure_rolls_back_state_and_registry(self):
        simulation = TorchSimulation(
            space=gmes.Cartesian((2, 2, 2), 2),
            geometry=_geometry()
            + [
                gmes.Block(
                    gmes.Dm2(
                        eps_inf=1.4,
                        mu_inf=1.1,
                        omega=(0.7, 1.1),
                        n_atom=(0.2, 0.4),
                        rho30=-0.8,
                        gamma=0.15,
                        t1=2.5,
                        t2=1.7,
                        hbar=1.2,
                        rtol=1e-4,
                    ),
                    center=(0, 0, 0),
                    size=(1, 1, 1),
                )
            ],
            runtime=TorchRuntimeConfig(
                device="cuda:0",
                precision="float32",
                compile_policy="compile",
                cpu_threads=1,
            ),
            dt=0.025,
        )
        rng = np.random.default_rng(123)
        simulation.load_host_fields(
            {
                name: rng.normal(size=tuple(field.shape)).astype(np.float32) * 1e-3
                for name, field in simulation.state.fields().items()
            }
        )
        expected = simulation.checkpoint()
        addresses = simulation.buffer_addresses()
        normal_dm2_updates = simulation._dm2_updates
        original_load_checkpoint = simulation.load_checkpoint
        restore_calls = 0

        def fail_first_restore(checkpoint):
            nonlocal restore_calls
            restore_calls += 1
            if restore_calls == 1:
                raise RuntimeError("injected CUDA graph checkpoint restore failure")
            return original_load_checkpoint(checkpoint)

        with (
            mock.patch.object(
                simulation,
                "load_checkpoint",
                side_effect=fail_first_restore,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "injected CUDA graph checkpoint restore failure",
            ),
        ):
            simulation.capture_cuda_graphs()

        self.assertEqual(restore_calls, 2)
        self.assertEqual(simulation._cuda_graphs, {})
        self.assertIs(simulation._dm2_updates, normal_dm2_updates)
        self.assertEqual(simulation.buffer_addresses(), addresses)
        actual = simulation.checkpoint()
        self.assertEqual(actual["metadata"], expected["metadata"])
        self.assertEqual(actual["auxiliaries"], expected["auxiliaries"])
        for name, value in expected["state"].items():
            self.assertTrue(torch.equal(actual["state"][name], value), name)
        for name, value in expected["probes"].items():
            self.assertTrue(torch.equal(actual["probes"][name], value), name)


class TorchOracleTest(unittest.TestCase):
    def test_heterogeneous_dielectric_cells_match_scalar_inverse_equations(self):
        resolution = 4
        geometry = [
            gmes.DefaultMedium(gmes.Dielectric(eps_inf=1.7, mu_inf=1.05)),
            gmes.Block(
                gmes.Dielectric(eps_inf=3.4, mu_inf=1.05),
                center=(0, 0, 0),
                size=(1, 1, 1),
            ),
        ]
        simulation = _simulation(
            size=(2, 2, 2),
            resolution=resolution,
            geometry=geometry,
        )
        fields = _seeded_fields(simulation, complex_fields=False)
        simulation.load_host_fields(fields)
        spacing = 1.0 / resolution
        dt = 0.99 * np.sqrt(1.7 * 1.05) / np.sqrt(3 * resolution**2)
        targets = (
            ((3, 4, 4), 3.4),  # Ex at (-0.125, 0, 0), inside the block.
            ((0, 4, 4), 1.7),  # Ex at (-0.875, 0, 0), in the default medium.
        )
        expected = {}
        for target, eps_inf in targets:
            i, j, k = target
            curl = (
                fields["Hz"][i + 1, j + 1, k] - fields["Hz"][i + 1, j, k]
            ) / spacing - (
                fields["Hy"][i + 1, j, k + 1] - fields["Hy"][i + 1, j, k]
            ) / spacing
            expected[target] = fields["Ex"][target] + dt * curl / eps_inf

        simulation.advance(1)
        actual = simulation.state.host_snapshot()["Ex"]
        for target, eps_inf in targets:
            with self.subTest(target=target, eps_inf=eps_inf):
                self.assertAlmostEqual(actual[target], expected[target], places=14)
                wrong = fields["Ex"][target] + (
                    expected[target] - fields["Ex"][target]
                ) * eps_inf / (1.7 if eps_inf == 3.4 else 3.4)
                self.assertGreater(abs(expected[target] - wrong), 1e-10)

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
        reference = _simulation(
            size=size,
            resolution=resolution,
            device="cpu",
            precision="float64",
            compile_policy="eager",
            execution_policy="dense",
            bloch=bloch,
        )
        fields = _seeded_fields(reference, complex_fields=bloch is not None)
        simulation = _simulation(
            size=size,
            resolution=resolution,
            device=device,
            precision=precision,
            compile_policy=compile_policy,
            execution_policy="compact",
            bloch=bloch,
        )
        reference.load_host_fields(fields)
        simulation.load_host_fields(fields)
        expected, expected_dt = _numpy_dielectric_step(
            fields,
            resolution=resolution,
            bloch=bloch,
        )
        self.assertAlmostEqual(simulation.plan.dt, expected_dt, places=15)
        simulation.advance(1)
        actual = simulation.state.host_snapshot()
        if np.iscomplexobj(expected["Ex"]):
            tolerance_name = "complex128" if precision == "float64" else "complex64"
        else:
            tolerance_name = precision
        tolerance = _TOLERANCES[tolerance_name]
        for name in _COMPONENTS:
            np.testing.assert_allclose(
                actual[name],
                expected[name],
                rtol=tolerance["rtol"],
                atol=tolerance["atol"],
                err_msg=f"{name} independent NumPy Yee step",
            )
        reference.advance(1)
        _assert_matches_reference(
            self,
            simulation,
            reference,
            steps=2,
            precision=precision,
        )
        return simulation

    def test_cpu_eager_float64_matches_numpy_step_and_dense_reference(self):
        self._compare(device="cpu", precision="float64", compile_policy="eager")

    def test_cpu_eager_float32_matches_numpy_step_with_separate_tolerance(self):
        self._compare(device="cpu", precision="float32", compile_policy="eager")

    def test_cpu_fullgraph_collapsed_z_matches_numpy_step_and_dense_reference(self):
        self._compare(
            device="cpu",
            precision="float64",
            compile_policy="compile",
            size=(4, 4, 0),
            resolution=4,
        )

    def test_cpu_fullgraph_bloch_matches_numpy_step_and_dense_reference(self):
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
    def test_cuda_fullgraph_collapsed_z_matches_cpu_reference(self):
        self._compare(
            device="cuda:0",
            precision="float32",
            compile_policy="compile",
            size=(4, 4, 0),
            resolution=4,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_eager_paired_real_matches_cpu_reference(self):
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
