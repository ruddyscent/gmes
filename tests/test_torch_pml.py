"""Oracle and storage tests for Torch UPML/CPML tensor buckets."""

import json
import unittest
from pathlib import Path

import numpy as np
import torch

import gmes
from benchmarks.native_oracle import (
    _build_sources,
    _coverage_geometry,
    find_case,
    initial_field_values,
)

_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
_MANIFEST = json.loads(
    (
        Path(__file__).parents[1] / "benchmarks" / "native_oracle_workloads.json"
    ).read_text()
)
_TOLERANCES = _MANIFEST["tolerances"]["torch"]["pml"]


def _geometry(material_type):
    return [
        gmes.DefaultMedium(gmes.Dielectric(eps_inf=2.5, mu_inf=1.2)),
        gmes.Shell(material=material_type(), thickness=0.5),
    ]


def _native_and_torch(
    material_type,
    *,
    size=(2, 2, 2),
    resolution=2,
    precision="float64",
    compile_policy="eager",
    policy="auto",
    bloch=None,
    device="cpu",
    geometry=None,
):
    geometry = _geometry(material_type) if geometry is None else geometry
    native = gmes.FDTD(
        gmes.Cartesian(size, resolution),
        geometry,
        bloch=bloch,
        verbose=False,
    )
    native.init()
    simulation = gmes.TorchSimulation(
        space=gmes.Cartesian(size, resolution),
        geometry=geometry,
        runtime=gmes.TorchRuntimeConfig(
            device=device,
            precision=precision,
            compile_policy=compile_policy,
            execution_policy=policy,
            planner_tile_size=16,
            cpu_threads=2,
        ),
        bloch=bloch,
    )
    rng = np.random.default_rng(118)
    fields = {}
    for component, values in native.field.items():
        seeded = rng.normal(size=values.shape) * 1e-3
        if bloch is not None:
            seeded = seeded + 1j * rng.normal(size=values.shape) * 1e-3
        values[...] = seeded
        fields[component.__name__] = seeded.copy()
    simulation.load_host_fields(fields)
    return native, simulation


def _native_pml_updater(native, component, model):
    return next(
        updater
        for updater in native.pw_material[getattr(gmes, component)].values()
        if model.__name__ in type(updater).__name__
    )


def _assert_oracle(test, native, simulation, model, precision):
    complex_values = native.cmplx
    tolerance_name = (
        ("complex128" if precision == "float64" else "complex64")
        if complex_values
        else precision
    )
    tolerance = _TOLERANCES[tolerance_name]
    snapshot = simulation.state.host_snapshot()
    states = simulation.state.pml_state_snapshot()
    width = 1 if model is gmes.Upml else 2
    model_name = model.__name__.lower()
    for component in _COMPONENTS:
        np.testing.assert_allclose(
            snapshot[component],
            native.field[getattr(gmes, component)],
            rtol=tolerance["rtol"],
            atol=tolerance["atol"],
            err_msg=component,
        )
        component_plan = simulation.plan.components[component]
        bucket_index, bucket = next(
            (index, bucket)
            for index, bucket in enumerate(component_plan.buckets)
            if bucket.signature.model == model_name
        )
        updater = _native_pml_updater(native, component, model)
        indices = np.asarray(updater.oracle_indices(), dtype=np.int64).reshape(-1, 3)
        np.testing.assert_array_equal(
            bucket.targets,
            np.ravel_multi_index(indices.T, component_plan.shape),
        )
        expected_state = np.asarray(updater.oracle_state()).reshape(-1, width)
        np.testing.assert_allclose(
            states[f"pml_{component.lower()}_{bucket_index}_state"],
            expected_state,
            rtol=tolerance["rtol"],
            atol=tolerance["atol"],
            err_msg=f"{component} state",
        )


def _native_state_rows(native, component_name, updater_prefix, width):
    component = getattr(gmes, component_name)
    updater = next(
        updater
        for updater in native.pw_material[component].values()
        if type(updater).__name__.startswith(updater_prefix)
    )
    indices = np.asarray(updater.oracle_indices(), dtype=np.int64).reshape(-1, 3)
    targets = np.ravel_multi_index(indices.T, native.field[component].shape)
    state = np.asarray(updater.oracle_state(), dtype=np.complex128).reshape(
        len(targets), width
    )
    return targets, state


def _select_native_rows(native_targets, native_state, torch_targets):
    positions = {int(target): index for index, target in enumerate(native_targets)}
    return native_state[[positions[int(target)] for target in torch_targets]]


def _host_array(value):
    return value.detach().cpu().numpy()


def _torch_dispersive_rows(simulation, descriptor):
    prefix = descriptor.prefix
    poles = descriptor.pole_count
    points = descriptor.point_count
    names = []

    def persistent(suffix):
        name = f"{prefix}_{suffix}"
        names.append(name)
        return _host_array(getattr(simulation.state, name))

    if descriptor.model in {"drude", "lorentz"}:
        state = np.concatenate(
            (
                persistent("previous")[..., 0].T,
                persistent("current")[..., 0].T,
            ),
            axis=1,
        )
        width = 2 * poles
    elif descriptor.model == "dcp-ade":
        state = np.concatenate(
            (
                persistent("field_old"),
                persistent("pole_old")[..., 0].T,
                persistent("pole_now")[..., 0].T,
                persistent("point_old")[..., 0].T,
                persistent("point_now")[..., 0].T,
            ),
            axis=1,
        )
        width = 1 + 2 * poles + 2 * points
    else:
        pole_state = persistent("pole_state")[..., 0].T
        point_values = persistent("point_state")
        point_state = (point_values[..., 0] + 1j * point_values[..., 1])[..., 0].T
        # Native real-field PLRC/RC updaters retain structurally zero state for
        # the absent imaginary field channel; Torch deliberately omits it.
        state = np.concatenate(
            (
                pole_state,
                np.zeros_like(pole_state),
                point_state,
                np.zeros_like(point_state),
            ),
            axis=1,
        )
        width = 2 * poles + 2 * points
    return state, width, set(names)


class TorchPmlOracleTest(unittest.TestCase):
    def test_nonzero_fields_and_state_match_at_reference_steps(self):
        for model in (gmes.Upml, gmes.Cpml):
            with self.subTest(model=model.__name__):
                native, simulation = _native_and_torch(model)
                completed = 0
                for target in (1, 2, 5, 20, 100):
                    delta = target - completed
                    simulation.advance(delta)
                    for _ in range(delta):
                        native.step()
                    _assert_oracle(self, native, simulation, model, "float64")
                    completed = target

    def test_float32_and_paired_real_bloch_use_model_tolerances(self):
        cases = (
            ("float32", None),
            ("float64", (0.07, 0.11, 0.13)),
            ("float32", (0.07, 0.11, 0.13)),
        )
        for model in (gmes.Upml, gmes.Cpml):
            for precision, bloch in cases:
                with self.subTest(
                    model=model.__name__, precision=precision, bloch=bool(bloch)
                ):
                    native, simulation = _native_and_torch(
                        model, precision=precision, bloch=bloch
                    )
                    simulation.advance(5)
                    for _ in range(5):
                        native.step()
                    _assert_oracle(self, native, simulation, model, precision)

    def test_collapsed_1d_2d_and_3d_modes_match_native(self):
        for size in ((4, 0, 0), (4, 3, 0), (3, 3, 2)):
            for model in (gmes.Upml, gmes.Cpml):
                with self.subTest(size=size, model=model.__name__):
                    native, simulation = _native_and_torch(model, size=size)
                    simulation.advance(5)
                    for _ in range(5):
                        native.step()
                    _assert_oracle(self, native, simulation, model, "float64")

    def test_shell_corners_use_mixed_underlying_media(self):
        geometry = [
            gmes.DefaultMedium(gmes.Dielectric(eps_inf=2.5, mu_inf=1.2)),
            gmes.Block(
                gmes.Dielectric(eps_inf=3.2, mu_inf=1.15),
                center=(0, 0, 0),
                size=(1.2, 1.2, 1.2),
            ),
            gmes.Shell(gmes.Cpml(), thickness=0.75),
        ]
        native, simulation = _native_and_torch(
            gmes.Cpml,
            size=(2, 2, 2),
            resolution=4,
            geometry=geometry,
        )
        simulation.advance(5)
        for _ in range(5):
            native.step()
        _assert_oracle(self, native, simulation, gmes.Cpml, "float64")
        for component in simulation.plan.components.values():
            bucket = next(
                bucket
                for bucket in component.buckets
                if bucket.signature.model == "cpml"
            )
            self.assertEqual(set(bucket.region_keys[:, 1]), {0, 1})
            self.assertEqual(len(np.unique(bucket.cell_coefficients[:, 0])), 2)

    def test_forced_execution_policies_are_oracle_equivalent(self):
        for policy in ("dense", "compact", "tiled"):
            with self.subTest(policy=policy):
                native, simulation = _native_and_torch(gmes.Cpml, policy=policy)
                simulation.advance(5)
                for _ in range(5):
                    native.step()
                _assert_oracle(self, native, simulation, gmes.Cpml, "float64")
                self.assertEqual(
                    {
                        bucket.selected_policy
                        for component in simulation.plan.components.values()
                        for bucket in component.buckets
                    },
                    {policy},
                )

    def test_compiled_z_collapsed_specialization_matches_native(self):
        for model in (gmes.Upml, gmes.Cpml):
            with self.subTest(model=model.__name__):
                torch._dynamo.reset()
                native, simulation = _native_and_torch(
                    model,
                    size=(4, 4, 0),
                    resolution=3,
                    compile_policy="compile",
                )
                self.assertEqual(
                    simulation.diagnostics()["phase_specialization"],
                    "z-collapsed-v1",
                )
                simulation.advance(5)
                for _ in range(5):
                    native.step()
                _assert_oracle(self, native, simulation, model, "float64")

    def test_compiled_cpu_crossover_manifest_matches_complete_native_state(self):
        torch._dynamo.reset()
        spec = find_case(_MANIFEST, "cpu-crossover-2d")
        reference = _MANIFEST["reference"]

        def geometry():
            return _coverage_geometry(spec, gmes)

        def sources():
            return _build_sources(spec, gmes)

        native = gmes.FDTD(
            gmes.Cartesian(tuple(spec["size"]), spec["resolution"]),
            geometry(),
            sources(),
            verbose=False,
        )
        native.init()
        simulation = gmes.TorchSimulation(
            space=gmes.Cartesian(tuple(spec["size"]), spec["resolution"]),
            geometry=geometry(),
            sources=sources(),
            dt=native.time_step.dt,
            runtime=gmes.TorchRuntimeConfig(
                device="cpu",
                precision="float64",
                compile_policy="compile",
                cpu_threads=2,
            ),
        )
        fields = initial_field_values(
            simulation.plan.shapes,
            reference["seed"],
            reference["field_scale"],
        )
        for component, native_field in native.field.items():
            native_field[...] = fields[component.__name__]
        simulation.load_host_fields(fields)

        simulation.advance(5)
        for _ in range(5):
            native.step()

        for bucket_state in simulation.state.dm2_buckets:
            self.assertGreater(
                int(torch.count_nonzero(bucket_state.u)),
                0,
                bucket_state.metadata.component,
            )

        diagnostics = simulation.diagnostics()
        self.assertEqual(diagnostics["phase_specialization"], "z-collapsed-v1")
        self.assertEqual(diagnostics["sources"]["target_rows"], 1)
        self.assertEqual(
            set(diagnostics["dispersive"]["models"]),
            {"drude", "lorentz", "dcp-ade", "dcp-plrc", "dcp-rc"},
        )
        self.assertEqual(len(diagnostics["dm2"]), 3)

        tolerance = _MANIFEST["tolerances"]["torch"]["mixed"]["float64"]
        persistent = simulation.state.state_dict()
        compared_dm2 = set()
        for index, bucket_state in enumerate(simulation.state.dm2_buckets):
            metadata = bucket_state.metadata
            native_targets, native_values = _native_state_rows(
                native,
                metadata.component,
                "Dm2",
                metadata.transition_count * 3,
            )
            native_values = native_values.real.reshape(
                len(native_targets), metadata.transition_count, 3
            )
            torch_targets = _host_array(
                getattr(simulation.plan, f"{metadata.prefix}_targets")
            )
            self.assertEqual(set(torch_targets), set(native_targets))
            np.testing.assert_allclose(
                _host_array(bucket_state.u).transpose(1, 2, 0),
                _select_native_rows(
                    native_targets, native_values, torch_targets
                ),
                rtol=tolerance["rtol"],
                atol=tolerance["atol"],
                err_msg=f"{metadata.component} DM2 state",
            )
            compared_dm2.add(f"dm2_buckets.{index}.u")
        expected_dm2 = {
            name for name in persistent if name.startswith("dm2_buckets.")
        }
        self.assertEqual(compared_dm2, expected_dm2)
        pml_state = simulation.state.pml_state_snapshot()
        compared_pml = set()
        for component_name in _COMPONENTS:
            component_plan = simulation.plan.components[component_name]
            bucket_index, bucket = next(
                (index, bucket)
                for index, bucket in enumerate(component_plan.buckets)
                if bucket.signature.model == "cpml"
            )
            native_targets, native_values = _native_state_rows(
                native, component_name, "Cpml", 2
            )
            torch_targets = np.asarray(bucket.targets, dtype=np.int64)
            self.assertEqual(set(torch_targets), set(native_targets))
            state_name = f"pml_{component_name.lower()}_{bucket_index}_state"
            np.testing.assert_allclose(
                pml_state[state_name],
                _select_native_rows(
                    native_targets, native_values, torch_targets
                ),
                rtol=tolerance["rtol"],
                atol=tolerance["atol"],
                err_msg=f"{component_name} CPML state",
            )
            compared_pml.add(state_name)
        expected_pml = {
            name for name in persistent if name.startswith("pml_")
        }
        self.assertEqual(compared_pml, expected_pml)

        updater_prefixes = {
            "drude": "Drude",
            "lorentz": "Lorentz",
            "dcp-ade": "DcpAde",
            "dcp-plrc": "DcpPlrc",
            "dcp-rc": "DcpPlrc",
        }
        compared_dispersive = set()
        covered_targets = {}
        native_target_sets = {}
        for descriptor in simulation.plan.dispersive_buckets:
            actual, width, names = _torch_dispersive_rows(simulation, descriptor)
            updater_prefix = updater_prefixes[descriptor.model]
            native_targets, native_values = _native_state_rows(
                native,
                descriptor.component,
                updater_prefix,
                width,
            )
            torch_targets = _host_array(
                getattr(simulation.plan, f"{descriptor.prefix}_targets")
            )
            np.testing.assert_allclose(
                actual,
                _select_native_rows(
                    native_targets, native_values, torch_targets
                ),
                rtol=tolerance["rtol"],
                atol=tolerance["atol"],
                err_msg=f"{descriptor.component} {descriptor.model} state",
            )
            key = (descriptor.component, updater_prefix)
            covered_targets.setdefault(key, set()).update(map(int, torch_targets))
            native_target_sets.setdefault(key, set()).update(map(int, native_targets))
            compared_dispersive.update(names)
        self.assertEqual(covered_targets, native_target_sets)
        expected_dispersive = {
            name for name in persistent if name.startswith("bucket_")
        }
        self.assertEqual(compared_dispersive, expected_dispersive)

        actual_fields = simulation.state.host_snapshot()
        for component_name in _COMPONENTS:
            np.testing.assert_allclose(
                actual_fields[component_name],
                native.field[getattr(gmes, component_name)],
                rtol=tolerance["rtol"],
                atol=tolerance["atol"],
                err_msg=f"{component_name} field",
            )


        for names in (expected_pml, expected_dispersive, expected_dm2):
            self.assertTrue(names)
            self.assertTrue(any(np.any(_host_array(persistent[name])) for name in names))
        self.assertEqual(int(simulation.state.step_count), 5)
        self.assertAlmostEqual(
            float(simulation.state.source_time), native.time_step.t, places=14
        )

    def test_long_run_absorbs_seeded_energy_like_native(self):
        native, simulation = _native_and_torch(gmes.Cpml, size=(4, 4, 0), resolution=3)
        initial_energy = sum(
            float(np.square(np.abs(values)).sum())
            for values in simulation.state.host_snapshot().values()
        )
        simulation.advance(200)
        for _ in range(200):
            native.step()
        _assert_oracle(self, native, simulation, gmes.Cpml, "float64")
        final_energy = sum(
            float(np.square(np.abs(values)).sum())
            for values in simulation.state.host_snapshot().values()
        )
        self.assertLess(final_energy, initial_energy)


class TorchPmlStorageTest(unittest.TestCase):
    def test_state_is_active_only_contiguous_and_uses_underlying_medium(self):
        simulation = gmes.TorchSimulation(
            space=gmes.Cartesian((4, 4, 4), 3),
            geometry=_geometry(gmes.Cpml),
            runtime=gmes.TorchRuntimeConfig(device="cpu", cpu_threads=2),
        )
        expected_values = 0
        full_grid_values = 0
        for component_name, component in simulation.plan.components.items():
            full_grid_values += int(np.prod(component.shape)) * 2
            for index, bucket in enumerate(component.buckets):
                if bucket.signature.model != "cpml":
                    continue
                expected_values += bucket.target_count * bucket.state_width
                state = getattr(
                    simulation.state,
                    f"pml_{component_name.lower()}_{index}_state",
                )
                self.assertTrue(state.is_contiguous())
                self.assertEqual(state.numel(), bucket.target_count * 2)
                self.assertTrue(np.all(bucket.region_keys[:, 1] >= 0))
                np.testing.assert_allclose(
                    bucket.cell_coefficients[:, 0],
                    1.0 / (2.5 if component_name in ("Ex", "Ey", "Ez") else 1.2),
                )
        diagnostics = simulation.diagnostics()["pml"]
        self.assertEqual(
            diagnostics["state_bytes"],
            expected_values * simulation.state.ex.element_size(),
        )
        self.assertLess(expected_values, full_grid_values)
        self.assertEqual(diagnostics["launches_per_step"], 6)

    def test_coordinate_coefficients_match_material_contract(self):
        component_types = {name: getattr(gmes, name) for name in _COMPONENTS}
        axes_by_component = {
            "Ex": (1, 2, 0),
            "Ey": (2, 0, 1),
            "Ez": (0, 1, 2),
            "Hx": (1, 2, 0),
            "Hy": (2, 0, 1),
            "Hz": (0, 1, 2),
        }
        for model in (gmes.Upml, gmes.Cpml):
            simulation = gmes.TorchSimulation(
                space=gmes.Cartesian((2, 2, 2), 2),
                geometry=_geometry(model),
                runtime=gmes.TorchRuntimeConfig(device="cpu", cpu_threads=2),
            )
            material = simulation.geometry[1].material
            for component_name, component in simulation.plan.components.items():
                bucket = next(
                    bucket
                    for bucket in component.buckets
                    if bucket.signature.model == model.__name__.lower()
                )
                indices = np.unravel_index(bucket.targets, component.shape)
                coordinate_axes = simulation.space.component_coordinate_axes(
                    component_types[component_name], component.shape
                )
                coordinates = np.column_stack(
                    [coordinate_axes[axis][indices[axis]] for axis in range(3)]
                )
                first, second, field = axes_by_component[component_name]
                base = bucket.cell_coefficients[:, 0]
                if model is gmes.Upml:
                    expected = np.column_stack(
                        (
                            base,
                            [
                                material.c1(value, first)
                                for value in coordinates[:, first]
                            ],
                            [
                                material.c2(value, first)
                                for value in coordinates[:, first]
                            ],
                            [
                                material.c3(value, second)
                                for value in coordinates[:, second]
                            ],
                            [
                                material.c4(value, second)
                                for value in coordinates[:, second]
                            ],
                            [
                                material.c5(value, field)
                                for value in coordinates[:, field]
                            ],
                            [
                                material.c6(value, field)
                                for value in coordinates[:, field]
                            ],
                        )
                    )
                else:
                    expected = np.column_stack(
                        (
                            base,
                            [
                                material.b(value, first)
                                for value in coordinates[:, first]
                            ],
                            [
                                material.c(value, first)
                                for value in coordinates[:, first]
                            ],
                            [
                                material.kappa(value, first)
                                for value in coordinates[:, first]
                            ],
                            [
                                material.b(value, second)
                                for value in coordinates[:, second]
                            ],
                            [
                                material.c(value, second)
                                for value in coordinates[:, second]
                            ],
                            [
                                material.kappa(value, second)
                                for value in coordinates[:, second]
                            ],
                        )
                    )
                np.testing.assert_allclose(
                    bucket.cell_coefficients,
                    expected,
                    rtol=2e-15,
                    atol=2e-15,
                    err_msg=f"{model.__name__} {component_name}",
                )

    def test_warm_execution_and_checkpoint_keep_fixed_device_storage(self):
        _native, simulation = _native_and_torch(gmes.Upml)
        simulation.advance(2)
        addresses = simulation.buffer_addresses()
        checkpoint = simulation.state.checkpoint()
        before = simulation.state.pml_state_snapshot(numpy=False)
        simulation.advance(8)
        self.assertEqual(addresses, simulation.buffer_addresses())
        simulation.state.load_checkpoint(checkpoint)
        self.assertEqual(addresses, simulation.buffer_addresses())
        after = simulation.state.pml_state_snapshot(numpy=False)
        for name in before:
            torch.testing.assert_close(before[name], after[name])

    def test_cpu_fullgraph_has_no_graph_break_or_storage_change(self):
        torch._dynamo.reset()
        native, simulation = _native_and_torch(gmes.Cpml, compile_policy="compile")
        simulation.advance(2)
        for _ in range(2):
            native.step()
        _assert_oracle(self, native, simulation, gmes.Cpml, "float64")
        self.assertEqual(torch._dynamo.utils.counters["graph_break"], {})
        addresses = simulation.buffer_addresses()
        simulation.advance(3)
        self.assertEqual(addresses, simulation.buffer_addresses())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_eager_and_fullgraph_have_stable_memory(self):
        for compile_policy in ("eager", "compile"):
            for model in (gmes.Upml, gmes.Cpml):
                with self.subTest(compile_policy=compile_policy, model=model.__name__):
                    native, simulation = _native_and_torch(
                        model,
                        precision="float32",
                        compile_policy=compile_policy,
                        device="cuda:0",
                    )
                    simulation.advance(2)
                    torch.cuda.synchronize(simulation.device)
                    allocated = torch.cuda.memory_allocated(simulation.device)
                    addresses = simulation.buffer_addresses()
                    simulation.advance(5)
                    for _ in range(7):
                        native.step()
                    torch.cuda.synchronize(simulation.device)
                    _assert_oracle(self, native, simulation, model, "float32")
                    self.assertEqual(addresses, simulation.buffer_addresses())
                    self.assertEqual(
                        allocated,
                        torch.cuda.memory_allocated(simulation.device),
                    )


if __name__ == "__main__":
    unittest.main()
