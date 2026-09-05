"""Oracle and storage tests for Torch dispersive tensor buckets."""

import json
import os
import unittest
from pathlib import Path

import numpy as np
import torch

import gmes
from gmes import torch_dispersive

_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
_CAPTURE_STEPS = (1, 2, 5, 20, 100)
_TOLERANCES = json.loads(
    (
        Path(__file__).parents[1] / "benchmarks" / "native_oracle_workloads.json"
    ).read_text()
)["tolerances"]["torch"]


def _poles(width):
    return tuple(
        gmes.DrudePole(omega=0.6 + 0.1 * index, gamma=0.03 + 0.01 * index)
        for index in range(width)
    )


def _lorentz_poles(width):
    return tuple(
        gmes.LorentzPole(
            amp=0.05 + 0.01 * index,
            omega=0.8 + 0.1 * index,
            gamma=0.03 + 0.01 * index,
        )
        for index in range(width)
    )


def _points(width=2):
    return tuple(
        gmes.CriticalPoint(
            amp=0.04 - 0.01 * index,
            phi=0.2 - 0.3 * index,
            omega=0.9 + 0.2 * index,
            gamma=0.03 + 0.01 * index,
        )
        for index in range(width)
    )


def _material(model, *, poles=1, points=2):
    factories = {
        "drude": lambda: gmes.Drude(eps_inf=1.2, sigma=0.01, dps=_poles(poles)),
        "lorentz": lambda: gmes.Lorentz(
            eps_inf=1.2, sigma=0.01, lps=_lorentz_poles(poles)
        ),
        "dcp-ade": lambda: gmes.DcpAde(
            eps_inf=1.2,
            sigma=0.01,
            dps=_poles(poles),
            cps=_points(points),
        ),
        "dcp-plrc": lambda: gmes.DcpPlrc(
            eps_inf=1.2,
            sigma=0.01,
            dps=_poles(poles),
            cps=_points(points),
        ),
        "dcp-rc": lambda: gmes.DcpRc(
            eps_inf=1.2,
            sigma=0.01,
            dps=_poles(poles),
            cps=_points(points),
        ),
    }
    return factories[model]()


def _geometry(model, *, poles=1, points=2):
    return [
        gmes.DefaultMedium(gmes.Dielectric(eps_inf=1.7, mu_inf=1.05)),
        gmes.Block(
            _material(model, poles=poles, points=points),
            center=(0, 0, 0),
            size=(1.4, 1.4, 1.4),
        ),
    ]


def _mixed_geometry():
    families = ("drude", "lorentz", "dcp-ade", "dcp-plrc", "dcp-rc")
    geometry = [gmes.DefaultMedium(gmes.Dielectric(eps_inf=1.7, mu_inf=1.05))]
    geometry.extend(
        gmes.Block(
            _material(model),
            center=(-3.2 + 1.6 * index, 0, 0),
            size=(1.2, 1.4, 1.4),
        )
        for index, model in enumerate(families)
    )
    return geometry


def _mixed_pml_geometry():
    families = ("drude", "lorentz", "dcp-ade", "dcp-plrc", "dcp-rc")
    geometry = [gmes.DefaultMedium(gmes.Dielectric(eps_inf=1.7, mu_inf=1.05))]
    geometry.extend(
        gmes.Block(
            _material(model),
            center=(-3.2 + 1.6 * index, 0, 0),
            size=(1.4, 5.8, 1),
        )
        for index, model in enumerate(families)
    )
    geometry.append(gmes.Shell(gmes.Cpml(), thickness=0.5))
    return geometry


def _seed_reference(reference, *, complex_fields):
    rng = np.random.default_rng(119)
    fields = {}
    for name, field in reference.state.host_snapshot().items():
        values = rng.normal(size=field.shape) * 1e-3
        if complex_fields:
            values = values + 1j * rng.normal(size=field.shape) * 1e-3
        fields[name] = values
    reference.load_host_fields(fields)
    return fields


def _reference_and_torch(
    model,
    *,
    bloch=None,
    precision="float64",
    policy="auto",
    compile_policy="eager",
    size=(2, 2, 2),
    resolution=3,
    poles=1,
    points=2,
    device="cpu",
):
    reference = gmes.TorchSimulation(
        space=gmes.Cartesian(size, resolution),
        geometry=_geometry(model, poles=poles, points=points),
        bloch=bloch,
        runtime=gmes.TorchRuntimeConfig(
            device="cpu",
            precision="float64",
            cpu_threads=2,
            execution_policy="dense",
        ),
    )
    simulation = gmes.TorchSimulation(
        space=gmes.Cartesian(size, resolution),
        geometry=_geometry(model, poles=poles, points=points),
        bloch=bloch,
        runtime=gmes.TorchRuntimeConfig(
            device=device,
            precision=precision,
            cpu_threads=2,
            execution_policy=policy,
            compile_policy=compile_policy,
        ),
    )
    fields = _seed_reference(reference, complex_fields=bloch is not None)
    simulation.load_host_fields(fields)
    return reference, simulation


def _assert_fields(test, reference, simulation, *, model, complex_fields):
    if complex_fields:
        tolerance_name = (
            "complex64" if simulation.dtype == torch.float32 else "complex128"
        )
    else:
        tolerance_name = "float32" if simulation.dtype == torch.float32 else "float64"
    tolerance = _TOLERANCES[model][tolerance_name]
    actual = simulation.state.host_snapshot()
    expected = reference.state.host_snapshot()
    for name, field in expected.items():
        test.assertTrue(np.all(np.isfinite(actual[name])))
        np.testing.assert_allclose(
            actual[name],
            field,
            rtol=tolerance["rtol"],
            atol=tolerance["atol"],
            err_msg=f"{model}:{name}",
        )


class DispersiveOracleTest(unittest.TestCase):
    def test_scalar_recurrences_match_explicit_independent_equations(self):
        for model, coefficients in (
            ("drude", (0.2, 0.7, -0.3)),
            ("lorentz", (-0.4, 0.6, 0.25)),
        ):
            with self.subTest(model=model):
                a = np.asarray(coefficients, dtype=np.float64).reshape(3, 1, 1)
                c = np.asarray((0.4, -0.2, 0.8), dtype=np.float64).reshape(3, 1)
                previous = np.asarray([[[0.3]]], dtype=np.float64)
                current = np.asarray([[[-0.1]]], dtype=np.float64)
                field_now = np.asarray([[0.5]], dtype=np.float64)
                curl = np.asarray([[-0.25]], dtype=np.float64)
                pole_work = (
                    a[0, :, :, None] * previous
                    + a[1, :, :, None] * current
                    + a[2, :, :, None] * field_now[None, :, :]
                )
                response = np.sum(pole_work - current, axis=0)
                expected_field = (
                    c[0, :, None] * curl
                    + c[1, :, None] * response
                    + c[2, :, None] * field_now
                )
                tensors = [
                    torch.from_numpy(value.copy())
                    for value in (a, c, previous, current)
                ]
                work = torch.zeros_like(tensors[2])
                delta = torch.zeros_like(tensors[2])
                actual_field = torch.zeros_like(torch.from_numpy(field_now))
                actual_response = torch.zeros_like(actual_field)
                torch_dispersive._update_two_level_tensors(
                    *tensors,
                    work,
                    delta,
                    torch.from_numpy(field_now),
                    actual_field,
                    torch.from_numpy(curl),
                    actual_response,
                )
                np.testing.assert_allclose(actual_field.numpy(), expected_field)
                np.testing.assert_allclose(tensors[2].numpy(), current)
                np.testing.assert_allclose(tensors[3].numpy(), pole_work)

        a = np.asarray((0.2, 0.7, -0.3), dtype=np.float64).reshape(3, 1, 1)
        b = np.asarray((0.1, 0.6, -0.2, 0.3, 0.4), dtype=np.float64).reshape(5, 1, 1)
        c = np.asarray((0.4, -0.2, 0.15, 0.8), dtype=np.float64).reshape(4, 1)
        field_old = np.asarray([[0.2]], dtype=np.float64)
        field_now = np.asarray([[0.5]], dtype=np.float64)
        pole_old = np.asarray([[[0.3]]], dtype=np.float64)
        pole_now = np.asarray([[[-0.1]]], dtype=np.float64)
        point_old = np.asarray([[[0.4]]], dtype=np.float64)
        point_now = np.asarray([[[-0.2]]], dtype=np.float64)
        curl = np.asarray([[-0.25]], dtype=np.float64)
        response = np.sum(
            pole_now - a[1, :, :, None] * pole_now - a[0, :, :, None] * pole_old,
            axis=0,
        )
        response += np.sum(
            point_now - b[1, :, :, None] * point_now - b[0, :, :, None] * point_old,
            axis=0,
        )
        expected_field = (
            c[0, :, None] * curl
            + c[1, :, None] * response
            + c[2, :, None] * field_old
            + c[3, :, None] * field_now
        )
        field_combo = field_old + 2.0 * field_now + expected_field
        expected_pole = (
            a[0, :, :, None] * pole_old
            + a[1, :, :, None] * pole_now
            + a[2, :, :, None] * field_combo[None, :, :]
        )
        expected_point = (
            b[0, :, :, None] * point_old
            + b[1, :, :, None] * point_now
            + b[2, :, :, None] * field_old[None, :, :]
            + b[3, :, :, None] * field_now[None, :, :]
            + b[4, :, :, None] * expected_field[None, :, :]
        )
        simulation = _reference_and_torch("dcp-ade", points=1)[0]
        descriptor = next(
            item
            for item in simulation.plan.dispersive_buckets
            if item.component == "Ex"
        )
        prefix = descriptor.prefix
        for suffix, value in (
            ("a", a),
            ("b", b),
            ("c", c),
        ):
            getattr(simulation.plan, f"{prefix}_{suffix}").copy_(
                torch.from_numpy(value)
            )
        for suffix, value in (
            ("field_old", field_old),
            ("pole_old", pole_old),
            ("pole_now", pole_now),
            ("point_old", point_old),
            ("point_now", point_now),
        ):
            target = getattr(simulation.state, f"{prefix}_{suffix}")
            target.copy_(torch.from_numpy(np.broadcast_to(value, target.shape).copy()))
        torch_dispersive._update_dcp_ade(
            simulation.plan,
            simulation.state,
            descriptor,
            torch.from_numpy(
                np.broadcast_to(field_now, (descriptor.target_count, 1)).copy()
            ),
            getattr(simulation.state, f"{prefix}_field_new"),
            torch.from_numpy(
                np.broadcast_to(curl, (descriptor.target_count, 1)).copy()
            ),
            getattr(simulation.state, f"{prefix}_response"),
            getattr(simulation.state, f"{prefix}_gather_a"),
        )
        np.testing.assert_allclose(
            getattr(simulation.state, f"{prefix}_field_new").numpy(),
            np.broadcast_to(expected_field, (descriptor.target_count, 1)),
        )
        np.testing.assert_allclose(
            getattr(simulation.state, f"{prefix}_pole_now").numpy(),
            np.broadcast_to(
                expected_pole,
                getattr(simulation.state, f"{prefix}_pole_now").shape,
            ),
        )
        np.testing.assert_allclose(
            getattr(simulation.state, f"{prefix}_point_now").numpy(),
            np.broadcast_to(
                expected_point,
                getattr(simulation.state, f"{prefix}_point_now").shape,
            ),
        )

        for model, scale in (("dcp-plrc", 1.0), ("dcp-rc", -0.75)):
            with self.subTest(model=model):
                a = np.asarray((0.2, -0.1, 0.7), dtype=np.float64).reshape(3, 1, 1)
                b = scale * np.asarray(
                    ((0.3, -0.2), (-0.1, 0.4), (0.6, 0.25)),
                    dtype=np.float64,
                ).reshape(3, 1, 1, 2)
                c = np.asarray((0.4, 0.8, -0.2), dtype=np.float64).reshape(3, 1)
                pole_state = np.asarray([[[0.15]]], dtype=np.float64)
                point_state = np.asarray([[[[0.25, -0.35]]]], dtype=np.float64)
                field_now = np.asarray([[0.5]], dtype=np.float64)
                curl = np.asarray([[-0.25]], dtype=np.float64)
                response = np.sum(pole_state, axis=0)
                response += np.sum(point_state[..., 0], axis=0)
                expected_field = (
                    c[0, :, None] * curl
                    + c[1, :, None] * field_now
                    + c[2, :, None] * response
                )
                expected_pole = (
                    a[0, :, :, None] * expected_field[None, :, :]
                    + a[1, :, :, None] * field_now[None, :, :]
                    + a[2, :, :, None] * pole_state
                )
                expected_point = np.empty_like(point_state)
                expected_point[..., 0] = (
                    b[0, ..., 0, None] * expected_field[None, :, :]
                    + b[1, ..., 0, None] * field_now[None, :, :]
                    + b[2, ..., 0, None] * point_state[..., 0]
                    - b[2, ..., 1, None] * point_state[..., 1]
                )
                expected_point[..., 1] = (
                    b[0, ..., 1, None] * expected_field[None, :, :]
                    + b[1, ..., 1, None] * field_now[None, :, :]
                    + b[2, ..., 0, None] * point_state[..., 1]
                    + b[2, ..., 1, None] * point_state[..., 0]
                )
                tensors = [
                    torch.from_numpy(value.copy())
                    for value in (a, b, c, pole_state, point_state)
                ]
                pole_work = torch.zeros_like(tensors[3])
                point_work = torch.zeros_like(tensors[4])
                actual_field = torch.zeros_like(torch.from_numpy(field_now))
                actual_response = torch.zeros_like(actual_field)
                point_response = torch.zeros_like(actual_field)
                torch_dispersive._update_dcp_convolution_tensors(
                    *tensors,
                    pole_work,
                    point_work,
                    torch.from_numpy(field_now),
                    actual_field,
                    torch.from_numpy(curl),
                    actual_response,
                    point_response,
                )
                np.testing.assert_allclose(actual_field.numpy(), expected_field)
                np.testing.assert_allclose(tensors[3].numpy(), expected_pole)
                np.testing.assert_allclose(tensors[4].numpy(), expected_point)

    def test_all_families_match_capture_steps_from_nonzero_fields_and_state(self):
        for model in ("drude", "lorentz", "dcp-ade", "dcp-plrc", "dcp-rc"):
            with self.subTest(model=model):
                reference, simulation = _reference_and_torch(model)
                reference.step()
                reference.step()
                simulation.advance(2)
                persistent = {
                    name: value
                    for name, value in simulation.state.state_dict().items()
                    if name.startswith("bucket_")
                }
                self.assertTrue(persistent)
                self.assertTrue(
                    any(torch.count_nonzero(value) for value in persistent.values())
                )

                completed = 0
                for capture in _CAPTURE_STEPS:
                    increment = capture - completed
                    simulation.advance(increment)
                    for _ in range(increment):
                        reference.step()
                    _assert_fields(
                        self,
                        reference,
                        simulation,
                        model=model,
                        complex_fields=False,
                    )
                    completed = capture

    def test_paired_real_complex_recurrences_match_dense_reference(self):
        bloch = (0.07, 0.11, 0.13)
        for model in ("drude", "lorentz", "dcp-ade", "dcp-plrc", "dcp-rc"):
            with self.subTest(model=model):
                reference, simulation = _reference_and_torch(
                    model,
                    bloch=bloch,
                    compile_policy="compile" if model == "dcp-plrc" else "eager",
                )
                simulation.advance(5)
                for _ in range(5):
                    reference.step()
                _assert_fields(
                    self,
                    reference,
                    simulation,
                    model=model,
                    complex_fields=True,
                )
                for name, value in simulation.state.named_buffers():
                    self.assertFalse(value.is_complex(), name)

        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        reference = gmes.TorchSimulation(
            space=gmes.Cartesian((8, 2, 2), 2),
            geometry=_mixed_geometry(),
            bloch=bloch,
            runtime=gmes.TorchRuntimeConfig(
                device="cpu",
                cpu_threads=2,
                execution_policy="dense",
            ),
        )
        simulation = gmes.TorchSimulation(
            space=gmes.Cartesian((8, 2, 2), 2),
            geometry=_mixed_geometry(),
            bloch=bloch,
            runtime=gmes.TorchRuntimeConfig(
                device="cpu",
                cpu_threads=2,
                compile_policy="compile",
                experimental_dispersive_grouping=True,
            ),
        )
        self.assertIsNotNone(simulation._dispersive_overlay)
        simulation.load_host_fields(_seed_reference(reference, complex_fields=True))
        simulation.advance(5)
        for _ in range(5):
            reference.step()
        _assert_fields(
            self,
            reference,
            simulation,
            model="mixed",
            complex_fields=True,
        )
        addresses = simulation.buffer_addresses()
        checkpoint = simulation.checkpoint()
        simulation.advance(1).load_checkpoint(checkpoint)
        self.assertEqual(simulation.buffer_addresses(), addresses)

    def test_forced_policies_are_exactly_equal(self):
        results = {}
        compile_cache_keys = set()
        representations = {}
        fields = None
        expected_operations = {
            "dense": "aten::masked_scatter_",
            "compact": "aten::index_copy_",
            "tiled": "aten::scatter_",
        }
        for policy in ("dense", "compact", "tiled"):
            _, simulation = _reference_and_torch("dcp-plrc", policy=policy)
            if fields is None:
                rng = np.random.default_rng(219)
                fields = {
                    name: rng.normal(size=tuple(field.shape)) * 1e-3
                    for name, field in simulation.state.fields().items()
                }
            simulation.load_host_fields(fields)
            addresses = simulation.buffer_addresses()
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                profile_memory=True,
            ) as profile:
                simulation.advance(20)
            self.assertEqual(addresses, simulation.buffer_addresses())
            results[policy] = simulation.state.host_snapshot()
            diagnostics = simulation.diagnostics()["dispersive"]
            representations[policy] = diagnostics["execution_representation"]
            self.assertTrue(diagnostics["policy_executions"])
            self.assertEqual(
                {item["policy"] for item in diagnostics["policy_executions"]},
                {policy},
            )
            self.assertEqual(
                {
                    item["execution_representation"]
                    for item in diagnostics["policy_executions"]
                },
                {gmes.torch_plan.EXECUTION_REPRESENTATIONS[policy]},
            )
            operation_names = {event.key for event in profile.key_averages()}
            observed_writes = operation_names & set(expected_operations.values())
            self.assertEqual(observed_writes, {expected_operations[policy]})
            expected_event = next(
                event
                for event in profile.key_averages()
                if event.key == expected_operations[policy]
            )
            self.assertEqual(expected_event.self_cpu_memory_usage, 0)
            compile_cache_keys.add(simulation.compile_cache_key)
            plan_buffers = dict(simulation.plan.named_buffers())
            for descriptor in simulation.plan.dispersive_buckets:
                mask_name = f"{descriptor.prefix}_execution_mask"
                targets_name = f"{descriptor.prefix}_execution_targets"
                self.assertEqual(mask_name in plan_buffers, policy == "dense")
                self.assertEqual(targets_name in plan_buffers, policy == "tiled")
        self.assertEqual(len(set(representations.values())), 3)
        self.assertEqual(len(compile_cache_keys), 3)
        for component in _COMPONENTS:
            np.testing.assert_array_equal(
                results["dense"][component], results["compact"][component]
            )
            np.testing.assert_array_equal(
                results["dense"][component], results["tiled"][component]
            )

    def test_compiled_forced_policies_keep_distinct_execution_identity(self):
        results = {}
        compile_cache_keys = set()
        fields = None
        for policy in ("dense", "compact", "tiled"):
            _, simulation = _reference_and_torch(
                "dcp-plrc", policy=policy, compile_policy="compile"
            )
            if fields is None:
                rng = np.random.default_rng(223)
                fields = {
                    name: rng.normal(size=tuple(field.shape)) * 1e-3
                    for name, field in simulation.state.fields().items()
                }
            simulation.load_host_fields(fields)
            addresses = simulation.buffer_addresses()
            simulation.advance(2)
            self.assertEqual(addresses, simulation.buffer_addresses())
            results[policy] = simulation.state.host_snapshot()
            compile_cache_keys.add(simulation.compile_cache_key)
            self.assertEqual(
                {
                    item["execution_representation"]
                    for item in simulation.diagnostics()["dispersive"][
                        "policy_executions"
                    ]
                },
                {gmes.torch_plan.EXECUTION_REPRESENTATIONS[policy]},
            )
        self.assertEqual(len(compile_cache_keys), 3)
        for component in _COMPONENTS:
            np.testing.assert_array_equal(
                results["dense"][component], results["compact"][component]
            )
            np.testing.assert_array_equal(
                results["dense"][component], results["tiled"][component]
            )

    def test_forced_policy_writes_preserve_paired_real_fields(self):
        results = {}
        for policy in ("dense", "compact", "tiled"):
            _, simulation = _reference_and_torch(
                "drude", bloch=(0.07, 0.11, 0.13), policy=policy
            )
            addresses = simulation.buffer_addresses()
            simulation.advance(5)
            self.assertEqual(addresses, simulation.buffer_addresses())
            results[policy] = simulation.state.host_snapshot()
        for component in _COMPONENTS:
            np.testing.assert_array_equal(
                results["dense"][component], results["compact"][component]
            )
            np.testing.assert_array_equal(
                results["dense"][component], results["tiled"][component]
            )

    def test_collapsed_1d_2d_and_3d_fields_match_dense_reference(self):
        for size in ((4, 0, 0), (4, 3, 0), (2, 2, 2)):
            for model in ("drude", "dcp-rc"):
                with self.subTest(size=size, model=model):
                    reference, simulation = _reference_and_torch(model, size=size)
                    simulation.advance(5)
                    for _ in range(5):
                        reference.step()
                    _assert_fields(
                        self,
                        reference,
                        simulation,
                        model=model,
                        complex_fields=False,
                    )

    def test_float32_all_families_match_performance_tolerance(self):
        for model in ("drude", "lorentz", "dcp-ade", "dcp-plrc", "dcp-rc"):
            with self.subTest(model=model):
                reference, simulation = _reference_and_torch(model, precision="float32")
                simulation.advance(20)
                for _ in range(20):
                    reference.step()
                _assert_fields(
                    self,
                    reference,
                    simulation,
                    model=model,
                    complex_fields=False,
                )

    def test_compiled_exact_schema_float32_matches_bucketed_execution(self):
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)

        def build(*, experimental):
            return gmes.TorchSimulation(
                space=gmes.Cartesian((8, 2, 2), 2),
                geometry=_mixed_geometry(),
                runtime=gmes.TorchRuntimeConfig(
                    device="cpu",
                    precision="float32",
                    cpu_threads=2,
                    compile_policy="compile" if experimental else "eager",
                    experimental_dispersive_grouping=experimental,
                ),
            )

        reference = build(experimental=False)
        simulation = build(experimental=True)
        self.assertIsNotNone(simulation._dispersive_overlay)
        rng = np.random.default_rng(887)
        fields = {
            name: rng.normal(size=tuple(field.shape)) * 1e-3
            for name, field in reference.state.fields().items()
        }
        reference.load_host_fields(fields).advance(20)
        simulation.load_host_fields(fields).advance(20)
        tolerances = tuple(
            _TOLERANCES[model]["float32"]
            for model in ("drude", "lorentz", "dcp-ade", "dcp-plrc", "dcp-rc")
        )
        rtol = max(value["rtol"] for value in tolerances)
        atol = max(value["atol"] for value in tolerances)
        for name, expected in reference.state.host_snapshot().items():
            np.testing.assert_allclose(
                simulation.state.host_snapshot()[name],
                expected,
                rtol=rtol,
                atol=atol,
            )
        for name, expected in reference.state.state_dict().items():
            if name.startswith("bucket_"):
                torch.testing.assert_close(
                    simulation.state.state_dict()[name],
                    expected,
                    rtol=rtol,
                    atol=atol,
                )
        for value in simulation._dispersive_overlay.buffers():
            if value.is_floating_point():
                self.assertEqual(value.dtype, torch.float32)

    def test_compiled_bulk_phases_preserve_dispersive_oracle_and_storage(self):
        torch._dynamo.reset()
        reference, simulation = _reference_and_torch(
            "dcp-plrc", compile_policy="compile"
        )
        simulation.advance(5)
        for _ in range(5):
            reference.step()
        _assert_fields(
            self,
            reference,
            simulation,
            model="dcp-plrc",
            complex_fields=False,
        )
        graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        addresses = simulation.buffer_addresses()
        simulation.advance(5)
        self.assertEqual(graphs, torch._dynamo.utils.counters["stats"]["unique_graphs"])
        self.assertEqual(addresses, simulation.buffer_addresses())

    def test_zero_width_conductive_variants_match_dense_reference(self):
        for model in ("drude", "lorentz", "dcp-ade", "dcp-plrc", "dcp-rc"):
            with self.subTest(model=model):
                reference, simulation = _reference_and_torch(model, poles=0, points=0)
                simulation.advance(20)
                for _ in range(20):
                    reference.step()
                _assert_fields(
                    self,
                    reference,
                    simulation,
                    model=model,
                    complex_fields=False,
                )

    def test_long_run_pulse_spectrum_boundary_energy_and_stability(self):
        reference, simulation = _reference_and_torch(
            "dcp-plrc", size=(12, 0, 0), resolution=4
        )
        fields = {}
        for name, field in reference.state.host_snapshot().items():
            values = np.zeros(field.shape)
            if name == "Ey":
                x = np.arange(field.shape[0])
                values[:, 0, 0] = (
                    np.exp(-0.5 * ((x - field.shape[0] * 0.28) / 2.0) ** 2) * 1e-3
                )
            fields[name] = values
        reference.load_host_fields(fields)
        simulation.load_host_fields(fields)

        completed = 0
        for capture in (100, 500):
            increment = capture - completed
            simulation.advance(increment)
            for _ in range(increment):
                reference.step()
            actual = simulation.state.host_snapshot()
            _assert_fields(
                self,
                reference,
                simulation,
                model="dcp-plrc",
                complex_fields=False,
            )
            reference_snapshot = reference.state.host_snapshot()
            reference_line = reference_snapshot["Ey"][:, 0, 0]
            torch_line = actual["Ey"][:, 0, 0]
            np.testing.assert_allclose(
                np.abs(np.fft.rfft(torch_line)),
                np.abs(np.fft.rfft(reference_line)),
                rtol=1e-11,
                atol=1e-12,
            )
            reference_energy = sum(
                float(np.sum(np.abs(field) ** 2))
                for field in reference_snapshot.values()
            )
            torch_energy = sum(
                float(np.sum(np.abs(actual[name]) ** 2)) for name in _COMPONENTS
            )
            self.assertAlmostEqual(torch_energy, reference_energy, places=16)
            self.assertLess(float(np.max(np.abs(torch_line))), 1e-3)
            completed = capture

    def test_mixed_families_share_one_complete_field_execution(self):
        reference = gmes.TorchSimulation(
            space=gmes.Cartesian((8, 2, 2), 2),
            geometry=_mixed_geometry(),
            runtime=gmes.TorchRuntimeConfig(
                device="cpu",
                cpu_threads=2,
                execution_policy="dense",
            ),
        )
        simulation = gmes.TorchSimulation(
            space=gmes.Cartesian((8, 2, 2), 2),
            geometry=_mixed_geometry(),
            runtime=gmes.TorchRuntimeConfig(device="cpu", cpu_threads=2),
        )
        simulation.load_host_fields(
            _seed_reference(reference, complex_fields=False)
        ).advance(20)
        for _ in range(20):
            reference.step()
        _assert_fields(
            self,
            reference,
            simulation,
            model="mixed",
            complex_fields=False,
        )
        self.assertEqual(
            {item.model for item in simulation.plan.dispersive_buckets},
            {"drude", "lorentz", "dcp-ade", "dcp-plrc", "dcp-rc"},
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_eager_float32_all_families_match_cpu_reference(self):
        for model in ("drude", "lorentz", "dcp-ade", "dcp-plrc", "dcp-rc"):
            with self.subTest(model=model):
                reference, simulation = _reference_and_torch(
                    model, precision="float32", device="cuda:0"
                )
                simulation.advance(5)
                for _ in range(5):
                    reference.step()
                _assert_fields(
                    self,
                    reference,
                    simulation,
                    model=model,
                    complex_fields=False,
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_compiled_float32_has_stable_storage_and_allocation(self):
        torch._dynamo.reset()
        reference, simulation = _reference_and_torch(
            "dcp-plrc",
            precision="float32",
            compile_policy="compile",
            device="cuda:0",
        )
        simulation.advance(5)
        for _ in range(5):
            reference.step()
        _assert_fields(
            self,
            reference,
            simulation,
            model="dcp-plrc",
            complex_fields=False,
        )
        torch.cuda.synchronize(simulation.device)
        graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        addresses = simulation.buffer_addresses()
        allocated = torch.cuda.memory_allocated(simulation.device)
        simulation.advance(8)
        torch.cuda.synchronize(simulation.device)
        self.assertEqual(graphs, torch._dynamo.utils.counters["stats"]["unique_graphs"])
        self.assertEqual(addresses, simulation.buffer_addresses())
        self.assertEqual(allocated, torch.cuda.memory_allocated(simulation.device))

    def test_mixed_pml_and_dispersive_underlying_match_capture_steps(self):
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        for experimental in (False, True):
            with self.subTest(experimental=experimental):
                reference = gmes.TorchSimulation(
                    space=gmes.Cartesian((8, 6, 0), 4),
                    geometry=_mixed_pml_geometry(),
                    runtime=gmes.TorchRuntimeConfig(
                        device="cpu",
                        cpu_threads=2,
                        execution_policy="dense",
                    ),
                )
                simulation = gmes.TorchSimulation(
                    space=gmes.Cartesian((8, 6, 0), 4),
                    geometry=_mixed_pml_geometry(),
                    runtime=gmes.TorchRuntimeConfig(
                        device="cpu",
                        cpu_threads=2,
                        compile_policy="compile" if experimental else "eager",
                        experimental_dispersive_grouping=experimental,
                    ),
                )
                if experimental:
                    self.assertIsNotNone(simulation._dispersive_overlay)
                else:
                    self.assertIsNone(simulation._dispersive_overlay)
                simulation.load_host_fields(
                    _seed_reference(reference, complex_fields=False)
                )
                completed = 0
                for capture in _CAPTURE_STEPS:
                    increment = capture - completed
                    simulation.advance(increment)
                    for _ in range(increment):
                        reference.step()
                    _assert_fields(
                        self,
                        reference,
                        simulation,
                        model="mixed",
                        complex_fields=False,
                    )
                    completed = capture
                self.assertEqual(
                    {item.model for item in simulation.plan.dispersive_buckets},
                    {"drude", "lorentz", "dcp-ade", "dcp-plrc", "dcp-rc"},
                )
                pml_buckets = [
                    bucket
                    for component in simulation.plan.components.values()
                    for bucket in component.buckets
                    if bucket.signature.model == "cpml"
                ]
                self.assertTrue(
                    all(len(bucket.region_keys) == 6 for bucket in pml_buckets)
                )


class DispersiveStorageTest(unittest.TestCase):
    def test_compiled_exact_schema_groups_preserve_logical_state_and_results(self):
        processors = os.cpu_count() or 1
        schema_threads = min(4, processors)
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch._dynamo.reset()
        graphs_before = torch._dynamo.utils.counters["stats"]["unique_graphs"]

        def build(compile_policy, *, experimental=False, scope="combined"):
            return gmes.TorchSimulation(
                space=gmes.Cartesian((8, 2, 2), 2),
                geometry=_mixed_geometry(),
                runtime=gmes.TorchRuntimeConfig(
                    device="cpu",
                    cpu_threads=schema_threads,
                    compile_policy=compile_policy,
                    experimental_dispersive_grouping=experimental,
                    experimental_dispersive_grouping_scope=scope,
                ),
            )

        eager = build("eager", experimental=True)
        compiled = build("compile", experimental=True)
        default_compiled = build("compile")
        two_level = build("compile", experimental=True, scope="two-level")
        dcp_convolution = build("compile", experimental=True, scope="dcp-convolution")
        overlay = compiled._dispersive_overlay
        self.assertIsNone(eager._dispersive_overlay)
        self.assertIsNone(default_compiled._dispersive_overlay)
        self.assertIsNotNone(overlay)
        self.assertEqual(len(overlay.groups), 6)
        self.assertEqual(len(overlay.entries), 9)
        self.assertEqual(len(two_level._dispersive_overlay.groups), 3)
        self.assertEqual(len(two_level._dispersive_overlay.entries), 12)
        self.assertEqual(
            {group.recurrence for group in two_level._dispersive_overlay.groups},
            {"two-level"},
        )
        self.assertEqual(len(dcp_convolution._dispersive_overlay.groups), 3)
        self.assertEqual(len(dcp_convolution._dispersive_overlay.entries), 12)
        self.assertEqual(
            {group.recurrence for group in dcp_convolution._dispersive_overlay.groups},
            {"dcp-convolution"},
        )
        self.assertEqual(
            tuple(
                tuple(span.descriptor.model for span in group.spans)
                for group in overlay.groups
            ),
            (
                ("dcp-plrc", "dcp-rc"),
                ("drude", "lorentz"),
            )
            * 3,
        )
        for group in overlay.groups:
            descriptors = tuple(span.descriptor for span in group.spans)
            for suffix in (
                "targets",
                "source_0_positive",
                "source_0_negative",
                "source_1_positive",
                "source_1_negative",
            ):
                expected = torch.cat(
                    tuple(
                        getattr(compiled.plan, f"{descriptor.prefix}_{suffix}")
                        for descriptor in descriptors
                    )
                )
                actual = getattr(overlay, f"{group.prefix}_{suffix}")
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            targets = getattr(overlay, f"{group.prefix}_targets")
            self.assertEqual(torch.unique(targets).numel(), targets.numel())
            persistent_suffixes = (
                ("previous", "current")
                if group.recurrence == "two-level"
                else ("pole_state", "point_state")
            )
            for span in group.spans:
                for suffix in persistent_suffixes:
                    logical = getattr(
                        compiled.state, f"{span.descriptor.prefix}_{suffix}"
                    )
                    physical = getattr(overlay, f"{group.prefix}_{suffix}")
                    self.assertEqual(
                        logical.untyped_storage().data_ptr(),
                        physical.untyped_storage().data_ptr(),
                    )

        eager_state = eager.state.state_dict()
        compiled_state = compiled.state.state_dict()
        self.assertEqual(tuple(eager_state), tuple(compiled_state))
        self.assertEqual(
            {name: tuple(value.shape) for name, value in eager_state.items()},
            {name: tuple(value.shape) for name, value in compiled_state.items()},
        )
        for name, expected in eager_state.items():
            if name.startswith("bucket_"):
                actual = compiled_state[name]
                self.assertEqual(actual.dtype, expected.dtype)
                self.assertEqual(actual.device, expected.device)
                self.assertTrue(actual.is_contiguous())
                self.assertEqual(actual.stride(), expected.stride())
                self.assertEqual(actual.view(-1).shape, expected.view(-1).shape)
        for name in compiled.state._grouped_dispersive_state_names:
            self.assertNotEqual(
                compiled_state[name].untyped_storage().data_ptr(),
                getattr(compiled.state, name).untyped_storage().data_ptr(),
            )
        self.assertEqual(eager.plan_identity, compiled.plan_identity)
        self.assertEqual(eager.plan_identity, default_compiled.plan_identity)
        self.assertNotEqual(
            compiled.compile_cache_key,
            default_compiled.compile_cache_key,
        )
        self.assertNotEqual(
            compiled.compile_cache_key,
            two_level.compile_cache_key,
        )
        self.assertNotEqual(
            two_level.compile_cache_key,
            dcp_convolution.compile_cache_key,
        )
        compiled.advance(1)
        self.assertEqual(
            torch._dynamo.utils.counters["stats"]["unique_graphs"] - graphs_before,
            2,
        )
        compiled.load_checkpoint(eager.checkpoint())
        graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        addresses = compiled.buffer_addresses()
        compiled.advance(1)
        self.assertEqual(
            torch._dynamo.utils.counters["stats"]["unique_graphs"],
            graphs,
        )
        self.assertEqual(compiled.buffer_addresses(), addresses)
        compiled.load_checkpoint(eager.checkpoint())
        eager.load_checkpoint(compiled.checkpoint())
        compiled.load_checkpoint(eager.checkpoint())

        rng = np.random.default_rng(319)
        fields = {
            name: rng.normal(size=tuple(field.shape)) * 1e-3
            for name, field in eager.state.fields().items()
        }
        eager.load_host_fields(fields).advance(2)
        compiled.load_host_fields(fields).advance(2)
        compiled_fields = compiled.state.host_snapshot()
        for name, expected in eager.state.host_snapshot().items():
            np.testing.assert_allclose(
                compiled_fields[name],
                expected,
                rtol=1e-13,
                atol=1e-15,
            )
        for name, expected in eager.state.state_dict().items():
            if name.startswith("bucket_"):
                torch.testing.assert_close(
                    compiled.state.state_dict()[name],
                    expected,
                    rtol=1e-13,
                    atol=1e-15,
                )

        addresses = compiled.buffer_addresses()
        checkpoint = eager.checkpoint()
        compiled.advance(1).load_checkpoint(checkpoint)
        self.assertEqual(compiled.buffer_addresses(), addresses)
        for name, expected in eager.state.state_dict().items():
            torch.testing.assert_close(
                compiled.state.state_dict()[name],
                expected,
                rtol=0,
                atol=0,
            )
        incompatible = compiled.state.load_state_dict(eager.state.state_dict())
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertEqual(compiled.buffer_addresses(), addresses)
        incompatible = eager.state.load_state_dict(compiled.state.state_dict())
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        with self.assertRaisesRegex(ValueError, "assign=False"):
            compiled.state.load_state_dict(eager.state.state_dict(), assign=True)

        diagnostics = compiled.diagnostics()["dispersive"]
        self.assertEqual(
            diagnostics["execution_representation"],
            "exact-schema-grouped-io-v1",
        )
        self.assertEqual(diagnostics["execution_entries_per_step"], 9)
        self.assertEqual(
            diagnostics["logical_buckets_per_step"],
            len(compiled.plan.dispersive_buckets),
        )
        self.assertEqual(diagnostics["exact_schema_groups"], 6)
        self.assertEqual(diagnostics["experimental_grouping_scope"], "combined")
        self.assertEqual(len(diagnostics["exact_schema_spans"]), 6)
        self.assertEqual(
            tuple(
                span["model"]
                for span in diagnostics["exact_schema_spans"][0]["logical_spans"]
            ),
            ("dcp-plrc", "dcp-rc"),
        )
        self.assertEqual(
            diagnostics["launches_per_step"],
            len(compiled.plan.dispersive_buckets),
        )
        self.assertTrue(
            any(
                name.startswith("dispersive_overlay.")
                for name in compiled.buffer_addresses()
            )
        )

    def test_state_uses_exact_width_real_soa_and_fixed_storage(self):
        simulation = gmes.TorchSimulation(
            space=gmes.Cartesian((6, 2, 2), 2),
            geometry=[
                gmes.DefaultMedium(gmes.Dielectric(eps_inf=1.7, mu_inf=1.05)),
                gmes.Block(
                    _material("drude", poles=1),
                    center=(-2.2, 0, 0),
                    size=(0.8, 1.4, 1.4),
                ),
                gmes.Block(
                    _material("drude", poles=4),
                    center=(-1.0, 0, 0),
                    size=(0.8, 1.4, 1.4),
                ),
                gmes.Block(
                    _material("lorentz", poles=4),
                    center=(0.2, 0, 0),
                    size=(0.8, 1.4, 1.4),
                ),
                gmes.Block(
                    _material("dcp-ade", poles=2, points=3),
                    center=(1.4, 0, 0),
                    size=(0.8, 1.4, 1.4),
                ),
                gmes.Block(
                    _material("dcp-rc", poles=2, points=3),
                    center=(2.6, 0, 0),
                    size=(0.6, 1.4, 1.4),
                ),
            ],
            bloch=(0.03, 0.05, 0.07),
            runtime=gmes.TorchRuntimeConfig(device="cpu", cpu_threads=2),
        )
        self.assertGreater(len(simulation.plan.dispersive_buckets), 5)
        persistent = simulation.state.state_dict()
        for descriptor in simulation.plan.dispersive_buckets:
            values = [
                tensor
                for name, tensor in persistent.items()
                if name.startswith(f"{descriptor.prefix}_")
            ]
            actual = sum(value.numel() for value in values)
            expected = descriptor.target_count * descriptor.state_width * 2
            self.assertEqual(actual, expected, descriptor)
            self.assertTrue(all(value.dtype == torch.float64 for value in values))
            coefficients = getattr(simulation.plan, f"{descriptor.prefix}_a")
            self.assertTrue(coefficients.is_contiguous())
            self.assertFalse(coefficients.is_complex())

        addresses = simulation.buffer_addresses()
        simulation.advance(20)
        self.assertEqual(addresses, simulation.buffer_addresses())
        self.assertEqual(simulation.plan.unsupported_models, ())
        diagnostics = simulation.diagnostics()["dispersive"]
        self.assertEqual(
            diagnostics["models"],
            ("dcp-ade", "dcp-rc", "drude", "lorentz"),
        )
        self.assertEqual(diagnostics["state_width_policy"], "exact")
        self.assertEqual(diagnostics["padding_elements"], 0)
        self.assertGreater(diagnostics["padding_elements_avoided"], 0)
        self.assertEqual(
            diagnostics["launches_per_step"],
            len(simulation.plan.dispersive_buckets),
        )


if __name__ == "__main__":
    unittest.main()
