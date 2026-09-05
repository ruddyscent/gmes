"""Differential and state-layout tests for Torch Maxwell--Bloch execution."""

import unittest

import numpy as np
import torch

import gmes
from benchmarks import native_oracle
from gmes import torch_dm2
from gmes.torch_dm2 import (
    DM2_ITERATIONS_PER_CHUNK,
    DM2_MAX_ITERATIONS,
    DM2_PACKED_ITERATIONS_PER_CONDITION,
)

_DM2_FLOAT32_TOLERANCE = native_oracle.load_manifest()["tolerances"]["torch"]["dm2"][
    "float32"
]


def _geometry(material):
    return [
        gmes.DefaultMedium(gmes.Dielectric(eps_inf=1.7, mu_inf=1.05)),
        gmes.Block(material, center=(0, 0, 0), size=(2, 2, 2)),
    ]


def _copy_material(material):
    return gmes.Dm2(
        eps_inf=material.eps_inf,
        mu_inf=material.mu_inf,
        omega=material.omega,
        n_atom=material.n_atom,
        rho30=material.rho30,
        gamma=material.gamma,
        t1=material.t1,
        t2=material.t2,
        hbar=material.hbar,
        rtol=material.rtol,
    )


def _simulations(
    material,
    *,
    precision="float64",
    compile_policy="eager",
    execution_policy="auto",
):
    reference = gmes.TorchSimulation(
        space=gmes.Cartesian((2, 2, 2), 2),
        geometry=_geometry(_copy_material(material)),
        runtime=gmes.TorchRuntimeConfig(
            device="cpu",
            precision="float64",
            execution_policy="dense",
            cpu_threads=2,
        ),
        dt=0.025,
    )
    torch_simulation = gmes.TorchSimulation(
        space=gmes.Cartesian((2, 2, 2), 2),
        geometry=_geometry(_copy_material(material)),
        runtime=gmes.TorchRuntimeConfig(
            device="cpu",
            precision=precision,
            compile_policy=compile_policy,
            execution_policy=execution_policy,
            cpu_threads=2,
        ),
        dt=0.025,
    )
    return reference, torch_simulation


class TorchDm2Test(unittest.TestCase):
    def test_zero_field_is_an_exact_equilibrium_with_one_corrector_iteration(self):
        material = gmes.Dm2(
            eps_inf=1.4,
            omega=(0.7, 1.1),
            n_atom=(0.2, 0.4),
            rho30=-0.75,
            gamma=0.15,
            t1=2.5,
            t2=1.7,
            rtol=1e-8,
        )
        _, simulation = _simulations(material)
        initial_targets = {
            snapshot["component"]: snapshot["targets"].copy()
            for snapshot in simulation.dm2_state_snapshot()
        }

        # With E = curl(E) = u = 0, every term in the implicit Bloch
        # recurrence is zero. The fixed point is therefore exact on the first
        # corrector iteration, independent of execution policy.
        simulation.step()

        for name, field in simulation.state.host_snapshot().items():
            np.testing.assert_array_equal(field, np.zeros_like(field), err_msg=name)
        snapshots = simulation.dm2_state_snapshot()
        self.assertEqual({item["component"] for item in snapshots}, {"Ex", "Ey", "Ez"})
        for snapshot in snapshots:
            targets = snapshot["targets"]
            np.testing.assert_array_equal(
                targets,
                initial_targets[snapshot["component"]],
            )
            self.assertTrue(np.all(targets[1:] > targets[:-1]))
            np.testing.assert_array_equal(snapshot["u"], np.zeros_like(snapshot["u"]))
        self.assertTrue(
            torch.equal(
                simulation.state._dm2_status,
                torch.zeros_like(simulation.state._dm2_status),
            )
        )
        self.assertTrue(
            torch.equal(
                simulation.state._dm2_iterations,
                torch.ones_like(simulation.state._dm2_iterations),
            )
        )

    def test_nonzero_state_matches_explicit_scalar_corrector_recurrence(self):
        metadata = torch_dm2.Dm2BucketMetadata(
            component="Ex",
            bucket_index=0,
            transition_count=1,
            target_count=1,
            prefix="dm2_scalar",
            source_component="Hz",
            status_offset=0,
        )
        status = torch.zeros(1, dtype=torch.int8)
        iterations = torch.zeros(1, dtype=torch.int32)
        state = torch_dm2.TorchDm2BucketState(
            metadata,
            status=status,
            iterations=iterations,
            device="cpu",
            dtype=torch.float64,
        )
        initial_u = np.asarray((0.05, -0.02, 0.01), dtype=np.float64).reshape(3, 1, 1)
        state.u.copy_(torch.from_numpy(initial_u))
        field = torch.tensor([0.2], dtype=torch.float64)
        source = torch.tensor([0.4, -0.1], dtype=torch.float64)
        target = torch.tensor([0], dtype=torch.int64)
        positive = torch.tensor([0], dtype=torch.int64)
        negative = torch.tensor([1], dtype=torch.int64)
        dt = 0.025
        rho30, gamma, t1, t2, hbar = -0.7, 0.15, 2.5, 1.7, 1.0
        omega, density, curl_scale, tolerance = 0.9, 0.3, 0.4, 1e-10

        time = dt
        decay = np.exp(-time / t2)
        coefficient_a = density * gamma / t2 * decay
        coefficient_b = density * gamma * omega * decay
        coefficient_plus = 2.0 * gamma / hbar * np.exp(-(1.0 / t1 - 1.0 / t2) * time)
        coefficient_minus = 2.0 * gamma / hbar * np.exp(-(1.0 / t2 - 1.0 / t1) * time)
        coefficient_d = 2.0 * gamma * rho30 / hbar * np.exp(time / t2)
        old_field = 0.2
        base_field = old_field + (0.4 - -0.1) * curl_scale
        old_u = initial_u[:, 0, 0].copy()
        expected_field = old_field
        expected_u = old_u.copy()
        expected_iterations = 0
        for _ in range(DM2_MAX_ITERATIONS):
            previous_field = expected_field
            previous_u = expected_u.copy()
            expected_field = (
                base_field
                - 0.5 * dt * (expected_u[0] + old_u[0]) * coefficient_a
                + 0.5 * dt * (expected_u[1] + old_u[1]) * coefficient_b
            )
            field_sum = expected_field + old_field
            u0 = old_u[0] + (expected_u[1] + old_u[1]) * omega * 0.5 * dt
            u1 = old_u[1] - (u0 + old_u[0]) * omega * 0.5 * dt
            u1 += (expected_u[2] + old_u[2]) * coefficient_plus * field_sum * 0.25 * dt
            u1 += coefficient_d * field_sum * 0.5 * dt
            u2 = old_u[2] - (u1 + old_u[1]) * coefficient_minus * field_sum * 0.25 * dt
            expected_u = np.asarray((u0, u1, u2))
            numerator = np.sqrt(
                (expected_field - previous_field) ** 2
                + np.sum((expected_u - previous_u) ** 2)
            )
            denominator = np.sqrt(previous_field**2 + np.sum(previous_u**2))
            error = 0.0 if numerator == denominator == 0.0 else numerator / denominator
            expected_iterations += 1
            if error <= tolerance:
                break

        scalar = lambda value: torch.tensor([value], dtype=torch.float64)
        state.prepare(
            field,
            source,
            torch.tensor(0, dtype=torch.int64),
            torch.tensor(dt, dtype=torch.float64),
            target,
            positive,
            negative,
            scalar(rho30),
            scalar(gamma),
            scalar(t1),
            scalar(t2),
            scalar(hbar),
            torch.tensor([[omega]], dtype=torch.float64),
            torch.tensor([[density]], dtype=torch.float64),
            scalar(curl_scale),
        )
        for _ in range(DM2_MAX_ITERATIONS // DM2_ITERATIONS_PER_CHUNK):
            state.iterate(
                0.5 * dt,
                0.25 * dt,
                scalar(tolerance),
                torch.tensor([[omega]], dtype=torch.float64),
            )
        state.finalize(field, target)

        self.assertEqual(int(status[0]), 0)
        self.assertEqual(int(iterations[0]), expected_iterations)
        self.assertAlmostEqual(float(field[0]), expected_field, places=14)
        np.testing.assert_allclose(
            state.u.numpy()[:, 0, 0], expected_u, rtol=0, atol=1e-14
        )

    def _build_failure_simulation(self, material, compile_policy):
        return gmes.TorchSimulation(
            space=gmes.Cartesian((3, 3, 3), 2),
            geometry=[
                gmes.DefaultMedium(gmes.Dielectric(eps_inf=1.7, mu_inf=1.05)),
                gmes.Block(
                    material,
                    center=(0, 0, 0),
                    size=(1, 1, 1),
                ),
                gmes.Shell(gmes.Cpml(), thickness=0.5),
            ],
            runtime=gmes.TorchRuntimeConfig(
                device="cpu",
                compile_policy=compile_policy,
                cpu_threads=2,
            ),
            dt=0.025,
        )

    def _assert_simulation_state_matches(self, actual, expected):
        actual_fields = actual.state.host_snapshot()
        expected_fields = expected.state.host_snapshot()
        for name in expected_fields:
            np.testing.assert_allclose(
                actual_fields[name], expected_fields[name], rtol=0, atol=2e-18
            )
        actual_pml = actual.state.pml_state_snapshot(numpy=False)
        expected_pml = expected.state.pml_state_snapshot(numpy=False)
        self.assertEqual(set(actual_pml), set(expected_pml))
        for name in expected_pml:
            self.assertTrue(torch.equal(actual_pml[name], expected_pml[name]), name)
        for actual_bucket, expected_bucket in zip(
            actual.state.dm2_buckets, expected.state.dm2_buckets
        ):
            self.assertTrue(torch.equal(actual_bucket.u, expected_bucket.u))
        self.assertTrue(
            torch.equal(actual.state._dm2_status, expected.state._dm2_status)
        )
        self.assertTrue(
            torch.equal(actual.state._dm2_iterations, expected.state._dm2_iterations)
        )
        self.assertTrue(torch.equal(actual.state.step_count, expected.state.step_count))
        self.assertTrue(
            torch.equal(actual.state.source_time, expected.state.source_time)
        )

    def test_exact_width_state_is_active_cell_only(self):
        material = gmes.Dm2(
            eps_inf=1.4,
            omega=(0.7, 0.9, 1.1, 1.3),
            n_atom=(0.2, 0.3, 0.4, 0.5),
        )
        _, simulation = _simulations(material)

        self.assertEqual(len(simulation.state.dm2_buckets), 3)
        for bucket in simulation.state.dm2_buckets:
            metadata = bucket.metadata
            self.assertEqual(metadata.transition_count, 4)
            self.assertEqual(tuple(bucket.u.shape), (3, metadata.target_count, 4))
            self.assertLess(
                metadata.target_count,
                int(np.prod(simulation.plan.shapes[metadata.component])),
            )

    def test_packed_cpu_iteration_schedule_is_public_and_exact(self):
        self.assertEqual(DM2_ITERATIONS_PER_CHUNK, 10)
        self.assertEqual(DM2_PACKED_ITERATIONS_PER_CONDITION, 3)
        self.assertIn("DM2_PACKED_ITERATIONS_PER_CONDITION", torch_dm2.__all__)

    def test_packed_cpu_workspace_is_exact_nonpersistent_and_fixed(self):
        material = gmes.Dm2(
            eps_inf=1.4,
            omega=(0.7, 0.9, 1.1, 1.3),
            n_atom=(0.2, 0.3, 0.4, 0.5),
            gamma=0.15,
            rtol=1e-9,
        )
        _, simulation = _simulations(material, compile_policy="compile")
        workspaces = []
        for bucket in simulation.state.dm2_buckets:
            metadata = bucket.metadata
            workspace = bucket._packed_loop_state
            expected_elements = metadata.target_count * (
                3 * metadata.transition_count + 2
            )
            self.assertEqual(tuple(workspace.shape), (expected_elements,))
            self.assertEqual(workspace.dtype, bucket.u.dtype)
            self.assertEqual(workspace.device, bucket.u.device)
            self.assertNotIn("_packed_loop_state", bucket.state_dict())
            workspaces.append((workspace.data_ptr(), workspace.numel()))

        rng = np.random.default_rng(146)
        simulation.load_host_fields(
            {
                name: rng.normal(size=tuple(field.shape)) * 1e-3
                for name, field in simulation.state.fields().items()
            }
        )
        simulation.step()

        self.assertEqual(
            workspaces,
            [
                (
                    bucket._packed_loop_state.data_ptr(),
                    bucket._packed_loop_state.numel(),
                )
                for bucket in simulation.state.dm2_buckets
            ],
        )
        self.assertFalse(
            any("_packed_loop_state" in name for name in simulation.state.state_dict())
        )

    def test_compiled_packed_corrector_preserves_early_convergence(self):
        _, simulation = _simulations(gmes.Dm2(), compile_policy="compile")

        simulation.step()

        self.assertTrue(torch.all(simulation.state._dm2_status == 0))
        self.assertTrue(torch.all(simulation.state._dm2_iterations == 1))

    def test_complete_fields_and_state_match_dense_reference(self):
        material = gmes.Dm2(
            eps_inf=1.4,
            mu_inf=1.1,
            omega=(0.7, 1.1),
            n_atom=(0.2, 0.4),
            rho30=-0.8,
            gamma=0.15,
            t1=2.5,
            t2=1.7,
            hbar=1.2,
            rtol=1e-10,
        )
        reference, simulation = _simulations(material)
        rng = np.random.default_rng(120)
        fields = {
            name: rng.normal(size=field.shape) * 1e-3
            for name, field in reference.state.host_snapshot().items()
        }
        reference.load_host_fields(fields)
        simulation.load_host_fields(fields)

        for _ in range(2):
            reference.step()
            simulation.step()

        actual = simulation.state.host_snapshot()
        for name, expected in reference.state.host_snapshot().items():
            np.testing.assert_allclose(
                actual[name],
                expected,
                rtol=2e-10,
                atol=2e-12,
                err_msg=name,
            )
        snapshots = {
            snapshot["component"]: snapshot
            for snapshot in simulation.dm2_state_snapshot()
        }
        reference_snapshots = {
            snapshot["component"]: snapshot
            for snapshot in reference.dm2_state_snapshot()
        }
        for component_name in ("Ex", "Ey", "Ez"):
            snapshot = snapshots[component_name]
            expected = reference_snapshots[component_name]
            np.testing.assert_array_equal(snapshot["targets"], expected["targets"])
            np.testing.assert_allclose(
                snapshot["u"], expected["u"], rtol=2e-10, atol=2e-12
            )

    def test_multiple_transition_widths_form_exact_buckets(self):
        geometry = [
            gmes.DefaultMedium(gmes.Dielectric(eps_inf=1.7, mu_inf=1.05)),
            gmes.Block(
                gmes.Dm2(omega=(0.7,), n_atom=(0.2,)),
                center=(-0.75, 0, 0),
                size=(1, 2, 2),
            ),
            gmes.Block(
                gmes.Dm2(
                    omega=(0.7, 0.9, 1.1, 1.3),
                    n_atom=(0.2, 0.3, 0.4, 0.5),
                ),
                center=(0.75, 0, 0),
                size=(1, 2, 2),
            ),
        ]
        simulation = gmes.TorchSimulation(
            space=gmes.Cartesian((3, 2, 2), 2),
            geometry=geometry,
            runtime=gmes.TorchRuntimeConfig(device="cpu", cpu_threads=2),
            dt=0.025,
        )

        for component in ("Ex", "Ey", "Ez"):
            widths = {
                bucket.metadata.transition_count
                for bucket in simulation.state.dm2_buckets
                if bucket.metadata.component == component
            }
            self.assertEqual(widths, {1, 4})
        self.assertEqual(
            sum(bucket.u.numel() for bucket in simulation.state.dm2_buckets),
            sum(
                3 * bucket.metadata.target_count * bucket.metadata.transition_count
                for bucket in simulation.state.dm2_buckets
            ),
        )

    def test_forced_planner_policies_produce_identical_dm2_results(self):
        material = gmes.Dm2(
            omega=(0.7, 1.1),
            n_atom=(0.2, 0.4),
            gamma=0.15,
            rtol=1e-9,
        )
        rng = np.random.default_rng(122)
        fields = None
        results = {}
        states = {}
        for policy in ("dense", "compact", "tiled"):
            simulation = gmes.TorchSimulation(
                space=gmes.Cartesian((2, 2, 2), 2),
                geometry=_geometry(material),
                runtime=gmes.TorchRuntimeConfig(
                    device="cpu",
                    cpu_threads=2,
                    execution_policy=policy,
                    planner_tile_size=8,
                ),
                dt=0.025,
            )
            if fields is None:
                fields = {
                    name: rng.normal(size=tuple(field.shape)) * 1e-3
                    for name, field in simulation.state.fields().items()
                }
            simulation.load_host_fields(fields).advance(2)
            results[policy] = simulation.state.host_snapshot()
            states[policy] = simulation.dm2_state_snapshot()
        for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            np.testing.assert_array_equal(
                results["dense"][component], results["compact"][component]
            )
            np.testing.assert_array_equal(
                results["dense"][component], results["tiled"][component]
            )
        for index in range(len(states["dense"])):
            np.testing.assert_array_equal(
                states["dense"][index]["u"], states["compact"][index]["u"]
            )
            np.testing.assert_array_equal(
                states["dense"][index]["u"], states["tiled"][index]["u"]
            )

    def test_float32_matches_dense_float64_tolerance(self):
        material = gmes.Dm2(
            omega=(0.7, 1.1),
            n_atom=(0.2, 0.4),
            gamma=0.15,
            rtol=1e-6,
        )
        reference, simulation = _simulations(material, precision="float32")
        rng = np.random.default_rng(123)
        fields = {
            name: rng.normal(size=field.shape) * 1e-3
            for name, field in reference.state.host_snapshot().items()
        }
        reference.load_host_fields(fields)
        simulation.load_host_fields(fields)
        reference.step()
        simulation.step()
        actual = simulation.state.host_snapshot()
        for name, expected in reference.state.host_snapshot().items():
            np.testing.assert_allclose(
                actual[name],
                expected,
                rtol=_DM2_FLOAT32_TOLERANCE["rtol"],
                atol=_DM2_FLOAT32_TOLERANCE["atol"],
                err_msg=name,
            )

    def test_compiled_preconditioned_state_matches_at_fixed_capture_steps(self):
        material = gmes.Dm2(
            omega=(0.7,),
            n_atom=(0.2,),
            gamma=0.15,
            t1=2.5,
            t2=1.7,
            rtol=1e-8,
        )
        reference, simulation = _simulations(material, compile_policy="compile")
        rng = np.random.default_rng(125)
        fields = {
            name: rng.normal(size=field.shape) * 1e-3
            for name, field in reference.state.host_snapshot().items()
        }
        reference.load_host_fields(fields).advance(2)
        fields = reference.state.host_snapshot()
        reference_state = {
            snapshot["component"]: snapshot["u"]
            for snapshot in reference.dm2_state_snapshot()
        }
        states = [
            reference_state[bucket.metadata.component]
            for bucket in simulation.state.dm2_buckets
        ]
        simulation.load_host_fields(fields)
        simulation.load_host_dm2_state(states, step_count=2)
        self.assertTrue(all(np.any(state) for state in states))

        completed = 0
        for capture in (1, 2, 5, 20, 100):
            delta = capture - completed
            reference.advance(delta)
            simulation.advance(delta)
            completed = capture
            actual = simulation.state.host_snapshot()
            for name, expected in reference.state.host_snapshot().items():
                np.testing.assert_allclose(
                    actual[name],
                    expected,
                    rtol=2e-10,
                    atol=2e-12,
                    err_msg=f"{name} at {capture}",
                )
            snapshots = {
                snapshot["component"]: snapshot
                for snapshot in simulation.dm2_state_snapshot()
            }
            reference_snapshots = {
                snapshot["component"]: snapshot
                for snapshot in reference.dm2_state_snapshot()
            }
            for component_name in ("Ex", "Ey", "Ez"):
                expected = reference_snapshots[component_name]["u"]
                np.testing.assert_allclose(
                    snapshots[component_name]["u"],
                    expected,
                    rtol=2e-10,
                    atol=2e-12,
                    err_msg=f"{component_name} state at {capture}",
                )

    def test_collapsed_axes_match_dense_reference(self):
        for size in ((2, 0, 0), (2, 2, 0)):
            with self.subTest(size=size):
                reference = gmes.TorchSimulation(
                    space=gmes.Cartesian(size, 2),
                    geometry=_geometry(
                        gmes.Dm2(
                            omega=(0.7,),
                            n_atom=(0.2,),
                            gamma=0.15,
                            rtol=1e-9,
                        )
                    ),
                    runtime=gmes.TorchRuntimeConfig(
                        device="cpu",
                        execution_policy="dense",
                        cpu_threads=2,
                    ),
                    dt=0.025,
                )
                simulation = gmes.TorchSimulation(
                    space=gmes.Cartesian(size, 2),
                    geometry=_geometry(
                        gmes.Dm2(
                            omega=(0.7,),
                            n_atom=(0.2,),
                            gamma=0.15,
                            rtol=1e-9,
                        )
                    ),
                    runtime=gmes.TorchRuntimeConfig(
                        device="cpu",
                        execution_policy="compact",
                        cpu_threads=2,
                    ),
                    dt=0.025,
                )
                rng = np.random.default_rng(124)
                fields = {
                    name: rng.normal(size=field.shape) * 1e-3
                    for name, field in reference.state.host_snapshot().items()
                }
                reference.load_host_fields(fields)
                simulation.load_host_fields(fields)
                reference.step()
                simulation.step()
                actual = simulation.state.host_snapshot()
                for name, expected in reference.state.host_snapshot().items():
                    np.testing.assert_allclose(
                        actual[name],
                        expected,
                        rtol=2e-10,
                        atol=2e-12,
                        err_msg=name,
                    )

    def test_zero_reference_converges_and_nan_retains_failed_state(self):
        _, simulation = _simulations(gmes.Dm2())
        fields = {
            name: np.zeros(tuple(field.shape))
            for name, field in simulation.state.fields().items()
        }
        fields["Hy"][1, 0, 1] = 1
        simulation.load_host_fields(fields).step()
        self.assertTrue(
            all(
                np.isfinite(value).all()
                for value in simulation.state.host_snapshot().values()
            )
        )

        _, invalid = _simulations(gmes.Dm2(gamma=np.nan))
        before = invalid.state.host_snapshot()
        with self.assertRaisesRegex(RuntimeError, "invalid error.*Ex/width=1"):
            invalid.step()
        after = invalid.state.host_snapshot()
        for snapshot in invalid.dm2_state_snapshot():
            self.assertFalse(np.any(snapshot["u"]))
            field_before = before[snapshot["component"]].reshape(-1)
            field_after = after[snapshot["component"]].reshape(-1)
            np.testing.assert_array_equal(
                field_after[snapshot["targets"]],
                field_before[snapshot["targets"]],
            )

    def test_compiled_invalid_error_commits_the_same_state_as_eager(self):
        eager = self._build_failure_simulation(gmes.Dm2(gamma=np.nan), "eager")
        compiled = self._build_failure_simulation(gmes.Dm2(gamma=np.nan), "compile")
        rng = np.random.default_rng(145)
        fields = {
            name: rng.normal(size=tuple(field.shape)) * 1e-3
            for name, field in eager.state.fields().items()
        }
        errors = []
        for simulation in (eager, compiled):
            simulation.load_host_fields(fields)
            with self.assertRaisesRegex(RuntimeError, "invalid error") as caught:
                simulation.step()
            errors.append(str(caught.exception))

        self.assertEqual(errors[1], errors[0])
        self._assert_simulation_state_matches(compiled, eager)

    def test_compiled_nonconvergence_commits_the_same_state_as_eager(self):
        eager = self._build_failure_simulation(gmes.Dm2(rtol=-1), "eager")
        compiled = self._build_failure_simulation(gmes.Dm2(rtol=-1), "compile")
        rng = np.random.default_rng(147)
        fields = {
            name: rng.normal(size=tuple(field.shape)) * 1e-3
            for name, field in eager.state.fields().items()
        }
        errors = []
        for simulation in (eager, compiled):
            simulation.load_host_fields(fields)
            with self.assertRaisesRegex(RuntimeError, "failed to converge") as caught:
                simulation.step()
            errors.append(str(caught.exception))

        self.assertEqual(errors[1], errors[0])
        self._assert_simulation_state_matches(compiled, eager)
        self.assertTrue(torch.all(compiled.state._dm2_status == 2))
        self.assertTrue(torch.all(compiled.state._dm2_iterations == DM2_MAX_ITERATIONS))

    def test_compiled_fullgraph_matches_dense_reference(self):
        material = gmes.Dm2(
            omega=(0.7,),
            n_atom=(0.2,),
            gamma=0.15,
            t1=2.5,
            t2=1.7,
            rtol=1e-8,
        )
        reference, simulation = _simulations(material, compile_policy="compile")
        rng = np.random.default_rng(121)
        fields = {
            name: rng.normal(size=field.shape) * 1e-3
            for name, field in reference.state.host_snapshot().items()
        }
        reference.load_host_fields(fields)
        simulation.load_host_fields(fields)

        reference.step()
        simulation.step()
        graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        addresses = simulation.buffer_addresses()

        actual = simulation.state.host_snapshot()
        for name, expected in reference.state.host_snapshot().items():
            np.testing.assert_allclose(
                actual[name],
                expected,
                rtol=2e-10,
                atol=2e-12,
                err_msg=name,
            )
        reference.step()
        simulation.step()
        self.assertEqual(graphs, torch._dynamo.utils.counters["stats"]["unique_graphs"])
        self.assertEqual(addresses, simulation.buffer_addresses())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_compiled_float32_has_fixed_storage_and_allocation(self):
        simulation = gmes.TorchSimulation(
            space=gmes.Cartesian((2, 2, 0), 2),
            geometry=_geometry(
                gmes.Dm2(
                    omega=(0.7, 1.1),
                    n_atom=(0.2, 0.4),
                    gamma=0.15,
                    rtol=1e-6,
                )
            ),
            runtime=gmes.TorchRuntimeConfig(
                device="cuda:0",
                precision="float32",
                compile_policy="compile",
                cpu_threads=1,
            ),
            dt=0.025,
        )
        rng = np.random.default_rng(126)
        simulation.load_host_fields(
            {
                name: rng.normal(size=tuple(field.shape)) * 1e-3
                for name, field in simulation.state.fields().items()
            }
        )
        simulation.step()
        torch.cuda.synchronize(simulation.device)
        addresses = simulation.buffer_addresses()
        allocated = torch.cuda.memory_allocated(simulation.device)
        simulation.advance(5)
        torch.cuda.synchronize(simulation.device)

        self.assertEqual(addresses, simulation.buffer_addresses())
        self.assertEqual(allocated, torch.cuda.memory_allocated(simulation.device))

    def test_device_mask_reports_nonconverged_targets_after_phase(self):
        _, simulation = _simulations(gmes.Dm2(rtol=-1))

        with self.assertRaisesRegex(RuntimeError, r"failed to converge.*Ex/width=1:\["):
            simulation.step()
        self.assertTrue(
            torch.all(simulation.state._dm2_iterations == DM2_MAX_ITERATIONS)
        )
        self.assertEqual(int(simulation.state.step_count), 0)

    def test_real_field_restriction_is_explicit(self):
        with self.assertRaisesRegex(gmes.TorchConfigurationError, "real fields"):
            gmes.TorchSimulation(
                space=gmes.Cartesian((2, 2, 0), 2),
                geometry=_geometry(gmes.Dm2()),
                runtime=gmes.TorchRuntimeConfig(device="cpu", cpu_threads=2),
                bloch=(0.1, 0.2, 0),
            )

    def test_iteration_distribution_is_reported_per_bucket(self):
        _, simulation = _simulations(gmes.Dm2())
        simulation.step()

        diagnostics = simulation.diagnostics()["dm2"]
        self.assertEqual(len(diagnostics), 3)
        for bucket in diagnostics:
            self.assertEqual(
                sum(bucket["iteration_distribution"].values()), bucket["targets"]
            )
            self.assertLessEqual(max(bucket["iteration_distribution"]), 100)


if __name__ == "__main__":
    unittest.main()
