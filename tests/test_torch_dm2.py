"""Differential and state-layout tests for Torch Maxwell--Bloch execution."""

import unittest

import numpy as np
import torch

import gmes


def _geometry(material):
    return [
        gmes.DefaultMedium(gmes.Dielectric(eps_inf=1.7, mu_inf=1.05)),
        gmes.Block(material, center=(0, 0, 0), size=(2, 2, 2)),
    ]


def _simulations(material, *, precision="float64", compile_policy="eager"):
    geometry = _geometry(material)
    native = gmes.FDTD(
        gmes.Cartesian((2, 2, 2), 2),
        geometry,
        dt=0.025,
        verbose=False,
    )
    native.init()
    torch_simulation = gmes.TorchSimulation(
        space=gmes.Cartesian((2, 2, 2), 2),
        geometry=_geometry(
            gmes.Dm2(
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
        ),
        runtime=gmes.TorchRuntimeConfig(
            device="cpu",
            precision=precision,
            compile_policy=compile_policy,
            cpu_threads=2,
        ),
        dt=0.025,
    )
    return native, torch_simulation


class TorchDm2Test(unittest.TestCase):
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

    def test_complete_fields_match_native_from_nonzero_input(self):
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
        native, simulation = _simulations(material)
        rng = np.random.default_rng(120)
        fields = {}
        for component, native_field in native.field.items():
            values = rng.normal(size=native_field.shape) * 1e-3
            native_field[...] = values
            fields[component.__name__] = values.copy()
        simulation.load_host_fields(fields)

        for _ in range(2):
            native.step()
            simulation.step()

        actual = simulation.state.host_snapshot()
        for component, native_field in native.field.items():
            np.testing.assert_allclose(
                actual[component.__name__],
                native_field,
                rtol=2e-10,
                atol=2e-12,
                err_msg=component.__name__,
            )
        snapshots = {
            snapshot["component"]: snapshot
            for snapshot in simulation.dm2_state_snapshot()
        }
        for component_name in ("Ex", "Ey", "Ez"):
            component = getattr(gmes, component_name)
            updater = next(
                updater
                for updater in native.pw_material[component].values()
                if type(updater).__name__.startswith("Dm2")
            )
            indices = np.asarray(updater.oracle_indices(), dtype=np.int64).reshape(
                -1, 3
            )
            targets = np.ravel_multi_index(indices.T, native.field[component].shape)
            snapshot = snapshots[component_name]
            np.testing.assert_array_equal(snapshot["targets"], targets)
            expected_state = np.asarray(
                updater.oracle_state(), dtype=np.complex128
            ).real.reshape(-1, len(material.omega), 3)
            np.testing.assert_allclose(
                snapshot["u"], expected_state, rtol=2e-10, atol=2e-12
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

    def test_float32_matches_native_tolerance(self):
        material = gmes.Dm2(
            omega=(0.7, 1.1),
            n_atom=(0.2, 0.4),
            gamma=0.15,
            rtol=1e-6,
        )
        native, simulation = _simulations(material, precision="float32")
        rng = np.random.default_rng(123)
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
                rtol=3e-4,
                atol=3e-6,
                err_msg=component.__name__,
            )

    def test_preconditioned_nonzero_state_matches_at_fixed_capture_steps(self):
        material = gmes.Dm2(
            omega=(0.7,),
            n_atom=(0.2,),
            gamma=0.15,
            t1=2.5,
            t2=1.7,
            rtol=1e-8,
        )
        native, simulation = _simulations(material)
        rng = np.random.default_rng(125)
        for native_field in native.field.values():
            native_field[...] = rng.normal(size=native_field.shape) * 1e-3
        native.step()
        native.step()

        fields = {
            component.__name__: field.copy()
            for component, field in native.field.items()
        }
        native_state = {}
        for component_name in ("Ex", "Ey", "Ez"):
            component = getattr(gmes, component_name)
            updater = next(
                updater
                for updater in native.pw_material[component].values()
                if type(updater).__name__.startswith("Dm2")
            )
            native_state[component_name] = np.asarray(
                updater.oracle_state(), dtype=np.complex128
            ).real.reshape(-1, 1, 3)
        states = [
            native_state[bucket.metadata.component]
            for bucket in simulation.state.dm2_buckets
        ]
        simulation.load_host_fields(fields)
        simulation.load_host_dm2_state(states, step_count=2)
        self.assertTrue(all(np.any(state) for state in states))

        completed = 0
        for capture in (1, 2, 5, 20, 100):
            delta = capture - completed
            for _ in range(delta):
                native.step()
            simulation.advance(delta)
            completed = capture
            actual = simulation.state.host_snapshot()
            for component, native_field in native.field.items():
                np.testing.assert_allclose(
                    actual[component.__name__],
                    native_field,
                    rtol=2e-10,
                    atol=2e-12,
                    err_msg=f"{component.__name__} at {capture}",
                )
            snapshots = {
                snapshot["component"]: snapshot
                for snapshot in simulation.dm2_state_snapshot()
            }
            for component_name in ("Ex", "Ey", "Ez"):
                component = getattr(gmes, component_name)
                updater = next(
                    updater
                    for updater in native.pw_material[component].values()
                    if type(updater).__name__.startswith("Dm2")
                )
                expected = np.asarray(
                    updater.oracle_state(), dtype=np.complex128
                ).real.reshape(-1, 1, 3)
                np.testing.assert_allclose(
                    snapshots[component_name]["u"],
                    expected,
                    rtol=2e-10,
                    atol=2e-12,
                    err_msg=f"{component_name} state at {capture}",
                )

    def test_collapsed_axes_match_native(self):
        for size in ((2, 0, 0), (2, 2, 0)):
            with self.subTest(size=size):
                geometry = _geometry(
                    gmes.Dm2(
                        omega=(0.7,),
                        n_atom=(0.2,),
                        gamma=0.15,
                        rtol=1e-9,
                    )
                )
                native = gmes.FDTD(
                    gmes.Cartesian(size, 2),
                    geometry,
                    dt=0.025,
                    verbose=False,
                )
                native.init()
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
                    runtime=gmes.TorchRuntimeConfig(device="cpu", cpu_threads=2),
                    dt=0.025,
                )
                rng = np.random.default_rng(124)
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
                        rtol=2e-10,
                        atol=2e-12,
                        err_msg=component.__name__,
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

    def test_compiled_fullgraph_matches_native(self):
        material = gmes.Dm2(
            omega=(0.7,),
            n_atom=(0.2,),
            gamma=0.15,
            t1=2.5,
            t2=1.7,
            rtol=1e-8,
        )
        native, simulation = _simulations(material, compile_policy="compile")
        rng = np.random.default_rng(121)
        fields = {}
        for component, native_field in native.field.items():
            values = rng.normal(size=native_field.shape) * 1e-3
            native_field[...] = values
            fields[component.__name__] = values.copy()
        simulation.load_host_fields(fields)

        native.step()
        simulation.step()
        graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        addresses = simulation.buffer_addresses()

        actual = simulation.state.host_snapshot()
        for component, native_field in native.field.items():
            np.testing.assert_allclose(
                actual[component.__name__],
                native_field,
                rtol=2e-10,
                atol=2e-12,
                err_msg=component.__name__,
            )
        native.step()
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
        self.assertTrue(torch.all(simulation.state._dm2_iterations == 100))
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
