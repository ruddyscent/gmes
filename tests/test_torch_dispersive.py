"""Oracle and storage tests for Torch dispersive tensor buckets."""

import json
import unittest
from pathlib import Path

import numpy as np
import torch

import gmes

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


def _seed_native(native, *, complex_fields):
    rng = np.random.default_rng(119)
    fields = {}
    for component, field in native.field.items():
        values = rng.normal(size=field.shape) * 1e-3
        if complex_fields:
            values = values + 1j * rng.normal(size=field.shape) * 1e-3
        field[...] = values
        fields[component.__name__] = values.copy()
    return fields


def _native_and_torch(
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
    native = gmes.FDTD(
        gmes.Cartesian(size, resolution),
        _geometry(model, poles=poles, points=points),
        bloch=bloch,
        verbose=False,
    )
    native.init()
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
    fields = _seed_native(native, complex_fields=bloch is not None)
    simulation.load_host_fields(fields)
    return native, simulation


def _assert_fields(test, native, simulation, *, model, complex_fields):
    if complex_fields:
        tolerance_name = (
            "complex64" if simulation.dtype == torch.float32 else "complex128"
        )
    else:
        tolerance_name = "float32" if simulation.dtype == torch.float32 else "float64"
    tolerance = _TOLERANCES[model][tolerance_name]
    actual = simulation.state.host_snapshot()
    for component, field in native.field.items():
        test.assertTrue(np.all(np.isfinite(actual[component.__name__])))
        np.testing.assert_allclose(
            actual[component.__name__],
            field,
            rtol=tolerance["rtol"],
            atol=tolerance["atol"],
            err_msg=f"{model}:{component.__name__}",
        )


class DispersiveOracleTest(unittest.TestCase):
    def test_all_families_match_capture_steps_from_nonzero_fields_and_state(self):
        for model in ("drude", "lorentz", "dcp-ade", "dcp-plrc", "dcp-rc"):
            with self.subTest(model=model):
                native, simulation = _native_and_torch(model)
                native.step()
                native.step()
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
                        native.step()
                    _assert_fields(
                        self,
                        native,
                        simulation,
                        model=model,
                        complex_fields=False,
                    )
                    completed = capture

    def test_paired_real_complex_recurrences_match_native(self):
        bloch = (0.07, 0.11, 0.13)
        for model in ("drude", "lorentz", "dcp-ade", "dcp-plrc", "dcp-rc"):
            with self.subTest(model=model):
                native, simulation = _native_and_torch(model, bloch=bloch)
                simulation.advance(5)
                for _ in range(5):
                    native.step()
                _assert_fields(
                    self,
                    native,
                    simulation,
                    model=model,
                    complex_fields=True,
                )
                for name, value in simulation.state.named_buffers():
                    self.assertFalse(value.is_complex(), name)

    def test_forced_policies_are_exactly_equal(self):
        results = {}
        fields = None
        for policy in ("dense", "compact", "tiled"):
            _, simulation = _native_and_torch("dcp-plrc", policy=policy)
            if fields is None:
                rng = np.random.default_rng(219)
                fields = {
                    name: rng.normal(size=tuple(field.shape)) * 1e-3
                    for name, field in simulation.state.fields().items()
                }
            simulation.load_host_fields(fields).advance(20)
            results[policy] = simulation.state.host_snapshot()
        for component in _COMPONENTS:
            np.testing.assert_array_equal(
                results["dense"][component], results["compact"][component]
            )
            np.testing.assert_array_equal(
                results["dense"][component], results["tiled"][component]
            )

    def test_collapsed_1d_2d_and_3d_fields_match_native(self):
        for size in ((4, 0, 0), (4, 3, 0), (2, 2, 2)):
            for model in ("drude", "dcp-rc"):
                with self.subTest(size=size, model=model):
                    native, simulation = _native_and_torch(model, size=size)
                    simulation.advance(5)
                    for _ in range(5):
                        native.step()
                    _assert_fields(
                        self,
                        native,
                        simulation,
                        model=model,
                        complex_fields=False,
                    )

    def test_float32_all_families_match_performance_tolerance(self):
        for model in ("drude", "lorentz", "dcp-ade", "dcp-plrc", "dcp-rc"):
            with self.subTest(model=model):
                native, simulation = _native_and_torch(model, precision="float32")
                simulation.advance(20)
                for _ in range(20):
                    native.step()
                _assert_fields(
                    self,
                    native,
                    simulation,
                    model=model,
                    complex_fields=False,
                )

    def test_compiled_bulk_phases_preserve_dispersive_oracle_and_storage(self):
        torch._dynamo.reset()
        native, simulation = _native_and_torch("dcp-plrc", compile_policy="compile")
        simulation.advance(5)
        for _ in range(5):
            native.step()
        _assert_fields(
            self,
            native,
            simulation,
            model="dcp-plrc",
            complex_fields=False,
        )
        graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        addresses = simulation.buffer_addresses()
        simulation.advance(5)
        self.assertEqual(graphs, torch._dynamo.utils.counters["stats"]["unique_graphs"])
        self.assertEqual(addresses, simulation.buffer_addresses())

    def test_zero_width_conductive_variants_match_native(self):
        for model in ("drude", "lorentz", "dcp-ade", "dcp-plrc", "dcp-rc"):
            with self.subTest(model=model):
                native, simulation = _native_and_torch(model, poles=0, points=0)
                simulation.advance(20)
                for _ in range(20):
                    native.step()
                _assert_fields(
                    self,
                    native,
                    simulation,
                    model=model,
                    complex_fields=False,
                )

    def test_long_run_pulse_spectrum_boundary_energy_and_stability(self):
        native, simulation = _native_and_torch(
            "dcp-plrc", size=(12, 0, 0), resolution=4
        )
        fields = {}
        for component, field in native.field.items():
            values = np.zeros(field.shape)
            if component.__name__ == "Ey":
                x = np.arange(field.shape[0])
                values[:, 0, 0] = (
                    np.exp(-0.5 * ((x - field.shape[0] * 0.28) / 2.0) ** 2) * 1e-3
                )
            field[...] = values
            fields[component.__name__] = values.copy()
        simulation.load_host_fields(fields)

        completed = 0
        for capture in (100, 500):
            increment = capture - completed
            simulation.advance(increment)
            for _ in range(increment):
                native.step()
            actual = simulation.state.host_snapshot()
            _assert_fields(
                self,
                native,
                simulation,
                model="dcp-plrc",
                complex_fields=False,
            )
            native_line = next(
                field[:, 0, 0]
                for component, field in native.field.items()
                if component.__name__ == "Ey"
            )
            torch_line = actual["Ey"][:, 0, 0]
            np.testing.assert_allclose(
                np.abs(np.fft.rfft(torch_line)),
                np.abs(np.fft.rfft(native_line)),
                rtol=1e-11,
                atol=1e-12,
            )
            native_energy = sum(
                float(np.sum(np.abs(field) ** 2)) for field in native.field.values()
            )
            torch_energy = sum(
                float(np.sum(np.abs(actual[name]) ** 2)) for name in _COMPONENTS
            )
            self.assertAlmostEqual(torch_energy, native_energy, places=16)
            self.assertLess(float(np.max(np.abs(torch_line))), 1e-3)
            completed = capture

    def test_mixed_families_share_one_complete_field_execution(self):
        native = gmes.FDTD(
            gmes.Cartesian((8, 2, 2), 2), _mixed_geometry(), verbose=False
        )
        native.init()
        simulation = gmes.TorchSimulation(
            space=gmes.Cartesian((8, 2, 2), 2),
            geometry=_mixed_geometry(),
            runtime=gmes.TorchRuntimeConfig(device="cpu", cpu_threads=2),
        )
        simulation.load_host_fields(_seed_native(native, complex_fields=False)).advance(
            20
        )
        for _ in range(20):
            native.step()
        _assert_fields(
            self,
            native,
            simulation,
            model="mixed",
            complex_fields=False,
        )
        self.assertEqual(
            {item.model for item in simulation.plan.dispersive_buckets},
            {"drude", "lorentz", "dcp-ade", "dcp-plrc", "dcp-rc"},
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_eager_float32_all_families_match_native(self):
        for model in ("drude", "lorentz", "dcp-ade", "dcp-plrc", "dcp-rc"):
            with self.subTest(model=model):
                native, simulation = _native_and_torch(
                    model, precision="float32", device="cuda:0"
                )
                simulation.advance(5)
                for _ in range(5):
                    native.step()
                _assert_fields(
                    self,
                    native,
                    simulation,
                    model=model,
                    complex_fields=False,
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_compiled_float32_has_stable_storage_and_allocation(self):
        torch._dynamo.reset()
        native, simulation = _native_and_torch(
            "dcp-plrc",
            precision="float32",
            compile_policy="compile",
            device="cuda:0",
        )
        simulation.advance(5)
        for _ in range(5):
            native.step()
        _assert_fields(
            self,
            native,
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
        native = gmes.FDTD(
            gmes.Cartesian((8, 6, 0), 4),
            _mixed_pml_geometry(),
            verbose=False,
        )
        native.init()
        simulation = gmes.TorchSimulation(
            space=gmes.Cartesian((8, 6, 0), 4),
            geometry=_mixed_pml_geometry(),
            runtime=gmes.TorchRuntimeConfig(device="cpu", cpu_threads=2),
        )
        simulation.load_host_fields(_seed_native(native, complex_fields=False))
        completed = 0
        for capture in _CAPTURE_STEPS:
            increment = capture - completed
            simulation.advance(increment)
            for _ in range(increment):
                native.step()
            _assert_fields(
                self,
                native,
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
        self.assertTrue(all(len(bucket.region_keys) == 6 for bucket in pml_buckets))


class DispersiveStorageTest(unittest.TestCase):
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
