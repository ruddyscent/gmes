"""Source, boundary, probe, and checkpoint coverage for Torch execution."""

import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import gmes
from gmes.torch_fdtd import FUSED_SOURCE_REPRESENTATION

_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def _geometry():
    return [gmes.DefaultMedium(gmes.Dielectric(eps_inf=1.5, mu_inf=1.1))]


def _point_sources():
    return [
        gmes.PointSource(
            gmes.Continuous(0.2, phase=0.3, width=1),
            (0, 0, 0),
            gmes.Ex,
            amp=2,
        ),
        # Native PwSource.merge() makes this the deterministic winner at Ex.
        gmes.PointSource(gmes.Bandpass(0.3, 0.1), (0, 0, 0), gmes.Jx, amp=0.4),
        gmes.PointSource(
            gmes.DifferentiatedGaussian(1, 2),
            (0, 0, 0),
            gmes.Hy,
            amp=0.7,
        ),
    ]


def _tfsf(source_time=None):
    return gmes.TotalFieldScatteredField(
        source_time or gmes.Continuous(0.2, phase=0.2, width=1),
        center=(0, 0, 0),
        size=(2, 2, 2),
        direction=(1, 0.2, 0.1),
        polarization=(0, 1, 0),
        amp=0.3,
    )


def _gaussian():
    return gmes.GaussianBeam(
        gmes.Continuous(0.2, width=0.2),
        directivity=gmes.PlusX,
        center=(0, 0, 0),
        size=(1, 1, 1),
        direction=(1, 0, 0),
        polarization=(0, 1, 0),
        waist=0.7,
        amp=0.3,
    )


def _native_and_torch(
    source_factory,
    *,
    size=(4, 4, 4),
    bloch=None,
    probes=(),
    compile_policy="eager",
):
    native = gmes.FDTD(
        gmes.Cartesian(size, 2),
        _geometry(),
        src_list=source_factory(),
        bloch=bloch,
        verbose=False,
    )
    native.init()
    simulation = gmes.TorchSimulation(
        space=gmes.Cartesian(size, 2),
        geometry=_geometry(),
        sources=source_factory(),
        probes=probes,
        runtime=gmes.TorchRuntimeConfig(
            device="cpu", cpu_threads=2, compile_policy=compile_policy
        ),
        dt=native.time_step.dt,
        bloch=bloch,
    )
    return native, simulation


def _assert_fields(test, native, simulation, steps, *, tolerance=2e-12):
    for _ in range(steps):
        native.step()
        simulation.step()
    actual = simulation.host_snapshot()
    for name in _COMPONENTS:
        np.testing.assert_allclose(
            actual[name],
            native.field[getattr(gmes, name)],
            rtol=tolerance,
            atol=tolerance,
            err_msg=name,
        )


class TorchPointSourceTest(unittest.TestCase):
    def test_time_models_currents_overlap_and_half_steps_match_native(self):
        native, simulation = _native_and_torch(_point_sources, size=(2, 2, 2))
        _assert_fields(self, native, simulation, 5)
        self.assertEqual(int(simulation.state.step_count), 5)
        source_time_address = simulation.state.source_time.data_ptr()
        simulation.advance(95)
        self.assertEqual(int(simulation.state.step_count), 100)
        self.assertEqual(float(simulation.state.source_time), 100 * simulation.plan.dt)
        self.assertEqual(simulation.state.source_time.data_ptr(), source_time_address)

    def test_compiled_material_phases_keep_source_storage_fixed(self):
        native, simulation = _native_and_torch(
            _point_sources,
            size=(2, 2, 2),
            compile_policy="compile",
        )
        addresses = simulation.buffer_addresses()
        _assert_fields(self, native, simulation, 4)
        self.assertEqual(addresses, simulation.buffer_addresses())
        self.assertEqual(float(simulation.state.source_time), 4 * simulation.plan.dt)
        self.assertEqual(
            simulation.diagnostics()["sources"]["execution_representation"],
            FUSED_SOURCE_REPRESENTATION,
        )

    def test_compile_cache_key_tracks_source_component(self):
        def build(component):
            return gmes.TorchSimulation(
                space=gmes.Cartesian((2, 2, 2), 1),
                geometry=_geometry(),
                sources=[
                    gmes.PointSource(
                        gmes.Continuous(0.2, width=1),
                        (0, 0, 0),
                        component,
                    )
                ],
                runtime=gmes.TorchRuntimeConfig(
                    device="cpu", cpu_threads=1, compile_policy="compile"
                ),
            )

        electric_x = build(gmes.Ex)
        electric_y = build(gmes.Ey)
        self.assertNotEqual(
            electric_x.compile_cache_key,
            electric_y.compile_cache_key,
        )

    def test_explicit_source_extension_lowers_once(self):
        class Extension:
            calls = 0

            def lower_torch_source(self, context):
                self.calls += 1
                self.context = context
                return (
                    gmes.TorchPointSourceRecord(
                        "Ez",
                        (1, 1, 1),
                        gmes.Continuous(0.2, phase=0.1, width=1),
                        amplitude=0.4,
                    ),
                )

        source = Extension()
        simulation = gmes.TorchSimulation(
            space=gmes.Cartesian((2, 2, 2), 1),
            geometry=_geometry(),
            sources=[source],
            runtime=gmes.TorchRuntimeConfig(device="cpu", cpu_threads=1),
        )
        simulation.step()
        source_time = gmes.Continuous(0.2, phase=0.1, width=1)
        source_time.init(False)
        expected = 0.4 * source_time.oscillator(0.5 * simulation.plan.dt)
        self.assertAlmostEqual(float(simulation.state.ez[1, 1, 1]), expected)
        self.assertEqual(source.calls, 1)
        self.assertEqual(source.context.device, simulation.device)

    def test_unsupported_callback_and_legacy_filename_fail_before_advance(self):
        class Callback:
            pass

        with self.assertRaisesRegex(TypeError, "lower_torch_source"):
            gmes.TorchSimulation(
                space=gmes.Cartesian((1, 1, 1), 1),
                geometry=_geometry(),
                sources=[Callback()],
                runtime=gmes.TorchRuntimeConfig(device="cpu", cpu_threads=1),
            )
        with self.assertRaisesRegex(ValueError, "bounded probe"):
            gmes.TorchSimulation(
                space=gmes.Cartesian((1, 1, 1), 1),
                geometry=_geometry(),
                sources=[
                    gmes.PointSource(
                        gmes.Continuous(1), (0, 0, 0), gmes.Ex, filename="x.dat"
                    )
                ],
                runtime=gmes.TorchRuntimeConfig(device="cpu", cpu_threads=1),
            )

    def test_all_yee_components_and_collapsed_axes_match_native(self):
        components = (gmes.Ex, gmes.Ey, gmes.Ez, gmes.Hx, gmes.Hy, gmes.Hz)

        def sources():
            return [
                gmes.PointSource(
                    gmes.Continuous(0.15 + 0.01 * index, width=0.8),
                    (0, 0, 0),
                    component,
                    amp=0.1 * (index + 1),
                )
                for index, component in enumerate(components)
            ]

        for size in ((3, 0, 0), (3, 3, 0), (3, 3, 3)):
            with self.subTest(size=size):
                native, simulation = _native_and_torch(sources, size=size)
                _assert_fields(self, native, simulation, 3)

    def test_source_composes_with_cpml_and_mixed_dispersive_material(self):
        def geometry():
            return [
                gmes.DefaultMedium(gmes.Dielectric(eps_inf=1.7, mu_inf=1.05)),
                gmes.Block(
                    gmes.Drude(
                        eps_inf=1.2,
                        sigma=0.01,
                        dps=(gmes.DrudePole(omega=0.7, gamma=0.03),),
                    ),
                    center=(0, 0, 0),
                    size=(1, 1, 1),
                ),
                gmes.Shell(gmes.Cpml(), thickness=0.4),
            ]

        def sources():
            return [
                gmes.PointSource(
                    gmes.Bandpass(0.2, 0.08),
                    (0, 0, 0),
                    gmes.Jz,
                    amp=0.2,
                )
            ]

        native = gmes.FDTD(
            gmes.Cartesian((3, 3, 3), 3),
            geometry(),
            src_list=sources(),
            verbose=False,
        )
        native.init()
        simulation = gmes.TorchSimulation(
            space=gmes.Cartesian((3, 3, 3), 3),
            geometry=geometry(),
            sources=sources(),
            runtime=gmes.TorchRuntimeConfig(device="cpu", cpu_threads=2),
            dt=native.time_step.dt,
        )
        _assert_fields(self, native, simulation, 2, tolerance=3e-12)
        checkpoint = simulation.checkpoint()
        _assert_fields(self, native, simulation, 3, tolerance=3e-12)
        expected = simulation.host_snapshot()
        simulation.load_checkpoint(checkpoint).advance(3)
        for name in _COMPONENTS:
            np.testing.assert_allclose(
                simulation.host_snapshot()[name],
                expected[name],
                rtol=3e-12,
                atol=3e-12,
            )


class TorchTransparentSourceTest(unittest.TestCase):
    def test_all_tfsf_faces_and_paired_real_auxiliary_match_native(self):
        native, simulation = _native_and_torch(
            lambda: [_tfsf()],
            bloch=(0.03, 0.04, 0.05),
            compile_policy="compile",
        )
        self.assertEqual(len(simulation.sources.auxiliaries), 1)
        auxiliary = simulation.sources.auxiliaries[0]
        self.assertEqual(auxiliary.device, simulation.device)
        self.assertEqual(auxiliary.dtype, simulation.dtype)
        self.assertTrue(auxiliary.state.paired_real)
        face_rows = sum(
            batch.targets.numel()
            for batch in simulation.sources.batches
            if hasattr(batch, "targets")
        )
        self.assertGreater(face_rows, 0)
        _assert_fields(self, native, simulation, 3)
        checkpoint = simulation.checkpoint()
        simulation.advance(2)
        expected_fields = simulation.host_snapshot()
        expected_auxiliary = auxiliary.host_snapshot()
        checkpoint["state"]["source_time"] = checkpoint["state"]["source_time"] + 1
        checkpoint["auxiliaries"][0]["state"]["source_time"] = (
            checkpoint["auxiliaries"][0]["state"]["source_time"] - 1
        )
        simulation.load_checkpoint(checkpoint)
        self.assertEqual(
            float(simulation.state.source_time),
            int(simulation.state.step_count) * simulation.plan.dt,
        )
        self.assertEqual(
            float(auxiliary.state.source_time),
            int(auxiliary.state.step_count) * auxiliary.plan.dt,
        )
        simulation.advance(2)
        for name in _COMPONENTS:
            np.testing.assert_array_equal(
                simulation.host_snapshot()[name], expected_fields[name]
            )
            np.testing.assert_array_equal(
                auxiliary.host_snapshot()[name], expected_auxiliary[name]
            )

    def test_tfsf_surface_intersecting_material_boundary_matches_native(self):
        def geometry():
            return [
                gmes.DefaultMedium(gmes.Dielectric(eps_inf=1.5, mu_inf=1.1)),
                gmes.Sphere(
                    gmes.Dielectric(eps_inf=2.2, mu_inf=1.3),
                    center=(1, 0, 0),
                    radius=0.8,
                ),
            ]

        native = gmes.FDTD(
            gmes.Cartesian((4, 4, 4), 2),
            geometry(),
            src_list=[_tfsf()],
            verbose=False,
        )
        native.init()
        simulation = gmes.TorchSimulation(
            space=gmes.Cartesian((4, 4, 4), 2),
            geometry=geometry(),
            sources=[_tfsf()],
            runtime=gmes.TorchRuntimeConfig(device="cpu", cpu_threads=2),
            dt=native.time_step.dt,
        )
        _assert_fields(self, native, simulation, 3)

    def test_gaussian_mode_prewarm_and_envelope_match_native(self):
        native, simulation = _native_and_torch(lambda: [_gaussian()], size=(3, 3, 3))
        self.assertGreater(int(simulation.sources.auxiliaries[0].state.step_count), 0)
        _assert_fields(self, native, simulation, 100)

        native_auxiliary = native.src_list[0].aux_fdtd.aux_fdtd
        torch_auxiliary = simulation.sources.auxiliaries[0]
        actual = torch_auxiliary.host_snapshot()
        for name in ("Ex", "Hy"):
            np.testing.assert_allclose(
                actual[name],
                native_auxiliary.field[getattr(gmes, name)],
                rtol=2e-12,
                atol=1e-12,
                err_msg=name,
            )
        step_count = int(torch_auxiliary.state.step_count)
        self.assertEqual(
            float(torch_auxiliary.state.source_time),
            step_count * torch_auxiliary.plan.dt,
        )


class TorchBoundaryTest(unittest.TestCase):
    def test_collapsed_paired_real_boundary_scratch_matches_native(self):
        bloch = (0.07, 0.11, 0.13)
        for compile_policy in ("eager", "compile"):
            with self.subTest(compile_policy=compile_policy):
                native, simulation = _native_and_torch(
                    lambda: [],
                    size=(4, 4, 0),
                    bloch=bloch,
                    compile_policy=compile_policy,
                )
                rng = np.random.default_rng(7)
                fields = {}
                for component, field in native.field.items():
                    field[...] = (
                        rng.normal(size=field.shape) * 1e-3
                        + 1j * rng.normal(size=field.shape) * 1e-3
                    )
                    fields[component.__name__] = field.copy()
                simulation.load_host_fields(fields)
                addresses = simulation.buffer_addresses()
                _assert_fields(self, native, simulation, 3)
                self.assertEqual(addresses, simulation.buffer_addresses())


class TorchProbeCheckpointTest(unittest.TestCase):
    def _simulation(self):
        return gmes.TorchSimulation(
            space=gmes.Cartesian((2, 2, 2), 2),
            geometry=_geometry(),
            sources=_point_sources(),
            probes=[
                gmes.TorchProbeSpec("Ex", (2, 2, 2), capacity=2),
                gmes.TorchProbeSpec(gmes.Hy, (0, 0, 0), capacity=3),
            ],
            runtime=gmes.TorchRuntimeConfig(device="cpu", cpu_threads=2),
            bloch=(0.01, 0.02, 0.03),
        )

    def test_bounded_ring_flush_and_versioned_resume_preserve_state(self):
        simulation = self._simulation()
        addresses = simulation.buffer_addresses()
        simulation.advance(2)
        checkpoint = simulation.checkpoint()
        expected = simulation.host_snapshot()
        with tempfile.TemporaryDirectory() as directory:
            path = simulation.save_checkpoint(Path(directory) / "restart.npz")
            with np.load(path, allow_pickle=False) as archive:
                self.assertIn("__metadata__", archive.files)
            simulation.advance(1)
            simulation.load_checkpoint_file(path)
            for name, values in expected.items():
                np.testing.assert_array_equal(simulation.host_snapshot()[name], values)
        simulation.advance(3)
        batches = simulation.flush_probes()
        self.assertEqual(
            [(len(item.times), item.dropped) for item in batches], [(2, 3), (3, 2)]
        )
        spectrum = gmes.probe_spectrum(batches[1], window="hann")
        self.assertEqual(spectrum.frequencies.shape, spectrum.amplitudes.shape)
        self.assertTrue(np.isfinite(spectrum.amplitudes).all())
        self.assertEqual(addresses, simulation.buffer_addresses())

        simulation.load_checkpoint(checkpoint)
        for name, values in expected.items():
            np.testing.assert_array_equal(simulation.host_snapshot()[name], values)
        resumed = self._simulation().load_checkpoint(checkpoint)
        simulation.advance(2)
        resumed.advance(2)
        for name in _COMPONENTS:
            torch.testing.assert_close(
                simulation.state.field(name), resumed.state.field(name)
            )

        for version in (0, 2):
            invalid = dict(checkpoint, version=version)
            with self.assertRaisesRegex(ValueError, "checkpoint version"):
                resumed.load_checkpoint(invalid)

    def test_advance_source_contains_no_host_or_output_calls(self):
        callables = (
            gmes.TorchSimulation.advance,
            gmes.TorchSimulation._update_dm2,
            gmes.torch_source.TorchSourcePlan.apply,
            gmes.torch_output._TorchProbeRing.record,
        )
        body = "\n".join(inspect.getsource(item) for item in callables)
        for forbidden in (".cpu(", ".numpy(", ".item(", "open(", "plot("):
            self.assertNotIn(forbidden, body)


if __name__ == "__main__":
    unittest.main()
