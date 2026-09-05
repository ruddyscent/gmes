"""Source, boundary, probe, and checkpoint coverage for Torch execution."""

import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import gmes
from gmes.source import TfsfFaceRule
from gmes.torch_fdtd import FUSED_SOURCE_REPRESENTATION
from gmes.torch_source import (
    TorchSourceLoweringContext,
    TorchTransparentBatch,
    prepare_sources,
)

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
        # The later source deterministically owns this overlapping Ex target.
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


def _gaussian(*, width=0.2):
    return gmes.GaussianBeam(
        gmes.Continuous(0.2, width=width),
        directivity=gmes.PlusX,
        center=(0, 0, 0),
        size=(1, 1, 1),
        direction=(1, 0, 0),
        polarization=(0, 1, 0),
        waist=0.7,
        amp=0.3,
    )


def _torch_simulation(
    source_factory,
    *,
    size=(4, 4, 4),
    bloch=None,
    probes=(),
    compile_policy="eager",
    compile_mode="default",
    device="cpu",
    precision="float64",
    geometry_factory=_geometry,
):
    return gmes.TorchSimulation(
        space=gmes.Cartesian(size, 2),
        geometry=geometry_factory(),
        sources=source_factory(),
        probes=probes,
        runtime=gmes.TorchRuntimeConfig(
            device=device,
            precision=precision,
            cpu_threads=2,
            compile_policy=compile_policy,
            compile_mode=compile_mode,
        ),
        bloch=bloch,
    )


def _advance_and_assert_finite(test, simulation, steps):
    simulation.advance(steps)
    for name, values in simulation.host_snapshot().items():
        test.assertTrue(np.isfinite(values).all(), name)


def _continuous_value(time, *, frequency, phase=0.0, width=1.0):
    rise = np.sin(0.5 * np.pi * time / width) ** 2 if time < width else 1.0
    return rise * np.cos(2 * np.pi * frequency * time + phase)


def _bandpass_value(time, *, frequency, fwidth, phase=0.0, sharpness=10.0):
    width = 1.0 / fwidth
    offset = time - width * sharpness
    cutoff = 2 * width * sharpness
    if abs(offset) > cutoff:
        return 0.0
    envelope = np.exp(-0.5 * (offset / width) ** 2)
    return (
        -envelope
        * np.sin(2 * np.pi * frequency * time + phase)
        / (2 * np.pi * frequency)
    )


def _differentiated_gaussian_value(time, *, half_width, delay):
    offset = (time - delay) / half_width
    return -2 * offset * np.exp(-(offset**2))


def _yee_shapes(size, resolution):
    cells = tuple(max(1, round(length * resolution)) for length in size)
    nx, ny, nz = cells
    return {
        "Ex": (nx, ny + 1, nz + 1),
        "Ey": (nx + 1, ny, nz + 1),
        "Ez": (nx + 1, ny + 1, nz),
        "Hx": (nx, ny + 1, nz + 1),
        "Hy": (nx + 1, ny, nz + 1),
        "Hz": (nx + 1, ny + 1, nz),
    }


def _yee_index(component, center, size, resolution):
    spacing = 1.0 / resolution
    half = tuple(0.5 * length if length else 0.5 * spacing for length in size)
    offsets = {
        "Ex": (-0.5, 0.0, 0.0),
        "Ey": (0.0, -0.5, 0.0),
        "Ez": (0.0, 0.0, -0.5),
        "Hx": (0.0, 0.5, 0.5),
        "Hy": (0.5, 0.0, 0.5),
        "Hz": (0.5, 0.5, 0.0),
    }[component]
    component_axis = "xyz".index(component[1].lower())
    index = []
    for axis, (value, extent, offset) in enumerate(
        zip(center, half, offsets, strict=True)
    ):
        if size[axis] == 0:
            index.append(
                1 if component.startswith("H") and axis != component_axis else 0
            )
        else:
            index.append(int(np.floor((value + extent) / spacing + offset + 0.5)))
    return tuple(index)


def _yee_coordinate(component, target, size, resolution):
    spacing = 1.0 / resolution
    half = tuple(0.5 * length if length else 0.5 * spacing for length in size)
    offsets = {
        "Ex": (0.5, 0.0, 0.0),
        "Ey": (0.0, 0.5, 0.0),
        "Ez": (0.0, 0.0, 0.5),
        "Hx": (0.0, -0.5, -0.5),
        "Hy": (-0.5, 0.0, -0.5),
        "Hz": (-0.5, -0.5, 0.0),
    }[component]
    return tuple(
        (index + offset) * spacing - extent
        for index, offset, extent in zip(target, offsets, half, strict=True)
    )


class TorchPointSourceTest(unittest.TestCase):
    def test_overwrite_current_material_scaling_and_last_wins_are_exact(self):
        size = (4, 4, 4)
        resolution = 2
        baseline = 0.125

        def geometry():
            return [
                gmes.DefaultMedium(gmes.Dielectric(eps_inf=1.5, mu_inf=1.1)),
                gmes.Block(
                    gmes.Dielectric(eps_inf=2.5, mu_inf=1.7),
                    center=(0, 0, 0),
                    size=(2, 2, 2),
                ),
            ]

        def sources():
            return [
                gmes.PointSource(
                    gmes.Continuous(0.2, phase=0.3, width=1),
                    (-1.5, 0, 0),
                    gmes.Ex,
                    amp=2.0,
                ),
                gmes.PointSource(
                    gmes.Bandpass(0.3, 0.1),
                    (0, 0, 0),
                    gmes.Jx,
                    amp=0.4,
                ),
                gmes.PointSource(
                    gmes.DifferentiatedGaussian(1, 2),
                    (-1.5, 0, 0),
                    gmes.Hy,
                    amp=0.7,
                ),
                gmes.PointSource(
                    gmes.Continuous(0.18, phase=-0.1, width=0.9),
                    (0, 0, 0),
                    gmes.My,
                    amp=0.6,
                ),
                gmes.PointSource(
                    gmes.Continuous(0.11, phase=0.4, width=0.7),
                    (0.5, 0, 0),
                    gmes.Ey,
                    amp=4.0,
                ),
                gmes.PointSource(
                    gmes.Bandpass(0.35, 0.12),
                    (0.5, 0, 0),
                    gmes.Jy,
                    amp=0.25,
                ),
            ]

        simulation = _torch_simulation(
            sources,
            size=size,
            geometry_factory=geometry,
        )
        shapes = _yee_shapes(size, resolution)
        self.assertEqual(dict(simulation.plan.shapes), shapes)
        fields = {
            component: np.full(shape, baseline, dtype=np.float64)
            for component, shape in shapes.items()
        }
        simulation.load_host_fields(fields)
        dt = 0.99 * np.sqrt(1.5 * 1.1) / np.sqrt(3 * resolution**2)
        self.assertAlmostEqual(simulation.plan.dt, dt, places=15)
        electric_time = 0.5 * dt
        magnetic_time = dt
        simulation.sources.apply(
            simulation,
            electric=True,
            time=torch.tensor(electric_time, dtype=simulation.dtype),
            transparent_time=torch.tensor(0.0, dtype=simulation.dtype),
        )
        simulation.sources.apply(
            simulation,
            electric=False,
            time=torch.tensor(magnetic_time, dtype=simulation.dtype),
            transparent_time=torch.tensor(magnetic_time, dtype=simulation.dtype),
        )
        snapshot = simulation.host_snapshot()

        expected = (
            (
                "Ex",
                (-1.5, 0, 0),
                2.0
                * _continuous_value(electric_time, frequency=0.2, phase=0.3, width=1),
            ),
            (
                "Ex",
                (0, 0, 0),
                baseline
                - dt
                * 0.4
                * _bandpass_value(electric_time, frequency=0.3, fwidth=0.1)
                / 2.5,
            ),
            (
                "Hy",
                (-1.5, 0, 0),
                0.7
                * _differentiated_gaussian_value(magnetic_time, half_width=1, delay=2),
            ),
            (
                "Hy",
                (0, 0, 0),
                baseline
                - dt
                * 0.6
                * _continuous_value(
                    magnetic_time, frequency=0.18, phase=-0.1, width=0.9
                )
                / 1.7,
            ),
            (
                "Ey",
                (0.5, 0, 0),
                baseline
                - dt
                * 0.25
                * _bandpass_value(electric_time, frequency=0.35, fwidth=0.12)
                / 2.5,
            ),
        )
        for component, center, value in expected:
            target = _yee_index(component, center, size, resolution)
            with self.subTest(component=component, center=center):
                self.assertAlmostEqual(snapshot[component][target], value, places=14)

        ey_batch = next(
            batch for batch in simulation.sources.batches if batch.component == "Ey"
        )
        self.assertEqual(ey_batch.overwrite_targets.numel(), 0)
        self.assertEqual(ey_batch.additive_targets.numel(), 1)

    def test_time_models_currents_overlap_and_half_steps_are_finite(self):
        simulation = _torch_simulation(_point_sources, size=(2, 2, 2))
        _advance_and_assert_finite(self, simulation, 5)
        self.assertTrue(
            any(
                np.count_nonzero(field) for field in simulation.host_snapshot().values()
            )
        )
        self.assertEqual(int(simulation.state.step_count), 5)
        source_time_address = simulation.state.source_time.data_ptr()
        simulation.advance(95)
        self.assertEqual(int(simulation.state.step_count), 100)
        self.assertEqual(float(simulation.state.source_time), 100 * simulation.plan.dt)
        self.assertEqual(simulation.state.source_time.data_ptr(), source_time_address)

    def test_compiled_material_phases_keep_source_storage_fixed(self):
        reference = _torch_simulation(_point_sources, size=(2, 2, 2))
        simulation = _torch_simulation(
            _point_sources,
            size=(2, 2, 2),
            compile_policy="compile",
        )
        addresses = simulation.buffer_addresses()
        reference.advance(4)
        simulation.advance(4)
        for name in _COMPONENTS:
            np.testing.assert_allclose(
                simulation.host_snapshot()[name],
                reference.host_snapshot()[name],
                rtol=2e-15,
                atol=2e-15,
            )
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

    def test_all_yee_components_and_collapsed_axes_are_driven(self):
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
                simulation = _torch_simulation(sources, size=size)
                shapes = _yee_shapes(size, 2)
                self.assertEqual(dict(simulation.plan.shapes), shapes)
                zeros = {
                    component: np.zeros(shape, dtype=np.float64)
                    for component, shape in shapes.items()
                }
                simulation.load_host_fields(zeros)
                electric_time = 0.5 * simulation.plan.dt
                magnetic_time = simulation.plan.dt
                simulation.sources.apply(
                    simulation,
                    electric=True,
                    time=torch.tensor(electric_time, dtype=simulation.dtype),
                    transparent_time=torch.tensor(0.0, dtype=simulation.dtype),
                )
                simulation.sources.apply(
                    simulation,
                    electric=False,
                    time=torch.tensor(magnetic_time, dtype=simulation.dtype),
                    transparent_time=torch.tensor(
                        magnetic_time, dtype=simulation.dtype
                    ),
                )
                snapshot = simulation.host_snapshot()
                for index, component in enumerate(_COMPONENTS):
                    time = electric_time if component.startswith("E") else magnetic_time
                    expected = (
                        0.1
                        * (index + 1)
                        * _continuous_value(
                            time,
                            frequency=0.15 + 0.01 * index,
                            width=0.8,
                        )
                    )
                    target = _yee_index(component, (0, 0, 0), size, 2)
                    with self.subTest(size=size, component=component):
                        self.assertAlmostEqual(
                            snapshot[component][target], expected, places=14
                        )

                simulation.load_host_fields(zeros)
                _advance_and_assert_finite(self, simulation, 3)
                snapshot = simulation.host_snapshot()
                for component in _COMPONENTS:
                    self.assertGreater(np.count_nonzero(snapshot[component]), 0)

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

        simulation = gmes.TorchSimulation(
            space=gmes.Cartesian((3, 3, 3), 3),
            geometry=geometry(),
            sources=sources(),
            runtime=gmes.TorchRuntimeConfig(device="cpu", cpu_threads=2),
        )
        _advance_and_assert_finite(self, simulation, 2)
        checkpoint = simulation.checkpoint()
        _advance_and_assert_finite(self, simulation, 3)
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
    def test_tfsf_all_24_component_face_coefficients_apply_exactly(self):
        source = _tfsf()
        simulation = _torch_simulation(lambda: [source])
        auxiliary = simulation.sources.auxiliaries[0]
        context = TorchSourceLoweringContext(
            simulation.space,
            simulation.geom_tree,
            False,
            simulation.dtype,
            simulation.device,
            simulation.plan.dt,
        )
        auxiliary_spec = source.lower_torch_source(context)[0].auxiliary_spec
        component_plans = tuple(
            SimpleNamespace(
                name=name,
                shape=shape,
                ownership=np.zeros(shape, dtype=np.int16),
            )
            for name, shape in simulation.plan.shapes.items()
        )
        faces = {
            "Ex": (
                (gmes.MinusY, -1),
                (gmes.PlusY, 1),
                (gmes.MinusZ, 1),
                (gmes.PlusZ, -1),
            ),
            "Ey": (
                (gmes.MinusZ, -1),
                (gmes.PlusZ, 1),
                (gmes.MinusX, 1),
                (gmes.PlusX, -1),
            ),
            "Ez": (
                (gmes.MinusX, -1),
                (gmes.PlusX, 1),
                (gmes.MinusY, 1),
                (gmes.PlusY, -1),
            ),
            "Hx": (
                (gmes.MinusY, 1),
                (gmes.PlusY, -1),
                (gmes.MinusZ, -1),
                (gmes.PlusZ, 1),
            ),
            "Hy": (
                (gmes.MinusZ, 1),
                (gmes.PlusZ, -1),
                (gmes.MinusX, -1),
                (gmes.PlusX, 1),
            ),
            "Hz": (
                (gmes.MinusX, 1),
                (gmes.PlusX, -1),
                (gmes.MinusY, -1),
                (gmes.PlusY, 1),
            ),
        }
        face_axes = {
            gmes.MinusX: 0,
            gmes.PlusX: 0,
            gmes.MinusY: 1,
            gmes.PlusY: 1,
            gmes.MinusZ: 2,
            gmes.PlusZ: 2,
        }
        checked = 0
        for component, component_faces in faces.items():
            for ordinal, (face, sign) in enumerate(component_faces):
                with self.subTest(component=component, face=face.__name__):
                    amplitude = 0.3 + 0.01 * ordinal
                    material = 2.5 if component.startswith("E") else 1.7
                    coefficient = (
                        sign
                        * simulation.plan.dt
                        * amplitude
                        / (material * simulation.plan.dr[face_axes[face]])
                    )
                    rule = TfsfFaceRule(
                        component,
                        (0, 0, 0),
                        "Hy" if component.startswith("E") else "Ex",
                        (0, 0, 0),
                        (0, 0, 1),
                        0.25,
                        0.75,
                        coefficient,
                        auxiliary_spec,
                    )
                    original_lower = source.lower_torch_source
                    source.lower_torch_source = lambda _context: (rule,)
                    try:
                        prepared = prepare_sources(
                            [source],
                            context=context,
                            component_plans=component_plans,
                        )
                    finally:
                        source.lower_torch_source = original_lower
                    batch = TorchTransparentBatch(
                        component,
                        prepared.transparent[component][id(auxiliary_spec)],
                        auxiliary=auxiliary,
                        gaussian_width=None,
                        paired_real=False,
                        device=simulation.device,
                        dtype=simulation.dtype,
                    )
                    np.testing.assert_allclose(
                        batch.weights.cpu().numpy(),
                        [[0.25 * coefficient, 0.75 * coefficient]],
                        rtol=0,
                        atol=1e-15,
                    )
                    np.testing.assert_array_equal(batch.targets.cpu(), [0])
                    np.testing.assert_array_equal(
                        batch.samples.cpu(),
                        [[0, 1]],
                    )

                    auxiliary_field = auxiliary.state.field(
                        batch.auxiliary_component
                    ).reshape(-1)
                    auxiliary_field.zero_()
                    auxiliary_field[int(batch.samples[0, 0])] = 2.0
                    auxiliary_field[int(batch.samples[0, 1])] = -1.0
                    outer = torch.zeros_like(simulation.state.field(component))
                    batch.apply(outer, torch.zeros((), dtype=simulation.dtype))
                    self.assertAlmostEqual(
                        float(outer.reshape(-1)[0]),
                        coefficient * (0.25 * 2.0 - 0.75),
                        places=14,
                    )
                    checked += 1
        self.assertEqual(checked, 24)

    def test_gaussian_spatial_weights_match_radial_equation(self):
        size = (3, 3, 3)
        resolution = 2

        def plane_wave():
            return [
                gmes.TotalFieldScatteredField(
                    gmes.Continuous(0.2, width=0.2),
                    center=(0, 0, 0),
                    size=(1, 1, 1),
                    direction=(1, 0, 0),
                    polarization=(0, 1, 0),
                    amp=0.3,
                )
            ]

        gaussian = _torch_simulation(lambda: [_gaussian()], size=size)
        uniform = _torch_simulation(plane_wave, size=size)
        gaussian_batches = {
            batch.component: batch
            for batch in gaussian.sources.batches
            if isinstance(batch, TorchTransparentBatch)
            and bool(torch.count_nonzero(batch.weights))
        }
        uniform_batches = {
            batch.component: batch
            for batch in uniform.sources.batches
            if isinstance(batch, TorchTransparentBatch)
            and bool(torch.count_nonzero(batch.weights))
        }
        self.assertEqual(set(gaussian_batches), {"Ey", "Hz"})
        for component, gaussian_batch in gaussian_batches.items():
            uniform_batch = uniform_batches[component]
            uniform_rows = {
                int(target): row
                for row, target in enumerate(uniform_batch.targets.cpu().numpy())
            }
            for row, target_tensor in enumerate(gaussian_batch.targets):
                target = int(target_tensor)
                with self.subTest(component=component, target=target):
                    uniform_row = uniform_rows[target]
                    np.testing.assert_array_equal(
                        gaussian_batch.samples[row].cpu(),
                        uniform_batch.samples[uniform_row].cpu(),
                    )
                    gaussian_weights = gaussian_batch.weights[row].cpu().numpy()
                    uniform_weights = uniform_batch.weights[uniform_row].cpu().numpy()
                    active = uniform_weights != 0
                    target_index = np.unravel_index(
                        target, gaussian.plan.shapes[component]
                    )
                    _, y, z = _yee_coordinate(
                        component,
                        target_index,
                        size,
                        resolution,
                    )
                    expected_mode = np.exp(-((y * y + z * z) / 0.7**2))
                    np.testing.assert_allclose(
                        gaussian_weights[active] / uniform_weights[active],
                        expected_mode,
                        rtol=2e-14,
                        atol=2e-14,
                    )

    def test_float32_tfsf_uses_double_auxiliary_and_fixed_cast_storage(self):
        simulation = _torch_simulation(lambda: [_tfsf()], precision="float32")
        self.assertEqual(simulation.dtype, torch.float32)
        self.assertEqual(len(simulation.sources.auxiliaries), 1)
        auxiliary = simulation.sources.auxiliaries[0]
        self.assertEqual(auxiliary.dtype, torch.float64)
        self.assertEqual(auxiliary.runtime.precision, "float64")
        self.assertEqual(
            simulation.diagnostics()["sources"]["auxiliary_precisions"],
            ("float64",),
        )
        for label, module in (
            ("plan", auxiliary.plan),
            ("state", auxiliary.state),
            ("source", auxiliary.sources),
        ):
            for name, value in module.named_buffers():
                if value.is_floating_point():
                    with self.subTest(module=label, buffer=name):
                        self.assertEqual(value.dtype, torch.float64)

        transparent = tuple(
            batch
            for batch in simulation.sources.batches
            if isinstance(batch, TorchTransparentBatch)
        )
        self.assertTrue(transparent)
        for batch in transparent:
            self.assertEqual(batch.weights.dtype, torch.float64)
            self.assertEqual(batch._sample_values.dtype, torch.float64)
            self.assertEqual(batch._values.dtype, torch.float64)
            self.assertEqual(batch._outer_values.dtype, torch.float32)

        addresses = simulation.buffer_addresses()
        simulation.advance(100)
        self.assertEqual(addresses, simulation.buffer_addresses())
        for state in (simulation.state, auxiliary.state):
            expected_time = state.step_count.to(state.source_time.dtype).mul(
                state.time_step
            )
            self.assertTrue(torch.equal(state.source_time, expected_time))

        actual = simulation.host_snapshot()
        for name in _COMPONENTS:
            self.assertTrue(np.isfinite(actual[name]).all(), name)
        self.assertTrue(any(np.count_nonzero(value) for value in actual.values()))
        actual_auxiliary = auxiliary.host_snapshot()
        for name in ("Ex", "Hy"):
            self.assertTrue(np.isfinite(actual_auxiliary[name]).all(), name)
            self.assertGreater(np.count_nonzero(actual_auxiliary[name]), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_float32_tfsf_eager_and_graph_keep_double_auxiliary(self):
        torch._dynamo.reset()
        modes = (
            ("eager", "default", False),
            ("compile", "reduce-overhead", True),
        )
        for compile_policy, compile_mode, capture_graphs in modes:
            with self.subTest(compile_policy=compile_policy, compile_mode=compile_mode):
                simulation = _torch_simulation(
                    lambda: [_tfsf()],
                    device="cuda:0",
                    precision="float32",
                    compile_policy=compile_policy,
                    compile_mode=compile_mode,
                )
                auxiliary = simulation.sources.auxiliaries[0]
                self.assertEqual(simulation.dtype, torch.float32)
                self.assertEqual(auxiliary.dtype, torch.float64)
                addresses = simulation.buffer_addresses()
                if capture_graphs:
                    simulation.capture_cuda_graphs()
                    self.assertTrue(simulation.diagnostics()["cuda_graph_regions"])
                simulation.advance(100)
                torch.cuda.synchronize(simulation.device)
                self.assertEqual(addresses, simulation.buffer_addresses())

                actual = simulation.host_snapshot()
                for name in _COMPONENTS:
                    self.assertTrue(np.isfinite(actual[name]).all(), name)
                actual_auxiliary = auxiliary.host_snapshot()
                for name in ("Ex", "Hy"):
                    self.assertTrue(np.isfinite(actual_auxiliary[name]).all(), name)

    def test_all_tfsf_faces_and_paired_real_auxiliary_replay_exactly(self):
        simulation = _torch_simulation(
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
        _advance_and_assert_finite(self, simulation, 3)
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

    def test_tfsf_surface_intersecting_material_boundary_is_finite(self):
        def geometry():
            return [
                gmes.DefaultMedium(gmes.Dielectric(eps_inf=1.5, mu_inf=1.1)),
                gmes.Sphere(
                    gmes.Dielectric(eps_inf=2.2, mu_inf=1.3),
                    center=(1, 0, 0),
                    radius=0.8,
                ),
            ]

        simulation = gmes.TorchSimulation(
            space=gmes.Cartesian((4, 4, 4), 2),
            geometry=geometry(),
            sources=[_tfsf()],
            runtime=gmes.TorchRuntimeConfig(device="cpu", cpu_threads=2),
        )
        uniform = _torch_simulation(lambda: [_tfsf()])
        uniform_batches = {
            batch.component: batch
            for batch in uniform.sources.batches
            if isinstance(batch, TorchTransparentBatch)
        }
        inside = outside = 0
        for batch in simulation.sources.batches:
            if not isinstance(batch, TorchTransparentBatch):
                continue
            uniform_batch = uniform_batches[batch.component]
            uniform_rows = {
                int(target): row
                for row, target in enumerate(uniform_batch.targets.cpu().numpy())
            }
            for row, target_tensor in enumerate(batch.targets):
                target = int(target_tensor)
                reference_row = uniform_rows[target]
                np.testing.assert_array_equal(
                    batch.samples[row].cpu(),
                    uniform_batch.samples[reference_row].cpu(),
                )
                point = _yee_coordinate(
                    batch.component,
                    np.unravel_index(target, simulation.plan.shapes[batch.component]),
                    (4, 4, 4),
                    2,
                )
                in_sphere = (
                    np.linalg.norm(np.asarray(point) - np.asarray((1, 0, 0))) <= 0.8
                )
                outer = uniform_batch.weights[reference_row].cpu().numpy()
                actual = batch.weights[row].cpu().numpy()
                active = outer != 0
                if in_sphere:
                    expected_ratio = (
                        1.5 / 2.2 if batch.component.startswith("E") else 1.1 / 1.3
                    )
                    inside += 1
                else:
                    expected_ratio = 1.0
                    outside += 1
                np.testing.assert_allclose(
                    actual[active] / outer[active],
                    expected_ratio,
                    rtol=2e-14,
                    atol=2e-14,
                )
        self.assertGreater(inside, 0)
        self.assertGreater(outside, 0)
        _advance_and_assert_finite(self, simulation, 3)
        self.assertTrue(
            any(
                np.count_nonzero(value) for value in simulation.host_snapshot().values()
            )
        )

    def test_gaussian_mode_prewarm_and_envelope_are_finite(self):
        simulation = _torch_simulation(lambda: [_gaussian()], size=(3, 3, 3))
        self.assertGreater(int(simulation.sources.auxiliaries[0].state.step_count), 0)
        _advance_and_assert_finite(self, simulation, 100)

        torch_auxiliary = simulation.sources.auxiliaries[0]
        actual = torch_auxiliary.host_snapshot()
        for name in ("Ex", "Hy"):
            self.assertTrue(np.isfinite(actual[name]).all(), name)
            self.assertGreater(np.count_nonzero(actual[name]), 0)
        step_count = int(torch_auxiliary.state.step_count)
        self.assertEqual(
            float(torch_auxiliary.state.source_time),
            step_count * torch_auxiliary.plan.dt,
        )

    def test_float32_gaussian_envelope_uses_exact_auxiliary_step_offset(self):
        simulation = _torch_simulation(
            lambda: [_gaussian()], size=(3, 3, 3), precision="float32"
        )
        auxiliary = simulation.sources.auxiliaries[0]
        gaussian_batches = tuple(
            batch
            for batch in simulation.sources.batches
            if isinstance(batch, TorchTransparentBatch)
            and batch.gaussian_width is not None
        )
        self.assertTrue(gaussian_batches)
        initial_auxiliary_step = int(auxiliary.state.step_count)
        for batch in gaussian_batches:
            self.assertEqual(int(batch._envelope_step_offset), initial_auxiliary_step)
            self.assertEqual(batch._envelope.dtype, torch.float64)

        initial_checkpoint = simulation.checkpoint()
        unrelated_outer_time = torch.tensor(
            12345.0, device=simulation.device, dtype=torch.float32
        )
        electric = next(
            batch for batch in gaussian_batches if batch.component.startswith("E")
        )
        electric.apply(
            torch.zeros_like(simulation.state.field(electric.component)),
            unrelated_outer_time,
        )
        self.assertEqual(float(electric._envelope), 0.0)

        auxiliary.step()
        magnetic = next(
            batch
            for batch in gaussian_batches
            if batch.component.startswith("H")
            and bool(torch.count_nonzero(batch.weights))
        )
        auxiliary_values = auxiliary.state.field(magnetic.auxiliary_component).reshape(
            -1
        )
        auxiliary_values.copy_(
            torch.linspace(
                -0.3,
                0.4,
                auxiliary_values.numel(),
                dtype=auxiliary_values.dtype,
                device=auxiliary_values.device,
            )
        )
        envelope_time = min(auxiliary.plan.dt, magnetic.gaussian_width)
        expected_envelope = (
            np.sin(0.5 * np.pi * envelope_time / magnetic.gaussian_width) ** 2
        )
        expected_values = (
            auxiliary_values[magnetic.samples].cpu().numpy()
            * magnetic.weights.cpu().numpy()
        ).sum(axis=1) * expected_envelope
        outer = torch.zeros_like(simulation.state.field(magnetic.component))
        magnetic.apply(outer, unrelated_outer_time)
        self.assertAlmostEqual(float(magnetic._envelope), expected_envelope)
        np.testing.assert_allclose(
            outer.reshape(-1)[magnetic.targets].cpu().numpy(),
            expected_values,
            rtol=2e-6,
            atol=2e-7,
        )

        simulation.load_checkpoint(initial_checkpoint)
        self.assertEqual(int(auxiliary.state.step_count), initial_auxiliary_step)
        addresses = simulation.buffer_addresses()
        simulation.advance(7)
        checkpoint = simulation.checkpoint()
        simulation.advance(3)
        expected_fields = simulation.host_snapshot()
        expected_auxiliary = auxiliary.host_snapshot()
        simulation.load_checkpoint(checkpoint).advance(3)
        for name in _COMPONENTS:
            np.testing.assert_array_equal(
                simulation.host_snapshot()[name], expected_fields[name]
            )
            np.testing.assert_array_equal(
                auxiliary.host_snapshot()[name], expected_auxiliary[name]
            )

        simulation.advance(90)
        self.assertEqual(addresses, simulation.buffer_addresses())
        self.assertEqual(int(simulation.state.step_count), 100)
        self.assertEqual(
            int(auxiliary.state.step_count) - initial_auxiliary_step,
            100,
        )
        expected_auxiliary_time = auxiliary.state.step_count.to(
            auxiliary.state.source_time.dtype
        ).mul(auxiliary.state.time_step)
        self.assertTrue(
            torch.equal(auxiliary.state.source_time, expected_auxiliary_time)
        )
        actual = simulation.host_snapshot()
        for name in _COMPONENTS:
            self.assertTrue(np.isfinite(actual[name]).all(), name)
        actual_auxiliary = auxiliary.host_snapshot()
        for name in ("Ex", "Hy"):
            self.assertTrue(np.isfinite(actual_auxiliary[name]).all(), name)

    def test_gaussian_zero_width_is_unwindowed(self):
        simulation = _torch_simulation(
            lambda: [_gaussian(width=0.0)],
            size=(3, 3, 3),
            precision="float32",
        )
        gaussian_batches = tuple(
            batch
            for batch in simulation.sources.batches
            if isinstance(batch, TorchTransparentBatch)
            and batch.gaussian_width is not None
        )
        self.assertTrue(gaussian_batches)
        unrelated_outer_time = torch.tensor(
            12345.0, device=simulation.device, dtype=torch.float32
        )
        for batch in gaussian_batches:
            batch.apply(
                torch.zeros_like(simulation.state.field(batch.component)),
                unrelated_outer_time,
            )
            self.assertEqual(float(batch._envelope), 1.0)
        _advance_and_assert_finite(self, simulation, 5)


class TorchBoundaryTest(unittest.TestCase):
    def test_collapsed_paired_real_boundaries_match_bloch_phase_equation(self):
        bloch = (0.07, 0.11, 0.13)
        for compile_policy in ("eager", "compile"):
            with self.subTest(compile_policy=compile_policy):
                simulation = _torch_simulation(
                    lambda: [],
                    size=(4, 4, 0),
                    bloch=bloch,
                    compile_policy=compile_policy,
                )
                rng = np.random.default_rng(7)
                fields = {
                    component: (
                        rng.normal(size=values.shape) * 1e-3
                        + 1j * rng.normal(size=values.shape) * 1e-3
                    )
                    for component, values in simulation.host_snapshot().items()
                }
                simulation.load_host_fields(fields)
                addresses = simulation.buffer_addresses()
                _advance_and_assert_finite(self, simulation, 3)
                simulation._sync_electric_boundaries(skip_axis=2)
                simulation._sync_magnetic_boundaries(skip_axis=2)
                self.assertEqual(addresses, simulation.buffer_addresses())
                snapshot = simulation.host_snapshot()
                for component, values in snapshot.items():
                    component_axis = "xyz".index(component[1].lower())
                    for axis in (0, 1):
                        if axis == component_axis or values.shape[axis] <= 1:
                            continue
                        length = (values.shape[axis] - 1) * simulation.plan.dr[axis]
                        phase = np.exp(1j * bloch[axis] * length)
                        low = np.take(values, 0, axis=axis)
                        high = np.take(values, -1, axis=axis)
                        np.testing.assert_allclose(
                            high,
                            low * phase,
                            rtol=2e-14,
                            atol=2e-14,
                            err_msg=f"{component} axis {axis}",
                        )


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
