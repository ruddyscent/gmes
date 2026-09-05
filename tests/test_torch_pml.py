"""Reference and storage tests for Torch UPML/CPML tensor buckets."""

import copy
import json
import unittest
from pathlib import Path

import numpy as np
import torch

import gmes
from gmes.torch_fdtd import (
    DEFAULT_CPML_REPRESENTATION,
    SPARSE_CPML_REPRESENTATION,
)

_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
_MANIFEST = json.loads(
    (
        Path(__file__).parents[1] / "benchmarks" / "native_oracle_workloads.json"
    ).read_text()
)
_TOLERANCES = _MANIFEST["tolerances"]["torch"]["pml"]


def _crossover_material(name):
    """Build only the public material models used by cpu-crossover-2d."""
    pole = gmes.DrudePole(omega=0.6, gamma=0.03)
    critical_points = (
        gmes.CriticalPoint(amp=0.04, phi=0.2, omega=0.9, gamma=0.03),
        gmes.CriticalPoint(amp=0.02, phi=-0.1, omega=1.1, gamma=0.04),
    )
    factories = {
        "dielectric": lambda: gmes.Dielectric(eps_inf=1.7, mu_inf=1.05),
        "drude-1": lambda: gmes.Drude(eps_inf=1.2, dps=(pole,)),
        "lorentz-1": lambda: gmes.Lorentz(
            eps_inf=1.2,
            lps=(gmes.LorentzPole(amp=0.05, omega=0.8, gamma=0.03),),
        ),
        "dcp-ade": lambda: gmes.DcpAde(eps_inf=1.2, dps=(pole,), cps=critical_points),
        "dcp-plrc": lambda: gmes.DcpPlrc(eps_inf=1.2, dps=(pole,), cps=critical_points),
        "dcp-rc": lambda: gmes.DcpRc(eps_inf=1.2, dps=(pole,), cps=critical_points),
        "dm2-1": lambda: gmes.Dm2(
            eps_inf=1.2,
            omega=(0.8,),
            n_atom=(0.01,),
            gamma=0.02,
            rtol=1e-4,
        ),
    }
    return factories[name]()


def _crossover_geometry(spec):
    """Construct the fixed fragmented coverage workload through public models."""
    size = np.maximum(np.asarray(spec["size"], dtype=float), 1.0)
    families = (
        "drude-1",
        "lorentz-1",
        "dcp-ade",
        "dcp-plrc",
        "dcp-rc",
        "dm2-1",
    )
    geometry = [
        gmes.DefaultMedium(_crossover_material("dielectric")),
        gmes.Shell(gmes.Cpml(), thickness=max(0.25, min(size) * 0.04)),
    ]
    total_width = max(
        size[0] * float(spec["coverage_percent"]) / 100,
        len(families) / spec["resolution"],
    )
    family_width = total_width / len(families)
    origin = -0.5 * total_width
    fragments = 4
    for family_index, family in enumerate(families):
        for fragment in range(fragments):
            width = family_width / fragments
            center_x = (
                origin + (family_index + (fragment + 0.5) / fragments) * family_width
            )
            height = size[1] / fragments
            center_y = -0.5 * size[1] + (fragment + 0.5) * height
            geometry.append(
                gmes.Block(
                    _crossover_material(family),
                    center=(center_x, center_y, 0),
                    size=(width * 0.92, height * 0.82, size[2]),
                )
            )
    return geometry


def _crossover_sources(spec):
    return [
        gmes.PointSource(
            src_time=gmes.Continuous(freq=0.35),
            center=(0, 0, 0),
            component=gmes.Ex,
            amp=float(spec.get("source_amp", 1e-3)),
        )
    ]


def _crossover_initial_fields(shapes, *, seed, scale):
    """Return deterministic nonzero fields without an executable oracle import."""
    rng = np.random.default_rng(seed)
    fields = {}
    for name in _COMPONENTS:
        shape = tuple(int(length) for length in shapes[name])
        values = scale * (1 + 0.1 * rng.random())
        for axis, length in enumerate(shape):
            ramp_shape = [1] * len(shape)
            ramp_shape[axis] = length
            values = values + (
                scale
                * 1e-6
                * (axis + 1)
                * np.linspace(0, 1, length).reshape(ramp_shape)
            )
        fields[name] = np.broadcast_to(values, shape).copy()
    return fields


def _geometry(material_type):
    return [
        gmes.DefaultMedium(gmes.Dielectric(eps_inf=2.5, mu_inf=1.2)),
        gmes.Shell(material=material_type(), thickness=0.5),
    ]


def _reference_and_torch(
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
    reference = gmes.TorchSimulation(
        space=gmes.Cartesian(size, resolution),
        geometry=copy.deepcopy(geometry),
        runtime=gmes.TorchRuntimeConfig(
            device="cpu",
            precision="float64",
            execution_policy="dense",
            planner_tile_size=16,
            cpu_threads=2,
        ),
        bloch=bloch,
    )
    simulation = gmes.TorchSimulation(
        space=gmes.Cartesian(size, resolution),
        geometry=copy.deepcopy(geometry),
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
    for name, values in reference.state.host_snapshot().items():
        seeded = rng.normal(size=values.shape) * 1e-3
        if bloch is not None:
            seeded = seeded + 1j * rng.normal(size=values.shape) * 1e-3
        fields[name] = seeded
    reference.load_host_fields(fields)
    simulation.load_host_fields(fields)
    return reference, simulation


def _assert_reference(test, reference, simulation, model, precision):
    expected_fields = reference.state.host_snapshot()
    complex_values = np.iscomplexobj(expected_fields["Ex"])
    tolerance_name = (
        ("complex128" if precision == "float64" else "complex64")
        if complex_values
        else precision
    )
    tolerance = _TOLERANCES[tolerance_name]
    snapshot = simulation.state.host_snapshot()
    states = simulation.state.pml_state_snapshot()
    reference_states = reference.state.pml_state_snapshot()
    model_name = model.__name__.lower()
    for component in _COMPONENTS:
        np.testing.assert_allclose(
            snapshot[component],
            expected_fields[component],
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
        reference_plan = reference.plan.components[component]
        reference_index, reference_bucket = next(
            (index, bucket)
            for index, bucket in enumerate(reference_plan.buckets)
            if bucket.signature.model == model_name
        )
        np.testing.assert_array_equal(bucket.targets, reference_bucket.targets)
        np.testing.assert_allclose(
            states[f"pml_{component.lower()}_{bucket_index}_state"],
            reference_states[f"pml_{component.lower()}_{reference_index}_state"],
            rtol=tolerance["rtol"],
            atol=tolerance["atol"],
            err_msg=f"{component} state",
        )


def _host_array(value):
    return value.detach().cpu().numpy()


def _pml_profile_formula(material, coordinate, axis):
    """Return sigma, kappa, and alpha from the published grading equations."""
    offset = coordinate - material.center[axis]
    half_size = material.half_size[axis]
    if offset <= material.d - half_size:
        depth = np.clip((half_size + offset) / material.d, 0.0, 1.0)
        boundary = True
    elif half_size - material.d <= offset:
        depth = np.clip((half_size - offset) / material.d, 0.0, 1.0)
        boundary = True
    else:
        depth = 1.0
        boundary = False
    graded = (1.0 - depth) ** material.m
    sigma = material.sigma_max[axis] * graded
    kappa = 1.0 + (material.kappa_max - 1.0) * graded
    alpha = (
        material.a_max * depth**material.m_a
        if boundary and isinstance(material, gmes.Cpml)
        else 0.0
    )
    return sigma, kappa, alpha


def _upml_coefficients_formula(material, coordinate, axis):
    sigma, kappa, _ = _pml_profile_formula(material, coordinate, axis)
    denominator = 2.0 * kappa + sigma * material.dt
    return (
        (2.0 * kappa - sigma * material.dt) / denominator,
        2.0 * material.dt / denominator,
        1.0 / denominator,
    )


def _cpml_coefficients_formula(material, coordinate, axis):
    sigma, kappa, alpha = _pml_profile_formula(material, coordinate, axis)
    decay = np.exp(-(sigma / kappa + alpha) * material.dt)
    denominator = (sigma + kappa * alpha) * kappa
    memory = 0.0 if denominator == 0.0 else sigma * (decay - 1.0) / denominator
    return decay, memory, kappa


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
        # The full complex PLRC/RC row layout reserves a structurally zero
        # imaginary channel; the real-only Torch state stores only live values.
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
    def test_one_cell_state_recurrences_match_explicit_scalar_equations(self):
        field = torch.tensor([0.25], dtype=torch.float64)
        source1 = torch.tensor([0.8, -0.2], dtype=torch.float64)
        source2 = torch.tensor([0.4, -0.6], dtype=torch.float64)
        targets = torch.tensor([0], dtype=torch.int64)
        stencil = torch.tensor([[0, 1, 0, 1]], dtype=torch.int64)
        scratch = tuple(torch.zeros(1, dtype=torch.float64) for _ in range(3))
        scale1, scale2, direction = 2.0, 0.5, -1.0
        difference1 = (0.8 - -0.2) * scale1
        difference2 = (0.4 - -0.6) * scale2

        upml_coefficients = np.asarray(
            [[0.7, 0.8, 0.3, 0.9, 0.4, 1.2, 0.6]], dtype=np.float64
        )
        upml_state = torch.tensor([[0.15]], dtype=torch.float64)
        old_memory = float(upml_state[0, 0])
        curl = (difference1 - difference2) * direction
        new_memory = 0.8 * old_memory + 0.3 * curl
        expected_field = 0.9 * 0.25 + 0.4 * 0.7 * (1.2 * new_memory - 0.6 * old_memory)
        gmes.torch_fdtd._upml_bucket_update(
            field,
            source1,
            source2,
            targets,
            stencil,
            torch.from_numpy(upml_coefficients),
            upml_state,
            *scratch,
            scale1,
            scale2,
            direction,
            False,
        )
        self.assertAlmostEqual(float(field[0]), expected_field)
        self.assertAlmostEqual(float(upml_state[0, 0]), new_memory)

        field.fill_(0.25)
        cpml_coefficients = np.asarray(
            [[0.7, 0.8, -0.3, 1.4, 0.6, -0.2, 1.5]], dtype=np.float64
        )
        cpml_state = torch.tensor([[0.15, -0.05]], dtype=torch.float64)
        psi1 = 0.8 * 0.15 - 0.3 * difference1
        psi2 = 0.6 * -0.05 - 0.2 * difference2
        expected_field = 0.25 + 0.7 * 0.1 * direction * (
            difference1 / 1.4 + psi1 - difference2 / 1.5 - psi2
        )
        gmes.torch_fdtd._cpml_bucket_update(
            field,
            source1,
            source2,
            targets,
            stencil,
            torch.from_numpy(cpml_coefficients),
            cpml_state,
            *scratch,
            scale1,
            scale2,
            direction,
            0.1,
            False,
        )
        self.assertAlmostEqual(float(field[0]), expected_field)
        np.testing.assert_allclose(
            cpml_state.numpy(), [[psi1, psi2]], rtol=0, atol=1e-15
        )

    def test_nonzero_fields_and_state_match_at_reference_steps(self):
        cases = (
            (gmes.Upml, "eager"),
            (gmes.Cpml, "eager"),
            (gmes.Cpml, "compile"),
        )
        for model, compile_policy in cases:
            with self.subTest(model=model.__name__, compile_policy=compile_policy):
                if compile_policy == "compile":
                    torch._dynamo.reset()
                reference, simulation = _reference_and_torch(
                    model,
                    compile_policy=compile_policy,
                )
                completed = 0
                for target in (1, 2, 5, 20, 100):
                    delta = target - completed
                    simulation.advance(delta)
                    for _ in range(delta):
                        reference.step()
                    _assert_reference(self, reference, simulation, model, "float64")
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
                    reference, simulation = _reference_and_torch(
                        model,
                        precision=precision,
                        bloch=bloch,
                        compile_policy=(
                            "compile"
                            if model is gmes.Cpml
                            or (precision == "float64" and bloch is not None)
                            else "eager"
                        ),
                    )
                    simulation.advance(5)
                    for _ in range(5):
                        reference.step()
                    _assert_reference(self, reference, simulation, model, precision)

    def test_collapsed_1d_2d_and_3d_modes_match_dense_reference(self):
        for size in ((4, 0, 0), (4, 3, 0), (3, 3, 2)):
            for model in (gmes.Upml, gmes.Cpml):
                with self.subTest(size=size, model=model.__name__):
                    reference, simulation = _reference_and_torch(
                        model,
                        size=size,
                        compile_policy="compile" if model is gmes.Cpml else "eager",
                    )
                    simulation.advance(5)
                    for _ in range(5):
                        reference.step()
                    _assert_reference(self, reference, simulation, model, "float64")

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
        reference, simulation = _reference_and_torch(
            gmes.Cpml,
            size=(2, 2, 2),
            resolution=4,
            geometry=geometry,
        )
        simulation.advance(5)
        for _ in range(5):
            reference.step()
        _assert_reference(self, reference, simulation, gmes.Cpml, "float64")
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
                reference, simulation = _reference_and_torch(gmes.Cpml, policy=policy)
                simulation.advance(5)
                for _ in range(5):
                    reference.step()
                _assert_reference(self, reference, simulation, gmes.Cpml, "float64")
                self.assertEqual(
                    {
                        bucket.selected_policy
                        for component in simulation.plan.components.values()
                        for bucket in component.buckets
                    },
                    {policy},
                )

    def test_compiled_z_collapsed_specialization_matches_dense_reference(self):
        for model in (gmes.Upml, gmes.Cpml):
            with self.subTest(model=model.__name__):
                torch._dynamo.reset()
                reference, simulation = _reference_and_torch(
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
                    reference.step()
                _assert_reference(self, reference, simulation, model, "float64")

    def test_compiled_custom_kappa_sparse_residual_matches_dense_reference(self):
        for precision in ("float64", "float32"):
            with self.subTest(precision=precision):
                geometry = [
                    gmes.DefaultMedium(gmes.Dielectric(eps_inf=2.5, mu_inf=1.2)),
                    gmes.Block(
                        gmes.Dielectric(eps_inf=3.2, mu_inf=1.15),
                        center=(0, 0, 0),
                        size=(1.2, 1.2, 1.2),
                    ),
                    gmes.Shell(gmes.Cpml(kappa_max=3.0), thickness=0.75),
                ]
                reference, simulation = _reference_and_torch(
                    gmes.Cpml,
                    size=(2, 2, 2),
                    resolution=4,
                    precision=precision,
                    compile_policy="compile",
                    geometry=geometry,
                )
                completed = 0
                for target in (1, 2, 5, 20, 100):
                    delta = target - completed
                    simulation.advance(delta)
                    for _ in range(delta):
                        reference.step()
                    _assert_reference(
                        self,
                        reference,
                        simulation,
                        gmes.Cpml,
                        precision,
                    )
                    completed = target
                self.assertEqual(
                    simulation.diagnostics()["pml"]["execution_representation"],
                    SPARSE_CPML_REPRESENTATION,
                )

    def test_extreme_float32_kappa_uses_stable_compact_fallback(self):
        geometry = [
            gmes.DefaultMedium(gmes.Dielectric(eps_inf=2.5, mu_inf=1.2)),
            gmes.Shell(gmes.Cpml(kappa_max=1e8), thickness=0.5),
        ]
        reference, simulation = _reference_and_torch(
            gmes.Cpml,
            precision="float32",
            compile_policy="compile",
            geometry=geometry,
        )
        simulation.advance(5)
        for _ in range(5):
            reference.step()
        _assert_reference(self, reference, simulation, gmes.Cpml, "float32")
        self.assertEqual(
            simulation.diagnostics()["pml"]["execution_representation"],
            DEFAULT_CPML_REPRESENTATION,
        )

    def test_compiled_cpu_crossover_manifest_matches_dense_reference_state(self):
        torch._dynamo.reset()
        spec = next(
            item
            for item in _MANIFEST["benchmarks"]
            if item["name"] == "cpu-crossover-2d"
        )
        reference = _MANIFEST["reference"]

        def geometry():
            return _crossover_geometry(spec)

        def sources():
            return _crossover_sources(spec)

        reference_simulation = gmes.TorchSimulation(
            space=gmes.Cartesian(tuple(spec["size"]), spec["resolution"]),
            geometry=geometry(),
            sources=sources(),
            runtime=gmes.TorchRuntimeConfig(
                device="cpu",
                precision="float64",
                execution_policy="dense",
                cpu_threads=2,
            ),
        )
        simulation = gmes.TorchSimulation(
            space=gmes.Cartesian(tuple(spec["size"]), spec["resolution"]),
            geometry=geometry(),
            sources=sources(),
            runtime=gmes.TorchRuntimeConfig(
                device="cpu",
                precision="float64",
                compile_policy="compile",
                cpu_threads=2,
            ),
        )
        fields = _crossover_initial_fields(
            simulation.plan.shapes,
            seed=reference["seed"],
            scale=reference["field_scale"],
        )
        reference_simulation.load_host_fields(fields)
        simulation.load_host_fields(fields)

        simulation.advance(5)
        reference_simulation.advance(5)

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
        persistent = simulation.state.checkpoint()
        compared_dm2 = set()
        actual_dm2 = simulation.dm2_state_snapshot()
        reference_dm2 = reference_simulation.dm2_state_snapshot()
        self.assertEqual(len(actual_dm2), len(reference_dm2))
        for index, (actual, expected) in enumerate(zip(actual_dm2, reference_dm2)):
            self.assertEqual(actual["component"], expected["component"])
            np.testing.assert_array_equal(actual["targets"], expected["targets"])
            np.testing.assert_allclose(
                actual["u"],
                expected["u"],
                rtol=tolerance["rtol"],
                atol=tolerance["atol"],
                err_msg=f"{actual['component']} DM2 state",
            )
            compared_dm2.add(f"dm2_buckets.{index}.u")
        expected_dm2 = {name for name in persistent if name.startswith("dm2_buckets.")}
        self.assertEqual(compared_dm2, expected_dm2)
        pml_state = simulation.state.pml_state_snapshot()
        reference_pml = reference_simulation.state.pml_state_snapshot()
        self.assertEqual(set(pml_state), set(reference_pml))
        for state_name, actual in pml_state.items():
            np.testing.assert_allclose(
                actual,
                reference_pml[state_name],
                rtol=tolerance["rtol"],
                atol=tolerance["atol"],
                err_msg=state_name,
            )
        compared_pml = set(pml_state)
        expected_pml = {name for name in persistent if name.startswith("pml_")}
        self.assertEqual(compared_pml, expected_pml)

        compared_dispersive = set()
        reference_descriptors = reference_simulation.plan.dispersive_buckets
        self.assertEqual(
            len(simulation.plan.dispersive_buckets), len(reference_descriptors)
        )
        for descriptor, reference_descriptor in zip(
            simulation.plan.dispersive_buckets, reference_descriptors
        ):
            actual, width, names = _torch_dispersive_rows(simulation, descriptor)
            expected, reference_width, reference_names = _torch_dispersive_rows(
                reference_simulation, reference_descriptor
            )
            torch_targets = _host_array(
                getattr(simulation.plan, f"{descriptor.prefix}_targets")
            )
            reference_targets = _host_array(
                getattr(
                    reference_simulation.plan,
                    f"{reference_descriptor.prefix}_targets",
                )
            )
            self.assertEqual(descriptor.component, reference_descriptor.component)
            self.assertEqual(descriptor.model, reference_descriptor.model)
            self.assertEqual(width, reference_width)
            self.assertEqual(names, reference_names)
            np.testing.assert_array_equal(torch_targets, reference_targets)
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=tolerance["rtol"],
                atol=tolerance["atol"],
                err_msg=f"{descriptor.component} {descriptor.model} state",
            )
            compared_dispersive.update(names)
        expected_dispersive = {
            name for name in persistent if name.startswith("bucket_")
        }
        self.assertEqual(compared_dispersive, expected_dispersive)

        actual_fields = simulation.state.host_snapshot()
        reference_fields = reference_simulation.state.host_snapshot()
        for component_name in _COMPONENTS:
            np.testing.assert_allclose(
                actual_fields[component_name],
                reference_fields[component_name],
                rtol=tolerance["rtol"],
                atol=tolerance["atol"],
                err_msg=f"{component_name} field",
            )

        for names in (expected_pml, expected_dispersive, expected_dm2):
            self.assertTrue(names)
            self.assertTrue(
                any(np.any(_host_array(persistent[name])) for name in names)
            )
        self.assertEqual(int(simulation.state.step_count), 5)
        self.assertAlmostEqual(
            float(simulation.state.source_time),
            float(reference_simulation.state.source_time),
            places=14,
        )

    def test_long_run_absorbs_seeded_energy_like_dense_reference(self):
        reference, simulation = _reference_and_torch(
            gmes.Cpml,
            size=(4, 4, 0),
            resolution=3,
            compile_policy="compile",
        )
        initial_energy = sum(
            float(np.square(np.abs(values)).sum())
            for values in simulation.state.host_snapshot().values()
        )
        simulation.advance(200)
        for _ in range(200):
            reference.step()
        _assert_reference(self, reference, simulation, gmes.Cpml, "float64")
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
                    first_coefficients = [
                        _upml_coefficients_formula(material, value, first)
                        for value in coordinates[:, first]
                    ]
                    second_coefficients = [
                        _upml_coefficients_formula(material, value, second)
                        for value in coordinates[:, second]
                    ]
                    field_profiles = [
                        _pml_profile_formula(material, value, field)
                        for value in coordinates[:, field]
                    ]
                    expected = np.column_stack(
                        (
                            base,
                            [value[0] for value in first_coefficients],
                            [value[1] for value in first_coefficients],
                            [value[0] for value in second_coefficients],
                            [value[2] for value in second_coefficients],
                            [
                                2.0 * kappa + sigma * material.dt
                                for sigma, kappa, _ in field_profiles
                            ],
                            [
                                2.0 * kappa - sigma * material.dt
                                for sigma, kappa, _ in field_profiles
                            ],
                        )
                    )
                else:
                    first_coefficients = [
                        _cpml_coefficients_formula(material, value, first)
                        for value in coordinates[:, first]
                    ]
                    second_coefficients = [
                        _cpml_coefficients_formula(material, value, second)
                        for value in coordinates[:, second]
                    ]
                    expected = np.column_stack(
                        (
                            base,
                            [value[0] for value in first_coefficients],
                            [value[1] for value in first_coefficients],
                            [value[2] for value in first_coefficients],
                            [value[0] for value in second_coefficients],
                            [value[1] for value in second_coefficients],
                            [value[2] for value in second_coefficients],
                        )
                    )
                np.testing.assert_allclose(
                    bucket.cell_coefficients,
                    expected,
                    rtol=2e-15,
                    atol=2e-15,
                    err_msg=f"{model.__name__} {component_name}",
                )

    def test_compiled_cpml_uses_sparse_state_with_canonical_checkpoint(self):
        simulation = gmes.TorchSimulation(
            space=gmes.Cartesian((4, 4, 4), 3),
            geometry=_geometry(gmes.Cpml),
            runtime=gmes.TorchRuntimeConfig(
                device="cpu",
                compile_policy="compile",
                cpu_threads=1,
            ),
        )
        logical_states = 0
        physical_states = 0
        for metadata in simulation.plan.cpml_residual_buckets:
            logical_states += 2 * metadata.target_count
            for axis in metadata.axes:
                physical_states += axis.target_count
                state = getattr(simulation.state, f"{axis.state_prefix}_state")
                self.assertTrue(state.is_contiguous())
                if state.numel():
                    state.copy_(
                        torch.linspace(
                            0.001,
                            0.002,
                            state.numel(),
                            dtype=state.dtype,
                            device=state.device,
                        ).reshape_as(state)
                    )
        self.assertLess(physical_states, logical_states)
        diagnostics = simulation.diagnostics()["pml"]
        self.assertEqual(
            diagnostics["state_bytes"],
            physical_states * simulation.state.ex.element_size(),
        )
        self.assertEqual(diagnostics["active_axis_states"], physical_states)
        self.assertEqual(
            diagnostics["execution_representation"], SPARSE_CPML_REPRESENTATION
        )

        addresses = simulation.buffer_addresses()
        state_dict = {
            name: value.clone() for name, value in simulation.state.state_dict().items()
        }
        self.assertFalse(any(name.startswith("_pml_") for name in state_dict))
        before_state_dict = simulation.state.pml_state_snapshot(numpy=False)
        for metadata in simulation.plan.cpml_residual_axes:
            getattr(simulation.state, f"{metadata.state_prefix}_state").zero_()
        incompatible = simulation.state.load_state_dict(state_dict)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertEqual(addresses, simulation.buffer_addresses())
        after_state_dict = simulation.state.pml_state_snapshot(numpy=False)
        for name in before_state_dict:
            self.assertIn(name, state_dict)
            torch.testing.assert_close(before_state_dict[name], after_state_dict[name])

        checkpoint = simulation.checkpoint()
        self.assertEqual(checkpoint["format"], "gmes.torch.simulation")
        self.assertEqual(checkpoint["version"], 1)
        state_checkpoint = checkpoint["state"]
        before = simulation.state.pml_state_snapshot(numpy=False)
        for metadata in simulation.plan.cpml_residual_buckets:
            self.assertEqual(
                state_checkpoint[metadata.state_name].shape,
                (metadata.target_count, 2),
            )
        for metadata in simulation.plan.cpml_residual_axes:
            getattr(simulation.state, f"{metadata.state_prefix}_state").zero_()
        simulation.load_checkpoint(checkpoint)
        self.assertEqual(addresses, simulation.buffer_addresses())
        after = simulation.state.pml_state_snapshot(numpy=False)
        self.assertEqual(set(before), set(after))
        for name in before:
            self.assertIn(name, state_checkpoint)
            self.assertEqual(state_checkpoint[name].shape[1], 2)
            torch.testing.assert_close(before[name], after[name])

        metadata = simulation.plan.cpml_residual_buckets[0]
        axis = metadata.axes[0]
        positions = getattr(simulation.plan, f"{axis.prefix}_positions")
        inactive = torch.ones(metadata.target_count, dtype=torch.bool)
        inactive[positions.cpu()] = False
        inactive_row = int(torch.nonzero(inactive, as_tuple=False)[0])
        invalid_state = {
            name: value.clone() for name, value in state_checkpoint.items()
        }
        invalid_state[metadata.state_name][inactive_row, axis.axis] = 1.0
        invalid = {**checkpoint, "state": invalid_state}
        with self.assertRaisesRegex(ValueError, "nonzero inactive CPML"):
            simulation.load_checkpoint(invalid)

    def test_warm_execution_and_checkpoint_keep_fixed_device_storage(self):
        _reference, simulation = _reference_and_torch(gmes.Upml)
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
        reference, simulation = _reference_and_torch(
            gmes.Cpml, compile_policy="compile"
        )
        simulation.advance(2)
        for _ in range(2):
            reference.step()
        _assert_reference(self, reference, simulation, gmes.Cpml, "float64")
        self.assertEqual(torch._dynamo.utils.counters["graph_break"], {})
        addresses = simulation.buffer_addresses()
        simulation.advance(3)
        self.assertEqual(addresses, simulation.buffer_addresses())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_eager_and_fullgraph_have_stable_memory(self):
        for compile_policy in ("eager", "compile"):
            for model in (gmes.Upml, gmes.Cpml):
                with self.subTest(compile_policy=compile_policy, model=model.__name__):
                    reference, simulation = _reference_and_torch(
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
                        reference.step()
                    torch.cuda.synchronize(simulation.device)
                    _assert_reference(self, reference, simulation, model, "float32")
                    self.assertEqual(addresses, simulation.buffer_addresses())
                    self.assertEqual(
                        allocated,
                        torch.cuda.memory_allocated(simulation.device),
                    )


if __name__ == "__main__":
    unittest.main()
