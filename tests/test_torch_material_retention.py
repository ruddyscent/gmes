"""Retained material descriptor and eager Maxwell--Bloch regressions."""

import pickle
import unittest
from types import SimpleNamespace

import numpy as np

import gmes


def _dm2_simulation(material, dt):
    """Build the smallest real eager simulation containing every E component."""
    return gmes.TorchSimulation(
        space=gmes.Cartesian((2, 2, 2), 2),
        geometry=[
            gmes.DefaultMedium(gmes.Dielectric()),
            gmes.Block(material, center=(0, 0, 0), size=(2, 2, 2)),
        ],
        runtime=gmes.TorchRuntimeConfig(
            device="cpu",
            precision="float64",
            execution_policy="dense",
            cpu_threads=2,
        ),
        dt=dt,
    )


def _uniform_electric_fields(simulation, value):
    """Return complete real host fields with all three electric components set."""
    fields = {
        name: np.zeros(tuple(field.shape))
        for name, field in simulation.state.fields().items()
    }
    for component in ("Ex", "Ey", "Ez"):
        fields[component].fill(value)
    return fields


def _uniform_component_fields(simulation, component, value):
    """Return complete real host fields with just one Yee component set."""
    fields = {
        name: np.zeros(tuple(field.shape))
        for name, field in simulation.state.fields().items()
    }
    fields[component].fill(value)
    return fields


class InitializedDescriptorRetentionTest(unittest.TestCase):
    """Keep pure initialized-descriptor behavior independent of native updaters."""

    def _assert_pml_pickle(self, material, scalar_names):
        space = gmes.Cartesian((0, 0, 0))
        space.dt = 1
        material.init(space, ((0, 0, 0), (1, 1, 1), 0.5))
        restored = pickle.loads(pickle.dumps(material))

        for name in scalar_names:
            self.assertEqual(getattr(restored, name), getattr(material, name))
        for name in ("center", "half_size", "dw", "sigma_max"):
            np.testing.assert_array_equal(
                getattr(restored, name), getattr(material, name)
            )
            self.assertIsNot(getattr(restored, name), getattr(material, name))
        return material

    def test_initialized_cpml_pickle_and_rounded_outer_grading_are_finite(self):
        cpml = self._assert_pml_pickle(
            gmes.Cpml(),
            (
                "eps_inf",
                "mu_inf",
                "initialized",
                "d",
                "dt",
                "m",
                "kappa_max",
                "m_a",
                "a_max",
                "sigma_max_ratio",
            ),
        )
        epsilon = np.finfo(float).eps
        for coordinate in (-1 - epsilon, 1 + epsilon):
            coefficients = (
                cpml.sigma(coordinate, 0),
                cpml.kappa(coordinate, 0),
                cpml.a(coordinate, 0),
                cpml.b(coordinate, 0),
                cpml.c(coordinate, 0),
            )
            self.assertTrue(np.isfinite(coefficients).all())

    def test_initialized_upml_pickle_round_trip(self):
        self._assert_pml_pickle(
            gmes.Upml(),
            (
                "eps_inf",
                "mu_inf",
                "initialized",
                "d",
                "dt",
                "m",
                "kappa_max",
                "sigma_max_ratio",
            ),
        )

    def test_lorentz_pickle_round_trip_before_and_after_initialization(self):
        fresh = gmes.Lorentz(
            eps_inf=2,
            mu_inf=3,
            sigma=0.25,
            lps=(gmes.LorentzPole(amp=4, omega=5, gamma=6),),
        )
        space = gmes.Cartesian((0, 0, 0))
        space.dt = 1
        initialized = gmes.Lorentz(
            eps_inf=1,
            mu_inf=1,
            sigma=0,
            lps=(
                gmes.LorentzPole(omega=1.1, gamma=1e-5, amp=0.5),
                gmes.LorentzPole(omega=0.5, gamma=0.1, amp=2e-5),
            ),
        )
        initialized.init(space)
        for material in (fresh, initialized):
            with self.subTest(initialized=material.initialized):
                restored = pickle.loads(pickle.dumps(material))
                for name in ("eps_inf", "mu_inf", "sigma", "initialized"):
                    self.assertEqual(getattr(restored, name), getattr(material, name))
                self.assertEqual(
                    [(pole.amp, pole.omega, pole.gamma) for pole in restored.lps],
                    [(pole.amp, pole.omega, pole.gamma) for pole in material.lps],
                )
                if material.initialized:
                    self.assertEqual(restored.dt, material.dt)
                    np.testing.assert_array_equal(restored.a, material.a)
                    np.testing.assert_array_equal(restored.c, material.c)
                    self.assertIsNot(restored.a, material.a)
                    self.assertIsNot(restored.c, material.c)

    def test_dm2_pickle_round_trip_before_and_after_initialization(self):
        for initialized in (False, True):
            with self.subTest(initialized=initialized):
                material = gmes.Dm2(
                    eps_inf=2,
                    mu_inf=3,
                    omega=(4, 5),
                    n_atom=(6, 7),
                    rho30=-0.5,
                    gamma=0.25,
                    t1=8,
                    t2=9,
                    hbar=10,
                    rtol=1e-6,
                )
                if initialized:
                    material.init(SimpleNamespace(dt=0.125))
                restored = pickle.loads(pickle.dumps(material))
                for name in (
                    "eps_inf",
                    "mu_inf",
                    "omega",
                    "n_atom",
                    "rho30",
                    "gamma",
                    "t1",
                    "t2",
                    "hbar",
                    "rtol",
                    "initialized",
                ):
                    self.assertEqual(getattr(restored, name), getattr(material, name))
                if initialized:
                    self.assertEqual(restored.dt, material.dt)

    def test_const_pickle_round_trip_preserves_real_and_complex_values(self):
        for material in (gmes.Const(2.5, eps_inf=3, mu_inf=4), gmes.Const(2 + 3j)):
            with self.subTest(value=material.value):
                restored = pickle.loads(pickle.dumps(material))
                self.assertEqual(restored.value, material.value)
                self.assertEqual(restored.eps_inf, material.eps_inf)
                self.assertEqual(restored.mu_inf, material.mu_inf)


class TorchDm2PhysicalRetentionTest(unittest.TestCase):
    """Port the direct native Maxwell--Bloch physical snapshots to eager Torch."""

    def test_initial_bloch_drive_does_not_gain_an_inverse_t1_factor(self):
        dt = 1e-7
        expected = 2 * 0.25 / 2 * 4 * -1
        slopes = {component: [] for component in ("Ex", "Ey", "Ez")}
        for t1 in (0.5, 2.0):
            simulation = _dm2_simulation(
                gmes.Dm2(
                    omega=(0,),
                    n_atom=(0,),
                    rho30=-1,
                    gamma=0.25,
                    t1=t1,
                    t2=3,
                    hbar=2,
                    rtol=1e-12,
                ),
                dt,
            )
            simulation.load_host_fields(_uniform_electric_fields(simulation, 4)).step()
            for snapshot in simulation.dm2_state_snapshot():
                slopes[snapshot["component"]].append(snapshot["rho"][:, 0, 1] / dt)
        for component, component_slopes in slopes.items():
            with self.subTest(component=component):
                for slope in component_slopes:
                    np.testing.assert_allclose(slope, expected, rtol=0, atol=1e-5)
                np.testing.assert_allclose(
                    component_slopes[0], component_slopes[1], rtol=0, atol=1e-5
                )

    def test_lossless_bloch_sphere_invariant(self):
        simulation = _dm2_simulation(
            gmes.Dm2(
                omega=(1.25,),
                n_atom=(0,),
                rho30=-1,
                gamma=0.2,
                t1=np.inf,
                t2=np.inf,
                hbar=1,
                rtol=1e-10,
            ),
            0.01,
        )
        simulation.load_host_fields(
            _uniform_component_fields(simulation, "Ex", 0.75)
        ).advance(500)
        snapshot = next(
            item
            for item in simulation.dm2_state_snapshot()
            if item["component"] == "Ex"
        )
        rho = snapshot["rho"][0, 0, :]
        self.assertAlmostEqual(float(rho @ rho), 1, places=8)


if __name__ == "__main__":
    unittest.main()
