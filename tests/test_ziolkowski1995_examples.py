import unittest
from math import pi

import numpy as np

from examples.ziolkowski1995_common import (
    F0_HZ,
    GAIN_T1_S,
    GAIN_T2_S,
    GAMMA_C_M,
    LAMBDA0_UM,
    N_ATOM_M3,
    OMEGA0_RAD_S,
    PAPER_EPS0,
    PERIOD_S,
    SIT_T1_S,
    SIT_T2_S,
    UNITS,
    PumpProbe,
    SechSinePulse,
    SmoothSine,
    UltrafastPulse,
    carrier_intensity,
    gain_scenario,
    make_simulation,
    pump_probe_scenario,
    sit_scenario,
    ultrafast_scenario,
)


class ZiolkowskiUnitsTest(unittest.TestCase):
    def test_si_round_trips(self):
        for value in (1e-15, 5e-14, 1e-10):
            with self.subTest(time=value):
                self.assertAlmostEqual(UNITS.time_si(UNITS.time(value)), value)
        for value in (1.0, 4.2186e9, 2.272e10):
            with self.subTest(field=value):
                self.assertTrue(
                    np.isclose(
                        UNITS.electric_field_si(UNITS.electric_field(value)),
                        value,
                        rtol=1e-15,
                    )
                )

    def test_dm2_conversion_preserves_both_couplings(self):
        parameters = UNITS.dm2_parameters(
            omega_rad_s=OMEGA0_RAD_S,
            atom_density_m3=N_ATOM_M3,
            dipole_c_m=GAMMA_C_M,
            t1_s=SIT_T1_S,
            t2_s=SIT_T2_S,
        )
        gamma = parameters["gamma"]
        atom_density = parameters["n_atom"][0]

        expected_bloch = (
            GAMMA_C_M
            * UNITS.electric_field_v_m
            * UNITS.length_m
            / (UNITS.reduced_planck_j_s * UNITS.wave_speed_m_s)
        )
        expected_maxwell = (
            N_ATOM_M3 * GAMMA_C_M / (PAPER_EPS0 * UNITS.electric_field_v_m)
        )
        self.assertAlmostEqual(gamma, expected_bloch)
        self.assertAlmostEqual(atom_density * gamma, expected_maxwell)
        self.assertAlmostEqual(parameters["omega"][0], 2 * pi / LAMBDA0_UM)


class ZiolkowskiSourceTest(unittest.TestCase):
    def test_carrier_intensity_recovers_unit_sinusoid(self):
        sample_interval = PERIOD_S / 100
        times = np.arange(1_000) * sample_interval
        field = 3.0 * np.sin(OMEGA0_RAD_S * times)

        intensity = carrier_intensity(field, sample_interval, 3.0)
        hann_intensity = carrier_intensity(
            field, sample_interval, 3.0, periods=3, window="hann"
        )

        self.assertTrue(np.allclose(intensity[100:-100], 1.0, atol=0.01))
        self.assertTrue(np.allclose(hann_intensity[200:-200], 1.0, atol=0.01))
        with self.assertRaisesRegex(ValueError, "period count"):
            carrier_intensity(field, sample_interval, 3.0, periods=0)
        with self.assertRaisesRegex(ValueError, "unsupported envelope window"):
            carrier_intensity(field, sample_interval, 3.0, window="triangle")

    def test_sech_pulse_support_and_envelope_area(self):
        width = UNITS.time(20 / F0_HZ)
        pulse = SechSinePulse(UNITS.angular_frequency(OMEGA0_RAD_S), width)
        self.assertEqual(pulse.oscillator(-1), 0)
        self.assertEqual(pulse.oscillator(width + 1), 0)
        self.assertEqual(pulse.envelope(width / 2), 1)

        times = np.linspace(0, width, 100_001)
        numerical = np.trapezoid([pulse.envelope(time) for time in times], times)
        analytic = width / 10 * np.arctan(np.sinh(10))
        self.assertAlmostEqual(numerical, analytic, places=10)

    def test_ultrafast_pulse_has_zero_area_and_smooth_endpoints(self):
        width = UNITS.time(PERIOD_S)
        pulse = UltrafastPulse(width)
        times = np.linspace(0, width, 100_001)
        values = np.array([pulse.oscillator(time) for time in times])

        self.assertEqual(values[0], 0)
        self.assertEqual(values[-1], 0)
        self.assertAlmostEqual(np.trapezoid(values, times), 0, places=12)
        self.assertAlmostEqual(
            (values[1] - values[0]) / (times[1] - times[0]), 0, places=6
        )
        self.assertAlmostEqual(
            (values[-1] - values[-2]) / (times[-1] - times[-2]), 0, places=6
        )

    def test_gain_and_pump_probe_delays(self):
        width = UNITS.time(PERIOD_S)
        omega = UNITS.angular_frequency(OMEGA0_RAD_S)
        sine = SmoothSine(omega, width)
        self.assertEqual(sine.oscillator(-1), 0)
        turn_on = np.array(
            [sine.envelope(time) for time in np.linspace(0, sine.rise_time, 101)]
        )
        self.assertEqual(turn_on[0], 0)
        self.assertEqual(turn_on[-1], 1)
        self.assertTrue(np.all(np.diff(turn_on) >= 0))
        self.assertAlmostEqual(sine.oscillator(5 * width + width / 4), 1)

        signal = PumpProbe(omega, width, beta=1e-4, delay=20 * width)
        delayed_turn_on = np.array(
            [
                signal.probe.envelope(time - signal.delay)
                for time in np.linspace(signal.delay, signal.delay + 5 * width, 101)
            ]
        )
        self.assertTrue(np.all(np.diff(delayed_turn_on) >= 0))
        self.assertEqual(signal.probe.oscillator(-1), 0)
        self.assertEqual(signal.oscillator(10 * width), 0)
        self.assertNotEqual(signal.oscillator(20 * width + width / 4), 0)


class ZiolkowskiScenarioTest(unittest.TestCase):
    def test_paper_cell_counts_and_resolutions(self):
        self.assertEqual(sit_scenario(2).cells, 20_000)
        self.assertEqual(ultrafast_scenario(5).cells, 2_000)
        self.assertEqual(ultrafast_scenario(9).cells, 5_000)
        self.assertEqual(gain_scenario().cells, 2_000)
        pump_probe = pump_probe_scenario(20)
        self.assertEqual(pump_probe.cells, 4_000)
        self.assertAlmostEqual(
            pump_probe.domain_um / pump_probe.cells, LAMBDA0_UM / 400
        )

    def test_material_and_probe_geometry(self):
        scenario = gain_scenario(quick=True)
        simulation = make_simulation(scenario)
        self.assertEqual(int(simulation.space.whole_field_size[2]), scenario.cells)
        self.assertAlmostEqual(simulation.time_step.dt, 0.5 / scenario.resolution)

        for distance, expected_rho30 in (
            (2.0, None),
            (5.0, 1.0),
            (13.0, None),
        ):
            coordinate = distance - scenario.domain_um / 2
            material, _ = simulation.geom_tree.material_of_point((0, 0, coordinate))
            if expected_rho30 is None:
                self.assertNotEqual(material.__class__.__name__, "Dm2")
            else:
                self.assertEqual(material.rho30, expected_rho30)

    def test_relaxation_parameters_are_figure_specific(self):
        self.assertEqual(gain_scenario().t1_s, GAIN_T1_S)
        self.assertEqual(gain_scenario().t2_s, GAIN_T2_S)
        self.assertEqual(sit_scenario(2).t1_s, SIT_T1_S)
        self.assertEqual(sit_scenario(2).t2_s, SIT_T2_S)


if __name__ == "__main__":
    unittest.main()
