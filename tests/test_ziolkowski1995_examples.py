import os
from importlib.util import find_spec
from math import pi
from pathlib import Path
from subprocess import run
from sys import executable
from tempfile import TemporaryDirectory
from textwrap import dedent

import numpy as np
import pytest

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
    SpatialSnapshot,
    UltrafastPulse,
    carrier_intensity,
    gain_scenario,
    make_simulation,
    plot_population,
    pump_probe_scenario,
    run_gain,
    run_snapshots,
    sample_snapshot,
    sit_scenario,
    ultrafast_scenario,
)
from gmes import TorchSimulation
from tests.test_torch_fdtd import restore_torch_runtime as restore_torch_runtime

MATPLOTLIB_AVAILABLE = find_spec("matplotlib") is not None


class TestZiolkowskiOptionalDependency:
    def test_computational_helpers_import_without_plot_extra(self):
        script = dedent("""
            import sys

            class BlockMatplotlib:
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.partition(".")[0] == "matplotlib":
                        raise ModuleNotFoundError(
                            "No module named 'matplotlib'", name="matplotlib"
                        )
                    return None

            sys.meta_path.insert(0, BlockMatplotlib())

            from examples import (
                ziolkowski1995,
                ziolkowski1995_gain,
                ziolkowski1995_pump_probe,
                ziolkowski1995_sit,
                ziolkowski1995_ultrafast,
            )
            from examples.ziolkowski1995_common import gain_scenario

            assert gain_scenario().cells == 2_000
            """)

        result = run(
            [executable, "-c", script],
            cwd=Path(__file__).parents[1],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_cli_help_is_headless_and_import_safe(self):
        root = Path(__file__).parents[1]
        result = run(
            [executable, "examples/ziolkowski1995.py", "--help"],
            cwd=root,
            env={**os.environ, "PYTHONPATH": str(root)},
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "--quick" in result.stdout

    @pytest.mark.skipif(
        not (MATPLOTLIB_AVAILABLE), reason="plot extra is not installed"
    )
    def test_plot_helpers_render_with_plot_extra(self):
        distance = np.array([0.0, 0.5, 1.0])
        snapshot = SpatialSnapshot(
            distance,
            np.zeros(3),
            np.array([0.0, 0.5, 0.0]),
            np.zeros(3),
            -np.ones(3),
        )

        with TemporaryDirectory() as directory:
            output = Path(directory) / "population.png"
            plot_population(snapshot, 1, output, title="Population")
            assert output.stat().st_size > 0


class TestZiolkowskiUnits:
    @pytest.mark.parametrize(
        ("quantity", "value"),
        (
            pytest.param("time", 1e-15, id="time-femtosecond"),
            pytest.param("time", 5e-14, id="time-fifty-femtoseconds"),
            pytest.param("time", 1e-10, id="time-hundred-picoseconds"),
            pytest.param("field", 1.0, id="field-unit"),
            pytest.param("field", 4.2186e9, id="field-sit"),
            pytest.param("field", 2.272e10, id="field-ultrafast"),
        ),
    )
    def test_si_round_trips(self, quantity, value):
        if quantity == "time":
            assert round(abs(UNITS.time_si(UNITS.time(value)) - value), 7) == 0
        else:
            assert np.isclose(
                UNITS.electric_field_si(UNITS.electric_field(value)), value, rtol=1e-15
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
        assert round(abs(gamma - expected_bloch), 7) == 0
        assert round(abs(atom_density * gamma - expected_maxwell), 7) == 0
        assert round(abs(parameters["omega"][0] - 2 * pi / LAMBDA0_UM), 7) == 0


class TestZiolkowskiSource:
    def test_carrier_intensity_recovers_unit_sinusoid(self):
        sample_interval = PERIOD_S / 100
        times = np.arange(1_000) * sample_interval
        field = 3.0 * np.sin(OMEGA0_RAD_S * times)

        intensity = carrier_intensity(field, sample_interval, 3.0)
        hann_intensity = carrier_intensity(
            field, sample_interval, 3.0, periods=3, window="hann"
        )

        assert np.allclose(intensity[100:-100], 1.0, atol=0.01)
        assert np.allclose(hann_intensity[200:-200], 1.0, atol=0.01)
        with pytest.raises(ValueError, match="period count"):
            carrier_intensity(field, sample_interval, 3.0, periods=0)
        with pytest.raises(ValueError, match="unsupported envelope window"):
            carrier_intensity(field, sample_interval, 3.0, window="triangle")

    def test_sech_pulse_support_and_envelope_area(self):
        width = UNITS.time(20 / F0_HZ)
        pulse = SechSinePulse(UNITS.angular_frequency(OMEGA0_RAD_S), width)
        assert pulse.oscillator(-1) == 0
        assert pulse.oscillator(width + 1) == 0
        assert pulse.envelope(width / 2) == 1

        times = np.linspace(0, width, 100_001)
        numerical = np.trapezoid([pulse.envelope(time) for time in times], times)
        analytic = width / 10 * np.arctan(np.sinh(10))
        assert round(abs(numerical - analytic), 10) == 0

    def test_ultrafast_pulse_has_zero_area_and_smooth_endpoints(self):
        width = UNITS.time(PERIOD_S)
        pulse = UltrafastPulse(width)
        times = np.linspace(0, width, 100_001)
        values = np.array([pulse.oscillator(time) for time in times])

        assert values[0] == 0
        assert values[-1] == 0
        assert round(abs(np.trapezoid(values, times) - 0), 12) == 0
        assert round(abs((values[1] - values[0]) / (times[1] - times[0]) - 0), 6) == 0
        assert (
            round(abs((values[-1] - values[-2]) / (times[-1] - times[-2]) - 0), 6) == 0
        )

    def test_gain_and_pump_probe_delays(self):
        width = UNITS.time(PERIOD_S)
        omega = UNITS.angular_frequency(OMEGA0_RAD_S)
        sine = SmoothSine(omega, width)
        assert sine.oscillator(-1) == 0
        turn_on = np.array(
            [sine.envelope(time) for time in np.linspace(0, sine.rise_time, 101)]
        )
        assert turn_on[0] == 0
        assert turn_on[-1] == 1
        assert np.all(np.diff(turn_on) >= 0)
        assert round(abs(sine.oscillator(5 * width + width / 4) - 1), 7) == 0

        signal = PumpProbe(omega, width, beta=1e-4, delay=20 * width)
        delayed_turn_on = np.array(
            [
                signal.probe.envelope(time - signal.delay)
                for time in np.linspace(signal.delay, signal.delay + 5 * width, 101)
            ]
        )
        assert np.all(np.diff(delayed_turn_on) >= 0)
        assert signal.probe.oscillator(-1) == 0
        assert signal.oscillator(10 * width) == 0
        assert signal.oscillator(20 * width + width / 4) != 0


class TestZiolkowskiScenario:
    def test_paper_cell_counts_and_resolutions(self):
        assert sit_scenario(2).cells == 20_000
        assert ultrafast_scenario(5).cells == 2_000
        assert ultrafast_scenario(9).cells == 5_000
        assert gain_scenario().cells == 2_000
        pump_probe = pump_probe_scenario(20)
        assert pump_probe.cells == 4_000
        assert (
            round(abs(pump_probe.domain_um / pump_probe.cells - LAMBDA0_UM / 400), 7)
            == 0
        )

    def test_material_and_probe_geometry(self):
        scenario = gain_scenario(quick=True)
        simulation = make_simulation(scenario)
        assert isinstance(simulation, TorchSimulation)
        assert simulation.plan.shapes["Ex"][2] - 1 == scenario.cells
        expected_dt = 0.5 / np.sqrt(sum(delta**-2 for delta in simulation.plan.dr))
        assert round(abs(simulation.plan.dt - expected_dt), 7) == 0

        dm2 = next(
            state
            for state in simulation.dm2_state_snapshot()
            if state["component"] == "Ex"
        )
        dm2_z_indices = set(
            np.unravel_index(dm2["targets"], simulation.plan.shapes["Ex"])[2]
        )
        for distance, expected_rho30 in (
            (2.0, None),
            (5.0, 1.0),
            (13.0, None),
        ):
            coordinate = distance - scenario.domain_um / 2
            index = int(simulation.space.space_to_ex_index(0, 0, coordinate)[2])
            if expected_rho30 is None:
                assert index not in dm2_z_indices
            else:
                assert index in dm2_z_indices
                assert sample_snapshot(simulation).rho3[index] == expected_rho30

    def test_torch_snapshot_checkpoint_and_sampling_clock(self):
        scenario = gain_scenario(quick=True)
        simulation = make_simulation(scenario)
        checkpoint = simulation.checkpoint()
        first_time = 2 * UNITS.time_si(simulation.plan.dt)
        snapshots = run_snapshots(simulation, (first_time,))
        snapshot = snapshots[first_time]
        assert snapshot.electric.shape == (scenario.cells,)
        assert np.isfinite(snapshot.electric).all()
        assert int(simulation.state.step_count.detach().cpu()) == 2
        assert np.isclose(
            simulation.dm2_state_snapshot()[0]["time"],
            2 * simulation.plan.dt,
        )
        simulation.load_checkpoint(checkpoint)
        assert int(simulation.state.step_count.detach().cpu()) == 0

        result = run_gain(
            scenario,
            duration_s=2 * UNITS.time_si(simulation.plan.dt),
            sample_stride=1,
        )
        assert np.allclose(
            result.time_s,
            UNITS.time_si(simulation.plan.dt * np.array((0.5, 1.5))),
        )
        assert np.isfinite(result.input_intensity).all()
        assert np.isfinite(result.output_intensity).all()

    def test_relaxation_parameters_are_figure_specific(self):
        assert gain_scenario().t1_s == GAIN_T1_S
        assert gain_scenario().t2_s == GAIN_T2_S
        assert sit_scenario(2).t1_s == SIT_T1_S
        assert sit_scenario(2).t2_s == SIT_T2_S
