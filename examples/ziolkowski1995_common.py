#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared machinery for reproducing Ziolkowski et al. (1995).

The paper reports SI quantities, while GMES advances Maxwell's equations in
units where the vacuum wave speed is one.  This module keeps the conversion in
one place and exposes the four simulation families used by Figs. 1-12.

Reference:
    R. W. Ziolkowski, J. M. Arnold, and D. M. Gogny, "Ultrafast pulse
    interactions with two-level atoms," Phys. Rev. A 52, 3082-3094 (1995).
    https://doi.org/10.1103/PhysRevA.52.3082
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sin
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from gmes import (
    Block,
    Cartesian,
    Cpml,
    DefaultMedium,
    Dielectric,
    Dm2,
    Ex,
    PointSource,
    Shell,
    TEMzFDTD,
)
from gmes.source import SrcTime

PAPER_C = 3.0e8
PAPER_EPS0 = 8.854187817e-12
PAPER_HBAR = 1.0546e-34
F0_HZ = 2.0e14
OMEGA0_RAD_S = 2 * pi * F0_HZ
LAMBDA0_UM = PAPER_C / F0_HZ / 1.0e-6
PERIOD_S = 1 / F0_HZ
N_ATOM_M3 = 1.0e24
GAMMA_C_M = 1.0e-29
SIT_T1_S = 1.0e-10
SIT_T2_S = 1.0e-10
GAIN_T1_S = 1.0e-10
GAIN_T2_S = 5.0e-14
FIELD_SCALE_V_M = 1.0e9
# Figs. 5-8 label the snapshot as 12.5 fs, but the plotted pulse is 6-7.5 um
# from a z=0 source.  At c this requires 25 fs of propagation.  Treat the
# first 12.5 fs as the paper's unreported pre-roll and retain its time label.
ULTRAFAST_PREROLL_S = 12.5e-15


@dataclass(frozen=True)
class GmesUnits:
    """Conversion between SI and the dimensionless GMES Maxwell system."""

    length_m: float = 1.0e-6
    electric_field_v_m: float = FIELD_SCALE_V_M
    wave_speed_m_s: float = PAPER_C
    vacuum_permittivity_f_m: float = PAPER_EPS0
    reduced_planck_j_s: float = PAPER_HBAR

    def length(self, value_m: float) -> float:
        return value_m / self.length_m

    def time(self, value_s: float) -> float:
        return value_s * self.wave_speed_m_s / self.length_m

    def time_si(self, value: float) -> float:
        return value * self.length_m / self.wave_speed_m_s

    def angular_frequency(self, value_rad_s: float) -> float:
        return value_rad_s * self.length_m / self.wave_speed_m_s

    def electric_field(self, value_v_m: float) -> float:
        return value_v_m / self.electric_field_v_m

    def electric_field_si(self, value: float) -> float:
        return value * self.electric_field_v_m

    def dm2_parameters(
        self,
        *,
        omega_rad_s: float,
        atom_density_m3: float,
        dipole_c_m: float,
        t1_s: float,
        t2_s: float,
    ) -> dict[str, float | tuple[float, ...]]:
        """Return a self-consistent dimensionless ``Dm2`` parameter set.

        With ``E = E_ref E_g``, ``z = L_ref z_g``, and
        ``t = L_ref t_g / c``, the Bloch coupling becomes
        ``gamma_g / hbar_g = gamma E_ref L_ref / (hbar c)``.  Choosing
        ``hbar_g = 1`` then requires the density-like Maxwell coefficient
        below so that ``n_g gamma_g = N gamma / (eps0 E_ref)``.
        """

        gamma = (
            dipole_c_m
            * self.electric_field_v_m
            * self.length_m
            / (self.reduced_planck_j_s * self.wave_speed_m_s)
        )
        atom_density = (
            atom_density_m3
            * self.reduced_planck_j_s
            * self.wave_speed_m_s
            / (
                self.vacuum_permittivity_f_m
                * self.electric_field_v_m**2
                * self.length_m
            )
        )
        return {
            "omega": (self.angular_frequency(omega_rad_s),),
            "n_atom": (atom_density,),
            "gamma": gamma,
            "hbar": 1.0,
            "t1": self.time(t1_s),
            "t2": self.time(t2_s),
        }


UNITS = GmesUnits()


class PaperSourceTime(SrcTime):
    """Base class for real-valued source functions used by the paper."""

    def init(self, cmplx):
        if cmplx:
            raise ValueError("Ziolkowski reproductions require real fields")

    def display_info(self, indent=0):
        print(" " * indent + self.__class__.__name__)


class SechSinePulse(PaperSourceTime):
    """Finite 20-cycle SIT pulse from Eqs. (21), (22), and (25)."""

    def __init__(self, omega: float, pulse_width: float):
        self.omega = float(omega)
        self.pulse_width = float(pulse_width)

    def envelope(self, time: float) -> float:
        if not 0 <= time <= self.pulse_width:
            return 0.0
        gamma = (time - self.pulse_width / 2) / (self.pulse_width / 2)
        return 1 / np.cosh(10 * gamma)

    def oscillator(self, time):
        return self.envelope(time) * sin(self.omega * time)


class UltrafastPulse(PaperSourceTime):
    """Twice continuously differentiable zero-area pulse from Eq. (28)."""

    SHAPE_FACTOR = -4.201355

    def __init__(self, pulse_width: float):
        self.pulse_width = float(pulse_width)

    def oscillator(self, time):
        if not 0 <= time <= self.pulse_width:
            return 0.0
        x_value = 2 * time / self.pulse_width - 1
        return self.SHAPE_FACTOR * x_value * (1 - x_value**2) ** 3


class UltrafastPulseTrain(PaperSourceTime):
    """Two-pulse excitation used for Fig. 9."""

    def __init__(
        self, pulse_width: float, alpha: float = 0.96, delay_periods: float = 3
    ):
        self.pulse = UltrafastPulse(pulse_width)
        self.pulse_width = float(pulse_width)
        self.alpha = float(alpha)
        self.delay = float(delay_periods) * self.pulse_width

    def oscillator(self, time):
        return self.pulse.oscillator(time) + self.alpha * self.pulse.oscillator(
            time - self.delay
        )


class SmoothSine(PaperSourceTime):
    """Resonant sinusoid from Ziolkowski et al. (1995), Eq. (29).

    Interpreting the printed five-period interval as ``x = 2t/(5Tp) - 1``
    makes the polynomial envelope return to zero before the continuous-wave
    branch starts. This produces the narrow precursor visible at the start of
    the input and output traces in Fig. 10. See ``VERIFICATION.md`` for the
    ambiguity in the paper's description of this function as a smooth turn-on.
    """

    def __init__(self, omega: float, period: float):
        self.omega = float(omega)
        self.period = float(period)
        self.rise_time = 5 * self.period

    def oscillator(self, time):
        if time < 0:
            return 0.0
        envelope = 1.0
        if time <= self.rise_time:
            x_value = 2 * time / self.rise_time - 1
            envelope = (1 - x_value**2) ** 4
        return envelope * sin(self.omega * time)


class PumpProbe(PaperSourceTime):
    """Ultrafast pump followed by a weak resonant probe."""

    def __init__(self, omega: float, pulse_width: float, beta: float, delay: float):
        self.pump = UltrafastPulse(pulse_width)
        self.probe = SmoothSine(omega, pulse_width)
        self.beta = float(beta)
        self.delay = float(delay)

    def oscillator(self, time):
        return self.pump.oscillator(time) + self.beta * self.probe.oscillator(
            time - self.delay
        )


@dataclass(frozen=True)
class Scenario:
    """Resolved physical and numerical configuration for a figure family."""

    domain_um: float
    cells: int
    medium_start_um: float
    medium_end_um: float
    t1_s: float
    t2_s: float
    rho30: float
    source_time: PaperSourceTime
    amplitude_v_m: float
    pml_um: float

    @property
    def resolution(self) -> float:
        return self.cells / self.domain_um


@dataclass
class SpatialSnapshot:
    distance_um: np.ndarray
    electric: np.ndarray
    rho1: np.ndarray
    rho2: np.ndarray
    rho3: np.ndarray


@dataclass
class GainResult:
    time_s: np.ndarray
    input_intensity: np.ndarray
    output_intensity: np.ndarray
    snapshot: SpatialSnapshot
    input_point_um: float
    output_point_um: float


def carrier_intensity(
    field: np.ndarray,
    sample_interval_s: float,
    normalization_amplitude: float,
) -> np.ndarray:
    """Return a one-carrier-period moving intensity envelope.

    A local average of ``2 E**2`` reproduces the envelope detector used for
    Figs. 10 and 12 without the nonlocal edge ringing that a finite-record
    Hilbert transform introduces after the much larger pump pulse.
    """

    window_size = max(2, round(PERIOD_S / sample_interval_s))
    left_padding = (window_size - 1) // 2
    right_padding = window_size - 1 - left_padding
    padded = np.pad(
        np.asarray(field) ** 2,
        (left_padding, right_padding),
        mode="reflect",
    )
    mean_square = np.convolve(
        padded, np.full(window_size, 1 / window_size), mode="valid"
    )
    return 2 * mean_square / normalization_amplitude**2


def paper_dm2(t1_s: float, t2_s: float, rho30: float) -> Dm2:
    parameters = UNITS.dm2_parameters(
        omega_rad_s=OMEGA0_RAD_S,
        atom_density_m3=N_ATOM_M3,
        dipole_c_m=GAMMA_C_M,
        t1_s=t1_s,
        t2_s=t2_s,
    )
    return Dm2(eps_inf=1, mu_inf=1, rho30=rho30, rtol=1.0e-8, **parameters)


def _quick_cells(full_cells: int, quick: bool) -> int:
    return max(80, full_cells // 10) if quick else full_cells


def sit_scenario(area_pi: int, quick: bool = False) -> Scenario:
    amplitude = 4.2186e9 * area_pi / 2
    return Scenario(
        domain_um=150.0,
        cells=_quick_cells(20_000, quick),
        medium_start_um=7.5,
        medium_end_um=142.5,
        t1_s=SIT_T1_S,
        t2_s=SIT_T2_S,
        rho30=-1.0,
        source_time=SechSinePulse(
            UNITS.angular_frequency(OMEGA0_RAD_S), UNITS.time(20 / F0_HZ)
        ),
        amplitude_v_m=amplitude,
        pml_um=3.0,
    )


def ultrafast_scenario(figure: int, quick: bool = False) -> Scenario:
    pulse_width = UNITS.time(PERIOD_S)
    if figure == 9:
        return Scenario(
            domain_um=37.5,
            cells=_quick_cells(5_000, quick),
            medium_start_um=3.75,
            medium_end_um=33.75,
            t1_s=SIT_T1_S,
            t2_s=SIT_T2_S,
            rho30=-1.0,
            source_time=UltrafastPulseTrain(pulse_width),
            amplitude_v_m=8.235e9,
            pml_um=1.5,
        )
    amplitude = 2.272e10 if figure == 8 else 8.205e9
    return Scenario(
        domain_um=15.0,
        cells=_quick_cells(2_000, quick),
        medium_start_um=1.5,
        medium_end_um=13.5,
        t1_s=SIT_T1_S,
        t2_s=SIT_T2_S,
        rho30=-1.0,
        source_time=UltrafastPulse(pulse_width),
        amplitude_v_m=amplitude,
        pml_um=0.75,
    )


def gain_scenario(quick: bool = False) -> Scenario:
    """Return the parameters reported for Figs. 10-11 and Eq. (29)."""

    return Scenario(
        domain_um=15.0,
        cells=_quick_cells(2_000, quick),
        medium_start_um=3.0,
        medium_end_um=12.0,
        t1_s=GAIN_T1_S,
        t2_s=GAIN_T2_S,
        rho30=1.0,
        source_time=SmoothSine(
            UNITS.angular_frequency(OMEGA0_RAD_S), UNITS.time(PERIOD_S)
        ),
        amplitude_v_m=1.0,
        pml_um=0.75,
    )


def pump_probe_scenario(delay_periods: int, quick: bool = False) -> Scenario:
    pulse_width = UNITS.time(PERIOD_S)
    return Scenario(
        domain_um=15.0,
        cells=_quick_cells(4_000, quick),
        medium_start_um=3.0,
        medium_end_um=12.0,
        t1_s=GAIN_T1_S,
        t2_s=GAIN_T2_S,
        rho30=-1.0,
        source_time=PumpProbe(
            UNITS.angular_frequency(OMEGA0_RAD_S),
            pulse_width,
            beta=1.0e-4,
            delay=delay_periods * pulse_width,
        ),
        amplitude_v_m=8.232e9,
        pml_um=0.75,
    )


def make_simulation(scenario: Scenario, verbose: bool = False) -> TEMzFDTD:
    """Construct the paper's one-dimensional Ex/Hy configuration."""

    space = Cartesian(size=(0, 0, scenario.domain_um), resolution=scenario.resolution)
    medium_center = (
        scenario.medium_start_um + scenario.medium_end_um
    ) / 2 - scenario.domain_um / 2
    medium_length = scenario.medium_end_um - scenario.medium_start_um
    geometry = [
        DefaultMedium(material=Dielectric()),
        Block(
            material=paper_dm2(scenario.t1_s, scenario.t2_s, scenario.rho30),
            center=(0, 0, medium_center),
            size=(1, 1, medium_length),
        ),
        Shell(
            material=Cpml(),
            thickness=scenario.pml_um,
            minus_x=False,
            plus_x=False,
            minus_y=False,
            plus_y=False,
            minus_z=False,
            plus_z=True,
        ),
    ]
    source = PointSource(
        src_time=scenario.source_time,
        center=(0, 0, -scenario.domain_um / 2),
        component=Ex,
        amp=UNITS.electric_field(scenario.amplitude_v_m),
    )
    simulation = TEMzFDTD(
        space,
        geometry,
        [source],
        courant_ratio=0.5,
        verbose=verbose,
    )
    simulation.init()
    return simulation


def _dm2_electric(simulation: TEMzFDTD):
    return next(
        material
        for material in simulation.pw_material[Ex].values()
        if material.name() == "Dm2Electric"
    )


def sample_snapshot(simulation: TEMzFDTD) -> SpatialSnapshot:
    count = int(simulation.space.whole_field_size[2])
    distance = np.arange(count, dtype=float) * simulation.dz
    electric = np.array(simulation.ex[0, 0, :count], copy=True)
    dm2 = _dm2_electric(simulation)
    time = simulation.time_step.t
    populations = []
    for rho_index in range(3):
        populations.append(
            np.fromiter(
                (
                    dm2.get_rho((0, 0, index), 0, rho_index, time)
                    for index in range(count)
                ),
                dtype=float,
                count=count,
            )
        )
    return SpatialSnapshot(distance, electric, *populations)


def run_snapshots(
    simulation: TEMzFDTD, times_s: tuple[float, ...]
) -> dict[float, SpatialSnapshot]:
    """Advance once and capture snapshots at the nearest time steps."""

    targets = {
        int(round(UNITS.time(time_s) / simulation.time_step.dt)): time_s
        for time_s in times_s
    }
    snapshots = {}
    final_step = max(targets)
    started = perf_counter()
    while simulation.time_step.n < final_step:
        simulation.step()
        step = int(round(simulation.time_step.n))
        if step in targets:
            snapshots[targets[step]] = sample_snapshot(simulation)
    elapsed = perf_counter() - started
    print(
        f"completed {final_step:,} steps over "
        f"{simulation.space.whole_field_size[2]:,} cells in {elapsed:.2f} s"
    )
    return snapshots


def _nearest_index(simulation: TEMzFDTD, distance_um: float) -> int:
    coordinate = distance_um - simulation.space.half_size[2]
    return int(simulation.space.space_to_ex_index(0, 0, coordinate)[2])


def run_gain(
    scenario: Scenario,
    duration_s: float,
    sample_stride: int = 10,
    verbose: bool = False,
    normalization_amplitude_v_m: float | None = None,
) -> GainResult:
    """Run a gain case and record the probes used by Figs. 10 and 12."""

    simulation = make_simulation(scenario, verbose=verbose)
    input_point = scenario.medium_start_um - 10 / scenario.resolution
    output_point = scenario.medium_end_um + 10 / scenario.resolution
    input_index = _nearest_index(simulation, input_point)
    output_index = _nearest_index(simulation, output_point)
    final_step = int(round(UNITS.time(duration_s) / simulation.time_step.dt))
    times = []
    input_field = []
    output_field = []
    started = perf_counter()
    while simulation.time_step.n < final_step:
        simulation.step()
        if int(round(simulation.time_step.n)) % sample_stride == 0:
            times.append(UNITS.time_si(simulation.time_step.t))
            input_field.append(simulation.ex[0, 0, input_index])
            output_field.append(simulation.ex[0, 0, output_index])
    elapsed = perf_counter() - started
    print(
        f"completed {final_step:,} steps over {scenario.cells:,} cells in "
        f"{elapsed:.2f} s"
    )
    if normalization_amplitude_v_m is None:
        normalization_amplitude_v_m = scenario.amplitude_v_m
    normalization_amplitude = UNITS.electric_field(normalization_amplitude_v_m)
    sample_interval_s = sample_stride * UNITS.time_si(simulation.time_step.dt)
    input_intensity = carrier_intensity(
        np.asarray(input_field), sample_interval_s, normalization_amplitude
    )
    output_intensity = carrier_intensity(
        np.asarray(output_field), sample_interval_s, normalization_amplitude
    )
    return GainResult(
        np.asarray(times),
        input_intensity,
        output_intensity,
        sample_snapshot(simulation),
        input_point,
        output_point,
    )


def population_mask(snapshot: SpatialSnapshot, scenario: Scenario) -> np.ndarray:
    inside = (snapshot.distance_um >= scenario.medium_start_um) & (
        snapshot.distance_um <= scenario.medium_end_um
    )
    return np.where(inside, scenario.rho30, 0.0)


def print_population_summary(snapshot: SpatialSnapshot, scenario: Scenario):
    inside = (snapshot.distance_um >= scenario.medium_start_um) & (
        snapshot.distance_um <= scenario.medium_end_um
    )
    rho3 = snapshot.rho3[inside]
    bloch_norm = (
        snapshot.rho1[inside] ** 2
        + snapshot.rho2[inside] ** 2
        + snapshot.rho3[inside] ** 2
    )
    print(
        f"rho3 range={rho3.min():.6f}:{rho3.max():.6f}, "
        f"Bloch norm range={bloch_norm.min():.6f}:{bloch_norm.max():.6f}"
    )


def print_intensity_summary(
    result: GainResult,
    *,
    peak_window_s: tuple[float, float],
    late_window_s: tuple[float, float],
):
    def values(window):
        mask = (result.time_s >= window[0]) & (result.time_s <= window[1])
        return result.output_intensity[mask]

    peak = values(peak_window_s)
    late = values(late_window_s)
    print(
        f"output intensity peak={peak.max():.6f}, " f"late median={np.median(late):.6f}"
    )


def configure_axes(axis, *, x_label=r"Distance ($\mu$m)"):
    axis.set_xlabel(x_label)
    axis.set_ylabel("Normalized amplitude")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")


def figure_title(title: str, quick: bool) -> str:
    return f"{title} (quick smoke test)" if quick else title


def save_plot(figure, path: Path, show: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    if show:
        plt.show()
    plt.close(figure)
    print(f"wrote {path}")


def plot_spatial_pair(
    snapshot: SpatialSnapshot,
    scenario: Scenario,
    path: Path,
    *,
    x_limits: tuple[float, float],
    title: str,
    show: bool = False,
):
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(
        snapshot.distance_um,
        snapshot.electric / UNITS.electric_field(scenario.amplitude_v_m),
        linestyle="--",
        label=r"$E_x / E_{\max}$",
    )
    axis.plot(snapshot.distance_um, snapshot.rho3, label=r"$\rho_3$")
    axis.set_xlim(*x_limits)
    axis.set_ylim(-1.2, 1.2)
    axis.set_title(title)
    configure_axes(axis)
    save_plot(figure, path, show)


def plot_population(
    snapshot: SpatialSnapshot,
    rho_index: int,
    path: Path,
    *,
    title: str,
    x_limits: tuple[float, float] | None = None,
    show: bool = False,
):
    values = (snapshot.rho1, snapshot.rho2)[rho_index - 1]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(snapshot.distance_um, values, label=rf"$\rho_{{{rho_index}}}$")
    axis.set_xlim(*(x_limits or (0, snapshot.distance_um[-1])))
    axis.set_title(title)
    configure_axes(axis)
    save_plot(figure, path, show)


def plot_gain(
    result: GainResult,
    path: Path,
    *,
    title: str,
    show: bool = False,
):
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(result.time_s / 1.0e-12, result.output_intensity, label="output")
    axis.plot(
        result.time_s / 1.0e-12,
        result.input_intensity,
        linestyle="--",
        label="input",
    )
    axis.set_title(title)
    axis.set_xlabel("Time (ps)")
    axis.set_ylabel("Normalized intensity")
    axis.set_xlim(0, 0.85)
    axis.set_ylim(0, 1.6)
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    save_plot(figure, path, show)


def print_scenario(scenario: Scenario):
    parameters = UNITS.dm2_parameters(
        omega_rad_s=OMEGA0_RAD_S,
        atom_density_m3=N_ATOM_M3,
        dipole_c_m=GAMMA_C_M,
        t1_s=scenario.t1_s,
        t2_s=scenario.t2_s,
    )
    print(
        f"domain={scenario.domain_um:g} um, cells={scenario.cells:,}, "
        f"dz={scenario.domain_um / scenario.cells:g} um"
    )
    print(
        f"medium={scenario.medium_start_um:g}:{scenario.medium_end_um:g} um, "
        f"Emax={scenario.amplitude_v_m:.7g} V/m"
    )
    print(f"normalized Dm2 parameters: {parameters}")
