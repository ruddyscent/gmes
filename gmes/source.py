#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Define temporal waveforms and spatial sources for legacy FDTD simulations."""

from cmath import exp as cexp
from collections.abc import Callable, Iterator, Sequence
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, replace
from math import cos, exp, pi, sin, sqrt
from os import PathLike
from typing import Any, Protocol, cast

import numpy as np
from numpy import cross, dot, inf
from numpy.linalg import norm
from numpy.typing import NDArray

# scipy-stubs does not yet support the project's Python 3.14 environment.
from scipy.optimize import bisect  # type: ignore[import-untyped]

from . import constant as const
from .geometry import Cartesian, DefaultMedium, GeomBoxTree, Shell
from .material import Cpml, Dielectric
from .pygeom import Material

type Vector3 = Sequence[float] | NDArray[np.float64]
type Coordinate3 = tuple[float | np.float64, float | np.float64, float | np.float64]
type Index3 = tuple[int, int, int]
type ComponentType = (
    type[const.Ex]
    | type[const.Ey]
    | type[const.Ez]
    | type[const.Hx]
    | type[const.Hy]
    | type[const.Hz]
    | type[const.Jx]
    | type[const.Jy]
    | type[const.Jz]
    | type[const.Mx]
    | type[const.My]
    | type[const.Mz]
)
type DirectionType = (
    type[const.PlusX]
    | type[const.MinusX]
    | type[const.PlusY]
    | type[const.MinusY]
    | type[const.PlusZ]
    | type[const.MinusZ]
)
type RealArray = NDArray[np.float64]
type ComplexArray = NDArray[np.complex128]
type FieldArray = RealArray | ComplexArray
type RealScalar = float | np.float64
type FieldScalar = float | complex | np.float64 | np.complex128
type SourcePoint = tuple[Index3, Coordinate3, Material, Material | None]


_COMPONENT_SHAPE_OFFSETS: dict[str, Index3] = {
    "Ex": (0, 1, 1),
    "Ey": (1, 0, 1),
    "Ez": (1, 1, 0),
    "Hx": (1, 0, 0),
    "Hy": (0, 1, 0),
    "Hz": (0, 0, 1),
}


def _is_valid_nonlocal_point_target(
    space: Cartesian, component: str, target: Index3
) -> bool:
    """Return whether a globally valid Yee target belongs to another rank."""

    if int(space.numprocs) == 1:
        return False
    shape_offset = _COMPONENT_SHAPE_OFFSETS[component]
    local_shape = tuple(
        int(size) + offset
        for size, offset in zip(space.my_field_size, shape_offset, strict=True)
    )
    if all(0 <= index < size for index, size in zip(target, local_shape, strict=True)):
        return False
    global_target = tuple(
        index + int(offset)
        for index, offset in zip(target, space.global_field_offset, strict=True)
    )
    global_shape = tuple(
        int(size) + offset
        for size, offset in zip(space.whole_field_size, shape_offset, strict=True)
    )
    return all(
        0 <= index < size
        for index, size in zip(global_target, global_shape, strict=True)
    )


@dataclass(frozen=True)
class AuxiliarySourceSpec:
    """Torch-neutral description of a TFSF auxiliary simulation."""

    space: Cartesian
    geometry: tuple[object, ...]
    sources: tuple[object, ...]
    gaussian_width: float | None = None
    prewarm_steps: int = 0


@dataclass(frozen=True)
class PointSourceRecord:
    """Direct point/current lowering independent of a pointwise proxy."""

    component: str
    target: Index3
    source_time: object
    amplitude: float
    current_scale: float | None


@dataclass(frozen=True)
class TfsfFaceRule:
    """One direct TFSF interpolation rule with its immutable auxiliary spec."""

    component: str
    target: Index3
    sample_component: str
    sample0: Index3
    sample1: Index3
    weight0: float
    weight1: float
    coefficient: float
    auxiliary_spec: AuxiliarySourceSpec


@dataclass(frozen=True)
class _TorchTfsfLowering:
    geometry_tree: GeomBoxTree
    auxiliary_spec: AuxiliarySourceSpec


_torch_tfsf_lowering: ContextVar[_TorchTfsfLowering | None] = ContextVar(
    "torch_tfsf_lowering", default=None
)

#
# SrcTime: Continuous, Bandpass
# Src: PointSource, GaussianBeam, TotalFieldScatteredField
#


class SrcTime(object):
    """Time-dependent part of a source."""

    def init(self, cmplx: bool) -> None:
        """Configure whether oscillator values must retain a complex phase."""

        raise NotImplementedError

    def oscillator(self, time: float) -> float | complex:
        """Return the source amplitude at a physical simulation time."""

        raise NotImplementedError

    def display_info(self, indent: int = 0) -> None:
        """Print a human-readable waveform summary."""

        raise NotImplementedError


class PaperSourceTime(SrcTime):
    """Base class for the real-valued Ziolkowski et al. (1995) waveforms."""

    def init(self, cmplx: bool) -> None:
        if cmplx:
            raise ValueError("Ziolkowski reproductions require real fields")

    def display_info(self, indent: int = 0) -> None:
        print(" " * indent + self.__class__.__name__)


class SechSinePulse(PaperSourceTime):
    """Finite 20-cycle SIT pulse from Eqs. (21), (22), and (25)."""

    def __init__(self, omega: float, pulse_width: float) -> None:
        self.omega = float(omega)
        self.pulse_width = float(pulse_width)

    def envelope(self, time: float) -> float:
        """Return the compact sech envelope at ``time``."""

        if not 0 <= time <= self.pulse_width:
            return 0.0
        gamma = (time - self.pulse_width / 2) / (self.pulse_width / 2)
        return float(1 / np.cosh(10 * gamma))

    def oscillator(self, time: float) -> float:
        return self.envelope(time) * sin(self.omega * time)


class UltrafastPulse(PaperSourceTime):
    """Twice continuously differentiable zero-area pulse from Eq. (28)."""

    SHAPE_FACTOR = -4.201355

    def __init__(self, pulse_width: float) -> None:
        self.pulse_width = float(pulse_width)

    def oscillator(self, time: float) -> float:
        if not 0 <= time <= self.pulse_width:
            return 0.0
        x_value = 2 * time / self.pulse_width - 1
        return self.SHAPE_FACTOR * x_value * (1 - x_value**2) ** 3


class UltrafastPulseTrain(PaperSourceTime):
    """Two-pulse excitation used for Fig. 9."""

    def __init__(
        self, pulse_width: float, alpha: float = 0.96, delay_periods: float = 3
    ) -> None:
        self.pulse = UltrafastPulse(pulse_width)
        self.pulse_width = float(pulse_width)
        self.alpha = float(alpha)
        self.delay = float(delay_periods) * self.pulse_width

    def oscillator(self, time: float) -> float:
        return self.pulse.oscillator(time) + self.alpha * self.pulse.oscillator(
            time - self.delay
        )


class SmoothSine(PaperSourceTime):
    """Resonant sinusoid from Eq. (29) with the paper's monotone turn-on."""

    def __init__(self, omega: float, period: float) -> None:
        self.omega = float(omega)
        self.period = float(period)
        self.rise_time = 5 * self.period

    def envelope(self, time: float) -> float:
        """Return the monotone resonant turn-on envelope at ``time``."""

        if time < 0:
            return 0.0
        if time >= self.rise_time:
            return 1.0
        x_value = time / self.rise_time - 1
        return (1 - x_value**2) ** 4

    def oscillator(self, time: float) -> float:
        return self.envelope(time) * sin(self.omega * time)


class PumpProbe(PaperSourceTime):
    """Ultrafast pump followed by the weak resonant probe used in Fig. 12."""

    def __init__(
        self, omega: float, pulse_width: float, beta: float, delay: float
    ) -> None:
        self.pump = UltrafastPulse(pulse_width)
        self.probe = SmoothSine(omega, pulse_width)
        self.beta = float(beta)
        self.delay = float(delay)

    def oscillator(self, time: float) -> float:
        return self.pump.oscillator(time) + self.beta * self.probe.oscillator(
            time - self.delay
        )


class PlaneWaveSrcTime(Protocol):
    """Temporal source contract required by total-field plane waves."""

    freq: float
    width: float

    def init(self, cmplx: bool) -> None:
        """Initialize the waveform for real or complex fields."""

    def oscillator(self, time: float) -> float | complex:
        """Evaluate the waveform at ``time``."""

    def display_info(self, indent: int = 0) -> None:
        """Print a human-readable waveform summary."""


class Src(object):
    """Space-dependent part of a source."""

    def display_info(self, indent: int = 0) -> None:
        """Print a human-readable spatial source summary."""

        raise NotImplementedError

    def init(self, geom_tree: GeomBoxTree, space: Cartesian, cmplx: bool) -> None:
        """Bind geometry and grid state before point-wise source lowering."""

        raise NotImplementedError

    def step(self) -> None:
        """Advance any auxiliary simulation owned by the source."""

        raise NotImplementedError


class Continuous(SrcTime):
    """Continuous (CW) source with (optional) slow turn-on and/or turn-off."""

    def __init__(
        self,
        freq: float,
        phase: float = 0,
        start: float = 0,
        end: float = inf,
        width: float | None = None,
    ) -> None:
        self.freq = float(freq)
        self.phase = float(phase)
        self.start = float(start)
        self.end = float(end)

        if width is None:
            self.width = 5 / self.freq
        else:
            self.width = float(width)

    def init(self, cmplx: bool) -> None:
        self.cmplx = cmplx

    def display_info(self, indent: int = 0) -> None:
        print(" " * indent, "continuous source:")
        print(" " * indent, end=" ")
        print("frequency:", self.freq, end=" ")
        print("initial phase advance:", self.phase, end=" ")
        print("start time:", self.start, end=" ")
        print("end time:", self.end, end=" ")
        print("raising duration:", self.width)

    def oscillator(self, time: float) -> float | complex:
        ts = time - self.start
        te = self.end - time

        if ts < 0 or te < 0:
            return 0

        # Use Hanning window function to reduce the transition.
        # D. T. Prescott and N. V. Shuley, "Reducing solution time in
        # monochromatic FDTD waveguide simulations", IEEE Trans. Microwave
        # Theory Tech., vol. 42, no. 8, pp. 1582-1584, 8. 1994.
        rise = sin(0.5 * pi * ts / self.width) ** 2 if ts < self.width else 1
        fall = sin(0.5 * pi * te / self.width) ** 2 if te < self.width else 1
        env = rise * fall

        osc = env * cexp(2j * pi * self.freq * time + 1j * self.phase)
        if self.cmplx:
            return osc
        else:
            return osc.real


class Bandpass(SrcTime):
    """a pulse source with Gaussian-envelope"""

    def __init__(
        self, freq: float, fwidth: float, s: float = 10, phase: float = 0
    ) -> None:
        self.freq = float(freq)
        self.phase = float(phase)
        self.fwidth = float(fwidth)
        self.width = 1 / self.fwidth
        self.peak_time = self.width * s
        self.cutoff = 2 * self.width * s

        # Makes the last_source_time as small as possible.
        while exp(-0.5 * (self.cutoff / self.width) ** 2) == 0:
            self.cutoff *= 0.9

    def init(self, cmplx: bool) -> None:
        self.cmplx = cmplx

    def display_info(self, indent: int = 0) -> None:
        print(" " * indent, "bandpass source")
        print(" " * indent, end=" ")
        print("center frequency:", self.freq, end=" ")
        print("bandwidth:", self.fwidth, end=" ")
        print("peak time:", self.peak_time, end=" ")
        print("cutoff:", self.cutoff)

    def oscillator(self, time: float) -> float | complex:
        tt = time - self.peak_time
        if abs(tt) > self.cutoff:
            return 0

        # correction factor so that current amplitude (= d(oscillator)/dt)
        # is ~1 near the peak of the Gaussian.
        cfactor = 1.0 / (-2j * pi * self.freq)

        osc = (
            cfactor
            * exp(-0.5 * (tt / self.width) ** 2)
            * cexp(2j * pi * self.freq * time + 1j * self.phase)
        )
        if self.cmplx:
            return osc
        else:
            return osc.real


class DifferentiatedGaussian(SrcTime):
    """a differentiated Gaussian pulse

    -2((t-t0)/tw)exp(-((t-t0)/tw)**2)

    """

    def __init__(self, tw: float, t0: float) -> None:
        """Initialize a differentiated Gaussian pulse.

        Args:
            tw: Pulse half-width in simulation time units.
            t0: Delay to the pulse center in simulation time units.
        """
        self.tw = float(tw)
        self.t0 = float(t0)

    def init(self, cmplx: bool) -> None:
        self.cmplx = bool(cmplx)

    def oscillator(self, time: float) -> float | complex:
        exponent = -(((time - self.t0) / self.tw) ** 2)
        osc = -2 * (time - self.t0) / self.tw * cexp(exponent)
        if self.cmplx:
            return osc
        else:
            return osc.real

    def display_info(self, indent: int = 0) -> None:
        print(" " * indent, end=" ")
        print("differentiated Gaussian pulse")
        print(" " * indent, end=" ")
        print("half-width:", self.tw, end=" ")
        print("delay time:", self.t0)


class PointSource(Src):
    """Inject a temporal waveform into one field or current component.

    Args:
        src_time: Temporal waveform implementing the SrcTime contract.
        center: Three-dimensional physical source location.
        component: Field or current component class to excite.
        amp: Scalar source amplitude.
        filename: Optional path for recording evaluated source values.
    """

    def __init__(
        self,
        src_time: SrcTime,
        center: Vector3,
        component: ComponentType,
        amp: float = 1,
        filename: str | PathLike[str] | None = None,
    ) -> None:
        self.center = np.array(center, np.double)
        self.comp = component
        self.src_time = src_time
        self.amp = float(amp)
        if filename:
            self.filename: str | None = str(filename)
        else:
            self.filename = None

    def init(self, geom_tree: GeomBoxTree, space: Cartesian, cmplx: bool) -> None:
        self.geom_tree = geom_tree
        self.src_time.init(cmplx)

    def step(self) -> None:
        pass

    def display_info(self, indent: int = 0) -> None:
        print(" " * indent, "point source:")
        print(" " * indent, "center:", self.center)
        print(" " * indent, "component:", self.comp.str())
        print(" " * indent, "maximum amp.:", self.amp)
        print(" " * indent, "source recording:", self.filename)

        self.src_time.display_info(4)

    def lower_torch_source(self, context: object) -> tuple[PointSourceRecord, ...]:
        """Lower this built-in directly without a pointwise source proxy."""
        if self.filename is not None:
            raise ValueError(
                "PointSource filename output is unsupported by TorchSimulation; "
                "use an explicit bounded probe/output adapter"
            )
        space = cast(Any, context).space
        geometry_tree = cast(Any, context).geometry_tree
        dt = float(cast(Any, context).dt)
        components = {
            const.Ex: ("Ex", space.space_to_ex_index, space.ex_index_to_space, False),
            const.Ey: ("Ey", space.space_to_ey_index, space.ey_index_to_space, False),
            const.Ez: ("Ez", space.space_to_ez_index, space.ez_index_to_space, False),
            const.Hx: ("Hx", space.space_to_hx_index, space.hx_index_to_space, False),
            const.Hy: ("Hy", space.space_to_hy_index, space.hy_index_to_space, False),
            const.Hz: ("Hz", space.space_to_hz_index, space.hz_index_to_space, False),
            const.Jx: ("Ex", space.space_to_ex_index, space.ex_index_to_space, True),
            const.Jy: ("Ey", space.space_to_ey_index, space.ey_index_to_space, True),
            const.Jz: ("Ez", space.space_to_ez_index, space.ez_index_to_space, True),
            const.Mx: ("Hx", space.space_to_hx_index, space.hx_index_to_space, True),
            const.My: ("Hy", space.space_to_hy_index, space.hy_index_to_space, True),
            const.Mz: ("Hz", space.space_to_hz_index, space.hz_index_to_space, True),
        }
        component, to_index, to_space, current = components[self.comp]
        target = cast(Index3, tuple(int(value) for value in to_index(*self.center)))
        if _is_valid_nonlocal_point_target(space, component, target):
            return ()
        material, _underneath = geometry_tree.material_of_point(to_space(*target))
        inverse = material.eps_inf if component.startswith("E") else material.mu_inf
        return (
            PointSourceRecord(
                component,
                target,
                self.src_time,
                self.amp,
                -dt / inverse if current else None,
            ),
        )


class TotalFieldScatteredField(Src):
    """Set a total and scattered field zone to launch a plane wave."""

    _MAPPING_TILE_SIZE = 65536
    auxiliary_spec: AuxiliarySourceSpec

    def __init__(
        self,
        src_time: SrcTime,
        center: Vector3,
        size: Vector3,
        direction: Vector3,
        polarization: Vector3,
        amp: float = 1,
    ) -> None:
        """Constructor

        Arguments:
        center -- center of the incidence interface. The beam axis crosses
                  this point.
           type: a tuple with three real numbers.
        size --  size of the incidence interface plane.
           type: a tuple with three real numbers.
        direction -- propagation direction of the beam.
           type: a tuple with three real numbers.
        freq -- oscillating frequency of the beam.
           type: a real number
        polarization -- electric field direction of the beam.
           type: a tuple with three real numbers.
        amp -- amplitude of the plane wave. The default is 1.
           type: a real number

        """
        if isinstance(src_time, SrcTime):
            self.src_time: PlaneWaveSrcTime = cast(PlaneWaveSrcTime, src_time)
        else:
            raise TypeError("src_time must be an instance of SrcTime.")

        self.k = np.array(direction, np.double) / norm(direction)
        self.center = np.array(center, np.double)
        if not np.isfinite(self.center).all():
            raise ValueError("center must contain only finite values")

        self.size = np.array(size, np.double)
        if not np.isfinite(self.size).all():
            raise ValueError("size must contain only finite values")

        self.half_size = 0.5 * self.size
        self.e_direction = np.array(polarization, np.double) / norm(polarization)

        # direction of h field
        self.h_direction = cross(self.k, self.e_direction)

        # maximum amplitude of stimulus
        self.amp = float(amp)

        self.on_axis_k = self._axis_in_k()

    def init(self, geom_tree: GeomBoxTree, space: Cartesian, cmplx: bool) -> None:
        self.geom_tree = geom_tree
        self.src_time.init(cmplx)

        self.auxiliary_spec = self._get_auxiliary_spec(space, geom_tree)

    def step(self) -> None:
        pass

    def display_info(self, indent: int = 0) -> None:
        print(" " * indent, "plane-wave source:")
        print(" " * indent, "propagation direction:", self.k)
        print(" " * indent, "center:", self.center)
        print(" " * indent, "source plane size:", self.size)
        print(" " * indent, "polarization direction:", self.e_direction)
        print(" " * indent, "amplitude:", self.amp)

        self.src_time.display_info(4)

    def mode_function(self, x: float, y: float, z: float) -> float:
        """Return the unit spatial mode amplitude for a plane wave."""

        return 1.0

    def _dist_from_center(self, point: Vector3) -> float:
        """Calculate distance from the interface plane center.

        Arguments:
        point -- location in the space coordinate

        """
        return cast(float, norm(self.center - point))

    def _metric_from_center_along_beam_axis(self, point: Vector3) -> float:
        """Calculate projected distance from center along the beam axis.

        Returns positive value when the point is located in
        the k direction to the center.

        Keyword arguments:
        point -- location in the space coordinate

        """
        return cast(float, dot(self.k, point - self.center))

    def _dist_from_beam_axis(self, x: float, y: float, z: float) -> float:
        """Calculate distance from the beam axis.

        Keyword arguments:
        point -- location in the space coordinate

        """
        return cast(float, norm(cross(self.k, (x, y, z) - self.center)))

    def _axis_in_k(self) -> DirectionType:
        """Return the biggest component direction of k."""
        dot_with_axis: dict[float, type[const.Directional]] = {}

        dot_with_axis[dot(const.PlusX.vector, self.k)] = const.PlusX
        dot_with_axis[dot(const.PlusY.vector, self.k)] = const.PlusY
        dot_with_axis[dot(const.PlusZ.vector, self.k)] = const.PlusZ
        dot_with_axis[dot(const.MinusX.vector, self.k)] = const.MinusX
        dot_with_axis[dot(const.MinusY.vector, self.k)] = const.MinusY
        dot_with_axis[dot(const.MinusZ.vector, self.k)] = const.MinusZ

        return cast(DirectionType, dot_with_axis[max(dot_with_axis)])

    def _get_wave_number(
        self, k: Vector3, eps_inf: float, mu_inf: float, space: Cartesian
    ) -> float:
        """Calculate the wave number for auxiliary fdtd using Newton's method.

        Keyword arguments:
        k -- normalized wave vector
        eps_inf -- permittivity which fills the auxiliary fdtd
        mu_inf -- permeability which fills the auxiliary fdtd
        space -- Cartesian instance

        """
        ds = np.array(space.dr)
        dt = space.dt
        v = 1 / sqrt(eps_inf * mu_inf)
        omega = 2 * pi * self.src_time.freq
        wave_number = omega / v
        wave_vector = wave_number * np.array(k)

        zeta = bisect(
            self._3d_dispersion_relation, 0, 2, (v, omega, ds, dt, wave_vector)
        )

        return cast(float, zeta * wave_number)

    def _3d_dispersion_relation(
        self,
        zeta: float,
        v: float,
        omega: float,
        ds: Sequence[float],
        dt: float,
        k: Vector3,
    ) -> float:
        """Evaluate the three-dimensional numerical dispersion residual.

        Keyword arguments:
        zeta: a scalar factor which is yet to be determined.
        v: the phase speed of the wave in the default medium.
        omega: the angular frequency of the input wave.
        ds: the space-cell size, (dx, dy, dz)
        dt: the time step
        k: the true wave vector, (kx, ky, kz)

        Equation 5.65 at p.214 of 'A. Taflove and S. C. Hagness, Computational
        Electrodynamics: The Finite-Difference Time-Domain Method, Third
        Edition, 3rd ed. Artech House Publishers, 2005'.

        """
        lhs = (sin(0.5 * dt * omega) / v / dt) ** 2
        rhs = sum((np.sin(0.5 * zeta * np.array(ds) * k) / ds) ** 2)

        return lhs - rhs

    def _1d_dispersion_relation(
        self, ds: float, zeta: float, v: float, omega: float, dt: float, k: float
    ) -> float:
        """Evaluate the one-dimensional numerical dispersion residual.

        Keyword arguments:
        ds: an 1D cell-size which is yet to be determined
        zeta: the scalar factor which relates the true and numerical
              wavenumber
        v: the phase speed of the input wave in the default medium
        omega: the angular frequency of the input wave
        dt: the time step
        k: the true wavenumber

        Equation 5.67 at p.215 of A. Taflove and S. C. Hagness, Computational
        Electrodynamics: The Finite-Difference Time-Domain Method, Third
        Edition, 3rd ed. Artech House Publishers, 2005.

        """
        lhs = sin(0.5 * omega * dt) / v / dt
        if ds == 0:
            rhs = 0.5 * k * zeta
        else:
            rhs = sin(0.5 * k * zeta * ds) / ds
        return lhs - rhs

    def _get_auxiliary_spec(
        self, space: Cartesian, geom_tree: GeomBoxTree
    ) -> AuxiliarySourceSpec:
        """Describe the legacy matched-dispersion auxiliary without creating it."""
        default_medium = geom_tree.object_of_point((inf, inf, inf))[0]
        eps_inf = default_medium.material.eps_inf
        mu_inf = default_medium.material.mu_inf
        v = 1 / sqrt(eps_inf * mu_inf)
        ds = tuple(space.dr)
        dt = space.dt
        omega = 2 * pi * self.src_time.freq
        wave_vector = omega * self.k / v
        zeta = bisect(
            self._3d_dispersion_relation, 0, 2, (v, omega, ds, dt, wave_vector)
        )
        wave_number = omega / v
        delta_1d = bisect(
            self._1d_dispersion_relation,
            0,
            2 * max(ds),
            (zeta, v, omega, dt, wave_number),
        )
        pml_thickness = 50 * delta_1d
        vertices = [
            self.center + (x, y, z)
            for x in (0.5 * self.size[0], -0.5 * self.size[0])
            for y in (0.5 * self.size[1], -0.5 * self.size[1])
            for z in (0.5 * self.size[2], -0.5 * self.size[2])
        ]
        max_dist = max(
            map(abs, map(self._metric_from_center_along_beam_axis, vertices))
        )
        aux_space = Cartesian(
            size=(0, 0, 2 * (max_dist + pml_thickness + 2 * delta_1d)),
            resolution=1 / delta_1d,
            parallel=False,
        )
        # The nested Torch runtime uses the outer Courant step explicitly.
        # Store it on this inert description so Gaussian prewarm is determined
        # before that runtime is constructed.
        aux_space.dt = space.dt
        material = geom_tree.material_of_point((inf, inf, inf))[0]
        return AuxiliarySourceSpec(
            aux_space,
            (DefaultMedium(material=material), Shell(Cpml(), thickness=pml_thickness)),
            (
                PointSource(
                    src_time=cast(SrcTime, deepcopy(self.src_time)),
                    component=const.Ex,
                    center=(0, 0, -max_dist - delta_1d),
                ),
            ),
        )

    def _transparent(
        self,
        space: Cartesian,
        component: ComponentType,
        cosine: RealScalar,
        shape: FieldArray | Index3,
        low_idx: Index3,
        high_idx: Index3,
        samp_i2s: Callable[[int, int, int], Coordinate3],
        face: DirectionType,
    ) -> list[TfsfFaceRule]:
        """Lower one interface face directly to interpolation rules."""
        lowering = _torch_tfsf_lowering.get()
        if lowering is None:
            raise RuntimeError("direct TFSF lowering context is unavailable")
        rules: list[TfsfFaceRule] = []
        electric = issubclass(component, const.Electric)
        axis = {
            const.MinusX: 0,
            const.PlusX: 0,
            const.MinusY: 1,
            const.PlusY: 1,
            const.MinusZ: 2,
            const.PlusZ: 2,
        }[face]
        signs = {
            "Ex": {"-y": -1, "+y": 1, "-z": 1, "+z": -1},
            "Ey": {"-z": -1, "+z": 1, "-x": 1, "+x": -1},
            "Ez": {"-x": -1, "+x": 1, "-y": -1, "+y": 1},
            "Hx": {"-y": 1, "+y": -1, "-z": -1, "+z": 1},
            "Hy": {"-z": 1, "+z": -1, "-x": -1, "+x": 1},
            "Hz": {"-x": 1, "+x": -1, "-y": -1, "+y": 1},
        }
        for idx, point, material, underneath in self._mapped_source_points(
            space, component, shape, low_idx, high_idx
        ):
            medium = underneath if underneath is not None else material
            sample = (
                lowering.auxiliary_spec.space.spc_to_exact_hy_idx
                if electric
                else lowering.auxiliary_spec.space.spc_to_exact_ex_idx
            )(*((0, 0, self._metric_from_center_along_beam_axis(samp_i2s(*idx)))))
            low = np.floor(sample).astype(np.intp)
            weight1 = float(sample[2] - low[2])
            coefficient = (
                signs[component.__name__][face.str()]
                * space.dt
                * float(cosine)
                * self.amp
                * self.mode_function(*point)
                / ((medium.eps_inf if electric else medium.mu_inf) * space.dr[axis])
            )
            rules.append(
                TfsfFaceRule(
                    component.__name__,
                    idx,
                    "Hy" if electric else "Ex",
                    tuple(low),
                    tuple(low + (0, 0, 1)),
                    1.0 - weight1,
                    weight1,
                    coefficient,
                    lowering.auxiliary_spec,
                )
            )
        return rules

    def _mapped_source_points(
        self,
        space: Cartesian,
        component: ComponentType,
        shape: FieldArray | Index3,
        low_idx: Index3,
        high_idx: Index3,
    ) -> Iterator[SourcePoint]:
        """Yield source indices, coordinates, and materials in C-order."""
        shape = cast(Index3, tuple(getattr(shape, "shape", shape)))
        electric_high_trim: dict[ComponentType, Index3] = {
            const.Ex: (0, 1, 1),
            const.Ey: (1, 0, 1),
            const.Ez: (1, 1, 0),
        }
        magnetic_low: dict[ComponentType, Index3] = {
            const.Hx: (0, 1, 1),
            const.Hy: (1, 0, 1),
            const.Hz: (1, 1, 0),
        }
        if component in electric_high_trim:
            allowed_low = np.zeros(3, dtype=np.intp)
            allowed_high = np.array(shape) - electric_high_trim[component]
        else:
            allowed_low = np.array(magnetic_low[component], dtype=np.intp)
            allowed_high = np.array(shape, dtype=np.intp)

        low = np.maximum(np.array(low_idx, dtype=np.intp), allowed_low)
        high = np.minimum(np.array(high_idx, dtype=np.intp), allowed_high)
        tile_shape = cast(Index3, tuple(high - low))
        if any(length <= 0 for length in tile_shape):
            return

        field_axes = space.component_coordinate_axes(component, shape)
        axes = cast(
            tuple[RealArray, RealArray, RealArray],
            tuple(
                np.ascontiguousarray(axis[start:stop])
                for axis, start, stop in zip(field_axes, low, high, strict=True)
            ),
        )
        total = int(np.prod(tile_shape))
        plane = tile_shape[1] * tile_shape[2]
        for start in range(0, total, self._MAPPING_TILE_SIZE):
            stop = min(start + self._MAPPING_TILE_SIZE, total)
            lowering = _torch_tfsf_lowering.get()
            if lowering is None:
                raise RuntimeError("direct TFSF lowering context is unavailable")
            geometry_map = lowering.geometry_tree.lower_grid(
                *axes, start, stop, component=component
            )
            geometries = geometry_map.geometries
            for offset, (material_id, underlying_id) in enumerate(
                zip(
                    geometry_map.material_ids,
                    geometry_map.underlying_ids,
                    strict=True,
                )
            ):
                material = geometries[material_id].material
                underneath = (
                    None if underlying_id < 0 else geometries[underlying_id].material
                )
                linear = start + offset
                i, remainder = divmod(linear, plane)
                j, k = divmod(remainder, tile_shape[2])
                local_idx = i, j, k
                idx = tuple(
                    value + origin for value, origin in zip(local_idx, low, strict=True)
                )
                point = tuple(
                    axis[value] for axis, value in zip(axes, local_idx, strict=True)
                )
                yield idx, point, material, underneath

    def _component_shapes(self, space: Cartesian) -> dict[str, Index3]:
        nx, ny, nz = (int(value) for value in space.my_field_size)
        return {
            "Ex": (nx, ny + 1, nz + 1),
            "Ey": (nx + 1, ny, nz + 1),
            "Ez": (nx + 1, ny + 1, nz),
            "Hx": (nx, ny + 1, nz + 1),
            "Hy": (nx + 1, ny, nz + 1),
            "Hz": (nx + 1, ny + 1, nz),
        }

    def _auxiliary_spec_for_context(self, context: object) -> AuxiliarySourceSpec:
        """Build the nested description without a mutable Torch lifecycle."""
        value = cast(Any, context)
        return self._get_auxiliary_spec(value.space, value.geometry_tree)

    def _lower_rules(
        self,
        context: object,
        helpers: Sequence[str],
        *,
        include_zero_cosine: bool = False,
    ) -> tuple[TfsfFaceRule, ...]:
        space = cast(Any, context).space
        shapes = self._component_shapes(space)
        directions = {
            "_transparent_ex_minus_y": ("Ex", self.h_direction, (0, 0, 1)),
            "_transparent_ex_plus_y": ("Ex", self.h_direction, (0, 0, 1)),
            "_transparent_ex_minus_z": ("Ex", self.h_direction, (0, 1, 0)),
            "_transparent_ex_plus_z": ("Ex", self.h_direction, (0, 1, 0)),
            "_transparent_ey_minus_z": ("Ey", self.h_direction, (1, 0, 0)),
            "_transparent_ey_plus_z": ("Ey", self.h_direction, (1, 0, 0)),
            "_transparent_ey_minus_x": ("Ey", self.h_direction, (0, 0, 1)),
            "_transparent_ey_plus_x": ("Ey", self.h_direction, (0, 0, 1)),
            "_transparent_ez_minus_x": ("Ez", self.h_direction, (0, 1, 0)),
            "_transparent_ez_plus_x": ("Ez", self.h_direction, (0, 1, 0)),
            "_transparent_ez_minus_y": ("Ez", self.h_direction, (1, 0, 0)),
            "_transparent_ez_plus_y": ("Ez", self.h_direction, (1, 0, 0)),
            "_transparent_hx_minus_y": ("Hx", self.e_direction, (0, 0, 1)),
            "_transparent_hx_plus_y": ("Hx", self.e_direction, (0, 0, 1)),
            "_transparent_hx_minus_z": ("Hx", self.e_direction, (0, 1, 0)),
            "_transparent_hx_plus_z": ("Hx", self.e_direction, (0, 1, 0)),
            "_transparent_hy_minus_z": ("Hy", self.e_direction, (1, 0, 0)),
            "_transparent_hy_plus_z": ("Hy", self.e_direction, (1, 0, 0)),
            "_transparent_hy_minus_x": ("Hy", self.e_direction, (0, 0, 1)),
            "_transparent_hy_plus_x": ("Hy", self.e_direction, (0, 0, 1)),
            "_transparent_hz_minus_x": ("Hz", self.e_direction, (0, 1, 0)),
            "_transparent_hz_plus_x": ("Hz", self.e_direction, (0, 1, 0)),
            "_transparent_hz_minus_y": ("Hz", self.e_direction, (1, 0, 0)),
            "_transparent_hz_plus_y": ("Hz", self.e_direction, (1, 0, 0)),
        }
        lowering = _TorchTfsfLowering(
            cast(Any, context).geometry_tree,
            self._auxiliary_spec_for_context(context),
        )
        token = _torch_tfsf_lowering.set(lowering)
        try:
            records: list[TfsfFaceRule] = []
            for name in helpers:
                component, direction, axis = directions[name]
                cosine = float(dot(direction, axis))
                if cosine or include_zero_cosine:
                    records.extend(
                        getattr(self, name)(shapes[component], space, cosine)
                    )
            return tuple(records)
        finally:
            _torch_tfsf_lowering.reset(token)

    def lower_torch_source(self, context: object) -> tuple[TfsfFaceRule, ...]:
        """Emit direct TFSF face rules for the retained Torch source plan."""

        return self._lower_rules(
            context,
            (
                "_transparent_ex_minus_y",
                "_transparent_ex_plus_y",
                "_transparent_ex_minus_z",
                "_transparent_ex_plus_z",
                "_transparent_ey_minus_z",
                "_transparent_ey_plus_z",
                "_transparent_ey_minus_x",
                "_transparent_ey_plus_x",
                "_transparent_ez_minus_x",
                "_transparent_ez_plus_x",
                "_transparent_ez_minus_y",
                "_transparent_ez_plus_y",
                "_transparent_hx_minus_y",
                "_transparent_hx_plus_y",
                "_transparent_hx_minus_z",
                "_transparent_hx_plus_z",
                "_transparent_hy_minus_z",
                "_transparent_hy_plus_z",
                "_transparent_hy_minus_x",
                "_transparent_hy_plus_x",
                "_transparent_hz_minus_x",
                "_transparent_hz_plus_x",
                "_transparent_hz_minus_y",
                "_transparent_hz_plus_y",
            ),
        )

    def _transparent_ex_minus_y(
        self, shape: Index3, space: Cartesian, cosine: RealScalar
    ) -> list[TfsfFaceRule]:
        if 2 * space.half_size[1] > space.dr[1]:
            low = self.center - self.half_size
            high = self.center + self.half_size * (1, -1, 1)

            low_idx = space.space_to_ex_index(*low)
            high_idx = cast(
                Index3, tuple(x + 1 for x in space.space_to_ex_index(*high))
            )

            hz_i2s = lambda i, j, k: space.hz_index_to_space(i + 1, j, k)

            pw_src = self._transparent(
                space,
                const.Ex,
                cosine,
                shape,
                low_idx,
                high_idx,
                hz_i2s,
                const.MinusY,
            )
        else:
            pw_src = []

        return pw_src

    def _transparent_ex_plus_y(
        self, shape: FieldArray, space: Cartesian, cosine: RealScalar
    ) -> list[TfsfFaceRule]:
        if 2 * space.half_size[1] > space.dr[1]:
            low = self.center - self.half_size * (1, -1, 1)
            high = self.center + self.half_size

            low_idx = space.space_to_ex_index(*low)
            high_idx = cast(
                Index3, tuple(x + 1 for x in space.space_to_ex_index(*high))
            )

            hz_i2s = lambda i, j, k: space.hz_index_to_space(i + 1, j + 1, k)

            pw_src = self._transparent(
                space,
                const.Ex,
                cosine,
                shape,
                low_idx,
                high_idx,
                hz_i2s,
                const.PlusY,
            )

        else:
            pw_src = []

        return pw_src

    def _transparent_ex_minus_z(
        self, shape: FieldArray, space: Cartesian, cosine: RealScalar
    ) -> list[TfsfFaceRule]:
        if 2 * space.half_size[2] > space.dr[2]:
            low = self.center - self.half_size
            high = self.center + self.half_size * (1, 1, -1)

            low_idx = space.space_to_ex_index(*low)
            high_idx = cast(
                Index3, tuple(x + 1 for x in space.space_to_ex_index(*high))
            )

            hy_i2s = lambda i, j, k: space.hy_index_to_space(i + 1, j, k)

            pw_src = self._transparent(
                space,
                const.Ex,
                cosine,
                shape,
                low_idx,
                high_idx,
                hy_i2s,
                const.MinusZ,
            )
        else:
            pw_src = []

        return pw_src

    def _transparent_ex_plus_z(
        self, shape: FieldArray, space: Cartesian, cosine: RealScalar
    ) -> list[TfsfFaceRule]:
        if 2 * space.half_size[2] > space.dr[2]:
            low = self.center - self.half_size * (1, 1, -1)
            high = self.center + self.half_size

            low_idx = space.space_to_ex_index(*low)
            high_idx = cast(
                Index3, tuple(x + 1 for x in space.space_to_ex_index(*high))
            )

            i2s = lambda i, j, k: space.hy_index_to_space(i + 1, j, k + 1)

            pw_src = self._transparent(
                space,
                const.Ex,
                cosine,
                shape,
                low_idx,
                high_idx,
                i2s,
                const.PlusZ,
            )
        else:
            pw_src = []

        return pw_src

    def _transparent_ey_minus_z(
        self, shape: FieldArray, space: Cartesian, cosine: RealScalar
    ) -> list[TfsfFaceRule]:
        if 2 * space.half_size[2] > space.dr[2]:
            low = self.center - self.half_size
            high = self.center + self.half_size * (1, 1, -1)

            low_idx = space.space_to_ey_index(*low)
            high_idx = cast(
                Index3, tuple(x + 1 for x in space.space_to_ey_index(*high))
            )

            hx_i2s = lambda i, j, k: space.hx_index_to_space(i, j + 1, k)

            pw_src = self._transparent(
                space,
                const.Ey,
                cosine,
                shape,
                low_idx,
                high_idx,
                hx_i2s,
                const.MinusZ,
            )
        else:
            pw_src = []

        return pw_src

    def _transparent_ey_plus_z(
        self, shape: FieldArray, space: Cartesian, cosine: RealScalar
    ) -> list[TfsfFaceRule]:
        if 2 * space.half_size[2] > space.dr[2]:
            low = self.center - self.half_size * (1, 1, -1)
            high = self.center + self.half_size

            low_idx = space.space_to_ey_index(*low)
            high_idx = cast(
                Index3, tuple(x + 1 for x in space.space_to_ey_index(*high))
            )

            hx_i2s = lambda i, j, k: space.hx_index_to_space(i, j + 1, k + 1)

            pw_src = self._transparent(
                space,
                const.Ey,
                cosine,
                shape,
                low_idx,
                high_idx,
                hx_i2s,
                const.PlusZ,
            )
        else:
            pw_src = []

        return pw_src

    def _transparent_ey_minus_x(
        self, shape: FieldArray, space: Cartesian, cosine: RealScalar
    ) -> list[TfsfFaceRule]:
        if 2 * space.half_size[0] > space.dr[0]:
            low = self.center - self.half_size
            high = self.center + self.half_size * (-1, 1, 1)

            low_idx = space.space_to_ey_index(*low)
            high_idx = cast(
                Index3, tuple(x + 1 for x in space.space_to_ey_index(*high))
            )

            hz_i2s = lambda i, j, k: space.hz_index_to_space(i, j + 1, k)

            pw_src = self._transparent(
                space,
                const.Ey,
                cosine,
                shape,
                low_idx,
                high_idx,
                hz_i2s,
                const.MinusX,
            )
        else:
            pw_src = []

        return pw_src

    def _transparent_ey_plus_x(
        self, shape: FieldArray, space: Cartesian, cosine: RealScalar
    ) -> list[TfsfFaceRule]:
        if 2 * space.half_size[0] > space.dr[0]:
            low = self.center - self.half_size * (-1, 1, 1)
            high = self.center + self.half_size

            low_idx = space.space_to_ey_index(*low)
            high_idx = cast(
                Index3, tuple(x + 1 for x in space.space_to_ey_index(*high))
            )

            hz_i2s = lambda i, j, k: space.hz_index_to_space(i + 1, j + 1, k)

            pw_src = self._transparent(
                space,
                const.Ey,
                cosine,
                shape,
                low_idx,
                high_idx,
                hz_i2s,
                const.PlusX,
            )
        else:
            pw_src = []

        return pw_src

    def _transparent_ez_minus_x(
        self, shape: FieldArray, space: Cartesian, cosine: RealScalar
    ) -> list[TfsfFaceRule]:
        if 2 * space.half_size[0] > space.dr[0]:
            low = self.center - self.half_size
            high = self.center + self.half_size * (-1, 1, 1)

            low_idx = space.space_to_ez_index(*low)
            high_idx = cast(
                Index3, tuple(x + 1 for x in space.space_to_ez_index(*high))
            )

            hy_i2s = lambda i, j, k: space.hy_index_to_space(i, j, k + 1)

            pw_src = self._transparent(
                space,
                const.Ez,
                cosine,
                shape,
                low_idx,
                high_idx,
                hy_i2s,
                const.MinusX,
            )
        else:
            pw_src = []

        return pw_src

    def _transparent_ez_plus_x(
        self, shape: FieldArray, space: Cartesian, cosine: RealScalar
    ) -> list[TfsfFaceRule]:
        if 2 * space.half_size[0] > space.dr[0]:
            low = self.center - self.half_size * (-1, 1, 1)
            high = self.center + self.half_size

            low_idx = space.space_to_ez_index(*low)
            high_idx = cast(
                Index3, tuple(x + 1 for x in space.space_to_ez_index(*high))
            )

            hy_i2s = lambda i, j, k: space.hy_index_to_space(i + 1, j, k + 1)

            pw_src = self._transparent(
                space,
                const.Ez,
                cosine,
                shape,
                low_idx,
                high_idx,
                hy_i2s,
                const.PlusX,
            )
        else:
            pw_src = []

        return pw_src

    def _transparent_ez_minus_y(
        self, shape: FieldArray, space: Cartesian, cosine: RealScalar
    ) -> list[TfsfFaceRule]:
        if 2 * space.half_size[1] > space.dr[1]:
            low = self.center - self.half_size
            high = self.center + self.half_size * (1, -1, 1)

            low_idx = space.space_to_ez_index(*low)
            high_idx = cast(
                Index3, tuple(x + 1 for x in space.space_to_ez_index(*high))
            )

            hx_i2s = lambda i, j, k: space.hx_index_to_space(i, j, k + 1)

            pw_src = self._transparent(
                space,
                const.Ez,
                cosine,
                shape,
                low_idx,
                high_idx,
                hx_i2s,
                const.MinusY,
            )
        else:
            pw_src = []

        return pw_src

    def _transparent_ez_plus_y(
        self, shape: FieldArray, space: Cartesian, cosine: RealScalar
    ) -> list[TfsfFaceRule]:
        if 2 * space.half_size[1] > space.dr[1]:
            low = self.center - self.half_size * (1, -1, 1)
            high = self.center + self.half_size

            low_idx = space.space_to_ez_index(*low)
            high_idx = cast(
                Index3, tuple(x + 1 for x in space.space_to_ez_index(*high))
            )

            hx_i2s = lambda i, j, k: space.hx_index_to_space(i, j + 1, k + 1)

            pw_src = self._transparent(
                space,
                const.Ez,
                cosine,
                shape,
                low_idx,
                high_idx,
                hx_i2s,
                const.PlusY,
            )
        else:
            pw_src = []

        return pw_src

    def _transparent_hx_minus_y(
        self, shape: FieldArray, space: Cartesian, cosine: RealScalar
    ) -> list[TfsfFaceRule]:
        if 2 * space.half_size[1] > space.dr[1]:
            low = self.center - self.half_size
            high = self.center + self.half_size * (1, -1, 1)

            low_idx = space.space_to_ez_index(*low)
            high_idx = cast(
                Index3, tuple(x + 1 for x in space.space_to_ez_index(*high))
            )

            low_idx = (low_idx[0], low_idx[1], low_idx[2] + 1)
            high_idx = (high_idx[0], high_idx[1], high_idx[2] + 1)

            ez_i2s = lambda i, j, k: space.ez_index_to_space(i, j, k - 1)

            pw_src = self._transparent(
                space,
                const.Hx,
                cosine,
                shape,
                low_idx,
                high_idx,
                ez_i2s,
                const.MinusY,
            )
        else:
            pw_src = []

        return pw_src

    def _transparent_hx_plus_y(
        self, shape: FieldArray, space: Cartesian, cosine: RealScalar
    ) -> list[TfsfFaceRule]:
        if 2 * space.half_size[1] > space.dr[1]:
            low = self.center - self.half_size * (1, -1, 1)
            high = self.center + self.half_size

            low_idx = space.space_to_ez_index(*low)
            high_idx = cast(
                Index3, tuple(x + 1 for x in space.space_to_ez_index(*high))
            )

            low_idx = (low_idx[0], low_idx[1] + 1, low_idx[2] + 1)
            high_idx = (high_idx[0], high_idx[1] + 1, high_idx[2] + 1)

            ez_i2s = lambda i, j, k: space.ez_index_to_space(i, j - 1, k - 1)

            pw_src = self._transparent(
                space,
                const.Hx,
                cosine,
                shape,
                low_idx,
                high_idx,
                ez_i2s,
                const.PlusY,
            )

        else:
            pw_src = []

        return pw_src

    def _transparent_hx_minus_z(
        self, shape: FieldArray, space: Cartesian, cosine: RealScalar
    ) -> list[TfsfFaceRule]:
        if 2 * space.half_size[2] > space.dr[2]:
            low = self.center - self.half_size
            high = self.center + self.half_size * (1, 1, -1)

            low_idx = space.space_to_ey_index(*low)
            high_idx = cast(
                Index3, tuple(x + 1 for x in space.space_to_ey_index(*high))
            )

            low_idx = (low_idx[0], low_idx[1] + 1, low_idx[2])
            high_idx = (high_idx[0], high_idx[1] + 1, high_idx[2])

            ey_i2s = lambda i, j, k: space.ey_index_to_space(i, j - 1, k - 1)

            pw_src = self._transparent(
                space,
                const.Hx,
                cosine,
                shape,
                low_idx,
                high_idx,
                ey_i2s,
                const.MinusZ,
            )
        else:
            pw_src = []

        return pw_src

    def _transparent_hx_plus_z(
        self, shape: FieldArray, space: Cartesian, cosine: RealScalar
    ) -> list[TfsfFaceRule]:
        if 2 * space.half_size[2] > space.dr[2]:
            low = self.center - self.half_size * (1, 1, -1)
            high = self.center + self.half_size

            ey_low_idx = space.space_to_ey_index(*low)
            ey_high_idx = cast(
                Index3, tuple(x + 1 for x in space.space_to_ey_index(*high))
            )

            low_idx = (ey_low_idx[0], ey_low_idx[1] + 1, ey_low_idx[2] + 1)
            high_idx = (ey_high_idx[0], ey_high_idx[1] + 1, ey_high_idx[2] + 1)

            ey_i2s = lambda i, j, k: space.ey_index_to_space(i, j - 1, k)

            pw_src = self._transparent(
                space,
                const.Hx,
                cosine,
                shape,
                low_idx,
                high_idx,
                ey_i2s,
                const.PlusZ,
            )
        else:
            pw_src = []

        return pw_src

    def _transparent_hy_minus_z(
        self, shape: FieldArray, space: Cartesian, cosine: RealScalar
    ) -> list[TfsfFaceRule]:
        if 2 * space.half_size[2] > space.dr[2]:
            low = self.center - self.half_size
            high = self.center + self.half_size * (1, 1, -1)

            low_idx = space.space_to_ex_index(*low)
            high_idx = cast(
                Index3, tuple(x + 1 for x in space.space_to_ex_index(*high))
            )

            low_idx = (low_idx[0] + 1, low_idx[1], low_idx[2])
            high_idx = (high_idx[0] + 1, high_idx[1], high_idx[2])

            ex_i2s = lambda i, j, k: space.ex_index_to_space(i - 1, j, k)

            pw_src = self._transparent(
                space,
                const.Hy,
                cosine,
                shape,
                low_idx,
                high_idx,
                ex_i2s,
                const.MinusZ,
            )
        else:
            pw_src = []

        return pw_src

    def _transparent_hy_plus_z(
        self, shape: FieldArray, space: Cartesian, cosine: RealScalar
    ) -> list[TfsfFaceRule]:
        if 2 * space.half_size[2] > space.dr[2]:
            low = self.center - self.half_size * (1, 1, -1)
            high = self.center + self.half_size

            low_idx = space.space_to_ex_index(*low)
            high_idx = cast(
                Index3, tuple(x + 1 for x in space.space_to_ex_index(*high))
            )

            low_idx = (low_idx[0] + 1, low_idx[1], low_idx[2] + 1)
            high_idx = (high_idx[0] + 1, high_idx[1], high_idx[2] + 1)

            ex_i2s = lambda i, j, k: space.ex_index_to_space(i - 1, j, k - 1)

            pw_src = self._transparent(
                space,
                const.Hy,
                cosine,
                shape,
                low_idx,
                high_idx,
                ex_i2s,
                const.PlusZ,
            )
        else:
            pw_src = []

        return pw_src

    def _transparent_hy_minus_x(
        self, shape: FieldArray, space: Cartesian, cosine: RealScalar
    ) -> list[TfsfFaceRule]:
        if 2 * space.half_size[0] > space.dr[0]:
            low = self.center - self.half_size
            high = self.center + self.half_size * (-1, 1, 1)

            ez_low_idx = space.space_to_ez_index(*low)
            ez_high_idx = cast(
                Index3, tuple(x + 1 for x in space.space_to_ez_index(*high))
            )

            low_idx = (ez_low_idx[0], ez_low_idx[1], ez_low_idx[2] + 1)
            high_idx = (ez_high_idx[0], ez_high_idx[1], ez_high_idx[2] + 1)

            ez_i2s = lambda i, j, k: space.ez_index_to_space(i, j, k - 1)

            pw_src = self._transparent(
                space,
                const.Hy,
                cosine,
                shape,
                low_idx,
                high_idx,
                ez_i2s,
                const.MinusX,
            )
        else:
            pw_src = []

        return pw_src

    def _transparent_hy_plus_x(
        self, shape: FieldArray, space: Cartesian, cosine: RealScalar
    ) -> list[TfsfFaceRule]:
        if 2 * space.half_size[0] > space.dr[0]:
            low = self.center - self.half_size * (-1, 1, 1)
            high = self.center + self.half_size

            ez_low_idx = space.space_to_ez_index(*low)
            ez_high_idx = cast(
                Index3, tuple(x + 1 for x in space.space_to_ez_index(*high))
            )

            low_idx = (ez_low_idx[0] + 1, ez_low_idx[1], ez_low_idx[2] + 1)
            high_idx = (ez_high_idx[0] + 1, ez_high_idx[1], ez_high_idx[2] + 1)

            ez_i2s = lambda i, j, k: space.ez_index_to_space(i - 1, j, k - 1)

            pw_src = self._transparent(
                space,
                const.Hy,
                cosine,
                shape,
                low_idx,
                high_idx,
                ez_i2s,
                const.PlusX,
            )
        else:
            pw_src = []

        return pw_src

    def _transparent_hz_minus_x(
        self, shape: FieldArray, space: Cartesian, cosine: RealScalar
    ) -> list[TfsfFaceRule]:
        if 2 * space.half_size[0] > space.dr[0]:
            low = self.center - self.half_size
            high = self.center + self.half_size * (-1, 1, 1)

            low_idx = space.space_to_ey_index(*low)
            high_idx = cast(
                Index3, tuple(x + 1 for x in space.space_to_ey_index(*high))
            )

            low_idx = (low_idx[0], low_idx[1] + 1, low_idx[2])
            high_idx = (high_idx[0], high_idx[1] + 1, high_idx[2])

            ey_i2s = lambda i, j, k: space.ey_index_to_space(i, j - 1, k)

            pw_src = self._transparent(
                space,
                const.Hz,
                cosine,
                shape,
                low_idx,
                high_idx,
                ey_i2s,
                const.MinusX,
            )
        else:
            pw_src = []

        return pw_src

    def _transparent_hz_plus_x(
        self, shape: FieldArray, space: Cartesian, cosine: RealScalar
    ) -> list[TfsfFaceRule]:
        if 2 * space.half_size[0] > space.dr[0]:
            low = self.center - self.half_size * (-1, 1, 1)
            high = self.center + self.half_size

            low_idx = space.space_to_ey_index(*low)
            high_idx = cast(
                Index3, tuple(x + 1 for x in space.space_to_ey_index(*high))
            )

            low_idx = (low_idx[0] + 1, low_idx[1] + 1, low_idx[2])
            high_idx = (high_idx[0] + 1, high_idx[1] + 1, high_idx[2])

            ey_i2s = lambda i, j, k: space.ey_index_to_space(i - 1, j - 1, k)

            pw_src = self._transparent(
                space,
                const.Hz,
                cosine,
                shape,
                low_idx,
                high_idx,
                ey_i2s,
                const.PlusX,
            )
        else:
            pw_src = []

        return pw_src

    def _transparent_hz_minus_y(
        self, shape: FieldArray, space: Cartesian, cosine: RealScalar
    ) -> list[TfsfFaceRule]:
        if 2 * space.half_size[1] > space.dr[1]:
            low = self.center - self.half_size
            high = self.center + self.half_size * (1, -1, 1)

            low_idx = space.space_to_ex_index(*low)
            high_idx = cast(
                Index3, tuple(x + 1 for x in space.space_to_ex_index(*high))
            )

            low_idx = (low_idx[0] + 1, low_idx[1], low_idx[2])
            high_idx = (high_idx[0] + 1, high_idx[1], high_idx[2])

            ex_i2s = lambda i, j, k: space.ex_index_to_space(i, j - 1, k)

            pw_src = self._transparent(
                space,
                const.Hz,
                cosine,
                shape,
                low_idx,
                high_idx,
                ex_i2s,
                const.MinusY,
            )
        else:
            pw_src = []

        return pw_src

    def _transparent_hz_plus_y(
        self, shape: FieldArray, space: Cartesian, cosine: RealScalar
    ) -> list[TfsfFaceRule]:
        if 2 * space.half_size[1] > space.dr[1]:
            low = self.center - self.half_size * (1, -1, 1)
            high = self.center + self.half_size

            low_idx = space.space_to_ex_index(*low)
            high_idx = cast(
                Index3, tuple(x + 1 for x in space.space_to_ex_index(*high))
            )

            low_idx = (low_idx[0] + 1, low_idx[1] + 1, low_idx[2])
            high_idx = (high_idx[0] + 1, high_idx[1] + 1, high_idx[2])

            ex_i2s = lambda i, j, k: space.ex_index_to_space(i - 1, j - 1, k)

            pw_src = self._transparent(
                space,
                const.Hz,
                cosine,
                shape,
                low_idx,
                high_idx,
                ex_i2s,
                const.PlusY,
            )
        else:
            pw_src = []

        return pw_src


class GaussianBeam(TotalFieldScatteredField):
    """Launch a transparent Gaussian beam.

    It works as a guided mode with Gaussian profile is launched through the
    incidence interface. The incidence interface is transparent, thus the
    scattered wave can penetrate through the interface plane.

    """

    def __init__(
        self,
        src_time: SrcTime,
        directivity: DirectionType,
        center: Vector3,
        size: Vector3,
        direction: Vector3,
        polarization: Vector3,
        waist: float = inf,
        amp: float = 1,
    ) -> None:
        """Initialize a transparent Gaussian-beam interface.

        Args:
            src_time: Temporal waveform of the incident beam.
            directivity: Directional class selecting the interface face.
            center: Three-dimensional center of the interface and beam axis.
            size: Three-dimensional interface extents.
            direction: Beam propagation vector.
            polarization: Electric-field polarization vector.
            waist: Gaussian beam radius in space units.
            amp: Peak incident-wave amplitude.
        """
        TotalFieldScatteredField.__init__(
            self, src_time, center, size, direction, polarization, amp
        )

        if issubclass(directivity, const.Directional):
            self.directivity = directivity
        else:
            raise TypeError("directivity must be a Directional type.")

        # spot size of Gaussian beam
        self.waist = float(waist)

    def init(self, geom_tree: GeomBoxTree, space: Cartesian, cmplx: bool) -> None:
        self.geom_tree = geom_tree
        self.src_time.init(cmplx)

        spec = self._get_auxiliary_spec(space, geom_tree)
        raising = cast(
            PlaneWaveSrcTime, cast(PointSource, spec.sources[0]).src_time
        ).width
        dist = 2 * spec.space.half_size[2]
        default_medium = next(
            (i for i in spec.geometry if isinstance(i, DefaultMedium))
        )
        eps_inf = default_medium.material.eps_inf
        mu_inf = default_medium.material.mu_inf
        v_p = 1 / sqrt(eps_inf * mu_inf)
        passby = raising + dist / v_p

        self.auxiliary_spec = replace(
            spec,
            gaussian_width=float(raising),
            prewarm_steps=int(np.ceil(2 * passby / spec.space.dt)),
        )

    def display_info(self, indent: int = 0) -> None:
        print(" " * indent, "Gaussian beam source:")
        print(" " * indent, end=" ")
        print("propagation direction:", self.k, end=" ")
        print("center:", self.center)
        print("source plane size:", self.size)
        print("polarization direction:", self.e_direction)
        print("beam waist:", self.waist)
        print("maximum amp.:", self.amp)

        self.src_time.display_info(indent + 4)

    def _auxiliary_spec_for_context(self, context: object) -> AuxiliarySourceSpec:
        spec = super()._auxiliary_spec_for_context(context)
        raising = cast(
            PlaneWaveSrcTime, cast(PointSource, spec.sources[0]).src_time
        ).width
        dist = 2 * spec.space.half_size[2]
        default_medium = next(
            item for item in spec.geometry if isinstance(item, DefaultMedium)
        )
        eps_inf = default_medium.material.eps_inf
        mu_inf = default_medium.material.mu_inf
        passby = raising + dist * sqrt(eps_inf * mu_inf)
        return replace(
            spec,
            gaussian_width=float(raising),
            prewarm_steps=int(np.ceil(2 * passby / spec.space.dt)),
        )

    def lower_torch_source(self, context: object) -> tuple[TfsfFaceRule, ...]:
        """Emit Gaussian-weighted direct TFSF face rules for Torch execution."""

        selected = {
            const.PlusY: (
                "_transparent_ex_minus_y",
                "_transparent_ez_minus_y",
                "_transparent_hx_minus_y",
                "_transparent_hz_minus_y",
            ),
            const.MinusY: (
                "_transparent_ex_plus_y",
                "_transparent_ez_plus_y",
                "_transparent_hx_plus_y",
                "_transparent_hz_plus_y",
            ),
            const.PlusZ: (
                "_transparent_ex_minus_z",
                "_transparent_ey_minus_z",
                "_transparent_hx_minus_z",
                "_transparent_hy_minus_z",
            ),
            const.MinusZ: (
                "_transparent_ex_plus_z",
                "_transparent_ey_plus_z",
                "_transparent_hx_plus_z",
                "_transparent_hy_plus_z",
            ),
            const.PlusX: (
                "_transparent_ey_minus_x",
                "_transparent_ez_minus_x",
                "_transparent_hy_minus_x",
                "_transparent_hz_minus_x",
            ),
            const.MinusX: (
                "_transparent_ey_plus_x",
                "_transparent_ez_plus_x",
                "_transparent_hy_plus_x",
                "_transparent_hz_plus_x",
            ),
        }[self.directivity]
        return self._lower_rules(context, selected, include_zero_cosine=True)

    def mode_function(self, x: float, y: float, z: float) -> float:
        r = self._dist_from_beam_axis(x, y, z)
        return exp(-((r / self.waist) ** 2))
