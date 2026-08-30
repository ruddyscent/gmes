"""Device-resident source lowering and execution for :mod:`gmes.torch_fdtd`."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import torch
from torch import nn

from . import constant as const
from .pw_source import PointSourceElectric, PointSourceMagnetic, TransparentParam
from .source import (
    Bandpass,
    Continuous,
    DifferentiatedGaussian,
    GaussianBeam,
    PointSource,
    TotalFieldScatteredField,
)
from .torch_plan import COMPONENTS

_POINT_COMPONENTS = MappingProxyType(
    {
        const.Ex: ("Ex", False),
        const.Ey: ("Ey", False),
        const.Ez: ("Ez", False),
        const.Hx: ("Hx", False),
        const.Hy: ("Hy", False),
        const.Hz: ("Hz", False),
        const.Jx: ("Ex", True),
        const.Jy: ("Ey", True),
        const.Jz: ("Ez", True),
        const.Mx: ("Hx", True),
        const.My: ("Hy", True),
        const.Mz: ("Hz", True),
    }
)
_FACE_NAMES = {
    const.MinusX: "-x",
    const.PlusX: "+x",
    const.MinusY: "-y",
    const.PlusY: "+y",
    const.MinusZ: "-z",
    const.PlusZ: "+z",
}


@dataclass(frozen=True)
class TorchSourceLoweringContext:
    """Read-only context passed to explicit third-party source lowerers.

    A custom source may implement ``lower_torch_source(context)`` and return
    one or more :class:`TorchPointSourceRecord` instances.  The method is
    invoked once during construction and is never called from ``advance()``.
    """

    space: object
    geometry_tree: object
    paired_real: bool
    dtype: torch.dtype
    device: torch.device
    dt: float


@dataclass(frozen=True)
class TorchPointSourceRecord:
    """A single point source emitted by the explicit lowering protocol."""

    component: str
    target: tuple[int, int, int]
    source_time: object
    amplitude: float = 1.0
    current_scale: float | None = None


def _time_parameters(source_time):
    if isinstance(source_time, Continuous):
        return 0, (
            source_time.freq,
            source_time.phase,
            source_time.start,
            source_time.end,
            source_time.width,
            0.0,
        )
    if isinstance(source_time, Bandpass):
        return 1, (
            source_time.freq,
            source_time.phase,
            source_time.width,
            source_time.peak_time,
            source_time.cutoff,
            0.0,
        )
    if isinstance(source_time, DifferentiatedGaussian):
        return 2, (source_time.tw, source_time.t0, 0.0, 0.0, 0.0, 0.0)
    raise TypeError(
        f"unsupported Torch source-time model {type(source_time).__name__!r}; "
        "use Continuous, Bandpass, DifferentiatedGaussian, or an explicit "
        "lower_torch_source() extension"
    )


def _evaluate_time(model, parameters, time, paired_real, output):
    """Evaluate all built-in source-time models without a per-source callback."""
    frequency = parameters[:, 0]
    phase = parameters[:, 1]
    angle = 2 * torch.pi * frequency * time + phase

    continuous_ts = time - parameters[:, 2]
    continuous_te = parameters[:, 3] - time
    continuous_width = parameters[:, 4]
    rise = torch.where(
        continuous_ts < continuous_width,
        torch.sin(0.5 * torch.pi * continuous_ts / continuous_width).square(),
        torch.ones_like(continuous_ts),
    )
    fall = torch.where(
        continuous_te < continuous_width,
        torch.sin(0.5 * torch.pi * continuous_te / continuous_width).square(),
        torch.ones_like(continuous_te),
    )
    continuous_envelope = torch.where(
        (continuous_ts >= 0) & (continuous_te >= 0),
        rise * fall,
        torch.zeros_like(rise),
    )
    continuous_real = continuous_envelope * torch.cos(angle)
    continuous_imag = continuous_envelope * torch.sin(angle)

    bandpass_offset = time - parameters[:, 3]
    bandpass_envelope = torch.where(
        torch.abs(bandpass_offset) <= parameters[:, 4],
        torch.exp(-0.5 * (bandpass_offset / parameters[:, 2]).square()),
        torch.zeros_like(bandpass_offset),
    )
    bandpass_norm = 1.0 / (2 * torch.pi * frequency)
    bandpass_real = -bandpass_norm * bandpass_envelope * torch.sin(angle)
    bandpass_imag = bandpass_norm * bandpass_envelope * torch.cos(angle)

    gaussian_offset = (time - parameters[:, 1]) / parameters[:, 0]
    gaussian_real = -2 * gaussian_offset * torch.exp(-gaussian_offset.square())
    zero = torch.zeros_like(gaussian_real)

    real = torch.where(
        model == 0,
        continuous_real,
        torch.where(model == 1, bandpass_real, gaussian_real),
    )
    if not paired_real:
        output[:, 0].copy_(real)
        return output
    imaginary = torch.where(
        model == 0,
        continuous_imag,
        torch.where(model == 1, bandpass_imag, zero),
    )
    output[:, 0].copy_(real)
    output[:, 1].copy_(imaginary)
    return output


class TorchPointSourceBatch(nn.Module):
    """Unique-target point/current source batch for one Yee component."""

    def __init__(self, component, records, *, shape, paired_real, device, dtype):
        super().__init__()
        self.component = component
        self.paired_real = paired_real
        overwrite = []
        additive = []
        # PwSource.merge() has last-source-wins semantics at one target.  Make
        # that decision here, before any indexed device write occurs.
        normalized = {}
        for record in records:
            normalized[tuple(record.target)] = record
        for target, record in normalized.items():
            model, parameters = _time_parameters(record.source_time)
            linear = int(np.ravel_multi_index(target, shape))
            item = (linear, model, parameters, float(record.amplitude))
            scale = 1.0 if record.current_scale is None else record.current_scale
            (additive if record.current_scale is not None else overwrite).append(
                item + (scale,)
            )
        self._register(
            "overwrite",
            overwrite,
            paired_real=paired_real,
            device=device,
            dtype=dtype,
        )
        self._register(
            "additive",
            additive,
            paired_real=paired_real,
            device=device,
            dtype=dtype,
        )

    def _register(self, prefix, records, *, paired_real, device, dtype):
        self.register_buffer(
            f"{prefix}_targets",
            torch.tensor(
                [item[0] for item in records], device=device, dtype=torch.int64
            ),
        )
        self.register_buffer(
            f"{prefix}_models",
            torch.tensor(
                [item[1] for item in records], device=device, dtype=torch.int8
            ),
        )
        parameters = [item[2] for item in records]
        self.register_buffer(
            f"{prefix}_parameters",
            torch.tensor(parameters, device=device, dtype=dtype).reshape(-1, 6),
        )
        self.register_buffer(
            f"{prefix}_amplitudes",
            torch.tensor(
                [item[3] * item[4] for item in records], device=device, dtype=dtype
            ),
        )
        self.register_buffer(
            f"_{prefix}_values",
            torch.zeros(
                (len(records), 2 if paired_real else 1),
                device=device,
                dtype=dtype,
            ),
            persistent=False,
        )

    def apply(self, field, time):
        plane = 2 if self.paired_real else 1
        flat = field.reshape(-1, plane)
        for prefix, additive in (("additive", True), ("overwrite", False)):
            targets = getattr(self, f"{prefix}_targets")
            if targets.numel() == 0:
                continue
            values = _evaluate_time(
                getattr(self, f"{prefix}_models"),
                getattr(self, f"{prefix}_parameters"),
                time,
                self.paired_real,
                getattr(self, f"_{prefix}_values"),
            )
            amplitudes = getattr(self, f"{prefix}_amplitudes")
            values.mul_(amplitudes[:, None])
            if additive:
                flat.index_add_(0, targets, values)
            else:
                flat.index_copy_(0, targets, values)


def _transparent_coefficient(component, face, parameter, dt, dr):
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
        "Ez": {"-x": -1, "+x": 1, "-y": 1, "+y": -1},
        "Hx": {"-y": 1, "+y": -1, "-z": -1, "+z": 1},
        "Hy": {"-z": 1, "+z": -1, "-x": -1, "+x": 1},
        "Hz": {"-x": 1, "+x": -1, "-y": -1, "+y": 1},
    }
    material = parameter.eps_inf if component.startswith("E") else parameter.mu_inf
    return (
        signs[component][_FACE_NAMES[face]]
        * dt
        * parameter.amp[face]
        / (material * dr[axis])
    )


class TorchTransparentBatch(nn.Module):
    """Unique-target TFSF face plan sampling one device-resident auxiliary solver."""

    def __init__(
        self,
        component,
        parameters,
        *,
        shape,
        auxiliary,
        gaussian_width,
        dt,
        dr,
        paired_real,
        device,
        dtype,
    ):
        super().__init__()
        self.component = component
        self.auxiliary = auxiliary
        self.auxiliary_component = "Hy" if component.startswith("E") else "Ex"
        self.paired_real = paired_real
        self.gaussian_width = gaussian_width
        aux_shape = auxiliary.plan.shapes[self.auxiliary_component]
        targets = []
        terms = []
        for target, parameter in parameters.items():
            target_terms = []
            for face in parameter.face_list:
                coefficient = _transparent_coefficient(
                    component, face, parameter, dt, dr
                )
                for sample, weight in (
                    (parameter.samp_idx0[face], parameter.r0[face]),
                    (parameter.samp_idx1[face], parameter.r1[face]),
                ):
                    target_terms.append(
                        (
                            int(np.ravel_multi_index(sample, aux_shape)),
                            coefficient * weight,
                        )
                    )
            consolidated = {}
            for sample, weight in target_terms:
                consolidated[sample] = consolidated.get(sample, 0.0) + weight
            targets.append(int(np.ravel_multi_index(target, shape)))
            terms.append(tuple(consolidated.items()))
        width = max((len(row) for row in terms), default=0)
        samples = np.zeros((len(terms), width), dtype=np.int64)
        weights = np.zeros((len(terms), width), dtype=np.float64)
        for row, values in enumerate(terms):
            for column, (sample, weight) in enumerate(values):
                samples[row, column] = sample
                weights[row, column] = weight
        self.register_buffer(
            "targets", torch.tensor(targets, device=device, dtype=torch.int64)
        )
        self.register_buffer(
            "samples", torch.tensor(samples, device=device, dtype=torch.int64)
        )
        self.register_buffer(
            "weights", torch.tensor(weights, device=device, dtype=dtype)
        )
        plane = 2 if paired_real else 1
        self.register_buffer(
            "_sample_values",
            torch.zeros(tuple(samples.shape) + (plane,), device=device, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "_values",
            torch.zeros((len(targets), plane), device=device, dtype=dtype),
            persistent=False,
        )

    def apply(self, field, source_time):
        plane = 2 if self.paired_real else 1
        auxiliary = self.auxiliary.state.field(self.auxiliary_component).reshape(
            -1, plane
        )
        torch.index_select(
            auxiliary,
            0,
            self.samples.reshape(-1),
            out=self._sample_values.reshape(-1, plane),
        )
        self._sample_values.mul_(self.weights[..., None])
        torch.sum(self._sample_values, dim=1, out=self._values)
        if self.gaussian_width is not None:
            envelope = torch.where(
                source_time < self.gaussian_width,
                torch.sin(0.5 * torch.pi * source_time / self.gaussian_width).square(),
                torch.ones_like(source_time),
            )
            self._values.mul_(envelope)
        field.reshape(-1, plane).index_add_(0, self.targets, self._values)


class TorchSourcePlan(nn.Module):
    """Static source batches and their device-resident auxiliary simulations."""

    def __init__(self, batches, auxiliaries):
        super().__init__()
        self.batches = nn.ModuleList(batches)
        self.auxiliaries = tuple(auxiliaries)

    @property
    def empty(self):
        """Return whether the plan contains no source batches."""

        return not self.batches

    def apply(self, simulation, *, electric, time, transparent_time):
        prefix = "E" if electric else "H"
        for batch in self.batches:
            if batch.component.startswith(prefix):
                batch_time = (
                    transparent_time
                    if isinstance(batch, TorchTransparentBatch)
                    else time
                )
                batch.apply(simulation.state.field(batch.component), batch_time)

    def step_auxiliaries(self):
        """Advance every device-resident auxiliary source simulation once."""

        for auxiliary in self.auxiliaries:
            auxiliary.step()


def _is_owned_target(simulation, component, target):
    shape = simulation.plan.shapes[component]
    target = tuple(int(value) for value in target)
    if len(target) != 3 or any(
        value < 0 or value >= limit for value, limit in zip(target, shape)
    ):
        return False
    linear = int(np.ravel_multi_index(target, shape))
    return simulation.plan.components[component].ownership.reshape(-1)[linear] >= 0


def lower_sources(
    sources,
    *,
    simulation,
    simulation_factory,
    runtime,
    bloch,
):
    """Lower legacy built-ins or explicit extension records exactly once."""
    sources = tuple(sources)
    if not sources:
        return TorchSourcePlan((), ())
    context = TorchSourceLoweringContext(
        simulation.space,
        simulation.geom_tree,
        simulation.state.paired_real,
        simulation.dtype,
        simulation.device,
        simulation.plan.dt,
    )
    extension_records = {name: [] for name in COMPONENTS}
    legacy_sources = []
    for source in sources:
        if type(source) in (PointSource, TotalFieldScatteredField, GaussianBeam):
            if isinstance(source, PointSource) and source.filename is not None:
                raise ValueError(
                    "PointSource filename output is unsupported by TorchSimulation; "
                    "use an explicit bounded probe/output adapter"
                )
            source.init(
                simulation.geom_tree, simulation.space, simulation.state.paired_real
            )
            legacy_sources.append(source)
            continue
        lower = getattr(source, "lower_torch_source", None)
        if lower is None:
            raise TypeError(
                f"unsupported legacy source {type(source).__name__!r}; implement "
                "lower_torch_source(context) to emit TorchPointSourceRecord values"
            )
        records = tuple(lower(context))
        for record in records:
            if not isinstance(record, TorchPointSourceRecord):
                raise TypeError(
                    "lower_torch_source() must emit TorchPointSourceRecord values"
                )
            if record.component not in COMPONENTS:
                raise ValueError(f"unknown source component {record.component!r}")
            extension_records[record.component].append(record)

    auxiliary_by_native = {}
    auxiliaries = []
    for source in legacy_sources:
        if not isinstance(source, TotalFieldScatteredField):
            continue
        native_auxiliary = source.aux_fdtd
        base = (
            native_auxiliary.aux_fdtd
            if isinstance(source, GaussianBeam)
            else native_auxiliary
        )
        auxiliary = simulation_factory(
            space=base.space,
            geometry=base.geom_list,
            sources=base.src_list,
            runtime=runtime,
            dt=simulation.plan.dt,
            bloch=(0.0, 0.0, 0.0) if bloch is not None else None,
            _is_auxiliary=True,
        )
        if isinstance(source, GaussianBeam) and base.time_step.n:
            auxiliary.advance(int(base.time_step.n))
        auxiliary_by_native[id(native_auxiliary)] = (
            auxiliary,
            source.src_time.width if isinstance(source, GaussianBeam) else None,
        )
        auxiliaries.append(auxiliary)

    batches = []
    for component in COMPONENTS:
        merged = {}
        getter_name = f"get_pw_source_{component.lower()}"
        for source in legacy_sources:
            pointwise = getattr(source, getter_name)(
                np.empty(simulation.plan.shapes[component]),
                simulation.space,
                simulation.geom_tree,
            )
            if pointwise is None:
                continue
            source_type = type(pointwise)
            if source_type in merged:
                merged[source_type].merge(pointwise)
            else:
                merged[source_type] = pointwise
        for pointwise in merged.values():
            if isinstance(pointwise, (PointSourceElectric, PointSourceMagnetic)):
                records = []
                for target, parameter in pointwise._param.items():
                    if not _is_owned_target(simulation, component, target):
                        continue
                    _, current = _POINT_COMPONENTS[parameter.comp]
                    scale = None
                    if current:
                        material = (
                            parameter.eps_inf
                            if component.startswith("E")
                            else parameter.mu_inf
                        )
                        scale = -simulation.plan.dt / material
                    records.append(
                        TorchPointSourceRecord(
                            component,
                            tuple(int(value) for value in target),
                            parameter.src_time,
                            parameter.amp,
                            scale,
                        )
                    )
                batches.append(
                    TorchPointSourceBatch(
                        component,
                        records,
                        shape=simulation.plan.shapes[component],
                        paired_real=simulation.state.paired_real,
                        device=simulation.device,
                        dtype=simulation.dtype,
                    )
                )
            elif all(
                isinstance(value, TransparentParam)
                for value in pointwise._param.values()
            ):
                by_auxiliary = {}
                for target, parameter in pointwise._param.items():
                    if not _is_owned_target(simulation, component, target):
                        continue
                    by_auxiliary.setdefault(id(parameter.aux_fdtd), {})[
                        target
                    ] = parameter
                for native_auxiliary, parameters in by_auxiliary.items():
                    auxiliary, gaussian_width = auxiliary_by_native[native_auxiliary]
                    batches.append(
                        TorchTransparentBatch(
                            component,
                            parameters,
                            shape=simulation.plan.shapes[component],
                            auxiliary=auxiliary,
                            gaussian_width=gaussian_width,
                            dt=simulation.plan.dt,
                            dr=simulation.plan.dr,
                            paired_real=simulation.state.paired_real,
                            device=simulation.device,
                            dtype=simulation.dtype,
                        )
                    )
            else:  # pragma: no cover - guards future legacy subclasses
                raise TypeError(
                    f"unsupported pointwise source {type(pointwise).__name__!r}"
                )
        owned_extension_records = [
            record
            for record in extension_records[component]
            if _is_owned_target(simulation, component, record.target)
        ]
        if owned_extension_records:
            batches.append(
                TorchPointSourceBatch(
                    component,
                    owned_extension_records,
                    shape=simulation.plan.shapes[component],
                    paired_real=simulation.state.paired_real,
                    device=simulation.device,
                    dtype=simulation.dtype,
                )
            )
    return TorchSourcePlan(batches, auxiliaries)


__all__ = [
    "TorchPointSourceRecord",
    "TorchSourceLoweringContext",
    "TorchSourcePlan",
]
