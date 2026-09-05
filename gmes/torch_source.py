"""Device-resident source lowering and execution for :mod:`gmes.torch_fdtd`."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any, Protocol, cast

import numpy as np
import torch
from torch import nn

from . import constant as const
from .source import (
    AuxiliarySourceSpec,
    Bandpass,
    Continuous,
    DifferentiatedGaussian,
    GaussianBeam,
    PaperSourceTime,
    PointSource,
    PointSourceRecord,
    PumpProbe,
    SechSinePulse,
    SmoothSine,
    TfsfFaceRule,
    TotalFieldScatteredField,
    UltrafastPulse,
    UltrafastPulseTrain,
)
from .torch_plan import COMPONENTS, ComponentPlan


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


class _TorchPointSourceLowerer(Protocol):
    """Third-party hook that emits exact Torch point-source records."""

    def lower_torch_source(
        self, context: TorchSourceLoweringContext
    ) -> Iterable[TorchPointSourceRecord]:
        """Lower this source once using the immutable host context."""


class _SourceRuntime(Protocol):
    """Runtime property required while constructing nested sources."""

    @property
    def precision(self) -> str:
        """Return the immutable precision selected for this runtime."""


def _finite_scalar(value: object, name: str) -> float:
    """Convert a required source scalar while rejecting malformed values."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real scalar")
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a finite real scalar") from error
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class _PreparedPointRecord:
    """Fully host-validated point update ready for tensor allocation."""

    component: str
    target: int
    model: int
    parameters: tuple[float, ...]
    amplitude: float
    current_scale: float | None


@dataclass(frozen=True)
class _PreparedTransparentRecord:
    """Fully host-validated transparent update ready for tensor allocation."""

    target: int
    terms: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class _PreparedSourceSet:
    """Host-only lowering result; deliberately contains no Torch tensors."""

    context: TorchSourceLoweringContext
    paired_real: bool
    points: dict[str, tuple[_PreparedPointRecord, ...]]
    transparent: dict[str, dict[int, tuple[_PreparedTransparentRecord, ...]]]
    specs: dict[int, AuxiliarySourceSpec]


def _index3(
    value: object, *, name: str, shape: tuple[int, int, int]
) -> tuple[int, int, int]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError(f"{name} must be a three-integer tuple")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, np.integer))
        for item in value
    ):
        raise TypeError(f"{name} must be a three-integer tuple")
    result = tuple(int(item) for item in value)
    if any(
        item < 0 or item >= limit for item, limit in zip(result, shape, strict=True)
    ):
        raise ValueError(f"{name} is outside its component shape")
    return cast(tuple[int, int, int], result)


def _component_shapes(space: Any) -> dict[str, tuple[int, int, int]]:
    nx, ny, nz = (int(value) for value in space.my_field_size)
    return {
        "Ex": (nx, ny + 1, nz + 1),
        "Ey": (nx + 1, ny, nz + 1),
        "Ez": (nx + 1, ny + 1, nz),
        "Hx": (nx, ny + 1, nz + 1),
        "Hy": (nx + 1, ny, nz + 1),
        "Hz": (nx + 1, ny + 1, nz),
    }


def _time_parameters(source_time: Any) -> tuple[int, tuple[float, ...]]:
    """Validate a built-in waveform before it can reach tensor allocation."""
    if isinstance(source_time, Continuous):
        freq = _finite_scalar(source_time.freq, "Continuous.freq")
        phase = _finite_scalar(source_time.phase, "Continuous.phase")
        start = _finite_scalar(source_time.start, "Continuous.start")
        end = float(source_time.end)
        if not (isfinite(end) or end == float("inf")):
            raise ValueError("Continuous.end must be finite or +inf")
        width = _finite_scalar(source_time.width, "Continuous.width")
        if freq < 0 or width < 0:
            raise ValueError("Continuous frequency and width must be nonnegative")
        return 0, (freq, phase, start, end, width, 0.0)
    if isinstance(source_time, Bandpass):
        freq = _finite_scalar(source_time.freq, "Bandpass.freq")
        phase = _finite_scalar(source_time.phase, "Bandpass.phase")
        width = _finite_scalar(source_time.width, "Bandpass.width")
        peak = _finite_scalar(source_time.peak_time, "Bandpass.peak_time")
        cutoff = _finite_scalar(source_time.cutoff, "Bandpass.cutoff")
        if freq <= 0 or width <= 0 or cutoff < 0:
            raise ValueError(
                "Bandpass frequency/width must be positive and cutoff nonnegative"
            )
        return 1, (freq, phase, width, peak, cutoff, 0.0)
    if isinstance(source_time, DifferentiatedGaussian):
        width = _finite_scalar(source_time.tw, "DifferentiatedGaussian.tw")
        center = _finite_scalar(source_time.t0, "DifferentiatedGaussian.t0")
        if width <= 0:
            raise ValueError("DifferentiatedGaussian.tw must be positive")
        return 2, (width, center, 0.0, 0.0, 0.0, 0.0)
    if isinstance(source_time, SechSinePulse):
        omega = _finite_scalar(source_time.omega, "SechSinePulse.omega")
        width = _finite_scalar(source_time.pulse_width, "SechSinePulse.pulse_width")
        if width <= 0:
            raise ValueError("SechSinePulse.pulse_width must be positive")
        return 3, (omega, width, 0.0, 0.0, 0.0, 0.0)
    if isinstance(source_time, UltrafastPulse):
        width = _finite_scalar(source_time.pulse_width, "UltrafastPulse.pulse_width")
        if width <= 0:
            raise ValueError("UltrafastPulse.pulse_width must be positive")
        return 4, (width, 0.0, 0.0, 0.0, 0.0, 0.0)
    if isinstance(source_time, UltrafastPulseTrain):
        width = _finite_scalar(
            source_time.pulse_width, "UltrafastPulseTrain.pulse_width"
        )
        alpha = _finite_scalar(source_time.alpha, "UltrafastPulseTrain.alpha")
        delay = _finite_scalar(source_time.delay, "UltrafastPulseTrain.delay")
        if width <= 0:
            raise ValueError("UltrafastPulseTrain.pulse_width must be positive")
        return 5, (width, alpha, delay, 0.0, 0.0, 0.0)
    if isinstance(source_time, SmoothSine):
        omega = _finite_scalar(source_time.omega, "SmoothSine.omega")
        period = _finite_scalar(source_time.period, "SmoothSine.period")
        if period <= 0:
            raise ValueError("SmoothSine.period must be positive")
        return 6, (omega, period, 0.0, 0.0, 0.0, 0.0)
    if isinstance(source_time, PumpProbe):
        omega = _finite_scalar(source_time.probe.omega, "PumpProbe.omega")
        width = _finite_scalar(source_time.pump.pulse_width, "PumpProbe.pulse_width")
        beta = _finite_scalar(source_time.beta, "PumpProbe.beta")
        delay = _finite_scalar(source_time.delay, "PumpProbe.delay")
        if width <= 0:
            raise ValueError("PumpProbe.pulse_width must be positive")
        return 7, (omega, width, beta, delay, 0.0, 0.0)
    raise TypeError(
        f"unsupported Torch source-time model {type(source_time).__name__!r}; "
        "use Continuous, Bandpass, DifferentiatedGaussian, or an explicit "
        "lower_torch_source() extension"
    )


def _evaluate_time(
    model: Any, parameters: Any, time: Any, paired_real: Any, output: Any
) -> Any:
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

    sech_gamma = (time - 0.5 * parameters[:, 1]) / (0.5 * parameters[:, 1])
    sech_real = torch.where(
        (time >= 0) & (time <= parameters[:, 1]),
        torch.sin(parameters[:, 0] * time) / torch.cosh(10 * sech_gamma),
        torch.zeros_like(time + parameters[:, 0]),
    )

    ultrafast_x = 2 * time / parameters[:, 0] - 1
    ultrafast_shape = -4.201355 * ultrafast_x * (1 - ultrafast_x.square()).pow(3)
    ultrafast_real = torch.where(
        (time >= 0) & (time <= parameters[:, 0]),
        ultrafast_shape,
        torch.zeros_like(ultrafast_shape),
    )

    delayed_ultrafast_time = time - parameters[:, 2]
    delayed_ultrafast_x = 2 * delayed_ultrafast_time / parameters[:, 0] - 1
    delayed_ultrafast = (
        -4.201355 * delayed_ultrafast_x * (1 - delayed_ultrafast_x.square()).pow(3)
    )
    train_real = ultrafast_real + parameters[:, 1] * torch.where(
        (delayed_ultrafast_time >= 0) & (delayed_ultrafast_time <= parameters[:, 0]),
        delayed_ultrafast,
        torch.zeros_like(delayed_ultrafast),
    )

    smooth_rise = 5 * parameters[:, 1]
    smooth_x = time / smooth_rise - 1
    smooth_envelope = torch.where(
        time < 0,
        torch.zeros_like(smooth_x),
        torch.where(
            time >= smooth_rise,
            torch.ones_like(smooth_x),
            (1 - smooth_x.square()).pow(4),
        ),
    )
    smooth_real = smooth_envelope * torch.sin(parameters[:, 0] * time)

    probe_time = time - parameters[:, 3]
    probe_rise = 5 * parameters[:, 1]
    probe_x = probe_time / probe_rise - 1
    probe_envelope = torch.where(
        probe_time < 0,
        torch.zeros_like(probe_x),
        torch.where(
            probe_time >= probe_rise,
            torch.ones_like(probe_x),
            (1 - probe_x.square()).pow(4),
        ),
    )
    pump_x = 2 * time / parameters[:, 1] - 1
    pump_real = torch.where(
        (time >= 0) & (time <= parameters[:, 1]),
        -4.201355 * pump_x * (1 - pump_x.square()).pow(3),
        torch.zeros_like(pump_x),
    )
    pump_probe_real = pump_real + parameters[:, 2] * probe_envelope * torch.sin(
        parameters[:, 0] * probe_time
    )
    zero = torch.zeros_like(gaussian_real)

    real = torch.where(
        model == 0,
        continuous_real,
        torch.where(
            model == 1,
            bandpass_real,
            torch.where(
                model == 2,
                gaussian_real,
                torch.where(
                    model == 3,
                    sech_real,
                    torch.where(
                        model == 4,
                        ultrafast_real,
                        torch.where(
                            model == 5,
                            train_real,
                            torch.where(model == 6, smooth_real, pump_probe_real),
                        ),
                    ),
                ),
            ),
        ),
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

    component: str
    paired_real: bool
    overwrite_targets: torch.Tensor
    overwrite_models: torch.Tensor
    overwrite_parameters: torch.Tensor
    overwrite_amplitudes: torch.Tensor
    _overwrite_values: torch.Tensor
    additive_targets: torch.Tensor
    additive_models: torch.Tensor
    additive_parameters: torch.Tensor
    additive_amplitudes: torch.Tensor
    _additive_values: torch.Tensor

    def __init__(
        self,
        component: Any,
        records: Any,
        *,
        paired_real: Any,
        device: Any,
        dtype: Any,
    ) -> None:
        super().__init__()
        self.component = component
        self.paired_real = paired_real
        overwrite: list[tuple[int, int, tuple[float, ...], float, float]] = []
        additive: list[tuple[int, int, tuple[float, ...], float, float]] = []
        # Point-source overwrite semantics are last-source-wins at one target.
        # Make that decision here, before any indexed device write occurs.
        normalized: dict[int, _PreparedPointRecord] = {}
        for record in records:
            normalized[record.target] = record
        for record in normalized.values():
            item = (record.target, record.model, record.parameters, record.amplitude)
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

    def _register(
        self, prefix: Any, records: Any, *, paired_real: Any, device: Any, dtype: Any
    ) -> Any:
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

    def apply(  # type: ignore[override]  # Source execution, not Module traversal
        self, field: torch.Tensor, time: torch.Tensor
    ) -> None:
        """Apply this point-source batch to ``field`` in place."""

        plane = 2 if self.paired_real else 1
        flat = field.reshape(-1, plane)
        for prefix, additive in (("overwrite", False), ("additive", True)):
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


class TorchTransparentBatch(nn.Module):
    """Unique-target TFSF face plan sampling one device-resident auxiliary solver."""

    component: str
    auxiliary: Any
    auxiliary_component: str
    paired_real: bool
    gaussian_width: float | None
    targets: torch.Tensor
    samples: torch.Tensor
    weights: torch.Tensor
    _sample_values: torch.Tensor
    _values: torch.Tensor
    _outer_values: torch.Tensor
    _envelope_step: torch.Tensor
    _envelope_step_offset: torch.Tensor
    _envelope: torch.Tensor

    def __init__(
        self,
        component: Any,
        records: Iterable[_PreparedTransparentRecord],
        *,
        auxiliary: Any,
        gaussian_width: Any,
        paired_real: Any,
        device: Any,
        dtype: Any,
    ) -> None:
        super().__init__()
        records = tuple(records)
        self.component = component
        self.auxiliary = auxiliary
        self.auxiliary_component = "Hy" if component.startswith("E") else "Ex"
        self.paired_real = paired_real
        self.gaussian_width = gaussian_width
        grouped: dict[int, list[tuple[int, float]]] = {}
        for record in records:
            terms = grouped.setdefault(record.target, [])
            terms.extend(record.terms)
        targets = []
        rows = []
        for target, terms in grouped.items():
            merged: dict[int, float] = {}
            for sample, weight in terms:
                merged[sample] = merged.get(sample, 0.0) + weight
            targets.append(target)
            rows.append(tuple(merged.items()))
        width = max((len(row) for row in rows), default=0)
        samples = np.zeros((len(rows), width), dtype=np.int64)
        weights = np.zeros((len(rows), width), dtype=np.float64)
        for row, values in enumerate(rows):
            for column, (sample, weight) in enumerate(values):
                samples[row, column] = sample
                weights[row, column] = weight
        auxiliary_dtype = auxiliary.dtype
        self.register_buffer(
            "targets", torch.tensor(targets, device=device, dtype=torch.int64)
        )
        self.register_buffer(
            "samples", torch.tensor(samples, device=device, dtype=torch.int64)
        )
        self.register_buffer(
            "weights", torch.tensor(weights, device=device, dtype=auxiliary_dtype)
        )
        plane = 2 if paired_real else 1
        self.register_buffer(
            "_sample_values",
            torch.zeros(
                tuple(samples.shape) + (plane,), device=device, dtype=auxiliary_dtype
            ),
            persistent=False,
        )
        self.register_buffer(
            "_values",
            torch.zeros((len(targets), plane), device=device, dtype=auxiliary_dtype),
            persistent=False,
        )
        self.register_buffer(
            "_outer_values",
            torch.zeros((len(targets), plane), device=device, dtype=dtype),
            persistent=False,
        )
        if gaussian_width is not None:
            self.register_buffer(
                "_envelope_step",
                torch.zeros((), device=device, dtype=torch.int64),
                persistent=False,
            )
            self.register_buffer(
                "_envelope_step_offset", auxiliary.state.step_count.detach().clone()
            )
            self.register_buffer(
                "_envelope",
                torch.zeros((), device=device, dtype=auxiliary_dtype),
                persistent=False,
            )

    def apply(  # type: ignore[override]  # Source execution, not Module traversal
        self, field: torch.Tensor, _source_time: torch.Tensor
    ) -> None:
        """Apply this transparent-source batch to ``field`` in place."""

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
            self._envelope_step.copy_(self.auxiliary.state.step_count).sub_(
                self._envelope_step_offset
            )
            self._envelope.copy_(self._envelope_step).mul_(
                self.auxiliary.state.time_step
            )
            if self.gaussian_width > 0:
                self._envelope.clamp_(max=self.gaussian_width)
                self._envelope.mul_(
                    0.5 * torch.pi / self.gaussian_width
                ).sin_().square_()
            else:
                self._envelope.fill_(1.0)
            self._values.mul_(self._envelope)
        self._outer_values.copy_(self._values)
        field.reshape(-1, plane).index_add_(0, self.targets, self._outer_values)


class _SourceState(Protocol):
    """Field access required while executing a lowered source plan."""

    def field(self, component: str) -> torch.Tensor:
        """Return a live component field."""


class _SourceSimulation(Protocol):
    """Simulation surface consumed by source batches."""

    @property
    def state(self) -> _SourceState:
        """Return live simulation state."""


class _AuxiliarySimulation(_SourceSimulation, Protocol):
    """Nested simulation lifecycle owned by transparent sources."""

    runtime: Any
    dtype: torch.dtype
    plan_identity: str
    compile_cache_key: str

    def step(self) -> object:
        """Advance the auxiliary simulation once."""

    def advance(self, steps: int) -> object:
        """Advance the auxiliary simulation by a positive step count."""

    def checkpoint(self) -> Mapping[str, object]:
        """Return auxiliary checkpoint data."""

    def load_checkpoint(self, checkpoint: Mapping[str, object]) -> object:
        """Restore auxiliary checkpoint data."""

    def buffer_addresses(self) -> Mapping[str, int]:
        """Return persistent buffer addresses."""


class TorchSourcePlan(nn.Module):
    """Static source batches and their device-resident auxiliary simulations."""

    batches: nn.ModuleList
    auxiliaries: tuple[_AuxiliarySimulation, ...]

    def __init__(
        self,
        batches: Iterable[TorchPointSourceBatch | TorchTransparentBatch],
        auxiliaries: Iterable[_AuxiliarySimulation],
    ) -> None:
        super().__init__()
        self.batches = nn.ModuleList(batches)
        self.auxiliaries = tuple(auxiliaries)

    @property
    def empty(self) -> bool:
        """Return whether the plan contains no source batches."""

        return not self.batches

    def apply(  # type: ignore[override]  # Source execution, not Module traversal
        self,
        simulation: _SourceSimulation,
        *,
        electric: bool,
        time: torch.Tensor,
        transparent_time: torch.Tensor,
    ) -> None:
        """Execute source batches for the selected field phase."""

        prefix = "E" if electric else "H"
        for batch in cast(
            Iterable[TorchPointSourceBatch | TorchTransparentBatch], self.batches
        ):
            if batch.component.startswith(prefix):
                batch_time = (
                    transparent_time
                    if isinstance(batch, TorchTransparentBatch)
                    else time
                )
                batch.apply(simulation.state.field(batch.component), batch_time)

    def step_auxiliaries(self) -> None:
        """Advance every device-resident auxiliary source simulation once."""

        for auxiliary in self.auxiliaries:
            auxiliary.step()


def _validate_auxiliary_spec(spec: AuxiliarySourceSpec) -> None:
    if not isinstance(spec, AuxiliarySourceSpec):
        raise TypeError("TFSF auxiliary_spec must be an AuxiliarySourceSpec")
    if not isinstance(spec.prewarm_steps, (int, np.integer)) or spec.prewarm_steps < 0:
        raise ValueError("TFSF prewarm_steps must be a nonnegative integer")
    if spec.gaussian_width is not None:
        _finite_scalar(spec.gaussian_width, "TFSF gaussian_width")
    for source in spec.sources:
        if type(source) is not PointSource:
            raise TypeError(
                "TFSF auxiliary sources must be built-in PointSource values"
            )
        _time_parameters(source.src_time)
        _finite_scalar(source.amp, "TFSF auxiliary point amplitude")


def prepare_sources(
    sources: Iterable[object],
    *,
    context: TorchSourceLoweringContext,
    component_plans: Iterable[ComponentPlan],
) -> _PreparedSourceSet:
    """Lower and validate every source before any Torch plan/state allocation."""
    plans = {plan.name: plan for plan in component_plans}
    if set(plans) != set(COMPONENTS):
        raise ValueError("component_plans must describe every Yee component")
    points: dict[str, list[_PreparedPointRecord]] = {name: [] for name in COMPONENTS}
    transparent: dict[str, dict[int, list[_PreparedTransparentRecord]]] = {
        name: {} for name in COMPONENTS
    }
    specs: dict[int, AuxiliarySourceSpec] = {}
    owners: dict[tuple[str, int], int] = {}
    for source in tuple(sources):
        builtin = type(source) in (PointSource, TotalFieldScatteredField, GaussianBeam)
        lower = getattr(source, "lower_torch_source", None)
        if lower is None:
            raise TypeError(
                f"unsupported source {type(source).__name__!r}; implement "
                "lower_torch_source(context)"
            )
        output = lower(context)  # Extensions are called and consumed exactly once.
        for record in output:
            if builtin and type(record) is PointSourceRecord:
                record = TorchPointSourceRecord(
                    record.component,
                    record.target,
                    record.source_time,
                    record.amplitude,
                    record.current_scale,
                )
            if type(record) is TorchPointSourceRecord:
                if record.component not in plans:
                    raise ValueError(f"unknown source component {record.component!r}")
                target = _index3(
                    record.target,
                    name="source target",
                    shape=plans[record.component].shape,
                )
                model, parameters = _time_parameters(record.source_time)
                if context.paired_real and isinstance(
                    record.source_time, PaperSourceTime
                ):
                    raise ValueError("Ziolkowski reproductions require real fields")
                amplitude = _finite_scalar(record.amplitude, "source amplitude")
                scale = (
                    None
                    if record.current_scale is None
                    else _finite_scalar(record.current_scale, "source current_scale")
                )
                linear = int(
                    np.ravel_multi_index(target, plans[record.component].shape)
                )
                if plans[record.component].ownership.reshape(-1)[linear] >= 0:
                    points[record.component].append(
                        _PreparedPointRecord(
                            record.component,
                            linear,
                            model,
                            parameters,
                            amplitude,
                            scale,
                        )
                    )
            elif builtin and type(record) is TfsfFaceRule:
                if record.component not in plans:
                    raise ValueError(f"unknown TFSF component {record.component!r}")
                expected = "Hy" if record.component.startswith("E") else "Ex"
                if record.sample_component != expected:
                    raise ValueError("incompatible direct TFSF sample component")
                _validate_auxiliary_spec(record.auxiliary_spec)
                target = _index3(
                    record.target,
                    name="TFSF target",
                    shape=plans[record.component].shape,
                )
                aux_shape = _component_shapes(record.auxiliary_spec.space)[expected]
                sample0 = _index3(record.sample0, name="TFSF sample0", shape=aux_shape)
                sample1 = _index3(record.sample1, name="TFSF sample1", shape=aux_shape)
                coefficient = _finite_scalar(record.coefficient, "TFSF coefficient")
                weight0 = _finite_scalar(record.weight0, "TFSF weight0")
                weight1 = _finite_scalar(record.weight1, "TFSF weight1")
                linear = int(
                    np.ravel_multi_index(target, plans[record.component].shape)
                )
                spec_id = id(record.auxiliary_spec)
                owner = owners.setdefault((record.component, linear), spec_id)
                if owner != spec_id:
                    raise ValueError(
                        "overlapping TFSF rules for one field target must use the same auxiliary specification"
                    )
                specs.setdefault(spec_id, record.auxiliary_spec)
                if plans[record.component].ownership.reshape(-1)[linear] >= 0:
                    terms = (
                        (
                            int(np.ravel_multi_index(sample0, aux_shape)),
                            coefficient * weight0,
                        ),
                        (
                            int(np.ravel_multi_index(sample1, aux_shape)),
                            coefficient * weight1,
                        ),
                    )
                    transparent[record.component].setdefault(spec_id, []).append(
                        _PreparedTransparentRecord(linear, terms)
                    )
            else:
                if not builtin and type(record) is not TorchPointSourceRecord:
                    raise TypeError(
                        "third-party lower_torch_source() must emit exact TorchPointSourceRecord values"
                    )
                raise TypeError(
                    "built-in source lowering emitted an unsupported record"
                )
    return _PreparedSourceSet(
        context,
        context.paired_real,
        {name: tuple(records) for name, records in points.items()},
        {
            name: {spec_id: tuple(records) for spec_id, records in values.items()}
            for name, values in transparent.items()
        },
        specs,
    )


def materialize_sources(
    prepared: _PreparedSourceSet,
    *,
    simulation_factory: Callable[..., _AuxiliarySimulation],
    runtime: _SourceRuntime,
    bloch: Sequence[float] | None,
) -> TorchSourcePlan:
    """Allocate a source plan from an already validated host-only lowering."""
    auxiliaries: list[_AuxiliarySimulation] = []
    auxiliary_by_spec: dict[int, _AuxiliarySimulation] = {}
    for spec_id, spec in prepared.specs.items():
        auxiliary = simulation_factory(
            space=spec.space,
            geometry=spec.geometry,
            sources=spec.sources,
            runtime=runtime,
            dt=prepared.context.dt,
            bloch=(0.0, 0.0, 0.0) if prepared.paired_real else None,
            _is_auxiliary=True,
        )
        if spec.prewarm_steps:
            auxiliary.advance(spec.prewarm_steps)
        auxiliary_by_spec[spec_id] = auxiliary
        auxiliaries.append(auxiliary)
    batches: list[TorchPointSourceBatch | TorchTransparentBatch] = []
    for component in COMPONENTS:
        if prepared.points[component]:
            batches.append(
                TorchPointSourceBatch(
                    component,
                    prepared.points[component],
                    paired_real=prepared.paired_real,
                    device=prepared.context.device,
                    dtype=prepared.context.dtype,
                )
            )
        for spec_id, records in prepared.transparent[component].items():
            batches.append(
                TorchTransparentBatch(
                    component,
                    records,
                    auxiliary=auxiliary_by_spec[spec_id],
                    gaussian_width=prepared.specs[spec_id].gaussian_width,
                    paired_real=prepared.paired_real,
                    device=prepared.context.device,
                    dtype=prepared.context.dtype,
                )
            )
    return TorchSourcePlan(batches, auxiliaries)


__all__ = [
    "TorchPointSourceRecord",
    "TorchSourceLoweringContext",
    "TorchSourcePlan",
    "materialize_sources",
    "prepare_sources",
]
