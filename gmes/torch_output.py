"""Bounded device probes and explicit host-output adapters for Torch FDTD."""

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from .torch_plan import COMPONENTS

type Index3 = tuple[int, int, int]
type RealArray = NDArray[np.float64]
type SampleArray = NDArray[np.float64] | NDArray[np.complex128]


class _ProbeState(Protocol):
    @property
    def paired_real(self) -> bool:
        """Return whether complex fields use paired real storage."""

    def field(self, component: str) -> torch.Tensor:
        """Return the tensor for a field component."""


class _ProbePlan(Protocol):
    @property
    def shapes(self) -> Mapping[str, Index3]:
        """Return component field shapes."""


class _ProbeSimulation(Protocol):
    @property
    def space(self) -> object:
        """Return the simulation space."""

    @property
    def plan(self) -> _ProbePlan:
        """Return the lowered execution plan."""

    @property
    def state(self) -> _ProbeState:
        """Return the device field state."""

    @property
    def device(self) -> torch.device:
        """Return the execution device."""

    @property
    def dtype(self) -> torch.dtype:
        """Return the field tensor dtype."""


@dataclass(frozen=True)
class TorchProbeSpec:
    """Describe a bounded probe at an index or spatial coordinate."""

    component: str | type[object]
    location: tuple[float, float, float]
    capacity: int = 1024
    coordinates: str = "index"


@dataclass(frozen=True)
class TorchProbeSpectrum:
    """Frequency-domain view produced at an explicit host boundary."""

    frequencies: RealArray
    amplitudes: NDArray[np.complex128]


@dataclass(frozen=True)
class TorchProbeSamples:
    """One explicitly synchronized host batch from a probe ring."""

    component: str
    index: tuple[int, int, int]
    times: RealArray
    values: SampleArray
    dropped: int
    total: int


def _component_name(component: str | type[object]) -> str:
    name = component if isinstance(component, str) else component.__name__
    if name not in COMPONENTS:
        raise ValueError(f"unknown probe component {name!r}")
    return name


def _resolve_index(spec: TorchProbeSpec, space: object, shape: Sequence[int]) -> Index3:
    if spec.coordinates not in {"index", "space"}:
        raise ValueError("probe coordinates must be 'index' or 'space'")
    if spec.coordinates == "space":
        method = getattr(
            space, f"space_to_{_component_name(spec.component).lower()}_index"
        )
        index = method(*spec.location)
    else:
        index = spec.location
        if any(float(value) != int(value) for value in index):
            raise ValueError("index-coordinate probe locations must be integral")
    index = tuple(int(value) for value in index)
    if len(index) != 3 or any(
        value < 0 or value >= limit for value, limit in zip(index, shape)
    ):
        raise ValueError(
            f"probe index {index!r} is outside field shape {tuple(shape)!r}"
        )
    return index


class _TorchProbeRing(nn.Module):
    component: str
    index: tuple[int, int, int]
    capacity: int
    paired_real: bool
    samples: torch.Tensor
    times: torch.Tensor
    write_count: torch.Tensor
    total_count: torch.Tensor

    def __init__(
        self,
        spec: TorchProbeSpec,
        *,
        space: object,
        shape: Sequence[int],
        paired_real: bool,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if (
            isinstance(spec.capacity, bool)
            or not isinstance(spec.capacity, int)
            or spec.capacity < 1
        ):
            raise ValueError("probe capacity must be a positive integer")
        self.component = _component_name(spec.component)
        self.index = _resolve_index(spec, space, shape)
        self.capacity = spec.capacity
        self.paired_real = paired_real
        plane = (2,) if paired_real else ()
        self.register_buffer(
            "samples", torch.zeros((spec.capacity,) + plane, device=device, dtype=dtype)
        )
        self.register_buffer(
            "times", torch.zeros(spec.capacity, device=device, dtype=dtype)
        )
        self.register_buffer(
            "write_count", torch.zeros((), device=device, dtype=torch.int64)
        )
        self.register_buffer(
            "total_count", torch.zeros((), device=device, dtype=torch.int64)
        )

    def record(self, field: torch.Tensor, time: torch.Tensor) -> None:
        """Copy one field sample and its time into the device ring."""

        slot = torch.remainder(self.write_count, self.capacity).reshape(1)
        self.samples.index_copy_(0, slot, field[self.index].unsqueeze(0))
        self.times.index_copy_(0, slot, time.reshape(1))
        self.write_count.add_(1)
        self.total_count.add_(1)

    @torch.inference_mode()
    def flush(self) -> TorchProbeSamples:
        """Return chronological host samples and reset the ring cursor."""

        count = int(self.write_count.detach().cpu())
        total = int(self.total_count.detach().cpu())
        available = min(count, self.capacity)
        start = count % self.capacity if count > self.capacity else 0
        order = (np.arange(available, dtype=np.int64) + start) % self.capacity
        order_device = torch.as_tensor(order, device=self.samples.device)
        times = self.times.index_select(0, order_device).detach().cpu().numpy().copy()
        value_tensor = self.samples.index_select(0, order_device).detach().cpu()
        if self.paired_real:
            value_tensor = torch.complex(value_tensor[..., 0], value_tensor[..., 1])
        values = value_tensor.numpy().copy()
        self.write_count.zero_()
        return TorchProbeSamples(
            self.component,
            self.index,
            times,
            values,
            max(0, count - self.capacity),
            total,
        )


class TorchProbeBuffer(nn.Module):
    """A set of bounded overwrite-on-backpressure device probe rings."""

    rings: nn.ModuleList

    def __init__(
        self, specs: Iterable[TorchProbeSpec], *, simulation: _ProbeSimulation
    ) -> None:
        """Allocate bounded device rings for probe specifications.

        Args:
            specs: Iterable of TorchProbeSpec values.
            simulation: Simulation providing field shapes, device, and dtype.
        """

        super().__init__()
        rings = []
        for spec in specs:
            if not isinstance(spec, TorchProbeSpec):
                raise TypeError("probes must contain TorchProbeSpec values")
            name = _component_name(spec.component)
            rings.append(
                _TorchProbeRing(
                    spec,
                    space=simulation.space,
                    shape=simulation.plan.shapes[name],
                    paired_real=simulation.state.paired_real,
                    device=simulation.device,
                    dtype=simulation.dtype,
                )
            )
        self.rings = nn.ModuleList(rings)

    @property
    def empty(self) -> bool:
        """Return whether no probe rings are configured."""

        return not self.rings

    def record(
        self, simulation: _ProbeSimulation, *, electric: bool, time: torch.Tensor
    ) -> None:
        """Record probes belonging to one electric or magnetic half step."""

        prefix = "E" if electric else "H"
        for ring in cast(Iterable[_TorchProbeRing], self.rings):
            if ring.component.startswith(prefix):
                ring.record(simulation.state.field(ring.component), time)

    @torch.inference_mode()
    def flush(self) -> tuple[TorchProbeSamples, ...]:
        """Synchronize buffered samples explicitly and reset each ring cursor."""
        return tuple(
            ring.flush() for ring in cast(Iterable[_TorchProbeRing], self.rings)
        )

    @torch.inference_mode()
    def checkpoint(self) -> dict[str, torch.Tensor]:
        """Clone device ring state for inclusion in a simulation checkpoint."""

        return {
            name: value.detach().clone() for name, value in self.state_dict().items()
        }

    @torch.inference_mode()
    def load_checkpoint(self, values: Mapping[str, torch.Tensor]) -> None:
        """Restore compatible ring tensors from a checkpoint mapping."""

        expected = self.state_dict()
        if set(values) != set(expected):
            raise ValueError("probe checkpoint keys do not match the configured probes")
        for name, target in expected.items():
            source = values[name]
            if source.shape != target.shape or source.dtype != target.dtype:
                raise ValueError(f"probe checkpoint tensor {name!r} is incompatible")
            target.copy_(source.to(target.device))


def probe_spectrum(
    samples: TorchProbeSamples, *, window: Literal["hann"] | None = None
) -> TorchProbeSpectrum:
    """Transform one flushed probe batch without entering the solver hot path."""
    if len(samples.times) < 2:
        raise ValueError("at least two probe samples are required for a spectrum")
    spacing = np.diff(samples.times)
    if not np.allclose(spacing, spacing[0], rtol=1e-7, atol=0):
        raise ValueError("probe times must be uniformly spaced")
    values = samples.values
    if window is not None:
        if window != "hann":
            raise ValueError("the supported probe spectrum window is 'hann'")
        values = values * np.hanning(len(values))
    if np.iscomplexobj(values):
        frequencies = np.fft.fftfreq(len(values), d=float(spacing[0]))
        amplitudes = cast(NDArray[np.complex128], np.fft.fft(values))
    else:
        frequencies = np.fft.rfftfreq(len(values), d=float(spacing[0]))
        amplitudes = cast(NDArray[np.complex128], np.fft.rfft(cast(RealArray, values)))
    return TorchProbeSpectrum(frequencies, amplitudes)


def _encode_checkpoint(value: Any, arrays: Any) -> Any:
    if isinstance(value, torch.Tensor):
        name = f"tensor_{len(arrays):06d}"
        arrays[name] = value.detach().to(device="cpu").contiguous().numpy()
        return {"kind": "tensor", "name": name}
    if isinstance(value, dict):
        return {
            "kind": "dict",
            "items": {
                str(key): _encode_checkpoint(item, arrays)
                for key, item in value.items()
            },
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [_encode_checkpoint(item, arrays) for item in value],
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return {"kind": "value", "value": value}
    raise TypeError(f"unsupported checkpoint value {type(value).__name__!r}")


def _decode_checkpoint(description: Any, arrays: Any, device: Any) -> Any:
    if not isinstance(description, dict):
        raise ValueError("checkpoint node descriptions must be mappings")
    kind = description.get("kind")
    if kind == "tensor":
        name = description["name"]
        if name not in arrays:
            raise ValueError(f"checkpoint tensor {name!r} is missing")
        return torch.from_numpy(np.array(arrays[name], copy=True)).to(device=device)
    if kind == "dict":
        return {
            key: _decode_checkpoint(value, arrays, device)
            for key, value in description["items"].items()
        }
    if kind == "tuple":
        return tuple(
            _decode_checkpoint(value, arrays, device) for value in description["items"]
        )
    if kind == "value":
        return description["value"]
    raise ValueError(f"unsupported checkpoint node kind {kind!r}")


def write_torch_checkpoint(
    checkpoint: Mapping[str, object], filename: str | PathLike[str]
) -> Path:
    """Persist a versioned tensor/JSON schema without arbitrary object pickle."""
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("format") != "gmes.torch.simulation"
        or checkpoint.get("version") != 1
    ):
        raise ValueError("only gmes.torch.simulation checkpoint version 1 is writable")
    arrays: dict[str, np.ndarray] = {}
    description = _encode_checkpoint(checkpoint, arrays)
    metadata = np.frombuffer(
        json.dumps(description, separators=(",", ":"), sort_keys=True).encode(),
        dtype=np.uint8,
    )
    path = Path(filename)
    with path.open("wb") as output:
        np.savez_compressed(output, __metadata__=metadata, **arrays)  # type: ignore[arg-type]  # NumPy stubs type all **values as bool.
    return path


def read_torch_checkpoint(
    filename: str | PathLike[str], *, device: torch.device | str = "cpu"
) -> dict[str, object]:
    """Read the safe tensor/JSON checkpoint schema with pickle disabled."""
    with np.load(Path(filename), allow_pickle=False) as archive:
        if "__metadata__" not in archive:
            raise ValueError("Torch checkpoint metadata is missing")
        try:
            description = json.loads(archive["__metadata__"].tobytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Torch checkpoint metadata is invalid") from error
        arrays = {
            name: archive[name] for name in archive.files if name != "__metadata__"
        }
        try:
            checkpoint = _decode_checkpoint(description, arrays, torch.device(device))
        except (AttributeError, KeyError, TypeError) as error:
            raise ValueError("Torch checkpoint schema is invalid") from error
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("format") != "gmes.torch.simulation"
        or checkpoint.get("version") != 1
    ):
        raise ValueError("unsupported Torch checkpoint format or version")
    return checkpoint


def write_probe_text(
    samples: Iterable[TorchProbeSamples], directory: str | PathLike[str]
) -> tuple[Path, ...]:
    """Write flushed probe batches; never called by the solver hot path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for sequence, batch in enumerate(samples):
        path = directory / f"{batch.component.lower()}-{sequence}.dat"
        values = batch.values
        if np.iscomplexobj(values):
            data = np.column_stack((batch.times, values.real, values.imag))
        else:
            data = np.column_stack((batch.times, values))
        np.savetxt(path, data)
        paths.append(path)
    return tuple(paths)


__all__ = [
    "TorchProbeBuffer",
    "TorchProbeSamples",
    "TorchProbeSpec",
    "TorchProbeSpectrum",
    "probe_spectrum",
    "read_torch_checkpoint",
    "write_probe_text",
    "write_torch_checkpoint",
]
