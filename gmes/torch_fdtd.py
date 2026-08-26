"""Torch-native state and execution for the minimal dielectric FDTD slice."""

import os
from dataclasses import dataclass, field
from math import sqrt
from types import MappingProxyType

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as error:  # pragma: no cover
    raise ImportError(
        "GMES Torch execution requires PyTorch 2.13; run `uv sync --locked`."
    ) from error

from .constant import Ex, Ey, Ez, Hx, Hy, Hz
from .geometry import DefaultMedium, GeomBoxTree
from .material import Dielectric

_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
_ELECTRIC = ("Ex", "Ey", "Ez")
_COMPONENT_TYPES = {"Ex": Ex, "Ey": Ey, "Ez": Ez, "Hx": Hx, "Hy": Hy, "Hz": Hz}


class TorchConfigurationError(ValueError):
    """A requested Torch execution configuration cannot be honored."""


@dataclass(frozen=True)
class DistributedLaunch:
    """Process metadata reserved for later distributed execution."""

    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    local_world_size: int = 1

    def validate(self):
        """Validate launch metadata without initializing a process group."""
        if self.world_size < 1 or self.local_world_size < 1:
            raise TorchConfigurationError("world sizes must be positive")
        if not 0 <= self.rank < self.world_size:
            raise TorchConfigurationError("rank must be inside world_size")
        if not 0 <= self.local_rank < self.local_world_size:
            raise TorchConfigurationError("local_rank must be inside local_world_size")


@dataclass(frozen=True)
class TorchRuntimeConfig:
    """Explicit device, precision, compilation, and launch configuration."""

    device: str
    precision: str = "float64"
    compile_policy: str = "eager"
    cpu_threads: int | None = None
    launch: DistributedLaunch = field(default_factory=DistributedLaunch)
    autograd: bool = False

    def validate_static(self):
        """Reject invalid requests before tensors or compiler state are created."""
        if not isinstance(self.device, str) or not self.device.strip():
            raise TorchConfigurationError(
                "device must be an explicit 'cpu' or 'cuda:N'"
            )
        if self.precision not in {"float32", "float64"}:
            raise TorchConfigurationError("precision must be 'float32' or 'float64'")
        if self.compile_policy not in {"eager", "compile"}:
            raise TorchConfigurationError("compile_policy must be 'eager' or 'compile'")
        if self.cpu_threads is not None and self.cpu_threads < 1:
            raise TorchConfigurationError("cpu_threads must be positive")
        if self.autograd:
            raise TorchConfigurationError(
                "autograd is unsupported; GMES field buffers are inference-only"
            )
        self.launch.validate()

    @property
    def dtype(self):
        """Return the requested real Torch dtype."""
        return {"float32": torch.float32, "float64": torch.float64}[self.precision]


def _resolved_device(config):
    try:
        device = torch.device(config.device)
    except RuntimeError as error:
        raise TorchConfigurationError(
            f"invalid Torch device {config.device!r}"
        ) from error
    if device.type not in {"cpu", "cuda"}:
        raise TorchConfigurationError(
            f"unsupported device {config.device!r}; this slice supports CPU and CUDA"
        )
    if device.type == "cpu":
        if device.index is not None:
            raise TorchConfigurationError(
                "CPU device indices are unsupported; use 'cpu'"
            )
        return device
    index = 0 if device.index is None else device.index
    if not torch.cuda.is_available():
        raise TorchConfigurationError(
            f"CUDA device cuda:{index} was requested, but PyTorch {torch.__version__} "
            "reports CUDA unavailable; install a CUDA 12.6/13.0 wheel and a compatible "
            "NVIDIA driver, or explicitly request device='cpu'."
        )
    count = torch.cuda.device_count()
    if index < 0 or index >= count:
        raise TorchConfigurationError(
            f"CUDA device cuda:{index} was requested, but only {count} device(s) are visible"
        )
    return torch.device("cuda", index)


def torch_runtime_diagnostics(config):
    """Return focused Torch/device diagnostics without unrelated environment data."""
    config.validate_static()
    requested = torch.device(config.device)
    result = {
        "torch": torch.__version__,
        "requested_device": config.device,
        "requested_precision": config.precision,
        "compile_policy": config.compile_policy,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "nccl_available": torch.distributed.is_nccl_available(),
    }
    if requested.type == "cuda" and torch.cuda.is_available():
        index = 0 if requested.index is None else requested.index
        if 0 <= index < torch.cuda.device_count():
            result["device_name"] = torch.cuda.get_device_name(index)
            result["device_capability"] = torch.cuda.get_device_capability(index)
    return result


def _field_shapes(space):
    nx, ny, nz = (int(value) for value in space.my_field_size)
    return MappingProxyType(
        {
            "Ex": (nx, ny + 1, nz + 1),
            "Ey": (nx + 1, ny, nz + 1),
            "Ez": (nx + 1, ny + 1, nz),
            "Hx": (nx, ny + 1, nz + 1),
            "Hy": (nx + 1, ny, nz + 1),
            "Hz": (nx + 1, ny + 1, nz),
        }
    )


def _boundary_mask(name, shape, start, stop):
    linear = np.arange(start, stop, dtype=np.intp)
    plane = shape[1] * shape[2]
    i, remainder = np.divmod(linear, plane)
    j, k = np.divmod(remainder, shape[2])
    if name == "Ex":
        return (j == shape[1] - 1) | (k == shape[2] - 1)
    if name == "Ey":
        return (k == shape[2] - 1) | (i == shape[0] - 1)
    if name == "Ez":
        return (i == shape[0] - 1) | (j == shape[1] - 1)
    if name == "Hx":
        return (j == 0) | (k == 0)
    if name == "Hy":
        return (k == 0) | (i == 0)
    return (i == 0) | (j == 0)


def _lower_component(geom_tree, space, name, shape, material_tile_size=65536):
    """Lower bounded GeometryMap tiles to dense dielectric coefficients and IDs."""
    component = _COMPONENT_TYPES[name]
    axes = space.component_coordinate_axes(component, shape)
    total = int(np.prod(shape))
    inverse = np.empty(total, dtype=np.float64)
    material_ids = np.empty(total, dtype=np.int32)
    coefficient = "eps_inf" if name in _ELECTRIC else "mu_inf"
    for start in range(0, total, material_tile_size):
        stop = min(start + material_tile_size, total)
        lowered = geom_tree.lower_grid(*axes, start, stop, component=component)
        tile = np.empty(stop - start, dtype=np.float64)
        for material_id in np.unique(lowered.material_ids):
            material = lowered.geometries[int(material_id)].material
            if type(material) is not Dielectric:
                raise NotImplementedError(
                    "the Torch minimal slice supports only simple Dielectric materials; "
                    f"{type(material).__name__} was mapped for {name}"
                )
            value = float(getattr(material, coefficient))
            if not np.isfinite(value) or value <= 0:
                raise TorchConfigurationError(
                    f"{coefficient} must be finite and positive for {name}"
                )
            tile[lowered.material_ids == material_id] = 1.0 / value
        tile[_boundary_mask(name, shape, start, stop)] = 0.0
        inverse[start:stop] = tile
        material_ids[start:stop] = lowered.material_ids
    return inverse.reshape(shape), material_ids.reshape(shape)


class TorchSimulationPlan(nn.Module):
    """Fixed-shape non-trainable buffers lowered from GeometryMap tiles."""

    def __init__(self, coefficients, material_ids, *, dr, dt, bloch, device, dtype):
        super().__init__()
        self.shapes = MappingProxyType(
            {name: tuple(coefficients[name].shape) for name in _COMPONENTS}
        )
        self.dr = tuple(float(value) for value in dr)
        self.dt = float(dt)
        self.bloch = None if bloch is None else tuple(float(value) for value in bloch)
        for name in _COMPONENTS:
            family = "inv_eps" if name in _ELECTRIC else "inv_mu"
            coefficient = torch.as_tensor(
                coefficients[name], dtype=dtype, device=device
            ).contiguous()
            if self.bloch is not None:
                coefficient = coefficient.unsqueeze(-1)
            ids = torch.as_tensor(
                material_ids[name], dtype=torch.int32, device=device
            ).contiguous()
            self.register_buffer(f"{family}_{name.lower()}", coefficient)
            self.register_buffer(f"material_ids_{name.lower()}", ids)
        self._sealed_names = frozenset(self._buffers) | {"shapes", "dr", "dt", "bloch"}

    def __setattr__(self, name, value):
        sealed = self.__dict__.get("_sealed_names", ())
        if name in sealed:
            raise AttributeError(f"Torch simulation plan buffer {name!r} is immutable")
        super().__setattr__(name, value)


class TorchSimulationState(nn.Module):
    """Device-resident fields, clocks, coefficients, and fixed scratch storage."""

    def __init__(self, plan, *, paired_real, device, dtype):
        super().__init__()
        self.plan = plan
        self.paired_real = bool(paired_real)
        plane = (2,) if paired_real else ()
        ex_shape = plan.shapes["Ex"]
        scratch_shape = (ex_shape[0], ex_shape[1] - 1, ex_shape[2] - 1) + plane
        for name in _COMPONENTS:
            shape = plan.shapes[name] + plane
            self.register_buffer(
                name.lower(), torch.zeros(shape, device=device, dtype=dtype)
            )
            self.register_buffer(
                f"_scratch_{name.lower()}",
                torch.zeros(scratch_shape, device=device, dtype=dtype),
                persistent=False,
            )
        self.register_buffer(
            "step_count", torch.zeros((), device=device, dtype=torch.int64)
        )
        self.register_buffer("source_time", torch.zeros((), device=device, dtype=dtype))
        self.register_buffer(
            "time_step", torch.tensor(plan.dt, device=device, dtype=dtype)
        )
        self.requires_grad_(False)

    def field(self, component):
        """Return one live device field by component type or canonical name."""
        name = component if isinstance(component, str) else component.__name__
        if name not in _COMPONENTS:
            raise KeyError(f"unknown field component {name!r}")
        return getattr(self, name.lower())

    def fields(self):
        """Return a read-only name-to-live-buffer view."""
        return MappingProxyType(
            {name: getattr(self, name.lower()) for name in _COMPONENTS}
        )

    @torch.inference_mode()
    def checkpoint(self):
        """Extract an independent device-resident checkpoint."""
        return {
            name: value.detach().clone() for name, value in self.state_dict().items()
        }

    @torch.inference_mode()
    def load_checkpoint(self, checkpoint):
        """Restore a checkpoint without replacing any registered buffer."""
        expected = set(self.state_dict())
        if set(checkpoint) != expected:
            raise ValueError("checkpoint keys do not match the simulation state")
        for name, target in self.state_dict().items():
            value = checkpoint[name]
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError(
                    f"checkpoint tensor {name!r} has an incompatible shape or dtype"
                )
            target.copy_(value.to(device=target.device))

    @torch.inference_mode()
    def snapshot(self):
        """Clone all live fields on their current device."""
        return {name: value.detach().clone() for name, value in self.fields().items()}

    @torch.inference_mode()
    def host_snapshot(self, *, numpy=True, complex_fields=True):
        """Explicitly transfer cloned fields to host memory."""
        result = {}
        for name, field_value in self.fields().items():
            host = field_value.detach().to(device="cpu", copy=True)
            if self.paired_real and complex_fields:
                host = torch.complex(host[..., 0], host[..., 1])
            result[name] = host.numpy() if numpy else host
        return result


def _electric_phase(
    ex,
    ey,
    ez,
    hx,
    hy,
    hz,
    inv_ex,
    inv_ey,
    inv_ez,
    scratch_ex,
    scratch_ey,
    scratch_ez,
    dt,
    dx,
    dy,
    dz,
):
    ex_target = ex[:, :-1, :-1]
    scratch = scratch_ex
    torch.sub(hz[1:, 1:, :], hz[1:, :-1, :], out=scratch)
    scratch.mul_(1.0 / dy)
    ex_target.addcmul_(inv_ex[:, :-1, :-1], scratch, value=dt)
    torch.sub(hy[1:, :, 1:], hy[1:, :, :-1], out=scratch)
    scratch.mul_(-1.0 / dz)
    ex_target.addcmul_(inv_ex[:, :-1, :-1], scratch, value=dt)

    ey_target = ey[:-1, :, :-1]
    scratch = scratch_ey
    torch.sub(hx[:, 1:, 1:], hx[:, 1:, :-1], out=scratch)
    scratch.mul_(1.0 / dz)
    ey_target.addcmul_(inv_ey[:-1, :, :-1], scratch, value=dt)
    torch.sub(hz[1:, 1:, :], hz[:-1, 1:, :], out=scratch)
    scratch.mul_(-1.0 / dx)
    ey_target.addcmul_(inv_ey[:-1, :, :-1], scratch, value=dt)

    ez_target = ez[:-1, :-1, :]
    scratch = scratch_ez
    torch.sub(hy[1:, :, 1:], hy[:-1, :, 1:], out=scratch)
    scratch.mul_(1.0 / dx)
    ez_target.addcmul_(inv_ez[:-1, :-1, :], scratch, value=dt)
    torch.sub(hx[:, 1:, 1:], hx[:, :-1, 1:], out=scratch)
    scratch.mul_(-1.0 / dy)
    ez_target.addcmul_(inv_ez[:-1, :-1, :], scratch, value=dt)


def _magnetic_phase(
    ex,
    ey,
    ez,
    hx,
    hy,
    hz,
    inv_hx,
    inv_hy,
    inv_hz,
    scratch_hx,
    scratch_hy,
    scratch_hz,
    dt,
    dx,
    dy,
    dz,
):
    hx_target = hx[:, 1:, 1:]
    scratch = scratch_hx
    torch.sub(ey[:-1, :, 1:], ey[:-1, :, :-1], out=scratch)
    scratch.mul_(1.0 / dz)
    hx_target.addcmul_(inv_hx[:, 1:, 1:], scratch, value=dt)
    torch.sub(ez[:-1, 1:, :], ez[:-1, :-1, :], out=scratch)
    scratch.mul_(-1.0 / dy)
    hx_target.addcmul_(inv_hx[:, 1:, 1:], scratch, value=dt)

    hy_target = hy[1:, :, 1:]
    scratch = scratch_hy
    torch.sub(ez[1:, :-1, :], ez[:-1, :-1, :], out=scratch)
    scratch.mul_(1.0 / dx)
    hy_target.addcmul_(inv_hy[1:, :, 1:], scratch, value=dt)
    torch.sub(ex[:, :-1, 1:], ex[:, :-1, :-1], out=scratch)
    scratch.mul_(-1.0 / dz)
    hy_target.addcmul_(inv_hy[1:, :, 1:], scratch, value=dt)

    hz_target = hz[1:, 1:, :]
    scratch = scratch_hz
    torch.sub(ex[:, 1:, :-1], ex[:, :-1, :-1], out=scratch)
    scratch.mul_(1.0 / dy)
    hz_target.addcmul_(inv_hz[1:, 1:, :], scratch, value=dt)
    torch.sub(ey[1:, :, :-1], ey[:-1, :, :-1], out=scratch)
    scratch.mul_(-1.0 / dx)
    hz_target.addcmul_(inv_hz[1:, 1:, :], scratch, value=dt)


class TorchSimulation:
    """Breaking Torch-only construction and execution API.

    Geometry construction and host conversion are deliberately separate from
    the compiled electric and magnetic phases. Sources and distributed domain
    decomposition are not part of this minimal dielectric slice.
    """

    def __init__(
        self,
        *,
        space,
        geometry,
        runtime,
        courant_ratio=0.99,
        dt=None,
        bloch=None,
    ):
        if not isinstance(runtime, TorchRuntimeConfig):
            raise TypeError("runtime must be a TorchRuntimeConfig")
        runtime.validate_static()
        processors = os.cpu_count() or 1
        per_process_limit = max(1, processors // runtime.launch.local_world_size)
        threads = (
            per_process_limit if runtime.cpu_threads is None else runtime.cpu_threads
        )
        if threads * runtime.launch.local_world_size > processors:
            raise TorchConfigurationError(
                f"{threads} CPU threads across {runtime.launch.local_world_size} local "
                f"processes oversubscribe {processors} available CPUs"
            )
        torch.set_num_threads(threads)
        if runtime.launch.world_size != 1:
            raise TorchConfigurationError(
                "distributed execution is not implemented in this slice; use world_size=1"
            )
        if int(space.numprocs) != 1:
            raise TorchConfigurationError(
                "MPI-decomposed Cartesian spaces are unsupported by TorchSimulation"
            )
        device = _resolved_device(runtime)

        geometry = tuple(geometry)
        default_medium = next(
            (item for item in geometry if isinstance(item, DefaultMedium)), None
        )
        if default_medium is None:
            raise ValueError("geometry must contain a DefaultMedium")
        if type(default_medium.material) is not Dielectric:
            raise NotImplementedError(
                "the Torch minimal slice requires a simple Dielectric DefaultMedium"
            )
        eps_inf = float(default_medium.material.eps_inf)
        mu_inf = float(default_medium.material.mu_inf)
        dr = tuple(float(value) for value in space.dr)
        dt_limit = sqrt(eps_inf * mu_inf) / sqrt(sum(value**-2 for value in dr))
        time_step = float(courant_ratio) * dt_limit if dt is None else float(dt)
        if not np.isfinite(time_step) or time_step <= 0:
            raise TorchConfigurationError("time step must be finite and positive")
        if time_step > dt_limit * (1 + 1e-14):
            raise TorchConfigurationError(
                f"time step {time_step:g} exceeds the Courant limit {dt_limit:g}"
            )

        space.dt = time_step
        for geometric_object in geometry:
            geometric_object.init(space)
        geom_tree = GeomBoxTree(geometry)
        shapes = _field_shapes(space)
        coefficients = {}
        material_ids = {}
        for name in _COMPONENTS:
            coefficients[name], material_ids[name] = _lower_component(
                geom_tree, space, name, shapes[name]
            )

        with torch.inference_mode():
            plan = TorchSimulationPlan(
                coefficients,
                material_ids,
                dr=dr,
                dt=time_step,
                bloch=bloch,
                device=device,
                dtype=runtime.dtype,
            )
            state = TorchSimulationState(
                plan,
                paired_real=bloch is not None,
                device=device,
                dtype=runtime.dtype,
            )

        self.runtime = runtime
        self.cpu_threads = threads
        self.device = device
        self.dtype = runtime.dtype
        self.space = space
        self.geometry = geometry
        self.geom_tree = geom_tree
        self.plan = plan
        self.state = state
        self._electric = _electric_phase
        self._magnetic = _magnetic_phase
        if runtime.compile_policy == "compile":
            self._electric = torch.compile(
                _electric_phase, fullgraph=True, dynamic=False
            )
            self._magnetic = torch.compile(
                _magnetic_phase, fullgraph=True, dynamic=False
            )

    def _electric_arguments(self):
        state = self.state
        plan = self.plan
        return (
            state.ex,
            state.ey,
            state.ez,
            state.hx,
            state.hy,
            state.hz,
            plan.inv_eps_ex,
            plan.inv_eps_ey,
            plan.inv_eps_ez,
            state._scratch_ex,
            state._scratch_ey,
            state._scratch_ez,
            plan.dt,
            *plan.dr,
        )

    def _magnetic_arguments(self):
        state = self.state
        plan = self.plan
        return (
            state.ex,
            state.ey,
            state.ez,
            state.hx,
            state.hy,
            state.hz,
            plan.inv_mu_hx,
            plan.inv_mu_hy,
            plan.inv_mu_hz,
            state._scratch_hx,
            state._scratch_hy,
            state._scratch_hz,
            plan.dt,
            *plan.dr,
        )

    def _boundary_angle(self, name, axis, direction):
        if self.plan.bloch is None:
            return None
        length = (self.plan.shapes[name][axis] - 1) * self.plan.dr[axis]
        return direction * self.plan.bloch[axis] * length

    @staticmethod
    def _rotate_or_copy(destination, source, angle):
        if angle is None:
            destination.copy_(source)
            return
        cosine = float(np.cos(angle))
        sine = float(np.sin(angle))
        destination[..., 0].copy_(source[..., 0]).mul_(cosine)
        destination[..., 0].add_(source[..., 1], alpha=-sine)
        destination[..., 1].copy_(source[..., 0]).mul_(sine)
        destination[..., 1].add_(source[..., 1], alpha=cosine)

    def _sync_electric_boundaries(self):
        state = self.state
        self._rotate_or_copy(
            state.ex[:, -1, :],
            state.ex[:, 0, :],
            self._boundary_angle("Ex", 1, 1),
        )
        self._rotate_or_copy(
            state.ex[:, :, -1],
            state.ex[:, :, 0],
            self._boundary_angle("Ex", 2, 1),
        )
        self._rotate_or_copy(
            state.ey[:, :, -1],
            state.ey[:, :, 0],
            self._boundary_angle("Ey", 2, 1),
        )
        self._rotate_or_copy(
            state.ey[-1, :, :],
            state.ey[0, :, :],
            self._boundary_angle("Ey", 0, 1),
        )
        self._rotate_or_copy(
            state.ez[-1, :, :],
            state.ez[0, :, :],
            self._boundary_angle("Ez", 0, 1),
        )
        self._rotate_or_copy(
            state.ez[:, -1, :],
            state.ez[:, 0, :],
            self._boundary_angle("Ez", 1, 1),
        )

    def _sync_magnetic_boundaries(self):
        state = self.state
        self._rotate_or_copy(
            state.hx[:, 0, :],
            state.hx[:, -1, :],
            self._boundary_angle("Hx", 1, -1),
        )
        self._rotate_or_copy(
            state.hx[:, :, 0],
            state.hx[:, :, -1],
            self._boundary_angle("Hx", 2, -1),
        )
        self._rotate_or_copy(
            state.hy[:, :, 0],
            state.hy[:, :, -1],
            self._boundary_angle("Hy", 2, -1),
        )
        self._rotate_or_copy(
            state.hy[0, :, :],
            state.hy[-1, :, :],
            self._boundary_angle("Hy", 0, -1),
        )
        self._rotate_or_copy(
            state.hz[0, :, :],
            state.hz[-1, :, :],
            self._boundary_angle("Hz", 0, -1),
        )
        self._rotate_or_copy(
            state.hz[:, 0, :],
            state.hz[:, -1, :],
            self._boundary_angle("Hz", 1, -1),
        )

    @torch.inference_mode()
    def advance(self, steps):
        """Advance a fixed state in place without implicit host conversion."""
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        for _ in range(steps):
            self.state.source_time.add_(self.state.time_step, alpha=0.5)
            self._sync_magnetic_boundaries()
            self._electric(*self._electric_arguments())
            self.state.source_time.add_(self.state.time_step, alpha=0.5)
            self._sync_electric_boundaries()
            self._magnetic(*self._magnetic_arguments())
            self.state.step_count.add_(1)
        return self

    def step(self):
        """Advance one step as a convenience wrapper."""
        return self.advance(1)

    @torch.inference_mode()
    def load_host_fields(self, fields):
        """Explicitly copy NumPy or CPU Torch field values into live buffers."""
        if set(fields) != set(_COMPONENTS):
            raise ValueError("host fields must contain Ex, Ey, Ez, Hx, Hy, and Hz")
        for name, values in fields.items():
            target = self.state.field(name)
            source = torch.as_tensor(values)
            if self.state.paired_real:
                if source.is_complex():
                    source = torch.stack((source.real, source.imag), dim=-1)
                elif source.shape != target.shape:
                    raise ValueError(
                        f"paired-real field {name} requires complex values or shape "
                        f"{tuple(target.shape)}"
                    )
            if source.shape != target.shape:
                raise ValueError(
                    f"field {name} has shape {tuple(source.shape)}; "
                    f"expected {tuple(target.shape)}"
                )
            target.copy_(source.to(device=self.device, dtype=self.dtype))
        return self

    def diagnostics(self):
        """Return the focused runtime diagnostic record for this simulation."""
        result = torch_runtime_diagnostics(self.runtime)
        result["resolved_device"] = str(self.device)
        result["cpu_threads"] = self.cpu_threads
        return result

    def buffer_addresses(self):
        """Return fixed storage addresses for explicit capture diagnostics."""
        return {name: tensor.data_ptr() for name, tensor in self.state.named_buffers()}


__all__ = [
    "DistributedLaunch",
    "TorchConfigurationError",
    "TorchRuntimeConfig",
    "TorchSimulation",
    "TorchSimulationPlan",
    "TorchSimulationState",
    "torch_runtime_diagnostics",
]
