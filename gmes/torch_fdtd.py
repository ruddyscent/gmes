"""Torch-native state and execution for the supported FDTD material slice."""

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

from .geometry import DefaultMedium, GeomBoxTree
from .material import Dielectric
from .torch_dispersive import (
    DISPERSIVE_MODELS,
    register_plan_buffers,
    register_state_buffers,
    update_bucket,
)
from .torch_plan import (
    COMPONENTS,
    ELECTRIC_COMPONENTS,
    EXECUTION_POLICIES,
    ComponentPlan,
    ExecutionSignature,
    FlattenedStencilTerm,
    MaterialBucketPlan,
    TorchExecutionPlanner,
)


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
    execution_policy: str = "auto"
    planner_tile_size: int = 4096

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
        if self.execution_policy not in EXECUTION_POLICIES:
            raise TorchConfigurationError(
                "execution_policy must be 'auto', 'dense', 'compact', or 'tiled'"
            )
        if self.planner_tile_size < 1:
            raise TorchConfigurationError("planner_tile_size must be positive")
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
        "execution_policy": config.execution_policy,
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


class TorchSimulationPlan(nn.Module):
    """Finalized immutable tensors and metadata for mixed-material execution."""

    def __init__(self, component_plans, *, dr, dt, bloch, device, dtype):
        super().__init__()
        plans = {component.name: component for component in component_plans}
        if set(plans) != set(COMPONENTS):
            raise ValueError("the execution plan must contain all six Yee components")
        self.components = MappingProxyType(plans)
        self.shapes = MappingProxyType(
            {name: tuple(plans[name].shape) for name in COMPONENTS}
        )
        self.dr = tuple(float(value) for value in dr)
        self.dt = float(dt)
        self.bloch = None if bloch is None else tuple(float(value) for value in bloch)
        dispersive_buckets = []
        for name in COMPONENTS:
            component = plans[name]
            family = "inv_eps" if name in ELECTRIC_COMPONENTS else "inv_mu"
            coefficient = torch.tensor(
                component.dense_inverse, dtype=dtype, device=device
            ).contiguous()
            if self.bloch is not None:
                coefficient = coefficient.unsqueeze(-1)
            self.register_buffer(f"{family}_{name.lower()}", coefficient)
            self.register_buffer(
                f"material_ids_{name.lower()}",
                torch.tensor(
                    component.material_ids, dtype=torch.int32, device=device
                ).contiguous(),
            )
            self.register_buffer(
                f"underlying_ids_{name.lower()}",
                torch.tensor(
                    component.underlying_ids, dtype=torch.int32, device=device
                ).contiguous(),
            )
            self.register_buffer(
                f"ownership_{name.lower()}",
                torch.tensor(
                    component.ownership, dtype=torch.int16, device=device
                ).contiguous(),
            )
            constant_values = component.constant_values
            if self.bloch is None:
                if np.any(constant_values[:, 1]):
                    raise TorchConfigurationError(
                        "a complex Const value requires paired-real Bloch fields"
                    )
                constant_values = constant_values[:, 0]
            self.register_buffer(
                f"constant_targets_{name.lower()}",
                torch.tensor(
                    component.constant_targets, dtype=torch.int64, device=device
                ).contiguous(),
            )
            self.register_buffer(
                f"constant_values_{name.lower()}",
                torch.tensor(constant_values, dtype=dtype, device=device).contiguous(),
            )
            for index, bucket in enumerate(component.buckets):
                prefix = f"bucket_{name.lower()}_{index}"
                arrays = [
                    ("region_keys", bucket.region_keys, torch.int32),
                    (
                        "region_coefficient_indices",
                        bucket.region_coefficient_indices,
                        torch.int32,
                    ),
                    ("coefficients", bucket.coefficient_table, dtype),
                ]
                if bucket.signature.model in {"upml", "cpml"}:
                    cell_coefficients = bucket.cell_coefficients
                    if self.bloch is not None:
                        cell_coefficients = cell_coefficients[..., np.newaxis]
                    arrays.extend(
                        (
                            ("targets", bucket.targets, torch.int64),
                            ("stencil_indices", bucket.stencil_indices, torch.int64),
                            ("cell_coefficients", cell_coefficients, dtype),
                        )
                    )
                elif bucket.selected_policy == "compact":
                    arrays.extend(
                        (
                            ("targets", bucket.targets, torch.int64),
                            (
                                "target_region_indices",
                                bucket.target_region_indices,
                                torch.int32,
                            ),
                        )
                    )
                elif bucket.selected_policy == "tiled":
                    arrays.extend(
                        (
                            ("tile_origins", bucket.tile_origins, torch.int64),
                            (
                                "tile_region_indices",
                                bucket.tile_region_indices,
                                torch.int32,
                            ),
                        )
                    )
                for suffix, values, tensor_dtype in arrays:
                    self.register_buffer(
                        f"{prefix}_{suffix}",
                        torch.tensor(
                            values, dtype=tensor_dtype, device=device
                        ).contiguous(),
                    )
                descriptor = register_plan_buffers(
                    self,
                    bucket,
                    component,
                    prefix,
                    dtype=dtype,
                    device=device,
                )
                if descriptor is not None:
                    dispersive_buckets.append(descriptor)
        self.dispersive_buckets = tuple(dispersive_buckets)
        self._sealed_names = frozenset(self._buffers) | {
            "components",
            "shapes",
            "dr",
            "dt",
            "bloch",
            "dispersive_buckets",
        }

    def __setattr__(self, name, value):
        sealed = self.__dict__.get("_sealed_names", ())
        if name in sealed:
            raise AttributeError(f"Torch simulation plan buffer {name!r} is immutable")
        super().__setattr__(name, value)

    @property
    def unsupported_models(self):
        """Return planned models without a Torch execution implementation."""
        supported = {"dielectric", "const", "dummy", "upml", "cpml"} | DISPERSIVE_MODELS
        return tuple(
            sorted(
                {
                    bucket.signature.model
                    for component in self.components.values()
                    for bucket in component.buckets
                    if bucket.signature.model not in supported
                }
            )
        )

    def decision_report(self):
        """Return policy choices and their occupancy/cost evidence."""
        return tuple(self.components[name].decision_record() for name in COMPONENTS)


class TorchSimulationState(nn.Module):
    """Device-resident fields, clocks, coefficients, and fixed scratch storage."""

    def __init__(self, plan, *, paired_real, device, dtype):
        super().__init__()
        self.plan = plan
        self.paired_real = bool(paired_real)
        plane = (2,) if paired_real else ()
        ex_shape = plan.shapes["Ex"]
        scratch_shape = (ex_shape[0], ex_shape[1] - 1, ex_shape[2] - 1) + plane
        for name in COMPONENTS:
            shape = plan.shapes[name] + plane
            self.register_buffer(
                name.lower(), torch.zeros(shape, device=device, dtype=dtype)
            )
            self.register_buffer(
                f"_scratch_{name.lower()}",
                torch.zeros(scratch_shape, device=device, dtype=dtype),
                persistent=False,
            )
        register_state_buffers(
            self,
            plan.dispersive_buckets,
            paired_real=paired_real,
            dtype=dtype,
            device=device,
        )
        self.register_buffer(
            "step_count", torch.zeros((), device=device, dtype=torch.int64)
        )
        self.register_buffer("source_time", torch.zeros((), device=device, dtype=dtype))
        self.register_buffer(
            "time_step", torch.tensor(plan.dt, device=device, dtype=dtype)
        )
        for component_name, component in plan.components.items():
            for index, bucket in enumerate(component.buckets):
                if bucket.signature.model not in {"upml", "cpml"}:
                    continue
                prefix = f"pml_{component_name.lower()}_{index}"
                state_shape = (bucket.target_count, bucket.state_width) + plane
                self.register_buffer(
                    f"{prefix}_state",
                    torch.zeros(state_shape, device=device, dtype=dtype),
                )
                scratch_shape = (bucket.target_count,) + plane
                for scratch_index in range(3):
                    self.register_buffer(
                        f"_{prefix}_scratch{scratch_index}",
                        torch.zeros(scratch_shape, device=device, dtype=dtype),
                        persistent=False,
                    )
        self.requires_grad_(False)

    def field(self, component):
        """Return one live device field by component type or canonical name."""
        name = component if isinstance(component, str) else component.__name__
        if name not in COMPONENTS:
            raise KeyError(f"unknown field component {name!r}")
        return getattr(self, name.lower())

    def fields(self):
        """Return a read-only name-to-live-buffer view."""
        return MappingProxyType(
            {name: getattr(self, name.lower()) for name in COMPONENTS}
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
    def pml_state_snapshot(self, *, numpy=True, complex_state=True):
        """Explicitly extract active-cell PML state for oracle comparison."""
        result = {}
        for name, value in self.named_buffers(recurse=False):
            if not name.startswith("pml_") or not name.endswith("_state"):
                continue
            host = value.detach().to(device="cpu", copy=True)
            if self.paired_real and complex_state:
                host = torch.complex(host[..., 0], host[..., 1])
            result[name] = host.numpy() if numpy else host
        return result

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


def _electric_phase_2d_z(
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
    del scratch_ex, scratch_ey, scratch_ez, dz
    ex[:, :-1, :-1].add_(
        inv_ex[:, :-1, :-1] * (hz[1:, 1:, :] - hz[1:, :-1, :]),
        alpha=dt / dy,
    )
    ey[:-1, :, :-1].add_(
        inv_ey[:-1, :, :-1] * (hz[1:, 1:, :] - hz[:-1, 1:, :]),
        alpha=-dt / dx,
    )
    ez[:-1, :-1, :].add_(
        inv_ez[:-1, :-1, :]
        * (
            (hy[1:, :, 1:] - hy[:-1, :, 1:]) / dx
            - (hx[:, 1:, 1:] - hx[:, :-1, 1:]) / dy
        ),
        alpha=dt,
    )


def _magnetic_phase_2d_z(
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
    del scratch_hx, scratch_hy, scratch_hz, dz
    hx[:, 1:, 1:].add_(
        inv_hx[:, 1:, 1:] * (ez[:-1, 1:, :] - ez[:-1, :-1, :]),
        alpha=-dt / dy,
    )
    hy[1:, :, 1:].add_(
        inv_hy[1:, :, 1:] * (ez[1:, :-1, :] - ez[:-1, :-1, :]),
        alpha=dt / dx,
    )
    hz[1:, 1:, :].add_(
        inv_hz[1:, 1:, :]
        * (
            (ex[:, 1:, :-1] - ex[:, :-1, :-1]) / dy
            - (ey[1:, :, :-1] - ey[:-1, :, :-1]) / dx
        ),
        alpha=dt,
    )


def _electric_periodic_phase(*arguments):
    ex, ey, ez, hx, hy, hz = arguments[:6]
    hx[:, 0, :].copy_(hx[:, -1, :])
    hx[:, :, 0].copy_(hx[:, :, -1])
    hy[:, :, 0].copy_(hy[:, :, -1])
    hy[0, :, :].copy_(hy[-1, :, :])
    hz[0, :, :].copy_(hz[-1, :, :])
    hz[:, 0, :].copy_(hz[:, -1, :])
    _electric_phase(*arguments)


def _magnetic_periodic_phase(*arguments):
    ex, ey, ez = arguments[:3]
    ex[:, -1, :].copy_(ex[:, 0, :])
    ex[:, :, -1].copy_(ex[:, :, 0])
    ey[:, :, -1].copy_(ey[:, :, 0])
    ey[-1, :, :].copy_(ey[0, :, :])
    ez[-1, :, :].copy_(ez[0, :, :])
    ez[:, -1, :].copy_(ez[:, 0, :])
    _magnetic_phase(*arguments)


def _electric_periodic_phase_2d_z(*arguments):
    ex, ey, ez, hx, hy, hz = arguments[:6]
    hx[:, 0, :].copy_(hx[:, -1, :])
    hy[0, :, :].copy_(hy[-1, :, :])
    hz[0, :, :].copy_(hz[-1, :, :])
    hz[:, 0, :].copy_(hz[:, -1, :])
    _electric_phase_2d_z(*arguments)


def _magnetic_periodic_phase_2d_z(*arguments):
    ex, ey, ez = arguments[:3]
    ex[:, -1, :].copy_(ex[:, 0, :])
    ey[-1, :, :].copy_(ey[0, :, :])
    ez[-1, :, :].copy_(ez[0, :, :])
    ez[:, -1, :].copy_(ez[:, 0, :])
    _magnetic_phase_2d_z(*arguments)


def _gather_difference(source, positive, negative, output, temporary, scale):
    torch.index_select(source, 0, positive, out=output)
    torch.index_select(source, 0, negative, out=temporary)
    output.sub_(temporary).mul_(scale)


def _upml_bucket_update(
    field,
    source1,
    source2,
    targets,
    stencil,
    coefficients,
    state,
    scratch0,
    scratch1,
    scratch2,
    scale1,
    scale2,
    direction,
):
    _gather_difference(
        source1, stencil[:, 0], stencil[:, 1], scratch0, scratch2, scale1
    )
    _gather_difference(
        source2, stencil[:, 2], stencil[:, 3], scratch1, scratch2, scale2
    )
    scratch0.sub_(scratch1).mul_(direction)
    memory = state[:, 0]
    scratch1.copy_(memory)
    memory.mul_(coefficients[:, 1]).addcmul_(coefficients[:, 2], scratch0)
    torch.index_select(field, 0, targets, out=scratch2)
    scratch0.copy_(memory).mul_(coefficients[:, 5])
    scratch0.addcmul_(coefficients[:, 6], scratch1, value=-1.0)
    scratch0.mul_(coefficients[:, 4]).mul_(coefficients[:, 0])
    scratch2.mul_(coefficients[:, 3]).add_(scratch0)
    field.index_copy_(0, targets, scratch2)


def _cpml_bucket_update(
    field,
    source1,
    source2,
    targets,
    stencil,
    coefficients,
    state,
    scratch0,
    scratch1,
    scratch2,
    scale1,
    scale2,
    direction,
    dt,
):
    _gather_difference(
        source1, stencil[:, 0], stencil[:, 1], scratch0, scratch2, scale1
    )
    _gather_difference(
        source2, stencil[:, 2], stencil[:, 3], scratch1, scratch2, scale2
    )
    psi1 = state[:, 0]
    psi2 = state[:, 1]
    psi1.mul_(coefficients[:, 1]).addcmul_(coefficients[:, 2], scratch0)
    psi2.mul_(coefficients[:, 4]).addcmul_(coefficients[:, 5], scratch1)
    torch.index_select(field, 0, targets, out=scratch2)
    scratch0.div_(coefficients[:, 3]).add_(psi1)
    scratch1.div_(coefficients[:, 6]).add_(psi2)
    scratch0.sub_(scratch1).mul_(coefficients[:, 0]).mul_(dt * direction)
    scratch2.add_(scratch0)
    field.index_copy_(0, targets, scratch2)


class TorchSimulation:
    """Breaking Torch-only construction and execution API.

    Geometry construction and host conversion are deliberately separate from
    the electric, magnetic, PML, and dispersive phases. Sources and distributed
    domain decomposition are not part of this single-device material slice.
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
                "TorchSimulation requires a simple Dielectric DefaultMedium"
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
        planner = TorchExecutionPlanner(
            geom_tree=geom_tree,
            space=space,
            shapes=shapes,
            precision=runtime.precision,
            device_type=device.type,
            policy=runtime.execution_policy,
            execution_tile_size=runtime.planner_tile_size,
        )
        component_plans = planner.build()
        unsupported_models = sorted(
            {
                bucket.signature.model
                for component in component_plans
                for bucket in component.buckets
                if bucket.signature.model
                not in (
                    {"dielectric", "const", "dummy", "upml", "cpml"} | DISPERSIVE_MODELS
                )
            }
        )
        if unsupported_models:
            raise NotImplementedError(
                "the mixed-material planner lowered these models, but their state "
                "equations belong to follow-up issue #120: "
                + ", ".join(unsupported_models)
            )

        with torch.inference_mode():
            plan = TorchSimulationPlan(
                component_plans,
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
        self._dispersive_buckets = plan.dispersive_buckets
        self.cpu_threads = threads
        self.device = device
        self.dtype = runtime.dtype
        self.space = space
        self.geometry = geometry
        self.geom_tree = geom_tree
        self.plan = plan
        self.state = state
        self._phase_includes_boundaries = bloch is None
        self._has_electric_constants = any(
            getattr(plan, f"constant_targets_{name.lower()}").numel()
            for name in ("Ex", "Ey", "Ez")
        )
        self._has_magnetic_constants = any(
            getattr(plan, f"constant_targets_{name.lower()}").numel()
            for name in ("Hx", "Hy", "Hz")
        )
        self._has_pml = any(
            bucket.signature.model in {"upml", "cpml"}
            for component in component_plans
            for bucket in component.buckets
        )
        z_collapsed = (
            shapes["Ez"][2] == 1
            and bloch is None
            and runtime.compile_policy == "compile"
            and not self._has_pml
        )
        if z_collapsed:
            electric_function = _electric_periodic_phase_2d_z
            magnetic_function = _magnetic_periodic_phase_2d_z
        else:
            electric_function = (
                _electric_periodic_phase
                if self._phase_includes_boundaries
                else _electric_phase
            )
            magnetic_function = (
                _magnetic_periodic_phase
                if self._phase_includes_boundaries
                else _magnetic_phase
            )
        self._electric = electric_function
        self._magnetic = magnetic_function
        if runtime.compile_policy == "compile":
            compile_options = {"cpp_wrapper": True} if device.type == "cpu" else None
            self._electric = torch.compile(
                electric_function,
                fullgraph=True,
                dynamic=False,
                options=compile_options,
            )
            self._magnetic = torch.compile(
                magnetic_function,
                fullgraph=True,
                dynamic=False,
                options=compile_options,
            )
        self._electric_args = self._electric_arguments()
        self._magnetic_args = self._magnetic_arguments()
        pml_functions = {
            "upml": _upml_bucket_update,
            "cpml": _cpml_bucket_update,
        }
        if runtime.compile_policy == "compile":
            pml_functions = {
                model: torch.compile(
                    function,
                    fullgraph=True,
                    dynamic=False,
                    options=compile_options,
                )
                for model, function in pml_functions.items()
            }
        self._electric_pml = self._pml_executions(("Ex", "Ey", "Ez"), pml_functions)
        self._magnetic_pml = self._pml_executions(("Hx", "Hy", "Hz"), pml_functions)
        self._defer_collapsed_ghosts = z_collapsed

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

    def _pml_executions(self, names, functions):
        executions = []
        paired_width = 2 if self.state.paired_real else None
        for name in names:
            component = self.plan.components[name]
            for index, bucket in enumerate(component.buckets):
                model = bucket.signature.model
                if model not in functions:
                    continue
                prefix = f"bucket_{name.lower()}_{index}"
                state_prefix = f"pml_{name.lower()}_{index}"
                flatten = (
                    (lambda value: value.reshape(-1, paired_width))
                    if paired_width is not None
                    else (lambda value: value.reshape(-1))
                )
                terms = component.stencil
                direction = 1.0
                if name not in ELECTRIC_COMPONENTS:
                    terms = tuple(reversed(terms))
                    direction = -1.0
                term1, term2 = terms
                arguments = [
                    flatten(self.state.field(name)),
                    flatten(self.state.field(term1.source)),
                    flatten(self.state.field(term2.source)),
                    getattr(self.plan, f"{prefix}_targets"),
                    getattr(self.plan, f"{prefix}_stencil_indices"),
                    getattr(self.plan, f"{prefix}_cell_coefficients"),
                    getattr(self.state, f"{state_prefix}_state"),
                    getattr(self.state, f"_{state_prefix}_scratch0"),
                    getattr(self.state, f"_{state_prefix}_scratch1"),
                    getattr(self.state, f"_{state_prefix}_scratch2"),
                    1.0 / self.plan.dr[term1.scale_axis],
                    1.0 / self.plan.dr[term2.scale_axis],
                    direction,
                ]
                if model == "cpml":
                    arguments.append(self.plan.dt)
                executions.append((functions[model], tuple(arguments)))
        return tuple(executions)

    @staticmethod
    def _run_pml(executions):
        for function, arguments in executions:
            function(*arguments)

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

    def _apply_constants(self, names):
        for name in names:
            targets = getattr(self.plan, f"constant_targets_{name.lower()}")
            if targets.numel() == 0:
                continue
            values = getattr(self.plan, f"constant_values_{name.lower()}")
            field = self.state.field(name)
            if self.state.paired_real:
                field.reshape(-1, 2).index_copy_(0, targets, values)
            else:
                field.reshape(-1).index_copy_(0, targets, values)

    def _apply_dispersive(self):
        for descriptor in self._dispersive_buckets:
            update_bucket(self.plan, self.state, descriptor)

    @torch.inference_mode()
    def advance(self, steps):
        """Advance a fixed state in place without implicit host conversion."""
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        for step_index in range(steps):
            if self._defer_collapsed_ghosts and step_index == steps - 1:
                self.state.hx[:, :, 0].copy_(self.state.hx[:, :, -1])
                self.state.hy[:, :, 0].copy_(self.state.hy[:, :, -1])
            if not self._phase_includes_boundaries:
                self._sync_magnetic_boundaries()
            self._electric(*self._electric_args)
            self._run_pml(self._electric_pml)
            if self._dispersive_buckets:
                self._apply_dispersive()
            if self._has_electric_constants:
                self._apply_constants(("Ex", "Ey", "Ez"))
            if not self._phase_includes_boundaries:
                self._sync_electric_boundaries()
            self._magnetic(*self._magnetic_args)
            self._run_pml(self._magnetic_pml)
            if self._has_magnetic_constants:
                self._apply_constants(("Hx", "Hy", "Hz"))
        if steps:
            if self._defer_collapsed_ghosts:
                self.state.ex[:, :, -1].copy_(self.state.ex[:, :, 0])
                self.state.ey[:, :, -1].copy_(self.state.ey[:, :, 0])
            self.state.source_time.add_(self.state.time_step, alpha=steps)
            self.state.step_count.add_(steps)
        return self

    def step(self):
        """Advance one step as a convenience wrapper."""
        return self.advance(1)

    @torch.inference_mode()
    def load_host_fields(self, fields):
        """Explicitly copy NumPy or CPU Torch field values into live buffers."""
        if set(fields) != set(COMPONENTS):
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
        result["material_plan"] = self.plan.decision_report()
        result["pml"] = {
            "active_cells": sum(
                bucket.target_count
                for component in self.plan.components.values()
                for bucket in component.buckets
                if bucket.signature.model in {"upml", "cpml"}
            ),
            "state_bytes": sum(
                value.numel() * value.element_size()
                for name, value in self.state.named_buffers(recurse=False)
                if name.startswith("pml_") and name.endswith("_state")
            ),
            "launches_per_step": len(self._electric_pml) + len(self._magnetic_pml),
        }
        result["dispersive"] = {
            "models": tuple(sorted({item.model for item in self._dispersive_buckets})),
            "active_cells": sum(item.target_count for item in self._dispersive_buckets),
            "state_bytes": sum(
                value.numel() * value.element_size()
                for name, value in self.state.state_dict().items()
                if name.startswith("bucket_")
            ),
            "launches_per_step": len(self._dispersive_buckets),
            "state_width_policy": "exact",
            "padding_elements": 0,
            "padding_elements_avoided": sum(
                bucket.padding_elements_avoided
                for component in self.plan.components.values()
                for bucket in component.buckets
                if bucket.signature.model in DISPERSIVE_MODELS
            ),
        }
        return result

    def buffer_addresses(self):
        """Return fixed storage addresses for explicit capture diagnostics."""
        return {name: tensor.data_ptr() for name, tensor in self.state.named_buffers()}


__all__ = [
    "ComponentPlan",
    "DistributedLaunch",
    "ExecutionSignature",
    "FlattenedStencilTerm",
    "MaterialBucketPlan",
    "TorchConfigurationError",
    "TorchExecutionPlanner",
    "TorchRuntimeConfig",
    "TorchSimulation",
    "TorchSimulationPlan",
    "TorchSimulationState",
    "torch_runtime_diagnostics",
]
