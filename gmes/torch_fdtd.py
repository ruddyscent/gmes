"""Torch-native state and execution for the supported FDTD material slice."""

import hashlib
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
from .torch_dm2 import (
    DM2_ITERATIONS_PER_CHUNK,
    DM2_MAX_ITERATIONS,
    Dm2BucketMetadata,
    TorchDm2BucketState,
)
from .torch_output import (
    TorchProbeBuffer,
    TorchProbeSamples,
    TorchProbeSpec,
    TorchProbeSpectrum,
    probe_spectrum,
    read_torch_checkpoint,
    write_probe_text,
    write_torch_checkpoint,
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
from .torch_source import (
    TorchPointSourceRecord,
    TorchSourceLoweringContext,
    lower_sources,
)


class TorchConfigurationError(ValueError):
    """A requested Torch execution configuration cannot be honored."""


COMPILE_MODES = ("default", "reduce-overhead", "max-autotune")
TORCH_SOLVER_ABI = "torch-fdtd-regions-v5"
DIRECT_VIEW_MUTATION_REPRESENTATION = "direct-nonoverlapping-as-strided-v1"
DEFAULT_VIEW_MUTATION_REPRESENTATION = "slice-views-v1"
PACKED_DM2_REPRESENTATION = "single-carry-packed-loop-v1"
FUNCTIONAL_DM2_REPRESENTATION = "functional-multi-carry-loop-v1"
FUSED_SOURCE_REPRESENTATION = "fused-half-step-v1"
EXTERNAL_SOURCE_REPRESENTATION = "external-v1"


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
    compile_mode: str = "default"
    cpu_threads: int | None = None
    cpu_interop_threads: int | None = None
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
        if self.compile_mode not in COMPILE_MODES:
            raise TorchConfigurationError(
                "compile_mode must be 'default', 'reduce-overhead', or 'max-autotune'"
            )
        if self.compile_policy == "eager" and self.compile_mode != "default":
            raise TorchConfigurationError(
                "compile_mode is meaningful only when compile_policy='compile'"
            )
        if self.device.startswith("cpu") and self.compile_mode != "default":
            raise TorchConfigurationError(
                "non-default compile modes are supported only on CUDA"
            )
        if self.cpu_threads is not None and self.cpu_threads < 1:
            raise TorchConfigurationError("cpu_threads must be positive")
        if self.cpu_interop_threads is not None and self.cpu_interop_threads < 1:
            raise TorchConfigurationError("cpu_interop_threads must be positive")
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
        "compile_mode": config.compile_mode,
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


def _compile_fullgraph(function, runtime, device, *, dynamic):
    """Compile one fixed solver region under the explicit runtime policy."""
    arguments = {"fullgraph": True, "dynamic": dynamic}
    if device.type == "cuda":
        arguments["mode"] = runtime.compile_mode
    return torch.compile(function, **arguments)


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
        dm2_buckets = []
        dm2_status_offset = 0
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
                elif (
                    bucket.selected_policy == "compact"
                    or bucket.signature.model == "dm2"
                ):
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
                if bucket.selected_policy == "tiled":
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
                if bucket.signature.model == "dm2":
                    target_coefficients = bucket.region_coefficient_indices[
                        bucket.target_region_indices
                    ]
                    target_rows = bucket.coefficient_table[target_coefficients]
                    columns = {
                        column_name: target_rows[:, column]
                        for column, column_name in enumerate(bucket.coefficient_names)
                    }
                    transition_count = bucket.signature.state_shape[0]
                    if transition_count:
                        omega = np.column_stack(
                            [
                                columns[f"transition{transition}_omega"]
                                for transition in range(transition_count)
                            ]
                        )
                        n_atom = np.column_stack(
                            [
                                columns[f"transition{transition}_density"]
                                for transition in range(transition_count)
                            ]
                        )
                    else:
                        omega = np.empty((bucket.target_count, 0), dtype=np.float64)
                        n_atom = np.empty((bucket.target_count, 0), dtype=np.float64)
                    for coefficient_name in (
                        "rho30",
                        "gamma",
                        "t1",
                        "t2",
                        "hbar",
                        "rtol",
                    ):
                        self.register_buffer(
                            f"{prefix}_{coefficient_name}",
                            torch.tensor(
                                columns[coefficient_name],
                                dtype=dtype,
                                device=device,
                            ).contiguous(),
                        )
                    self.register_buffer(
                        f"{prefix}_omega",
                        torch.tensor(omega, dtype=dtype, device=device).contiguous(),
                    )
                    self.register_buffer(
                        f"{prefix}_n_atom",
                        torch.tensor(n_atom, dtype=dtype, device=device).contiguous(),
                    )
                    term = component.stencil[1]
                    coordinates = np.unravel_index(bucket.targets, component.shape)
                    source_bases = sum(
                        coordinate * stride
                        for coordinate, stride in zip(coordinates, term.source_strides)
                    )
                    source_positive_indices = (
                        source_bases + term.positive_offset
                    ).astype(np.int64, copy=False)
                    source_negative_indices = (
                        source_bases + term.negative_offset
                    ).astype(np.int64, copy=False)
                    source_size = int(np.prod(term.source_shape))
                    if (
                        np.any(source_positive_indices < 0)
                        or np.any(source_positive_indices >= source_size)
                        or np.any(source_negative_indices < 0)
                        or np.any(source_negative_indices >= source_size)
                    ):
                        raise ValueError("DM2 source stencil is outside its component")
                    self.register_buffer(
                        f"{prefix}_source_positive_indices",
                        torch.tensor(
                            source_positive_indices,
                            dtype=torch.int64,
                            device=device,
                        ).contiguous(),
                    )
                    self.register_buffer(
                        f"{prefix}_source_negative_indices",
                        torch.tensor(
                            source_negative_indices,
                            dtype=torch.int64,
                            device=device,
                        ).contiguous(),
                    )
                    self.register_buffer(
                        f"{prefix}_curl_scale",
                        torch.full(
                            (bucket.target_count,),
                            (
                                0.0
                                if (
                                    self.shapes["Ex"][0],
                                    self.shapes["Ey"][1],
                                    self.shapes["Ez"][2],
                                )[term.scale_axis]
                                == 1
                                else term.sign * self.dt / self.dr[term.scale_axis]
                            ),
                            dtype=dtype,
                            device=device,
                        ),
                    )
                    dm2_buckets.append(
                        Dm2BucketMetadata(
                            component=name,
                            bucket_index=index,
                            transition_count=transition_count,
                            target_count=bucket.target_count,
                            prefix=prefix,
                            source_component=term.source,
                            status_offset=dm2_status_offset,
                        )
                    )
                    dm2_status_offset += bucket.target_count
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
        self.dm2_buckets = tuple(dm2_buckets)
        self.dm2_target_count = dm2_status_offset
        self.dispersive_buckets = tuple(dispersive_buckets)
        self._sealed_names = frozenset(self._buffers) | {
            "components",
            "shapes",
            "dr",
            "dt",
            "bloch",
            "dm2_buckets",
            "dm2_target_count",
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
        supported = {
            "dielectric",
            "const",
            "dm2",
            "dummy",
            "upml",
            "cpml",
        } | DISPERSIVE_MODELS
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
        if paired_real:
            boundary_axes = {
                "Ex": (1, 2),
                "Ey": (2, 0),
                "Ez": (0, 1),
                "Hx": (1, 2),
                "Hy": (2, 0),
                "Hz": (0, 1),
            }
            for name, axes in boundary_axes.items():
                for axis in axes:
                    shape = list(plan.shapes[name])
                    del shape[axis]
                    self.register_buffer(
                        f"_boundary_{name.lower()}_{axis}",
                        torch.zeros(tuple(shape) + (2,), device=device, dtype=dtype),
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
        self.register_buffer(
            "_step_increment",
            torch.ones((), device=device, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer("source_time", torch.zeros((), device=device, dtype=dtype))
        self.register_buffer(
            "time_step", torch.tensor(plan.dt, device=device, dtype=dtype)
        )
        self.register_buffer(
            "_dm2_status",
            torch.zeros(plan.dm2_target_count, device=device, dtype=torch.int8),
            persistent=False,
        )
        self.register_buffer(
            "_dm2_iterations",
            torch.zeros(plan.dm2_target_count, device=device, dtype=torch.int32),
            persistent=False,
        )
        dm2_states = []
        for metadata in plan.dm2_buckets:
            start = metadata.status_offset
            stop = start + metadata.target_count
            dm2_states.append(
                TorchDm2BucketState(
                    metadata,
                    status=self._dm2_status[start:stop],
                    iterations=self._dm2_iterations[start:stop],
                    device=device,
                    dtype=dtype,
                )
            )
        self.dm2_buckets = nn.ModuleList(dm2_states)
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


def _field_region(field, starts, trims):
    # Solver fields are contiguous registered buffers with zero storage offset.
    # A single non-overlapping view lets Inductor safely reinplace the mutation.
    strides = field.stride()
    size = tuple(
        field.shape[axis] - starts[axis] - trims[axis] for axis in range(3)
    ) + tuple(field.shape[3:])
    offset = sum(starts[axis] * strides[axis] for axis in range(3))
    return torch.as_strided(field, size=size, stride=strides, storage_offset=offset)


def _boundary_plane(field, axis, index):
    strides = field.stride()
    size = tuple(field.shape[:axis]) + tuple(field.shape[axis + 1 :])
    stride = tuple(strides[:axis]) + tuple(strides[axis + 1 :])
    offset = (field.shape[axis] - 1 if index == -1 else index) * strides[axis]
    return torch.as_strided(field, size=size, stride=stride, storage_offset=offset)


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
    direct_view_mutations,
):
    ex_target = (
        _field_region(ex, (0, 0, 0), (0, 1, 1))
        if direct_view_mutations
        else ex[:, :-1, :-1]
    )
    scratch = scratch_ex
    torch.sub(hz[1:, 1:, :], hz[1:, :-1, :], out=scratch)
    scratch.mul_(1.0 / dy)
    ex_target.addcmul_(inv_ex[:, :-1, :-1], scratch, value=dt)
    torch.sub(hy[1:, :, 1:], hy[1:, :, :-1], out=scratch)
    scratch.mul_(-1.0 / dz)
    ex_target.addcmul_(inv_ex[:, :-1, :-1], scratch, value=dt)

    ey_target = (
        _field_region(ey, (0, 0, 0), (1, 0, 1))
        if direct_view_mutations
        else ey[:-1, :, :-1]
    )
    scratch = scratch_ey
    torch.sub(hx[:, 1:, 1:], hx[:, 1:, :-1], out=scratch)
    scratch.mul_(1.0 / dz)
    ey_target.addcmul_(inv_ey[:-1, :, :-1], scratch, value=dt)
    torch.sub(hz[1:, 1:, :], hz[:-1, 1:, :], out=scratch)
    scratch.mul_(-1.0 / dx)
    ey_target.addcmul_(inv_ey[:-1, :, :-1], scratch, value=dt)

    ez_target = (
        _field_region(ez, (0, 0, 0), (1, 1, 0))
        if direct_view_mutations
        else ez[:-1, :-1, :]
    )
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
    direct_view_mutations,
):
    hx_target = (
        _field_region(hx, (0, 1, 1), (0, 0, 0))
        if direct_view_mutations
        else hx[:, 1:, 1:]
    )
    scratch = scratch_hx
    torch.sub(ey[:-1, :, 1:], ey[:-1, :, :-1], out=scratch)
    scratch.mul_(1.0 / dz)
    hx_target.addcmul_(inv_hx[:, 1:, 1:], scratch, value=dt)
    torch.sub(ez[:-1, 1:, :], ez[:-1, :-1, :], out=scratch)
    scratch.mul_(-1.0 / dy)
    hx_target.addcmul_(inv_hx[:, 1:, 1:], scratch, value=dt)

    hy_target = (
        _field_region(hy, (1, 0, 1), (0, 0, 0))
        if direct_view_mutations
        else hy[1:, :, 1:]
    )
    scratch = scratch_hy
    torch.sub(ez[1:, :-1, :], ez[:-1, :-1, :], out=scratch)
    scratch.mul_(1.0 / dx)
    hy_target.addcmul_(inv_hy[1:, :, 1:], scratch, value=dt)
    torch.sub(ex[:, :-1, 1:], ex[:, :-1, :-1], out=scratch)
    scratch.mul_(-1.0 / dz)
    hy_target.addcmul_(inv_hy[1:, :, 1:], scratch, value=dt)

    hz_target = (
        _field_region(hz, (1, 1, 0), (0, 0, 0))
        if direct_view_mutations
        else hz[1:, 1:, :]
    )
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
    direct_view_mutations,
):
    del scratch_ex, scratch_ey, scratch_ez, dz
    ex_target = (
        _field_region(ex, (0, 0, 0), (0, 1, 1))
        if direct_view_mutations
        else ex[:, :-1, :-1]
    )
    ex_target.add_(
        inv_ex[:, :-1, :-1] * (hz[1:, 1:, :] - hz[1:, :-1, :]),
        alpha=dt / dy,
    )
    ey_target = (
        _field_region(ey, (0, 0, 0), (1, 0, 1))
        if direct_view_mutations
        else ey[:-1, :, :-1]
    )
    ey_target.add_(
        inv_ey[:-1, :, :-1] * (hz[1:, 1:, :] - hz[:-1, 1:, :]),
        alpha=-dt / dx,
    )
    ez_target = (
        _field_region(ez, (0, 0, 0), (1, 1, 0))
        if direct_view_mutations
        else ez[:-1, :-1, :]
    )
    ez_target.add_(
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
    direct_view_mutations,
):
    del scratch_hx, scratch_hy, scratch_hz, dz
    hx_target = (
        _field_region(hx, (0, 1, 1), (0, 0, 0))
        if direct_view_mutations
        else hx[:, 1:, 1:]
    )
    hx_target.add_(
        inv_hx[:, 1:, 1:] * (ez[:-1, 1:, :] - ez[:-1, :-1, :]),
        alpha=-dt / dy,
    )
    hy_target = (
        _field_region(hy, (1, 0, 1), (0, 0, 0))
        if direct_view_mutations
        else hy[1:, :, 1:]
    )
    hy_target.add_(
        inv_hy[1:, :, 1:] * (ez[1:, :-1, :] - ez[:-1, :-1, :]),
        alpha=dt / dx,
    )
    hz_target = (
        _field_region(hz, (1, 1, 0), (0, 0, 0))
        if direct_view_mutations
        else hz[1:, 1:, :]
    )
    hz_target.add_(
        inv_hz[1:, 1:, :]
        * (
            (ex[:, 1:, :-1] - ex[:, :-1, :-1]) / dy
            - (ey[1:, :, :-1] - ey[:-1, :, :-1]) / dx
        ),
        alpha=dt,
    )


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
    reshape_fields,
):
    if reshape_fields and field.ndim == 4:
        field = field.reshape(-1, field.shape[-1])
        source1 = source1.reshape(-1, source1.shape[-1])
        source2 = source2.reshape(-1, source2.shape[-1])
    elif reshape_fields:
        field = field.reshape(-1)
        source1 = source1.reshape(-1)
        source2 = source2.reshape(-1)
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
    reshape_fields,
):
    if reshape_fields and field.ndim == 4:
        field = field.reshape(-1, field.shape[-1])
        source1 = source1.reshape(-1, source1.shape[-1])
        source2 = source2.reshape(-1, source2.shape[-1])
    elif reshape_fields:
        field = field.reshape(-1)
        source1 = source1.reshape(-1)
        source2 = source2.reshape(-1)
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
    the electric, magnetic, PML, dispersive, source, and observation phases.
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
        sources=(),
        probes=(),
        _is_auxiliary=False,
        _distributed_partition=None,
        _auxiliary_factory=None,
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
        if runtime.cpu_interop_threads is not None:
            current_interop_threads = torch.get_num_interop_threads()
            if current_interop_threads != runtime.cpu_interop_threads:
                try:
                    torch.set_num_interop_threads(runtime.cpu_interop_threads)
                except RuntimeError as error:
                    raise TorchConfigurationError(
                        "cpu_interop_threads must be configured before Torch parallel "
                        "work starts"
                    ) from error
        if runtime.launch.world_size == 1:
            if int(space.numprocs) != 1:
                raise TorchConfigurationError(
                    "MPI-decomposed Cartesian spaces are unsupported by TorchSimulation"
                )
        elif (
            runtime.launch.world_size != 2
            or _distributed_partition is None
            or int(space.numprocs) != 2
        ):
            raise TorchConfigurationError(
                "world_size=2 is available only through TorchDistributedSimulation"
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
        if time_step > dt_limit * (1 + 1e-14) and not _is_auxiliary:
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
        has_dm2 = any(
            bucket.signature.model == "dm2"
            for component in component_plans
            for bucket in component.buckets
        )
        if has_dm2 and bloch is not None:
            raise TorchConfigurationError("Dm2 supports real fields only")
        unsupported_models = sorted(
            {
                bucket.signature.model
                for component in component_plans
                for bucket in component.buckets
                if bucket.signature.model
                not in (
                    {"dielectric", "const", "dm2", "dummy", "upml", "cpml"}
                    | DISPERSIVE_MODELS
                )
            }
        )
        if unsupported_models:
            raise NotImplementedError(
                "the mixed-material planner lowered these models, but their state "
                "equations have no Torch execution implementation: "
                + ", ".join(unsupported_models)
            )

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
        self._has_dm2 = has_dm2
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
        self._fused_local_phases = (
            runtime.compile_policy == "compile" and _distributed_partition is None
        )
        self._packed_dm2 = (
            runtime.compile_policy == "compile" and device.type == "cpu"
        )
        self._dm2_execution_representation = (
            PACKED_DM2_REPRESENTATION
            if self._packed_dm2
            else FUNCTIONAL_DM2_REPRESENTATION
        )
        self._fused_source_updates = (
            self._fused_local_phases and device.type == "cpu"
        )
        self._source_execution_representation = (
            FUSED_SOURCE_REPRESENTATION
            if self._fused_source_updates
            else EXTERNAL_SOURCE_REPRESENTATION
        )
        self._direct_view_mutations = (
            self._fused_local_phases and device.type == "cpu"
        )
        self._view_mutation_representation = (
            DIRECT_VIEW_MUTATION_REPRESENTATION
            if self._direct_view_mutations
            else DEFAULT_VIEW_MUTATION_REPRESENTATION
        )
        z_collapsed = (
            shapes["Ez"][2] == 1
            and bloch is None
            and runtime.compile_policy == "compile"
            and (
                not self._has_pml
                or (device.type == "cpu" and _distributed_partition is None)
            )
        )
        if z_collapsed:
            electric_function = _electric_phase_2d_z
            magnetic_function = _magnetic_phase_2d_z
            self._phase_specialization = "z-collapsed-v1"
        else:
            electric_function = _electric_phase
            magnetic_function = _magnetic_phase
            self._phase_specialization = "three-axis-v1"
        self._electric = electric_function
        self._magnetic = magnetic_function
        if runtime.compile_policy == "compile" and not self._fused_local_phases:
            self._electric = _compile_fullgraph(
                electric_function,
                runtime,
                device,
                dynamic=False,
            )
            self._magnetic = _compile_fullgraph(
                magnetic_function,
                runtime,
                device,
                dynamic=False,
            )
        self._electric_args = self._electric_arguments()
        self._magnetic_args = self._magnetic_arguments()
        pml_functions = {
            "upml": _upml_bucket_update,
            "cpml": _cpml_bucket_update,
        }
        self._electric_pml = self._pml_executions(("Ex", "Ey", "Ez"), pml_functions)
        self._magnetic_pml = self._pml_executions(("Hx", "Hy", "Hz"), pml_functions)
        self._dm2_updates = []
        for bucket_state in state.dm2_buckets:
            metadata = bucket_state.metadata
            prefix = metadata.prefix
            repetitions = DM2_MAX_ITERATIONS // DM2_ITERATIONS_PER_CHUNK
            prepare_args = (
                state.field(metadata.component),
                state.field(metadata.source_component),
                state.step_count,
                state.time_step,
                getattr(plan, f"{prefix}_targets"),
                getattr(plan, f"{prefix}_source_positive_indices"),
                getattr(plan, f"{prefix}_source_negative_indices"),
                getattr(plan, f"{prefix}_rho30"),
                getattr(plan, f"{prefix}_gamma"),
                getattr(plan, f"{prefix}_t1"),
                getattr(plan, f"{prefix}_t2"),
                getattr(plan, f"{prefix}_hbar"),
                getattr(plan, f"{prefix}_omega"),
                getattr(plan, f"{prefix}_n_atom"),
                getattr(plan, f"{prefix}_curl_scale"),
            )
            iterate_args = (
                0.5 * plan.dt,
                0.25 * plan.dt,
                getattr(plan, f"{prefix}_rtol"),
                getattr(plan, f"{prefix}_omega"),
            )
            finalize_args = (
                state.field(metadata.component),
                getattr(plan, f"{prefix}_targets"),
            )
            if runtime.compile_policy == "compile":
                solve = (
                    bucket_state.solve_packed_cpu
                    if self._packed_dm2
                    else bucket_state.solve
                )
                if not self._fused_local_phases:
                    solve = _compile_fullgraph(
                        solve,
                        runtime,
                        device,
                        dynamic=False,
                    )
                self._dm2_updates.append(
                    (
                        solve,
                        (
                            *prepare_args,
                            *iterate_args[:3],
                            DM2_MAX_ITERATIONS,
                        ),
                    )
                )
            else:
                self._dm2_updates.append(
                    (
                        bucket_state.prepare,
                        prepare_args,
                        bucket_state.iterate,
                        iterate_args,
                        bucket_state.finalize,
                        finalize_args,
                        repetitions,
                    )
                )
        if has_dm2 and device.type == "cuda":
            self._dm2_iterations_host = torch.empty(
                plan.dm2_target_count,
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )
        else:
            self._dm2_iterations_host = state._dm2_iterations

        self._dispersive = self._apply_dispersive
        self._electric_material = self._electric_material_update
        self._magnetic_material = self._magnetic_material_update
        if runtime.compile_policy == "compile" and not self._fused_local_phases:
            if (
                self._electric_pml
                or self._dispersive_buckets
                or self._has_electric_constants
            ):
                self._electric_material = _compile_fullgraph(
                    self._electric_material_update,
                    runtime,
                    device,
                    dynamic=False,
                )
            if self._magnetic_pml or self._has_magnetic_constants:
                self._magnetic_material = _compile_fullgraph(
                    self._magnetic_material_update,
                    runtime,
                    device,
                    dynamic=False,
                )
        self._distributed_partition = _distributed_partition
        self._distributed_exchange = None
        self._electric_half = None
        self._magnetic_half = None
        if self._fused_local_phases:
            self._electric_half = _compile_fullgraph(
                self._electric_half_update,
                runtime,
                device,
                dynamic=False,
            )
            self._magnetic_half = _compile_fullgraph(
                self._magnetic_half_update,
                runtime,
                device,
                dynamic=False,
            )
        simulation_factory = (
            type(self) if _auxiliary_factory is None else _auxiliary_factory
        )
        self.sources = lower_sources(
            sources,
            simulation=self,
            simulation_factory=simulation_factory,
            runtime=runtime,
            bloch=bloch,
        )
        self._is_auxiliary = bool(_is_auxiliary)
        self.probes = TorchProbeBuffer(probes, simulation=self)
        self.plan_identity = self._compute_plan_identity()
        self.compile_cache_key = self._compute_compile_cache_key()
        self._cuda_graphs = {}

    def _compute_compile_cache_key(self):
        if self.device.type == "cuda":
            capability = torch.cuda.get_device_capability(self.device)
        else:
            capability = torch.backends.cpu.get_cpu_capability()
        planner_only_suffixes = (
            "_region_keys",
            "_region_coefficient_indices",
            "_target_region_indices",
            "_tile_origins",
            "_tile_region_indices",
        )
        inactive_prefixes = tuple(
            f"bucket_{component_name.lower()}_{index}_"
            for component_name, component in self.plan.components.items()
            for index, bucket in enumerate(component.buckets)
            if bucket.signature.model in {"const", "dielectric", "dummy"}
        )

        def buffer_layouts(module, *, plan=False, recurse=True):
            layouts = []
            for name, value in module.named_buffers(recurse=recurse):
                if plan:
                    if name.startswith(
                        ("material_ids_", "underlying_ids_", "ownership_")
                    ):
                        continue
                    if name.startswith(inactive_prefixes):
                        continue
                    if name.endswith(planner_only_suffixes):
                        continue
                    if name.endswith("_coefficients") and not name.endswith(
                        "_cell_coefficients"
                    ):
                        continue
                layouts.append(
                    (
                        name,
                        str(value.device),
                        str(value.dtype),
                        str(value.layout),
                        tuple(value.shape),
                        tuple(value.stride()),
                    )
                )
            return tuple(layouts)

        signatures = tuple(
            (component_name, index, repr(bucket.signature))
            for component_name, component in self.plan.components.items()
            for index, bucket in enumerate(component.buckets)
        )
        source_topology = tuple(
            (
                f"{type(batch).__module__}.{type(batch).__qualname__}",
                batch.component,
                getattr(batch, "paired_real", None),
                getattr(batch, "auxiliary_component", None),
                getattr(batch, "gaussian_width", None),
            )
            for batch in self.sources.batches
        )
        if self._fused_local_phases:
            region_topology = "local-two-fused-half-steps"
        elif self._distributed_partition is not None:
            region_topology = "distributed-stencil-and-material-regions"
        else:
            region_topology = "local-eager-stencil-and-material-phases"
        self._compiled_region_topology = region_topology
        payload = (
            TORCH_SOLVER_ABI,
            torch.__version__,
            capability,
            str(self.dtype),
            self.runtime.compile_policy,
            self.runtime.compile_mode,
            region_topology,
            "dense-base+compact-indexed-materials-v1",
            self._view_mutation_representation,
            self._dm2_execution_representation,
            self._source_execution_representation,
            self._phase_specialization,
            tuple(self.plan.dr),
            self.plan.dt,
            self.plan.bloch,
            self.state.paired_real,
            self._fused_local_phases,
            (
                None
                if self._distributed_partition is None
                else self._distributed_partition.identity
            ),
            (
                self._has_dm2,
                DM2_MAX_ITERATIONS,
                DM2_ITERATIONS_PER_CHUNK,
                self._has_electric_constants,
                self._has_magnetic_constants,
                self._has_pml,
            ),
            tuple(sorted(self.plan.shapes.items())),
            signatures,
            source_topology,
            tuple(
                auxiliary.compile_cache_key
                for auxiliary in self.sources.auxiliaries
            ),
            buffer_layouts(self.plan, plan=True),
            buffer_layouts(self.state, recurse=False),
            tuple(buffer_layouts(bucket) for bucket in self.state.dm2_buckets),
            buffer_layouts(self.sources) if self._fused_source_updates else (),
        )
        return hashlib.sha256(repr(payload).encode()).hexdigest()

    def _compute_plan_identity(self):
        digest = hashlib.sha256()
        digest.update(repr((self.plan.dr, self.plan.dt, self.plan.bloch)).encode())
        for module in (self.plan, self.sources):
            for name, value in module.named_buffers():
                digest.update(name.encode())
                digest.update(str(value.dtype).encode())
                digest.update(repr(tuple(value.shape)).encode())
                digest.update(
                    value.detach().to(device="cpu").contiguous().numpy().tobytes()
                )
        for auxiliary in self.sources.auxiliaries:
            digest.update(auxiliary.plan_identity.encode())
        return digest.hexdigest()

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
            self._direct_view_mutations,
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
            self._direct_view_mutations,
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
                    (lambda value: value)
                    if self._direct_view_mutations
                    else (
                        (lambda value: value.reshape(-1, paired_width))
                        if paired_width is not None
                        else (lambda value: value.reshape(-1))
                    )
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
                arguments.append(self._direct_view_mutations)
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
    def _rotate_or_copy(destination, source, angle, scratch):
        if scratch is not None:
            scratch.copy_(source)
            source = scratch
        if angle is None:
            destination.copy_(source)
            return
        cosine = float(np.cos(angle))
        sine = float(np.sin(angle))
        destination[..., 0].copy_(source[..., 0]).mul_(cosine)
        destination[..., 0].add_(source[..., 1], alpha=-sine)
        destination[..., 1].copy_(source[..., 0]).mul_(sine)
        destination[..., 1].add_(source[..., 1], alpha=cosine)

    def _sync_boundary_family(self, names, *, high_from_low, skip_axis=None):
        for name in names:
            component_axis = ("x", "y", "z").index(name[1].lower())
            field = self.state.field(name)
            for axis in range(3):
                if axis == component_axis or axis == skip_axis:
                    continue
                destination_index = -1 if high_from_low else 0
                source_index = 0 if high_from_low else -1
                direction = 1 if high_from_low else -1
                if self._direct_view_mutations:
                    destination = _boundary_plane(field, axis, destination_index)
                    source = _boundary_plane(field, axis, source_index)
                else:
                    destination_slice = [slice(None)] * field.ndim
                    source_slice = [slice(None)] * field.ndim
                    destination_slice[axis] = destination_index
                    source_slice[axis] = source_index
                    destination = field[tuple(destination_slice)]
                    source = field[tuple(source_slice)]
                self._rotate_or_copy(
                    destination,
                    source,
                    self._boundary_angle(name, axis, direction),
                    getattr(self.state, f"_boundary_{name.lower()}_{axis}", None),
                )

    def _sync_electric_boundaries(self, *, skip_axis=None):
        self._sync_boundary_family(
            ("Ex", "Ey", "Ez"), high_from_low=True, skip_axis=skip_axis
        )

    def _sync_magnetic_boundaries(self, *, skip_axis=None):
        self._sync_boundary_family(
            ("Hx", "Hy", "Hz"), high_from_low=False, skip_axis=skip_axis
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

    def _update_dm2(self):
        for update in self._dm2_updates:
            if len(update) == 2:
                solve, solve_args = update
                solve(*solve_args)
                continue
            (
                prepare,
                prepare_args,
                iterate,
                iterate_args,
                finalize,
                finalize_args,
                repetitions,
            ) = update
            prepare(*prepare_args)
            for _ in range(repetitions):
                iterate(*iterate_args)
            finalize(*finalize_args)
        if not self._has_dm2:
            return
        for metadata in self.plan.dm2_buckets:
            start = metadata.status_offset
            stop = start + metadata.target_count
            status = self.state._dm2_status[start:stop]
            label = (
                f"{metadata.component}/width={metadata.transition_count}:"
                f"[{metadata.target_count} target(s)]"
            )
            torch._assert_async(
                torch.all(status != 1),
                "Dm2 corrector produced an invalid error for " + label,
            )
            torch._assert_async(
                torch.all(status != 2),
                "Dm2 corrector failed to converge for " + label,
            )

    def _apply_dispersive(self):
        for descriptor in self._dispersive_buckets:
            update_bucket(self.plan, self.state, descriptor)

    def _electric_material_update(self):
        self._run_pml(self._electric_pml)
        if self._dispersive_buckets:
            self._dispersive()
        if self._has_electric_constants:
            self._apply_constants(("Ex", "Ey", "Ez"))

    def _magnetic_material_update(self):
        self._run_pml(self._magnetic_pml)
        if self._has_magnetic_constants:
            self._apply_constants(("Hx", "Hy", "Hz"))

    def _electric_post_update(self):
        self._electric_material()
        self._update_dm2()

    def _magnetic_post_update(self):
        self._magnetic_material()

    def _electric_half_update(self):
        self._sync_magnetic_boundaries()
        self._electric(*self._electric_args)
        self._electric_post_update()
        if self._fused_source_updates and not self.sources.empty:
            self.sources.apply(
                self,
                electric=True,
                time=self.state.source_time + 0.5 * self.state.time_step,
                transparent_time=self.state.source_time,
            )

    def _magnetic_half_update(self):
        self._sync_electric_boundaries()
        self._magnetic(*self._magnetic_args)
        self._magnetic_post_update()
        if self._fused_source_updates and not self.sources.empty:
            self.sources.apply(
                self,
                electric=False,
                time=self.state.source_time + self.state.time_step,
                transparent_time=self.state.source_time + self.state.time_step,
            )

    def _run_compute_region(self, name, function):
        graph = self._cuda_graphs.get(name)
        if graph is None:
            function()
        else:
            graph.replay()

    @torch.inference_mode()
    def capture_cuda_graphs(self):
        """Capture fixed-storage compute regions without NCCL or host I/O."""
        if self.device.type != "cuda":
            raise TorchConfigurationError("CUDA graph capture requires a CUDA runtime")
        if self._cuda_graphs:
            return self
        checkpoint = self.checkpoint()
        if self._electric_half is not None:
            regions = [
                ("electric_half", self._electric_half),
                ("magnetic_half", self._magnetic_half),
            ]
        else:
            regions = [
                ("electric", lambda: self._electric(*self._electric_args)),
                ("magnetic", lambda: self._magnetic(*self._magnetic_args)),
            ]
            if (
                self._has_pml
                or self._dispersive_buckets
                or self._has_dm2
                or self._has_electric_constants
            ):
                regions.append(("electric_post", self._electric_post_update))
            if self._has_pml or self._has_magnetic_constants:
                regions.append(("magnetic_post", self._magnetic_post_update))
        for _name, function in regions:
            function()
        torch.cuda.synchronize(self.device)
        graphs = {}
        for name, function in regions:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                function()
            graphs[name] = graph
        self._cuda_graphs = graphs
        self.load_checkpoint(checkpoint)
        torch.cuda.synchronize(self.device)
        return self

    @torch.inference_mode()
    def advance(self, steps):
        """Advance fixed rank-local state without implicit host conversion."""
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        exchange = self._distributed_exchange
        split_axis = None if exchange is None else exchange.axis
        for _ in range(steps):
            if self.device.type == "cuda" and self.runtime.compile_policy == "compile":
                torch.compiler.cudagraph_mark_step_begin()
            if self._electric_half is not None:
                self._run_compute_region("electric_half", self._electric_half)
            else:
                self._sync_magnetic_boundaries(skip_axis=split_axis)
                if exchange is not None:
                    exchange.begin("magnetic")
                self._run_compute_region(
                    "electric", lambda: self._electric(*self._electric_args)
                )
                if exchange is not None:
                    exchange.finish("magnetic")
                self._run_compute_region("electric_post", self._electric_post_update)
            if not self.sources.empty:
                if not self._fused_source_updates:
                    self.sources.apply(
                        self,
                        electric=True,
                        time=self.state.source_time + 0.5 * self.state.time_step,
                        transparent_time=self.state.source_time,
                    )
                self.sources.step_auxiliaries()
            if not self.probes.empty:
                self.probes.record(
                    self,
                    electric=True,
                    time=self.state.source_time + 0.5 * self.state.time_step,
                )
            if self._magnetic_half is not None:
                self._run_compute_region("magnetic_half", self._magnetic_half)
            else:
                self._sync_electric_boundaries(skip_axis=split_axis)
                if exchange is not None:
                    exchange.begin("electric")
                self._run_compute_region(
                    "magnetic", lambda: self._magnetic(*self._magnetic_args)
                )
                if exchange is not None:
                    exchange.finish("electric")
                self._run_compute_region("magnetic_post", self._magnetic_post_update)
            if not self.sources.empty:
                if not self._fused_source_updates:
                    self.sources.apply(
                        self,
                        electric=False,
                        time=self.state.source_time + self.state.time_step,
                        transparent_time=self.state.source_time
                        + self.state.time_step,
                    )
            if not self.probes.empty:
                self.probes.record(
                    self,
                    electric=False,
                    time=self.state.source_time + self.state.time_step,
                )
            self.state.source_time.add_(self.state.time_step)
            self.state.step_count.add_(self.state._step_increment)
        return self

    def step(self):
        """Advance one step as a convenience wrapper."""
        return self.advance(1)

    @torch.inference_mode()
    def checkpoint(self):
        """Return a versioned tensor/metadata checkpoint, never a simulation pickle."""
        return {
            "format": "gmes.torch.simulation",
            "version": 1,
            "metadata": {
                "plan_identity": self.plan_identity,
                "device": str(self.device),
                "dtype": str(self.dtype),
                "paired_real": self.state.paired_real,
            },
            "state": self.state.checkpoint(),
            "auxiliaries": tuple(
                item.checkpoint() for item in self.sources.auxiliaries
            ),
            "probes": self.probes.checkpoint(),
        }

    @torch.inference_mode()
    def load_checkpoint(self, checkpoint):
        """Restore a trusted versioned checkpoint into fixed live buffers."""
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("format") != "gmes.torch.simulation"
        ):
            raise ValueError("unsupported Torch checkpoint format")
        if checkpoint.get("version") != 1:
            raise ValueError(
                f"unsupported Torch checkpoint version {checkpoint.get('version')!r}"
            )
        expected_keys = {
            "format",
            "version",
            "metadata",
            "state",
            "auxiliaries",
            "probes",
        }
        if set(checkpoint) != expected_keys:
            raise ValueError("Torch checkpoint schema keys do not match version 1")
        metadata = checkpoint["metadata"]
        expected_metadata = {
            "plan_identity": self.plan_identity,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "paired_real": self.state.paired_real,
        }
        if metadata != expected_metadata:
            raise ValueError(
                "Torch checkpoint metadata does not match this execution plan"
            )
        auxiliaries = checkpoint["auxiliaries"]
        if len(auxiliaries) != len(self.sources.auxiliaries):
            raise ValueError("Torch checkpoint auxiliary solver count is incompatible")
        self.state.load_checkpoint(checkpoint["state"])
        for auxiliary, values in zip(
            self.sources.auxiliaries, auxiliaries, strict=True
        ):
            auxiliary.load_checkpoint(values)
        self.probes.load_checkpoint(checkpoint["probes"])
        return self

    def save_checkpoint(self, filename):
        """Explicitly persist the versioned tensor/metadata checkpoint."""
        return write_torch_checkpoint(self.checkpoint(), filename)

    def load_checkpoint_file(self, filename):
        """Explicitly load and restore a pickle-free checkpoint file."""
        return self.load_checkpoint(read_torch_checkpoint(filename, device=self.device))

    def flush_probes(self):
        """Explicitly synchronize and drain bounded probe rings to host arrays."""
        return self.probes.flush()

    def host_snapshot(self, *, numpy=True, complex_fields=True):
        """Explicit host adapter for file, plotting, and NumPy consumers."""
        return self.state.host_snapshot(numpy=numpy, complex_fields=complex_fields)

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

    @torch.inference_mode()
    def load_host_dm2_state(self, states, *, step_count=None):
        """Copy cell-major (cell, transition, u/v/w) DM2 state arrays."""
        if len(states) != len(self.state.dm2_buckets):
            raise ValueError("DM2 state input must contain one array per bucket")
        for bucket, values in zip(self.state.dm2_buckets, states):
            source = torch.as_tensor(values)
            expected = (
                bucket.metadata.target_count,
                bucket.metadata.transition_count,
                3,
            )
            if tuple(source.shape) != expected:
                raise ValueError(
                    f"DM2 state for {bucket.metadata.component} has shape "
                    f"{tuple(source.shape)}; expected {expected}"
                )
            bucket.u.copy_(
                source.permute(2, 0, 1).to(device=self.device, dtype=self.dtype)
            )
        if step_count is not None:
            if (
                isinstance(step_count, bool)
                or not isinstance(step_count, int)
                or step_count < 0
            ):
                raise ValueError("DM2 state step_count must be a non-negative integer")
            self.state.step_count.fill_(step_count)
            self.state.source_time.copy_(self.state.time_step).mul_(step_count)
        return self

    @torch.inference_mode()
    def dm2_state_snapshot(self):
        """Return explicit host snapshots of transformed and physical DM2 state."""
        time = float(self.state.source_time.detach().cpu().numpy())
        snapshots = []
        for metadata, bucket in zip(self.plan.dm2_buckets, self.state.dm2_buckets):
            prefix = metadata.prefix
            u = bucket.u.detach().cpu().permute(1, 2, 0).contiguous().numpy().copy()
            t1 = getattr(self.plan, f"{prefix}_t1").detach().cpu().numpy()
            t2 = getattr(self.plan, f"{prefix}_t2").detach().cpu().numpy()
            rho30 = getattr(self.plan, f"{prefix}_rho30").detach().cpu().numpy()
            rho = u.copy()
            rho[:, :, :2] *= np.exp(-time / t2)[:, None, None]
            rho[:, :, 2] *= np.exp(-time / t1)[:, None]
            rho[:, :, 2] += rho30[:, None]
            snapshots.append(
                {
                    "component": metadata.component,
                    "transition_count": metadata.transition_count,
                    "targets": (
                        getattr(self.plan, f"{prefix}_targets")
                        .detach()
                        .cpu()
                        .numpy()
                        .copy()
                    ),
                    "u": u,
                    "rho": rho,
                    "time": time,
                }
            )
        return tuple(snapshots)

    @torch.inference_mode()
    def diagnostics(self):
        """Return the focused runtime diagnostic record for this simulation."""
        result = torch_runtime_diagnostics(self.runtime)
        result["resolved_device"] = str(self.device)
        result["cpu_threads"] = self.cpu_threads
        result["cpu_interop_threads"] = torch.get_num_interop_threads()
        result["compile_cache_key"] = self.compile_cache_key
        result["compile_solver_abi"] = TORCH_SOLVER_ABI
        result["compiled_region_topology"] = self._compiled_region_topology
        result["material_execution_representation"] = (
            "dense-base+compact-indexed-materials-v1"
        )
        result["view_mutation_representation"] = self._view_mutation_representation
        result["dm2_execution_representation"] = (
            self._dm2_execution_representation
        )
        result["phase_specialization"] = self._phase_specialization
        result["cuda_graph_regions"] = tuple(sorted(self._cuda_graphs))
        result["material_plan"] = self.plan.decision_report()
        if self._has_dm2:
            self._dm2_iterations_host.copy_(self.state._dm2_iterations)
            iterations = self._dm2_iterations_host.numpy()
            dm2 = []
            for metadata in self.plan.dm2_buckets:
                start = metadata.status_offset
                stop = start + metadata.target_count
                values, counts = np.unique(iterations[start:stop], return_counts=True)
                dm2.append(
                    {
                        "component": metadata.component,
                        "transition_count": metadata.transition_count,
                        "targets": metadata.target_count,
                        "iteration_distribution": {
                            int(value): int(count)
                            for value, count in zip(values, counts)
                        },
                    }
                )
            result["dm2"] = tuple(dm2)
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
        result["sources"] = {
            "execution_representation": self._source_execution_representation,
            "batches": len(self.sources.batches),
            "target_rows": sum(
                value.numel()
                for name, value in self.sources.named_buffers()
                if name.endswith("targets")
            ),
            "auxiliaries": len(self.sources.auxiliaries),
            "plan_bytes": sum(
                value.numel() * value.element_size() for value in self.sources.buffers()
            ),
        }
        result["probes"] = {
            "rings": len(self.probes.rings),
            "capacity": sum(ring.capacity for ring in self.probes.rings),
            "device_bytes": sum(
                value.numel() * value.element_size() for value in self.probes.buffers()
            ),
            "backpressure": "overwrite-oldest",
        }
        result["boundaries"] = {
            "scheduling": "external",
            "paired_real_scratch_bytes": sum(
                value.numel() * value.element_size()
                for name, value in self.state.named_buffers()
                if name.startswith("_boundary_")
            ),
        }
        result["checkpoint_schema"] = {
            "format": "gmes.torch.simulation",
            "version": 1,
            "plan_identity": self.plan_identity,
        }
        return result

    def buffer_addresses(self):
        """Return fixed storage addresses for explicit capture diagnostics."""
        result = {
            f"state.{name}": tensor.data_ptr()
            for name, tensor in self.state.named_buffers()
        }
        result.update(
            {
                f"plan.{name}": tensor.data_ptr()
                for name, tensor in self.plan.named_buffers()
            }
        )
        result.update(
            {
                f"sources.{name}": tensor.data_ptr()
                for name, tensor in self.sources.named_buffers()
            }
        )
        result.update(
            {
                f"probes.{name}": tensor.data_ptr()
                for name, tensor in self.probes.named_buffers()
            }
        )
        for index, auxiliary in enumerate(self.sources.auxiliaries):
            result.update(
                {
                    f"auxiliary.{index}.{name}": address
                    for name, address in auxiliary.buffer_addresses().items()
                }
            )
        return result


__all__ = [
    "ComponentPlan",
    "DistributedLaunch",
    "ExecutionSignature",
    "FlattenedStencilTerm",
    "MaterialBucketPlan",
    "TorchConfigurationError",
    "TorchExecutionPlanner",
    "TorchPointSourceRecord",
    "TorchProbeSamples",
    "TorchProbeSpec",
    "TorchProbeSpectrum",
    "TorchRuntimeConfig",
    "TorchSimulation",
    "TorchSimulationPlan",
    "TorchSimulationState",
    "TorchSourceLoweringContext",
    "probe_spectrum",
    "read_torch_checkpoint",
    "torch_runtime_diagnostics",
    "write_probe_text",
    "write_torch_checkpoint",
]
