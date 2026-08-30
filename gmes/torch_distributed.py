"""Two-GPU spatial decomposition and NCCL halo exchange for Torch FDTD."""

from __future__ import annotations

import hashlib
import os
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from math import prod
from types import MappingProxyType

import numpy as np
import torch
import torch.distributed as dist

from . import constant as const
from .geometry import GeomBoxTree
from .torch_fdtd import (
    DistributedLaunch,
    TorchConfigurationError,
    TorchRuntimeConfig,
    TorchSimulation,
)
from .torch_output import TorchProbeSpec
from .torch_plan import COMPONENTS

_AXIS_NAMES = ("x", "y", "z")
_AXIS_COMPONENT = ("Ex", "Ey", "Ez")
_HALO_COMPONENTS = MappingProxyType(
    {
        0: ("Ey", "Ez"),
        1: ("Ex", "Ez"),
        2: ("Ex", "Ey"),
    }
)
_MATERIAL_COSTS = MappingProxyType(
    {
        "dummy": 0.1,
        "const": 0.25,
        "dielectric": 1.0,
        "upml": 3.0,
        "cpml": 3.5,
        "drude": 3.0,
        "lorentz": 3.5,
        "dcp": 4.5,
        "dm2": 12.0,
    }
)


class TorchDistributedError(RuntimeError):
    """A two-rank execution or communication contract failed."""


@dataclass(frozen=True)
class TwoGpuDecomposition:
    """One Cartesian cut with deterministic rank ownership."""

    global_shape: tuple[int, int, int]
    axis: int
    cut: int
    rank_costs: tuple[float, float]
    device_weights: tuple[float, float]
    communication_cells: int
    source_crossings: int

    def __post_init__(self):
        if self.axis not in (0, 1, 2):
            raise ValueError("split axis must be 0, 1, or 2")
        if len(self.global_shape) != 3 or any(value < 1 for value in self.global_shape):
            raise ValueError("global shape must contain three positive cell counts")
        if not 0 < self.cut < self.global_shape[self.axis]:
            raise ValueError("cut must leave nonempty partitions on both ranks")

    @property
    def identity(self):
        """Return a stable hash of the partition geometry and device weights."""

        payload = (
            self.global_shape,
            self.axis,
            self.cut,
            tuple(round(value, 12) for value in self.device_weights),
        )
        return hashlib.sha256(repr(payload).encode()).hexdigest()

    def offset(self, rank):
        """Return the rank's global cell offset along the split axis."""

        self._validate_rank(rank)
        result = [0, 0, 0]
        if rank == 1:
            result[self.axis] = self.cut
        return tuple(result)

    def local_shape(self, rank):
        """Return the rank-local three-dimensional cell shape."""

        self._validate_rank(rank)
        result = list(self.global_shape)
        result[self.axis] = self.cut if rank == 0 else result[self.axis] - self.cut
        return tuple(result)

    def metadata(self):
        """Return serializable decomposition metadata including its identity."""

        result = asdict(self)
        result["identity"] = self.identity
        result["axis_name"] = _AXIS_NAMES[self.axis]
        return result

    @staticmethod
    def _validate_rank(rank):
        if rank not in (0, 1):
            raise ValueError("two-GPU decomposition rank must be 0 or 1")


class _TwoRankCartesianComm:
    """Minimal Cartesian metadata used by legacy geometry/source lowering."""

    def __init__(self, axis, rank):
        dims = [1, 1, 1]
        coords = [0, 0, 0]
        dims[axis] = 2
        coords[axis] = rank
        self.rank = rank
        self._dims = tuple(dims)
        self._coords = tuple(coords)

    def Get_topo(self):
        """Return MPI-compatible Cartesian topology metadata."""

        return self._dims, (1, 1, 1), self._coords

    def Get_size(self):
        """Return the fixed communicator size of two."""

        return 2

    def Get_coords(self, rank):
        """Return the Cartesian coordinate for a rank."""

        coords = [0, 0, 0]
        coords[self._dims.index(2)] = rank
        return tuple(coords)

    def Get_cart_rank(self, coords):
        """Return the rank at a Cartesian coordinate."""

        return int(coords[self._dims.index(2)])

    def Shift(self, direction, _disp):
        """Return source and destination ranks for a Cartesian shift."""

        if direction != self._dims.index(2):
            return self.rank, self.rank
        return 1 - self.rank, 1 - self.rank

    def bcast(self, obj=None, root=0):
        """Return the object as the local stand-in for a broadcast."""

        del root
        return obj

    def allgather(self, obj=None):
        """Return one local object for each emulated rank."""

        return [obj, obj]


def rank_local_space(global_space, decomposition, rank):
    """Clone a global Cartesian description into one non-replicated rank view."""
    local = deepcopy(global_space)
    local.numprocs = 2
    local.my_id = rank
    local.my_cart_idx = np.asarray(
        _TwoRankCartesianComm(decomposition.axis, rank).Get_coords(rank),
        dtype=np.intp,
    )
    local.cart_comm = _TwoRankCartesianComm(decomposition.axis, rank)
    local.my_field_size = np.asarray(decomposition.local_shape(rank), dtype=np.intp)
    local.global_field_offset = np.asarray(decomposition.offset(rank), dtype=np.intp)
    return local


def _material_weight(material):
    name = type(material).__name__.lower()
    if name.startswith("dcp"):
        return _MATERIAL_COSTS["dcp"]
    for marker, weight in _MATERIAL_COSTS.items():
        if marker in name:
            return weight
    return 2.0


def _axis_cost_profile(space, geom_tree, axis, *, sample_limit=32):
    shape = tuple(int(value) for value in space.whole_field_size)
    indices = []
    for current_axis, length in enumerate(shape):
        if current_axis == axis or length <= sample_limit:
            indices.append(np.arange(length, dtype=np.intp))
        else:
            indices.append(
                np.unique(
                    np.linspace(0, length - 1, sample_limit).round().astype(np.intp)
                )
            )
    axes = tuple(
        (values.astype(np.float64) + 0.5) * space.dr[current_axis]
        - space.half_size[current_axis]
        for current_axis, values in enumerate(indices)
    )
    sampled_shape = tuple(len(values) for values in indices)
    total = prod(sampled_shape)
    profile = np.zeros(shape[axis], dtype=np.float64)
    geometries = None
    weight_table = None
    tile_size = 65536
    for start in range(0, total, tile_size):
        stop = min(start + tile_size, total)
        lowered = geom_tree.lower_grid(*axes, start, stop, component=const.Ex)
        if geometries is None:
            geometries = lowered.geometries
            weight_table = np.asarray(
                [_material_weight(item.material) for item in geometries],
                dtype=np.float64,
            )
        weights = weight_table[lowered.material_ids]
        flat = np.arange(start, stop, dtype=np.int64)
        sampled_axis_indices = np.unravel_index(flat, sampled_shape)[axis]
        global_axis_indices = indices[axis][sampled_axis_indices]
        profile += np.bincount(
            global_axis_indices, weights=weights, minlength=shape[axis]
        )
    sampled_cross_section = prod(
        sampled_shape[current_axis] for current_axis in range(3) if current_axis != axis
    )
    full_cross_section = prod(
        shape[current_axis] for current_axis in range(3) if current_axis != axis
    )
    profile *= full_cross_section / sampled_cross_section
    return profile


def _source_crossings(sources, space, axis, cut):
    coordinate = cut * space.dr[axis] - space.half_size[axis]
    crossings = 0
    for source in sources:
        center = getattr(source, "center", None)
        if center is None:
            continue
        size = getattr(source, "size", None)
        half_width = 0.5 * float(size[axis]) if size is not None else 0.0
        if abs(float(center[axis]) - coordinate) <= half_width + space.dr[axis]:
            crossings += 1
    return crossings


def choose_two_gpu_decomposition(
    space,
    geometry,
    *,
    sources=(),
    device_weights=(1.0, 1.0),
    split_axis=None,
    cut=None,
):
    """Choose a material-, source-, surface-, and device-aware two-rank cut."""
    shape = tuple(int(value) for value in space.whole_field_size)
    weights = tuple(float(value) for value in device_weights)
    if len(weights) != 2 or any(
        not np.isfinite(value) or value <= 0 for value in weights
    ):
        raise ValueError("device_weights must contain two finite positive values")
    scale = sum(weights)
    weights = tuple(value / scale for value in weights)
    axes = (
        (int(split_axis),)
        if split_axis is not None
        else tuple(axis for axis, length in enumerate(shape) if length >= 4)
    )
    if not axes:
        raise TorchConfigurationError(
            "two-GPU decomposition needs at least one axis with four cells"
        )
    if any(axis not in (0, 1, 2) for axis in axes):
        raise ValueError("split_axis must be 0, 1, or 2")
    if not hasattr(space, "dt"):
        space.dt = 0.5 * min(float(value) for value in space.dr)
    for item in geometry:
        item.init(space)
    geom_tree = GeomBoxTree(geometry)
    best = None
    for axis in axes:
        length = shape[axis]
        if length < 4:
            continue
        profile = _axis_cost_profile(space, geom_tree, axis)
        prefix = np.cumsum(profile)
        candidates = (int(cut),) if cut is not None else range(2, length - 1)
        communication = prod(shape[current] for current in range(3) if current != axis)
        for candidate in candidates:
            if not 1 < candidate < length - 1:
                raise ValueError("cut must leave at least two cells on each rank")
            costs = (
                float(prefix[candidate - 1]),
                float(prefix[-1] - prefix[candidate - 1]),
            )
            crossings = _source_crossings(sources, space, axis, candidate)
            predicted = max(costs[0] / weights[0], costs[1] / weights[1])
            # A leading-axis plane is contiguous in the component tensors;
            # the other axes require strided packing and have a larger PHB cost.
            surface_weight = 16.0 if axis == 0 else 64.0
            predicted += communication * (surface_weight + 16.0 * crossings)
            record = (predicted, axis, candidate, costs, communication, crossings)
            if best is None or record[:3] < best[:3]:
                best = record
    if best is None:
        raise TorchConfigurationError(
            "no nontrivial two-GPU Cartesian cut is available"
        )
    _, axis, selected_cut, costs, communication, crossings = best
    return TwoGpuDecomposition(
        shape,
        axis,
        selected_cut,
        costs,
        weights,
        communication,
        crossings,
    )


def distributed_launch_from_environment():
    """Resolve the four torchrun rank variables without implicit defaults."""
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise TorchConfigurationError(
            "two-GPU execution must be launched with torchrun; missing "
            + ", ".join(missing)
        )
    return DistributedLaunch(
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
        local_rank=int(os.environ["LOCAL_RANK"]),
        local_world_size=int(os.environ["LOCAL_WORLD_SIZE"]),
    )


def _initialize_nccl(runtime, timeout_seconds, *, require_peer_access=False):
    launch = runtime.launch
    if (launch.world_size, launch.local_world_size) != (2, 2):
        raise TorchConfigurationError(
            "advertised distributed mode requires exactly two local torchrun ranks"
        )
    if torch.cuda.device_count() != 2:
        raise TorchConfigurationError(
            f"two-GPU mode requires exactly two visible CUDA devices; found {torch.cuda.device_count()}"
        )
    expected_device = torch.device("cuda", launch.local_rank)
    if torch.device(runtime.device) != expected_device:
        raise TorchConfigurationError(
            f"rank {launch.rank} must use {expected_device}, not {runtime.device}"
        )
    if not torch.distributed.is_nccl_available():
        raise TorchConfigurationError("PyTorch was built without NCCL support")
    torch.cuda.set_device(expected_device)
    peer = 1 - launch.local_rank
    if require_peer_access and not torch.cuda.can_device_access_peer(
        launch.local_rank, peer
    ):
        raise TorchConfigurationError(
            f"cuda:{launch.local_rank} cannot directly access cuda:{peer}; "
            "disable require_peer_access for a topology-qualified NCCL transport"
        )
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(seconds=float(timeout_seconds)),
            device_id=expected_device,
        )
    if dist.get_backend() != "nccl" or dist.get_world_size() != 2:
        raise TorchConfigurationError(
            "the active process group is not a two-rank NCCL group"
        )
    if dist.get_rank() != launch.rank:
        raise TorchConfigurationError(
            "runtime rank does not match the active process group"
        )
    precision_code = 32 if runtime.dtype == torch.float32 else 64
    contract = torch.tensor(
        [precision_code, launch.local_rank],
        device=expected_device,
        dtype=torch.int32,
    )
    contracts = [torch.empty_like(contract) for _ in range(2)]
    dist.all_gather(contracts, contract)
    rows = [tuple(int(value) for value in item.cpu().tolist()) for item in contracts]
    if {row[0] for row in rows} != {precision_code}:
        raise TorchConfigurationError(
            "both ranks must use the same floating-point precision"
        )
    if {row[1] for row in rows} != {0, 1}:
        raise TorchConfigurationError(
            "each local CUDA device must be bound to exactly one rank"
        )
    return expected_device


def _device_weight(device):
    props = torch.cuda.get_device_properties(device)
    return float(props.multi_processor_count * props.max_threads_per_multi_processor)


def _collect_device_weights(device):
    local = torch.tensor([_device_weight(device)], device=device, dtype=torch.float64)
    gathered = [torch.empty_like(local) for _ in range(2)]
    dist.all_gather(gathered, local)
    return tuple(float(value) for tensor in gathered for value in tensor.tolist())


def _plane(value, axis, index):
    return value.select(axis, index)


def _rotate_in_place(value, angle, scratch):
    if angle is None or angle == 0.0:
        return
    scratch.copy_(value)
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    value[..., 0].copy_(scratch[..., 0]).mul_(cosine)
    value[..., 0].add_(scratch[..., 1], alpha=-sine)
    value[..., 1].copy_(scratch[..., 0]).mul_(sine)
    value[..., 1].add_(scratch[..., 1], alpha=cosine)


class TorchHaloExchange:
    """Persistent CUDA halo buffers and overlapped two-rank NCCL scheduling."""

    def __init__(self, simulation, decomposition, *, group=None):
        """Allocate persistent send, receive, and phase-rotation buffers.

        Args:
            simulation: Rank-local Torch simulation whose fields are exchanged.
            decomposition: Two-rank domain decomposition.
            group: Optional initialized PyTorch distributed process group.
        """

        self.simulation = simulation
        self.decomposition = decomposition
        self.axis = decomposition.axis
        self.rank = simulation.runtime.launch.rank
        self.peer = 1 - self.rank
        self.group = dist.group.WORLD if group is None else group
        self._buffers = {}
        self._scratch = {}
        self._works = None
        self._phase = None
        self.profile_annotations = False
        for phase, prefix in (("magnetic", "H"), ("electric", "E")):
            for component in _HALO_COMPONENTS[self.axis]:
                name = prefix + component[1]
                field = simulation.state.field(name)
                template = _plane(field, self.axis, 0)
                key = (phase, name)
                self._buffers[key] = (
                    torch.empty_like(template),
                    torch.empty_like(template),
                )
                if simulation.state.paired_real:
                    self._scratch[key] = torch.empty_like(template)

    def buffers(self):
        """Yield every persistent tensor owned by the exchange."""

        for send, receive in self._buffers.values():
            yield send
            yield receive
        yield from self._scratch.values()

    def _send_index(self, phase):
        return -1 if phase == "magnetic" else 0

    def _receive_index(self, phase):
        return 0 if phase == "magnetic" else -1

    def _wrap_angle(self, phase):
        bloch = self.simulation.plan.bloch
        if bloch is None:
            return None
        crosses_wrap = (phase == "magnetic" and self.rank == 1) or (
            phase == "electric" and self.rank == 0
        )
        if not crosses_wrap:
            return None
        direction = -1.0 if phase == "magnetic" else 1.0
        length = (
            self.decomposition.global_shape[self.axis]
            * self.simulation.plan.dr[self.axis]
        )
        return direction * bloch[self.axis] * length

    def begin(self, phase):
        """Pack a boundary and launch nonblocking peer communication.

        Args:
            phase: Either "magnetic" or "electric".

        Raises:
            ValueError: If the phase name is unsupported.
            TorchDistributedError: If an earlier exchange is still active.
        """

        if phase not in {"magnetic", "electric"}:
            raise ValueError("halo phase must be 'magnetic' or 'electric'")
        if self._works is not None:
            raise TorchDistributedError("a halo exchange is already active")
        context = (
            torch.profiler.record_function(f"gmes::halo_{phase}_pack_launch")
            if self.profile_annotations
            else nullcontext()
        )
        with context:
            prefix = "H" if phase == "magnetic" else "E"
            angle = self._wrap_angle(phase)
            operations = []
            for component in _HALO_COMPONENTS[self.axis]:
                name = prefix + component[1]
                key = (phase, name)
                send, receive = self._buffers[key]
                send.copy_(
                    _plane(
                        self.simulation.state.field(name),
                        self.axis,
                        self._send_index(phase),
                    )
                )
                if angle is not None:
                    _rotate_in_place(send, angle, self._scratch[key])
                operations.extend(
                    (
                        dist.P2POp(dist.isend, send, self.peer, group=self.group),
                        dist.P2POp(dist.irecv, receive, self.peer, group=self.group),
                    )
                )
            self._works = dist.batch_isend_irecv(operations)
            self._phase = phase

    def finish(self, phase):
        """Wait for the active exchange and apply received boundary values.

        Args:
            phase: Phase passed to the matching begin call.

        Raises:
            TorchDistributedError: If no matching exchange is active.
        """

        if phase != self._phase or self._works is None:
            raise TorchDistributedError(
                "halo completion does not match the active phase"
            )
        try:
            wait_context = (
                torch.profiler.record_function(f"gmes::halo_{phase}_exposed_wait")
                if self.profile_annotations
                else nullcontext()
            )
            with wait_context:
                for work in self._works:
                    work.wait()
            unpack_context = (
                torch.profiler.record_function(f"gmes::halo_{phase}_boundary_unpack")
                if self.profile_annotations
                else nullcontext()
            )
            with unpack_context:
                self._apply_dense_boundary_correction(phase)
                for component in _HALO_COMPONENTS[self.axis]:
                    prefix = "H" if phase == "magnetic" else "E"
                    name = prefix + component[1]
                    _, receive = self._buffers[(phase, name)]
                    _plane(
                        self.simulation.state.field(name),
                        self.axis,
                        self._receive_index(phase),
                    ).copy_(receive)
        except Exception as error:
            raise TorchDistributedError(
                f"rank {self.rank} failed the {phase} NCCL halo exchange"
            ) from error
        finally:
            self._works = None
            self._phase = None

    def _delta(self, phase, name):
        _, receive = self._buffers[(phase, name)]
        ghost = _plane(
            self.simulation.state.field(name), self.axis, self._receive_index(phase)
        )
        return receive - ghost

    def _apply_dense_boundary_correction(self, phase):
        sim = self.simulation
        state = sim.state
        plan = sim.plan
        axis = self.axis
        if phase == "magnetic":
            if axis == 0:
                state.ey[0, :, :-1].addcmul_(
                    plan.inv_eps_ey[0, :, :-1],
                    self._delta(phase, "Hz")[1:, :],
                    value=plan.dt / plan.dr[0],
                )
                state.ez[0, :-1, :].addcmul_(
                    plan.inv_eps_ez[0, :-1, :],
                    self._delta(phase, "Hy")[:, 1:],
                    value=-plan.dt / plan.dr[0],
                )
            elif axis == 1:
                state.ex[:, 0, :-1].addcmul_(
                    plan.inv_eps_ex[:, 0, :-1],
                    self._delta(phase, "Hz")[1:, :],
                    value=-plan.dt / plan.dr[1],
                )
                state.ez[:-1, 0, :].addcmul_(
                    plan.inv_eps_ez[:-1, 0, :],
                    self._delta(phase, "Hx")[:, 1:],
                    value=plan.dt / plan.dr[1],
                )
            else:
                state.ex[:, :-1, 0].addcmul_(
                    plan.inv_eps_ex[:, :-1, 0],
                    self._delta(phase, "Hy")[1:, :],
                    value=plan.dt / plan.dr[2],
                )
                state.ey[:-1, :, 0].addcmul_(
                    plan.inv_eps_ey[:-1, :, 0],
                    self._delta(phase, "Hx")[:, 1:],
                    value=-plan.dt / plan.dr[2],
                )
            return
        if axis == 0:
            state.hy[-1, :, 1:].addcmul_(
                plan.inv_mu_hy[-1, :, 1:],
                self._delta(phase, "Ez")[:-1, :],
                value=plan.dt / plan.dr[0],
            )
            state.hz[-1, 1:, :].addcmul_(
                plan.inv_mu_hz[-1, 1:, :],
                self._delta(phase, "Ey")[:, :-1],
                value=-plan.dt / plan.dr[0],
            )
        elif axis == 1:
            state.hx[:, -1, 1:].addcmul_(
                plan.inv_mu_hx[:, -1, 1:],
                self._delta(phase, "Ez")[:-1, :],
                value=-plan.dt / plan.dr[1],
            )
            state.hz[1:, -1, :].addcmul_(
                plan.inv_mu_hz[1:, -1, :],
                self._delta(phase, "Ex")[:, :-1],
                value=plan.dt / plan.dr[1],
            )
        else:
            state.hx[:, 1:, -1].addcmul_(
                plan.inv_mu_hx[:, 1:, -1],
                self._delta(phase, "Ey")[:-1, :],
                value=plan.dt / plan.dr[2],
            )
            state.hy[1:, :, -1].addcmul_(
                plan.inv_mu_hy[1:, :, -1],
                self._delta(phase, "Ex")[:, :-1],
                value=-plan.dt / plan.dr[2],
            )


class TorchDistributedSimulation:
    """Two-rank facade over non-replicated rank-local :class:`TorchSimulation`."""

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
        split_axis=None,
        cut=None,
        timeout_seconds=120,
        require_peer_access=False,
        _decomposition=None,
        _is_auxiliary=False,
    ):
        """Initialize a non-replicated two-rank CUDA simulation.

        Args:
            space: Global Cartesian simulation space.
            geometry: Legacy geometric objects to lower on each rank.
            runtime: Runtime configuration with a two-rank distributed launch.
            courant_ratio: Fraction of the Courant stability limit.
            dt: Optional explicit time step; inferred from space when omitted.
            bloch: Optional three-component Bloch wave vector.
            sources: Source specifications in global coordinates.
            probes: Probe specifications in global coordinates.
            split_axis: Optional Cartesian axis index to force.
            cut: Optional global cell index at which to split the domain.
            timeout_seconds: NCCL process-group initialization timeout.
            require_peer_access: Require direct CUDA peer access between devices.

        Raises:
            TorchConfigurationError: If ranks disagree or launch requirements fail.
        """

        if not isinstance(runtime, TorchRuntimeConfig):
            raise TypeError("runtime must be a TorchRuntimeConfig")
        device = _initialize_nccl(
            runtime,
            timeout_seconds,
            require_peer_access=require_peer_access,
        )
        device_weights = _collect_device_weights(device)
        decomposition = _decomposition or choose_two_gpu_decomposition(
            space,
            geometry,
            sources=sources,
            device_weights=device_weights,
            split_axis=split_axis,
            cut=cut,
        )
        identity = torch.tensor(
            list(bytes.fromhex(decomposition.identity)),
            dtype=torch.uint8,
            device=device,
        )
        identities = [torch.empty_like(identity) for _ in range(2)]
        dist.all_gather(identities, identity)
        if any(not torch.equal(identity, item) for item in identities):
            raise TorchConfigurationError(
                "ranks selected different domain decompositions"
            )
        self.decomposition = decomposition
        self.runtime = runtime
        self.rank = runtime.launch.rank
        self.group = dist.group.WORLD
        local_space = rank_local_space(space, decomposition, self.rank)
        local_probes = self._localize_probes(probes, space, local_space)

        def auxiliary_factory(**kwargs):
            auxiliary_space = kwargs.pop("space")
            auxiliary_runtime = kwargs.pop("runtime")
            serial_runtime = replace(
                auxiliary_runtime,
                launch=DistributedLaunch(),
            )
            return TorchSimulation(
                space=auxiliary_space,
                runtime=serial_runtime,
                **kwargs,
            )

        self.local = TorchSimulation(
            space=local_space,
            geometry=geometry,
            runtime=runtime,
            courant_ratio=courant_ratio,
            dt=dt,
            bloch=bloch,
            sources=sources,
            probes=local_probes,
            _is_auxiliary=_is_auxiliary,
            _distributed_partition=decomposition,
            _auxiliary_factory=auxiliary_factory,
        )
        self.exchange = TorchHaloExchange(self.local, decomposition, group=self.group)
        self.local._distributed_exchange = self.exchange
        self.plan_identity = hashlib.sha256(
            (self.local.plan_identity + decomposition.identity).encode()
        ).hexdigest()

    def __getattr__(self, name):
        if name == "local":
            raise AttributeError(name)
        return getattr(self.local, name)

    def _localize_probes(self, probes, global_space, local_space):
        result = []
        axis = self.decomposition.axis
        offset = self.decomposition.offset(self.rank)[axis]
        local_n = self.decomposition.local_shape(self.rank)[axis]
        cut = self.decomposition.cut
        global_n = self.decomposition.global_shape[axis]
        for spec in probes:
            if not isinstance(spec, TorchProbeSpec):
                raise TypeError("probes must contain TorchProbeSpec values")
            name = (
                spec.component
                if isinstance(spec.component, str)
                else spec.component.__name__
            )
            if spec.coordinates == "space":
                method = getattr(global_space, f"space_to_{name.lower()}_index")
                global_index = list(method(*spec.location))
            else:
                global_index = [int(value) for value in spec.location]
            position = global_index[axis]
            perpendicular = name[1].lower() != _AXIS_NAMES[axis]
            if not perpendicular:
                owner = 0 if position < cut else 1
            elif name.startswith("E"):
                owner = 0 if position < cut else 1
                if position == global_n:
                    owner = 0
                    global_index[axis] = 0
            else:
                owner = 0 if position <= cut else 1
                if position == 0:
                    owner = 1
                    global_index[axis] = global_n
            if owner != self.rank:
                continue
            global_index[axis] -= offset
            if not 0 <= global_index[axis] <= local_n:
                continue
            result.append(
                TorchProbeSpec(
                    spec.component,
                    tuple(global_index),
                    capacity=spec.capacity,
                    coordinates="index",
                )
            )
        return tuple(result)

    def advance(self, steps):
        """Advance the distributed simulation by a number of complete steps."""

        try:
            self.local.advance(steps)
        except Exception:
            self._abort_group()
            raise
        return self

    def step(self):
        """Advance the distributed simulation by one complete step."""

        return self.advance(1)

    def load_host_fields(self, fields):
        """Copy complete global host fields into rank-local slabs and ghosts."""
        if set(fields) != set(COMPONENTS):
            raise ValueError("host fields must contain Ex, Ey, Ez, Hx, Hy, and Hz")
        axis = self.decomposition.axis
        offset = self.decomposition.offset(self.rank)[axis]
        local_fields = {}
        for name, values in fields.items():
            source = np.asarray(values)
            expected = list(self.decomposition.global_shape)
            component_axis = _AXIS_NAMES.index(name[1].lower())
            for current_axis in range(3):
                if current_axis != component_axis:
                    expected[current_axis] += 1
            if tuple(source.shape) != tuple(expected):
                raise ValueError(
                    f"global field {name} has shape {tuple(source.shape)}; "
                    f"expected {tuple(expected)}"
                )
            slices = [slice(None)] * 3
            local_length = self.plan.shapes[name][axis]
            slices[axis] = slice(offset, offset + local_length)
            local_fields[name] = source[tuple(slices)]
        self.local.load_host_fields(local_fields)
        return self

    def flush_probes(self):
        """Return this rank's explicitly synchronized owned probe samples."""
        return {
            "rank": self.rank,
            "samples": self.local.flush_probes(),
        }

    def capture_cuda_graphs(self):
        """Collectively capture only rank-local fixed compute regions."""
        try:
            dist.barrier(group=self.group)
            self.local.capture_cuda_graphs()
            dist.barrier(group=self.group)
        except Exception:
            self._abort_group()
            raise
        return self

    def checkpoint(self):
        """Return rank-local tensors and decomposition metadata for restart."""

        return {
            "format": "gmes.torch.distributed",
            "version": 1,
            "rank": self.rank,
            "world_size": 2,
            "decomposition": self.decomposition.metadata(),
            "plan_identity": self.plan_identity,
            "local": self.local.checkpoint(),
        }

    def load_checkpoint(self, checkpoint):
        """Restore a collectively compatible rank-local checkpoint.

        Args:
            checkpoint: Mapping returned by checkpoint on this rank.

        Raises:
            TorchDistributedError: If any rank reports incompatible metadata.
        """

        valid = (
            isinstance(checkpoint, dict)
            and checkpoint.get("format") == "gmes.torch.distributed"
            and checkpoint.get("version") == 1
            and checkpoint.get("rank") == self.rank
            and checkpoint.get("world_size") == 2
            and checkpoint.get("plan_identity") == self.plan_identity
            and checkpoint.get("decomposition", {}).get("identity")
            == self.decomposition.identity
        )
        status = torch.tensor(int(not valid), device=self.device, dtype=torch.int32)
        dist.all_reduce(status, op=dist.ReduceOp.MAX, group=self.group)
        if int(status.cpu()):
            raise TorchDistributedError(
                "distributed checkpoint metadata does not match every rank"
            )
        self.local.load_checkpoint(checkpoint["local"])
        return self

    def global_field_snapshot(self, *, root=0, numpy=True, complex_fields=True):
        """Explicitly gather uneven owned CUDA slabs and reconstruct global fields."""
        result = {}
        for name in COMPONENTS:
            owned = self._owned_field(name).contiguous()
            length = torch.tensor(
                [owned.numel()], device=self.device, dtype=torch.int64
            )
            lengths = [torch.empty_like(length) for _ in range(2)]
            dist.all_gather(lengths, length, group=self.group)
            sizes = [int(value.cpu()) for value in lengths]
            maximum = max(sizes)
            padded = torch.zeros(maximum, device=self.device, dtype=owned.dtype)
            padded[: owned.numel()].copy_(owned.reshape(-1))
            gathered = [torch.empty_like(padded) for _ in range(2)]
            dist.all_gather(gathered, padded, group=self.group)
            if self.rank != root:
                continue
            slabs = []
            for rank, values in enumerate(gathered):
                shape = self._owned_shape(name, rank)
                slabs.append(values[: sizes[rank]].reshape(shape))
            joined = torch.cat(slabs, dim=self.decomposition.axis)
            host = joined.detach().to(device="cpu", copy=True)
            if self.state.paired_real and complex_fields:
                host = torch.complex(host[..., 0], host[..., 1])
            result[name] = host.numpy() if numpy else host
        return result if self.rank == root else None

    def _owned_shape(self, name, rank):
        shape = list(self.plan.shapes[name])
        local_n = self.decomposition.local_shape(rank)[self.decomposition.axis]
        perpendicular = name[1].lower() != _AXIS_NAMES[self.decomposition.axis]
        shape[self.decomposition.axis] = local_n
        if perpendicular and rank == 0 and name.startswith("H"):
            shape[self.decomposition.axis] += 1
        if perpendicular and rank == 1 and name.startswith("E"):
            shape[self.decomposition.axis] += 1
        if self.state.paired_real:
            shape.append(2)
        return tuple(shape)

    def _owned_field(self, name):
        field = self.state.field(name)
        axis = self.decomposition.axis
        perpendicular = name[1].lower() != _AXIS_NAMES[axis]
        if not perpendicular:
            return field
        slices = [slice(None)] * field.ndim
        if name.startswith("E"):
            if self.rank == 0:
                slices[axis] = slice(0, -1)
        elif self.rank == 1:
            slices[axis] = slice(1, None)
        return field[tuple(slices)]

    def diagnostics(self):
        """Return device, NCCL, decomposition, and halo-allocation diagnostics."""

        props = torch.cuda.get_device_properties(self.device)
        return {
            "rank": self.rank,
            "device": str(self.device),
            "device_name": props.name,
            "device_memory": props.total_memory,
            "device_capability": (props.major, props.minor),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nccl": torch.cuda.nccl.version(),
            "peer_access": torch.cuda.can_device_access_peer(self.rank, 1 - self.rank),
            "decomposition": self.decomposition.metadata(),
            "local_field_shape": tuple(
                int(value) for value in self.space.my_field_size
            ),
            "halo_bytes": sum(
                value.numel() * value.element_size()
                for value in self.exchange.buffers()
            ),
        }

    @staticmethod
    def _abort_group():
        if not dist.is_initialized():
            return
        group = dist.group.WORLD
        abort = getattr(group, "abort", None)
        if abort is not None:
            abort()
        else:
            dist.destroy_process_group()

    @staticmethod
    def close():
        """Destroy the active PyTorch distributed process group, if any."""

        if dist.is_initialized():
            dist.destroy_process_group()


__all__ = [
    "TorchDistributedError",
    "TorchDistributedSimulation",
    "TorchHaloExchange",
    "TwoGpuDecomposition",
    "choose_two_gpu_decomposition",
    "distributed_launch_from_environment",
    "rank_local_space",
]
