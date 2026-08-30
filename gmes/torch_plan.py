"""Occupancy-aware host lowering for Torch material execution plans."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from .constant import Ex, Ey, Ez, Hx, Hy, Hz
from .geometry import Cartesian, GeomBoxTree
from .material import (
    Const,
    Cpml,
    DcpAde,
    DcpPlrc,
    DcpRc,
    Dielectric,
    Dm2,
    Drude,
    Dummy,
    Lorentz,
    Upml,
)

COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
ELECTRIC_COMPONENTS = frozenset(("Ex", "Ey", "Ez"))
COMPONENT_TYPES = {"Ex": Ex, "Ey": Ey, "Ez": Ez, "Hx": Hx, "Hy": Hy, "Hz": Hz}
SINGLE_BUCKET_MODELS = frozenset(
    ("upml", "cpml", "drude", "lorentz", "dcp-ade", "dcp-plrc", "dcp-rc")
)
EXECUTION_POLICIES = frozenset(("auto", "dense", "compact", "tiled"))

type Shape3 = tuple[int, int, int]
type Bounds3 = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
type StencilCoordinate = tuple[str, Shape3, Shape3, int, int]
type EstimatedCosts = tuple[tuple[str, float], ...]


def _readonly(values: Any, dtype: Any = None) -> Any:
    result = np.ascontiguousarray(values, dtype=dtype)
    result.flags.writeable = False
    return result


def _cpml_residual_is_numerically_stable(cell_coefficients: Any, precision: Any) -> Any:
    """Return whether base-plus-residual preserves direct reciprocal kappa."""
    dtype = np.dtype(precision)
    kappas = np.asarray(cell_coefficients[:, (3, 6)], dtype=dtype)
    direct = np.asarray(1.0, dtype=dtype) / kappas
    residual = np.asarray(
        1.0 / cell_coefficients[:, (3, 6)] - 1.0,
        dtype=dtype,
    )
    reconstructed = np.asarray(
        np.asarray(1.0, dtype=dtype) + residual,
        dtype=dtype,
    )
    return bool(
        np.all(
            np.isclose(
                reconstructed,
                direct,
                rtol=8.0 * np.finfo(dtype).eps,
                atol=0.0,
            )
        )
    )


@dataclass(frozen=True, order=True)
class ExecutionSignature:
    """Object-independent material/state execution identity."""

    model: str
    component: str
    precision: str
    state_shape: tuple[int, ...]


@dataclass(frozen=True)
class FlattenedStencilTerm:
    """One signed finite-difference term in flattened-source coordinates."""

    source: str
    source_shape: Shape3
    source_strides: Shape3
    positive_offset: int
    negative_offset: int
    scale_axis: int
    sign: int


@dataclass(frozen=True)
class CpmlResidualAxisPlan:
    """One active CPML curl axis lowered as a sparse residual update."""

    axis: int
    positions: np.ndarray
    targets: np.ndarray
    stencil_indices: np.ndarray
    parameters: np.ndarray

    def __post_init__(self) -> None:
        if self.axis not in (0, 1):
            raise ValueError("a CPML residual axis must be zero or one")
        count = len(self.targets)
        if (
            self.positions.dtype != np.int64
            or self.positions.shape != (count,)
            or self.targets.dtype != np.int64
            or self.targets.shape != (count,)
        ):
            raise ValueError("CPML residual rows and targets must be flat int64 arrays")
        if self.stencil_indices.shape != (count, 2):
            raise ValueError("a CPML residual axis requires two stencil indices")
        if self.stencil_indices.dtype != np.int64:
            raise ValueError("CPML residual stencil indices must use int64")
        if self.parameters.shape != (count, 4) or self.parameters.dtype != np.float64:
            raise ValueError(
                "CPML residual parameters must contain inv_base, b, c, and 1/kappa-1"
            )
        for values in (
            self.positions,
            self.targets,
            self.stencil_indices,
            self.parameters,
        ):
            if values.flags.writeable or not values.flags.c_contiguous:
                raise ValueError(
                    "CPML residual arrays must be immutable and contiguous"
                )


@dataclass(frozen=True)
class MaterialBucketPlan:
    """One normalized signature with dense, compact, and tiled layouts."""

    signature: ExecutionSignature
    selected_policy: str
    decision: str
    estimated_costs: EstimatedCosts
    occupancy: float
    fragmentation: float
    target_count: int
    state_width: int
    padded_state_width: int
    padding_elements_avoided: int
    width_decision: str
    coefficient_names: tuple[str, ...]
    targets: np.ndarray
    target_region_indices: np.ndarray
    region_keys: np.ndarray
    region_coefficient_indices: np.ndarray
    coefficient_table: np.ndarray
    cell_coefficient_names: tuple[str, ...]
    cell_coefficients: np.ndarray
    stencil_indices: np.ndarray
    cpml_residual_axes: tuple[CpmlResidualAxisPlan, ...]
    tile_origins: np.ndarray
    tile_region_indices: np.ndarray
    estimated_bytes: int

    def __post_init__(self) -> None:
        if self.selected_policy not in EXECUTION_POLICIES - {"auto"}:
            raise ValueError("a material bucket must have a concrete execution policy")
        if self.targets.dtype != np.int64 or self.targets.ndim != 1:
            raise ValueError("bucket targets must be a flat int64 array")
        if self.target_region_indices.shape != self.targets.shape:
            raise ValueError("every bucket target must map to one region key")
        if self.region_keys.ndim != 2 or self.region_keys.shape[1] != 2:
            raise ValueError("region keys must contain material and underlying IDs")
        if len(self.region_coefficient_indices) != len(self.region_keys):
            raise ValueError("every region key must map to one coefficient row")
        if self.coefficient_table.ndim != 2 or self.coefficient_table.shape[1] != len(
            self.coefficient_names
        ):
            raise ValueError("coefficient table width does not match its descriptor")
        if (
            self.cell_coefficients.ndim != 2
            or self.cell_coefficients.shape[0] not in (0, self.target_count)
            or self.cell_coefficients.shape[1] != len(self.cell_coefficient_names)
        ):
            raise ValueError(
                "cell coefficients must be empty or completely described per target"
            )
        if self.stencil_indices.shape not in ((0, 4), (self.target_count, 4)):
            raise ValueError("stencil indices must be empty or cover every target")
        if self.cpml_residual_axes:
            if self.signature.model != "cpml" or len(self.cpml_residual_axes) != 2:
                raise ValueError(
                    "only CPML buckets may define both sparse residual axes"
                )
            for expected_axis, residual in enumerate(self.cpml_residual_axes):
                if residual.axis != expected_axis:
                    raise ValueError("CPML residual axes must retain curl-term order")
                if len(residual.positions) and (
                    residual.positions[0] < 0
                    or residual.positions[-1] >= self.target_count
                    or np.any(np.diff(residual.positions) <= 0)
                ):
                    raise ValueError("CPML residual rows must be sorted and unique")
                if not np.array_equal(
                    residual.targets, self.targets[residual.positions]
                ):
                    raise ValueError("CPML residual targets changed bucket row order")
        if self.tile_region_indices.ndim != 2:
            raise ValueError("tile-dense region indices must be two-dimensional")
        if len(self.tile_origins) != len(self.tile_region_indices):
            raise ValueError("every tile-dense row must have one flattened origin")
        for values in (
            self.targets,
            self.target_region_indices,
            self.region_keys,
            self.region_coefficient_indices,
            self.coefficient_table,
            self.cell_coefficients,
            self.stencil_indices,
            self.tile_origins,
            self.tile_region_indices,
        ):
            if values.flags.writeable or not values.flags.c_contiguous:
                raise ValueError("bucket arrays must be immutable and contiguous")

    @property
    def launch_count(self) -> int:
        """Return the steady-state material launch estimate for this signature."""
        if self.target_count == 0 or self.signature.model == "dummy":
            return 0
        if self.cpml_residual_axes:
            return sum(bool(len(axis.targets)) for axis in self.cpml_residual_axes)
        if (
            self.selected_policy == "tiled"
            and self.signature.model not in SINGLE_BUCKET_MODELS
        ):
            return len(self.tile_origins)
        return 1


@dataclass(frozen=True)
class ComponentPlan:
    """Immutable ownership and execution plan for one Yee component."""

    name: str
    shape: Shape3
    active_bounds: Bounds3
    stencil: tuple[FlattenedStencilTerm, ...]
    material_ids: np.ndarray
    underlying_ids: np.ndarray
    ownership: np.ndarray
    dense_inverse: np.ndarray
    constant_targets: np.ndarray
    constant_values: np.ndarray
    buckets: tuple[MaterialBucketPlan, ...]
    requested_policy: str
    tile_size: int

    def __post_init__(self) -> None:
        if self.name not in COMPONENTS or len(self.shape) != 3:
            raise ValueError("unknown Yee component plan")
        if self.requested_policy not in EXECUTION_POLICIES:
            raise ValueError("unknown execution policy")
        total = int(np.prod(self.shape))
        expected_shape = self.shape
        arrays = (
            (self.material_ids, np.int32),
            (self.underlying_ids, np.int32),
            (self.ownership, np.int16),
            (self.dense_inverse, np.float64),
        )
        for values, dtype in arrays:
            if (
                values.shape != expected_shape
                or values.dtype != dtype
                or values.flags.writeable
                or not values.flags.c_contiguous
            ):
                raise ValueError("component planes must be immutable contiguous arrays")
        if (
            self.constant_targets.dtype != np.int64
            or self.constant_targets.ndim != 1
            or self.constant_values.shape != (len(self.constant_targets), 2)
            or self.constant_values.dtype != np.float64
        ):
            raise ValueError("constant targets require paired real values")
        if len(self.constant_targets) and (
            self.constant_targets[0] < 0 or self.constant_targets[-1] >= total
        ):
            raise ValueError("constant target is outside the component")

        active = _active_targets(self.name, self.shape)
        owned = np.flatnonzero(self.ownership.reshape(-1) >= 0)
        if not np.array_equal(active, owned):
            raise ValueError(
                "component ownership does not exactly cover active targets"
            )
        bucket_targets = (
            np.sort(np.concatenate([bucket.targets for bucket in self.buckets]))
            if self.buckets
            else np.empty(0, dtype=np.int64)
        )
        if not np.array_equal(active, bucket_targets):
            raise ValueError("material buckets must uniquely cover every active target")
        if len(bucket_targets) != len(np.unique(bucket_targets)):
            raise ValueError("material buckets contain duplicate write targets")
        flat_material = self.material_ids.reshape(-1)
        flat_underlying = self.underlying_ids.reshape(-1)
        for owner, bucket in enumerate(self.buckets):
            if not np.all(self.ownership.reshape(-1)[bucket.targets] == owner):
                raise ValueError("bucket ownership plane does not match target arrays")
            keys = np.column_stack(
                (flat_material[bucket.targets], flat_underlying[bucket.targets])
            )
            if not np.array_equal(
                keys, bucket.region_keys[bucket.target_region_indices]
            ):
                raise ValueError("bucket region indirection changed geometry mapping")
        inactive = np.ones(total, dtype=np.bool_)
        inactive[active] = False
        if np.any(self.dense_inverse.reshape(-1)[inactive]):
            raise ValueError(
                "dense coefficient plane must preserve inactive boundaries"
            )

    @property
    def active_count(self) -> int:
        """Number of complete, uniquely owned update destinations."""
        return int(np.count_nonzero(self.ownership >= 0))

    @property
    def launch_count(self) -> int:
        """Object-independent material launch estimate for this component."""
        dense = np.any(self.dense_inverse)
        non_dense = sum(
            bucket.launch_count
            for bucket in self.buckets
            if bucket.signature.model != "dielectric"
        )
        return int(dense) + non_dense

    def decision_record(self) -> dict[str, object]:
        """Return JSON-compatible policy diagnostics for benchmarks and debugging."""
        return {
            "component": self.name,
            "shape": self.shape,
            "active_bounds": self.active_bounds,
            "requested_policy": self.requested_policy,
            "active_cells": self.active_count,
            "launches": self.launch_count,
            "buckets": [
                {
                    "signature": {
                        "model": bucket.signature.model,
                        "component": bucket.signature.component,
                        "precision": bucket.signature.precision,
                        "state_shape": bucket.signature.state_shape,
                    },
                    "selected_policy": bucket.selected_policy,
                    "decision": bucket.decision,
                    "targets": bucket.target_count,
                    "occupancy": bucket.occupancy,
                    "fragmentation": bucket.fragmentation,
                    "state_width": bucket.state_width,
                    "padded_state_width": bucket.padded_state_width,
                    "padding_elements_avoided": bucket.padding_elements_avoided,
                    "width_decision": bucket.width_decision,
                    "estimated_costs": dict(bucket.estimated_costs),
                    "estimated_bytes": bucket.estimated_bytes,
                    "coefficient_rows": len(bucket.coefficient_table),
                    "cell_coefficient_width": len(bucket.cell_coefficient_names),
                    "cpml_residual_axis_targets": tuple(
                        len(axis.targets) for axis in bucket.cpml_residual_axes
                    ),
                }
                for bucket in self.buckets
            ],
        }


def _active_bounds(name: Any, shape: Any) -> Any:
    if name == "Ex":
        return ((0, shape[0]), (0, shape[1] - 1), (0, shape[2] - 1))
    if name == "Ey":
        return ((0, shape[0] - 1), (0, shape[1]), (0, shape[2] - 1))
    if name == "Ez":
        return ((0, shape[0] - 1), (0, shape[1] - 1), (0, shape[2]))
    if name == "Hx":
        return ((0, shape[0]), (1, shape[1]), (1, shape[2]))
    if name == "Hy":
        return ((1, shape[0]), (0, shape[1]), (1, shape[2]))
    return ((1, shape[0]), (1, shape[1]), (0, shape[2]))


def _active_targets(name: Any, shape: Any) -> Any:
    bounds = _active_bounds(name, shape)
    mask = np.zeros(shape, dtype=np.bool_)
    slices = tuple(slice(start, stop) for start, stop in bounds)
    mask[slices] = True
    return np.flatnonzero(mask.reshape(-1)).astype(np.int64, copy=False)


_STENCIL_COORDINATES = {
    "Ex": (("Hz", (1, 1, 0), (1, 0, 0), 1, 1), ("Hy", (1, 0, 1), (1, 0, 0), 2, -1)),
    "Ey": (("Hx", (0, 1, 1), (0, 1, 0), 2, 1), ("Hz", (1, 1, 0), (0, 1, 0), 0, -1)),
    "Ez": (("Hy", (1, 0, 1), (0, 0, 1), 0, 1), ("Hx", (0, 1, 1), (0, 0, 1), 1, -1)),
    "Hx": (
        ("Ey", (0, -1, 0), (0, -1, -1), 2, 1),
        ("Ez", (0, 0, -1), (0, -1, -1), 1, -1),
    ),
    "Hy": (
        ("Ez", (0, 0, -1), (-1, 0, -1), 0, 1),
        ("Ex", (-1, 0, 0), (-1, 0, -1), 2, -1),
    ),
    "Hz": (
        ("Ex", (-1, 0, 0), (-1, -1, 0), 1, 1),
        ("Ey", (0, -1, 0), (-1, -1, 0), 0, -1),
    ),
}


def _stencil(name: Any, shapes: Any) -> Any:
    terms = []
    for source, positive, negative, axis, sign in _STENCIL_COORDINATES[name]:
        source_shape = shapes[source]
        strides = (source_shape[1] * source_shape[2], source_shape[2], 1)
        terms.append(
            FlattenedStencilTerm(
                source=source,
                source_shape=source_shape,
                source_strides=strides,
                positive_offset=sum(a * b for a, b in zip(positive, strides)),
                negative_offset=sum(a * b for a, b in zip(negative, strides)),
                scale_axis=axis,
                sign=sign,
            )
        )
    return tuple(terms)


def _model_and_state(material: Any, component: Any) -> Any:
    magnetic = component not in ELECTRIC_COMPONENTS
    if isinstance(material, Dummy):
        return "dummy", ()
    if isinstance(material, Const):
        return "const", ()
    if isinstance(material, Upml):
        return "upml", (1,)
    if isinstance(material, Cpml):
        return "cpml", (2,)
    if magnetic and isinstance(material, (Drude, Lorentz, DcpAde, DcpPlrc, DcpRc, Dm2)):
        return "dielectric", ()
    if type(material) is Dielectric:
        return "dielectric", ()
    if isinstance(material, Drude):
        return "drude", (len(material.dps),)
    if isinstance(material, Lorentz):
        return "lorentz", (len(material.lps),)
    if isinstance(material, DcpAde):
        return "dcp-ade", (len(material.dps), len(material.cps))
    if isinstance(material, DcpRc):
        return "dcp-rc", (len(material.dps), len(material.cps))
    if isinstance(material, DcpPlrc):
        return "dcp-plrc", (len(material.dps), len(material.cps))
    if isinstance(material, Dm2):
        if len(material.omega) != len(material.n_atom):
            raise ValueError("Dm2 omega and n_atom must have equal lengths")
        return "dm2", (len(material.omega),)
    return f"custom:{type(material).__module__}.{type(material).__qualname__}", ()


def _coefficient_descriptor(material: Any, component: Any, underlying: Any) -> Any:
    effective = underlying if underlying is not None else material
    base_name = "inv_eps" if component in ELECTRIC_COMPONENTS else "inv_mu"
    base_parameter = float(
        effective.eps_inf if component in ELECTRIC_COMPONENTS else effective.mu_inf
    )
    if not np.isfinite(base_parameter) or base_parameter <= 0:
        raise ValueError(f"{base_name[4:]} must be finite and positive for {component}")
    base_value = 1.0 / base_parameter
    model, _ = _model_and_state(material, component)
    names = [base_name]
    values = [base_value]
    if model == "const":
        value = complex(material.value)
        names.extend(("value_real", "value_imag"))
        values.extend((value.real, value.imag))
    elif model in {"upml", "cpml"}:
        for name in ("m", "kappa_max", "sigma_max_ratio"):
            names.append(name)
            values.append(float(getattr(material, name)))
        if model == "cpml":
            names.extend(("m_a", "a_max"))
            values.extend((float(material.m_a), float(material.a_max)))
    elif model in {"drude", "lorentz"}:
        for pole, row in enumerate(material.a):
            for term, value in enumerate(row):
                names.append(f"a{pole}_{term}")
                values.append(float(value))
        for term, value in enumerate(material.c):
            names.append(f"c{term}")
            values.append(float(value))
    elif model == "dcp-ade":
        for pole, row in enumerate(material.a):
            for term, value in enumerate(row):
                names.append(f"a{pole}_{term}")
                values.append(float(value))
        for point, row in enumerate(material.b):
            for term, value in enumerate(row):
                names.append(f"b{point}_{term}")
                values.append(float(value))
        for term, value in enumerate(material.c):
            names.append(f"c{term}")
            values.append(float(value))
    elif model in {"dcp-plrc", "dcp-rc"}:
        for pole, row in enumerate(material.a):
            for term, value in enumerate(row):
                names.append(f"a{pole}_{term}")
                values.append(float(value))
        for point, row in enumerate(material.b):
            for term, value in enumerate(row):
                value = complex(value)
                names.extend((f"b{point}_{term}_real", f"b{point}_{term}_imag"))
                values.extend((value.real, value.imag))
        for term, value in enumerate(material.c):
            names.append(f"c{term}")
            values.append(float(value))
    elif model == "dm2":
        names.extend(("rho30", "gamma", "t1", "t2", "hbar", "rtol"))
        values.extend(
            (
                material.rho30,
                material.gamma,
                material.t1,
                material.t2,
                material.hbar,
                material.rtol,
            )
        )
        for index, (omega, density) in enumerate(zip(material.omega, material.n_atom)):
            names.extend((f"transition{index}_omega", f"transition{index}_density"))
            values.extend((omega, density))
    return tuple(names), tuple(float(value) for value in values)


_PML_CELL_COEFFICIENT_NAMES = {
    "upml": ("inv_base", "c1", "c2", "c3", "c4", "c5", "c6"),
    "cpml": ("inv_base", "b1", "c1", "kappa1", "b2", "c2", "kappa2"),
}

_PML_AXES = {
    "Ex": (1, 2, 0),
    "Ey": (2, 0, 1),
    "Ez": (0, 1, 2),
    "Hx": (1, 2, 0),
    "Hy": (2, 0, 1),
    "Hz": (0, 1, 2),
}


def _pml_profile(material: Any, coordinates: Any, axis: Any) -> Any:
    """Vectorize the native PML grading functions over one coordinate axis."""
    offset = coordinates[:, axis] - material.center[axis]
    half_size = material.half_size[axis]
    low = offset <= material.d - half_size
    high = np.logical_and(~low, half_size - material.d <= offset)
    depth = np.zeros(len(coordinates), dtype=np.float64)
    depth[low] = np.clip((half_size + offset[low]) / material.d, 0.0, 1.0)
    depth[high] = np.clip((half_size - offset[high]) / material.d, 0.0, 1.0)
    graded = np.zeros(len(coordinates), dtype=np.float64)
    boundary = np.logical_or(low, high)
    graded[boundary] = (1.0 - depth[boundary]) ** material.m
    sigma = material.sigma_max[axis] * graded
    kappa = 1.0 + (material.kappa_max - 1.0) * graded
    return sigma, kappa, depth, boundary


def _pml_cell_coefficients(
    material: Any, component: Any, coordinates: Any, inverse_base: Any
) -> Any:
    """Lower coordinate-dependent PML parameters without per-cell objects."""
    first_axis, second_axis, field_axis = _PML_AXES[component]
    profiles = {
        axis: _pml_profile(material, coordinates, axis)
        for axis in {first_axis, second_axis, field_axis}
    }
    base = np.full(len(coordinates), inverse_base, dtype=np.float64)
    if isinstance(material, Upml):
        sigma1, kappa1, _, _ = profiles[first_axis]
        sigma2, kappa2, _, _ = profiles[second_axis]
        sigmaf, kappaf, _, _ = profiles[field_axis]

        def pair(sigma: Any, kappa: Any) -> Any:
            denominator = 2.0 * kappa + sigma * material.dt
            return (
                (2.0 * kappa - sigma * material.dt) / denominator,
                2.0 * material.dt / denominator,
                1.0 / denominator,
            )

        c1, c2, _ = pair(sigma1, kappa1)
        c3, _, c4 = pair(sigma2, kappa2)
        c5 = 2.0 * kappaf + sigmaf * material.dt
        c6 = 2.0 * kappaf - sigmaf * material.dt
        return np.column_stack((base, c1, c2, c3, c4, c5, c6))

    rows = [base]
    for axis in (first_axis, second_axis):
        sigma, kappa, depth, boundary = profiles[axis]
        a = np.zeros(len(coordinates), dtype=np.float64)
        a[boundary] = material.a_max * depth[boundary] ** material.m_a
        b = np.exp(-(sigma / kappa + a) * material.dt)
        denominator = (sigma + kappa * a) * kappa
        c = np.zeros(len(coordinates), dtype=np.float64)
        nonzero = denominator != 0.0
        c[nonzero] = sigma[nonzero] * (b[nonzero] - 1.0) / denominator[nonzero]
        rows.extend((b, c, kappa))
    return np.column_stack(rows)


def _pml_stencil_indices(
    component: Any, targets: Any, target_shape: Any, shapes: Any
) -> Any:
    coordinates = np.unravel_index(targets, target_shape)
    columns = []
    stencil: tuple[StencilCoordinate, ...] = _STENCIL_COORDINATES[component]
    if component not in ELECTRIC_COMPONENTS:
        stencil = tuple(reversed(stencil))
    for source, positive, negative, _axis, _sign in stencil:
        for delta in (positive, negative):
            source_coordinates = tuple(
                coordinates[axis] + delta[axis] for axis in range(3)
            )
            columns.append(
                np.ravel_multi_index(source_coordinates, shapes[source]).astype(
                    np.int64, copy=False
                )
            )
    return np.column_stack(columns)


def _state_width(signature: Any) -> Any:
    if signature.model in {"drude", "lorentz"}:
        return 2 * signature.state_shape[0]
    if signature.model == "dcp-ade":
        poles, points = signature.state_shape
        return 1 + 2 * poles + 2 * points
    if signature.model in {"dcp-plrc", "dcp-rc"}:
        poles, points = signature.state_shape
        return poles + 2 * points
    return int(sum(signature.state_shape))


def _select_policy(
    requested: Any,
    *,
    device_type: Any,
    count: Any,
    active_count: Any,
    runs: Any,
    tiles: Any,
    width: Any,
) -> Any:
    occupancy = count / active_count if active_count else 0.0
    fragmentation = runs / count if count else 0.0
    launch = 4096.0 if device_type == "cuda" else 256.0
    tile_cells = tiles
    costs = {
        "dense": active_count * (1.0 + 0.10 * width) + launch,
        "compact": count * (2.15 + 0.18 * width) + launch,
        "tiled": tile_cells * (1.20 + 0.12 * width) + launch * 1.15,
    }
    selected = min(costs, key=costs.__getitem__) if requested == "auto" else requested
    reason = (
        f"{selected} minimizes the static {device_type} cost model at "
        f"occupancy={occupancy:.4f}, fragmentation={fragmentation:.4f}, "
        f"state_width={width}"
        if requested == "auto"
        else f"{selected} was forced for benchmark/debugging"
    )
    return selected, reason, tuple(sorted(costs.items())), occupancy, fragmentation


class TorchExecutionPlanner:
    """Lower GeometryMap tiles into immutable signature-bucketed component plans."""

    def __init__(
        self,
        *,
        geom_tree: GeomBoxTree,
        space: Cartesian,
        shapes: Mapping[str, Shape3],
        precision: Literal["float32", "float64"],
        device_type: str,
        policy: str = "auto",
        material_tile_size: int = 65536,
        execution_tile_size: int = 4096,
        cpml_sparse_residual: bool = False,
    ) -> None:
        if policy not in EXECUTION_POLICIES:
            raise ValueError(
                "execution policy must be 'auto', 'dense', 'compact', or 'tiled'"
            )
        if material_tile_size < 1 or execution_tile_size < 1:
            raise ValueError("planner tile sizes must be positive")
        self.geom_tree = geom_tree
        self.space = space
        self.shapes = MappingProxyType(dict(shapes))
        self.precision = precision
        self.device_type = device_type
        self.policy = policy
        self.material_tile_size = int(material_tile_size)
        self.execution_tile_size = int(execution_tile_size)
        self.cpml_sparse_residual = bool(cpml_sparse_residual)

    def build(self) -> tuple[ComponentPlan, ...]:
        """Build and validate all six component plans before tensor finalization."""
        return tuple(self._lower_component(name) for name in COMPONENTS)

    def _lower_component(self, name: Any) -> Any:
        shape = self.shapes[name]
        component = COMPONENT_TYPES[name]
        axes = self.space.component_coordinate_axes(component, shape)
        total = int(np.prod(shape))
        material_ids = np.empty(total, dtype=np.int32)
        underlying_ids = np.empty(total, dtype=np.int32)
        geometries = None
        for start in range(0, total, self.material_tile_size):
            stop = min(start + self.material_tile_size, total)
            lowered = self.geom_tree.lower_grid(*axes, start, stop, component=component)
            if geometries is None:
                geometries = lowered.geometries
            elif geometries != lowered.geometries:
                raise ValueError("geometry table changed between lowering tiles")
            material_ids[start:stop] = lowered.material_ids
            underlying_ids[start:stop] = lowered.underlying_ids
        if geometries is None:
            lowered = self.geom_tree.lower_grid(*axes, 0, 0, component=component)
            geometries = lowered.geometries
        if np.any(material_ids < 0) or np.any(material_ids >= len(geometries)):
            raise ValueError("geometry lowering did not completely map the component")
        if np.any(underlying_ids < -1) or np.any(underlying_ids >= len(geometries)):
            raise ValueError("geometry lowering returned an invalid underlying region")

        active_targets = _active_targets(name, shape)
        flat_material = material_ids
        flat_underlying = underlying_ids
        signatures = []
        for geometry in geometries:
            model, state_shape = _model_and_state(geometry.material, name)
            signatures.append(
                ExecutionSignature(model, name, self.precision, state_shape)
            )
        unique_signatures = tuple(
            sorted({signatures[index] for index in flat_material[active_targets]})
        )
        signature_widths = {
            signature: _state_width(signature) for signature in unique_signatures
        }
        padded_widths: dict[tuple[str, str, str], int] = {}
        signature_counts = {}
        width_groups: dict[tuple[str, str, str], list[ExecutionSignature]] = {}
        for signature, width in signature_widths.items():
            key = (signature.model, signature.component, signature.precision)
            padded_widths[key] = max(padded_widths.get(key, 0), width)
            width_groups.setdefault(key, []).append(signature)
            region_ids = np.asarray(
                [index for index, item in enumerate(signatures) if item == signature],
                dtype=np.int32,
            )
            signature_counts[signature] = int(
                np.count_nonzero(np.isin(flat_material[active_targets], region_ids))
            )
        width_decisions = {}
        for key, group in width_groups.items():
            exact_elements = sum(
                signature_counts[item] * signature_widths[item] for item in group
            )
            padded_elements = (
                sum(signature_counts[item] for item in group) * padded_widths[key]
            )
            width_decisions[key] = (
                f"exact selected: {len(group)} signature launch(es), "
                f"{exact_elements} state elements; bounded max-width merge: "
                f"1 launch, {padded_elements} elements "
                f"(+{padded_elements - exact_elements})"
            )
        ownership = np.full(total, -1, dtype=np.int16)
        dense_inverse = np.zeros(total, dtype=np.float64)
        buckets = []
        constant_targets = []
        constant_values = []

        for owner, signature in enumerate(unique_signatures):
            region_ids = np.asarray(
                [index for index, item in enumerate(signatures) if item == signature],
                dtype=np.int32,
            )
            selected = np.isin(flat_material[active_targets], region_ids)
            targets = active_targets[selected]
            ownership[targets] = owner
            keys, target_region_indices = np.unique(
                np.column_stack(
                    (flat_material[targets], flat_underlying[targets])
                ).astype(np.int32, copy=False),
                axis=0,
                return_inverse=True,
            )
            descriptor_names = None
            coefficient_rows = []
            for material_id, underlying_id in keys:
                material = geometries[int(material_id)].material
                underlying = (
                    None
                    if underlying_id < 0
                    else geometries[int(underlying_id)].material
                )
                names, row = _coefficient_descriptor(material, name, underlying)
                if descriptor_names is None:
                    descriptor_names = names
                elif descriptor_names != names:
                    raise ValueError(
                        "one execution signature produced ragged coefficients"
                    )
                coefficient_rows.append(row)
            coefficient_row_array = np.asarray(coefficient_rows, dtype=np.float64)
            coefficient_table, region_coefficient_indices = np.unique(
                coefficient_row_array, axis=0, return_inverse=True
            )
            cell_coefficient_names: tuple[str, ...] = ()
            cell_coefficients = np.empty((0, 0), dtype=np.float64)
            stencil_indices = np.empty((0, 4), dtype=np.int64)
            cpml_residual_axes: tuple[CpmlResidualAxisPlan, ...] = ()
            if signature.model in {"upml", "cpml"}:
                cell_coefficient_names = _PML_CELL_COEFFICIENT_NAMES[signature.model]
                linear_coordinates = np.unravel_index(targets, shape)
                target_coordinates = np.column_stack(
                    tuple(axes[axis][linear_coordinates[axis]] for axis in range(3))
                )
                cell_coefficients = np.empty((len(targets), 7), dtype=np.float64)
                for region_index, (material_id, _underlying_id) in enumerate(keys):
                    selected_region = target_region_indices == region_index
                    material = geometries[int(material_id)].material
                    descriptor_row = coefficient_table[
                        region_coefficient_indices[region_index]
                    ]
                    cell_coefficients[selected_region] = _pml_cell_coefficients(
                        material,
                        name,
                        target_coordinates[selected_region],
                        descriptor_row[0],
                    )
                stencil_indices = _pml_stencil_indices(
                    name, targets, shape, self.shapes
                )
                if (
                    signature.model == "cpml"
                    and self.cpml_sparse_residual
                    and _cpml_residual_is_numerically_stable(
                        cell_coefficients, self.precision
                    )
                ):
                    residual_axes: list[CpmlResidualAxisPlan] = []
                    for axis, (b_column, c_column, kappa_column) in enumerate(
                        ((1, 2, 3), (4, 5, 6))
                    ):
                        active = np.logical_or(
                            cell_coefficients[:, c_column] != 0.0,
                            cell_coefficients[:, kappa_column] != 1.0,
                        )
                        positions = np.flatnonzero(active).astype(np.int64, copy=False)
                        parameters = np.column_stack(
                            (
                                cell_coefficients[positions, 0],
                                cell_coefficients[positions, b_column],
                                cell_coefficients[positions, c_column],
                                1.0 / cell_coefficients[positions, kappa_column] - 1.0,
                            )
                        )
                        residual_axes.append(
                            CpmlResidualAxisPlan(
                                axis=axis,
                                positions=_readonly(positions, np.int64),
                                targets=_readonly(targets[positions], np.int64),
                                stencil_indices=_readonly(
                                    stencil_indices[positions, 2 * axis : 2 * axis + 2],
                                    np.int64,
                                ),
                                parameters=_readonly(parameters, np.float64),
                            )
                        )
                    cpml_residual_axes = tuple(residual_axes)

            tile_ids = np.unique(targets // self.execution_tile_size)
            tile_cells = len(tile_ids) * self.execution_tile_size
            selected_positions = np.flatnonzero(selected)
            runs = (
                0
                if not len(targets)
                else 1 + int(np.count_nonzero(np.diff(selected_positions) != 1))
            )
            state_width = signature_widths[signature]
            padded_state_width = padded_widths[
                (signature.model, signature.component, signature.precision)
            ]
            padding_elements_avoided = len(targets) * (padded_state_width - state_width)
            selected_policy, reason, costs, occupancy, fragmentation = _select_policy(
                self.policy,
                device_type=self.device_type,
                count=len(targets),
                active_count=len(active_targets),
                runs=runs,
                tiles=tile_cells,
                width=state_width,
            )
            if selected_policy == "tiled":
                tile_origins = tile_ids.astype(np.int64) * self.execution_tile_size
                tile_region_indices = np.full(
                    (len(tile_origins), self.execution_tile_size),
                    -1,
                    dtype=np.int32,
                )
                tile_rows = np.searchsorted(
                    tile_ids, targets // self.execution_tile_size
                )
                tile_offsets = targets % self.execution_tile_size
                tile_region_indices[tile_rows, tile_offsets] = target_region_indices
            else:
                tile_origins = np.empty(0, dtype=np.int64)
                tile_region_indices = np.empty(
                    (0, self.execution_tile_size), dtype=np.int32
                )
            if signature.model == "dielectric":
                target_coefficients = region_coefficient_indices[target_region_indices]
                dense_inverse[targets] = coefficient_table[target_coefficients, 0]
            elif signature.model == "cpml" and cpml_residual_axes:
                dense_inverse[targets] = cell_coefficients[:, 0]
            elif signature.model == "const":
                target_coefficients = region_coefficient_indices[target_region_indices]
                values = coefficient_table[target_coefficients, 1:3]
                constant_targets.append(targets)
                constant_values.append(values)

            estimated_bytes = sum(
                values.nbytes
                for values in (
                    targets,
                    target_region_indices,
                    keys,
                    region_coefficient_indices,
                    coefficient_table,
                    cell_coefficients,
                    stencil_indices,
                    tile_origins,
                    tile_region_indices,
                )
            ) + sum(
                values.nbytes
                for residual in cpml_residual_axes
                for values in (
                    residual.positions,
                    residual.targets,
                    residual.stencil_indices,
                    residual.parameters,
                )
            )
            buckets.append(
                MaterialBucketPlan(
                    signature=signature,
                    selected_policy=selected_policy,
                    decision=reason,
                    estimated_costs=costs,
                    occupancy=occupancy,
                    fragmentation=fragmentation,
                    target_count=len(targets),
                    state_width=state_width,
                    padded_state_width=padded_state_width,
                    padding_elements_avoided=padding_elements_avoided,
                    width_decision=width_decisions[
                        (signature.model, signature.component, signature.precision)
                    ],
                    coefficient_names=descriptor_names or (),
                    targets=_readonly(targets, np.int64),
                    target_region_indices=_readonly(target_region_indices, np.int32),
                    region_keys=_readonly(keys, np.int32),
                    region_coefficient_indices=_readonly(
                        region_coefficient_indices, np.int32
                    ),
                    coefficient_table=_readonly(coefficient_table, np.float64),
                    cell_coefficient_names=cell_coefficient_names,
                    cell_coefficients=_readonly(cell_coefficients, np.float64),
                    stencil_indices=_readonly(stencil_indices, np.int64),
                    cpml_residual_axes=cpml_residual_axes,
                    tile_origins=_readonly(tile_origins, np.int64),
                    tile_region_indices=_readonly(tile_region_indices, np.int32),
                    estimated_bytes=estimated_bytes,
                )
            )

        if constant_targets:
            constant_target_array = np.concatenate(constant_targets)
            constant_value_array = np.concatenate(constant_values)
            order = np.argsort(constant_target_array)
            constant_target_array = constant_target_array[order]
            constant_value_array = constant_value_array[order]
        else:
            constant_target_array = np.empty(0, dtype=np.int64)
            constant_value_array = np.empty((0, 2), dtype=np.float64)

        return ComponentPlan(
            name=name,
            shape=shape,
            active_bounds=_active_bounds(name, shape),
            stencil=_stencil(name, self.shapes),
            material_ids=_readonly(material_ids.reshape(shape), np.int32),
            underlying_ids=_readonly(underlying_ids.reshape(shape), np.int32),
            ownership=_readonly(ownership.reshape(shape), np.int16),
            dense_inverse=_readonly(dense_inverse.reshape(shape), np.float64),
            constant_targets=_readonly(constant_target_array, np.int64),
            constant_values=_readonly(constant_value_array, np.float64),
            buckets=tuple(buckets),
            requested_policy=self.policy,
            tile_size=self.execution_tile_size,
        )


__all__ = [
    "COMPONENTS",
    "ELECTRIC_COMPONENTS",
    "EXECUTION_POLICIES",
    "ComponentPlan",
    "CpmlResidualAxisPlan",
    "ExecutionSignature",
    "FlattenedStencilTerm",
    "MaterialBucketPlan",
    "TorchExecutionPlanner",
]
