#!/usr/bin/env python3
"""Capture, compare, and benchmark the immutable native FDTD oracle."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import sysconfig
from pathlib import Path
from statistics import median, pstdev
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "native_oracle_workloads.json"
COMPONENT_NAMES = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
FIELD_INITIALIZER = "native-affine-ramp-v1"


def load_manifest(path=DEFAULT_MANIFEST):
    """Load and validate the portable workload and frozen gate description."""
    data = json.loads(Path(path).read_text())
    if data.get("schema_version") != 2:
        raise ValueError("unsupported native oracle workload schema")
    reference = data["reference"]
    observer_commit = reference.get("observer_commit", "")
    if len(observer_commit) != 40 or any(
        character not in "0123456789abcdef" for character in observer_commit
    ):
        raise ValueError("observer_commit must be a full lowercase Git commit")
    performance_summary_sha256 = reference.get("performance_summary_sha256", "")
    if len(performance_summary_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in performance_summary_sha256
    ):
        raise ValueError(
            "performance_summary_sha256 must be a lowercase SHA-256 digest"
        )
    if reference.get("field_initializer") != FIELD_INITIALIZER:
        raise ValueError("unsupported native oracle field initializer")
    for name in (
        "performance_warmup_steps",
        "performance_steps_per_repeat",
        "performance_repetitions",
        "performance_profile_steps",
    ):
        if not isinstance(reference.get(name), int) or reference[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    acceptance = data.get("performance_gates", {}).get("cpu_acceptance", {})
    known_cases = {
        case["name"]
        for group in ("correctness", "benchmarks")
        for case in data.get(group, ())
    }
    cases = acceptance.get("cases")
    if (
        not isinstance(cases, list)
        or not cases
        or len(cases) != len(set(cases))
        or any(case not in known_cases for case in cases)
    ):
        raise ValueError("cpu_acceptance cases must be unique known workloads")
    if acceptance.get("thread_modes") != ["one", "physical"]:
        raise ValueError("cpu_acceptance thread_modes must be one and physical")
    if acceptance.get("precision") != "float64":
        raise ValueError("cpu_acceptance precision must be float64")
    ratio = acceptance.get("max_individual_ratio")
    if not isinstance(ratio, (int, float)) or ratio < 1:
        raise ValueError("cpu_acceptance max_individual_ratio must be at least one")
    statistics = acceptance.get("statistics", {})
    if statistics.get("method") != "independent-stratified-bootstrap-log-geomean-v1":
        raise ValueError("unsupported cpu_acceptance statistics method")
    if not isinstance(statistics.get("resamples"), int) or statistics["resamples"] < 1:
        raise ValueError("cpu_acceptance resamples must be a positive integer")
    if not isinstance(statistics.get("seed"), int):
        raise ValueError("cpu_acceptance seed must be an integer")
    confidence = statistics.get("one_sided_confidence")
    if not isinstance(confidence, (int, float)) or not 0 < confidence < 1:
        raise ValueError("cpu_acceptance confidence must be between zero and one")
    regression_ratio = statistics.get("regression_ratio")
    if not isinstance(regression_ratio, (int, float)) or regression_ratio <= 0:
        raise ValueError("cpu_acceptance regression_ratio must be positive")
    relative_mad = statistics.get("max_relative_mad")
    if not isinstance(relative_mad, (int, float)) or not 0 <= relative_mad < 1:
        raise ValueError("cpu_acceptance max_relative_mad must be in [0, 1)")
    if reference["capture_steps"] != sorted(set(reference["capture_steps"])):
        raise ValueError("capture_steps must be unique and increasing")
    cases = (
        data["correctness"]
        + data.get("benchmarks", [])
        + data.get("physical_checks", [])
    )
    names = [case["name"] for case in cases]
    if len(names) != len(set(names)):
        raise ValueError("workload names must be unique")
    known = set(names)
    for gate_name, gate in data["performance_gates"].items():
        if isinstance(gate, dict) and "cases" in gate:
            unknown = set(gate["cases"]) - known
            if unknown:
                raise ValueError(
                    f"{gate_name} references unknown workloads: {sorted(unknown)}"
                )
    return data


def _json_value(value):
    """Convert coefficient and configuration metadata to stable JSON values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if hasattr(value, "__dict__"):
        return {
            "type": type(value).__name__,
            "values": {
                name: _json_value(item)
                for name, item in sorted(vars(value).items())
                if not name.startswith("_")
                and name not in {"f", "geom_tree", "space", "aux_fdtd"}
            },
        }
    return repr(value)


def _command_output(command):
    try:
        return subprocess.run(
            command, check=True, capture_output=True, text=True
        ).stdout.strip()
    except FileNotFoundError, subprocess.CalledProcessError:
        return None


def _timing_summary(samples):
    values = [float(value) for value in samples]
    sample_median = median(values)
    absolute_deviations = [abs(value - sample_median) for value in values]
    return {
        "raw_seconds": values,
        "median_seconds": sample_median,
        "p95_seconds": _percentile95(values),
        "population_stdev_seconds": pstdev(values),
        "relative_mad": (
            median(absolute_deviations) / sample_median if sample_median else 0.0
        ),
        "repetitions": len(values),
    }


def material_from_name(name, gmes):
    """Build one material strategy from its backend-neutral manifest name."""
    drude_poles = tuple(
        gmes.DrudePole(omega=0.6 + 0.1 * index, gamma=0.03 + 0.01 * index)
        for index in range(4 if name.endswith("-4") else 1)
    )
    lorentz_poles = tuple(
        gmes.LorentzPole(
            amp=0.05 + 0.01 * index,
            omega=0.8 + 0.1 * index,
            gamma=0.03 + 0.01 * index,
        )
        for index in range(4 if name.endswith("-4") else 1)
    )
    critical_points = (
        gmes.CriticalPoint(amp=0.04, phi=0.2, omega=0.9, gamma=0.03),
        gmes.CriticalPoint(amp=0.02, phi=-0.1, omega=1.1, gamma=0.04),
    )
    factories = {
        "dummy": lambda: gmes.Dummy(eps_inf=1.1, mu_inf=1.05),
        "const": lambda: gmes.Const(value=0.25, eps_inf=1.1, mu_inf=1.05),
        "dielectric": lambda: gmes.Dielectric(eps_inf=1.7, mu_inf=1.05),
        "upml": gmes.Upml,
        "cpml": gmes.Cpml,
        "drude-1": lambda: gmes.Drude(eps_inf=1.2, dps=drude_poles),
        "drude-4": lambda: gmes.Drude(eps_inf=1.2, dps=drude_poles),
        "lorentz-1": lambda: gmes.Lorentz(eps_inf=1.2, lps=lorentz_poles),
        "lorentz-4": lambda: gmes.Lorentz(eps_inf=1.2, lps=lorentz_poles),
        "dcp-ade": lambda: gmes.DcpAde(
            eps_inf=1.2, dps=drude_poles[:1], cps=critical_points
        ),
        "dcp-plrc": lambda: gmes.DcpPlrc(
            eps_inf=1.2, dps=drude_poles[:1], cps=critical_points
        ),
        "dcp-rc": lambda: gmes.DcpRc(
            eps_inf=1.2, dps=drude_poles[:1], cps=critical_points
        ),
        "dm2-1": lambda: gmes.Dm2(
            eps_inf=1.2,
            omega=(0.8,),
            n_atom=(0.01,),
            gamma=0.02,
            rtol=1e-4,
        ),
        "dm2-4": lambda: gmes.Dm2(
            eps_inf=1.2,
            omega=(0.7, 0.8, 0.9, 1.0),
            n_atom=(0.01, 0.01, 0.01, 0.01),
            gamma=0.02,
            rtol=1e-4,
        ),
    }
    try:
        return factories[name]()
    except KeyError as error:
        raise ValueError(f"unknown material strategy: {name}") from error


def _mixed_geometry(spec, gmes):
    size = spec["size"]
    width = float(size[0]) / 9
    names = (
        "upml",
        "drude-1",
        "lorentz-1",
        "dcp-ade",
        "dcp-plrc",
        "dcp-rc",
        "dm2-1",
    )
    geometry = [gmes.DefaultMedium(material_from_name("dielectric", gmes))]
    geometry.append(
        gmes.Shell(
            material=material_from_name("cpml", gmes),
            thickness=max(0.25, width / 4),
        )
    )
    z_size = max(float(size[2]) * 0.65, 1.0)
    for index, name in enumerate(names):
        center_x = -float(size[0]) / 2 + (index + 1.5) * width
        region_size = (width * 0.72, float(size[1]) * 0.65, z_size)
        if name == "dm2-1" and float(size[2]) > 0:
            geometry.append(
                gmes.Sphere(
                    material=material_from_name(name, gmes),
                    center=(
                        float(size[0]) / 2 - 0.5 / spec["resolution"],
                        0,
                        0,
                    ),
                    radius=0.2 / spec["resolution"],
                )
            )
        elif name == "upml":
            geometry.append(
                gmes.Shell(
                    material=material_from_name(name, gmes),
                    center=(center_x, 0, 0),
                    size=region_size,
                    thickness=max(0.25, width * 0.2),
                )
            )
        else:
            geometry.append(
                gmes.Block(
                    material=material_from_name(name, gmes),
                    center=(center_x, 0, 0),
                    size=region_size,
                )
            )
    return geometry


def _coverage_geometry(spec, gmes):
    """Build measured contiguous or fragmented material coverage workloads."""
    size = np.maximum(np.asarray(spec["size"], dtype=float), 1.0)
    fraction = float(spec["coverage_percent"]) / 100
    families = spec.get(
        "families",
        [
            "drude-1",
            "lorentz-1",
            "dcp-ade",
            "dcp-plrc",
            "dcp-rc",
            "dm2-1",
        ],
    )
    if float(spec["size"][2]) > 0:
        families = [name for name in families if not name.startswith("dm2-")]
    geometry = [gmes.DefaultMedium(material_from_name("dielectric", gmes))]
    if spec.get("include_pml", True):
        geometry.append(
            gmes.Shell(
                material=material_from_name("cpml", gmes),
                thickness=max(0.25, min(size) * 0.04),
            )
        )
    total_width = max(size[0] * fraction, len(families) / spec["resolution"])
    family_width = total_width / len(families)
    origin = -0.5 * total_width
    fragments = 1 if spec.get("layout") == "contiguous" else 4
    for family_index, family in enumerate(families):
        for fragment in range(fragments):
            width = family_width / fragments
            center_x = (
                origin + (family_index + (fragment + 0.5) / fragments) * family_width
            )
            if fragments == 1:
                center_y = 0.0
                height = size[1]
            else:
                height = size[1] / fragments
                center_y = -0.5 * size[1] + (fragment + 0.5) * height
            geometry.append(
                gmes.Block(
                    material=material_from_name(family, gmes),
                    center=(center_x, center_y, 0),
                    size=(width * 0.92, height * 0.82, size[2]),
                )
            )
    return geometry


def _heterogeneous_geometry(spec, gmes):
    geometry = [gmes.DefaultMedium(material_from_name("dielectric", gmes))]
    count = int(spec.get("region_count", 16))
    side = int(np.ceil(np.sqrt(count)))
    spacing_x = float(spec["size"][0]) / (side + 1)
    spacing_y = max(float(spec["size"][1]), 1.0) / (side + 1)
    for index in range(count):
        x_index, y_index = divmod(index, side)
        geometry.append(
            gmes.Cylinder(
                material=gmes.Dielectric(eps_inf=2.0 + 0.1 * (index % 5)),
                center=(
                    -0.5 * float(spec["size"][0]) + (x_index + 1) * spacing_x,
                    -0.5 * float(spec["size"][1]) + (y_index + 1) * spacing_y,
                    0,
                ),
                axis=(0, 0, 1),
                radius=0.22 * min(spacing_x, spacing_y),
            )
        )
    if spec.get("overlap"):
        geometry.extend(
            gmes.Block(
                material=gmes.Dielectric(eps_inf=3.0 + index),
                center=(0.08 * index, -0.08 * index, 0),
                size=(
                    float(spec["size"][0]) * (0.75 - index * 0.08),
                    max(float(spec["size"][1]), 1.0) * (0.75 - index * 0.08),
                    max(float(spec["size"][2]), 1.0),
                ),
            )
            for index in range(5)
        )
    return geometry


def _build_sources(spec, gmes):
    source_name = spec.get("source", "point")
    if source_name == "none":
        return []
    source_time = gmes.Continuous(freq=0.35)
    amplitude = float(spec.get("source_amp", 1e-3))
    if source_name in {"point", "overlap-point"}:
        component = getattr(gmes, spec.get("source_component", "Ex"))
        sources = [
            gmes.PointSource(
                src_time=source_time,
                center=(0, 0, 0),
                component=component,
                amp=amplitude,
            )
        ]
        if source_name == "overlap-point":
            sources.append(
                gmes.PointSource(
                    src_time=gmes.Continuous(freq=0.55, phase=0.2),
                    center=(0, 0, 0),
                    component=component,
                    amp=0.25 * amplitude,
                )
            )
        return sources
    size = np.asarray(spec["size"], dtype=float)
    interface_size = tuple(max(value * 0.55, 1.0) for value in size)
    common = {
        "src_time": source_time,
        "center": (0, 0, 0),
        "size": interface_size,
        "direction": (1, 0, 0),
        "polarization": (0, 1, 0),
        "amp": amplitude,
    }
    if source_name == "tfsf":
        return [gmes.TotalFieldScatteredField(**common)]
    if source_name == "gaussian":
        return [gmes.GaussianBeam(directivity=gmes.PlusX, **common)]
    raise ValueError(f"unknown source recipe: {source_name}")


def build_simulation(spec, gmes):
    """Build a native simulation solely from one JSON workload object."""
    space = gmes.Cartesian(size=tuple(spec["size"]), resolution=spec["resolution"])
    if spec["recipe"] == "mixed":
        geometry = _mixed_geometry(spec, gmes)
    elif spec["recipe"] == "coverage":
        geometry = _coverage_geometry(spec, gmes)
    elif spec["recipe"] == "heterogeneous":
        geometry = _heterogeneous_geometry(spec, gmes)
    else:
        material_name = spec["material"]
        if material_name in {"upml", "cpml"}:
            geometry = [
                gmes.DefaultMedium(material_from_name("dielectric", gmes)),
                gmes.Shell(
                    material=material_from_name(material_name, gmes),
                    thickness=spec.get("pml_thickness", 1),
                ),
            ]
        else:
            geometry = [gmes.DefaultMedium(material_from_name(material_name, gmes))]
    kwargs = {"bloch": (0.07, 0.11, 0.13)} if spec.get("complex") else {}
    return gmes.FDTD(
        space, geometry, _build_sources(spec, gmes), verbose=False, **kwargs
    )


def initial_field_values(shapes, seed, scale=1e-3, *, complex_fields=False):
    """Build backend-neutral fixed-seed fields in canonical component order."""
    if set(shapes) != set(COMPONENT_NAMES):
        raise ValueError("field shapes must contain all canonical Yee components")
    rng = np.random.default_rng(seed)
    result = {}
    for name in COMPONENT_NAMES:
        shape = tuple(int(length) for length in shapes[name])
        values = scale * (1 + 0.1 * rng.random())
        for axis, length in enumerate(shape):
            ramp_shape = [1] * len(shape)
            ramp_shape[axis] = length
            values = values + (
                scale
                * 1e-6
                * (axis + 1)
                * np.linspace(0, 1, length).reshape(ramp_shape)
            )
        if complex_fields:
            values = values + 1j * scale * (1 + 0.1 * rng.random())
        result[name] = np.broadcast_to(values, shape).copy()
    return result


def initialize_fields(simulation, seed, scale=1e-3):
    """Fill every active field with fixed-seed nonzero values."""
    shapes = {
        component.__name__: tuple(field.shape)
        for component, field in simulation.field.items()
    }
    values = initial_field_values(
        shapes,
        seed,
        scale,
        complex_fields=simulation.cmplx,
    )
    for component, field in simulation.field.items():
        field[...] = values[component.__name__]
        if not np.all(field != 0):
            raise AssertionError("oracle seed unexpectedly produced a zero field value")


def _component_maps(simulation):
    arrays = {}
    metadata = {}
    for component, field in simulation.field.items():
        axes = simulation.space.component_coordinate_axes(component, field.shape)
        lowered = simulation.geom_tree.lower_grid(
            *axes, 0, field.size, component=component
        )
        name = component.__name__
        arrays[f"map/{name}/material_ids"] = np.asarray(lowered.material_ids).copy()
        arrays[f"map/{name}/underlying_ids"] = np.asarray(lowered.underlying_ids).copy()
        metadata[name] = {
            "shape": list(field.shape),
            "dtype": str(field.dtype),
            "active_cells": int(field.size),
            "material_regions": int(np.unique(lowered.material_ids).size),
            "underlying_regions": int(
                np.unique(lowered.underlying_ids[lowered.underlying_ids >= 0]).size
            ),
        }
    return arrays, metadata


def _linear_run_count(indices, shape):
    if not len(indices):
        return 0
    linear = np.sort(np.ravel_multi_index(indices.T, shape))
    return int(1 + np.count_nonzero(np.diff(linear) != 1))


def _updater_strategies(simulation, native_type):
    family = native_type
    for component in COMPONENT_NAMES:
        family = family.replace(f"{component}Real", "").replace(f"{component}Cmplx", "")
    compatible = {
        "DcpPlrc": {"DcpPlrc", "DcpRc"},
    }.get(family, {family})
    strategies = sorted(
        {
            type(geometry.material).__name__
            for geometry in simulation.geom_list
            if type(geometry.material).__name__ in compatible
        }
    )
    return strategies or [family]


def _updater_records(simulation, step, arrays, prefix="state"):
    records = []
    for component, updaters in sorted(
        simulation.pw_material.items(), key=lambda item: item[0].__name__
    ):
        ordered = sorted(
            updaters.items(),
            key=lambda item: (type(item[0]).__name__, type(item[1]).__name__),
        )
        for ordinal, (material_descriptor, updater) in enumerate(ordered):
            native_type = type(updater).__name__
            strategies = _updater_strategies(simulation, native_type)
            strategy = "+".join(strategies)
            record_prefix = (
                f"step/{step}/{prefix}/{component.__name__}/{ordinal}-{strategy}"
            )
            indices = np.asarray(updater.oracle_indices(), dtype=np.intc).reshape(-1, 3)
            state = np.asarray(updater.oracle_state(), dtype=np.complex128)
            arrays[f"{record_prefix}/indices"] = indices
            arrays[f"{record_prefix}/values"] = state
            run_count = _linear_run_count(indices, simulation.field[component].shape)
            cells = int(updater.idx_size())
            plan_bytes = int(updater.plan_bytes())
            index_bytes = int(updater.oracle_index_bytes())
            parameter_bytes = int(updater.oracle_parameter_bytes())
            records.append(
                {
                    "component": component.__name__,
                    "strategy": strategy,
                    "strategies": strategies,
                    "native_type": native_type,
                    "cells": cells,
                    "coverage": cells / simulation.field[component].size,
                    "fragmentation_runs": run_count,
                    "fragmentation_ratio": run_count / cells if cells else 0.0,
                    "state_values": int(state.size),
                    "state_nonzero_values": int(np.count_nonzero(state)),
                    "state_width": state.size / cells if cells else 0.0,
                    "state_key": f"{record_prefix}/values",
                    "state_bytes": int(updater.oracle_state_bytes()),
                    "plan_bytes": plan_bytes,
                    "index_bytes": index_bytes,
                    "parameter_bytes": parameter_bytes,
                    "live_updater_bytes": (plan_bytes + index_bytes + parameter_bytes),
                    "plan_runs": int(updater.plan_run_count()),
                    "bucket_signature": [
                        component.__name__,
                        strategy,
                        native_type,
                        int(updater.idx_size()),
                        int(state.size),
                    ],
                }
            )
    return records


def _source_param_values(parameter, time):
    values = []
    for name in ("amp", "eps_inf", "mu_inf"):
        value = getattr(parameter, name, None)
        if isinstance(value, dict):
            values.extend(complex(item) for item in value.values())
        elif value is not None:
            values.append(complex(value))
    source_time = getattr(parameter, "src_time", None)
    if source_time is not None:
        values.append(complex(source_time.oscillator(time)))
    for name in ("r0", "r1"):
        values.extend(complex(value) for value in getattr(parameter, name, {}).values())
    for name in ("samp_idx0", "samp_idx1"):
        for index in getattr(parameter, name, {}).values():
            values.extend(complex(value) for value in index)
    return values


def _source_records(simulation, step, arrays):
    records = []
    for component, updaters in sorted(
        simulation.pw_source.items(), key=lambda item: item[0].__name__
    ):
        for ordinal, updater in enumerate(
            sorted(updaters.values(), key=lambda value: type(value).__name__)
        ):
            ordered = sorted(updater._param.items())
            indices = np.asarray([index for index, _ in ordered], dtype=np.intc)
            if not len(indices):
                indices = np.empty((0, 3), dtype=np.intc)
            values = np.asarray(
                [
                    value
                    for _, parameter in ordered
                    for value in _source_param_values(parameter, simulation.time_step.t)
                ],
                dtype=np.complex128,
            )
            record_prefix = (
                f"step/{step}/source/{component.__name__}/"
                f"{ordinal}-{type(updater).__name__}"
            )
            arrays[f"{record_prefix}/indices"] = indices
            arrays[f"{record_prefix}/values"] = values
            records.append(
                {
                    "component": component.__name__,
                    "native_type": type(updater).__name__,
                    "cells": len(ordered),
                    "state_values": int(values.size),
                }
            )

    auxiliary = []
    for ordinal, source in enumerate(simulation.src_list):
        aux_fdtd = getattr(source, "aux_fdtd", None)
        if aux_fdtd is None:
            continue
        aux_prefix = f"step/{step}/source_aux/{ordinal}-{type(source).__name__}"
        arrays[f"{aux_prefix}/time"] = np.asarray(
            [aux_fdtd.time_step.n, aux_fdtd.time_step.t, aux_fdtd.time_step.dt]
        )
        for component, field in aux_fdtd.field.items():
            arrays[f"{aux_prefix}/field/{component.__name__}"] = field.copy()
        aux_records = _updater_records(
            aux_fdtd, step, arrays, prefix=f"source_aux_material/{ordinal}"
        )
        auxiliary.append(
            {
                "source": type(source).__name__,
                "fields": {
                    component.__name__: list(field.shape)
                    for component, field in aux_fdtd.field.items()
                },
                "materials": aux_records,
            }
        )
    return {"updaters": records, "auxiliary": auxiliary}


def _physical_observables(simulation, step, arrays):
    prefix = f"step/{step}/physical"
    energy = 0.0
    maximum = 0.0
    boundary_low = 0.0
    boundary_high = 0.0
    finite = True
    for component, field in simulation.field.items():
        magnitude = np.abs(field)
        energy += float(np.sum(magnitude * magnitude))
        maximum = max(maximum, float(np.max(magnitude)))
        boundary_low += float(np.sum(magnitude[0] * magnitude[0]))
        boundary_high += float(np.sum(magnitude[-1] * magnitude[-1]))
        finite = finite and bool(np.isfinite(field).all())
        transverse_axes = tuple(range(1, field.ndim))
        line = np.mean(field, axis=transverse_axes) if transverse_axes else field
        arrays[f"{prefix}/spectrum/{component.__name__}"] = np.abs(np.fft.fft(line))
    arrays[f"{prefix}/summary"] = np.asarray(
        [energy, maximum, boundary_low, boundary_high, float(finite)]
    )
    return {
        "energy": energy,
        "maximum_abs_field": maximum,
        "boundary_low_energy": boundary_low,
        "boundary_high_energy": boundary_high,
        "finite": finite,
    }


def _snapshot(simulation, step, arrays):
    for component, field in simulation.field.items():
        arrays[f"step/{step}/field/{component.__name__}"] = field.copy()
    arrays[f"step/{step}/time"] = np.asarray(
        [simulation.time_step.n, simulation.time_step.t, simulation.time_step.dt]
    )
    return {
        "materials": _updater_records(simulation, step, arrays),
        "sources": _source_records(simulation, step, arrays),
        "physical": _physical_observables(simulation, step, arrays),
    }


def capture_case(spec, manifest, output):
    """Capture complete maps, fields, and persistent updater states into NPZ."""
    import gmes

    reference_source = Path(gmes.__file__).resolve()
    expected_checkout = os.environ.get("GMES_ORACLE_EXPECTED_CHECKOUT")
    if expected_checkout and not reference_source.is_relative_to(
        Path(expected_checkout).resolve()
    ):
        raise RuntimeError(
            "isolated oracle imported gmes outside the requested checkout: "
            f"{reference_source}"
        )
    simulation = build_simulation(spec, gmes)
    simulation.init()
    reference = manifest["reference"]
    initialize_fields(simulation, reference["seed"], reference["field_scale"])
    for _ in range(reference["precondition_steps"]):
        simulation.step()

    arrays, map_metadata = _component_maps(simulation)
    step_records = {"0": _snapshot(simulation, 0, arrays)}
    current = 0
    capture_steps = spec.get("capture_steps", reference["capture_steps"])
    for target in capture_steps:
        while current < target:
            simulation.step()
            current += 1
        step_records[str(target)] = _snapshot(simulation, target, arrays)

    all_records = [
        record for snapshot in step_records.values() for record in snapshot["materials"]
    ]
    final_records = step_records[str(current)]["materials"]
    active_cells = sum(record["cells"] for record in final_records)
    state_bytes = sum(record["state_bytes"] for record in final_records)
    plan_bytes = sum(record["plan_bytes"] for record in final_records)
    index_bytes = sum(record["index_bytes"] for record in final_records)
    parameter_bytes = sum(record["parameter_bytes"] for record in final_records)
    geometry_metadata = [
        {
            "geometry": type(geometry).__name__,
            "material": _json_value(geometry.material),
        }
        for geometry in simulation.geom_list
    ]
    metadata = {
        "schema_version": 1,
        "backend": "native",
        "workload": spec,
        "reference": reference,
        "capture_steps": capture_steps,
        "input_state": {
            "archive_prefix": "step/0",
            "precondition_steps": reference["precondition_steps"],
            "relative_capture_steps": True,
        },
        "maps": map_metadata,
        "steps": step_records,
        "geometry_and_coefficients": geometry_metadata,
        "active_cells": active_cells,
        "state_bytes": state_bytes,
        "plan_bytes": plan_bytes,
        "index_bytes": index_bytes,
        "parameter_bytes": parameter_bytes,
        "live_updater_bytes": plan_bytes + index_bytes + parameter_bytes,
        "archive_array_bytes": int(sum(value.nbytes for value in arrays.values())),
        "nonzero_seed": True,
        "nonzero_persistent_state": all(
            record["state_values"] == 0 or record["state_nonzero_values"] > 0
            for record in all_records
        ),
        "reference_source": str(reference_source),
    }
    arrays["metadata.json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    return metadata


def read_metadata(archive):
    return json.loads(str(archive["metadata.json"]))


def compare_archives(reference_path, candidate_path, manifest):
    """Compare every complete array; raise with the precise failing key."""
    failures = []
    with (
        np.load(reference_path, allow_pickle=False) as reference,
        np.load(candidate_path, allow_pickle=False) as candidate,
    ):
        candidate_metadata = read_metadata(candidate)
        backend = candidate_metadata.get("backend", "native")
        reference_keys = set(reference.files) - {"metadata.json"}
        candidate_keys = set(candidate.files) - {"metadata.json"}
        if reference_keys != candidate_keys:
            failures.append(
                {
                    "key": "archive",
                    "missing": sorted(reference_keys - candidate_keys),
                    "unexpected": sorted(candidate_keys - reference_keys),
                }
            )
        for key in sorted(reference_keys & candidate_keys):
            expected = reference[key]
            actual = candidate[key]
            tolerance = tolerance_for_key(manifest, backend, key, str(expected.dtype))
            same_shape = expected.shape == actual.shape
            if np.issubdtype(expected.dtype, np.number):
                equal = same_shape and np.allclose(
                    expected,
                    actual,
                    rtol=tolerance["rtol"],
                    atol=tolerance["atol"],
                    equal_nan=True,
                )
            else:
                equal = same_shape and np.array_equal(expected, actual)
            if not equal:
                difference = (
                    np.abs(expected - actual)
                    if expected.shape == actual.shape
                    else np.array([np.inf])
                )
                failures.append(
                    {
                        "key": key,
                        "expected_shape": list(expected.shape),
                        "actual_shape": list(actual.shape),
                        "max_abs_error": float(np.max(difference)),
                        "rtol": tolerance["rtol"],
                        "atol": tolerance["atol"],
                    }
                )
    return {"passed": not failures, "failures": failures}


def tolerance_for_key(manifest, backend, key, dtype):
    tolerances = manifest["tolerances"]
    if backend == "native":
        return tolerances["native"].get(dtype, {"rtol": 0.0, "atol": 0.0})
    backend_tolerances = tolerances[backend]
    normalized = key.lower()
    for model in (
        "dcp-plrc",
        "dcp-ade",
        "dcp-rc",
        "lorentz",
        "drude",
        "dm2",
        "pml",
        "dielectric",
    ):
        if model in normalized:
            return backend_tolerances[model].get(dtype, {"rtol": 0.0, "atol": 0.0})
    return backend_tolerances["mixed"].get(dtype, {"rtol": 0.0, "atol": 0.0})


def peak_rss_bytes():
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(maximum if platform.system() == "Darwin" else maximum * 1024)


def current_rss_bytes():
    if platform.system() == "Linux":
        fields = Path("/proc/self/statm").read_text().split()
        return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
    output = _command_output(["ps", "-o", "rss=", "-p", str(os.getpid())])
    return int(output) * 1024 if output else None


def _percentile95(samples):
    return float(np.percentile(np.asarray(samples), 95))


def environment_metadata(gmes):
    """Return reproducibility metadata without requiring optional Torch/CUDA."""
    try:
        gpu = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,driver_version,memory.total,pstate,power.limit",
                    "--format=csv,noheader",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
            .splitlines()
        )
        topology = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except FileNotFoundError, subprocess.CalledProcessError:
        gpu, topology = [], None
    checkout = ROOT.parent
    lockfile = checkout / "uv.lock"
    try:
        import torch

        torch_metadata = {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
        }
    except ImportError:
        torch_metadata = None
    cpu_topology = _command_output(["lscpu", "-p=CORE,SOCKET"])
    physical_cores = None
    if cpu_topology:
        physical_cores = len(
            {
                line
                for line in cpu_topology.splitlines()
                if line and not line.startswith("#")
            }
        )
    return {
        "platform": platform.platform(),
        "hostname": platform.node(),
        "os": platform.uname()._asdict(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "gmes_version": importlib.metadata.version("gmes"),
        "gmes_source": str(Path(gmes.__file__).resolve()),
        "native_extension": str(Path(gmes._pw_material.__file__).resolve()),
        "git_commit": _command_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"]
        ),
        "git_status": _command_output(
            ["git", "-C", str(checkout), "status", "--short"]
        ),
        "uv_lock_sha256": (
            hashlib.sha256(lockfile.read_bytes()).hexdigest()
            if lockfile.is_file()
            else None
        ),
        "python_compiler": platform.python_compiler(),
        "python_build_cflags": sysconfig.get_config_var("CFLAGS"),
        "cxx_version": _command_output(["c++", "--version"]),
        "swig_version": _command_output(["swig", "-version"]),
        "extension_compile_standard": "c++23",
        "build_environment": {
            name: os.environ.get(name)
            for name in (
                "CC",
                "CXX",
                "CFLAGS",
                "CXXFLAGS",
                "LDFLAGS",
                "GMES_ENABLE_OPENMP",
                "GMES_OPENMP_PREFIX",
                "MACOSX_DEPLOYMENT_TARGET",
            )
        },
        "openmp_enabled": bool(gmes.pw_material.openmp_enabled()),
        "openmp_threads": int(gmes.pw_material.openmp_max_threads()),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_physical": physical_cores,
        "cpu_topology": cpu_topology,
        "cpu_model": _command_output(["lscpu"]),
        "memory_bytes": int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")),
        "gpu": gpu,
        "gpu_topology": topology,
        "torch": torch_metadata,
    }


def benchmark_case(spec, manifest, repeats, warmup, steps):
    """Emit the backend-neutral timing schema used by native and Torch runners."""
    import gmes

    construction = []
    lowering = []
    geometry_mapping = []
    simulation = None
    for _ in range(repeats):
        start = perf_counter()
        simulation = build_simulation(spec, gmes)
        construction.append(perf_counter() - start)
        start = perf_counter()
        simulation.init()
        lowering.append(perf_counter() - start)
        start = perf_counter()
        _component_maps(simulation)
        geometry_mapping.append(perf_counter() - start)
        initialize_fields(
            simulation,
            manifest["reference"]["seed"],
            manifest["reference"]["field_scale"],
        )
    start = perf_counter()
    for _ in range(warmup):
        simulation.step()
    warmup_seconds = perf_counter() - start
    one_step_samples = []
    for _ in range(repeats):
        simulation = build_simulation(spec, gmes)
        simulation.init()
        initialize_fields(
            simulation,
            manifest["reference"]["seed"],
            manifest["reference"]["field_scale"],
        )
        for _ in range(warmup):
            simulation.step()
        start = perf_counter()
        simulation.step()
        one_step_samples.append(perf_counter() - start)
    samples = []
    rss_samples = [current_rss_bytes()]
    for _ in range(repeats):
        simulation = build_simulation(spec, gmes)
        simulation.init()
        initialize_fields(
            simulation,
            manifest["reference"]["seed"],
            manifest["reference"]["field_scale"],
        )
        for _ in range(warmup):
            simulation.step()
        start = perf_counter()
        for _ in range(steps):
            simulation.step()
        samples.append(perf_counter() - start)
        rss_samples.append(current_rss_bytes())
    final_records = _updater_records(simulation, "benchmark", {})
    elapsed_cells = sum(record["cells"] for record in final_records) * steps
    return {
        "schema_version": 2,
        "backend": "native",
        "workload": spec,
        "benchmark_contract": {
            "initializer": FIELD_INITIALIZER,
            "seed": manifest["reference"]["seed"],
            "field_scale": manifest["reference"]["field_scale"],
            "warmup_steps": warmup,
            "steps_per_repeat": steps,
            "repetitions": repeats,
            "timer": "time.perf_counter",
            "sample_start": "independently-rebuilt-post-warmup-state",
        },
        "environment": environment_metadata(gmes),
        "measurements": {
            "construction": _timing_summary(construction),
            "geometry_mapping": _timing_summary(geometry_mapping),
            "native_initialization_and_plan_lowering": _timing_summary(lowering),
            "host_to_device_transfer": {
                "raw_seconds": [0.0] * repeats,
                "median_seconds": 0.0,
                "p95_seconds": 0.0,
            },
            "eager_warmup_seconds": warmup_seconds,
            "cold_compile": None,
            "cached_compile": None,
            "one_step": _timing_summary(one_step_samples),
            "advance": {
                **_timing_summary(samples),
                "steps_per_repeat": steps,
                "steps_per_second": steps / median(samples),
                "cells_per_second": elapsed_cells / median(samples),
            },
        },
        "memory": {
            "peak_rss_bytes": peak_rss_bytes(),
            "rss_samples_bytes": rss_samples,
            "rss_growth_bytes": (
                rss_samples[-1] - rss_samples[0]
                if all(value is not None for value in rss_samples)
                else None
            ),
            "live_field_bytes": sum(
                field.nbytes for field in simulation.field.values()
            ),
            "live_plan_bytes": sum(record["plan_bytes"] for record in final_records),
            "live_index_bytes": sum(record["index_bytes"] for record in final_records),
            "live_parameter_bytes": sum(
                record["parameter_bytes"] for record in final_records
            ),
            "live_updater_bytes": sum(
                record["live_updater_bytes"] for record in final_records
            ),
            "live_state_bytes": sum(record["state_bytes"] for record in final_records),
            "cuda_allocated_peak_bytes": None,
            "cuda_reserved_peak_bytes": None,
        },
        "updaters": final_records,
        "profiler": None,
    }


def find_case(manifest, name):
    try:
        return next(
            case
            for group in ("correctness", "benchmarks", "physical_checks")
            for case in manifest.get(group, [])
            if case["name"] == name
        )
    except StopIteration as error:
        raise ValueError(f"unknown correctness workload: {name}") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    subparsers = parser.add_subparsers(dest="command", required=True)
    describe = subparsers.add_parser("describe")
    describe.add_argument("--case")
    capture = subparsers.add_parser("capture")
    capture.add_argument("--case", required=True)
    capture.add_argument("--output", type=Path, required=True)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--case", required=True)
    benchmark.add_argument("--repeats", type=int, default=11)
    benchmark.add_argument("--warmup", type=int, default=3)
    benchmark.add_argument("--steps", type=int, default=20)
    benchmark.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    if args.command == "describe":
        value = find_case(manifest, args.case) if args.case else manifest
    elif args.command == "capture":
        value = capture_case(find_case(manifest, args.case), manifest, args.output)
    elif args.command == "compare":
        value = compare_archives(args.reference, args.candidate, manifest)
        if not value["passed"]:
            print(json.dumps(value, indent=2, sort_keys=True))
            raise SystemExit(1)
    else:
        if args.repeats < 1 or args.warmup < 0 or args.steps < 1 or args.threads < 1:
            parser.error(
                "repeats, steps, and threads must be positive; "
                "warmup must be nonnegative"
            )
        for variable in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[variable] = str(args.threads)
        value = benchmark_case(
            find_case(manifest, args.case),
            manifest,
            args.repeats,
            args.warmup,
            args.steps,
        )
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
