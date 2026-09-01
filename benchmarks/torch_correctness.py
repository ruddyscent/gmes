#!/usr/bin/env python3
"""Produce and bind independently-derived Torch correctness archives."""

import argparse
import hashlib
import json
import zipfile
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path, PurePosixPath

import numpy as np

import gmes
import gmes.torch_fdtd
from benchmarks import native_oracle
from benchmarks.native_oracle import COMPONENT_NAMES
from gmes.torch_source import TorchPointSourceBatch, TorchTransparentBatch

ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "native_oracle_workloads.json"
INDEX_KIND = "torch-correctness-evidence-index"
INDEX_CONTRACT = "complete-field-and-persistent-state-v1"
PRODUCER = "gmes-torch-correctness-v2"
TORCH_ARRAY_CONTRACT = "torch-plan-state-source-arrays-v1"

_STRATEGY_MODELS = {
    "Const": "const",
    "Cpml": "cpml",
    "DcpAde": "dcp-ade",
    "DcpPlrc": "dcp-plrc",
    "DcpRc": "dcp-rc",
    "Dielectric": "dielectric",
    "Dm2": "dm2",
    "Drude": "drude",
    "Dummy": "dummy",
    "Lorentz": "lorentz",
    "Upml": "upml",
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_sha256(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _exact_keys(value, expected, label, optional=()):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    actual = set(value)
    expected = set(expected)
    optional = set(optional)
    if not expected <= actual <= expected | optional:
        raise ValueError(
            f"{label} keys are invalid: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected - optional)}"
        )


def _required_cases(manifest):
    return tuple(
        (group, case)
        for group in ("correctness", "physical_checks")
        for case in manifest.get(group, ())
    )


def _candidate_geometry(spec):
    if spec["recipe"] == "mixed":
        return native_oracle._mixed_geometry(spec, gmes)
    if spec["recipe"] == "coverage":
        return native_oracle._coverage_geometry(spec, gmes)
    if spec["recipe"] == "heterogeneous":
        return native_oracle._heterogeneous_geometry(spec, gmes)
    material = native_oracle.material_from_name(spec["material"], gmes)
    if spec["material"] in {"upml", "cpml"}:
        return [
            gmes.DefaultMedium(native_oracle.material_from_name("dielectric", gmes)),
            gmes.Shell(
                material=material,
                thickness=spec.get("pml_thickness", 1),
            ),
        ]
    if spec["material"] == "dielectric":
        return [gmes.DefaultMedium(material)]

    # Torch uses a dense Dielectric base.  Preserve the homogeneous native
    # coefficients at the base and overlay the stateful strategy on every
    # active cell.  The canonical step/0 maps remain the archive input.
    base = gmes.Dielectric(eps_inf=material.eps_inf, mu_inf=material.mu_inf)
    minimum = 1.0 / spec["resolution"]
    extent = tuple(max(abs(float(value)), minimum) for value in spec["size"])
    return [
        gmes.DefaultMedium(base),
        gmes.Block(material=material, center=(0, 0, 0), size=extent),
    ]


def _candidate_sources(spec):
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
    common = {
        "src_time": source_time,
        "center": (0, 0, 0),
        "size": tuple(max(value * 0.55, 1.0) for value in size),
        "direction": (1, 0, 0),
        "polarization": (0, 1, 0),
        "amp": amplitude,
    }
    if source_name == "tfsf":
        return [gmes.TotalFieldScatteredField(**common)]
    if source_name == "gaussian":
        return [gmes.GaussianBeam(directivity=gmes.PlusX, **common)]
    raise ValueError(f"unknown source recipe: {source_name}")


def _runtime_contract(device, precision, graph_mode, compile_mode):
    if not isinstance(device, str) or (
        device != "cpu"
        and not (
            device.startswith("cuda:")
            and device[5:].isdigit()
            and str(int(device[5:])) == device[5:]
        )
    ):
        raise ValueError("device must be 'cpu' or an explicit canonical 'cuda:N'")
    if precision not in {"float32", "float64"}:
        raise ValueError("precision must be float32 or float64")
    if graph_mode not in {"eager", "graph"}:
        raise ValueError("graph_mode must be eager or graph")
    if compile_mode not in {"default", "reduce-overhead", "max-autotune"}:
        raise ValueError("compile_mode is invalid")
    if graph_mode == "eager" and compile_mode != "default":
        raise ValueError("a non-default compile mode requires graph mode")
    if device == "cpu" and compile_mode != "default":
        raise ValueError("CPU correctness graph mode requires default compile mode")
    return {
        "device": device,
        "precision": precision,
        "graph_mode": graph_mode,
        "compile_policy": "compile" if graph_mode == "graph" else "eager",
        "compile_mode": compile_mode,
    }


def _build_torch_simulation(
    spec,
    *,
    dt,
    threads=1,
    device="cpu",
    precision="float64",
    graph_mode="eager",
    compile_mode="default",
):
    mode = _runtime_contract(device, precision, graph_mode, compile_mode)
    bloch = (0.07, 0.11, 0.13) if spec.get("complex") else None
    return gmes.TorchSimulation(
        space=gmes.Cartesian(tuple(spec["size"]), spec["resolution"]),
        geometry=_candidate_geometry(spec),
        sources=_candidate_sources(spec),
        runtime=gmes.TorchRuntimeConfig(
            device=mode["device"],
            precision=mode["precision"],
            compile_policy=mode["compile_policy"],
            compile_mode=mode["compile_mode"],
            cpu_threads=threads,
            cpu_interop_threads=1,
        ),
        dt=dt,
        bloch=bloch,
    )


def _host(value):
    return value.detach().cpu().numpy().copy()


def _complex_channels(value, paired_real):
    value = _host(value)
    if paired_real:
        return value[..., 0] + 1j * value[..., 1]
    return value[..., 0].astype(np.complex128)


def _dispersive_rows(simulation, descriptor):
    prefix = descriptor.prefix
    paired = simulation.state.paired_real
    poles = descriptor.pole_count
    points = descriptor.point_count

    def state(suffix):
        return getattr(simulation.state, f"{prefix}_{suffix}")

    if descriptor.model in {"drude", "lorentz"}:
        return np.concatenate(
            (
                _complex_channels(state("previous"), paired).T,
                _complex_channels(state("current"), paired).T,
            ),
            axis=1,
        )
    if descriptor.model == "dcp-ade":
        return np.concatenate(
            (
                _complex_channels(state("field_old"), paired)[:, None],
                _complex_channels(state("pole_old"), paired).T,
                _complex_channels(state("pole_now"), paired).T,
                _complex_channels(state("point_old"), paired).T,
                _complex_channels(state("point_now"), paired).T,
            ),
            axis=1,
        )

    pole = _host(state("pole_state"))
    point = _host(state("point_state"))
    channels = 2 if paired else 1
    pole_blocks = [pole[..., channel].T for channel in range(channels)]
    point_blocks = [
        (point[..., channel, 0] + 1j * point[..., channel, 1]).T
        for channel in range(channels)
    ]
    if not paired:
        pole_blocks.append(np.zeros((descriptor.target_count, poles)))
        point_blocks.append(np.zeros((descriptor.target_count, points)))
    return np.concatenate((*pole_blocks, *point_blocks), axis=1)


def _state_parts(simulation, component, models):
    parts = []
    plan = simulation.plan.components[component]
    pml = simulation.state.pml_state_snapshot()
    for index, bucket in enumerate(plan.buckets):
        model = bucket.signature.model
        if model not in models or model not in {"upml", "cpml"}:
            continue
        key = f"pml_{component.lower()}_{index}_state"
        targets = np.asarray(bucket.targets, dtype=np.int64)

        parts.append(
            {
                "model": model,
                "targets": targets,
                "rows": np.asarray(pml[key], dtype=np.complex128),
                "representation": key,
            }
        )

    for descriptor in simulation.plan.dispersive_buckets:
        if descriptor.component != component or descriptor.model not in models:
            continue
        targets = _host(
            getattr(simulation.plan, f"{descriptor.prefix}_targets")
        ).astype(np.int64)

        parts.append(
            {
                "model": descriptor.model,
                "targets": targets,
                "rows": _dispersive_rows(simulation, descriptor),
                "representation": descriptor.prefix,
            }
        )

    dm2_snapshots = {
        (snapshot["component"], tuple(snapshot["targets"])): snapshot
        for snapshot in simulation.dm2_state_snapshot()
    }
    for bucket in simulation.state.dm2_buckets:
        metadata = bucket.metadata
        if component != metadata.component or "dm2" not in models:
            continue
        targets = _host(getattr(simulation.plan, f"{metadata.prefix}_targets")).astype(
            np.int64
        )
        snapshot = dm2_snapshots[(component, tuple(targets))]

        parts.append(
            {
                "model": "dm2",
                "targets": targets,
                "rows": np.asarray(snapshot["u"], dtype=np.complex128).reshape(
                    len(targets), -1
                ),
                "representation": metadata.prefix,
            }
        )
    return parts


def _physical(fields, step, arrays):
    prefix = f"step/{step}/physical"
    summary = [0.0, 0.0, 0.0, 0.0, 1.0]
    for name in COMPONENT_NAMES:
        field = np.asarray(fields[name])
        magnitude = np.abs(field)
        summary[0] += float(np.sum(magnitude * magnitude))
        summary[1] = max(summary[1], float(np.max(magnitude)))
        summary[2] += float(np.sum(magnitude[0] * magnitude[0]))
        summary[3] += float(np.sum(magnitude[-1] * magnitude[-1]))
        summary[4] = float(bool(summary[4]) and np.isfinite(field).all())
        axes = tuple(range(1, field.ndim))
        line = np.mean(field, axis=axes) if axes else field
        arrays[f"{prefix}/spectrum/{name}"] = np.abs(np.fft.fft(line))
    arrays[f"{prefix}/summary"] = np.asarray(summary)
    return {
        "energy": summary[0],
        "maximum_abs_field": summary[1],
        "boundary_low_energy": summary[2],
        "boundary_high_energy": summary[3],
        "finite": bool(summary[4]),
    }


def _strategy_family(strategy):
    return "DcpPlrc" if strategy in {"DcpPlrc", "DcpRc"} else strategy


def _material_plan_groups(simulation, component):
    plan = simulation.plan.components[component]
    material_ids = np.asarray(plan.material_ids).reshape(-1)
    groups = {}
    target_families = {}
    for bucket_index, bucket in enumerate(plan.buckets):
        for target in np.asarray(bucket.targets, dtype=np.int64):
            geometry_index = int(material_ids[int(target)])
            strategy = type(simulation.geometry[geometry_index].material).__name__
            family = _strategy_family(strategy)
            group = groups.setdefault(
                family,
                {"targets": [], "strategies": set(), "buckets": set()},
            )
            group["targets"].append(int(target))
            group["strategies"].add(strategy)
            group["buckets"].add(bucket_index)
            if int(target) in target_families:
                raise ValueError(f"Torch material target is duplicated for {component}")
            target_families[int(target)] = family
    inactive = np.flatnonzero(np.asarray(plan.ownership).reshape(-1) < 0)
    if len(inactive):
        group = groups.setdefault(
            "Dummy",
            {"targets": [], "strategies": set(), "buckets": set()},
        )
        group["strategies"].add("Dummy")
        for value in inactive:
            target = int(value)
            if target in target_families:
                raise ValueError(f"Torch material target is duplicated for {component}")
            group["targets"].append(target)
            target_families[target] = "Dummy"
    if len(target_families) != int(np.prod(plan.shape)):
        raise ValueError(f"Torch material topology is incomplete for {component}")
    normalized = []
    for family, group in sorted(groups.items()):
        targets = np.asarray(sorted(group["targets"]), dtype=np.int64)
        if len(targets) != len(np.unique(targets)):
            raise ValueError(f"Torch material group {family} contains duplicates")
        normalized.append(
            {
                "family": family,
                "strategies": sorted(group["strategies"]),
                "targets": targets,
                "buckets": sorted(group["buckets"]),
            }
        )
    return normalized, target_families


def _group_state_rows(simulation, component, group, target_families):
    parts = []
    for part in _state_parts(simulation, component, set(_STRATEGY_MODELS.values())):
        families = {target_families[int(value)] for value in part["targets"]}
        if len(families) != 1:
            raise ValueError("one Torch persistent-state buffer spans material groups")
        if families == {group["family"]}:
            parts.append(part)
    if not parts:
        return np.empty((len(group["targets"]), 0), dtype=np.complex128), []
    by_target = {}
    width = None
    for part in parts:
        rows = np.asarray(part["rows"], dtype=np.complex128).reshape(
            len(part["targets"]), -1
        )
        if width is None:
            width = rows.shape[1]
        elif rows.shape[1] != width:
            raise ValueError("Torch material state widths differ inside one group")
        for target, row in zip(part["targets"], rows, strict=True):
            target = int(target)
            if target in by_target:
                raise ValueError("Torch material state target is duplicated")
            by_target[target] = row
    expected = {int(value) for value in group["targets"]}
    if set(by_target) != expected:
        raise ValueError("Torch persistent state does not cover its material group")
    return np.asarray([by_target[int(value)] for value in group["targets"]]), parts


def _independent_material_records(
    simulation, step, arrays, *, prefix="state", components=COMPONENT_NAMES
):
    records = []
    suffix = "Cmplx" if simulation.state.paired_real else "Real"
    for component in components:
        groups, target_families = _material_plan_groups(simulation, component)
        shape = tuple(simulation.plan.shapes[component])
        for ordinal, group in enumerate(groups):
            strategy = "+".join(group["strategies"])
            record_prefix = f"step/{step}/{prefix}/{component}/{ordinal}-{strategy}"
            targets = group["targets"]
            indices = np.column_stack(np.unravel_index(targets, shape)).astype(
                np.int64, copy=False
            )
            rows, parts = _group_state_rows(
                simulation, component, group, target_families
            )
            values = np.asarray(rows, dtype=np.complex128).reshape(-1)
            arrays[f"{record_prefix}/indices"] = indices.copy()
            arrays[f"{record_prefix}/values"] = values.copy()
            cells = len(targets)
            runs = native_oracle._linear_run_count(indices, shape)
            plan_bytes = sum(
                int(
                    simulation.plan.components[component].buckets[index].estimated_bytes
                )
                for index in group["buckets"]
            )
            state_bytes = int(values.nbytes)
            native_type = f"{group['family']}{component}{suffix}"
            records.append(
                {
                    "component": component,
                    "strategy": strategy,
                    "strategies": group["strategies"],
                    "native_type": native_type,
                    "cells": cells,
                    "coverage": cells / int(np.prod(shape)),
                    "fragmentation_runs": runs,
                    "fragmentation_ratio": runs / cells if cells else 0.0,
                    "state_values": int(values.size),
                    "state_nonzero_values": int(np.count_nonzero(values)),
                    "state_width": values.size / cells if cells else 0.0,
                    "state_key": f"{record_prefix}/values",
                    "state_bytes": state_bytes,
                    "plan_bytes": plan_bytes,
                    "index_bytes": int(indices.nbytes),
                    "parameter_bytes": 0,
                    "live_updater_bytes": plan_bytes + int(indices.nbytes),
                    "plan_runs": runs,
                    "bucket_signature": [
                        component,
                        strategy,
                        native_type,
                        cells,
                        int(values.size),
                    ],
                    "backend_metadata": {
                        "producer": PRODUCER,
                        "representations": [part["representation"] for part in parts],
                        "plan_bucket_ordinals": group["buckets"],
                        "state_origin": "live-torch-state-buffers",
                    },
                }
            )
    return records


def _source_batch_payload(simulation, batch):
    component = batch.component
    shape = tuple(simulation.plan.shapes[component])
    if isinstance(batch, TorchPointSourceBatch):
        rows = []
        for kind, prefix in enumerate(("overwrite", "additive")):
            targets = _host(getattr(batch, f"{prefix}_targets")).astype(np.int64)
            models = _host(getattr(batch, f"{prefix}_models")).astype(np.float64)
            parameters = _host(getattr(batch, f"{prefix}_parameters")).astype(
                np.float64
            )
            amplitudes = _host(getattr(batch, f"{prefix}_amplitudes")).astype(
                np.float64
            )
            for target, model, parameter, amplitude in zip(
                targets, models, parameters, amplitudes, strict=True
            ):
                rows.append(
                    (
                        int(target),
                        np.concatenate(
                            (
                                np.asarray([kind, model], dtype=np.float64),
                                parameter,
                                np.asarray([amplitude], dtype=np.float64),
                            )
                        ),
                    )
                )
        rows.sort(key=lambda item: (item[0], item[1][0]))
        native_type = f"PointSource{component}"
        representation = "torch-point-source-batch-v1"
    elif isinstance(batch, TorchTransparentBatch):
        targets = _host(batch.targets).astype(np.int64)
        samples = _host(batch.samples).astype(np.float64)
        weights = _host(batch.weights).astype(np.float64)
        width = -1.0 if batch.gaussian_width is None else batch.gaussian_width
        rows = [
            (
                int(target),
                np.concatenate(
                    (np.asarray([width]), sample.reshape(-1), weight.reshape(-1))
                ),
            )
            for target, sample, weight in zip(targets, samples, weights, strict=True)
        ]
        rows.sort(key=lambda item: item[0])
        native_type = f"Transparent{component}"
        representation = "torch-transparent-source-batch-v1"
    else:
        raise TypeError(f"unsupported Torch source batch {type(batch).__name__}")
    targets = np.asarray([target for target, _row in rows], dtype=np.int64)
    indices = (
        np.column_stack(np.unravel_index(targets, shape)).astype(np.int64, copy=False)
        if len(targets)
        else np.empty((0, 3), dtype=np.int64)
    )
    # Preserve the exact IEEE-754 payload while satisfying the base archive's
    # finite-number contract. Point-source plans legitimately use infinities
    # as open-interval sentinels; the raw named buffers are archived separately.
    values = (
        np.concatenate([row for _target, row in rows])
        .astype("<f8", copy=False)
        .view("<u8")
        .copy()
        if rows
        else np.empty(0, dtype="<u8")
    )
    return native_type, representation, indices, values


def _auxiliary_source_name(simulation, auxiliary):
    batches = [
        batch
        for batch in simulation.sources.batches
        if isinstance(batch, TorchTransparentBatch) and batch.auxiliary is auxiliary
    ]
    if not batches:
        raise ValueError("Torch auxiliary simulation has no transparent source batch")
    return (
        "GaussianBeam"
        if any(batch.gaussian_width is not None for batch in batches)
        else "TotalFieldScatteredField"
    )


def _independent_source_records(simulation, step, arrays):
    records = []
    ordinals = {component: 0 for component in COMPONENT_NAMES}
    ordered = sorted(
        enumerate(simulation.sources.batches),
        key=lambda item: (item[1].component, type(item[1]).__name__, item[0]),
    )
    for _source_ordinal, batch in ordered:
        native_type, representation, indices, values = _source_batch_payload(
            simulation, batch
        )
        component = batch.component
        ordinal = ordinals[component]
        ordinals[component] += 1
        prefix = f"step/{step}/source/{component}/{ordinal}-{native_type}"
        arrays[f"{prefix}/indices"] = indices.copy()
        arrays[f"{prefix}/values"] = values.copy()
        records.append(
            {
                "component": component,
                "native_type": native_type,
                "cells": len(indices),
                "state_values": int(values.size),
                "backend_metadata": {
                    "producer": PRODUCER,
                    "representation": representation,
                    "source_origin": "live-torch-source-batch",
                },
            }
        )

    auxiliaries = []
    for ordinal, auxiliary in enumerate(simulation.sources.auxiliaries):
        source_name = _auxiliary_source_name(simulation, auxiliary)
        prefix = f"step/{step}/source_aux/{ordinal}-{source_name}"
        fields = auxiliary.host_snapshot()
        active_components = tuple(
            component
            for component in COMPONENT_NAMES
            if np.count_nonzero(fields[component])
        )
        if not active_components:
            raise ValueError("Torch auxiliary simulation has no active field component")
        field_shapes = {}
        for component in COMPONENT_NAMES:
            values = np.asarray(fields[component]).copy()
            canonical = (
                values
                if component in active_components
                else np.zeros((1, 1, 1), dtype=values.dtype)
            )
            arrays[f"{prefix}/field/{component}"] = canonical
            field_shapes[component] = list(canonical.shape)
        arrays[f"{prefix}/time"] = np.asarray(
            [
                int(auxiliary.state.step_count.detach().cpu()),
                int(auxiliary.state.step_count.detach().cpu()) * auxiliary.plan.dt,
                auxiliary.plan.dt,
            ]
        )
        auxiliaries.append(
            {
                "source": source_name,
                "fields": field_shapes,
                "materials": _independent_material_records(
                    auxiliary,
                    step,
                    arrays,
                    components=active_components,
                    prefix=f"source_aux_material/{ordinal}",
                ),
                "backend_metadata": {
                    "producer": PRODUCER,
                    "canonical_components": list(active_components),
                    "precision": auxiliary.runtime.precision,
                    "raw_state_origin": "torch-named-buffers",
                    "representation": "live-torch-auxiliary-simulation-v1",
                },
            }
        )
    return {"updaters": records, "auxiliary": auxiliaries}


def _safe_buffer_name(name):
    parts = name.split(".")
    if not parts or any(not part or "/" in part for part in parts):
        raise ValueError(f"Torch buffer name is not canonical: {name!r}")
    return "/".join(parts)


def _store_module_buffers(module, prefix, arrays):
    for name, value in module.named_buffers():
        key = f"{prefix}/{_safe_buffer_name(name)}"
        if key in arrays:
            raise ValueError(f"duplicate Torch buffer archive key: {key}")
        arrays[key] = _host(value)


def _store_planner_arrays(simulation, arrays):
    for component, plan in simulation.plan.components.items():
        prefix = f"torch/planner/{component}"
        for name in (
            "material_ids",
            "underlying_ids",
            "ownership",
            "dense_inverse",
            "constant_targets",
            "constant_values",
        ):
            arrays[f"{prefix}/{name}"] = np.asarray(getattr(plan, name)).copy()
        for ordinal, bucket in enumerate(plan.buckets):
            bucket_prefix = f"{prefix}/bucket/{ordinal}-{bucket.signature.model}"
            for name in (
                "targets",
                "target_region_indices",
                "region_keys",
                "region_coefficient_indices",
                "coefficient_table",
                "cell_coefficients",
                "stencil_indices",
                "tile_origins",
                "tile_region_indices",
            ):
                arrays[f"{bucket_prefix}/{name}"] = np.asarray(
                    getattr(bucket, name)
                ).copy()
            for axis, residual in enumerate(bucket.cpml_residual_axes):
                residual_prefix = f"{bucket_prefix}/cpml/{axis}"
                for name in ("positions", "targets", "stencil_indices", "parameters"):
                    arrays[f"{residual_prefix}/{name}"] = np.asarray(
                        getattr(residual, name)
                    ).copy()


def _store_live_torch_state(simulation, step, arrays):
    _store_module_buffers(simulation.state, f"torch/step/{step}/state", arrays)
    _store_module_buffers(simulation.sources, f"torch/step/{step}/sources", arrays)
    for ordinal, auxiliary in enumerate(simulation.sources.auxiliaries):
        prefix = f"torch/step/{step}/auxiliary/{ordinal}"
        _store_module_buffers(auxiliary.state, f"{prefix}/state", arrays)
        _store_module_buffers(auxiliary.sources, f"{prefix}/sources", arrays)


def _independent_snapshot(simulation, step, arrays):
    fields = simulation.host_snapshot()
    for name in COMPONENT_NAMES:
        arrays[f"step/{step}/field/{name}"] = np.asarray(fields[name]).copy()
    arrays[f"step/{step}/time"] = np.asarray(
        [
            int(simulation.state.step_count.detach().cpu()),
            int(simulation.state.step_count.detach().cpu()) * simulation.plan.dt,
            simulation.plan.dt,
        ]
    )
    snapshot = {
        "materials": _independent_material_records(simulation, step, arrays),
        "sources": _independent_source_records(simulation, step, arrays),
        "physical": _physical(fields, step, arrays),
    }
    _store_live_torch_state(simulation, step, arrays)
    return snapshot


def _component_maps(simulation, fields, arrays):
    metadata = {}
    for component in COMPONENT_NAMES:
        plan = simulation.plan.components[component]
        material_ids = np.asarray(plan.material_ids).reshape(-1).copy()
        underlying_ids = np.asarray(plan.underlying_ids).reshape(-1).copy()
        arrays[f"map/{component}/material_ids"] = material_ids
        arrays[f"map/{component}/underlying_ids"] = underlying_ids
        underlying = underlying_ids[underlying_ids >= 0]
        metadata[component] = {
            "shape": list(plan.shape),
            "dtype": str(np.asarray(fields[component]).dtype),
            "active_cells": int(material_ids.size),
            "material_regions": int(np.unique(material_ids).size),
            "underlying_regions": int(np.unique(underlying).size),
        }
    return metadata


def _logical_geometry(spec):
    if spec["recipe"] in {"mixed", "coverage", "heterogeneous"}:
        return _candidate_geometry(spec)
    material = native_oracle.material_from_name(spec["material"], gmes)
    if spec["material"] in {"upml", "cpml"}:
        return [
            gmes.DefaultMedium(native_oracle.material_from_name("dielectric", gmes)),
            gmes.Shell(
                material=material,
                thickness=spec.get("pml_thickness", 1),
            ),
        ]
    return [gmes.DefaultMedium(material)]


def _geometry_metadata(geometry):
    return [
        {
            "geometry": type(item).__name__,
            "material": native_oracle._json_value(item.material),
        }
        for item in geometry
    ]


def _initialized_geometry_metadata(spec, dt):
    space = gmes.Cartesian(tuple(spec["size"]), spec["resolution"])
    space.dt = float(dt)
    geometry = _candidate_geometry(spec)
    for geometric_object in geometry:
        geometric_object.init(space)
    return _geometry_metadata(geometry)


def _logical_geometry_metadata(spec, dt):
    space = gmes.Cartesian(tuple(spec["size"]), spec["resolution"])
    space.dt = float(dt)
    geometry = _logical_geometry(spec)
    for geometric_object in geometry:
        geometric_object.init(space)
    return _geometry_metadata(geometry)


def _array_descriptor(value):
    contiguous = np.ascontiguousarray(value)
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "size_bytes": int(contiguous.nbytes),
        "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }


def _array_descriptors(arrays, predicate):
    return {
        key: _array_descriptor(arrays[key]) for key in sorted(arrays) if predicate(key)
    }


def _candidate_provenance():
    source = Path(gmes.torch_fdtd.__file__).resolve()
    controller = Path(__file__).resolve()
    records = {
        "source": native_oracle._checkout_provenance(
            native_oracle._git_checkout(source), source
        ),
        "controller": native_oracle._checkout_provenance(
            native_oracle._git_checkout(controller), controller
        ),
    }
    if any(record["clean"] is not True for record in records.values()):
        raise RuntimeError("Torch correctness evidence requires clean Git checkouts")
    if records["source"]["commit"] != records["controller"]["commit"]:
        raise RuntimeError("Torch source and correctness controller commits differ")
    return records, source


def capture_torch_candidate(
    reference_path,
    manifest,
    output,
    *,
    threads=1,
    device="cpu",
    precision="float64",
    graph_mode="eager",
    compile_mode="default",
):
    """Independently execute Torch and emit a strict schema-2 candidate."""
    mode = _runtime_contract(device, precision, graph_mode, compile_mode)
    reference_path = Path(reference_path).resolve(strict=True)
    output = Path(output).resolve()
    if output == reference_path:
        raise ValueError("candidate output must differ from the native reference")
    with np.load(reference_path, allow_pickle=False) as reference:
        reference_metadata = native_oracle._validate_archive(
            reference, manifest, "reference"
        )
        spec = reference_metadata["workload"]
        dt = float(np.asarray(reference["step/0/time"])[2])
        input_arrays = {
            key: _array_descriptor(reference[key])
            for key in sorted(reference.files)
            if key.startswith(("map/", "step/0/"))
        }
    simulation = _build_torch_simulation(
        spec,
        dt=dt,
        threads=threads,
        device=device,
        precision=precision,
        graph_mode=graph_mode,
        compile_mode=compile_mode,
    )
    for name in COMPONENT_NAMES:
        shape = tuple(reference_metadata["maps"][name]["shape"])
        if tuple(simulation.plan.shapes[name]) != shape:
            raise ValueError(f"Torch field shape differs for {name}")
    initial_fields = native_oracle.initial_field_values(
        simulation.plan.shapes,
        manifest["reference"]["seed"],
        manifest["reference"]["field_scale"],
        complex_fields=simulation.state.paired_real,
    )
    simulation.load_host_fields(initial_fields)
    if graph_mode == "graph" and simulation.device.type == "cuda":
        simulation.capture_cuda_graphs()
    simulation.advance(manifest["reference"]["precondition_steps"])

    arrays = {}
    _store_planner_arrays(simulation, arrays)
    _store_module_buffers(simulation.plan, "torch/plan", arrays)
    fields = simulation.host_snapshot()
    map_metadata = _component_maps(simulation, fields, arrays)
    steps = {"0": _independent_snapshot(simulation, 0, arrays)}
    completed = 0
    capture_steps = spec.get("capture_steps", manifest["reference"]["capture_steps"])
    for target in capture_steps:
        simulation.advance(target - completed)
        completed = target
        steps[str(target)] = _independent_snapshot(simulation, target, arrays)

    final_records = steps[str(completed)]["materials"]
    all_records = [record for value in steps.values() for record in value["materials"]]
    totals = {
        "active_cells": sum(record["cells"] for record in final_records),
        "state_bytes": sum(record["state_bytes"] for record in final_records),
        "plan_bytes": sum(record["plan_bytes"] for record in final_records),
        "index_bytes": sum(record["index_bytes"] for record in final_records),
        "parameter_bytes": sum(record["parameter_bytes"] for record in final_records),
    }
    totals["live_updater_bytes"] = (
        totals["plan_bytes"] + totals["index_bytes"] + totals["parameter_bytes"]
    )
    torch_arrays = _array_descriptors(arrays, lambda key: key.startswith("torch/"))
    source_arrays = _array_descriptors(
        arrays,
        lambda key: len(key.split("/")) > 2 and key.split("/")[2] == "source",
    )
    base_bytes = sum(
        value.nbytes for key, value in arrays.items() if not key.startswith("torch/")
    )
    actual_geometry = _geometry_metadata(simulation.geometry)
    provenance, source = _candidate_provenance()
    metadata = {
        "schema_version": native_oracle.ARCHIVE_SCHEMA_VERSION,
        "backend": "torch",
        "backend_metadata": {
            "producer": PRODUCER,
            "solver_abi": gmes.torch_fdtd.TORCH_SOLVER_ABI,
            "cuda_graph_execution_representation": (
                gmes.torch_fdtd.CUDA_GRAPH_EXECUTION_REPRESENTATION
            ),
            **mode,
            "resolved_device": str(simulation.device),
            "paired_real": bool(simulation.state.paired_real),
            "auxiliary_precisions": [
                auxiliary.runtime.precision
                for auxiliary in simulation.sources.auxiliaries
            ],
            "manifest_contract_sha256": _canonical_sha256(manifest),
            "input_archive": {
                "sha256": _sha256(reference_path),
                "size_bytes": reference_path.stat().st_size,
                "media_type": "application/x-npz",
                "prefix": "step/0",
            },
            "input_step_zero_contract": {
                "array_contract": "native-step-zero-and-maps-v1",
                "array_bytes": sum(
                    descriptor["size_bytes"] for descriptor in input_arrays.values()
                ),
                "arrays": input_arrays,
                "reconstruction": {
                    "mode": "independent-workload-replay-v1",
                    "field_initializer": manifest["reference"]["field_initializer"],
                    "seed": manifest["reference"]["seed"],
                    "field_scale": manifest["reference"]["field_scale"],
                    "precondition_steps": manifest["reference"]["precondition_steps"],
                    "field_origin": "manifest-initializer",
                    "source_origin": "live-torch-plan",
                    "material_origin": "live-torch-plan",
                },
            },
            "logical_map_source": "live-torch-plan",
            "actual_geometry_and_coefficients": actual_geometry,
            "plan_identity": simulation.plan_identity,
            "compile_cache_key": simulation.compile_cache_key,
            "cuda_graph_regions": list(sorted(simulation._cuda_graphs)),
            "array_contract": TORCH_ARRAY_CONTRACT,
            "torch_array_bytes": sum(
                descriptor["size_bytes"] for descriptor in torch_arrays.values()
            ),
            "torch_arrays": torch_arrays,
            "source_arrays": source_arrays,
        },
        "workload": spec,
        "reference": native_oracle._correctness_reference_contract(
            manifest["reference"]
        ),
        "capture_steps": capture_steps,
        "input_state": {
            "archive_prefix": "step/0",
            "precondition_steps": manifest["reference"]["precondition_steps"],
            "relative_capture_steps": True,
        },
        "maps": map_metadata,
        "steps": steps,
        "geometry_and_coefficients": _logical_geometry_metadata(spec, dt),
        **totals,
        "archive_array_bytes": int(base_bytes),
        "nonzero_seed": all(
            np.count_nonzero(arrays[f"step/0/field/{name}"])
            == arrays[f"step/0/field/{name}"].size
            for name in COMPONENT_NAMES
        ),
        "nonzero_persistent_state": all(
            record["state_values"] == 0 or record["state_nonzero_values"] > 0
            for record in all_records
        ),
        "provenance": provenance,
        "reference_source": str(source),
    }
    arrays["metadata.json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    validation = compare_torch_archives(reference_path, output, manifest)
    if validation != {"passed": True, "failures": []}:
        raise ValueError(
            "independent Torch reconstruction differs from native step/0 contract: "
            f"{validation['failures'][:1]}"
        )
    return metadata


class _BaseArchiveView:
    def __init__(self, archive):
        self._archive = archive
        self.files = [key for key in archive.files if not key.startswith("torch/")]

    def __getitem__(self, key):
        if key not in self.files:
            raise KeyError(key)
        return self._archive[key]


def _descriptor_map_complete(archive, descriptors, expected_keys, label):
    _exact_keys(descriptors, expected_keys, label)
    for key in sorted(expected_keys):
        descriptor = descriptors[key]
        _exact_keys(
            descriptor,
            {"shape", "dtype", "size_bytes", "sha256"},
            f"{label}.{key}",
        )
        if not native_oracle._same_json_value(
            descriptor, _array_descriptor(archive[key])
        ):
            raise ValueError(f"{label} descriptor differs for {key}")


def _input_step_zero_contract_complete(contract, manifest):
    try:
        _exact_keys(
            contract,
            {"array_contract", "array_bytes", "arrays", "reconstruction"},
            "Torch native step/0 input contract",
        )
        arrays = contract["arrays"]
        if not isinstance(arrays, dict) or not arrays:
            return False
        required = {"step/0/time"}
        required.update(f"step/0/field/{name}" for name in COMPONENT_NAMES)
        required.update(
            f"map/{name}/{suffix}"
            for name in COMPONENT_NAMES
            for suffix in ("material_ids", "underlying_ids")
        )
        if not required <= set(arrays):
            return False
        size_bytes = 0
        for key, descriptor in arrays.items():
            if not isinstance(key, str) or not key.startswith(("map/", "step/0/")):
                return False
            _exact_keys(
                descriptor,
                {"shape", "dtype", "size_bytes", "sha256"},
                f"Torch native step/0 descriptor {key}",
            )
            shape = descriptor["shape"]
            if not isinstance(shape, list) or any(
                type(dimension) is not int or dimension < 0 for dimension in shape
            ):
                return False
            dtype_name = descriptor["dtype"]
            if not isinstance(dtype_name, str) or not dtype_name:
                return False
            dtype = np.dtype(dtype_name)
            if dtype.hasobject:
                return False
            expected_size = int(dtype.itemsize)
            for dimension in shape:
                expected_size *= dimension
            if (
                type(descriptor["size_bytes"]) is not int
                or descriptor["size_bytes"] != expected_size
                or not _hex_string(descriptor["sha256"], 64)
            ):
                return False
            size_bytes += expected_size
        reconstruction = contract["reconstruction"]
        expected_reconstruction = {
            "mode": "independent-workload-replay-v1",
            "field_initializer": manifest["reference"]["field_initializer"],
            "seed": manifest["reference"]["seed"],
            "field_scale": manifest["reference"]["field_scale"],
            "precondition_steps": manifest["reference"]["precondition_steps"],
            "field_origin": "manifest-initializer",
            "source_origin": "live-torch-plan",
            "material_origin": "live-torch-plan",
        }
        return (
            contract["array_contract"] == "native-step-zero-and-maps-v1"
            and type(contract["array_bytes"]) is int
            and contract["array_bytes"] == size_bytes
            and native_oracle._same_json_value(reconstruction, expected_reconstruction)
        )
    except AttributeError, KeyError, TypeError, ValueError, OverflowError:
        return False


def _validate_live_clock(archive, prefix, precision, canonical_time):
    step_count = np.asarray(archive[f"{prefix}/step_count"])
    source_time = np.asarray(archive[f"{prefix}/source_time"])
    time_step = np.asarray(archive[f"{prefix}/time_step"])
    clock_dtype = np.dtype(precision)
    if (
        step_count.shape != ()
        or step_count.dtype != np.dtype("int64")
        or int(step_count) < 0
        or source_time.shape != ()
        or time_step.shape != ()
        or source_time.dtype != clock_dtype
        or time_step.dtype != clock_dtype
        or not bool(np.isfinite(source_time))
        or not bool(np.isfinite(time_step))
        or float(time_step) <= 0.0
    ):
        raise ValueError(f"Torch live clock is invalid: {prefix}")
    expected = np.multiply(step_count.astype(clock_dtype), time_step, dtype=clock_dtype)
    canonical_time = np.asarray(canonical_time)
    if canonical_time.shape != (3,):
        raise ValueError(f"Torch canonical clock is invalid: {prefix}")
    canonical_dt = np.asarray(canonical_time[2], dtype=clock_dtype)
    if (
        float(canonical_time[0]) != int(step_count)
        or not np.array_equal(time_step, canonical_dt)
        or not np.array_equal(source_time, expected)
    ):
        raise ValueError(f"Torch live clock differs from canonical n * dt: {prefix}")


def _validate_torch_candidate_archive(archive, manifest):
    metadata = native_oracle._validate_archive(
        _BaseArchiveView(archive), manifest, "candidate"
    )
    if metadata["backend"] != "torch":
        raise ValueError("candidate correctness archive is not Torch")
    backend = metadata.get("backend_metadata")
    _exact_keys(
        backend,
        {
            "producer",
            "solver_abi",
            "cuda_graph_execution_representation",
            "device",
            "precision",
            "graph_mode",
            "compile_policy",
            "compile_mode",
            "resolved_device",
            "paired_real",
            "auxiliary_precisions",
            "manifest_contract_sha256",
            "input_archive",
            "input_step_zero_contract",
            "logical_map_source",
            "actual_geometry_and_coefficients",
            "plan_identity",
            "compile_cache_key",
            "cuda_graph_regions",
            "array_contract",
            "torch_array_bytes",
            "torch_arrays",
            "source_arrays",
        },
        "Torch correctness backend_metadata",
    )
    mode = _runtime_contract(
        backend["device"],
        backend["precision"],
        backend["graph_mode"],
        backend["compile_mode"],
    )
    if any(backend[key] != value for key, value in mode.items()):
        raise ValueError("Torch correctness runtime mode is inconsistent")
    if (
        backend["producer"] != PRODUCER
        or backend["solver_abi"] != gmes.torch_fdtd.TORCH_SOLVER_ABI
        or backend["cuda_graph_execution_representation"]
        != gmes.torch_fdtd.CUDA_GRAPH_EXECUTION_REPRESENTATION
        or backend["resolved_device"] != backend["device"]
        or type(backend["paired_real"]) is not bool
        or backend["paired_real"] is not bool(metadata["workload"].get("complex"))
        or backend["manifest_contract_sha256"] != _canonical_sha256(manifest)
        or backend["logical_map_source"] != "live-torch-plan"
        or backend["array_contract"] != TORCH_ARRAY_CONTRACT
        or not isinstance(backend["plan_identity"], str)
        or not backend["plan_identity"]
        or not isinstance(backend["compile_cache_key"], str)
        or not backend["compile_cache_key"]
    ):
        raise ValueError("Torch correctness backend identity is invalid")
    auxiliary_precisions = backend["auxiliary_precisions"]
    if not isinstance(auxiliary_precisions, list) or any(
        precision != "float64" for precision in auxiliary_precisions
    ):
        raise ValueError("Torch auxiliary precision metadata is invalid")
    _exact_keys(
        backend["input_archive"],
        {"sha256", "size_bytes", "media_type", "prefix"},
        "Torch correctness input_archive",
    )
    input_archive = backend["input_archive"]
    if (
        not _hex_string(input_archive["sha256"], 64)
        or type(input_archive["size_bytes"]) is not int
        or input_archive["size_bytes"] < 1
        or input_archive["media_type"] != "application/x-npz"
        or input_archive["prefix"] != "step/0"
    ):
        raise ValueError("Torch correctness input archive descriptor is invalid")
    if not _input_step_zero_contract_complete(
        backend["input_step_zero_contract"], manifest
    ):
        raise ValueError("Torch native step/0 input contract is invalid")
    expected_geometry = _initialized_geometry_metadata(
        metadata["workload"], np.asarray(archive["step/0/time"])[2]
    )
    if not native_oracle._same_json_value(
        backend["actual_geometry_and_coefficients"], expected_geometry
    ):
        raise ValueError("Torch correctness actual geometry differs from workload")
    graph_regions = backend["cuda_graph_regions"]
    if not isinstance(graph_regions, list) or not all(
        isinstance(value, str) and value for value in graph_regions
    ):
        raise ValueError("Torch correctness CUDA graph regions are invalid")
    if backend["device"] == "cpu" or backend["graph_mode"] == "eager":
        if graph_regions != []:
            raise ValueError("eager/CPU correctness cannot claim CUDA graphs")
    elif not graph_regions:
        raise ValueError("CUDA graph correctness has no captured graph regions")

    torch_keys = {key for key in archive.files if key.startswith("torch/")}
    _descriptor_map_complete(
        archive, backend["torch_arrays"], torch_keys, "Torch arrays"
    )
    if type(backend["torch_array_bytes"]) is not int or backend[
        "torch_array_bytes"
    ] != sum(value["size_bytes"] for value in backend["torch_arrays"].values()):
        raise ValueError("Torch array byte accounting is inaccurate")
    source_keys = {
        key
        for key in archive.files
        if len(key.split("/")) > 2 and key.split("/")[2] == "source"
    }
    _descriptor_map_complete(
        archive, backend["source_arrays"], source_keys, "Torch source arrays"
    )
    expected_steps = {"0"} | {str(value) for value in metadata["capture_steps"]}
    expected_field_dtype = np.dtype(
        "complex64"
        if backend["paired_real"] and backend["precision"] == "float32"
        else "complex128" if backend["paired_real"] else backend["precision"]
    )
    for step in expected_steps:
        if not any(key.startswith(f"torch/step/{step}/state/") for key in torch_keys):
            raise ValueError(f"Torch raw state arrays are absent for step {step}")
        _validate_live_clock(
            archive,
            f"torch/step/{step}/state",
            backend["precision"],
            archive[f"step/{step}/time"],
        )
        auxiliaries = metadata["steps"][step]["sources"]["auxiliary"]
        if len(auxiliaries) != len(auxiliary_precisions):
            raise ValueError("Torch auxiliary precision count is inconsistent")
        for ordinal, record in enumerate(auxiliaries):
            auxiliary_precision = record["backend_metadata"]["precision"]
            if auxiliary_precision != auxiliary_precisions[ordinal]:
                raise ValueError("Torch auxiliary precision is inconsistent")
            expected_auxiliary_dtype = np.dtype(auxiliary_precision)
            state_prefix = f"torch/step/{step}/auxiliary/{ordinal}/state"
            required_state_fields = {
                f"{state_prefix}/{component.lower()}" for component in COMPONENT_NAMES
            }
            if not required_state_fields.issubset(torch_keys):
                raise ValueError("Torch auxiliary raw state fields are incomplete")
            for key in required_state_fields:
                if archive[key].dtype != expected_auxiliary_dtype:
                    raise ValueError(f"Torch auxiliary raw precision differs for {key}")
            for raw_prefix in (
                f"{state_prefix}/",
                f"torch/step/{step}/auxiliary/{ordinal}/sources/",
            ):
                for key in torch_keys:
                    if not key.startswith(raw_prefix):
                        continue
                    dtype = archive[key].dtype
                    if np.issubdtype(dtype, np.inexact) and dtype != (
                        expected_auxiliary_dtype
                    ):
                        raise ValueError(
                            f"Torch auxiliary raw precision differs for {key}"
                        )
            _validate_live_clock(
                archive,
                state_prefix,
                auxiliary_precision,
                archive[f"step/{step}/source_aux/{ordinal}-{record['source']}/time"],
            )
            auxiliary_field_dtype = np.dtype(
                "complex128" if backend["paired_real"] else auxiliary_precision
            )
            for component in COMPONENT_NAMES:
                key = (
                    f"step/{step}/source_aux/{ordinal}-{record['source']}/"
                    f"field/{component}"
                )
                if archive[key].dtype != auxiliary_field_dtype:
                    raise ValueError(
                        f"Torch auxiliary field precision differs for {key}"
                    )
        source_batch_root = f"torch/step/{step}/sources/batches/"
        transparent_batch_prefixes = set()
        for key in torch_keys:
            if not key.startswith(source_batch_root):
                continue
            remainder = key.removeprefix(source_batch_root)
            parts = remainder.split("/")
            if (
                len(parts) == 2
                and parts[0].isdigit()
                and parts[0] == str(int(parts[0]))
                and parts[1] == "weights"
            ):
                transparent_batch_prefixes.add(f"{source_batch_root}{parts[0]}")
        transparent_updaters = sum(
            record["native_type"].startswith("Transparent")
            for record in metadata["steps"][step]["sources"]["updaters"]
        )
        if len(transparent_batch_prefixes) != transparent_updaters:
            raise ValueError("Torch transparent raw batch count is inconsistent")
        for batch_prefix in transparent_batch_prefixes:
            expected_dtypes = {
                "targets": np.dtype("int64"),
                "samples": np.dtype("int64"),
                "weights": np.dtype("float64"),
                "_sample_values": np.dtype("float64"),
                "_values": np.dtype("float64"),
                "_outer_values": np.dtype(backend["precision"]),
            }
            for name, expected_dtype in expected_dtypes.items():
                key = f"{batch_prefix}/{name}"
                if key not in torch_keys:
                    raise ValueError(
                        f"Torch transparent raw batch is incomplete: {batch_prefix}"
                    )
                if archive[key].dtype != expected_dtype:
                    raise ValueError(
                        f"Torch transparent raw precision differs for {key}"
                    )
            envelope_dtypes = {
                "_envelope_step": np.dtype("int64"),
                "_envelope_step_offset": np.dtype("int64"),
                "_envelope": np.dtype("float64"),
            }
            envelope_keys = {f"{batch_prefix}/{name}" for name in envelope_dtypes}
            present_envelope_keys = envelope_keys & torch_keys
            expected_envelope_keys = (
                envelope_keys
                if metadata["workload"].get("source") == "gaussian"
                else set()
            )
            if present_envelope_keys != expected_envelope_keys:
                raise ValueError(
                    f"Torch Gaussian envelope raw state is incomplete: {batch_prefix}"
                )
            for name, expected_dtype in envelope_dtypes.items():
                key = f"{batch_prefix}/{name}"
                if key in torch_keys and archive[key].dtype != expected_dtype:
                    raise ValueError(
                        f"Torch Gaussian envelope raw precision differs for {key}"
                    )
        for component in COMPONENT_NAMES:
            if archive[f"step/{step}/field/{component}"].dtype != expected_field_dtype:
                raise ValueError(
                    f"Torch field precision differs for step/{step}/field/{component}"
                )

    for component in COMPONENT_NAMES:
        for suffix in ("material_ids", "underlying_ids"):
            base = np.asarray(archive[f"map/{component}/{suffix}"]).reshape(-1)
            planner = np.asarray(
                archive[f"torch/planner/{component}/{suffix}"]
            ).reshape(-1)
            if not np.array_equal(base, planner):
                raise ValueError(
                    f"Torch logical map differs from planner array for {component}"
                )
    if not _candidate_source_metadata_complete(metadata, torch_keys):
        raise ValueError("Torch source topology metadata is invalid")
    return metadata


def _candidate_source_metadata_complete(metadata, torch_keys):
    try:
        for step in ("0", *(str(value) for value in metadata["capture_steps"])):
            sources = metadata["steps"][step]["sources"]
            for record in sources["updaters"]:
                backend = record["backend_metadata"]
                _exact_keys(
                    backend,
                    {"producer", "representation", "source_origin"},
                    "Torch source backend_metadata",
                )
                native_type = record["native_type"]
                representation = (
                    "torch-point-source-batch-v1"
                    if native_type.startswith("PointSource")
                    else (
                        "torch-transparent-source-batch-v1"
                        if native_type.startswith("Transparent")
                        else None
                    )
                )
                if (
                    backend["producer"] != PRODUCER
                    or backend["source_origin"] != "live-torch-source-batch"
                    or backend["representation"] != representation
                ):
                    return False
            for ordinal, record in enumerate(sources["auxiliary"]):
                backend = record["backend_metadata"]
                _exact_keys(
                    backend,
                    {
                        "producer",
                        "representation",
                        "canonical_components",
                        "precision",
                        "raw_state_origin",
                    },
                    "Torch auxiliary backend_metadata",
                )
                components = backend["canonical_components"]
                if (
                    backend["producer"] != PRODUCER
                    or backend["representation"] != "live-torch-auxiliary-simulation-v1"
                    or backend["precision"] != "float64"
                    or backend["raw_state_origin"] != "torch-named-buffers"
                    or not isinstance(components, list)
                    or components
                    != [name for name in COMPONENT_NAMES if name in components]
                    or not components
                ):
                    return False
                for component in COMPONENT_NAMES:
                    canonical = component in components
                    if (record["fields"][component] != [1, 1, 1]) is not canonical:
                        return False
                    key = (
                        f"torch/step/{step}/auxiliary/{ordinal}/state/"
                        f"{component.lower()}"
                    )
                    if key not in torch_keys:
                        return False
        return True
    except AttributeError, KeyError, TypeError, ValueError:
        return False


def _runtime_mode(metadata):
    backend = metadata["backend_metadata"]
    return {
        key: backend[key]
        for key in (
            "device",
            "precision",
            "graph_mode",
            "compile_policy",
            "compile_mode",
        )
    }


_MATERIAL_TOPOLOGY_KEYS = (
    "component",
    "strategies",
    "native_type",
    "cells",
    "state_values",
    "state_width",
    "state_key",
    "bucket_signature",
)

_STRATEGY_TOLERANCE_MODEL = {
    "Const": "dielectric",
    "Cpml": "pml",
    "DcpAde": "dcp-ade",
    "DcpPlrc": "dcp-plrc",
    "DcpRc": "dcp-rc",
    "Dielectric": "dielectric",
    "Dm2": "dm2",
    "Drude": "drude",
    "Lorentz": "lorentz",
    "Upml": "pml",
}


def _material_topology(records):
    return [{key: record[key] for key in _MATERIAL_TOPOLOGY_KEYS} for record in records]


def _material_topology_matches(reference_metadata, candidate_metadata):
    for step in ("0", *(str(value) for value in reference_metadata["capture_steps"])):
        expected = reference_metadata["steps"][step]
        actual = candidate_metadata["steps"][step]
        if not native_oracle._same_json_value(
            _material_topology(expected["materials"]),
            _material_topology(actual["materials"]),
        ):
            return False
        expected_auxiliary = expected["sources"]["auxiliary"]
        actual_auxiliary = actual["sources"]["auxiliary"]
        if len(expected_auxiliary) != len(actual_auxiliary):
            return False
        for left, right in zip(expected_auxiliary, actual_auxiliary, strict=True):
            if not native_oracle._same_json_value(
                _material_topology(left["materials"]),
                _material_topology(right["materials"]),
            ):
                return False
    return True


def _reference_strategies(metadata):
    strategies = set()
    for step in ("0", *(str(value) for value in metadata["capture_steps"])):
        snapshot = metadata["steps"][step]
        records = list(snapshot["materials"])
        for auxiliary in snapshot["sources"]["auxiliary"]:
            records.extend(auxiliary["materials"])
        for record in records:
            strategies.update(record["strategies"])
    return strategies


def _manifest_tolerance(manifest, key, dtype, strategies, workload_name):
    if "/source_aux/" in key:
        source = (
            manifest["tolerances"]["torch"]
            .get("source_auxiliary", {})
            .get(workload_name, {})
            .get(dtype)
        )
        if source is not None:
            if not isinstance(source, dict) or set(source) != {"rtol", "atol"}:
                raise ValueError("source/auxiliary tolerance is invalid")
            values = {name: float(source[name]) for name in ("rtol", "atol")}
            if any(not np.isfinite(value) or value < 0 for value in values.values()):
                raise ValueError("source/auxiliary tolerance is invalid")
            return {
                **values,
                "scope": f"source_auxiliary/{workload_name}/{dtype}",
            }

    selected = set(strategies)
    for segment in key.split("/"):
        if "-" not in segment:
            continue
        candidate = segment.split("-", 1)[1]
        if candidate in selected:
            selected = {candidate}
            break
    models = {
        _STRATEGY_TOLERANCE_MODEL[strategy]
        for strategy in selected
        if strategy != "Dummy"
    }
    scope = None
    if not models:
        parts = key.split("/")
        dummy_source_numerics = (
            selected == {"Dummy"}
            and workload_name == "dummy"
            and len(parts) > 3
            and parts[0] == "step"
            and parts[1].isdigit()
            and parts[2] in {"field", "physical"}
        )
        if dummy_source_numerics:
            # CPython and compiled Torch can use different libm oscillator
            # kernels, so the driven cell may differ at the ULP scale. Reuse
            # the pinned non-dispersive tolerance without relaxing topology.
            models = {"dielectric"}
            scope = f"dummy-source-numerics/dielectric/{dtype}"
        else:
            return {"rtol": 0.0, "atol": 0.0, "scope": "exact/dummy"}
    tolerances = []
    for model in sorted(models):
        model_dtype = "float64" if model == "dm2" and dtype == "complex128" else dtype
        try:
            tolerance = manifest["tolerances"]["torch"][model][model_dtype]
            if set(tolerance) != {"rtol", "atol"}:
                raise KeyError(model_dtype)
            values = {name: float(tolerance[name]) for name in ("rtol", "atol")}
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"manifest has no pinned {model_dtype} tolerance for {model}"
            ) from error
        if any(not np.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError(f"manifest tolerance is invalid for {model}")
        tolerances.append(values)
    result = {
        name: max(tolerance[name] for tolerance in tolerances)
        for name in ("rtol", "atol")
    }
    result["scope"] = scope or f"strategies/{','.join(sorted(models))}/{dtype}"
    return result


def _source_topology_matches(reference_metadata, candidate_metadata):
    for step in ("0", *(str(value) for value in reference_metadata["capture_steps"])):
        expected = reference_metadata["steps"][step]["sources"]
        actual = candidate_metadata["steps"][step]["sources"]
        expected_updaters = expected["updaters"]
        actual_updaters = actual["updaters"]
        if len(expected_updaters) != len(actual_updaters):
            return False
        for left, right in zip(expected_updaters, actual_updaters, strict=True):
            if {key: left[key] for key in ("component", "native_type", "cells")} != {
                key: right[key] for key in ("component", "native_type", "cells")
            }:
                return False
            backend = right.get("backend_metadata")
            if (
                not isinstance(backend, dict)
                or backend.get("producer") != PRODUCER
                or backend.get("source_origin") != "live-torch-source-batch"
            ):
                return False
        expected_aux = expected["auxiliary"]
        actual_aux = actual["auxiliary"]
        if len(expected_aux) != len(actual_aux):
            return False
        for left, right in zip(expected_aux, actual_aux, strict=True):
            if left["source"] != right["source"] or left["fields"] != right["fields"]:
                return False
            backend = right.get("backend_metadata")
            if (
                not isinstance(backend, dict)
                or backend.get("producer") != PRODUCER
                or backend.get("representation") != "live-torch-auxiliary-simulation-v1"
            ):
                return False
    return True


def _comparison_precision(candidate_metadata, key):
    parts = key.split("/")
    if (
        len(parts) > 3
        and parts[0] == "step"
        and parts[2] in {"source_aux", "source_aux_material"}
    ):
        ordinal_value = parts[3].split("-", 1)[0]
        if ordinal_value.isdigit():
            auxiliaries = candidate_metadata["steps"][parts[1]]["sources"]["auxiliary"]
            ordinal = int(ordinal_value)
            if ordinal < len(auxiliaries):
                return auxiliaries[ordinal]["backend_metadata"]["precision"]
    return candidate_metadata["backend_metadata"]["precision"]


def _comparison_dtype(candidate_metadata, key, expected, actual):
    if _comparison_precision(candidate_metadata, key) == "float32":
        return "float32"
    if np.iscomplexobj(expected) or np.iscomplexobj(actual):
        return "complex128"
    return "float64"


def _is_source_array(key):
    parts = key.split("/")
    return len(parts) > 2 and parts[2] == "source"


def compare_torch_archives(
    reference_path, candidate_path, manifest, *, include_tolerances=False
):
    """Fail closed while strictly comparing two untrusted evidence archives."""
    try:
        return _compare_torch_archives_loaded(
            reference_path,
            candidate_path,
            manifest,
            include_tolerances=include_tolerances,
        )
    except (
        AttributeError,
        EOFError,
        IndexError,
        KeyError,
        OSError,
        OverflowError,
        TypeError,
        UnicodeError,
        ValueError,
        zipfile.BadZipFile,
    ) as error:
        result = {
            "passed": False,
            "failures": [{"key": "archive/container", "error": str(error)}],
        }
        if include_tolerances:
            result["tolerance_results"] = []
        return result


def _compare_torch_archives_loaded(
    reference_path, candidate_path, manifest, *, include_tolerances=False
):
    """Strictly compare native schema-2 state with a Torch schema-2 archive."""
    failures = []
    tolerance_results = []
    with ExitStack() as stack:
        reference = stack.enter_context(np.load(reference_path, allow_pickle=False))
        candidate = stack.enter_context(np.load(candidate_path, allow_pickle=False))
        try:
            reference_metadata = native_oracle._validate_archive(
                reference, manifest, "reference"
            )
        except (KeyError, TypeError, ValueError, IndexError, OverflowError) as error:
            failures.append({"key": "reference/archive-contract", "error": str(error)})
            reference_metadata = None
        try:
            candidate_metadata = _validate_torch_candidate_archive(candidate, manifest)
        except (KeyError, TypeError, ValueError, IndexError, OverflowError) as error:
            failures.append({"key": "candidate/archive-contract", "error": str(error)})
            candidate_metadata = None
        if reference_metadata is None or candidate_metadata is None:
            return {"passed": False, "failures": failures}
        input_archive = candidate_metadata["backend_metadata"]["input_archive"]
        if (
            input_archive["sha256"] != _sha256(reference_path)
            or input_archive["size_bytes"] != Path(reference_path).stat().st_size
        ):
            failures.append(
                {"key": "candidate/input_archive", "error": "reference bytes differ"}
            )
        reference_inputs = {
            key: _array_descriptor(reference[key])
            for key in sorted(reference.files)
            if key.startswith(("map/", "step/0/"))
        }
        input_contract = candidate_metadata["backend_metadata"][
            "input_step_zero_contract"
        ]
        if not native_oracle._same_json_value(
            input_contract["arrays"], reference_inputs
        ) or input_contract["array_bytes"] != sum(
            value["size_bytes"] for value in reference_inputs.values()
        ):
            failures.append(
                {
                    "key": "candidate/input_step_zero_contract",
                    "error": "native step/0 or map array descriptors differ",
                }
            )
        if not native_oracle._same_json_value(
            candidate_metadata["geometry_and_coefficients"],
            reference_metadata["geometry_and_coefficients"],
        ):
            failures.append(
                {
                    "key": "geometry_and_coefficients",
                    "error": "logical geometry and coefficients differ",
                }
            )
        if not _source_topology_matches(reference_metadata, candidate_metadata):
            failures.append(
                {"key": "source/topology", "error": "Torch source topology differs"}
            )
        if not _material_topology_matches(reference_metadata, candidate_metadata):
            failures.append(
                {
                    "key": "material/topology",
                    "error": "Torch material topology differs",
                }
            )
        strategies = _reference_strategies(reference_metadata)
        reference_source_indices = {
            key
            for key in reference.files
            if _is_source_array(key) and key.endswith("/indices")
        }
        candidate_source_indices = {
            key
            for key in candidate.files
            if _is_source_array(key) and key.endswith("/indices")
        }
        if reference_source_indices != candidate_source_indices:
            failures.append(
                {
                    "key": "source/index-topology",
                    "missing": sorted(
                        reference_source_indices - candidate_source_indices
                    ),
                    "unexpected": sorted(
                        candidate_source_indices - reference_source_indices
                    ),
                }
            )
        for key in sorted(reference_source_indices & candidate_source_indices):
            if not np.array_equal(reference[key], candidate[key]):
                failures.append({"key": key, "error": "source indices differ"})

        reference_keys = {
            key
            for key in reference.files
            if key != "metadata.json"
            and not key.startswith("map/")
            and not _is_source_array(key)
        }
        candidate_keys = {
            key
            for key in candidate.files
            if key != "metadata.json"
            and not key.startswith(("map/", "torch/"))
            and not _is_source_array(key)
        }
        if reference_keys != candidate_keys:
            failures.append(
                {
                    "key": "archive/topology",
                    "missing": sorted(reference_keys - candidate_keys),
                    "unexpected": sorted(candidate_keys - reference_keys),
                }
            )
        for key in sorted(reference_keys & candidate_keys):
            expected = reference[key]
            actual = candidate[key]
            same_shape = expected.shape == actual.shape
            if np.issubdtype(expected.dtype, np.integer):
                equal = same_shape and np.array_equal(expected, actual)
                tolerance = {
                    "rtol": 0.0,
                    "atol": 0.0,
                    "scope": "exact/integer",
                }
                dtype_label = str(expected.dtype)
            else:
                dtype = _comparison_dtype(candidate_metadata, key, expected, actual)
                dtype_label = dtype
                try:
                    tolerance = _manifest_tolerance(
                        manifest,
                        key,
                        dtype,
                        strategies,
                        reference_metadata["workload"]["name"],
                    )
                except ValueError as error:
                    failures.append({"key": key, "error": str(error)})
                    continue
                equal = same_shape and np.allclose(
                    expected,
                    actual,
                    rtol=tolerance["rtol"],
                    atol=tolerance["atol"],
                    equal_nan=False,
                )
            if expected.shape == actual.shape:
                if np.issubdtype(expected.dtype, np.integer):
                    difference = np.abs(
                        expected.astype(np.float64) - actual.astype(np.float64)
                    )
                else:
                    difference = np.abs(expected - actual)
            else:
                difference = np.asarray([np.inf])
            tolerance_results.append(
                {
                    "key": key,
                    "dtype": dtype_label,
                    "scope": tolerance["scope"],
                    "rtol": tolerance["rtol"],
                    "atol": tolerance["atol"],
                    "max_abs_error": (
                        float(np.max(difference)) if difference.size else 0.0
                    ),
                }
            )
            if not equal:
                failures.append(
                    {
                        "key": key,
                        "expected_shape": list(expected.shape),
                        "actual_shape": list(actual.shape),
                        "max_abs_error": float(np.max(difference)),
                        **tolerance,
                    }
                )
    result = {"passed": not failures, "failures": failures}
    if include_tolerances:
        result["tolerance_results"] = tolerance_results
    return result


def _archive_record(path, manifest, role):
    path = Path(path).resolve(strict=True)
    with np.load(path, allow_pickle=False) as archive:
        metadata = (
            _validate_torch_candidate_archive(archive, manifest)
            if role == "candidate"
            else native_oracle._validate_archive(archive, manifest, role)
        )
    return path, metadata


def _descriptor_candidate_evidence(candidate_evidence):
    value = {
        key: candidate_evidence.get(key)
        for key in (
            "candidate_git_commit",
            "candidate_git_status",
            "manifest_sha256",
        )
    }
    if (
        not _hex_string(value["candidate_git_commit"], 40)
        or value["candidate_git_status"] != ""
        or not _hex_string(value["manifest_sha256"], 64)
    ):
        raise ValueError("candidate evidence has no portable three-key binding")
    return value


def _canonical_descriptor_path(value):
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "\x00" in value
        or (len(value) >= 2 and value[0].isalpha() and value[1] == ":")
    ):
        raise ValueError("artifact descriptor path is not canonical POSIX")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("artifact descriptor path contains a dot or empty segment")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise ValueError("artifact descriptor path must be canonical and relative")
    return path


def _descriptor_root(path):
    root = Path(path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("correctness descriptor root must be a directory")
    return root


def _relative_descriptor_path(path, root):
    try:
        relative = Path(path).resolve(strict=True).relative_to(root)
    except ValueError as error:
        raise ValueError("correctness artifact is outside descriptor root") from error
    value = PurePosixPath(*relative.parts).as_posix()
    _canonical_descriptor_path(value)
    return value


def _artifact_descriptor(path, root, candidate_evidence, media_type):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError("correctness artifact must be a regular file")
    return {
        "path": _relative_descriptor_path(path, root),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "media_type": media_type,
        "candidate_evidence": _descriptor_candidate_evidence(candidate_evidence),
    }


def _resolve_artifact_descriptor(root, descriptor):
    portable = _canonical_descriptor_path(descriptor["path"])
    path = root.joinpath(*portable.parts).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("correctness artifact escapes descriptor root") from error
    if not path.is_file():
        raise ValueError("correctness artifact must be a regular file")
    return path


def build_correctness_evidence_index(
    reference_paths,
    candidate_paths,
    manifest,
    candidate_evidence,
    *,
    descriptor_root,
):
    """Recompare the exact correctness matrix and return a deterministic index."""
    if not isinstance(candidate_evidence, dict):
        raise ValueError("candidate evidence must be an object")
    descriptor_root = _descriptor_root(descriptor_root)
    _descriptor_candidate_evidence(candidate_evidence)
    commit = candidate_evidence.get("candidate_git_commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        or candidate_evidence.get("candidate_git_status") != ""
    ):
        raise ValueError("candidate evidence must identify one clean full commit")

    references = {}
    for path in reference_paths:
        resolved, metadata = _archive_record(path, manifest, "reference")
        name = metadata["workload"]["name"]
        if name in references:
            raise ValueError(f"duplicate reference archive for {name}")
        references[name] = (resolved, metadata)
    candidates = {}
    for path in candidate_paths:
        resolved, metadata = _archive_record(path, manifest, "candidate")
        name = metadata["workload"]["name"]
        if name in candidates:
            raise ValueError(f"duplicate candidate archive for {name}")
        if metadata["backend"] != "torch":
            raise ValueError(f"candidate archive for {name} is not Torch")
        backend = metadata.get("backend_metadata", {})
        if backend.get("producer") != PRODUCER or backend.get(
            "solver_abi"
        ) != candidate_evidence.get("solver_abi"):
            raise ValueError(f"candidate backend metadata differs for {name}")
        provenance = metadata["provenance"]
        if any(
            provenance[key]["commit"] != commit
            or provenance[key]["git_status"] != ""
            or provenance[key]["clean"] is not True
            for key in ("source", "controller")
        ):
            raise ValueError(f"candidate provenance differs for {name}")
        candidates[name] = (resolved, metadata)

    required = _required_cases(manifest)
    required_names = [case["name"] for _group, case in required]
    if set(references) != set(required_names) or set(candidates) != set(required_names):
        raise ValueError(
            "correctness evidence archives do not exactly cover correctness and "
            "physical_checks"
        )
    runtime_modes = [_runtime_mode(metadata) for _path, metadata in candidates.values()]
    runtime_mode = runtime_modes[0]
    if any(
        not native_oracle._same_json_value(value, runtime_mode)
        for value in runtime_modes[1:]
    ):
        raise ValueError("candidate correctness archives mix runtime modes")
    artifacts = []
    for group, case in required:
        name = case["name"]
        reference_path, reference_metadata = references[name]
        candidate_path, candidate_metadata = candidates[name]
        comparison = compare_torch_archives(
            reference_path,
            candidate_path,
            manifest,
            include_tolerances=True,
        )
        tolerance_results = comparison.pop("tolerance_results", None)
        if comparison.get("passed") is not True or comparison.get("failures") != []:
            raise ValueError(f"Torch correctness comparison failed for {name}")
        if not _tolerance_results_complete(tolerance_results):
            raise ValueError(f"Torch tolerance results are invalid for {name}")
        artifacts.append(
            {
                "case": name,
                "group": group,
                "reference": _artifact_descriptor(
                    reference_path,
                    descriptor_root,
                    candidate_evidence,
                    "application/x-npz",
                ),
                "reference_observer_commit": reference_metadata["provenance"]["source"][
                    "commit"
                ],
                "candidate": _artifact_descriptor(
                    candidate_path,
                    descriptor_root,
                    candidate_evidence,
                    "application/x-npz",
                ),
                "candidate_provenance": {
                    "commit": candidate_metadata["provenance"]["source"]["commit"],
                    "source_sha256": candidate_metadata["provenance"]["source"][
                        "source_sha256"
                    ],
                    "controller_sha256": candidate_metadata["provenance"]["controller"][
                        "source_sha256"
                    ],
                },
                "comparison": comparison,
                "tolerance_results": tolerance_results,
            }
        )
    return {
        "schema_version": 1,
        "kind": INDEX_KIND,
        "contract_id": INDEX_CONTRACT,
        "manifest_contract_sha256": _canonical_sha256(manifest),
        "candidate_evidence": candidate_evidence,
        "runtime_mode": runtime_mode,
        "required_cases": required_names,
        "artifacts": artifacts,
        "suite_acceptance": {
            "correctness_case_count": len(manifest.get("correctness", ())),
            "physical_check_case_count": len(manifest.get("physical_checks", ())),
            "evaluated_case_count": len(artifacts),
            "complete_fields": True,
            "persistent_state": True,
            "source_and_auxiliary_state": True,
            "physical_observables": True,
            "passed": True,
        },
    }


def _hex_string(value, width):
    return (
        isinstance(value, str)
        and len(value) == width
        and all(character in "0123456789abcdef" for character in value)
    )


def _portable_descriptor_complete(descriptor, expected_evidence, media_type):
    try:
        _exact_keys(
            descriptor,
            {"path", "sha256", "size_bytes", "media_type", "candidate_evidence"},
            "correctness artifact descriptor",
        )
        _canonical_descriptor_path(descriptor["path"])
        return (
            _hex_string(descriptor["sha256"], 64)
            and type(descriptor["size_bytes"]) is int
            and descriptor["size_bytes"] > 0
            and descriptor["media_type"] == media_type
            and native_oracle._same_json_value(
                descriptor["candidate_evidence"],
                _descriptor_candidate_evidence(expected_evidence),
            )
        )
    except AttributeError, KeyError, TypeError, ValueError, OverflowError:
        return False


def _tolerance_results_complete(results):
    try:
        if not isinstance(results, list) or not results:
            return False
        keys = []
        for result in results:
            _exact_keys(
                result,
                {"key", "dtype", "scope", "rtol", "atol", "max_abs_error"},
                "correctness tolerance result",
            )
            keys.append(result["key"])
            if not all(
                isinstance(result[name], str) and result[name]
                for name in ("key", "dtype", "scope")
            ):
                return False
            for name in ("rtol", "atol", "max_abs_error"):
                value = result[name]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not np.isfinite(value)
                    or value < 0
                ):
                    return False
        return keys == sorted(set(keys))
    except AttributeError, KeyError, TypeError, ValueError:
        return False


def _artifact_binding_complete(artifact, group, name, manifest, expected_evidence):
    try:
        _exact_keys(
            artifact,
            {
                "case",
                "group",
                "reference",
                "reference_observer_commit",
                "candidate",
                "candidate_provenance",
                "comparison",
                "tolerance_results",
            },
            f"correctness artifact {name}",
        )
        reference = artifact["reference"]
        candidate = artifact["candidate"]
        provenance = artifact["candidate_provenance"]
        _exact_keys(
            provenance,
            {"commit", "source_sha256", "controller_sha256"},
            f"correctness candidate provenance {name}",
        )
        return (
            artifact["case"] == name
            and artifact["group"] == group
            and native_oracle._same_json_value(
                artifact["comparison"], {"passed": True, "failures": []}
            )
            and _portable_descriptor_complete(
                reference, expected_evidence, "application/x-npz"
            )
            and artifact["reference_observer_commit"]
            == manifest["reference"]["observer_commit"]
            and _portable_descriptor_complete(
                candidate, expected_evidence, "application/x-npz"
            )
            and reference["path"] != candidate["path"]
            and _tolerance_results_complete(artifact["tolerance_results"])
            and provenance["commit"] == expected_evidence["candidate_git_commit"]
            and _hex_string(provenance["commit"], 40)
            and _hex_string(provenance["source_sha256"], 64)
            and _hex_string(provenance["controller_sha256"], 64)
        )
    except AttributeError, KeyError, TypeError, ValueError:
        return False


def _source_artifact_complete(source, expected_evidence):
    return _portable_descriptor_complete(source, expected_evidence, "application/json")


def _runtime_mode_complete(mode):
    try:
        _exact_keys(
            mode,
            {"device", "precision", "graph_mode", "compile_policy", "compile_mode"},
            "correctness runtime_mode",
        )
        expected = _runtime_contract(
            mode["device"],
            mode["precision"],
            mode["graph_mode"],
            mode["compile_mode"],
        )
        return native_oracle._same_json_value(mode, expected)
    except AttributeError, KeyError, TypeError, ValueError:
        return False


def correctness_binding_complete(
    index, manifest, expected_evidence, *, require_source_artifact=False
):
    """Return whether an already-revalidated evidence index binds exactly."""
    try:
        _exact_keys(
            index,
            {
                "schema_version",
                "kind",
                "contract_id",
                "manifest_contract_sha256",
                "candidate_evidence",
                "runtime_mode",
                "required_cases",
                "artifacts",
                "suite_acceptance",
            },
            "correctness index",
            optional={"source_artifact"},
        )
        required_pairs = _required_cases(manifest)
        required = [case["name"] for _group, case in required_pairs]
        suite = index["suite_acceptance"]
        _exact_keys(
            suite,
            {
                "correctness_case_count",
                "physical_check_case_count",
                "evaluated_case_count",
                "complete_fields",
                "persistent_state",
                "source_and_auxiliary_state",
                "physical_observables",
                "passed",
            },
            "correctness index suite_acceptance",
        )
        artifacts = index["artifacts"]
        candidate_binding_complete = (
            isinstance(expected_evidence, dict)
            and _hex_string(expected_evidence.get("candidate_git_commit"), 40)
            and expected_evidence.get("candidate_git_status") == ""
            and _hex_string(expected_evidence.get("manifest_sha256"), 64)
            and isinstance(expected_evidence.get("solver_abi"), str)
            and bool(expected_evidence["solver_abi"])
        )
        artifacts_complete = (
            isinstance(artifacts, list)
            and len(artifacts) == len(required_pairs)
            and all(
                _artifact_binding_complete(
                    artifact,
                    group,
                    case["name"],
                    manifest,
                    expected_evidence,
                )
                for artifact, (group, case) in zip(
                    artifacts, required_pairs, strict=True
                )
            )
        )
        artifact_paths_complete = artifacts_complete and len(
            {
                artifact[role]["path"]
                for artifact in artifacts
                for role in ("reference", "candidate")
            }
        ) == 2 * len(artifacts)
        source = index.get("source_artifact")
        source_complete = source is None or _source_artifact_complete(
            source, expected_evidence
        )
        return (
            type(index["schema_version"]) is int
            and index["schema_version"] == 1
            and index["kind"] == INDEX_KIND
            and index["contract_id"] == INDEX_CONTRACT
            and index["manifest_contract_sha256"] == _canonical_sha256(manifest)
            and native_oracle._same_json_value(
                index["candidate_evidence"], expected_evidence
            )
            and candidate_binding_complete
            and _runtime_mode_complete(index["runtime_mode"])
            and index["required_cases"] == required
            and artifacts_complete
            and artifact_paths_complete
            and source_complete
            and (not require_source_artifact or source is not None)
            and native_oracle._same_json_value(
                suite,
                {
                    "correctness_case_count": len(manifest.get("correctness", ())),
                    "physical_check_case_count": len(
                        manifest.get("physical_checks", ())
                    ),
                    "evaluated_case_count": len(required),
                    "complete_fields": True,
                    "persistent_state": True,
                    "source_and_auxiliary_state": True,
                    "physical_observables": True,
                    "passed": True,
                },
            )
        )
    except AttributeError, KeyError, TypeError, ValueError:
        return False


def load_correctness_evidence_index(
    path, manifest, expected_evidence, *, descriptor_root
):
    path = Path(path).resolve(strict=True)
    descriptor_root = _descriptor_root(descriptor_root)
    _relative_descriptor_path(path, descriptor_root)
    try:
        document = json.loads(
            path.read_bytes(),
            object_pairs_hook=native_oracle._object_without_duplicate_keys,
            parse_constant=native_oracle._reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("correctness evidence index is not valid JSON") from error
    if not isinstance(document, dict):
        raise ValueError("correctness evidence index must be an object")
    if not correctness_binding_complete(document, manifest, expected_evidence):
        raise ValueError("correctness evidence index differs from recomputed evidence")
    references = [
        _resolve_artifact_descriptor(descriptor_root, item["reference"])
        for item in document.get("artifacts", ())
    ]
    candidates = [
        _resolve_artifact_descriptor(descriptor_root, item["candidate"])
        for item in document.get("artifacts", ())
    ]
    rebuilt = build_correctness_evidence_index(
        references,
        candidates,
        manifest,
        expected_evidence,
        descriptor_root=descriptor_root,
    )
    if not native_oracle._same_json_value(document, rebuilt):
        raise ValueError("correctness evidence index differs from recomputed evidence")
    result = deepcopy(rebuilt)
    result["source_artifact"] = _artifact_descriptor(
        path, descriptor_root, expected_evidence, "application/json"
    )
    return result


def _load_candidate_evidence(path):
    try:
        value = json.loads(
            Path(path).read_bytes(),
            object_pairs_hook=native_oracle._object_without_duplicate_keys,
            parse_constant=native_oracle._reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("candidate evidence is not valid JSON") from error
    if isinstance(value, dict) and "evidence" in value:
        value = value["evidence"]
    if not isinstance(value, dict):
        raise ValueError("candidate evidence JSON must contain an object")
    if not native_oracle._finite_json_value(value):
        raise ValueError("candidate evidence contains a non-finite value")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("--reference", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--threads", type=int, default=1)
    capture.add_argument("--device", default="cpu")
    capture.add_argument(
        "--precision", choices=("float32", "float64"), default="float64"
    )
    capture.add_argument("--graph-mode", choices=("eager", "graph"), default="eager")
    capture.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    index = commands.add_parser("index")
    index.add_argument("--references", type=Path, nargs="+", required=True)
    index.add_argument("--candidates", type=Path, nargs="+", required=True)
    index.add_argument("--candidate-evidence", type=Path, required=True)
    index.add_argument("--descriptor-root", type=Path, required=True)
    index.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate-index")
    validate.add_argument("--index", type=Path, required=True)
    validate.add_argument("--candidate-evidence", type=Path, required=True)
    validate.add_argument("--descriptor-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = native_oracle.load_manifest(args.manifest)
    if args.command == "capture":
        if args.threads < 1:
            parser.error("threads must be positive")
        value = capture_torch_candidate(
            args.reference,
            manifest,
            args.output,
            threads=args.threads,
            device=args.device,
            precision=args.precision,
            graph_mode=args.graph_mode,
            compile_mode=args.compile_mode,
        )
    elif args.command == "index":
        evidence = _load_candidate_evidence(args.candidate_evidence)
        if evidence.get("manifest_sha256") != _sha256(args.manifest):
            raise ValueError("candidate evidence manifest bytes differ")
        value = build_correctness_evidence_index(
            args.references,
            args.candidates,
            manifest,
            evidence,
            descriptor_root=args.descriptor_root,
        )
        rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        evidence = _load_candidate_evidence(args.candidate_evidence)
        if evidence.get("manifest_sha256") != _sha256(args.manifest):
            raise ValueError("candidate evidence manifest bytes differ")
        value = load_correctness_evidence_index(
            args.index,
            manifest,
            evidence,
            descriptor_root=args.descriptor_root,
        )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
