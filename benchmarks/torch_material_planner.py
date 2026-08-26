#!/usr/bin/env python3
"""Benchmark occupancy-aware Torch material planning and simple execution."""

import argparse
import json
import os
import platform
import resource
from statistics import median
from time import perf_counter

import numpy as np
import torch

COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
EXECUTABLE_CASES = frozenset(
    ("homogeneous", "heterogeneous-16", "many-regions", "collapsed-bloch")
)
CASES = (
    "homogeneous",
    "heterogeneous-16",
    "many-regions",
    "coverage-contiguous-1",
    "coverage-contiguous-10",
    "coverage-contiguous-50",
    "coverage-contiguous-90",
    "coverage-fragmented-1",
    "coverage-fragmented-10",
    "coverage-fragmented-50",
    "coverage-fragmented-90",
    "state-widths",
    "collapsed-bloch",
)


def _peak_rss_bytes():
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(maximum if platform.system() == "Darwin" else maximum * 1024)


def _default_geometry(gmes):
    return [gmes.DefaultMedium(gmes.Dielectric(eps_inf=1.7, mu_inf=1.05))]


def _heterogeneous_geometry(gmes):
    geometry = _default_geometry(gmes)
    geometry.extend(
        gmes.Cylinder(
            material=gmes.Dielectric(eps_inf=2.0 + 0.125 * index),
            center=(x, y, 0),
            axis=(0, 0, 1),
            radius=0.55,
        )
        for index, (x, y) in enumerate(
            (x, y) for x in (-3, -1, 1, 3) for y in (-3, -1, 1, 3)
        )
    )
    return geometry


def _many_region_geometry(gmes, count=1000):
    geometry = _default_geometry(gmes)
    columns = 40
    for index in range(count):
        x_index, y_index = divmod(index, columns)
        geometry.append(
            gmes.Block(
                material=gmes.Dielectric(eps_inf=2.5, mu_inf=1.1),
                center=(-9.75 + 0.5 * y_index, -6.0 + 0.5 * x_index, 0),
                size=(0.2, 0.2, 1.0),
            )
        )
    return geometry


def _coverage_geometry(gmes, coverage, layout):
    geometry = _default_geometry(gmes)
    fraction = coverage / 100
    if layout == "contiguous":
        geometry.append(
            gmes.Block(
                material=gmes.Drude(
                    eps_inf=1.2,
                    dps=(gmes.DrudePole(omega=0.8, gamma=0.03),),
                ),
                center=(-5 + 10 * fraction / 2, 0, 0),
                size=(10 * fraction, 10, 1.0),
            )
        )
        return geometry
    columns = 120
    selected_count = {1: 1, 10: 8, 50: 45, 90: 95}[coverage]
    selected = np.random.default_rng(117 + coverage).choice(
        columns, size=selected_count, replace=False
    )
    material = gmes.Drude(
        eps_inf=1.2,
        dps=(gmes.DrudePole(omega=0.8, gamma=0.03),),
    )
    cell_width = 10 / columns
    for column in selected:
        geometry.append(
            gmes.Block(
                material=material,
                center=(-5 + (column + 0.5) * cell_width, 0, 0),
                size=(1.1 * cell_width, 10, 1.0),
            )
        )
    return geometry


def _state_width_geometry(gmes):
    geometry = _default_geometry(gmes)
    for index, width in enumerate((1, 2, 4, 8)):
        geometry.append(
            gmes.Block(
                material=gmes.Drude(
                    eps_inf=1.2 + 0.05 * index,
                    dps=tuple(
                        gmes.DrudePole(omega=0.6 + 0.05 * pole, gamma=0.03)
                        for pole in range(width)
                    ),
                ),
                center=(-3 + 2 * index, 0, 0),
                size=(1.5, 6, 1.0),
            )
        )
    return geometry


def build_case(case, gmes):
    """Return space, geometry, and optional Bloch vector for one fixed workload."""
    if case == "homogeneous":
        return gmes.Cartesian((10, 10, 0), 12), _default_geometry(gmes), None
    if case == "heterogeneous-16":
        return (
            gmes.Cartesian((10, 10, 0), 20),
            _heterogeneous_geometry(gmes),
            None,
        )
    if case == "many-regions":
        return (
            gmes.Cartesian((20, 14, 0), 8),
            _many_region_geometry(gmes),
            None,
        )
    if case.startswith("coverage-"):
        _, layout, coverage = case.split("-")
        return (
            gmes.Cartesian((10, 10, 0), 12),
            _coverage_geometry(gmes, int(coverage), layout),
            None,
        )
    if case == "state-widths":
        return (
            gmes.Cartesian((10, 8, 0), 12),
            _state_width_geometry(gmes),
            None,
        )
    if case == "collapsed-bloch":
        return (
            gmes.Cartesian((10, 0, 0), 32),
            _heterogeneous_geometry(gmes),
            (0.07, 0, 0),
        )
    raise ValueError(f"unknown planner benchmark case: {case}")


def _time_step(space, geometry):
    material = geometry[0].material
    dr = tuple(float(value) for value in space.dr)
    return (
        0.99
        * np.sqrt(material.eps_inf * material.mu_inf)
        / np.sqrt(sum(value**-2 for value in dr))
    )


def build_host_plan(
    case,
    *,
    policy,
    precision,
    device_type,
    tile_size,
    gmes,
):
    """Build a validated host plan, including stateful synthetic workloads."""
    from gmes.geometry import GeomBoxTree
    from gmes.torch_fdtd import _field_shapes
    from gmes.torch_plan import TorchExecutionPlanner

    space, geometry, bloch = build_case(case, gmes)
    space.dt = _time_step(space, geometry)
    start = perf_counter()
    for geometric_object in geometry:
        geometric_object.init(space)
    tree = GeomBoxTree(tuple(geometry))
    plans = TorchExecutionPlanner(
        geom_tree=tree,
        space=space,
        shapes=_field_shapes(space),
        precision=precision,
        device_type=device_type,
        policy=policy,
        execution_tile_size=tile_size,
    ).build()
    seconds = perf_counter() - start
    return space, geometry, bloch, plans, seconds


def plan_summary(plans):
    """Return stable launch, memory, signature, and policy metadata."""
    active = sum(plan.active_count for plan in plans)
    bytes_used = sum(
        plan.material_ids.nbytes
        + plan.underlying_ids.nbytes
        + plan.ownership.nbytes
        + plan.dense_inverse.nbytes
        + plan.constant_targets.nbytes
        + plan.constant_values.nbytes
        + sum(bucket.estimated_bytes for bucket in plan.buckets)
        for plan in plans
    )
    signatures = sorted(
        {
            (
                bucket.signature.model,
                bucket.signature.component,
                bucket.signature.precision,
                bucket.signature.state_shape,
            )
            for plan in plans
            for bucket in plan.buckets
        }
    )
    return {
        "active_component_cells": active,
        "estimated_plan_bytes": bytes_used,
        "bytes_per_active_component_cell": bytes_used / max(1, active),
        "material_launches_per_step": sum(plan.launch_count for plan in plans),
        "signature_count": len(signatures),
        "signatures": [
            {
                "model": model,
                "component": component,
                "precision": precision,
                "state_shape": state_shape,
            }
            for model, component, precision, state_shape in signatures
        ],
        "decisions": [plan.decision_record() for plan in plans],
    }


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _profile(simulation, steps):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if simulation.device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        profile_memory=True,
        record_shapes=False,
    ) as profile:
        simulation.advance(steps)
        _synchronize(simulation.device)
    events = profile.key_averages()
    return {
        "event_count": sum(event.count for event in events),
        "positive_self_memory_events": sum(
            event.count for event in events if event.self_cpu_memory_usage > 0
        ),
        "top_operations": [
            {"name": event.key, "count": event.count}
            for event in sorted(
                events, key=lambda event: event.self_cpu_time_total, reverse=True
            )[:12]
        ],
    }


def run_native_case(
    case,
    *,
    threads,
    warmup,
    steps,
    repeats,
    gmes,
):
    """Measure the unchanged native simple-material execution on the same case."""
    if case not in EXECUTABLE_CASES:
        return None
    space, geometry, bloch = build_case(case, gmes)
    simulation = gmes.FDTD(space, geometry, bloch=bloch, verbose=False)
    simulation.init()
    rng = np.random.default_rng(117)
    for field in simulation.field.values():
        values = rng.normal(size=field.shape) * 1e-3
        if bloch is not None:
            values = values + 1j * rng.normal(size=field.shape) * 1e-3
        field[...] = values
    for _ in range(warmup):
        simulation.step()
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        for _ in range(steps):
            simulation.step()
        samples.append((perf_counter() - start) / steps)
    cells = int(np.prod(space.my_field_size))
    return {
        "threads": threads,
        "seconds_per_step": samples,
        "median_seconds_per_step": median(samples),
        "cells_per_second": cells / median(samples),
        "field_checksum": float(
            sum(np.abs(field).sum() for field in simulation.field.values())
        ),
    }


def run_case(
    case,
    *,
    policy="auto",
    device="cpu",
    precision="float64",
    compile_policy="eager",
    threads=1,
    warmup=2,
    steps=10,
    repeats=3,
    tile_size=4096,
    profile=False,
    gmes,
):
    """Measure plan creation and supported steady-state execution."""
    device_type = torch.device(device).type
    space, geometry, bloch, plans, plan_seconds = build_host_plan(
        case,
        policy=policy,
        precision=precision,
        device_type=device_type,
        tile_size=tile_size,
        gmes=gmes,
    )
    result = {
        "case": case,
        "policy": policy,
        "device": device,
        "precision": precision,
        "compile_policy": compile_policy,
        "geometry_objects": len(geometry),
        "component_shapes": {plan.name: plan.shape for plan in plans},
        "plan_creation_seconds": plan_seconds,
        "peak_rss_bytes": _peak_rss_bytes(),
        **plan_summary(plans),
    }
    if case not in EXECUTABLE_CASES:
        result["execution"] = "plan-only; state equations are implemented by #118-#120"
        return result

    start = perf_counter()
    simulation = gmes.TorchSimulation(
        space=space,
        geometry=geometry,
        runtime=gmes.TorchRuntimeConfig(
            device=device,
            precision=precision,
            compile_policy=compile_policy,
            cpu_threads=threads,
            execution_policy=policy,
            planner_tile_size=tile_size,
        ),
        bloch=bloch,
        dt=space.dt,
    )
    finalization_seconds = perf_counter() - start
    rng = np.random.default_rng(117)
    fields = {
        name: rng.normal(size=tuple(field.shape)) * 1e-3
        for name, field in simulation.state.fields().items()
    }
    simulation.load_host_fields(fields)
    simulation.advance(warmup)
    _synchronize(simulation.device)
    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats(simulation.device)
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        simulation.advance(steps)
        _synchronize(simulation.device)
        samples.append((perf_counter() - start) / steps)
    cells = int(np.prod(space.my_field_size))
    result.update(
        {
            "tensor_finalization_seconds": finalization_seconds,
            "seconds_per_step": samples,
            "median_seconds_per_step": median(samples),
            "cells_per_second": cells / median(samples),
            "plan_tensor_bytes": sum(
                value.numel() * value.element_size()
                for _, value in simulation.plan.named_buffers()
            ),
            "field_checksum": float(
                sum(
                    value.detach().abs().sum().cpu().item()
                    for value in simulation.state.fields().values()
                )
            ),
            "peak_rss_bytes": _peak_rss_bytes(),
            "peak_device_bytes": (
                torch.cuda.max_memory_allocated(simulation.device)
                if device_type == "cuda"
                else 0
            ),
        }
    )
    if profile:
        result["profile"] = _profile(simulation, max(1, min(steps, 5)))
    return result


def run_policy_matrix(case, **kwargs):
    """Measure auto and all forced candidates from the same workload contract."""
    results = {
        policy: run_case(case, policy=policy, **kwargs)
        for policy in ("auto", "dense", "compact", "tiled")
    }
    if case in EXECUTABLE_CASES:
        fastest = min(
            results[policy]["median_seconds_per_step"]
            for policy in ("dense", "compact", "tiled")
        )
        auto = results["auto"]["median_seconds_per_step"]
        ratio = auto / fastest
    else:
        ratio = None
    return {
        "case": case,
        "auto_to_fastest_forced_ratio": ratio,
        "within_ten_percent": ratio is None or ratio <= 1.10,
        "results": results,
    }


def main():
    """Parse arguments and print machine-readable benchmark evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES + ("all",), default="homogeneous")
    parser.add_argument(
        "--policy",
        choices=("auto", "dense", "compact", "tiled", "matrix"),
        default="auto",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--precision", choices=("float32", "float64"), default="float64"
    )
    parser.add_argument(
        "--compile-policy", choices=("eager", "compile"), default="eager"
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tile-size", type=int, default=4096)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--native-reference", action="store_true")
    args = parser.parse_args()
    if (
        args.threads < 1
        or args.warmup < 0
        or args.steps < 1
        or args.repeats < 1
        or args.tile_size < 1
    ):
        parser.error("thread, step, repeat, and tile counts must be positive")

    if args.native_reference and torch.device(args.device).type != "cpu":
        parser.error("--native-reference requires a CPU Torch run")

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    import gmes

    kwargs = {
        "device": args.device,
        "precision": args.precision,
        "compile_policy": args.compile_policy,
        "threads": args.threads,
        "warmup": args.warmup,
        "steps": args.steps,
        "repeats": args.repeats,
        "tile_size": args.tile_size,
        "profile": args.profile,
        "gmes": gmes,
    }
    cases = CASES if args.case == "all" else (args.case,)
    if args.policy == "matrix":
        torch_output = [run_policy_matrix(case, **kwargs) for case in cases]
    else:
        torch_output = [run_case(case, policy=args.policy, **kwargs) for case in cases]
    if args.native_reference:
        native_output = {
            case: run_native_case(
                case,
                threads=args.threads,
                warmup=args.warmup,
                steps=args.steps,
                repeats=args.repeats,
                gmes=gmes,
            )
            for case in cases
            if case in EXECUTABLE_CASES
        }
        gates = {}
        for item in torch_output:
            torch_result = item["results"]["auto"] if args.policy == "matrix" else item
            native_result = native_output.get(item["case"])
            if native_result is None:
                continue
            ratio = (
                torch_result["median_seconds_per_step"]
                / native_result["median_seconds_per_step"]
            )
            gates[item["case"]] = {
                "torch_to_native_step_ratio": ratio,
                "within_five_percent": ratio <= 1.05,
            }
        output = {
            "torch": torch_output,
            "native": native_output,
            "native_non_regression": gates,
        }
    else:
        output = torch_output
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
