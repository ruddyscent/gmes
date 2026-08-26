#!/usr/bin/env python3
"""Benchmark Torch DM2 width, coverage, fragmentation, and convergence."""

import argparse
import json
import platform
import resource
from statistics import median
from time import perf_counter

import numpy as np
import torch

CASES = (
    "width-1",
    "width-4",
    "width-8",
    "mixed-widths",
    "coverage-10-contiguous",
    "coverage-10-fragmented",
    "coverage-50-contiguous",
    "coverage-50-fragmented",
    "hard-nonconverging",
)


def _peak_rss_bytes():
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(maximum if platform.system() == "Darwin" else maximum * 1024)


def _material(gmes, width, *, hard=False, offset=0.0):
    return gmes.Dm2(
        eps_inf=1.4,
        omega=tuple(0.7 + offset + 0.08 * index for index in range(width)),
        n_atom=tuple(0.2 + 0.02 * index for index in range(width)),
        rho30=-0.8,
        gamma=0.15,
        t1=2.5,
        t2=1.7,
        hbar=1.2,
        rtol=-1 if hard else 1e-6,
    )


def _default_geometry(gmes):
    return [gmes.DefaultMedium(gmes.Dielectric(eps_inf=1.7, mu_inf=1.05))]


def _coverage_geometry(gmes, coverage, layout):
    geometry = _default_geometry(gmes)
    material = _material(gmes, 4)
    fraction = coverage / 100
    if layout == "contiguous":
        geometry.append(
            gmes.Block(
                material,
                center=(-4 + 8 * fraction / 2, 0, 0),
                size=(8 * fraction, 8, 1),
            )
        )
        return geometry
    columns = 80
    selected_count = round(columns * fraction)
    selected = np.random.default_rng(120 + coverage).choice(
        columns, size=selected_count, replace=False
    )
    width = 8 / columns
    for column in selected:
        geometry.append(
            gmes.Block(
                material,
                center=(-4 + (column + 0.5) * width, 0, 0),
                size=(1.05 * width, 8, 1),
            )
        )
    return geometry


def build_case(case, gmes):
    """Return a fixed 2-D DM2 workload."""
    space = gmes.Cartesian((8, 8, 0), 6)
    if case.startswith("width-"):
        width = int(case.split("-")[1])
        geometry = _default_geometry(gmes)
        geometry.append(
            gmes.Block(_material(gmes, width), center=(0, 0, 0), size=(8, 8, 1))
        )
        return space, geometry
    if case == "mixed-widths":
        geometry = _default_geometry(gmes)
        for index, width in enumerate((1, 2, 4, 8)):
            geometry.append(
                gmes.Block(
                    _material(gmes, width, offset=0.02 * index),
                    center=(-3 + 2 * index, 0, 0),
                    size=(2, 8, 1),
                )
            )
        return space, geometry
    if case.startswith("coverage-"):
        _, coverage, layout = case.split("-")
        return space, _coverage_geometry(gmes, int(coverage), layout)
    if case == "hard-nonconverging":
        geometry = _default_geometry(gmes)
        geometry.append(
            gmes.Block(
                _material(gmes, 4, hard=True),
                center=(0, 0, 0),
                size=(8, 8, 1),
            )
        )
        return space, geometry
    raise ValueError(f"unknown DM2 benchmark case: {case}")


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _state_metrics(simulation):
    persistent = sum(
        bucket.u.numel() * bucket.u.element_size()
        for bucket in simulation.state.dm2_buckets
    )
    scratch = sum(
        value.numel() * value.element_size()
        for name, value in simulation.state.named_buffers()
        if "dm2_buckets" in name and not name.endswith(".u")
    )
    targets = sum(
        bucket.metadata.target_count for bucket in simulation.state.dm2_buckets
    )
    widths = sorted(
        {bucket.metadata.transition_count for bucket in simulation.state.dm2_buckets}
    )
    exact_elements = sum(
        bucket.metadata.target_count * bucket.metadata.transition_count
        for bucket in simulation.state.dm2_buckets
    )
    padded_elements = sum(
        bucket.metadata.target_count * max(widths, default=0)
        for bucket in simulation.state.dm2_buckets
    )
    return {
        "targets": targets,
        "persistent_state_bytes": persistent,
        "scratch_bytes": scratch,
        "transition_widths": widths,
        "exact_transition_elements": exact_elements,
        "bounded_padding_elements": padded_elements,
        "bounded_padding_overhead": (
            padded_elements / exact_elements if exact_elements else 1.0
        ),
    }


def run_case(args, gmes):
    """Construct, warm, and measure one fixed benchmark case."""
    space, geometry = build_case(args.case, gmes)
    runtime = gmes.TorchRuntimeConfig(
        device=args.device,
        precision=args.precision,
        compile_policy=args.compile_policy,
        cpu_threads=args.threads,
        execution_policy=args.policy,
        planner_tile_size=args.tile_size,
    )
    start = perf_counter()
    simulation = gmes.TorchSimulation(
        space=space,
        geometry=geometry,
        runtime=runtime,
    )
    construction_seconds = perf_counter() - start
    rng = np.random.default_rng(120)
    fields = {
        name: rng.normal(size=tuple(field.shape)) * 1e-3
        for name, field in simulation.state.fields().items()
    }
    simulation.load_host_fields(fields)
    hard = args.case == "hard-nonconverging"

    def advance():
        try:
            simulation.advance(args.steps)
        except RuntimeError:
            if not hard:
                raise

    for _ in range(args.warmup):
        advance()
    _synchronize(simulation.device)
    addresses = simulation.buffer_addresses()
    samples = []
    for _ in range(args.repeats):
        start = perf_counter()
        advance()
        _synchronize(simulation.device)
        samples.append((perf_counter() - start) / (1 if hard else args.steps))
    if addresses != simulation.buffer_addresses():
        raise RuntimeError("DM2 storage addresses changed during steady state")

    state = _state_metrics(simulation)
    diagnostics = simulation.diagnostics()
    result = {
        "case": args.case,
        "device": str(simulation.device),
        "precision": args.precision,
        "compile_policy": args.compile_policy,
        "execution_policy": args.policy,
        "construction_seconds": construction_seconds,
        "seconds_per_step": {
            "samples": samples,
            "median": median(samples),
            "minimum": min(samples),
            "maximum": max(samples),
        },
        "dm2_cells_per_second": state["targets"] / median(samples),
        "state": state,
        "iteration_distributions": diagnostics.get("dm2", ()),
        "peak_rss_bytes": _peak_rss_bytes(),
        "fixed_storage": True,
        "hard_failure_observed": bool(
            hard and np.any(simulation.state._dm2_status.cpu().numpy())
        ),
    }
    if simulation.device.type == "cuda":
        result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(
            simulation.device
        )
        result["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(
            simulation.device
        )
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, default="mixed-widths")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--precision", choices=("float32", "float64"), default="float64"
    )
    parser.add_argument(
        "--compile-policy", choices=("eager", "compile"), default="compile"
    )
    parser.add_argument(
        "--policy", choices=("auto", "dense", "compact", "tiled"), default="auto"
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--tile-size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if min(args.warmup, args.steps, args.repeats) < 1:
        raise ValueError("warmup, steps, and repeats must be positive")
    import gmes

    print(json.dumps(run_case(args, gmes), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
