#!/usr/bin/env python3
"""Benchmark FDTD initialization and field updates without simulation output."""

import argparse
import json
import os
import platform
import resource
from statistics import median
from time import perf_counter

import numpy as np


def build_simulation(case, gmes):
    """Construct one representative benchmark simulation without initializing it."""
    if case == "small":
        space = gmes.Cartesian(size=(2, 2, 0), resolution=20)
        medium = gmes.Dielectric()
        simulation_type = gmes.TMzFDTD
        source_component = gmes.Ez
    elif case == "2d":
        space = gmes.Cartesian(size=(10, 10, 0), resolution=20)
        medium = gmes.Dielectric()
        simulation_type = gmes.TMzFDTD
        source_component = gmes.Ez
    elif case == "3d":
        space = gmes.Cartesian(size=(5, 5, 5), resolution=10)
        medium = gmes.Dielectric()
        simulation_type = gmes.FDTD
        source_component = gmes.Ex
    elif case == "dispersive":
        space = gmes.Cartesian(size=(12, 12, 0), resolution=20)
        medium = gmes.Drude(dps=(gmes.DrudePole(omega=1.0, gamma=0.1),))
        simulation_type = gmes.TMzFDTD
        source_component = gmes.Ez
    elif case == "lorentz":
        space = gmes.Cartesian(size=(12, 12, 0), resolution=20)
        medium = gmes.Lorentz(lps=(gmes.LorentzPole(amp=0.2, omega=1.0, gamma=0.1),))
        simulation_type = gmes.TMzFDTD
        source_component = gmes.Ez
    elif case == "dcp":
        space = gmes.Cartesian(size=(12, 12, 0), resolution=20)
        medium = gmes.DcpAde(
            dps=(gmes.DrudePole(omega=1.0, gamma=0.1),),
            cps=(gmes.CriticalPoint(amp=0.1, phi=0.2, omega=1.5, gamma=0.1),),
        )
        simulation_type = gmes.TMzFDTD
        source_component = gmes.Ez
    elif case == "dm2":
        space = gmes.Cartesian(size=(4, 4, 0), resolution=10)
        medium = gmes.Dm2(
            omega=(0.9, 1.1),
            n_atom=(0.05, 0.05),
            gamma=0.05,
            rtol=1e-4,
        )
        simulation_type = gmes.TMzFDTD
        source_component = gmes.Ez
    elif case == "pml":
        space = gmes.Cartesian(size=(10, 10, 0), resolution=20)
        medium = gmes.Dielectric()
        simulation_type = gmes.TMzFDTD
        source_component = gmes.Ez
    elif case == "heterogeneous":
        space = gmes.Cartesian(size=(10, 10, 0), resolution=20)
        medium = gmes.Dielectric()
        simulation_type = gmes.TMzFDTD
        source_component = gmes.Ez
    elif case == "complex":
        space = gmes.Cartesian(size=(10, 10, 0), resolution=20)
        medium = gmes.Dielectric()
        simulation_type = gmes.TMzFDTD
        source_component = gmes.Ez
    else:
        raise ValueError(f"unknown benchmark case: {case}")

    geometry = [gmes.DefaultMedium(material=medium)]
    if case == "heterogeneous":
        geometry.extend(
            gmes.Cylinder(
                material=gmes.Dielectric(eps_inf=4.0),
                center=(x, y, 0),
                axis=(0, 0, 1),
                radius=0.6,
            )
            for x in (-3, -1, 1, 3)
            for y in (-3, -1, 1, 3)
        )
    shell_material = gmes.Upml() if case == "pml" else gmes.Cpml()
    geometry.append(gmes.Shell(material=shell_material))
    sources = [
        gmes.PointSource(
            src_time=gmes.Continuous(freq=0.8),
            center=(0, 0, 0),
            component=source_component,
        )
    ]
    kwargs = {"bloch": (0.1, 0.2, 0)} if case == "complex" else {}
    return simulation_type(space, geometry, sources, verbose=False, **kwargs)


def field_checksum(simulation):
    """Return a deterministic scalar summary of all active field arrays."""
    return float(
        sum(
            np.sum(np.abs(simulation.field[component]))
            for component in simulation.field
        )
    )


def material_update_sizes(simulation):
    """Return the cell count handled by each native material updater."""
    return sorted(
        (
            {
                "cells": updater.idx_size(),
                "component": component.__name__,
                "material": type(updater).__name__,
                "plan_bytes": updater.plan_bytes(),
                "plan_runs": updater.plan_run_count(),
            }
            for component, updaters in simulation.pw_material.items()
            for updater in updaters.values()
        ),
        key=lambda item: (item["component"], item["material"]),
    )


def field_shapes(simulation):
    """Return array shapes and dtypes for every field component."""
    return {
        component.__name__: {
            "dtype": str(field.dtype),
            "shape": list(field.shape),
        }
        for component, field in simulation.field.items()
    }


def initialize_simulation(case, gmes):
    """Time construction and FDTD.init() independently for one simulation."""
    start = perf_counter()
    simulation = build_simulation(case, gmes)
    construction_seconds = perf_counter() - start

    start = perf_counter()
    simulation.init()
    initialization_seconds = perf_counter() - start
    return simulation, construction_seconds, initialization_seconds


def peak_rss_bytes():
    """Return the process peak resident-set size in bytes."""
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(maximum if platform.system() == "Darwin" else maximum * 1024)


def run_case(case, warmup, steps, repeats, gmes):
    """Measure repeated initialization and step timings for one case."""
    construction_samples = []
    initialization_samples = []
    simulation = None
    for _ in range(repeats):
        simulation = None
        simulation, construction_seconds, initialization_seconds = (
            initialize_simulation(case, gmes)
        )
        construction_samples.append(construction_seconds)
        initialization_samples.append(initialization_seconds)

    for _ in range(warmup):
        simulation.step()

    step_samples = []
    for _ in range(repeats):
        start = perf_counter()
        for _ in range(steps):
            simulation.step()
        step_samples.append((perf_counter() - start) / steps)

    return {
        "case": case,
        "seconds_per_construction": construction_samples,
        "median_seconds_per_construction": median(construction_samples),
        "seconds_per_initialization": initialization_samples,
        "median_seconds_per_initialization": median(initialization_samples),
        "seconds_per_step": step_samples,
        "median_seconds_per_step": median(step_samples),
        "checksum": field_checksum(simulation),
        "field_shapes": field_shapes(simulation),
        "material_update_sizes": material_update_sizes(simulation),
        "native_update_plan_bytes": sum(
            updater.plan_bytes()
            for updaters in simulation.pw_material.values()
            for updater in updaters.values()
        ),
        "peak_rss_bytes": peak_rss_bytes(),
    }


def main():
    """Parse benchmark options and emit machine-readable JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=(
            "small",
            "2d",
            "3d",
            "dispersive",
            "lorentz",
            "dcp",
            "dm2",
            "pml",
            "heterogeneous",
            "complex",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--threshold",
        type=int,
        help="override GMES_OPENMP_THRESHOLD for this process",
    )
    args = parser.parse_args()

    if args.warmup < 0 or args.steps < 1 or args.repeats < 1 or args.threads < 1:
        parser.error(
            "warmup must be nonnegative; steps, repeats, and threads must be positive"
        )
    if args.threshold is not None and args.threshold < 0:
        parser.error("threshold must be nonnegative")

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    if args.threshold is not None:
        os.environ["GMES_OPENMP_THRESHOLD"] = str(args.threshold)

    import gmes

    cases = (
        (
            "small",
            "2d",
            "3d",
            "dispersive",
            "lorentz",
            "dcp",
            "dm2",
            "pml",
            "heterogeneous",
            "complex",
        )
        if args.case == "all"
        else (args.case,)
    )
    result = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "openmp_enabled": gmes.pw_material.openmp_enabled(),
        "openmp_max_threads": gmes.pw_material.openmp_max_threads(),
        "openmp_cell_threshold": gmes.pw_material.openmp_cell_threshold(),
        "openmp_threads": args.threads,
        "warmup_steps": args.warmup,
        "steps_per_repeat": args.steps,
        "repeats": args.repeats,
        "cases": [
            run_case(case, args.warmup, args.steps, args.repeats, gmes)
            for case in cases
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
