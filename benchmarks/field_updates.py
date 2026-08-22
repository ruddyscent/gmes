#!/usr/bin/env python3
"""Benchmark native FDTD field updates without visualization or file output."""

import argparse
import json
import os
import platform
from statistics import median
from time import perf_counter

import numpy as np


def build_simulation(case):
    """Construct and initialize one representative benchmark simulation."""
    from gmes import (
        FDTD,
        Cartesian,
        Continuous,
        Cpml,
        DefaultMedium,
        Dielectric,
        Drude,
        DrudePole,
        Ex,
        Ez,
        PointSource,
        Shell,
        TMzFDTD,
    )

    if case == "small":
        space = Cartesian(size=(2, 2, 0), resolution=20)
        medium = Dielectric()
        simulation_type = TMzFDTD
        source_component = Ez
    elif case == "2d":
        space = Cartesian(size=(10, 10, 0), resolution=20)
        medium = Dielectric()
        simulation_type = TMzFDTD
        source_component = Ez
    elif case == "3d":
        space = Cartesian(size=(5, 5, 5), resolution=10)
        medium = Dielectric()
        simulation_type = FDTD
        source_component = Ex
    elif case == "dispersive":
        space = Cartesian(size=(12, 12, 0), resolution=20)
        medium = Drude(dps=(DrudePole(omega=1.0, gamma=0.1),))
        simulation_type = TMzFDTD
        source_component = Ez
    else:
        raise ValueError(f"unknown benchmark case: {case}")

    geometry = [DefaultMedium(material=medium), Shell(material=Cpml())]
    sources = [
        PointSource(
            src_time=Continuous(freq=0.8),
            center=(0, 0, 0),
            component=source_component,
        )
    ]
    simulation = simulation_type(space, geometry, sources, verbose=False)
    simulation.init()
    return simulation


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
            }
            for component, updaters in simulation.pw_material.items()
            for updater in updaters.values()
        ),
        key=lambda item: (item["component"], item["material"]),
    )


def run_case(case, warmup, steps, repeats):
    """Measure one case and return timing samples plus its final checksum."""
    simulation = build_simulation(case)
    for _ in range(warmup):
        simulation.step()

    samples = []
    for _ in range(repeats):
        start = perf_counter()
        for _ in range(steps):
            simulation.step()
        samples.append((perf_counter() - start) / steps)

    return {
        "case": case,
        "seconds_per_step": samples,
        "median_seconds_per_step": median(samples),
        "checksum": field_checksum(simulation),
        "material_update_sizes": material_update_sizes(simulation),
    }


def main():
    """Parse benchmark options and emit machine-readable JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("small", "2d", "3d", "dispersive", "all"),
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

    from gmes import pw_material

    cases = ("small", "2d", "3d", "dispersive") if args.case == "all" else (args.case,)
    result = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "openmp_enabled": pw_material.openmp_enabled(),
        "openmp_max_threads": pw_material.openmp_max_threads(),
        "openmp_cell_threshold": pw_material.openmp_cell_threshold(),
        "openmp_threads": args.threads,
        "cases": [
            run_case(case, args.warmup, args.steps, args.repeats) for case in cases
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
