#!/usr/bin/env python3
"""Benchmark bounded geometry-to-region lowering independently of FDTD updates."""

import argparse
import hashlib
import json
import platform
import resource
from statistics import geometric_mean, median
from time import perf_counter

import numpy as np

COMPONENT_NAMES = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def geometry_for_case(case, gmes):
    """Return a space, geometry list, and complex-mode flag for one case."""
    complex_mode = case == "complex"
    if case in {"default-2d", "complex"}:
        space = gmes.Cartesian(size=(10, 10, 0), resolution=20)
        geometry = [gmes.DefaultMedium(gmes.Dielectric())]
    elif case == "default-3d":
        space = gmes.Cartesian(size=(5, 5, 5), resolution=10)
        geometry = [gmes.DefaultMedium(gmes.Dielectric())]
    elif case == "heterogeneous":
        space = gmes.Cartesian(size=(10, 10, 0), resolution=20)
        geometry = [gmes.DefaultMedium(gmes.Dielectric())]
        geometry.extend(
            gmes.Cylinder(gmes.Dielectric(4), center=(x, y, 0), radius=0.6, height=1)
            for x in (-3, -1, 1, 3)
            for y in (-3, -1, 1, 3)
        )
    elif case == "many-small":
        space = gmes.Cartesian(size=(8, 8, 0), resolution=20)
        geometry = [gmes.DefaultMedium(gmes.Dielectric())]
        geometry.extend(
            gmes.Sphere(gmes.Dielectric(2), center=(x, y, 0), radius=0.12)
            for x in np.linspace(-3.5, 3.5, 20)
            for y in np.linspace(-3.5, 3.5, 20)
        )
    elif case == "overlap":
        space = gmes.Cartesian(size=(6, 6, 6), resolution=12)
        geometry = [gmes.DefaultMedium(gmes.Dielectric())]
        geometry.extend(
            gmes.Sphere(gmes.Dielectric(index + 2), radius=2.5 - 0.08 * index)
            for index in range(24)
        )
    elif case == "primitives":
        space = gmes.Cartesian(size=(14, 4, 4), resolution=12)
        geometry = [
            gmes.DefaultMedium(gmes.Dielectric()),
            gmes.Cone(
                gmes.Dielectric(2),
                center=(-5, 0, 0),
                axis=(1, 1, 1),
                radius=0.9,
                radius2=0.2,
                height=2,
            ),
            gmes.Cylinder(
                gmes.Dielectric(3),
                center=(-2.5, 0, 0),
                axis=(1, -1, 1),
                radius=0.9,
                height=2,
            ),
            gmes.Block(
                gmes.Dielectric(4),
                e1=(1, 0, 0),
                e2=(1, 1, 0),
                e3=(0, 0, 1),
                size=(2, 2, 2),
            ),
            gmes.Ellipsoid(
                gmes.Dielectric(5),
                center=(2.5, 0, 0),
                e1=(1, 0, 0),
                e2=(1, 1, 0),
                e3=(0, 0, 1),
                size=(2, 2, 2),
            ),
            gmes.Sphere(gmes.Dielectric(6), center=(5, 0, 0), radius=1),
            gmes.Shell(
                gmes.Cpml(), center=(0.5, 0, 0), size=(13, 3.5, 3.5), thickness=0.4
            ),
        ]
    elif case == "collapsed-1d":
        space = gmes.Cartesian(size=(20, 0, 0), resolution=30)
        geometry = [
            gmes.DefaultMedium(gmes.Dielectric()),
            gmes.Block(gmes.Dielectric(3), size=(5, 1, 1)),
        ]
    elif case == "custom-fallback":

        class CustomSphere(gmes.Sphere):
            pass

        space = gmes.Cartesian(size=(8, 8, 0), resolution=20)
        geometry = [
            gmes.DefaultMedium(gmes.Dielectric()),
            CustomSphere(gmes.Dielectric(3), radius=2),
        ]
    else:
        raise ValueError(f"unknown case: {case}")
    return space, geometry, complex_mode


def build_workload(case, gmes):
    """Initialize geometry and return all six component coordinate grids."""
    space, geometry, complex_mode = geometry_for_case(case, gmes)
    simulation = gmes.FDTD(
        space,
        geometry,
        verbose=False,
        **({"bloch": (0.1, 0.2, 0)} if complex_mode else {}),
    )
    fields = {
        gmes.Ex: space.get_ex_storage(simulation.e_field_compnt, complex_mode),
        gmes.Ey: space.get_ey_storage(simulation.e_field_compnt, complex_mode),
        gmes.Ez: space.get_ez_storage(simulation.e_field_compnt, complex_mode),
        gmes.Hx: space.get_hx_storage(simulation.h_field_compnt, complex_mode),
        gmes.Hy: space.get_hy_storage(simulation.h_field_compnt, complex_mode),
        gmes.Hz: space.get_hz_storage(simulation.h_field_compnt, complex_mode),
    }
    grids = tuple(
        (
            component,
            field.shape,
            space.component_coordinate_axes(component, field.shape),
        )
        for component, field in fields.items()
    )
    return simulation.geom_tree, grids


def lower_tile(tree, axes, start, stop):
    """Return reference-compatible material and underlying integer IDs."""
    if hasattr(tree, "lower_grid"):
        geometry_map = tree.lower_grid(*axes, start, stop)
        return geometry_map.material_ids, geometry_map.underlying_ids

    materials, underlying = tree.material_of_grid(*axes, start, stop)
    geometries = tree.root.geom_list
    material_ids = {
        id(geometry.material): index for index, geometry in enumerate(geometries)
    }
    return (
        np.fromiter(
            (material_ids[id(material)] for material in materials),
            dtype=np.int32,
            count=len(materials),
        ),
        np.fromiter(
            (
                -1 if material is None else material_ids[id(material)]
                for material in underlying
            ),
            dtype=np.int32,
            count=len(underlying),
        ),
    )


def map_workload(tree, grids, tile_size):
    """Lower every component and return a complete-map digest."""
    material_digest = hashlib.sha256()
    underlying_digest = hashlib.sha256()
    cells = 0
    for component, shape, axes in grids:
        material_digest.update(component.__name__.encode())
        underlying_digest.update(component.__name__.encode())
        total = int(np.prod(shape))
        cells += total
        for start in range(0, total, tile_size):
            material_ids, underlying_ids = lower_tile(
                tree, axes, start, min(start + tile_size, total)
            )
            material_digest.update(material_ids.tobytes())
            underlying_digest.update(underlying_ids.tobytes())
    digest = hashlib.sha256(material_digest.digest() + underlying_digest.digest())
    return digest.hexdigest(), cells


def peak_rss_bytes():
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(maximum if platform.system() == "Darwin" else maximum * 1024)


def run_case(case, repeats, warmup, tile_size, gmes):
    tree, grids = build_workload(case, gmes)
    for _ in range(warmup):
        map_workload(tree, grids, tile_size)

    samples = []
    digest = None
    cells = None
    for _ in range(repeats):
        start = perf_counter()
        digest, cells = map_workload(tree, grids, tile_size)
        samples.append(perf_counter() - start)
    return {
        "case": case,
        "cells": cells,
        "map_sha256": digest,
        "median_seconds": median(samples),
        "seconds": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="all")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--tile-size", type=int, default=65536)
    parser.add_argument(
        "--reference-commit", default="f3f062761fdbec8906b691806cb8e2c29bcb2bdd"
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0 or args.tile_size < 1:
        parser.error(
            "repeats and tile-size must be positive; warmup must be nonnegative"
        )

    import gmes

    all_cases = (
        "default-2d",
        "default-3d",
        "heterogeneous",
        "many-small",
        "overlap",
        "primitives",
        "collapsed-1d",
        "complex",
        "custom-fallback",
    )
    cases = all_cases if args.case == "all" else (args.case,)
    results = [
        run_case(case, args.repeats, args.warmup, args.tile_size, gmes)
        for case in cases
    ]
    gating = [
        item["median_seconds"] for item in results if item["case"] != "custom-fallback"
    ]
    print(
        json.dumps(
            {
                "candidate_has_region_lowering": hasattr(
                    gmes.pygeom.GeomBoxTree, "lower_grid"
                ),
                "geometric_mean_seconds": geometric_mean(gating),
                "numpy": np.__version__,
                "peak_rss_bytes": peak_rss_bytes(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "reference_commit": args.reference_commit,
                "repeats": args.repeats,
                "tile_size": args.tile_size,
                "warmup": args.warmup,
                "cases": results,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
