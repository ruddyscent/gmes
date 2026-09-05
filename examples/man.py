"""Construct the man-shaped geometry and inspect its Torch plan."""

import argparse

from numpy import cross

import gmes


def run(resolution=20):
    geometry = [
        gmes.DefaultMedium(gmes.Dielectric(10)),
        gmes.Block(gmes.Dielectric(1), size=(1, 1, 2)),
        gmes.Sphere(gmes.Dielectric(2), center=(0, 0, -1.5), radius=0.5),
        gmes.Cone(
            gmes.Dielectric(3),
            center=(0, 0.2, -2.15),
            axis=(0, 0.2, -1),
            radius=0.7,
            height=0.5,
        ),
        gmes.Cylinder(
            gmes.Dielectric(4),
            center=(0, 0.5, 2),
            axis=(0, -0.2, -1),
            radius=0.2,
            height=2,
        ),
        gmes.Cylinder(
            gmes.Dielectric(5),
            center=(0, -0.5, 2),
            axis=(0, 0.2, -1),
            radius=0.2,
            height=2,
        ),
        gmes.Ellipsoid(
            gmes.Dielectric(6),
            center=(0, 1.3, -0.2),
            e1=(1, 0, 0),
            e2=(0, 1, -1),
            e3=cross((1, 0, 0), (0, 1, -1)),
            size=(0.5, 0.5, 1.5),
        ),
        gmes.Ellipsoid(
            gmes.Dielectric(7),
            center=(0, -1.3, -0.2),
            e1=(1, 0, 0),
            e2=(0, 1, 1),
            e3=cross((1, 0, 0), (0, 1, 1)),
            size=(0.5, 0.5, 1.5),
        ),
    ]
    return gmes.TorchSimulation(
        space=gmes.Cartesian((6, 6, 6), resolution),
        geometry=geometry,
        runtime=gmes.TorchRuntimeConfig(device="cpu", cpu_threads=1),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    run(4 if parser.parse_args().quick else 20)
