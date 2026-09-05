#!/usr/bin/env python3

"""Simulate a photonic crystal slab waveguide.

The photonic crystal slab consists of a silicon-on-insulator
substrate with a triangular array of holes. The whole waveguide is
in the air. This structure is presented at

'N. Moll and G.-L. Bona, "Comparison of three-dimensional photonic
crystal slab waveguides with two-dimensional photonic crystal
waveguides: Efficient butt coupling into these photonic crystal
waveguides," J. Appl. Phys., vol. 93, no. 9, pp. 4986-4991, 2003.'

This script requires about 1.3GB of memory.

"""

import argparse

from numpy import *

from gmes import *

# Define simulation parameters.

FREQ = 0.3
RADIUS = 0.35
SLAB_CORE = 0.5 / sqrt(11.8336)
SLAB_THICK = 3 * SLAB_CORE
PML_THICK = 0.5
ZLENGTH = 3 * SLAB_THICK + 2 * PML_THICK
SIZE = (15, 15, ZLENGTH)
RESOLUTION = (25, 25, 10)
UNTIL = 50
QUICK_SIZE = (5, 5, ZLENGTH)
QUICK_RESOLUTION = (5, 5, 4)
QUICK_UNTIL = 1

AIR = Dielectric(1)
SiO2 = Dielectric(2.1316)
Si = Dielectric(11.8336)


def make_hole(center):
    """Punch a air hole on the slab."""
    return Cylinder(
        material=AIR, center=center, axis=(0, 0, 1), radius=RADIUS, height=SLAB_THICK
    )


def make_crystals(x_size, y_size):
    """Punch holes on the lattice positions."""
    a1 = array((cos(pi / 3), sin(pi / 3)))
    a2 = array((cos(pi / 3), -sin(pi / 3)))
    crystals = []

    for i in arange(-ceil(x_size), ceil(x_size)):
        for j in arange(-ceil(y_size), ceil(y_size)):
            center = tuple(i * a1 + j * a2) + (0,)
            if (
                fabs(center[0]) <= 0.5 * x_size + RADIUS
                and fabs(center[1]) <= 0.5 * y_size + RADIUS
            ):
                crystals.append(make_hole(center))
    return crystals


def fill_hole(center):
    """Remove a air hole."""
    filler = [
        Cylinder(
            material=SiO2,
            axis=(0, 0, 1),
            radius=RADIUS,
            height=SLAB_THICK,
            center=center,
        ),
        Cylinder(
            material=Si, axis=(0, 0, 1), radius=RADIUS, height=SLAB_CORE, center=center
        ),
    ]

    return filler


def make_line_defect(length):
    """Remove air holes to form a line defect."""
    line_defect = []

    for i in range(int(-length / 2), int(length / 2 + 1)):
        line_defect += fill_hole((i, 0, 0))

    return line_defect


def run(size=SIZE, resolution=RESOLUTION, until=UNTIL):
    """Run the three-dimensional photonic-crystal slab simulation."""
    geom_list = (
        [
            DefaultMedium(material=AIR),
            Block(material=SiO2, size=(size[0], size[1], SLAB_THICK)),
            Block(material=Si, size=(size[0], size[1], SLAB_CORE)),
        ]
        + make_crystals(*size[:2])
        + make_line_defect(size[0])
        + [Shell(material=Cpml(), thickness=PML_THICK)]
    )
    space = Cartesian(size=size, resolution=resolution)
    src_list = [
        PointSource(
            src_time=Continuous(freq=FREQ),
            center=(-size[0] / 2 + 1, 0, 0),
            component=Hz,
        )
    ]

    simulation = TorchSimulation(
        space=space,
        geometry=geom_list,
        sources=src_list,
        runtime=TorchRuntimeConfig(device="cpu", cpu_threads=1),
    )
    simulation.advance(max(1, round(until / simulation.plan.dt)))
    return simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use a smaller lattice, lower resolution, and shorter duration",
    )
    args = parser.parse_args()
    if args.quick:
        run(QUICK_SIZE, QUICK_RESOLUTION, QUICK_UNTIL)
    else:
        run()


if __name__ == "__main__":
    main()
