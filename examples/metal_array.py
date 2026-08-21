#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""A plasmon waveguide consisting of six silver nanospheres in the
air.

This script models a plasmon waveguide consisting of six silver
nanospheres in the air. A dipole source oscilating along the array
is used for field excitation. A simple on-time visualization
display will show the Ey fields of the propagating longitudinal
mode. Run this script through ``mpiexec`` after installing GMES with
the ``mpi`` optional dependency to use multiple processes. This script
requires about 1.1 GB of memory.

"""

import argparse

from gmes import *

FULL_RESOLUTION = 40
FULL_UNTIL = 50
QUICK_RESOLUTION = 8
QUICK_UNTIL = 1


class Silver(DcpPlrc):
    """This silver permittivity value in the range of 200-1,000 nm.

    This parameters has a fitness value of 0.0266134 to the real
    permittivity data in the range of 200-1,000 nm of

    P. B. Johnson and R. W. Christy, "Optical constants of the
    noble metals,"  Phys. Rev. B 6, 4370 (1972).

    """

    def __init__(self, a):
        """
        a: lattice constant in meters.

        """
        dp1 = DrudePole(omega=1.38737e16 * a / c0, gamma=2.07331e13 * a / c0)
        cp1 = CriticalPoint(
            amp=1.3735,
            phi=-0.504658,
            omega=7.59914e15 * a / c0,
            gamma=4.28431e15 * a / c0,
        )
        cp2 = CriticalPoint(
            amp=0.304478,
            phi=-1.48944,
            omega=6.15009e15 * a / c0,
            gamma=6.59262e14 * a / c0,
        )
        DcpPlrc.__init__(
            self, eps_inf=0.89583, mu_inf=1, sigma=0, dps=(dp1,), cps=(cp1, cp2)
        )


def run(resolution=FULL_RESOLUTION, until=FULL_UNTIL):
    """Run the six-particle plasmon-waveguide simulation."""
    space = Cartesian(size=(2, 8, 2), resolution=resolution, parallel=True)
    geom_list = [DefaultMedium(Dielectric())]
    for y in range(-2, 4):
        geom_list.append(Sphere(Silver(75 * NANO), radius=1.0 / 3, center=(0, y, 0)))
    geom_list.append(Shell(Cpml(), thickness=0.5))
    src_list = [PointSource(Continuous(freq=0.207), center=(0, -3, 0), component=Jy)]
    simulation = FDTD(space, geom_list, src_list, courant_ratio=0.5)
    simulation.init()
    simulation.show_field(Ey, Z, 0, (-1e-5, 1e-5))
    simulation.step_until_t(until)
    return simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use reduced resolution and duration for a fast smoke test",
    )
    args = parser.parse_args()
    if args.quick:
        run(QUICK_RESOLUTION, QUICK_UNTIL)
    else:
        run()


if __name__ == "__main__":
    main()
