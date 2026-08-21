#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Models two-dimensional TMz cylindrical-wave propagation in air.

This script models two-dimensional TMz cylindrical-wave
propagation in air. A single Ez component located at the center of
the space oscillates sinusoidally. A simple on-time visualization
display will show the Ez, Hx, and Hy fields of the outgoing wave
distributed within the grid. You can compare the spatial-symmetry
properties of these fields with respect to the center of the space
where the excitation is applied.

"""

from gmes import *

SIZE = (10, 10, 0)
DISPLAY_COMPONENTS = (Ez, Hx, Hy)


def make_simulation(verbose=True):
    space = Cartesian(size=SIZE, resolution=20)
    geom_list = [DefaultMedium(material=Dielectric()), Shell(material=Cpml())]
    src_list = [
        PointSource(src_time=Continuous(freq=0.8), center=(0, 0, 0), component=Ez)
    ]
    return TMzFDTD(space, geom_list, src_list, verbose=verbose)


def main():
    simulation = make_simulation()
    simulation.init()
    for component in DISPLAY_COMPONENTS:
        simulation.show_field(component, Z, 0)
    simulation.step_until_t(10)
    simulation.write_field(Ez, (-5, -5, 0), (5, 5, 0))


if __name__ == "__main__":
    main()
