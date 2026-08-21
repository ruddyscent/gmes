#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shows an Ez field in a dielectric slab waveguide.

A simple example showing the Ez field in a dielectric slab
waveguide. This is a GMES version of the script in Fig.12 of

A. F. Oskooi, D. Roundy, M. Ibanescu, P. Bermel, J. D.
Joannopoulos, and S. G. Johnson, "Meep: A flexible free-software
package for electromagnetic simulations by the FDTD method,"
Comput. Phys. Commun. 181, 687-702 (2010).

"""

from gmes import *


def make_simulation(verbose=True):
    space = Cartesian(size=(16, 8, 0), resolution=10)
    geom_list = [
        DefaultMedium(material=Dielectric()),
        Block(material=Dielectric(12), size=(16, 1, 1)),
        Shell(material=Cpml()),
    ]
    src_list = [
        PointSource(src_time=Continuous(freq=0.15), component=Ez, center=(-7, 0, 0))
    ]
    return TMzFDTD(space, geom_list, src_list, verbose=verbose)


def main():
    simulation = make_simulation()
    simulation.init()
    simulation.show_field(Ez, Z, 0)
    simulation.step_until_t(200)


if __name__ == "__main__":
    main()
