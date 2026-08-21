#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-or-later

##    gmes - GIST Maxwell's Equations Solver
##    Copyright (C) 2007-2012  Kyungwon Chun
##
##    This program is free software; you can redistribute it and/or
##    modify it under the terms of the GNU General Public
##    License as published by the Free Software Foundation; either
##    version 3 of the License, or (at your option) any later version.
##
##    This program is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
##    General Public License for more details.
##
##    You should have received a copy of the GNU General Public
##    License along with this program. If not, see
##    <https://www.gnu.org/licenses/>.
##
##    Kyungwon Chun
##    kwchun@gist.ac.kr

"""A Python implementation of the explicit FDTD method

GMES is a Python implementation of the explicit finite-difference
time-domain (FDTD) method. It is designed to simulate the photonic
device in 1, 2, and 3-d Cartesian coordinates.

Modules:
    fdtd --- Provide various simulation classes suitable for 1, 2, and 3-d.
    geometry --- Provide coordinate and geometric primitives.
    show --- Real-time display classes
    constant --- Physical and simulation constants
    source --- Define the input sources
    pw_source --- Source update mechanism
    material --- Define the propagating medium
    pw_material --- Provide the update mechanism

"""

from .fdtd import *
from .geometry import *
from .constant import *
from .source import *
from .material import *

from . import constant, fdtd, geometry, material, pw_material, pw_source, source

# List here only the objects we want to be publicly available
_module = ['fdtd', 'geometry', 'constant', 'source', 'pw_source', 'material', 'pw_material']
_class = ['TimeStep', 'FDTD', 'TExFDTD', 'TEyFDTD', 'TEzFDTD', 'TMxFDTD', 'TMyFDTD', 'TMzFDTD', 'TEMxFDTD', 'TEMyFDTD', 'TEMzFDTD',
          'Cartesian', 'DefaultMedium', 'Cone', 'Cylinder', 'Block', 'Ellipsoid', 'Sphere', 'Shell',
          'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz', 'Jx', 'Jy', 'Jz', 'Mx', 'My', 'Mz', 'X', 'Y', 'Z', 'PlusX', 'MinusX', 'PlusY', 'MinusY', 'PlusZ', 'MinusZ',
          'Continuous', 'Bandpass', 'DifferentiatedGaussian', 'PointSource', 'TotalFieldScatteredField', 'GaussianBeam',
          'Dummy', 'Const', 'Dielectric', 'Upml', 'Cpml', 'DrudePole', 'LorentzPole', 'CriticalPoint', 'DcpAde', 'DcpPlrc', 'DcpRc', 'Drude', 'Lorentz', 'Dm2']
_constant = ['pi', 'c0', 'mu0', 'eps0', 'Z0', 'PETA', 'TERA', 'GIGA', 'MEGA', 'KILO', 'MILLI', 'MICRO', 'NANO', 'PICO', 'FEMTO', 'ATTO',
             'inf']
__all__ = []
__all__.extend(_module)
__all__.extend(_class)
__all__.extend(_constant)
