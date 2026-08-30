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

"""Simulate electromagnetic fields with legacy native and PyTorch FDTD backends.

GMES implements the explicit finite-difference time-domain (FDTD) method
for one-, two-, and three-dimensional Cartesian photonic simulations. The
legacy API maps NumPy field arrays to native point-wise update kernels. The
PyTorch API lowers the same geometry and material models into reusable tensor
plans for CPU, CUDA, and optional two-rank distributed execution.

Modules:
    fdtd: Legacy FDTD simulation classes backed by native update kernels.
    geometry: Cartesian grids and geometric primitives.
    constant: Physical constants and field, axis, and direction identifiers.
    source: Temporal waveforms and spatial source definitions.
    material: Nondispersive, absorbing, and dispersive materials.
    torch_fdtd: Planned PyTorch simulations and runtime configuration.
    torch_distributed: Two-rank CUDA domain decomposition and halo exchange.
    torch_output: Device probe buffers, spectra, and checkpoint persistence.

Examples:
    Run one source-free legacy step in a two-dimensional TMz domain:

    >>> space = Cartesian(size=(2, 2, 0), resolution=2)
    >>> geometry = [DefaultMedium(material=Dielectric())]
    >>> simulation = TMzFDTD(space, geometry, verbose=False)
    >>> simulation.init()
    >>> simulation.step()
"""

from . import constant, fdtd, geometry, material, pw_material, pw_source, source
from .constant import *
from .fdtd import *
from .geometry import *
from .material import *
from .source import *
from .torch_distributed import *
from .torch_fdtd import *

# List here only the objects we want to be publicly available
_module = [
    "fdtd",
    "geometry",
    "constant",
    "source",
    "pw_source",
    "material",
    "pw_material",
]
_class = [
    "TimeStep",
    "FDTD",
    "TExFDTD",
    "TEyFDTD",
    "TEzFDTD",
    "TMxFDTD",
    "TMyFDTD",
    "TMzFDTD",
    "TEMxFDTD",
    "TEMyFDTD",
    "TEMzFDTD",
    "Cartesian",
    "DefaultMedium",
    "Cone",
    "Cylinder",
    "Block",
    "Ellipsoid",
    "Sphere",
    "Shell",
    "Ex",
    "Ey",
    "Ez",
    "Hx",
    "Hy",
    "Hz",
    "Jx",
    "Jy",
    "Jz",
    "Mx",
    "My",
    "Mz",
    "X",
    "Y",
    "Z",
    "PlusX",
    "MinusX",
    "PlusY",
    "MinusY",
    "PlusZ",
    "MinusZ",
    "Continuous",
    "Bandpass",
    "DifferentiatedGaussian",
    "PointSource",
    "TotalFieldScatteredField",
    "GaussianBeam",
    "Dummy",
    "Const",
    "Dielectric",
    "Upml",
    "Cpml",
    "DrudePole",
    "LorentzPole",
    "CriticalPoint",
    "DcpAde",
    "DcpPlrc",
    "DcpRc",
    "Drude",
    "Lorentz",
    "Dm2",
    "ComponentPlan",
    "DistributedLaunch",
    "ExecutionSignature",
    "FlattenedStencilTerm",
    "MaterialBucketPlan",
    "TorchConfigurationError",
    "TorchDistributedError",
    "TorchDistributedSimulation",
    "TorchHaloExchange",
    "TorchExecutionPlanner",
    "TorchPointSourceRecord",
    "TorchProbeSamples",
    "TorchProbeSpec",
    "TorchProbeSpectrum",
    "TorchRuntimeConfig",
    "TorchSimulation",
    "TorchSimulationPlan",
    "TorchSimulationState",
    "TorchSourceLoweringContext",
    "TwoGpuDecomposition",
]
_constant = [
    "pi",
    "c0",
    "mu0",
    "eps0",
    "Z0",
    "PETA",
    "TERA",
    "GIGA",
    "MEGA",
    "KILO",
    "MILLI",
    "MICRO",
    "NANO",
    "PICO",
    "FEMTO",
    "ATTO",
    "inf",
]
_function = [
    "choose_two_gpu_decomposition",
    "distributed_launch_from_environment",
    "probe_spectrum",
    "rank_local_space",
    "read_torch_checkpoint",
    "torch_runtime_diagnostics",
    "write_probe_text",
    "write_torch_checkpoint",
]
__all__ = []
__all__.extend(_module)
__all__.extend(_class)
__all__.extend(_constant)
__all__.extend(_function)
