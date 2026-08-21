# GMES

GMES (GIST Maxwell's Equations Solver) is a free electromagnetic simulator that solves Maxwell's equations with the explicit finite-difference time-domain (FDTD) method. It provides a Python interface backed by C++, SWIG, and Cython extensions for modeling photonic devices in one-, two-, and three-dimensional Cartesian domains.

> [!IMPORTANT]
> The current development line targets Python 3.14, C++17, NumPy 2, Cython 3, and SWIG 4. Python 2 and the former Distutils build are no longer supported.

## Features

- 1D, 2D, and 3D Cartesian FDTD simulations
- TE, TM, and TEM simulation classes
- Dielectric, Drude, Lorentz, critical-point, and related dispersive material models
- UPML and CPML absorbing boundary layers
- Point, continuous-wave, Gaussian, bandpass, and total-field/scattered-field sources
- Geometric primitives including blocks, spheres, cylinders, cones, ellipsoids, and shells
- Bloch-periodic simulations with complex-valued fields
- Optional MPI-based parallel execution
- Field visualization and HDF5 output utilities

## Requirements

- Python 3.14 or newer
- A C++17 compiler
- SWIG 4
- NumPy 2.3 or newer
- SciPy 1.16 or newer

Matplotlib, mpi4py, and PyTables are available through the `plot`, `mpi`, and `hdf5` optional dependency groups.

## Installation

Install SWIG with the package manager for your operating system, then create an isolated Python environment:

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

For development, install the package in editable mode with the development and optional runtime dependencies needed for the tests:

```sh
python -m pip install -e ".[dev,hdf5]"
```

Other supported combinations include `.[plot]`, `.[mpi]`, and `.[all]`. The build uses the PEP 517 configuration in `pyproject.toml`; invoking `setup.py` directly is not supported.

## Quick start

The following example creates a two-dimensional TMz domain in air, surrounds it with a CPML absorbing boundary, and excites an `Ez` point source:

```python
from gmes import Cartesian, Continuous, Cpml, DefaultMedium, Dielectric
from gmes import Ez, PointSource, Shell, TMzFDTD

space = Cartesian(size=(10, 10, 0), resolution=20)
geometry = [
    DefaultMedium(material=Dielectric()),
    Shell(material=Cpml()),
]
sources = [
    PointSource(
        src_time=Continuous(freq=0.8),
        center=(0, 0, 0),
        component=Ez,
    ),
]

simulation = TMzFDTD(space, geometry, sources)
simulation.init()
simulation.step_until_t(10)
simulation.write_field(Ez, (-5, -5, 0), (5, 5, 0))
```

After installing GMES, run examples from the repository root:

```sh
python examples/air2d.py
```

See [`examples/`](examples/) for simulations of wave propagation, Fresnel reflection, photonic-crystal waveguides, slab waveguides, plasmonic arrays, and total-field/scattered-field excitation. Some three-dimensional examples require more than 1 GB of memory and are not suitable as routine smoke tests.

## Testing and packaging

Run the complete test suite and build both distribution formats with:

```sh
python -m unittest discover -v
python -m build
```

The tests include component coverage, geometry and source-time checks, a deterministic FDTD regression, and optional HDF5 output coverage. The HDF5 tests are skipped when PyTables is not installed.

## Parallel execution

Install GMES with its MPI dependency and use the launcher supplied by your MPI implementation:

```sh
python -m pip install ".[mpi]"
mpiexec -n <process-count> python <simulation.py>
```

## Repository layout

```text
gmes/       Python package and public simulation API
src/        C++, SWIG, and Cython extension sources
examples/   Example electromagnetic simulations
tests/      Unit and numerical regression tests
utils/      Data-processing and diagnostic utilities
docs/       Maintenance and migration notes
```

## Known limitations

- Do not use `numpy.inf` for simulation bounds; use a sufficiently large finite value instead. GMES does not consistently treat `numpy.inf` as infinity.
- Some large examples retain their historical problem sizes and can consume substantial memory and execution time.
- Linux and macOS are exercised by CI; other platforms may require build-system adjustments.

## Contributing and support

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the development workflow. Bug reports and patches are welcome through the [GitHub issue tracker](https://github.com/ruddyscent/gmes/issues). Historical releases and discussions remain available on the [GMES SourceForge project](https://sourceforge.net/projects/gmes/).

## License

GMES is distributed under the GNU General Public License version 3 or later (`GPL-3.0-or-later`). See [`LICENSE`](LICENSE) for the full license text.

Copyright (C) 2007-2012 Kyungwon Chun.
