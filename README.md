# GMES

GMES (GIST Maxwell's Equations Solver) is a free electromagnetic simulator that solves Maxwell's equations with the explicit finite-difference time-domain (FDTD) method. It provides a Python interface backed by C++ and SWIG extensions for modeling photonic devices in one-, two-, and three-dimensional Cartesian domains.

> [!IMPORTANT]
> The current development line targets Python 3.14, C++23, NumPy 2, and SWIG 4. Python 2 and the former Distutils build are no longer supported.

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

- Python 3.14 or newer (the tested 0.10.0 release target is Python 3.14)
- A C++23 compiler and standard library
- SWIG 4
- NumPy 2.3 or newer
- SciPy 1.16 or newer
- PyTorch 2.13 (CPU, CUDA 12.6, or CUDA 13.0 wheel)

GMES 0.10.0 publishes binary wheels for the following combinations:

| Python | Operating system | Architecture | Minimum platform |
| --- | --- | --- | --- |
| CPython 3.14 | Linux | x86_64 | glibc 2.34 (`manylinux_2_34`) |
| CPython 3.14 | macOS | arm64 (Apple silicon) | macOS 11 |

Source installations are supported on current Linux x86_64 and macOS arm64
systems with the native toolchain documented below. Windows and macOS x86_64
are not supported by the 0.10.0 release because they do not have tested wheel
builds. Python versions newer than 3.14 may satisfy the package metadata but
are not part of the 0.10.0 tested release matrix.

Matplotlib, mpi4py, and PyTables are available through the `plot`, `mpi`, and `hdf5` optional dependency groups.

### System prerequisites

On Ubuntu 24.04 or newer, install the compiler toolchain and SWIG with:

```sh
sudo apt-get update
sudo apt-get install --yes build-essential swig
c++ --version
swig -version
```

On macOS, install the current Xcode Command Line Tools and SWIG. Install
Homebrew `libomp` as well to enable native OpenMP field updates:

```sh
xcode-select --install
brew install swig libomp
c++ --version
swig -version
```

The native extensions are always compiled in C++23 mode. They use
`std::mdspan` when the standard library provides `<mdspan>` and otherwise use
the internal contiguous-indexing fallback. That fallback does not add support
for older C++ language modes.

OpenMP support is detected while building the native material extension. The
default `GMES_ENABLE_OPENMP=auto` mode uses OpenMP when a compile-and-link
probe succeeds and otherwise builds the serial fallback. On macOS, auto mode
also rejects a `libomp` whose minimum deployment target is newer than the
extension target. Set the variable to `0` to require a serial build or to `1`
to require OpenMP and fail the build when the toolchain or runtime is
unavailable or incompatible. `GMES_OPENMP_PREFIX` can point to a nonstandard
`libomp` installation.

## Installation

On a supported wheel platform, create an isolated Python environment and
install the release from PyPI; a compiler and SWIG are not needed for this
path:

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "gmes==0.10.0"
```

For a source checkout or source distribution, install SWIG and the compiler
toolchain first, then install the local project:

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

For development, install [uv](https://docs.astral.sh/uv/), then create the
locked Python 3.14 environment with the optional runtime dependencies needed
by the tests:

```sh
uv python install 3.14
uv sync --locked --extra torch-cpu --extra hdf5
uv run --no-sync python -m unittest discover -v
uv build
```

This checkout requires uv 0.12.5; uv exits with an actionable version error
when a different release is used.

The `dev` dependency group is installed by default. Select exactly one of `torch-cpu`, `torch-cu126`, or `torch-cu130` for an explicit development runtime; see [`docs/torch-execution.md`](docs/torch-execution.md) for the breaking Torch-only API and wheel-selection contract.

Other supported runtime combinations include `--extra plot`, `--extra mpi`, and `--extra all`. The
build uses the PEP 517 configuration in `pyproject.toml`; invoking `setup.py`
directly is not supported.

If an existing checkout used `python -m pip install -e ".[dev,hdf5]"`, switch
to the commands above. `dev` is now a PEP 735 dependency group rather than a
package extra, and `uv sync` installs it by default. Extras such as `hdf5`,
`plot`, and `mpi` remain explicit `--extra` options. Use `uv sync --locked`
when consuming the committed lockfile; reserve `uv lock --upgrade` for a
deliberate dependency-update change.

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
```

The quick-start code above uses only the base dependencies. To run the
visualizing `air2d.py` example, install its plotting and HDF5 dependencies and
then launch it from the repository root:

```sh
uv sync --locked --extra plot --extra hdf5
uv run --no-sync python examples/air2d.py
```

See [`examples/`](examples/) for simulations of wave propagation, Fresnel reflection, photonic-crystal waveguides, slab waveguides, plasmonic arrays, and total-field/scattered-field excitation. Some three-dimensional examples require more than 1 GB of memory and are not suitable as routine smoke tests.

## Testing and packaging

Run the complete test suite and build both distribution formats with:

```sh
uv run --no-sync python -m mypy
uv run --no-sync python -m mypy.stubtest gmes.constant gmes.pw_material
uv run --no-sync python -m unittest discover -v
uv build
```

The tests include component coverage, geometry and source-time checks, a deterministic FDTD regression, and optional HDF5 output coverage. The HDF5 tests are skipped when PyTables is not installed.

macOS wheels target macOS 11 by default. Set `MACOSX_DEPLOYMENT_TARGET`
explicitly before building only when a wheel intentionally requires a newer
macOS release; the build verifies both the wheel platform tag and every native
extension's minimum OS load command.

Release artifacts are built only by the tag-triggered GitHub Actions release
workflow. Maintainers must not upload files from a local `dist/` directory.
See [`docs/releasing.md`](docs/releasing.md) for the release checklist.

## Parallel execution

When OpenMP is present, native material-update loops with at least 8,192 cells
run in parallel. Select the thread count before starting Python:

```sh
OMP_NUM_THREADS=4 uv run --no-sync python examples/air3d.py
```

Use `GMES_OPENMP_THRESHOLD` to tune the cutoff at runtime; `0` forces every
eligible loop through OpenMP. The following functions report the active build
and runtime configuration:

```python
from gmes import pw_material

pw_material.openmp_enabled()
pw_material.openmp_max_threads()
pw_material.openmp_cell_threshold()
```

Build-time controls require reinstalling the native extension. For example,
the locked development environment can switch to its serial fallback with:

```sh
GMES_ENABLE_OPENMP=0 uv sync --locked --extra hdf5 --reinstall-package gmes
```

See [`benchmarks/`](benchmarks/) for repeatable performance measurements and
the threshold-selection record.

### MPI

Install an MPI implementation (`libopenmpi-dev openmpi-bin` on Ubuntu or
`open-mpi` with Homebrew on macOS), then install the Python extra and use its
launcher through the uv environment:

```sh
# Ubuntu
sudo apt-get install --yes libopenmpi-dev openmpi-bin

# macOS
brew install open-mpi
```

```sh
uv sync --locked --extra mpi
uv run --no-sync mpiexec -n <process-count> python <simulation.py>
```

When combining MPI and OpenMP, set `OMP_NUM_THREADS` explicitly and keep the
process count times the thread count within the available physical cores to
avoid oversubscription.

## Repository layout

```text
gmes/       Python package and public simulation API
src/        C++ and SWIG extension sources
examples/   Example electromagnetic simulations
tests/      Unit and numerical regression tests
benchmarks/ Repeatable field-update performance measurements
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
