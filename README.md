# GMES

GMES (GIST Maxwell's Equations Solver) is a pure-Python, PyTorch-based
electromagnetic simulator for explicit finite-difference time-domain (FDTD)
work in one-, two-, and three-dimensional Cartesian domains. The Torch runtime
is the supported solver API; the former C++/SWIG solver classes, OpenMP
controls, and MPI launcher are retired.

## Features

- 1D, 2D, and 3D Cartesian FDTD simulations with dielectric and dispersive descriptors
- CPML absorbing layers, point/TFSF sources, Bloch fields, probes, and checkpoints
- Torch CPU execution and supported single-/two-GPU execution

## Requirements and installation

- Python 3.14 or newer
- NumPy 2.3 or newer, SciPy 1.16 or newer, and PyTorch `>=2.13,<2.14`

GMES itself is a universal pure-Python package: installation and source builds
do not require a C/C++ compiler, SWIG, Cython, OpenMP, or system headers.
`plot` and `hdf5` remain optional extras for external plotting and generic HDF5
use.

Use one explicit Torch runtime extra in a checkout; the extras select the
locked CPU, CUDA 12.6, or CUDA 13.0 PyTorch index and must not be combined:

```sh
uv python install 3.14
uv sync --locked --extra torch-cpu --extra hdf5
uv run --no-sync python -m unittest discover -v
uv build
```

The `dev` dependency group is installed by default and is a PEP 735 group, not
a package extra. Consume the committed lock with `uv sync --locked`; use
`uv lock --upgrade` only in a dedicated dependency update. Select
`torch-cu126` or `torch-cu130` instead of `torch-cpu` only on a Linux CUDA
target. The package has no CUDA suffix and bundles no CUDA or native GMES
library.

This checkout requires uv 0.12.5. The current verified platform scope remains
Linux x86_64 and macOS arm64; a universal wheel does not broaden that evidence.

The staged release contract is one `py3-none-any` wheel and one sdist. That
artifact shape does not claim that every operating system or accelerator gate
has passed: Linux/macOS CPU and trusted CUDA/two-GPU installed-artifact
validation remain release gates.

## Quick start

```python
from gmes import Cartesian, Continuous, Cpml, DefaultMedium, Dielectric, Ez, Shell
from gmes import PointSource, TorchRuntimeConfig, TorchSimulation

simulation = TorchSimulation(
    space=Cartesian(size=(10, 10, 0), resolution=20),
    geometry=[DefaultMedium(material=Dielectric()), Shell(material=Cpml())],
    sources=[PointSource(Continuous(freq=0.8), (0, 0, 0), Ez)],
    runtime=TorchRuntimeConfig(device="cpu", precision="float64", cpu_threads=1),
)
simulation.advance(10)
fields = simulation.host_snapshot()  # explicit host observation boundary
```

For the bounded headless example smoke, run
`uv run --no-sync python examples/air2d.py --no-plot --steps 2`.

Choose the device, real dtype, CPU thread count, and compilation policy before
construction. `compile_policy="compile"` has a first-use compilation warmup;
benchmark steady state only after that warmup. Field/probe export, checkpoint
I/O, and plotting are explicit boundaries: use `flush_probes()`,
`save_checkpoint()`, `host_snapshot()`, and an external plotting library
rather than an in-solver display API. See
[`docs/torch-execution.md`](docs/torch-execution.md) for one- and two-GPU
launch, fixed storage, checkpoints, and compile policy.

The two-GPU path is a Linux/NCCL `torchrun` launch with exactly two visible
NVIDIA devices; it is not an MPI invocation and it fails when those resources
are absent. Large scientific examples remain unsuitable as routine smoke
tests.

## Testing, packaging, and layout

Run focused tests while developing, then the complete locked suite and pure
build before submitting a compatible tree. `uv build` uses the PEP 517
configuration; do not publish local `dist/` files.

```sh
uv run --no-sync python -m mypy
uv run --no-sync python -m pylint $(git ls-files 'gmes/*.py') setup.py
uv run --no-sync python -m unittest discover -v
uv build
```

```text
gmes/       Python package and public Torch simulation API
examples/   Example electromagnetic simulations
tests/      Unit and numerical regression tests
benchmarks/ Historical and repeatable performance evidence
utils/      Data-processing and diagnostic utilities
docs/       Runtime, release, and maintenance notes
```

Historical native measurements and releases remain available for differential
context; they are not active installation or solver instructions. See
[`docs/releasing.md`](docs/releasing.md), [`CONTRIBUTING.md`](CONTRIBUTING.md),
and [`SECURITY.md`](SECURITY.md) for their respective policies.

## Contributing and support

Bug reports and patches are welcome through the [GitHub issue tracker](https://github.com/ruddyscent/gmes/issues). Historical releases and discussions remain available on the [GMES SourceForge project](https://sourceforge.net/projects/gmes/).

## License

GMES is distributed under the GNU General Public License version 3 or later
(`GPL-3.0-or-later`). See [`LICENSE`](LICENSE).

Copyright (C) 2007-2012 Kyungwon Chun.
