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
- NumPy 2.3 or newer, SciPy 1.16 or newer, and PyTorch `>=2.14,<2.15`

GMES itself is a universal pure-Python package: installation and source builds
do not require a C/C++ compiler, SWIG, Cython, OpenMP, or system headers.
`plot` and `hdf5` remain optional extras for external plotting and generic HDF5
use.

Use one explicit Torch runtime extra in a checkout; the extras select the
locked CPU, CUDA 12.6, or CUDA 13.0 PyTorch index and must not be combined:

```sh
uv python install 3.14
uv sync --locked --extra torch-cpu --extra hdf5
uv run --no-sync python -m pytest -v
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

## Custom vectorized geometry

Decorate a `GeometricObject` subclass with `@gmes.vectorized_geometry` to use
bounded array containment during material planning. The decorator is also
available from `gmes.geometry`. See the complete hemisphere implementation in
[`examples/vectorized_geometry.py`](examples/vectorized_geometry.py), runnable
with `uv run --no-sync python examples/vectorized_geometry.py`.

The supported array hook is `_contains_points(x, y, z)` (its historical name is
retained). It receives three same-length, one-dimensional NumPy `float64`
coordinate arrays and must return a NumPy Boolean array of exactly the same
shape. Every entry must equal `in_object((x[i], y[i], z[i]))`, including boundary
points. The scalar method must return a Python `bool`. Preserve a bounding box
that encloses the object. Inputs must not be mutated; predicates must not depend
on tile boundaries, point ordering, or call counts. A direct call with empty
arrays must return a Boolean array of shape `(0,)`; empty lowering tiles return
empty maps without calling predicates. Lowering rejects non-Boolean/non-array
results with `TypeError` and wrong shapes with `ValueError`. These checks cannot
prove semantic equivalence; test the scalar and array implementations together.

Parameter-only subclasses inherit the public opt-in. A subclass that replaces
`in_object()` or `_contains_points()` falls back to scalar containment until it
is decorated again, even if the replacement is semantically equivalent. Normal
Python method resolution order selects the inherited contract, so a mixin that
changes either effective method also triggers fallback. Renew opt-in only after
ensuring both predicates agree. Classmethod predicates are compared by their
underlying implementation, so binding an inherited method to a subclass does
not require renewed opt-in. Undecorated custom classes, including bare
subclasses of built-in shapes, retain scalar fallback.

The legacy `_gmes_vectorized_geometry = True` marker remains supported with its
original inherited semantics, and a legacy child can set it to `False` to opt
out. Legacy authors must maintain equivalent predicates when overriding either
method; the marker alone does not detect stale inherited containment. Once a
public decorator contract is inherited or declared, it takes precedence over
both `True` and `False` legacy markers and applies the method-change policy
above. Migration can be as small as decorating the existing valid legacy class.

The decorator requires a geometry class with scalar and array predicates; the
base class's scalar-loop array implementation is not an array opt-in. Invalid
classes or instances raise `TypeError`. Decorating returns the original class,
wraps no methods, and can be repeated. Names, signatures, class identity, and
existing serialization hooks remain intact. Custom classes still need the usual
importable class definition and serialization of their own additional state.

Array opt-in and result validation run during host geometry lowering, before
compiled stepping and during distributed cost planning. They add no timestep
callbacks or device synchronization; the planner's existing tile bound remains
in effect.

## Testing, packaging, and layout

Run focused tests while developing, then the complete locked suite and pure
build before submitting a compatible tree. `uv build` uses the PEP 517
configuration; do not publish local `dist/` files.

```sh
uv run --no-sync python -m mypy
uv run --no-sync python -m pylint $(git ls-files 'gmes/*.py') setup.py
uv run --no-sync python -m pytest -v
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
