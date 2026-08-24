# Contributing to GMES

GMES targets the latest stable Python 3 release. Python 2 compatibility and the legacy Distutils workflow are intentionally out of scope.

## Development setup

Install a C++23 compiler, SWIG 4, and [uv](https://docs.astral.sh/uv/). On
Ubuntu 24.04 or newer use `sudo apt-get install --yes build-essential swig`;
on macOS install the current Xcode Command Line Tools and run
`brew install swig libomp`. Verify the selected tools with `c++ --version` and
`swig -version`. Homebrew `libomp` is optional for serial-only development but
required when testing OpenMP changes on macOS.

The build uses `std::mdspan` when `<mdspan>` is available and otherwise uses
its internal contiguous-indexing fallback; C++23 mode itself is required.
Create and verify the locked Python 3.14 development environment from the
repository root:

```sh
uv python install 3.14
uv sync --locked --extra hdf5
uv run --no-sync python -m unittest discover -v
uv build
```

The `dev` dependency group is installed by default. Use the following command
when work also needs plotting, MPI, and HDF5 support:

```sh
uv sync --locked --extra all
```

Developers migrating from `python -m pip install -e ".[dev,hdf5]"` should use
the uv setup above. `dev` is a PEP 735 dependency group, not an installable
extra; runtime extras such as `hdf5` are still selected with `--extra`.

uv's editable-project cache tracks the C++ and SWIG inputs under `src/`.
Changing a `*.cc`, `*.hh`, or `*.i` file makes the next
`uv sync` rebuild the native extensions. Pure Python files remain directly
editable and do not trigger a native rebuild. If a build input outside the
configured cache keys changes, force a rebuild with:

```sh
uv sync --locked --extra hdf5 --reinstall-package gmes
```

## Dependency updates

Local development and CI use uv 0.12.5. Check `uv --version` before updating
the environment. Update runtime and development dependencies only in a
dedicated change:

```sh
uv lock --upgrade
uv sync --locked --extra hdf5
uv run --no-sync python -m unittest discover -v
uv build
```

The native build constraints in `pyproject.toml` pin setuptools, wheel, and
NumPy for isolated builds. Build-time NumPy must remain identical
to the NumPy version in `uv.lock`; update both together, then verify an
editable sync and a distribution build. When changing the required uv release,
update both `[tool.uv].required-version` and the `setup-uv` workflow input in
the same pull request.

Keep the `build` package in the `dev` group. The packaging regression in
`tests/test_packaging.py` invokes `python -m build --sdist`, while `uv build`
remains the documented command for contributor distribution builds.

## Verification

Run the narrowest relevant tests while developing, followed by the complete suite before submitting a change:

```sh
uv run --no-sync python -m unittest tests.test_geometry -v
uv run --no-sync python -m isort --check-only gmes examples tests benchmarks utils setup.py
uv run --no-sync python -m black --check gmes examples tests benchmarks utils setup.py
uv run --no-sync python -m pylint gmes setup.py
uv run --no-sync python -m unittest discover -v
uv build
```

Changes to native update loops must also cover the forced-parallel and serial
build paths:

```sh
GMES_ENABLE_OPENMP=1 uv sync --locked --extra hdf5 --reinstall-package gmes
OMP_NUM_THREADS=4 GMES_OPENMP_THRESHOLD=0 uv run --no-sync python -m unittest discover -v
GMES_ENABLE_OPENMP=0 uv sync --locked --extra hdf5 --reinstall-package gmes
GMES_ENABLE_OPENMP=0 uv run --no-sync python -m unittest discover -v
```

Numerical behavior changes must include a deterministic regression test. Avoid using the large examples as routine tests because some need more than 1 GB of memory and run for a long time.

Generated SWIG proxies, compiled extensions, distributions, and simulation results must not be committed.

Use `benchmarks/field_updates.py` to evaluate field-update performance. Run
each thread count or threshold in a separate process and compare both timing
and checksum output. Keep visualization, generated JSON, and simulation output
out of the repository. The reference measurements and threshold rationale are
recorded in [`docs/openmp-benchmark.md`](docs/openmp-benchmark.md).

Use `benchmarks/geometry_mapping.py` to compare complete geometry region maps,
mapping time, and peak RSS against the Cython reference implementation.

Production distributions are created only by the tag-triggered release
workflow. Follow [`docs/releasing.md`](docs/releasing.md) when preparing a
release; never publish artifacts from a developer workstation or reuse files
already present in `dist/`.

## Change scope and compatibility

- Keep public API changes explicit and document their migration impact.
- Preserve numerical behavior unless the change is intentional and covered by tests.
- Keep C++ and SWIG sources compatible with the current stable toolchain.
- Update the README or examples whenever a user-facing workflow changes.

## Commit messages

Use Conventional Commit titles in the form `<type>(<scope>): <summary>` or `<type>: <summary>`. Keep the title imperative and add a body that explains why the change is needed, its compatibility or numerical implications, and the verification performed.
