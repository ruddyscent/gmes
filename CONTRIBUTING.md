# Contributing to GMES

GMES targets the latest stable Python 3 release. Python 2 compatibility and the legacy Distutils workflow are intentionally out of scope.

## Development setup

Install a C++23 compiler and standard library with `std::mdspan` support,
SWIG 4, and [uv](https://docs.astral.sh/uv/). Then create the locked Python
3.14 development environment from the repository root:

```sh
uv python install 3.14
uv sync --locked --extra hdf5
```

The `dev` dependency group is installed by default. Use the following command
when work also needs plotting, MPI, and HDF5 support:

```sh
uv sync --locked --extra all
```

## Verification

Run the narrowest relevant tests while developing, followed by the complete suite before submitting a change:

```sh
uv run --no-sync python -m unittest tests.test_geometry -v
uv run --no-sync python -m isort --check-only gmes examples tests utils setup.py
uv run --no-sync python -m black --check gmes examples tests utils setup.py
uv run --no-sync python -m pylint gmes setup.py
uv run --no-sync python -m unittest discover -v
uv build
```

Numerical behavior changes must include a deterministic regression test. Avoid using the large examples as routine tests because some need more than 1 GB of memory and run for a long time.

Generated SWIG proxies, Cython C/C++ output, compiled extensions, distributions, and simulation results must not be committed.

## Change scope and compatibility

- Keep public API changes explicit and document their migration impact.
- Preserve numerical behavior unless the change is intentional and covered by tests.
- Keep C++, SWIG, and Cython sources compatible with the current stable toolchain.
- Update the README or examples whenever a user-facing workflow changes.

## Commit messages

Use Conventional Commit titles in the form `<type>(<scope>): <summary>` or `<type>: <summary>`. Keep the title imperative and add a body that explains why the change is needed, its compatibility or numerical implications, and the verification performed.
