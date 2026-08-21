# Contributing to GMES

GMES targets the latest stable Python 3 release. Python 2 compatibility and the legacy Distutils workflow are intentionally out of scope.

## Development setup

Install a C++17 compiler and SWIG 4, then create an isolated Python 3.14 environment from the repository root:

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,hdf5]"
```

Use `.[all,dev]` when work also needs plotting and MPI support.

## Verification

Run the narrowest relevant tests while developing, followed by the complete suite before submitting a change:

```sh
python -m unittest tests.test_geometry -v
python -m isort --check-only gmes examples tests utils setup.py
python -m black --check gmes examples tests utils setup.py
python -m pylint gmes setup.py
python -m unittest discover -v
python -m build
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
