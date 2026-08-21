# AGENTS.md

This file provides guidance for contributors and coding agents working in this repository.

## Project overview

GMES (GIST Maxwell's Equations Solver) is a Python package for electromagnetic simulation using the explicit finite-difference time-domain (FDTD) method. The Python API is implemented in `gmes/`, while performance-sensitive extensions live in `src/` and are built with C++, SWIG, and Cython. Example simulations are in `examples/`, and component-level tests are in `tests/`.

## Compatibility

- Target Python 3.14 or newer and a C++23 toolchain with `std::mdspan` support.
- Python 2 compatibility is not required.
- Preserve numerical behavior unless a change is explicitly documented and covered by regression tests.
- Use the PEP 517 build declared in `pyproject.toml`; keep SWIG and Cython sources compatible with their current stable releases.
- Be mindful that some examples require substantial memory and execution time; do not use them as routine smoke tests.

## Building and testing

Build and install the package in an isolated environment from the repository root:

```sh
python -m pip install -e ".[dev,hdf5]"
```

Run the test suite:

```sh
python -m unittest discover -v
```

Choose the narrowest tests that cover the change. If the required native build toolchain is unavailable, report which checks could not be run instead of claiming full verification.

## Change guidelines

- Keep changes focused on the requested behavior and avoid unrelated refactoring.
- Follow the style of the surrounding code, including its naming and Python-version conventions.
- Add or update tests when changing numerical behavior, material models, geometry, sources, or boundary conditions.
- Update documentation or examples when a public API or user-facing workflow changes.
- Do not commit generated extension artifacts, build outputs, or simulation results.

## Commit messages

- Write the commit title with an appropriate Conventional Commit prefix, using the form `<type>: <summary>` or `<type>(<scope>): <summary>`.
- Use a concise, imperative summary. Do not end the title with a period.
- Common prefixes include `feat`, `fix`, `docs`, `test`, `refactor`, `build`, `ci`, `perf`, `style`, `chore`, and `revert`.
- Include a commit description (body) explaining what changed and why, unless the change is trivial or the entire commit is very short and self-explanatory.
- Separate the title and description with a blank line.
- In the description, call out compatibility implications, numerical behavior changes, and tests performed when relevant.

Example:

```text
fix(material): handle Lorentz pole coefficients consistently

Use the same time-step normalization in the electric and magnetic
update paths to avoid divergent results. Add coverage for both paths.
```
