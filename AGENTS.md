# AGENTS.md

This file provides guidance for contributors and coding agents working in this repository.

## Project overview

GMES (GIST Maxwell's Equations Solver) is a Python package for electromagnetic simulation using the explicit finite-difference time-domain (FDTD) method. The Python API is implemented in `gmes/`, while performance-sensitive extensions live in `src/` and are built with C++, SWIG, and Cython. Example simulations are in `examples/`, and component-level tests are in `tests/`.

## Compatibility

- Treat the current Python 2.7 and C++11 compatibility as intentional unless a task explicitly requests modernization.
- Preserve the existing public API and numerical behavior when making internal changes.
- Do not replace the Distutils, SWIG, or Cython build system as part of an unrelated change.
- Be mindful that some examples require substantial memory and execution time; do not use them as routine smoke tests.

## Building and testing

Build extensions in place from the repository root:

```sh
python setup.py build_ext --inplace
```

Run relevant tests individually:

```sh
python tests/<test_file>.py
```

Choose the narrowest tests that cover the change. If the required legacy toolchain is unavailable, report which checks could not be run instead of claiming full verification.

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
