# AGENTS.md

This file provides guidance for contributors and coding agents working in this repository.

## Project overview

GMES (GIST Maxwell's Equations Solver) is a Python package for electromagnetic simulation using the explicit finite-difference time-domain (FDTD) method. The Python API is implemented in `gmes/`, while performance-sensitive extensions live in `src/` and are built with C++, SWIG, and Cython. Example simulations are in `examples/`, and component-level tests are in `tests/`.

## Compatibility

- Target Python 3.14 or newer and a C++23 toolchain.
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

OpenMP is enabled automatically on Linux and when Homebrew `libomp` is found
on macOS. Use `GMES_ENABLE_OPENMP=0` for a serial build. For changes to native
field updates, also exercise the parallel path explicitly:

```sh
OMP_NUM_THREADS=4 GMES_OPENMP_THRESHOLD=0 python -m unittest discover -v
```

Choose the narrowest tests that cover the change. If the required native build toolchain is unavailable, report which checks could not be run instead of claiming full verification.

## Change guidelines

- Keep changes focused on the requested behavior and avoid unrelated refactoring.
- Follow the style of the surrounding code, including its naming and Python-version conventions.
- Add or update tests when changing numerical behavior, material models, geometry, sources, or boundary conditions.
- Update documentation or examples when a public API or user-facing workflow changes.
- Do not commit generated extension artifacts, build outputs, or simulation results.
- Keep performance measurements in `benchmarks/` free of visualization and
  generated simulation output.

## Branch and merge workflow

The active `Protect master` GitHub ruleset governs changes to `master`:

- Make commits on a non-target feature or fix branch. Do not push commits directly to `master`.
- Open a pull request targeting `master`; approving reviews are not currently required.
- Bring the pull request branch up to date with the latest `master` before merging.
- Wait for both required checks to pass: `Python 3.14 / ubuntu-latest` and `Python 3.14 / macos-latest`.
- Resolve every pull request review conversation before merging.
- Use **Squash and merge**. Merge commits and rebase merges are not allowed because `master` must retain a linear history and squash is the only permitted merge method.
- Never force-push to or delete `master`. The ruleset has no bypass actors.

Verify the current settings in the [Protect master ruleset](https://github.com/ruddyscent/gmes/settings/rules/21130311) if GitHub reports different requirements.

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
