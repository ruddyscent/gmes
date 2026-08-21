# AGENTS.md

This file provides guidance for contributors and coding agents working in this repository.

## Project overview

GMES (GIST Maxwell's Equations Solver) is a Python package for electromagnetic simulation using the explicit finite-difference time-domain (FDTD) method. The Python API is implemented in `gmes/`, while performance-sensitive extensions live in `src/` and are built with C++, SWIG, and Cython. Example simulations are in `examples/`, and component-level tests are in `tests/`.

## Compatibility

- Target Python 3.14 or newer and a C++23 toolchain. Use `std::mdspan` when the standard library provides it; preserve the internal contiguous-indexing fallback when it does not.
- Python 2 compatibility is not required.
- Preserve numerical behavior unless a change is explicitly documented and covered by regression tests.
- Use the PEP 517 build declared in `pyproject.toml`; keep SWIG and Cython sources compatible with their current stable releases.
- Be mindful that some examples require substantial memory and execution time; do not use them as routine smoke tests.

## Building and testing

Install the locked development environment and optional HDF5 dependency from
the repository root. System prerequisites are `build-essential` and `swig` on
Ubuntu 24.04 or newer, or the current Xcode Command Line Tools and Homebrew
`swig` on macOS; verify them with `c++ --version` and `swig -version`.

```sh
uv python install 3.14
uv sync --locked --extra hdf5
uv run --no-sync python -m unittest discover -v
uv build
```

Choose the narrowest tests that cover the change. If the required native build toolchain is unavailable, report which checks could not be run instead of claiming full verification.

`uv sync --locked` consumes the committed lockfile. Use `uv lock --upgrade`
only in a dedicated dependency-update change. The `dev` dependencies are a
PEP 735 group installed by uv by default, not a package extra; do not restore
the former `.[dev,hdf5]` pip workflow.

## Change guidelines

- Keep changes focused on the requested behavior and avoid unrelated refactoring.
- Follow the style of the surrounding code, including its naming and Python-version conventions.
- Add or update tests when changing numerical behavior, material models, geometry, sources, or boundary conditions.
- Update documentation or examples when a public API or user-facing workflow changes.
- Do not commit generated extension artifacts, build outputs, or simulation results.

## GitHub CLI authentication

Before performing GitHub operations with the CLI, run `gh auth status`. If
the active account has an expired or invalid token, pause the GitHub operation
and refresh authentication with:

```sh
gh auth login -h github.com -w
```

Complete the browser authorization flow, then rerun `gh auth status` and a
read-only repository command such as `gh pr list` to verify access. Do not
continue with GitHub mutations such as creating, closing, or merging pull
requests until both checks succeed.

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
