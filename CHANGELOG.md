# Changelog

Notable changes to GMES are documented in this file.

## [Unreleased]

### Changed

- Make the Torch runtime the supported solver API and remove the retired
  C++/SWIG solver, generated proxies, OpenMP controls, and MPI solver launch.
- Build one universal pure-Python wheel and one sdist; GMES no longer requires
  a compiler, SWIG, Cython, OpenMP, or system headers to install.
- Make device, dtype, thread, compilation, probe, checkpoint, and external
  observation boundaries explicit. This is a breaking API migration.

### Historical

- Keep recorded OpenMP measurements as pre-cutover evidence only; they are not
  current build, tuning, or support promises.

## [0.10.0] - 2026-08-22

### Changed

- Require Python 3.14 or newer and remove Python 2 compatibility.
- Replace direct Distutils builds with a PEP 517 setuptools build.
- Update native bindings for C++23, SWIG 4, Cython 3, and NumPy 2.
- Replace field-indexing macros with `std::mdspan` views, validate index spans,
  and use zipped ranges for synchronized material updates.
- Modernize package imports, iterator behavior, integer indexing, examples, and optional I/O integrations.

### Added

- Deterministic geometry, source-time, FDTD, and HDF5 regression tests.
- Linux and macOS CI for tests and distribution builds.
- An allowed-to-fail Python prerelease compatibility check.
- Automated monthly dependency update checks.
- isort and Black formatting checks plus a focused Pylint quality gate.
- Reproducible Python 3.14 wheel and source-distribution releases through
  GitHub Actions and PyPI Trusted Publishing.

### Fixed

- Prevent shared NumPy field buffers caused by repeated array references.
- Restore field snapshot dispatch and make probe cleanup reliable.
