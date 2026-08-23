# Changelog

Notable changes to GMES are documented in this file.

## [Unreleased]

### Added

- Optional OpenMP parallelism for large native material-update loops, with
  build-time fallback, runtime introspection, and a tunable cell threshold.
- Repeatable field-update benchmarks for dielectric and dispersive workloads.

### Changed

- Replace the untyped Cython material configuration layer with an
  import-compatible Python module while keeping native field-update kernels.

### Fixed

- macOS auto-detection now skips `libomp` runtimes whose minimum deployment
  target is newer than the extension target instead of producing a
  deceptively tagged wheel with a linker warning.

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
