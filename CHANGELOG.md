# Changelog

Notable changes to GMES are documented in this file.

## [0.10.0] - Unreleased

### Changed

- Require Python 3.14 or newer and remove Python 2 compatibility.
- Replace direct Distutils builds with a PEP 517 setuptools build.
- Update native bindings for C++17, SWIG 4, Cython 3, and NumPy 2.
- Modernize package imports, iterator behavior, integer indexing, examples, and optional I/O integrations.

### Added

- Deterministic geometry, source-time, FDTD, and HDF5 regression tests.
- Linux and macOS CI for tests and distribution builds.
- An allowed-to-fail Python prerelease compatibility check.
- Automated monthly dependency update checks.
- isort and Black formatting checks plus a focused Pylint quality gate.

### Fixed

- Prevent shared NumPy field buffers caused by repeated array references.
- Restore field snapshot dispatch and make probe cleanup reliable.
