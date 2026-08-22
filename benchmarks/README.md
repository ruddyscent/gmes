# Performance benchmarks

`field_updates.py` measures simulation construction, `FDTD.init()`, and
complete FDTD time steps as separate metrics without visualization or
simulation output. It covers a small-grid control case, larger two- and
three-dimensional dielectric grids, a two-dimensional Drude dispersive
medium, a heterogeneous dielectric geometry, and a Bloch-periodic complex
field. Each result also records field shapes and the native material-update
loop sizes.

Run each configuration in a separate process so the OpenMP runtime reads the
thread count and GMES reads the threshold before the first native update:

```sh
uv run --no-sync python benchmarks/field_updates.py --threads 1
uv run --no-sync python benchmarks/field_updates.py --threads 4
uv run --no-sync python benchmarks/field_updates.py --threads 4 --threshold 0
uv run --no-sync python benchmarks/field_updates.py --threads 4 --threshold 1000000000
```

The benchmark imports GMES after applying the requested runtime settings and
before any timer starts. For every case and repeat, it constructs a fresh
simulation and then times `FDTD.init()` independently. The initialization
metric includes field allocation plus material and source mapping. After the
last initialization sample, the existing warmup and repeated step measurement
runs on that simulation.

Compare `median_seconds_per_construction`,
`median_seconds_per_initialization`, and `median_seconds_per_step` for
performance. Each median is computed from the samples in its corresponding
`seconds_per_*` list. Compare `checksum` for numerical equivalence, and use
`field_shapes` and `material_update_sizes` to confirm that runs have matching
workloads. The top-level `warmup_steps`, `steps_per_repeat`, and `repeats`
values record the command's workload settings. Results depend on the
processor, compiler, OpenMP runtime, and system load, so generated JSON output
should not be committed.

The default `GMES_OPENMP_THRESHOLD=8192` keeps small loops serial. Setting it
to `0` forces every eligible loop through OpenMP; a value larger than the
benchmark grids provides an OpenMP-linked serial reference. The package must
be rebuilt after changing `GMES_ENABLE_OPENMP`, but the threshold and
`OMP_NUM_THREADS` are runtime settings.

When benchmarking an MPI run, choose the OpenMP thread count explicitly and
avoid assigning more total workers than physical cores.

See [`../docs/openmp-benchmark.md`](../docs/openmp-benchmark.md) for the
reference measurements and the reasoning behind the default threshold.
