# Performance benchmarks

`field_updates.py` measures complete FDTD time steps without visualization or
simulation output. It covers a small-grid control case, larger two- and
three-dimensional dielectric grids, and a two-dimensional Drude dispersive
medium. Each result also records the native material-update loop sizes.

Run each configuration in a separate process so the OpenMP runtime reads the
thread count and GMES reads the threshold before the first native update:

```sh
uv run --no-sync python benchmarks/field_updates.py --threads 1
uv run --no-sync python benchmarks/field_updates.py --threads 4
uv run --no-sync python benchmarks/field_updates.py --threads 4 --threshold 0
uv run --no-sync python benchmarks/field_updates.py --threads 4 --threshold 1000000000
```

Compare `median_seconds_per_step` for performance and `checksum` for numerical
equivalence. Results depend on the processor, compiler, OpenMP runtime, and
system load, so generated JSON output should not be committed.

The default `GMES_OPENMP_THRESHOLD=8192` keeps small loops serial. Setting it
to `0` forces every eligible loop through OpenMP; a value larger than the
benchmark grids provides an OpenMP-linked serial reference. The package must
be rebuilt after changing `GMES_ENABLE_OPENMP`, but the threshold and
`OMP_NUM_THREADS` are runtime settings.

When benchmarking an MPI run, choose the OpenMP thread count explicitly and
avoid assigning more total workers than physical cores.

See [`../docs/openmp-benchmark.md`](../docs/openmp-benchmark.md) for the
reference measurements and the reasoning behind the default threshold.
