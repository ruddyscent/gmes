# Performance benchmarks

`field_updates.py` measures complete FDTD time steps without visualization or
simulation output. It covers a small two-dimensional dielectric grid, a
three-dimensional dielectric grid, and a two-dimensional Drude dispersive
medium.

Run each thread count in a separate process so the OpenMP runtime reads the
requested setting before its first parallel region:

```sh
python benchmarks/field_updates.py --threads 1
python benchmarks/field_updates.py --threads 2
python benchmarks/field_updates.py --threads 4
python benchmarks/field_updates.py --threads 8
```

Compare `median_seconds_per_step` for performance and `checksum` for numerical
equivalence. Benchmark results depend on the processor, compiler, OpenMP
runtime, and system load, so generated output should not be committed.

The default `GMES_OPENMP_THRESHOLD=32768` keeps small loops serial. Override
the value to study the crossover point, or set it to `0` to force every
eligible loop through OpenMP. The package must be rebuilt after changing
`GMES_ENABLE_OPENMP`, but the threshold and `OMP_NUM_THREADS` are runtime
settings that must be set before the first field update.

When benchmarking an MPI run, choose the OpenMP thread count explicitly and
avoid assigning more total workers than physical cores.

See [`../docs/openmp-benchmark.md`](../docs/openmp-benchmark.md) for the initial
reference measurements and the reasoning behind the default threshold.
