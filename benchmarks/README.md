# Performance benchmarks

`geometry_mapping.py` isolates bounded geometry-to-region lowering for all
built-in primitives, default-only 2-D and 3-D grids, heterogeneous and
overlapping scenes, collapsed dimensions, complex-mode coordinates, and an
independently reported custom pointwise fallback. It hashes complete material
and underlying-region maps so reference and candidate runs also verify mapping
parity. Run the Cython reference and candidate in separate checkouts with the
same options:

```sh
uv run --no-sync python benchmarks/geometry_mapping.py --repeats 7
```

Compare every case median, the built-in geometric mean, map hashes, and peak
RSS. Generated JSON results are machine-specific and should not be committed.

`field_updates.py` measures simulation construction, `FDTD.init()`, and
complete FDTD time steps as separate metrics without visualization or
simulation output. It covers a small-grid control case, larger two- and
three-dimensional dielectric grids, UPML, Drude, Lorentz, DCP ADE, and DM2
media, a heterogeneous dielectric geometry, and a Bloch-periodic complex
field. Each result also records field shapes, native material-update loop
sizes, finalized-plan run and byte counts, and process peak resident memory.

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
workloads. `native_update_plan_bytes` reports the native offset/run storage,
while `peak_rss_bytes` reports the process high-water resident set. The
top-level `warmup_steps`, `steps_per_repeat`, and `repeats` values record the
command's workload settings. Results depend on the processor, compiler,
OpenMP runtime, and system load, so generated JSON output should not be
committed.

The default `GMES_OPENMP_THRESHOLD=8192` keeps small loops serial. Setting it
to `0` forces every eligible loop through OpenMP; a value larger than the
benchmark grids provides an OpenMP-linked serial reference. The package must
be rebuilt after changing `GMES_ENABLE_OPENMP`, but the threshold and
`OMP_NUM_THREADS` are runtime settings.

When benchmarking an MPI run, choose the OpenMP thread count explicitly and
avoid assigning more total workers than physical cores.

See [`../docs/openmp-benchmark.md`](../docs/openmp-benchmark.md) for the
reference measurements and the reasoning behind the default threshold.
