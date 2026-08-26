# Native migration oracle

The machine-readable workload contract is
`native_oracle_workloads.json`.  It fixes the post-#113 physics commit,
correctness capture steps, model-specific tolerances, measured material
coverage/fragmentation cases, and the CPU/single-GPU/two-GPU gate sizes before
Torch tuning starts.

## Reference lifecycle

`native-oracle-d87d25a` points exactly to
`d87d25afd160d96b1fa0890cacecd90802448d57`, the final post-#113 native
solver.  It is the immutable physics source and can be built in isolation:

```sh
git worktree add --detach /tmp/gmes-native-source native-oracle-d87d25a
cd /tmp/gmes-native-source
uv sync --locked --extra hdf5
uv run --no-sync python -m unittest discover -v
uv build
```

That commit predates the read-only C++ state exporters required to inspect
private dispersive/PML state.  The companion
`native-oracle-observer-v4` tag contains only the #115 observer, workload,
and runner changes on top of the frozen source.  Use it to produce reference
archives; the archive metadata records both the frozen physics reference and
the actual observer checkout commit.

Keep controller, reference, and candidate environments separate.  The
controller starts Python with `-I`, removes import-related environment
variables, changes to an empty temporary directory, and rejects a `gmes`
import that is not below the requested checkout.

```sh
git worktree add --detach /tmp/gmes-native-observer native-oracle-observer-v4
cd /tmp/gmes-native-observer
uv sync --locked --extra hdf5

cd /path/to/controller-checkout
uv run --no-sync python benchmarks/run_isolated_oracle.py \
  --checkout /tmp/gmes-native-observer \
  --python /tmp/gmes-native-observer/.venv/bin/python \
  --manifest "$PWD/benchmarks/native_oracle_workloads.json" \
  --case mixed-3d \
  --output /tmp/gmes-oracle/reference-mixed-3d.npz
```

Repeat the command with a separately built candidate checkout and output
path.  Compare every field, map, time value, source/auxiliary state, and
persistent material state with:

```sh
uv run --no-sync python benchmarks/native_oracle.py compare \
  --reference /tmp/gmes-oracle/reference-mixed-3d.npz \
  --candidate /tmp/gmes-oracle/candidate-mixed-3d.npz
```

Each archive includes a `step/0` canonical input checkpoint.  It contains
the fixed-seed nonzero fields and the complete state after the manifest's
deterministic preconditioning steps.  The named 1, 2, 5, 20, and 100 captures
are relative to that checkpoint.  A candidate backend loads `step/0`
directly, so it does not need to run or import the native reference to create
its starting state.

The mixed-material cases are source-free because PointSource, TFSF, and
Gaussian source behavior has dedicated cases. Native DM2 is stable for its
homogeneous and Ziolkowski workloads but does not converge after waves from
adjacent dispersive volumes reach a 3-D DM2 volume. The 3-D mixed case therefore
keeps a normal, nonzero-state DM2 updater on one Ex Yee point and fixes a long
x-domain that isolates it for the 100-step differential window. This preserves
all material dispatcher and state-storage coverage without redefining the DM2
corrector or weakening its tolerance. DM2 transition-count and coverage
performance is measured in 2-D; 3-D coverage gates exclude volumetric DM2.

## Measurements and gates

Set thread counts before importing the native module or Torch.  Use an
otherwise idle host, fixed clock/power settings where available, and the same
workload, dtype, thread count, and replicate count for both sides.

```sh
uv run --no-sync python benchmarks/native_oracle.py benchmark \
  --case cpu-crossover-2d --threads 1 --warmup 5 --steps 100 --repeats 15 \
  > /tmp/gmes-oracle/native-cpu-crossover-2d-t1.json
```

Run each CPU gate once with one thread and once with the physical-core count
reported in `environment.cpu_count_physical`.  Raw construction, geometry
mapping, native plan initialization, one-step, and batched timings are kept
separate. Each batched timing replicate starts from an independently rebuilt,
identically seeded state so later samples do not measure a progressively older
simulation. The output also records median, p95, population standard deviation,
relative MAD, RSS samples/growth, exact field/update-run/index/parameter bytes
(with mutable state reported as a parameter-storage subset), compiler and SWIG
versions, build flags, CPU topology, GPU inventory/topology, and the
locked-environment hash.

The single- and two-GPU sizes in the manifest are frozen now but are executed
by the Torch runner introduced by later issues.  That runner must preserve
this schema while adding transfer, eager warm-up, cold/cached compile, CUDA
allocated/reserved peaks, profiler/graph-break, device-copy, allocator, and
kernel-launch data.  Synchronize CUDA only at measurement boundaries.  Use
one process per GPU for the two-GPU cases and record link topology.

The long-running `physical_checks` cases archive field energy, boundary
energy (reflection/transmission observables), maximum amplitude, finiteness,
component spectra, complete fields, and DM2 state at fixed steps.  Large NPZ,
JSON, and profiler outputs belong in CI/run artifacts or `/tmp`; do not add
them to the repository, source distribution, or wheel.

## Torch material planner matrix

`torch_material_planner.py` measures the immutable lowering contract and the
currently executable simple-material path. Its fixed cases cover homogeneous
and 16-cylinder dielectric grids, 1,000 equivalent geometry objects, 1/10/50/90
percent contiguous and fragmented stateful coverage, exact state-width
buckets, collapsed paired-real Bloch fields, thin and thick CPML shells, and a
mixed Dielectric+CPML workload. It reports plan creation and tensor-finalization
time, cells/s, launches/step, bytes/cell, PML active cells/state bytes/
gather-scatter bytes, peak host/device memory, signature normalization, and the
complete `auto` decision record.

Run every forced candidate to verify complete-field equality and the 10 percent
automatic-policy gate:

```sh
uv run --no-sync python benchmarks/torch_material_planner.py \
  --case heterogeneous-16 --policy matrix --compile-policy compile \
  --warmup 10 --steps 100 --repeats 15 --native-reference
```

Add `--device cuda:0 --precision float32 --profile` for the single-GPU matrix.
CUDA is synchronized only at measurement boundaries. The profiler record
includes operation counts and positive allocation events; generated JSON and
trace artifacts belong in `/tmp` or CI artifacts, not the repository.
PML cases execute eagerly or as full graphs; dispersive coverage and width
cases remain plan-only until #119 and #120 supply their model equations.

## Existing native microbenchmarks

`geometry_mapping.py` isolates bounded geometry-to-region lowering for all
built-in primitives, default-only 2-D and 3-D grids, heterogeneous and
overlapping scenes, collapsed dimensions, complex-mode coordinates, and an
independently reported custom pointwise fallback. It hashes complete material
and underlying-region maps so reference and candidate runs also verify mapping
parity:

```sh
uv run --no-sync python benchmarks/geometry_mapping.py --repeats 7
```

Compare every case median, the built-in geometric mean, map hashes, and peak
RSS. Generated JSON results are machine-specific and should not be committed.

`field_updates.py` measures simulation construction, `FDTD.init()`, and
complete FDTD time steps separately. It retains the #99 small, 2-D, 3-D,
dispersive, Lorentz, DCP, DM2, PML, heterogeneous, and Bloch/complex cases.
Run each configuration in a separate process so OpenMP and GMES read their
runtime settings before the first native update:

```sh
uv run --no-sync python benchmarks/field_updates.py --threads 1
uv run --no-sync python benchmarks/field_updates.py --threads 4
uv run --no-sync python benchmarks/field_updates.py --threads 4 --threshold 0
uv run --no-sync python benchmarks/field_updates.py \
  --threads 4 --threshold 1000000000
```

Compare the raw and median construction, initialization, and step samples,
field checksum/shapes, material update sizes, native update-plan bytes, and
peak RSS. The default `GMES_OPENMP_THRESHOLD=8192` keeps small loops serial.
A threshold of zero forces eligible loops through OpenMP; a value larger than
the grids provides an OpenMP-linked serial reference. Rebuild after changing
`GMES_ENABLE_OPENMP`, but not for threshold or `OMP_NUM_THREADS` changes.
For MPI runs, set the OpenMP thread count explicitly and avoid assigning more
total workers than physical cores.

See [the OpenMP benchmark notes](../docs/openmp-benchmark.md) for the reference
measurements and threshold rationale.
