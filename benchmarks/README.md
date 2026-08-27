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

Schema 2 also records the initializer, seed and scale, warm-up count, timed
steps, replicate count, timer, and sample-start semantics as an explicit
`benchmark_contract`. Native/Torch hard-gate comparisons must reject a
mismatched contract and must authenticate the frozen observer commit.
The legacy v4 summary predates the embedded field; it is accepted only with
its exact clean observer commit and the SHA-256 of the ANSI-sanitized JSON
artifact published in #115. A same-commit rerun with different warm-up or
sample semantics is rejected. Every replacement summary, including one with an
embedded contract, requires a separately frozen observer commit and exact
content digest in the manifest before it can participate in a hard gate.

The single- and two-GPU sizes in the manifest are frozen now but are executed
by the Torch runner introduced by later issues.  That runner must preserve
this schema while adding transfer, eager warm-up, cold/cached compile, CUDA
allocated/reserved peaks, profiler/graph-break, device-copy, allocator, and
kernel-launch data.  Synchronize CUDA only at measurement boundaries.  Use
one process per GPU for the two-GPU cases and record link topology.

## Torch compiler and runtime tuning

`torch_tuning.py` consumes the frozen manifest and native summary. Authoritative
native ratios use an explicit `time.perf_counter` plus device synchronization
loop so Torch does not receive `Timer.timeit()`'s two hidden statement warmups.
Every authoritative replicate restores the seeded pre-warm-up checkpoint and
executes the real manifest warm-up before timing, matching the native repeat
contract. Repeated `torch.utils.benchmark.Timer` samples are retained in a separate,
exploratory record and never feed the native gate. The runner records
construction, H2D, cold and cached compilation, one-step latency, batched
throughput, compiler counters, raw samples, buffer addresses, memory growth,
topology, and a Chrome profiler trace. The strict acceptance summary rejects graph breaks,
recompilation after warm-up, host-device transfers, storage changes, or unbounded
device-memory growth. CPU hard-gate evidence covers crossover, large mixed, and
paired-real Bloch cases in both 2-D and 3-D at one thread and at the physical-core
thread count. Each isolated thread slice must contain all six cases; both slices
are required before epic acceptance. Schema 3 binds each CPU slice to the exact manifest, runner inputs, solver
inputs and ABI, clean candidate commit, and host identity. Aggregation requires
`--native-summary`, reopens the exact SHA-pinned artifact, and recomputes every
native comparison instead of trusting embedded booleans. Raw samples are also checked for positivity,
replicate count, relative MAD, the 5% individual limit, and a deterministic
one-sided bootstrap test of the log-geometric-mean ratio.

The current dense/compact/tiled selector changes planner metadata and optional
storage only; material execution still uses one dense dielectric base plus
compact indexed material updates. Policy-matrix timings are therefore retained
as exploratory raw data and fail closed instead of claiming the 10% automatic
policy gate. That gate can be enabled only after the forced policies select
distinct executable representations.

```sh
uv run --no-sync python -m benchmarks.torch_tuning \
  --case cpu-crossover-2d --device cpu --precision float64 \
  --threads 1 --interop-threads 1 --native-summary native-summary.json \
  --trace-directory /tmp/gmes-tuning-traces --output /tmp/gmes-tuning.json

uv run --no-sync python -m benchmarks.torch_tuning \
  --case single-gpu-3d --device cuda:0 --precision float32 \
  --compile-mode matrix --capture-graphs \
  --trace-directory /tmp/gmes-tuning-traces --output /tmp/gmes-cuda.json
```

Use `--policy matrix` for the forced dense/compact/tiled comparison and
`--enforce` in automation. Keep JSON and profiler traces in CI artifacts or
`/tmp`; the files contain the complete environment metadata and can be large.
Run the six-case `cpu-gates` command in separate processes for `--threads 1`
and the affinity-aware physical-core count, then aggregate them. A single case
or thread slice is diagnostic-only and cannot report epic CPU suite success:

```sh
uv run --no-sync python -m benchmarks.torch_tuning \
  --case cpu-gates --device cpu --threads 1 --interop-threads 1 \
  --native-summary native-summary.json --output /tmp/cpu-one.json

uv run --no-sync python -m benchmarks.torch_tuning \
  --case cpu-gates --device cpu --threads 4 --interop-threads 1 \
  --native-summary native-summary.json --output /tmp/cpu-physical.json

uv run --no-sync python -m benchmarks.torch_tuning \
  --case cpu-gates --native-summary native-summary.json \
  --cpu-slice-artifacts /tmp/cpu-one.json /tmp/cpu-physical.json \
  --output /tmp/cpu-acceptance.json --enforce
```

The long-running `physical_checks` cases archive field energy, boundary
energy (reflection/transmission observables), maximum amplitude, finiteness,
component spectra, complete fields, and DM2 state at fixed steps.  Large NPZ,
JSON, and profiler outputs belong in CI/run artifacts or `/tmp`; do not add
them to the repository, source distribution, or wheel.

## Torch material planner matrix

`torch_material_planner.py` measures the immutable lowering contract and the
executable simple, PML, and dispersive paths. Its fixed cases cover homogeneous
and 16-cylinder dielectric grids, 1,000 equivalent dielectric or Drude objects,
1/10/50/90 percent contiguous and fragmented Drude coverage, exact-width
buckets, Drude/Lorentz pole widths 1 and 4, every DCP strategy, collapsed
paired-real Bloch fields, thin and thick CPML shells, and a mixed
Dielectric+PML+Drude+Lorentz+DCP workload. It reports plan creation and tensor
finalization time, cells/s, launches/step, bytes/cell, PML and dispersive state
bytes, gather/scatter bytes, peak host/device memory, signature normalization,
and the complete `auto` decision record.

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
PML, dispersive, and DM2 cases execute eagerly or with compiled bulk/PML
phases. Results include exact-width state bytes, the bounded-padding
alternative's element count, avoided padding, and signature-bounded launch
counts; use `--native-reference` on CPU to record the native step ratio.

## Torch DM2 matrix

`torch_dm2.py` executes the DM2 state equations over exact transition widths
1, 4, and 8; a mixed 1/2/4/8-width case; 10 and 50 percent contiguous and
fragmented coverage; all-material 2-D and 3-D workloads combining Dielectric,
CPML, Drude, Lorentz, every DCP strategy, and DM2; and a deliberately
nonconverging workload. It reports
cells/s, persistent and scratch bytes, fixed-address validation, actual
iteration distributions, deterministic failure status, and the exact-width
versus bounded-padding storage/arithmetic ratio.

```sh
uv run --no-sync python benchmarks/torch_dm2.py \
  --case mixed-widths --compile-policy compile \
  --warmup 2 --steps 10 --repeats 5

uv run --no-sync python benchmarks/torch_dm2.py \
  --case hard-nonconverging --device cuda:0 --precision float32

uv run --no-sync python benchmarks/torch_dm2.py \
  --case all-material-3d --compile-policy compile
```

Run all forced planner policies for differential checks and repeat the matrix
on CPU with one/tuned threads and on CUDA in eager/compiled modes. CUDA is
synchronized only at measurement boundaries; generated JSON belongs in
`/tmp` or CI artifacts.

## Two-GPU Torch scaling

The fixed two-GPU runner compares one and two GPUs inside the same
`torchrun` job, restores the same captured checkpoint for every replicate,
and keeps construction, compute-graph capture, warm steady-state timing,
memory, and profiler evidence separate. The strong mixed case uses the frozen
`two-gpu-3d` size from `native_oracle_workloads.json`; the weak case doubles
the 96x96x96 single-GPU volume along x. Homogeneous and deliberately
imbalanced mixed cases validate the decomposition decision separately.

Install the CUDA 12.6 lock variant and capture the physical inventory before
the run. PyTorch device enumeration may differ from the physical
`nvidia-smi -L` order, so the JSON device table and rank diagnostics are the
authoritative binding record.

```sh
uv sync --locked --extra torch-cu126 --extra hdf5
nvidia-smi -L
nvidia-smi topo -m

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
uv run --no-sync torchrun --standalone --nproc-per-node=2 \
  --module benchmarks.torch_two_gpu \
  --case strong-mixed --warmup 5 --steps 100 --repeats 15 \
  --profile-steps 10 --enforce \
  --output /tmp/gmes-two-gpu/strong-mixed.json \
  --trace-directory /tmp/gmes-two-gpu/traces-strong

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
uv run --no-sync torchrun --standalone --nproc-per-node=2 \
  --module benchmarks.torch_two_gpu \
  --case weak-mixed --warmup 5 --steps 100 --repeats 15 \
  --profile-steps 10 --enforce \
  --output /tmp/gmes-two-gpu/weak-mixed.json \
  --trace-directory /tmp/gmes-two-gpu/traces-weak
```

Repeat with `--case strong-homogeneous` and
`--case strong-imbalanced`. JSON output records every timing sample,
one-versus-two throughput, the 1.6 strong or 0.8 weak gate, rank-local memory,
halo bytes, fixed-address validation, exact decomposition, PyTorch/CUDA/NCCL
versions, both GPU models, peer-access status, and `nvidia-smi topo -m`.
Per-rank Chrome traces remain in the requested trace directory and separate
pack/launch, boundary wait, unpack, NCCL device time, compute overlap, and
exposed communication. Generated JSON and traces belong in `/tmp` or CI
artifacts, not in the repository.

The frozen native reference/candidate commands earlier in this document remain
the correctness-oracle handoff. The two-GPU runner is the candidate performance
handoff: it consumes the same mixed-coverage recipe and fixed GPU sizes without
importing or launching the native solver in either CUDA rank.

Run the complete deterministic field/material/source/restart matrix with:

```sh
OMP_NUM_THREADS=1 uv run --no-sync torchrun --standalone \
  --nproc-per-node=2 --module benchmarks.torch_two_gpu_correctness \
  --output /tmp/gmes-two-gpu/correctness.json

OMP_NUM_THREADS=1 uv run --no-sync torchrun --standalone \
  --nproc-per-node=2 --module benchmarks.torch_two_gpu_correctness \
  --capture-graphs --output /tmp/gmes-two-gpu/correctness-graphs.json
```

The failure-contract runner expects the first three modes to report
`"passed": true`; `rank-failure` intentionally returns nonzero so `torchrun`
can terminate its peer rank:

```sh
uv run --no-sync torchrun --standalone --nproc-per-node=2 \
  --module benchmarks.torch_two_gpu_failures strict-peer
uv run --no-sync torchrun --standalone --nproc-per-node=2 \
  --module benchmarks.torch_two_gpu_failures dtype-mismatch
uv run --no-sync torchrun --standalone --nproc-per-node=2 \
  --module benchmarks.torch_two_gpu_failures checkpoint-mismatch
uv run --no-sync torchrun --standalone --nproc-per-node=2 \
  --module benchmarks.torch_two_gpu_failures rank-failure
```

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
