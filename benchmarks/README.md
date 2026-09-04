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
`native-oracle-observer-v6` contains the #115 observer plus the strict-JSON
metadata serializer used for correctness archives. Use v6 for correctness
capture; archive metadata records both the frozen physics reference and the
actual observer checkout commit. The already completed 12-cell native timing
matrix remains byte-pinned to `native-oracle-observer-v5`; the manifest keeps
that performance observer provenance separate from the v6 correctness pin.

Keep controller, reference, and candidate environments separate.  The
controller starts Python with `-I`, removes import-related environment
variables, changes to an empty temporary directory, and rejects a `gmes`
import that is not below the requested checkout.

```sh
git worktree add --detach /tmp/gmes-native-observer native-oracle-observer-v6
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

`native_oracle.py capture` always constructs the legacy native `gmes.FDTD`
solver. Repeating this command from another checkout is useful only as a
native-to-native runner smoke test; it is not Torch correctness evidence.
`torch_correctness.py capture` independently reconstructs the canonical
starting state from the backend-neutral manifest, verifies that reconstruction
against every descriptor-bound native `step/0` array, and then advances the
live Torch plan, source, auxiliary, material, and persistent-state buffers. It
emits archive schema 2 with `backend: "torch"` and the complete required key
closure. Compare the reference and candidate with:

```sh
uv run --no-sync python benchmarks/native_oracle.py compare \
  --reference /tmp/gmes-oracle/reference-mixed-3d.npz \
  --candidate /tmp/gmes-oracle/candidate-mixed-3d.npz
```

The comparator validates both archives independently before comparing values.
It rejects dirty or malformed provenance, a reference checkout other than the
manifest's observer commit, unknown backends, workload/capture-contract drift,
missing or extra arrays, reshaped maps, inconsistent physical summaries, and
incorrect byte accounting. A native archive compared with itself is only a
validator smoke test and does not satisfy the Torch evidence requirement.
Every source-free mixed archive must also keep field energy below 100 times
its `step/0` value at every capture, so a finite but numerically runaway
reference cannot become an acceptance oracle.

Each archive includes a `step/0` canonical input checkpoint. It contains the
fixed-seed nonzero fields and the complete state after the manifest's
deterministic preconditioning steps. The named 1, 2, 5, 20, and 100 captures
are relative to that checkpoint. The Torch candidate derives the same state
independently from the manifest and uses the reference `step/0` only as an
exact, descriptor-bound validation input; it never imports or executes the
native reference implementation.

The mixed-material cases are source-free because PointSource, TFSF, and
Gaussian source behavior has dedicated cases. Native DM2 is stable for its
homogeneous and Ziolkowski workloads but does not converge after waves from
adjacent dispersive volumes reach a 3-D DM2 volume. The 3-D mixed case therefore
keeps a normal, nonzero-state DM2 updater on one Ex Yee point and fixes a long
x-domain that isolates it for the 100-step differential window. This preserves
all material dispatcher and state-storage coverage without redefining the DM2
corrector or omitting its state from the comparison contract. Its background
matches the instantaneous
dispersive coefficients, and both PML shells are capped by the shortest active
domain dimension so their opposite faces remain disjoint. DM2 transition-count
and coverage performance is measured in 2-D; 3-D coverage gates exclude
volumetric DM2.

## Measurements and gates

Set thread counts before importing the native module or Torch.  Use an
otherwise idle host, fixed clock/power settings where available, and the same
workload, dtype, thread count, and replicate count for both sides.

```sh
uv run --no-sync python benchmarks/native_oracle.py benchmark \
  --case cpu-crossover-2d --threads 1 --warmup 5 --steps 100 --repeats 15 \
  > /tmp/gmes-oracle/native-cpu-crossover-2d-t1.json
```

Capture all six CPU acceptance cases at one and four threads into a directory
containing only those 12 cell JSON files. Assemble them without sanitizing or
rewriting the source bytes:

```sh
uv run --no-sync python benchmarks/native_summary.py \
  --output /tmp/gmes-oracle/native-summary.json \
  /tmp/gmes-oracle/cells/*.json
# Linux
sha256sum /tmp/gmes-oracle/native-summary.json
# macOS
shasum -a 256 /tmp/gmes-oracle/native-summary.json
```

The assembler requires the exact six-case × two-thread matrix, the frozen
benchmark contract and 1/4-thread pins, clean observer commit, one normalized
host/toolchain identity, internally consistent raw statistics, updater/memory
accounting, and source-file SHA-256 provenance. It rejects ANSI-bearing input
rather than silently changing bytes and writes a deterministic case/thread
order. It does not update the manifest pin. A new required informational
summary must first be generated from a separately frozen observer-only commit,
published, and then have both that commit and the exact summary digest reviewed
and pinned. The accepted matrix now contains all 12 cells from
`native-oracle-observer-v5` at
`1ab94e579dc52861db7ecdcd55f24f8af1977de7`; its deterministic summary SHA-256
is `1c9bdce2717ba858fd03b2e40302a5b2d19a29920496f969e33aee36e34e1baa`.

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
`benchmark_contract`. Native/Torch comparisons must reject a mismatched
contract and must authenticate the frozen performance observer commit. The
historical four-cell v4 summary remains an audit record but is not accepted by
the current #123 completion contract. A same-commit rerun with different
warm-up or sample semantics is rejected. Every replacement summary requires a
separately frozen observer commit and exact content digest in the manifest
before it can serve as required comparison evidence.

The single- and two-GPU sizes remain frozen in the manifest.
`torch_tuning.py`, `torch_two_gpu.py`, and
`torch_two_gpu_correctness.py` execute those cases and preserve the shared
schema while recording transfer, eager warm-up, cold/cached compile, CUDA
allocated/reserved peaks, profiler/graph-break, device-copy, allocator, and
kernel-launch data. They synchronize CUDA only at measurement boundaries; the
two-GPU runners use one process per device and record link topology.

## Torch compiler and runtime tuning

`torch_tuning.py` consumes the frozen manifest, the informational native
summary, and the two corrected-runner Torch baseline slices rooted at commit
`821c075b9328e02c3f3e5d16488a44b64ff08c04`. Authoritative timings use an
explicit `time.perf_counter` plus device synchronization loop so Torch does not
receive `Timer.timeit()`'s two hidden statement warmups.
Every authoritative replicate restores the seeded pre-warm-up checkpoint and
executes the real manifest warm-up before timing, matching the native repeat
contract. Repeated `torch.utils.benchmark.Timer` samples are retained in a
separate, exploratory record and never feed an acceptance timing comparison.
The runner records
construction, H2D, cold and cached compilation, one-step latency, batched
throughput, compiler counters, raw samples, buffer addresses, memory growth,
topology, and a Chrome profiler trace. The strict acceptance summary rejects
graph breaks, recompilation after warm-up, host-device transfers, storage
changes, or unbounded device-memory growth. CPU performance evidence covers
crossover, large mixed, and paired-real Bloch cases in both 2-D and 3-D at one
thread and at the physical-core thread count. Each isolated thread slice must
contain all six cases; both slices are required for the CPU performance
subgate. `torch_correctness.py` binds the separate complete-field,
persistent-state, source/auxiliary-state, and physical-observable archives;
supplying its validated index promotes the aggregate scope to
`cpu-performance-and-correctness`. Schema 4 binds each CPU slice to the exact
manifest, runner inputs, solver inputs and ABI, clean candidate commit, host
identity, and byte hashes of its Torch baseline inputs.

Native comparisons remain required, recomputed, and published for all 12
workload/thread cells, but their timing ratios are informational. The blocking
timing checks compare the candidate with the matching same-host Torch baseline:
no individual cell may exceed `1.05x`, and the deterministic one-sided bootstrap
of the 12-cell log-geometric-mean ratio must find no significant regression.
Both reference and candidate raw samples are checked for positivity, replicate
count, reported-summary consistency, and relative MAD before comparison.

The baseline JSON bytes are part of the frozen contract, not merely examples.
They are sanitized public Release assets, not tracked repository files. The
one-thread asset is
[`torch-cpu-baseline-one.json`](https://github.com/ruddyscent/gmes/releases/download/issue-123-torch-cpu-baseline-v3/torch-cpu-baseline-one.json)
(18,281 bytes, SHA-256
`c8eba3c17ccae5ba744a8fbc90b89d72a77dcf0624339cda1deb4d7f594395ed`).
The four-physical-core asset is
[`torch-cpu-baseline-physical.json`](https://github.com/ruddyscent/gmes/releases/download/issue-123-torch-cpu-baseline-v3/torch-cpu-baseline-physical.json)
(18,292 bytes, SHA-256
`b1a3c82a069c2475560468a7b8d0a237db89e857bdce96f8fc812449b5c35602`).
The manifest freezes each Release URL, byte size, and digest; the loader
performs no network access and rejects any URL, schema, size, or content
substitution. Download the two files explicitly, verify their SHA-256 values,
and pass those local paths to `--torch-baseline-slice-artifacts`.

The public copies retain only the data required to reproduce the timing and
allocation checks. They omit hostname, PID values, raw CPU/GPU topology,
compiler-cache metadata, RSS records, profiler traces, local paths, and all
private commitment material. A release-scoped host commitment lets a local
candidate prove equality with the frozen CPU host and upstream PyTorch release
without publishing that identity directly. It excludes the PyTorch local-build
suffix, CUDA runtime, device inventory, and GPU topology because those
device-specific values differ between CPU and CUDA builds and are validated by
the corresponding runtime contracts. It still commits to the platform, Python
version, PyTorch public version, CPU counts, affinity, topology, and normalized
model. Public documentation and artifacts disclose only that safe commitment;
the material required to open it remains protected.
CPU timing adds a separate fail-closed runtime identity in the manifest. Both
candidate slices must use exactly PyTorch `2.13.0+cpu` with no CUDA runtime;
switching to a CUDA-enabled wheel, changing the local build suffix, or reporting
a CUDA runtime invalidates the comparison even when the privacy-preserving host
commitment still matches. The pinned native summary is informational and may
use a different local PyTorch build suffix, but its public PyTorch version must
remain exactly `2.13.0`.
The legacy artifacts remain timing references only: their recurring allocations
are reported, while the revised fixed-temporary and full-field-clone rules apply
to the candidate rather than retroactively to the baseline. Neither asset is
included in Git, an sdist, or a wheel.

CPU RSS acceptance uses 28 consecutive five-step windows in a fresh process.
The first 16 windows are a fixed stabilization phase. The remaining 12 windows
form two non-overlapping six-window evaluation blocks. A run is bounded only
when its upward excursion from the first evaluation sample is at most 1 MiB
and an exact one-sided permutation test does not report a positive-order trend
in both blocks. Linux probes use the `proc-self-statm-preadv-v1` provider,
which keeps `/proc/self/statm` open and reuses one fixed buffer and iovec for
every window after one prevalidation read outside the measured windows.
Downward returns remain in the absolute-envelope diagnostics;
they are not treated as leaks, and unavailable measurements fail closed.
Linux reads `/proc/self/statm`; Darwin uses `proc_pid_rusage` only after its
byte value is validated against a `ps` KiB reference within the same 1 MiB bound.

The dense, compact, and tiled policies execute distinct `aten::masked_scatter_`,
`aten::index_copy_`, and `aten::scatter_` representations. Each forced run also
creates a candidate-bound, uncompiled diagnostic trace that profiles only the
live dispersive dispatcher. The final evaluator reopens those raw Chrome traces
and requires the expected operation count with both alternative operations
absent. It also reconstructs every compile-cache SHA-256 from its exact runtime
tuple preimage and verifies a canonical policy/configuration preimage. The
matrix rejects reused traces, metadata-only representation changes, mismatched
cache preimages, differing target topologies, or failed raw runtime gates.
`auto / fastest-forced` is recomputed from raw timing samples and must be at
most `1.10`.

Compiled CPU `auto` deliberately selects `compact` rather than the static-cost
winner when necessary: both dense `masked_scatter_` and tiled `scatter_` can
materialize a recurring full-field temporary in the supported PyTorch release.
The decision record retains all three cost estimates and names both exclusions.
Forced dense and tiled runs remain available for the CUDA policy matrix and
diagnostic comparison; they are not silently remapped.


The official policy, paired-real, and equivalent-region evidence is one
same-host CUDA float32 family. Every run binds the clean candidate/manifest and
the canonical Linux host/toolchain contract. The policy matrix contains exactly
eight ordered workloads x four ordered policies and 32 exact compiled Chrome
traces. Its forced runs embed another 24 exact, candidate-bound uncompiled
operation traces. List all 56 trace files as bundle payloads; the 24 diagnostic
descriptors are consumed transitively from `policy.json`.
The paired-real suite is one atomic JSON with ordered bloch-2d/bloch-3d cases
and two traces. The region suite is one atomic JSON with ordered
one-region/32-region cases and two traces; it publishes the raw effective
material plan so the final evaluator can verify identical active targets and
material/kernel launch counts while geometry object count increases.

The blocking single-GPU suite revalidates, rather than merely trusts, two
pre-created strict correctness indexes in fixed eager then compiled-graph
order. Both indexes must bind the same clean candidate, manifest, solver ABI,
complete case set, and raw candidate NPZ archives.

The final completion bundle embeds all three correctness indexes as full
artifact descriptors: the CPU scope owns the CPU index, and the single-GPU
scope owns the two CUDA indexes in fixed eager then compiled-graph order.
Bundle closure follows those descriptors into each index and then into every
referenced raw NPZ, so neither an index hash nor a summarized pass flag can
stand in for the CPU, eager, or compiled-graph correctness bytes.

The `cuda-gates` command keeps `--precision float32` as the suite selector, but
the v2 suite contract pins the effective precision per case. Five cases remain
CUDA float32; `single-gpu-3d` alone runs in CUDA float64 because its fixed
native step-100 magnitude exceeds the float32 range. The workload, initializer,
five-step warm-up, 100 timed steps, 15 repetitions, and profiler contract are
unchanged. Every recorded checkpoint must keep all six fields and every
non-plan dynamic checkpoint tensor finite, including nested source auxiliaries
and probe state. Static plan sentinels are not mistaken for dynamic state. The
blocking CUDA memory measure is the net allocation before and after the
steady-state run, after evidence-only checkpoint clones are released. Reported
peak allocation is diagnostic and can include checkpoint and finiteness
validation storage; it is not presented as the solver-only steady-state peak.
The complete eager/compiled-graph correctness matrix remains CUDA float32.

~~~sh
uv run --no-sync python -m benchmarks.torch_tuning \
  --case cuda-gates --device cuda:0 --precision float32 \
  --compile-mode default --policy auto --threads 1 --interop-threads 1 \
  --cuda-correctness-index /tmp/issue-123/correctness-eager.json \
    /tmp/issue-123/correctness-graph.json \
  --cuda-correctness-receipt /trusted/issue-123-runtime-receipts/cuda-eager.json \
    /trusted/issue-123-runtime-receipts/cuda-graph.json \
  --trace-directory /tmp/issue-123/cuda-traces \
  --output /tmp/issue-123/cuda.json --enforce

uv run --no-sync python -m benchmarks.torch_tuning \
  --case policy-gates --device cuda:0 --precision float32 \
  --compile-mode default --policy matrix --threads 1 --interop-threads 1 \
  --descriptor-root /tmp/issue-123 \
  --trace-directory /tmp/issue-123/policy-traces \
  --output /tmp/issue-123/policy.json --enforce

uv run --no-sync python -m benchmarks.torch_tuning \
  --case paired-real-gates --device cuda:0 --precision float32 \
  --compile-mode default --policy auto --threads 1 --interop-threads 1 \
  --trace-directory /tmp/issue-123/paired-real-traces \
  --output /tmp/issue-123/paired-real.json --enforce

uv run --no-sync python -m benchmarks.torch_tuning \
  --case region-invariance-gates --device cuda:0 --precision float32 \
  --compile-mode default --policy auto --threads 1 --interop-threads 1 \
  --trace-directory /tmp/issue-123/region-traces \
  --output /tmp/issue-123/region-invariance.json --enforce
~~~


```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
uv run --no-sync python -m benchmarks.torch_tuning \
  --case cpu-crossover-2d --device cpu --precision float64 \
  --threads 1 --interop-threads 1 --native-summary native-summary.json \
  --torch-baseline-slice-artifacts baseline-one.json baseline-physical.json \
  --trace-directory /tmp/gmes-tuning-traces --output /tmp/gmes-tuning.json

uv run --no-sync python -m benchmarks.torch_tuning \
  --case single-gpu-3d --device cuda:0 --precision float64 \
  --compile-mode matrix --capture-graphs \
  --trace-directory /tmp/gmes-tuning-traces --output /tmp/gmes-cuda.json
```

Use `--policy matrix` for the forced dense/compact/tiled comparison and
`--enforce` in automation. Keep JSON and profiler traces in CI artifacts or
`/tmp`; the files contain the complete environment metadata and can be large.
Run the six-case `cpu-gates` command in separate processes for `--threads 1`
and the affinity-aware physical-core count, then aggregate them. A single case
or thread slice is diagnostic-only and cannot report CPU performance
acceptance:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
uv run --no-sync python -m benchmarks.torch_tuning \
  --case cpu-gates --device cpu --threads 1 --interop-threads 1 \
  --native-summary native-summary.json \
  --torch-baseline-slice-artifacts baseline-one.json baseline-physical.json \
  --output /tmp/cpu-one.json

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
uv run --no-sync python -m benchmarks.torch_tuning \
  --case cpu-gates --device cpu --threads 4 --interop-threads 1 \
  --native-summary native-summary.json \
  --torch-baseline-slice-artifacts baseline-one.json baseline-physical.json \
  --output /tmp/cpu-physical.json

uv run --no-sync python -m benchmarks.torch_tuning \
  --case cpu-gates --native-summary native-summary.json \
  --torch-baseline-slice-artifacts baseline-one.json baseline-physical.json \
  --cpu-slice-artifacts /tmp/cpu-one.json /tmp/cpu-physical.json \
  --allocation-provenance allocation-provenance.json \
  --correctness-evidence-index /tmp/torch-correctness-index.json \
  --correctness-runtime-receipt /trusted/issue-123-runtime-receipts/cpu.json \
  --output /tmp/cpu-acceptance.json --enforce
```

Capture one Torch candidate archive from each frozen native step/0 archive,
then build and revalidate the exact correctness index:

```sh
uv run --no-sync python -m benchmarks.torch_correctness capture \
  --reference /tmp/native/dielectric-1d.npz \
  --output /tmp/torch/dielectric-1d.npz

uv run --no-sync python -m benchmarks.torch_correctness index \
  --references /tmp/native/*.npz --candidates /tmp/torch/*.npz \
  --candidate-evidence /tmp/cpu-one.json \
  --runtime-receipt /tmp/runtime-receipts/cpu.json \
  --descriptor-root /tmp \
  --output /tmp/torch-correctness-index.json

uv run --no-sync python -m benchmarks.torch_correctness validate-index \
  --index /tmp/torch-correctness-index.json \
  --descriptor-root /tmp \
  --candidate-evidence /tmp/cpu-one.json \
  --runtime-receipt /trusted/issue-123-runtime-receipts/cpu.json
```

Correctness index schema 2 requires the ordered union of all 31 manifest
`correctness` cases and all three `physical_checks` cases. It reopens both NPZ
archives, compares complete fields and persistent/source/auxiliary state, and
records the exact archive hashes. Repeat the index for the CPU, CUDA eager, and
CUDA compiled-graph runtime modes. Each index has 34 native-reference and 34
Torch-candidate descriptors. The final completion validator
`_validate_global_correctness_archive_topology` requires one identical ordered
34-reference set across CPU, CUDA eager, and CUDA graph, with every reference
record identical by case, path, SHA-256, size, media type, and payload identity.
CPU, CUDA eager, and CUDA graph must each use a distinct 34-candidate set. All
three candidate sets must be mutually disjoint and disjoint from the shared
references by path, digest, and payload identity. The resulting 204 descriptor
occurrences therefore resolve to exactly 34 shared references plus 102
candidates, for 136 globally unique archives. Do not replace this closure with
summaries or grouped differential projections.

The schema-2 contract is `complete-field-state-and-runtime-receipt-v2`.
Construction embeds the descriptor of the matching canonical runtime receipt;
`validate-index` additionally requires a caller-supplied, byte-identical copy
outside `--descriptor-root`. It repeats every comparison and does not trust an
embedded `"passed": true` or receipt assertion. Every nested archive descriptor
is a canonical POSIX path relative to `--descriptor-root`, plus its exact byte
size, media type, SHA-256, and candidate binding; absolute paths and path
escapes fail closed.

### Candidate-bound differential evidence

Differential schema 5 retains a complete `reference_source` and
`candidate_source` descriptor on every record in addition to the compact
grouped projections. Create both scopes from a clean candidate checkout. The
isolated runner must use the pinned `native-oracle-observer-v6` worktree for
every native reference; running `native_oracle.py capture` in the candidate
checkout is not an equivalent reference. Start with a fresh bundle directory
so the producer's exact source and output closures cannot include stale files:

```sh
GMES_CANDIDATE_CHECKOUT=$PWD
GMES_ISSUE123_BUNDLE=/tmp/gmes-issue123-bundle
mkdir -p \
  "$GMES_ISSUE123_BUNDLE/sources/native/paired" \
  "$GMES_ISSUE123_BUNDLE/sources/native/single" \
  "$GMES_ISSUE123_BUNDLE/sources/torch/paired" \
  "$GMES_ISSUE123_BUNDLE/sources/torch/single"

uv run --no-sync python -m benchmarks.issue123_differential candidate \
  --output "$GMES_ISSUE123_BUNDLE/candidate.json"

for GMES_CASE in bloch-2d bloch-3d upml-bloch cpml-bloch lorentz-bloch \
  dcp-ade-bloch dcp-plrc-bloch dcp-rc-bloch; do
  uv run --no-sync python benchmarks/run_isolated_oracle.py \
    --checkout /tmp/gmes-native-observer \
    --python /tmp/gmes-native-observer/.venv/bin/python \
    --manifest "$GMES_CANDIDATE_CHECKOUT/benchmarks/native_oracle_workloads.json" \
    --case "$GMES_CASE" \
    --output "$GMES_ISSUE123_BUNDLE/sources/native/paired/$GMES_CASE.npz"
done

for GMES_CASE in single-gpu-2d single-gpu-3d; do
  uv run --no-sync python benchmarks/run_isolated_oracle.py \
    --checkout /tmp/gmes-native-observer \
    --python /tmp/gmes-native-observer/.venv/bin/python \
    --manifest "$GMES_CANDIDATE_CHECKOUT/benchmarks/native_oracle_workloads.json" \
    --case "$GMES_CASE" \
    --output "$GMES_ISSUE123_BUNDLE/sources/native/single/$GMES_CASE.npz"
done

for GMES_CASE in bloch-2d bloch-3d upml-bloch cpml-bloch lorentz-bloch \
  dcp-ade-bloch dcp-plrc-bloch dcp-rc-bloch; do
  uv run --no-sync python -m benchmarks.torch_correctness capture \
    --reference "$GMES_ISSUE123_BUNDLE/sources/native/paired/$GMES_CASE.npz" \
    --output "$GMES_ISSUE123_BUNDLE/sources/torch/paired/$GMES_CASE-cpu.npz" \
    --device cpu --precision float64 --graph-mode eager --compile-mode default
  uv run --no-sync python -m benchmarks.torch_correctness capture \
    --reference "$GMES_ISSUE123_BUNDLE/sources/native/paired/$GMES_CASE.npz" \
    --output "$GMES_ISSUE123_BUNDLE/sources/torch/paired/$GMES_CASE-cuda.npz" \
    --device cuda:0 --precision float32 --graph-mode eager --compile-mode default
done

uv run --no-sync python -m benchmarks.torch_correctness capture \
  --reference "$GMES_ISSUE123_BUNDLE/sources/native/single/single-gpu-2d.npz" \
  --output "$GMES_ISSUE123_BUNDLE/sources/torch/single/single-gpu-2d.npz" \
  --device cuda:0 --precision float32 --graph-mode eager --compile-mode default

uv run --no-sync python -m benchmarks.torch_correctness capture \
  --reference "$GMES_ISSUE123_BUNDLE/sources/native/single/single-gpu-3d.npz" \
  --output "$GMES_ISSUE123_BUNDLE/sources/torch/single/single-gpu-3d.npz" \
  --device cuda:0 --precision float64 --graph-mode eager --compile-mode default
```

Build and then independently reopen each raw projection:

```sh
uv run --no-sync python -m benchmarks.issue123_differential build \
  --scope paired-real \
  --references "$GMES_ISSUE123_BUNDLE"/sources/native/paired/*.npz \
  --candidates "$GMES_ISSUE123_BUNDLE"/sources/torch/paired/*.npz \
  --candidate-evidence "$GMES_ISSUE123_BUNDLE/candidate.json" \
  --descriptor-root "$GMES_ISSUE123_BUNDLE" \
  --output-directory "$GMES_ISSUE123_BUNDLE/artifacts/paired-real/raw" \
  --output "$GMES_ISSUE123_BUNDLE/artifacts/paired-real/index.json"

uv run --no-sync python -m benchmarks.issue123_differential build \
  --scope single-gpu-cuda \
  --references "$GMES_ISSUE123_BUNDLE"/sources/native/single/*.npz \
  --candidates "$GMES_ISSUE123_BUNDLE"/sources/torch/single/*.npz \
  --candidate-evidence "$GMES_ISSUE123_BUNDLE/candidate.json" \
  --descriptor-root "$GMES_ISSUE123_BUNDLE" \
  --output-directory "$GMES_ISSUE123_BUNDLE/artifacts/single-gpu/raw" \
  --output "$GMES_ISSUE123_BUNDLE/artifacts/single-gpu/index.json"

for GMES_SCOPE in paired-real single-gpu-cuda; do
  GMES_SCOPE_DIRECTORY=${GMES_SCOPE%-cuda}
  uv run --no-sync python -m benchmarks.issue123_differential validate \
    --scope "$GMES_SCOPE" \
    --candidate-evidence "$GMES_ISSUE123_BUNDLE/candidate.json" \
    --descriptor-root "$GMES_ISSUE123_BUNDLE" \
    --index "$GMES_ISSUE123_BUNDLE/artifacts/$GMES_SCOPE_DIRECTORY/index.json"
done
```

The full schema-5 source topology is fixed. `paired-real` has eight native
sources shared by CPU/CUDA records and 16 Torch sources (eight CPU float64 and
eight CUDA eager float32), producing 16 records. Its source descriptors have
32 occurrences over 24 unique source NPZs; the one group per role adds 32
grouped NPZs, for 64 descriptor occurrences over 56 unique files.
`single-gpu-cuda` has two native and two Torch sources. Across both scopes the
exact 18-record index closure is 16 paired records plus two single-GPU records,
and the complete source closure is exactly ten native plus 18 Torch NPZs. The
32 paired and eight single-GPU grouped projections make 40 compact NPZs.

The builder first validates every complete source archive and its
runtime/provenance contract. It writes the final-step fields and an exact,
case-derived inventory of persistent material state for ordinary records. The
`single-gpu-3d` record instead preserves the fields, physical spectra and
summary, and the same exact persistent inventory in exactly three groups per
role: `[0, 1]`, `[2, 5]`, and `[20, 100]`. Step 0 is an implicit initialized
capture; the frozen manifest `capture_steps` remains the positive list
`[1, 2, 5, 20, 100]`. Together with the one final-step group per role for
`single-gpu-2d`, the single-GPU index has eight grouped NPZs (two for 2-D and
six for 3-D) plus four complete source NPZs, for an exact 12-file closure.
Schema-5 validation byte-for-byte regenerates every canonical grouped NPZ and
checks its complete ZIP member coverage and archive-comment identity. Raw
native and Torch PointSource payloads intentionally use different
representations, so they are not compared as if they were the same numerical
state. Instead, the builder decodes every capture's live Torch targets, model,
parameters, amplitude, and update mode, checks its evaluated waveform at the
Yee half-step, and cross-checks those buffers against the packed nine-word
Torch source record. It also verifies the native source indices, amplitude,
source-cell `eps_inf` and `mu_inf`, and waveform at every capture against the
frozen workload. The two representations are canonicalized independently into
an exact semantic JSON record. A second canonical raw-proof record retains
every capture's native time/indices/values and Torch time/packed values/all ten
live buffers. It records a separate SHA-256 for each role's retained canonical
preimage; the strict validator reconstructs and rehashes both preimages instead
of claiming that the compact bundle contains or can rehash the complete source
archives. Both records are stored byte-for-byte and independently reopened in
every grouped NPZ by the strict validator.

All differential records except `single-gpu-3d` retain the manifest's
elementwise `rtol`/`atol` comparison. For `single-gpu-3d`, captures 0, 1, 2,
and 5 must still pass those strict checks. Captures 20 and 100 use the
case-scoped float64 residual contract because the fixed workload's unstable
mode amplifies ULP-scale association differences and makes a late-step
elementwise tolerance meaningless. For every floating field, physical
observable, and persistent array, the builder requires
`||candidate-reference||inf / max(||reference||inf, 2e-12)` and
`||candidate-reference||2 / max(||reference||2, 2e-12*sqrt(N))` to be at most
`1e-6`. The late group retains both steps and records both maxima
for independent recomputation. It also recomputes the maximum absolute native
step-100 field value and requires it to exceed the float32 finite range, binding
the reviewed float64 exception to the evidence itself. An all-zero reference
array, every integer array, topology, shape, dtype, and finiteness remain
exact/fail-closed; the limit is pinned and is never calibrated from the
candidate.

Each NPZ descriptor has exactly the five bundle-relative
keys `path`, `sha256`, `size_bytes`, `media_type`, and `candidate_evidence`.
Validation derives the workload/device dtype and comparison contract, reopens
the canonical source bytes, and recomputes every array metric; recorded
comparison values, metrics, and pass flags cannot weaken the gate.

For a nonzero allocation trace, the sidecar is necessarily a second-stage
review. First create and preserve the slice JSON, Chrome trace, and generated
source without a provenance input. Then review those exact bytes, write the
sidecar with their hashes and selector, and pass it while aggregating the saved
slices. Do not rerun the slice to apply the sidecar: a new profiler trace has a
different digest. Aggregation reopens and hashes the saved trace, reconstructs
all trace-derived allocation metrics, verifies the generated source and compile
cache key, and replaces only the originally failed allocation decision. Any
other runtime failure remains blocking.

Without a correctness index the aggregate reports
`acceptance_scope: "cpu-performance-only"`. With a fully revalidated index it
reports `acceptance_scope: "cpu-performance-and-correctness"` and sets
`cpu_correctness_satisfied`; final issue completion remains the responsibility
of the all-scope aggregator below.

Zero-allocation CPU traces need no provenance record. A nonzero trace passes
only when every fixed-shape temporary is accounted for in a reviewed record
selected by workload, device, precision, compile mode, execution policy, and
thread count. The JSON document has schema 1 and kind
`torch-cpu-allocation-provenance`; each record binds the Chrome-trace SHA-256,
compile-cache key, profile-step count, exact allocation-size histogram,
per-step counts and generated operation, and at least one generated-source path
and SHA-256. Every nonzero reviewed record also names at least one public
`https://github.com/pytorch/pytorch/issues/<number>` URL; the current indexed
workspace allocation is tracked by pytorch/pytorch#195330. The trace must have
zero final live growth and bounded repeated RSS. Every allocation matching a
live field/domain buffer fails. Direct, registered, nonpersistent
`state._boundary_*` scratch is inventoried and bound into the sidecar, but it
is preallocated before profiling and never authorizes a dynamic allocation.
The cached paired-real boundary plan stores its zero-dimensional phase views in
that reserved fixed workspace, avoiding per-step scalar tensor allocation. The
`paired_real_scratch_bytes` key is retained as a diagnostic compatibility name;
it inventories reserved boundary storage rather than recurring scratch writes.
Generated-source review must still record zero full-field/domain clone events.

```json
{
  "schema_version": 1,
  "kind": "torch-cpu-allocation-provenance",
  "method": "reviewed-fixed-temporary-provenance-v1",
  "records": [
    {
      "workload": "cpu-crossover-2d",
      "device": "cpu",
      "precision": "float64",
      "compile_mode": "default",
      "execution_policy": "auto",
      "threads": 1,
      "method": "reviewed-fixed-temporary-provenance-v1",
      "reviewed": true,
      "trace_sha256": "<64 lowercase hex characters>",
      "compile_cache_key": "<reported compile cache key>",
      "profile_steps": 5,
      "allocation_size_histogram": {"792": 5},
      "fixed_boundary_buffer_sizes_bytes": {},
      "full_field_or_domain_clone_events": 0,
      "upstream_issue_urls": [
        "https://github.com/pytorch/pytorch/issues/195330"
      ],
      "allocations": [
        {
          "size_bytes": 792,
          "events_per_step": 1,
          "classification": "allowed-plan-bounded-temporary",
          "generated_operation": "allocate indexed-update values buffer"
        }
      ],
      "generated_sources": [
        {"path": "/tmp/torchinductor/generated.cpp", "sha256": "<SHA-256>"}
      ]
    }
  ]
}
```

### Inductor allocation reproducers

Run the reproducers with Python optimization enabled to confirm that their
checks do not depend on removable Python `assert` statements. Both scripts
raise an explicit exception and return nonzero when requested equivalence,
compiler, allocation, trace, or provenance evidence is missing or differs
from the affected behavior.

#### Composed mutation

`torch_inductor_composed_mutation.py` is a standalone, Torch-only reproducer
for the full-field CPU allocation observed when an otherwise allocation-free
slice mutation is composed with a compact indexed update. It compares the
compiled result with eager sequential execution and reports allocation traces
and timings for the isolated and composed graphs. The asserted baseline
requires exact repeated equivalence, no steady-state compiler-counter changes,
no allocations outside the measured scopes, one full-field allocation per
profiled composed call, and no full-field allocation for the isolated compact
update or public `torch.as_strided` formulation:

```sh
uv run --no-sync python -O benchmarks/torch_inductor_composed_mutation.py \
  --assert-affected --trace-directory /tmp/gmes-inductor-repro
```

`--force-reinplace` replaces only Inductor's private generalized-scatter
profitability check and is useful for confirming the responsible pass. It is a
diagnostic, not a GMES runtime workaround. Run it in a separate process so it
uses a fresh Inductor cache:

```sh
uv run --no-sync python -O benchmarks/torch_inductor_composed_mutation.py \
  --force-reinplace --assert-affected \
  --trace-directory /tmp/gmes-inductor-repro-forced
```

The selected GMES CPU representation follows the public path demonstrated by
this reproducer: non-overlapping interior-field and boundary-plane mutations
use storage-sharing `torch.as_strided` views. CUDA, distributed, and eager
execution retain their existing slice representation.

#### While-loop state

`torch_inductor_while_loop_allocation.py` is a standalone, Torch-only
reproducer for recurring allocations in a compiled CPU `torch.while_loop`. It
measures two full-graph variants independently: the original multiple tensor
carries, and a GMES-style single carry backed by one exact-size, caller-owned
workspace containing field, history, and per-cell completion-code state. It
verifies exact repeated eager equivalence, stable workspace storage, no warm
steady-state recompilation, and writes a raw Chrome allocation trace for each
variant:

```sh
uv run --no-sync python -O \
  benchmarks/torch_inductor_while_loop_allocation.py \
  --assert-affected --trace-directory /tmp/gmes-while-loop-repro
```

Explicit `--cache-directory` and `--trace-directory` paths must be new or
empty; omitting either uses temporary directories. The assertion mode fails
closed if any expected predicate or carry allocation disappears, equivalence
or workspace checks fail, a graph break or new steady-state graph appears, or
either trace is absent. Only after collecting and freezing that functional
evidence, the script also compiles a minimal packed carry whose loop body
mutates the caller-owned input in place. Assertion mode requires the public
higher-order-op tracer to reject that attempted allocation-free formulation
specifically because its body mutates an input; the report records only the
normalized exception type and one-line reason. The functional packed variant
shows that coalescing loop state removes the separate field/history/counter
carries, but Inductor still creates the one-byte condition result and packed
carry storage on every iteration even when the initial workspace belongs to
the caller. The selected CPU DM2 representation uses this bounded single-carry
topology; the functional multi-carry implementation remains available to
non-CPU execution.

The long-running `physical_checks` cases archive field energy, boundary
energy (reflection/transmission observables), maximum amplitude, finiteness,
component spectra, complete fields, and DM2 state at fixed steps. The
Ziolkowski DM2 case fixes captures at steps 100 and 500. Its float32 DM2
comparison uses `rtol=6e-4` and `atol=3e-6`, calibrated to the observed
long-horizon float32 Maxwell coupling; this model-scoped tolerance applies to
the workload's floating field, transformed-state, and physical arrays. The
float64 DM2 tolerance remains `rtol=2e-10` and `atol=2e-12`. This does not
remove any evidence: complete fields, physical observables and reconstructed
density state, exact status/iteration arrays, and every archived
persistent-state descriptor remain required and fail closed. Large NPZ, JSON,
and profiler outputs belong in CI/run artifacts or `/tmp`; do not add them to
the repository, source distribution, or wheel.

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

The fixed parent runner completes an isolated one-GPU child before it starts a
separate two-rank `torch.distributed.run` child. It restores the same captured
checkpoint for every replicate and keeps construction, graph capture,
steady-state timing, rank-local memory/storage/halo evidence, raw child
JSON/stdout/stderr, and profiler evidence separate. The strong mixed case uses the frozen
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
uv run --no-sync python -m benchmarks.torch_two_gpu \
  --case strong-mixed --warmup 5 --steps 100 --repeats 15 \
  --profile-steps 10 --enforce \
  --descriptor-root /tmp/gmes-issue123-bundle \
  --output /tmp/gmes-issue123-bundle/artifacts/two-gpu/performance/strong-mixed/result.json \
  --trace-directory /tmp/gmes-issue123-bundle/artifacts/two-gpu/performance/strong-mixed/traces

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
uv run --no-sync python -m benchmarks.torch_two_gpu \
  --case weak-mixed --warmup 5 --steps 100 --repeats 15 \
  --profile-steps 10 --enforce \
  --descriptor-root /tmp/gmes-issue123-bundle \
  --output /tmp/gmes-issue123-bundle/artifacts/two-gpu/performance/weak-mixed/result.json \
  --trace-directory /tmp/gmes-issue123-bundle/artifacts/two-gpu/performance/weak-mixed/traces
```

Repeat with `--case strong-homogeneous` and
`--case strong-imbalanced`. JSON output records every timing sample,
one-versus-two throughput, the 1.6 strong or 0.8 weak gate, rank-local memory,
halo bytes, fixed-address validation, exact decomposition, PyTorch/CUDA/NCCL
versions, both GPU models, peer-access status, and `nvidia-smi topo -m`.
The parent binds each completed child JSON and its exact raw stdout/stderr by
bundle-relative typed descriptors. Per-rank Chrome traces remain in the
requested trace directory and separate
pack/launch, boundary wait, unpack, NCCL device time, compute overlap, and
exposed communication. Generated JSON and traces belong in `/tmp` or CI
artifacts, not in the repository.

The frozen native reference/candidate commands earlier in this document remain
the correctness-oracle handoff. The two-GPU runner is the candidate performance
handoff: it consumes the same mixed-coverage recipe and fixed GPU sizes without
importing or launching the native solver in either CUDA rank.

Run the complete deterministic field/material/source/restart matrix with:

```sh
mkdir -p /tmp/gmes-issue123-bundle

OMP_NUM_THREADS=1 uv run --no-sync torchrun --standalone \
  --nproc-per-node=2 --module benchmarks.torch_two_gpu_correctness \
  --long-steps 1000 --enforce \
  --descriptor-root /tmp/gmes-issue123-bundle \
  --output /tmp/gmes-issue123-bundle/artifacts/two-gpu/correctness/eager.json

OMP_NUM_THREADS=1 uv run --no-sync torchrun --standalone \
  --nproc-per-node=2 --module benchmarks.torch_two_gpu_correctness \
  --capture-graphs --long-steps 1000 --enforce \
  --descriptor-root /tmp/gmes-issue123-bundle \
  --output /tmp/gmes-issue123-bundle/artifacts/two-gpu/correctness/graph.json
```

Each schema-3 JSON has a sibling `eager-raw` or `graph-raw` directory. Every
case NPZ contains complete distributed/serial fields at steps 1, 2, 5, 20, and
100; checkpoint expected/replay/serial fields; and raw initial/final rank and
serial address arrays. The long-stability NPZ contains initial,
distributed-final, and serial-final fields. Exact five-key bundle-relative NPZ
descriptors bind those bytes to the clean candidate. The completion evaluator
recomputes all errors, energies, address digests/stability, dtype, shape, and
array closure instead of trusting the JSON summaries.

The canonical capture wrapper runs each exact two-rank command, preserves raw
stdout/stderr and its exit code, and records only the clean candidate, canonical
host contract, and structured expected failure reason (never a full environment
dump). The first three probes must emit their specific collective error and exit
zero; `rank-failure` must propagate the injected rank-local error through
`torchrun` and exit nonzero:

```sh
uv run --no-sync python -m benchmarks.two_gpu_failure_evidence strict-peer \
  --descriptor-root /tmp/gmes-issue123-bundle \
  --output-directory /tmp/gmes-issue123-bundle/artifacts/two-gpu/failures
uv run --no-sync python -m benchmarks.two_gpu_failure_evidence dtype-mismatch \
  --descriptor-root /tmp/gmes-issue123-bundle \
  --output-directory /tmp/gmes-issue123-bundle/artifacts/two-gpu/failures
uv run --no-sync python -m benchmarks.two_gpu_failure_evidence checkpoint-mismatch \
  --descriptor-root /tmp/gmes-issue123-bundle \
  --output-directory /tmp/gmes-issue123-bundle/artifacts/two-gpu/failures
uv run --no-sync python -m benchmarks.two_gpu_failure_evidence rank-failure \
  --descriptor-root /tmp/gmes-issue123-bundle \
  --output-directory /tmp/gmes-issue123-bundle/artifacts/two-gpu/failures
```

## Issue #123 final evidence aggregation

`issue123_completion.py` consumes one canonical index rooted at the clean
candidate commit. The top-level `candidate_evidence` has exactly
`candidate_git_commit`, empty `candidate_git_status`, and the raw manifest
`manifest_sha256`. Every referenced file uses the exact descriptor
`{path, sha256, size_bytes, media_type, candidate_evidence}`; paths are
canonical bundle-relative POSIX paths. Missing fields, duplicate JSON keys,
non-finite numbers, changed bytes, or cross-candidate bindings fail closed.

The top-level `artifacts` object has exactly six scopes:

- `cpu`: the correctness-bound aggregate, pinned native summary, both source
  slices, correctness index, and 12 Chrome traces;
- `policy_paired_real`: the CUDA float32 8x4 policy matrix, compiled and raw
  operation traces, paired-real differential, atomic tuning, and region
  invariance evidence;
- `single_gpu`: all six CUDA gates, the two-case differential index and its
  eight grouped NPZ payloads, and six traces;
- `two_gpu`: four scaling results, eager/graph schema-3 raw correctness
  matrices, four collective-failure wrappers, and eight rank traces;
- `macos`: the schema-2 package/runtime index and raw Actions ZIP containing
  the sdist, CPython 3.14 arm64 wheel, and all 12 stdout/stderr logs;
- `operations`: the schema-2 descriptor-only index over the exact 22 GitHub
  response roles described below, including issues #123 and #115 with all
  comments, PR #167 comments and closing references, the signed candidate,
  CI/CodeQL, reviews/threads, and the immutable technical publication.

Single-GPU pass flags are not inputs. Every CUDA case reopens its exact trace
and recomputes raw timing/MAD, kernel launches, zero H2D/D2H events, and
bounded memory. The evaluator compares those samples with the best
one/physical-thread CPU slice, treats the two tiny crossover cases as
published diagnostics, blocks unless both above-crossover CPU-large cases are
faster on CUDA, and recomputes the large 3-D `>=2x` gate from the exact pinned
native summary's physical `cpu-large-3d` samples.

The evaluator reopens every exact byte, recomputes CPU statistics, NPZ
comparisons, profiler operations, two-rank reductions, package/runtime results,
and GitHub operational requirements. Linux CPU, policy, single-GPU, and
two-GPU evidence share one common host identity; CUDA scopes additionally
share the exact runtime and GPU inventory. macOS is the only separate host.
The operations scope cross-binds the macOS archive to the exact successful CI
job. Passing all six scopes establishes offline structural validity only. An
offline evaluation always emits `issue_completion_satisfied=false` and
`final_acceptance=false`; only the completion-level same-process live path
described below can make either field true.

Download the pull-request macOS artifact and assemble its schema-2 index:

~~~sh
uv run --no-sync python -m benchmarks.macos_ci_evidence assemble \
  --runtime-index /tmp/issue-123-macos/runtime-index.json \
  --actions-archive /tmp/issue-123-macos/actions-archive.zip \
  --repository ruddyscent/gmes --ci-run-id <CI> \
  --output /tmp/issue-123-macos/index.json \
  --scope-output /tmp/issue-123-macos/scope.json
~~~

### Operations and publication closure

The final-SHA publication and release-dependent operations steps in this
section are the six-item chain governed by the
[Recommendation A OWNER amendment](https://github.com/ruddyscent/gmes/issues/123#issuecomment-5523144396) and deferred to open
[#169](https://github.com/ruddyscent/gmes/issues/169).
They are deferred, unperformed, unsatisfied, still required, and owned by
that open issue; none is a prerequisite for #123 technical closure. Every
retained non-deferred #123 gate remains mandatory.

Publish the technical evidence before operations capture under the exact
lightweight, non-version tag
`issue-123-technical-evidence-<FINAL_SHA>`. Create the new tag through the
release with `--target <FINAL_SHA>` so both `target_commitish` and the tag's Git
commit object name the candidate. Use `--latest=false`. Immutable releases must
be enabled before this future release is created; schema 2 rejects a mutable,
draft, prerelease, annotated-tag, bot-authored, or wrong-target release.
The repository's immutable-release policy is currently disabled, so enabling
it is an explicit operator prerequisite rather than an action performed by the
evaluator.

The immutable release has exactly four OWNER-uploaded assets:

- `issue-123-public-technical-evidence.zip`, containing the five
  pre-operations public technical projections, event-complete
  privacy-normalized traces, and public correctness commitments, but no private
  correctness arrays;
- `issue-123-technical-summary.json`;
- `issue-115-raw-timing.json`, containing actual fixed-workload raw
  `torch.utils.benchmark` timing evidence;
- `issue-115-event-level-profiler.json`, containing actual fixed-workload
  event-level profiler evidence.

Every asset must be `uploaded`, have a unique positive ID, name, and SHA-256,
and expose the canonical GitHub API and browser-download URLs. The paginated
asset endpoint and the release's embedded asset ledger must agree exactly.
Their sizes and GitHub `sha256:` digests are copied into the structured OWNER
comments below. The public technical ZIP intentionally excludes the private
correctness arrays; those remain in the local evaluator bundle while their
commitments are public. The release must not contain the operations scope or
the final six-scope bundle, because either would make operations publication
circular.

For the deferred release-dependent #123 record, post one OWNER comment,
copying the following lines without the documentation fence, indentation,
extra text, blank fields, or reordering:

~~~text
GMES_ISSUE_123_FINAL_CONTRACT_AMENDMENT_V2
FINAL_SHA=<FINAL_SHA>
PR=167
TARGET_ISSUE=123
TECHNICAL_RELEASE_URL=https://github.com/ruddyscent/gmes/releases/tag/issue-123-technical-evidence-<FINAL_SHA>
BASELINE_V3_ROOT_COMMIT=821c075b9328e02c3f3e5d16488a44b64ff08c04
BASELINE_V3_ONE_URL=https://github.com/ruddyscent/gmes/releases/download/issue-123-torch-cpu-baseline-v3/torch-cpu-baseline-one.json
BASELINE_V3_ONE_SIZE_BYTES=18281
BASELINE_V3_ONE_SHA256=c8eba3c17ccae5ba744a8fbc90b89d72a77dcf0624339cda1deb4d7f594395ed
BASELINE_V3_PHYSICAL_URL=https://github.com/ruddyscent/gmes/releases/download/issue-123-torch-cpu-baseline-v3/torch-cpu-baseline-physical.json
BASELINE_V3_PHYSICAL_SIZE_BYTES=18292
BASELINE_V3_PHYSICAL_SHA256=b1a3c82a069c2475560468a7b8d0a237db89e857bdce96f8fc812449b5c35602
BASELINE_V3_HOSTNAME=redacted
BASELINE_V3_HOST_IDENTITY_SCHEMA=torch-cpu-host-identity-v2
BASELINE_V3_HOST_COMMITMENT_SHA256=f7b3b1b0eb13531682ea0381698c60aa9a97c7a3a0dfffc5344b828772f67a56
BASELINE_V3_DISPOSITION=authoritative-published-privacy-sanitized
SUPERSEDES_BASELINE_ISSUE_COMMENT=<BASELINE_CLOSURE_COMMENT_ID>
SUPERSEDES_DM2_ISSUE_COMMENT=<DM2_ISSUE_AMENDMENT_COMMENT_ID>
SUPERSEDES_DM2_PR_COMMENT=<DM2_PR_INSIGHT_COMMENT_ID>
SUPERSEDES_SINGLE_GPU_ISSUE_COMMENT=<SINGLE_GPU_AMENDMENT_COMMENT_ID>
PRIOR_CONTRACT_DISPOSITION=superseded-by-this-amendment
SOLVER_ABI=torch-fdtd-regions-v15
EXECUTION_REPRESENTATION=external-no-inner-cudagraph-regions+dm2-raw-fixed-masked-v1
DIFFERENTIAL_SCHEMA_VERSION=5
DIFFERENTIAL_EARLY_STEPS=0,1,2,5
DIFFERENTIAL_EARLY_CONTRACT=manifest-strict-elementwise
DIFFERENTIAL_LATE_STEPS=20,100
DIFFERENTIAL_LATE_CONTRACT=normalized-linf-l2-at-most-1e-6
SINGLE_GPU_3D_CASE=single-gpu-3d
SINGLE_GPU_3D_PRECISION=float64
SINGLE_GPU_3D_LATE_STEPS=20,100
SINGLE_GPU_3D_LATE_RESIDUAL_CONTRACT=normalized-linf-l2-at-most-1e-6
SINGLE_GPU_3D_RESIDUAL_DENOMINATOR_FLOOR=2e-12
SINGLE_GPU_3D_L2_DENOMINATOR_SCALE=sqrt(N)
SINGLE_GPU_3D_ZERO_REFERENCE_CONTRACT=exact
PUBLIC_TRACE_DISPOSITION=published-event-complete-privacy-normalized
CORRECTNESS_ARRAY_DISPOSITION=private
CORRECTNESS_COMMITMENT_DISPOSITION=published-in-technical-evidence
~~~

For the deferred #169 chain, after issue #115 has
`state_reason=completed`, no unchecked boxes, and both exact runtime/profiler
checklist lines checked, post its distinct OWNER handoff comment. Substitute
values from the captured release ledger:

~~~text
GMES_ISSUE_115_FINAL_RUNTIME_HANDOFF_V2
FINAL_SHA=<FINAL_SHA>
PR=167
TARGET_ISSUE=123
HANDOFF_ISSUE=115
TECHNICAL_RELEASE_URL=https://github.com/ruddyscent/gmes/releases/tag/issue-123-technical-evidence-<FINAL_SHA>
RAW_TIMING_CONTRACT=torch-utils-benchmark-fixed-workloads
RAW_TIMING_ASSET_NAME=issue-115-raw-timing.json
RAW_TIMING_ASSET_URL=https://github.com/ruddyscent/gmes/releases/download/issue-123-technical-evidence-<FINAL_SHA>/issue-115-raw-timing.json
RAW_TIMING_ASSET_SIZE_BYTES=<RAW_TIMING_SIZE>
RAW_TIMING_ASSET_SHA256=<RAW_TIMING_SHA256>
EVENT_PROFILER_CONTRACT=event-level-profiler-fixed-workloads
EVENT_PROFILER_ASSET_NAME=issue-115-event-level-profiler.json
EVENT_PROFILER_ASSET_URL=https://github.com/ruddyscent/gmes/releases/download/issue-123-technical-evidence-<FINAL_SHA>/issue-115-event-level-profiler.json
EVENT_PROFILER_ASSET_SIZE_BYTES=<EVENT_PROFILER_SIZE>
EVENT_PROFILER_ASSET_SHA256=<EVENT_PROFILER_SHA256>
HANDOFF_DISPOSITION=complete
~~~

For the deferred #169 chain, post the independent PR #167 OWNER insight only
after the exact CI and CodeQL runs have passed. It does not reuse or infer
fields from the issue record:

~~~text
GMES_PR_167_FINAL_CANDIDATE_INSIGHT_V2
FINAL_SHA=<FINAL_SHA>
PR=167
TARGET_ISSUE=123
FINAL_COMMIT_URL=https://github.com/ruddyscent/gmes/commit/<FINAL_SHA>
FINAL_COMMIT_VERIFICATION=verified:valid
CI_RUN_URL=https://github.com/ruddyscent/gmes/actions/runs/<CI>
CODEQL_RUN_URL=https://github.com/ruddyscent/gmes/actions/runs/<CODEQL>
TEST_SUMMARY=required-ci-and-regression-tests-pass
EVIDENCE_SUMMARY=five-technical-scopes-pass-private-arrays-commitment-published
TECHNICAL_RELEASE_URL=https://github.com/ruddyscent/gmes/releases/tag/issue-123-technical-evidence-<FINAL_SHA>
TECHNICAL_EVIDENCE_ASSET_NAME=issue-123-public-technical-evidence.zip
TECHNICAL_EVIDENCE_ASSET_URL=https://github.com/ruddyscent/gmes/releases/download/issue-123-technical-evidence-<FINAL_SHA>/issue-123-public-technical-evidence.zip
TECHNICAL_EVIDENCE_ASSET_SIZE_BYTES=<TECHNICAL_EVIDENCE_SIZE>
TECHNICAL_EVIDENCE_ASSET_SHA256=<TECHNICAL_EVIDENCE_SHA256>
TECHNICAL_SUMMARY_ASSET_NAME=issue-123-technical-summary.json
TECHNICAL_SUMMARY_ASSET_URL=https://github.com/ruddyscent/gmes/releases/download/issue-123-technical-evidence-<FINAL_SHA>/issue-123-technical-summary.json
TECHNICAL_SUMMARY_ASSET_SIZE_BYTES=<TECHNICAL_SUMMARY_SIZE>
TECHNICAL_SUMMARY_ASSET_SHA256=<TECHNICAL_SUMMARY_SHA256>
~~~

The four superseded-comment fields are semantic roles, not labels alone.
Operations resolves each typed placeholder to one exact captured
issue-comment or pull-request-comment stream record, canonical API and HTML
URL, repository-OWNER association, timestamp pair, role, and body
digest/content contract. The private capture retains those exact identifiers;
public documentation and publication assets do not reproduce account or
comment identities.

Chronology is fail-closed. Every GitHub object exposing both timestamps must
have `created_at <= updated_at`. The deferred technical release must eventually
precede the release-dependent #115 handoff, structured #123 record, and PR
#167 insight. Issue #115 must satisfy issue creation <= closure <= issue
`updated_at` (the completed checklist state) <= handoff comment; retained local
chronology checks remain mandatory. Candidate commit `verified_at`, release
publication, both selected workflow-run completions, and every selected CI and
CodeQL job completion must precede the PR insight.

Production binding readiness is currently fail-closed. Production
evaluator-binding authority is intentionally deferred to
[#169](https://github.com/ruddyscent/gmes/issues/169). That follow-up owns the
six deferred items: (1) production-bound final-SHA generation of the four
public assets; (2) actual-public-byte schema, cardinality, commitment, digest,
recursive-privacy, Unicode-collision, and exact-byte validation; (3) the
final-SHA immutable release, four OWNER uploads, and byte-for-byte read-back;
(4) release link/tag/ID/URL/size/hash fields in the #115 handoff, structured
#123 record, and PR #167 insight; (5) release-dependent O0/B0/ack/O1/B1,
technical/live receipt, and amended/final aggregate chronology; and (6)
production publication, cutover, a nonempty production registry, and the
production-authority end state. These six items are deferred, unperformed,
unsatisfied, still required, and owned by open #169; none is a #123 closure
prerequisite.

These deferrals change no runtime authority.
`CODE_OWNED_LITERAL_TARGET_BINDINGS` remains empty, the production literal
binding registry remains empty, and the current production policy contains
targets with no code-owned evaluator binding. Publication preparation therefore
remains unavailable for that production target closure, and completion live
verification cannot set `final_acceptance` or `issue_completion_satisfied`
through this path. Caller roles, selectors, paths, descriptors, digests,
selected values, fixtures, and policy iteration are never substitutes for
OWNER authority.

Issue #123 may accept and close for its technical work only after every
retained, non-deferred performance, correctness, evidence, operations, privacy,
security, CI, CodeQL, review, and clean-candidate gate passes. Its closure
record must state that all six items above remain deferred, unperformed,
unsatisfied, and still required by open #169, and that no final-SHA public
release, release-dependent receipt, production publication, cutover,
production readiness, or production authority was established. That technical
acceptance remains distinct from the deferred production-publication chain and
does not claim production readiness. Production evaluator authority remains
deferred, and production publication and cutover remain blocked on #169. The
commands below document the executable fail-closed interface; they do not
claim present production readiness.

Once #169 discharges the deferred chain with exact-byte OWNER-adopted
policy/profile bytes and its required tests, the adapter may become
production-authoritative without changing the public
schemas or cardinalities. It accepts one canonical private source specification
and derives the five projection scopes from their exact completion-index
descriptors. A source record's role and selector fields are redundant
assertions: code owns the target-to-evaluator-role map, rejects unmapped targets,
and never discovers authority by scope, path, digest, or selected value. Prepare
the exact four public assets and the protected binding sidecar before the
release. The sidecar output directory must already exist with mode `0700`; its
file is mode `0600` and must remain outside every public asset and bundle
directory:

~~~sh
mkdir -m 0700 /tmp/issue-123-private-authority

uv run --no-sync python -m benchmarks.issue123_publication prepare \
  --source-spec /trusted/issue-123-publication-source-spec.json \
  --completion-index /tmp/issue-123-projection-source/completion-index.json \
  --policy /trusted/issue-123-publication-policy.json \
  --policy-sha256 <CALLER_OWNED_POLICY_SHA256> \
  --runtime-receipt cpu=/trusted/issue-123-runtime-receipts/cpu.json \
  --runtime-receipt cuda-eager=/trusted/issue-123-runtime-receipts/cuda-eager.json \
  --runtime-receipt cuda-graph=/trusted/issue-123-runtime-receipts/cuda-graph.json \
  --runtime-receipt single-gpu-2d=/trusted/issue-123-runtime-receipts/single-gpu-2d.json \
  --runtime-receipt single-gpu-3d=/trusted/issue-123-runtime-receipts/single-gpu-3d.json \
  --asset-output-directory /tmp/issue-123-publication/assets \
  --private-openings-output /tmp/issue-123-private-authority/publication-openings.json
~~~

Successful stdout is exactly `issue123-publication-prepare-ok`; it contains no
resolved path. Salts, openings, raw arrays, private paths, host/device
identities, and source identities exist only in protected inputs or memory.
Public v1 bytes contain safe commitments and descriptors only.

The documented `main(argv)` boundary and the module process entry use the same
fixed, path-free success and failure tokens. Expected publication path and OS
failures reach direct library callers only as a fixed typed `PublicationError`
without a path-bearing exception chain.

After uploading those four bytes and independently capturing the immutable
release identity and external asset ledger, finalize the private publication
receipt:

~~~sh
uv run --no-sync python -m benchmarks.issue123_publication finalize \
  --asset-directory /tmp/issue-123-publication/assets \
  --release-capture /trusted/issue-123-release-capture.json \
  --release-identity /trusted/issue-123-release-identity.json \
  --policy /trusted/issue-123-publication-policy.json \
  --policy-sha256 <CALLER_OWNED_POLICY_SHA256> \
  --receipt-output /tmp/issue-123-private-authority/publication-receipt.json
~~~

Successful stdout is exactly `issue123-publication-finalize-ok`. The release
identity and external byte ledger are caller-owned values, not values recovered
from the receipt. Capture operations only after that receipt, the immutable
release, and all three final semantic-role comments exist; PR #167 must be
current with `master`, required CI and CodeQL must be complete, and all review
requests and conversations must be clear:

~~~sh
uv run --no-sync python -m benchmarks.issue123_operations capture \
  --repository ruddyscent/gmes --pull-request 167 \
  --ci-run-id <CI> --codeql-run-id <CODEQL> \
  --technical-release-tag issue-123-technical-evidence-<FINAL_SHA> \
  --publication-receipt /tmp/issue-123-private-authority/publication-receipt.json \
  --manifest benchmarks/native_oracle_workloads.json \
  --output-directory /tmp/issue-123-operations-o0
~~~

`capture` sends each fixed `gh api --hostname github.com` request directly. If
an API request fails because its credential state needs repair, repair the
intended credential state outside this noninteractive command and rerun
`capture` with a nonexistent output directory. Both production modes pin every
internal API request to `--hostname github.com`.

The schema-v2 producer preserves the exact response roles, in canonical query
order: `technical_release`, `technical_release_assets`,
`technical_release_tag`, `issue_123`, `issue_123_comments`, `issue_115`,
`issue_115_comments`, `pull_request`, `pull_request_comments`,
`candidate_commit`, `base_compare`, `ci_run`, `ci_jobs`, `codeql_run`,
`codeql_jobs`, `codeql_analyses`, `codeql_alerts`, `ruleset`, `check_runs`,
`reviews`, `requested_reviewers`, and `review_threads`. Every role retains its
raw canonical response and a page ledger containing ordinal, HTTP status,
canonical page-body SHA-256 and size, item count, and the exact has-next/next
relationship. The only retained response-header fields are `content-type`,
`etag`, `last-modified`, canonicalized `link`,
`x-github-api-version-selected`, and `x-github-media-type`; GraphQL records a
null selected-version header when GitHub omits it. Authorization, cookie,
OAuth-scope, token, and secret material is neither captured nor emitted.

REST Link URLs must use the exact named repository route or GitHub's numeric
`/repositories/<ID>/...` equivalent, preserve the closed request-filter set,
and identify the next page for every intermediate page. GraphQL must identify
the next cursor for every intermediate page. The terminal page must have no
next relation, and any declared REST last page must equal the ledger's terminal
ordinal. REST review count is independently matched to GraphQL
`reviews.totalCount`; deleting the terminal reviews, CodeQL analyses, or CodeQL
alerts page therefore fails.

Ordinary `evaluate_operations` and a schema-v2 operations index with its
embedded public schema-v1 publication receipt are structural checks only:
their result always has `final_acceptance=false`. They cannot be replayed to
authorize final acceptance. The non-circular recapture protocol has one stable
fixed point and this exact chronology:

`O0/B0 reopen -> authorized two-line acknowledgment -> O1 recapture -> B1 reopen -> offline evaluate -> live verify`

O0 is the operations capture above while the issue has the exact two authorized
checklist markers still unchecked. Create a six-scope B0 bundle whose first
five artifact descriptors and ordered five runtime receipts are the protected
frozen inventory, and whose operations scope contains O0. Reopen B0 as a
distinct copy and authenticate the exact unchecked issue response, both bundle
inventories, and the protected publication openings:

~~~sh
uv run --no-sync python -m benchmarks.issue123_completion assemble \
  --specification /tmp/gmes-issue123-b0-spec.json \
  --bundle /tmp/gmes-issue123-b0-source

cp -a /tmp/gmes-issue123-b0-source /tmp/gmes-issue123-b0-reopened

uv run --no-sync python -m benchmarks.issue123_completion record-reopen \
  --source-index /tmp/gmes-issue123-b0-source/completion-index.json \
  --reopened-index /tmp/gmes-issue123-b0-reopened/completion-index.json \
  --stage pre-acknowledgment \
  --private-openings /tmp/issue-123-private-authority/publication-openings.json \
  --pre-ack-response /tmp/issue-123-operations-o0/raw/issue_123.json \
  --output /tmp/issue-123-private-authority/b0-reopen-receipt.json
~~~

Only then may the authorized acknowledgment change the two designated checklist
markers from unchecked to checked. The authenticated transition commits the
complete canonical O0/O1 response projection: it neutralizes only those two
exact marker tokens and the separately authenticated top-level `updated_at`;
nested timestamps, every other field, all other body text, ordering, and line
endings remain committed. Recapture all 22 operations roles as O1, then assemble
B1 with O1 while preserving exactly the B0 first-five mappings and the same
ordered five runtime receipts. Reopen B1 as a distinct copy and link it to the
authenticated B0 transition:

~~~sh
uv run --no-sync python -m benchmarks.issue123_completion assemble \
  --specification /tmp/gmes-issue123-b1-spec.json \
  --bundle /tmp/gmes-issue123-b1-source

cp -a /tmp/gmes-issue123-b1-source /tmp/gmes-issue123-b1-reopened

uv run --no-sync python -m benchmarks.issue123_completion record-reopen \
  --source-index /tmp/gmes-issue123-b1-source/completion-index.json \
  --reopened-index /tmp/gmes-issue123-b1-reopened/completion-index.json \
  --stage final \
  --private-openings /tmp/issue-123-private-authority/publication-openings.json \
  --pre-ack-receipt /tmp/issue-123-private-authority/b0-reopen-receipt.json \
  --output /tmp/issue-123-private-authority/b1-reopen-receipt.json
~~~

This is the fixed point: there is no B2 and no third operations capture. Both
reopen receipts and the publication openings are protected mode-`0600` private
authority files and never public assets or bundle payloads. Receipt links are
SHA-256 over exact canonical receipt bytes; the receipt bodies themselves are
authenticated with the private binding key.

Write each bundle specification with schema
`issue-123-completion-bundle-specification` version 1. List every source as
`{source_path, bundle_path, media_type}`, and make the six `artifacts` entries
refer only to those bundle paths. Embedded descriptors, including nested
correctness and differential NPZ files, must also name registered payloads.
Run the offline structural preflight against B1:

~~~sh
uv run --no-sync python -m benchmarks.issue123_completion evaluate \
  --index /tmp/gmes-issue123-b1-source/completion-index.json \
  --manifest benchmarks/native_oracle_workloads.json \
  --runtime-receipts \
    /trusted/issue-123-runtime-receipts/cpu.json \
    /trusted/issue-123-runtime-receipts/cuda-eager.json \
    /trusted/issue-123-runtime-receipts/cuda-graph.json \
    /trusted/issue-123-runtime-receipts/single-gpu-2d.json \
    /trusted/issue-123-runtime-receipts/single-gpu-3d.json \
  --output /tmp/gmes-issue123-evaluation.json --enforce-structural
~~~

For `evaluate`, `--enforce-structural` exits successfully only when the six
scopes, their cross-scope bindings, and exactly five external runtime receipts
are structurally valid. The fixed order is CPU, CUDA eager, CUDA graph,
single-GPU 2-D, and single-GPU 3-D. The legacy `--enforce` name remains a
final-acceptance gate and therefore always exits 2 in offline mode. The JSON
field `structural_validation_satisfied` reports the structural result; both
final-authority fields remain false even when it is true.

Immediately before the production decision, download exactly the four immutable
release assets into a clean directory. The policy file and
`<CALLER_OWNED_POLICY_SHA256>` must remain outside every bundle and receipt; do
not recover that digest from an untrusted artifact. A standalone operations
check is available for diagnosis with authenticated B1 authority inputs:

~~~sh
mkdir -p /tmp/issue-123-publication/downloaded
gh release download issue-123-technical-evidence-<FINAL_SHA> \
  --repo ruddyscent/gmes \
  --dir /tmp/issue-123-publication/downloaded

uv run --no-sync python -m benchmarks.issue123_operations verify-live \
  --index /tmp/issue-123-operations-o1/operations-index.json \
  --manifest benchmarks/native_oracle_workloads.json \
  --publication-policy /trusted/issue-123-publication-policy.json \
  --publication-policy-sha256 <CALLER_OWNED_POLICY_SHA256> \
  --technical-evidence-asset /tmp/issue-123-publication/downloaded/issue-123-public-technical-evidence.zip \
  --technical-summary-asset /tmp/issue-123-publication/downloaded/issue-123-technical-summary.json \
  --raw-timing-asset /tmp/issue-123-publication/downloaded/issue-115-raw-timing.json \
  --event-profiler-asset /tmp/issue-123-publication/downloaded/issue-115-event-level-profiler.json \
  --source-index /tmp/gmes-issue123-b1-source/completion-index.json \
  --reopened-index /tmp/gmes-issue123-b1-reopened/completion-index.json \
  --private-openings /tmp/issue-123-private-authority/publication-openings.json \
  --pre-ack-bundle-reopen-receipt /tmp/issue-123-private-authority/b0-reopen-receipt.json \
  --final-bundle-reopen-receipt /tmp/issue-123-private-authority/b1-reopen-receipt.json \
  --runtime-receipts \
    /trusted/issue-123-runtime-receipts/cpu.json \
    /trusted/issue-123-runtime-receipts/cuda-eager.json \
    /trusted/issue-123-runtime-receipts/cuda-graph.json \
    /trusted/issue-123-runtime-receipts/single-gpu-2d.json \
    /trusted/issue-123-runtime-receipts/single-gpu-3d.json \
  --baseline-authority live-release \
  --receipt-output /tmp/issue-123-private-authority/operations-only-live-receipt.json
~~~

Only `--baseline-authority live-release` is implemented. Immutable-mirror mode
is rejected unless and until an authorized mirror-receipt contract is added.
The standalone command derives its typed expectation from the authenticated B1
inputs; no caller-supplied expectation JSON exists. Its schema-v3 receipt is
non-replayable operations provenance and cannot set completion authority.
Both protected B1 roots come from that retained authenticated lease, never raw
CLI path spellings. Completion, standalone operations, and publication reject
equal, nested, dot-segment, and symlink-alias output locations before creating a
directory, temporary, sidecar, receipt, or result.

The authoritative completion command consumes the protected openings, both
ordered reopen receipts, both retained B1 trees, and the five runtime receipts:

~~~sh
umask 077
uv run --no-sync python -m benchmarks.issue123_completion verify-live \
  --index /tmp/gmes-issue123-b1-source/completion-index.json \
  --reopened-index /tmp/gmes-issue123-b1-reopened/completion-index.json \
  --private-openings /tmp/issue-123-private-authority/publication-openings.json \
  --pre-ack-bundle-reopen-receipt /tmp/issue-123-private-authority/b0-reopen-receipt.json \
  --final-bundle-reopen-receipt /tmp/issue-123-private-authority/b1-reopen-receipt.json \
  --manifest benchmarks/native_oracle_workloads.json \
  --runtime-receipts \
    /trusted/issue-123-runtime-receipts/cpu.json \
    /trusted/issue-123-runtime-receipts/cuda-eager.json \
    /trusted/issue-123-runtime-receipts/cuda-graph.json \
    /trusted/issue-123-runtime-receipts/single-gpu-2d.json \
    /trusted/issue-123-runtime-receipts/single-gpu-3d.json \
  --publication-policy /trusted/issue-123-publication-policy.json \
  --publication-policy-sha256 <CALLER_OWNED_POLICY_SHA256> \
  --technical-evidence-asset /tmp/issue-123-publication/downloaded/issue-123-public-technical-evidence.zip \
  --technical-summary-asset /tmp/issue-123-publication/downloaded/issue-123-technical-summary.json \
  --raw-timing-asset /tmp/issue-123-publication/downloaded/issue-115-raw-timing.json \
  --event-profiler-asset /tmp/issue-123-publication/downloaded/issue-115-event-level-profiler.json \
  --output-directory /tmp/gmes-issue123-production-verification \
  --enforce
~~~

The completion-owned retained lease holds file descriptors and full registered
file/directory-closure snapshots for both source and reopened B1 trees across
the live operations call, durable receipt validation, final decision, and
result serialization. It revalidates the shared frozen inventory and both trees
at the final authority-elevation boundary immediately before the private result
is atomically linked, leaving no fallible authority-producing gap. Append,
same-size rewrite, inode replacement, and extra-file changes fail closed on
either copy.

The live operations step reruns all 22 fixed GitHub roles, revalidates exactly
the four downloaded public assets, requires the provider's exact two-asset
order, and retains the two downloaded baseline-v3 file identities and exact
bytes across evaluation, receipt validation, the final barrier, the authority
link, durability checks, and final reopen. Any reorder, rewrite, replacement,
mode change, or extra file fails closed before either completion flag can be
true. The private destination contains the schema-v3
`operations-live-receipt.json` and `completion-live-result.json`; both have
`receipt_replay_authority=false` and serialize no private paths, raw identities,
keys, or openings. A later evaluator must rerun the complete live command with
fresh responses and a new private destination.

Pending private bytes, including serialized true claims, are not authority.
The atomic no-replace link immediately after the last retained-input callback
is the sole authority linearization point. The linked leaf is then durably
checked and reopened while the retained baseline and both B1 leases remain
open; a failure after the link is reported as a committed-authority custody
failure and the final leaf is never silently removed.

The preserved contracts are public projection/publication schema v1, bundle
specification v1, completion index v2, exactly four ordered public assets,
exactly five ordered runtime receipts, and exactly 22 operations roles. The
completion live output and private operations live receipt advance to v3.

Assembly refuses an existing destination and rejects absolute bundle paths as
well as dotted, backslash, symlinked, duplicate, unused, or unregistered
payload paths. A specification may use an absolute `source_path`; its
`bundle_path` must still satisfy the canonical relative-path contract. Reads
use bounded exact-size file descriptors with `O_NOFOLLOW` where available and
verify file identity before and after the read. The local threat model assumes
an unprivileged evaluator workspace without a privileged process concurrently
replacing ancestor directories; run the final evaluation from a private,
non-shared directory.

Generated evidence remains in CI artifacts, `/tmp`, or the SHA-256-pinned
Release assets documented above; no generated baseline artifact belongs in the
repository.

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
