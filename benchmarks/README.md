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
compiler-cache metadata, RSS records, profiler traces, and local paths. A
release-scoped, 32-byte-salted SHA-256 host commitment lets a local candidate
prove equality with the frozen CPU host and upstream PyTorch release without
publishing that identity directly; each Release uses a fresh salt shared by its
two slices. The commitment ignores the PyTorch local-build suffix, CUDA runtime,
device inventory, and GPU topology because those device-specific values differ
between CPU and CUDA builds and are validated by the corresponding runtime
contracts. It still binds the platform, Python version, PyTorch public version,
CPU counts, affinity, topology, and normalized model. This is a public equality
commitment, not a guarantee that a party with an independently captured
candidate fingerprint cannot test equality.
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

~~~sh
uv run --no-sync python -m benchmarks.torch_tuning \
  --case cuda-gates --device cuda:0 --precision float32 \
  --compile-mode default --policy auto --threads 1 --interop-threads 1 \
  --cuda-correctness-index /tmp/issue-123/correctness-eager.json \
    /tmp/issue-123/correctness-graph.json \
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
  --case single-gpu-3d --device cuda:0 --precision float32 \
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
  --descriptor-root /tmp \
  --output /tmp/torch-correctness-index.json

uv run --no-sync python -m benchmarks.torch_correctness validate-index \
  --index /tmp/torch-correctness-index.json \
  --descriptor-root /tmp \
  --candidate-evidence /tmp/cpu-one.json
```

The index command requires the ordered union of every manifest `correctness`
and `physical_checks` case. It reopens both NPZ archives, compares complete
fields and persistent/source/auxiliary state, and records the exact archive
hashes. `validate-index` repeats that comparison; it does not trust an embedded
`"passed": true`. Every nested archive descriptor is a canonical POSIX path
relative to `--descriptor-root`, plus its exact byte size, media type, SHA-256,
and candidate binding; absolute paths and path escapes fail closed.

### Candidate-bound differential evidence

Create both differential scopes from a clean candidate checkout. The isolated
runner must use the pinned `native-oracle-observer-v6` worktree for every native
reference; running `native_oracle.py capture` in the candidate checkout is not
an equivalent reference. Start with a fresh bundle directory so the producer's
exact source and output closures cannot include stale files:

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

for GMES_CASE in single-gpu-2d single-gpu-3d; do
  uv run --no-sync python -m benchmarks.torch_correctness capture \
    --reference "$GMES_ISSUE123_BUNDLE/sources/native/single/$GMES_CASE.npz" \
    --output "$GMES_ISSUE123_BUNDLE/sources/torch/single/$GMES_CASE.npz" \
    --device cuda:0 --precision float32 --graph-mode eager --compile-mode default
done
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

The builder first validates every complete native/Torch source archive and its
runtime/provenance contract, then writes final-step fields plus every persistent
state/source array. Each NPZ descriptor has exactly the five bundle-relative
keys `path`, `sha256`, `size_bytes`, `media_type`, and `candidate_evidence`.
Validation derives workload/device dtype and tolerance from the manifest and
recomputes every array comparison; recorded tolerances and pass flags cannot
weaken the gate.

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
- `single_gpu`: all six CUDA gates, the two large-case differential index,
  and six traces;
- `two_gpu`: four scaling results, eager/graph schema-3 raw correctness
  matrices, four collective-failure wrappers, and eight rank traces;
- `macos`: the schema-2 package/runtime index and raw Actions ZIP containing
  the sdist, CPython 3.14 arm64 wheel, and all 12 stdout/stderr logs;
- `operations`: the descriptor-only index over exact GitHub API response
  bytes for issue #115, the pull request, base comparison, CI and CodeQL,
  ruleset, checks, reviews, requested reviewers, and review threads.

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
job. Completion is true only when all six scopes pass.

Download the pull-request macOS artifact and assemble its schema-2 index:

~~~sh
uv run --no-sync python -m benchmarks.macos_ci_evidence assemble \
  --runtime-index /tmp/issue-123-macos/runtime-index.json \
  --actions-archive /tmp/issue-123-macos/actions-archive.zip \
  --repository ruddyscent/gmes --ci-run-id <CI> \
  --output /tmp/issue-123-macos/index.json \
  --scope-output /tmp/issue-123-macos/scope.json
~~~

Capture operational evidence only after issue #115 is closed, its final owner
handoff comment exists, the pull request is current with `master`, required CI
and CodeQL are complete, and all review requests and conversations are clear.
Check runs are queried at the candidate head; CodeQL analyses and alerts remain
bound to the synthetic merge:

~~~sh
uv run --no-sync python -m benchmarks.issue123_operations \
  --repository ruddyscent/gmes --pull-request <PR> \
  --ci-run-id <CI> --codeql-run-id <CODEQL> \
  --output-directory /tmp/issue-123-operations
~~~

The macOS archive has exactly 15 non-directory members: the runtime index, two
packages, and 12 stdout/stderr logs. The operations producer stores the raw API
responses and their digests; it emits no self-reported acceptance booleans.
The evaluator derives the active ruleset's strict checks, squash-only merge,
CodeQL `errors` quality threshold, `high_or_higher` security threshold,
candidate/base/merge bindings, and unresolved-review count from those bytes.

Write a bundle specification with schema
`issue-123-completion-bundle-specification` version 1. List every source as
`{source_path, bundle_path, media_type}`, and make the six `artifacts`
entries refer only to those bundle paths. Embedded descriptors, including
nested correctness and differential NPZ files, must also name registered
payloads. Then assemble once into a new destination and evaluate the generated
index:

~~~sh
uv run --no-sync python -m benchmarks.issue123_completion assemble \
  --specification /tmp/gmes-issue123-spec.json \
  --bundle /tmp/gmes-issue123-final

uv run --no-sync python -m benchmarks.issue123_completion evaluate \
  --index /tmp/gmes-issue123-final/completion-index.json \
  --output /tmp/gmes-issue123-evaluation.json --enforce
~~~

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
