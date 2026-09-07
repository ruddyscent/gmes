# Issue #114 local correctness qualification

`python -m benchmarks.full_state_qualification` records local numerical evidence.
It does not produce or authorize an Issue #169 publication/completion result.
Use an explicit interpreter and this checkout as the working directory. Results
bind HEAD, current imported runtime/harness/test bytes, and the frozen manifest;
an uncommitted harness is reported honestly rather than declared clean.

For an independent CPU supplement, run `--mode analytic --precision float64`
(then float32) with `--output-dir` naming a new private directory outside the
checkout. It compares all six whole fields with the existing independent NumPy
Yee equations in 1-D/2-D/3-D, real/paired-real, at 1/2/5/20/100 steps. These are
analytic comparisons, not native qualification and not stateful-material proof.

For native qualification, provide `--mode native --reference-dir DIRECTORY
--cases dielectric-1d drude-1 --precision float64 --output-dir NEW_DIRECTORY`.
Each selected case requires its complete `CASE.npz` from the manifest-pinned
`native-oracle-observer-v6`, including step zero and every prescribed capture.
The original reference archive is read without modification. No sampled fixture,
Torch reference, fake clean provenance, or locally invented tolerance substitutes
for it. The original observer alone executes native capture in its own verified
environment. JSON summaries contain portable identities and hashes; raw arrays
remain outside the repository.

The retained v6 native observer uses an older CPU-descriptor schema. Before a
capture, `--mode prepare-native-manifest` derives a separate temporary manifest
from the current frozen manifest, preserves every scientific input (including
the v6 observer pin and current tolerances), and substitutes only the legacy
CPU-acceptance descriptor after verifying its two local historical artifacts by
size and SHA-256. Both artifacts explicitly have `diagnostic_acceptance.passed`
and `suite_acceptance.passed` false: they are schema inputs only, never
performance evidence or Issue #169 publication. Native mode re-derives this
manifest byte-for-byte before opening a reference archive.

The comparator checks exact keysets and shapes before every-element comparison.
Integer topology is exact; numeric tolerances come from the frozen manifest.
Material/auxiliary-state arrays and clocks are included. Native transparent face
parameter arrays (four components at each capture) and packed Torch transparent
batch parameters have different layouts; until an exact projection is qualified,
those cases report incomplete. This does not discard other compared fields,
material state, source-auxiliary state, or source records. Missing dtype
tolerances or reference archives are not passes. Negative controls cover an
interior field element, a material-state element, a source-state element, a
source-auxiliary-state element, missing/extra state, a broadcastable wrong
shape, nonfinite values, missing captures, and incorrect clocks.

The `two-gpu` mode is a separate Torch serial-versus-two-rank check. It defaults
to eager execution; `--compile-policy compile` explicitly requests its scoped
compiled variant. Launch either with exactly two visible CUDA devices via
`torchrun` and a new private `--output-dir`. It compares every field, owned
material-state row, source payload, nested transparent-source auxiliary
checkpoint state, and live source clock at every standard capture. Auxiliary
capture retains the existing field/material projection and separately includes
every buffer returned by each auxiliary state checkpoint, including inactive
component PML state. Static source-plan descriptors, such as nested point
amplitudes, are not checkpoint state and are not included by this capture. Its
five-step checkpoint replay compares the same complete distributed persistent
state, not fields alone. The fixed uneven split has a Drude block crossing the
partition, point sources owned on both ranks including same-target last-wins
ordering, and one TFSF face region with owned targets on both sides of the cut.
It is neither a native comparison nor production authority; eager results do
not qualify the explicit compiled scope.

The first local TFSF two-GPU capture predates the per-capture auxiliary and
complete replay checks above, so it remains limited positive eager evidence.
The corrected eager capture at collector
`3e179f47a1c292a8298d8eb372c67d44fb9107b14067f3a540826cf99549b766`
passed with result SHA-256
`d0b422a6b8c80f4b87e5e7dfefb5adf35cc9d6acb47ec003eb44cb5ee6c8bc76`
and raw-array SHA-256
`4e82515db493c0b9d3fd79c58212c0bc66ad968f4d6d8af457813d207c18abc0`.
The final exact-clock validator at
`93b329803f1632640f6952911b434c539019b02fbcb872ed8191007e7369cd5e`
passed 45 focused CPU tests and 56 offline exact scheduling-clock comparisons
using validator SHA-256
`bde5b62d960ed1c2635a73b284b425477a0b277cede4efdd0415e01167b27f94`
and attestation SHA-256
`2d8c15bf20a437c788cd46ae2894717e4c44f3f4bbe418b7155a8b0e7dc871f2`.
That attestation is not a fresh final-byte GPU capture. Reuse the validated
capture and offline result unless affected execution bytes invalidate their
provenance or an exact final-byte execution record is required.

Remaining qualification includes unexecuted native cases, the prescribed longer
physical cases, the paper reproductions beyond bounded API tests, transparent
parameter projection, and any separately advertised compiled distributed scope.
The original #114 correctness gate expressly retains longer physical invariants
and Ziolkowski/DM2 reproductions, so those are closure evidence rather than new
criteria. Timing/scaling targets are unrelated to these checks.

## Issue #114 closure evidence matrix

This index maps the retained correctness and stability conditions in
[Issue #114](https://github.com/ruddyscent/gmes/issues/114) to public code and
review records on master `7286f7f` plus this local follow-up candidate. It does
not expose local captures or turn test fixtures into historical authority, and
does not claim uncommitted follow-up changes are already on master. The
performance amendment defers
timing and scaling goals to [#172](https://github.com/ruddyscent/gmes/issues/172);
it does not defer the rows below. [#169](https://github.com/ruddyscent/gmes/issues/169)
remains separate production-authority work.

| Retained #114 condition | Current evidence | Residual evidence gap |
| --- | --- | --- |
| Complete fields, clocks, and persistent state at 1/2/5/20/100 steps | This harness checks exact keys, shapes, clocks, finite values, and every numeric element. Its unit controls reject an interior field, material-state, source-state, source-auxiliary-state, shape, key, and clock mutation. The analytic mode independently covers six fields in 1-D/2-D/3-D and real/paired-real dielectric cases; `tests/test_torch_dispersive.py` covers nonzero Drude/Lorentz/DCP state at the same captures. | These are not one full historical matrix. Reuse valid bound summaries instead of rerunning passing captures; run only missing coverage or a case whose provenance is invalidated. Unexecuted stateful, longer physical, and paper cases remain incomplete rather than failed. |
| Sources, boundaries, and checkpoints | `tests/test_torch_sources_output.py` checks all 24 Torch TFSF component/face coefficients, paired-real auxiliary replay, float32 auxiliary storage, and a material-boundary case. `tests/test_torch_correctness.py` labels its sampled TFSF probe fixtures test-only. Native mode deliberately marks unprojected transparent parameter arrays incomplete. | A native “24-array mapping” is not a separate #114 gate. The needed completion is an exact full-array/source-state comparison or a qualified parameter projection for the selected TFSF case; sampled probes and an omitted parameter layout cannot certify it. |
| Compiled numerical/stability behavior and bounded repeated advance | [PR #174](https://github.com/ruddyscent/gmes/pull/174) records focused full-state/memory/CPML coverage, advancing CPU RSS evidence, and compiled CUDA field/CPML/checkpoint evidence. `benchmarks/torch_memory_stability.py` binds live buffers, finite state, RSS, allocated/reserved CUDA totals, and compiler counters. | The merged result remains diagnostic-only because `benchmarks/torch_lowered_materialization.py` cannot yet classify all required lowered output paths. This is missing provenance, not a demonstrated clone defect; computed full-field workspaces are not automatically clones. |
| Two-GPU field, material, source, and checkpoint equivalence | The corrected eager capture reconstructs six global fields, canonicalizes owned material/source rows, checks rank-local transparent auxiliary clocks and state at every capture, and compares complete persistent state after five-step replay. The later validator independently rechecks all available scheduling clocks exactly from the immutable raw arrays. | The first TFSF capture remains limited. The corrected capture plus offline clock attestation are reusable local evidence, not a fresh final-byte capture. Compiled and native distributed coverage remain incomplete. |
| Pure Torch package and retained historical reference boundary | The #124 package cutover is recorded as complete in #114; the Torch-only package and historical-reference integrity are not reopened by this harness. | No additional #114 evidence is created here. #169 publication/registry adoption is neither supplied nor required by this local qualification index. |

### Minimal next evidence, not new acceptance criteria

1. Produce one fail-closed lowered-wrapper provenance record for the selected
   compiled CUDA workload. It must bind runtime/config/regions and distinguish a
   literal input snapshot from a computed fixed-shape output; opaque paths stay
   unverified.
2. Use the existing native mode and frozen observer inputs for the selected
   full-state cases. Keep raw arrays outside Git and record only portable
   identities, hashes, captures, comparison status, and any explicit incomplete
   projection.
3. Reuse the validated corrected two-rank capture and exact-clock attestation.
   Run a fresh capture only when affected execution bytes invalidate provenance
   or exact final-byte evidence is required. If compiled distributed execution
   is advertised, record it separately rather than treating eager evidence as
   compiler evidence.

### Reusable verified scope

The following bounded evidence is already available and should not be repeated
merely to recreate a passing capture: eager CPU float64 native full-array
comparisons for `dielectric-1d`, `drude-1`, `dm2-1`, and `cpml`; twelve
independent analytic dielectric cases (two precisions, three dimensions, and
real/paired-real); a selected compiled CUDA 15-by-100 memory/numerical case;
single-GPU CPML float32/float64 field/state/checkpoint checks; and the
corrected eager two-GPU TFSF crossing/full-state capture with its offline exact
clock attestation. The public [PR #174
record](https://github.com/ruddyscent/gmes/pull/174) covers the merged focused
tests and selected CPU/CUDA evidence; raw arrays and local-only summaries are
intentionally not public archives or #169 authority. Preserve each local
summary's portable hashes with its evidence, and rerun only missing scope or
invalidated provenance.
