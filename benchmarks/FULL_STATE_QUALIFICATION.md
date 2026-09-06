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

The `two-gpu` mode is a separate, eager-only Torch serial-versus-two-rank
check. Launch it with exactly two visible CUDA devices via `torchrun` and a new
private `--output-dir`. It compares every field and owned material-state row at
the standard captures, ordered point source plans plus live source clocks, and
five-step checkpoint replay. Its fixed uneven split has a Drude block crossing
the partition and point sources owned on both ranks, including same-target
last-wins ordering. Point sources have no extended footprint, so this is not an
extended-source-boundary claim. It is neither a native comparison nor compiled
qualification.

Remaining qualification includes unexecuted native cases, the prescribed longer
physical cases, the paper reproductions beyond bounded API tests, transparent
parameter projection, and extended-source-boundary GPU field/state comparisons.
Timing/scaling targets are unrelated to these checks.
