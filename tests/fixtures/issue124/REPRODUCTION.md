# V5.4 isolated reproduction

This bundle is test-only sampled/aggregate historical evidence. It is not a
full-array equality oracle and does not publish, adopt, or alter the #169
production evidence schema or bindings.

Use the repository's locked Python 3.14 environment and an exact, clean local
checkout of commit `66a0a1aa8d6f163134967e8b8a7e9dc46530717b`:

```sh
PYTHONDONTWRITEBYTECODE=1 /path/to/gmes/.venv/bin/python \
  generator.py --checkout /path/to/exact-clean-checkout --output /tmp/v5-4-repeat
PYTHONDONTWRITEBYTECODE=1 /path/to/gmes/.venv/bin/python \
  self_test.py --bundle /tmp/v5-4-repeat
```

The reproducer verifies the checkout commit, clean staged, unstaged, and
untracked `gmes/` and `benchmarks/` paths, observer tag object and peeled commit, source manifest,
all 33 consumer call sites, six default-float64 consumers, two CUDA runtime
contracts, privacy, deterministic gzip level 9 with `mtime=0`, exact counts,
and removal of transient raw NPZ files. It imports the producer only from the
caller-provided checkout and performs no network access. Compare the generated
bundle and fixture SHA-256/byte counts with the code-owned trust constants in
`self_test.py`; never derive those constants from the bundle being loaded.
The final artifact name is `pre-cutover-native-numeric-probes-v5-4.json.gz`.
Generation is ordered: immutable support descriptors and BUNDLE checksums are
written before the literal external anchors in `self_test.py`.  The self-test
is present in the synthetic sdist, but deliberately excluded from the side
manifest support descriptors so the anchor file cannot form a descriptor/hash
cycle.

V5.4 serializes no transient raw NPZ identity. It derives a
`validated-probe-projection-v5-storage-dtype-contract` SHA-256 from the canonical `profile`, `case`,
archive schema version, and validated sampled arrays, prefixed with
`gmes-issue124-probe-projection-v5\\0`. It retains 4,377 public records,
omits exactly 145 paired direct-source `values` payloads, and retains all 145
direct-source `indices` records. Float/complex candidate samples use the
immutable validated reference selector sequence
`reference-selector-candidate-fixed-positions-v3`; integer/map selectors stay
exact and candidate-derived. The three reviewed coordinate-index
grammars are canonical signed `int64`; map IDs remain strict `int32`. Two
separately clean captures must
produce identical projection objects, canonical JSON, gzip, checksums, side
manifest, BUNDLE, and caller anchors; otherwise do not choose one result.

V5.4 keeps nonzero count diagnostic for every float/complex descriptor,
including `step/*/time` and `source_aux/*/time`; only reviewed integral map
and coordinate-index descriptors enforce it exactly. Storage dtype is validated
strictly by semantic grammar before tolerance selection; tolerance precision
remains the pinned runtime/auxiliary contract and no candidate is cast.
