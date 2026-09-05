# V5.4 test-only historical probes

This is sampled/aggregate historical evidence, not full-array physics equality
and not #169 production publication, adoption, registry, or binding authority.

Reproduce and verify with the exact locked Python 3.14 interpreter (not an
ambient `python`):

```sh
PYTHONDONTWRITEBYTECODE=1 /path/to/gmes/.venv/bin/python generator.py \
  --checkout /path/to/clean-exact-checkout --output /tmp/v5-4-repeat
PYTHONDONTWRITEBYTECODE=1 /path/to/gmes/.venv/bin/python self_test.py \
  --bundle /tmp/v5-4-repeat
```

See `REPRODUCTION.md` for the complete isolated contract. A caller must pin
the literal bundle-manifest and fixture byte counts and SHA-256 values; the
loader never derives trust from the mutable bundle itself.

The final artifact is `pre-cutover-native-numeric-probes-v5-4.json.gz`.  The
generator writes the immutable fixture, support descriptors, side manifest,
and checksum files first, then rewrites the literal caller anchors in
`self_test.py`.  That anchor carrier is required in the synthetic sdist but is
intentionally not described by the side manifest: including it there would
make its final anchor values depend on their own descriptor hash.

Each capture contains only its validated sampled/aggregate array projection
and a domain-separated V5 projection digest. It retains 4,377 public
records and omits exactly the 145 paired direct-source `values` payloads while
retaining their 145 `indices` topology records. For float/complex candidates,
the loader samples at the validated immutable reference selector coordinates;
integer/map selectors remain exact and dynamic. Coordinate-index arrays in the
three reviewed source, source-auxiliary-material, and persistent-state
grammars are represented canonically as signed `int64`; map IDs remain strict
`int32`. It intentionally contains no raw
NPZ identity, raw archive bytes, raw archive path, or raw metadata.  A matching
projection digest does not make this sampled fixture full native-reference or
#169 production authority.

V5.4 treats `nonzero_count` as a diagnostic for every float/complex record,
including both public time families. Only strict integral map and coordinate
index descriptors retain exact nonzero-count enforcement. Candidate storage
dtype is a strict semantic table separate from tolerance dtype: primary fields
and spectra use runtime storage, primary summary/time are float64, state values
are complex128, auxiliary fields/time are float64, auxiliary material values
are complex128, map IDs are int32, and all reviewed coordinate indices are
int64. No descriptor is cast to satisfy this contract.
