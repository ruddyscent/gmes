#!/usr/bin/env python3
"""Bounded V5.4 fixture, comparator, and package-placement self-tests."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

import numpy as np


# generator.py replaces this one-line marker only after BUNDLE-MANIFEST is final.
TRUST = {'expected_manifest_sha256': 'cce4820ccc0e8050db6baf47d18fe646ee2c59bb64c89c0279cd38be1ba17d41', 'expected_manifest_bytes': 28865, 'expected_fixture_sha256': '9fa7d54d63c4e0c6bceca37ed2c9c392d398f4ebe5ae94748de3b25b73b6b289', 'expected_fixture_bytes': 202209}
FIXTURE_ROOT = "tests/fixtures/issue124"
FIXTURE_NAME = "pre-cutover-native-numeric-probes-v5-4.json.gz"
FIXTURE_CHECKSUM_NAME = "pre-cutover-native-numeric-probes-v5-4.sha256"
TOLERANCES_NUMERIC_SHA256 = "208b8816eafd97b8a3a295bdff0c68db8b2a6cccc47ffc3301fcc895cbeb4c3f"
SYNTHETIC_STRATEGY_KEY = "step/1/state/Ex/0-Dielectric/values"
SYNTHETIC_STRATEGY_VALUES = np.asarray([0.0j], dtype=np.complex128)
FORBIDDEN = re.compile(r"(?:@|(?:github_pat_|gh[opsu]_)[A-Za-z0-9_]{8,}|(?i:password|secret|token|hostname|host_identity|environment|command|/home/|\\\\Users\\\\))")


def module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    try:
        spec.loader.exec_module(value)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return value


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def expect_value_error(callback, label: str) -> None:
    try:
        callback()
    except ValueError:
        return
    raise AssertionError(f"{label} was accepted")


def copied_bundle(root: Path):
    temporary = tempfile.TemporaryDirectory()
    target = Path(temporary.name) / "bundle"
    shutil.copytree(root, target)
    return temporary, target


def write_side(root: Path, side: dict) -> dict:
    payload = canonical(side)
    (root / "BUNDLE-MANIFEST.json").write_bytes(payload)
    (root / "BUNDLE-MANIFEST.sha256").write_text(f"{digest(payload)}  BUNDLE-MANIFEST.json\n")
    return {
        "expected_manifest_sha256": digest(payload),
        "expected_manifest_bytes": len(payload),
        "expected_fixture_sha256": side["fixture"]["compressed_sha256"],
        "expected_fixture_bytes": side["fixture"]["compressed_bytes"],
    }


def rewrite_fixture(root: Path, side: dict, fixture: dict) -> dict:
    raw = canonical(fixture)
    encoded = gzip.compress(raw, compresslevel=9, mtime=0)
    fixture_path = root / side["fixture"]["file"]
    fixture_path.write_bytes(encoded)
    side["fixture"].update(
        compressed_bytes=len(encoded),
        compressed_sha256=digest(encoded),
        uncompressed_bytes=len(raw),
        uncompressed_sha256=digest(raw),
    )
    (root / FIXTURE_CHECKSUM_NAME).write_text(
        f"{side['fixture']['compressed_sha256']}  {fixture_path.name}\n"
    )
    return write_side(root, side)


def fixture_document(root: Path, side: dict) -> dict:
    return json.loads(gzip.decompress((root / side["fixture"]["file"]).read_bytes()))


def synthetic_resolved(loader, *, complex_values: bool = False, atol: float = 1e-6, rtol: float = 0.0, capture: dict | None = None):
    key = "step/1/field/Ex"
    values = np.asarray([1.0 + 1.0j, 2.0 + 2.0j] if complex_values else [1.0, 2.0], dtype=np.complex128 if complex_values else np.float64)
    if capture is None:
        capture = {"arrays": [
            loader.probe_array(key, values),
            loader.probe_array(SYNTHETIC_STRATEGY_KEY, SYNTHETIC_STRATEGY_VALUES),
        ]}
    manifest = {"tolerances": {"torch": {"dielectric": {"float64": {"rtol": rtol, "atol": atol}}}}}
    return key, values, loader.ResolvedCase(
        "synthetic-v5", "synthetic", manifest,
        {"device": "cpu", "precision": "float64", "graph_mode": "eager", "compile_mode": "default"},
        1, capture,
    )


def write_candidate(path: Path, key: str, values: np.ndarray) -> None:
    arrays = {key: values}
    if key == "step/1/field/Ex":
        arrays[SYNTHETIC_STRATEGY_KEY] = SYNTHETIC_STRATEGY_VALUES
    write_arrays(path, arrays)


def write_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **arrays)


def assert_comparator(loader) -> None:
    key, values, resolved = synthetic_resolved(loader)
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        good, within, outside = base / "good.npz", base / "within.npz", base / "outside.npz"
        write_candidate(good, key, values)
        write_candidate(within, key, values + np.asarray([0.0, 0.5e-6]))
        write_candidate(outside, key, values + np.asarray([0.0, 2.0e-6]))
        assert loader.compare_candidate(good, resolved)["passed"]
        assert loader.compare_candidate_bytes(good.read_bytes(), resolved)["passed"]
        assert not loader.compare_candidate_bytes(bytearray(good.read_bytes()), resolved)["passed"]
        assert loader.compare_candidate(within, resolved)["passed"], "atol-compliant candidate failed"
        assert not loader.compare_candidate(outside, resolved)["passed"], "atol-violating candidate passed"
        key2, values2, relative = synthetic_resolved(loader, rtol=0.1, atol=0.0)
        relative_ok, relative_bad = base / "relative-ok.npz", base / "relative-bad.npz"
        write_candidate(relative_ok, key2, values2 * 1.05)
        write_candidate(relative_bad, key2, values2 * 1.11)
        assert loader.compare_candidate(relative_ok, relative)["passed"], "rtol-compliant candidate failed"
        assert not loader.compare_candidate(relative_bad, relative)["passed"], "rtol-violating candidate passed"
        ckey, cvalues, complex_resolved = synthetic_resolved(loader, complex_values=True)
        complex_ok, complex_bad = base / "complex-ok.npz", base / "complex-bad.npz"
        write_candidate(complex_ok, ckey, cvalues + 0.5e-6j)
        write_candidate(complex_bad, ckey, cvalues + 2.0e-6j)
        assert loader.compare_candidate(complex_ok, complex_resolved)["passed"], "complex component tolerance failed"
        assert not loader.compare_candidate(complex_bad, complex_resolved)["passed"], "complex component violation passed"
        shortened = copy.deepcopy(resolved.capture)
        shortened["arrays"][0][5] = shortened["arrays"][0][5][:-1]
        _, _, short_resolved = synthetic_resolved(loader, capture=shortened)
        assert not loader.compare_candidate(good, short_resolved)["passed"], "sample length mismatch passed"
        oversized = base / "oversized.npz"
        header = io.BytesIO()
        np.lib.format.write_array_header_2_0(header, {"descr": "<f8", "fortran_order": False, "shape": (loader.MAX_NPY_PAYLOAD_BYTES // 8 + 1,)})
        with zipfile.ZipFile(oversized, "w") as archive:
            archive.writestr(f"{key}.npy", header.getvalue())
        assert not loader.compare_candidate(oversized, resolved)["passed"], "oversized NPY payload passed"
        assert not loader.compare_candidate_bytes(oversized.read_bytes(), resolved)["passed"], "oversized immutable snapshot passed"
        duplicate = base / "duplicate.npz"
        payload = io.BytesIO(); np.save(payload, values)
        with zipfile.ZipFile(duplicate, "w") as archive:
            archive.writestr(f"{key}.npy", payload.getvalue())
            archive.writestr(f"{key}.npy", payload.getvalue())
        assert not loader.compare_candidate(duplicate, resolved)["passed"], "duplicate NPZ member passed"
        assert not loader.compare_candidate_bytes(duplicate.read_bytes(), resolved)["passed"], "duplicate immutable snapshot passed"


def assert_reference_selector_regressions(loader) -> None:
    """Float/complex candidates use fixture positions, never their own extrema."""
    key = SYNTHETIC_STRATEGY_KEY
    reference_values = np.zeros(10, dtype=np.complex128)
    reference_values[4] = reference_values[8] = 2.0
    reference = loader.probe_array(key, reference_values)
    positions = [sample[0] for sample in reference[5]]
    assert positions == [0, 9, 5, 4, 1, 2, 3, 6, 7]
    assert loader.reference_value_selected_positions(reference) == [4]
    _, _, resolved = synthetic_resolved(loader, capture={"arrays": [reference]})
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        earlier = reference_values.copy(); earlier[1] = 0.5e-6
        assert np.flatnonzero(earlier)[0] == 1
        assert [sample[0] for sample in loader.probe_array(key, earlier)[5]] != positions
        earlier_path = base / "earlier-nonzero.npz"; write_arrays(earlier_path, {key: earlier})
        assert loader.compare_candidate_bytes(earlier_path.read_bytes(), resolved)["passed"], "tolerated earlier nonzero changed selector acceptance"
        shifted_max = reference_values.copy(); shifted_max[8] += 0.5e-6
        assert int(np.argmax(np.abs(shifted_max))) == 8
        max_path = base / "shifted-max.npz"; write_arrays(max_path, {key: shifted_max})
        assert loader.compare_candidate(max_path, resolved)["passed"], "tolerated argmax shift changed selector acceptance"
        beyond = reference_values.copy(); beyond[4] += 2e-6
        beyond_path = base / "beyond-reference-max.npz"; write_arrays(beyond_path, {key: beyond})
        assert not loader.compare_candidate(beyond_path, resolved)["passed"], "reference max beyond tolerance passed"
        complex_reference = reference_values.astype(np.complex128) * (1.0 + 1.0j)
        complex_probe = loader.probe_array(key, complex_reference)
        _, _, complex_resolved = synthetic_resolved(loader, capture={"arrays": [complex_probe]})
        complex_earlier = complex_reference.copy(); complex_earlier[1] = 0.5e-6j
        complex_path = base / "complex-earlier-nonzero.npz"; write_arrays(complex_path, {key: complex_earlier})
        assert loader.compare_candidate(complex_path, complex_resolved)["passed"], "complex tolerated earlier nonzero failed"
        complex_beyond = complex_reference.copy(); complex_beyond[4] += 2e-6j
        complex_beyond_path = base / "complex-beyond-reference.npz"; write_arrays(complex_beyond_path, {key: complex_beyond})
        assert not loader.compare_candidate(complex_beyond_path, complex_resolved)["passed"], "complex frozen reference position beyond tolerance passed"
    tie = np.zeros(10, dtype=np.complex128); tie[4] = 2.0; tie[5] = -2.0
    tie_probe = loader.probe_array(key, tie)
    assert [sample[0] for sample in tie_probe[5]] == [0, 9, 5, 4, 1, 2, 3, 6, 7]
    expect_value_error(lambda: loader.probe_array(key, tie, sample_positions=[0, 0]), "duplicate immutable selector")


def assert_nonzero_count_contract(loader, bundle) -> None:
    """Only integral descriptors make nonzero count a pass/fail constraint."""
    records = [record for capture in bundle.fixture["captures"].values() for record in capture["arrays"]]
    integral = [record for record in records if loader.nonzero_count_is_exact(record[1])]
    nonintegral = [record for record in records if not loader.nonzero_count_is_exact(record[1])]
    assert len(integral) == 1813 and len(nonintegral) == 2564
    assert all(record[1].startswith(("bool", "int", "uint")) for record in integral)
    assert all(not loader.nonzero_count_is_exact(record[1]) for record in nonintegral)
    time_records = [record for record in records if record[0].endswith("/time")]
    direct_time = [record for record in time_records if "/source_aux/" not in record[0]]
    auxiliary_time = [record for record in time_records if "/source_aux/" in record[0]]
    assert (len(direct_time), len(auxiliary_time)) == (83, 22)
    assert all(record[1] == "float64" and not loader.nonzero_count_is_exact(record[1]) for record in time_records)
    choices = (("step-time", direct_time[0][0]), ("source-aux-time", auxiliary_time[0][0]))
    companion_values = np.asarray([0.0j], dtype=np.complex128)
    manifest = {
        "tolerances": {
            "torch": {
                "dielectric": {"float64": {"rtol": 0.0, "atol": 1e-6}},
                "source_auxiliary": {"synthetic": {"float64": {"rtol": 0.0, "atol": 1e-6}}},
            }
        }
    }
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        for label, key in choices:
            reference_values = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
            reference = loader.probe_array(key, reference_values)
            companion = loader.probe_array(SYNTHETIC_STRATEGY_KEY, companion_values)
            resolved = loader.ResolvedCase(
                "synthetic-v5-r2", "synthetic", manifest,
                {"device": "cpu", "precision": "float64", "graph_mode": "eager", "compile_mode": "default"},
                1, {"arrays": [reference, companion]},
            )
            within = reference_values.copy(); within[0] = 0.5e-6
            within_path = base / f"{label}-within.npz"
            write_arrays(within_path, {key: within, SYNTHETIC_STRATEGY_KEY: companion_values})
            result = loader.compare_candidate_bytes(within_path.read_bytes(), resolved)
            assert result["passed"], (label, result["failures"])
            nonzero = next(item for item in result["tolerance_results"] if item["key"] == key and item["label"] == "nonzero_count")
            assert nonzero["diagnostic"] is True and nonzero["limit"] is None
            beyond = reference_values.copy(); beyond[0] = 2e-6
            beyond_path = base / f"{label}-beyond.npz"
            write_arrays(beyond_path, {key: beyond, SYNTHETIC_STRATEGY_KEY: companion_values})
            result = loader.compare_candidate(beyond_path, resolved)
            assert not result["passed"], f"{label} beyond tolerance passed"
            nonzero = next(item for item in result["tolerance_results"] if item["key"] == key and item["label"] == "nonzero_count")
            assert nonzero["diagnostic"] is True and nonzero["limit"] is None
        map_key = "map/Ex/material_ids"
        map_reference = loader.probe_array(map_key, np.asarray([[0, 1]], dtype=np.int32))
        resolved = loader.ResolvedCase(
            "synthetic-v5-r2", "synthetic", {},
            {"device": "cpu", "precision": "float64", "graph_mode": "eager", "compile_mode": "default"},
            1, {"arrays": [map_reference]},
        )
        map_path = base / "integer-nonzero-drift.npz"
        write_arrays(map_path, {map_key: np.asarray([[1, 1]], dtype=np.int32)})
        result = loader.compare_candidate(map_path, resolved)
        assert not result["passed"], "integer nonzero drift passed"
        nonzero = next(item for item in result["tolerance_results"] if item["label"] == "nonzero_count")
        assert nonzero["diagnostic"] is False and nonzero["limit"] == 0.0


def assert_loader_negatives(root: Path, loader, bundle) -> None:
    side = bundle.side
    temporary, copied = copied_bundle(root)
    try:
        item = copied / side["fixture"]["file"]
        item.write_bytes(item.read_bytes() + b"x")
        expect_value_error(lambda: loader.load_bundle(copied, **TRUST), "tampered fixture")
    finally:
        temporary.cleanup()
    temporary, copied = copied_bundle(root)
    try:
        changed = json.loads((copied / "BUNDLE-MANIFEST.json").read_text())
        changed["sampling"]["max_samples_per_array"] = 12
        write_side(copied, changed)
        expect_value_error(lambda: loader.load_bundle(copied, **TRUST), "coherent side-manifest rewrite")
    finally:
        temporary.cleanup()
    temporary, copied = copied_bundle(root)
    try:
        changed = json.loads((copied / "BUNDLE-MANIFEST.json").read_text())
        changed["profiles"][next(iter(changed["profiles"]))]["file"] = "../escape.json"
        anchors = write_side(copied, changed)
        expect_value_error(lambda: loader.load_bundle(copied, **anchors), "descriptor traversal")
    finally:
        temporary.cleanup()
    temporary, copied = copied_bundle(root)
    try:
        profile = side["profiles"][next(iter(side["profiles"]))]["file"]
        target = copied / profile
        target.unlink(); target.symlink_to(root / profile)
        expect_value_error(lambda: loader.load_bundle(copied, **TRUST), "descriptor symlink")
    finally:
        temporary.cleanup()
    temporary, copied = copied_bundle(root)
    try:
        profile = side["profiles"][next(iter(side["profiles"]))]["file"]
        target, twin = copied / profile, copied / "hardlink-source.json"
        twin.write_bytes(target.read_bytes()); target.unlink(); os.link(twin, target)
        expect_value_error(lambda: loader.load_bundle(copied, **TRUST), "descriptor hard-link")
    finally:
        temporary.cleanup()
    temporary, copied = copied_bundle(root)
    try:
        changed = json.loads((copied / "BUNDLE-MANIFEST.json").read_text())
        fixture = fixture_document(copied, changed); fixture["fixture_schema_version"] = 999
        anchors = rewrite_fixture(copied, changed, fixture)
        expect_value_error(lambda: loader.load_bundle(copied, **anchors), "fixture schema mutation")
    finally:
        temporary.cleanup()
    temporary, copied = copied_bundle(root)
    try:
        changed = json.loads((copied / "BUNDLE-MANIFEST.json").read_text())
        fixture = fixture_document(copied, changed)
        next(iter(fixture["captures"].values()))["ephemeral_raw"] = {"bytes": 1, "sha256": "0" * 64}
        anchors = rewrite_fixture(copied, changed, fixture)
        expect_value_error(lambda: loader.load_bundle(copied, **anchors), "ephemeral raw revival")
    finally:
        temporary.cleanup()
    temporary, copied = copied_bundle(root)
    try:
        changed = json.loads((copied / "BUNDLE-MANIFEST.json").read_text())
        fixture = fixture_document(copied, changed)
        next(iter(fixture["captures"].values()))["unknown_capture_field"] = True
        anchors = rewrite_fixture(copied, changed, fixture)
        expect_value_error(lambda: loader.load_bundle(copied, **anchors), "unknown capture field")
    finally:
        temporary.cleanup()
    temporary, copied = copied_bundle(root)
    try:
        changed = json.loads((copied / "BUNDLE-MANIFEST.json").read_text())
        fixture = fixture_document(copied, changed)
        next(iter(fixture["captures"].values()))["probe_projection"]["sha256"] = "0" * 64
        anchors = rewrite_fixture(copied, changed, fixture)
        expect_value_error(lambda: loader.load_bundle(copied, **anchors), "projection digest mutation")
    finally:
        temporary.cleanup()
    temporary, copied = copied_bundle(root)
    try:
        changed = json.loads((copied / "BUNDLE-MANIFEST.json").read_text())
        fixture = fixture_document(copied, changed)
        next(iter(fixture["captures"].values()))["arrays"][0][5][0][2][0] += 0.25
        anchors = rewrite_fixture(copied, changed, fixture)
        expect_value_error(lambda: loader.load_bundle(copied, **anchors), "numeric probe mutation")
    finally:
        temporary.cleanup()
    temporary, copied = copied_bundle(root)
    try:
        changed = json.loads((copied / "BUNDLE-MANIFEST.json").read_text())
        fixture = fixture_document(copied, changed); fixture["provenance"]["repository"] = "unreviewed/example"
        anchors = rewrite_fixture(copied, changed, fixture)
        expect_value_error(lambda: loader.load_bundle(copied, **anchors), "fixture provenance mutation")
    finally:
        temporary.cleanup()
    temporary, copied = copied_bundle(root)
    try:
        payload = (copied / "BUNDLE-MANIFEST.json").read_bytes().rstrip()[:-1] + b',"fixture":null}\n'
        (copied / "BUNDLE-MANIFEST.json").write_bytes(payload)
        (copied / "BUNDLE-MANIFEST.sha256").write_text(f"{digest(payload)}  BUNDLE-MANIFEST.json\n")
        anchors = {**TRUST, "expected_manifest_sha256": digest(payload), "expected_manifest_bytes": len(payload)}
        expect_value_error(lambda: loader.load_bundle(copied, **anchors), "duplicate side key")
    finally:
        temporary.cleanup()
    temporary, copied = copied_bundle(root)
    try:
        changed = json.loads((copied / "BUNDLE-MANIFEST.json").read_text())
        fixture = fixture_document(copied, changed)
        first = next(iter(fixture["captures"].values()))["arrays"]
        first[0], first[1] = first[1], first[0]
        anchors = rewrite_fixture(copied, changed, fixture)
        expect_value_error(lambda: loader.load_bundle(copied, **anchors), "noncontiguous probe ordering")
    finally:
        temporary.cleanup()
    temporary, copied = copied_bundle(root)
    try:
        changed = json.loads((copied / "BUNDLE-MANIFEST.json").read_text())
        fixture = fixture_document(copied, changed)
        next(iter(fixture["captures"].values()))["arrays"][0][5] = []
        anchors = rewrite_fixture(copied, changed, fixture)
        expect_value_error(lambda: loader.load_bundle(copied, **anchors), "truncated sample list")
    finally:
        temporary.cleanup()


def assert_runtime_authority(loader, bundle) -> None:
    binding = bundle.side["runtime_bindings"][0]
    manifest = copy.deepcopy(bundle.profiles[binding["profile"]])
    resolved = loader.resolve_case(bundle, profile_id=binding["profile"], case=binding["case"], manifest=manifest, runtime=binding["runtime"], consumer_line=binding["test_line"])
    assert resolved.capture["profile"] == binding["profile"]
    changed = copy.deepcopy(manifest); changed["_unreviewed"] = True
    expect_value_error(lambda: loader.resolve_case(bundle, profile_id=binding["profile"], case=binding["case"], manifest=changed, runtime=binding["runtime"], consumer_line=binding["test_line"]), "current manifest drift")
    runtime = dict(binding["runtime"]); runtime["precision"] = "float32"
    expect_value_error(lambda: loader.resolve_case(bundle, profile_id=binding["profile"], case=binding["case"], manifest=manifest, runtime=runtime, consumer_line=binding["test_line"]), "runtime fallback")
    expect_value_error(lambda: loader.resolve_case(bundle, profile_id=binding["profile"], case="not-a-reviewed-case", manifest=manifest, runtime=binding["runtime"], consumer_line=binding["test_line"]), "case fallback")


def assert_binding_schedule_validation(root: Path, loader, bundle) -> None:
    """A binding may select an explicit subset, but never another case schedule."""
    cuda_index = next(
        index
        for index, binding in enumerate(bundle.side["runtime_bindings"])
        if binding["runtime"]["device"] == "cuda:0"
    )
    temporary, copied = copied_bundle(root)
    try:
        changed = json.loads((copied / "BUNDLE-MANIFEST.json").read_text())
        fixture = fixture_document(copied, changed)
        changed["runtime_bindings"][cuda_index]["capture_steps"] = [100]
        fixture["runtime_bindings"][cuda_index]["capture_steps"] = [100]
        anchors = rewrite_fixture(copied, changed, fixture)
        subset = loader.load_bundle(copied, **anchors)
        assert subset.side["runtime_bindings"][cuda_index]["capture_steps"] == [100]
    finally:
        temporary.cleanup()
    temporary, copied = copied_bundle(root)
    try:
        changed = json.loads((copied / "BUNDLE-MANIFEST.json").read_text())
        fixture = fixture_document(copied, changed)
        changed["runtime_bindings"][cuda_index]["capture_steps"] = [1]
        fixture["runtime_bindings"][cuda_index]["capture_steps"] = [1]
        anchors = rewrite_fixture(copied, changed, fixture)
        expect_value_error(lambda: loader.load_bundle(copied, **anchors), "invalid binding schedule")
    finally:
        temporary.cleanup()


def assert_index_canonicalization(root: Path, loader, bundle) -> None:
    """Exercise all and only the adjudicated coordinate-index semantics."""
    generator = module(root / "generator.py", "generator_index_projection_selftest")
    records = [
        record
        for capture in bundle.fixture["captures"].values()
        for record in capture["arrays"]
    ]
    coordinate = [record for record in records if loader.coordinate_index_family(record[0]) is not None]
    map_ids = [record for record in records if loader.MAP_ID_PATH.fullmatch(record[0])]
    assert len(coordinate) == 1393 and len(map_ids) == 420
    assert {loader.coordinate_index_family(record[0]) for record in coordinate} == {0, 1, 2}
    assert not [record[0] for record in records if record[0].endswith("/indices") and loader.coordinate_index_family(record[0]) is None]
    assert all(record[1] == "int64" for record in coordinate)
    assert all(record[1] == "int32" for record in map_ids)
    families = (
        "step/1/source/Ex/0-PointSourceEx/indices",
        "step/1/source_aux_material/0/Ex/0-Dielectric/indices",
        "step/1/state/Ex/0-Dielectric/indices",
    )
    values = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    for family, key in enumerate(families):
        native = generator.array_probe(key, values)
        torch = loader.probe_array(key, values.astype(np.int64))
        assert native == torch and native[1] == "int64", family
        _, _, resolved = synthetic_resolved(loader, capture={"arrays": [native]})
        def candidate_arrays(item):
            arrays = {key: item}
            if family == 0:
                arrays[key.removesuffix("/indices") + "/values"] = np.asarray([0], dtype=np.uint64)
            return arrays
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / f"family-{family}.npz"
            write_arrays(path, candidate_arrays(values.astype(np.int64)))
            result = loader.compare_candidate(path, resolved)
            assert result["passed"], result["failures"]
            changed = values.astype(np.int64); changed[0, 0] += 1
            write_arrays(path, candidate_arrays(changed))
            result = loader.compare_candidate(path, resolved)
            assert not result["passed"]
            assert all(record["rtol"] == record["atol"] == 0.0 for record in result["tolerance_results"] if record["dtype"] == "int64")
            for label, invalid in (
                ("negative", np.asarray([[-1, 1, 2], [3, 4, 5]], dtype=np.int64)),
                ("out-of-field", np.asarray([[1_000_000, 1, 2], [3, 4, 5]], dtype=np.int64)),
            ):
                write_arrays(path, candidate_arrays(invalid))
                assert not loader.compare_candidate(path, resolved)["passed"], label
        for label, invalid in (
            ("float", values.astype(np.float64)),
            ("bool", values.astype(bool)),
            ("complex", values.astype(np.complex128)),
            ("object", values.astype(object)),
            ("uint32", values.astype(np.uint32)),
            ("uint64-overflow", np.asarray([[2**63, 1, 2]], dtype=np.uint64)),
            ("rank", values.reshape(-1)),
            ("shape", values.reshape(1, 2, 3)),
            ("big-endian", values.astype(">i4")),
        ):
            expect_value_error(lambda invalid=invalid: loader.probe_array(key, invalid), f"{key}/{label}")
    map_key = "map/Ex/material_ids"
    map_values = np.asarray([[0, 1], [2, 3]], dtype=np.int32)
    _, _, resolved = synthetic_resolved(loader, capture={"arrays": [loader.probe_array(map_key, map_values)]})
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "map-width.npz"
        write_candidate(path, map_key, map_values.astype(np.int64))
        assert not loader.compare_candidate(path, resolved)["passed"], "map IDs must remain int32"


def _storage_values(key: str, dtype: str) -> np.ndarray:
    """Use exactly representable values for strict storage-dtype comparisons."""
    if key.startswith("map/"):
        return np.asarray([[0, 1], [2, 3]], dtype=dtype)
    if key.endswith("/indices"):
        return np.asarray([[0, 1, 2], [3, 4, 5]], dtype=dtype)
    if dtype.startswith("complex"):
        return np.asarray([1.0 + 1.0j, 2.0 + 2.0j], dtype=dtype)
    return np.asarray([1.0, 2.0], dtype=dtype)


def _storage_manifest() -> dict:
    tolerance = {"rtol": 0.0, "atol": 1e-6}
    return {
        "tolerances": {
            "torch": {
                "dielectric": {
                    "float32": dict(tolerance),
                    "float64": dict(tolerance),
                    "complex128": dict(tolerance),
                },
                "source_auxiliary": {
                    "synthetic": {
                        "float64": dict(tolerance),
                        "complex128": dict(tolerance),
                    }
                },
            }
        }
    }


def _storage_resolved(loader, key: str, reference_dtype: str, precision: str):
    reference = loader.probe_array(key, _storage_values(key, reference_dtype))
    arrays = [reference]
    if key != SYNTHETIC_STRATEGY_KEY:
        arrays.append(loader.probe_array(SYNTHETIC_STRATEGY_KEY, SYNTHETIC_STRATEGY_VALUES))
    resolved = loader.ResolvedCase(
        "synthetic-v5-4", "synthetic", _storage_manifest(),
        {"device": "cpu", "precision": precision, "graph_mode": "eager", "compile_mode": "default"},
        1, {"arrays": arrays},
    )
    return reference, resolved


def _storage_candidate_arrays(key: str, value: np.ndarray) -> dict[str, np.ndarray]:
    arrays = {key: value}
    if key != SYNTHETIC_STRATEGY_KEY:
        arrays[SYNTHETIC_STRATEGY_KEY] = SYNTHETIC_STRATEGY_VALUES
    if "/source/" in key and key.endswith("/indices"):
        arrays[key.removesuffix("/indices") + "/values"] = np.asarray([0], dtype=np.uint64)
    return arrays


def assert_comparison_storage_dtype_contract(loader, bundle) -> None:
    """Exercise every V5.4 storage table row separately from tolerances."""
    rows = (
        ("map/Ex/material_ids", "int32", "int32", "int32", "int64"),
        ("step/1/source/Ex/0-Dielectric/indices", "int64", "int64", "int64", "int32"),
        ("step/1/source_aux_material/0/Ex/0-Dielectric/indices", "int64", "int64", "int64", "int32"),
        ("step/1/state/Ex/0-Dielectric/indices", "int64", "int64", "int64", "int32"),
        ("step/1/field/Ex", "float64", "float32", "float64", "float64"),
        ("step/1/field/Ey", "complex128", "complex64", "complex128", "complex128"),
        ("step/1/physical/spectrum/Ex", "float64", "float32", "float64", "float64"),
        ("step/1/physical/summary", "float64", "float64", "float64", "float32"),
        ("step/1/time", "float64", "float64", "float64", "float32"),
        ("step/1/source_aux/0-Synthetic/field/Ex", "float64", "float64", "float64", "float32"),
        ("step/1/source_aux/0-Synthetic/time", "float64", "float64", "float64", "float32"),
        ("step/1/source_aux_material/0/Ex/0-Dielectric/values", "complex128", "complex128", "complex128", "complex64"),
        ("step/1/state/Ex/0-Dielectric/values", "complex128", "complex128", "complex128", "complex64"),
    )
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "storage-contract.npz"
        for key, reference_dtype, expected32, expected64, wrong32 in rows:
            reference, resolved32 = _storage_resolved(loader, key, reference_dtype, "float32")
            _, resolved64 = _storage_resolved(loader, key, reference_dtype, "float64")
            assert loader.validate_storage_reference_dtype(key, reference[1])
            assert loader.comparison_storage_dtype(resolved32, key, reference[1]) == expected32
            assert loader.comparison_storage_dtype(resolved64, key, reference[1]) == expected64
            write_arrays(path, _storage_candidate_arrays(key, _storage_values(key, expected32)))
            result = loader.compare_candidate(path, resolved32)
            assert result["passed"], (key, expected32, result["failures"])
            write_arrays(path, _storage_candidate_arrays(key, _storage_values(key, wrong32)))
            assert not loader.compare_candidate_bytes(path.read_bytes(), resolved32)["passed"], (key, wrong32)
    primary_reference, primary = _storage_resolved(loader, "step/1/physical/summary", "float64", "float32")
    time_reference, clocks = _storage_resolved(loader, "step/1/time", "float64", "float32")
    state_reference, state = _storage_resolved(loader, SYNTHETIC_STRATEGY_KEY, "complex128", "float32")
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "tolerance-scope.npz"
        for key, reference, resolved in (("step/1/physical/summary", primary_reference, primary), ("step/1/time", time_reference, clocks), (SYNTHETIC_STRATEGY_KEY, state_reference, state)):
            values = _storage_values(key, "complex128" if reference[1] == "complex128" else "float64")
            values[0] += 2e-6
            write_arrays(path, _storage_candidate_arrays(key, values))
            result = loader.compare_candidate(path, resolved)
            assert not result["passed"], key
            numeric = [item for item in result["tolerance_results"] if item["key"] == key]
            assert numeric and {item["dtype"] for item in numeric} == {"float32"}, key
    for key in (
        "map/Ex/material_ids",
        "step/1/source/Ex/0-Dielectric/indices",
        "step/1/source_aux_material/0/Ex/0-Dielectric/indices",
        "step/1/state/Ex/0-Dielectric/indices",
    ):
        reference_dtype = "int32" if key.startswith("map/") else "int64"
        _, resolved = _storage_resolved(loader, key, reference_dtype, "float32")
        values = _storage_values(key, reference_dtype)
        for label, invalid in (
            ("uint", values.astype(np.uint64)),
            ("float", values.astype(np.float64)),
            ("bool", values.astype(bool)),
            ("rank", values.reshape(-1)),
            ("shape", values.reshape((1,) + values.shape)),
        ):
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / f"{label}.npz"
                write_arrays(path, _storage_candidate_arrays(key, invalid))
                assert not loader.compare_candidate(path, resolved)["passed"], (key, label)
        if key.endswith("/indices"):
            overflow = np.asarray([[2**63, 1, 2]], dtype=np.uint64)
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "overflow.npz"
                write_arrays(path, _storage_candidate_arrays(key, overflow))
                assert not loader.compare_candidate(path, resolved)["passed"], (key, "overflow")
    bindings = bundle.side["runtime_bindings"]
    assert len(bindings) == 56
    float32_bindings = [binding for binding in bindings if binding["runtime"]["precision"] == "float32"]
    assert len(float32_bindings) == 8
    legacy_categories = {"physical_summary": 0, "primary_time": 0, "state_values": 0}
    for binding in bindings:
        capture = bundle.fixture["captures"][binding["capture_id"]]
        resolved = loader.ResolvedCase(binding["profile"], binding["case"], {}, binding["runtime"], binding["test_line"], capture)
        for reference in capture["arrays"]:
            key, reference_dtype = reference[0], reference[1]
            expected = loader.comparison_storage_dtype(resolved, key, reference_dtype)
            assert isinstance(expected, str)
            if binding["runtime"]["precision"] != "float32" or reference_dtype.startswith(("bool", "int", "uint")):
                continue
            legacy = ("complex64" if reference_dtype.startswith("complex") else "float32")
            if "/source_aux/" in key or "/source_aux_material/" in key:
                legacy = "complex128" if reference_dtype.startswith("complex") else "float64"
            if legacy != expected:
                family = loader.validate_storage_reference_dtype(key, reference_dtype)
                assert family in legacy_categories, (key, family)
                legacy_categories[family] += 1
    assert legacy_categories == {"physical_summary": 26, "primary_time": 26, "state_values": 312}
    assert sum(legacy_categories.values()) == 364


def assert_projection_classifier(root: Path, loader, bundle) -> None:
    """Exercise the one opaque family and strict allowlist boundary."""
    records = [
        record
        for capture in bundle.fixture["captures"].values()
        for record in capture["arrays"]
    ]
    classes = [loader.projection_class(record[0]) for record in records]
    assert all(item in loader.EXPECTED_FAMILY_COUNTS for item in classes)
    counts = {name: classes.count(name) for name in loader.EXPECTED_FAMILY_COUNTS}
    assert counts == loader.EXPECTED_FAMILY_COUNTS
    assert len(records) == 4377
    assert counts["source_indices"] == 145
    assert len(records) + counts["source_indices"] == 4522
    expected = {
        "map/Ex/material_ids": "map",
        "step/1/field/Ex": "field",
        "step/1/physical/spectrum/Ex": "physical",
        "step/1/physical/summary": "physical",
        "step/0/time": "time",
        "step/0/source/Ex/0-PointSourceEx/indices": "source_indices",
        "step/0/source_aux/0-PointSourceEx/field/Ex": "source_aux",
        "step/0/source_aux/0-PointSourceEx/time": "source_aux",
        "step/0/source_aux_material/0/Ex/0-Dielectric/indices": "source_aux_material",
        "step/0/source_aux_material/0/Ex/0-Dielectric/values": "source_aux_material",
        "step/0/state/Ex/0-Dielectric/indices": "state",
        "step/0/state/Ex/0-Dielectric/values": "state",
        "step/0/source/Ex/0-PointSourceEx/values": "opaque_source_value",
    }
    for key, classification in expected.items():
        assert loader.projection_class(key) == classification, key
    for key in (
        "metadata.json",
        "torch/Ex/packed",
        "map/Ex/values",
        "step/0/source/Ex/0-PointSourceEx/value",
        "step/0/source/Ex/0-PointSourceEx/values/extra",
        "step/0/source/Ex/0-PointSourceEx/indices/extra",
        "step/0/source_aux/0-PointSourceEx/values",
        "step/0/source_aux_material/0/Ex/0-Dielectric/value",
        "step/0/state/Ex/0-Dielectric/unknown",
        "step/0/field/Ex/step_count",
        "step/0/source/Ex/0-PointSourceEx/source_time",
        "step/0/time_step",
        "step/0/unknown/Ex",
    ):
        assert loader.projection_class(key) is None, key
    source_index = "step/1/source/Ex/0-PointSourceEx/indices"
    source_values = source_index.removesuffix("/indices") + "/values"
    source_array = np.asarray([[0, 1, 2]], dtype=np.int64)
    _, _, resolved = synthetic_resolved(loader, capture={"arrays": [loader.probe_array(source_index, source_array)]})
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        good = base / "paired-source.npz"
        write_arrays(good, {source_index: source_array, source_values: np.asarray([0], dtype=np.uint64)})
        assert loader.compare_candidate(good, resolved)["passed"]
        missing = base / "missing-source-values.npz"
        write_arrays(missing, {source_index: source_array})
        assert not loader.compare_candidate(missing, resolved)["passed"]
        opaque_only = base / "opaque-only.npz"
        write_arrays(opaque_only, {source_values: np.asarray([0], dtype=np.uint64)})
        assert not loader.compare_candidate(opaque_only, resolved)["passed"]
        renamed = base / "renamed-source.npz"
        write_arrays(renamed, {source_index.removesuffix("/indices") + "/index": source_array, source_values: np.asarray([0], dtype=np.uint64)})
        assert not loader.compare_candidate(renamed, resolved)["passed"]
        extra = base / "extra-source-suffix.npz"
        write_arrays(extra, {source_index: source_array, source_values: np.asarray([0], dtype=np.uint64), source_index.removesuffix("/indices") + "/source_time": np.asarray([0.0])})
        assert not loader.compare_candidate(extra, resolved)["passed"]
    temporary, copied = copied_bundle(root)
    try:
        changed = json.loads((copied / "BUNDLE-MANIFEST.json").read_text())
        fixture = fixture_document(copied, changed)
        capture_id, capture = next(
            (capture_id, capture)
            for capture_id, capture in fixture["captures"].items()
            if any(loader.projection_class(record[0]) == "source_indices" for record in capture["arrays"])
        )
        source_record = next(record for record in capture["arrays"] if loader.projection_class(record[0]) == "source_indices")
        serialized_opaque = copy.deepcopy(source_record)
        serialized_opaque[0] = source_record[0].removesuffix("/indices") + "/values"
        capture["arrays"].append(serialized_opaque)
        capture["arrays"].sort(key=lambda record: record[0])
        capture["probe_projection"]["sha256"] = loader.projection_sha256(capture["profile"], capture["case"], capture["archive_schema_version"], capture["arrays"])
        for bindings in (fixture["runtime_bindings"], changed["runtime_bindings"]):
            for binding in bindings:
                if binding["capture_id"] == capture_id:
                    binding["probe_projection_sha256"] = capture["probe_projection"]["sha256"]
        anchors = rewrite_fixture(copied, changed, fixture)
        expect_value_error(lambda: loader.load_bundle(copied, **anchors), "serialized opaque direct-source descriptor")
    finally:
        temporary.cleanup()


def assert_privacy_and_shape(root: Path, loader, bundle) -> None:
    side, fixture = bundle.side, bundle.fixture
    assert (len(side["profiles"]), len(fixture["captures"]), sum(len(item["arrays"]) for item in fixture["captures"].values()), len(side["runtime_bindings"])) == (14, 35, 4377, 56)
    assert (side["bundle_schema_version"], side["fixture_schema_version"], fixture["fixture_schema_version"]) == (3, 8, 8)
    assert side["sampling"]["algorithm"] == "reference-selector-candidate-fixed-positions-v3"
    assert set(side["support"]) == {"package_assertions", "schema", "readme", "reproduction"}
    float64_lines = {1426, 1753, 1930, 1975, 1990, 2012}
    assert {item["test_line"] for item in side["runtime_bindings"] if item["runtime"]["precision"] == "float64" and item["test_line"] in float64_lines} == float64_lines
    cuda = [item for item in side["runtime_bindings"] if item["runtime"]["device"] == "cuda:0"]
    assert [item["runtime"] for item in cuda] == [{"device": "cuda:0", "precision": "float32", "graph_mode": "eager", "compile_mode": "default"}, {"device": "cuda:0", "precision": "float32", "graph_mode": "graph", "compile_mode": "reduce-overhead"}]
    assert [item["capture_steps"] for item in cuda] == [[100, 500], [100, 500]]
    assert hashlib.sha256(canonical(fixture["tolerances_numeric"])).hexdigest() == TOLERANCES_NUMERIC_SHA256
    assert not (root / ".transient").exists() and not list(root.rglob("*.npz")) and not list(root.rglob("__pycache__"))
    compressed = (root / side["fixture"]["file"]).read_bytes()
    assert b".npz" not in compressed and b"PK\x03\x04" not in compressed
    data = [gzip.decompress(compressed), (root / "BUNDLE-MANIFEST.json").read_bytes(), (root / "inputs/native_oracle_workloads.json").read_bytes(), *((root / descriptor["file"]).read_bytes() for descriptor in side["profiles"].values())]
    if any(FORBIDDEN.search(item.decode()) for item in data):
        raise AssertionError("privacy scan failed")
    for capture in fixture["captures"].values():
        assert set(capture) == {"profile", "case", "archive_schema_version", "probe_projection", "arrays"}
        assert set(capture["probe_projection"]) == {"algorithm", "sha256"}
        assert capture["probe_projection"]["algorithm"] == "validated-probe-projection-v5-storage-dtype-contract"
        assert capture["probe_projection"]["sha256"] == loader.projection_sha256(capture["profile"], capture["case"], capture["archive_schema_version"], capture["arrays"])
    records = [record for capture in fixture["captures"].values() for record in capture["arrays"]]
    numeric = [record for record in records if not record[1].startswith(("bool", "int", "uint"))]
    dynamic = [loader.reference_value_selected_positions(record) for record in numeric]
    assert (len(numeric), sum(bool(item) for item in dynamic), sum(len(item) for item in dynamic)) == (2564, 729, 771)


def assert_sdist_and_wheel(root: Path, loader, package) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary); sdist, wheel = base / "issue124-v5-4.tar.gz", base / "issue124-v5-4.whl"
        with tarfile.open(sdist, "w:gz") as archive:
            for name in package.REQUIRED:
                relative = name.removeprefix(f"{FIXTURE_ROOT}/")
                source = root / "historical_probe_loader.py" if name == "benchmarks/historical_probes.py" else root / relative
                info = tarfile.TarInfo(f"issue124-v5-4/{name}"); data = source.read_bytes(); info.size = len(data); archive.addfile(info, io.BytesIO(data))
        package.assert_sdist(sdist, trust=TRUST)
        with tarfile.open(sdist) as archive:
            archive.extractall(base / "extracted", filter="data")
        packaged = module(base / "extracted" / "issue124-v5-4" / "benchmarks/historical_probes.py", "packaged_historical_probes_selftest")
        packaged.load_bundle(base / "extracted" / "issue124-v5-4" / FIXTURE_ROOT, **TRUST)
        key, values, resolved = synthetic_resolved(packaged)
        candidate = base / "packaged-compare.npz"; write_candidate(candidate, key, values)
        assert packaged.compare_candidate(candidate, resolved)["passed"], "extracted public loader comparison failed"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("gmes/__init__.py", "")
        package.assert_wheel(wheel)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--bundle", type=Path, required=True); args = parser.parse_args(); root = args.bundle.resolve()
    if set(TRUST) != {"expected_manifest_sha256", "expected_manifest_bytes", "expected_fixture_sha256", "expected_fixture_bytes"}:
        raise ValueError("generator has not written final caller trust anchors")
    loader = module(root / "historical_probe_loader.py", "historical_probe_loader_selftest")
    bundle = loader.load_bundle(root, **TRUST)
    assert_privacy_and_shape(root, loader, bundle)
    assert_loader_negatives(root, loader, bundle)
    assert_runtime_authority(loader, bundle)
    assert_binding_schedule_validation(root, loader, bundle)
    assert_index_canonicalization(root, loader, bundle)
    assert_comparison_storage_dtype_contract(loader, bundle)
    assert_projection_classifier(root, loader, bundle)
    assert_comparator(loader)
    assert_reference_selector_regressions(loader)
    assert_nonzero_count_contract(loader, bundle)
    package = module(root / "package_integration_assertions.py", "package_assertions_selftest")
    assert_sdist_and_wheel(root, loader, package)
    print(json.dumps({"profiles": len(bundle.side["profiles"]), "captures": len(bundle.fixture["captures"]), "runtime_bindings": len(bundle.side["runtime_bindings"]), "result": "PASS"}, sort_keys=True))


if __name__ == "__main__":
    main()
