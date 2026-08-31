#!/usr/bin/env python3
"""Build strict, candidate-bound GPU differential evidence for issue #123."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import zipfile
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path, PurePosixPath

import numpy as np

from benchmarks import host_contract, native_oracle, torch_correctness
from benchmarks.host_contract import DEFAULT_MANIFEST

INDEX_KIND = "issue-123-differential-evidence"
INDEX_SCHEMA_VERSION = 1
MEDIA_TYPE_NPZ = "application/x-npz"
FIELD_ARRAYS = tuple(native_oracle.COMPONENT_NAMES)
SCOPES = ("paired-real", "single-gpu-cuda")
SINGLE_GPU_CASES = ("single-gpu-2d", "single-gpu-3d")
DESCRIPTOR_KEYS = {
    "path",
    "sha256",
    "size_bytes",
    "media_type",
    "candidate_evidence",
}
INDEX_KEYS = {
    "schema_version",
    "kind",
    "scope",
    "candidate_evidence",
    "required_cases",
    "cases",
    "passed",
}
RECORD_KEYS = {
    "case",
    "device",
    "reference",
    "candidate",
    "field_arrays",
    "persistent_arrays",
    "rtol",
    "atol",
    "maximum_abs_error",
    "maximum_relative_error",
    "passed",
}
MAX_INDEX_BYTES = 16 * 1024 * 1024
MAX_NPZ_MEMBERS = 512
MAX_NPZ_MEMBER_BYTES = 256 * 1024 * 1024
MAX_NPZ_TOTAL_BYTES = 512 * 1024 * 1024
MAX_NPZ_COMPRESSION_RATIO = 10_000.0


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _exact_keys(value, expected, label):
    _require(isinstance(value, dict), f"{label} must be an object")
    _require(set(value) == set(expected), f"{label} keys are not exact")


def _hex_string(value, width):
    return (
        isinstance(value, str)
        and len(value) == width
        and all(character in "0123456789abcdef" for character in value)
    )


def _candidate_projection(value):
    _require(isinstance(value, dict), "candidate evidence must be an object")
    projection = {
        name: value.get(name)
        for name in (
            "candidate_git_commit",
            "candidate_git_status",
            "manifest_sha256",
        )
    }
    _require(
        _hex_string(projection["candidate_git_commit"], 40)
        and projection["candidate_git_status"] == ""
        and _hex_string(projection["manifest_sha256"], 64),
        "candidate evidence has no clean portable three-key binding",
    )
    return projection


def _canonical_path(value):
    _require(
        isinstance(value, str)
        and bool(value)
        and "\\" not in value
        and "\x00" not in value
        and not (len(value) >= 2 and value[0].isalpha() and value[1] == ":"),
        "differential descriptor path is not canonical POSIX",
    )
    path = PurePosixPath(value)
    _require(
        not path.is_absolute()
        and path.as_posix() == value
        and all(part not in {"", ".", ".."} for part in path.parts),
        "differential descriptor path must be canonical and relative",
    )
    return path


def _descriptor_root(value):
    root = Path(value).resolve(strict=True)
    _require(root.is_dir(), "differential descriptor root must be a directory")
    return root


def _relative_path(path, root):
    resolved = Path(path).resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("differential artifact is outside descriptor root") from error
    value = PurePosixPath(*relative.parts).as_posix()
    _canonical_path(value)
    return value


def _descriptor(path, root, candidate):
    path = Path(path).resolve(strict=True)
    _require(path.is_file(), "differential artifact must be a regular file")
    raw = path.read_bytes()
    return {
        "path": _relative_path(path, root),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "media_type": MEDIA_TYPE_NPZ,
        "candidate_evidence": candidate,
    }


def _resolve_descriptor(root, descriptor, candidate):
    _exact_keys(descriptor, DESCRIPTOR_KEYS, "differential artifact descriptor")
    portable = _canonical_path(descriptor["path"])
    path = root.joinpath(*portable.parts)
    _require(not path.is_symlink(), "differential artifacts cannot be symlinks")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("differential artifact escapes descriptor root") from error
    _require(resolved.is_file(), "differential artifact must be a regular file")
    raw = resolved.read_bytes()
    _require(
        _hex_string(descriptor["sha256"], 64)
        and descriptor["sha256"] == hashlib.sha256(raw).hexdigest()
        and type(descriptor["size_bytes"]) is int
        and descriptor["size_bytes"] == len(raw)
        and descriptor["media_type"] == MEDIA_TYPE_NPZ
        and descriptor["candidate_evidence"] == candidate,
        "differential artifact descriptor does not bind exact bytes and candidate",
    )
    return resolved, raw


def _object_without_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json_bytes(raw, label):
    _require(len(raw) <= MAX_INDEX_BYTES, f"{label} exceeds the size limit")

    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        return json.loads(
            raw,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict JSON") from error


def expected_records(manifest, scope):
    """Return the frozen ordered workload/device closure for one scope."""
    _require(scope in SCOPES, f"unknown differential scope: {scope}")
    if scope == "paired-real":
        names = [
            case["name"]
            for case in manifest.get("correctness", ())
            if case.get("complex") is True
        ]
        _require(bool(names), "paired-real manifest case closure is empty")
        return [
            {"case": name, "device": device}
            for name in names
            for device in ("cpu", "cuda:0")
        ]
    manifest_names = {case.get("name") for case in manifest.get("benchmarks", ())}
    _require(
        set(SINGLE_GPU_CASES) <= manifest_names,
        "single-GPU differential workloads are absent from the manifest",
    )
    return [{"case": name, "device": "cuda:0"} for name in SINGLE_GPU_CASES]


def _workload(manifest, name):
    matches = [
        case
        for group in ("correctness", "benchmarks", "physical_checks")
        for case in manifest.get(group, ())
        if case.get("name") == name
    ]
    _require(len(matches) == 1, f"manifest workload {name!r} is not unique")
    return matches[0]


def _model_names(workload):
    recipe = workload.get("recipe")
    material = workload.get("material")
    if recipe in {"coverage", "mixed", "heterogeneous"}:
        return (
            "dielectric",
            "pml",
            "drude",
            "lorentz",
            "dcp-ade",
            "dcp-plrc",
            "dcp-rc",
            "dm2",
        )
    if material in {"upml", "cpml"}:
        return ("dielectric", "pml")
    if isinstance(material, str):
        if material.startswith("drude-"):
            return ("drude",)
        if material.startswith("lorentz-"):
            return ("lorentz",)
        if material.startswith("dm2-"):
            return ("dm2",)
        if material in {"dcp-ade", "dcp-plrc", "dcp-rc"}:
            return (material,)
    return ("dielectric",)


def frozen_tolerance(manifest, scope, case, device):
    """Resolve the immutable workload/device tolerance from the manifest."""
    _require(
        {"case": case, "device": device} in expected_records(manifest, scope),
        "differential workload/device is outside the frozen scope",
    )
    dtype = "complex128" if scope == "paired-real" and device == "cpu" else "float32"
    values = []
    for model in _model_names(_workload(manifest, case)):
        try:
            model_dtype = (
                "float64" if model == "dm2" and dtype == "complex128" else dtype
            )
            record = manifest["tolerances"]["torch"][model][model_dtype]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"manifest has no pinned {model_dtype} tolerance for {model}"
            ) from error
        _exact_keys(record, {"rtol", "atol"}, f"{model} {model_dtype} tolerance")
        parsed = {name: float(record[name]) for name in ("rtol", "atol")}
        _require(
            all(math.isfinite(value) and value >= 0 for value in parsed.values()),
            f"manifest {model} {model_dtype} tolerance is invalid",
        )
        values.append(parsed)
    return {name: max(record[name] for record in values) for name in ("rtol", "atol")}


def _expected_field_dtype(scope, device):
    if scope == "paired-real":
        return np.dtype("complex128" if device == "cpu" else "complex64")
    return np.dtype("float32")


def _safe_projection_name(value):
    _require(isinstance(value, str) and bool(value), "projection name is empty")
    path = PurePosixPath(value)
    _require(
        not path.is_absolute()
        and path.as_posix() == value
        and all(part not in {"", ".", ".."} for part in path.parts),
        "projection array name is not canonical",
    )
    return value


def _persistent_source_keys(archive, step):
    prefixes = (f"step/{step}/state/", f"step/{step}/source/")
    auxiliary_prefix = f"step/{step}/source_aux/"
    keys = sorted(
        key
        for key in archive.files
        if key.startswith(prefixes) or key.startswith(auxiliary_prefix)
    )
    _require(bool(keys), "differential source has no persistent arrays")
    return keys


def _projected_name(key, step):
    prefix = f"step/{step}/"
    _require(key.startswith(prefix), "source array is outside the final capture step")
    return _safe_projection_name(f"persistent/{key[len(prefix):]}")


def _numeric_array(value, label):
    array = np.asarray(value)
    _require(
        array.dtype.fields is None
        and array.dtype.subdtype is None
        and array.dtype.kind in {"b", "i", "u", "f", "c"}
        and array.ndim <= 8
        and array.flags.c_contiguous,
        f"{label} is not a plain contiguous numeric array",
    )
    _require(np.isfinite(array).all(), f"{label} contains non-finite values")
    return array


def _projection_arrays(reference_path, candidate_path, manifest, scope, expected):
    reference_path, reference_metadata = torch_correctness._archive_record(
        reference_path, manifest, "reference"
    )
    candidate_path, candidate_metadata = torch_correctness._archive_record(
        candidate_path, manifest, "candidate"
    )
    case = expected["case"]
    device = expected["device"]
    _require(
        reference_metadata["workload"]
        == candidate_metadata["workload"]
        == _workload(manifest, case),
        "differential source workload differs from the manifest",
    )
    runtime = candidate_metadata["backend_metadata"]
    expected_precision = "float64" if device == "cpu" else "float32"
    _require(
        runtime["device"] == device
        and runtime["resolved_device"] == device
        and runtime["precision"] == expected_precision
        and runtime["graph_mode"] == "eager"
        and runtime["compile_policy"] == "eager"
        and runtime["compile_mode"] == "default",
        "differential source runtime mode differs from the frozen contract",
    )
    candidate_evidence = candidate_metadata["provenance"]["source"]
    comparison = torch_correctness.compare_torch_archives(
        reference_path, candidate_path, manifest
    )
    _require(
        comparison == {"passed": True, "failures": []},
        "differential source archives fail full correctness validation",
    )
    step = str(reference_metadata["capture_steps"][-1])
    reference_arrays = {}
    candidate_arrays = {}
    with ExitStack() as stack:
        reference = stack.enter_context(np.load(reference_path, allow_pickle=False))
        candidate = stack.enter_context(np.load(candidate_path, allow_pickle=False))
        reference_keys = _persistent_source_keys(reference, step)
        candidate_keys = _persistent_source_keys(candidate, step)
        _require(
            reference_keys == candidate_keys,
            "differential persistent source topology differs",
        )
        key_pairs = [(name, f"step/{step}/field/{name}") for name in FIELD_ARRAYS] + [
            (_projected_name(key, step), key) for key in reference_keys
        ]
        for projected, source in key_pairs:
            left = _numeric_array(reference[source], f"reference {source}")
            right = _numeric_array(candidate[source], f"candidate {source}")
            _require(left.shape == right.shape, f"{source} shape differs")
            if right.dtype.kind in {"b", "i", "u"}:
                _require(
                    left.dtype.kind in {"b", "i", "u"} and np.array_equal(left, right),
                    f"{source} exact integer values differ",
                )
            reference_arrays[projected] = np.ascontiguousarray(
                left.astype(right.dtype, copy=False)
            )
            candidate_arrays[projected] = np.ascontiguousarray(right)
    field_dtype = _expected_field_dtype(scope, device)
    _require(
        all(candidate_arrays[name].dtype == field_dtype for name in FIELD_ARRAYS),
        "differential field dtype differs from the frozen scope",
    )
    return (
        reference_arrays,
        candidate_arrays,
        candidate_evidence,
        [name for name in reference_arrays if name not in FIELD_ARRAYS],
    )


def _write_npz(path, arrays):
    path = Path(path)
    _require(not path.exists(), f"differential output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _compare_arrays(reference, candidate, rtol, atol):
    maximum_abs = 0.0
    maximum_relative = 0.0
    passed = True
    for name in reference:
        left = reference[name]
        right = candidate[name]
        _require(
            left.shape == right.shape and left.dtype == right.dtype,
            f"differential projection array {name} shape or dtype differs",
        )
        _require(
            np.isfinite(left).all() and np.isfinite(right).all(),
            f"differential projection array {name} contains non-finite values",
        )
        difference = np.abs(right - left)
        maximum_abs = max(maximum_abs, float(np.max(difference, initial=0.0)))
        denominator = np.maximum(
            np.abs(left), atol if atol > 0 else np.finfo(float).tiny
        )
        maximum_relative = max(
            maximum_relative,
            float(np.max(difference / denominator, initial=0.0)),
        )
        if left.dtype.kind in {"b", "i", "u"}:
            passed = passed and bool(np.array_equal(left, right))
        else:
            passed = passed and bool(np.allclose(right, left, rtol=rtol, atol=atol))
    return maximum_abs, maximum_relative, passed


def build_differential_evidence(
    reference_paths,
    candidate_paths,
    manifest,
    candidate_evidence,
    *,
    scope,
    descriptor_root,
    output_directory,
):
    """Validate full source archives and write deterministic compact projections."""
    candidate = _candidate_projection(candidate_evidence)
    descriptor_root = _descriptor_root(descriptor_root)
    supplied_output_directory = Path(output_directory)
    _require(
        not supplied_output_directory.is_symlink(),
        "differential output directory cannot be a symlink",
    )
    output_directory = supplied_output_directory.resolve()
    try:
        output_directory.relative_to(descriptor_root)
    except ValueError as error:
        raise ValueError(
            "differential output directory escapes descriptor root"
        ) from error
    if output_directory.exists():
        _require(
            output_directory.is_dir() and not any(output_directory.iterdir()),
            "differential output directory must be empty",
        )
    else:
        output_directory.mkdir(parents=True)
    references = {}
    for path in reference_paths:
        resolved, metadata = torch_correctness._archive_record(
            path, manifest, "reference"
        )
        name = metadata["workload"]["name"]
        _require(name not in references, f"duplicate differential reference: {name}")
        references[name] = resolved
    candidates = {}
    for path in candidate_paths:
        resolved, metadata = torch_correctness._archive_record(
            path, manifest, "candidate"
        )
        key = (metadata["workload"]["name"], metadata["backend_metadata"]["device"])
        _require(key not in candidates, f"duplicate differential candidate: {key}")
        provenance = metadata["provenance"]
        _require(
            all(
                provenance[name]["commit"] == candidate["candidate_git_commit"]
                and provenance[name]["git_status"] == ""
                and provenance[name]["clean"] is True
                for name in ("source", "controller")
            ),
            "differential candidate archive provenance differs",
        )
        candidates[key] = resolved
    required = expected_records(manifest, scope)
    required_names = {record["case"] for record in required}
    required_keys = {(record["case"], record["device"]) for record in required}
    _require(
        set(references) == required_names and set(candidates) == required_keys,
        "differential source archive closure differs",
    )
    records = []
    for expected in required:
        name = expected["case"]
        device = expected["device"]
        left, right, _source, persistent = _projection_arrays(
            references[name], candidates[(name, device)], manifest, scope, expected
        )
        suffix = device.replace(":", "-")
        reference_output = output_directory / f"{name}-{suffix}-reference.npz"
        candidate_output = output_directory / f"{name}-{suffix}-candidate.npz"
        _write_npz(reference_output, left)
        _write_npz(candidate_output, right)
        tolerance = frozen_tolerance(manifest, scope, name, device)
        maximum_abs, maximum_relative, passed = _compare_arrays(
            left, right, tolerance["rtol"], tolerance["atol"]
        )
        _require(passed, f"differential projection failed for {name} on {device}")
        records.append(
            {
                "case": name,
                "device": device,
                "reference": _descriptor(reference_output, descriptor_root, candidate),
                "candidate": _descriptor(candidate_output, descriptor_root, candidate),
                "field_arrays": list(FIELD_ARRAYS),
                "persistent_arrays": persistent,
                **tolerance,
                "maximum_abs_error": maximum_abs,
                "maximum_relative_error": maximum_relative,
                "passed": True,
            }
        )
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "kind": INDEX_KIND,
        "scope": scope,
        "candidate_evidence": candidate,
        "required_cases": required,
        "cases": records,
        "passed": True,
    }


def _preflight_npz(raw, names, label):
    expected = [f"{name}.npy" for name in names]
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = archive.infolist()
            _require(
                0 < len(members) <= MAX_NPZ_MEMBERS,
                f"{label} NPZ member count is invalid",
            )
            actual = [member.filename for member in members]
            _require(actual == expected, f"{label} NPZ array closure differs")
            _require(len(actual) == len(set(actual)), f"{label} repeats NPZ members")
            total = 0
            for member in members:
                _safe_projection_name(member.filename.removesuffix(".npy"))
                _require(
                    not member.is_dir()
                    and 0 <= member.file_size <= MAX_NPZ_MEMBER_BYTES
                    and member.compress_size >= 0,
                    f"{label} NPZ member exceeds bounds",
                )
                if member.file_size:
                    _require(
                        member.compress_size > 0
                        and member.file_size / member.compress_size
                        <= MAX_NPZ_COMPRESSION_RATIO,
                        f"{label} NPZ compression ratio exceeds the bound",
                    )
                total += member.file_size
            _require(total <= MAX_NPZ_TOTAL_BYTES, f"{label} NPZ is too large")
    except zipfile.BadZipFile as error:
        raise ValueError(f"{label} is not a readable NPZ archive") from error


def _load_projection(raw, names, label):
    _preflight_npz(raw, names, label)
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            _require(archive.files == names, f"{label} NPZ array order differs")
            arrays = {
                name: _numeric_array(archive[name], f"{label} {name}") for name in names
            }
    except (EOFError, MemoryError, OSError, TypeError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith(label):
            raise
        raise ValueError(f"{label} is not a safe numeric NPZ archive") from error
    _require(
        sum(array.nbytes for array in arrays.values()) <= MAX_NPZ_TOTAL_BYTES,
        f"{label} uncompressed arrays exceed the bound",
    )
    return arrays


def validate_differential_document(
    document,
    manifest,
    candidate_evidence,
    *,
    descriptor_root,
    expected_scope=None,
    artifact_loader=None,
):
    """Reopen every projection and recompute the complete differential result."""
    candidate = _candidate_projection(candidate_evidence)
    root = _descriptor_root(descriptor_root)
    _exact_keys(document, INDEX_KEYS, "differential index")
    scope = document["scope"]
    _require(
        document["schema_version"] == INDEX_SCHEMA_VERSION
        and type(document["schema_version"]) is int
        and document["kind"] == INDEX_KIND
        and scope in SCOPES
        and (expected_scope is None or scope == expected_scope)
        and document["candidate_evidence"] == candidate,
        "differential index identity differs",
    )
    required = expected_records(manifest, scope)
    _require(
        document["required_cases"] == required, "differential case contract differs"
    )
    records = document["cases"]
    _require(
        isinstance(records, list)
        and [
            {"case": record.get("case"), "device": record.get("device")}
            for record in records
            if isinstance(record, dict)
        ]
        == required,
        "differential evaluated case closure differs",
    )
    used_paths = set()
    recomputed_records = []
    for index, (record, expected) in enumerate(zip(records, required, strict=True)):
        label = f"{scope} differential case {index}"
        _exact_keys(record, RECORD_KEYS, label)
        _require(
            record["case"] == expected["case"]
            and record["device"] == expected["device"],
            f"{label} identity differs",
        )
        fields = record["field_arrays"]
        persistent = record["persistent_arrays"]
        _require(fields == list(FIELD_ARRAYS), f"{label} field closure differs")
        _require(
            isinstance(persistent, list)
            and bool(persistent)
            and persistent == sorted(set(persistent))
            and all(
                _safe_projection_name(name).startswith("persistent/")
                for name in persistent
            ),
            f"{label} persistent array closure differs",
        )
        names = fields + persistent
        _require(len(names) == len(set(names)), f"{label} repeats array names")
        if artifact_loader is None:
            reference_path, reference_raw = _resolve_descriptor(
                root, record["reference"], candidate
            )
            candidate_path, candidate_raw = _resolve_descriptor(
                root, record["candidate"], candidate
            )
        else:
            reference_path, reference_raw = artifact_loader(
                record["reference"], f"{label} reference"
            )
            candidate_path, candidate_raw = artifact_loader(
                record["candidate"], f"{label} candidate"
            )
            reference_path = Path(reference_path).resolve(strict=True)
            candidate_path = Path(candidate_path).resolve(strict=True)
            _require(
                isinstance(reference_raw, bytes) and isinstance(candidate_raw, bytes),
                f"{label} artifact loader did not return exact bytes",
            )
        _require(
            reference_path != candidate_path
            and record["reference"]["path"] not in used_paths
            and record["candidate"]["path"] not in used_paths,
            f"{label} reuses an artifact path",
        )
        used_paths.update((record["reference"]["path"], record["candidate"]["path"]))
        left = _load_projection(reference_raw, names, f"{label} reference")
        right = _load_projection(candidate_raw, names, f"{label} candidate")
        expected_dtype = _expected_field_dtype(scope, expected["device"])
        _require(
            all(
                left[name].dtype == right[name].dtype == expected_dtype
                for name in FIELD_ARRAYS
            ),
            f"{label} field dtype differs from the frozen runtime",
        )
        tolerance = frozen_tolerance(
            manifest, scope, expected["case"], expected["device"]
        )
        _require(
            type(record["rtol"]) is float
            and type(record["atol"]) is float
            and record["rtol"] == tolerance["rtol"]
            and record["atol"] == tolerance["atol"],
            f"{label} tolerance differs from the manifest",
        )
        maximum_abs, maximum_relative, passed = _compare_arrays(
            left, right, tolerance["rtol"], tolerance["atol"]
        )
        _require(
            isinstance(record["maximum_abs_error"], (int, float))
            and not isinstance(record["maximum_abs_error"], bool)
            and math.isclose(
                float(record["maximum_abs_error"]),
                maximum_abs,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            and isinstance(record["maximum_relative_error"], (int, float))
            and not isinstance(record["maximum_relative_error"], bool)
            and math.isclose(
                float(record["maximum_relative_error"]),
                maximum_relative,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            and record["passed"] is passed
            and passed,
            f"{label} recomputed differential failed",
        )
        recomputed_records.append(
            {
                "case": expected["case"],
                "device": expected["device"],
                "rtol": tolerance["rtol"],
                "atol": tolerance["atol"],
                "maximum_abs_error": maximum_abs,
                "maximum_relative_error": maximum_relative,
                "passed": passed,
            }
        )
    _require(document["passed"] is True, "differential suite pass is false")
    return {
        "scope": scope,
        "candidate_evidence": candidate,
        "cases": recomputed_records,
        "passed": True,
    }


def load_differential_evidence_index(
    path,
    manifest,
    candidate_evidence,
    *,
    descriptor_root,
    expected_scope=None,
):
    """Strictly load and independently recompute a differential index."""
    path = Path(path).resolve(strict=True)
    _require(
        path.is_file() and not path.is_symlink(), "differential index is not regular"
    )
    document = _strict_json_bytes(path.read_bytes(), "differential index")
    result = validate_differential_document(
        document,
        manifest,
        candidate_evidence,
        descriptor_root=descriptor_root,
        expected_scope=expected_scope,
    )
    result["document"] = deepcopy(document)
    result["source_artifact"] = {
        "path": _relative_path(path, _descriptor_root(descriptor_root)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
        "media_type": "application/json",
        "candidate_evidence": _candidate_projection(candidate_evidence),
    }
    return result


def _load_candidate_evidence(path):
    value = _strict_json_bytes(Path(path).read_bytes(), "candidate evidence")
    if isinstance(value, dict) and "evidence" in value:
        value = value["evidence"]
    return _candidate_projection(value)


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    commands = parser.add_subparsers(dest="command", required=True)
    candidate = commands.add_parser("candidate")
    candidate.add_argument("--output", type=Path, required=True)
    build = commands.add_parser("build")
    build.add_argument("--scope", choices=SCOPES, required=True)
    build.add_argument("--references", type=Path, nargs="+", required=True)
    build.add_argument("--candidates", type=Path, nargs="+", required=True)
    build.add_argument("--candidate-evidence", type=Path, required=True)
    build.add_argument("--descriptor-root", type=Path, required=True)
    build.add_argument("--output-directory", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--scope", choices=SCOPES, required=True)
    validate.add_argument("--candidate-evidence", type=Path, required=True)
    validate.add_argument("--descriptor-root", type=Path, required=True)
    validate.add_argument("--index", type=Path, required=True)
    return parser.parse_args()


def main():
    args = _arguments()
    manifest = native_oracle.load_manifest(args.manifest)
    if args.command == "candidate":
        _require(not args.output.exists(), "candidate evidence already exists")
        value = _candidate_projection(host_contract.candidate_evidence(args.manifest))
        rendered = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 0
    candidate = _load_candidate_evidence(args.candidate_evidence)
    _require(
        candidate["manifest_sha256"]
        == hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        "candidate evidence manifest bytes differ",
    )
    if args.command == "build":
        _require(not args.output.exists(), "differential index already exists")
        value = build_differential_evidence(
            args.references,
            args.candidates,
            manifest,
            candidate,
            scope=args.scope,
            descriptor_root=args.descriptor_root,
            output_directory=args.output_directory,
        )
        rendered = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        value = load_differential_evidence_index(
            args.index,
            manifest,
            candidate,
            descriptor_root=args.descriptor_root,
            expected_scope=args.scope,
        )
    print(json.dumps(value, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
