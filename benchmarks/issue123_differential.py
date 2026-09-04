#!/usr/bin/env python3
"""Build strict, candidate-bound GPU differential evidence for issue #123."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import stat
import zipfile
import zlib
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path, PurePosixPath

import numpy as np

from benchmarks import host_contract, native_oracle, torch_correctness
from benchmarks.host_contract import DEFAULT_MANIFEST

INDEX_KIND = "issue-123-differential-evidence"
INDEX_SCHEMA_VERSION = 5
MEDIA_TYPE_NPZ = "application/x-npz"
FIELD_ARRAYS = tuple(native_oracle.COMPONENT_NAMES)
SCOPES = ("paired-real", "single-gpu-cuda")
TRUSTED_MANIFEST_SHA256 = (
    "0766dbf932882dfec7a40abfbcd78eb67978ed8cd65e38625193a16502cc29a9"
)
PAIRED_CASES = (
    "bloch-2d",
    "bloch-3d",
    "upml-bloch",
    "cpml-bloch",
    "lorentz-bloch",
    "dcp-ade-bloch",
    "dcp-plrc-bloch",
    "dcp-rc-bloch",
)
SINGLE_GPU_CASES = ("single-gpu-2d", "single-gpu-3d")
SINGLE_GPU_PRECISION_BY_CASE = {
    "single-gpu-2d": "float32",
    "single-gpu-3d": "float64",
}
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
    "precision",
    "projection_steps",
    "projection_groups",
    "reference",
    "candidate",
    "reference_source",
    "candidate_source",
    "field_arrays",
    "physical_arrays",
    "persistent_arrays",
    "contract_arrays",
    "comparison",
    "precision_limitation",
    "metrics",
    "passed",
}
ELEMENTWISE_COMPARISON_MODE = "elementwise-allclose-v1"
NORMALIZED_COMPARISON_MODE = "normalized-linf-l2-v1"
NORMALIZED_RESIDUAL_LIMIT = 1e-6
# The reviewed late-step residual contract retains this denominator floor even
# when the active early-step updater union has a smaller elementwise tolerance.
NORMALIZED_ABSOLUTE_SCALE_FLOOR = 2e-12
NORMALIZED_RESIDUAL_CASE = ("single-gpu-cuda", "single-gpu-3d", "cuda:0")
NORMALIZED_STRICT_STEPS = (0, 1, 2, 5)
NORMALIZED_RESIDUAL_STEPS = (20, 100)
NORMALIZED_PROJECTION_GROUPS = ((0, 1), (2, 5), (20, 100))
NORMALIZED_PROJECTION_STEPS = tuple(
    step for group in NORMALIZED_PROJECTION_GROUPS for step in group
)
FROZEN_CAPTURE_STEPS = (1, 2, 5, 20, 100)
PHYSICAL_ARRAY_SUFFIXES = tuple(
    f"physical/spectrum/{name}" for name in FIELD_ARRAYS
) + ("physical/summary",)
SOURCE_CONTRACT_ARRAY = "persistent/source/semantic-contract.json"
SOURCE_CONTRACT_SCHEMA = "point-source-semantic-v1"
SOURCE_PROOF_ARRAY = "persistent/source/raw-proof.json"
SOURCE_PROOF_SCHEMA = "point-source-raw-proof-v2"
SOURCE_PREIMAGE_SCHEMA = "point-source-role-preimage-v1"
GROUP_ARCHIVE_SCHEMA = "issue-123-differential-group-npz-v1"
POINT_SOURCE_LIVE_ARRAYS = (
    "overwrite_targets",
    "overwrite_models",
    "overwrite_parameters",
    "overwrite_amplitudes",
    "_overwrite_values",
    "additive_targets",
    "additive_models",
    "additive_parameters",
    "additive_amplitudes",
    "_additive_values",
)
CASE_UPDATER_LABELS = {
    "bloch-2d": ("0-Dielectric", "1-Dummy"),
    "bloch-3d": ("0-Drude", "1-Dummy"),
    "upml-bloch": ("0-Dielectric", "1-Dummy", "2-Upml"),
    "cpml-bloch": ("0-Cpml", "1-Dielectric", "2-Dummy"),
    "lorentz-bloch": ("0-Dummy", "1-Lorentz"),
    "dcp-ade-bloch": ("0-DcpAde", "1-Dummy"),
    "dcp-plrc-bloch": ("0-DcpPlrc", "1-Dummy"),
    "dcp-rc-bloch": ("0-DcpRc", "1-Dummy"),
    "single-gpu-2d": (
        "0-Cpml",
        "1-DcpAde",
        "2-DcpPlrc+DcpRc",
        "3-Dielectric",
        "4-Dm2",
        "5-Drude",
        "6-Dummy",
        "7-Lorentz",
    ),
    "single-gpu-3d": (
        "0-Cpml",
        "1-DcpAde",
        "2-DcpPlrc+DcpRc",
        "3-Dielectric",
        "4-Drude",
        "5-Dummy",
        "6-Lorentz",
    ),
}
FROZEN_PERSISTENT_GEOMETRY_SHA256_BY_CASE = {
    "bloch-2d": "c9f3fa5ff684335642cebf6e3ec08a5bf3d6a55ac2bd0a6cd9aba4ea4b3cf492",
    "bloch-3d": "f761f31aa18052eee53c4d2729b1e7e889daa6a45896d4e33303254c07a9ea4f",
    "upml-bloch": "f65f673b73e423993004ab41ec258f61d2b39e7c8f18d21bbfe2d0a6833ff53f",
    "cpml-bloch": "53ba4c204c0049a162b2062413da5c072548cfbb047173830bc0b9c07c99c94f",
    "lorentz-bloch": "57b24806b7909d4e80ea9e53ae1ed51230345248cd19488f28ddd18b92877195",
    "dcp-ade-bloch": "b8f74ed09738c270ccf51844e2b0620107f913b87d56f00852fdb0add477204f",
    "dcp-plrc-bloch": "213b7620064d703cd50c3d375eb0ba6d89a19b40dc19c4e4ae2a5a949b078bde",
    "dcp-rc-bloch": "4f59c27cf0dd16f7ade60f143460857865538506919fd687396151309a78a245",
    "single-gpu-2d": "69afb7d7a8d902eadfa82c133a3db50ec0b54396cc1ffdde097fb3ae090b74b1",
    "single-gpu-3d": "2fa7d81021a44428f361db1707b5192089b4ac7fa92285e306b20a6a8944b765",
}
STRATEGY_TOLERANCE_MODELS = {
    "Cpml": ("pml",),
    "DcpAde": ("dcp-ade",),
    "DcpPlrc": ("dcp-plrc",),
    "DcpPlrc+DcpRc": ("dcp-plrc", "dcp-rc"),
    "DcpRc": ("dcp-rc",),
    "Dielectric": ("dielectric",),
    "Dm2": ("dm2",),
    "Drude": ("drude",),
    "Dummy": (),
    "Lorentz": ("lorentz",),
    "Upml": ("pml",),
}
FROZEN_TORCH_TOLERANCES = {
    "dcp-ade": {
        "float32": {"rtol": 6e-4, "atol": 1e-4},
        "float64": {"rtol": 5e-12, "atol": 5e-13},
        "complex128": {"rtol": 1e-11, "atol": 1e-12},
    },
    "dcp-plrc": {
        "float32": {"rtol": 6e-4, "atol": 1e-4},
        "float64": {"rtol": 5e-12, "atol": 5e-13},
        "complex128": {"rtol": 1e-11, "atol": 1e-12},
    },
    "dcp-rc": {
        "float32": {"rtol": 6e-4, "atol": 1e-4},
        "float64": {"rtol": 5e-12, "atol": 5e-13},
        "complex128": {"rtol": 1e-11, "atol": 1e-12},
    },
    "dielectric": {
        "float32": {"rtol": 3e-5, "atol": 3e-6},
        "float64": {"rtol": 1e-13, "atol": 1e-14},
        "complex64": {"rtol": 3e-5, "atol": 3e-6},
        "complex128": {"rtol": 2e-13, "atol": 2e-14},
    },
    "dm2": {
        "float32": {"rtol": 6e-4, "atol": 3e-6},
        "float64": {"rtol": 2e-10, "atol": 2e-12},
    },
    "drude": {
        "float32": {"rtol": 6e-4, "atol": 1e-4},
        "float64": {"rtol": 5e-12, "atol": 5e-13},
        "complex128": {"rtol": 6e-12, "atol": 6e-13},
    },
    "lorentz": {
        "float32": {"rtol": 6e-4, "atol": 1e-4},
        "float64": {"rtol": 5e-12, "atol": 5e-13},
        "complex128": {"rtol": 6e-12, "atol": 6e-13},
    },
    "pml": {
        "float32": {"rtol": 5e-5, "atol": 5e-6},
        "float64": {"rtol": 2e-12, "atol": 2e-13},
        "complex64": {"rtol": 8e-5, "atol": 8e-6},
        "complex128": {"rtol": 4e-12, "atol": 4e-13},
    },
}
PRECISION_LIMITATION_CONTRACT_ID = "single-gpu-3d-float32-dynamic-range-review-v1"
PRECISION_LIMITATION_REASON = "native-step-100-magnitude-exceeds-float32-range"
PRECISION_LIMITATION_KEYS = {
    "contract_id",
    "rejected_precision",
    "accepted_precision",
    "reference_step",
    "reference_field_max_abs",
    "rejected_precision_max",
    "range_exceeded",
    "reason",
}
POINT_SOURCE_FREQUENCY = 0.35
POINT_SOURCE_PHASE = 0.0
POINT_SOURCE_START = 0.0
POINT_SOURCE_END = math.inf
POINT_SOURCE_WIDTH = 5.0 / POINT_SOURCE_FREQUENCY
POINT_SOURCE_VALUE_ULP_FACTOR = 256
POINT_SOURCE_COURANT_RATIO = 0.99
MAX_INDEX_BYTES = 16 * 1024 * 1024
MAX_NPZ_MEMBERS = 512
MAX_NPZ_MEMBER_BYTES = 256 * 1024 * 1024
MAX_NPZ_TOTAL_BYTES = 512 * 1024 * 1024
MAX_NPZ_COMPRESSION_RATIO = 10_000.0
MAX_NPY_HEADER_BYTES = 64 * 1024
MAX_NPZ_ARCHIVE_BYTES = (
    MAX_NPZ_TOTAL_BYTES + MAX_NPZ_MEMBERS * (MAX_NPY_HEADER_BYTES + 1024) + 65535
)
MAX_SOURCE_NPZ_ARCHIVE_BYTES = 1024 * 1024 * 1024
MAX_SOURCE_NPZ_MEMBERS = 8192
MAX_SOURCE_NPZ_MEMBER_BYTES = 256 * 1024 * 1024
MAX_SOURCE_NPZ_TOTAL_BYTES = 8 * 1024 * 1024 * 1024
MAX_SOURCE_METADATA_BYTES = 16 * 1024 * 1024
MAX_SOURCE_PROOF_BYTES = 1024 * 1024
MAX_SOURCE_PROOF_ARRAY_BYTES = 64 * 1024


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
    _require(
        projection["manifest_sha256"] == TRUSTED_MANIFEST_SHA256,
        "candidate evidence is not bound to the trusted repository manifest",
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
    raw = _bounded_regular_file_bytes(
        path,
        MAX_NPZ_ARCHIVE_BYTES,
        "differential artifact",
    )
    return {
        "path": _relative_path(path, root),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "media_type": MEDIA_TYPE_NPZ,
        "candidate_evidence": candidate,
    }


def _source_descriptor(path, root, candidate):
    path = Path(path).resolve(strict=True)
    _require(path.is_file(), "differential source must be a regular file")
    raw = _bounded_regular_file_bytes(
        path,
        MAX_SOURCE_NPZ_ARCHIVE_BYTES,
        "differential source archive",
    )
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
    raw = _bounded_regular_file_bytes(
        resolved,
        MAX_NPZ_ARCHIVE_BYTES,
        "differential artifact",
    )
    _require(
        len(raw) == descriptor["size_bytes"],
        "differential artifact exceeds the NPZ archive size bound",
    )
    _validate_descriptor_payload(descriptor, raw, candidate)
    return resolved, raw


def _validate_descriptor_payload(descriptor, raw, candidate):
    _exact_keys(descriptor, DESCRIPTOR_KEYS, "differential artifact descriptor")
    _canonical_path(descriptor["path"])
    _require(
        _hex_string(descriptor["sha256"], 64)
        and descriptor["sha256"] == hashlib.sha256(raw).hexdigest()
        and type(descriptor["size_bytes"]) is int
        and 0 < descriptor["size_bytes"] <= MAX_NPZ_ARCHIVE_BYTES
        and descriptor["size_bytes"] == len(raw)
        and descriptor["media_type"] == MEDIA_TYPE_NPZ
        and descriptor["candidate_evidence"] == candidate,
        "differential artifact descriptor does not bind exact bytes and candidate",
    )


def _validate_source_descriptor_payload(descriptor, raw, candidate):
    _exact_keys(descriptor, DESCRIPTOR_KEYS, "differential source descriptor")
    _canonical_path(descriptor["path"])
    _require(
        type(raw) is bytes
        and _hex_string(descriptor["sha256"], 64)
        and descriptor["sha256"] == hashlib.sha256(raw).hexdigest()
        and type(descriptor["size_bytes"]) is int
        and 0 < descriptor["size_bytes"] <= MAX_SOURCE_NPZ_ARCHIVE_BYTES
        and descriptor["size_bytes"] == len(raw)
        and descriptor["media_type"] == MEDIA_TYPE_NPZ
        and descriptor["candidate_evidence"] == candidate,
        "differential source descriptor does not bind exact bytes and candidate",
    )


def _resolve_source_descriptor(root, descriptor, candidate):
    _exact_keys(descriptor, DESCRIPTOR_KEYS, "differential source descriptor")
    portable = _canonical_path(descriptor["path"])
    supplied = root.joinpath(*portable.parts)
    _require(not supplied.is_symlink(), "differential sources cannot be symlinks")
    resolved = supplied.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("differential source escapes descriptor root") from error
    raw = _bounded_regular_file_bytes(
        resolved,
        MAX_SOURCE_NPZ_ARCHIVE_BYTES,
        "differential source archive",
    )
    _validate_source_descriptor_payload(descriptor, raw, candidate)
    return resolved, raw


def _load_bound_descriptor(
    root,
    descriptor,
    candidate,
    label,
    artifact_loader,
    *,
    source,
):
    if artifact_loader is None:
        resolver = _resolve_source_descriptor if source else _resolve_descriptor
        return resolver(root, descriptor, candidate)
    path, raw = artifact_loader(descriptor, label)
    path = Path(path).resolve(strict=True)
    _require(
        type(raw) is bytes,
        f"{label} artifact loader did not return exact bytes",
    )
    validator = (
        _validate_source_descriptor_payload if source else _validate_descriptor_payload
    )
    validator(descriptor, raw, candidate)
    return path, raw


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


def _bounded_regular_file_bytes(path, limit, label):
    supplied = Path(path)
    _require(not supplied.is_symlink(), f"{label} cannot be a symlink")
    resolved = supplied.resolve(strict=True)
    _require(resolved.is_file(), f"{label} must be a regular file")
    before = _source_archive_identity(resolved)
    _require(
        0 < before[2] <= limit,
        f"{label} exceeds the size limit",
    )
    try:
        with resolved.open("rb") as stream:
            raw = stream.read(limit + 1)
    except (MemoryError, OSError, OverflowError) as error:
        raise ValueError(f"{label} could not be read safely") from error
    _require(
        len(raw) == before[2] <= limit and _source_archive_identity(resolved) == before,
        f"{label} changed while it was read",
    )
    return raw


def expected_records(manifest, scope):
    """Return the frozen ordered workload/device closure for one scope."""
    _require(scope in SCOPES, f"unknown differential scope: {scope}")
    _validate_frozen_manifest_contract(manifest)
    if scope == "paired-real":
        manifest_names = [
            case["name"]
            for case in manifest.get("correctness", ())
            if case.get("complex") is True
        ]
        _require(
            manifest_names == list(PAIRED_CASES),
            "paired-real manifest identity differs from the frozen case closure",
        )
        return [
            {
                "case": name,
                "device": device,
                "precision": "float64" if device == "cpu" else "float32",
            }
            for name in PAIRED_CASES
            for device in ("cpu", "cuda:0")
        ]
    manifest_names = {case.get("name") for case in manifest.get("benchmarks", ())}
    _require(
        set(SINGLE_GPU_CASES) <= manifest_names,
        "single-GPU differential workloads are absent from the manifest",
    )
    return [
        {
            "case": name,
            "device": "cuda:0",
            "precision": SINGLE_GPU_PRECISION_BY_CASE[name],
        }
        for name in SINGLE_GPU_CASES
    ]


def _validate_frozen_manifest_contract(manifest):
    _require(isinstance(manifest, dict), "differential manifest must be an object")
    _require(
        [
            case.get("name")
            for case in manifest.get("correctness", ())
            if isinstance(case, dict) and case.get("complex") is True
        ]
        == list(PAIRED_CASES),
        "paired-real manifest identity differs from the frozen case closure",
    )
    for name in (*PAIRED_CASES, *SINGLE_GPU_CASES):
        workload = _workload(manifest, name)
        steps = workload.get(
            "capture_steps", manifest.get("reference", {}).get("capture_steps")
        )
        _require(
            steps == list(FROZEN_CAPTURE_STEPS),
            f"manifest capture steps for {name} differ from the frozen contract",
        )
    for model, expected_dtypes in FROZEN_TORCH_TOLERANCES.items():
        try:
            actual_dtypes = manifest["tolerances"]["torch"][model]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"manifest has no frozen tolerance contract for {model}"
            ) from error
        _require(
            actual_dtypes == expected_dtypes,
            f"manifest {model} tolerance contract differs from the frozen values",
        )


def _workload(manifest, name):
    matches = [
        case
        for group in ("correctness", "benchmarks", "physical_checks")
        for case in manifest.get(group, ())
        if case.get("name") == name
    ]
    _require(len(matches) == 1, f"manifest workload {name!r} is not unique")
    return matches[0]


def _frozen_capture_steps(manifest, case):
    steps = _workload(manifest, case).get(
        "capture_steps", manifest["reference"]["capture_steps"]
    )
    _require(
        isinstance(steps, list)
        and bool(steps)
        and all(type(step) is int and step > 0 for step in steps)
        and steps == sorted(set(steps)),
        "differential capture-step contract is malformed",
    )
    _require(
        steps == list(FROZEN_CAPTURE_STEPS),
        "differential capture steps differ from the frozen contract",
    )
    return list(FROZEN_CAPTURE_STEPS)


def frozen_projection_step(manifest, case):
    """Resolve the final required capture step, including a case override."""
    steps = _frozen_capture_steps(manifest, case)
    return steps[-1]


def frozen_projection_steps(manifest, scope, case, device):
    """Resolve the ordered capture-step projection for one frozen record."""
    return [
        step
        for group in frozen_projection_groups(manifest, scope, case, device)
        for step in group
    ]


def frozen_projection_groups(manifest, scope, case, device):
    """Resolve exact bounded compact-projection groups for one record."""
    steps = _frozen_capture_steps(manifest, case)
    if (scope, case, device) == NORMALIZED_RESIDUAL_CASE:
        _require(
            all(step in steps for step in NORMALIZED_PROJECTION_STEPS if step != 0),
            "normalized projection steps differ from the manifest",
        )
        return [list(group) for group in NORMALIZED_PROJECTION_GROUPS]
    return [[steps[-1]]]


def _active_tolerance_models(case):
    try:
        labels = CASE_UPDATER_LABELS[case]
    except KeyError as error:
        raise ValueError(f"no frozen updater topology for {case!r}") from error
    models = []
    for label in labels:
        strategy = label.split("-", 1)[-1]
        try:
            strategy_models = STRATEGY_TOLERANCE_MODELS[strategy]
        except KeyError as error:
            raise ValueError(f"no frozen tolerance model for {strategy!r}") from error
        for model in strategy_models:
            if model not in models:
                models.append(model)
    return tuple(models)


def _comparison_dtype(scope, expected):
    return (
        "complex128"
        if scope == "paired-real" and expected["device"] == "cpu"
        else expected["precision"]
    )


def _model_tolerance(manifest, models, dtype):
    if not models:
        return {"rtol": 0.0, "atol": 0.0}
    values = []
    for model in models:
        try:
            model_dtype = (
                "float64" if model == "dm2" and dtype == "complex128" else dtype
            )
            record = manifest["tolerances"]["torch"][model][model_dtype]
            frozen_record = FROZEN_TORCH_TOLERANCES[model][model_dtype]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"manifest has no pinned {model_dtype} tolerance for {model}"
            ) from error
        _exact_keys(record, {"rtol", "atol"}, f"{model} {model_dtype} tolerance")
        _require(
            record == frozen_record
            and all(
                type(record[name]) in {int, float}
                and math.isfinite(float(record[name]))
                and float(record[name]) >= 0
                for name in ("rtol", "atol")
            ),
            f"manifest {model} {model_dtype} tolerance differs from the frozen value",
        )
        values.append({name: float(frozen_record[name]) for name in ("rtol", "atol")})
    return {name: max(record[name] for record in values) for name in ("rtol", "atol")}


def frozen_tolerance(manifest, scope, case, device):
    """Resolve the immutable workload/device tolerance from the manifest."""
    _require(
        any(
            record["case"] == case and record["device"] == device
            for record in expected_records(manifest, scope)
        ),
        "differential workload/device is outside the frozen scope",
    )
    expected = next(
        record
        for record in expected_records(manifest, scope)
        if record["case"] == case and record["device"] == device
    )
    return _model_tolerance(
        manifest,
        _active_tolerance_models(case),
        _comparison_dtype(scope, expected),
    )


def frozen_comparison_contract(manifest, scope, case, device):
    """Return the immutable comparison rule for one differential record."""
    tolerance = frozen_tolerance(manifest, scope, case, device)
    if (scope, case, device) == NORMALIZED_RESIDUAL_CASE:
        return {
            "mode": NORMALIZED_COMPARISON_MODE,
            "linf_limit": NORMALIZED_RESIDUAL_LIMIT,
            "l2_limit": NORMALIZED_RESIDUAL_LIMIT,
            "absolute_scale_floor": NORMALIZED_ABSOLUTE_SCALE_FLOOR,
            "all_zero_reference": "exact",
        }
    return {
        "mode": ELEMENTWISE_COMPARISON_MODE,
        "rtol": tolerance["rtol"],
        "atol": tolerance["atol"],
    }


def _elementwise_comparison_contract(manifest, scope, case, device):
    tolerance = frozen_tolerance(manifest, scope, case, device)
    return {
        "mode": ELEMENTWISE_COMPARISON_MODE,
        "rtol": tolerance["rtol"],
        "atol": tolerance["atol"],
    }


def _projection_array_step(name):
    parts = name.split("/", 2)
    if (
        len(parts) == 3
        and parts[0] == "step"
        and parts[1].isdigit()
        and str(int(parts[1])) == parts[1]
    ):
        return int(parts[1])
    return None


def frozen_array_comparison_contract(manifest, scope, case, device, name):
    """Resolve a model-scoped immutable comparison rule for one projected array."""
    comparison = frozen_comparison_contract(manifest, scope, case, device)
    if (scope, case, device) == NORMALIZED_RESIDUAL_CASE:
        step = _projection_array_step(name)
        _require(
            step in NORMALIZED_PROJECTION_STEPS
            or name
            in {
                SOURCE_CONTRACT_ARRAY,
                SOURCE_PROOF_ARRAY,
            },
            f"no frozen normalized projection step for {name!r}",
        )
        if step in NORMALIZED_STRICT_STEPS:
            comparison = _elementwise_comparison_contract(manifest, scope, case, device)
        else:
            return comparison
    parts = name.split("/")
    if len(parts) != 6 or parts[2] != "state" or parts[-1] != "values":
        return comparison
    strategy = parts[4].split("-", 1)[-1]
    try:
        models = STRATEGY_TOLERANCE_MODELS[strategy]
    except KeyError as error:
        raise ValueError(f"no frozen tolerance model for {strategy!r}") from error
    expected = next(
        record
        for record in expected_records(manifest, scope)
        if record["case"] == case and record["device"] == device
    )
    tolerance = _model_tolerance(manifest, models, _comparison_dtype(scope, expected))
    return {
        "mode": ELEMENTWISE_COMPARISON_MODE,
        "rtol": tolerance["rtol"],
        "atol": tolerance["atol"],
    }


def _frozen_array_comparisons(manifest, scope, case, device, names):
    return {
        name: frozen_array_comparison_contract(manifest, scope, case, device, name)
        for name in names
    }


def _expected_field_dtype(scope, case, device):
    if scope == "paired-real":
        return np.dtype("complex128" if device == "cpu" else "complex64")
    return np.dtype(SINGLE_GPU_PRECISION_BY_CASE[case])


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
    prefixes = (
        f"step/{step}/state/",
        f"step/{step}/source_aux/",
        f"step/{step}/source_aux_material/",
    )
    keys = sorted(key for key in archive.files if key.startswith(prefixes))
    _require(bool(keys), "differential source has no persistent arrays")
    return keys


def _expected_persistent_suffixes(case):
    try:
        labels = CASE_UPDATER_LABELS[case]
    except KeyError as error:
        raise ValueError(f"no frozen persistent inventory for {case!r}") from error
    return sorted(
        f"state/{component}/{label}/{kind}"
        for component in FIELD_ARRAYS
        for label in labels
        for kind in ("indices", "values")
    )


def _whole_shape(workload):
    size = workload.get("size")
    resolution = workload.get("resolution")
    _require(
        isinstance(size, list)
        and len(size) == 3
        and all(
            type(length) in {int, float}
            and not isinstance(length, bool)
            and math.isfinite(float(length))
            and float(length) >= 0.0
            for length in size
        )
        and type(resolution) in {int, float}
        and not isinstance(resolution, bool)
        and math.isfinite(float(resolution))
        and float(resolution) > 0.0,
        "differential workload grid contract is invalid",
    )
    shape = tuple(
        1 if float(length) == 0.0 else int(np.rint(float(length) * resolution))
        for length in size
    )
    _require(
        all(value > 0 for value in shape),
        "differential workload grid shape is invalid",
    )
    return shape


def _expected_field_shapes(workload):
    nx, ny, nz = _whole_shape(workload)
    return {
        "Ex": (nx, ny + 1, nz + 1),
        "Ey": (nx + 1, ny, nz + 1),
        "Ez": (nx + 1, ny + 1, nz),
        "Hx": (nx, ny + 1, nz + 1),
        "Hy": (nx + 1, ny, nz + 1),
        "Hz": (nx + 1, ny + 1, nz),
    }


def _strategy_pole_count(workload, prefix):
    names = [workload.get("material", ""), *workload.get("families", ())]
    return 4 if f"{prefix}-4" in names else 1


def _persistent_state_width(workload, component, updater_label):
    strategy = updater_label.split("-", 1)[-1]
    if strategy == "Cpml":
        return 2
    if strategy == "Upml":
        return 1
    if strategy in {"Dielectric", "Dummy"} or component.startswith("H"):
        return 0
    if strategy == "Drude":
        return 2 * _strategy_pole_count(workload, "drude")
    if strategy == "Lorentz":
        return 2 * _strategy_pole_count(workload, "lorentz")
    if strategy == "DcpAde":
        return 7
    if strategy in {"DcpPlrc", "DcpPlrc+DcpRc", "DcpRc"}:
        return 6
    if strategy == "Dm2":
        _require(
            _strategy_pole_count(workload, "dm2") == 1,
            "no frozen differential state width exists for multi-transition DM2",
        )
        return 3
    raise ValueError(f"no frozen persistent-state width for {strategy!r}")


def _float_token(value):
    value = float(value)
    if math.isinf(value):
        return {"kind": "infinity", "sign": 1 if value > 0 else -1}
    _require(math.isfinite(value), "PointSource semantic value is non-finite")
    return {"kind": "finite", "hex": value.hex()}


def _expected_point_source_medium(workload):
    recipe = workload.get("recipe")
    material = workload.get("material")
    if recipe == "coverage" or material in {"dielectric", "upml", "cpml"}:
        return (1.7, 1.05)
    if (
        recipe == "homogeneous"
        and isinstance(material, str)
        and material.startswith(("drude-", "lorentz-", "dcp-", "dm2-"))
    ):
        return (1.2, 1.0)
    raise ValueError("PointSource source-cell medium is outside the frozen contract")


def _expected_point_source_time_step(workload):
    resolution = float(workload["resolution"])
    eps_inf, mu_inf = _expected_point_source_medium(workload)
    value = (
        POINT_SOURCE_COURANT_RATIO
        * math.sqrt(eps_inf * mu_inf)
        / (resolution * math.sqrt(3.0))
    )
    _require(
        math.isfinite(value) and value > 0.0,
        "PointSource workload has no finite Courant time step",
    )
    return value


def _expected_point_source_contract(workload):
    _require(
        workload.get("source", "point") == "point"
        and workload.get("source_component", "Ex") == "Ex",
        "differential source semantic contract only supports the frozen Ex PointSource",
    )
    resolution = float(workload["resolution"])
    whole_shape = [
        1 if float(length) == 0 else int(np.rint(float(length) * resolution))
        for length in workload["size"]
    ]
    indices = [
        whole_shape[0] // 2,
        0 if whole_shape[1] == 1 else int(math.floor(whole_shape[1] / 2 + 0.5)),
        0 if whole_shape[2] == 1 else int(math.floor(whole_shape[2] / 2 + 0.5)),
    ]
    parameters = (
        POINT_SOURCE_FREQUENCY,
        POINT_SOURCE_PHASE,
        POINT_SOURCE_START,
        POINT_SOURCE_END,
        POINT_SOURCE_WIDTH,
        0.0,
    )
    return {
        "schema": SOURCE_CONTRACT_SCHEMA,
        "workload": workload["name"],
        "schedule": "yee-point-source-overwrite-v1",
        "sources": [
            {
                "component": "Ex",
                "native_type": "PointSourceEx",
                "target_index": indices,
                "operation": "overwrite",
                "model": {"id": 0, "name": "Continuous"},
                "parameters": [_float_token(value) for value in parameters],
                "amplitude": _float_token(workload.get("source_amp", 1e-3)),
                "source_cell_medium": {
                    "eps_inf": _float_token(_expected_point_source_medium(workload)[0]),
                    "mu_inf": _float_token(_expected_point_source_medium(workload)[1]),
                },
            }
        ],
    }


def _canonical_source_bytes(value):
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _continuous_point_source_value(time, paired_real):
    ts = time - POINT_SOURCE_START
    te = POINT_SOURCE_END - time
    if ts < 0 or te < 0:
        return 0j if paired_real else 0.0
    rise = (
        math.sin(0.5 * math.pi * ts / POINT_SOURCE_WIDTH) ** 2
        if ts < POINT_SOURCE_WIDTH
        else 1.0
    )
    fall = (
        math.sin(0.5 * math.pi * te / POINT_SOURCE_WIDTH) ** 2
        if te < POINT_SOURCE_WIDTH
        else 1.0
    )
    angle = 2.0 * math.pi * POINT_SOURCE_FREQUENCY * time + POINT_SOURCE_PHASE
    value = rise * fall * complex(math.cos(angle), math.sin(angle))
    return value if paired_real else value.real


def _point_source_value_matches(actual, expected, dtype, scale):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        return False
    if np.all(expected == 0):
        return bool(np.array_equal(actual, expected))
    real_dtype = np.empty((), dtype=dtype).real.dtype
    tolerance = (
        POINT_SOURCE_VALUE_ULP_FACTOR
        * np.finfo(real_dtype).eps
        * max(float(np.max(np.abs(expected))), abs(float(scale)))
    )
    return bool(np.all(np.abs(actual - expected) <= tolerance))


def _validate_native_point_source(archive, metadata, workload, expected):
    expected_medium = _expected_point_source_medium(workload)
    for step in ("0", *(str(value) for value in metadata["capture_steps"])):
        records = metadata["steps"][step]["sources"]["updaters"]
        _require(len(records) == 1, "native PointSource closure differs")
        record = records[0]
        source = expected["sources"][0]
        _require(
            {name: record[name] for name in ("component", "native_type", "cells")}
            == {
                "component": source["component"],
                "native_type": source["native_type"],
                "cells": 1,
            },
            "native PointSource topology differs",
        )
        _require(
            record["state_values"] == 4,
            "native PointSource state-value closure differs",
        )
        prefix = f"step/{step}/source/Ex/0-PointSourceEx"
        indices = _numeric_array(archive[f"{prefix}/indices"], "native source indices")
        values = _numeric_array(archive[f"{prefix}/values"], "native source values")
        time = _numeric_array(archive[f"step/{step}/time"], "native source time")
        amplitude = float(workload.get("source_amp", 1e-3))
        oscillator = _continuous_point_source_value(
            float(time[1]), bool(workload.get("complex"))
        )
        _require(
            indices.shape == (1, 3)
            and indices.dtype == np.dtype(np.intc)
            and indices.tolist() == [source["target_index"]],
            "native PointSource target differs from the workload",
        )
        _require(
            time.shape == (3,)
            and time.dtype == np.dtype("float64")
            and values.shape == (4,)
            and values.dtype == np.dtype("complex128")
            and _point_source_value_matches(
                values[0], complex(amplitude), np.dtype("complex128"), amplitude
            )
            and _point_source_value_matches(
                values[1],
                complex(expected_medium[0]),
                np.dtype("complex128"),
                expected_medium[0],
            )
            and _point_source_value_matches(
                values[2],
                complex(expected_medium[1]),
                np.dtype("complex128"),
                expected_medium[1],
            )
            and _point_source_value_matches(
                values[3], oscillator, np.dtype("complex128"), 1.0
            ),
            "native PointSource amplitude or waveform differs from the workload",
        )
    return _canonical_source_bytes(expected)


def _validate_candidate_point_source(archive, metadata, workload, expected):
    precision = np.dtype(metadata["backend_metadata"]["precision"])
    source = expected["sources"][0]
    expected_parameters = np.asarray(
        [
            POINT_SOURCE_FREQUENCY,
            POINT_SOURCE_PHASE,
            POINT_SOURCE_START,
            POINT_SOURCE_END,
            POINT_SOURCE_WIDTH,
            0.0,
        ],
        dtype=precision,
    )
    expected_amplitude = np.asarray(workload.get("source_amp", 1e-3), dtype=precision)
    channels = 2 if workload.get("complex") else 1
    suffixes = set(POINT_SOURCE_LIVE_ARRAYS)
    for step in ("0", *(str(value) for value in metadata["capture_steps"])):
        records = metadata["steps"][step]["sources"]["updaters"]
        _require(len(records) == 1, "Torch PointSource closure differs")
        record = records[0]
        _require(
            record["component"] == source["component"]
            and record["native_type"] == source["native_type"]
            and record["cells"] == 1,
            "Torch PointSource topology differs",
        )
        _require(
            record["state_values"] == 9,
            "Torch PointSource packed state-value closure differs",
        )
        batches_root = f"torch/step/{step}/sources/batches/"
        batch_ordinals = {
            key.removeprefix(batches_root).split("/", 1)[0]
            for key in archive.files
            if key.startswith(batches_root)
        }
        _require(batch_ordinals == {"0"}, "Torch PointSource batch closure differs")
        root = f"{batches_root}0/"
        actual_suffixes = {
            key.removeprefix(root) for key in archive.files if key.startswith(root)
        }
        _require(actual_suffixes == suffixes, "Torch PointSource live buffers differ")
        values = {
            name: np.asarray(archive[f"{root}{name}"])
            for name in POINT_SOURCE_LIVE_ARRAYS
        }
        time = _numeric_array(archive[f"step/{step}/time"], "Torch source time")
        evaluated_time = float(time[1]) - 0.5 * float(time[2])
        oscillator = _continuous_point_source_value(
            evaluated_time, bool(workload.get("complex"))
        )
        expected_evaluated = expected_amplitude.item() * np.asarray(
            (
                [complex(oscillator).real, complex(oscillator).imag]
                if channels == 2
                else [float(oscillator)]
            )
        )
        _require(
            time.shape == (3,)
            and time.dtype == np.dtype("float64")
            and values["overwrite_targets"].dtype == np.dtype("int64")
            and values["overwrite_targets"].shape == (1,)
            and values["overwrite_models"].dtype == np.dtype("int8")
            and values["overwrite_models"].shape == (1,)
            and int(values["overwrite_models"][0]) == source["model"]["id"]
            and values["overwrite_parameters"].dtype == precision
            and values["overwrite_parameters"].shape == (1, 6)
            and np.array_equal(values["overwrite_parameters"][0], expected_parameters)
            and values["overwrite_amplitudes"].dtype == precision
            and values["overwrite_amplitudes"].shape == (1,)
            and np.array_equal(values["overwrite_amplitudes"][0], expected_amplitude)
            and values["_overwrite_values"].dtype == precision
            and values["_overwrite_values"].shape == (1, channels)
            and _point_source_value_matches(
                values["_overwrite_values"][0],
                expected_evaluated,
                precision,
                expected_amplitude.item(),
            ),
            "Torch PointSource overwrite semantics differ from the workload",
        )
        for name, shape, dtype in (
            ("additive_targets", (0,), np.dtype("int64")),
            ("additive_models", (0,), np.dtype("int8")),
            ("additive_parameters", (0, 6), precision),
            ("additive_amplitudes", (0,), precision),
            ("_additive_values", (0, channels), precision),
        ):
            _require(
                values[name].shape == shape and values[name].dtype == dtype,
                "Torch additive source is unexpected",
            )
        field_shape = archive[f"step/{step}/field/Ex"].shape
        target = np.unravel_index(int(values["overwrite_targets"][0]), field_shape)
        packed_root = f"step/{step}/source/Ex/0-PointSourceEx"
        packed_indices = _numeric_array(
            archive[f"{packed_root}/indices"], "Torch packed source indices"
        )
        packed_values = _numeric_array(
            archive[f"{packed_root}/values"], "Torch packed source values"
        )
        expected_packed_values = (
            np.concatenate(
                (
                    np.asarray(
                        [0.0, float(values["overwrite_models"][0])],
                        dtype=np.float64,
                    ),
                    values["overwrite_parameters"][0].astype(np.float64),
                    values["overwrite_amplitudes"].astype(np.float64),
                )
            )
            .astype("<f8", copy=False)
            .view("<u8")
        )
        _require(
            list(target) == source["target_index"]
            and packed_indices.dtype == np.dtype("int64")
            and packed_indices.shape == (1, 3)
            and packed_indices.tolist() == [source["target_index"]]
            and packed_values.dtype == np.dtype("<u8")
            and packed_values.shape == (9,)
            and np.array_equal(packed_values, expected_packed_values),
            "Torch PointSource packed/live semantics differ from the workload",
        )
    return _canonical_source_bytes(expected)


def _point_source_contract_arrays(
    reference, candidate, reference_metadata, candidate_metadata, workload
):
    expected = _expected_point_source_contract(workload)
    reference_payload = _validate_native_point_source(
        reference, reference_metadata, workload, expected
    )
    candidate_payload = _validate_candidate_point_source(
        candidate, candidate_metadata, workload, expected
    )
    return (
        np.frombuffer(reference_payload, dtype=np.uint8).copy(),
        np.frombuffer(candidate_payload, dtype=np.uint8).copy(),
    )


def _source_array_record(value, label):
    array = np.asarray(value)
    _require(
        array.dtype.fields is None
        and array.dtype.subdtype is None
        and array.dtype.kind in {"b", "i", "u", "f", "c"}
        and array.ndim <= 8
        and array.flags.c_contiguous
        and array.nbytes <= MAX_SOURCE_PROOF_ARRAY_BYTES,
        f"{label} is outside the raw source proof array contract",
    )
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "data_hex": array.tobytes(order="C").hex(),
    }


def _source_array_from_record(value, label):
    _exact_keys(value, {"dtype", "shape", "data_hex"}, label)
    dtype_name = value["dtype"]
    _require(
        isinstance(dtype_name, str) and 0 < len(dtype_name) <= 16,
        f"{label} dtype is invalid",
    )
    try:
        dtype = np.dtype(dtype_name)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} dtype is invalid") from error
    _require(
        dtype.str == dtype_name
        and dtype.fields is None
        and dtype.subdtype is None
        and dtype.kind in {"b", "i", "u", "f", "c"}
        and 0 < dtype.itemsize <= 16,
        f"{label} dtype is outside the raw source proof contract",
    )
    shape = value["shape"]
    _require(
        isinstance(shape, list)
        and len(shape) <= 8
        and all(type(size) is int and 0 <= size <= 2**31 - 1 for size in shape),
        f"{label} shape is invalid",
    )
    size_bytes = math.prod(shape) * dtype.itemsize
    data_hex = value["data_hex"]
    _require(
        size_bytes <= MAX_SOURCE_PROOF_ARRAY_BYTES
        and _hex_string(data_hex, size_bytes * 2),
        f"{label} exact bytes are invalid",
    )
    return np.frombuffer(bytes.fromhex(data_hex), dtype=dtype).reshape(shape).copy()


def _point_source_field_shape(workload):
    resolution = float(workload["resolution"])
    whole_shape = [
        1 if float(length) == 0 else int(np.rint(float(length) * resolution))
        for length in workload["size"]
    ]
    return (whole_shape[0], whole_shape[1] + 1, whole_shape[2] + 1)


def _source_role_preimage_sha256(workload, role, captures):
    _require(role in {"reference", "candidate"}, "PointSource proof role is invalid")
    capture_role = "native" if role == "reference" else "candidate"
    payload = {
        "schema": SOURCE_PREIMAGE_SCHEMA,
        "workload": workload["name"],
        "role": role,
        "captures": [
            {"step": capture["step"], "arrays": capture[capture_role]}
            for capture in captures
        ],
    }
    return hashlib.sha256(_canonical_source_bytes(payload)).hexdigest()


def _build_point_source_raw_proof(
    reference,
    candidate,
    reference_metadata,
    candidate_metadata,
    workload,
):
    capture_steps = reference_metadata["capture_steps"]
    _require(
        candidate_metadata["capture_steps"] == capture_steps,
        "PointSource raw proof capture steps differ",
    )
    captures = []
    for captured_step in (0, *capture_steps):
        step = str(captured_step)
        native_root = f"step/{step}/source/Ex/0-PointSourceEx"
        packed_root = f"step/{step}/source/Ex/0-PointSourceEx"
        live_root = f"torch/step/{step}/sources/batches/0"
        captures.append(
            {
                "step": captured_step,
                "native": {
                    "time": _source_array_record(
                        reference[f"step/{step}/time"], "native source time"
                    ),
                    "indices": _source_array_record(
                        reference[f"{native_root}/indices"],
                        "native source indices",
                    ),
                    "values": _source_array_record(
                        reference[f"{native_root}/values"], "native source values"
                    ),
                },
                "candidate": {
                    "time": _source_array_record(
                        candidate[f"step/{step}/time"], "candidate source time"
                    ),
                    "packed_indices": _source_array_record(
                        candidate[f"{packed_root}/indices"],
                        "candidate packed source indices",
                    ),
                    "packed_values": _source_array_record(
                        candidate[f"{packed_root}/values"],
                        "candidate packed source values",
                    ),
                    "live": {
                        name: _source_array_record(
                            candidate[f"{live_root}/{name}"],
                            f"candidate live source {name}",
                        )
                        for name in POINT_SOURCE_LIVE_ARRAYS
                    },
                },
            }
        )
    proof = {
        "schema": SOURCE_PROOF_SCHEMA,
        "workload": workload["name"],
        "reference_preimage_sha256": _source_role_preimage_sha256(
            workload, "reference", captures
        ),
        "candidate_preimage_sha256": _source_role_preimage_sha256(
            workload, "candidate", captures
        ),
        "captures": captures,
    }
    _require(
        proof["reference_preimage_sha256"] != proof["candidate_preimage_sha256"],
        "native and candidate PointSource raw preimages are identical",
    )
    return proof


class _ProofArchive(dict):
    @property
    def files(self):
        return list(self)


class _ProofField:
    def __init__(self, shape):
        self.shape = shape


def _validate_source_raw_proof_bytes(
    raw,
    workload,
    capture_steps,
    precision,
    precondition_steps,
):
    _require(
        isinstance(raw, bytes) and 0 < len(raw) <= MAX_SOURCE_PROOF_BYTES,
        "projected PointSource raw proof exceeds the size bound",
    )
    proof = _strict_json_bytes(raw, "projected PointSource raw proof")
    _exact_keys(
        proof,
        {
            "schema",
            "workload",
            "reference_preimage_sha256",
            "candidate_preimage_sha256",
            "captures",
        },
        "projected PointSource raw proof",
    )
    reference_digest = proof["reference_preimage_sha256"]
    candidate_digest = proof["candidate_preimage_sha256"]
    _require(
        proof["schema"] == SOURCE_PROOF_SCHEMA
        and proof["workload"] == workload["name"]
        and _hex_string(reference_digest, 64)
        and _hex_string(candidate_digest, 64)
        and reference_digest != candidate_digest
        and raw == _canonical_source_bytes(proof),
        "projected PointSource raw proof identity differs",
    )
    expected_steps = [0, *capture_steps]
    _require(
        type(precondition_steps) is int and precondition_steps > 0,
        "PointSource raw proof precondition-step contract is invalid",
    )
    captures = proof["captures"]
    _require(
        isinstance(captures, list)
        and len(captures) == len(expected_steps)
        and [capture.get("step") for capture in captures if isinstance(capture, dict)]
        == expected_steps,
        "projected PointSource raw proof capture closure differs",
    )
    native = _ProofArchive()
    candidate = _ProofArchive()
    native_steps = {}
    candidate_steps = {}
    field_shape = _point_source_field_shape(workload)
    expected_time_step = _expected_point_source_time_step(workload)
    for capture, captured_step in zip(captures, expected_steps, strict=True):
        label = f"PointSource raw proof step {captured_step}"
        _exact_keys(capture, {"step", "native", "candidate"}, label)
        _require(
            type(capture["step"]) is int and capture["step"] == captured_step,
            f"{label} identity differs",
        )
        native_record = capture["native"]
        candidate_record = capture["candidate"]
        _exact_keys(native_record, {"time", "indices", "values"}, f"{label} native")
        _exact_keys(
            candidate_record,
            {"time", "packed_indices", "packed_values", "live"},
            f"{label} candidate",
        )
        live = candidate_record["live"]
        _exact_keys(live, set(POINT_SOURCE_LIVE_ARRAYS), f"{label} candidate live")
        step = str(captured_step)
        native_root = f"step/{step}/source/Ex/0-PointSourceEx"
        packed_root = f"step/{step}/source/Ex/0-PointSourceEx"
        live_root = f"torch/step/{step}/sources/batches/0"
        native[f"step/{step}/time"] = _source_array_from_record(
            native_record["time"], f"{label} native time"
        )
        native[f"{native_root}/indices"] = _source_array_from_record(
            native_record["indices"], f"{label} native indices"
        )
        native[f"{native_root}/values"] = _source_array_from_record(
            native_record["values"], f"{label} native values"
        )
        candidate[f"step/{step}/time"] = _source_array_from_record(
            candidate_record["time"], f"{label} candidate time"
        )
        candidate[f"{packed_root}/indices"] = _source_array_from_record(
            candidate_record["packed_indices"], f"{label} candidate packed indices"
        )
        candidate[f"{packed_root}/values"] = _source_array_from_record(
            candidate_record["packed_values"], f"{label} candidate packed values"
        )
        for name in POINT_SOURCE_LIVE_ARRAYS:
            candidate[f"{live_root}/{name}"] = _source_array_from_record(
                live[name], f"{label} candidate live {name}"
            )
        candidate[f"step/{step}/field/Ex"] = _ProofField(field_shape)
        expected_clock_step = precondition_steps + captured_step
        for source_time, source_label in (
            (native[f"step/{step}/time"], "native"),
            (candidate[f"step/{step}/time"], "candidate"),
        ):
            _require(
                source_time.shape == (3,)
                and source_time.dtype == np.dtype("float64")
                and np.isfinite(source_time).all()
                and float(source_time[2]) == expected_time_step
                and float(source_time[0]) == expected_clock_step
                and float(source_time[1]) == expected_clock_step * expected_time_step,
                f"{label} {source_label} time differs from the capture clock",
            )
        _require(
            np.array_equal(native[f"step/{step}/time"], candidate[f"step/{step}/time"]),
            f"{label} native and candidate time preimages differ",
        )
        native_steps[step] = {
            "sources": {
                "updaters": [
                    {
                        "component": "Ex",
                        "native_type": "PointSourceEx",
                        "cells": 1,
                        "state_values": 4,
                    }
                ]
            }
        }
        candidate_steps[step] = {
            "sources": {
                "updaters": [
                    {
                        "component": "Ex",
                        "native_type": "PointSourceEx",
                        "cells": 1,
                        "state_values": 9,
                    }
                ]
            }
        }
    _require(
        reference_digest
        == _source_role_preimage_sha256(workload, "reference", captures)
        and candidate_digest
        == _source_role_preimage_sha256(workload, "candidate", captures),
        "PointSource raw preimage digest differs from canonical bytes",
    )
    expected = _expected_point_source_contract(workload)
    native_payload = _validate_native_point_source(
        native,
        {"capture_steps": capture_steps, "steps": native_steps},
        workload,
        expected,
    )
    candidate_payload = _validate_candidate_point_source(
        candidate,
        {
            "capture_steps": capture_steps,
            "backend_metadata": {"precision": precision},
            "steps": candidate_steps,
        },
        workload,
        expected,
    )
    _require(
        native_payload == candidate_payload,
        "PointSource raw proof semantic projections differ",
    )
    return native_payload


def _validate_projected_source_proof(
    reference, candidate, workload, capture_steps, precision, precondition_steps
):
    _require(
        reference.dtype == candidate.dtype == np.dtype("uint8")
        and reference.ndim == candidate.ndim == 1
        and 0 < reference.size <= MAX_SOURCE_PROOF_BYTES
        and np.array_equal(reference, candidate),
        "projected PointSource raw proof bytes differ",
    )
    return _validate_source_raw_proof_bytes(
        reference.tobytes(),
        workload,
        capture_steps,
        precision,
        precondition_steps,
    )


def _validate_projected_source_contract(reference, candidate, workload):
    _require(
        reference.dtype == candidate.dtype == np.dtype("uint8")
        and reference.ndim == candidate.ndim == 1
        and 0 < reference.size <= 64 * 1024
        and np.array_equal(reference, candidate),
        "projected PointSource contract bytes differ",
    )
    raw = reference.tobytes()
    parsed = _strict_json_bytes(raw, "projected PointSource contract")
    expected = _expected_point_source_contract(workload)
    _require(
        parsed == expected and raw == _canonical_source_bytes(expected),
        "projected PointSource contract differs from the workload",
    )
    return raw


def _field_projection_names(steps):
    return [
        f"step/{step}/field/{component}" for step in steps for component in FIELD_ARRAYS
    ]


def _physical_projection_names(steps, scope, case, device):
    if (scope, case, device) != NORMALIZED_RESIDUAL_CASE:
        return []
    return [
        f"step/{step}/{suffix}" for step in steps for suffix in PHYSICAL_ARRAY_SUFFIXES
    ]


def _persistent_projection_names(steps, suffixes):
    return [f"step/{step}/{suffix}" for step in steps for suffix in suffixes]


def _projection_group_names(steps, scope, case, device):
    return [
        *_field_projection_names(steps),
        *_physical_projection_names(steps, scope, case, device),
        *_persistent_projection_names(steps, _expected_persistent_suffixes(case)),
        SOURCE_CONTRACT_ARRAY,
        SOURCE_PROOF_ARRAY,
    ]


def _group_archive_comment(
    scope, case, device, role, ordinal, steps, candidate_evidence
):
    _require(role in {"reference", "candidate"}, "projection group role is invalid")
    return _canonical_source_bytes(
        {
            "schema": GROUP_ARCHIVE_SCHEMA,
            "scope": scope,
            "case": case,
            "device": device,
            "role": role,
            "ordinal": ordinal,
            "steps": steps,
            "candidate_evidence": _candidate_projection(candidate_evidence),
        }
    )


def _persistent_geometry_sha256(arrays, workload, case, step):
    try:
        labels = CASE_UPDATER_LABELS[case]
    except KeyError as error:
        raise ValueError(f"no frozen persistent geometry for {case!r}") from error
    digest = hashlib.sha256()
    digest.update(b"issue-123-persistent-geometry-v1\0")
    digest.update(case.encode())
    digest.update(b"\0")
    digest.update(_canonical_source_bytes(workload))
    digest.update(b"\0")
    suffixes = sorted(
        f"state/{component}/{updater_label}/indices"
        for component in FIELD_ARRAYS
        for updater_label in labels
    )
    for suffix in suffixes:
        indices = np.asarray(arrays[f"step/{step}/{suffix}"])
        _require(
            indices.dtype == np.dtype("int64")
            and indices.ndim == 2
            and indices.shape[1:] == (3,)
            and indices.flags.c_contiguous,
            f"persistent geometry input is invalid for {suffix}",
        )
        canonical = indices.astype("<i8", copy=False)
        digest.update(suffix.encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(canonical.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _recomputed_physical_arrays(arrays, step):
    spectra = {}
    summary = [0.0, 0.0, 0.0, 0.0, 1.0]
    for component in FIELD_ARRAYS:
        field = np.asarray(arrays[f"step/{step}/field/{component}"])
        magnitude = np.abs(field)
        summary[0] += float(np.sum(magnitude * magnitude))
        summary[1] = max(summary[1], float(np.max(magnitude)))
        summary[2] += float(np.sum(magnitude[0] * magnitude[0]))
        summary[3] += float(np.sum(magnitude[-1] * magnitude[-1]))
        summary[4] = float(bool(summary[4]) and np.isfinite(field).all())
        axes = tuple(range(1, field.ndim))
        line = np.mean(field, axis=axes) if axes else field
        spectra[component] = np.abs(np.fft.fft(line))
    return spectra, np.asarray(summary, dtype=np.float64)


def _validate_projected_array_contract(
    arrays,
    workload,
    scope,
    case,
    device,
    projection_steps,
    label,
    *,
    persistent_index_bindings=None,
):
    field_shapes = _expected_field_shapes(workload)
    field_dtype = _expected_field_dtype(scope, case, device)
    physical_dtype = np.empty((), dtype=field_dtype).real.dtype
    updater_labels = CASE_UPDATER_LABELS[case]
    if persistent_index_bindings is None:
        persistent_index_bindings = {}
    for step in projection_steps:
        for component, shape in field_shapes.items():
            field = arrays[f"step/{step}/field/{component}"]
            _require(
                field.shape == shape and field.dtype == field_dtype,
                f"{label} field shape or dtype differs for step/{step}/{component}",
            )
        if (scope, case, device) == NORMALIZED_RESIDUAL_CASE:
            expected_spectra, expected_summary = _recomputed_physical_arrays(
                arrays, step
            )
            for component, shape in field_shapes.items():
                spectrum = arrays[f"step/{step}/physical/spectrum/{component}"]
                _require(
                    spectrum.shape == (shape[0],)
                    and spectrum.dtype == physical_dtype
                    and np.array_equal(spectrum, expected_spectra[component]),
                    f"{label} physical spectrum differs from its field for "
                    f"step/{step}/{component}",
                )
            summary = arrays[f"step/{step}/physical/summary"]
            _require(
                summary.shape == (5,)
                and summary.dtype == np.dtype("float64")
                and np.array_equal(summary, expected_summary),
                f"{label} physical summary differs from its fields for step/{step}",
            )
        for component, shape in field_shapes.items():
            field_size = math.prod(shape)
            covered = np.zeros(field_size, dtype=np.bool_)
            for updater_label in updater_labels:
                prefix = f"step/{step}/state/{component}/{updater_label}"
                indices = arrays[f"{prefix}/indices"]
                values = arrays[f"{prefix}/values"]
                _require(
                    indices.dtype == np.dtype("int64")
                    and indices.ndim == 2
                    and indices.shape[1:] == (3,)
                    and indices.shape[0] > 0,
                    f"{label} persistent index shape or dtype differs for {prefix}",
                )
                _require(
                    all(
                        bool(np.all(indices[:, axis] >= 0))
                        and bool(np.all(indices[:, axis] < shape[axis]))
                        for axis in range(3)
                    ),
                    f"{label} persistent indices are outside the field for {prefix}",
                )
                linear = np.ravel_multi_index(tuple(indices.T), shape)
                _require(
                    len(np.unique(linear)) == len(linear)
                    and not bool(np.any(covered[linear])),
                    f"{label} persistent indices overlap for {prefix}",
                )
                covered[linear] = True
                width = _persistent_state_width(workload, component, updater_label)
                _require(
                    values.dtype == np.dtype("complex128")
                    and values.shape == (indices.shape[0] * width,),
                    f"{label} persistent value shape or dtype differs for {prefix}",
                )
                suffix = f"state/{component}/{updater_label}/indices"
                binding = (
                    indices.shape,
                    hashlib.sha256(indices.tobytes(order="C")).digest(),
                )
                previous = persistent_index_bindings.get(suffix)
                if previous is None:
                    persistent_index_bindings[suffix] = binding
                else:
                    _require(
                        previous == binding,
                        f"{label} persistent indices change across capture steps "
                        f"for {suffix}",
                    )
            _require(
                bool(np.all(covered)),
                f"{label} persistent indices do not cover the complete field for "
                f"step/{step}/{component}",
            )
        try:
            expected_geometry = FROZEN_PERSISTENT_GEOMETRY_SHA256_BY_CASE[case]
        except KeyError as error:
            raise ValueError(
                f"no frozen persistent geometry digest for {case!r}"
            ) from error
        _require(
            _persistent_geometry_sha256(arrays, workload, case, step)
            == expected_geometry,
            f"{label} persistent geometry differs from the frozen case for "
            f"step/{step}",
        )


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


def _precision_limitation(reference, scope, case, device, projection_steps):
    if (scope, case, device) != NORMALIZED_RESIDUAL_CASE:
        return None
    _require(
        projection_steps == list(NORMALIZED_PROJECTION_STEPS),
        "precision limitation projection steps differ",
    )
    names = [
        f"step/{NORMALIZED_RESIDUAL_STEPS[-1]}/field/{component}"
        for component in FIELD_ARRAYS
    ]
    _require(
        all(name in reference for name in names),
        "precision limitation reference fields are incomplete",
    )
    maximum = max(float(np.max(np.abs(reference[name]), initial=0.0)) for name in names)
    rejected_maximum = float(np.finfo(np.float32).max)
    _require(
        math.isfinite(maximum) and maximum > rejected_maximum,
        "native step-100 fields do not prove the float32 dynamic-range limitation",
    )
    return {
        "contract_id": PRECISION_LIMITATION_CONTRACT_ID,
        "rejected_precision": "float32",
        "accepted_precision": "float64",
        "reference_step": NORMALIZED_RESIDUAL_STEPS[-1],
        "reference_field_max_abs": maximum,
        "rejected_precision_max": rejected_maximum,
        "range_exceeded": True,
        "reason": PRECISION_LIMITATION_REASON,
    }


def _stable_l2_norm(value):
    magnitudes = np.abs(np.asarray(value)).reshape(-1)
    scale = float(np.max(magnitudes, initial=0.0))
    if scale == 0.0:
        return 0.0
    scaled = magnitudes / scale
    return scale * math.sqrt(float(np.sum(scaled * scaled, dtype=np.float64)))


def _normalized_array_errors(reference, candidate, floor, *, zero_exact):
    _require(floor >= 0.0 and math.isfinite(floor), "normalized scale floor is invalid")
    if reference.size == 0:
        return (0.0, 0.0) if np.array_equal(reference, candidate) else (math.inf,) * 2
    difference = np.abs(candidate - reference)
    difference_linf = float(np.max(difference, initial=0.0))
    reference_linf = float(np.max(np.abs(reference), initial=0.0))
    if zero_exact and reference_linf == 0.0:
        return (0.0, 0.0) if np.array_equal(reference, candidate) else (math.inf,) * 2
    linf_denominator = max(reference_linf, floor)
    l2_denominator = max(_stable_l2_norm(reference), floor * math.sqrt(reference.size))
    linf = (
        0.0
        if difference_linf == 0.0
        else math.inf if linf_denominator == 0.0 else difference_linf / linf_denominator
    )
    difference_l2 = _stable_l2_norm(difference)
    l2 = (
        0.0
        if difference_l2 == 0.0
        else math.inf if l2_denominator == 0.0 else difference_l2 / l2_denominator
    )
    return linf, l2


def _source_archive_identity(path):
    status = Path(path).stat()
    return (
        status.st_dev,
        status.st_ino,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _validate_zip_byte_coverage(raw, archive, label):
    """Require one canonical, completely indexed single-disk ZIP byte stream."""
    comment = archive.comment
    eocd_offset = len(raw) - 22 - len(comment)
    _require(
        raw.startswith(b"PK\x03\x04")
        and eocd_offset >= 0
        and raw[eocd_offset : eocd_offset + 4] == b"PK\x05\x06"
        and int.from_bytes(raw[eocd_offset + 4 : eocd_offset + 6], "little") == 0
        and int.from_bytes(raw[eocd_offset + 6 : eocd_offset + 8], "little") == 0
        and int.from_bytes(raw[eocd_offset + 20 : eocd_offset + 22], "little")
        == len(comment),
        f"{label} ZIP has prepended, trailing, or multi-disk bytes",
    )
    members = archive.infolist()
    entry_count = int.from_bytes(raw[eocd_offset + 8 : eocd_offset + 10], "little")
    total_count = int.from_bytes(raw[eocd_offset + 10 : eocd_offset + 12], "little")
    central_size = int.from_bytes(raw[eocd_offset + 12 : eocd_offset + 16], "little")
    central_offset = int.from_bytes(raw[eocd_offset + 16 : eocd_offset + 20], "little")
    _require(
        entry_count == total_count == len(members)
        and central_offset > 0
        and central_offset + central_size == eocd_offset,
        f"{label} ZIP central-directory closure differs",
    )

    local_cursor = 0
    central_cursor = central_offset
    for member in members:
        offset = member.header_offset
        _require(
            offset == local_cursor
            and offset + 30 <= central_offset
            and raw[offset : offset + 4] == b"PK\x03\x04",
            f"{label} ZIP local records have an unindexed gap or overlap",
        )
        flags = int.from_bytes(raw[offset + 6 : offset + 8], "little")
        compression = int.from_bytes(raw[offset + 8 : offset + 10], "little")
        crc = int.from_bytes(raw[offset + 14 : offset + 18], "little")
        compressed_size = int.from_bytes(raw[offset + 18 : offset + 22], "little")
        file_size = int.from_bytes(raw[offset + 22 : offset + 26], "little")
        name_size = int.from_bytes(raw[offset + 26 : offset + 28], "little")
        extra_size = int.from_bytes(raw[offset + 28 : offset + 30], "little")
        header_end = offset + 30 + name_size + extra_size
        encoding = "utf-8" if flags & 0x800 else "cp437"
        try:
            encoded_name = member.filename.encode(encoding)
        except UnicodeEncodeError as error:
            raise ValueError(f"{label} ZIP member name is not canonical") from error
        extra = raw[offset + 30 + name_size : header_end]
        if compressed_size == file_size == 0xFFFFFFFF:
            expected_extra = (
                b"\x01\x00\x10\x00"
                + member.file_size.to_bytes(8, "little")
                + member.compress_size.to_bytes(8, "little")
            )
            local_sizes_match = extra == expected_extra
        else:
            local_sizes_match = (
                extra == b""
                and compressed_size == member.compress_size
                and file_size == member.file_size
            )
        local_cursor = header_end + member.compress_size
        _require(
            header_end <= central_offset
            and local_cursor <= central_offset
            and raw[offset + 30 : offset + 30 + name_size] == encoded_name
            and flags == member.flag_bits
            and not (flags & (0x1 | 0x8 | 0x40))
            and compression == member.compress_type
            and crc == member.CRC
            and local_sizes_match,
            f"{label} ZIP local record is not canonical",
        )

        _require(
            central_cursor + 46 <= eocd_offset
            and raw[central_cursor : central_cursor + 4] == b"PK\x01\x02",
            f"{label} ZIP central record is invalid",
        )
        central_flags = int.from_bytes(
            raw[central_cursor + 8 : central_cursor + 10], "little"
        )
        central_compression = int.from_bytes(
            raw[central_cursor + 10 : central_cursor + 12], "little"
        )
        central_crc = int.from_bytes(
            raw[central_cursor + 16 : central_cursor + 20], "little"
        )
        central_compressed = int.from_bytes(
            raw[central_cursor + 20 : central_cursor + 24], "little"
        )
        central_file_size = int.from_bytes(
            raw[central_cursor + 24 : central_cursor + 28], "little"
        )
        central_name_size = int.from_bytes(
            raw[central_cursor + 28 : central_cursor + 30], "little"
        )
        central_extra_size = int.from_bytes(
            raw[central_cursor + 30 : central_cursor + 32], "little"
        )
        central_comment_size = int.from_bytes(
            raw[central_cursor + 32 : central_cursor + 34], "little"
        )
        central_disk = int.from_bytes(
            raw[central_cursor + 34 : central_cursor + 36], "little"
        )
        central_local_offset = int.from_bytes(
            raw[central_cursor + 42 : central_cursor + 46], "little"
        )
        central_end = (
            central_cursor
            + 46
            + central_name_size
            + central_extra_size
            + central_comment_size
        )
        _require(
            central_end <= eocd_offset
            and raw[central_cursor + 46 : central_cursor + 46 + central_name_size]
            == encoded_name
            and central_flags == member.flag_bits
            and central_compression == member.compress_type
            and central_crc == member.CRC
            and central_compressed == member.compress_size
            and central_file_size == member.file_size
            and central_extra_size == 0
            and central_comment_size == 0
            and central_disk == 0
            and central_local_offset == offset
            and member.extra == b""
            and member.comment == b"",
            f"{label} ZIP central record is not canonical",
        )
        central_cursor = central_end
    _require(
        local_cursor == central_offset
        and central_cursor == central_offset + central_size,
        f"{label} ZIP contains unindexed, gap, or trailing records",
    )


def _require_source_archive_binding(path, binding, label):
    _require(
        _source_archive_identity(path) == binding,
        f"{label} changed after source NPZ preflight",
    )


def _preflight_source_npz(path, label):
    supplied = Path(path)
    _require(not supplied.is_symlink(), f"{label} cannot be a symlink")
    resolved = supplied.resolve(strict=True)
    _require(resolved.is_file(), f"{label} must be a regular file")
    binding = _source_archive_identity(resolved)
    _require(
        0 < binding[2] <= MAX_SOURCE_NPZ_ARCHIVE_BYTES,
        f"{label} exceeds the source NPZ archive bound",
    )
    raw = _bounded_regular_file_bytes(
        resolved,
        MAX_SOURCE_NPZ_ARCHIVE_BYTES,
        label,
    )
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            _require(archive.comment == b"", f"{label} source NPZ has a comment")
            _validate_zip_byte_coverage(raw, archive, f"{label} source NPZ")
            members = archive.infolist()
            _require(
                0 < len(members) <= MAX_SOURCE_NPZ_MEMBERS,
                f"{label} source NPZ member count is invalid",
            )
            names = [member.filename for member in members]
            _require(
                len(names) == len(set(names))
                and all(name.endswith(".npy") for name in names),
                f"{label} source NPZ member closure is invalid",
            )
            _require(
                sum(member.file_size for member in members)
                <= MAX_SOURCE_NPZ_TOTAL_BYTES,
                f"{label} source NPZ expanded size exceeds the bound",
            )
            total_payload_bytes = 0
            for member in members:
                _safe_projection_name(member.filename.removesuffix(".npy"))
                offset = member.header_offset
                _require(
                    type(offset) is int and 0 <= offset <= binding[2] - 30,
                    f"{label} source NPZ local header is outside the archive",
                )
                local_header = raw[offset : offset + 30]
                _require(
                    len(local_header) == 30 and local_header[:4] == b"PK\x03\x04",
                    f"{label} source NPZ local header is invalid",
                )
                local_flags = int.from_bytes(local_header[6:8], "little")
                local_compression = int.from_bytes(local_header[8:10], "little")
                local_name_size = int.from_bytes(local_header[26:28], "little")
                local_extra_size = int.from_bytes(local_header[28:30], "little")
                local_name = raw[offset + 30 : offset + 30 + local_name_size]
                encoding = "utf-8" if local_flags & 0x800 else "cp437"
                try:
                    encoded_name = member.filename.encode(encoding)
                except UnicodeEncodeError as error:
                    raise ValueError(
                        f"{label} source NPZ local name is invalid"
                    ) from error
                mode = member.external_attr >> 16
                _require(
                    local_name == encoded_name
                    and offset + 30 + local_name_size + local_extra_size <= binding[2]
                    and local_flags == member.flag_bits
                    and not (local_flags & (0x1 | 0x40))
                    and local_compression == member.compress_type
                    and member.compress_type
                    in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                    and not member.is_dir()
                    and stat.S_IFMT(mode) in {0, stat.S_IFREG}
                    and 0 <= member.file_size <= MAX_SOURCE_NPZ_MEMBER_BYTES,
                    f"{label} source NPZ member contract is invalid",
                )
                if member.file_size:
                    _require(
                        member.compress_size > 0
                        and member.file_size / member.compress_size
                        <= MAX_NPZ_COMPRESSION_RATIO,
                        f"{label} source NPZ compression ratio exceeds the bound",
                    )
                try:
                    with archive.open(member) as payload:
                        version = np.lib.format.read_magic(payload)
                        _require(
                            version in {(1, 0), (2, 0), (3, 0)},
                            f"{label} source NPY version is unsupported",
                        )
                        header_reader = (
                            np.lib.format.read_array_header_1_0
                            if version == (1, 0)
                            else np.lib.format.read_array_header_2_0
                        )
                        shape, fortran_order, dtype = header_reader(
                            payload, max_header_size=MAX_NPY_HEADER_BYTES
                        )
                        header_bytes = payload.tell()
                        observed = header_bytes
                        while chunk := payload.read(1024 * 1024):
                            observed += len(chunk)
                            _require(
                                observed <= member.file_size
                                and observed <= MAX_SOURCE_NPZ_MEMBER_BYTES,
                                f"{label} source NPY expanded beyond its bound",
                            )
                        _require(
                            observed == member.file_size,
                            f"{label} source NPY size differs from ZIP metadata",
                        )
                except (
                    EOFError,
                    MemoryError,
                    NotImplementedError,
                    OSError,
                    OverflowError,
                    RecursionError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                    zipfile.BadZipFile,
                    zlib.error,
                ) as error:
                    if isinstance(error, ValueError) and str(error).startswith(label):
                        raise
                    raise ValueError(
                        f"{label} source NPZ has an invalid NPY member"
                    ) from error
                dtype = np.dtype(dtype)
                _require(
                    isinstance(shape, tuple)
                    and len(shape) <= 8
                    and all(
                        type(size) is int and 0 <= size <= 2**31 - 1 for size in shape
                    )
                    and fortran_order is False
                    and dtype.fields is None
                    and dtype.subdtype is None,
                    f"{label} source NPY shape or dtype is invalid",
                )
                array_bytes = math.prod(shape) * dtype.itemsize
                numeric = dtype.kind in {"b", "i", "u", "f", "c"} and (
                    0 < dtype.itemsize <= 16
                )
                metadata = (
                    member.filename == "metadata.json.npy"
                    and shape == ()
                    and dtype.kind == "U"
                    and dtype.str.startswith("<U")
                    and 0 < array_bytes <= MAX_SOURCE_METADATA_BYTES
                )
                _require(
                    (numeric or metadata)
                    and array_bytes <= MAX_SOURCE_NPZ_MEMBER_BYTES
                    and header_bytes + array_bytes == member.file_size,
                    f"{label} source NPY payload contract is invalid",
                )
                total_payload_bytes += array_bytes
                _require(
                    total_payload_bytes <= MAX_SOURCE_NPZ_TOTAL_BYTES,
                    f"{label} source NPY allocation exceeds the bound",
                )
    except (
        MemoryError,
        NotImplementedError,
        OSError,
        OverflowError,
        RecursionError,
        RuntimeError,
        TypeError,
        zipfile.BadZipFile,
        zlib.error,
    ) as error:
        raise ValueError(f"{label} is not a safe source NPZ archive") from error
    _require_source_archive_binding(resolved, binding, label)
    return resolved, binding


def _normalized_archive_comparison_passes(
    reference_path,
    candidate_path,
    manifest,
    comparison,
    source_bindings,
):
    _require_source_archive_binding(
        reference_path, source_bindings[reference_path], "differential reference"
    )
    _require_source_archive_binding(
        candidate_path, source_bindings[candidate_path], "differential candidate"
    )
    result = torch_correctness.compare_torch_archives(
        reference_path, candidate_path, manifest
    )
    allowed_steps = {str(step) for step in NORMALIZED_RESIDUAL_STEPS}
    allowed_prefixes = tuple(
        f"step/{step}/{category}/"
        for step in allowed_steps
        for category in (
            "field",
            "physical",
            "state",
            "source_aux",
            "source_aux_material",
        )
    )
    _require_source_archive_binding(
        reference_path, source_bindings[reference_path], "differential reference"
    )
    _require_source_archive_binding(
        candidate_path, source_bindings[candidate_path], "differential candidate"
    )
    with ExitStack() as stack:
        reference = stack.enter_context(np.load(reference_path, allow_pickle=False))
        candidate = stack.enter_context(np.load(candidate_path, allow_pickle=False))
        normalized_keys = sorted(
            key
            for key in reference.files
            if key.startswith(allowed_prefixes) and key in candidate.files
        )
        _require(bool(normalized_keys), "normalized archive array closure is empty")
        allowed_failure_keys = set()
        for key in normalized_keys:
            left = _numeric_array(reference[key], f"reference {key}")
            right = _numeric_array(candidate[key], f"candidate {key}")
            _require(left.shape == right.shape, f"{key} shape differs")
            if left.dtype.kind in {"b", "i", "u"}:
                _require(
                    right.dtype.kind in {"b", "i", "u"} and np.array_equal(left, right),
                    f"{key} exact integer values differ",
                )
                continue
            _require(
                left.dtype.kind in {"f", "c"}
                and right.dtype.kind in {"f", "c"}
                and left.dtype == right.dtype,
                f"{key} numerical dtype differs",
            )
            allowed_failure_keys.add(key)
            linf, l2 = _normalized_array_errors(
                left,
                right,
                comparison["absolute_scale_floor"],
                zero_exact=True,
            )
            _require(
                math.isfinite(linf)
                and math.isfinite(l2)
                and linf <= comparison["linf_limit"]
                and l2 <= comparison["l2_limit"],
                f"normalized archive residual failed for {key}",
            )
    _require(
        all(
            failure.get("key") in allowed_failure_keys for failure in result["failures"]
        ),
        "differential source archives fail strict early-step or topology validation",
    )
    return True


def _projection_arrays(
    reference_path,
    candidate_path,
    manifest,
    scope,
    expected,
    *,
    group_consumer=None,
    source_bindings=None,
):
    if source_bindings is None:
        reference_path, reference_binding = _preflight_source_npz(
            reference_path, "differential reference"
        )
        candidate_path, candidate_binding = _preflight_source_npz(
            candidate_path, "differential candidate"
        )
        source_bindings = {
            reference_path: reference_binding,
            candidate_path: candidate_binding,
        }
    else:
        reference_path = Path(reference_path).resolve(strict=True)
        candidate_path = Path(candidate_path).resolve(strict=True)
        _require(
            reference_path in source_bindings and candidate_path in source_bindings,
            "differential source preflight binding is incomplete",
        )
        _require_source_archive_binding(
            reference_path,
            source_bindings[reference_path],
            "differential reference",
        )
        _require_source_archive_binding(
            candidate_path,
            source_bindings[candidate_path],
            "differential candidate",
        )
    reference_path, reference_metadata = torch_correctness._archive_record(
        reference_path, manifest, "reference"
    )
    reference_path = Path(reference_path).resolve(strict=True)
    _require_source_archive_binding(
        reference_path, source_bindings[reference_path], "differential reference"
    )
    candidate_path, candidate_metadata = torch_correctness._archive_record(
        candidate_path, manifest, "candidate"
    )
    candidate_path = Path(candidate_path).resolve(strict=True)
    _require_source_archive_binding(
        candidate_path, source_bindings[candidate_path], "differential candidate"
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
    expected_precision = expected["precision"]
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
    comparison = frozen_comparison_contract(manifest, scope, case, device)
    if comparison["mode"] == NORMALIZED_COMPARISON_MODE:
        _normalized_archive_comparison_passes(
            reference_path,
            candidate_path,
            manifest,
            comparison,
            source_bindings,
        )
    else:
        _require_source_archive_binding(
            reference_path,
            source_bindings[reference_path],
            "differential reference",
        )
        _require_source_archive_binding(
            candidate_path,
            source_bindings[candidate_path],
            "differential candidate",
        )
        archive_comparison = torch_correctness.compare_torch_archives(
            reference_path, candidate_path, manifest
        )
        _require(
            archive_comparison == {"passed": True, "failures": []},
            "differential source archives fail full correctness validation",
        )
    capture_steps = _frozen_capture_steps(manifest, case)
    _require(
        reference_metadata["capture_steps"]
        == candidate_metadata["capture_steps"]
        == capture_steps,
        "differential source capture-step closure differs from the manifest",
    )
    projection_steps = frozen_projection_steps(manifest, scope, case, device)
    projection_groups = frozen_projection_groups(manifest, scope, case, device)
    field_names = _field_projection_names(projection_steps)
    physical_names = _physical_projection_names(projection_steps, scope, case, device)
    contract_names = [SOURCE_CONTRACT_ARRAY, SOURCE_PROOF_ARRAY]
    persistent_suffixes = _expected_persistent_suffixes(case)
    persistent_names = _persistent_projection_names(
        projection_steps, persistent_suffixes
    )
    reference_groups = [] if group_consumer is None else None
    candidate_groups = [] if group_consumer is None else None
    reference_index_bindings = {}
    candidate_index_bindings = {}
    _require_source_archive_binding(
        reference_path, source_bindings[reference_path], "differential reference"
    )
    _require_source_archive_binding(
        candidate_path, source_bindings[candidate_path], "differential candidate"
    )
    with ExitStack() as stack:
        reference = stack.enter_context(np.load(reference_path, allow_pickle=False))
        candidate = stack.enter_context(np.load(candidate_path, allow_pickle=False))
        for projected_step in projection_steps:
            step = str(projected_step)
            reference_keys = _persistent_source_keys(reference, step)
            candidate_keys = _persistent_source_keys(candidate, step)
            expected_persistent_keys = [
                f"step/{step}/{suffix}" for suffix in persistent_suffixes
            ]
            _require(
                reference_keys == candidate_keys == expected_persistent_keys,
                "differential persistent source topology differs from the frozen case",
            )
            expected_fields = {
                f"step/{step}/field/{component}" for component in FIELD_ARRAYS
            }
            for archive, label in ((reference, "reference"), (candidate, "candidate")):
                actual_fields = {
                    key
                    for key in archive.files
                    if key.startswith(f"step/{step}/field/")
                }
                _require(
                    actual_fields == expected_fields,
                    f"{label} projected field closure differs",
                )
            if physical_names:
                expected_physical = {
                    f"step/{step}/{suffix}" for suffix in PHYSICAL_ARRAY_SUFFIXES
                }
                for archive, label in (
                    (reference, "reference"),
                    (candidate, "candidate"),
                ):
                    actual_physical = {
                        key
                        for key in archive.files
                        if key.startswith(f"step/{step}/physical/")
                    }
                    _require(
                        actual_physical == expected_physical,
                        f"{label} projected physical closure differs",
                    )
        reference_contract, candidate_contract = _point_source_contract_arrays(
            reference,
            candidate,
            reference_metadata,
            candidate_metadata,
            _workload(manifest, case),
        )
        proof = _build_point_source_raw_proof(
            reference,
            candidate,
            reference_metadata,
            candidate_metadata,
            _workload(manifest, case),
        )
        proof_payload = _canonical_source_bytes(proof)
        proof_semantic = _validate_source_raw_proof_bytes(
            proof_payload,
            _workload(manifest, case),
            capture_steps,
            expected_precision,
            manifest["reference"]["precondition_steps"],
        )
        _require(
            proof_semantic
            == reference_contract.tobytes()
            == candidate_contract.tobytes(),
            "PointSource semantic contract differs from its raw proof",
        )
        for ordinal, group in enumerate(projection_groups):
            projected_names = _projection_group_names(group, scope, case, device)[
                : -len(contract_names)
            ]
            reference_arrays = {}
            candidate_arrays = {}
            for name in projected_names:
                left = _numeric_array(reference[name], f"reference {name}")
                right = _numeric_array(candidate[name], f"candidate {name}")
                _require(left.shape == right.shape, f"{name} shape differs")
                if right.dtype.kind in {"b", "i", "u"}:
                    _require(
                        left.dtype.kind in {"b", "i", "u"}
                        and np.array_equal(left, right),
                        f"{name} exact integer values differ",
                    )
                reference_arrays[name] = np.ascontiguousarray(
                    left.astype(right.dtype, copy=False)
                )
                candidate_arrays[name] = np.ascontiguousarray(right)
            reference_arrays[SOURCE_CONTRACT_ARRAY] = reference_contract
            candidate_arrays[SOURCE_CONTRACT_ARRAY] = candidate_contract
            reference_arrays[SOURCE_PROOF_ARRAY] = np.frombuffer(
                proof_payload, dtype=np.uint8
            ).copy()
            candidate_arrays[SOURCE_PROOF_ARRAY] = np.frombuffer(
                proof_payload, dtype=np.uint8
            ).copy()
            for arrays, label, bindings in (
                (
                    reference_arrays,
                    f"reference projection group {ordinal}",
                    reference_index_bindings,
                ),
                (
                    candidate_arrays,
                    f"candidate projection group {ordinal}",
                    candidate_index_bindings,
                ),
            ):
                _validate_projected_array_contract(
                    arrays,
                    _workload(manifest, case),
                    scope,
                    case,
                    device,
                    group,
                    label,
                    persistent_index_bindings=bindings,
                )
            if group_consumer is None:
                reference_groups.append(reference_arrays)
                candidate_groups.append(candidate_arrays)
            else:
                group_consumer(
                    ordinal,
                    group,
                    reference_arrays,
                    candidate_arrays,
                )
    return (
        reference_groups,
        candidate_groups,
        candidate_evidence,
        projection_steps,
        projection_groups,
        field_names,
        physical_names,
        persistent_names,
        contract_names,
    )


def _render_npz(arrays, archive_comment):
    _require(
        isinstance(archive_comment, bytes) and 0 < len(archive_comment) <= 65535,
        "differential NPZ group identity must be bounded bytes",
    )
    payload = io.BytesIO()
    np.savez_compressed(payload, **arrays)
    with zipfile.ZipFile(payload, "a") as archive:
        archive.comment = archive_comment
    raw = payload.getvalue()
    _preflight_npz(raw, list(arrays), archive_comment, "differential output")
    return raw


def _write_npz(path, arrays, archive_comment):
    path = Path(path)
    _require(not path.exists(), f"differential output already exists: {path}")
    raw = _render_npz(arrays, archive_comment)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def _compare_arrays(reference, candidate, comparison, *, array_comparisons=None):
    maximum_abs = 0.0
    maximum_relative = 0.0
    maximum_linf = 0.0
    maximum_l2 = 0.0
    passed = True
    _require(
        set(reference) == set(candidate),
        "differential projection array closure differs",
    )
    if array_comparisons is not None:
        _require(
            set(array_comparisons) == set(reference),
            "differential per-array comparison closure differs",
        )
    for name in reference:
        left = reference[name]
        right = candidate[name]
        array_comparison = (
            comparison if array_comparisons is None else array_comparisons[name]
        )
        mode = array_comparison["mode"]
        floor = (
            array_comparison["absolute_scale_floor"]
            if mode == NORMALIZED_COMPARISON_MODE
            else array_comparison["atol"]
        )
        _require(
            left.shape == right.shape and left.dtype == right.dtype,
            f"differential projection array {name} shape or dtype differs",
        )
        _require(
            np.isfinite(left).all() and np.isfinite(right).all(),
            f"differential projection array {name} contains non-finite values",
        )
        exact = left.dtype.kind in {"b", "i", "u"}
        difference = (
            np.abs(right.astype(np.float64) - left.astype(np.float64))
            if exact
            else np.abs(right - left)
        )
        maximum_abs = max(maximum_abs, float(np.max(difference, initial=0.0)))
        denominator = np.maximum(
            np.abs(left), floor if floor > 0 else np.finfo(float).tiny
        )
        maximum_relative = max(
            maximum_relative,
            float(np.max(difference / denominator, initial=0.0)),
        )
        if exact:
            passed = passed and bool(np.array_equal(left, right))
            linf = l2 = 0.0 if np.array_equal(left, right) else math.inf
        else:
            linf, l2 = _normalized_array_errors(
                left,
                right,
                floor,
                zero_exact=True,
            )
            if not bool(np.any(left)):
                passed = passed and bool(np.array_equal(left, right))
            elif mode == ELEMENTWISE_COMPARISON_MODE:
                passed = passed and bool(
                    np.allclose(
                        right,
                        left,
                        rtol=array_comparison["rtol"],
                        atol=array_comparison["atol"],
                    )
                )
            else:
                passed = passed and (
                    linf <= array_comparison["linf_limit"]
                    and l2 <= array_comparison["l2_limit"]
                )
        maximum_linf = max(maximum_linf, linf)
        maximum_l2 = max(maximum_l2, l2)
    return (
        {
            "maximum_abs_error": maximum_abs,
            "maximum_relative_error": maximum_relative,
            "maximum_normalized_linf_error": maximum_linf,
            "maximum_normalized_l2_error": maximum_l2,
        },
        passed,
    )


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
    source_bindings = {}
    references = {}
    for path in reference_paths:
        path, binding = _preflight_source_npz(path, "differential reference")
        source_bindings[path] = binding
        resolved, metadata = torch_correctness._archive_record(
            path, manifest, "reference"
        )
        resolved = Path(resolved).resolve(strict=True)
        _require_source_archive_binding(
            resolved, source_bindings[resolved], "differential reference"
        )
        name = metadata["workload"]["name"]
        _require(name not in references, f"duplicate differential reference: {name}")
        references[name] = resolved
    candidates = {}
    for path in candidate_paths:
        path, binding = _preflight_source_npz(path, "differential candidate")
        source_bindings[path] = binding
        resolved, metadata = torch_correctness._archive_record(
            path, manifest, "candidate"
        )
        resolved = Path(resolved).resolve(strict=True)
        _require_source_archive_binding(
            resolved, source_bindings[resolved], "differential candidate"
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
    reference_source_descriptors = {
        name: _source_descriptor(path, descriptor_root, candidate)
        for name, path in references.items()
    }
    candidate_source_descriptors = {
        key: _source_descriptor(path, descriptor_root, candidate)
        for key, path in candidates.items()
    }
    records = []
    used_output_paths = set()
    used_output_digests = set()
    for expected in required:
        name = expected["case"]
        device = expected["device"]
        projection_steps = frozen_projection_steps(manifest, scope, name, device)
        projection_groups = frozen_projection_groups(manifest, scope, name, device)
        comparison = frozen_comparison_contract(manifest, scope, name, device)
        reference_descriptors = []
        candidate_descriptors = []
        metrics = {
            "maximum_abs_error": 0.0,
            "maximum_relative_error": 0.0,
            "maximum_normalized_linf_error": 0.0,
            "maximum_normalized_l2_error": 0.0,
        }
        precision_limitation = None
        suffix = device.replace(":", "-")

        def consume_group(ordinal, group, left, right):
            nonlocal precision_limitation
            first_step = group[0]
            last_step = group[-1]
            stem = f"{name}-{suffix}-group-{ordinal}-{first_step}-{last_step}"
            reference_output = output_directory / f"{stem}-reference.npz"
            candidate_output = output_directory / f"{stem}-candidate.npz"
            _write_npz(
                reference_output,
                left,
                _group_archive_comment(
                    scope,
                    name,
                    device,
                    "reference",
                    ordinal,
                    group,
                    candidate,
                ),
            )
            _write_npz(
                candidate_output,
                right,
                _group_archive_comment(
                    scope,
                    name,
                    device,
                    "candidate",
                    ordinal,
                    group,
                    candidate,
                ),
            )
            reference_descriptor = _descriptor(
                reference_output, descriptor_root, candidate
            )
            candidate_descriptor = _descriptor(
                candidate_output, descriptor_root, candidate
            )
            for descriptor in (reference_descriptor, candidate_descriptor):
                _require(
                    descriptor["path"] not in used_output_paths
                    and descriptor["sha256"] not in used_output_digests,
                    "differential group descriptor reuses output bytes",
                )
                used_output_paths.add(descriptor["path"])
                used_output_digests.add(descriptor["sha256"])
            reference_descriptors.append(reference_descriptor)
            candidate_descriptors.append(candidate_descriptor)
            group_metrics, group_passed = _compare_arrays(
                left,
                right,
                comparison,
                array_comparisons=_frozen_array_comparisons(
                    manifest, scope, name, device, left
                ),
            )
            _require(
                group_passed,
                f"differential projection group {ordinal} failed for {name} on {device}",
            )
            for metric_name, value in group_metrics.items():
                metrics[metric_name] = max(metrics[metric_name], value)
            if 100 in group:
                precision_limitation = _precision_limitation(
                    left, scope, name, device, projection_steps
                )

        (
            reference_groups,
            candidate_groups,
            _source,
            actual_projection_steps,
            actual_projection_groups,
            fields,
            physical,
            persistent,
            contracts,
        ) = _projection_arrays(
            references[name],
            candidates[(name, device)],
            manifest,
            scope,
            expected,
            group_consumer=consume_group,
            source_bindings=source_bindings,
        )
        _require(
            reference_groups is candidate_groups is None
            and actual_projection_steps == projection_steps
            and actual_projection_groups == projection_groups
            and len(reference_descriptors)
            == len(candidate_descriptors)
            == len(projection_groups),
            "differential projection group closure differs",
        )
        records.append(
            {
                "case": name,
                "device": device,
                "precision": expected["precision"],
                "projection_steps": projection_steps,
                "projection_groups": projection_groups,
                "reference": reference_descriptors,
                "candidate": candidate_descriptors,
                "reference_source": reference_source_descriptors[name],
                "candidate_source": candidate_source_descriptors[(name, device)],
                "field_arrays": fields,
                "physical_arrays": physical,
                "persistent_arrays": persistent,
                "contract_arrays": contracts,
                "comparison": comparison,
                "precision_limitation": precision_limitation,
                "metrics": metrics,
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


def _preflight_npz(raw, names, archive_comment, label):
    _require(
        type(raw) is bytes and 0 < len(raw) <= MAX_NPZ_ARCHIVE_BYTES,
        f"{label} NPZ archive size exceeds the bound",
    )
    expected = [f"{name}.npy" for name in names]
    _require(
        raw.startswith(b"PK\x03\x04"),
        f"{label} is not a canonical NPZ archive",
    )
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            _require(
                archive.comment == archive_comment,
                f"{label} NPZ group identity differs",
            )
            _validate_zip_byte_coverage(raw, archive, f"{label} NPZ")
            eocd_offset = len(raw) - 22 - len(archive.comment)
            _require(
                eocd_offset >= 0
                and raw[eocd_offset : eocd_offset + 4] == b"PK\x05\x06"
                and int.from_bytes(raw[eocd_offset + 20 : eocd_offset + 22], "little")
                == len(archive.comment),
                f"{label} NPZ has prepended or trailing bytes",
            )
            members = archive.infolist()
            _require(
                0 < len(members) <= MAX_NPZ_MEMBERS,
                f"{label} NPZ member count is invalid",
            )
            actual = [member.filename for member in members]
            _require(actual == expected, f"{label} NPZ array closure differs")
            _require(len(actual) == len(set(actual)), f"{label} repeats NPZ members")
            _require(
                sum(member.file_size for member in members) <= MAX_NPZ_TOTAL_BYTES,
                f"{label} NPZ is too large",
            )
            total = 0
            total_array_bytes = 0
            metadata = {}
            for member in members:
                _safe_projection_name(member.filename.removesuffix(".npy"))
                offset = member.header_offset
                _require(
                    type(offset) is int
                    and 0 <= offset <= len(raw) - 30
                    and raw[offset : offset + 4] == b"PK\x03\x04",
                    f"{label} NPZ member has no bounded local header",
                )
                local_flags = int.from_bytes(raw[offset + 6 : offset + 8], "little")
                local_compression = int.from_bytes(
                    raw[offset + 8 : offset + 10], "little"
                )
                local_name_size = int.from_bytes(
                    raw[offset + 26 : offset + 28], "little"
                )
                local_extra_size = int.from_bytes(
                    raw[offset + 28 : offset + 30], "little"
                )
                local_header_end = offset + 30 + local_name_size + local_extra_size
                encoding = "utf-8" if local_flags & 0x800 else "cp437"
                try:
                    encoded_name = member.filename.encode(encoding)
                except UnicodeEncodeError as error:
                    raise ValueError(
                        f"{label} NPZ local member name is not canonical"
                    ) from error
                _require(
                    local_header_end <= len(raw)
                    and raw[offset + 30 : offset + 30 + local_name_size] == encoded_name
                    and local_flags == member.flag_bits
                    and not (local_flags & (0x1 | 0x40))
                    and local_compression == member.compress_type
                    and local_compression == zipfile.ZIP_DEFLATED,
                    f"{label} NPZ local member contract differs",
                )
                mode = member.external_attr >> 16
                _require(
                    not member.is_dir()
                    and stat.S_IFMT(mode) in {0, stat.S_IFREG}
                    and member.compress_type == zipfile.ZIP_DEFLATED
                    and not (member.flag_bits & (0x1 | 0x40))
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
                try:
                    with archive.open(member) as payload:
                        version = np.lib.format.read_magic(payload)
                        _require(
                            version == (1, 0),
                            f"{label} NPY version is unsupported",
                        )
                        shape, fortran_order, dtype = (
                            np.lib.format.read_array_header_1_0(
                                payload, max_header_size=MAX_NPY_HEADER_BYTES
                            )
                        )
                        header_bytes = payload.tell()
                        observed = header_bytes
                        while chunk := payload.read(1024 * 1024):
                            observed += len(chunk)
                            _require(
                                observed <= member.file_size
                                and observed <= MAX_NPZ_MEMBER_BYTES,
                                f"{label} NPY member expanded beyond its "
                                "declared size",
                            )
                        _require(
                            observed == member.file_size,
                            f"{label} NPY member size differs from its ZIP metadata",
                        )
                except (
                    EOFError,
                    MemoryError,
                    NotImplementedError,
                    OSError,
                    OverflowError,
                    RecursionError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                    zipfile.BadZipFile,
                    zlib.error,
                ) as error:
                    if isinstance(error, ValueError) and str(error).startswith(label):
                        raise
                    raise ValueError(f"{label} has an invalid NPY header") from error
                dtype = np.dtype(dtype)
                _require(
                    isinstance(shape, tuple)
                    and len(shape) <= 8
                    and all(
                        type(size) is int and 0 <= size <= 2**31 - 1 for size in shape
                    )
                    and fortran_order is False
                    and dtype.fields is None
                    and dtype.subdtype is None
                    and dtype.kind in {"b", "i", "u", "f", "c"}
                    and 0 < dtype.itemsize <= 16,
                    f"{label} NPY array contract is invalid",
                )
                array_bytes = math.prod(shape) * dtype.itemsize
                _require(
                    array_bytes <= MAX_NPZ_MEMBER_BYTES
                    and header_bytes + array_bytes == member.file_size,
                    f"{label} NPY payload size differs from its header",
                )
                name = member.filename.removesuffix(".npy")
                metadata[name] = (shape, dtype, array_bytes)
                total += member.file_size
                total_array_bytes += array_bytes
            _require(
                total <= MAX_NPZ_TOTAL_BYTES
                and total_array_bytes <= MAX_NPZ_TOTAL_BYTES,
                f"{label} NPZ is too large",
            )
    except (
        MemoryError,
        NotImplementedError,
        OSError,
        OverflowError,
        RecursionError,
        RuntimeError,
        TypeError,
        zipfile.BadZipFile,
        zlib.error,
    ) as error:
        raise ValueError(f"{label} is not a readable NPZ archive") from error
    return metadata


def _load_projection(raw, names, archive_comment, label):
    metadata = _preflight_npz(raw, names, archive_comment, label)
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            _require(archive.files == names, f"{label} NPZ array order differs")
            arrays = {}
            for name in names:
                array = _numeric_array(archive[name], f"{label} {name}")
                shape, dtype, size_bytes = metadata[name]
                _require(
                    array.shape == shape
                    and array.dtype == dtype
                    and array.flags.c_contiguous
                    and array.nbytes == size_bytes,
                    f"{label} {name} differs from its preflight metadata",
                )
                arrays[name] = array
    except (
        EOFError,
        MemoryError,
        OSError,
        TypeError,
        ValueError,
        zlib.error,
    ) as error:
        if isinstance(error, ValueError) and str(error).startswith(label):
            raise
        raise ValueError(f"{label} is not a safe numeric NPZ archive") from error
    _require(
        sum(array.nbytes for array in arrays.values()) <= MAX_NPZ_TOTAL_BYTES,
        f"{label} uncompressed arrays exceed the bound",
    )
    return arrays


def _regenerate_record_projections(
    root,
    record,
    manifest,
    candidate,
    scope,
    expected,
    projection_groups,
    artifact_loader,
    used_paths,
    used_digests,
    reference_sources,
):
    """Reopen full sources and reproduce every indexed compact group byte-for-byte."""
    label = f"{scope} differential {expected['case']} {expected['device']}"
    reference_descriptor = record["reference_source"]
    candidate_descriptor = record["candidate_source"]
    reference_path, reference_raw = _load_bound_descriptor(
        root,
        reference_descriptor,
        candidate,
        f"{label} reference source",
        artifact_loader,
        source=True,
    )
    candidate_path, candidate_raw = _load_bound_descriptor(
        root,
        candidate_descriptor,
        candidate,
        f"{label} candidate source",
        artifact_loader,
        source=True,
    )
    _require(
        reference_path != candidate_path
        and reference_descriptor["path"] != candidate_descriptor["path"]
        and reference_descriptor["sha256"] != candidate_descriptor["sha256"],
        f"{label} source roles reuse a path or bytes",
    )
    source_identity = (
        reference_descriptor["path"],
        reference_descriptor["sha256"],
        reference_descriptor["size_bytes"],
    )
    prior_reference = reference_sources.get(expected["case"])
    if prior_reference is None:
        _require(
            source_identity[0] not in used_paths
            and source_identity[1] not in used_digests,
            f"{label} reference source reuses unrelated artifact bytes",
        )
        reference_sources[expected["case"]] = source_identity
        used_paths.add(source_identity[0])
        used_digests.add(source_identity[1])
    else:
        _require(
            prior_reference == source_identity,
            f"{label} does not share the exact case reference source",
        )
    _require(
        candidate_descriptor["path"] not in used_paths
        and candidate_descriptor["sha256"] not in used_digests,
        f"{label} candidate source reuses artifact path or bytes",
    )
    used_paths.add(candidate_descriptor["path"])
    used_digests.add(candidate_descriptor["sha256"])

    _require(
        _bounded_regular_file_bytes(
            reference_path,
            MAX_SOURCE_NPZ_ARCHIVE_BYTES,
            f"{label} reference source",
        )
        == reference_raw
        and _bounded_regular_file_bytes(
            candidate_path,
            MAX_SOURCE_NPZ_ARCHIVE_BYTES,
            f"{label} candidate source",
        )
        == candidate_raw,
        f"{label} source path differs from its exact descriptor bytes",
    )
    reference_path, reference_binding = _preflight_source_npz(
        reference_path, f"{label} reference source"
    )
    candidate_path, candidate_binding = _preflight_source_npz(
        candidate_path, f"{label} candidate source"
    )
    source_bindings = {
        reference_path: reference_binding,
        candidate_path: candidate_binding,
    }
    artifacts = {}

    def consume_group(ordinal, group, left, right):
        _require(
            ordinal == len(artifacts)
            and ordinal < len(projection_groups)
            and group == projection_groups[ordinal],
            f"{label} regenerated projection group order differs",
        )
        group_label = f"{label} projection group {ordinal}"
        descriptors = (record["reference"][ordinal], record["candidate"][ordinal])
        loaded = tuple(
            _load_bound_descriptor(
                root,
                descriptor,
                candidate,
                f"{group_label} {role}",
                artifact_loader,
                source=False,
            )
            for role, descriptor in zip(
                ("reference", "candidate"), descriptors, strict=True
            )
        )
        descriptor_paths = tuple(descriptor["path"] for descriptor in descriptors)
        descriptor_digests = tuple(descriptor["sha256"] for descriptor in descriptors)
        _require(
            loaded[0][0] != loaded[1][0]
            and len(set(descriptor_paths)) == 2
            and len(set(descriptor_digests)) == 2
            and not any(path in used_paths for path in descriptor_paths)
            and not any(digest in used_digests for digest in descriptor_digests),
            f"{group_label} reuses artifact path or bytes",
        )
        used_paths.update(descriptor_paths)
        used_digests.update(descriptor_digests)
        for role, arrays, (_path, raw) in zip(
            ("reference", "candidate"),
            (left, right),
            loaded,
            strict=True,
        ):
            rendered = _render_npz(
                arrays,
                _group_archive_comment(
                    scope,
                    expected["case"],
                    expected["device"],
                    role,
                    ordinal,
                    group,
                    candidate,
                ),
            )
            _require(
                raw == rendered,
                f"{group_label} {role} bytes differ from the complete source archive",
            )
        artifacts[ordinal] = loaded

    regenerated = _projection_arrays(
        reference_path,
        candidate_path,
        manifest,
        scope,
        expected,
        group_consumer=consume_group,
        source_bindings=source_bindings,
    )
    _require(
        len(artifacts) == len(projection_groups)
        and regenerated[0] is regenerated[1] is None
        and regenerated[2] == candidate
        and regenerated[3] == [step for group in projection_groups for step in group]
        and regenerated[4] == projection_groups,
        f"{label} regenerated projection closure differs",
    )
    _require(
        _bounded_regular_file_bytes(
            reference_path,
            MAX_SOURCE_NPZ_ARCHIVE_BYTES,
            f"{label} reference source",
        )
        == reference_raw
        and _bounded_regular_file_bytes(
            candidate_path,
            MAX_SOURCE_NPZ_ARCHIVE_BYTES,
            f"{label} candidate source",
        )
        == candidate_raw,
        f"{label} source bytes changed during projection regeneration",
    )
    return artifacts, regenerated[5:]


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
            {
                "case": record.get("case"),
                "device": record.get("device"),
                "precision": record.get("precision"),
            }
            for record in records
            if isinstance(record, dict)
        ]
        == required,
        "differential evaluated case closure differs",
    )
    used_paths = set()
    used_digests = set()
    reference_sources = {}
    recomputed_records = []
    for index, (record, expected) in enumerate(zip(records, required, strict=True)):
        label = f"{scope} differential case {index}"
        _exact_keys(record, RECORD_KEYS, label)
        _require(
            record["case"] == expected["case"]
            and record["device"] == expected["device"]
            and record["precision"] == expected["precision"],
            f"{label} identity differs",
        )
        projection_steps = frozen_projection_steps(
            manifest, scope, expected["case"], expected["device"]
        )
        projection_groups = frozen_projection_groups(
            manifest, scope, expected["case"], expected["device"]
        )
        _require(
            record["projection_steps"] == projection_steps
            and isinstance(record["projection_steps"], list)
            and all(type(step) is int for step in record["projection_steps"]),
            f"{label} projection steps differ from the manifest",
        )
        _require(
            record["projection_groups"] == projection_groups
            and isinstance(record["projection_groups"], list)
            and all(
                isinstance(group, list)
                and bool(group)
                and all(type(step) is int for step in group)
                for group in record["projection_groups"]
            ),
            f"{label} projection groups differ from the manifest",
        )
        fields = record["field_arrays"]
        physical = record["physical_arrays"]
        persistent = record["persistent_arrays"]
        contracts = record["contract_arrays"]
        expected_fields = _field_projection_names(projection_steps)
        expected_physical = _physical_projection_names(
            projection_steps, scope, expected["case"], expected["device"]
        )
        expected_persistent = _persistent_projection_names(
            projection_steps, _expected_persistent_suffixes(expected["case"])
        )
        _require(
            fields == expected_fields,
            f"{label} field array closure differs",
        )
        _require(
            physical == expected_physical,
            f"{label} physical array closure differs",
        )
        _require(
            contracts == [SOURCE_CONTRACT_ARRAY, SOURCE_PROOF_ARRAY]
            and isinstance(contracts, list),
            f"{label} contract array closure differs",
        )
        _require(
            isinstance(persistent, list) and persistent == expected_persistent,
            f"{label} persistent array closure differs from the frozen case",
        )
        names = fields + physical + persistent + contracts
        _require(len(names) == len(set(names)), f"{label} repeats array names")
        reference_descriptors = record["reference"]
        candidate_descriptors = record["candidate"]
        _require(
            isinstance(reference_descriptors, list)
            and isinstance(candidate_descriptors, list)
            and len(reference_descriptors)
            == len(candidate_descriptors)
            == len(projection_groups),
            f"{label} projection descriptor groups differ",
        )
        comparison = frozen_comparison_contract(
            manifest, scope, expected["case"], expected["device"]
        )
        _require(
            record["comparison"] == comparison
            and all(
                type(value) is float
                for name, value in record["comparison"].items()
                if name
                in {
                    "rtol",
                    "atol",
                    "linf_limit",
                    "l2_limit",
                    "absolute_scale_floor",
                }
            ),
            f"{label} comparison contract differs from the manifest",
        )
        projection_artifacts, regenerated_names = _regenerate_record_projections(
            root,
            record,
            manifest,
            candidate,
            scope,
            expected,
            projection_groups,
            artifact_loader,
            used_paths,
            used_digests,
            reference_sources,
        )
        _require(
            regenerated_names == (fields, physical, persistent, contracts),
            f"{label} regenerated array inventory differs",
        )
        expected_dtype = _expected_field_dtype(
            scope, expected["case"], expected["device"]
        )
        metrics = {
            "maximum_abs_error": 0.0,
            "maximum_relative_error": 0.0,
            "maximum_normalized_linf_error": 0.0,
            "maximum_normalized_l2_error": 0.0,
        }
        passed = True
        expected_limitation = None
        contract_bindings = None
        reference_index_bindings = {}
        candidate_index_bindings = {}
        for ordinal, (group, reference_descriptor, candidate_descriptor) in enumerate(
            zip(
                projection_groups,
                reference_descriptors,
                candidate_descriptors,
                strict=True,
            )
        ):
            group_label = f"{label} projection group {ordinal}"
            group_names = _projection_group_names(
                group, scope, expected["case"], expected["device"]
            )
            (reference_path, reference_raw), (candidate_path, candidate_raw) = (
                projection_artifacts[ordinal]
            )
            left = _load_projection(
                reference_raw,
                group_names,
                _group_archive_comment(
                    scope,
                    expected["case"],
                    expected["device"],
                    "reference",
                    ordinal,
                    group,
                    candidate,
                ),
                f"{group_label} reference",
            )
            right = _load_projection(
                candidate_raw,
                group_names,
                _group_archive_comment(
                    scope,
                    expected["case"],
                    expected["device"],
                    "candidate",
                    ordinal,
                    group,
                    candidate,
                ),
                f"{group_label} candidate",
            )
            group_fields = _field_projection_names(group)
            _require(
                all(
                    left[name].dtype == right[name].dtype == expected_dtype
                    for name in group_fields
                ),
                f"{group_label} field dtype differs from the frozen runtime",
            )
            for arrays, projection_label, bindings in (
                (left, f"{group_label} reference", reference_index_bindings),
                (right, f"{group_label} candidate", candidate_index_bindings),
            ):
                _validate_projected_array_contract(
                    arrays,
                    _workload(manifest, expected["case"]),
                    scope,
                    expected["case"],
                    expected["device"],
                    group,
                    projection_label,
                    persistent_index_bindings=bindings,
                )
            semantic_payload = _validate_projected_source_contract(
                left[SOURCE_CONTRACT_ARRAY],
                right[SOURCE_CONTRACT_ARRAY],
                _workload(manifest, expected["case"]),
            )
            proof_semantic_payload = _validate_projected_source_proof(
                left[SOURCE_PROOF_ARRAY],
                right[SOURCE_PROOF_ARRAY],
                _workload(manifest, expected["case"]),
                _frozen_capture_steps(manifest, expected["case"]),
                expected["precision"],
                manifest["reference"]["precondition_steps"],
            )
            _require(
                semantic_payload == proof_semantic_payload,
                f"{group_label} PointSource semantic contract differs from its raw proof",
            )
            group_contract_bindings = tuple(
                hashlib.sha256(left[name].tobytes(order="C")).digest()
                for name in contracts
            )
            if contract_bindings is None:
                contract_bindings = group_contract_bindings
            else:
                _require(
                    contract_bindings == group_contract_bindings,
                    f"{group_label} contract bytes change across groups",
                )
            if 100 in group:
                expected_limitation = _precision_limitation(
                    left,
                    scope,
                    expected["case"],
                    expected["device"],
                    projection_steps,
                )
            group_metrics, group_passed = _compare_arrays(
                left,
                right,
                comparison,
                array_comparisons=_frozen_array_comparisons(
                    manifest,
                    scope,
                    expected["case"],
                    expected["device"],
                    left,
                ),
            )
            passed = passed and group_passed
            for metric_name, value in group_metrics.items():
                metrics[metric_name] = max(metrics[metric_name], value)
        _require(
            contract_bindings is not None,
            f"{label} has no projected contract group",
        )
        if expected_limitation is None:
            _require(
                record["precision_limitation"] is None,
                f"{label} has an unexpected precision limitation",
            )
        else:
            limitation = record["precision_limitation"]
            _exact_keys(limitation, PRECISION_LIMITATION_KEYS, f"{label} precision")
            _require(
                limitation == expected_limitation
                and type(limitation["reference_step"]) is int
                and type(limitation["reference_field_max_abs"]) is float
                and type(limitation["rejected_precision_max"]) is float
                and limitation["range_exceeded"] is True,
                f"{label} precision limitation differs from projected reference",
            )
        _exact_keys(
            record["metrics"],
            {
                "maximum_abs_error",
                "maximum_relative_error",
                "maximum_normalized_linf_error",
                "maximum_normalized_l2_error",
            },
            f"{label} metrics",
        )
        _require(
            all(
                isinstance(record["metrics"][name], (int, float))
                and not isinstance(record["metrics"][name], bool)
                and math.isfinite(float(record["metrics"][name]))
                and math.isclose(
                    float(record["metrics"][name]),
                    value,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
                for name, value in metrics.items()
            )
            and record["passed"] is passed
            and passed,
            f"{label} recomputed differential failed",
        )
        recomputed_records.append(
            {
                "case": expected["case"],
                "device": expected["device"],
                "precision": expected["precision"],
                "projection_steps": projection_steps,
                "projection_groups": projection_groups,
                "comparison": comparison,
                "precision_limitation": expected_limitation,
                "metrics": metrics,
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
    supplied = Path(path)
    _require(not supplied.is_symlink(), "differential index cannot be a symlink")
    path = supplied.resolve(strict=True)
    raw = _bounded_regular_file_bytes(path, MAX_INDEX_BYTES, "differential index")
    document = _strict_json_bytes(raw, "differential index")
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
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "media_type": "application/json",
        "candidate_evidence": _candidate_projection(candidate_evidence),
    }
    return result


def _load_candidate_evidence(path):
    value = _strict_json_bytes(
        _bounded_regular_file_bytes(path, MAX_INDEX_BYTES, "candidate evidence"),
        "candidate evidence",
    )
    if isinstance(value, dict) and "evidence" in value:
        value = value["evidence"]
    return _candidate_projection(value)


def _load_trusted_manifest(path):
    raw = _bounded_regular_file_bytes(path, MAX_INDEX_BYTES, "differential manifest")
    _require(
        hashlib.sha256(raw).hexdigest() == TRUSTED_MANIFEST_SHA256,
        "differential manifest differs from the trusted repository manifest",
    )
    value = _strict_json_bytes(raw, "differential manifest")
    _validate_frozen_manifest_contract(value)
    return value, raw


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
    manifest, manifest_raw = _load_trusted_manifest(args.manifest)
    if args.command == "candidate":
        _require(not args.output.exists(), "candidate evidence already exists")
        value = _candidate_projection(
            {
                "candidate_git_commit": host_contract._command_text(
                    "git", "rev-parse", "HEAD"
                ),
                "candidate_git_status": host_contract._command_text(
                    "git", "status", "--short", allow_empty=True
                ),
                "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
            }
        )
        rendered = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 0
    candidate = _load_candidate_evidence(args.candidate_evidence)
    _require(
        candidate["manifest_sha256"] == hashlib.sha256(manifest_raw).hexdigest(),
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
