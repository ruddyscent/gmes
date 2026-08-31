#!/usr/bin/env python3
"""Fail-closed final evidence aggregator for GitHub issue #123."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import stat
import tarfile
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from statistics import median
from typing import Any
from urllib.parse import urlsplit

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "benchmarks" / "native_oracle_workloads.json"

INDEX_KIND = "issue-123-completion-evidence-index"
BUNDLE_SPEC_KIND = "issue-123-completion-bundle-specification"
OUTPUT_KIND = "issue-123-completion-evaluation"
DIFFERENTIAL_KIND = "issue-123-differential-evidence"
MACOS_INDEX_KIND = "issue-123-macos-evidence-index"
FAILURE_RUN_KIND = "two-gpu-failure-run"
TORCH_SOLVER_ABI = "torch-fdtd-regions-v9"
LOCAL_COMPILED_REGION_TOPOLOGY = (
    "local-two-static-half-step-regions+external-boundary-sync-v1"
)

INDEX_SCHEMA_VERSION = 2
BUNDLE_SPEC_SCHEMA_VERSION = 1
BUNDLE_FORMAT = "canonical-directory-v1"
PATH_CONTRACT = "bundle-relative-canonical-posix-v1"
MEDIA_TYPE_JSON = "application/json"
MEDIA_TYPE_NPZ = "application/x-npz"
MEDIA_TYPE_ZIP = "application/zip"
MEDIA_TYPE_WHEEL = "application/vnd.python.wheel+zip"
MEDIA_TYPE_GZIP = "application/gzip"
MEDIA_TYPE_TEXT = "text/plain; charset=utf-8"
MEDIA_TYPE_CPP = "text/x-c++; charset=utf-8"
MEDIA_TYPE_BINARY = "application/octet-stream"
ALLOWED_MEDIA_TYPES = {
    MEDIA_TYPE_JSON,
    MEDIA_TYPE_NPZ,
    MEDIA_TYPE_ZIP,
    MEDIA_TYPE_WHEEL,
    MEDIA_TYPE_GZIP,
    MEDIA_TYPE_TEXT,
    MEDIA_TYPE_CPP,
    MEDIA_TYPE_BINARY,
}
MEDIA_TYPE_SUFFIXES = {
    MEDIA_TYPE_JSON: ".json",
    MEDIA_TYPE_NPZ: ".npz",
    MEDIA_TYPE_ZIP: ".zip",
    MEDIA_TYPE_WHEEL: ".whl",
    MEDIA_TYPE_GZIP: ".tar.gz",
    MEDIA_TYPE_TEXT: ".txt",
    MEDIA_TYPE_CPP: ".cpp",
    MEDIA_TYPE_BINARY: ".bin",
}
MAX_INDEX_BYTES = 16 * 1024 * 1024
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_JSON_BYTES = 256 * 1024 * 1024
MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024
MAX_JSON_DEPTH = 64
MAX_JSON_NODES = 2_000_000
MAX_JSON_COLLECTION_ITEMS = 1_000_000
MAX_JSON_STRING_BYTES = 16 * 1024 * 1024
MAX_ZIP_MEMBERS = 4096
MAX_ZIP_MEMBER_BYTES = 512 * 1024 * 1024
MAX_ZIP_TOTAL_BYTES = 1024 * 1024 * 1024
MAX_ZIP_COMPRESSION_RATIO = 10_000.0
MAX_NPZ_MEMBERS = 512
MAX_NPZ_ARRAY_BYTES = 256 * 1024 * 1024
MAX_NPZ_TOTAL_BYTES = 512 * 1024 * 1024
MAX_NPZ_DIMENSIONS = 8
GPU_TOPOLOGY_NORMALIZATION_RULE = "nvidia-smi-topology-underline-sgr-v1"
GPU_TOPOLOGY_NORMALIZATION_PATH = "/environment/gpu_topology"
DARWIN_SYSTEM_PATH_ALIASES = {
    "/tmp": "/private/tmp",
    "/var": "/private/var",
}

CPU_CASES = (
    "cpu-crossover-2d",
    "cpu-crossover-3d",
    "cpu-large-2d",
    "cpu-large-3d",
    "bloch-2d",
    "bloch-3d",
)
POLICY_CASES = tuple(
    f"coverage-{coverage}-{layout}"
    for coverage in (1, 10, 50, 90)
    for layout in ("contiguous", "fragmented")
)
PAIRED_REAL_CASES = ("bloch-2d", "bloch-3d")
REGION_INVARIANCE_CASES = ("equivalent-region-1", "equivalent-region-32")
PAIRED_REAL_CONTRACT = "cuda-float32-paired-real-tuning-v1"
REGION_INVARIANCE_CONTRACT = "material-region-launch-invariance-v1"
REGION_EQUIVALENCE_GROUP = "overlapping-identical-drude-block-v1"
POLICIES = ("auto", "dense", "compact", "tiled")
FORCED_REPRESENTATIONS = {
    "dense": "dense-mask-scatter-v1",
    "compact": "compact-index-copy-v1",
    "tiled": "tiled-index-scatter-v1",
}
POLICY_WRITE_OPERATIONS = {
    "dense": "aten::masked_scatter_",
    "compact": "aten::index_copy_",
    "tiled": "aten::scatter_",
}
POLICY_DIAGNOSTIC_KIND = "torch-policy-uncompiled-operation-trace"
POLICY_DIAGNOSTIC_CONTRACT = "forced-dispersive-write-op-v1"
COMPILE_CACHE_PREIMAGE_ALGORITHM = "sha256-python-repr-nested-tuples-v1"
CUDA_CASES = (
    "cpu-crossover-2d",
    "cpu-crossover-3d",
    "cpu-large-2d",
    "cpu-large-3d",
    "single-gpu-2d",
    "single-gpu-3d",
)
SINGLE_GPU_CASES = ("single-gpu-2d", "single-gpu-3d")
TWO_GPU_CASES = {
    "strong-mixed": ("strong", 1.6),
    "weak-mixed": ("weak", 0.8),
    "strong-homogeneous": ("informational", None),
    "strong-imbalanced": ("strong", 1.6),
}
TWO_GPU_SIZE_CONTRACTS = {
    "strong-mixed": {
        "serial": [128, 96, 96],
        "distributed": [128, 96, 96],
    },
    "weak-mixed": {
        "serial": [96, 96, 96],
        "distributed": [192, 96, 96],
    },
    "strong-homogeneous": {
        "serial": [128, 96, 96],
        "distributed": [128, 96, 96],
    },
    "strong-imbalanced": {
        "serial": [128, 96, 96],
        "distributed": [128, 96, 96],
    },
}


TWO_GPU_CORRECTNESS_CASES = (
    "axis-0-real",
    "axis-0-bloch",
    "axis-1-real",
    "axis-1-bloch",
    "axis-2-real",
    "axis-2-bloch",
    "collapsed-1d",
    "collapsed-2d",
    "upml",
    "cpml",
    "drude",
    "lorentz",
    "dcp-ade",
    "dcp-plrc",
    "dcp-rc",
    "dm2",
    "tfsf",
    "gaussian",
)
HALO_ANNOTATIONS = tuple(
    f"gmes::halo_{phase}_{operation}"
    for phase in ("magnetic", "electric")
    for operation in ("pack_launch", "exposed_wait", "boundary_unpack")
)
FAILURE_MODES = (
    "strict-peer",
    "dtype-mismatch",
    "checkpoint-mismatch",
    "rank-failure",
)
FAILURE_REASON_CONTRACTS = {
    "strict-peer": {
        "reason_id": "strict-peer-access-unavailable",
        "exit_code_contract": "zero",
        "required_tokens": [
            "cannot directly access",
            "disable require_peer_access",
        ],
    },
    "dtype-mismatch": {
        "reason_id": "rank-precision-mismatch",
        "exit_code_contract": "zero",
        "required_tokens": [
            "both ranks must use the same floating-point precision",
        ],
    },
    "checkpoint-mismatch": {
        "reason_id": "distributed-checkpoint-metadata-mismatch",
        "exit_code_contract": "zero",
        "required_tokens": [
            "distributed checkpoint metadata does not match every rank",
        ],
    },
    "rank-failure": {
        "reason_id": "rank-local-failure-propagated",
        "exit_code_contract": "nonzero",
        "required_tokens": ["injected rank-local failure"],
    },
}
REQUIRED_JOBS = (
    "Python 3.14 / ubuntu-latest",
    "Python 3.14 / macos-latest",
    "CodeQL / python",
    "CodeQL / c-cpp",
)
REQUIRED_RUNTIME_ROLES = (
    "wheel-import",
    "wheel-default-suite",
    "wheel-serial-suite",
    "sdist-import",
    "sdist-default-suite",
    "sdist-serial-suite",
)
FIELD_ARRAYS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
TWO_GPU_CAPTURE_STEPS = (1, 2, 5, 20, 100)
TWO_GPU_CORRECTNESS_ATOL = 2e-10
TUNING_ACCEPTANCE_FIELDS = {
    "compiler_clean",
    "compiled_hot_path_complete",
    "external_indexed_writes_only_sources",
    "steady_state_transfers_zero",
    "storage_stable",
    "memory_bounded",
    "fixed_temporary_contract_satisfied",
    "measurement_contract_matches_manifest",
    "state_progressed",
    "passed",
}
COMPILER_COUNTER_FIELDS = {
    "graph_breaks",
    "unique_graphs",
    "calls_captured",
    "frames_total",
    "frames_ok",
    "fxgraph_cache_hit",
    "fxgraph_cache_miss",
}
CPU_RECOMPUTED_RUNTIME_FIELDS = TUNING_ACCEPTANCE_FIELDS - {
    "fixed_temporary_contract_satisfied",
    "passed",
}
ALLOCATION_CONTRACT_FIELDS = {
    "method",
    "applied",
    "satisfied",
    "status",
    "zero_allocation",
    "checks",
    "errors",
    "provenance",
    "verified_generated_sources",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
NATIVE_SUMMARY_KEYS = {
    "schema_version",
    "kind",
    "physics_reference",
    "observer_tag",
    "observer_commit",
    "benchmark_contract",
    "environment",
    "assembly_contract",
    "source_artifacts",
    "samples",
}
COMMON_HOST_IDENTITY_KEYS = {
    "hostname",
    "platform",
    "os",
    "python",
    "cxx_version",
    "swig_version",
    "uv_lock_sha256",
}
RUNTIME_IDENTITY_KEYS = {"torch", "cuda_runtime"}
HOST_CONTRACT_KEYS = {"schema_version", "common_identity", "runtime_identity"}


class EvidenceError(ValueError):
    """An evidence contract is absent, malformed, or internally inconsistent."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def _exact_keys(
    value: Any,
    required: set[str],
    label: str,
    *,
    optional: set[str] = frozenset(),
) -> None:
    _require(isinstance(value, dict), f"{label} must be an object")
    actual = set(value)
    _require(
        required <= actual <= required | optional,
        f"{label} has an invalid schema: {sorted(actual)!r}",
    )


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise EvidenceError(f"non-finite JSON number: {value}")


def _finite_json_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceError(f"non-finite JSON number: {value}")
    return result


def _strict_json_int(value: str) -> int:
    result = int(value)
    if not -(2**63) <= result <= 2**64 - 1:
        raise EvidenceError("JSON integer is outside the 64-bit evidence bound")
    return result


def _audit_json_bounds(value: Any, label: str) -> None:
    nodes = 0
    stack = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        nodes += 1
        _require(nodes <= MAX_JSON_NODES, f"{label} exceeds the JSON node bound")
        _require(depth <= MAX_JSON_DEPTH, f"{label} exceeds the JSON depth bound")
        if isinstance(current, dict):
            _require(
                len(current) <= MAX_JSON_COLLECTION_ITEMS,
                f"{label} object exceeds the item bound",
            )
            for key, child in current.items():
                _require(type(key) is str, f"{label} object key is not text")
                _require(
                    len(key.encode("utf-8")) <= MAX_JSON_STRING_BYTES,
                    f"{label} object key exceeds the string bound",
                )
                stack.append((child, depth + 1))
        elif isinstance(current, list):
            _require(
                len(current) <= MAX_JSON_COLLECTION_ITEMS,
                f"{label} array exceeds the item bound",
            )
            stack.extend((child, depth + 1) for child in current)
        elif isinstance(current, str):
            _require(
                len(current.encode("utf-8")) <= MAX_JSON_STRING_BYTES,
                f"{label} string exceeds the byte bound",
            )
        else:
            _require(
                current is None or type(current) in {bool, int, float},
                f"{label} contains an unsupported JSON value type",
            )


def _strict_json_bytes(
    raw: bytes,
    label: str,
    *,
    max_bytes: int = MAX_JSON_BYTES,
) -> Any:
    _require(type(raw) is bytes, f"{label} JSON input must be exact bytes")
    _require(len(raw) <= max_bytes, f"{label} exceeds the JSON byte bound")
    try:
        text = raw.decode("utf-8")
        _require(not text.startswith("\ufeff"), f"{label} must not contain a BOM")
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
            parse_float=_finite_json_float,
            parse_int=_strict_json_int,
        )
        _audit_json_bounds(value, label)
        return value
    except EvidenceError:
        raise
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        MemoryError,
        OverflowError,
        RecursionError,
        ValueError,
    ) as error:
        raise EvidenceError(f"{label} is not strict JSON") from error


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return _sha256(raw)


def _finite_float(value: Any, label: str, *, positive: bool = False) -> float:
    _require(
        not isinstance(value, bool) and isinstance(value, (int, float)),
        f"{label} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} must be finite")
    if positive:
        _require(result > 0.0, f"{label} must be positive")
    return result


def _timestamp(value: Any, label: str) -> dt.datetime:
    _require(isinstance(value, str) and bool(value), f"{label} is absent")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise EvidenceError(f"{label} is not an ISO timestamp") from error
    _require(parsed.tzinfo is not None, f"{label} must include a timezone")
    return parsed


def _positive_int(value: Any, label: str) -> int:
    _require(type(value) is int and value > 0, f"{label} must be a positive integer")
    return value


def _is_exact_int(value: Any, expected: int) -> bool:
    return type(value) is int and value == expected


def _type_exact_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _type_exact_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(
            _type_exact_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def _close(actual: Any, expected: float, label: str) -> None:
    actual_value = _finite_float(actual, label)
    _require(
        math.isclose(actual_value, expected, rel_tol=1e-12, abs_tol=0.0),
        f"{label} differs from raw evidence",
    )


def _candidate_evidence(value: Any, manifest_sha256: str) -> dict[str, str]:
    _exact_keys(
        value,
        {"candidate_git_commit", "candidate_git_status", "manifest_sha256"},
        "candidate evidence",
    )
    _require(
        isinstance(value["candidate_git_commit"], str)
        and COMMIT_RE.fullmatch(value["candidate_git_commit"]) is not None,
        "candidate commit must be one full lowercase Git object id",
    )
    _require(
        value["candidate_git_status"] == "",
        "candidate checkout must be clean",
    )
    _require(
        value["manifest_sha256"] == manifest_sha256,
        "candidate manifest digest differs from the supplied manifest bytes",
    )
    return dict(value)


def _document_candidate_matches(
    document: Any,
    expected: dict[str, str],
    *,
    required: bool = False,
) -> None:
    _require(isinstance(document, dict), "artifact root must be an object")
    if "candidate_evidence" in document:
        evidence = document["candidate_evidence"]
    elif "evidence" in document:
        evidence = document["evidence"]
    else:
        _require(not required, "artifact has no embedded candidate evidence")
        return
    _require(
        isinstance(evidence, dict),
        "embedded candidate evidence must be an object",
    )
    for key, value in expected.items():
        _require(
            evidence.get(key) == value,
            f"embedded candidate evidence differs at {key}",
        )


def _validate_host_contract(
    value: Any,
    label: str,
    *,
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        from benchmarks.host_contract import host_contract_complete
    except ImportError as error:
        raise EvidenceError(f"{label} validator is unavailable") from error
    _require(host_contract_complete(value), f"{label} is not canonical schema v2")
    _exact_keys(value, HOST_CONTRACT_KEYS, label)
    common = value["common_identity"]
    runtime = value["runtime_identity"]
    _exact_keys(common, COMMON_HOST_IDENTITY_KEYS, f"{label} common identity")
    _exact_keys(runtime, RUNTIME_IDENTITY_KEYS, f"{label} runtime identity")
    os_record = common["os"]
    _exact_keys(os_record, {"system", "release", "machine"}, f"{label} OS")
    _require(
        _is_exact_int(value["schema_version"], 2)
        and all(
            isinstance(common[name], str) and bool(common[name])
            for name in (
                "hostname",
                "platform",
                "python",
                "cxx_version",
                "swig_version",
            )
        )
        and os_record["system"] == "Linux"
        and all(
            isinstance(os_record[name], str) and bool(os_record[name])
            for name in ("release", "machine")
        )
        and isinstance(common["uv_lock_sha256"], str)
        and SHA256_RE.fullmatch(common["uv_lock_sha256"]) is not None
        and isinstance(runtime["torch"], str)
        and bool(runtime["torch"])
        and (
            runtime["cuda_runtime"] is None
            or isinstance(runtime["cuda_runtime"], str)
            and bool(runtime["cuda_runtime"])
        ),
        f"{label} is incomplete or not Linux",
    )
    if environment is not None:
        _require(
            environment.get("hostname") == common["hostname"]
            and environment.get("platform") == common["platform"]
            and environment.get("python") == common["python"]
            and environment.get("torch") == runtime["torch"]
            and (
                "cuda_runtime" not in environment
                or environment["cuda_runtime"] == runtime["cuda_runtime"]
            ),
            f"{label} differs from its producer environment",
        )
    return dict(value)


def _common_host_identity(host_contract: dict[str, Any]) -> dict[str, Any]:
    """Return host/toolchain identity without a device-specific Torch build."""

    return copy.deepcopy(host_contract["common_identity"])


def _require_successful_command_statuses(
    environment: dict[str, Any],
    label: str,
    *,
    require_cuda_topology: bool,
) -> None:
    statuses = {
        key: value
        for key, value in environment.items()
        if key.endswith("_command_status")
    }
    for key, value in statuses.items():
        if key == "gpu_topology_command_status" and not require_cuda_topology:
            continue
        _require(
            value is None or _is_exact_int(value, 0),
            f"{label} command {key} did not succeed",
        )
    if require_cuda_topology:
        topology_statuses = [
            value
            for key, value in statuses.items()
            if key in {"gpu_topology_command_status", "topology_command_status"}
        ]
        _require(
            len(topology_statuses) == 1 and _is_exact_int(topology_statuses[0], 0),
            f"{label} GPU topology command did not succeed",
        )


def _canonical_bundle_path(value: Any, label: str) -> PurePosixPath:
    _require(isinstance(value, str) and bool(value), f"{label} must be non-empty")
    _require("\\" not in value and "\x00" not in value, f"{label} is not POSIX")
    _require(re.match(r"^[A-Za-z]:", value) is None, f"{label} is absolute")
    parts = value.split("/")
    _require(
        all(part not in {"", ".", ".."} for part in parts),
        f"{label} contains an empty or dot segment",
    )
    path = PurePosixPath(value)
    _require(not path.is_absolute(), f"{label} must be bundle-relative")
    _require(path.as_posix() == value, f"{label} is not canonical POSIX")
    return path


def _permitted_system_path_alias(path: Path, metadata: Any) -> bool:
    """Accept only Darwin's fixed root-level compatibility aliases."""
    target = DARWIN_SYSTEM_PATH_ALIASES.get(str(path))
    if (
        platform.system() != "Darwin"
        or target is None
        or not stat.S_ISLNK(metadata.st_mode)
    ):
        return False
    try:
        return path.resolve(strict=True) == Path(target)
    except OSError:
        return False


def _path_without_symlinks(path: Path, label: str) -> tuple[Path, Any]:
    """Resolve one lexical path only after every existing component is audited."""

    absolute = path if path.is_absolute() else Path.cwd() / path
    anchor = Path(absolute.anchor)
    try:
        metadata = anchor.lstat()
    except OSError as error:
        raise EvidenceError(f"{label} path root is unreadable") from error
    if stat.S_ISLNK(metadata.st_mode):
        _require(
            _permitted_system_path_alias(anchor, metadata),
            f"{label} path uses a symlink",
        )
        metadata = anchor.resolve(strict=True).stat()
    current = anchor
    for part in absolute.parts[1:]:
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as error:
            raise EvidenceError(f"{label} path is unreadable") from error
        if stat.S_ISLNK(metadata.st_mode):
            _require(
                _permitted_system_path_alias(current, metadata),
                f"{label} path uses a symlink",
            )
            metadata = current.resolve(strict=True).stat()
    return current.resolve(strict=True), metadata


def _read_opened_regular_file(
    path: Path,
    label: str,
    *,
    max_bytes: int,
    expected_size: int,
) -> bytes:
    """Read one already-audited file through a no-follow descriptor."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise EvidenceError(
            f"{label} could not be opened without following links"
        ) from error
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file")
        _require(
            before.st_size == expected_size and 0 <= before.st_size <= max_bytes,
            f"{label} byte size differs",
        )
        chunks = []
        remaining = before.st_size + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
    except OSError as error:
        raise EvidenceError(f"{label} bytes are unreadable") from error
    finally:
        os.close(descriptor)
    _require(
        len(raw) == expected_size
        and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        f"{label} changed while being read",
    )
    return raw


def _bounded_regular_file_bytes(
    path: Path,
    label: str,
    *,
    max_bytes: int,
) -> tuple[Path, bytes]:
    checked, metadata = _path_without_symlinks(path, label)
    _require(stat.S_ISREG(metadata.st_mode), f"{label} is not a regular file")
    _require(
        0 <= metadata.st_size <= max_bytes,
        f"{label} exceeds the byte bound",
    )
    raw = _read_opened_regular_file(
        checked, label, max_bytes=max_bytes, expected_size=metadata.st_size
    )
    return checked, raw


def _ensure_directory_without_symlinks(path: Path, label: str) -> Path:
    """Create missing path components without traversing an existing symlink."""

    absolute = path if path.is_absolute() else Path.cwd() / path
    missing: list[Path] = []
    current = absolute
    while True:
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            _require(current.parent != current, f"{label} has no existing ancestor")
            missing.append(current)
            current = current.parent
            continue
        except OSError as error:
            raise EvidenceError(f"{label} is unreadable") from error
        if stat.S_ISLNK(metadata.st_mode):
            _require(
                _permitted_system_path_alias(current, metadata),
                f"{label} uses a symlink",
            )
            metadata = current.resolve(strict=True).stat()
        _require(stat.S_ISDIR(metadata.st_mode), f"{label} ancestor is not a directory")
        _path_without_symlinks(current, label)
        break
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            pass
        try:
            metadata = directory.lstat()
        except OSError as error:
            raise EvidenceError(f"{label} could not be created") from error
        _require(
            not stat.S_ISLNK(metadata.st_mode) and stat.S_ISDIR(metadata.st_mode),
            f"{label} appeared as a symlink or non-directory",
        )
    return absolute.resolve(strict=True)


def _preflight_zip(
    raw: bytes,
    label: str,
    *,
    max_members: int = MAX_ZIP_MEMBERS,
    max_member_bytes: int = MAX_ZIP_MEMBER_BYTES,
    max_total_bytes: int = MAX_ZIP_TOTAL_BYTES,
    expected_files: set[str] | None = None,
) -> list[zipfile.ZipInfo]:
    """Validate ZIP structure and CRCs before any consumer extracts a member."""

    _require(type(raw) is bytes, f"{label} ZIP input must be exact bytes")
    _require(raw.startswith(b"PK"), f"{label} is not ZIP bytes")
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            infos = archive.infolist()
            _require(len(infos) <= max_members, f"{label} has too many ZIP members")
            names: set[str] = set()
            files: set[str] = set()
            total = 0
            for index, info in enumerate(infos):
                member_label = f"{label} ZIP member {index}"
                _require(type(info.filename) is str, f"{member_label} name is invalid")
                name = info.filename[:-1] if info.is_dir() else info.filename
                _canonical_bundle_path(name, f"{member_label} name")
                _require(name not in names, f"{label} repeats ZIP member {name!r}")
                names.add(name)
                mode = info.external_attr >> 16
                _require(
                    not stat.S_ISLNK(mode),
                    f"{member_label} is a symbolic link",
                )
                _require(not info.flag_bits & 0x1, f"{member_label} is encrypted")
                _require(
                    type(info.file_size) is int
                    and 0 <= info.file_size <= max_member_bytes,
                    f"{member_label} exceeds the uncompressed byte bound",
                )
                _require(
                    type(info.compress_size) is int and info.compress_size >= 0,
                    f"{member_label} compressed size is invalid",
                )
                if info.file_size:
                    ratio = info.file_size / max(1, info.compress_size)
                    _require(
                        ratio <= MAX_ZIP_COMPRESSION_RATIO,
                        f"{member_label} exceeds the compression-ratio bound",
                    )
                total += info.file_size
                _require(
                    total <= max_total_bytes, f"{label} exceeds the ZIP byte bound"
                )
                if info.is_dir():
                    continue
                files.add(name)
                observed = 0
                with archive.open(info, "r") as stream:
                    while chunk := stream.read(1024 * 1024):
                        observed += len(chunk)
                        _require(
                            observed <= info.file_size and observed <= max_member_bytes,
                            f"{member_label} expanded beyond its declared size",
                        )
                _require(
                    observed == info.file_size,
                    f"{member_label} uncompressed size differs",
                )
            if expected_files is not None:
                _require(files == expected_files, f"{label} ZIP file closure differs")
            return infos
    except EvidenceError:
        raise
    except (MemoryError, OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise EvidenceError(f"{label} is not a bounded valid ZIP archive") from error


def _validate_media_payload(raw: bytes, media_type: str, label: str) -> None:
    if media_type == MEDIA_TYPE_JSON:
        _strict_json_bytes(raw, label)
    elif media_type == MEDIA_TYPE_NPZ:
        _preflight_zip(
            raw,
            label,
            max_members=MAX_NPZ_MEMBERS,
            max_member_bytes=MAX_NPZ_ARRAY_BYTES,
            max_total_bytes=MAX_NPZ_TOTAL_BYTES,
        )
    elif media_type in {MEDIA_TYPE_ZIP, MEDIA_TYPE_WHEEL}:
        _preflight_zip(raw, label)
    elif media_type == MEDIA_TYPE_GZIP:
        _require(raw.startswith(b"\x1f\x8b"), f"{label} is not gzip bytes")
    elif media_type in {MEDIA_TYPE_TEXT, MEDIA_TYPE_CPP}:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise EvidenceError(f"{label} is not UTF-8 text") from error
        _require(not text.startswith("\ufeff"), f"{label} text must not contain a BOM")


def _validate_descriptor(
    descriptor: Any,
    candidate: dict[str, str],
    label: str,
) -> dict[str, Any]:
    _exact_keys(
        descriptor,
        {"path", "sha256", "size_bytes", "media_type", "candidate_evidence"},
        f"{label} descriptor",
    )
    _canonical_bundle_path(descriptor["path"], f"{label} path")
    _require(
        descriptor["candidate_evidence"] == candidate,
        f"{label} descriptor is bound to a different candidate",
    )
    _require(
        isinstance(descriptor["sha256"], str)
        and SHA256_RE.fullmatch(descriptor["sha256"]) is not None,
        f"{label} digest must be lowercase SHA-256",
    )
    _require(
        type(descriptor["size_bytes"]) is int
        and 0 <= descriptor["size_bytes"] <= MAX_ARTIFACT_BYTES,
        f"{label} size is outside the artifact bound",
    )
    _require(
        descriptor["media_type"] in ALLOWED_MEDIA_TYPES,
        f"{label} media type is unsupported",
    )
    return dict(descriptor)


@dataclass(frozen=True)
class LoadedArtifact:
    descriptor: dict[str, Any]
    path: Path
    raw: bytes
    document: Any | None = None


class ArtifactReader:
    """Read exact-byte artifacts from one relocatable, symlink-free bundle."""

    def __init__(
        self,
        base: Path,
        candidate: dict[str, str],
        registry: list[dict[str, Any]] | None = None,
    ):
        self.base, metadata = _path_without_symlinks(base, "evidence bundle root")
        _require(
            stat.S_ISDIR(metadata.st_mode),
            "evidence bundle root is not a directory",
        )
        self.candidate = candidate
        self._seen: dict[Path, tuple[int, str]] = {}
        self._registry: dict[str, dict[str, Any]] | None = None
        if registry is not None:
            validated = [
                _validate_descriptor(item, candidate, f"payload registry[{index}]")
                for index, item in enumerate(registry)
            ]
            paths = [item["path"] for item in validated]
            _require(paths == sorted(paths), "payload registry is not path-sorted")
            _require(len(paths) == len(set(paths)), "payload registry repeats a path")
            self._registry = {item["path"]: item for item in validated}

    def load(
        self,
        descriptor: Any,
        label: str,
        *,
        json_document: bool = True,
        expected_media_types: set[str] | frozenset[str] | None = None,
        require_embedded_candidate: bool = True,
    ) -> LoadedArtifact:
        descriptor = _validate_descriptor(descriptor, self.candidate, label)
        if self._registry is not None:
            _require(
                self._registry.get(descriptor["path"]) == descriptor,
                f"{label} descriptor is absent from the payload registry",
            )
        if expected_media_types is None and json_document:
            expected_media_types = {MEDIA_TYPE_JSON}
        if expected_media_types is not None:
            _require(
                descriptor["media_type"] in expected_media_types,
                f"{label} media type differs from its role",
            )
        portable = _canonical_bundle_path(descriptor["path"], f"{label} path")
        path = self.base
        for part in portable.parts:
            path = path / part
            try:
                metadata = path.lstat()
            except OSError as error:
                raise EvidenceError(f"{label} path is unreadable") from error
            _require(not stat.S_ISLNK(metadata.st_mode), f"{label} path uses a symlink")
        _require(stat.S_ISREG(metadata.st_mode), f"{label} is not a regular file")
        _require(
            metadata.st_size == descriptor["size_bytes"],
            f"{label} byte size differs",
        )
        path, audited = _path_without_symlinks(path, label)
        _require(stat.S_ISREG(audited.st_mode), f"{label} is not a regular file")
        raw = _read_opened_regular_file(
            path,
            label,
            max_bytes=MAX_ARTIFACT_BYTES,
            expected_size=descriptor["size_bytes"],
        )
        _require(_sha256(raw) == descriptor["sha256"], f"{label} digest differs")
        _validate_media_payload(raw, descriptor["media_type"], label)
        identity = (len(raw), descriptor["sha256"])
        previous = self._seen.setdefault(path, identity)
        _require(previous == identity, f"{label} path has conflicting descriptors")
        document = _strict_json_bytes(raw, label) if json_document else None
        if document is not None and require_embedded_candidate:
            _document_candidate_matches(document, self.candidate)
        return LoadedArtifact(dict(descriptor), path, raw, document)

    def load_many(
        self,
        descriptors: Any,
        label: str,
        *,
        count: int | None = None,
        json_document: bool = True,
        expected_media_types: set[str] | frozenset[str] | None = None,
    ) -> list[LoadedArtifact]:
        _require(isinstance(descriptors, list), f"{label} must be a list")
        if count is not None:
            _require(len(descriptors) == count, f"{label} must contain exactly {count}")
        loaded = [
            self.load(
                item,
                f"{label}[{index}]",
                json_document=json_document,
                expected_media_types=expected_media_types,
            )
            for index, item in enumerate(descriptors)
        ]
        identities = [(item.path, item.descriptor["sha256"]) for item in loaded]
        _require(
            len(set(identities)) == len(identities), f"{label} contains duplicates"
        )
        return loaded


def _raw_values(value: Any, label: str, *, count: int | None = None) -> list[float]:
    _require(isinstance(value, list), f"{label} must be a list")
    if count is not None:
        _require(len(value) == count, f"{label} must contain exactly {count} samples")
    result = [
        _finite_float(item, f"{label}[{index}]", positive=True)
        for index, item in enumerate(value)
    ]
    _require(bool(result), f"{label} must not be empty")
    return result


def _raw_summary(
    summary: Any,
    label: str,
    *,
    steps: int,
    repeats: int,
) -> tuple[list[float], float]:
    _require(isinstance(summary, dict), f"{label} must be an object")
    raw = _raw_values(summary.get("raw_seconds"), f"{label}.raw_seconds", count=repeats)
    middle = median(raw)
    if "median_seconds" in summary:
        _close(summary["median_seconds"], middle, f"{label}.median_seconds")
    if "seconds_per_step" in summary:
        _close(
            summary["seconds_per_step"],
            middle / steps,
            f"{label}.seconds_per_step",
        )
    if "steps_per_second" in summary:
        _close(summary["steps_per_second"], steps / middle, f"{label}.steps_per_second")
    if "repetitions" in summary:
        _require(summary["repetitions"] == repeats, f"{label}.repetitions differs")
    if "steps_per_repeat" in summary:
        _require(
            summary["steps_per_repeat"] == steps, f"{label}.steps_per_repeat differs"
        )
    return raw, middle / steps


def _relative_mad(values: list[float]) -> float:
    middle = median(values)
    return median(abs(value - middle) for value in values) / middle


def _bootstrap_geomean(
    gates: list[dict[str, Any]],
    statistics: dict[str, Any],
) -> dict[str, Any]:
    _require(
        statistics.get("method") == "independent-stratified-bootstrap-log-geomean-v1",
        "CPU bootstrap method differs",
    )
    resamples = _positive_int(statistics.get("resamples"), "CPU bootstrap resamples")
    seed = _positive_int(statistics.get("seed"), "CPU bootstrap seed")
    confidence = _finite_float(
        statistics.get("one_sided_confidence"), "CPU bootstrap confidence"
    )
    threshold = _finite_float(
        statistics.get("regression_ratio"), "CPU regression ratio", positive=True
    )
    _require(0.0 < confidence < 1.0, "CPU bootstrap confidence is outside (0, 1)")
    rng = np.random.default_rng(seed)
    log_ratios = []
    point_ratios = []
    for index, gate in enumerate(gates):
        reference = np.asarray(
            _raw_values(
                gate.get("reference_raw_seconds_per_step"),
                f"CPU gate {index} reference",
            ),
            dtype=np.float64,
        )
        candidate = np.asarray(
            _raw_values(
                gate.get("candidate_raw_seconds_per_step"),
                f"CPU gate {index} candidate",
                count=len(reference),
            ),
            dtype=np.float64,
        )
        reference_indices = rng.integers(
            0, len(reference), size=(resamples, len(reference))
        )
        candidate_indices = rng.integers(
            0, len(candidate), size=(resamples, len(candidate))
        )
        log_ratios.append(
            np.log(
                np.median(candidate[candidate_indices], axis=1)
                / np.median(reference[reference_indices], axis=1)
            )
        )
        point_ratios.append(float(np.median(candidate) / np.median(reference)))
    distribution = np.exp(np.mean(np.stack(log_ratios), axis=0))
    lower = float(np.quantile(distribution, 1.0 - confidence))
    geomean = math.exp(
        sum(math.log(value) for value in point_ratios) / len(point_ratios)
    )
    return {
        "geometric_mean_ratio": geomean,
        "one_sided_lower_bound": lower,
        "significant_regression": lower > threshold,
        "passed": lower <= threshold,
    }


def _walk(value: Any):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk(child)


def _trace_summary(raw: bytes, label: str) -> dict[str, Any]:
    trace = _strict_json_bytes(raw, label)
    _require(isinstance(trace, dict), f"{label} trace root must be an object")
    events = trace.get("traceEvents")
    _require(isinstance(events, list) and bool(events), f"{label} has no trace events")
    kernels = 0
    h2d = 0
    d2h = 0
    device_copies = 0
    allocated = 0
    freed = 0
    allocation_events = 0
    compiled_regions = 0
    cuda_graph_launches = 0
    policy_write_operations = Counter()
    nccl_intervals: list[tuple[float, float]] = []
    compute_intervals: list[tuple[float, float]] = []
    halo_annotations = {
        name: {"count": 0, "duration_us": 0.0} for name in HALO_ANNOTATIONS
    }
    for event in events:
        _require(isinstance(event, dict), f"{label} trace event must be an object")
        name = str(event.get("name", ""))
        lowered = name.lower()
        category = str(event.get("cat", "")).lower()
        if name == "[memory]":
            size = event.get("args", {}).get("Bytes")
            if not isinstance(size, bool) and isinstance(size, (int, float)):
                amount = int(size)
                if amount > 0:
                    allocation_events += 1
                    allocated += amount
                elif amount < 0:
                    freed -= amount
        if event.get("ph") != "X":
            continue
        if name in POLICY_WRITE_OPERATIONS.values():
            policy_write_operations[name] += 1
        if name in halo_annotations:
            duration = _finite_float(
                event.get("dur"),
                f"{label} annotation {name} duration",
            )
            _require(duration >= 0.0, f"{label} annotation duration is negative")
            halo_annotations[name]["count"] += 1
            halo_annotations[name]["duration_us"] += duration
        if lowered.startswith("torch-compiled region:"):
            compiled_regions += 1
        if lowered == "cudagraphlaunch":
            cuda_graph_launches += 1
        if category == "kernel":
            kernels += 1
            if all(key in event for key in ("ts", "dur")):
                start = _finite_float(event["ts"], f"{label} event timestamp")
                duration = _finite_float(event["dur"], f"{label} event duration")
                interval = (start, start + duration)
                if "nccl" in lowered:
                    nccl_intervals.append(interval)
                elif all(token not in lowered for token in ("memcpy", "memset")):
                    compute_intervals.append(interval)
        if "memcpy" in category or "memcpy" in lowered:
            device_copies += 1
            if "htod" in lowered or "host to device" in lowered:
                h2d += 1
            if "dtoh" in lowered or "device to host" in lowered:
                d2h += 1
    nccl = _interval_duration(nccl_intervals)
    overlap = _intersection_duration(nccl_intervals, compute_intervals)
    return {
        "chrome_trace_size_bytes": len(raw),
        "chrome_trace_sha256": _sha256(raw),
        "kernel_launches": kernels,
        "device_copy_events": device_copies,
        "host_to_device_events": h2d,
        "device_to_host_events": d2h,
        "positive_allocation_events": allocation_events,
        "allocated_bytes": allocated,
        "freed_bytes": freed,
        "allocation_net_bytes": allocated - freed,
        "compiled_region_events": compiled_regions,
        "cuda_graph_launches": cuda_graph_launches,
        "policy_write_operations": {
            name: policy_write_operations[name]
            for name in POLICY_WRITE_OPERATIONS.values()
        },
        "nccl_kernel_launches": len(nccl_intervals),
        "nccl_device_us": nccl,
        "nccl_compute_overlap_us": overlap,
        "nccl_exposed_us": max(0.0, nccl - overlap),
        "overlap_fraction": overlap / nccl if nccl else 0.0,
        "halo_annotations": halo_annotations,
    }


def _interval_union(intervals: list[tuple[float, float]]) -> list[list[float]]:
    merged: list[list[float]] = []
    for start, stop in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)
    return merged


def _interval_duration(intervals: list[tuple[float, float]]) -> float:
    return sum(stop - start for start, stop in _interval_union(intervals))


def _intersection_duration(
    first: list[tuple[float, float]],
    second: list[tuple[float, float]],
) -> float:
    left = _interval_union(first)
    right = _interval_union(second)
    result = 0.0
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        start = max(left[left_index][0], right[right_index][0])
        stop = min(left[left_index][1], right[right_index][1])
        result += max(0.0, stop - start)
        if left[left_index][1] < right[right_index][1]:
            left_index += 1
        else:
            right_index += 1
    return result


def _load_traces(
    reader: ArtifactReader,
    descriptors: Any,
    label: str,
    *,
    count: int,
    require_nccl: bool = False,
) -> dict[str, tuple[LoadedArtifact, dict[str, Any]]]:
    loaded = reader.load_many(
        descriptors,
        label,
        count=count,
        json_document=False,
        expected_media_types={MEDIA_TYPE_JSON},
    )
    result: dict[str, tuple[LoadedArtifact, dict[str, Any]]] = {}
    for index, artifact in enumerate(loaded):
        summary = _trace_summary(artifact.raw, f"{label}[{index}]")
        if require_nccl:
            _require(
                summary["nccl_device_us"] > 0.0, f"{label}[{index}] has no NCCL kernel"
            )
        key = artifact.descriptor["sha256"]
        _require(key not in result, f"{label} repeats a trace digest")
        result[key] = (artifact, summary)
    return result


def _bind_tuning_traces(
    document: dict[str, Any],
    traces: dict[str, tuple[LoadedArtifact, dict[str, Any]]],
    label: str,
) -> None:
    profilers = [
        value
        for value in _walk(document)
        if isinstance(value, dict)
        and {"chrome_trace_sha256", "chrome_trace_size_bytes"} <= set(value)
    ]
    _require(len(profilers) == len(traces), f"{label} profiler/trace count differs")
    used: set[str] = set()
    for profiler in profilers:
        digest = profiler["chrome_trace_sha256"]
        _require(digest in traces, f"{label} references an unindexed trace")
        _require(digest not in used, f"{label} repeats one profiler trace")
        used.add(digest)
        _artifact, recomputed = traces[digest]
        for name in (
            "chrome_trace_sha256",
            "chrome_trace_size_bytes",
            "kernel_launches",
            "device_copy_events",
            "host_to_device_events",
            "device_to_host_events",
            "positive_allocation_events",
            "allocated_bytes",
            "freed_bytes",
            "allocation_net_bytes",
            "compiled_region_events",
            "cuda_graph_launches",
        ):
            if name in profiler:
                _require(
                    profiler[name] == recomputed[name],
                    f"{label} profiler field {name} differs from trace bytes",
                )
    _require(used == set(traces), f"{label} contains unreferenced trace artifacts")


def _validate_correctness_index(
    artifact: LoadedArtifact,
    manifest: dict[str, Any],
    candidate: dict[str, str],
    reader: ArtifactReader,
) -> dict[str, Any]:
    document = artifact.document
    _require(isinstance(document, dict), "correctness index root must be an object")
    _document_candidate_matches(document, candidate, required=True)
    evidence = document.get("candidate_evidence")
    _require(isinstance(evidence, dict), "correctness index has no candidate evidence")
    _require(
        document.get("manifest_contract_sha256") == _canonical_sha256(manifest),
        "correctness index canonical manifest digest differs",
    )
    required = [
        case["name"]
        for group in ("correctness", "physical_checks")
        for case in manifest.get(group, ())
    ]
    _require(
        document.get("required_cases") == required, "correctness case closure differs"
    )
    suite = document.get("suite_acceptance")
    _require(isinstance(suite, dict), "correctness suite acceptance is absent")
    expected_suite = {
        "correctness_case_count": len(manifest.get("correctness", ())),
        "physical_check_case_count": len(manifest.get("physical_checks", ())),
        "evaluated_case_count": len(required),
        "complete_fields": True,
        "persistent_state": True,
        "source_and_auxiliary_state": True,
        "physical_observables": True,
        "passed": True,
    }
    _require(
        _type_exact_equal(suite, expected_suite),
        "correctness suite acceptance is not exact",
    )
    artifacts = document.get("artifacts")
    _require(
        isinstance(artifacts, list) and len(artifacts) == len(required),
        "correctness archive descriptor closure differs",
    )
    nested = []
    for index, record in enumerate(artifacts):
        _require(isinstance(record, dict), "correctness artifact record differs")
        for role in ("reference", "candidate"):
            nested.append(
                reader.load(
                    record.get(role),
                    f"correctness {index} {role} archive",
                    json_document=False,
                    expected_media_types={MEDIA_TYPE_NPZ},
                )
            )
    try:
        from benchmarks.torch_correctness import load_correctness_evidence_index

        rebuilt = load_correctness_evidence_index(
            artifact.path,
            manifest,
            evidence,
            descriptor_root=reader.base,
        )
    except (
        ImportError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise EvidenceError(
            f"correctness archives could not be independently recomputed: {error}"
        ) from error
    _require(
        rebuilt.get("source_artifact", {}).get("sha256")
        == artifact.descriptor["sha256"],
        "correctness source index digest differs after recomputation",
    )
    for index, loaded in enumerate(nested):
        checked = reader.load(
            loaded.descriptor,
            f"correctness archive post-validation {index}",
            json_document=False,
            expected_media_types={MEDIA_TYPE_NPZ},
        )
        _require(
            checked.raw == loaded.raw, "correctness archive changed during validation"
        )
    return rebuilt


def _validate_tuning_acceptance(
    result: Any,
    label: str,
    *,
    allow_allocation_override: bool = False,
) -> None:
    _require(isinstance(result, dict), f"{label} must be an object")
    acceptance = result.get("acceptance")
    _exact_keys(acceptance, TUNING_ACCEPTANCE_FIELDS, f"{label} acceptance")
    expected_pass = all(
        acceptance[name] for name in TUNING_ACCEPTANCE_FIELDS if name != "passed"
    )
    _require(
        acceptance["passed"] is expected_pass
        and all(
            value is True
            for name, value in acceptance.items()
            if not allow_allocation_override
            or name not in {"fixed_temporary_contract_satisfied", "passed"}
        ),
        f"{label} embedded runtime acceptance is not wholly true",
    )
    compiler = result.get("compiler")
    _exact_keys(
        compiler,
        {
            "after_cold",
            "after_warmup",
            "after_steady",
            "steady_state_delta",
            "fullgraph_clean",
        },
        f"{label} compiler evidence",
    )
    for name in ("after_cold", "after_warmup", "after_steady"):
        snapshot = compiler[name]
        _exact_keys(snapshot, COMPILER_COUNTER_FIELDS, f"{label} compiler {name}")
        _require(
            all(type(value) is int and value >= 0 for value in snapshot.values()),
            f"{label} compiler {name} counters are malformed",
        )
    expected_delta = {
        name: compiler["after_steady"][name] - compiler["after_warmup"][name]
        for name in COMPILER_COUNTER_FIELDS
    }
    _require(
        compiler["steady_state_delta"] == expected_delta
        and all(value == 0 for value in expected_delta.values())
        and compiler["after_steady"]["graph_breaks"] == 0
        and compiler["fullgraph_clean"] is True,
        f"{label} compiler evidence is not clean",
    )
    memory = result.get("memory")
    _require(
        isinstance(memory, dict)
        and isinstance(memory.get("storage_addresses_before"), dict)
        and bool(memory["storage_addresses_before"])
        and memory.get("storage_addresses_after") == memory["storage_addresses_before"]
        and memory.get("storage_addresses_stable") is True
        and memory.get("bounded") is True,
        f"{label} storage or memory evidence failed",
    )
    allocation = result.get("allocation_contract")
    _require(
        isinstance(allocation, dict)
        and (
            allocation.get("satisfied") is True
            or allow_allocation_override
            and allocation.get("satisfied") is False
        ),
        f"{label} allocation contract failed",
    )


def _validate_cpu_recomputed_acceptance(
    summary: Any,
    label: str,
    manifest: dict[str, Any],
) -> None:
    _require(isinstance(summary, dict), f"{label} aggregate summary is malformed")
    runtime_records = summary.get("recomputed_runtime_acceptance")
    _require(
        isinstance(runtime_records, list) and len(runtime_records) == len(CPU_CASES),
        f"{label} recomputed runtime acceptance closure differs",
    )
    for index, record in enumerate(runtime_records):
        record_label = f"{label} runtime acceptance {CPU_CASES[index]}"
        _exact_keys(record, CPU_RECOMPUTED_RUNTIME_FIELDS, record_label)
        _require(
            all(value is True for value in record.values()),
            f"{record_label} failed",
        )

    allocation_records = summary.get("recomputed_allocation_contracts")
    _require(
        isinstance(allocation_records, list)
        and len(allocation_records) == len(CPU_CASES),
        f"{label} recomputed allocation closure differs",
    )
    expected_method = manifest["performance_gates"]["cpu_acceptance"][
        "allocation_contract"
    ]["method"]
    for index, record in enumerate(allocation_records):
        record_label = f"{label} allocation {CPU_CASES[index]}"
        _exact_keys(record, ALLOCATION_CONTRACT_FIELDS, record_label)
        checks = record["checks"]
        sources = record["verified_generated_sources"]
        _require(
            record["method"] == expected_method
            and record["applied"] is True
            and record["satisfied"] is True
            and record["errors"] == []
            and isinstance(checks, dict)
            and bool(checks)
            and all(value is True for value in checks.values())
            and type(record["zero_allocation"]) is bool
            and isinstance(sources, list),
            f"{record_label} failed",
        )
        if record["zero_allocation"]:
            _require(
                record["status"] == "zero-allocation"
                and record["provenance"] is None
                and sources == [],
                f"{record_label} zero-allocation evidence differs",
            )
        else:
            _require(
                record["status"] == "reviewed-fixed-temporary"
                and isinstance(record["provenance"], dict)
                and bool(sources)
                and all(
                    isinstance(source, dict)
                    and source.get("matches_provenance") is True
                    and isinstance(source.get("path"), str)
                    and bool(source["path"])
                    and isinstance(source.get("sha256"), str)
                    and SHA256_RE.fullmatch(source["sha256"]) is not None
                    for source in sources
                ),
                f"{record_label} reviewed provenance differs",
            )


def _all_traces_have_zero_allocations(
    traces: dict[str, tuple[LoadedArtifact, dict[str, Any]]],
) -> bool:
    metrics = (
        "positive_allocation_events",
        "allocated_bytes",
        "freed_bytes",
        "allocation_net_bytes",
    )
    return all(
        all(summary.get(name) == 0 for name in metrics)
        for _artifact, summary in traces.values()
    )


def _load_cpu_allocation_evidence(
    scope: dict[str, Any],
    reader: ArtifactReader,
    aggregate: dict[str, Any],
    *,
    all_zero: bool,
) -> tuple[dict[str, Any] | None, dict[str, LoadedArtifact]]:
    binding = aggregate.get("allocation_provenance_artifact")
    if all_zero:
        _require(
            scope["allocation_sidecars"] == []
            and scope["generated_sources"] == []
            and binding is None,
            "zero-allocation CPU evidence must not carry provenance artifacts",
        )
        return None, {}
    sidecars = reader.load_many(
        scope["allocation_sidecars"],
        "CPU allocation sidecars",
        count=1,
        expected_media_types={MEDIA_TYPE_JSON},
    )
    generated = reader.load_many(
        scope["generated_sources"],
        "CPU generated sources",
        json_document=False,
        expected_media_types={MEDIA_TYPE_CPP, MEDIA_TYPE_TEXT, MEDIA_TYPE_BINARY},
    )
    _require(bool(generated), "CPU generated source closure is empty")
    by_sha = {item.descriptor["sha256"]: item for item in generated}
    _require(len(by_sha) == len(generated), "CPU generated sources repeat a digest")
    _exact_keys(binding, {"path", "sha256"}, "CPU allocation sidecar binding")
    _require(
        binding["sha256"] == sidecars[0].descriptor["sha256"],
        "CPU aggregate allocation sidecar digest differs",
    )
    try:
        from benchmarks.torch_tuning import _load_allocation_provenance

        document = _load_allocation_provenance(sidecars[0].path)
    except (ImportError, KeyError, OSError, TypeError, ValueError) as error:
        raise EvidenceError(f"CPU allocation sidecar is invalid: {error}") from error
    records = document.get("records")
    _require(isinstance(records, list), "CPU allocation records are absent")
    used: set[str] = set()
    for record_index, record in enumerate(records):
        sources = record.get("generated_sources")
        if sources is None:
            continue
        _require(
            isinstance(sources, list) and bool(sources),
            f"CPU allocation record {record_index} generated sources are malformed",
        )
        for source_index, source in enumerate(sources):
            label = f"CPU allocation record {record_index} source {source_index}"
            _exact_keys(source, {"path", "sha256"}, label)
            digest = source["sha256"]
            _require(digest in by_sha, f"{label} is absent from the bundle")
            source["path"] = str(by_sha[digest].path)
            used.add(digest)
    _require(
        used == set(by_sha), "CPU bundle contains an unreferenced generated source"
    )
    return document, by_sha


def _normalize_allocation_source_paths(
    value: Any,
    sources_by_sha: dict[str, LoadedArtifact],
    label: str,
) -> Any:
    result = copy.deepcopy(value)

    def normalize(items: Any, item_label: str) -> None:
        _require(isinstance(items, list), f"{item_label} must be a list")
        for index, item in enumerate(items):
            _require(isinstance(item, dict), f"{item_label}[{index}] is malformed")
            digest = item.get("sha256")
            _require(digest in sources_by_sha, f"{item_label}[{index}] is unindexed")
            _require(
                isinstance(item.get("path"), str),
                f"{item_label}[{index}] path is absent",
            )
            item["path"] = sources_by_sha[digest].descriptor["path"]

    provenance = result.get("provenance") if isinstance(result, dict) else None
    if isinstance(provenance, dict) and "generated_sources" in provenance:
        normalize(provenance["generated_sources"], f"{label} provenance sources")
    if isinstance(result, dict) and "verified_generated_sources" in result:
        normalize(result["verified_generated_sources"], f"{label} verified sources")
    return result


def _recompute_cpu_case_acceptance(
    cases: list[dict[str, Any]],
    summary: dict[str, Any],
    traces: dict[str, tuple[LoadedArtifact, dict[str, Any]]],
    allocation_document: dict[str, Any] | None,
    sources_by_sha: dict[str, LoadedArtifact],
    manifest: dict[str, Any],
    expected_evidence: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    try:
        from benchmarks import torch_tuning as tuning
    except ImportError as error:
        raise EvidenceError("CPU raw recomputation helpers are unavailable") from error
    recorded_runtime = summary["recomputed_runtime_acceptance"]
    recorded_allocation = summary["recomputed_allocation_contracts"]
    allocation_definition = manifest["performance_gates"]["cpu_acceptance"][
        "allocation_contract"
    ]
    recomputed_runtime = []
    recomputed_allocation = []
    for index, case in enumerate(cases):
        case_label = f"{label} {CPU_CASES[index]}"
        relocated = copy.deepcopy(case)
        profiler = relocated.get("profiler")
        _require(isinstance(profiler, dict), f"{case_label} profiler is absent")
        digest = profiler.get("chrome_trace_sha256")
        _require(digest in traces, f"{case_label} raw trace is unindexed")
        profiler["chrome_trace"] = str(traces[digest][0].path)
        _require(
            tuning._profiler_trace_matches(profiler),
            f"{case_label} profiler differs from its raw trace",
        )
        runtime, runtime_errors = tuning._recompute_cpu_runtime_acceptance(
            relocated,
            manifest,
            expected_evidence,
        )
        _require(not runtime_errors, f"{case_label} runtime failed: {runtime_errors!r}")
        _require(
            _type_exact_equal(runtime, recorded_runtime[index]),
            f"{case_label} recorded runtime acceptance differs from raw evidence",
        )

        positive_events = profiler.get("positive_allocation_events")
        _require(
            type(positive_events) is int and positive_events >= 0,
            f"{case_label} allocation event count is malformed",
        )
        provenance = None
        runtime_contract = relocated.get("runtime")
        _require(isinstance(runtime_contract, dict), f"{case_label} runtime is absent")
        if positive_events:
            try:
                provenance = tuning._select_allocation_provenance(
                    allocation_document,
                    workload=relocated.get("workload", {}).get("name"),
                    device=runtime_contract.get("device"),
                    precision=runtime_contract.get("precision"),
                    compile_mode=runtime_contract.get("compile_mode"),
                    execution_policy=runtime_contract.get("execution_policy"),
                    threads=runtime_contract.get("threads"),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise EvidenceError(
                    f"{case_label} allocation provenance selection failed: {error}"
                ) from error
        allocation = tuning._fixed_temporary_allocation_contract(
            tuning.torch.device("cpu"),
            profiler,
            compile_cache_key=runtime_contract.get("compile_cache_key"),
            allocation_provenance=provenance,
            public_upstream_issue_required=allocation_definition[
                "public_upstream_issue_required"
            ],
        )
        _require(
            allocation.get("method") == allocation_definition["method"]
            and allocation.get("applied") is True
            and allocation.get("satisfied") is True
            and allocation.get("errors") == [],
            f"{case_label} allocation contract failed raw recomputation",
        )
        normalized = _normalize_allocation_source_paths(
            allocation,
            sources_by_sha,
            f"{case_label} recomputed allocation",
        )
        expected = _normalize_allocation_source_paths(
            recorded_allocation[index],
            sources_by_sha,
            f"{case_label} recorded allocation",
        )
        _require(
            _type_exact_equal(normalized, expected),
            f"{case_label} recorded allocation differs from raw evidence",
        )
        recomputed_runtime.append(runtime)
        recomputed_allocation.append(normalized)
    return {
        "runtime": recomputed_runtime,
        "allocation": recomputed_allocation,
    }


def _native_benchmark_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    reference = manifest["reference"]
    return {
        "initializer": reference["field_initializer"],
        "seed": reference["seed"],
        "field_scale": reference["field_scale"],
        "warmup_steps": reference["performance_warmup_steps"],
        "steps_per_repeat": reference["performance_steps_per_repeat"],
        "repetitions": reference["performance_repetitions"],
        "timer": "time.perf_counter",
        "sample_start": "independently-rebuilt-post-warmup-state",
    }


def _validate_native_summary(
    artifact: LoadedArtifact,
    manifest: dict[str, Any],
    raw_artifacts: list[LoadedArtifact],
    manifest_path: Path,
) -> dict[str, Any]:
    document = artifact.document
    _exact_keys(document, NATIVE_SUMMARY_KEYS, "native summary")
    reference = manifest["reference"]
    _require(
        artifact.descriptor["sha256"] == reference["performance_summary_sha256"]
        and _is_exact_int(document["schema_version"], 3)
        and document["kind"] == "native-cpu-acceptance-summary"
        and document["physics_reference"] == reference["tag"]
        and document["observer_tag"] == reference["performance_observer_tag"]
        and document["observer_commit"] == reference["performance_observer_commit"],
        "native summary exact bytes or observer provenance differs",
    )
    contract = _native_benchmark_contract(manifest)
    _require(
        _type_exact_equal(document["benchmark_contract"], contract),
        "native summary benchmark contract differs",
    )
    _require(
        _type_exact_equal(
            document["assembly_contract"],
            {
                "id": "native-cpu-summary-v2",
                "input_schema_version": 2,
                "input_count": 12,
                "sample_order": "frozen-case-order-then-one-physical",
                "ansi_escape_handling": (
                    "reject-except-balanced-nvidia-smi-topology-underline-sgr"
                ),
                "gpu_topology_normalization_rule": (GPU_TOPOLOGY_NORMALIZATION_RULE),
                "gpu_topology_normalization_path": (GPU_TOPOLOGY_NORMALIZATION_PATH),
                "per_cell_environment_fields": [
                    "openmp_threads",
                    "omp_num_threads",
                ],
                "normalized_cpu_model_line_prefixes": ["CPU(s) scaling MHz:"],
            },
        ),
        "native summary assembly provenance differs",
    )
    environment = document["environment"]
    _require(
        isinstance(environment, dict)
        and environment.get("git_commit") == reference["performance_observer_commit"]
        and environment.get("git_status") == ""
        and environment.get("openmp_enabled") is True
        and all(
            isinstance(environment.get(name), str) and bool(environment[name])
            for name in (
                "hostname",
                "platform",
                "python",
                "cxx_version",
                "swig_version",
            )
        )
        and isinstance(environment.get("uv_lock_sha256"), str)
        and SHA256_RE.fullmatch(environment["uv_lock_sha256"]) is not None
        and isinstance(environment.get("os"), dict)
        and environment["os"].get("system") == "Linux"
        and isinstance(environment["os"].get("machine"), str)
        and bool(environment["os"]["machine"]),
        "native summary host/toolchain provenance is incomplete",
    )

    acceptance = manifest["performance_gates"]["cpu_acceptance"]
    pins = acceptance["timing_reference"]["slice_artifacts"]
    threads_by_mode = {
        pin["thread_mode"]: pin["threads"] for pin in pins if isinstance(pin, dict)
    }
    _require(
        threads_by_mode.keys() == {"one", "physical"}
        and _is_exact_int(threads_by_mode["one"], 1)
        and type(threads_by_mode["physical"]) is int
        and threads_by_mode["physical"] > 1,
        "native summary thread pins differ",
    )
    expected_cells = [
        (name, mode, threads_by_mode[mode])
        for name in CPU_CASES
        for mode in ("one", "physical")
    ]
    samples = document["samples"]
    sources = document["source_artifacts"]
    _require(
        isinstance(samples, list)
        and isinstance(sources, list)
        and len(samples) == len(sources) == len(expected_cells),
        "native summary must contain the exact twelve-cell matrix",
    )
    raw_by_mode = {"one": {}, "physical": {}}
    source_digests = set()
    for index, ((name, mode, threads), sample, source) in enumerate(
        zip(expected_cells, samples, sources, strict=True)
    ):
        label = f"native summary {name}/{mode}"
        _exact_keys(
            sample,
            {
                "workload",
                "threads",
                "openmp_threads",
                "benchmark_contract",
                "measurements",
                "memory",
                "updaters",
                "profiler",
            },
            label,
        )
        _exact_keys(
            source,
            {
                "workload",
                "thread_mode",
                "threads",
                "sha256",
                "normalization_provenance",
                "raw_environment",
            },
            f"{label} source",
        )
        raw_environment = source["raw_environment"]
        normalization = source["normalization_provenance"]
        _exact_keys(
            normalization,
            {
                "rule_id",
                "json_pointer",
                "source_sha256",
                "applied",
                "removed_pair_count",
            },
            f"{label} normalization provenance",
        )
        _exact_keys(
            raw_environment,
            {"cpu_model", "openmp_threads", "omp_num_threads"},
            f"{label} raw environment",
        )
        _require(
            _type_exact_equal(sample["workload"], _manifest_case(manifest, name))
            and sample["threads"] == str(threads)
            and _is_exact_int(sample["openmp_threads"], threads)
            and _type_exact_equal(sample["benchmark_contract"], contract)
            and sample["profiler"] is None
            and source["workload"] == name
            and source["thread_mode"] == mode
            and _is_exact_int(source["threads"], threads)
            and isinstance(source["sha256"], str)
            and SHA256_RE.fullmatch(source["sha256"]) is not None
            and source["sha256"] not in source_digests
            and normalization["rule_id"] == GPU_TOPOLOGY_NORMALIZATION_RULE
            and normalization["json_pointer"] == GPU_TOPOLOGY_NORMALIZATION_PATH
            and normalization["source_sha256"] == source["sha256"]
            and type(normalization["applied"]) is bool
            and type(normalization["removed_pair_count"]) is int
            and normalization["removed_pair_count"] >= 0
            and normalization["applied"] is (normalization["removed_pair_count"] > 0)
            and isinstance(raw_environment["cpu_model"], str)
            and bool(raw_environment["cpu_model"])
            and _is_exact_int(raw_environment["openmp_threads"], threads)
            and raw_environment["omp_num_threads"] == str(threads),
            f"{label} identity or per-cell provenance differs",
        )
        source_digests.add(source["sha256"])
        advance = sample.get("measurements", {}).get("advance")
        raw, _seconds = _raw_summary(
            advance,
            f"{label} advance",
            steps=contract["steps_per_repeat"],
            repeats=contract["repetitions"],
        )
        relative_mad = _relative_mad(raw)
        _close(advance.get("relative_mad"), relative_mad, f"{label} relative MAD")
        _require(
            relative_mad <= acceptance["statistics"]["max_relative_mad"],
            f"{label} relative MAD exceeds the frozen limit",
        )
        raw_by_mode[mode][name] = [
            value / contract["steps_per_repeat"] for value in raw
        ]
    _require(len(raw_artifacts) == 12, "native raw artifact closure differs")
    _require(
        {item.descriptor["sha256"] for item in raw_artifacts} == source_digests,
        "native raw descriptors differ from summary source digests",
    )
    try:
        from benchmarks.native_summary import assemble_summary

        rebuilt = assemble_summary(
            [item.path for item in raw_artifacts],
            manifest_path,
        )
    except (ImportError, KeyError, OSError, TypeError, ValueError) as error:
        raise EvidenceError(
            f"native raw matrix could not be independently assembled: {error}"
        ) from error
    _require(
        _type_exact_equal(rebuilt, document),
        "native summary differs from the twelve raw native artifacts",
    )
    return {
        "environment": environment,
        "raw_seconds_per_step": raw_by_mode,
        "summary_sha256": artifact.descriptor["sha256"],
    }


def _load_pinned_torch_baseline(
    manifest: dict[str, Any],
    reader: ArtifactReader,
    artifact_descriptors: Any,
) -> tuple[
    dict[str, dict[str, list[float]]],
    dict[str, str],
    str,
    dict[str, Any],
    dict[str, dict[str, str | None]],
    dict[str, dict[str, Any]],
]:
    try:
        from benchmarks.torch_cpu_baseline import load_torch_cpu_baseline

        pins = manifest["performance_gates"]["cpu_acceptance"]["timing_reference"][
            "slice_artifacts"
        ]
        artifacts = reader.load_many(
            artifact_descriptors,
            "pinned Torch CPU baseline artifacts",
            count=2,
            json_document=False,
            expected_media_types={MEDIA_TYPE_JSON},
        )
        _require(
            isinstance(pins, list) and len(pins) == len(artifacts),
            "pinned Torch CPU baseline manifest closure differs",
        )
        for index, (artifact, pin) in enumerate(zip(artifacts, pins, strict=True)):
            publication_url = (
                pin.get("publication_url") if isinstance(pin, dict) else None
            )
            parsed_url = (
                urlsplit(publication_url) if isinstance(publication_url, str) else None
            )
            asset_name = PurePosixPath(parsed_url.path).name if parsed_url else None
            _require(
                isinstance(pin, dict)
                and artifact.descriptor["sha256"] == pin.get("sha256")
                and artifact.descriptor["size_bytes"] == pin.get("size_bytes")
                and artifact.path.name == asset_name
                and bool(asset_name),
                f"pinned Torch CPU baseline artifact {index} differs from its Release pin",
            )
        baseline = load_torch_cpu_baseline(
            [artifact.path for artifact in artifacts], manifest
        )
    except (ImportError, KeyError, OSError, TypeError, ValueError) as error:
        raise EvidenceError(
            f"pinned Torch CPU Release baseline bytes could not be validated: {error}"
        ) from error
    raw_by_mode: dict[str, dict[str, list[float]]] = {}
    source_sha_by_mode = {}
    thread_environment_by_mode = {}
    sources_by_mode = {}
    for source in baseline["source_artifacts"]:
        mode = source["thread_mode"]
        source_sha_by_mode[mode] = source["sha256"]
        thread_environment_by_mode[mode] = source["thread_environment"]
        sources_by_mode[mode] = source
        raw_by_mode[mode] = {
            case["name"]: list(case["measurements"]["advance"]["raw_seconds_per_step"])
            for case in source["cases"]
        }
    pins_by_mode = {
        pin.get("thread_mode"): pin for pin in pins if isinstance(pin, dict)
    }
    _require(
        set(pins_by_mode) == set(sources_by_mode) == {"one", "physical"}
        and all(
            sources_by_mode[mode].get("publication_url")
            == pins_by_mode[mode].get("publication_url")
            and sources_by_mode[mode].get("size_bytes")
            == pins_by_mode[mode].get("size_bytes")
            and sources_by_mode[mode].get("sha256") == pins_by_mode[mode].get("sha256")
            for mode in ("one", "physical")
        ),
        "pinned Torch CPU Release baseline provenance differs",
    )
    _require(
        set(raw_by_mode) == {"one", "physical"}
        and all(set(cases) == set(CPU_CASES) for cases in raw_by_mode.values()),
        "pinned Torch CPU Release baseline cell closure differs",
    )
    return (
        raw_by_mode,
        source_sha_by_mode,
        baseline["timing_reference"]["root_commit"],
        baseline["environment"],
        thread_environment_by_mode,
        sources_by_mode,
    )


def _require_cpu_baseline_publication_provenance(
    aggregate: dict[str, Any],
    manifest: dict[str, Any],
    sources_by_mode: dict[str, dict[str, Any]],
) -> None:
    provenance = aggregate.get("torch_baseline")
    _exact_keys(
        provenance,
        {
            "kind",
            "cpu_acceptance_contract_id",
            "timing_reference",
            "source_artifacts",
        },
        "CPU aggregate Torch baseline provenance",
    )
    acceptance = manifest["performance_gates"]["cpu_acceptance"]
    _require(
        provenance["kind"] == "torch-cpu-baseline"
        and provenance["cpu_acceptance_contract_id"] == acceptance["contract_id"]
        and _type_exact_equal(
            provenance["timing_reference"], acceptance["timing_reference"]
        )
        and isinstance(provenance["source_artifacts"], list)
        and len(provenance["source_artifacts"]) == 2,
        "CPU aggregate Torch baseline provenance is incomplete",
    )
    reported_by_mode = {
        source.get("thread_mode"): source
        for source in provenance["source_artifacts"]
        if isinstance(source, dict)
    }
    _require(
        set(reported_by_mode) == set(sources_by_mode) == {"one", "physical"},
        "CPU aggregate Torch baseline publication modes differ",
    )
    for mode in ("one", "physical"):
        source = sources_by_mode[mode]
        reported = reported_by_mode[mode]
        _exact_keys(
            reported,
            {
                "publication_url",
                "size_bytes",
                "sha256",
                "thread_mode",
                "threads",
                "thread_environment",
                "root_commit",
            },
            f"CPU aggregate Torch baseline {mode} publication",
        )
        _require(
            all(
                _type_exact_equal(reported.get(name), source.get(name))
                for name in (
                    "publication_url",
                    "size_bytes",
                    "sha256",
                    "thread_mode",
                    "threads",
                    "thread_environment",
                    "root_commit",
                )
            ),
            f"CPU aggregate Torch baseline {mode} publication differs",
        )


def _require_frozen_baseline_host(
    environment: dict[str, Any],
    baseline_environment: dict[str, Any],
    baseline_thread_environment: dict[str, str | None],
    label: str,
) -> None:
    try:
        from benchmarks.torch_cpu_baseline import privacy_preserving_host_identity

        baseline_identity = privacy_preserving_host_identity(baseline_environment)
        candidate = privacy_preserving_host_identity(
            environment, salt=baseline_identity["salt"]
        )
    except (ImportError, KeyError, TypeError, ValueError) as error:
        raise EvidenceError(
            f"{label} baseline privacy identity is invalid: {error}"
        ) from error
    _require(
        _type_exact_equal(candidate, baseline_identity)
        and _type_exact_equal(
            environment.get("thread_environment"),
            baseline_thread_environment,
        ),
        f"{label} differs from the exact frozen Torch baseline host or thread environment",
    )


def _require_raw_match(actual: list[float], expected: list[float], label: str) -> None:
    _require(len(actual) == len(expected), f"{label} sample count differs")
    for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
        _close(left, right, f"{label}[{index}]")


def _validate_cpu_gate(
    gate: Any,
    label: str,
    *,
    max_ratio: float,
    max_relative_mad: float,
    expected_reference: list[float],
    expected_candidate: list[float],
    expected_source_sha256: str,
    expected_root_commit: str,
) -> float:
    _exact_keys(
        gate,
        {
            "comparison_valid",
            "contract_errors",
            "reference_source_artifact_sha256",
            "reference_root_commit",
            "reference_raw_seconds_per_step",
            "candidate_raw_seconds_per_step",
            "reference_seconds_per_step",
            "candidate_seconds_per_step",
            "candidate_to_torch_baseline_ratio",
            "individual_ratio_limit",
            "within_five_percent",
        },
        label,
    )
    reference = _raw_values(
        gate["reference_raw_seconds_per_step"],
        f"{label} reference samples",
        count=len(expected_reference),
    )
    candidate = _raw_values(
        gate["candidate_raw_seconds_per_step"],
        f"{label} candidate samples",
        count=len(expected_candidate),
    )
    _require_raw_match(reference, expected_reference, f"{label} reference binding")
    _require_raw_match(candidate, expected_candidate, f"{label} candidate binding")
    _require(
        _relative_mad(reference) <= max_relative_mad
        and _relative_mad(candidate) <= max_relative_mad,
        f"{label} relative MAD exceeds the manifest limit",
    )
    ratio = median(candidate) / median(reference)
    _close(gate["reference_seconds_per_step"], median(reference), f"{label} reference")
    _close(gate["candidate_seconds_per_step"], median(candidate), f"{label} candidate")
    _close(gate["candidate_to_torch_baseline_ratio"], ratio, f"{label} ratio")
    _close(gate["individual_ratio_limit"], max_ratio, f"{label} ratio limit")
    _require(
        gate["reference_source_artifact_sha256"] == expected_source_sha256
        and gate["reference_root_commit"] == expected_root_commit
        and gate["contract_errors"] == []
        and gate["comparison_valid"] is True,
        f"{label} baseline provenance or comparison is invalid",
    )
    _require(ratio <= max_ratio, f"{label} exceeds the individual ratio threshold")
    _require(
        gate["within_five_percent"] is True,
        f"{label} embedded individual pass is false",
    )
    return ratio


def _validate_native_gate(
    gate: Any,
    label: str,
    *,
    expected_reference: list[float],
    expected_candidate: list[float],
    summary_sha256: str,
    manifest: dict[str, Any],
) -> None:
    _exact_keys(
        gate,
        {
            "comparison_role",
            "reference_observer_tag",
            "reference_observer_commit",
            "reference_precision",
            "reference_contract",
            "reference_contract_provenance",
            "reference_sha256",
            "reference_seconds_per_step",
            "candidate_seconds_per_step",
            "reference_raw_seconds_per_step",
            "candidate_raw_seconds_per_step",
            "torch_to_native_ratio",
            "comparison_valid",
            "contract_errors",
        },
        label,
    )
    reference = _raw_values(
        gate["reference_raw_seconds_per_step"],
        f"{label} reference samples",
        count=len(expected_reference),
    )
    candidate = _raw_values(
        gate["candidate_raw_seconds_per_step"],
        f"{label} candidate samples",
        count=len(expected_candidate),
    )
    _require_raw_match(reference, expected_reference, f"{label} reference binding")
    _require_raw_match(candidate, expected_candidate, f"{label} candidate binding")
    ratio = median(candidate) / median(reference)
    _close(gate["reference_seconds_per_step"], median(reference), f"{label} reference")
    _close(gate["candidate_seconds_per_step"], median(candidate), f"{label} candidate")
    _close(gate["torch_to_native_ratio"], ratio, f"{label} ratio")
    reference_contract = _native_benchmark_contract(manifest)
    frozen = manifest["reference"]
    _require(
        gate["comparison_role"] == "informational"
        and gate["reference_observer_tag"] == frozen["performance_observer_tag"]
        and gate["reference_observer_commit"] == frozen["performance_observer_commit"]
        and gate["reference_precision"]
        == manifest["performance_gates"]["cpu_acceptance"]["precision"]
        and gate["reference_contract"] == reference_contract
        and gate["reference_contract_provenance"]
        == "exact SHA-256-pinned embedded contract"
        and gate["reference_sha256"] == summary_sha256
        and gate["comparison_valid"] is True
        and gate["contract_errors"] == [],
        f"{label} informational native provenance or comparison is invalid",
    )


def _validate_cpu_scope(
    scope: Any,
    reader: ArtifactReader,
    manifest: dict[str, Any],
    candidate: dict[str, str],
    manifest_path: Path,
) -> dict[str, Any]:
    _exact_keys(
        scope,
        {
            "aggregate",
            "native_summary",
            "native_raw",
            "torch_baseline_artifacts",
            "correctness_index",
            "slices",
            "traces",
            "allocation_sidecars",
            "generated_sources",
        },
        "CPU scope",
    )
    aggregate_artifact = reader.load(scope["aggregate"], "CPU aggregate")
    aggregate = aggregate_artifact.document
    _require(
        isinstance(aggregate, dict)
        and _is_exact_int(aggregate.get("schema_version"), 4)
        and aggregate.get("kind") == "cpu-acceptance-aggregate",
        "CPU aggregate schema differs",
    )
    _document_candidate_matches(aggregate, candidate, required=True)
    full_evidence = aggregate.get("evidence")
    _require(isinstance(full_evidence, dict), "CPU aggregate evidence is absent")
    _require(
        full_evidence.get("candidate_git_status") == "",
        "CPU aggregate was captured from a dirty checkout",
    )

    native_raw = reader.load_many(
        scope["native_raw"],
        "native raw matrix",
        count=12,
        json_document=False,
        expected_media_types={MEDIA_TYPE_JSON},
    )
    native_summary = _validate_native_summary(
        reader.load(scope["native_summary"], "pinned native summary"),
        manifest,
        native_raw,
        manifest_path,
    )
    (
        baseline_raw_by_mode,
        baseline_source_sha_by_mode,
        baseline_root_commit,
        baseline_environment,
        baseline_thread_environment_by_mode,
        baseline_sources_by_mode,
    ) = _load_pinned_torch_baseline(
        manifest,
        reader,
        scope["torch_baseline_artifacts"],
    )
    _require_cpu_baseline_publication_provenance(
        aggregate,
        manifest,
        baseline_sources_by_mode,
    )

    correctness_artifact = reader.load(
        scope["correctness_index"],
        "CPU correctness index",
    )
    correctness = _validate_correctness_index(
        correctness_artifact,
        manifest,
        candidate,
        reader,
    )
    embedded_correctness = aggregate.get("correctness_evidence")
    _require(
        isinstance(embedded_correctness, dict),
        "CPU aggregate does not bind correctness evidence",
    )
    source = embedded_correctness.get("source_artifact")
    _require(
        isinstance(source, dict)
        and source.get("sha256") == correctness_artifact.descriptor["sha256"],
        "CPU aggregate correctness source digest differs",
    )
    without_source = dict(embedded_correctness)
    without_source.pop("source_artifact", None)
    rebuilt_without_source = dict(correctness)
    rebuilt_without_source.pop("source_artifact", None)
    _require(
        without_source == rebuilt_without_source,
        "CPU aggregate embeds different correctness evidence",
    )

    slice_artifacts = reader.load_many(scope["slices"], "CPU slices", count=2)
    source_records = aggregate.get("candidate_slice_artifacts")
    _require(
        isinstance(source_records, list) and len(source_records) == 2,
        "CPU aggregate source slice bindings are incomplete",
    )
    for index, (artifact, source_record) in enumerate(
        zip(slice_artifacts, source_records, strict=True)
    ):
        _require(
            isinstance(source_record, dict)
            and source_record.get("sha256") == artifact.descriptor["sha256"]
            and Path(source_record.get("path", "")).name == artifact.path.name,
            f"CPU aggregate source slice {index} differs",
        )
    traces = _load_traces(reader, scope["traces"], "CPU traces", count=12)
    combined_slices = {"slices": [artifact.document for artifact in slice_artifacts]}
    _bind_tuning_traces(combined_slices, traces, "CPU")
    allocation_document, generated_sources = _load_cpu_allocation_evidence(
        scope,
        reader,
        aggregate,
        all_zero=_all_traces_have_zero_allocations(traces),
    )

    acceptance = manifest["performance_gates"]["cpu_acceptance"]
    _require(
        tuple(acceptance["cases"]) == CPU_CASES, "manifest CPU case contract differs"
    )
    _require(
        tuple(acceptance["thread_modes"]) == ("one", "physical"),
        "manifest CPU thread modes differ",
    )
    max_ratio = _finite_float(
        acceptance["max_individual_ratio"],
        "CPU individual threshold",
        positive=True,
    )
    max_relative_mad = _finite_float(
        acceptance["statistics"]["max_relative_mad"],
        "CPU relative MAD limit",
        positive=True,
    )
    modes = []
    all_gates: list[dict[str, Any]] = []
    environments = []
    torch_raw_by_mode: dict[str, dict[str, list[float]]] = {}
    native_raw_by_mode: dict[str, dict[str, list[float]]] = {}
    raw_acceptance_by_mode: dict[str, dict[str, Any]] = {}
    aggregate_slices = aggregate.get("cpu_slices")
    _require(
        isinstance(aggregate_slices, list) and len(aggregate_slices) == 2,
        "CPU aggregate thread slices are incomplete",
    )
    for slice_index, (artifact, summary) in enumerate(
        zip(slice_artifacts, aggregate_slices, strict=True)
    ):
        document = artifact.document
        _require(
            isinstance(document, dict)
            and _is_exact_int(document.get("schema_version"), 4)
            and document.get("kind") == "cpu-acceptance-thread-slice",
            f"CPU slice {slice_index} schema differs",
        )
        _require(
            document.get("evidence") == full_evidence,
            f"CPU slice {slice_index} candidate evidence differs",
        )
        cases = document.get("cases")
        _require(
            isinstance(cases, list)
            and [case.get("workload", {}).get("name") for case in cases]
            == list(CPU_CASES),
            f"CPU slice {slice_index} case closure differs",
        )
        threads = {case.get("runtime", {}).get("threads") for case in cases}
        _require(
            len(threads) == 1
            and all(type(value) is int and value > 0 for value in threads),
            f"CPU slice {slice_index} thread count differs",
        )
        thread_count = next(iter(threads))
        environment = document.get("environment")
        _require(
            isinstance(environment, dict),
            f"CPU slice {slice_index} environment is absent",
        )
        _require_successful_command_statuses(
            environment,
            f"CPU slice {slice_index}",
            require_cuda_topology=False,
        )
        physical = environment.get("cpu_count_physical_affinity")
        _require(
            type(physical) is int and physical > 1,
            f"CPU slice {slice_index} physical core count is malformed",
        )
        mode = (
            "one"
            if thread_count == 1
            else "physical" if thread_count == physical else None
        )
        _require(mode is not None, f"CPU slice {slice_index} thread mode is invalid")
        _require_frozen_baseline_host(
            environment,
            baseline_environment,
            baseline_thread_environment_by_mode[mode],
            f"CPU {mode} slice",
        )
        modes.append(mode)
        torch_raw_by_mode[mode] = {}
        native_raw_by_mode[mode] = {}
        environments.append(
            {
                key: environment.get(key)
                for key in (
                    "host_contract",
                    "hostname",
                    "platform",
                    "python",
                    "torch",
                    "cpu_affinity",
                    "cpu_count_physical_affinity",
                    "cpu_topology",
                )
            }
        )
        for case_index, case in enumerate(cases):
            label = f"CPU {mode} {CPU_CASES[case_index]}"
            runtime = case.get("runtime", {})
            _require(
                runtime.get("device") == "cpu"
                and runtime.get("precision") == acceptance["precision"]
                and _is_exact_int(runtime.get("interop_threads"), 1)
                and runtime.get("compile_policy") == "compile"
                and runtime.get("compile_mode") == "default"
                and runtime.get("execution_policy") == "auto",
                f"{label} runtime contract differs",
            )
            raw, _seconds = _raw_summary(
                case.get("measurements", {}).get("advance"),
                f"{label} timing",
                steps=manifest["reference"]["performance_steps_per_repeat"],
                repeats=manifest["reference"]["performance_repetitions"],
            )
            torch_raw_by_mode[mode][CPU_CASES[case_index]] = [
                value / manifest["reference"]["performance_steps_per_repeat"]
                for value in raw
            ]
            _validate_tuning_acceptance(case, label, allow_allocation_override=True)
        _validate_cpu_recomputed_acceptance(summary, f"CPU {mode}", manifest)
        raw_acceptance_by_mode[mode] = _recompute_cpu_case_acceptance(
            cases,
            summary,
            traces,
            allocation_document,
            generated_sources,
            manifest,
            full_evidence,
            f"CPU {mode}",
        )
        gates = summary.get("torch_baseline_comparisons")
        native_gates = summary.get("native_comparisons")
        _require(
            isinstance(gates, list) and len(gates) == len(CPU_CASES),
            f"CPU {mode} baseline gate count differs",
        )
        _require(
            isinstance(native_gates, list) and len(native_gates) == len(CPU_CASES),
            f"CPU {mode} native gate count differs",
        )
        for case_name, gate, native_gate in zip(
            CPU_CASES,
            gates,
            native_gates,
            strict=True,
        ):
            _validate_cpu_gate(
                gate,
                f"CPU {mode} {case_name} baseline",
                max_ratio=max_ratio,
                max_relative_mad=max_relative_mad,
                expected_reference=baseline_raw_by_mode[mode][case_name],
                expected_candidate=torch_raw_by_mode[mode][case_name],
                expected_source_sha256=baseline_source_sha_by_mode[mode],
                expected_root_commit=baseline_root_commit,
            )
            _validate_native_gate(
                native_gate,
                f"CPU {mode} {case_name} native",
                expected_reference=native_summary["raw_seconds_per_step"][mode][
                    case_name
                ],
                expected_candidate=torch_raw_by_mode[mode][case_name],
                summary_sha256=native_summary["summary_sha256"],
                manifest=manifest,
            )
            native_raw_by_mode[mode][case_name] = list(
                native_summary["raw_seconds_per_step"][mode][case_name]
            )
            all_gates.append(gate)
        _require(
            summary.get("errors") == [] and summary.get("passed") is True,
            f"CPU {mode} aggregate slice failed",
        )
    _require(sorted(modes) == ["one", "physical"], "CPU thread mode closure differs")
    _require(environments[0] == environments[1], "CPU slices came from different hosts")

    cpu_environment = aggregate.get("environment")
    _require(
        isinstance(cpu_environment, dict)
        and cpu_environment.get("host_contract") == environments[0]["host_contract"],
        "CPU aggregate and source slices use different host contracts",
    )
    host_contract = _validate_host_contract(
        cpu_environment["host_contract"],
        "CPU host contract",
        environment=cpu_environment,
    )
    common_host = host_contract["common_identity"]
    cpu_runtime = host_contract["runtime_identity"]
    native_environment = native_summary["environment"]
    native_torch = native_environment.get("torch")
    _require(
        common_host["hostname"] == native_environment["hostname"]
        and common_host["platform"] == native_environment["platform"]
        and common_host["python"] == native_environment["python"]
        and common_host["cxx_version"] == native_environment["cxx_version"]
        and common_host["swig_version"] == native_environment["swig_version"]
        and common_host["uv_lock_sha256"] == native_environment["uv_lock_sha256"]
        and common_host["os"]["system"] == native_environment["os"]["system"]
        and common_host["os"]["release"] == native_environment["os"]["release"]
        and common_host["os"]["machine"] == native_environment["os"]["machine"]
        and isinstance(native_torch, dict)
        and native_torch.get("version") == cpu_runtime["torch"],
        "candidate CPU and pinned native evidence use different hosts or toolchains",
    )

    recomputed_bootstrap = _bootstrap_geomean(
        all_gates,
        acceptance["statistics"],
    )
    _require(recomputed_bootstrap["passed"], "CPU bootstrap gate failed")
    suite = aggregate.get("suite_acceptance")
    _require(isinstance(suite, dict), "CPU suite acceptance is absent")
    recorded_bootstrap = suite.get("torch_baseline_geomean_statistics")
    _require(isinstance(recorded_bootstrap, dict), "CPU recorded bootstrap is absent")
    for key, value in recomputed_bootstrap.items():
        if isinstance(value, float):
            _close(recorded_bootstrap.get(key), value, f"CPU bootstrap {key}")
        else:
            _require(
                recorded_bootstrap.get(key) == value, f"CPU bootstrap {key} differs"
            )
    _require(
        aggregate.get("acceptance_scope") == "cpu-performance-and-correctness"
        and aggregate.get("cpu_correctness_satisfied") is True
        and aggregate.get("issue_completion_satisfied") is False
        and suite.get("correctness_evidence_bound") is True
        and suite.get("cpu_correctness_satisfied") is True
        and suite.get("errors") == []
        and suite.get("passed") is True,
        "CPU aggregate acceptance is not exact",
    )
    return {
        "candidate_evidence": candidate,
        "environment": aggregate.get("environment"),
        "host_contract": host_contract,
        "common_host_identity": _common_host_identity(host_contract),
        "runtime_identity": copy.deepcopy(cpu_runtime),
        "native_summary_sha256": native_summary["summary_sha256"],
        "cell_count": len(all_gates),
        "bootstrap": recomputed_bootstrap,
        "torch_raw_seconds_per_step": torch_raw_by_mode,
        "native_raw_seconds_per_step": native_raw_by_mode,
        "raw_recomputed_acceptance": raw_acceptance_by_mode,
    }


def _npz_arrays(
    artifact: LoadedArtifact,
    names: list[str],
    label: str,
) -> dict[str, np.ndarray]:
    expected_files = {f"{name}.npy" for name in names}
    _preflight_zip(
        artifact.raw,
        label,
        max_members=MAX_NPZ_MEMBERS,
        max_member_bytes=MAX_NPZ_ARRAY_BYTES,
        max_total_bytes=MAX_NPZ_TOTAL_BYTES,
        expected_files=expected_files,
    )
    try:
        with np.load(io.BytesIO(artifact.raw), allow_pickle=False) as archive:
            _require(
                archive.files == names,
                f"{label} NPZ array closure differs",
            )
            result: dict[str, np.ndarray] = {}
            total = 0
            for name in names:
                array = np.asarray(archive[name])
                _require(
                    array.dtype.fields is None
                    and array.dtype.subdtype is None
                    and array.dtype.kind in {"b", "i", "u", "f", "c"},
                    f"{label} array {name!r} dtype is not a plain numeric type",
                )
                _require(
                    array.ndim <= MAX_NPZ_DIMENSIONS
                    and all(
                        type(size) is int and 0 <= size <= 2**31 - 1
                        for size in array.shape
                    ),
                    f"{label} array {name!r} shape exceeds the bound",
                )
                _require(
                    array.flags.c_contiguous and array.nbytes <= MAX_NPZ_ARRAY_BYTES,
                    f"{label} array {name!r} storage exceeds the bound",
                )
                total += array.nbytes
                _require(
                    total <= MAX_NPZ_TOTAL_BYTES,
                    f"{label} arrays exceed the byte bound",
                )
                result[name] = array
            return result
    except (MemoryError, OSError, TypeError, ValueError) as error:
        if isinstance(error, EvidenceError):
            raise
        raise EvidenceError(f"{label} is not a readable safe NPZ archive") from error


def _validate_differential(
    artifact: LoadedArtifact,
    reader: ArtifactReader,
    manifest: dict[str, Any],
    candidate: dict[str, str],
    *,
    scope: str,
) -> None:
    try:
        from benchmarks.issue123_differential import (
            validate_differential_document,
        )

        def load_projection(descriptor, label):
            loaded = reader.load(
                descriptor,
                label,
                json_document=False,
                expected_media_types={MEDIA_TYPE_NPZ},
            )
            return loaded.path, loaded.raw

        validate_differential_document(
            artifact.document,
            manifest,
            candidate,
            descriptor_root=reader.base,
            expected_scope=scope,
            artifact_loader=load_projection,
        )
    except (ImportError, OSError, TypeError, ValueError) as error:
        if isinstance(error, EvidenceError):
            raise
        raise EvidenceError(
            f"{scope} differential could not be independently recomputed: {error}"
        ) from error
    document = artifact.document
    _exact_keys(
        document,
        {
            "schema_version",
            "kind",
            "scope",
            "candidate_evidence",
            "required_cases",
            "cases",
            "passed",
        },
        f"{scope} differential index",
    )
    _require(
        _is_exact_int(document["schema_version"], 1)
        and document["kind"] == DIFFERENTIAL_KIND
        and document["scope"] == scope
        and document["candidate_evidence"] == candidate,
        f"{scope} differential contract differs",
    )
    if scope == "paired-real":
        names = [
            case["name"]
            for case in manifest.get("correctness", ())
            if case.get("complex") is True
        ]
        expected = [
            {"case": name, "device": device}
            for name in names
            for device in ("cpu", "cuda:0")
        ]
    elif scope == "single-gpu-cuda":
        expected = [{"case": name, "device": "cuda:0"} for name in SINGLE_GPU_CASES]
    else:
        raise EvidenceError(f"unknown differential scope {scope!r}")
    _require(document["required_cases"] == expected, f"{scope} required cases differ")
    cases = document["cases"]
    _require(
        isinstance(cases, list)
        and [{"case": case.get("case"), "device": case.get("device")} for case in cases]
        == expected,
        f"{scope} evaluated case closure differs",
    )
    for index, (record, expected_record) in enumerate(
        zip(cases, expected, strict=True)
    ):
        label = f"{scope} case {index}"
        _exact_keys(
            record,
            {
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
            },
            label,
        )
        _require(
            record["case"] == expected_record["case"]
            and record["device"] == expected_record["device"],
            f"{label} identity differs",
        )
        field_arrays = record["field_arrays"]
        persistent_arrays = record["persistent_arrays"]
        _require(
            field_arrays == list(FIELD_ARRAYS),
            f"{label} complete field array list differs",
        )
        _require(
            isinstance(persistent_arrays, list)
            and bool(persistent_arrays)
            and all(isinstance(name, str) and name for name in persistent_arrays)
            and len(set(persistent_arrays)) == len(persistent_arrays),
            f"{label} persistent array closure is absent",
        )
        names = field_arrays + persistent_arrays
        _require(len(set(names)) == len(names), f"{label} repeats array names")
        reference = reader.load(
            record["reference"],
            f"{label} reference",
            json_document=False,
            expected_media_types={MEDIA_TYPE_NPZ},
        )
        actual = reader.load(
            record["candidate"],
            f"{label} candidate",
            json_document=False,
            expected_media_types={MEDIA_TYPE_NPZ},
        )
        reference_arrays = _npz_arrays(reference, names, f"{label} reference")
        actual_arrays = _npz_arrays(actual, names, f"{label} candidate")
        rtol = _finite_float(record["rtol"], f"{label} rtol")
        atol = _finite_float(record["atol"], f"{label} atol")
        _require(
            rtol >= 0.0 and atol >= 0.0, f"{label} tolerances must be non-negative"
        )
        maximum_abs = 0.0
        maximum_relative = 0.0
        all_close = True
        for name in names:
            left = reference_arrays[name]
            right = actual_arrays[name]
            _require(
                left.shape == right.shape and left.dtype == right.dtype,
                f"{label} array {name} shape or dtype differs",
            )
            _require(
                np.isfinite(left).all() and np.isfinite(right).all(),
                f"{label} array {name} contains non-finite values",
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
            all_close = all_close and bool(
                np.allclose(right, left, rtol=rtol, atol=atol)
            )
        _close(record["maximum_abs_error"], maximum_abs, f"{label} maximum abs error")
        _close(
            record["maximum_relative_error"],
            maximum_relative,
            f"{label} maximum relative error",
        )
        _require(all_close and record["passed"] is True, f"{label} differential failed")
    _require(document["passed"] is True, f"{scope} embedded suite pass is false")


def _tuning_cases(document: Any, label: str) -> list[dict[str, Any]]:
    _require(
        isinstance(document, dict)
        and _is_exact_int(document.get("schema_version"), 4)
        and document.get("kind") == "torch-tuning-diagnostic",
        f"{label} tuning schema differs",
    )
    cases = document.get("cases")
    _require(isinstance(cases, list), f"{label} cases must be a list")
    diagnostic = document.get("diagnostic_acceptance")
    _require(
        isinstance(diagnostic, dict) and diagnostic.get("passed") is True,
        f"{label} diagnostic acceptance failed",
    )
    return cases


def _tuple_cache_preimage(value: Any, label: str) -> Any:
    if isinstance(value, list):
        return tuple(
            _tuple_cache_preimage(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    _require(
        value is None or type(value) in {bool, int, float, str},
        f"{label} contains a non-canonical value",
    )
    if isinstance(value, float):
        _require(math.isfinite(value), f"{label} contains a non-finite value")
    return value


def _policy_config_preimage(
    result: dict[str, Any],
    case_name: str,
    policy: str,
    executions: list[dict[str, Any]],
) -> dict[str, Any]:
    runtime = result["runtime"]
    return {
        "workload": case_name,
        "device": runtime["device"],
        "precision": runtime["precision"],
        "compile_policy": runtime["compile_policy"],
        "compile_mode": runtime["compile_mode"],
        "explicit_cuda_graphs": runtime["explicit_cuda_graphs"],
        "requested_policy": policy,
        "resolved_policies": sorted({item["policy"] for item in executions}),
        "execution_representations": sorted(
            {item["execution_representation"] for item in executions}
        ),
        "execution_topology": sorted(
            [item["component"], item["model"], item["targets"]] for item in executions
        ),
    }


def _validate_compile_cache_key_evidence(
    result: dict[str, Any],
    label: str,
    *,
    policy: str,
    case_name: str,
    executions: list[dict[str, Any]],
    expected_representation: str,
) -> str:
    evidence = result.get("compile_cache_key_evidence")
    _exact_keys(
        evidence,
        {
            "schema_version",
            "algorithm",
            "runtime_preimage",
            "policy_config",
            "policy_config_sha256",
        },
        f"{label} compile cache evidence",
    )
    preimage = evidence["runtime_preimage"]
    _require(isinstance(preimage, list), f"{label} cache preimage must be a list")
    payload = _tuple_cache_preimage(preimage, f"{label} cache preimage")
    _require(
        isinstance(payload, tuple) and len(payload) == 31,
        f"{label} cache preimage shape differs",
    )
    runtime = result["runtime"]
    diagnostics = result.get("diagnostics")
    cache_key = runtime.get("compile_cache_key")
    recomputed = _sha256(repr(payload).encode())
    _require(
        _is_exact_int(evidence["schema_version"], 1)
        and evidence["algorithm"] == COMPILE_CACHE_PREIMAGE_ALGORITHM
        and isinstance(cache_key, str)
        and SHA256_RE.fullmatch(cache_key) is not None
        and recomputed == cache_key,
        f"{label} compile cache key does not match its preimage",
    )
    _require(
        isinstance(diagnostics, dict)
        and payload[0] == TORCH_SOLVER_ABI
        and diagnostics.get("compile_solver_abi") == TORCH_SOLVER_ABI
        and diagnostics.get("compiled_region_topology")
        == LOCAL_COMPILED_REGION_TOPOLOGY
        and payload[3] == runtime["field_storage_dtype"]
        and payload[4] == runtime["compile_policy"]
        and payload[5] == runtime["compile_mode"]
        and payload[6] == LOCAL_COMPILED_REGION_TOPOLOGY
        and payload[8] == expected_representation
        and payload[18] is runtime["paired_real"]
        and payload[19] is True
        and payload[20] is None,
        f"{label} cache preimage differs from the runtime policy/configuration",
    )
    expected_config = _policy_config_preimage(result, case_name, policy, executions)
    _require(
        _type_exact_equal(evidence["policy_config"], expected_config)
        and evidence["policy_config_sha256"] == _canonical_sha256(expected_config),
        f"{label} canonical policy/config preimage differs",
    )
    return cache_key


def _validate_policy_execution_diagnostic(
    value: Any,
    label: str,
    *,
    policy: str,
    executions: list[dict[str, Any]],
    reader: ArtifactReader,
    main_traces: dict[str, tuple[LoadedArtifact, dict[str, Any]]],
    used_digests: set[str],
) -> None:
    _exact_keys(
        value,
        {
            "schema_version",
            "kind",
            "contract_id",
            "execution_policy",
            "compile_policy",
            "profile_steps",
            "execution_records_per_step",
            "expected_operation",
            "expected_operation_count",
            "observed_operation_counts",
            "trace",
        },
        f"{label} operation diagnostic",
    )
    expected_operation = POLICY_WRITE_OPERATIONS[policy]
    steps = value["profile_steps"]
    records_per_step = len(executions)
    expected_count = records_per_step * steps if type(steps) is int else -1
    _require(
        _is_exact_int(value["schema_version"], 1)
        and value["kind"] == POLICY_DIAGNOSTIC_KIND
        and value["contract_id"] == POLICY_DIAGNOSTIC_CONTRACT
        and value["execution_policy"] == policy
        and value["compile_policy"] == "eager"
        and type(steps) is int
        and steps > 0
        and _is_exact_int(value["execution_records_per_step"], records_per_step)
        and value["expected_operation"] == expected_operation
        and _is_exact_int(value["expected_operation_count"], expected_count),
        f"{label} operation diagnostic contract differs",
    )
    artifact = reader.load(
        value["trace"],
        f"{label} uncompiled policy trace",
        json_document=False,
        expected_media_types={MEDIA_TYPE_JSON},
    )
    digest = artifact.descriptor["sha256"]
    _require(
        digest not in main_traces and digest not in used_digests,
        f"{label} reuses a policy profiler trace",
    )
    used_digests.add(digest)
    summary = _trace_summary(artifact.raw, f"{label} uncompiled policy trace")
    expected_counts = {
        operation: expected_count if operation == expected_operation else 0
        for operation in POLICY_WRITE_OPERATIONS.values()
    }
    _require(
        summary["compiled_region_events"] == 0
        and summary["cuda_graph_launches"] == 0
        and summary["policy_write_operations"] == expected_counts
        and _type_exact_equal(value["observed_operation_counts"], expected_counts),
        f"{label} raw trace does not exclusively execute the forced aten write op",
    )


def _validate_policy_run(
    result: Any,
    label: str,
    *,
    policy: str,
    case_name: str,
    traces: dict[str, tuple[LoadedArtifact, dict[str, Any]]],
    manifest: dict[str, Any],
    reader: ArtifactReader,
    diagnostic_trace_digests: set[str],
) -> tuple[
    float,
    tuple[tuple[str, str, int], ...],
    tuple[str, ...],
    tuple[str, ...],
    str,
]:
    _require(isinstance(result, dict), f"{label} must be an object")
    _require(
        _type_exact_equal(result.get("workload"), _manifest_case(manifest, case_name)),
        f"{label} workload differs from the manifest",
    )
    timing = _validate_cuda_tuning_case(
        result,
        traces,
        manifest,
        name=case_name,
        paired_real=False,
        execution_policy=policy,
    )
    seconds = timing["seconds_per_step"]
    runtime = result["runtime"]
    dispersive = result.get("diagnostics", {}).get("dispersive", {})
    executions = dispersive.get("policy_executions")
    _require(
        isinstance(executions, list) and bool(executions),
        f"{label} has no policy execution records",
    )
    topology = []
    representations = set()
    resolved_policies = set()
    for execution in executions:
        _require(isinstance(execution, dict), f"{label} execution must be an object")
        component = execution.get("component")
        model = execution.get("model")
        targets = execution.get("targets")
        resolved = execution.get("policy")
        representation = execution.get("execution_representation")
        _require(
            isinstance(component, str)
            and bool(component)
            and isinstance(model, str)
            and bool(model)
            and type(targets) is int
            and targets > 0
            and resolved in FORCED_REPRESENTATIONS
            and representation == FORCED_REPRESENTATIONS[resolved],
            f"{label} policy execution record differs",
        )
        topology.append((component, model, targets))
        representations.add(representation)
        resolved_policies.add(resolved)
    if policy != "auto":
        _require(resolved_policies == {policy}, f"{label} resolved a different policy")
        _require(
            representations == {FORCED_REPRESENTATIONS[policy]},
            f"{label} representation differs",
        )
        expected = f"policy-dispatched-bucket-io-v2[{FORCED_REPRESENTATIONS[policy]}]"
    else:
        expected = (
            "policy-dispatched-bucket-io-v2[" + ",".join(sorted(representations)) + "]"
        )
    _require(
        dispersive.get("execution_representation") == expected,
        f"{label} top-level execution representation differs",
    )
    cache_key = _validate_compile_cache_key_evidence(
        result,
        label,
        policy=policy,
        case_name=case_name,
        executions=executions,
        expected_representation=expected,
    )
    if policy == "auto":
        _require(
            "policy_execution_diagnostic" not in result,
            f"{label} unexpectedly contains a forced-policy diagnostic",
        )
    else:
        _validate_policy_execution_diagnostic(
            result.get("policy_execution_diagnostic"),
            label,
            policy=policy,
            executions=executions,
            reader=reader,
            main_traces=traces,
            used_digests=diagnostic_trace_digests,
        )
    return (
        seconds,
        tuple(sorted(topology)),
        tuple(sorted(representations)),
        tuple(sorted(resolved_policies)),
        cache_key,
    )


def _manifest_case(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [
        case
        for group in ("benchmarks", "correctness", "physical_checks")
        for case in manifest.get(group, ())
        if case.get("name") == name
    ]
    _require(len(matches) == 1, f"manifest case {name!r} is not unique")
    return matches[0]


def _bound_trace_summary(
    result: dict[str, Any],
    traces: dict[str, tuple[LoadedArtifact, dict[str, Any]]],
    label: str,
) -> dict[str, Any]:
    profiler = result.get("profiler")
    _require(isinstance(profiler, dict), f"{label} profiler is absent")
    digest = profiler.get("chrome_trace_sha256")
    _require(digest in traces, f"{label} trace is not exact-byte bound")
    return traces[digest][1]


def _validate_cuda_memory(result: dict[str, Any], label: str) -> None:
    memory = result.get("memory")
    _require(isinstance(memory, dict), f"{label} CUDA memory is absent")
    before = memory.get("cuda_allocated_before_bytes")
    after = memory.get("cuda_allocated_after_bytes")
    growth = memory.get("cuda_allocated_growth_bytes")
    peak = memory.get("cuda_peak_allocated_bytes")
    reserved = memory.get("cuda_peak_reserved_bytes")
    _require(
        memory.get("bounded") is True
        and type(before) is int
        and before >= 0
        and type(after) is int
        and after >= 0
        and type(growth) is int
        and growth == after - before
        and growth <= 1024 * 1024
        and type(peak) is int
        and peak > 0
        and type(reserved) is int
        and reserved >= peak,
        f"{label} CUDA memory gate failed",
    )


def _validate_cuda_tuning_case(
    result: Any,
    traces: dict[str, tuple[LoadedArtifact, dict[str, Any]]],
    manifest: dict[str, Any],
    *,
    name: str,
    paired_real: bool,
    execution_policy: str = "auto",
) -> dict[str, Any]:
    label = f"CUDA tuning {name}"
    _require(isinstance(result, dict), f"{label} must be an object")
    workload = result.get("workload")
    _require(
        isinstance(workload, dict) and workload.get("name") == name,
        f"{label} workload is absent or renamed",
    )
    if name not in REGION_INVARIANCE_CASES:
        _require(
            _type_exact_equal(workload, _manifest_case(manifest, name))
            and workload.get("complex") is paired_real,
            f"{label} manifest workload differs",
        )
    runtime = result.get("runtime")
    expected_representation = "paired-real-v1" if paired_real else "real-v1"
    expected_channels = 2 if paired_real else 1
    _require(
        isinstance(runtime, dict)
        and runtime.get("device") == "cuda:0"
        and runtime.get("precision") == "float32"
        and runtime.get("compile_policy") == "compile"
        and runtime.get("compile_mode") == "default"
        and runtime.get("explicit_cuda_graphs") is False
        and runtime.get("execution_policy") == execution_policy
        and runtime.get("experimental_dispersive_grouping") is False
        and _is_exact_int(runtime.get("threads"), 1)
        and _is_exact_int(runtime.get("interop_threads"), 1)
        and runtime.get("paired_real") is paired_real
        and runtime.get("field_storage_representation") == expected_representation
        and _is_exact_int(runtime.get("field_storage_channels"), expected_channels)
        and runtime.get("field_storage_dtype") == "torch.float32",
        f"{label} runtime/storage contract differs",
    )
    contract = result.get("benchmark_contract")
    reference = manifest["reference"]
    _require(
        isinstance(contract, dict)
        and contract.get("warmup_steps") == reference["performance_warmup_steps"]
        and contract.get("steps_per_repeat")
        == reference["performance_steps_per_repeat"]
        and contract.get("repetitions") == reference["performance_repetitions"]
        and contract.get("profile_steps") == reference["performance_profile_steps"]
        and contract.get("sample_start") == "independently-restored-pre-warmup-state",
        f"{label} benchmark contract differs",
    )
    _validate_tuning_acceptance(result, label)
    summary = result.get("measurements", {}).get("advance")
    raw, seconds = _raw_summary(
        summary,
        f"{label} timing",
        steps=reference["performance_steps_per_repeat"],
        repeats=reference["performance_repetitions"],
    )
    relative_mad = _relative_mad(raw)
    _close(summary.get("relative_mad"), relative_mad, f"{label} relative MAD")
    _require(
        relative_mad
        <= manifest["performance_gates"]["cpu_acceptance"]["statistics"][
            "max_relative_mad"
        ],
        f"{label} relative MAD exceeds the frozen limit",
    )
    profiler = result.get("profiler")
    _require(
        isinstance(profiler, dict)
        and profiler.get("profile_steps") == reference["performance_profile_steps"],
        f"{label} profiler step count differs",
    )
    trace = _bound_trace_summary(result, traces, label)
    _require(
        trace["kernel_launches"] > 0
        and trace["host_to_device_events"] == 0
        and trace["device_to_host_events"] == 0,
        f"{label} raw trace launch/transfer gate failed",
    )
    _validate_cuda_memory(result, label)
    return {
        "raw_seconds_per_step": [
            value / reference["performance_steps_per_repeat"] for value in raw
        ],
        "seconds_per_step": seconds,
        "relative_mad": relative_mad,
        "trace": trace,
    }


def _validate_paired_real_tuning(
    artifact: LoadedArtifact,
    traces: dict[str, tuple[LoadedArtifact, dict[str, Any]]],
    manifest: dict[str, Any],
    candidate: dict[str, str],
) -> dict[str, Any]:
    document = artifact.document
    _document_candidate_matches(document, candidate, required=True)
    cases = _tuning_cases(document, "paired-real CUDA tuning")
    _require(
        [case.get("workload", {}).get("name") for case in cases]
        == list(PAIRED_REAL_CASES),
        "paired-real CUDA case closure differs",
    )
    _bind_tuning_traces(document, traces, "paired-real CUDA tuning")
    gate = document.get("paired_real_cuda_gate")
    _exact_keys(
        gate,
        {
            "contract_id",
            "required_cases",
            "timing_statistics",
            "trace_contract",
            "errors",
            "passed",
        },
        "paired-real CUDA gate",
    )
    _require(
        gate["contract_id"] == PAIRED_REAL_CONTRACT
        and gate["required_cases"] == list(PAIRED_REAL_CASES)
        and gate["timing_statistics"] == "raw-median-relative-mad-v1"
        and gate["trace_contract"] == "sha256-bound-zero-transfer-kernel-count-v1"
        and gate["errors"] == []
        and gate["passed"] is True,
        "paired-real embedded suite gate differs",
    )
    suite = document.get("suite_acceptance")
    _require(
        isinstance(suite, dict)
        and suite.get("paired_real_suite_expected") is True
        and suite.get("paired_real_suite_complete") is True
        and suite.get("passed") is True,
        "paired-real suite acceptance failed",
    )
    details = {
        name: _validate_cuda_tuning_case(
            result,
            traces,
            manifest,
            name=name,
            paired_real=True,
        )
        for name, result in zip(PAIRED_REAL_CASES, cases, strict=True)
    }
    return {
        "environment": _gpu_environment(document, "paired-real CUDA"),
        "hostname": document.get("environment", {}).get("hostname"),
        "cases": details,
    }


def _numeric_json_tree(value: Any, label: str) -> None:
    for item in _walk(value):
        if isinstance(item, bool):
            raise EvidenceError(f"{label} contains a Boolean numeric value")
        if isinstance(item, float):
            _require(math.isfinite(item), f"{label} contains a non-finite value")


def _validate_effective_material_plan(value: Any, label: str) -> tuple[int, int]:
    _require(
        isinstance(value, list) and len(value) == len(FIELD_ARRAYS),
        f"{label} component closure differs",
    )
    _require(
        [record.get("component") for record in value if isinstance(record, dict)]
        == list(FIELD_ARRAYS),
        f"{label} component order differs",
    )
    active_targets = 0
    material_launches = 0
    for record in value:
        component = record["component"]
        _exact_keys(
            record,
            {
                "component",
                "shape",
                "dense_inverse",
                "constant_targets",
                "constant_values",
                "buckets",
            },
            f"{label} {component}",
        )
        _require(
            isinstance(record["shape"], list)
            and len(record["shape"]) == 3
            and all(type(item) is int and item > 0 for item in record["shape"])
            and isinstance(record["constant_targets"], list)
            and all(
                type(item) is int and 0 <= item < math.prod(record["shape"])
                for item in record["constant_targets"]
            )
            and len(set(record["constant_targets"])) == len(record["constant_targets"])
            and isinstance(record["buckets"], list),
            f"{label} {component} raw arrays are malformed",
        )
        _numeric_json_tree(
            record["dense_inverse"], f"{label} {component} dense inverse"
        )
        _numeric_json_tree(record["constant_values"], f"{label} {component} constants")
        dense_inverse = np.asarray(record["dense_inverse"])
        _require(
            dense_inverse.shape == tuple(record["shape"])
            and dense_inverse.dtype.kind in {"f", "i", "u"}
            and bool(np.isfinite(dense_inverse).all()),
            f"{label} {component} dense inverse shape or values differ",
        )
        _require(
            len(record["constant_targets"]) == len(record["constant_values"]),
            f"{label} {component} constant target/value closure differs",
        )
        material_launches += int(bool(np.any(dense_inverse != 0)))
        component_targets = set(record["constant_targets"])
        for index, bucket in enumerate(record["buckets"]):
            bucket_label = f"{label} {component} bucket {index}"
            _exact_keys(
                bucket,
                {
                    "signature",
                    "coefficient_names",
                    "targets",
                    "target_coefficients",
                    "cell_coefficient_names",
                    "cell_coefficients",
                },
                bucket_label,
            )
            signature = bucket["signature"]
            _exact_keys(
                signature,
                {"component", "model", "precision", "state_shape"},
                f"{bucket_label} signature",
            )
            targets = bucket["targets"]
            _require(
                signature["component"] == component
                and signature["model"] in {"dielectric", "drude"}
                and signature["precision"] == "float32"
                and isinstance(signature["state_shape"], list)
                and all(
                    type(item) is int and item >= 0 for item in signature["state_shape"]
                )
                and isinstance(bucket["coefficient_names"], list)
                and all(
                    isinstance(item, str) and item
                    for item in bucket["coefficient_names"]
                )
                and isinstance(bucket["cell_coefficient_names"], list)
                and all(
                    isinstance(item, str) and item
                    for item in bucket["cell_coefficient_names"]
                )
                and isinstance(targets, list)
                and bool(targets)
                and all(
                    type(item) is int and 0 <= item < math.prod(record["shape"])
                    for item in targets
                )
                and len(set(targets)) == len(targets)
                and component_targets.isdisjoint(targets)
                and isinstance(bucket["target_coefficients"], list)
                and len(bucket["target_coefficients"]) == len(targets),
                f"{bucket_label} signature or target closure differs",
            )
            _numeric_json_tree(
                bucket["target_coefficients"], f"{bucket_label} target coefficients"
            )
            _numeric_json_tree(
                bucket["cell_coefficients"], f"{bucket_label} cell coefficients"
            )
            if signature["model"] == "drude":
                material_launches += 1
            active_targets += len(targets)
            component_targets.update(targets)
    return active_targets, material_launches


def _validate_region_invariance(
    artifact: LoadedArtifact,
    traces: dict[str, tuple[LoadedArtifact, dict[str, Any]]],
    manifest: dict[str, Any],
    candidate: dict[str, str],
) -> dict[str, Any]:
    document = artifact.document
    _document_candidate_matches(document, candidate, required=True)
    cases = _tuning_cases(document, "equivalent-region tuning")
    _require(
        [case.get("workload", {}).get("name") for case in cases]
        == list(REGION_INVARIANCE_CASES),
        "equivalent-region case closure differs",
    )
    _bind_tuning_traces(document, traces, "equivalent-region tuning")
    gate = document.get("region_invariance_gate")
    _exact_keys(
        gate,
        {"contract_id", "required_cases", "errors", "passed"},
        "equivalent-region gate",
    )
    _require(
        gate["contract_id"] == REGION_INVARIANCE_CONTRACT
        and gate["required_cases"] == list(REGION_INVARIANCE_CASES)
        and gate["errors"] == []
        and gate["passed"] is True,
        "equivalent-region embedded suite gate differs",
    )
    suite = document.get("suite_acceptance")
    _require(
        isinstance(suite, dict)
        and suite.get("region_invariance_suite_expected") is True
        and suite.get("region_invariance_suite_complete") is True
        and suite.get("passed") is True,
        "equivalent-region suite acceptance failed",
    )
    records = []
    expected_counts = (1, 32)
    for name, expected_count, result in zip(
        REGION_INVARIANCE_CASES, expected_counts, cases, strict=True
    ):
        timing = _validate_cuda_tuning_case(
            result,
            traces,
            manifest,
            name=name,
            paired_real=False,
        )
        workload = result["workload"]
        evidence = result.get("region_equivalence")
        _exact_keys(
            evidence,
            {
                "contract_id",
                "equivalence_group",
                "geometry_region_count",
                "geometry_object_count",
                "material_compute_launches_per_step",
                "effective_material_plan",
                "effective_material_plan_sha256",
            },
            f"equivalent-region {name}",
        )

        _exact_keys(
            workload,
            {
                "name",
                "recipe",
                "size",
                "resolution",
                "material",
                "complex",
                "equivalence_contract_id",
                "equivalence_group",
                "geometry_region_count",
                "geometry_object_count",
            },
            f"equivalent-region {name} workload",
        )

        plan = evidence["effective_material_plan"]
        active_targets, material_launches = _validate_effective_material_plan(
            plan, f"equivalent-region {name} plan"
        )
        _require(
            workload.get("recipe") == "equivalent-overlap"
            and workload.get("size") == [4.0, 4.0, 0.0]
            and workload.get("resolution") == 4
            and workload.get("material") == "drude-1"
            and workload.get("name") == name
            and workload.get("complex") is False
            and workload.get("equivalence_contract_id") == REGION_INVARIANCE_CONTRACT
            and workload.get("equivalence_group") == REGION_EQUIVALENCE_GROUP
            and _is_exact_int(workload.get("geometry_region_count"), expected_count)
            and _is_exact_int(workload.get("geometry_object_count"), expected_count + 1)
            and evidence["contract_id"] == REGION_INVARIANCE_CONTRACT
            and evidence["equivalence_group"] == REGION_EQUIVALENCE_GROUP
            and _is_exact_int(evidence["geometry_region_count"], expected_count)
            and _is_exact_int(evidence["geometry_object_count"], expected_count + 1)
            and evidence["effective_material_plan_sha256"] == _canonical_sha256(plan)
            and _is_exact_int(
                evidence["material_compute_launches_per_step"], material_launches
            )
            and material_launches > 0,
            f"equivalent-region {name} normalization contract differs",
        )
        diagnostic_plan = result.get("diagnostics", {}).get("material_plan")
        _require(
            isinstance(diagnostic_plan, list)
            and [item.get("component") for item in diagnostic_plan]
            == list(FIELD_ARRAYS)
            and all(
                type(item.get("launches")) is int and item["launches"] >= 0
                for item in diagnostic_plan
            )
            and sum(item["launches"] for item in diagnostic_plan) == material_launches,
            f"equivalent-region {name} material launch summary differs",
        )
        records.append(
            {
                "region_count": expected_count,
                "plan": plan,
                "active_targets": active_targets,
                "material_launches": material_launches,
                "kernel_launches": timing["trace"]["kernel_launches"],
                "compiled_regions": timing["trace"]["compiled_region_events"],
            }
        )
    baseline, expanded = records
    _require(
        _type_exact_equal(baseline["plan"], expanded["plan"])
        and baseline["active_targets"] == expanded["active_targets"]
        and baseline["material_launches"] == expanded["material_launches"]
        and baseline["kernel_launches"] == expanded["kernel_launches"]
        and baseline["compiled_regions"] == expanded["compiled_regions"],
        "equivalent-region raw plan or launch count increased",
    )
    return {
        "environment": _gpu_environment(document, "equivalent-region CUDA"),
        "hostname": document.get("environment", {}).get("hostname"),
        "baseline_region_count": baseline["region_count"],
        "expanded_region_count": expanded["region_count"],
        "active_targets": baseline["active_targets"],
        "material_launches_per_step": baseline["material_launches"],
        "profile_kernel_launches": baseline["kernel_launches"],
    }


def _validate_policy_scope(
    scope: Any,
    reader: ArtifactReader,
    manifest: dict[str, Any],
    candidate: dict[str, str],
) -> dict[str, Any]:
    _exact_keys(
        scope,
        {
            "policy_matrix",
            "policy_traces",
            "paired_real_differential",
            "paired_real_tuning",
            "paired_real_traces",
            "region_invariance",
            "region_invariance_traces",
        },
        "policy/paired-real scope",
    )
    artifact = reader.load(scope["policy_matrix"], "policy matrix")
    document = artifact.document
    _document_candidate_matches(document, candidate, required=True)
    cases = _tuning_cases(document, "policy matrix")
    _require(
        [case.get("case") for case in cases] == list(POLICY_CASES),
        "policy matrix case closure differs",
    )
    suite = document.get("suite_acceptance")
    _require(
        isinstance(suite, dict)
        and suite.get("policy_matrix_expected") is True
        and suite.get("policy_matrix_complete") is True
        and suite.get("passed") is True,
        "policy matrix suite acceptance failed",
    )
    traces = _load_traces(reader, scope["policy_traces"], "policy traces", count=32)
    _bind_tuning_traces(document, traces, "policy matrix")
    environment = document.get("environment")
    _require(
        isinstance(environment, dict)
        and environment.get("cuda_runtime") is not None
        and isinstance(environment.get("devices"), list)
        and bool(environment["devices"]),
        "policy matrix GPU inventory is absent",
    )
    diagnostic_trace_digests: set[str] = set()
    for matrix in cases:
        label = f"policy {matrix['case']}"
        _exact_keys(
            matrix,
            {
                "case",
                "comparison_valid",
                "invalid_reason",
                "forced_execution_representations",
                "forced_compile_cache_keys",
                "forced_execution_topologies",
                "auto_compile_cache_key",
                "auto_policy_executions",
                "all_acceptance_passed",
                "auto_to_fastest_forced_ratio",
                "within_ten_percent",
                "results",
                "passed",
            },
            label,
        )
        results = matrix.get("results")
        _require(
            isinstance(results, dict) and set(results) == set(POLICIES),
            f"{label} policy closure differs",
        )
        timings = {}
        topologies = {}
        representations = {}
        resolved_policies = {}
        cache_keys = {}
        for policy in POLICIES:
            (
                timings[policy],
                topologies[policy],
                representations[policy],
                resolved_policies[policy],
                cache_keys[policy],
            ) = _validate_policy_run(
                results[policy],
                f"{label} {policy}",
                policy=policy,
                case_name=matrix["case"],
                traces=traces,
                manifest=manifest,
                reader=reader,
                diagnostic_trace_digests=diagnostic_trace_digests,
            )
        forced = ("dense", "compact", "tiled")
        _require(
            len({topologies[policy] for policy in forced}) == 1
            and topologies["auto"] == topologies["dense"],
            f"{label} policy target topology differs",
        )
        _require(
            len({representations[policy] for policy in forced}) == 3,
            f"{label} forced representations are not distinct",
        )
        _require(
            len({cache_keys[policy] for policy in forced}) == 3,
            f"{label} forced compile cache keys are not distinct",
        )
        if len(resolved_policies["auto"]) == 1:
            selected = resolved_policies["auto"][0]
            _require(
                cache_keys["auto"] == cache_keys[selected],
                f"{label} auto compile cache key differs from its forced path",
            )
        expected_representations = {
            policy: list(representations[policy]) for policy in forced
        }
        expected_cache_keys = {policy: cache_keys[policy] for policy in forced}
        expected_topologies = {
            policy: [list(item) for item in topologies[policy]] for policy in forced
        }
        auto_executions = (
            results["auto"]
            .get("diagnostics", {})
            .get("dispersive", {})
            .get("policy_executions")
        )
        _require(
            _type_exact_equal(
                matrix["forced_execution_representations"], expected_representations
            )
            and _type_exact_equal(
                matrix["forced_compile_cache_keys"], expected_cache_keys
            )
            and _type_exact_equal(
                matrix["forced_execution_topologies"], expected_topologies
            )
            and matrix["auto_compile_cache_key"] == cache_keys["auto"]
            and isinstance(auto_executions, list)
            and _type_exact_equal(matrix["auto_policy_executions"], auto_executions),
            f"{label} embedded policy summaries differ from raw executions",
        )
        ratio = timings["auto"] / min(timings[policy] for policy in forced)
        _close(
            matrix.get("auto_to_fastest_forced_ratio"),
            ratio,
            f"{label} auto ratio",
        )
        _require(ratio <= 1.10, f"{label} auto policy exceeds 1.10x")
        _require(
            matrix.get("comparison_valid") is True
            and matrix.get("invalid_reason") is None
            and matrix.get("all_acceptance_passed") is True
            and matrix.get("within_ten_percent") is True
            and matrix.get("passed") is True,
            f"{label} embedded acceptance is not exact",
        )
    _require(
        len(diagnostic_trace_digests)
        == len(POLICY_CASES) * len(POLICY_WRITE_OPERATIONS),
        "policy matrix uncompiled diagnostic trace closure differs",
    )
    paired = reader.load(scope["paired_real_differential"], "paired-real differential")
    _validate_differential(
        paired,
        reader,
        manifest,
        candidate,
        scope="paired-real",
    )
    paired_traces = _load_traces(
        reader,
        scope["paired_real_traces"],
        "paired-real CUDA traces",
        count=2,
    )
    paired_tuning = _validate_paired_real_tuning(
        reader.load(scope["paired_real_tuning"], "paired-real CUDA tuning"),
        paired_traces,
        manifest,
        candidate,
    )
    region_traces = _load_traces(
        reader,
        scope["region_invariance_traces"],
        "equivalent-region traces",
        count=2,
    )
    region = _validate_region_invariance(
        reader.load(scope["region_invariance"], "equivalent-region tuning"),
        region_traces,
        manifest,
        candidate,
    )
    policy_environment = _gpu_environment(document, "policy matrix")
    _require(
        paired_tuning["environment"] == region["environment"] == policy_environment
        and paired_tuning["hostname"] == region["hostname"]
        and isinstance(paired_tuning["hostname"], str)
        and bool(paired_tuning["hostname"]),
        "policy, paired-real, and equivalent-region GPU environments differ",
    )
    return {
        "candidate_evidence": candidate,
        "environment": policy_environment,
        "hostname": paired_tuning["hostname"],
        "matrix_count": len(cases),
        "paired_real": paired_tuning,
        "region_invariance": region,
    }


def _gpu_environment(document: dict[str, Any], label: str) -> dict[str, Any]:
    environment = document.get("environment")
    _require(isinstance(environment, dict), f"{label} environment is absent")
    host_contract = _validate_host_contract(
        environment.get("host_contract"),
        f"{label} host contract",
        environment=environment,
    )
    _require_successful_command_statuses(
        environment,
        label,
        require_cuda_topology=True,
    )
    devices = environment.get("devices")
    _require(
        isinstance(devices, list) and bool(devices),
        f"{label} GPU inventory is malformed",
    )
    for index, device in enumerate(devices):
        device_label = f"{label} GPU {index}"
        _exact_keys(
            device,
            {"index", "name", "memory_bytes", "capability", "multiprocessors"},
            device_label,
        )
        capability = device["capability"]
        _require(
            _is_exact_int(device["index"], index)
            and isinstance(device["name"], str)
            and bool(device["name"])
            and type(device["memory_bytes"]) is int
            and device["memory_bytes"] > 0
            and isinstance(capability, list)
            and len(capability) == 2
            and type(capability[0]) is int
            and capability[0] > 0
            and type(capability[1]) is int
            and capability[1] >= 0
            and type(device["multiprocessors"]) is int
            and device["multiprocessors"] > 0,
            f"{device_label} identity is malformed",
        )
    cuda_runtime = environment.get("cuda_runtime")
    _require(
        isinstance(cuda_runtime, str) and bool(cuda_runtime),
        f"{label} CUDA runtime is absent",
    )
    topology = environment.get("gpu_topology", environment.get("topology"))
    _require(
        isinstance(topology, str) and bool(topology.strip()),
        f"{label} GPU topology is absent",
    )
    return {
        "host_contract": host_contract,
        "common_host_identity": _common_host_identity(host_contract),
        "platform": environment.get("platform"),
        "python": environment.get("python"),
        "torch": environment.get("torch"),
        "cuda_runtime": cuda_runtime,
        "devices": devices,
        "topology": topology.strip(),
        "runtime_identity": {
            "kind": "cuda",
            "torch": host_contract["runtime_identity"]["torch"],
            "cuda_runtime": cuda_runtime,
            "devices": devices,
            "topology": topology.strip(),
        },
    }


def _validate_single_gpu_scope(
    scope: Any,
    reader: ArtifactReader,
    manifest: dict[str, Any],
    candidate: dict[str, str],
) -> dict[str, Any]:
    _exact_keys(
        scope,
        {"cuda_gates", "correctness", "traces"},
        "single-GPU scope",
    )
    artifact = reader.load(scope["cuda_gates"], "single-GPU CUDA gates")
    document = artifact.document
    _document_candidate_matches(document, candidate, required=True)
    cases = _tuning_cases(document, "single-GPU CUDA gates")
    _require(
        [case.get("workload", {}).get("name") for case in cases] == list(CUDA_CASES),
        "single-GPU CUDA case closure differs",
    )
    traces = _load_traces(reader, scope["traces"], "single-GPU traces", count=6)
    _bind_tuning_traces(document, traces, "single-GPU CUDA gates")
    cuda_raw_by_case = {}
    for index, result in enumerate(cases):
        name = CUDA_CASES[index]
        timing = _validate_cuda_tuning_case(
            result,
            traces,
            manifest,
            name=name,
            paired_real=False,
        )
        cuda_raw_by_case[name] = timing["raw_seconds_per_step"]
    correctness = reader.load(
        scope["correctness"],
        "single-GPU differential",
    )
    _validate_differential(
        correctness,
        reader,
        manifest,
        candidate,
        scope="single-gpu-cuda",
    )
    environment = _gpu_environment(document, "single-GPU")
    return {
        "candidate_evidence": candidate,
        "environment": environment,
        "cuda_raw_seconds_per_step": cuda_raw_by_case,
    }


TWO_GPU_ACCEPTANCE_CHECKS = {
    "independent_subprocesses",
    "worker_contract",
    "environment_complete",
    "timing_reduction_complete",
    "rank_evidence_complete",
    "decomposition_complete",
    "imbalance_evidence_complete",
    "device_memory_bounded",
    "storage_addresses_stable",
    "halos_device_resident",
    "peer_access_reported",
    "steady_state_transfers_zero",
    "nccl_phases_complete",
    "interior_halo_overlap_observed",
    "performance_threshold",
}


def _positive_ratio(values: list[Any], label: str) -> float:
    parsed = [
        _finite_float(value, f"{label}[{index}]", positive=True)
        for index, value in enumerate(values)
    ]
    _require(bool(parsed), f"{label} is empty")
    return max(parsed) / min(parsed)


def _validate_two_gpu_memory(value: Any, label: str) -> None:
    _exact_keys(
        value,
        {
            "allocated_before_bytes",
            "allocated_after_bytes",
            "allocated_growth_bytes",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "bounded",
        },
        label,
    )
    before = value["allocated_before_bytes"]
    after = value["allocated_after_bytes"]
    growth = value["allocated_growth_bytes"]
    peak = value["peak_allocated_bytes"]
    reserved = value["peak_reserved_bytes"]
    bounded = (
        type(before) is int
        and before >= 0
        and type(after) is int
        and after >= 0
        and type(growth) is int
        and growth == after - before
        and growth <= 1024 * 1024
        and type(peak) is int
        and peak > 0
        and type(reserved) is int
        and reserved >= peak
    )
    _require(
        bounded and value["bounded"] is bounded,
        f"{label} is not independently bounded",
    )


def _validate_two_gpu_storage(value: Any, label: str) -> None:
    _exact_keys(
        value,
        {
            "address_count",
            "initial_address_sha256",
            "final_address_sha256",
            "addresses_stable",
            "alias_count",
            "tracked_tensor_count",
            "device_resident",
            "resident_bytes",
            "category_bytes",
        },
        label,
    )
    categories = value["category_bytes"]
    stable = (
        isinstance(value["initial_address_sha256"], str)
        and SHA256_RE.fullmatch(value["initial_address_sha256"]) is not None
        and value["final_address_sha256"] == value["initial_address_sha256"]
    )
    _require(
        type(value["address_count"]) is int
        and value["address_count"] > 0
        and type(value["alias_count"]) is int
        and 0 <= value["alias_count"] <= value["address_count"]
        and type(value["tracked_tensor_count"]) is int
        and value["tracked_tensor_count"] > 0
        and type(value["resident_bytes"]) is int
        and value["resident_bytes"] > 0
        and isinstance(categories, dict)
        and bool(categories)
        and all(
            isinstance(name, str) and bool(name) and type(size) is int and size >= 0
            for name, size in categories.items()
        )
        and sum(categories.values()) == value["resident_bytes"]
        and stable
        and value["addresses_stable"] is stable
        and value["device_resident"] is True,
        f"{label} is malformed, unstable, or not device resident",
    )


def _validate_two_gpu_halo(value: Any, device: str, label: str) -> None:
    _exact_keys(
        value,
        {
            "buffer_count",
            "bytes",
            "initial_address_sha256",
            "final_address_sha256",
            "addresses_stable",
            "alias_count",
            "device",
            "device_resident",
        },
        label,
    )
    stable = (
        isinstance(value["initial_address_sha256"], str)
        and SHA256_RE.fullmatch(value["initial_address_sha256"]) is not None
        and value["final_address_sha256"] == value["initial_address_sha256"]
    )
    _require(
        type(value["buffer_count"]) is int
        and value["buffer_count"] > 0
        and type(value["bytes"]) is int
        and value["bytes"] > 0
        and type(value["alias_count"]) is int
        and 0 <= value["alias_count"] <= value["buffer_count"]
        and value["device"] == device
        and stable
        and value["addresses_stable"] is stable
        and value["device_resident"] is True,
        f"{label} is malformed, unstable, or not device resident",
    )


def _validate_two_gpu_decomposition(
    value: Any,
    expected_shape: list[int],
    label: str,
) -> dict[str, Any]:
    _exact_keys(
        value,
        {
            "global_shape",
            "axis",
            "cut",
            "rank_costs",
            "device_weights",
            "communication_cells",
            "source_crossings",
            "identity",
            "axis_name",
        },
        label,
    )
    axis = value["axis"]
    cut = value["cut"]
    costs = value["rank_costs"]
    weights = value["device_weights"]
    _require(
        _type_exact_equal(value["global_shape"], expected_shape)
        and type(axis) is int
        and axis in (0, 1, 2)
        and type(cut) is int
        and 1 < cut < expected_shape[axis] - 1
        and isinstance(costs, list)
        and len(costs) == 2
        and all(
            _finite_float(item, f"{label} rank cost", positive=True) for item in costs
        )
        and isinstance(weights, list)
        and len(weights) == 2
        and all(
            _finite_float(item, f"{label} device weight", positive=True)
            for item in weights
        )
        and math.isclose(sum(weights), 1.0, rel_tol=1e-12, abs_tol=1e-12)
        and type(value["communication_cells"]) is int
        and value["communication_cells"] > 0
        and type(value["source_crossings"]) is int
        and value["source_crossings"] >= 0
        and value["axis_name"] == "xyz"[axis],
        f"{label} geometry or cost evidence differs",
    )
    identity_payload = (
        tuple(expected_shape),
        axis,
        cut,
        tuple(round(item, 12) for item in weights),
    )
    _require(
        value["identity"] == _sha256(repr(identity_payload).encode()),
        f"{label} identity differs from its raw fields",
    )
    return value


def _validate_two_gpu_rank_evidence(
    value: Any,
    decomposition: dict[str, Any],
    repeats: int,
    label: str,
) -> list[dict[str, Any]]:
    _require(isinstance(value, list) and len(value) == 2, f"{label} closure differs")
    shape = decomposition["global_shape"]
    axis = decomposition["axis"]
    cut = decomposition["cut"]
    for rank, record in enumerate(value):
        rank_label = f"{label} rank {rank}"
        _exact_keys(
            record,
            {
                "rank",
                "local_rank",
                "device",
                "peer_rank",
                "peer_access",
                "construction_seconds",
                "capture_seconds",
                "raw_seconds",
                "memory",
                "storage",
                "halo",
                "decomposition_identity",
                "local_field_shape",
                "global_offset",
            },
            rank_label,
        )
        device = f"cuda:{rank}"
        expected_local_shape = [
            (cut if rank == 0 else length - cut) if index == axis else length
            for index, length in enumerate(shape)
        ]
        expected_offset = [
            cut if rank == 1 and index == axis else 0 for index in range(3)
        ]
        _require(
            _is_exact_int(record["rank"], rank)
            and _is_exact_int(record["local_rank"], rank)
            and record["device"] == device
            and _is_exact_int(record["peer_rank"], 1 - rank)
            and type(record["peer_access"]) is bool
            and bool(
                _finite_float(
                    record["construction_seconds"],
                    f"{rank_label} construction",
                    positive=True,
                )
            )
            and bool(
                _finite_float(
                    record["capture_seconds"],
                    f"{rank_label} capture",
                    positive=True,
                )
            )
            and record["decomposition_identity"] == decomposition["identity"]
            and _type_exact_equal(record["local_field_shape"], expected_local_shape)
            and _type_exact_equal(record["global_offset"], expected_offset),
            f"{rank_label} identity or decomposition differs",
        )
        _raw_values(record["raw_seconds"], f"{rank_label} samples", count=repeats)
        _validate_two_gpu_memory(record["memory"], f"{rank_label} memory")
        _validate_two_gpu_storage(record["storage"], f"{rank_label} storage")
        _validate_two_gpu_halo(record["halo"], device, f"{rank_label} halo")
    return value


def _validate_two_gpu_imbalance(
    value: Any,
    rank_samples: list[list[float]],
    ranks: list[dict[str, Any]],
    decomposition: dict[str, Any],
    label: str,
) -> None:
    _exact_keys(
        value,
        {
            "rank_seconds_ratio_per_repeat",
            "rank_seconds_ratio_median",
            "rank_seconds_ratio_maximum",
            "material_cost_ratio",
            "device_adjusted_cost_ratio",
            "resident_storage_ratio",
            "peak_allocated_ratio",
            "halo_bytes_ratio",
        },
        label,
    )
    timing = [
        _positive_ratio(sample, f"{label} timing replicate {index}")
        for index, sample in enumerate(rank_samples)
    ]
    recorded = _raw_values(
        value["rank_seconds_ratio_per_repeat"],
        f"{label} timing ratios",
        count=len(timing),
    )
    for index, (actual, expected) in enumerate(zip(recorded, timing, strict=True)):
        _close(actual, expected, f"{label} timing ratio {index}")
    expected = {
        "rank_seconds_ratio_median": median(timing),
        "rank_seconds_ratio_maximum": max(timing),
        "material_cost_ratio": _positive_ratio(
            decomposition["rank_costs"], f"{label} material costs"
        ),
        "device_adjusted_cost_ratio": _positive_ratio(
            [
                cost / weight
                for cost, weight in zip(
                    decomposition["rank_costs"],
                    decomposition["device_weights"],
                    strict=True,
                )
            ],
            f"{label} adjusted costs",
        ),
        "resident_storage_ratio": _positive_ratio(
            [record["storage"]["resident_bytes"] for record in ranks],
            f"{label} storage",
        ),
        "peak_allocated_ratio": _positive_ratio(
            [record["memory"]["peak_allocated_bytes"] for record in ranks],
            f"{label} allocation",
        ),
        "halo_bytes_ratio": _positive_ratio(
            [record["halo"]["bytes"] for record in ranks],
            f"{label} halo",
        ),
    }
    for name, expected_value in expected.items():
        _require(
            expected_value >= 1.0,
            f"{label} recomputed {name} is outside its ratio contract",
        )
        _close(value[name], expected_value, f"{label} {name}")


def _validate_two_gpu_profile_record(
    value: Any,
    profile_steps: int,
    label: str,
) -> None:
    _exact_keys(
        value,
        {
            "trace",
            "trace_size_bytes",
            "trace_sha256",
            "kernel_launches",
            "host_to_device_events",
            "device_to_host_events",
            "nccl_kernel_launches",
            "nccl_device_us",
            "nccl_compute_overlap_us",
            "nccl_exposed_us",
            "overlap_fraction",
            "halo_annotations",
        },
        label,
    )
    _canonical_bundle_path(value["trace"], f"{label} trace")
    device_us = _finite_float(value["nccl_device_us"], f"{label} NCCL", positive=True)
    overlap_us = _finite_float(
        value["nccl_compute_overlap_us"], f"{label} overlap", positive=True
    )
    exposed_us = _finite_float(value["nccl_exposed_us"], f"{label} exposed")
    fraction = _finite_float(
        value["overlap_fraction"], f"{label} fraction", positive=True
    )
    annotations = value["halo_annotations"]
    _require(
        type(value["trace_size_bytes"]) is int
        and value["trace_size_bytes"] > 0
        and isinstance(value["trace_sha256"], str)
        and SHA256_RE.fullmatch(value["trace_sha256"]) is not None
        and type(value["kernel_launches"]) is int
        and value["kernel_launches"] > 0
        and _is_exact_int(value["host_to_device_events"], 0)
        and _is_exact_int(value["device_to_host_events"], 0)
        and type(value["nccl_kernel_launches"]) is int
        and value["nccl_kernel_launches"] > 0
        and overlap_us <= device_us
        and exposed_us >= 0.0
        and 0.0 < fraction <= 1.0
        and math.isclose(
            exposed_us, device_us - overlap_us, rel_tol=1e-12, abs_tol=1e-9
        )
        and math.isclose(fraction, overlap_us / device_us, rel_tol=1e-12, abs_tol=1e-15)
        and isinstance(annotations, dict)
        and set(annotations) == set(HALO_ANNOTATIONS),
        f"{label} trace metrics are malformed",
    )
    for name, record in annotations.items():
        _exact_keys(record, {"count", "duration_us"}, f"{label} {name}")
        _require(
            type(record["count"]) is int
            and record["count"] >= profile_steps
            and bool(
                _finite_float(
                    record["duration_us"],
                    f"{label} {name} duration",
                    positive=True,
                )
            ),
            f"{label} {name} annotation evidence is incomplete",
        )


def _load_two_gpu_subprocesses(
    value: Any,
    reader: ArtifactReader,
    candidate: dict[str, str],
    label: str,
) -> dict[str, dict[str, Any]]:
    _exact_keys(value, {"serial", "distributed"}, f"{label} subprocesses")
    children: dict[str, dict[str, Any]] = {}
    paths: set[str] = set()
    commands: dict[str, list[str]] = {}
    for role in ("serial", "distributed"):
        record = value[role]
        role_label = f"{label} {role} subprocess"
        _exact_keys(
            record,
            {
                "role",
                "command",
                "exit_code",
                "stdout_sha256",
                "stdout_size_bytes",
                "stderr_sha256",
                "stderr_size_bytes",
                "artifact_sha256",
                "artifact_size_bytes",
                "artifact",
                "stdout",
                "stderr",
            },
            role_label,
        )
        command = record["command"]
        _require(
            record["role"] == role
            and isinstance(command, list)
            and bool(command)
            and all(isinstance(item, str) and bool(item) for item in command)
            and not any(Path(item).is_absolute() for item in command)
            and _is_exact_int(record["exit_code"], 0),
            f"{role_label} command or exit contract differs",
        )
        artifact = reader.load(
            record["artifact"],
            f"{role_label} artifact",
            expected_media_types={MEDIA_TYPE_JSON},
        )
        stdout = reader.load(
            record["stdout"],
            f"{role_label} stdout",
            expected_media_types={MEDIA_TYPE_JSON},
        )
        stderr = reader.load(
            record["stderr"],
            f"{role_label} stderr",
            json_document=False,
            expected_media_types={MEDIA_TYPE_TEXT},
        )
        _decode_log(stderr.raw, f"{role_label} stderr")
        for name, loaded in (
            ("artifact", artifact),
            ("stdout", stdout),
            ("stderr", stderr),
        ):
            descriptor = loaded.descriptor
            _require(
                record[f"{name}_sha256"] == descriptor["sha256"]
                and record[f"{name}_size_bytes"] == descriptor["size_bytes"],
                f"{role_label} {name} digest or size differs",
            )
            _require(descriptor["path"] not in paths, f"{label} reuses child bytes")
            paths.add(descriptor["path"])
        _require(
            _type_exact_equal(stdout.document, artifact.document),
            f"{role_label} stdout differs from its structured artifact",
        )
        _document_candidate_matches(artifact.document, candidate, required=True)
        children[role] = artifact.document
        commands[role] = command
    _require(
        commands["serial"] != commands["distributed"]
        and "benchmarks.torch_two_gpu" in commands["serial"]
        and "torch.distributed.run" not in commands["serial"]
        and "--worker" in commands["serial"]
        and "serial" in commands["serial"]
        and "torch.distributed.run" in commands["distributed"]
        and "--worker" in commands["distributed"]
        and "distributed" in commands["distributed"],
        f"{label} did not use independent serial and torchrun children",
    )
    return children


def _validate_two_gpu_performance(
    artifact: LoadedArtifact,
    reader: ArtifactReader,
    manifest: dict[str, Any],
    candidate: dict[str, str],
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    document = artifact.document
    _exact_keys(
        document,
        {
            "candidate_evidence",
            "schema_version",
            "case",
            "gate",
            "sizes",
            "measurement",
            "subprocesses",
            "serial",
            "distributed",
            "decomposition",
            "rank_evidence",
            "imbalance",
            "profiles",
            "environment",
            "acceptance",
        },
        "two-GPU performance v2",
    )
    _document_candidate_matches(document, candidate, required=True)
    case = document["case"]
    _require(case in TWO_GPU_CASES, "unknown two-GPU performance case")
    gate, frozen_threshold = TWO_GPU_CASES[case]
    _require(
        _is_exact_int(document["schema_version"], 2) and document["gate"] == gate,
        f"two-GPU {case} schema or gate differs",
    )
    label = f"two-GPU {case}"
    measurement = document["measurement"]
    _exact_keys(
        measurement,
        {"warmup", "steps", "repeats", "profile_steps", "threads_per_rank"},
        f"{label} measurement",
    )
    reference = manifest["reference"]
    _require(
        _is_exact_int(measurement["warmup"], reference["performance_warmup_steps"])
        and _is_exact_int(
            measurement["steps"], reference["performance_steps_per_repeat"]
        )
        and _is_exact_int(measurement["repeats"], reference["performance_repetitions"])
        and _is_exact_int(
            measurement["profile_steps"], reference["performance_profile_steps"]
        )
        and _is_exact_int(measurement["threads_per_rank"], 1),
        f"{label} measurement contract differs",
    )
    steps = measurement["steps"]
    repeats = measurement["repeats"]
    expected_sizes = TWO_GPU_SIZE_CONTRACTS[case]
    serial_cells = math.prod(expected_sizes["serial"])
    distributed_cells = math.prod(expected_sizes["distributed"])
    _require(
        _type_exact_equal(
            document["sizes"],
            {
                "serial": expected_sizes["serial"],
                "distributed": expected_sizes["distributed"],
                "serial_cells": serial_cells,
                "distributed_cells": distributed_cells,
            },
        ),
        f"{label} fixed size or cell counts differ",
    )

    children = _load_two_gpu_subprocesses(
        document["subprocesses"], reader, candidate, label
    )
    _exact_keys(
        children["serial"],
        {
            "schema_version",
            "kind",
            "candidate_evidence",
            "environment",
            "case",
            "size",
            "measurement",
            "serial",
        },
        f"{label} serial child",
    )
    _exact_keys(
        children["distributed"],
        {
            "schema_version",
            "kind",
            "candidate_evidence",
            "environment",
            "case",
            "size",
            "measurement",
            "distributed",
            "decomposition",
            "rank_evidence",
            "profiles",
        },
        f"{label} distributed child",
    )
    _require(
        _is_exact_int(children["serial"]["schema_version"], 2)
        and children["serial"]["kind"] == "two-gpu-serial-worker"
        and _is_exact_int(children["distributed"]["schema_version"], 2)
        and children["distributed"]["kind"] == "two-gpu-distributed-worker"
        and children["serial"]["candidate_evidence"] == candidate
        and children["distributed"]["candidate_evidence"] == candidate
        and children["serial"]["case"] == children["distributed"]["case"] == case
        and _type_exact_equal(children["serial"]["size"], expected_sizes["serial"])
        and _type_exact_equal(
            children["distributed"]["size"], expected_sizes["distributed"]
        )
        and _type_exact_equal(children["serial"]["measurement"], measurement)
        and _type_exact_equal(children["distributed"]["measurement"], measurement)
        and _type_exact_equal(
            children["serial"]["environment"], document["environment"]
        )
        and _type_exact_equal(
            children["distributed"]["environment"], document["environment"]
        )
        and _type_exact_equal(children["serial"]["serial"], document["serial"])
        and _type_exact_equal(
            children["distributed"]["distributed"], document["distributed"]
        )
        and _type_exact_equal(
            children["distributed"]["decomposition"], document["decomposition"]
        )
        and _type_exact_equal(
            children["distributed"]["rank_evidence"], document["rank_evidence"]
        )
        and _type_exact_equal(
            children["distributed"]["profiles"], document["profiles"]
        ),
        f"{label} parent differs from exact child artifacts",
    )

    serial = document["serial"]
    distributed = document["distributed"]
    _exact_keys(
        serial,
        {
            "raw_seconds",
            "median_seconds",
            "steps_per_second",
            "cells_per_second",
            "construction_seconds",
            "capture_seconds",
            "memory",
            "storage",
        },
        f"{label} serial",
    )
    _exact_keys(
        distributed,
        {
            "raw_seconds",
            "median_seconds",
            "steps_per_second",
            "cells_per_second",
            "rank_raw_seconds",
            "construction_seconds",
            "capture_seconds",
            "peak_allocated_bytes_rank0",
            "halo_bytes_rank0",
            "storage_addresses_stable",
        },
        f"{label} distributed",
    )
    serial_raw, serial_seconds_per_step = _raw_summary(
        serial, f"{label} serial", steps=steps, repeats=repeats
    )
    distributed_raw, distributed_seconds_per_step = _raw_summary(
        distributed, f"{label} distributed", steps=steps, repeats=repeats
    )
    serial_median = median(serial_raw)
    distributed_median = median(distributed_raw)
    _close(
        serial["cells_per_second"],
        serial_cells * steps / serial_median,
        f"{label} serial cells/s",
    )
    _close(
        distributed["cells_per_second"],
        distributed_cells * steps / distributed_median,
        f"{label} distributed cells/s",
    )
    _finite_float(
        serial["construction_seconds"], f"{label} serial construction", positive=True
    )
    _finite_float(serial["capture_seconds"], f"{label} serial capture", positive=True)
    _validate_two_gpu_memory(serial["memory"], f"{label} serial memory")
    _validate_two_gpu_storage(serial["storage"], f"{label} serial storage")

    decomposition = _validate_two_gpu_decomposition(
        document["decomposition"],
        expected_sizes["distributed"],
        f"{label} decomposition",
    )
    ranks = _validate_two_gpu_rank_evidence(
        document["rank_evidence"], decomposition, repeats, f"{label} rank evidence"
    )
    rank_samples_value = distributed["rank_raw_seconds"]
    _require(
        isinstance(rank_samples_value, list) and len(rank_samples_value) == repeats,
        f"{label} rank timing closure differs",
    )
    rank_samples: list[list[float]] = []
    for index, (sample, reduced) in enumerate(
        zip(rank_samples_value, distributed_raw, strict=True)
    ):
        values = _raw_values(sample, f"{label} rank timing {index}", count=2)
        _close(reduced, max(values), f"{label} reduced rank timing {index}")
        rank_samples.append(values)
    for rank in range(2):
        expected_rank_raw = [sample[rank] for sample in rank_samples]
        _require(
            _type_exact_equal(ranks[rank]["raw_seconds"], expected_rank_raw),
            f"{label} rank {rank} raw timings differ from reductions",
        )
    _close(
        distributed["construction_seconds"],
        max(record["construction_seconds"] for record in ranks),
        f"{label} distributed construction",
    )
    _close(
        distributed["capture_seconds"],
        max(record["capture_seconds"] for record in ranks),
        f"{label} distributed capture",
    )
    _require(
        distributed["peak_allocated_bytes_rank0"]
        == ranks[0]["memory"]["peak_allocated_bytes"]
        and distributed["halo_bytes_rank0"] == ranks[0]["halo"]["bytes"]
        and distributed["storage_addresses_stable"]
        is all(record["storage"]["addresses_stable"] for record in ranks)
        is True,
        f"{label} distributed memory/storage summary differs",
    )
    _validate_two_gpu_imbalance(
        document["imbalance"], rank_samples, ranks, decomposition, f"{label} imbalance"
    )

    profiles = document["profiles"]
    _require(
        isinstance(profiles, list) and len(profiles) == 2,
        f"{label} profile closure differs",
    )
    for rank, profile in enumerate(profiles):
        _validate_two_gpu_profile_record(
            profile, measurement["profile_steps"], f"{label} profile {rank}"
        )
        _require(
            PurePosixPath(profile["trace"]).name == f"gmes-two-gpu-{rank}.json",
            f"{label} profile {rank} filename differs",
        )

    ratio = (
        distributed["cells_per_second"] / (2.0 * serial["cells_per_second"])
        if gate == "weak"
        else serial_median / distributed_median
    )
    acceptance = document["acceptance"]
    _exact_keys(
        acceptance, {"ratio", "threshold", "checks", "passed"}, f"{label} acceptance"
    )
    _close(acceptance["ratio"], ratio, f"{label} acceptance ratio")
    _require(
        _type_exact_equal(acceptance["threshold"], frozen_threshold),
        f"{label} threshold differs",
    )
    checks = acceptance["checks"]
    _exact_keys(checks, TWO_GPU_ACCEPTANCE_CHECKS, f"{label} acceptance checks")
    recomputed_checks = {name: True for name in TWO_GPU_ACCEPTANCE_CHECKS}
    recomputed_checks["performance_threshold"] = (
        frozen_threshold is None or ratio >= frozen_threshold
    )
    _require(
        _type_exact_equal(checks, recomputed_checks)
        and acceptance["passed"] is all(recomputed_checks.values())
        and acceptance["passed"] is True,
        f"{label} independently recomputed acceptance failed",
    )
    environment = _gpu_environment(document, label)
    _require(
        serial_seconds_per_step > 0.0 and distributed_seconds_per_step > 0.0,
        f"{label} per-step timing is not positive",
    )
    return case, environment, profiles


def _bind_two_gpu_profiles(
    performance: list[tuple[LoadedArtifact, list[dict[str, Any]]]],
    traces: dict[str, tuple[LoadedArtifact, dict[str, Any]]],
) -> None:
    used: set[str] = set()
    for artifact, profiles in performance:
        case = artifact.document["case"]
        for rank, profile in enumerate(profiles):
            _require(
                isinstance(profile, dict), f"two-GPU {case} profile {rank} is malformed"
            )
            trace_path = profile.get("trace")
            _require(
                isinstance(trace_path, str) and bool(trace_path),
                f"two-GPU {case} profile {rank} path is absent",
            )
            matches = [
                (digest, item)
                for digest, item in traces.items()
                if item[0].descriptor["path"] == trace_path
            ]
            _require(
                len(matches) == 1,
                f"two-GPU {case} profile {rank} is not exactly indexed",
            )
            digest, (_trace, summary) = matches[0]
            _require(digest not in used, f"two-GPU {case} reuses a profiler trace")
            used.add(digest)
            _require(
                profile["trace_sha256"] == digest
                and profile["trace_size_bytes"] == summary["chrome_trace_size_bytes"]
                and profile["kernel_launches"] == summary["kernel_launches"]
                and profile["host_to_device_events"] == summary["host_to_device_events"]
                and profile["device_to_host_events"] == summary["device_to_host_events"]
                and profile["nccl_kernel_launches"] == summary["nccl_kernel_launches"],
                f"two-GPU {case} profile {rank} differs from raw trace bytes",
            )
            for name in (
                "nccl_device_us",
                "nccl_compute_overlap_us",
                "nccl_exposed_us",
                "overlap_fraction",
            ):
                _close(
                    profile.get(name),
                    summary[name],
                    f"two-GPU {case} profile {rank} {name}",
                )
            for name in HALO_ANNOTATIONS:
                recorded = profile["halo_annotations"][name]
                observed = summary["halo_annotations"][name]
                _require(
                    recorded["count"] == observed["count"],
                    f"two-GPU {case} profile {rank} {name} count differs",
                )
                _close(
                    recorded["duration_us"],
                    observed["duration_us"],
                    f"two-GPU {case} profile {rank} {name} duration",
                )
    _require(used == set(traces), "two-GPU contains unreferenced traces")


def _two_gpu_whole_shape(case: str) -> tuple[int, int, int]:
    if case.startswith("axis-"):
        return (7, 6, 5)
    if case == "collapsed-1d":
        return (8, 1, 1)
    if case == "collapsed-2d":
        return (8, 6, 1)
    _require(
        case in TWO_GPU_CORRECTNESS_CASES,
        f"unknown two-GPU correctness case {case!r}",
    )
    return (8, 8, 8)


def _two_gpu_field_shapes(
    whole_shape: tuple[int, int, int],
) -> dict[str, list[int]]:
    nx, ny, nz = whole_shape
    return {
        "Ex": [nx, ny + 1, nz + 1],
        "Ey": [nx + 1, ny, nz + 1],
        "Ez": [nx + 1, ny + 1, nz],
        "Hx": [nx, ny + 1, nz + 1],
        "Hy": [nx + 1, ny, nz + 1],
        "Hz": [nx + 1, ny + 1, nz],
    }


def _two_gpu_case_raw_names() -> list[str]:
    names = [
        f"capture/{step}/{role}/{field}"
        for step in TWO_GPU_CAPTURE_STEPS
        for field in FIELD_ARRAYS
        for role in ("distributed", "serial")
    ]
    names.extend(
        f"checkpoint/{phase}/{field}"
        for phase in ("expected", "replay", "serial")
        for field in FIELD_ARRAYS
    )
    names.extend(
        f"storage/rank/{rank}/{phase}"
        for rank in (0, 1)
        for phase in ("initial", "final")
    )
    names.extend(f"storage/serial/{phase}" for phase in ("initial", "final"))
    return names


def _two_gpu_long_raw_names() -> list[str]:
    return [
        f"{phase}/{field}"
        for phase in ("initial", "distributed", "serial")
        for field in FIELD_ARRAYS
    ]


def _load_two_gpu_raw_evidence(
    value: Any,
    reader: ArtifactReader,
    names: list[str],
    field_shapes: dict[str, list[int]],
    label: str,
    used_artifacts: set[tuple[Path, str]],
) -> dict[str, np.ndarray]:
    _exact_keys(
        value,
        {"artifact", "array_names", "field_shapes"},
        f"{label} raw evidence",
    )
    _require(
        _type_exact_equal(value["array_names"], names)
        and _type_exact_equal(value["field_shapes"], field_shapes),
        f"{label} raw array closure or field shapes differ",
    )
    artifact = reader.load(
        value["artifact"],
        f"{label} raw archive",
        json_document=False,
        expected_media_types={MEDIA_TYPE_NPZ},
    )
    identity = (artifact.path, artifact.descriptor["sha256"])
    _require(identity not in used_artifacts, f"{label} reuses a raw archive")
    used_artifacts.add(identity)
    return _npz_arrays(artifact, names, f"{label} raw archive")


def _two_gpu_field_snapshot(
    arrays: dict[str, np.ndarray],
    prefix: str,
    field_shapes: dict[str, list[int]],
    dtype: np.dtype[Any],
    label: str,
) -> dict[str, np.ndarray]:
    result = {}
    for field in FIELD_ARRAYS:
        array = arrays[f"{prefix}/{field}"]
        _require(
            array.shape == tuple(field_shapes[field]) and array.dtype == dtype,
            f"{label} {field} shape or dtype differs",
        )
        _require(
            bool(np.isfinite(array).all()),
            f"{label} {field} contains non-finite values",
        )
        result[field] = array
    return result


def _two_gpu_maximum_error(
    actual: dict[str, np.ndarray],
    expected: dict[str, np.ndarray],
) -> float:
    return max(
        float(np.max(np.abs(actual[field] - expected[field]), initial=0.0))
        for field in FIELD_ARRAYS
    )


def _two_gpu_storage_digest(names: list[str], addresses: np.ndarray) -> str:
    hasher = hashlib.sha256()
    hasher.update(json.dumps(names, separators=(",", ":")).encode())
    hasher.update(addresses.tobytes(order="C"))
    return hasher.hexdigest()


def _validate_two_gpu_raw_storage(
    summary: Any,
    arrays: dict[str, np.ndarray],
    prefix: str,
    label: str,
    *,
    rank: int | None,
) -> None:
    keys = {
        "address_names",
        "address_count",
        "initial_sha256",
        "final_sha256",
        "addresses_stable",
    }
    if rank is not None:
        keys.add("rank")
    _exact_keys(summary, keys, label)
    names = summary["address_names"]
    _require(
        isinstance(names, list)
        and bool(names)
        and names == sorted(names)
        and len(names) == len(set(names))
        and all(isinstance(name, str) and bool(name) for name in names),
        f"{label} address-name closure differs",
    )
    if rank is not None:
        _require(_is_exact_int(summary["rank"], rank), f"{label} rank differs")
    initial = arrays[f"{prefix}/initial"]
    final = arrays[f"{prefix}/final"]
    _require(
        initial.dtype == np.dtype(np.uint64)
        and final.dtype == np.dtype(np.uint64)
        and initial.shape == (len(names),)
        and final.shape == (len(names),)
        and _is_exact_int(summary["address_count"], len(names)),
        f"{label} raw address array differs",
    )
    stable = bool(np.array_equal(initial, final))
    _require(
        summary["initial_sha256"] == _two_gpu_storage_digest(names, initial)
        and summary["final_sha256"] == _two_gpu_storage_digest(names, final)
        and summary["addresses_stable"] is stable
        and stable,
        f"{label} raw storage digest or stability differs",
    )


def _validate_two_gpu_correctness_v3(
    artifact: LoadedArtifact,
    reader: ArtifactReader,
    candidate: dict[str, str],
) -> tuple[bool, dict[str, Any]]:
    document = artifact.document
    _require(isinstance(document, dict), "two-GPU correctness root must be an object")
    _exact_keys(
        document,
        {
            "candidate_evidence",
            "environment",
            "schema_version",
            "contract_id",
            "capture_steps",
            "capture_graphs",
            "execution_mode",
            "maximum_error",
            "passed",
            "cases",
            "long_stability",
            "suite_acceptance",
        },
        "two-GPU correctness",
    )
    _document_candidate_matches(document, candidate, required=True)
    _require(
        _is_exact_int(document["schema_version"], 3)
        and document["contract_id"] == "two-gpu-full-field-replay-v3"
        and _type_exact_equal(document["capture_steps"], list(TWO_GPU_CAPTURE_STEPS)),
        "two-GPU correctness schema differs",
    )
    capture_graphs = document["capture_graphs"]
    _require(
        type(capture_graphs) is bool
        and document["execution_mode"] == ("graph" if capture_graphs else "eager"),
        "two-GPU correctness graph mode is absent or inconsistent",
    )
    cases = document["cases"]
    _require(
        isinstance(cases, list)
        and [record.get("name") for record in cases] == list(TWO_GPU_CORRECTNESS_CASES),
        "two-GPU correctness case closure differs",
    )
    maximum = 0.0
    passed = True
    raw_names = _two_gpu_case_raw_names()
    used_raw_artifacts: set[tuple[Path, str]] = set()
    for record in cases:
        case = record["name"]
        label = f"two-GPU correctness {case}"
        _exact_keys(
            record,
            {
                "name",
                "axis",
                "cut",
                "capture_errors",
                "checkpoint_determinism_error",
                "checkpoint_reference_error",
                "rank0_probe_count",
                "checkpoint_replay_steps",
                "checkpoint_replay_fields",
                "rank_storage",
                "serial_storage",
                "raw_evidence",
            },
            label,
        )
        whole_shape = _two_gpu_whole_shape(case)
        field_shapes = _two_gpu_field_shapes(whole_shape)
        arrays = _load_two_gpu_raw_evidence(
            record["raw_evidence"],
            reader,
            raw_names,
            field_shapes,
            label,
            used_raw_artifacts,
        )
        dtype = np.dtype(np.complex128 if case.endswith("-bloch") else np.float64)
        recomputed_errors = {}
        for step in TWO_GPU_CAPTURE_STEPS:
            distributed = _two_gpu_field_snapshot(
                arrays,
                f"capture/{step}/distributed",
                field_shapes,
                dtype,
                f"{label} capture {step} distributed",
            )
            serial = _two_gpu_field_snapshot(
                arrays,
                f"capture/{step}/serial",
                field_shapes,
                dtype,
                f"{label} capture {step} serial",
            )
            recomputed_errors[str(step)] = _two_gpu_maximum_error(distributed, serial)
        errors = record["capture_errors"]
        _require(
            isinstance(errors, dict)
            and set(errors) == {str(step) for step in TWO_GPU_CAPTURE_STEPS},
            f"{label} capture error closure differs",
        )
        for step, error in recomputed_errors.items():
            _close(errors[step], error, f"{label} capture {step} maximum error")
        expected = _two_gpu_field_snapshot(
            arrays,
            "checkpoint/expected",
            field_shapes,
            dtype,
            f"{label} checkpoint expected",
        )
        replay = _two_gpu_field_snapshot(
            arrays,
            "checkpoint/replay",
            field_shapes,
            dtype,
            f"{label} checkpoint replay",
        )
        serial = _two_gpu_field_snapshot(
            arrays,
            "checkpoint/serial",
            field_shapes,
            dtype,
            f"{label} checkpoint serial",
        )
        determinism = _two_gpu_maximum_error(replay, expected)
        reference = _two_gpu_maximum_error(replay, serial)
        _close(
            record["checkpoint_determinism_error"],
            determinism,
            f"{label} checkpoint determinism",
        )
        _close(
            record["checkpoint_reference_error"],
            reference,
            f"{label} checkpoint reference",
        )
        rank_storage = record["rank_storage"]
        _require(
            _is_exact_int(record["checkpoint_replay_steps"], 5)
            and _type_exact_equal(
                record["checkpoint_replay_fields"], list(FIELD_ARRAYS)
            )
            and isinstance(rank_storage, list)
            and len(rank_storage) == 2,
            f"{label} replay/storage closure differs",
        )
        for rank, storage in enumerate(rank_storage):
            _validate_two_gpu_raw_storage(
                storage,
                arrays,
                f"storage/rank/{rank}",
                f"{label} rank {rank} storage",
                rank=rank,
            )
        _validate_two_gpu_raw_storage(
            record["serial_storage"],
            arrays,
            "storage/serial",
            f"{label} serial storage",
            rank=None,
        )
        expected_axis = int(case[5]) if case.startswith("axis-") else 0
        _require(
            _is_exact_int(record["axis"], expected_axis)
            and type(record["cut"]) is int
            and 1 < record["cut"] < whole_shape[expected_axis] - 1
            and type(record["rank0_probe_count"]) is int
            and record["rank0_probe_count"] >= 0,
            f"{label} topology/probe metadata differs",
        )
        case_maximum = max(recomputed_errors.values())
        maximum = max(maximum, case_maximum)
        passed = (
            passed
            and case_maximum <= TWO_GPU_CORRECTNESS_ATOL
            and determinism == 0.0
            and reference <= TWO_GPU_CORRECTNESS_ATOL
        )
    _close(document["maximum_error"], maximum, "two-GPU correctness maximum error")
    stability = document["long_stability"]
    _exact_keys(
        stability,
        {
            "steps",
            "maximum_error",
            "finite",
            "initial_energy",
            "final_energy",
            "energy_ratio",
            "raw_evidence",
        },
        "two-GPU long stability",
    )
    long_shapes = _two_gpu_field_shapes((16, 12, 8))
    long_arrays = _load_two_gpu_raw_evidence(
        stability["raw_evidence"],
        reader,
        _two_gpu_long_raw_names(),
        long_shapes,
        "two-GPU long stability",
        used_raw_artifacts,
    )
    initial = _two_gpu_field_snapshot(
        long_arrays,
        "initial",
        long_shapes,
        np.dtype(np.float64),
        "two-GPU long stability initial",
    )
    distributed = _two_gpu_field_snapshot(
        long_arrays,
        "distributed",
        long_shapes,
        np.dtype(np.float64),
        "two-GPU long stability distributed",
    )
    serial = _two_gpu_field_snapshot(
        long_arrays,
        "serial",
        long_shapes,
        np.dtype(np.float64),
        "two-GPU long stability serial",
    )
    stability_error = _two_gpu_maximum_error(distributed, serial)
    initial_energy = sum(
        float(np.square(np.abs(initial[field])).sum()) for field in FIELD_ARRAYS
    )
    final_energy = sum(
        float(np.square(np.abs(distributed[field])).sum()) for field in FIELD_ARRAYS
    )
    _require(
        math.isfinite(initial_energy)
        and initial_energy > 0.0
        and math.isfinite(final_energy)
        and final_energy > 0.0,
        "two-GPU long-stability energy is not finite and positive",
    )
    energy_ratio = final_energy / initial_energy
    _close(
        stability["maximum_error"],
        stability_error,
        "two-GPU long-stability maximum error",
    )
    _close(
        stability["initial_energy"],
        initial_energy,
        "two-GPU long-stability initial energy",
    )
    _close(
        stability["final_energy"],
        final_energy,
        "two-GPU long-stability final energy",
    )
    _close(
        stability["energy_ratio"],
        energy_ratio,
        "two-GPU long-stability energy ratio",
    )
    long_passed = (
        type(stability["steps"]) is int
        and stability["steps"] >= 1000
        and stability["finite"] is True
        and stability_error <= TWO_GPU_CORRECTNESS_ATOL
        and energy_ratio < 100.0
    )
    passed = passed and long_passed
    _require(document["passed"] is passed and passed, "two-GPU correctness failed")
    environment = _gpu_environment(document, "two-GPU correctness")
    suite = document["suite_acceptance"]
    _exact_keys(
        suite,
        {
            "required_cases",
            "required_capture_steps",
            "required_long_steps",
            "checks",
            "passed",
        },
        "two-GPU correctness suite acceptance",
    )
    expected_checks = {
        "environment_complete": True,
        "case_closure_complete": True,
        "capture_steps_complete": True,
        "complete_field_replay": True,
        "rank_storage_stable": True,
        "raw_full_fields_bound": True,
        "long_stability_complete": True,
        "numerical_acceptance": True,
    }
    _exact_keys(
        suite["checks"],
        set(expected_checks),
        "two-GPU correctness suite checks",
    )
    _require(
        _type_exact_equal(suite["required_cases"], list(TWO_GPU_CORRECTNESS_CASES))
        and _type_exact_equal(
            suite["required_capture_steps"], list(TWO_GPU_CAPTURE_STEPS)
        )
        and _is_exact_int(suite["required_long_steps"], 1000)
        and _type_exact_equal(suite["checks"], expected_checks)
        and suite["passed"] is True,
        "two-GPU correctness suite acceptance differs from raw evidence",
    )
    return capture_graphs, environment


def _decode_log(raw: bytes, label: str) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise EvidenceError(f"{label} is not UTF-8") from error


def _validate_failure_run(
    artifact: LoadedArtifact,
    reader: ArtifactReader,
    candidate: dict[str, str],
) -> tuple[str, dict[str, Any]]:
    document = artifact.document
    _exact_keys(
        document,
        {
            "schema_version",
            "kind",
            "mode",
            "candidate_evidence",
            "host_contract",
            "command",
            "exit_code",
            "stdout",
            "stderr",
            "expected_failure",
            "passed",
        },
        "two-GPU failure run",
    )
    mode = document["mode"]
    _require(
        _is_exact_int(document["schema_version"], 1)
        and document["kind"] == FAILURE_RUN_KIND
        and mode in FAILURE_MODES
        and document["candidate_evidence"] == candidate,
        "two-GPU failure wrapper contract differs",
    )
    host_contract = _validate_host_contract(
        document["host_contract"],
        f"two-GPU failure {mode} host contract",
    )
    expected_failure = document["expected_failure"]
    _exact_keys(
        expected_failure,
        {"reason_id", "exit_code_contract", "required_tokens"},
        f"two-GPU failure {mode} expected reason",
    )
    _require(
        _type_exact_equal(expected_failure, FAILURE_REASON_CONTRACTS[mode]),
        f"two-GPU failure {mode} reason contract differs",
    )
    expected_command = [
        "uv",
        "run",
        "--no-sync",
        "torchrun",
        "--standalone",
        "--nproc-per-node=2",
        "--module",
        "benchmarks.torch_two_gpu_failures",
        mode,
    ]
    _require(
        document["command"] == expected_command,
        f"two-GPU failure {mode} command is not canonical",
    )
    stdout = reader.load(
        document["stdout"],
        f"two-GPU failure {mode} stdout",
        json_document=False,
        expected_media_types={MEDIA_TYPE_TEXT},
    )
    stderr = reader.load(
        document["stderr"],
        f"two-GPU failure {mode} stderr",
        json_document=False,
        expected_media_types={MEDIA_TYPE_TEXT},
    )
    stdout_text = _decode_log(stdout.raw, f"two-GPU failure {mode} stdout")
    stderr_text = _decode_log(stderr.raw, f"two-GPU failure {mode} stderr")
    exit_code = document["exit_code"]
    _require(type(exit_code) is int, f"two-GPU failure {mode} exit code is malformed")
    required_tokens = expected_failure["required_tokens"]
    if mode == "rank-failure":
        text = (stdout_text + "\n" + stderr_text).lower()
        _require(
            exit_code != 0
            and all(token in text for token in required_tokens)
            and ("childfailederror" in text or "rank" in text),
            "two-GPU rank-local failure did not propagate collectively",
        )
    else:
        _require(exit_code == 0, f"two-GPU failure {mode} runner did not exit cleanly")
        records = []
        for line in stdout_text.splitlines():
            try:
                value = _strict_json_bytes(
                    line.encode(),
                    f"two-GPU failure {mode} stdout",
                )
            except EvidenceError:
                continue
            if isinstance(value, dict) and value.get("mode") == mode:
                records.append(value)
        _require(len(records) == 1, f"two-GPU failure {mode} record closure differs")
        record = records[0]
        _exact_keys(
            record,
            {"mode", "passed", "rank0_error"},
            f"two-GPU failure {mode} observed record",
        )
        error_text = record["rank0_error"]
        _require(
            record["passed"] is True
            and isinstance(error_text, str)
            and bool(error_text)
            and all(token in error_text.lower() for token in required_tokens),
            f"two-GPU failure {mode} expected reason was not observed",
        )
    _require(
        document["passed"] is True,
        f"two-GPU failure {mode} wrapper pass is false",
    )
    return mode, host_contract


def _validate_two_gpu_scope(
    scope: Any,
    reader: ArtifactReader,
    manifest: dict[str, Any],
    candidate: dict[str, str],
) -> dict[str, Any]:
    _exact_keys(
        scope,
        {"performance", "correctness", "failures", "traces"},
        "two-GPU scope",
    )
    performance_artifacts = reader.load_many(
        scope["performance"],
        "two-GPU performance",
        count=4,
    )
    records = []
    environments = []
    profiles = []
    for artifact in performance_artifacts:
        case, environment, case_profiles = _validate_two_gpu_performance(
            artifact,
            reader,
            manifest,
            candidate,
        )
        records.append(case)
        environments.append(environment)
        profiles.append((artifact, case_profiles))
    _require(
        set(records) == set(TWO_GPU_CASES) and len(set(records)) == 4,
        "two-GPU performance case closure differs",
    )
    _require(
        all(environment == environments[0] for environment in environments[1:]),
        "two-GPU performance artifacts came from different environments",
    )
    _require(
        {device.get("index") for device in environments[0]["devices"]} >= {0, 1},
        "two-GPU environment does not contain devices 0 and 1",
    )
    traces = _load_traces(
        reader,
        scope["traces"],
        "two-GPU traces",
        count=8,
        require_nccl=True,
    )
    _bind_two_gpu_profiles(profiles, traces)
    correctness = reader.load_many(
        scope["correctness"],
        "two-GPU correctness",
        count=2,
    )
    correctness_records = [
        _validate_two_gpu_correctness_v3(artifact, reader, candidate)
        for artifact in correctness
    ]
    graph_modes = {mode for mode, _environment in correctness_records}
    _require(
        graph_modes == {False, True},
        "two-GPU correctness must cover eager and graph capture",
    )
    _require(
        all(
            environment == environments[0] for _mode, environment in correctness_records
        ),
        "two-GPU correctness and performance environments differ",
    )
    failures = reader.load_many(
        scope["failures"],
        "two-GPU failures",
        count=4,
    )
    failure_records = [
        _validate_failure_run(artifact, reader, candidate) for artifact in failures
    ]
    modes = {mode for mode, _host_contract in failure_records}
    _require(modes == set(FAILURE_MODES), "two-GPU failure mode closure differs")
    _require(
        all(
            host_contract == environments[0]["host_contract"]
            for _mode, host_contract in failure_records
        ),
        "two-GPU failure and performance hosts differ",
    )
    return {
        "candidate_evidence": candidate,
        "environment": environments[0],
        "performance_cases": sorted(records),
    }


def _validate_job(
    value: Any,
    candidate: dict[str, str],
    label: str,
) -> str:
    _exact_keys(
        value,
        {
            "name",
            "workflow",
            "run_id",
            "run_attempt",
            "job_id",
            "event",
            "head_sha",
            "status",
            "conclusion",
            "html_url",
            "started_at",
            "completed_at",
        },
        label,
    )
    name = value["name"]
    _require(name in REQUIRED_JOBS, f"{label} has an unknown canonical job name")
    expected_workflow = "CI" if name.startswith("Python 3.14 / ") else "CodeQL"
    started = _timestamp(value["started_at"], f"{label} start")
    completed = _timestamp(value["completed_at"], f"{label} completion")
    _require(
        value["workflow"] == expected_workflow
        and type(value["run_id"]) is int
        and value["run_id"] > 0
        and type(value["run_attempt"]) is int
        and value["run_attempt"] > 0
        and type(value["job_id"]) is int
        and value["job_id"] > 0
        and value["event"] == "pull_request"
        and value["head_sha"] == candidate["candidate_git_commit"]
        and value["status"] == "completed"
        and value["conclusion"] == "success"
        and isinstance(value["html_url"], str)
        and value["html_url"].startswith("https://github.com/")
        and started <= completed,
        f"{label} GitHub run metadata is incomplete or failing",
    )
    return name


def _validate_code_scanning_analysis(
    value: Any,
    candidate: dict[str, str],
    pull_request: dict[str, Any],
    job: dict[str, Any],
    label: str,
) -> str:
    _exact_keys(
        value,
        {
            "language",
            "analysis_key",
            "category",
            "commit_sha",
            "ref",
            "environment",
            "created_at",
            "results_count",
            "rules_count",
            "error",
            "warning",
            "url",
            "sarif_id",
            "tool",
        },
        label,
    )
    language = value["language"]
    expected_modes = {"python": "none", "c-cpp": "manual"}
    _require(language in expected_modes, f"{label} language differs")
    environment = value["environment"]
    _exact_keys(environment, {"language", "build-mode"}, f"{label} environment")
    tool = value["tool"]
    _exact_keys(tool, {"name", "version"}, f"{label} tool")
    created = _timestamp(value["created_at"], f"{label} creation")
    started = _timestamp(job["started_at"], f"{label} job start")
    completed = _timestamp(job["completed_at"], f"{label} job completion")
    _require(
        pull_request["head_sha"] == candidate["candidate_git_commit"]
        and value["analysis_key"] == ".github/workflows/codeql.yml:analyze"
        and value["category"] == f"/language:{language}"
        and value["commit_sha"] == pull_request["merge_sha"]
        and value["ref"] == pull_request["merge_ref"]
        and started <= created <= completed
        and environment
        == {"language": language, "build-mode": expected_modes[language]}
        and type(value["results_count"]) is int
        and value["results_count"] == 0
        and type(value["rules_count"]) is int
        and value["rules_count"] > 0
        and value["error"] == ""
        and value["warning"] == ""
        and isinstance(value["url"], str)
        and value["url"].startswith(
            "https://api.github.com/repos/ruddyscent/gmes/code-scanning/analyses/"
        )
        and isinstance(value["sarif_id"], str)
        and bool(value["sarif_id"])
        and tool["name"] == "CodeQL"
        and isinstance(tool["version"], str)
        and bool(tool["version"]),
        f"{label} analysis metadata is incomplete, failing, or cross-run",
    )
    return language


def _validate_pull_request(
    value: Any,
    candidate: dict[str, str],
) -> dict[str, Any]:
    _exact_keys(
        value,
        {"number", "head_sha", "merge_sha", "merge_ref"},
        "macOS pull request",
    )
    _require(
        type(value["number"]) is int
        and value["number"] > 0
        and value["head_sha"] == candidate["candidate_git_commit"]
        and isinstance(value["merge_sha"], str)
        and COMMIT_RE.fullmatch(value["merge_sha"]) is not None
        and value["merge_ref"] == f"refs/pull/{value['number']}/merge",
        "macOS pull request candidate or synthetic merge binding differs",
    )
    return dict(value)


def _validate_actions_artifact(
    value: Any,
    archive: LoadedArtifact,
    candidate: dict[str, str],
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    _exact_keys(
        value,
        {
            "id",
            "name",
            "size_in_bytes",
            "archive_download_url",
            "expired",
            "created_at",
            "updated_at",
            "digest",
            "workflow_run",
        },
        "macOS Actions artifact",
    )
    workflow_run = value["workflow_run"]
    _exact_keys(
        workflow_run,
        {
            "head_branch",
            "head_repository_id",
            "head_sha",
            "id",
            "repository_id",
        },
        "macOS Actions artifact workflow run",
    )
    macos_job = next(job for job in jobs if job["name"] == REQUIRED_JOBS[1])
    started = _timestamp(macos_job["started_at"], "macOS job start")
    completed = _timestamp(macos_job["completed_at"], "macOS job completion")
    created = _timestamp(value["created_at"], "Actions artifact creation")
    updated = _timestamp(value["updated_at"], "Actions artifact update")
    _require(
        type(value["id"]) is int
        and value["id"] > 0
        and value["name"] == f"issue-123-macos-{candidate['candidate_git_commit']}"
        and type(value["size_in_bytes"]) is int
        and value["size_in_bytes"] == len(archive.raw)
        and value["digest"] == f"sha256:{archive.descriptor['sha256']}"
        and value["expired"] is False
        and value["archive_download_url"]
        == (
            "https://api.github.com/repos/ruddyscent/gmes/actions/artifacts/"
            f"{value['id']}/zip"
        )
        and isinstance(workflow_run["head_branch"], str)
        and bool(workflow_run["head_branch"])
        and type(workflow_run["head_repository_id"]) is int
        and workflow_run["head_repository_id"] > 0
        and type(workflow_run["repository_id"]) is int
        and workflow_run["repository_id"] > 0
        and workflow_run["head_repository_id"] == workflow_run["repository_id"]
        and workflow_run["head_sha"] == candidate["candidate_git_commit"]
        and type(workflow_run["id"]) is int
        and workflow_run["id"] > 0
        and workflow_run["id"] == macos_job["run_id"]
        and started <= created <= updated <= completed,
        "macOS Actions artifact bytes, candidate, run, or time binding differs",
    )
    return dict(value)


def _validate_actions_archive(
    archive: LoadedArtifact,
    document: dict[str, Any],
    payloads: list[LoadedArtifact],
) -> list[str]:
    payload_by_path = {payload.descriptor["path"]: payload for payload in payloads}
    _require(
        len(payload_by_path) == 8,
        "macOS Actions archive payload path closure differs",
    )
    expected = {"runtime-index.json", *payload_by_path}
    try:
        with zipfile.ZipFile(archive.path) as zipped:
            members = [item for item in zipped.infolist() if not item.is_dir()]
            names = [item.filename for item in members]
            _require(
                len(names) == 9 and len(set(names)) == 9 and set(names) == expected,
                "macOS Actions archive exact member closure differs",
            )
            _require(
                zipped.testzip() is None,
                "macOS Actions archive CRC check failed",
            )
            for path, payload in payload_by_path.items():
                _require(
                    zipped.read(path) == payload.raw,
                    f"macOS Actions archive payload bytes differ: {path}",
                )
            runtime = _strict_json_bytes(
                zipped.read("runtime-index.json"),
                "macOS archived runtime index",
            )
    except (OSError, zipfile.BadZipFile) as error:
        raise EvidenceError("macOS Actions archive is not a readable ZIP") from error
    _exact_keys(
        runtime,
        {
            "schema_version",
            "kind",
            "candidate_evidence",
            "packages",
            "runtime_checks",
            "passed",
        },
        "macOS archived runtime index",
    )
    _require(
        _is_exact_int(runtime["schema_version"], 1)
        and runtime["kind"] == "issue-123-macos-runtime-evidence"
        and runtime["candidate_evidence"] == document["candidate_evidence"]
        and _type_exact_equal(runtime["packages"], document["packages"])
        and _type_exact_equal(runtime["runtime_checks"], document["runtime_checks"])
        and runtime["passed"] is True,
        "macOS archived runtime index differs from the final index",
    )
    return names


def _validate_package(
    record: Any,
    reader: ArtifactReader,
    label: str,
) -> tuple[str, LoadedArtifact]:
    _exact_keys(record, {"role", "filename", "artifact"}, label)
    role = record["role"]
    _require(role in {"sdist", "wheel-macos-arm64"}, f"{label} role differs")
    artifact = reader.load(
        record["artifact"],
        f"{label} bytes",
        json_document=False,
        expected_media_types={MEDIA_TYPE_GZIP, MEDIA_TYPE_WHEEL},
    )
    _require(record["filename"] == artifact.path.name, f"{label} filename differs")
    if role == "sdist":
        _require(
            artifact.descriptor["media_type"] == MEDIA_TYPE_GZIP,
            "macOS sdist media type differs",
        )
        _require(
            artifact.path.name.endswith(".tar.gz"),
            "macOS sdist filename must end in .tar.gz",
        )
        try:
            with tarfile.open(artifact.path, "r:gz") as archive:
                _require(bool(archive.getmembers()), "macOS sdist is empty")
        except (OSError, tarfile.TarError) as error:
            raise EvidenceError("macOS sdist is not a readable gzip tar") from error
    else:
        _require(
            artifact.descriptor["media_type"] == MEDIA_TYPE_WHEEL,
            "macOS wheel media type differs",
        )
        _require(
            re.search(r"cp314[^/]*-macosx_[^/]*_arm64\.whl\Z", artifact.path.name)
            is not None,
            "macOS wheel is not a CPython 3.14 arm64 wheel",
        )
        try:
            with zipfile.ZipFile(artifact.path) as archive:
                _require(bool(archive.namelist()), "macOS wheel is empty")
                _require(archive.testzip() is None, "macOS wheel CRC check failed")
        except (OSError, zipfile.BadZipFile) as error:
            raise EvidenceError("macOS wheel is not a readable ZIP") from error
    return role, artifact


def _validate_macos_scope(
    scope: Any,
    reader: ArtifactReader,
    candidate: dict[str, str],
) -> dict[str, Any]:
    _exact_keys(scope, {"index", "actions_archive"}, "macOS scope")
    archive = reader.load(
        scope["actions_archive"],
        "macOS Actions archive",
        json_document=False,
        expected_media_types={MEDIA_TYPE_ZIP},
    )
    artifact = reader.load(scope["index"], "macOS evidence index")
    document = artifact.document
    _exact_keys(
        document,
        {
            "schema_version",
            "kind",
            "candidate_evidence",
            "pull_request",
            "jobs",
            "actions_artifact",
            "code_scanning_analyses",
            "packages",
            "runtime_checks",
            "passed",
        },
        "macOS evidence index",
    )
    _require(
        _is_exact_int(document["schema_version"], 1)
        and document["kind"] == MACOS_INDEX_KIND
        and document["candidate_evidence"] == candidate,
        "macOS evidence index contract differs",
    )
    pull_request = _validate_pull_request(document["pull_request"], candidate)
    jobs = document["jobs"]
    _require(isinstance(jobs, list), "macOS job index must be a list")
    job_names = [
        _validate_job(job, candidate, f"macOS job {index}")
        for index, job in enumerate(jobs)
    ]
    _require(
        job_names == list(REQUIRED_JOBS),
        "macOS required CI/CodeQL job closure or order differs",
    )
    _require(
        jobs[0]["run_id"] == jobs[1]["run_id"]
        and jobs[0]["run_attempt"] == jobs[1]["run_attempt"]
        and jobs[2]["run_id"] == jobs[3]["run_id"]
        and jobs[2]["run_attempt"] == jobs[3]["run_attempt"],
        "required CI/CodeQL jobs do not come from one exact run per workflow",
    )
    job_by_language = {
        name.removeprefix("CodeQL / "): job
        for name, job in zip(job_names, jobs, strict=True)
        if name.startswith("CodeQL / ")
    }
    analyses = document["code_scanning_analyses"]
    _require(
        isinstance(analyses, list) and len(analyses) == 2,
        "CodeQL analysis closure differs",
    )
    analysis_languages = [
        _validate_code_scanning_analysis(
            analysis,
            candidate,
            pull_request,
            job_by_language[language],
            f"CodeQL analysis {index}",
        )
        for index, (language, analysis) in enumerate(
            zip(("python", "c-cpp"), analyses, strict=True)
        )
    ]
    _require(
        analysis_languages == ["python", "c-cpp"],
        "CodeQL analysis language order differs",
    )
    packages = document["packages"]
    _require(
        isinstance(packages, list) and len(packages) == 2,
        "macOS package closure differs",
    )
    package_records = [
        _validate_package(record, reader, f"macOS package {index}")
        for index, record in enumerate(packages)
    ]
    _require(
        [role for role, _artifact in package_records] == ["sdist", "wheel-macos-arm64"],
        "macOS package roles or order differ",
    )
    package_digests = {
        role: item.descriptor["sha256"] for role, item in package_records
    }
    payloads = [item for _role, item in package_records]
    runtime_checks = document["runtime_checks"]
    _require(isinstance(runtime_checks, list), "macOS runtime checks must be a list")
    roles = []
    for index, check in enumerate(runtime_checks):
        label = f"macOS runtime check {index}"
        _exact_keys(
            check,
            {
                "role",
                "package_sha256",
                "platform",
                "exit_code",
                "log",
            },
            label,
        )
        role = check["role"]
        _require(role in REQUIRED_RUNTIME_ROLES, f"{label} role differs")
        package_role = "wheel-macos-arm64" if role.startswith("wheel-") else "sdist"
        _require(
            check["package_sha256"] == package_digests[package_role],
            f"{label} package digest differs",
        )
        platform_record = check["platform"]
        _exact_keys(
            platform_record,
            {"system", "machine", "python"},
            f"{label} platform",
        )
        _require(
            platform_record["system"] == "Darwin"
            and platform_record["machine"] == "arm64"
            and isinstance(platform_record["python"], str)
            and re.fullmatch(r"3\.14(?:\.\d+)?", platform_record["python"]) is not None
            and _is_exact_int(check["exit_code"], 0),
            f"{label} runtime platform or exit code differs",
        )
        log = reader.load(
            check["log"],
            f"{label} log",
            json_document=False,
            expected_media_types=(
                {MEDIA_TYPE_TEXT}
                if role in {"wheel-import", "sdist-import"}
                else {MEDIA_TYPE_JSON}
            ),
        )
        _require(bool(log.raw), f"{label} log is empty")
        _decode_log(log.raw, f"{label} log")

        if role not in {"wheel-import", "sdist-import"}:
            suite_log = _strict_json_bytes(log.raw, f"{label} suite log")
            _exact_keys(
                suite_log,
                {
                    "role",
                    "mode",
                    "package_sha256",
                    "platform",
                    "openmp_enabled",
                    "native_fdtd_steps",
                    "passed",
                },
                f"{label} suite log",
            )
            suite_platform = suite_log["platform"]
            _exact_keys(
                suite_platform,
                {"system", "machine", "python"},
                f"{label} suite log platform",
            )
            expected_mode = "default" if "-default-" in role else "serial"
            _require(
                suite_log["role"] == role
                and suite_log["mode"] == expected_mode
                and suite_log["package_sha256"] == check["package_sha256"]
                and suite_platform == platform_record
                and suite_log["openmp_enabled"] is False
                and type(suite_log["native_fdtd_steps"]) is int
                and suite_log["native_fdtd_steps"] == 1
                and suite_log["passed"] is True,
                f"{label} installed-package suite evidence differs",
            )

        payloads.append(log)
        roles.append(role)
    _require(
        roles == list(REQUIRED_RUNTIME_ROLES),
        "macOS runtime role closure or order differs",
    )
    actions = _validate_actions_artifact(
        document["actions_artifact"],
        archive,
        candidate,
        jobs,
    )
    archive_members = _validate_actions_archive(archive, document, payloads)
    _require(document["passed"] is True, "macOS embedded suite pass is false")
    return {
        "candidate_evidence": candidate,
        "pull_request": pull_request,
        "jobs": job_names,
        "actions_artifact_id": actions["id"],
        "actions_archive_sha256": archive.descriptor["sha256"],
        "actions_archive_members": archive_members,
        "code_scanning_analyses": analysis_languages,
        "packages": list(package_digests),
        "runtime_checks": roles,
    }


def _validate_macos_command(
    value: Any,
    role: str,
    package_filename: str,
) -> tuple[str, str]:
    _exact_keys(value, {"argv", "cwd", "environment"}, f"macOS {role} command")
    argv = value["argv"]
    expected_mode = None
    if role.endswith("-default-suite"):
        expected_mode = "default"
    elif role.endswith("-serial-suite"):
        expected_mode = "serial"
    expected_length = 16 if expected_mode is not None else 14
    _require(
        isinstance(argv, list)
        and len(argv) == expected_length
        and all(isinstance(item, str) and bool(item) for item in argv),
        f"macOS {role} argv differs",
    )
    executable = PurePosixPath(argv[0])
    script = PurePosixPath(argv[4])
    repository = PurePosixPath(argv[9])
    forbidden_root = PurePosixPath(argv[11])
    package = PurePosixPath(argv[13])
    cwd = PurePosixPath(value["cwd"]) if isinstance(value["cwd"], str) else None
    fixed = [
        argv[1:4] == ["-I", "-W", "error"],
        argv[5:9] == ["_probe", "--role", role, "--repository"],
        argv[10] == "--forbidden-root",
        argv[12] == "--expected-package",
        expected_mode is None or argv[14:] == ["--mode", expected_mode],
    ]
    _require(
        all(fixed)
        and executable.is_absolute()
        and script.is_absolute()
        and repository.is_absolute()
        and forbidden_root == repository
        and package.is_absolute()
        and package.name == package_filename
        and script == repository / "benchmarks" / "macos_ci_evidence.py"
        and cwd is not None
        and cwd.is_absolute()
        and not cwd.is_relative_to(repository),
        f"macOS {role} command provenance differs",
    )
    environment = value["environment"]
    _exact_keys(
        environment,
        {
            "GMES_ENABLE_OPENMP",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "TORCHINDUCTOR_CACHE_DIR",
        },
        f"macOS {role} command environment",
    )
    cache = environment["TORCHINDUCTOR_CACHE_DIR"]
    expected_openmp = "0" if expected_mode == "serial" else "auto"
    _require(
        environment["GMES_ENABLE_OPENMP"] == expected_openmp
        and all(
            environment[name] == "1"
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        )
        and isinstance(cache, str)
        and PurePosixPath(cache).is_absolute()
        and PurePosixPath(cache).parts[-2:] == ("torchinductor", role)
        and not PurePosixPath(cache).is_relative_to(repository),
        f"macOS {role} command environment differs",
    )
    return argv[0], argv[9]


def _validate_macos_actions_artifact_v2(
    value: Any,
    archive: LoadedArtifact,
    candidate: dict[str, str],
) -> dict[str, Any]:
    _exact_keys(
        value,
        {
            "id",
            "name",
            "size_in_bytes",
            "archive_download_url",
            "expired",
            "created_at",
            "updated_at",
            "digest",
            "workflow_run",
        },
        "macOS Actions artifact",
    )
    workflow_run = value["workflow_run"]
    _exact_keys(
        workflow_run,
        {"head_branch", "head_repository_id", "head_sha", "id", "repository_id"},
        "macOS Actions artifact workflow run",
    )
    created = _timestamp(value["created_at"], "macOS artifact creation")
    updated = _timestamp(value["updated_at"], "macOS artifact update")
    _require(
        type(value["id"]) is int
        and value["id"] > 0
        and value["name"] == f"issue-123-macos-{candidate['candidate_git_commit']}"
        and type(value["size_in_bytes"]) is int
        and value["size_in_bytes"] == len(archive.raw)
        and value["digest"] == f"sha256:{archive.descriptor['sha256']}"
        and value["expired"] is False
        and value["archive_download_url"]
        == (
            "https://api.github.com/repos/ruddyscent/gmes/actions/artifacts/"
            f"{value['id']}/zip"
        )
        and created <= updated
        and isinstance(workflow_run["head_branch"], str)
        and bool(workflow_run["head_branch"])
        and type(workflow_run["head_repository_id"]) is int
        and workflow_run["head_repository_id"] > 0
        and workflow_run["head_repository_id"] == workflow_run["repository_id"]
        and workflow_run["head_sha"] == candidate["candidate_git_commit"]
        and type(workflow_run["id"]) is int
        and workflow_run["id"] > 0,
        "macOS Actions artifact bytes or candidate binding differs",
    )
    return {
        "artifact_id": value["id"],
        "run_id": workflow_run["id"],
        "created_at": value["created_at"],
        "updated_at": value["updated_at"],
    }


def _validate_macos_archive_v2(
    archive: LoadedArtifact,
    document: dict[str, Any],
    payloads: list[LoadedArtifact],
) -> list[str]:
    payload_by_path = {item.descriptor["path"]: item for item in payloads}
    _require(
        len(payloads) == 14 and len(payload_by_path) == 14,
        "macOS Actions archive payload closure differs",
    )
    expected = {"runtime-index.json", *payload_by_path}
    try:
        with zipfile.ZipFile(io.BytesIO(archive.raw)) as zipped:
            members = [item for item in zipped.infolist() if not item.is_dir()]
            names = [item.filename for item in members]
            _require(
                len(names) == 15 and len(set(names)) == 15 and set(names) == expected,
                "macOS Actions archive exact member closure differs",
            )
            _require(zipped.testzip() is None, "macOS Actions archive CRC check failed")
            for path, payload in payload_by_path.items():
                _require(
                    zipped.read(path) == payload.raw,
                    f"macOS Actions archive payload bytes differ: {path}",
                )
            runtime = _strict_json_bytes(
                zipped.read("runtime-index.json"), "macOS archived runtime index"
            )
    except (OSError, zipfile.BadZipFile, KeyError) as error:
        raise EvidenceError("macOS Actions archive is not a readable ZIP") from error
    _exact_keys(
        runtime,
        {
            "schema_version",
            "kind",
            "candidate_evidence",
            "packages",
            "runtime_checks",
            "passed",
        },
        "macOS archived runtime index",
    )
    _require(
        _is_exact_int(runtime["schema_version"], 2)
        and runtime["kind"] == "issue-123-macos-runtime-evidence"
        and runtime["candidate_evidence"] == document["candidate_evidence"]
        and _type_exact_equal(runtime["packages"], document["packages"])
        and _type_exact_equal(runtime["runtime_checks"], document["runtime_checks"])
        and runtime["passed"] is True,
        "macOS archived runtime index differs from the final index",
    )
    return sorted(names)


def _validate_macos_scope(
    scope: Any,
    reader: ArtifactReader,
    candidate: dict[str, str],
) -> dict[str, Any]:
    _exact_keys(scope, {"index", "actions_archive"}, "macOS scope")
    archive = reader.load(
        scope["actions_archive"],
        "macOS Actions archive",
        json_document=False,
        expected_media_types={MEDIA_TYPE_ZIP},
    )
    document = reader.load(scope["index"], "macOS evidence index").document
    _exact_keys(
        document,
        {
            "schema_version",
            "kind",
            "candidate_evidence",
            "actions_artifact",
            "packages",
            "runtime_checks",
            "passed",
        },
        "macOS evidence index",
    )
    _require(
        _is_exact_int(document["schema_version"], 2)
        and document["kind"] == MACOS_INDEX_KIND
        and document["candidate_evidence"] == candidate,
        "macOS evidence index contract differs",
    )
    packages = document["packages"]
    _require(
        isinstance(packages, list) and len(packages) == 2,
        "macOS package closure differs",
    )
    package_records = [
        _validate_package(record, reader, f"macOS package {index}")
        for index, record in enumerate(packages)
    ]
    _require(
        [role for role, _item in package_records] == ["sdist", "wheel-macos-arm64"],
        "macOS package roles or order differ",
    )
    package_by_role = {role: item for role, item in package_records}
    package_names = {role: item.path.name for role, item in package_records}
    payloads = [item for _role, item in package_records]
    checks = document["runtime_checks"]
    _require(
        isinstance(checks, list) and len(checks) == len(REQUIRED_RUNTIME_ROLES),
        "macOS runtime check closure differs",
    )
    executables = set()
    repositories = set()
    platforms = []
    hosts = []
    try:
        from benchmarks import macos_ci_evidence as macos_contract
    except (ImportError, OSError) as error:
        raise EvidenceError("macOS schema validator is unavailable") from error
    for index, (check, role) in enumerate(
        zip(checks, REQUIRED_RUNTIME_ROLES, strict=True)
    ):
        label = f"macOS runtime check {index}"
        _exact_keys(
            check,
            {
                "role",
                "package_sha256",
                "platform",
                "command",
                "exit_code",
                "stdout",
                "stderr",
                "result",
            },
            label,
        )
        package_role = "wheel-macos-arm64" if role.startswith("wheel-") else "sdist"
        package = package_by_role[package_role]
        _require(
            check["role"] == role
            and check["package_sha256"] == package.descriptor["sha256"]
            and _is_exact_int(check["exit_code"], 0),
            f"{label} identity or exit code differs",
        )
        executable, repository = _validate_macos_command(
            check["command"], role, package_names[package_role]
        )
        executables.add(executable)
        repositories.add(repository)
        stdout = reader.load(
            check["stdout"],
            f"{label} stdout",
            expected_media_types={MEDIA_TYPE_JSON},
        )
        stderr = reader.load(
            check["stderr"],
            f"{label} stderr",
            json_document=False,
            expected_media_types={MEDIA_TYPE_TEXT},
        )
        _decode_log(stderr.raw, f"{label} stderr")
        _require(
            _type_exact_equal(stdout.document, check["result"]),
            f"{label} raw stdout and embedded result differ",
        )
        try:
            macos_contract._validate_probe_result(
                stdout.document, role, check["package_sha256"], check["platform"]
            )
        except (ValueError, TypeError, KeyError) as error:
            raise EvidenceError(f"{label} raw result contract differs") from error
        platforms.append(check["platform"])
        hosts.append(stdout.document["host_contract"])
        payloads.extend((stdout, stderr))
    _require(
        len(executables) == 1
        and len(repositories) == 1
        and all(value == platforms[0] for value in platforms[1:])
        and all(value == hosts[0] for value in hosts[1:]),
        "macOS runtime commands or host identities differ across checks",
    )
    actions = _validate_macos_actions_artifact_v2(
        document["actions_artifact"], archive, candidate
    )
    archive_members = _validate_macos_archive_v2(archive, document, payloads)
    recomputed_passed = all(check["result"]["passed"] is True for check in checks)
    _require(
        document["passed"] is recomputed_passed and recomputed_passed,
        "macOS suite pass summary differs",
    )
    return {
        "candidate_evidence": candidate,
        "platform": platforms[0],
        "host_contract": hosts[0],
        "actions_artifact": actions,
        "actions_archive_sha256": archive.descriptor["sha256"],
        "actions_archive_members": archive_members,
        "packages": [role for role, _item in package_records],
        "runtime_checks": list(REQUIRED_RUNTIME_ROLES),
    }


def _validate_operations_scope(
    scope: Any,
    reader: ArtifactReader,
    candidate: dict[str, str],
) -> dict[str, Any]:
    _exact_keys(scope, {"index"}, "operations scope")
    artifact = reader.load(scope["index"], "operations evidence index")
    document = artifact.document
    _require(isinstance(document, dict), "operations evidence index differs")
    records = document.get("responses")
    _require(isinstance(records, dict), "operations response index differs")
    raw_responses = {}
    for role, record in records.items():
        _require(
            isinstance(role, str) and bool(role) and isinstance(record, dict),
            "operations response index differs",
        )
        _exact_keys(record, {"request", "artifact"}, f"operations response {role}")
        loaded = reader.load(
            record["artifact"],
            f"operations raw response {role}",
            expected_media_types={MEDIA_TYPE_JSON},
            require_embedded_candidate=False,
        )
        raw_responses[role] = loaded.document
    try:
        from benchmarks.issue123_operations import evaluate_operations
    except (ImportError, OSError) as error:
        raise EvidenceError("operations schema validator is unavailable") from error
    try:
        return evaluate_operations(document, raw_responses, candidate)
    except (ValueError, TypeError, KeyError) as error:
        raise EvidenceError("raw GitHub operational evidence differs") from error


def _validate_cpu_gpu_contract(
    cpu: dict[str, Any],
    single_gpu: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    _require(
        cpu.get("common_host_identity")
        == single_gpu.get("environment", {}).get("common_host_identity"),
        "CPU and single-GPU evidence were not captured on the same Linux host",
    )
    cpu_by_mode = cpu.get("torch_raw_seconds_per_step")
    native_by_mode = cpu.get("native_raw_seconds_per_step")
    cuda_by_case = single_gpu.get("cuda_raw_seconds_per_step")
    _require(
        isinstance(cpu_by_mode, dict) and set(cpu_by_mode) == {"one", "physical"},
        "CPU/GPU comparison is missing the two CPU thread modes",
    )
    _require(
        isinstance(native_by_mode, dict) and set(native_by_mode) == {"one", "physical"},
        "CPU/GPU comparison is missing native thread modes",
    )
    _require(
        isinstance(cuda_by_case, dict) and set(cuda_by_case) == set(CUDA_CASES),
        "CPU/GPU comparison is missing CUDA raw cases",
    )
    comparisons = {}
    for case in CUDA_CASES[:4]:
        cuda = _raw_values(
            cuda_by_case[case],
            f"CPU/GPU {case} CUDA samples",
            count=manifest["reference"]["performance_repetitions"],
        )
        cpu_modes = {
            mode: _raw_values(
                cpu_by_mode[mode].get(case),
                f"CPU/GPU {case} CPU {mode} samples",
                count=manifest["reference"]["performance_repetitions"],
            )
            for mode in ("one", "physical")
        }
        best_mode = min(cpu_modes, key=lambda mode: median(cpu_modes[mode]))
        best_cpu = cpu_modes[best_mode]
        speedup = median(best_cpu) / median(cuda)
        above_crossover = case.startswith("cpu-large-")
        if above_crossover:
            _require(
                speedup > 1.0,
                f"CPU/GPU {case} does not beat the best same-host CPU result",
            )
        comparisons[case] = {
            "role": (
                "blocking-above-crossover"
                if above_crossover
                else "crossover-diagnostic"
            ),
            "best_cpu_thread_mode": best_mode,
            "best_cpu_seconds_per_step": median(best_cpu),
            "cuda_seconds_per_step": median(cuda),
            "best_cpu_to_cuda_speedup": speedup,
            "passed": speedup > 1.0 if above_crossover else True,
        }

    native = _raw_values(
        native_by_mode["physical"].get("cpu-large-3d"),
        "single-GPU native physical cpu-large-3d samples",
        count=manifest["reference"]["performance_repetitions"],
    )
    target_cuda = _raw_values(
        cuda_by_case.get("cpu-large-3d"),
        "cuda-gates cpu-large-3d CUDA samples",
        count=manifest["reference"]["performance_repetitions"],
    )
    threshold = _finite_float(
        manifest["performance_gates"]["single_gpu"][
            "large_mixed_minimum_native_cpu_speedup"
        ],
        "single-GPU native speedup threshold",
        positive=True,
    )
    target_speedup = median(native) / median(target_cuda)
    _require(
        target_speedup >= threshold,
        "single-GPU large mixed 3-D native CPU speedup gate failed",
    )
    return {
        "cpu_cuda_crossover": comparisons,
        "large_mixed_3d_native_comparison": {
            "native_reference": "physical/cpu-large-3d",
            "native_seconds_per_step": median(native),
            "cuda_reference": "cuda-gates/cpu-large-3d",
            "cuda_seconds_per_step": median(target_cuda),
            "native_cpu_to_cuda_speedup": target_speedup,
            "threshold": threshold,
            "passed": True,
        },
    }


def _empty_scope_result() -> dict[str, Any]:
    return {"satisfied": False, "errors": [], "details": None}


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _read_assembly_source(
    path_value: Any,
    base: Path,
    label: str,
    *,
    max_bytes: int = MAX_ARTIFACT_BYTES,
) -> tuple[Path, bytes]:
    _require(isinstance(path_value, str) and bool(path_value), f"{label} is absent")
    _require("\\" not in path_value and "\x00" not in path_value, f"{label} is invalid")
    supplied = Path(path_value)
    _require(
        all(part not in {".", ".."} for part in supplied.parts),
        f"{label} contains a dot segment",
    )
    path = supplied if supplied.is_absolute() else base / supplied
    return _bounded_regular_file_bytes(path, label, max_bytes=max_bytes)


def _materialize_artifact_references(
    value: Any,
    registry: dict[str, dict[str, Any]],
    used: set[str],
    label: str,
) -> Any:
    if isinstance(value, str):
        _canonical_bundle_path(value, label)
        _require(value in registry, f"{label} names an unregistered payload")
        used.add(value)
        return dict(registry[value])
    if isinstance(value, list):
        return [
            _materialize_artifact_references(item, registry, used, f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        _require(
            all(isinstance(key, str) and bool(key) for key in value),
            f"{label} contains an invalid key",
        )
        return {
            key: _materialize_artifact_references(
                item,
                registry,
                used,
                f"{label}.{key}",
            )
            for key, item in value.items()
        }
    raise EvidenceError(f"{label} must contain only mappings, lists, or payload paths")


def _bind_embedded_descriptors(
    value: Any,
    registry: dict[str, dict[str, Any]],
    candidate: dict[str, str],
    used: set[str],
    label: str,
) -> None:
    if isinstance(value, dict):
        descriptor_keys = {
            "path",
            "sha256",
            "size_bytes",
            "media_type",
            "candidate_evidence",
        }
        if set(value) == descriptor_keys:
            descriptor = _validate_descriptor(value, candidate, label)
            _require(
                registry.get(descriptor["path"]) == descriptor,
                f"{label} embedded descriptor is not registered",
            )
            used.add(descriptor["path"])
            return
        for key, item in value.items():
            _bind_embedded_descriptors(
                item,
                registry,
                candidate,
                used,
                f"{label}.{key}",
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _bind_embedded_descriptors(
                item,
                registry,
                candidate,
                used,
                f"{label}[{index}]",
            )


def assemble_evidence_bundle(
    specification_path: Path | str,
    output_directory: Path | str,
    manifest_path: Path | str = DEFAULT_MANIFEST,
) -> Path:
    """Copy exact evidence bytes into one deterministic relocatable bundle."""

    specification_path, specification_raw = _read_assembly_source(
        str(specification_path),
        Path.cwd(),
        "bundle specification",
        max_bytes=MAX_INDEX_BYTES,
    )
    specification = _strict_json_bytes(
        specification_raw,
        "bundle specification",
        max_bytes=MAX_INDEX_BYTES,
    )
    _exact_keys(
        specification,
        {
            "schema_version",
            "kind",
            "issue",
            "candidate_evidence",
            "payloads",
            "artifacts",
        },
        "bundle specification",
    )
    _require(
        _is_exact_int(specification["schema_version"], BUNDLE_SPEC_SCHEMA_VERSION)
        and specification["kind"] == BUNDLE_SPEC_KIND
        and _is_exact_int(specification["issue"], 123),
        "bundle specification identity differs",
    )
    manifest_path, manifest_raw = _read_assembly_source(
        str(manifest_path),
        Path.cwd(),
        "manifest",
        max_bytes=MAX_MANIFEST_BYTES,
    )
    _strict_json_bytes(manifest_raw, "manifest", max_bytes=MAX_MANIFEST_BYTES)
    candidate = _candidate_evidence(
        specification["candidate_evidence"],
        _sha256(manifest_raw),
    )
    scope_names = {
        "cpu",
        "policy_paired_real",
        "single_gpu",
        "two_gpu",
        "macos",
        "operations",
    }
    _exact_keys(specification["artifacts"], scope_names, "bundle artifact scopes")
    payload_specs = specification["payloads"]
    _require(isinstance(payload_specs, list), "bundle payloads must be a list")
    source_base = specification_path.resolve().parent
    prepared: list[tuple[dict[str, Any], bytes]] = []
    registry: dict[str, dict[str, Any]] = {}
    for index, source in enumerate(payload_specs):
        label = f"bundle payload[{index}]"
        _exact_keys(source, {"source_path", "bundle_path", "media_type"}, label)
        portable = _canonical_bundle_path(source["bundle_path"], f"{label} path")
        _require(
            source["media_type"] in ALLOWED_MEDIA_TYPES,
            f"{label} media type is unsupported",
        )
        _source_path, raw = _read_assembly_source(
            source["source_path"], source_base, f"{label} source"
        )
        _validate_media_payload(raw, source["media_type"], label)
        descriptor = {
            "path": portable.as_posix(),
            "sha256": _sha256(raw),
            "size_bytes": len(raw),
            "media_type": source["media_type"],
            "candidate_evidence": candidate,
        }
        _require(descriptor["path"] not in registry, f"{label} repeats a bundle path")
        registry[descriptor["path"]] = descriptor
        prepared.append((descriptor, raw))

    manifest_bundle_path = "manifest/native_oracle_workloads.json"
    _require(manifest_bundle_path not in registry, "payloads replace the manifest")
    manifest_descriptor = {
        "path": manifest_bundle_path,
        "sha256": _sha256(manifest_raw),
        "size_bytes": len(manifest_raw),
        "media_type": MEDIA_TYPE_JSON,
        "candidate_evidence": candidate,
    }
    registry[manifest_bundle_path] = manifest_descriptor
    prepared.append((manifest_descriptor, manifest_raw))

    used = {manifest_bundle_path}
    artifacts = _materialize_artifact_references(
        specification["artifacts"],
        registry,
        used,
        "bundle artifacts",
    )
    for descriptor, raw in prepared:
        if descriptor["media_type"] == MEDIA_TYPE_JSON:
            document = _strict_json_bytes(raw, f"payload {descriptor['path']}")
            _bind_embedded_descriptors(
                document,
                registry,
                candidate,
                used,
                f"payload {descriptor['path']}",
            )
    _require(
        used == set(registry),
        f"bundle payload closure differs: unused={sorted(set(registry) - used)!r}",
    )
    payloads = [registry[path] for path in sorted(registry)]
    index = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "kind": INDEX_KIND,
        "issue": 123,
        "bundle": {
            "format": BUNDLE_FORMAT,
            "path_contract": PATH_CONTRACT,
            "artifact_count": len(payloads),
            "artifact_bytes": sum(item["size_bytes"] for item in payloads),
        },
        "candidate_evidence": candidate,
        "manifest": manifest_descriptor,
        "payloads": payloads,
        "artifacts": artifacts,
    }
    output_directory = Path(output_directory)
    _require(
        output_directory.name not in {"", ".", ".."},
        "bundle output directory name is invalid",
    )
    _require(
        not output_directory.exists() and not output_directory.is_symlink(),
        "bundle output directory already exists",
    )
    output_parent = _ensure_directory_without_symlinks(
        output_directory.parent,
        "bundle output parent",
    )
    output_directory = output_parent / output_directory.name
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_directory.name}.", dir=output_parent)
    )
    try:
        for descriptor, raw in sorted(prepared, key=lambda item: item[0]["path"]):
            destination = temporary.joinpath(*PurePosixPath(descriptor["path"]).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
        index_path = temporary / "completion-index.json"
        index_path.write_bytes(_canonical_json_bytes(index))
        _require(
            not output_directory.exists() and not output_directory.is_symlink(),
            "bundle output directory appeared during assembly",
        )
        temporary.rename(output_directory)
    except Exception:
        shutil.rmtree(temporary)
        raise
    return output_directory / "completion-index.json"


def _structured_evidence_error(
    error: Exception,
    *,
    phase: str,
    scope: str | None = None,
) -> dict[str, Any]:
    if isinstance(error, MemoryError):
        code = "evidence-resource-limit"
    elif isinstance(error, OSError):
        code = "evidence-io-error"
    elif isinstance(error, EvidenceError):
        code = "invalid-evidence"
    else:
        code = "evidence-validation-error"
    return {
        "code": code,
        "phase": phase,
        "scope": scope,
        "message": str(error) or type(error).__name__,
    }


def evaluate_completion(
    index_path: Path | str,
    manifest_path: Path | str | None = None,
) -> dict[str, Any]:
    """Evaluate all issue #123 scopes; any ambiguity produces a false result."""

    supplied_index_path = Path(index_path)
    scope_names = (
        "cpu",
        "policy_paired_real",
        "single_gpu",
        "two_gpu",
        "macos",
        "operations",
    )
    output: dict[str, Any] = {
        "schema_version": 2,
        "kind": OUTPUT_KIND,
        "issue": 123,
        "evidence_index": {
            "path": supplied_index_path.name,
            "size_bytes": None,
            "sha256": None,
        },
        "manifest": {
            "path": None,
            "size_bytes": None,
            "sha256": None,
        },
        "candidate_evidence": None,
        "scopes": {name: _empty_scope_result() for name in scope_names},
        "cross_scope_details": {},
        "cross_scope_errors": [],
        "issue_completion_satisfied": False,
    }
    try:
        index_path, index_raw = _bounded_regular_file_bytes(
            supplied_index_path,
            "completion evidence index",
            max_bytes=MAX_INDEX_BYTES,
        )
        index = _strict_json_bytes(
            index_raw,
            "completion evidence index",
            max_bytes=MAX_INDEX_BYTES,
        )
        output["evidence_index"].update(
            path=index_path.name,
            size_bytes=len(index_raw),
            sha256=_sha256(index_raw),
        )
        _exact_keys(
            index,
            {
                "schema_version",
                "kind",
                "issue",
                "bundle",
                "candidate_evidence",
                "manifest",
                "payloads",
                "artifacts",
            },
            "completion evidence index",
        )
        _require(
            _is_exact_int(index["schema_version"], INDEX_SCHEMA_VERSION)
            and index["kind"] == INDEX_KIND
            and _is_exact_int(index["issue"], 123),
            "completion evidence index identity differs",
        )
        bundle = index["bundle"]
        _exact_keys(
            bundle,
            {"format", "path_contract", "artifact_count", "artifact_bytes"},
            "completion evidence bundle contract",
        )
        payloads = index["payloads"]
        _require(isinstance(payloads, list), "payload registry must be a list")
        _require(
            bundle["format"] == BUNDLE_FORMAT
            and bundle["path_contract"] == PATH_CONTRACT
            and _is_exact_int(bundle["artifact_count"], len(payloads))
            and type(bundle["artifact_bytes"]) is int
            and bundle["artifact_bytes"] >= 0,
            "completion evidence bundle contract differs",
        )
        candidate_value = index["candidate_evidence"]
        _require(
            isinstance(candidate_value, dict), "candidate evidence must be an object"
        )
        candidate = _candidate_evidence(
            candidate_value,
            candidate_value.get("manifest_sha256"),
        )
        reader = ArtifactReader(index_path.parent, candidate, payloads)
        manifest_artifact = reader.load(
            index["manifest"],
            "bundled manifest",
            expected_media_types={MEDIA_TYPE_JSON},
        )
        candidate = _candidate_evidence(
            candidate,
            manifest_artifact.descriptor["sha256"],
        )
        manifest_raw = manifest_artifact.raw
        manifest = manifest_artifact.document
        _require(isinstance(manifest, dict), "manifest root must be an object")
        if manifest_path is not None:
            _supplied_manifest_path, supplied_manifest_raw = (
                _bounded_regular_file_bytes(
                    Path(manifest_path),
                    "supplied manifest",
                    max_bytes=MAX_MANIFEST_BYTES,
                )
            )
            _require(
                supplied_manifest_raw == manifest_raw,
                "supplied manifest bytes differ from the bundled manifest",
            )
        output["manifest"].update(
            path=manifest_artifact.descriptor["path"],
            size_bytes=len(manifest_raw),
            sha256=_sha256(manifest_raw),
        )
        output["candidate_evidence"] = candidate
        _require(
            sum(item["size_bytes"] for item in payloads) == bundle["artifact_bytes"],
            "bundle artifact byte total differs",
        )
        artifacts = index["artifacts"]
        _exact_keys(artifacts, set(scope_names), "completion evidence scopes")
    except Exception as error:
        output["cross_scope_errors"].append(
            _structured_evidence_error(error, phase="bundle-index")
        )
        return output

    validators = {
        "cpu": lambda: _validate_cpu_scope(
            artifacts["cpu"],
            reader,
            manifest,
            candidate,
            manifest_artifact.path,
        ),
        "policy_paired_real": lambda: _validate_policy_scope(
            artifacts["policy_paired_real"], reader, manifest, candidate
        ),
        "single_gpu": lambda: _validate_single_gpu_scope(
            artifacts["single_gpu"], reader, manifest, candidate
        ),
        "two_gpu": lambda: _validate_two_gpu_scope(
            artifacts["two_gpu"], reader, manifest, candidate
        ),
        "macos": lambda: _validate_macos_scope(artifacts["macos"], reader, candidate),
        "operations": lambda: _validate_operations_scope(
            artifacts["operations"], reader, candidate
        ),
    }
    for name in scope_names:
        try:
            details = validators[name]()
        except Exception as error:
            output["scopes"][name]["errors"].append(
                _structured_evidence_error(error, phase="scope-validation", scope=name)
            )
        else:
            output["scopes"][name].update(satisfied=True, details=details)

    cpu = output["scopes"]["cpu"]
    policy = output["scopes"]["policy_paired_real"]
    single = output["scopes"]["single_gpu"]
    two = output["scopes"]["two_gpu"]
    macos = output["scopes"]["macos"]
    operations = output["scopes"]["operations"]
    if macos["satisfied"] and operations["satisfied"]:
        try:
            actions = macos["details"]["actions_artifact"]
            job = operations["details"]["macos_job"]
            _require(
                actions["run_id"] == job["run_id"]
                and _timestamp(job["started_at"], "macOS operations job start")
                <= _timestamp(actions["created_at"], "macOS artifact creation")
                <= _timestamp(actions["updated_at"], "macOS artifact update")
                <= _timestamp(job["completed_at"], "macOS operations job completion"),
                "macOS archive is not bound to the exact successful raw CI job",
            )
        except Exception as error:
            record = _structured_evidence_error(
                error,
                phase="macos-operations-binding",
                scope="operations",
            )
            output["cross_scope_errors"].append(record)
            for scoped in (macos, operations):
                scoped["satisfied"] = False
                scoped["errors"].append(record)
        else:
            output["cross_scope_details"]["macos_operations"] = {
                "ci_run_id": actions["run_id"],
                "artifact_id": actions["artifact_id"],
            }
    if cpu["satisfied"] and single["satisfied"]:
        try:
            comparison = _validate_cpu_gpu_contract(
                cpu["details"],
                single["details"],
                manifest,
            )
        except Exception as error:
            record = _structured_evidence_error(
                error,
                phase="cpu-gpu-comparison",
                scope="single_gpu",
            )
            output["cross_scope_errors"].append(record)
            single["satisfied"] = False
            single["errors"].append(record)
        else:
            output["cross_scope_details"]["cpu_gpu"] = comparison

    linux_scopes = {
        "cpu": cpu,
        "policy_paired_real": policy,
        "single_gpu": single,
        "two_gpu": two,
    }
    if all(scope["satisfied"] for scope in linux_scopes.values()):
        common_identities = {
            name: (
                scope["details"]["common_host_identity"]
                if name == "cpu"
                else scope["details"]["environment"]["common_host_identity"]
            )
            for name, scope in linux_scopes.items()
        }
        gpu_runtime_identities = {
            name: linux_scopes[name]["details"]["environment"]["runtime_identity"]
            for name in ("policy_paired_real", "single_gpu", "two_gpu")
        }
        candidate_bindings = {
            name: scope["details"]["candidate_evidence"]
            for name, scope in linux_scopes.items()
        }
        if (
            len({_canonical_sha256(value) for value in common_identities.values()}) != 1
            or len(
                {_canonical_sha256(value) for value in gpu_runtime_identities.values()}
            )
            != 1
            or any(value != candidate for value in candidate_bindings.values())
        ):
            record = _structured_evidence_error(
                EvidenceError(
                    "Linux scopes do not share one common host, exact CUDA runtime, "
                    "GPU inventory, candidate commit, and manifest"
                ),
                phase="linux-identity",
            )
            output["cross_scope_errors"].append(record)
            for scope in linux_scopes.values():
                scope["satisfied"] = False
                scope["errors"].append(record)
        else:
            output["cross_scope_details"]["linux_environment"] = {
                "common_host_identity": next(iter(common_identities.values())),
                "cpu_runtime_identity": cpu["details"]["runtime_identity"],
                "cuda_runtime_identity": next(iter(gpu_runtime_identities.values())),
            }
    if reader._registry is not None:
        consumed = {path.relative_to(reader.base).as_posix() for path in reader._seen}
        missing = sorted(set(reader._registry) - consumed)
        if missing:
            output["cross_scope_errors"].append(
                _structured_evidence_error(
                    EvidenceError(f"bundle contains unconsumed payloads: {missing!r}"),
                    phase="bundle-closure",
                )
            )
    output["issue_completion_satisfied"] = not output["cross_scope_errors"] and all(
        output["scopes"][name]["satisfied"] for name in scope_names
    )
    return output


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    assemble = commands.add_parser("assemble", help="create a relocatable bundle")
    assemble.add_argument("--specification", "--spec", type=Path, required=True)
    assemble.add_argument("--bundle", type=Path, required=True)
    assemble.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    evaluate = commands.add_parser("evaluate", help="validate a completed bundle")
    evaluate.add_argument("--index", type=Path, required=True)
    evaluate.add_argument("--manifest", type=Path)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument(
        "--enforce",
        action="store_true",
        help="exit 2 unless every issue #123 scope is independently satisfied",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    if args.command == "assemble":
        index_path = assemble_evidence_bundle(
            args.specification,
            args.bundle,
            args.manifest,
        )
        print(index_path)
        return 0
    result = evaluate_completion(args.index, args.manifest)
    rendered = json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        _require(
            output != args.index.resolve(),
            "output must not overwrite the evidence index",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 2 if args.enforce and not result["issue_completion_satisfied"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
