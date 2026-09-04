#!/usr/bin/env python3
"""Build and independently validate privacy-safe issue #123 publications.

The validator in this module deliberately consumes public bytes and a caller-owned
binding only.  It does not import or trust the private evidence projector.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import struct
import sys
import tempfile
import unicodedata
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from statistics import median
from typing import Any

SCHEMA_VERSION = 1
PROJECTION_KIND = "issue-123-publication-projection"
VALIDATION_KIND = "issue-123-publication-validation"
POLICY_KIND = "issue-123-publication-policy"
SCOPE_KIND = "issue-123-public-technical-scope"
COMMITMENTS_KIND = "issue-123-public-correctness-commitments"
EXECUTION_WITNESS_KIND = "issue-123-public-execution-witness"
ARCHIVE_MANIFEST_KIND = "issue-123-public-technical-archive-manifest"
SUMMARY_KIND = "issue-123-technical-summary"
RELEASE_CAPTURE_KIND = "issue-123-public-release-capture"
PUBLICATION_RECEIPT_KIND = "issue-123-publication-receipt"
RAW_TIMING_KIND = "issue-115-raw-timing"
EVENT_PROFILER_KIND = "issue-115-event-level-profiler"

RAW_TIMING_CONTRACT_ID = "torch-utils-benchmark-fixed-workloads"
EVENT_PROFILER_CONTRACT_ID = "event-level-profiler-fixed-workloads"
ARCHIVE_FORMAT = "canonical-zip-stored-v1"
COMMITMENT_ALGORITHM = "HMAC-SHA-256"
SCAN_CONTRACT = "recursive-personal-metadata-v1"
LOCAL_CLOCK = "local-origin-microseconds-v1"
JSON_MEDIA_TYPE = "application/json"
ZIP_MEDIA_TYPE = "application/zip"

TECHNICAL_EVIDENCE_ASSET = "issue-123-public-technical-evidence.zip"
TECHNICAL_SUMMARY_ASSET = "issue-123-technical-summary.json"
RAW_TIMING_ASSET = "issue-115-raw-timing.json"
EVENT_PROFILER_ASSET = "issue-115-event-level-profiler.json"

# This order is part of the completed operations-v2 release contract.  It is
# intentionally different from construction order (raw assets, archive, summary).
ASSET_ORDER = (
    ("technical_evidence", TECHNICAL_EVIDENCE_ASSET),
    ("technical_summary", TECHNICAL_SUMMARY_ASSET),
    ("raw_timing", RAW_TIMING_ASSET),
    ("event_profiler", EVENT_PROFILER_ASSET),
)

# The completed operations-v2 contract requires this exact closure and order.
# These display names intentionally contain spaces and slashes; they are not
# user-controlled filesystem names.
REQUIRED_JOB_NAMES = (
    "Python 3.14 / ubuntu-latest",
    "Python 3.14 / macos-latest",
    "CodeQL / python",
    "CodeQL / c-cpp",
)

TECHNICAL_SCOPE_ORDER = (
    "cpu",
    "policy_paired_real",
    "single_gpu",
    "two_gpu",
    "macos",
)
SCOPE_PATHS = {
    "cpu": "scopes/01-cpu.json",
    "policy_paired_real": "scopes/02-policy-paired-real.json",
    "single_gpu": "scopes/03-single-gpu.json",
    "two_gpu": "scopes/04-two-gpu.json",
    "macos": "scopes/05-macos.json",
}
COMMITMENTS_PATH = "correctness/commitments.json"
EXECUTION_WITNESS_PATH = "execution/witness.json"
MANIFEST_PATH = "manifest.json"
ARCHIVE_PAYLOAD_PATHS = (
    COMMITMENTS_PATH,
    EXECUTION_WITNESS_PATH,
    *(SCOPE_PATHS[scope] for scope in TECHNICAL_SCOPE_ORDER),
)
ARCHIVE_ENTRY_ORDER = tuple(sorted((*ARCHIVE_PAYLOAD_PATHS, MANIFEST_PATH)))

FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
FIXED_ZIP_MODE = stat.S_IFREG | 0o644
MAX_PUBLIC_JSON_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_TRACE_EVENTS = 2_000_000
MAX_TIMING_SAMPLES = 1_000_000
MAX_STRING_BYTES = 4096
MAX_ARRAY_BYTES = 1024 * 1024 * 1024
MAX_PAYLOAD_BYTES = 1024 * 1024 * 1024
MAX_LOCAL_TIMESTAMP_US = 24 * 60 * 60 * 1_000_000
MAX_RELEASE_CAPTURE_BYTES = 1024 * 1024
MAX_PUBLICATION_RECEIPT_BYTES = 4 * 1024 * 1024
MAX_PUBLICATION_TOTAL_BYTES = MAX_ARCHIVE_BYTES + 3 * MAX_PUBLIC_JSON_BYTES

REPOSITORY = "ruddyscent/gmes"
RELEASE_TAG_PREFIX = "issue-123-technical-evidence-"
EXECUTION_CLAIM_ORDER = ("cpu-eager", "cuda-eager", "cuda-graph")
EXECUTION_CLAIM_SCOPES = {
    "cpu-eager": "cpu",
    "cuda-eager": "single_gpu",
    "cuda-graph": "two_gpu",
}
EXECUTION_WORKFLOW = "CI"
EXECUTION_JOB_NAME = "Python 3.14 / ubuntu-latest"

_SHA40_RE = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_NAME_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_SEMANTIC_SEGMENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_UUID_RE = re.compile(
    r"(?i)(?:GPU-)?[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)
_EMAIL_RE = re.compile(r"(?i)(?<![A-Z0-9._%+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}")
_UNIX_PERSONAL_PATH_RE = re.compile(
    r"(?i)(?:file://)?/(?:home|users|root|volumes|private/(?:tmp|var/folders)|"
    r"var/folders|var/lib/jenkins/workspace|builds|tmp|workspace|workspaces|"
    r"github/workspace|runner/_work|home/runner/work|users/runner/work|"
    r"opt/(?:hostedtoolcache|actions-runner))(?:/|\Z)"
)
_WINDOWS_PERSONAL_PATH_RE = re.compile(
    r"(?i)(?:[A-Z]:[\\/](?:Users|a[\\/](?:_work|work)|"
    r"runner[\\/](?:_work|work)|actions-runner[\\/]_work)|\\\\[^\\/]+[\\/])"
)
_SECRET_RE = re.compile(
    r"(?i)(?:-----BEGIN [A-Z ]*(?:PRIVATE|SECRET) KEY-----|"
    r"github_pat_[A-Za-z0-9_]{20,}|gh[opusr]_[A-Za-z0-9]{20,}|"
    r"AKIA[0-9A-Z]{16}|bearer\s+[A-Za-z0-9._~+/=-]{12,}|"
    r"(?:password|passwd|secret|token|credential)\s*[:=])"
)
_SENSITIVE_KEY_RE = re.compile(
    r"(?i)(?:^|_)(?:hostname|host_name|username|user_name|home|homedir|"
    r"worktree|working_directory|cwd|pwd|stack|callstack|traceback|uuid|guid|"
    r"environment|env|password|passwd|secret|tokens?|credentials?|authorization|"
    r"cookie|signing_material|private_key|ssh_key|salt|private_openings)(?:$|_)"
)
_PRIVATE_METADATA_KEY_RE = re.compile(
    r"(?i)(?:hostname|host_name|computername|computer_name|username|user_name|user|login|"
    r"owner|actor|account|machine_id|machine_uuid|home|home_dir|homedir|cwd|pwd|"
    r"worktree|workspace|working_directory|environment|env|credentials?|tokens?|"
    r"passwords?|passwd|secrets?|(?:aws[_ -]?)?secret[_ -]?access[_ -]?keys?|"
    r"api[_ -]?keys?|access[_ -]?keys?(?:[_ -]?id)?|client[_ -]?secrets?|"
    r"authorization|cookie|signing[_ -]?material|private[_ -]?keys?|"
    r"path|cuda_visible_devices|runner_(?:name|os|arch|temp|tool_cache|workspace)|"
    r"github_(?:actor|triggering_actor|workspace|repository|repository_owner|"
    r"run_id|run_attempt|job|workflow|sha|ref|head_ref|base_ref)|"
    r"ssh_auth_sock|shell|tmpdir|temp|tmp|virtual_env|conda_prefix|pythonpath|"
    r"ld_library_path|dyld_library_path|"
    r"device_uuid|deviceuuid|device_id|serial_number)\Z"
)
_PRIVATE_ASSIGNMENT_RE = re.compile(
    r"(?i)(?<![a-z0-9_])(?:hostname|host_name|computername|computer_name|username|user_name|"
    r"user|logname|login|host|owner|actor|account|machine_id|machine_uuid|home|"
    r"home_dir|homedir|cwd|pwd|worktree|workspace|working_directory|"
    r"github_workspace|github_actor|github_triggering_actor|repository_owner|"
    r"runner_name|environment|env|device_uuid|deviceuuid|device_id|"
    r"serial_number)\s*=\s*['\"]?[^\x00\r\n\t '\"]{1,4096}"
)
_PRIVATE_ENV_ASSIGNMENT_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9_])(?:PATH|CUDA_VISIBLE_DEVICES|"
    r"RUNNER_(?:NAME|OS|ARCH|TEMP|TOOL_CACHE|WORKSPACE)|"
    r"GITHUB_(?:ACTOR|TRIGGERING_ACTOR|WORKSPACE|REPOSITORY|REPOSITORY_OWNER|"
    r"RUN_ID|RUN_ATTEMPT|JOB|WORKFLOW|SHA|REF|HEAD_REF|BASE_REF)|"
    r"SSH_AUTH_SOCK|SHELL|TMPDIR|TEMP|TMP|VIRTUAL_ENV|CONDA_PREFIX|PYTHONPATH|"
    r"LD_LIBRARY_PATH|DYLD_LIBRARY_PATH)\s*=\s*"
    r"['\"]?[^\x00\r\n\t '\"]{1,4096}"
)
_WITHHELD_ARRAY_RE = re.compile(
    r"(?i)(?:\.np[yz](?:\Z|[^A-Za-z0-9])|numpy\.lib\.format)"
)
_WITHHELD_PAYLOAD_RE = re.compile(
    r"(?i)(?:^|[-_.])(?:operations?|private|completion|correctness(?:[-_]?arrays?)?)"
    r"(?:[-_.]|$)"
)
_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        "clock$",
        "conin$",
        "conout$",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
)
_WINDOWS_INVALID_PATH_CHARACTERS = frozenset('<>:"|?*')

DTYPES = {
    "bool",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
}
DTYPE_ITEMSIZE = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "int32": 4,
    "uint32": 4,
    "int64": 8,
    "uint64": 8,
    "float16": 2,
    "float32": 4,
    "float64": 8,
    "complex64": 8,
    "complex128": 16,
}
FLOAT_DTYPES = {"float16", "float32", "float64", "complex64", "complex128"}
COMPARISON_CONTRACTS = {"exact", "elementwise", "normalized-linf-l2"}
PAYLOAD_MEDIA_TYPES = {
    "application/octet-stream",
    "application/zip",
    "application/vnd.python.wheel",
    "application/gzip",
    "application/x-tar",
}
PAYLOAD_SUFFIXES = {
    "application/octet-stream": (".bin",),
    "application/zip": (".zip",),
    "application/vnd.python.wheel": (".whl",),
    "application/gzip": (".tar.gz", ".tgz"),
    "application/x-tar": (".tar",),
}
TRACE_KINDS = {
    "metadata",
    "allocation",
    "graph-break",
    "recompile",
    "fallback",
    "halo-annotation",
    "cuda-graph",
    "compiled-region",
    "copy-h2d",
    "copy-d2h",
    "copy-device",
    "memset",
    "kernel",
    "nccl-kernel",
    "indexed-write",
    "cpu-operation",
    "cuda-runtime",
    "profiler-step",
    "correlation-flow",
}
TRACE_PHASES = {
    "complete",
    "instant",
    "counter",
    "metadata",
    "flow-start",
    "flow-step",
    "flow-end",
}
SPECIAL_SEMANTIC_TOKEN_KINDS = {
    "metadata-process-name": "metadata",
    "metadata-process-sort-index": "metadata",
    "metadata-thread-name": "metadata",
    "metadata-thread-sort-index": "metadata",
    "halo-magnetic-pack-launch": "halo-annotation",
    "halo-magnetic-exposed-wait": "halo-annotation",
    "halo-magnetic-boundary-unpack": "halo-annotation",
    "halo-electric-pack-launch": "halo-annotation",
    "halo-electric-exposed-wait": "halo-annotation",
    "halo-electric-boundary-unpack": "halo-annotation",
    "indexed-write-masked-scatter": "indexed-write",
    "indexed-write-index-copy": "indexed-write",
    "indexed-write-scatter": "indexed-write",
}
SEMANTIC_TOKEN_PHASES = {
    **{
        token: {"metadata"}
        for token, kind in SPECIAL_SEMANTIC_TOKEN_KINDS.items()
        if kind == "metadata"
    },
    **{
        token: {"complete"}
        for token, kind in SPECIAL_SEMANTIC_TOKEN_KINDS.items()
        if kind != "metadata"
    },
    "allocation": {"instant", "counter"},
    "graph-break": {"complete", "instant"},
    "recompile": {"complete", "instant"},
    "fallback": {"complete", "instant"},
    "cuda-graph": {"complete"},
    "compiled-region": {"complete"},
    "copy-h2d": {"complete"},
    "copy-d2h": {"complete"},
    "copy-device": {"complete"},
    "memset": {"complete"},
    "nccl-kernel": {"complete"},
    "kernel": {"complete"},
    "cpu-operation": {"complete"},
    "cuda-runtime": {"complete"},
    "profiler-step": {"complete"},
    "correlation-flow": {"flow-start", "flow-step", "flow-end"},
}
TRACE_SUMMARY_FIELDS = {
    "event_count",
    "allocation_events",
    "positive_allocation_events",
    "allocated_bytes",
    "freed_bytes",
    "allocation_net_bytes",
    "live_allocation_baseline_bytes",
    "peak_live_allocated_bytes",
    "final_live_allocated_bytes",
    "live_allocation_growth_bytes",
    "graph_breaks",
    "recompiles",
    "fallbacks",
    "device_copy_events",
    "host_to_device_events",
    "device_to_host_events",
    "kernel_launches",
    "compiled_region_events",
    "cuda_graph_launches",
    "nccl_kernel_launches",
    "nccl_device_us",
    "compute_device_us",
    "nccl_compute_overlap_us",
    "nccl_exposed_us",
    "overlap_fraction",
}
TRACE_EXPECTATION_FIELDS = {
    "name",
    "event_count",
    "semantic_signatures",
    "allocation_events",
    "positive_allocation_events",
    "allocated_bytes",
    "freed_bytes",
    "allocation_net_bytes",
    "live_allocation_baseline_bytes",
    "peak_live_allocated_bytes",
    "final_live_allocated_bytes",
    "live_allocation_growth_bytes",
    "graph_breaks",
    "recompiles",
    "fallbacks",
    "device_copy_events",
    "host_to_device_events",
    "device_to_host_events",
    "kernel_launches",
    "compiled_region_events",
    "cuda_graph_launches",
    "nccl_kernel_launches",
    "require_nccl_overlap",
}


class PublicationError(ValueError):
    """Public bytes do not satisfy the frozen publication contract."""


class PublicationCommitError(PublicationError):
    """A private publication leaf linked but final verification failed."""

    committed = True


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise PublicationError(message)


def _exact_keys(value: Any, expected: set[str], label: str) -> None:
    _require(isinstance(value, dict), f"{label} must be an object")
    actual = set(value)
    _require(actual == expected, f"{label} fields differ")


def _is_int(value: Any) -> bool:
    return type(value) is int


def _is_number(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_sha256(value: Any) -> str:
    _validate_json_tree(value, "digest preimage")
    raw = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _sha256(raw)


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one permitted JSON representation for a public document."""

    _validate_json_tree(value, "public JSON")
    try:
        rendered = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise PublicationError("public JSON is not canonicalizable") from error
    return (rendered + "\n").encode("utf-8")


def _validate_json_tree(value: Any, label: str) -> None:
    nodes = 0

    def visit(item: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        _require(nodes <= 4_000_000, f"{label} is too large")
        _require(depth <= 64, f"{label} is too deeply nested")
        if item is None or type(item) in (bool, int):
            return
        if type(item) is float:
            _require(math.isfinite(item), f"{label} contains a non-finite number")
            return
        if isinstance(item, str):
            _require(
                unicodedata.normalize("NFC", item) == item,
                f"{label} contains non-NFC text",
            )
            try:
                encoded = item.encode("utf-8")
            except UnicodeEncodeError as error:
                raise PublicationError(f"{label} contains invalid Unicode") from error
            _require(
                len(encoded) <= MAX_STRING_BYTES, f"{label} contains oversized text"
            )
            return
        if isinstance(item, list):
            for child in item:
                visit(child, depth + 1)
            return
        if isinstance(item, dict):
            for key, child in item.items():
                _require(isinstance(key, str), f"{label} contains a non-text key")
                visit(key, depth + 1)
                visit(child, depth + 1)
            return
        raise PublicationError(
            f"{label} contains unsupported value {type(item).__name__}"
        )

    visit(value, 0)


def _strict_json_bytes(raw: bytes, label: str) -> Any:
    _require(type(raw) is bytes, f"{label} must be bytes")
    _require(0 < len(raw) <= MAX_PUBLIC_JSON_BYTES, f"{label} byte size differs")
    _require(not raw.startswith(b"\xef\xbb\xbf"), f"{label} has a UTF-8 BOM")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise PublicationError(f"{label} repeats a JSON key")
            result[key] = value
        return result

    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                PublicationError(f"{label} contains a non-finite number")
            ),
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
    ) as error:
        raise PublicationError(f"{label} is not strict UTF-8 JSON") from error
    _validate_json_tree(document, label)
    _require(
        raw == canonical_json_bytes(document), f"{label} JSON bytes are not canonical"
    )
    return document


def _safe_name(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and _SAFE_NAME_RE.fullmatch(value) is not None
        and ".." not in value
        and not value.startswith(("/", "\\")),
        f"{label} is not a safe public name",
    )
    _scan_string(value, label)
    return value


def _validate_bindings(
    value: Any,
    label: str,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _exact_keys(value, {"final_sha", "manifest_sha256", "jobs"}, label)
    _require(
        isinstance(value["final_sha"], str)
        and _SHA40_RE.fullmatch(value["final_sha"]) is not None,
        f"{label} FINAL_SHA differs",
    )
    _require(
        isinstance(value["manifest_sha256"], str)
        and _SHA256_RE.fullmatch(value["manifest_sha256"]) is not None,
        f"{label} manifest SHA-256 differs",
    )
    jobs = value["jobs"]
    _require(
        isinstance(jobs, list) and len(jobs) == len(REQUIRED_JOB_NAMES),
        f"{label} job closure differs",
    )
    names: list[str] = []
    identifiers: list[int] = []
    for index, job in enumerate(jobs):
        job_label = f"{label}.jobs[{index}]"
        _exact_keys(job, {"name", "run_id", "run_attempt", "job_id"}, job_label)
        _require(
            job["name"] == REQUIRED_JOB_NAMES[index],
            f"{job_label} canonical name or order differs",
        )
        names.append(job["name"])
        _scan_string(job["name"], f"{job_label}.name")
        _require(
            _is_int(job["run_id"])
            and job["run_id"] > 0
            and _is_int(job["run_attempt"])
            and job["run_attempt"] > 0
            and _is_int(job["job_id"])
            and job["job_id"] > 0,
            f"{job_label} identifiers must be positive integers",
        )
        identifiers.append(job["job_id"])
    _require(len(names) == len(set(names)), f"{label} repeats a job name")
    _require(len(identifiers) == len(set(identifiers)), f"{label} repeats a job id")
    _require(
        jobs[0]["run_id"] == jobs[1]["run_id"]
        and jobs[0]["run_attempt"] == jobs[1]["run_attempt"]
        and jobs[2]["run_id"] == jobs[3]["run_id"]
        and jobs[2]["run_attempt"] == jobs[3]["run_attempt"]
        and jobs[0]["run_id"] != jobs[2]["run_id"],
        f"{label} jobs do not bind one exact run per workflow",
    )
    if expected is not None:
        normalized_expected = _validate_bindings(dict(expected), "expected bindings")
        _require(
            value == normalized_expected, f"{label} is stale or belongs to another run"
        )
    return value


def publication_bindings(
    final_sha: str,
    manifest_sha256: str,
    jobs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Create and validate the caller-owned immutable publication binding."""

    value = {
        "final_sha": final_sha,
        "manifest_sha256": manifest_sha256,
        "jobs": [dict(job) for job in jobs],
    }
    return _validate_bindings(value, "publication bindings")


def _scan_string_view(value: str, label: str) -> None:
    _require(
        "\n" not in value and "\r" not in value and "\t" not in value,
        f"{label} contains free-form text",
    )
    _require(
        _UUID_RE.search(value) is None, f"{label} contains a device or personal UUID"
    )
    _require(_EMAIL_RE.search(value) is None, f"{label} contains an email identity")
    _require(
        _UNIX_PERSONAL_PATH_RE.search(value) is None, f"{label} contains a host path"
    )
    _require(
        _WINDOWS_PERSONAL_PATH_RE.search(value) is None,
        f"{label} contains a Windows host path",
    )
    _require(
        _SECRET_RE.search(value) is None,
        f"{label} contains credential or signing material",
    )
    _require(
        _PRIVATE_ASSIGNMENT_RE.search(value) is None
        and _PRIVATE_ENV_ASSIGNMENT_RE.search(value) is None,
        f"{label} contains an environment or identity assignment",
    )
    _require(
        _WITHHELD_ARRAY_RE.search(value) is None,
        f"{label} names a withheld array payload",
    )
    _require("://" not in value, f"{label} contains an unapproved URL")


def _scan_string(value: str, label: str) -> None:
    _scan_string_view(value, label)
    normalized = unicodedata.normalize("NFKC", value).casefold()
    _scan_string_view(normalized, f"{label} normalized text")


def _scan_public_tree(value: Any, label: str, *, allow_path_key: bool = False) -> None:
    if isinstance(value, str):
        _scan_string(value, label)
        return
    if isinstance(value, list):
        for item in value:
            _scan_public_tree(item, f"{label} list item", allow_path_key=allow_path_key)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = unicodedata.normalize("NFKC", key).casefold()
            if key != "semantic_token" and not (allow_path_key and key == "path"):
                _require(
                    _SENSITIVE_KEY_RE.search(normalized_key) is None
                    and _PRIVATE_METADATA_KEY_RE.fullmatch(normalized_key) is None,
                    f"{label} contains a forbidden metadata key",
                )
            _scan_string(key, f"{label} object key")
            _scan_public_tree(
                item, f"{label} object value", allow_path_key=allow_path_key
            )


def scan_public_bytes(raw: bytes, label: str = "public bytes") -> None:
    """Reject identifying metadata and withheld arrays in canonical JSON bytes."""

    document = _strict_json_bytes(raw, label)
    _scan_public_tree(document, label)


def _descriptor(path: str, raw: bytes) -> dict[str, Any]:
    return {
        "path": path,
        "media_type": JSON_MEDIA_TYPE,
        "size_bytes": len(raw),
        "sha256": _sha256(raw),
    }


def _asset_descriptor(
    role: str, name: str, media_type: str, raw: bytes
) -> dict[str, Any]:
    return {
        "role": role,
        "name": name,
        "media_type": media_type,
        "size_bytes": len(raw),
        "sha256": _sha256(raw),
    }


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.create_version = 20
    info.extract_version = 20
    info.external_attr = FIXED_ZIP_MODE << 16
    info.internal_attr = 0
    info.flag_bits = 0
    info.extra = b""
    info.comment = b""
    return info


def _encode_archive(entries: Mapping[str, bytes]) -> bytes:
    _require(
        tuple(sorted(entries)) == ARCHIVE_ENTRY_ORDER, "archive entry closure differs"
    )
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", allowZip64=False) as archive:
        archive.comment = b""
        for name in ARCHIVE_ENTRY_ORDER:
            archive.writestr(_zip_info(name), entries[name])
    raw = stream.getvalue()
    _require(len(raw) <= MAX_ARCHIVE_BYTES, "public technical archive is too large")
    return raw


def _canonical_archive_name(name: Any, label: str) -> str:
    _require(isinstance(name, str) and name, f"{label} name is invalid")
    _require(name.isascii(), f"{label} name must be ASCII")
    _require("\\" not in name and "\x00" not in name, f"{label} name is aliased")
    path = PurePosixPath(name)
    _require(
        not path.is_absolute()
        and all(part not in ("", ".", "..") for part in path.parts)
        and path.as_posix() == name
        and not name.endswith("/"),
        f"{label} name is non-canonical or traverses",
    )
    return name


def _validate_eocd(raw: bytes, expected_count: int) -> None:
    _require(len(raw) >= 22, "public technical archive is truncated")
    eocd_offset = len(raw) - 22
    _require(
        raw[eocd_offset : eocd_offset + 4] == b"PK\x05\x06",
        "archive has a comment or trailing bytes",
    )
    (
        signature,
        disk_number,
        central_disk,
        disk_entries,
        total_entries,
        central_size,
        central_offset,
        comment_size,
    ) = struct.unpack_from("<4s4H2LH", raw, eocd_offset)
    _require(signature == b"PK\x05\x06", "archive EOCD signature differs")
    _require(
        disk_number == central_disk == 0
        and disk_entries == total_entries == expected_count
        and comment_size == 0
        and central_offset + central_size == eocd_offset,
        "archive EOCD, disk, or central-directory closure differs",
    )
    _require(
        b"PK\x06\x06" not in raw and b"PK\x06\x07" not in raw, "ZIP64 is not permitted"
    )


def _validate_local_headers(
    raw: bytes, infos: list[zipfile.ZipInfo], central_offset: int
) -> None:
    ordered = sorted(infos, key=lambda info: info.header_offset)
    expected_offset = 0
    for index, info in enumerate(ordered):
        label = f"archive local member {info.filename!r}"
        _require(info.header_offset == expected_offset, f"{label} has a prefix or gap")
        _require(
            raw[expected_offset : expected_offset + 4] == b"PK\x03\x04",
            f"{label} header is invalid",
        )
        try:
            (
                _signature,
                extract_version,
                flag_bits,
                compression,
                modified_time,
                modified_date,
                crc,
                compressed_size,
                file_size,
                name_size,
                extra_size,
            ) = struct.unpack_from("<4s5H3L2H", raw, expected_offset)
        except struct.error as error:
            raise PublicationError(f"{label} header is truncated") from error
        name_start = expected_offset + 30
        name_end = name_start + name_size
        extra_end = name_end + extra_size
        data_end = extra_end + compressed_size
        _require(
            extract_version == 20
            and flag_bits == 0
            and compression == zipfile.ZIP_STORED
            and modified_time == 0
            and modified_date == 33
            and crc == info.CRC
            and compressed_size == info.compress_size == file_size == info.file_size
            and raw[name_start:name_end] == info.filename.encode("ascii")
            and extra_size == 0,
            f"{label} metadata differs",
        )
        next_offset = (
            ordered[index + 1].header_offset
            if index + 1 < len(ordered)
            else central_offset
        )
        _require(data_end == next_offset, f"{label} data boundary differs")
        expected_offset = data_end


def _read_archive_entries(raw: bytes) -> dict[str, bytes]:
    _require(type(raw) is bytes, "public technical archive must be bytes")
    _require(
        0 < len(raw) <= MAX_ARCHIVE_BYTES, "public technical archive byte size differs"
    )
    _validate_eocd(raw, len(ARCHIVE_ENTRY_ORDER))
    try:
        with zipfile.ZipFile(io.BytesIO(raw), "r") as archive:
            _require(archive.comment == b"", "archive comment is not empty")
            infos = archive.infolist()
            names = [info.filename for info in infos]
            canonical = [
                _canonical_archive_name(name, f"archive member[{index}]")
                for index, name in enumerate(names)
            ]
            _require(
                tuple(names) == ARCHIVE_ENTRY_ORDER,
                "archive members are unsorted, missing, or extra",
            )
            _require(len(names) == len(set(names)), "archive repeats a member name")
            _require(
                len(canonical) == len(set(name.casefold() for name in canonical)),
                "archive member names have case aliases",
            )
            for info in infos:
                mode = info.external_attr >> 16
                _require(
                    info.date_time == FIXED_ZIP_TIMESTAMP
                    and info.compress_type == zipfile.ZIP_STORED
                    and info.compress_size == info.file_size
                    and info.create_system == 3
                    and info.create_version == 20
                    and info.extract_version == 20
                    and info.flag_bits == 0
                    and info.extra == b""
                    and info.comment == b""
                    and info.internal_attr == 0
                    and mode == FIXED_ZIP_MODE
                    and stat.S_ISREG(mode),
                    f"archive member {info.filename!r} metadata or compression differs",
                )
                _require(
                    info.file_size <= MAX_PUBLIC_JSON_BYTES,
                    f"archive member {info.filename!r} is too large",
                )
            _validate_local_headers(raw, infos, archive.start_dir)
            _require(archive.testzip() is None, "archive CRC validation failed")
            entries = {info.filename: archive.read(info) for info in infos}
    except (
        OSError,
        EOFError,
        UnicodeDecodeError,
        zipfile.BadZipFile,
        RuntimeError,
    ) as error:
        raise PublicationError("public technical archive is invalid") from error
    _require(
        raw == _encode_archive(entries),
        "archive bytes are not the deterministic canonical encoding",
    )
    return entries


def _validate_file_descriptor(value: Any, label: str) -> dict[str, Any]:
    _exact_keys(value, {"path", "media_type", "size_bytes", "sha256"}, label)
    path = _canonical_archive_name(value["path"], f"{label}.path")
    _require(path != MANIFEST_PATH, f"{label} self-describes the archive manifest")
    _require(value["media_type"] == JSON_MEDIA_TYPE, f"{label} media type differs")
    _require(
        _is_int(value["size_bytes"]) and value["size_bytes"] > 0,
        f"{label} size differs",
    )
    _require(
        isinstance(value["sha256"], str)
        and _SHA256_RE.fullmatch(value["sha256"]) is not None,
        f"{label} digest differs",
    )
    return value


def _archive_manifest(
    bindings: Mapping[str, Any], payloads: Mapping[str, bytes]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": ARCHIVE_MANIFEST_KIND,
        "archive_format": ARCHIVE_FORMAT,
        "bindings": dict(bindings),
        "scope_order": list(TECHNICAL_SCOPE_ORDER),
        "payloads": [_descriptor(path, payloads[path]) for path in sorted(payloads)],
    }


def _number(value: Any, label: str, *, positive: bool = False) -> int | float:
    _require(_is_number(value), f"{label} must be a finite JSON number")
    if positive:
        _require(value > 0, f"{label} must be positive")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    _require(_is_int(value) and value >= 0, f"{label} must be a nonnegative integer")
    return value


def _portable_payload_name(
    value: Any,
    label: str,
    media_type: str | None = None,
) -> str:
    _require(isinstance(value, str) and value, f"{label} must be text")
    _require(
        unicodedata.normalize("NFC", value) == value
        and len(value.encode("utf-8")) <= MAX_STRING_BYTES
        and not any(
            ord(character) < 0x20 or ord(character) == 0x7F for character in value
        ),
        f"{label} is not bounded NFC text",
    )
    _require(
        "\\" not in value and "\x00" not in value,
        f"{label} is not portable",
    )
    path = PurePosixPath(value)
    _require(
        not path.is_absolute()
        and path.as_posix() == value
        and all(part not in ("", ".", "..") for part in path.parts),
        f"{label} is not a canonical relative path",
    )
    alias = unicodedata.normalize("NFKC", value)
    _require(
        alias.count("/") == value.count("/")
        and "\\" not in alias
        and re.match(r"(?i)^[a-z]:", alias) is None,
        f"{label} is not portable",
    )
    alias_path = PurePosixPath(alias)
    _require(
        not alias_path.is_absolute()
        and alias == alias_path.as_posix()
        and all(part not in ("", ".", "..") for part in alias_path.parts),
        f"{label} has a nonportable alias",
    )
    for component in alias.split("/"):
        folded = component.casefold()
        device_stem = folded.split(".", 1)[0].rstrip(" ")
        _require(
            bool(component)
            and component == component.rstrip(" .")
            and not any(
                character in _WINDOWS_INVALID_PATH_CHARACTERS for character in component
            )
            and device_stem not in _WINDOWS_RESERVED_NAMES,
            f"{label} is not portable",
        )
    lowered = alias.casefold()
    _require(
        all(
            _WITHHELD_PAYLOAD_RE.search(part.casefold()) is None
            for part in alias_path.parts
        )
        and not lowered.endswith((".npy", ".npz")),
        f"{label} names withheld private evidence",
    )
    if media_type is not None:
        _require(
            isinstance(media_type, str) and media_type in PAYLOAD_MEDIA_TYPES,
            f"{label} media type differs",
        )
        _require(
            lowered.endswith(PAYLOAD_SUFFIXES[media_type]),
            f"{label} suffix does not match its media type",
        )
    _scan_string(value, label)
    return value


def _payload_alias(name: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", name)
    _require(
        normalized.count("/") == name.count("/") and "\\" not in normalized,
        "payload alias introduces a separator",
    )
    _require(
        all(part == part.rstrip(" .") for part in normalized.split("/")),
        "payload alias has a platform suffix",
    )
    canonical = _portable_payload_name(normalized, "payload alias")
    return tuple(part.casefold() for part in PurePosixPath(canonical).parts)


def _semantic_name(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and 0 < len(value) <= 1024,
        f"{label} is not a semantic name",
    )
    _require(
        unicodedata.normalize("NFC", value) == value
        and "\\" not in value
        and "\x00" not in value,
        f"{label} is not a canonical semantic name",
    )
    parts = value.split("/")
    _require(
        all(
            part not in ("", ".", "..")
            and ".." not in part
            and _SEMANTIC_SEGMENT_RE.fullmatch(part) is not None
            for part in parts
        ),
        f"{label} is not a canonical semantic name",
    )
    _scan_string(value, label)
    return value


def _capture_key(value: Any, label: str) -> str:
    if _is_int(value):
        _require(value >= 0, f"{label} must be nonnegative")
        return str(value)
    return _safe_name(value, label)


def _shape(value: Any, label: str) -> list[int]:
    _require(isinstance(value, list) and len(value) <= 16, f"{label} differs")
    for index, dimension in enumerate(value):
        _nonnegative_int(dimension, f"{label}[{index}]")
    return value


def _element_count(shape: Sequence[int]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _validate_comparison_policy(value: Any, label: str) -> dict[str, Any]:
    _exact_keys(
        value,
        {
            "name",
            "dtype",
            "shape",
            "comparison_contract",
            "rtol",
            "atol",
            "normalized_limit",
        },
        label,
    )
    _semantic_name(value["name"], f"{label}.name")
    _require(
        isinstance(value["dtype"], str) and value["dtype"] in DTYPES,
        f"{label} dtype differs",
    )
    shape = _shape(value["shape"], f"{label}.shape")
    element_count = _element_count(shape)
    _require(
        0 < element_count
        and element_count * DTYPE_ITEMSIZE[value["dtype"]] <= MAX_ARRAY_BYTES,
        f"{label} is empty or too large",
    )
    contract = value["comparison_contract"]
    _require(
        isinstance(contract, str) and contract in COMPARISON_CONTRACTS,
        f"{label} comparison contract differs",
    )
    rtol = _number(value["rtol"], f"{label}.rtol")
    atol = _number(value["atol"], f"{label}.atol")
    _require(rtol >= 0 and atol >= 0, f"{label} tolerances are negative")
    if contract == "normalized-linf-l2":
        _number(value["normalized_limit"], f"{label}.normalized_limit", positive=True)
        _require(rtol == 0 and atol > 0, f"{label} normalized tolerance differs")
    else:
        _require(
            value["normalized_limit"] is None,
            f"{label} has an inapplicable normalized limit",
        )
    if contract == "exact":
        _require(
            rtol == 0 and atol == 0 and value["dtype"] not in FLOAT_DTYPES,
            f"{label} exact tolerance differs",
        )
    return value


def _validate_trace_expectation(value: Any, label: str) -> dict[str, Any]:
    _exact_keys(value, TRACE_EXPECTATION_FIELDS, label)
    _safe_name(value["name"], f"{label}.name")
    semantic_signatures = value["semantic_signatures"]
    _require(
        isinstance(semantic_signatures, list)
        and len(semantic_signatures) == value["event_count"],
        f"{label} semantic inventory closure differs",
    )
    for index, signature in enumerate(semantic_signatures):
        signature_label = f"{label}.semantic_signatures[{index}]"
        _require(
            isinstance(signature, list)
            and len(signature) == 2
            and all(isinstance(item, str) for item in signature),
            f"{signature_label} differs",
        )
        token, phase = signature
        _safe_name(token, f"{signature_label}[0]")
        _semantic_token_kind(token, f"{signature_label}[0]")
        _require(
            phase in SEMANTIC_TOKEN_PHASES[token],
            f"{signature_label} token and phase are incompatible",
        )
    for field in TRACE_EXPECTATION_FIELDS - {
        "name",
        "semantic_signatures",
        "require_nccl_overlap",
    }:
        _nonnegative_int(value[field], f"{label}.{field}")
    _require(value["event_count"] > 0, f"{label} event closure is empty")
    _require(
        value["allocation_events"] >= value["positive_allocation_events"]
        and value["allocated_bytes"] - value["freed_bytes"]
        == value["allocation_net_bytes"]
        and value["final_live_allocated_bytes"]
        - value["live_allocation_baseline_bytes"]
        == value["live_allocation_growth_bytes"]
        and value["peak_live_allocated_bytes"]
        >= max(
            value["live_allocation_baseline_bytes"],
            value["final_live_allocated_bytes"],
        ),
        f"{label} allocation arithmetic differs",
    )
    _require(
        value["allocation_net_bytes"] == 0
        and value["live_allocation_growth_bytes"] == 0,
        f"{label} permits allocation growth",
    )
    for field in (
        "graph_breaks",
        "recompiles",
        "fallbacks",
        "host_to_device_events",
        "device_to_host_events",
    ):
        _require(value[field] == 0, f"{label} permits {field}")
    _require(
        type(value["require_nccl_overlap"]) is bool, f"{label} overlap policy differs"
    )
    if value["require_nccl_overlap"]:
        _require(
            value["nccl_kernel_launches"] > 0, f"{label} requires no NCCL launches"
        )
    return value


def _validate_execution_witness_policy(
    value: Any,
    scopes: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, Any],
) -> list[dict[str, Any]]:
    _require(isinstance(value, list), "execution witness policy must be a list")
    _require(
        [item.get("claim") if isinstance(item, dict) else None for item in value]
        == list(EXECUTION_CLAIM_ORDER),
        "execution witness claim order differs",
    )
    job_names = [job["name"] for job in bindings["jobs"]]
    validated: list[dict[str, Any]] = []
    traces_seen: set[tuple[str, str]] = set()
    for index, item in enumerate(value):
        label = f"execution witness policy[{index}]"
        _exact_keys(
            item,
            {
                "claim",
                "scope",
                "trace_name",
                "validation_workflow",
                "validator_job_name",
            },
            label,
        )
        claim = item["claim"]
        _require(
            item["scope"] == EXECUTION_CLAIM_SCOPES[claim],
            f"{label} scope differs",
        )
        trace_name = _safe_name(item["trace_name"], f"{label}.trace_name")
        _require(
            item["validation_workflow"] == EXECUTION_WORKFLOW,
            f"{label} workflow differs",
        )
        _require(
            item["validator_job_name"] == EXECUTION_JOB_NAME
            and item["validator_job_name"] in job_names,
            f"{label} job differs",
        )
        scope = scopes[TECHNICAL_SCOPE_ORDER.index(item["scope"])]
        _require(
            trace_name in [trace["name"] for trace in scope["traces"]],
            f"{label} references an absent trusted trace",
        )
        trace_key = (item["scope"], trace_name)
        _require(
            trace_key not in traces_seen,
            "execution witness policy reuses a trace across claims",
        )
        traces_seen.add(trace_key)
        validated.append(dict(item))
    return validated


def _validate_policy(
    policy: Any,
    expected_bindings: Mapping[str, Any] | None = None,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    _exact_keys(
        policy,
        {
            "schema_version",
            "kind",
            "bindings",
            "scopes",
            "issue115",
            "execution_witnesses",
        },
        "public validation policy",
    )
    _require(
        _is_int(policy["schema_version"])
        and policy["schema_version"] == SCHEMA_VERSION
        and policy["kind"] == POLICY_KIND,
        "public validation policy identity differs",
    )
    bindings = _validate_bindings(
        policy["bindings"],
        "public validation policy bindings",
        expected_bindings,
    )
    scopes = policy["scopes"]
    _require(
        isinstance(scopes, list)
        and [item.get("name") if isinstance(item, dict) else None for item in scopes]
        == list(TECHNICAL_SCOPE_ORDER),
        "public validation policy scope order differs",
    )
    validated: list[dict[str, Any]] = []
    for scope_index, scope in enumerate(scopes):
        scope_name = TECHNICAL_SCOPE_ORDER[scope_index]
        label = f"public validation policy scope {scope_name}"
        _exact_keys(
            scope,
            {"name", "identities", "timings", "traces", "payloads", "correctness"},
            label,
        )
        identities = scope["identities"]
        _require(isinstance(identities, list), f"{label}.identities must be a list")
        for index, name in enumerate(identities):
            _safe_name(name, f"{label}.identities[{index}]")
        _require(len(identities) == len(set(identities)), f"{label} repeats identities")
        timings = scope["timings"]
        _require(isinstance(timings, list), f"{label}.timings must be a list")
        timing_names: set[str] = set()
        for index, timing in enumerate(timings):
            timing_label = f"{label}.timings[{index}]"
            _exact_keys(
                timing,
                {"name", "sample_count", "samples_sha256"},
                timing_label,
            )
            name = _safe_name(timing["name"], f"{timing_label}.name")
            _require(name not in timing_names, f"{label} repeats timings")
            timing_names.add(name)
            _require(
                _is_int(timing["sample_count"])
                and 0 < timing["sample_count"] <= MAX_TIMING_SAMPLES,
                f"{timing_label} sample closure differs",
            )
            _require(
                isinstance(timing["samples_sha256"], str)
                and _SHA256_RE.fullmatch(timing["samples_sha256"]) is not None,
                f"{timing_label} sample digest differs",
            )
        traces = scope["traces"]
        _require(isinstance(traces, list), f"{label}.traces must be a list")
        for index, trace in enumerate(traces):
            _validate_trace_expectation(trace, f"{label}.traces[{index}]")
        _require(
            len({trace["name"] for trace in traces}) == len(traces),
            f"{label} repeats trace names",
        )
        payloads = scope["payloads"]
        _require(isinstance(payloads, list), f"{label}.payloads must be a list")
        aliases: set[tuple[str, ...]] = set()
        for index, payload in enumerate(payloads):
            payload_label = f"{label}.payloads[{index}]"
            _exact_keys(
                payload, {"name", "media_type", "size_bytes", "sha256"}, payload_label
            )
            name = _portable_payload_name(
                payload["name"],
                f"{payload_label}.name",
                payload["media_type"],
            )
            alias = _payload_alias(name)
            _require(alias not in aliases, f"{label} aliases payload names")
            aliases.add(alias)
            _require(
                isinstance(payload["media_type"], str)
                and payload["media_type"] in PAYLOAD_MEDIA_TYPES,
                f"{payload_label} media type differs",
            )
            _nonnegative_int(payload["size_bytes"], f"{payload_label}.size_bytes")
            _require(
                payload["size_bytes"] <= MAX_PAYLOAD_BYTES,
                f"{payload_label} exceeds its byte bound",
            )
            _require(
                isinstance(payload["sha256"], str)
                and _SHA256_RE.fullmatch(payload["sha256"]) is not None,
                f"{payload_label} digest differs",
            )
        cases = scope["correctness"]
        _require(isinstance(cases, list), f"{label}.correctness must be a list")
        case_names: set[str] = set()
        for case_index, case in enumerate(cases):
            case_label = f"{label}.correctness[{case_index}]"
            _exact_keys(case, {"name", "captures"}, case_label)
            case_name = _semantic_name(case["name"], f"{case_label}.name")
            _require(case_name not in case_names, f"{label} repeats a correctness case")
            case_names.add(case_name)
            captures = case["captures"]
            _require(
                isinstance(captures, list) and captures,
                f"{case_label} capture closure is empty",
            )
            capture_names: set[str] = set()
            for capture_index, capture in enumerate(captures):
                capture_label = f"{case_label}.captures[{capture_index}]"
                _exact_keys(capture, {"capture", "arrays"}, capture_label)
                capture_name = _capture_key(
                    capture["capture"], f"{capture_label}.capture"
                )
                _require(
                    capture_name not in capture_names, f"{case_label} repeats a capture"
                )
                capture_names.add(capture_name)
                arrays = capture["arrays"]
                _require(
                    isinstance(arrays, list) and arrays,
                    f"{capture_label} array closure is empty",
                )
                for array_index, array in enumerate(arrays):
                    _validate_comparison_policy(
                        array, f"{capture_label}.arrays[{array_index}]"
                    )
                _require(
                    len({array["name"] for array in arrays}) == len(arrays),
                    f"{capture_label} repeats an array",
                )
        validated.append(scope)

    issue115 = policy["issue115"]
    _exact_keys(issue115, {"timings", "profilers"}, "issue-115 validation policy")
    for role in ("timings", "profilers"):
        refs = issue115[role]
        _require(isinstance(refs, list) and refs, f"issue-115 {role} closure is empty")
        seen: set[tuple[str, str]] = set()
        for index, ref in enumerate(refs):
            _exact_keys(ref, {"scope", "name"}, f"issue-115 {role}[{index}]")
            _require(
                ref["scope"] in TECHNICAL_SCOPE_ORDER, f"issue-115 {role} scope differs"
            )
            _safe_name(ref["name"], f"issue-115 {role} name")
            key = (ref["scope"], ref["name"])
            _require(key not in seen, f"issue-115 {role} repeats a record")
            seen.add(key)
            expected_scope = validated[TECHNICAL_SCOPE_ORDER.index(ref["scope"])]
            available = (
                [timing["name"] for timing in expected_scope["timings"]]
                if role == "timings"
                else [trace["name"] for trace in expected_scope["traces"]]
            )
            _require(
                ref["name"] in available,
                f"issue-115 {role} references an absent record",
            )
    execution_witnesses = _validate_execution_witness_policy(
        policy["execution_witnesses"], validated, bindings
    )
    _scan_public_tree(policy, "public validation policy")
    return bindings, validated, issue115, execution_witnesses


def _validate_timing_record(
    value: Any,
    expected: Mapping[str, Any],
    label: str,
    *,
    expected_scope: str | None = None,
) -> dict[str, Any]:
    fields = {
        "name",
        "unit",
        "samples",
        "sample_count",
        "samples_sha256",
        "median_seconds",
        "mad_seconds",
        "relative_mad",
    }
    if expected_scope is not None:
        fields.add("scope")
    _exact_keys(value, fields, label)
    _require(value["name"] == expected["name"], f"{label} name differs")
    _safe_name(value["name"], f"{label}.name")
    if expected_scope is not None:
        _require(value["scope"] == expected_scope, f"{label} scope differs")
    _require(value["unit"] == "seconds", f"{label} unit differs")
    samples = value["samples"]
    _require(
        isinstance(samples, list) and samples and len(samples) <= MAX_TIMING_SAMPLES,
        f"{label} sample closure differs",
    )
    for index, sample in enumerate(samples):
        _number(sample, f"{label}.samples[{index}]", positive=True)
    observed_median = median(samples)
    observed_mad = median(abs(sample - observed_median) for sample in samples)
    _require(
        _is_int(value["sample_count"])
        and value["sample_count"] == len(samples)
        and value["sample_count"] == expected["sample_count"]
        and value["samples_sha256"] == _canonical_sha256(samples)
        and value["samples_sha256"] == expected["samples_sha256"]
        and _is_number(value["median_seconds"])
        and value["median_seconds"] == observed_median
        and _is_number(value["mad_seconds"])
        and value["mad_seconds"] == observed_mad
        and _is_number(value["relative_mad"])
        and value["relative_mad"] == observed_mad / observed_median,
        f"{label} timing arithmetic differs",
    )
    return value


def _interval_union(
    intervals: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, stop in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)
    return [(start, stop) for start, stop in merged]


def _interval_duration(intervals: Sequence[tuple[float, float]]) -> float:
    return sum(stop - start for start, stop in _interval_union(intervals))


def _intersection_duration(
    first: Sequence[tuple[float, float]],
    second: Sequence[tuple[float, float]],
) -> float:
    left = _interval_union(first)
    right = _interval_union(second)
    total = 0.0
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        start = max(left[left_index][0], right[right_index][0])
        stop = min(left[left_index][1], right[right_index][1])
        total += max(0.0, stop - start)
        if left[left_index][1] < right[right_index][1]:
            left_index += 1
        else:
            right_index += 1
    return total


def _ordinal(
    value: Any,
    seen: list[int],
    label: str,
    *,
    nullable: bool,
) -> int | None:
    if nullable and value is None:
        return None
    _require(_is_int(value) and value >= 0, f"{label} is not a dense ordinal")
    if value not in seen:
        _require(value == len(seen), f"{label} is not first-appearance dense")
        seen.append(value)
    return value


def _semantic_token_kind(token: str, label: str) -> str:
    if token in SPECIAL_SEMANTIC_TOKEN_KINDS:
        return SPECIAL_SEMANTIC_TOKEN_KINDS[token]
    _require(
        token in TRACE_KINDS - {"metadata", "halo-annotation", "indexed-write"},
        f"{label} is outside the closed semantic token taxonomy",
    )
    return token


def _bind_ordinal_context(
    ordinal: int | None,
    contexts: dict[int, int | None],
    context: int | None,
    label: str,
) -> None:
    if ordinal is None:
        return
    if ordinal in contexts:
        _require(contexts[ordinal] == context, f"{label} changes identifier context")
    else:
        contexts[ordinal] = context


def _recompute_trace_summary(
    events: Any,
    published: Any,
    label: str,
    semantic_signatures: Sequence[Sequence[str]],
) -> dict[str, Any]:
    _require(
        isinstance(events, list) and events and len(events) <= MAX_TRACE_EVENTS,
        f"{label} event closure differs",
    )
    _require(
        len(events) == len(semantic_signatures),
        f"{label} event closure differs from trusted semantic inventory",
    )
    _exact_keys(published, TRACE_SUMMARY_FIELDS, f"{label}.summary")
    count_fields = TRACE_SUMMARY_FIELDS - {
        "nccl_device_us",
        "compute_device_us",
        "nccl_compute_overlap_us",
        "nccl_exposed_us",
        "overlap_fraction",
    }
    for field in count_fields:
        _require(
            _is_int(published[field]), f"{label}.summary.{field} must be an integer"
        )
    for field in TRACE_SUMMARY_FIELDS - count_fields:
        _number(published[field], f"{label}.summary.{field}")
    live_by_context: dict[int, int] = {}
    baseline_by_context: dict[int, int] = {}
    peak_aggregate_live = 0
    process_ordinals: list[int] = []
    thread_ordinals: list[int] = []
    stream_ordinals: list[int] = []
    correlation_ordinals: list[int] = []
    allocation_ordinals: list[int] = []
    allocation_context_ordinals: list[int] = []
    graph_ordinals: list[int] = []
    thread_processes: dict[int, int] = {}
    stream_processes: dict[int, int] = {}
    allocation_processes: dict[int, int] = {}
    allocation_streams: dict[int, int | None] = {}
    allocation_context_processes: dict[int, int] = {}
    allocation_identity_contexts: dict[int, int | None] = {}
    graph_processes: dict[int, int] = {}
    graph_streams: dict[int, int | None] = {}
    allocated = freed = positive_allocations = allocation_events = 0
    graph_breaks = recompiles = fallbacks = 0
    copies = h2d = d2h = kernels = compiled = cuda_graphs = 0
    nccl_intervals: list[tuple[float, float]] = []
    compute_intervals: list[tuple[float, float]] = []
    clocked_starts: list[float] = []
    flow_phases: dict[int, list[tuple[str, float]]] = {}
    event_fields = {
        "ordinal",
        "kind",
        "semantic_token",
        "phase",
        "start_us",
        "duration_us",
        "process_ordinal",
        "thread_ordinal",
        "stream_ordinal",
        "correlation_ordinal",
        "allocation_ordinal",
        "allocation_context_ordinal",
        "graph_ordinal",
        "bytes",
        "live_allocated_bytes",
    }
    for index, event in enumerate(events):
        event_label = f"{label}.events[{index}]"
        _exact_keys(event, event_fields, event_label)
        _require(
            event["ordinal"] == index and _is_int(event["ordinal"]),
            f"{event_label} ordinal differs",
        )
        _require(
            isinstance(event["kind"], str) and event["kind"] in TRACE_KINDS,
            f"{event_label} kind is unknown",
        )
        _safe_name(event["semantic_token"], f"{event_label}.semantic_token")
        _require(
            [event["semantic_token"], event["phase"]]
            == list(semantic_signatures[index]),
            f"{event_label} semantic signature differs from trusted inventory",
        )
        _require(
            _semantic_token_kind(
                event["semantic_token"], f"{event_label}.semantic_token"
            )
            == event["kind"],
            f"{event_label} semantic token does not match its event kind",
        )
        _require(
            isinstance(event["phase"], str) and event["phase"] in TRACE_PHASES,
            f"{event_label} phase is unknown",
        )
        start = float(_number(event["start_us"], f"{event_label}.start_us"))
        duration = float(_number(event["duration_us"], f"{event_label}.duration_us"))
        _require(
            0 <= start <= MAX_LOCAL_TIMESTAMP_US
            and 0 <= duration
            and start + duration <= MAX_LOCAL_TIMESTAMP_US,
            f"{event_label} is not a bounded local timestamp",
        )
        if event["phase"] == "metadata":
            _require(
                event["kind"] == "metadata" and start == 0,
                f"{event_label} metadata semantics differ",
            )
        else:
            _require(
                event["kind"] != "metadata", f"{event_label} metadata phase is missing"
            )
            clocked_starts.append(start)
        if event["phase"] != "complete":
            _require(duration == 0, f"{event_label} non-complete event has duration")
        process_ordinal = _ordinal(
            event["process_ordinal"],
            process_ordinals,
            f"{event_label}.process_ordinal",
            nullable=False,
        )
        thread_ordinal = _ordinal(
            event["thread_ordinal"],
            thread_ordinals,
            f"{event_label}.thread_ordinal",
            nullable=False,
        )
        stream_ordinal = _ordinal(
            event["stream_ordinal"],
            stream_ordinals,
            f"{event_label}.stream_ordinal",
            nullable=True,
        )
        correlation_ordinal = _ordinal(
            event["correlation_ordinal"],
            correlation_ordinals,
            f"{event_label}.correlation_ordinal",
            nullable=True,
        )
        allocation_ordinal = _ordinal(
            event["allocation_ordinal"],
            allocation_ordinals,
            f"{event_label}.allocation_ordinal",
            nullable=True,
        )
        allocation_context_ordinal = _ordinal(
            event["allocation_context_ordinal"],
            allocation_context_ordinals,
            f"{event_label}.allocation_context_ordinal",
            nullable=True,
        )
        graph_ordinal = _ordinal(
            event["graph_ordinal"],
            graph_ordinals,
            f"{event_label}.graph_ordinal",
            nullable=True,
        )
        assert process_ordinal is not None and thread_ordinal is not None
        _bind_ordinal_context(
            thread_ordinal,
            thread_processes,
            process_ordinal,
            f"{event_label}.thread_ordinal",
        )
        _bind_ordinal_context(
            stream_ordinal,
            stream_processes,
            process_ordinal,
            f"{event_label}.stream_ordinal",
        )
        _bind_ordinal_context(
            allocation_ordinal,
            allocation_processes,
            process_ordinal,
            f"{event_label}.allocation_ordinal",
        )
        _bind_ordinal_context(
            allocation_context_ordinal,
            allocation_context_processes,
            process_ordinal,
            f"{event_label}.allocation_context_ordinal",
        )
        _bind_ordinal_context(
            allocation_ordinal,
            allocation_identity_contexts,
            allocation_context_ordinal,
            f"{event_label}.allocation_ordinal",
        )
        _bind_ordinal_context(
            graph_ordinal,
            graph_processes,
            process_ordinal,
            f"{event_label}.graph_ordinal",
        )
        _bind_ordinal_context(
            allocation_ordinal,
            allocation_streams,
            stream_ordinal,
            f"{event_label}.allocation_ordinal",
        )
        _bind_ordinal_context(
            graph_ordinal,
            graph_streams,
            stream_ordinal,
            f"{event_label}.graph_ordinal",
        )
        amount = event["bytes"]
        _require(amount is None or _is_int(amount), f"{event_label}.bytes differs")
        live_total = event["live_allocated_bytes"]
        kind = event["kind"]
        if kind == "correlation-flow":
            _require(
                correlation_ordinal is not None,
                f"{event_label} correlation flow has no ordinal",
            )
            flow_phases.setdefault(correlation_ordinal, []).append(
                (event["phase"], start)
            )
        if kind == "allocation":
            _require(
                amount is not None
                and amount != 0
                and allocation_context_ordinal is not None
                and _is_int(live_total)
                and live_total >= 0,
                f"{event_label} allocation delta or live total differs",
            )
            previous = live_by_context.get(
                allocation_context_ordinal, live_total - amount
            )
            if allocation_context_ordinal not in baseline_by_context:
                _require(previous >= 0, f"{event_label} allocation baseline underflows")
                baseline_by_context[allocation_context_ordinal] = previous
            allocation_events += 1
            _require(
                live_total == previous + amount,
                f"{event_label} allocation trajectory differs",
            )
            live_by_context[allocation_context_ordinal] = live_total
            peak_aggregate_live = max(
                peak_aggregate_live, sum(live_by_context.values())
            )
            if amount > 0:
                positive_allocations += 1
                allocated += amount
            else:
                freed -= amount
        else:
            _require(
                live_total is None,
                f"{event_label} non-allocation event exposes a live total",
            )
            _require(
                allocation_ordinal is None and allocation_context_ordinal is None,
                f"{event_label} non-allocation event has an allocation identity or context",
            )
            if kind.startswith("copy-"):
                _require(
                    amount is None or amount >= 0,
                    f"{event_label} copy byte count is negative",
                )
            else:
                _require(
                    amount is None,
                    f"{event_label} exposes an inapplicable byte count",
                )
        if kind == "graph-break":
            graph_breaks += 1
        elif kind == "recompile":
            recompiles += 1
        elif kind == "fallback":
            fallbacks += 1
        elif kind.startswith("copy-"):
            copies += 1
            h2d += kind == "copy-h2d"
            d2h += kind == "copy-d2h"
        elif kind == "compiled-region" and event["phase"] == "complete":
            compiled += 1
        elif kind == "cuda-graph" and event["phase"] == "complete":
            cuda_graphs += 1
        if kind in {"kernel", "nccl-kernel"} and event["phase"] == "complete":
            kernels += 1
            interval = (start, start + duration)
            if kind == "nccl-kernel":
                nccl_intervals.append(interval)
            else:
                compute_intervals.append(interval)
    for correlation_ordinal, phase_records in flow_phases.items():
        phases = [phase for phase, _start in phase_records]
        _require(
            len(phases) >= 2
            and phases[0] == "flow-start"
            and phases[-1] == "flow-end"
            and all(phase == "flow-step" for phase in phases[1:-1]),
            f"{label} correlation flow {correlation_ordinal} topology is incomplete",
        )
        starts = [start for _phase, start in phase_records]
        _require(
            all(left <= right for left, right in zip(starts, starts[1:])),
            f"{label} correlation flow {correlation_ordinal} timestamps decrease",
        )
    _require(
        bool(clocked_starts) and min(clocked_starts) == 0,
        f"{label} clock does not have a local origin",
    )
    nccl_us = _interval_duration(nccl_intervals)
    compute_us = _interval_duration(compute_intervals)
    overlap_us = _intersection_duration(nccl_intervals, compute_intervals)
    observed_baseline = sum(baseline_by_context.values())
    observed_final = sum(live_by_context.values())
    observed_peak = max(observed_baseline, peak_aggregate_live, observed_final)
    observed = {
        "event_count": len(events),
        "allocation_events": allocation_events,
        "positive_allocation_events": positive_allocations,
        "allocated_bytes": allocated,
        "freed_bytes": freed,
        "allocation_net_bytes": allocated - freed,
        "live_allocation_baseline_bytes": observed_baseline,
        "peak_live_allocated_bytes": observed_peak,
        "final_live_allocated_bytes": observed_final,
        "live_allocation_growth_bytes": observed_final - observed_baseline,
        "graph_breaks": graph_breaks,
        "recompiles": recompiles,
        "fallbacks": fallbacks,
        "device_copy_events": copies,
        "host_to_device_events": h2d,
        "device_to_host_events": d2h,
        "kernel_launches": kernels,
        "compiled_region_events": compiled,
        "cuda_graph_launches": cuda_graphs,
        "nccl_kernel_launches": len(nccl_intervals),
        "nccl_device_us": nccl_us,
        "compute_device_us": compute_us,
        "nccl_compute_overlap_us": overlap_us,
        "nccl_exposed_us": max(0.0, nccl_us - overlap_us),
        "overlap_fraction": overlap_us / nccl_us if nccl_us else 0.0,
    }
    _require(published == observed, f"{label} profiler semantic summary differs")
    _require(
        observed["allocation_net_bytes"] == 0
        and observed["live_allocation_growth_bytes"] == 0
        and observed["graph_breaks"] == 0
        and observed["recompiles"] == 0
        and observed["fallbacks"] == 0,
        f"{label} contains a failed allocation or graph semantic invariant",
    )
    return observed


def _validate_trace_record(
    value: Any,
    expectation: Mapping[str, Any],
    label: str,
    *,
    expected_scope: str | None = None,
) -> dict[str, Any]:
    fields = {"name", "clock", "events", "summary"}
    if expected_scope is not None:
        fields.add("scope")
    _exact_keys(value, fields, label)
    _require(value["name"] == expectation["name"], f"{label} name differs")
    if expected_scope is not None:
        _require(value["scope"] == expected_scope, f"{label} scope differs")
    _require(value["clock"] == LOCAL_CLOCK, f"{label} clock differs")
    summary = _recompute_trace_summary(
        value["events"],
        value["summary"],
        label,
        expectation["semantic_signatures"],
    )
    for field in TRACE_EXPECTATION_FIELDS - {
        "name",
        "semantic_signatures",
        "require_nccl_overlap",
    }:
        _require(
            summary[field] == expectation[field],
            f"{label} {field} differs from trusted closure",
        )
    if expectation["require_nccl_overlap"]:
        _require(
            summary["nccl_device_us"] > 0
            and summary["nccl_compute_overlap_us"] > 0
            and 0 < summary["overlap_fraction"] <= 1,
            f"{label} does not prove NCCL/compute overlap",
        )
    return value


def _policy_inventory(scope: Mapping[str, Any]) -> tuple[int, int, list[list[str]]]:
    captures = arrays = 0
    inventory: list[list[str]] = []
    for case in scope["correctness"]:
        for capture in case["captures"]:
            captures += 1
            capture_name = _capture_key(capture["capture"], "policy capture")
            for array in capture["arrays"]:
                arrays += 1
                inventory.append(
                    [scope["name"], case["name"], capture_name, array["name"]]
                )
    return captures, arrays, inventory


def _validate_comparison_result(
    value: Any,
    policy: Mapping[str, Any],
    element_count: int,
    label: str,
) -> dict[str, Any]:
    _exact_keys(
        value,
        {
            "contract",
            "rtol",
            "atol",
            "normalized_limit",
            "max_abs_error",
            "max_allowed_error",
            "max_tolerance_excess",
            "reference_abs_max",
            "reference_l2",
            "error_l2",
            "normalized_linf",
            "normalized_l2",
            "reference_all_zero",
            "zero_reference_exact",
            "passed",
        },
        label,
    )
    _require(isinstance(value["contract"], str), f"{label}.contract must be text")
    _number(value["rtol"], f"{label}.rtol")
    _number(value["atol"], f"{label}.atol")
    if value["normalized_limit"] is not None:
        _number(value["normalized_limit"], f"{label}.normalized_limit", positive=True)
    published_tolerance = {
        "comparison_contract": value["contract"],
        "rtol": value["rtol"],
        "atol": value["atol"],
        "normalized_limit": value["normalized_limit"],
    }
    trusted_tolerance = {
        key: policy[key]
        for key in ("comparison_contract", "rtol", "atol", "normalized_limit")
    }
    _require(
        canonical_json_bytes(published_tolerance)
        == canonical_json_bytes(trusted_tolerance),
        f"{label} differs from the trusted tolerance",
    )
    for field in ("max_abs_error", "reference_abs_max", "reference_l2", "error_l2"):
        _number(value[field], f"{label}.{field}")
        _require(value[field] >= 0, f"{label}.{field} is negative")
    _require(
        value["reference_l2"] >= value["reference_abs_max"]
        and value["error_l2"] >= value["max_abs_error"]
        and (value["reference_abs_max"] == 0) == (value["reference_l2"] == 0)
        and (value["max_abs_error"] == 0) == (value["error_l2"] == 0)
        and value["reference_all_zero"] == (value["reference_abs_max"] == 0),
        f"{label} norm aggregates are inconsistent",
    )
    reference_l2_bound = math.sqrt(element_count) * value["reference_abs_max"]
    error_l2_bound = math.sqrt(element_count) * value["max_abs_error"]
    _require(
        (
            value["reference_l2"] <= reference_l2_bound
            or math.isclose(value["reference_l2"], reference_l2_bound, rel_tol=1e-15)
        )
        and (
            value["error_l2"] <= error_l2_bound
            or math.isclose(value["error_l2"], error_l2_bound, rel_tol=1e-15)
        ),
        f"{label} L2 aggregates exceed their L-infinity bounds",
    )
    _require(
        type(value["reference_all_zero"]) is bool
        and value["zero_reference_exact"] is True
        and value["passed"] is True,
        f"{label} zero-reference or pass gate differs",
    )
    contract = value["contract"]
    if contract == "exact":
        _require(
            value["max_allowed_error"] == 0.0
            and value["max_tolerance_excess"] is None
            and value["normalized_linf"] is None
            and value["normalized_l2"] is None
            and value["max_abs_error"] == 0
            and value["error_l2"] == 0,
            f"{label} exact arithmetic differs",
        )
    elif contract == "elementwise":
        _number(value["max_allowed_error"], f"{label}.max_allowed_error")
        excess = _number(value["max_tolerance_excess"], f"{label}.max_tolerance_excess")
        expected_allowed = value["atol"] + value["rtol"] * value["reference_abs_max"]
        _require(
            value["max_allowed_error"] == expected_allowed
            and excess <= 0
            and value["max_abs_error"] - value["max_allowed_error"] <= excess
            and excess <= value["max_abs_error"] - value["atol"]
            and value["normalized_linf"] is None
            and value["normalized_l2"] is None,
            f"{label} elementwise arithmetic differs",
        )
    else:
        _require(
            value["max_allowed_error"] is None
            and value["max_tolerance_excess"] is None,
            f"{label} normalized bound differs",
        )
        linf = _number(value["normalized_linf"], f"{label}.normalized_linf")
        l2 = _number(value["normalized_l2"], f"{label}.normalized_l2")
        expected_linf = value["max_abs_error"] / max(
            value["reference_abs_max"], value["atol"]
        )
        expected_l2 = value["error_l2"] / max(
            value["reference_l2"], value["atol"] * math.sqrt(element_count)
        )
        _require(
            linf == expected_linf
            and l2 == expected_l2
            and 0 <= linf <= value["normalized_limit"]
            and 0 <= l2 <= value["normalized_limit"],
            f"{label} normalized tolerance failed",
        )
    if value["reference_all_zero"]:
        _require(
            value["max_abs_error"] == 0
            and value["error_l2"] == 0
            and (value["normalized_linf"] in (None, 0, 0.0))
            and (value["normalized_l2"] in (None, 0, 0.0)),
            f"{label} all-zero reference was not exact",
        )
    return value


def _validate_correctness(
    value: Any,
    policy_scopes: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    _exact_keys(
        value,
        {"schema_version", "kind", "bindings", "algorithm", "cases", "closure"},
        "public correctness commitments",
    )
    _require(
        _is_int(value["schema_version"])
        and value["schema_version"] == SCHEMA_VERSION
        and value["kind"] == COMMITMENTS_KIND
        and value["algorithm"] == COMMITMENT_ALGORITHM,
        "public correctness identity differs",
    )
    _validate_bindings(value["bindings"], "public correctness bindings", bindings)
    cases = value["cases"]
    expected_cases = [
        (scope, case) for scope in policy_scopes for case in scope["correctness"]
    ]
    _require(
        isinstance(cases, list) and len(cases) == len(expected_cases),
        "public correctness case closure differs from trusted policy",
    )
    inventory: list[list[str]] = []
    capture_count = array_count = 0
    for case_index, (case, (policy_scope, policy_case)) in enumerate(
        zip(cases, expected_cases, strict=True)
    ):
        case_label = f"public correctness cases[{case_index}]"
        _exact_keys(case, {"scope", "name", "captures"}, case_label)
        _require(
            case["scope"] == policy_scope["name"]
            and case["name"] == policy_case["name"],
            f"{case_label} identity or order differs",
        )
        captures = case["captures"]
        expected_captures = policy_case["captures"]
        _require(
            isinstance(captures, list) and len(captures) == len(expected_captures),
            f"{case_label} capture closure differs",
        )
        for capture_index, (capture, policy_capture) in enumerate(
            zip(captures, expected_captures, strict=True)
        ):
            capture_label = f"{case_label}.captures[{capture_index}]"
            _exact_keys(capture, {"capture", "arrays"}, capture_label)
            capture_name = _capture_key(
                policy_capture["capture"], f"{capture_label}.capture"
            )
            actual_capture_name = _capture_key(
                capture["capture"], f"{capture_label}.capture"
            )
            _require(
                actual_capture_name == capture_name
                and canonical_json_bytes(capture["capture"])
                == canonical_json_bytes(policy_capture["capture"]),
                f"{capture_label} identity or order differs",
            )
            capture_count += 1
            arrays = capture["arrays"]
            expected_arrays = policy_capture["arrays"]
            _require(
                isinstance(arrays, list) and len(arrays) == len(expected_arrays),
                f"{capture_label} array closure differs",
            )
            for array_index, (array, policy_array) in enumerate(
                zip(arrays, expected_arrays, strict=True)
            ):
                array_label = f"{capture_label}.arrays[{array_index}]"
                _exact_keys(
                    array,
                    {
                        "name",
                        "dtype",
                        "shape",
                        "element_count",
                        "comparison",
                        "commitments",
                    },
                    array_label,
                )
                actual_shape = _shape(array["shape"], f"{array_label}.shape")
                _require(
                    array["name"] == policy_array["name"]
                    and array["dtype"] == policy_array["dtype"]
                    and canonical_json_bytes(actual_shape)
                    == canonical_json_bytes(policy_array["shape"])
                    and _is_int(array["element_count"])
                    and array["element_count"] > 0
                    and array["element_count"] == _element_count(actual_shape),
                    f"{array_label} descriptor differs from trusted policy",
                )
                _validate_comparison_result(
                    array["comparison"],
                    policy_array,
                    array["element_count"],
                    f"{array_label}.comparison",
                )
                commitments = array["commitments"]
                _exact_keys(
                    commitments,
                    {"algorithm", "reference", "candidate"},
                    f"{array_label}.commitments",
                )
                _require(
                    commitments["algorithm"] == COMMITMENT_ALGORITHM
                    and isinstance(commitments["reference"], str)
                    and _SHA256_RE.fullmatch(commitments["reference"]) is not None
                    and isinstance(commitments["candidate"], str)
                    and _SHA256_RE.fullmatch(commitments["candidate"]) is not None
                    and commitments["reference"] != commitments["candidate"],
                    f"{array_label} commitment differs",
                )
                array_count += 1
                inventory.append(
                    [
                        policy_scope["name"],
                        policy_case["name"],
                        capture_name,
                        policy_array["name"],
                    ]
                )
    closure = value["closure"]
    _exact_keys(
        closure,
        {
            "scope_order",
            "case_count",
            "capture_count",
            "array_count",
            "inventory_sha256",
        },
        "public correctness closure",
    )
    _require(
        closure["scope_order"] == list(TECHNICAL_SCOPE_ORDER)
        and _is_int(closure["case_count"])
        and closure["case_count"] == len(cases)
        and _is_int(closure["capture_count"])
        and closure["capture_count"] == capture_count
        and _is_int(closure["array_count"])
        and closure["array_count"] == array_count
        and closure["inventory_sha256"] == _canonical_sha256(inventory),
        "public correctness closure arithmetic differs",
    )
    _scan_public_tree(value, "public correctness commitments")
    return value


def _validate_scope(
    value: Any,
    policy_scope: Mapping[str, Any],
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    scope_name = policy_scope["name"]
    label = f"public technical scope {scope_name}"
    _exact_keys(
        value,
        {
            "schema_version",
            "kind",
            "scope",
            "bindings",
            "identity_commitments",
            "timings",
            "traces",
            "payloads",
            "closure",
        },
        label,
    )
    _require(
        _is_int(value["schema_version"])
        and value["schema_version"] == SCHEMA_VERSION
        and value["kind"] == SCOPE_KIND
        and value["scope"] == scope_name,
        f"{label} identity differs",
    )
    _validate_bindings(value["bindings"], f"{label} bindings", bindings)
    identities = value["identity_commitments"]
    expected_identities = policy_scope["identities"]
    _require(
        isinstance(identities, list) and len(identities) == len(expected_identities),
        f"{label} identity commitment closure differs",
    )
    commitments_seen: set[str] = set()
    for index, (identity, expected_name) in enumerate(
        zip(identities, expected_identities, strict=True)
    ):
        identity_label = f"{label}.identity_commitments[{index}]"
        _exact_keys(identity, {"name", "algorithm", "commitment"}, identity_label)
        _require(
            identity["name"] == expected_name
            and identity["algorithm"] == COMMITMENT_ALGORITHM
            and isinstance(identity["commitment"], str)
            and _SHA256_RE.fullmatch(identity["commitment"]) is not None
            and identity["commitment"] not in commitments_seen,
            f"{identity_label} differs",
        )
        commitments_seen.add(identity["commitment"])
    timings = value["timings"]
    _require(
        isinstance(timings, list) and len(timings) == len(policy_scope["timings"]),
        f"{label} timing closure differs",
    )
    for index, (timing, expected_timing) in enumerate(
        zip(timings, policy_scope["timings"], strict=True)
    ):
        _validate_timing_record(
            timing,
            expected_timing,
            f"{label}.timings[{index}]",
        )
    traces = value["traces"]
    _require(
        isinstance(traces, list) and len(traces) == len(policy_scope["traces"]),
        f"{label} trace closure differs",
    )
    for index, (trace, expectation) in enumerate(
        zip(traces, policy_scope["traces"], strict=True)
    ):
        _validate_trace_record(trace, expectation, f"{label}.traces[{index}]")
    payloads = value["payloads"]
    expected_payloads = policy_scope["payloads"]
    _require(
        isinstance(payloads, list) and len(payloads) == len(expected_payloads),
        f"{label} payload closure differs",
    )
    for index, (payload, expected) in enumerate(
        zip(payloads, expected_payloads, strict=True)
    ):
        payload_label = f"{label}.payloads[{index}]"
        _exact_keys(
            payload,
            {"name", "media_type", "size_bytes", "sha256", "scan_contract"},
            payload_label,
        )
        _portable_payload_name(
            payload["name"], f"{payload_label}.name", payload["media_type"]
        )
        _nonnegative_int(payload["size_bytes"], f"{payload_label}.size_bytes")
        _require(
            payload["size_bytes"] <= MAX_PAYLOAD_BYTES
            and isinstance(payload["sha256"], str)
            and _SHA256_RE.fullmatch(payload["sha256"]) is not None,
            f"{payload_label} size or digest differs",
        )
        published_descriptor = {
            key: payload[key] for key in ("name", "media_type", "size_bytes", "sha256")
        }
        _require(
            canonical_json_bytes(published_descriptor) == canonical_json_bytes(expected)
            and payload["scan_contract"] == SCAN_CONTRACT,
            f"{payload_label} differs from trusted scanned payload",
        )
    expected_captures, expected_arrays, _inventory = _policy_inventory(policy_scope)
    closure = value["closure"]
    _exact_keys(
        closure,
        {
            "identity_names",
            "timing_names",
            "trace_names",
            "payload_names",
            "correctness_case_names",
            "correctness_capture_count",
            "correctness_array_count",
        },
        f"{label}.closure",
    )
    _require(
        closure["identity_names"] == list(policy_scope["identities"])
        and closure["timing_names"]
        == [timing["name"] for timing in policy_scope["timings"]]
        and closure["trace_names"]
        == [trace["name"] for trace in policy_scope["traces"]]
        and closure["payload_names"]
        == [payload["name"] for payload in policy_scope["payloads"]]
        and closure["correctness_case_names"]
        == [case["name"] for case in policy_scope["correctness"]]
        and _is_int(closure["correctness_capture_count"])
        and closure["correctness_capture_count"] == expected_captures
        and _is_int(closure["correctness_array_count"])
        and closure["correctness_array_count"] == expected_arrays,
        f"{label} closure differs from trusted policy",
    )
    _scan_public_tree(value, label)
    return value


def _validate_archive_manifest(
    value: Any,
    entries: Mapping[str, bytes],
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    _exact_keys(
        value,
        {
            "schema_version",
            "kind",
            "archive_format",
            "bindings",
            "scope_order",
            "payloads",
        },
        "public archive manifest",
    )
    _require(
        _is_int(value["schema_version"])
        and value["schema_version"] == SCHEMA_VERSION
        and value["kind"] == ARCHIVE_MANIFEST_KIND
        and value["archive_format"] == ARCHIVE_FORMAT
        and value["scope_order"] == list(TECHNICAL_SCOPE_ORDER),
        "public archive manifest identity differs",
    )
    _validate_bindings(value["bindings"], "public archive manifest bindings", bindings)
    descriptors = value["payloads"]
    _require(
        isinstance(descriptors, list)
        and [
            descriptor.get("path") if isinstance(descriptor, dict) else None
            for descriptor in descriptors
        ]
        == list(sorted(ARCHIVE_PAYLOAD_PATHS)),
        "public archive manifest payload order or closure differs",
    )
    for index, descriptor in enumerate(descriptors):
        label = f"public archive manifest payloads[{index}]"
        _validate_file_descriptor(descriptor, label)
        expected = _descriptor(descriptor["path"], entries[descriptor["path"]])
        _require(descriptor == expected, f"{label} does not bind exact member bytes")
    _require(
        all(descriptor["path"] != MANIFEST_PATH for descriptor in descriptors),
        "public archive manifest contains a self-reference",
    )
    _scan_public_tree(value, "public archive manifest", allow_path_key=True)
    return value


def _scope_index(
    scopes: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for scope in scopes:
        for timing in scope["timings"]:
            result[(scope["scope"], "timing", timing["name"])] = timing
        for trace in scope["traces"]:
            result[(scope["scope"], "trace", trace["name"])] = trace
    return result


def _derive_execution_witness(
    witness_policy: Sequence[Mapping[str, Any]],
    scopes: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    traces = _scope_index(scopes)
    jobs = {job["name"]: job for job in bindings["jobs"]}
    claims: list[dict[str, Any]] = []
    for mapping in witness_policy:
        claim = mapping["claim"]
        scope = mapping["scope"]
        trace_name = mapping["trace_name"]
        trace = traces[(scope, "trace", trace_name)]
        signatures = [
            [event["semantic_token"], event["phase"]] for event in trace["events"]
        ]
        tokens = {token for token, _phase in signatures}
        summary = trace["summary"]
        if claim == "cpu-eager":
            _require(
                summary["kernel_launches"] == 0
                and summary["cuda_graph_launches"] == 0
                and summary["compiled_region_events"] == 0
                and bool(
                    tokens
                    & {
                        "cpu-operation",
                        "indexed-write-masked-scatter",
                        "indexed-write-index-copy",
                        "indexed-write-scatter",
                    }
                )
                and not tokens
                & {
                    "kernel",
                    "nccl-kernel",
                    "cuda-graph",
                    "cuda-runtime",
                    "copy-h2d",
                    "copy-d2h",
                    "copy-device",
                    "memset",
                    "compiled-region",
                },
                "cpu-eager witness semantics differ",
            )
        elif claim == "cuda-eager":
            _require(
                summary["kernel_launches"] > 0
                and summary["cuda_graph_launches"] == 0
                and summary["compiled_region_events"] == 0
                and "cuda-runtime" in tokens
                and bool(tokens & {"kernel", "nccl-kernel"})
                and "compiled-region" not in tokens,
                "cuda-eager witness semantics differ",
            )
        else:
            _require(
                summary["kernel_launches"] > 0
                and summary["cuda_graph_launches"] > 0
                and "cuda-graph" in tokens
                and bool(tokens & {"kernel", "nccl-kernel"}),
                "cuda-graph witness semantics differ",
            )
        normalized_trace = {
            "clock": trace["clock"],
            "events": trace["events"],
            "summary": summary,
        }
        claims.append(
            {
                "claim": claim,
                "scope": scope,
                "trace_name": trace_name,
                "validation_workflow": mapping["validation_workflow"],
                "validator_job": dict(jobs[mapping["validator_job_name"]]),
                "event_count": summary["event_count"],
                "semantic_inventory_sha256": _canonical_sha256(signatures),
                "normalized_trace_sha256": _canonical_sha256(normalized_trace),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": EXECUTION_WITNESS_KIND,
        "bindings": dict(bindings),
        "claims": claims,
    }


def _validate_execution_witness(
    value: Any,
    witness_policy: Sequence[Mapping[str, Any]],
    scopes: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    _exact_keys(
        value,
        {"schema_version", "kind", "bindings", "claims"},
        "public execution witness",
    )
    _require(
        _is_int(value["schema_version"])
        and value["schema_version"] == SCHEMA_VERSION
        and value["kind"] == EXECUTION_WITNESS_KIND,
        "public execution witness identity differs",
    )
    _validate_bindings(value["bindings"], "public execution witness bindings", bindings)
    claims = value["claims"]
    _require(
        isinstance(claims, list)
        and [
            claim.get("claim") if isinstance(claim, dict) else None for claim in claims
        ]
        == list(EXECUTION_CLAIM_ORDER),
        "public execution witness claim order or closure differs",
    )
    expected = _derive_execution_witness(witness_policy, scopes, bindings)
    _require(
        canonical_json_bytes(value) == canonical_json_bytes(expected),
        "public execution witness differs from trusted traces",
    )
    for index, claim in enumerate(claims):
        _exact_keys(
            claim,
            {
                "claim",
                "scope",
                "trace_name",
                "validation_workflow",
                "validator_job",
                "event_count",
                "semantic_inventory_sha256",
                "normalized_trace_sha256",
            },
            f"public execution witness claims[{index}]",
        )
        job = claim["validator_job"]
        _exact_keys(
            job,
            {"name", "run_id", "run_attempt", "job_id"},
            f"public execution witness claims[{index}].validator_job",
        )
        _require(
            job["name"] == EXECUTION_JOB_NAME
            and all(
                _is_int(job[field]) and job[field] > 0
                for field in ("run_id", "run_attempt", "job_id")
            ),
            f"public execution witness claims[{index}] validator job differs",
        )
    _scan_public_tree(value, "public execution witness")
    return value


def _validate_issue115_timing(
    value: Any,
    issue115_policy: Mapping[str, Any],
    scopes: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    _exact_keys(
        value,
        {"schema_version", "kind", "contract_id", "bindings", "records", "closure"},
        "issue-115 raw timing",
    )
    _require(
        _is_int(value["schema_version"])
        and value["schema_version"] == SCHEMA_VERSION
        and value["kind"] == RAW_TIMING_KIND
        and value["contract_id"] == RAW_TIMING_CONTRACT_ID,
        "issue-115 raw timing identity differs",
    )
    _validate_bindings(value["bindings"], "issue-115 raw timing bindings", bindings)
    records = value["records"]
    refs = issue115_policy["timings"]
    _require(
        isinstance(records, list) and len(records) == len(refs),
        "issue-115 raw timing record closure differs",
    )
    index = _scope_index(scopes)
    inventory: list[list[str]] = []
    for record_index, (record, ref) in enumerate(zip(records, refs, strict=True)):
        label = f"issue-115 raw timing records[{record_index}]"
        technical_record = index[(ref["scope"], "timing", ref["name"])]
        _validate_timing_record(
            record,
            {
                "name": ref["name"],
                "sample_count": technical_record["sample_count"],
                "samples_sha256": _canonical_sha256(technical_record["samples"]),
            },
            label,
            expected_scope=ref["scope"],
        )
        _require(
            canonical_json_bytes(
                {key: item for key, item in record.items() if key != "scope"}
            )
            == canonical_json_bytes(technical_record),
            f"{label} does not preserve exact technical timing samples",
        )
        inventory.append([ref["scope"], ref["name"]])
    closure = value["closure"]
    _exact_keys(
        closure, {"record_count", "inventory_sha256"}, "issue-115 raw timing closure"
    )
    _require(
        _is_int(closure["record_count"])
        and closure["record_count"] == len(records)
        and closure["inventory_sha256"] == _canonical_sha256(inventory),
        "issue-115 raw timing closure differs",
    )
    _scan_public_tree(value, "issue-115 raw timing")
    return value


def _validate_issue115_profiler(
    value: Any,
    issue115_policy: Mapping[str, Any],
    policy_scopes: Sequence[Mapping[str, Any]],
    scopes: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    _exact_keys(
        value,
        {"schema_version", "kind", "contract_id", "bindings", "records", "closure"},
        "issue-115 event profiler",
    )
    _require(
        _is_int(value["schema_version"])
        and value["schema_version"] == SCHEMA_VERSION
        and value["kind"] == EVENT_PROFILER_KIND
        and value["contract_id"] == EVENT_PROFILER_CONTRACT_ID,
        "issue-115 event profiler identity differs",
    )
    _validate_bindings(value["bindings"], "issue-115 event profiler bindings", bindings)
    records = value["records"]
    refs = issue115_policy["profilers"]
    _require(
        isinstance(records, list) and len(records) == len(refs),
        "issue-115 event profiler record closure differs",
    )
    index = _scope_index(scopes)
    policy_by_scope = {scope["name"]: scope for scope in policy_scopes}
    inventory: list[list[str]] = []
    for record_index, (record, ref) in enumerate(zip(records, refs, strict=True)):
        label = f"issue-115 event profiler records[{record_index}]"
        expectation = next(
            trace
            for trace in policy_by_scope[ref["scope"]]["traces"]
            if trace["name"] == ref["name"]
        )
        _validate_trace_record(
            record,
            expectation,
            label,
            expected_scope=ref["scope"],
        )
        technical_record = index[(ref["scope"], "trace", ref["name"])]
        _require(
            canonical_json_bytes(
                {key: item for key, item in record.items() if key != "scope"}
            )
            == canonical_json_bytes(technical_record),
            f"{label} does not preserve exact normalized events",
        )
        inventory.append([ref["scope"], ref["name"]])
    closure = value["closure"]
    _exact_keys(
        closure,
        {"record_count", "inventory_sha256"},
        "issue-115 event profiler closure",
    )
    _require(
        _is_int(closure["record_count"])
        and closure["record_count"] == len(records)
        and closure["inventory_sha256"] == _canonical_sha256(inventory),
        "issue-115 event profiler closure differs",
    )
    _scan_public_tree(value, "issue-115 event profiler")
    return value


def validate_public_archive(
    archive_bytes: bytes,
    *,
    expected_policy: Mapping[str, Any],
    expected_bindings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Re-open and validate public archive bytes against a caller-owned policy."""

    bindings, policy_scopes, _issue115, witness_policy = _validate_policy(
        expected_policy, expected_bindings
    )
    entries = _read_archive_entries(archive_bytes)
    documents = {
        path: _strict_json_bytes(raw, f"archive member {path!r}")
        for path, raw in entries.items()
    }
    manifest = _validate_archive_manifest(documents[MANIFEST_PATH], entries, bindings)
    scopes = []
    for scope_name, policy_scope in zip(
        TECHNICAL_SCOPE_ORDER, policy_scopes, strict=True
    ):
        scopes.append(
            _validate_scope(documents[SCOPE_PATHS[scope_name]], policy_scope, bindings)
        )
    correctness = _validate_correctness(
        documents[COMMITMENTS_PATH], policy_scopes, bindings
    )
    execution_witness = _validate_execution_witness(
        documents[EXECUTION_WITNESS_PATH], witness_policy, scopes, bindings
    )
    return {
        "bindings": bindings,
        "manifest": manifest,
        "technical_scopes": scopes,
        "correctness_commitments": correctness,
        "execution_witness": execution_witness,
    }


def build_public_archive(
    technical_scopes: Sequence[Mapping[str, Any]],
    correctness_commitments: Mapping[str, Any],
    execution_witness: Mapping[str, Any],
    *,
    expected_policy: Mapping[str, Any],
    expected_bindings: Mapping[str, Any] | None = None,
) -> bytes:
    """Build deterministic ZIP_STORED bytes and independently re-open them."""

    bindings, policy_scopes, _issue115, witness_policy = _validate_policy(
        expected_policy, expected_bindings
    )
    _require(
        isinstance(technical_scopes, Sequence)
        and not isinstance(technical_scopes, (str, bytes, bytearray))
        and len(technical_scopes) == len(TECHNICAL_SCOPE_ORDER),
        "public technical scope closure differs",
    )
    payloads: dict[str, bytes] = {}
    for scope_name, scope, policy_scope in zip(
        TECHNICAL_SCOPE_ORDER,
        technical_scopes,
        policy_scopes,
        strict=True,
    ):
        _validate_scope(scope, policy_scope, bindings)
        payloads[SCOPE_PATHS[scope_name]] = canonical_json_bytes(scope)
    _validate_correctness(correctness_commitments, policy_scopes, bindings)
    payloads[COMMITMENTS_PATH] = canonical_json_bytes(correctness_commitments)
    _validate_execution_witness(
        execution_witness, witness_policy, technical_scopes, bindings
    )
    payloads[EXECUTION_WITNESS_PATH] = canonical_json_bytes(execution_witness)
    manifest = _archive_manifest(bindings, payloads)
    entries = {**payloads, MANIFEST_PATH: canonical_json_bytes(manifest)}
    archive = _encode_archive(entries)
    validate_public_archive(
        archive,
        expected_policy=expected_policy,
        expected_bindings=bindings,
    )
    return archive


def _technical_summary(
    bindings: Mapping[str, Any],
    archive: bytes,
    raw_timing: bytes,
    event_profiler: bytes,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": SUMMARY_KIND,
        "bindings": dict(bindings),
        "scope_order": list(TECHNICAL_SCOPE_ORDER),
        "assets": [
            _asset_descriptor(
                "technical_evidence",
                TECHNICAL_EVIDENCE_ASSET,
                ZIP_MEDIA_TYPE,
                archive,
            ),
            _asset_descriptor(
                "raw_timing",
                RAW_TIMING_ASSET,
                JSON_MEDIA_TYPE,
                raw_timing,
            ),
            _asset_descriptor(
                "event_profiler",
                EVENT_PROFILER_ASSET,
                JSON_MEDIA_TYPE,
                event_profiler,
            ),
        ],
        "exclusions": [
            "correctness-arrays",
            "operations",
            "private-openings",
            "private-six-scope-bundle",
        ],
    }


def _validate_summary(
    value: Any,
    bindings: Mapping[str, Any],
    archive: bytes,
    raw_timing: bytes,
    event_profiler: bytes,
) -> dict[str, Any]:
    _exact_keys(
        value,
        {
            "schema_version",
            "kind",
            "bindings",
            "scope_order",
            "assets",
            "exclusions",
        },
        "technical summary",
    )
    _require(
        _is_int(value["schema_version"])
        and value["schema_version"] == SCHEMA_VERSION
        and value["kind"] == SUMMARY_KIND
        and value["scope_order"] == list(TECHNICAL_SCOPE_ORDER)
        and value["exclusions"]
        == [
            "correctness-arrays",
            "operations",
            "private-openings",
            "private-six-scope-bundle",
        ],
        "technical summary identity or exclusions differ",
    )
    _validate_bindings(value["bindings"], "technical summary bindings", bindings)
    expected = _technical_summary(bindings, archive, raw_timing, event_profiler)[
        "assets"
    ]
    assets = value["assets"]
    _require(
        isinstance(assets, list)
        and len(assets) == len(expected)
        and canonical_json_bytes(assets) == canonical_json_bytes(expected),
        "technical summary asset bindings differ",
    )
    for index, asset in enumerate(assets):
        _exact_keys(
            asset,
            {"role", "name", "media_type", "size_bytes", "sha256"},
            f"technical summary assets[{index}]",
        )
        _require(
            asset["role"] != "technical_summary"
            and asset["name"] != TECHNICAL_SUMMARY_ASSET,
            "technical summary contains a self-reference",
        )
    _scan_public_tree(value, "technical summary")
    return value


def _asset_bytes(value: Any, name: str) -> bytes:
    limit = (
        MAX_ARCHIVE_BYTES if name == TECHNICAL_EVIDENCE_ASSET else MAX_PUBLIC_JSON_BYTES
    )
    _require(
        type(value) is bytes and 0 < len(value) <= limit,
        f"publication asset {name!r} byte size differs",
    )
    return value


def _validate_expected_assets(
    assets: Mapping[str, bytes],
    expected: Mapping[str, Mapping[str, Any]],
) -> None:
    _require(
        isinstance(expected, Mapping)
        and set(expected) == {role for role, _name in ASSET_ORDER},
        "expected publication asset ledger closure differs",
    )
    for role, name in ASSET_ORDER:
        record = expected[role]
        _exact_keys(
            record,
            {"name", "size_bytes", "sha256"},
            f"expected publication asset {role}",
        )
        raw = assets[name]
        _require(
            record["name"] == name
            and _is_int(record["size_bytes"])
            and record["size_bytes"] == len(raw)
            and record["sha256"] == _sha256(raw),
            f"publication asset {role} differs from the external ledger",
        )


def validate_publication_assets(
    assets: Mapping[str, bytes],
    *,
    expected_policy: Mapping[str, Any],
    expected_bindings: Mapping[str, Any] | None = None,
    expected_assets: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate downloaded bytes against a caller-owned four-asset ledger."""

    _require(isinstance(assets, Mapping), "publication assets must be a mapping")
    expected_names = [name for _role, name in ASSET_ORDER]
    _require(
        set(assets) == set(expected_names) and len(assets) == len(expected_names),
        "publication asset file closure differs",
    )
    raw_assets = {name: _asset_bytes(assets[name], name) for name in expected_names}
    _require(
        sum(len(raw) for raw in raw_assets.values()) <= MAX_PUBLICATION_TOTAL_BYTES,
        "publication asset aggregate byte size differs",
    )
    _validate_expected_assets(raw_assets, expected_assets)
    bindings, policy_scopes, issue115_policy, _witness_policy = _validate_policy(
        expected_policy, expected_bindings
    )
    archive_result = validate_public_archive(
        raw_assets[TECHNICAL_EVIDENCE_ASSET],
        expected_policy=expected_policy,
        expected_bindings=bindings,
    )
    scopes = archive_result["technical_scopes"]
    raw_timing = _strict_json_bytes(
        raw_assets[RAW_TIMING_ASSET], "issue-115 raw timing asset"
    )
    event_profiler = _strict_json_bytes(
        raw_assets[EVENT_PROFILER_ASSET], "issue-115 event profiler asset"
    )
    _validate_issue115_timing(raw_timing, issue115_policy, scopes, bindings)
    _validate_issue115_profiler(
        event_profiler,
        issue115_policy,
        policy_scopes,
        scopes,
        bindings,
    )
    summary = _strict_json_bytes(
        raw_assets[TECHNICAL_SUMMARY_ASSET], "technical summary asset"
    )
    _validate_summary(
        summary,
        bindings,
        raw_assets[TECHNICAL_EVIDENCE_ASSET],
        raw_assets[RAW_TIMING_ASSET],
        raw_assets[EVENT_PROFILER_ASSET],
    )
    descriptors = {
        role: {
            "name": name,
            "size_bytes": len(raw_assets[name]),
            "sha256": _sha256(raw_assets[name]),
        }
        for role, name in ASSET_ORDER
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": VALIDATION_KIND,
        "bindings": bindings,
        "asset_order": [role for role, _name in ASSET_ORDER],
        "assets": descriptors,
        "technical_scopes": scopes,
        "correctness_commitments": archive_result["correctness_commitments"],
        "execution_witness": archive_result["execution_witness"],
        "raw_timing": raw_timing,
        "event_profiler": event_profiler,
        "technical_summary": summary,
    }


def build_publication_assets(
    projection: Mapping[str, Any],
    *,
    expected_policy: Mapping[str, Any],
    expected_bindings: Mapping[str, Any] | None = None,
) -> dict[str, bytes]:
    """Construct four acyclic release assets from a trusted public projection."""

    _exact_keys(
        projection,
        {
            "schema_version",
            "kind",
            "bindings",
            "technical_scopes",
            "correctness_commitments",
            "execution_witness",
            "raw_timing",
            "event_profiler",
        },
        "public projection",
    )
    bindings, policy_scopes, issue115_policy, _witness_policy = _validate_policy(
        expected_policy, expected_bindings
    )
    _require(
        _is_int(projection["schema_version"])
        and projection["schema_version"] == SCHEMA_VERSION
        and projection["kind"] == PROJECTION_KIND,
        "public projection identity differs",
    )
    _validate_bindings(projection["bindings"], "public projection bindings", bindings)
    scopes = projection["technical_scopes"]
    _require(isinstance(scopes, list), "public projection scopes must be a list")

    # Construct raw issue-115 assets first, then the archive, then the one-way
    # summary.  No earlier byte sequence names or digests the summary.
    raw_timing = canonical_json_bytes(projection["raw_timing"])
    event_profiler = canonical_json_bytes(projection["event_profiler"])
    archive = build_public_archive(
        scopes,
        projection["correctness_commitments"],
        projection["execution_witness"],
        expected_policy=expected_policy,
        expected_bindings=bindings,
    )
    _validate_issue115_timing(
        projection["raw_timing"], issue115_policy, scopes, bindings
    )
    _validate_issue115_profiler(
        projection["event_profiler"],
        issue115_policy,
        policy_scopes,
        scopes,
        bindings,
    )
    summary_document = _technical_summary(bindings, archive, raw_timing, event_profiler)
    summary = canonical_json_bytes(summary_document)
    result = {
        TECHNICAL_EVIDENCE_ASSET: archive,
        TECHNICAL_SUMMARY_ASSET: summary,
        RAW_TIMING_ASSET: raw_timing,
        EVENT_PROFILER_ASSET: event_profiler,
    }
    self_ledger = {
        role: {
            "name": name,
            "size_bytes": len(result[name]),
            "sha256": _sha256(result[name]),
        }
        for role, name in ASSET_ORDER
    }
    validate_publication_assets(
        result,
        expected_policy=expected_policy,
        expected_bindings=bindings,
        expected_assets=self_ledger,
    )
    return result


def _release_capture_document(value: Any) -> tuple[dict[str, Any], bytes]:
    if type(value) is bytes:
        _require(
            0 < len(value) <= MAX_RELEASE_CAPTURE_BYTES,
            "release capture byte size differs",
        )
        document = _strict_json_bytes(value, "release capture")
        return document, value
    _require(isinstance(value, Mapping), "release capture must be bytes or a mapping")
    document = dict(value)
    raw = canonical_json_bytes(document)
    _require(
        len(raw) <= MAX_RELEASE_CAPTURE_BYTES,
        "release capture byte size differs",
    )
    return document, raw


def _canonical_release_urls(
    final_sha: str, release_id: int, asset_id: int, asset_name: str
) -> tuple[str, str, str, str, str, str]:
    tag = f"{RELEASE_TAG_PREFIX}{final_sha}"
    api_root = f"https://api.github.com/repos/{REPOSITORY}"
    web_root = f"https://github.com/{REPOSITORY}"
    return (
        f"{web_root}/releases/tag/{tag}",
        f"{api_root}/git/refs/tags/{tag}",
        f"{api_root}/git/commits/{final_sha}",
        f"{api_root}/releases/{release_id}",
        f"{api_root}/releases/assets/{asset_id}",
        f"{web_root}/releases/download/{tag}/{asset_name}",
    )


def _validate_release_identity_anchor(
    value: Any, bindings: Mapping[str, Any]
) -> dict[str, Any]:
    _exact_keys(
        value,
        {
            "repository",
            "release_id",
            "tag_name",
            "target_commitish",
            "api_url",
            "html_url",
            "tag_ref",
            "assets",
        },
        "expected release identity",
    )
    final_sha = bindings["final_sha"]
    release_id = value["release_id"]
    _require(
        value["repository"] == REPOSITORY
        and _is_int(release_id)
        and release_id > 0
        and value["tag_name"] == f"{RELEASE_TAG_PREFIX}{final_sha}"
        and value["target_commitish"] == final_sha,
        "expected release identity or target differs",
    )
    html_url, tag_api_url, object_url, release_api_url, _, _ = _canonical_release_urls(
        final_sha, release_id, 1, TECHNICAL_EVIDENCE_ASSET
    )
    _require(
        value["api_url"] == release_api_url and value["html_url"] == html_url,
        "expected release identity URLs differ",
    )
    tag_ref = value["tag_ref"]
    _exact_keys(
        tag_ref,
        {"ref", "api_url", "object_type", "object_sha", "object_url"},
        "expected release tag identity",
    )
    _require(
        tag_ref
        == {
            "ref": f"refs/tags/{value['tag_name']}",
            "api_url": tag_api_url,
            "object_type": "commit",
            "object_sha": final_sha,
            "object_url": object_url,
        },
        "expected release tag identity differs",
    )
    records = value["assets"]
    _require(
        isinstance(records, list)
        and [
            (
                (record.get("role"), record.get("name"))
                if isinstance(record, Mapping)
                else None
            )
            for record in records
        ]
        == list(ASSET_ORDER),
        "expected release asset identity closure differs",
    )
    asset_ids: set[int] = set()
    normalized_records = []
    for index, ((role, name), record) in enumerate(
        zip(ASSET_ORDER, records, strict=True)
    ):
        label = f"expected release asset identity[{index}]"
        _exact_keys(
            record,
            {
                "role",
                "asset_id",
                "release_id",
                "name",
                "api_url",
                "browser_download_url",
            },
            label,
        )
        asset_id = record["asset_id"]
        _, _, _, _, api_url, browser_download_url = _canonical_release_urls(
            final_sha, release_id, asset_id, name
        )
        _require(
            record["role"] == role
            and _is_int(asset_id)
            and asset_id > 0
            and asset_id not in asset_ids
            and _is_int(record["release_id"])
            and record["release_id"] == release_id
            and record["name"] == name
            and record["api_url"] == api_url
            and record["browser_download_url"] == browser_download_url,
            f"{label} differs",
        )
        asset_ids.add(asset_id)
        normalized_records.append(dict(record))
    return {
        "repository": value["repository"],
        "release_id": release_id,
        "tag_name": value["tag_name"],
        "target_commitish": value["target_commitish"],
        "api_url": value["api_url"],
        "html_url": value["html_url"],
        "tag_ref": dict(tag_ref),
        "assets": normalized_records,
    }


def _release_identity_from_capture(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "repository",
        "release_id",
        "tag_name",
        "target_commitish",
        "api_url",
        "html_url",
    }
    asset_fields = {
        "role",
        "asset_id",
        "release_id",
        "name",
        "api_url",
        "browser_download_url",
    }
    return {
        **{field: value[field] for field in fields},
        "tag_ref": dict(value["tag_ref"]),
        "assets": [
            {field: record[field] for field in asset_fields}
            for record in value["assets"]
        ],
    }


def _validate_release_capture(
    value: Any,
    assets: Mapping[str, bytes],
    expected_assets: Mapping[str, Mapping[str, Any]],
    bindings: Mapping[str, Any],
    expected_release_identity: Mapping[str, Any],
) -> dict[str, Any]:
    release_identity = _validate_release_identity_anchor(
        expected_release_identity, bindings
    )
    _exact_keys(
        value,
        {
            "schema_version",
            "kind",
            "repository",
            "release_id",
            "tag_name",
            "target_commitish",
            "api_url",
            "html_url",
            "immutable",
            "draft",
            "prerelease",
            "tag_ref",
            "assets",
        },
        "release capture",
    )
    final_sha = bindings["final_sha"]
    tag_name = f"{RELEASE_TAG_PREFIX}{final_sha}"
    release_id = value["release_id"]
    _require(
        _is_int(value["schema_version"])
        and value["schema_version"] == SCHEMA_VERSION
        and value["kind"] == RELEASE_CAPTURE_KIND
        and value["repository"] == REPOSITORY
        and _is_int(release_id)
        and release_id > 0
        and value["tag_name"] == tag_name
        and value["target_commitish"] == final_sha
        and type(value["immutable"]) is bool
        and value["immutable"] is True
        and type(value["draft"]) is bool
        and value["draft"] is False
        and type(value["prerelease"]) is bool
        and value["prerelease"] is False,
        "release capture identity, target, or state differs",
    )
    html_url, tag_api_url, object_url, release_api_url, _, _ = _canonical_release_urls(
        final_sha, release_id, 1, TECHNICAL_EVIDENCE_ASSET
    )
    _require(
        value["api_url"] == release_api_url and value["html_url"] == html_url,
        "release capture API or HTML URL differs",
    )
    tag_ref = value["tag_ref"]
    _exact_keys(
        tag_ref,
        {"ref", "api_url", "object_type", "object_sha", "object_url"},
        "release capture tag ref",
    )
    _require(
        tag_ref
        == {
            "ref": f"refs/tags/{tag_name}",
            "api_url": tag_api_url,
            "object_type": "commit",
            "object_sha": final_sha,
            "object_url": object_url,
        },
        "release capture lightweight tag binding differs",
    )
    records = value["assets"]
    _require(
        isinstance(records, list)
        and [
            (
                (record.get("role"), record.get("name"))
                if isinstance(record, dict)
                else None
            )
            for record in records
        ]
        == list(ASSET_ORDER),
        "release capture asset order or closure differs",
    )
    asset_ids: set[int] = set()
    digests: set[str] = set()
    api_urls: set[str] = set()
    download_urls: set[str] = set()
    for index, ((role, name), record) in enumerate(
        zip(ASSET_ORDER, records, strict=True)
    ):
        label = f"release capture assets[{index}]"
        _exact_keys(
            record,
            {
                "role",
                "asset_id",
                "release_id",
                "name",
                "api_url",
                "browser_download_url",
                "state",
                "size_bytes",
                "sha256",
            },
            label,
        )
        asset_id = record["asset_id"]
        _require(
            _is_int(asset_id) and asset_id > 0 and asset_id not in asset_ids,
            f"{label} asset id differs or repeats",
        )
        _, _, _, _, api_url, browser_download_url = _canonical_release_urls(
            final_sha, release_id, asset_id, name
        )
        external = expected_assets[role]
        raw = assets[name]
        _require(
            record["role"] == role
            and _is_int(record["release_id"])
            and record["release_id"] == release_id
            and record["name"] == name
            and record["api_url"] == api_url
            and record["browser_download_url"] == browser_download_url
            and record["state"] == "uploaded"
            and _is_int(record["size_bytes"])
            and record["size_bytes"] == len(raw) == external["size_bytes"]
            and record["sha256"] == _sha256(raw) == external["sha256"],
            f"{label} byte, ledger, release, or URL binding differs",
        )
        _require(
            record["sha256"] not in digests
            and api_url not in api_urls
            and browser_download_url not in download_urls,
            f"{label} aliases another release asset",
        )
        asset_ids.add(asset_id)
        digests.add(record["sha256"])
        api_urls.add(api_url)
        download_urls.add(browser_download_url)
    _require(
        _release_identity_from_capture(value) == release_identity,
        "release capture differs from the caller-owned release identity",
    )
    return value


def _ordered_asset_ledger(
    expected_assets: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {"role": role, **dict(expected_assets[role])} for role, _name in ASSET_ORDER
    ]


def _publication_receipt_document(
    validated: Mapping[str, Any],
    release_capture: Mapping[str, Any],
    expected_policy: Mapping[str, Any],
    expected_assets: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    ledger = _ordered_asset_ledger(expected_assets)
    witness = validated["execution_witness"]
    witness_member = _descriptor(EXECUTION_WITNESS_PATH, canonical_json_bytes(witness))
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": PUBLICATION_RECEIPT_KIND,
        "bindings": dict(validated["bindings"]),
        "asset_order": [role for role, _name in ASSET_ORDER],
        "asset_ledger": ledger,
        "release_capture": dict(release_capture),
        "execution_witness": witness,
        "execution_witness_member": witness_member,
        "hashes": {
            "release_capture_sha256": _sha256(canonical_json_bytes(release_capture)),
            "execution_witness_member_sha256": witness_member["sha256"],
            "trusted_policy_sha256": _sha256(canonical_json_bytes(expected_policy)),
            "asset_ledger_sha256": _sha256(canonical_json_bytes(ledger)),
        },
    }


def _validate_receipt_document(
    receipt: Any,
    assets: Mapping[str, bytes],
    *,
    expected_policy: Mapping[str, Any],
    expected_bindings: Mapping[str, Any] | None,
    expected_assets: Mapping[str, Mapping[str, Any]],
    expected_release_identity: Mapping[str, Any],
) -> dict[str, Any]:
    _exact_keys(
        receipt,
        {
            "schema_version",
            "kind",
            "bindings",
            "asset_order",
            "asset_ledger",
            "release_capture",
            "execution_witness",
            "execution_witness_member",
            "hashes",
        },
        "publication receipt",
    )
    _require(
        _is_int(receipt["schema_version"])
        and receipt["schema_version"] == SCHEMA_VERSION
        and receipt["kind"] == PUBLICATION_RECEIPT_KIND,
        "publication receipt identity differs",
    )
    validated = validate_publication_assets(
        assets,
        expected_policy=expected_policy,
        expected_bindings=expected_bindings,
        expected_assets=expected_assets,
    )
    bindings = validated["bindings"]
    _validate_bindings(receipt["bindings"], "publication receipt bindings", bindings)
    raw_assets = {name: assets[name] for _role, name in ASSET_ORDER}
    capture = _validate_release_capture(
        receipt["release_capture"],
        raw_assets,
        expected_assets,
        bindings,
        expected_release_identity,
    )
    expected = _publication_receipt_document(
        validated, capture, expected_policy, expected_assets
    )
    _require(
        canonical_json_bytes(receipt) == canonical_json_bytes(expected),
        "publication receipt bindings or hashes differ",
    )
    _exact_keys(
        receipt["hashes"],
        {
            "release_capture_sha256",
            "execution_witness_member_sha256",
            "trusted_policy_sha256",
            "asset_ledger_sha256",
        },
        "publication receipt hashes",
    )
    _require(
        not any("receipt" in key for key in receipt["hashes"]),
        "publication receipt contains a self digest",
    )
    return receipt


def finalize_publication(
    assets: Mapping[str, bytes],
    release_metadata: bytes | Mapping[str, Any],
    *,
    expected_policy: Mapping[str, Any],
    expected_release_identity: Mapping[str, Any],
    expected_bindings: Mapping[str, Any] | None = None,
    expected_assets: Mapping[str, Mapping[str, Any]],
) -> bytes:
    """Validate downloaded bytes and captured release state; return a receipt."""

    validated = validate_publication_assets(
        assets,
        expected_policy=expected_policy,
        expected_bindings=expected_bindings,
        expected_assets=expected_assets,
    )
    capture, _capture_raw = _release_capture_document(release_metadata)
    raw_assets = {name: assets[name] for _role, name in ASSET_ORDER}
    _validate_release_capture(
        capture,
        raw_assets,
        expected_assets,
        validated["bindings"],
        expected_release_identity,
    )
    receipt = _publication_receipt_document(
        validated, capture, expected_policy, expected_assets
    )
    raw = canonical_json_bytes(receipt)
    _require(
        len(raw) <= MAX_PUBLICATION_RECEIPT_BYTES,
        "publication receipt byte size differs",
    )
    validate_publication_receipt(
        raw,
        assets,
        expected_policy=expected_policy,
        expected_release_identity=expected_release_identity,
        expected_bindings=validated["bindings"],
        expected_assets=expected_assets,
    )
    return raw


def validate_publication_receipt(
    receipt_bytes: bytes,
    assets: Mapping[str, bytes],
    *,
    expected_policy: Mapping[str, Any],
    expected_release_identity: Mapping[str, Any],
    expected_bindings: Mapping[str, Any] | None = None,
    expected_assets: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Independently re-open a receipt against downloaded bytes and trust anchors."""

    _require(
        type(receipt_bytes) is bytes
        and 0 < len(receipt_bytes) <= MAX_PUBLICATION_RECEIPT_BYTES,
        "publication receipt byte size differs",
    )
    receipt = _strict_json_bytes(receipt_bytes, "publication receipt")
    return _validate_receipt_document(
        receipt,
        assets,
        expected_policy=expected_policy,
        expected_release_identity=expected_release_identity,
        expected_bindings=expected_bindings,
        expected_assets=expected_assets,
    )


def _production_file_bytes(path_value: Path | str, label: str, limit: int) -> bytes:
    failure: PublicationError | None = None
    try:
        from benchmarks import issue123_privacy as privacy

        _path, raw = privacy._private_file_bytes(path_value, label, maximum=limit)
    except ImportError, OSError, TypeError, ValueError:
        failure = PublicationError(f"{label} is unavailable")
    if failure is not None:
        raise failure from None
    return raw


def _write_exclusive(path: Path, raw: bytes, label: str, mode: int) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    failure: PublicationError | None = None
    try:
        descriptor = os.open(path, flags, mode)
        try:
            os.fchmod(descriptor, mode)
            view = memoryview(raw)
            while view:
                written = os.write(descriptor, view)
                _require(written > 0, f"{label} could not be written")
                view = view[written:]
            os.fsync(descriptor)
            metadata = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        failure = PublicationError(f"{label} could not be created exclusively")
    if failure is not None:
        raise failure from None
    _require(
        stat.S_ISREG(metadata.st_mode)
        and stat.S_IMODE(metadata.st_mode) == mode
        and metadata.st_size == len(raw),
        f"{label} mode or size differs",
    )


def _private_output_path(
    path_value: Path | str,
    *,
    forbidden_roots: Sequence[Path | str],
    prospective_forbidden_roots: Sequence[Path | str] = (),
) -> Path:
    failure: PublicationError | None = None
    try:
        from benchmarks import issue123_privacy as privacy

        output, parent = privacy.preflight_private_output_path(
            path_value,
            label="publication private authority output",
            forbidden_roots=forbidden_roots,
            prospective_forbidden_roots=prospective_forbidden_roots,
        )
        metadata = parent.lstat()
    except ImportError, OSError, TypeError, ValueError:
        failure = PublicationError("private authority output is unavailable")
    if failure is not None:
        raise failure from None
    _require(
        stat.S_ISDIR(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and stat.S_IMODE(metadata.st_mode) == 0o700,
        "private openings must use a distinct mode-0700 directory",
    )
    return output


def _existing_publication_directory(
    path_value: Path | str,
    label: str,
) -> Path:
    failure: PublicationError | None = None
    try:
        from benchmarks import issue123_privacy as privacy

        directory = privacy._lexical_path_without_symlinks(
            path_value,
            label,
            require_leaf=True,
        )
        metadata = directory.lstat()
    except ImportError, OSError, TypeError, ValueError:
        failure = PublicationError(f"{label} is unavailable")
    if failure is not None:
        raise failure from None
    _require(
        stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
        f"{label} is not a directory",
    )
    return directory


def _commit_private_authority_file(
    path: Path,
    raw: bytes,
    *,
    forbidden_roots: Sequence[Path | str],
) -> Path:
    try:
        from benchmarks import issue123_privacy as privacy
    except ImportError:
        raise PublicationError(
            "private authority file could not be committed"
        ) from None
    try:
        return privacy.write_private_authority_file(
            path,
            raw,
            label="publication private authority file",
            forbidden_roots=forbidden_roots,
        )
    except privacy.PrivateAuthorityCommitError:
        raise PublicationCommitError(
            "private authority file was committed but final verification failed"
        ) from None
    except OSError, TypeError, ValueError:
        failure = PublicationError("private authority file could not be committed")
    raise failure from None


def prepare_publication(
    *,
    source_specification: Path | str,
    completion_index: Path | str,
    policy_path: Path | str,
    policy_sha256: str,
    runtime_receipt_paths: Mapping[str, Path | str],
    asset_output_directory: Path | str,
    private_openings_output: Path | str,
    salt: bytes | None = None,
) -> dict[str, Any]:
    """Create the exact v1 public bytes plus a protected v1 binding sidecar."""

    try:
        from benchmarks import issue123_privacy as privacy
    except ImportError, OSError:
        raise PublicationError("private source adapter is unavailable") from None
    policy_raw = _production_file_bytes(
        policy_path, "trusted publication policy", MAX_PUBLIC_JSON_BYTES
    )
    policy = _strict_json_bytes(policy_raw, "trusted publication policy")
    _require(
        policy_raw == canonical_json_bytes(policy)
        and isinstance(policy_sha256, str)
        and _SHA256_RE.fullmatch(policy_sha256) is not None
        and hashlib.sha256(policy_raw).hexdigest() == policy_sha256,
        "trusted publication policy bytes or digest differ",
    )
    failure: PublicationError | None = None
    try:
        specification = privacy.load_publication_source_spec(
            source_specification,
            policy,
            completion_index=completion_index,
            policy_sha256=policy_sha256,
        )
        materialized = privacy.materialize_publication_inputs(
            specification, runtime_receipt_paths=runtime_receipt_paths
        )
        openings = privacy.PrivateOpenings(salt)
        projection = privacy.project_publication(
            materialized.private_bundle, policy, private_openings=openings
        )
    except AttributeError, OSError, TypeError, ValueError:
        failure = PublicationError("publication source preparation failed")
    if failure is not None:
        raise failure from None
    assets = build_publication_assets(
        projection,
        expected_policy=policy,
        expected_bindings=policy["bindings"],
    )
    ledger = [
        {
            "role": role,
            "name": name,
            "size_bytes": len(assets[name]),
            "sha256": hashlib.sha256(assets[name]).hexdigest(),
        }
        for role, name in ASSET_ORDER
    ]
    context = privacy.publication_binding_context(materialized, projection, ledger)
    protected_raw = privacy.serialize_private_openings(openings, context)
    failure = None
    try:
        checked_completion_index = privacy._lexical_path_without_symlinks(
            completion_index,
            "publication source completion index",
            require_leaf=True,
        )
        completion_bundle_root = checked_completion_index.parent
        output, output_parent = privacy.preflight_private_output_path(
            asset_output_directory,
            label="publication asset output directory",
            forbidden_roots=(completion_bundle_root,),
        )
    except OSError, TypeError, ValueError:
        failure = PublicationError("publication output location is unavailable")
    if failure is not None:
        raise failure from None
    _require(
        output.name not in {"", ".", ".."}
        and not output.exists()
        and not output.is_symlink(),
        "publication asset output directory already exists or is invalid",
    )
    private_path = _private_output_path(
        private_openings_output,
        forbidden_roots=(completion_bundle_root,),
        prospective_forbidden_roots=(output,),
    )
    _require(
        private_path.name not in {"", ".", ".."}
        and not private_path.exists()
        and not private_path.is_symlink(),
        "protected openings output already exists or is invalid",
    )
    temporary: Path | None = None
    file_failure: PublicationError | None = None
    try:
        temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output_parent))
        for _role, name in ASSET_ORDER:
            _write_exclusive(
                temporary / name, assets[name], f"publication asset {name}", 0o644
            )
        reopened = {
            name: _production_file_bytes(
                temporary / name,
                f"reopened publication asset {name}",
                MAX_ARCHIVE_BYTES,
            )
            for _role, name in ASSET_ORDER
        }
        validate_publication_assets(
            reopened,
            expected_policy=policy,
            expected_bindings=policy["bindings"],
            expected_assets={
                item["role"]: {
                    key: item[key] for key in ("name", "size_bytes", "sha256")
                }
                for item in ledger
            },
        )
        _require(reopened == assets, "reopened publication bytes differ")
        temporary.rename(output)
        _commit_private_authority_file(
            private_path,
            protected_raw,
            forbidden_roots=(completion_bundle_root, output),
        )
    except PublicationCommitError:
        raise
    except OSError, TypeError, ValueError:
        file_failure = PublicationError("publication files could not be prepared")
    if file_failure is not None:
        if temporary is not None:
            try:
                if temporary.exists():
                    shutil.rmtree(temporary)
            except OSError:
                pass
        raise file_failure from None
    return {
        "asset_directory": output,
        "asset_paths": {role: output / name for role, name in ASSET_ORDER},
        "asset_ledger": ledger,
        "private_openings": private_path,
        "technical_input_root": materialized.technical_input_root,
        "public_projection_sha256": context["public_projection_sha256"],
    }


def finalize_publication_files(
    *,
    asset_directory: Path | str,
    release_capture_path: Path | str,
    release_identity_path: Path | str,
    policy_path: Path | str,
    policy_sha256: str,
    receipt_output: Path | str,
) -> Path:
    """Finalize publication from exact files and emit canonical receipt bytes."""

    policy_raw = _production_file_bytes(
        policy_path, "trusted publication policy", MAX_PUBLIC_JSON_BYTES
    )
    policy = _strict_json_bytes(policy_raw, "trusted publication policy")
    _require(
        policy_raw == canonical_json_bytes(policy)
        and hashlib.sha256(policy_raw).hexdigest() == policy_sha256,
        "trusted publication policy bytes or digest differ",
    )
    directory = _existing_publication_directory(
        asset_directory,
        "publication asset directory",
    )
    assets = {
        name: _production_file_bytes(
            directory / name, f"publication asset {name}", MAX_ARCHIVE_BYTES
        )
        for _role, name in ASSET_ORDER
    }
    ledger = {
        role: {
            "name": name,
            "size_bytes": len(assets[name]),
            "sha256": hashlib.sha256(assets[name]).hexdigest(),
        }
        for role, name in ASSET_ORDER
    }
    release_raw = _production_file_bytes(
        release_capture_path, "release capture", MAX_RELEASE_CAPTURE_BYTES
    )
    identity_raw = _production_file_bytes(
        release_identity_path, "release identity", MAX_RELEASE_CAPTURE_BYTES
    )
    identity = _strict_json_bytes(identity_raw, "release identity")
    receipt = finalize_publication(
        assets,
        release_raw,
        expected_policy=policy,
        expected_release_identity=identity,
        expected_bindings=policy["bindings"],
        expected_assets=ledger,
    )
    output = _private_output_path(
        receipt_output,
        forbidden_roots=(directory,),
    )
    return _commit_private_authority_file(
        output,
        receipt,
        forbidden_roots=(directory,),
    )


def _role_paths(values: Sequence[str], expected: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        role, separator, path = value.partition("=")
        _require(separator == "=" and role not in result and path, "role path differs")
        result[role] = Path(path)
    _require(list(result) == list(expected), "role path closure or order differs")
    return result


class _CliUsageError(ValueError):
    pass


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise _CliUsageError("publication CLI usage differs") from None


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _SafeArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source-spec", type=Path, required=True)
    prepare.add_argument("--completion-index", type=Path, required=True)
    prepare.add_argument("--policy", type=Path, required=True)
    prepare.add_argument("--policy-sha256", required=True)
    prepare.add_argument("--runtime-receipt", action="append", required=True)
    prepare.add_argument("--asset-output-directory", type=Path, required=True)
    prepare.add_argument("--private-openings-output", type=Path, required=True)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--asset-directory", type=Path, required=True)
    finalize.add_argument("--release-capture", type=Path, required=True)
    finalize.add_argument("--release-identity", type=Path, required=True)
    finalize.add_argument("--policy", type=Path, required=True)
    finalize.add_argument("--policy-sha256", required=True)
    finalize.add_argument("--receipt-output", type=Path, required=True)
    return parser.parse_args(argv)


def _main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    if args.command == "prepare":
        from benchmarks import issue123_privacy as privacy

        prepare_publication(
            source_specification=args.source_spec,
            completion_index=args.completion_index,
            policy_path=args.policy,
            policy_sha256=args.policy_sha256,
            runtime_receipt_paths=_role_paths(
                args.runtime_receipt, privacy.RUNTIME_RECEIPT_ORDER
            ),
            asset_output_directory=args.asset_output_directory,
            private_openings_output=args.private_openings_output,
        )
        print("issue123-publication-prepare-ok")
    else:
        finalize_publication_files(
            asset_directory=args.asset_directory,
            release_capture_path=args.release_capture,
            release_identity_path=args.release_identity,
            policy_path=args.policy,
            policy_sha256=args.policy_sha256,
            receipt_output=args.receipt_output,
        )
        print("issue123-publication-finalize-ok")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fixed-token publication command boundary."""

    values = list(sys.argv[1:] if argv is None else argv)
    command = values[0] if values and values[0] in {"prepare", "finalize"} else None
    try:
        return _main(values)
    except _CliUsageError:
        print("issue123-publication-usage-failed", file=sys.stderr)
        return 2
    except ImportError, OSError, PublicationError, TypeError, ValueError:
        token = (
            f"issue123-publication-{command}-failed"
            if command is not None
            else "issue123-publication-usage-failed"
        )
        print(token, file=sys.stderr)
        return 2


def _cli(argv: Sequence[str] | None = None) -> int:
    return main(argv)


# Concise public aliases for callers that think in publication rather than ZIP.
assemble_publication = build_publication_assets
validate_publication = validate_publication_assets


__all__ = [
    "ARCHIVE_ENTRY_ORDER",
    "ARCHIVE_FORMAT",
    "ARCHIVE_MANIFEST_KIND",
    "ASSET_ORDER",
    "COMMITMENTS_KIND",
    "COMMITMENTS_PATH",
    "COMMITMENT_ALGORITHM",
    "EXECUTION_CLAIM_ORDER",
    "EXECUTION_WITNESS_KIND",
    "EXECUTION_WITNESS_PATH",
    "EVENT_PROFILER_ASSET",
    "EVENT_PROFILER_CONTRACT_ID",
    "EVENT_PROFILER_KIND",
    "FIXED_ZIP_TIMESTAMP",
    "LOCAL_CLOCK",
    "MANIFEST_PATH",
    "POLICY_KIND",
    "PROJECTION_KIND",
    "PUBLICATION_RECEIPT_KIND",
    "PublicationError",
    "RAW_TIMING_ASSET",
    "RAW_TIMING_CONTRACT_ID",
    "RAW_TIMING_KIND",
    "RELEASE_CAPTURE_KIND",
    "RELEASE_TAG_PREFIX",
    "REPOSITORY",
    "REQUIRED_JOB_NAMES",
    "SCAN_CONTRACT",
    "SCHEMA_VERSION",
    "SCOPE_KIND",
    "SCOPE_PATHS",
    "SUMMARY_KIND",
    "TECHNICAL_EVIDENCE_ASSET",
    "TECHNICAL_SCOPE_ORDER",
    "TECHNICAL_SUMMARY_ASSET",
    "VALIDATION_KIND",
    "assemble_publication",
    "build_public_archive",
    "build_publication_assets",
    "canonical_json_bytes",
    "finalize_publication",
    "finalize_publication_files",
    "main",
    "prepare_publication",
    "publication_bindings",
    "scan_public_bytes",
    "validate_public_archive",
    "validate_publication",
    "validate_publication_assets",
    "validate_publication_receipt",
]


if __name__ == "__main__":
    raise SystemExit(_cli())
