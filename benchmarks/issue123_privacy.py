#!/usr/bin/env python3
"""Project private issue #123 evidence into privacy-safe public documents.

This module deliberately does not create releases, access the network, or write
archives.  Its input is private and untrusted.  A separately constructed policy
is the trust anchor for bindings and closure; hashes supplied by the private
bundle never define what evidence is expected.
"""

from __future__ import annotations

import codecs
import copy
import hashlib
import hmac
import io
import json
import math
import os
import platform
import re
import secrets
import stat
import statistics
import struct
import tarfile
import unicodedata
import zipfile
import zlib
from bisect import bisect_right
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from fractions import Fraction
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Callable

import numpy as np

SCHEMA_VERSION = 1
POLICY_KIND = "issue-123-publication-policy"
PRIVATE_INPUT_KIND = "issue-123-private-publication-input"
PROJECTION_KIND = "issue-123-publication-projection"
SCOPE_KIND = "issue-123-public-technical-scope"
CORRECTNESS_KIND = "issue-123-public-correctness-commitments"
EXECUTION_WITNESS_KIND = "issue-123-public-execution-witness"
RAW_TIMING_KIND = "issue-115-raw-timing"
EVENT_PROFILER_KIND = "issue-115-event-level-profiler"
PUBLICATION_SOURCE_SPECIFICATION_KIND = "issue-123-publication-source-specification"
PROTECTED_OPENINGS_KIND = "issue-123-protected-publication-openings"
RAW_TIMING_CONTRACT = "torch-utils-benchmark-fixed-workloads"
EVENT_PROFILER_CONTRACT = "event-level-profiler-fixed-workloads"
COMMITMENT_ALGORITHM = "HMAC-SHA-256"
SCAN_CONTRACT = "recursive-personal-metadata-v1"
LOCAL_CLOCK = "local-origin-microseconds-v1"
PUBLICATION_SOURCE_SPECIFICATION_VERSION = 1
PROTECTED_OPENINGS_VERSION = 1
PRIVATE_OPENING_BINDING_DOMAIN = "gmes.issue123.private-opening-binding.v1"
TECHNICAL_INPUT_INVENTORY_DOMAIN = "gmes.issue123.technical-input-inventory.v1"
PUBLIC_PROJECTION_DOMAIN = "gmes.issue123.public-projection.v1"
PUBLIC_ASSET_LEDGER_DOMAIN = "gmes.issue123.public-asset-ledger.v1"
TAGGED_DIGEST_PREFIX = b"GMES-ISSUE123\x00"
DARWIN_SYSTEM_PATH_ALIASES = {"/tmp": "/private/tmp", "/var": "/private/var"}

TECHNICAL_SCOPE_ORDER = (
    "cpu",
    "policy_paired_real",
    "single_gpu",
    "two_gpu",
    "macos",
)
RUNTIME_RECEIPT_ORDER = (
    "cpu",
    "cuda-eager",
    "cuda-graph",
    "single-gpu-2d",
    "single-gpu-3d",
)
TECHNICAL_SCOPE_PATHS = (
    "scopes/01-cpu.json",
    "scopes/02-policy-paired-real.json",
    "scopes/03-single-gpu.json",
    "scopes/04-two-gpu.json",
    "scopes/05-macos.json",
)
CORRECTNESS_PATH = "correctness/commitments.json"
EXECUTION_WITNESS_PATH = "execution/witness.json"

EXECUTION_CLAIM_ORDER = ("cpu-eager", "cuda-eager", "cuda-graph")
EXECUTION_CLAIM_SCOPES = {
    "cpu-eager": "cpu",
    "cuda-eager": "single_gpu",
    "cuda-graph": "two_gpu",
}
EXECUTION_VALIDATION_WORKFLOW = "CI"
EXECUTION_VALIDATOR_JOB_NAME = "Python 3.14 / ubuntu-latest"

SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
SAFE_NAME_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
JOB_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._+:/()-]{0,255}\Z")
SEMANTIC_SEGMENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
UUID_RE = re.compile(
    rb"(?i)(?<![0-9a-f])[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    rb"[0-9a-f]{4}-[0-9a-f]{12}(?![0-9a-f])"
)
EMAIL_RE = re.compile(
    rb"(?i)(?<![a-z0-9._%+-])[a-z0-9._%+-]{1,64}@"
    rb"[a-z0-9.-]{1,253}\.[a-z]{2,63}(?![a-z0-9.-])"
)
PRIVATE_ROOT_RE = re.compile(
    rb"(?i)(?:/home/[^/\x00\s]+|/users/[^/\x00\s]+|"
    rb"/root(?:/[^\x00\s]*)?|/volumes/[^/\x00\s]+|"
    rb"/private/var/folders/[^\x00\s]+|/var/folders/[^\x00\s]+|"
    rb"/var/lib/jenkins/workspace(?:/[^\x00\s]*)?|/builds(?:/[^\x00\s]*)?|"
    rb"/tmp(?:/[^\x00\s]*)?|/workspaces?(?:/[^\x00\s]*)?|"
    rb"/runner/_work(?:/[^\x00\s]*)?|file://[^\x00\s]+|"
    rb"/opt/(?:hostedtoolcache|actions-runner)(?:/[^\x00\s]*)?|"
    rb"/home/runner/work(?:/[^\x00\s]*)?|/users/runner/work(?:/[^\x00\s]*)?|"
    rb"/github/workspace(?:/[^\x00\s]*)?|[a-z]:[\\/]users[\\/][^\\/\x00\s]+|"
    rb"[a-z]:[\\/](?:a|runner)[\\/](?:_work|work)(?:[\\/][^\x00\s]*)?|"
    rb"\\\\[^\\/\x00\s]+\\[^\x00\s]+)"
)
SECRET_RE = re.compile(
    rb"(?i)(?:github_pat_[a-z0-9_]{20,}|gh[pousr]_[a-z0-9]{20,}|"
    rb"akia[0-9a-z]{16}|bearer\s+[^\x00\r\n\t '\"]{1,4096}|"
    rb"(?:passwords?|passwd|secret[_ -]?(?:access[_ -]?)?keys?|secrets?|"
    rb"tokens?|credentials?|api[_ -]?keys?|access[_ -]?keys?(?:[_ -]?id)?|"
    rb"client[_ -]?secrets?|private[_ -]?keys?|authorization|cookie|"
    rb"signing[_ -]?material)\s*[:=]\s*['\"]?[^\x00\r\n\t '\"]{1,4096}|"
    rb"-----begin (?:rsa |ec |openssh )?private key-----)"
)
PRIVATE_JSON_KEY_RE = re.compile(
    rb'(?i)"(?:hostname|host_name|computername|computer_name|username|user_name|user|login|'
    rb"owner|actor|account|machine_id|machine_uuid|home|home_dir|homedir|cwd|pwd|"
    rb"worktree|workspace|working_directory|environment|env|credentials?|tokens?|"
    rb"passwords?|passwd|secrets?|(?:aws[_ -]?)?secret[_ -]?access[_ -]?keys?|"
    rb"api[_ -]?keys?|access[_ -]?keys?(?:[_ -]?id)?|client[_ -]?secrets?|"
    rb"authorization|cookie|signing[_ -]?material|private[_ -]?keys?|"
    rb"path|cuda_visible_devices|runner_(?:name|os|arch|temp|tool_cache|workspace)|"
    rb"github_(?:actor|triggering_actor|workspace|repository|repository_owner|"
    rb"run_id|run_attempt|job|workflow|sha|ref|head_ref|base_ref)|"
    rb"ssh_auth_sock|shell|tmpdir|temp|tmp|virtual_env|conda_prefix|pythonpath|"
    rb"ld_library_path|dyld_library_path|"
    rb'device_uuid|deviceuuid|device_id|serial_number)"\s*:'
)
PRIVATE_METADATA_KEY_RE = re.compile(
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
PRIVATE_ASSIGNMENT_RE = re.compile(
    rb"(?i)(?<![a-z0-9_])(?:hostname|host_name|computername|computer_name|username|user_name|"
    rb"user|logname|login|host|owner|actor|account|machine_id|machine_uuid|home|"
    rb"home_dir|homedir|cwd|pwd|worktree|workspace|working_directory|"
    rb"github_workspace|github_actor|github_triggering_actor|repository_owner|"
    rb"runner_name|environment|env|device_uuid|deviceuuid|device_id|"
    rb"serial_number)\s*=\s*['\"]?[^\x00\r\n\t '\"]{1,4096}"
)
PRIVATE_ENV_ASSIGNMENT_RE = re.compile(
    rb"(?i)(?<![A-Za-z0-9_])(?:PATH|CUDA_VISIBLE_DEVICES|"
    rb"RUNNER_(?:NAME|OS|ARCH|TEMP|TOOL_CACHE|WORKSPACE)|"
    rb"GITHUB_(?:ACTOR|TRIGGERING_ACTOR|WORKSPACE|REPOSITORY|REPOSITORY_OWNER|"
    rb"RUN_ID|RUN_ATTEMPT|JOB|WORKFLOW|SHA|REF|HEAD_REF|BASE_REF)|"
    rb"SSH_AUTH_SOCK|SHELL|TMPDIR|TEMP|TMP|VIRTUAL_ENV|CONDA_PREFIX|PYTHONPATH|"
    rb"LD_LIBRARY_PATH|DYLD_LIBRARY_PATH)\s*=\s*"
    rb"['\"]?[^\x00\r\n\t '\"]{1,4096}"
)
GITHUB_WORKSPACE_ASSIGNMENT_RE = re.compile(rb"(?i)github_workspace=")
TAR_CHECKSUM_FIELD_RE = re.compile(rb"(?=(?:[ 0-7]{6}\x00[ \x00]|[ 0-7]{7}[\x00 ]))")
WITHHELD_PAYLOAD_RE = re.compile(
    r"(?i)(?:^|[-_.])(?:operations?|private|completion|correctness(?:[-_]?arrays?)?)"
    r"(?:[-_.]|$)"
)

MAX_TRACE_BYTES = 256 * 1024 * 1024
MAX_TRACE_EVENTS = 2_000_000
MAX_TIMING_SAMPLES = 1_000_000
MAX_LOCAL_TIMESTAMP_US = 24 * 60 * 60 * 1_000_000
MAX_ARRAY_BYTES = 1024 * 1024 * 1024
MAX_PAYLOAD_BYTES = 1024 * 1024 * 1024
MAX_PUBLIC_JSON_BYTES = 64 * 1024 * 1024
MAX_PUBLIC_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 20_000
MAX_ARCHIVE_MEMBER_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
MAX_COMPRESSION_RATIO = 200
MAX_TAR_BYTES = MAX_PUBLIC_ARCHIVE_BYTES
MAX_TAR_RESIDENT_BYTES = MAX_PAYLOAD_BYTES
MAX_TAR_SCAN_TRANSIENT_BYTES = 8 * 1024 * 1024
MAX_TAR_PAX_HELPER_BYTES = MAX_PUBLIC_JSON_BYTES
MAX_TAR_HELPER_TOTAL_BYTES = MAX_PUBLIC_JSON_BYTES
MAX_TAR_PAX_RECORDS = MAX_ARCHIVE_MEMBERS
MAX_TAR_GNU_LONGNAME_BYTES = 4097
MAX_TAR_CONSECUTIVE_HELPERS = 8
MAX_TAR_EFFECTIVE_PAX_ASSOCIATIONS = 100_000
MIN_TAR_GZIP_RATIO_BYTES = 1024 * 1024
TAR_GZIP_INPUT_CHUNK_BYTES = 64 * 1024
TAR_GZIP_OUTPUT_CHUNK_BYTES = 64 * 1024
PRIVACY_TEXT_INPUT_CHUNK_BYTES = 128 * 1024
MAX_PRIVACY_NORMALIZATION_TAIL_CHARACTERS = 16 * 1024
MAX_PRIVACY_NORMALIZATION_TAIL_BYTES = 64 * 1024
_PRIVACY_NORMALIZATION_BOUNDARY_LOOKAHEAD_CHARACTERS = 2
_PRIVACY_NORMALIZATION_BOUNDARY_LOOKAHEAD_BYTES = 8
MAX_PRIVACY_NORMALIZATION_EXPANSION = 32
MAX_PRIVACY_BUILTIN_PATTERN_BYTES = 16 * 1024
MAX_FORBIDDEN_VALUE_CHARACTERS = 4096
MAX_CANONICAL_OPENING_BYTES = 16 * 1024
MAX_FORBIDDEN_PLAN_BYTES = 256 * 1024
MAX_FORBIDDEN_MATCHER_STATES = 256 * 1024
MAX_PRIVACY_SCAN_CANONICAL_BYTES = 64 * MAX_PAYLOAD_BYTES
_ALLOWED_TAR_PAX_KEYS = frozenset(
    {"path", "mtime", "comment", "hdrcharset", "uid", "gid", "uname", "gname"}
)
MAX_PRIVATE_JSON_DEPTH = 64
MAX_PRIVATE_JSON_NODES = 100_000
MAX_PRIVATE_JSON_STRING_BYTES = 64 * 1024
MAX_PRIVATE_JSON_TOTAL_STRING_BYTES = 16 * 1024 * 1024
MAX_PUBLIC_JSON_NODES = 5_000_000
MAX_IDENTITY_SCAN_VALUES = 4096
MAX_IDENTITY_SCAN_VALUE_BYTES = 4096
MAX_EMBEDDED_ARCHIVE_CANDIDATES = MAX_ARCHIVE_MEMBERS
MAX_TRACE_NUMBER_LITERAL_BYTES = 128
MAX_TRACE_NUMBER_DIGITS = 128
MAX_TRACE_NUMBER_EXPONENT = 128
_PAX_DECODED_BYTES_PER_SOURCE_BYTE = 4
_PAX_RECORD_OBJECT_RESERVE_BYTES = 512
# A physical association stores a pair plus the owning effective-items tuple.
# The logical parser retains a copied pax_headers mapping for the same pair.
_PAX_PHYSICAL_ASSOCIATION_RESERVE_BYTES = 96
_PAX_LOGICAL_ASSOCIATION_RESERVE_BYTES = 64
_PAX_PHYSICAL_ASSOCIATED_MEMBER_RESERVE_BYTES = 512
_PAX_LOGICAL_ASSOCIATED_MEMBER_RESERVE_BYTES = 512
# `tarfile.TarInfo._proc_pax` retains the complete helper body, a record
# slice, and partitioned raw fields while it creates its own decoded values.
# The physical ledger remains live during that reconciliation, so reserve all
# three raw copies plus a second decoded/object representation.
_PAX_LOGICAL_REPARSE_RAW_BYTES_PER_SOURCE_BYTE = 3
_PAX_LOGICAL_REPARSE_DECODED_BYTES_PER_SOURCE_BYTE = 4
_PAX_LOGICAL_REPARSE_RECORD_OBJECT_RESERVE_BYTES = 512

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

_PRIVACY_DECODE_BARRIER = "\udcff"
_PRIVACY_DECODE_ERROR_HANDLER = "gmes_issue123_privacy_barrier"


def _privacy_decode_error(error: UnicodeDecodeError) -> tuple[str, int]:
    return _PRIVACY_DECODE_BARRIER, error.end


codecs.register_error(_PRIVACY_DECODE_ERROR_HANDLER, _privacy_decode_error)

HALO_ANNOTATIONS = tuple(
    f"gmes::halo_{phase}_{operation}"
    for phase in ("magnetic", "electric")
    for operation in ("pack_launch", "exposed_wait", "boundary_unpack")
)
POLICY_WRITE_OPERATIONS = (
    "aten::masked_scatter_",
    "aten::index_copy_",
    "aten::scatter_",
)

TRACE_PHASES = {
    "X": "complete",
    "i": "instant",
    "I": "instant",
    "C": "counter",
    "M": "metadata",
    "s": "flow-start",
    "t": "flow-step",
    "f": "flow-end",
}
TRACE_EVENT_KEYS = {"name", "cat", "ph", "ts", "dur", "pid", "tid", "args", "id"}
METADATA_NAMES = {
    "process_name",
    "process_sort_index",
    "thread_name",
    "thread_sort_index",
}
KNOWN_CATEGORIES = {
    "",
    "cpu_op",
    "user_annotation",
    "cuda_runtime",
    "cuda_driver",
    "kernel",
    "gpu_memcpy",
    "gpu_memset",
    "ac2g",
    "trace",
}
ARG_KEYS = {
    "Bytes",
    "Total Allocated",
    "Total Reserved",
    "Total Active",
    "Total Requested",
    "Device Id",
    "Device Type",
    "Addr",
    "Allocation Id",
    "allocation_id",
    "stream",
    "Stream",
    "Stream Id",
    "correlation",
    "Correlation",
    "Correlation ID",
    "correlation_id",
    "External id",
    "Sequence number",
    "Fwd thread id",
    "Record function id",
    "Ev Idx",
    "graph_id",
    "Graph Id",
    "device",
    "context",
    "queued",
    "registers per thread",
    "shared memory",
    "blocks per SM",
    "est. achieved occupancy %",
    "grid",
    "block",
}
METADATA_ARG_KEYS = {"name", "sort_index"}

DTYPES = {
    "bool": ("?", 1, False),
    "int8": ("b", 1, False),
    "uint8": ("B", 1, False),
    "int16": ("h", 2, False),
    "uint16": ("H", 2, False),
    "int32": ("i", 4, False),
    "uint32": ("I", 4, False),
    "int64": ("q", 8, False),
    "uint64": ("Q", 8, False),
    "float16": ("e", 2, True),
    "float32": ("f", 4, True),
    "float64": ("d", 8, True),
    "complex64": ("ff", 8, True),
    "complex128": ("dd", 16, True),
}
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
OPERATIONS_V2_JOB_ORDER = (
    "Python 3.14 / ubuntu-latest",
    "Python 3.14 / macos-latest",
    "CodeQL / python",
    "CodeQL / c-cpp",
)

METADATA_SEMANTIC_TOKENS = {
    "process_name": "metadata-process-name",
    "process_sort_index": "metadata-process-sort-index",
    "thread_name": "metadata-thread-name",
    "thread_sort_index": "metadata-thread-sort-index",
}
HALO_SEMANTIC_TOKENS = {
    name: name.removeprefix("gmes::").replace("_", "-") for name in HALO_ANNOTATIONS
}
WRITE_SEMANTIC_TOKENS = {
    "aten::masked_scatter_": "indexed-write-masked-scatter",
    "aten::index_copy_": "indexed-write-index-copy",
    "aten::scatter_": "indexed-write-scatter",
}
SEMANTIC_TOKEN_PHASES = {
    **{token: {"metadata"} for token in METADATA_SEMANTIC_TOKENS.values()},
    **{token: {"complete"} for token in HALO_SEMANTIC_TOKENS.values()},
    **{token: {"complete"} for token in WRITE_SEMANTIC_TOKENS.values()},
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
SEMANTIC_EVENT_SIGNATURES = frozenset(
    (token, phase)
    for token, phases in SEMANTIC_TOKEN_PHASES.items()
    for phase in phases
)


class PrivacyError(ValueError):
    """Private evidence is unsafe, incomplete, or outside the trusted policy."""


class _TarOwnerPolicy(Enum):
    PUBLIC_NEUTRAL = "public-neutral"
    PRIVATE_STRUCTURAL = "private-structural"


class _PrivateSdistFailure(Enum):
    SOURCE_INVALID = "PRIVATE_SDIST_SOURCE_INVALID"
    SOURCE_CHANGED = "PRIVATE_SDIST_SOURCE_CHANGED"
    LIMIT_INVALID = "PRIVATE_SDIST_LIMIT_INVALID"
    READ_FAILED = "PRIVATE_SDIST_READ_FAILED"
    ARCHIVE_REJECTED = "PRIVATE_SDIST_ARCHIVE_REJECTED"
    CLOSE_FAILED = "PRIVATE_SDIST_CLOSE_FAILED"


class _PrivateSdistValidationError(PrivacyError):
    __slots__ = ("token",)

    def __init__(self, token: _PrivateSdistFailure) -> None:
        self.token = token
        super().__init__(token.value)


@dataclass(frozen=True, slots=True, repr=False)
class _PrivateSdistIdentity:
    device: int
    inode: int
    mode: int
    nlink: int
    size: int
    mtime_ns: int
    ctime_ns: int


_PRIVATE_SDIST_VIEW_SEAL = object()


@dataclass(frozen=True, slots=True, repr=False)
class _PrivateSdistReadView:
    fd: int = field(repr=False, compare=False)
    identity: _PrivateSdistIdentity = field(repr=False)
    _seal: object = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class _PrivateSdistValidationLimits:
    archive_bytes: int
    members: int
    member_bytes: int
    total_member_bytes: int
    helper_bytes: int
    helper_records: int
    consecutive_helpers: int
    effective_pax_associations: int
    normalization_work_bytes: int
    matcher_states: int


@dataclass(frozen=True, slots=True, repr=False)
class _TarOwnerValues:
    uid: int
    gid: int
    uname: str
    gname: str


@dataclass(frozen=True, slots=True)
class _PrivateSdistMember:
    name: str
    type_code: str
    size: int
    body_offset: int
    sha256: str


@dataclass(frozen=True, slots=True)
class _ValidatedPrivateSdist:
    archive_size: int
    archive_sha256: str
    members: tuple[_PrivateSdistMember, ...]
    total_member_bytes: int
    physical_ordinary_count: int
    logical_member_count: int


def _private_sdist_identity(stat_result: os.stat_result) -> _PrivateSdistIdentity:
    return _PrivateSdistIdentity(
        device=stat_result.st_dev,
        inode=stat_result.st_ino,
        mode=stat_result.st_mode,
        nlink=stat_result.st_nlink,
        size=stat_result.st_size,
        mtime_ns=stat_result.st_mtime_ns,
        ctime_ns=stat_result.st_ctime_ns,
    )


class _PrivateSdistLease:
    """Own a duplicated, redacted descriptor for one private validation."""

    __slots__ = ("_entered", "_exited", "_fd", "_identity", "_view")

    def __init__(self, fd: int, identity: _PrivateSdistIdentity) -> None:
        self._entered = False
        self._exited = False
        self._fd = fd
        self._identity = identity
        self._view = _PrivateSdistReadView(fd, identity, _PRIVATE_SDIST_VIEW_SEAL)

    def __repr__(self) -> str:
        return "<_PrivateSdistLease redacted>"

    __str__ = __repr__

    def __enter__(self) -> _PrivateSdistReadView:
        if self._entered or self._exited:
            raise _PrivateSdistValidationError(_PrivateSdistFailure.SOURCE_INVALID)
        self._entered = True
        return self._view

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if not self._entered or self._exited:
            raise _PrivateSdistValidationError(_PrivateSdistFailure.SOURCE_INVALID)
        self._exited = True
        close_failed = False
        try:
            os.close(self._fd)
        except OSError:
            close_failed = True
        if close_failed and exc_type is None:
            raise _PrivateSdistValidationError(
                _PrivateSdistFailure.CLOSE_FAILED
            ) from None
        return False


def _retain_private_sdist_fd(fd: int) -> _PrivateSdistLease:
    duplicated: int | None = None
    identity: _PrivateSdistIdentity | None = None
    invalid = type(fd) is not int or not 0 <= fd <= _PRIVATE_SDIST_MAX_FD
    if not invalid:
        try:
            duplicated = os.dup(fd)
            os.set_inheritable(duplicated, False)
            details = os.fstat(duplicated)
            identity = _private_sdist_identity(details)
            invalid = not _private_sdist_identity_is_valid(identity)
        except OSError:
            invalid = True
            details = None
    else:
        details = None
    if invalid or duplicated is None or details is None or identity is None:
        if duplicated is not None:
            try:
                os.close(duplicated)
            except OSError:
                pass
        raise _PrivateSdistValidationError(
            _PrivateSdistFailure.SOURCE_INVALID
        ) from None
    return _PrivateSdistLease(duplicated, identity)


def _default_private_sdist_validation_limits() -> _PrivateSdistValidationLimits:
    return _PrivateSdistValidationLimits(
        archive_bytes=MAX_PUBLIC_ARCHIVE_BYTES,
        members=MAX_ARCHIVE_MEMBERS,
        member_bytes=MAX_ARCHIVE_MEMBER_BYTES,
        total_member_bytes=MAX_ARCHIVE_TOTAL_BYTES,
        helper_bytes=MAX_TAR_HELPER_TOTAL_BYTES,
        helper_records=MAX_TAR_PAX_RECORDS,
        consecutive_helpers=MAX_TAR_CONSECUTIVE_HELPERS,
        effective_pax_associations=MAX_TAR_EFFECTIVE_PAX_ASSOCIATIONS,
        normalization_work_bytes=MAX_PRIVACY_SCAN_CANONICAL_BYTES,
        matcher_states=MAX_FORBIDDEN_MATCHER_STATES,
    )


def _private_sdist_limits_are_valid(limits: Any) -> bool:
    if type(limits) is not _PrivateSdistValidationLimits:
        return False
    maximum = _default_private_sdist_validation_limits()
    return all(
        type(getattr(limits, field_name)) is int
        and 0 < getattr(limits, field_name) <= getattr(maximum, field_name)
        for field_name in maximum.__dataclass_fields__
    )


class PrivateAuthorityCommitError(PrivacyError):
    """A final private leaf was linked but could not be durably reverified."""

    def __init__(self, message: str, *, committed: bool) -> None:
        super().__init__(message)
        self.committed = committed


class PrivateOpenings:
    """Optional caller-owned private commitment openings.

    Its representation is always redacted.  The private serializer below emits
    only the salt and binding metadata, never raw identities or arrays.
    """

    __slots__ = ("_salt", "_identities", "_arrays", "_populated")

    def __init__(self, salt: bytes | None = None) -> None:
        if salt is not None and (type(salt) is not bytes or len(salt) != 32):
            raise PrivacyError("commitment salt must be exactly 32 private bytes")
        self._salt: bytes | None = None if salt is None else bytes(salt)
        self._identities: dict[tuple[str, str], bytes] = {}
        self._arrays: dict[tuple[str, str, str, str, str], bytes] = {}
        self._populated = False

    def __repr__(self) -> str:
        return "<PrivateOpenings redacted>"

    __str__ = __repr__

    def salt_for_private_verification(self) -> bytes:
        """Return a defensive copy of the private salt."""

        if self._salt is None:
            raise PrivacyError("private openings have not been populated")
        return bytes(self._salt)

    def identity_for_private_verification(self, scope: str, name: str) -> Any:
        return json.loads(self._identities[(scope, name)].decode("utf-8"))

    def array_for_private_verification(
        self, scope: str, case: str, capture: str, array: str, role: str
    ) -> bytes:
        return bytes(self._arrays[(scope, case, capture, array, role)])


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise PrivacyError(message)


def _utf8_bytes(value: str, label: str) -> bytes:
    encoding_failed = False
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        encoding_failed = True
        encoded = b""
    if encoding_failed:
        raise PrivacyError(f"{label} is not valid Unicode") from None
    return encoded


def _exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    keys = set(value)
    _require(keys == expected, f"{label} fields differ")
    _require(all(type(key) is str for key in value), f"{label} keys must be strings")
    return value


def _strict_json(raw: bytes, label: str) -> Any:
    _require(type(raw) is bytes, f"{label} must be bytes")
    _require(len(raw) <= MAX_TRACE_BYTES, f"{label} exceeds its byte bound")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise PrivacyError(f"{label} repeats a JSON key")
            result[key] = value
        return result

    def exact_float(token: str) -> Decimal:
        _require(
            len(token.encode("ascii")) <= MAX_TRACE_NUMBER_LITERAL_BYTES,
            f"{label} contains an oversized number literal",
        )
        value = None
        try:
            value = Decimal(token)
        except InvalidOperation:
            pass
        if value is None:
            raise PrivacyError(f"{label} contains an invalid number literal") from None
        sign, digits, exponent = value.as_tuple()
        del sign
        _require(
            value.is_finite()
            and len(digits) <= MAX_TRACE_NUMBER_DIGITS
            and abs(exponent) <= MAX_TRACE_NUMBER_EXPONENT,
            f"{label} contains an out-of-bounds number literal",
        )
        return value

    decode_failed = False
    parse_failed = False
    try:
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=pairs,
            parse_float=exact_float,
            parse_constant=lambda token: (_ for _ in ()).throw(
                PrivacyError(f"{label} contains a non-finite number")
            ),
        )
    except PrivacyError:
        raise
    except UnicodeDecodeError:
        decode_failed = True
        value = None
    except ValueError, RecursionError:
        parse_failed = True
        value = None
    if decode_failed:
        raise PrivacyError(f"{label} is not UTF-8 JSON") from None
    if parse_failed:
        raise PrivacyError(f"{label} is not strict JSON") from None
    return value


def canonical_json_bytes(value: Any, *, label: str = "public document") -> bytes:
    """Encode canonical public JSON and apply the last-resort privacy scan."""

    try:
        raw = (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise PrivacyError(f"{label} is not canonical JSON data") from error
    scan_public_bytes(raw, label=label)
    _scan_json_metadata(raw, label, (), require_json=True)
    return raw


def _portable_path(value: Any, label: str) -> str:
    _require(type(value) is str and bool(value), f"{label} must be a path string")
    _require(
        len(_utf8_bytes(value, label)) <= 4096
        and "\\" not in value
        and not any(
            ord(character) < 0x20 or ord(character) == 0x7F for character in value
        ),
        f"{label} is not portable",
    )
    _require(unicodedata.normalize("NFC", value) == value, f"{label} is not NFC")
    path = PurePosixPath(value)
    _require(
        not path.is_absolute()
        and value == path.as_posix()
        and all(part not in {"", ".", ".."} for part in path.parts),
        f"{label} is not canonical and relative",
    )
    alias = unicodedata.normalize("NFKC", value)
    _require(
        alias.count("/") == value.count("/")
        and "\\" not in alias
        and re.match(r"(?i)^[a-z]:", alias) is None,
        f"{label} is not portable, canonical, and relative",
    )
    alias_path = PurePosixPath(alias)
    _require(
        not alias_path.is_absolute()
        and alias == alias_path.as_posix()
        and all(part not in {"", ".", ".."} for part in alias_path.parts),
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
    return value


def _safe_name(value: Any, label: str) -> str:
    _require(type(value) is str and SAFE_NAME_RE.fullmatch(value), f"{label} is unsafe")
    _require(".." not in value, f"{label} is unsafe")
    return value


def _job_name(value: Any, label: str) -> str:
    _require(type(value) is str and JOB_NAME_RE.fullmatch(value), f"{label} is unsafe")
    _require(
        unicodedata.normalize("NFC", value) == value
        and ".." not in value
        and not any(ord(character) < 0x20 for character in value),
        f"{label} is unsafe",
    )
    scan_public_bytes(_utf8_bytes(value, label), label=label)
    return value


def _semantic_name(value: Any, label: str) -> str:
    _require(type(value) is str and 0 < len(value) <= 1024, f"{label} is unsafe")
    _require(
        unicodedata.normalize("NFC", value) == value
        and "\\" not in value
        and "\x00" not in value,
        f"{label} is unsafe",
    )
    _require(len(_utf8_bytes(value, label)) <= 4096, f"{label} is unsafe")
    parts = value.split("/")
    _require(
        all(
            part not in {"", ".", ".."}
            and ".." not in part
            and SEMANTIC_SEGMENT_RE.fullmatch(part)
            for part in parts
        ),
        f"{label} is unsafe",
    )
    return value


def _number(value: Any, label: str, *, positive: bool = False) -> int | float | Decimal:
    _require(type(value) in {int, float, Decimal}, f"{label} must be a JSON number")
    if type(value) is Decimal:
        finite = value.is_finite()
    else:
        try:
            finite = math.isfinite(value)
        except OverflowError as error:
            raise PrivacyError(f"{label} is outside its numeric bound") from error
    _require(finite, f"{label} must be finite")
    if positive:
        _require(value > 0, f"{label} must be positive")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    _require(
        type(value) is int and value >= 0, f"{label} must be a nonnegative integer"
    )
    return value


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _private_json_bytes(value: Any, label: str) -> bytes:
    """Validate and canonically copy a bounded private JSON value."""

    state = {"nodes": 0, "string_bytes": 0}

    def copy_value(item: Any, depth: int) -> Any:
        _require(depth <= MAX_PRIVATE_JSON_DEPTH, f"{label} is too deeply nested")
        state["nodes"] += 1
        _require(
            state["nodes"] <= MAX_PRIVATE_JSON_NODES,
            f"{label} has too many JSON nodes",
        )
        if item is None or type(item) is bool:
            return item
        if type(item) is int:
            _require(
                -(1 << 63) <= item < (1 << 63),
                f"{label} integer is outside its bound",
            )
            return item
        if type(item) is float:
            _require(math.isfinite(item), f"{label} contains a non-finite number")
            return item
        if type(item) is str:
            encoded = _utf8_bytes(item, label)
            _require(
                unicodedata.normalize("NFC", item) == item
                and "\x00" not in item
                and len(encoded) <= MAX_PRIVATE_JSON_STRING_BYTES,
                f"{label} contains an invalid string",
            )
            state["string_bytes"] += len(encoded)
            _require(
                state["string_bytes"] <= MAX_PRIVATE_JSON_TOTAL_STRING_BYTES,
                f"{label} contains too much string data",
            )
            return item
        if type(item) is list:
            return [copy_value(child, depth + 1) for child in item]
        if type(item) is dict:
            result = {}
            for key, child in item.items():
                _require(type(key) is str, f"{label} object keys must be strings")
                copied_key = copy_value(key, depth + 1)
                result[copied_key] = copy_value(child, depth + 1)
            return result
        raise PrivacyError(f"{label} contains a non-JSON value")

    copied = copy_value(value, 0)
    try:
        return json.dumps(
            copied,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise PrivacyError(f"{label} is not canonical JSON") from error


def _privacy_nfkc_casefold(value: str) -> str:
    return unicodedata.normalize(
        "NFKC", unicodedata.normalize("NFKC", value).casefold()
    )


@dataclass(frozen=True, slots=True)
class _CanonicalByteMatcher:
    transitions: tuple[Mapping[int, int], ...]
    failures: tuple[int, ...]
    terminal: tuple[bool, ...]

    def scan(self, raw: bytes, state: int = 0) -> tuple[int, bool]:
        for byte in raw:
            while state and byte not in self.transitions[state]:
                state = self.failures[state]
            state = self.transitions[state].get(byte, 0)
            if self.terminal[state]:
                return state, True
        return state, False


@dataclass(frozen=True, slots=True)
class _ForbiddenValuePlan:
    matcher: _CanonicalByteMatcher
    maximum_needle_bytes: int
    value_count: int
    construction_work: int


@dataclass(slots=True)
class _PrivacyScanContext:
    plan: _ForbiddenValuePlan
    canonical_bytes: int = 0
    maximum_canonical_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.maximum_canonical_bytes is None:
            self.maximum_canonical_bytes = MAX_PRIVACY_SCAN_CANONICAL_BYTES
        _require(
            type(self.maximum_canonical_bytes) is int
            and self.maximum_canonical_bytes > 0,
            "private opening scan exceeds its bound",
        )
        self.charge(self.plan.construction_work)

    def charge(self, amount: int) -> None:
        _require(
            amount <= self.maximum_canonical_bytes - self.canonical_bytes,
            "private opening scan exceeds its bound",
        )
        self.canonical_bytes += amount


def _build_canonical_matcher(
    needles: Sequence[bytes],
    *,
    maximum_states: int | None = None,
) -> _CanonicalByteMatcher:
    if maximum_states is None:
        maximum_states = MAX_FORBIDDEN_MATCHER_STATES
    transitions: list[dict[int, int]] = [{}]
    failures = [0]
    terminal = [False]
    for needle in needles:
        state = 0
        for byte in needle:
            following = transitions[state].get(byte)
            if following is None:
                _require(
                    len(transitions) < maximum_states,
                    "private opening scan exceeds its bound",
                )
                following = len(transitions)
                transitions[state][byte] = following
                transitions.append({})
                failures.append(0)
                terminal.append(False)
            state = following
        terminal[state] = True

    queue = list(transitions[0].values())
    cursor = 0
    while cursor < len(queue):
        state = queue[cursor]
        cursor += 1
        for byte, following in transitions[state].items():
            queue.append(following)
            failure = failures[state]
            while failure and byte not in transitions[failure]:
                failure = failures[failure]
            failures[following] = transitions[failure].get(byte, 0)
            terminal[following] = terminal[following] or terminal[failures[following]]
    return _CanonicalByteMatcher(
        tuple(MappingProxyType(item) for item in transitions),
        tuple(failures),
        tuple(terminal),
    )


def _prepare_forbidden_value_plan(
    forbidden_values: Any,
    *,
    maximum_matcher_states: int | None = None,
) -> _ForbiddenValuePlan:
    if maximum_matcher_states is None:
        maximum_matcher_states = MAX_FORBIDDEN_MATCHER_STATES
    _require(
        type(maximum_matcher_states) is int and maximum_matcher_states > 0,
        "private opening scan exceeds its bound",
    )
    materialization_failed = isinstance(forbidden_values, (str, bytes, bytearray))
    iterator = None
    if not materialization_failed:
        try:
            iterator = iter(forbidden_values)
        except Exception:
            materialization_failed = True
    if materialization_failed:
        raise PrivacyError("private opening scan exceeds its bound") from None

    values_list: list[Any] = []
    overflow = False
    while True:
        exhausted = False
        item_failed = False
        try:
            value = next(iterator)
        except StopIteration:
            exhausted = True
        except Exception:
            item_failed = True
        if item_failed:
            raise PrivacyError("private opening scan exceeds its bound") from None
        if exhausted:
            break
        if len(values_list) == MAX_IDENTITY_SCAN_VALUES:
            overflow = True
            break
        values_list.append(value)
    _require(not overflow, "private opening scan exceeds its bound")
    values = tuple(values_list)

    needles: set[bytes] = set()
    construction_work = 0
    for value in values:
        _require(
            type(value) is str and 3 <= len(value) <= MAX_FORBIDDEN_VALUE_CHARACTERS,
            "private opening scan exceeds its bound",
        )
        encoding_failed = False
        try:
            original = value.encode("utf-8")
        except UnicodeEncodeError:
            encoding_failed = True
            original = b""
        if encoding_failed:
            raise PrivacyError("private opening scan exceeds its bound") from None
        _require(
            len(original) <= MAX_IDENTITY_SCAN_VALUE_BYTES,
            "private opening scan exceeds its bound",
        )
        for normalization in ("NFC", "NFD", "NFKC", "NFKD"):
            normalized = unicodedata.normalize(normalization, value)
            for candidate in (
                normalized,
                normalized.casefold(),
                normalized.lower(),
                normalized.upper(),
            ):
                encoded = _privacy_nfkc_casefold(candidate).encode("utf-8")
                _require(
                    0 < len(encoded) <= MAX_CANONICAL_OPENING_BYTES,
                    "private opening scan exceeds its bound",
                )
                _require(
                    len(encoded) <= MAX_FORBIDDEN_PLAN_BYTES - construction_work,
                    "private opening scan exceeds its bound",
                )
                construction_work += len(encoded)
                needles.add(encoded)
    ordered = tuple(sorted(needles))
    matcher = _build_canonical_matcher(ordered, maximum_states=maximum_matcher_states)
    return _ForbiddenValuePlan(
        matcher,
        max(map(len, ordered), default=0),
        len(values),
        construction_work + len(matcher.transitions),
    )


def _privacy_scan_context(
    forbidden_values: Any,
    *,
    maximum_canonical_bytes: int | None = None,
    maximum_matcher_states: int | None = None,
) -> _PrivacyScanContext:
    if maximum_canonical_bytes is None:
        maximum_canonical_bytes = MAX_PRIVACY_SCAN_CANONICAL_BYTES
    if maximum_matcher_states is None:
        maximum_matcher_states = MAX_FORBIDDEN_MATCHER_STATES
    if isinstance(forbidden_values, _PrivacyScanContext):
        _require(
            forbidden_values.maximum_canonical_bytes <= maximum_canonical_bytes,
            "private opening scan exceeds its bound",
        )
        return forbidden_values
    return _PrivacyScanContext(
        _prepare_forbidden_value_plan(
            forbidden_values, maximum_matcher_states=maximum_matcher_states
        ),
        maximum_canonical_bytes=maximum_canonical_bytes,
    )


def _scan_pattern_view(
    raw: bytes,
    label: str,
    *,
    assignment_exempt_starts: frozenset[int] = frozenset(),
) -> None:
    for pattern, description in (
        (UUID_RE, "UUID"),
        (PRIVATE_ROOT_RE, "personal/worktree path"),
        (SECRET_RE, "credential or signing material"),
    ):
        _require(pattern.search(raw) is None, f"{label} contains {description}")
    for match in EMAIL_RE.finditer(raw):
        local, domain = match.group().rsplit(b"@", 1)
        _require(
            len(local) > 64 or len(domain) > 253 or len(match.group()) > 254,
            f"{label} contains email address",
        )
    _require(
        PRIVATE_JSON_KEY_RE.search(raw) is None,
        f"{label} contains a private metadata key",
    )
    for pattern in (PRIVATE_ASSIGNMENT_RE, PRIVATE_ENV_ASSIGNMENT_RE):
        _require(
            all(
                match.start() in assignment_exempt_starts
                for match in pattern.finditer(raw)
            ),
            f"{label} contains an environment or identity assignment",
        )
    _require(
        GITHUB_WORKSPACE_ASSIGNMENT_RE.search(raw) is None,
        f"{label} contains an environment value",
    )


def _normalization_non_inert_codepoints() -> frozenset[int]:
    non_inert: set[int] = set()
    composition_participants: set[int] = set()
    for codepoint in range(0x110000):
        character = chr(codepoint)
        if (
            unicodedata.combining(character) != 0
            or unicodedata.category(character).startswith("M")
            or unicodedata.normalize("NFKD", character) != character
            or character.casefold() != character
        ):
            non_inert.add(codepoint)
        decomposition = unicodedata.decomposition(character)
        if decomposition and not decomposition.startswith("<"):
            parts = decomposition.split()
            if len(parts) == 2:
                composition_participants.update(int(part, 16) for part in parts)
    # Python's decomposition table omits algorithmic Hangul composition.
    composition_participants.update(range(0x1100, 0x1113))
    composition_participants.update(range(0x1161, 0x1176))
    composition_participants.update(range(0x11A8, 0x11C3))
    composition_participants.update(range(0xAC00, 0xD7A4, 28))
    non_inert.update(composition_participants)
    return frozenset(non_inert)


_NORMALIZATION_NON_INERT_CODEPOINTS = _normalization_non_inert_codepoints()


def _starts_normalization_starter_group(value: str, index: int) -> bool:
    # A boundary before two consecutive ASCII scalars is stable across both
    # NFKC passes: even if later text composes with the second scalar, the first
    # remains an ASCII CCC=0 starter after casefolding.  A single ASCII scalar
    # is insufficient because it may compose with a following nonstarter.  A
    # single scalar is also a stable barrier when it is NFKD/casefold-invariant,
    # CCC=0, not a mark, and absent from every canonical/Hangul composition pair.
    return ord(value[index]) not in _NORMALIZATION_NON_INERT_CODEPOINTS or (
        index + 1 < len(value)
        and ord(value[index]) < 128
        and ord(value[index + 1]) < 128
    )


def _normalization_prefix_end(value: str) -> int:
    # Retain four complete starter groups.  This is conservative for both
    # canonical composition and the three-scalar Hangul L/V/T composition.
    recent_starters: list[int] = []
    starter_count = 0
    for index in range(len(value)):
        if _starts_normalization_starter_group(value, index):
            starter_count += 1
            recent_starters.append(index)
            if len(recent_starters) > 4:
                recent_starters.pop(0)
    return recent_starters[0] if starter_count > 4 else 0


def _scan_decoded_privacy_view(
    raw: bytes | bytearray,
    encoding: str,
    offset: int,
    label: str,
    context: _PrivacyScanContext,
    *,
    check_patterns: bool,
    check_openings: bool,
    mask_spans: Sequence[tuple[int, int]] = (),
) -> None:
    decoder = codecs.getincrementaldecoder(encoding)(
        errors=_PRIVACY_DECODE_ERROR_HANDLER
    )
    normalization_tail = ""
    normalization_tail_bytes = 0
    normalization_lookahead: list[str] = []
    normalization_lookahead_bytes = 0
    matcher_state = 0
    pattern_tail = b""
    in_whitespace = False

    def emit(text: str) -> None:
        nonlocal matcher_state, pattern_tail, in_whitespace
        if not text:
            return
        canonical_text = _privacy_nfkc_casefold(text)
        encoded = canonical_text.encode("utf-8")
        source_size = len(text.encode("utf-8", errors="replace"))
        _require(
            len(encoded) <= max(64, source_size * MAX_PRIVACY_NORMALIZATION_EXPANSION),
            f"{label} privacy normalization exceeds its bound",
        )
        context.charge(len(encoded))
        if check_openings and context.plan.value_count:
            context.charge(len(encoded))
            matcher_state, matched = context.plan.matcher.scan(encoded, matcher_state)
            _require(not matched, f"{label} contains a private opening")
        if check_patterns:
            compacted: list[str] = []
            for character in canonical_text:
                if character.isspace():
                    if not in_whitespace:
                        compacted.append(" ")
                    in_whitespace = True
                else:
                    compacted.append(character)
                    in_whitespace = False
            candidate = pattern_tail + "".join(compacted).encode("utf-8")
            _scan_pattern_view(candidate, f"{label} normalized text")
            pattern_tail = candidate[-MAX_PRIVACY_BUILTIN_PATTERN_BYTES:]

    def accept_segment(decoded: str, *, stable_boundary: bool) -> None:
        nonlocal normalization_tail, normalization_tail_bytes
        nonlocal normalization_lookahead, normalization_lookahead_bytes

        def retain(character: str, *, already_charged: bool) -> None:
            nonlocal normalization_tail, normalization_tail_bytes
            encoded = character.encode("utf-8")
            _require(
                len(normalization_tail) < MAX_PRIVACY_NORMALIZATION_TAIL_CHARACTERS
                and len(encoded)
                <= MAX_PRIVACY_NORMALIZATION_TAIL_BYTES - normalization_tail_bytes,
                f"{label} privacy normalization exceeds its bound",
            )
            if not already_charged:
                context.charge(1 + len(encoded))
            normalization_tail += character
            normalization_tail_bytes += len(encoded)

        def emit_stable_prefix() -> None:
            nonlocal normalization_tail, normalization_tail_bytes
            end = _normalization_prefix_end(normalization_tail)
            if end:
                emit(normalization_tail[:end])
                normalization_tail = normalization_tail[end:]
                normalization_tail_bytes = len(normalization_tail.encode("utf-8"))

        def release_lookahead() -> None:
            nonlocal normalization_tail, normalization_tail_bytes
            nonlocal normalization_lookahead, normalization_lookahead_bytes
            lookahead = "".join(normalization_lookahead)
            _require(
                _starts_normalization_starter_group(lookahead, 0),
                f"{label} privacy normalization exceeds its bound",
            )
            emit(normalization_tail)
            normalization_tail = ""
            normalization_tail_bytes = 0
            normalization_lookahead = []
            normalization_lookahead_bytes = 0
            for character in lookahead:
                retain(character, already_charged=True)
                emit_stable_prefix()

        cursor = 0
        while cursor < len(decoded):
            if not normalization_lookahead:
                scalar_room = MAX_PRIVACY_NORMALIZATION_TAIL_CHARACTERS - len(
                    normalization_tail
                )
                byte_room = (
                    MAX_PRIVACY_NORMALIZATION_TAIL_BYTES - normalization_tail_bytes
                )
                if scalar_room > 0 and byte_room > 0:
                    take = min(1024, scalar_room, len(decoded) - cursor)
                    piece = decoded[cursor : cursor + take]
                    piece_bytes = piece.encode("utf-8")
                    while len(piece_bytes) > byte_room and take > 1:
                        take //= 2
                        piece = decoded[cursor : cursor + take]
                        piece_bytes = piece.encode("utf-8")
                    if len(piece_bytes) <= byte_room:
                        context.charge(len(piece) + len(piece_bytes))
                        normalization_tail += piece
                        normalization_tail_bytes += len(piece_bytes)
                        cursor += take
                        emit_stable_prefix()
                        continue
            character = decoded[cursor]
            cursor += 1
            encoded = character.encode("utf-8")
            if normalization_lookahead:
                _require(
                    len(normalization_lookahead)
                    < _PRIVACY_NORMALIZATION_BOUNDARY_LOOKAHEAD_CHARACTERS
                    and normalization_lookahead_bytes + len(encoded)
                    <= _PRIVACY_NORMALIZATION_BOUNDARY_LOOKAHEAD_BYTES,
                    f"{label} privacy normalization exceeds its bound",
                )
                context.charge(1 + len(encoded))
                normalization_lookahead.append(character)
                normalization_lookahead_bytes += len(encoded)
                lookahead = "".join(normalization_lookahead)
                if _starts_normalization_starter_group(lookahead, 0):
                    release_lookahead()
                elif len(normalization_lookahead) == (
                    _PRIVACY_NORMALIZATION_BOUNDARY_LOOKAHEAD_CHARACTERS
                ):
                    raise PrivacyError(
                        f"{label} privacy normalization exceeds its bound"
                    ) from None
                continue
            if (
                len(normalization_tail) == MAX_PRIVACY_NORMALIZATION_TAIL_CHARACTERS
                or normalization_tail_bytes + len(encoded)
                > MAX_PRIVACY_NORMALIZATION_TAIL_BYTES
            ):
                _require(
                    len(encoded) <= _PRIVACY_NORMALIZATION_BOUNDARY_LOOKAHEAD_BYTES,
                    f"{label} privacy normalization exceeds its bound",
                )
                context.charge(1 + len(encoded))
                normalization_lookahead = [character]
                normalization_lookahead_bytes = len(encoded)
                if _starts_normalization_starter_group(character, 0):
                    release_lookahead()
                elif ord(character) >= 128:
                    raise PrivacyError(
                        f"{label} privacy normalization exceeds its bound"
                    ) from None
                continue
        if stable_boundary:
            _require(
                not normalization_lookahead,
                f"{label} privacy normalization exceeds its bound",
            )
        if stable_boundary and normalization_tail:
            emit(normalization_tail)
            normalization_tail = ""
            normalization_tail_bytes = 0

    def reset_at_barrier() -> None:
        nonlocal matcher_state, pattern_tail, in_whitespace
        accept_segment("", stable_boundary=True)
        matcher_state = 0
        pattern_tail = b""
        in_whitespace = False

    def accept(decoded: str, *, final: bool) -> None:
        segments = decoded.split(_PRIVACY_DECODE_BARRIER)
        for segment in segments[:-1]:
            accept_segment(segment, stable_boundary=True)
            reset_at_barrier()
        accept_segment(segments[-1], stable_boundary=final)

    chunk_bytes = max(2, PRIVACY_TEXT_INPUT_CHUNK_BYTES)
    if encoding.startswith("utf-16"):
        chunk_bytes -= chunk_bytes % 2
    position = offset
    mask_index = 0
    while position < len(raw):
        end = min(len(raw), position + chunk_bytes)
        supplied: bytes | bytearray | memoryview = memoryview(raw)[position:end]
        if mask_spans:
            while (
                mask_index < len(mask_spans) and mask_spans[mask_index][1] <= position
            ):
                mask_index += 1
            index = mask_index
            masked: bytearray | None = None
            while index < len(mask_spans) and mask_spans[index][0] < end:
                if masked is None:
                    masked = bytearray(supplied)
                start, stop = mask_spans[index]
                masked[max(start, position) - position : min(stop, end) - position] = (
                    b"\0" * (min(stop, end) - max(start, position))
                )
                index += 1
            if masked is not None:
                supplied = masked
        context.charge(len(supplied))
        accept(decoder.decode(supplied, final=False), final=False)
        position = end
    accept(decoder.decode(b"", final=True), final=True)


def _scan_nfkc_utf8(
    raw: bytes | bytearray,
    label: str,
    context: _PrivacyScanContext,
    *,
    check_patterns: bool = True,
    check_openings: bool = True,
    mask_spans: Sequence[tuple[int, int]] = (),
) -> None:
    _scan_decoded_privacy_view(
        raw,
        "utf-8",
        0,
        label,
        context,
        check_patterns=check_patterns,
        check_openings=check_openings,
        mask_spans=mask_spans,
    )


def _scan_patterns(raw: bytes, label: str) -> None:
    _scan_pattern_view(raw, label)


def _scan_utf16_ascii(
    raw: bytes | bytearray,
    label: str,
    context: _PrivacyScanContext,
    *,
    check_openings: bool = True,
) -> None:
    for encoding in ("utf-16-le", "utf-16-be"):
        for offset in (0, 1):
            _scan_decoded_privacy_view(
                raw,
                encoding,
                offset,
                label,
                context,
                check_patterns=True,
                check_openings=check_openings,
            )


def _scan_public_bytes_with_context(
    raw: bytes,
    label: str,
    context: _PrivacyScanContext,
    *,
    check_openings: bool,
) -> None:
    _scan_patterns(raw, label)
    _scan_nfkc_utf8(
        raw,
        label,
        context,
        check_openings=check_openings,
    )
    _scan_utf16_ascii(raw, label, context, check_openings=check_openings)


def scan_public_bytes(
    raw: bytes,
    *,
    label: str = "public bytes",
    forbidden_values: Any = (),
    _check_openings: bool = True,
) -> None:
    """Reject text or binary bytes containing personal/device metadata."""

    _require(type(raw) is bytes, f"{label} must be immutable bytes")
    context = _privacy_scan_context(forbidden_values)
    _scan_public_bytes_with_context(
        raw,
        label,
        context,
        check_openings=_check_openings,
    )


def _payload_name(value: Any, label: str, media_type: str | None = None) -> str:
    name = _portable_path(value, label)
    lowered = unicodedata.normalize("NFKC", name).casefold()
    _require(
        all(
            WITHHELD_PAYLOAD_RE.search(part) is None
            for part in PurePosixPath(lowered).parts
        ),
        f"{label} names withheld private evidence",
    )
    _require(
        not lowered.endswith((".npy", ".npz")),
        f"{label} names a withheld array container",
    )
    if media_type is not None:
        _require(media_type in PAYLOAD_MEDIA_TYPES, f"{label} media type differs")
        _require(
            lowered.endswith(PAYLOAD_SUFFIXES[media_type]),
            f"{label} suffix does not match its media type",
        )
    return name


def _scan_json_metadata(
    raw: bytes,
    label: str,
    forbidden_values: Any,
    *,
    require_json: bool,
) -> None:
    context = _privacy_scan_context(forbidden_values)
    stripped = raw.lstrip()
    if not stripped.startswith((b"{", b"[")):
        _require(not require_json, f"{label} is not JSON")
        return
    _require(len(raw) <= MAX_PUBLIC_JSON_BYTES, f"{label} JSON exceeds its byte bound")

    def pairs(items):
        result = {}
        for key, value in items:
            _require(
                type(key) is str and key not in result, f"{label} JSON keys differ"
            )
            result[key] = value
        return result

    parse_failed = False
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                PrivacyError(f"{label} contains {token}")
            ),
        )
    except PrivacyError:
        raise
    except UnicodeDecodeError, ValueError, RecursionError:
        parse_failed = True
        value = None
    if parse_failed:
        if require_json:
            raise PrivacyError(f"{label} is not strict JSON") from None
        return

    state = {"nodes": 0}

    def walk(item: Any, depth: int) -> None:
        _require(depth <= MAX_PRIVATE_JSON_DEPTH, f"{label} JSON is too deep")
        state["nodes"] += 1
        _require(state["nodes"] <= MAX_PUBLIC_JSON_NODES, f"{label} JSON is too large")
        if isinstance(item, Mapping):
            for key, child in item.items():
                key_raw = _utf8_bytes(key, "JSON object key")
                scan_public_bytes(
                    key_raw,
                    label="JSON object key",
                    forbidden_values=context,
                )
                normalized_key = _privacy_nfkc_casefold(key)
                if PRIVATE_METADATA_KEY_RE.fullmatch(normalized_key):
                    _require(
                        child in (None, "", [], {}),
                        f"{label} contains a private metadata value",
                    )
                walk(child, depth + 1)
        elif isinstance(item, list):
            for child in item:
                walk(child, depth + 1)
        elif type(item) is str:
            scan_public_bytes(
                _utf8_bytes(item, "JSON string value"),
                label="JSON string value",
                forbidden_values=context,
            )

    walk(value, 0)


def _scan_contextual_json(
    name: str, raw: bytes, label: str, forbidden_values: Sequence[str]
) -> None:
    _scan_json_metadata(
        raw,
        label,
        forbidden_values,
        require_json=name.casefold().endswith(".json"),
    )


def _archive_alias(name: str) -> str:
    normalized = unicodedata.normalize("NFKC", name)
    _require(
        normalized.count("/") == name.count("/") and "\\" not in normalized,
        "archive member alias introduces a separator",
    )
    _require(
        all(part == part.rstrip(" .") for part in normalized.split("/")),
        "archive member alias has a platform suffix",
    )
    return _payload_name(normalized, "archive member alias").casefold()


def _archive_member_name(value: Any, label: str) -> tuple[str, bool]:
    _require(type(value) is str and bool(value), f"{label} must be a path string")
    is_directory = value.endswith("/")
    stripped = value[:-1] if is_directory else value
    _require(
        not is_directory or bool(stripped) and not stripped.endswith("/"),
        f"{label} is not canonical",
    )
    member = _payload_name(stripped, label)
    _require(
        all(part == part.rstrip(" .") for part in member.split("/")),
        f"{label} has a platform alias",
    )
    return member, is_directory


def _looks_like_zip(raw: bytes) -> bool:
    return raw.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"))


def _is_zip_container(raw: bytes) -> bool:
    try:
        return zipfile.is_zipfile(io.BytesIO(raw))
    except OSError, ValueError:
        return False


def _looks_like_gzip(raw: bytes) -> bool:
    return raw.startswith(b"\x1f\x8b")


def _looks_like_tar(raw: bytes) -> bool:
    return len(raw) >= 265 and raw[257:262] == b"ustar"


def _has_embedded_gzip(raw: bytes) -> bool:
    cursor = 0
    candidate_count = 0
    while True:
        offset = raw.find(b"\x1f\x8b\x08", cursor)
        if offset < 0:
            return False
        candidate_count += 1
        if candidate_count > MAX_EMBEDDED_ARCHIVE_CANDIDATES:
            return True
        if offset + 18 <= len(raw) and not (raw[offset + 3] & 0xE0):
            return True
        cursor = offset + 1


def _tar_checksum_matches(header: bytes, *, require_ustar: bool = True) -> bool:
    if len(header) != 512 or require_ustar and header[257:262] != b"ustar":
        return False
    field = header[148:156].strip(b" \x00")
    if not field or any(byte not in b"01234567" for byte in field):
        return False
    stored = int(field, 8)
    spaced = header[:148] + b" " * 8 + header[156:]
    unsigned = sum(spaced)
    signed = sum(byte if byte < 128 else byte - 256 for byte in spaced)
    return stored in {unsigned, signed}


def _has_embedded_tar(raw: bytes) -> bool:
    candidate_count = 0
    for match in TAR_CHECKSUM_FIELD_RE.finditer(raw):
        offset = match.start() - 148
        if offset < 0 or offset + 512 > len(raw):
            continue
        candidate_count += 1
        if candidate_count > MAX_ARCHIVE_MEMBERS:
            return True
        header = raw[offset : offset + 512]
        if not _tar_checksum_matches(header, require_ustar=False):
            continue
        try:
            info = tarfile.TarInfo.frombuf(
                header, encoding="utf-8", errors="surrogateescape"
            )
        except OSError, tarfile.HeaderError, ValueError, OverflowError:
            continue
        if bool(info.name) and type(info.size) is int and info.size >= 0:
            return True
    return False


def _contains_archive(raw: bytes) -> bool:
    return _is_zip_container(raw) or _has_embedded_gzip(raw) or _has_embedded_tar(raw)


def _reject_array_or_nested_archive(member: str, raw: bytes, label: str) -> None:
    lowered = member.casefold()
    _require(b"\x93NUMPY" not in raw, f"{label} contains NPY array bytes")
    _require(
        not lowered.endswith((".npy", ".npz")),
        f"{label} contains a withheld array container",
    )
    nested_suffix = lowered.endswith((".zip", ".whl", ".tar", ".tar.gz", ".tgz", ".gz"))
    _require(
        not (nested_suffix or _contains_archive(raw)),
        f"{label} contains a nested archive",
    )


def _zip_extra_fields(raw: bytes, label: str) -> None:
    _require(raw == b"", f"{label} contains unsupported ZIP metadata")


_ZIP_LIBRARY_ERRORS = (
    EOFError,
    OSError,
    OverflowError,
    RecursionError,
    RuntimeError,
    TypeError,
    ValueError,
    struct.error,
    zipfile.BadZipFile,
    zipfile.LargeZipFile,
    zlib.error,
)


def _zip_unpack(
    format_string: str, raw: bytes, offset: int, message: str
) -> tuple[Any, ...]:
    fields = None
    try:
        fields = struct.unpack_from(format_string, raw, offset)
    except struct.error:
        pass
    if fields is None:
        raise PrivacyError(message) from None
    return fields


def _zip_decode(raw: bytes, encoding: str, message: str) -> str:
    value = None
    try:
        value = raw.decode(encoding)
    except UnicodeDecodeError:
        pass
    if value is None:
        raise PrivacyError(message) from None
    return value


def _validate_zip_member_stream(
    raw: bytes, info: zipfile.ZipInfo, member_raw: bytes, label: str
) -> None:
    name_size, extra_size = _zip_unpack(
        "<HH",
        raw,
        info.header_offset + 26,
        f"{label} local header is truncated",
    )
    data_start = info.header_offset + 30 + name_size + extra_size
    data_end = data_start + info.compress_size
    _require(data_end <= len(raw), f"{label} compressed stream is truncated")
    compressed = raw[data_start:data_end]
    if info.compress_type == zipfile.ZIP_STORED:
        _require(
            compressed == member_raw,
            f"{label} stored stream differs from its reopened bytes",
        )
        return
    _require(
        info.compress_type == zipfile.ZIP_DEFLATED,
        f"{label} uses an unsupported compression method",
    )
    decompressor = None
    decoded = None
    stream_failed = False
    try:
        decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
        decoded = decompressor.decompress(compressed, info.file_size + 1)
        decoded += decompressor.flush(info.file_size - len(decoded) + 1)
    except _ZIP_LIBRARY_ERRORS:
        stream_failed = True
    if stream_failed:
        raise PrivacyError(f"{label} compressed stream is malformed") from None
    assert decompressor is not None and decoded is not None
    _require(
        decompressor.eof
        and not decompressor.unused_data
        and not decompressor.unconsumed_tail
        and decoded == member_raw,
        f"{label} compressed stream has trailing or inconsistent content",
    )


def _validate_zip_structure(
    name: str,
    raw: bytes,
    archive: zipfile.ZipFile,
    infos: list[zipfile.ZipInfo],
    forbidden_values: Sequence[str],
) -> None:
    comment = archive.comment
    eocd_offset = len(raw) - 22 - len(comment)
    _require(eocd_offset >= 0, f"{name} has a truncated EOCD")
    (
        signature,
        disk_number,
        central_disk,
        disk_entries,
        total_entries,
        central_size,
        central_offset,
        comment_size,
    ) = _zip_unpack("<4s4H2LH", raw, eocd_offset, f"{name} has a truncated EOCD")
    _require(signature == b"PK\x05\x06", f"{name} has trailing or prefixed bytes")
    _require(
        raw[eocd_offset + 22 :] == comment and comment_size == len(comment),
        f"{name} archive comment differs",
    )
    scan_public_bytes(
        comment,
        label=f"{name} archive comment",
        forbidden_values=forbidden_values,
    )
    _require(
        disk_number == central_disk == 0
        and disk_entries == total_entries == len(infos)
        and total_entries != 0xFFFF
        and central_size != 0xFFFFFFFF
        and central_offset != 0xFFFFFFFF
        and central_offset + central_size == eocd_offset
        and archive.start_dir == central_offset,
        f"{name} disk or central-directory closure differs",
    )
    _require(
        eocd_offset < 20 or raw[eocd_offset - 20 : eocd_offset - 16] != b"PK\x06\x07",
        f"{name} uses ZIP64",
    )

    central_position = central_offset
    for index, info in enumerate(infos):
        label = f"{name} central member {index}"
        fields = _zip_unpack(
            "<4s6H3L5H2L", raw, central_position, f"{label} is truncated"
        )
        (
            central_signature,
            _create_version,
            _extract_version,
            flags,
            compression,
            _time,
            _date,
            crc,
            compressed_size,
            file_size,
            name_size,
            extra_size,
            member_comment_size,
            disk_start,
            _internal_attr,
            _external_attr,
            local_offset,
        ) = fields
        _require(central_signature == b"PK\x01\x02", f"{label} header differs")
        name_start = central_position + 46
        name_end = name_start + name_size
        extra_end = name_end + extra_size
        comment_end = extra_end + member_comment_size
        _require(comment_end <= eocd_offset, f"{label} metadata is truncated")
        encoding = "utf-8" if flags & 0x800 else "cp437"
        central_name = _zip_decode(
            raw[name_start:name_end],
            encoding,
            f"{label} name encoding is invalid",
        )
        central_extra = raw[name_end:extra_end]
        central_comment = raw[extra_end:comment_end]
        _zip_extra_fields(central_extra, f"{label} extra")
        scan_public_bytes(
            central_extra,
            label=f"{label} extra",
            forbidden_values=forbidden_values,
        )
        scan_public_bytes(
            central_comment,
            label=f"{label} comment",
            forbidden_values=forbidden_values,
        )
        _require(
            central_name == info.orig_filename
            and central_extra == info.extra
            and central_comment == info.comment
            and flags == info.flag_bits
            and compression == info.compress_type
            and crc == info.CRC
            and compressed_size == info.compress_size
            and file_size == info.file_size
            and disk_start == 0
            and local_offset == info.header_offset
            and 0xFFFFFFFF not in {compressed_size, file_size, local_offset},
            f"{label} metadata differs",
        )
        central_position = comment_end
    _require(
        central_position == eocd_offset,
        f"{name} central directory has gaps or hidden records",
    )

    ordered = sorted(infos, key=lambda item: item.header_offset)
    expected_offset = 0
    for index, info in enumerate(ordered):
        label = f"{name} local member {index}"
        _require(info.header_offset == expected_offset, f"{label} has a prefix or gap")
        (
            local_signature,
            _extract_version,
            local_flags,
            local_compression,
            _time,
            _date,
            local_crc,
            local_compressed_size,
            local_file_size,
            local_name_size,
            local_extra_size,
        ) = _zip_unpack("<4s5H3L2H", raw, expected_offset, f"{label} is truncated")
        _require(local_signature == b"PK\x03\x04", f"{label} header differs")
        name_start = expected_offset + 30
        name_end = name_start + local_name_size
        extra_end = name_end + local_extra_size
        data_end = extra_end + local_compressed_size
        _require(data_end <= central_offset, f"{label} data is truncated")
        encoding = "utf-8" if local_flags & 0x800 else "cp437"
        local_name = _zip_decode(
            raw[name_start:name_end],
            encoding,
            f"{label} name encoding is invalid",
        )
        local_extra = raw[name_end:extra_end]
        _zip_extra_fields(local_extra, f"{label} extra")
        scan_public_bytes(
            local_extra,
            label=f"{label} extra",
            forbidden_values=forbidden_values,
        )
        _require(
            local_name == info.orig_filename
            and local_flags == info.flag_bits
            and local_compression == info.compress_type
            and local_crc == info.CRC
            and local_compressed_size == info.compress_size
            and local_file_size == info.file_size
            and not (local_flags & (0x1 | 0x8 | 0x40)),
            f"{label} metadata differs or uses a data descriptor/encryption",
        )
        next_offset = (
            ordered[index + 1].header_offset
            if index + 1 < len(ordered)
            else central_offset
        )
        _require(data_end == next_offset, f"{label} overlaps or has hidden bytes")
        expected_offset = data_end


def _scan_zip(name: str, raw: bytes, forbidden_values: Sequence[str]) -> None:
    archive: zipfile.ZipFile | None = None
    failure_message: str | None = None
    try:
        try:
            archive = zipfile.ZipFile(io.BytesIO(raw))
        except _ZIP_LIBRARY_ERRORS:
            failure_message = f"{name} is not a valid ZIP payload"
        if archive is not None:
            try:
                seen: set[str] = set()
                aliases: set[str] = set()
                infos = archive.infolist()
                _require(
                    len(infos) <= MAX_ARCHIVE_MEMBERS,
                    f"{name} has too many members",
                )
                _validate_zip_structure(name, raw, archive, infos, forbidden_values)
                total = 0
                for info in infos:
                    member, is_directory = _archive_member_name(
                        info.orig_filename, f"{name} member"
                    )
                    scan_public_bytes(
                        _utf8_bytes(member, f"{name} member name"),
                        label=f"{name} member name",
                        forbidden_values=forbidden_values,
                    )
                    alias = _archive_alias(member)
                    _require(
                        member not in seen and alias not in aliases,
                        f"{name} aliases a member",
                    )
                    seen.add(member)
                    aliases.add(alias)
                    mode = info.external_attr >> 16
                    file_type = stat.S_IFMT(mode)
                    _require(
                        file_type in {0, stat.S_IFREG, stat.S_IFDIR},
                        f"{name} contains a special member",
                    )
                    _require(
                        is_directory == info.is_dir()
                        and (not is_directory or file_type in {0, stat.S_IFDIR}),
                        f"{name} member type differs from its path",
                    )
                    _require(
                        not (info.flag_bits & (0x1 | 0x8 | 0x40)),
                        f"{name} contains encryption or a data descriptor",
                    )
                    _require(
                        info.compress_type
                        in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED},
                        f"{name} contains an unsupported compression method",
                    )
                    _require(
                        info.file_size <= MAX_ARCHIVE_MEMBER_BYTES,
                        f"{name} member exceeds its byte bound",
                    )
                    if info.compress_size == 0:
                        _require(
                            info.file_size == 0,
                            f"{name} contains an impossible compression ratio",
                        )
                    elif info.file_size > 1024 * 1024:
                        _require(
                            info.file_size
                            <= info.compress_size * MAX_COMPRESSION_RATIO,
                            f"{name} contains a compression bomb",
                        )
                    total += info.file_size
                    _require(
                        total <= MAX_ARCHIVE_TOTAL_BYTES,
                        f"{name} expands past its byte bound",
                    )
                    member_raw = archive.read(info)
                    _require(
                        len(member_raw) == info.file_size,
                        f"{name} member size differs",
                    )
                    _validate_zip_member_stream(
                        raw, info, member_raw, f"{name}:{member}"
                    )
                    scan_public_bytes(
                        member_raw,
                        label=f"{name}:{member}",
                        forbidden_values=forbidden_values,
                    )
                    if is_directory:
                        _require(
                            info.file_size == 0 and member_raw == b"",
                            f"{name} directory member has a body",
                        )
                        continue
                    _scan_contextual_json(
                        member, member_raw, f"{name}:{member}", forbidden_values
                    )
                    _reject_array_or_nested_archive(
                        member, member_raw, f"{name}:{member}"
                    )
                _require(
                    archive.testzip() is None,
                    f"{name} CRC validation failed",
                )
            except PrivacyError as error:
                failure_message = str(error)
            except _ZIP_LIBRARY_ERRORS:
                failure_message = f"{name} is not a valid ZIP payload"
    finally:
        if archive is not None:
            try:
                archive.close()
            except _ZIP_LIBRARY_ERRORS:
                if failure_message is None:
                    failure_message = f"{name} is not a valid ZIP payload"
    if failure_message is not None:
        raise PrivacyError(failure_message) from None


def _scan_tar_metadata_fields(
    value: Any,
    label: str,
    forbidden_values: Any,
    *,
    owner_policy: _TarOwnerPolicy = _TarOwnerPolicy.PUBLIC_NEUTRAL,
    scan_raw: bool = True,
) -> None:
    if type(value) is str:
        scan_public_bytes(
            _utf8_bytes(value, label),
            label=label,
            forbidden_values=forbidden_values,
            _check_openings=False,
        )
        return
    _require(isinstance(value, Mapping), f"{label} is malformed")
    for key, item in value.items():
        _require(type(key) is str and type(item) is str, f"{label} is malformed")
        normalized_key = unicodedata.normalize("NFKC", key).casefold()
        normalized_value = (
            unicodedata.normalize("NFKC", item).casefold()
            if normalized_key == "schily.filetype"
            else ""
        )
        private_key = (
            PRIVATE_METADATA_KEY_RE.fullmatch(normalized_key) is not None
            and normalized_key != "path"
        )
        owner_key = normalized_key.rsplit(".", 1)[-1] in {"uname", "gname"}
        structural_override = (
            normalized_key == "size"
            or normalized_key.startswith("gnu.sparse.")
            or normalized_key == "schily.realsize"
            or (normalized_key == "schily.filetype" and normalized_value == "sparse")
        )
        security_override = normalized_key.startswith(
            (
                "schily.xattr.",
                "schily.acl.",
                "libarchive.xattr.",
                "rht.security.",
                "security.",
                "trusted.",
            )
        ) or normalized_key in {
            "sun.holesdata",
            "libarchive.symlinktype",
            "schily.devmajor",
            "schily.devminor",
            "schily.filetype",
            "schily.fflags",
            "schily.ino",
        }
        _require(not private_key or item == "", f"{label} contains private metadata")
        _require(
            owner_policy is _TarOwnerPolicy.PRIVATE_STRUCTURAL
            or not owner_key
            or item == "",
            f"{label} contains private metadata",
        )
        _require(
            not structural_override,
            f"{label} contains an unsupported tar size override",
        )
        _require(
            not security_override,
            f"{label} contains an unsupported tar metadata override",
        )
        terminal_key = normalized_key.rsplit(".", 1)[-1]
        _require(
            terminal_key != "linkpath",
            f"{label} contains an unsupported link path",
        )
        if terminal_key in {"uid", "gid"}:
            _require(
                (
                    item == "0"
                    if owner_policy is _TarOwnerPolicy.PUBLIC_NEUTRAL
                    else bool(re.fullmatch(r"(?:0|[1-9][0-9]{0,18})", item))
                    and int(item) <= (1 << 63) - 1
                ),
                f"{label} contains non-neutral owner identifiers",
            )
        if owner_key and owner_policy is _TarOwnerPolicy.PRIVATE_STRUCTURAL:
            _require(
                not any(
                    ord(character) < 32 or 0xD800 <= ord(character) <= 0xDFFF
                    for character in item
                )
                and len(_utf8_bytes(item, label)) <= 255,
                f"{label} contains private metadata",
            )
        if normalized_key == "path":
            _require(key == "path", f"{label} contains a noncanonical path field")
            _validate_tar_helper_path(item, label, forbidden_values)
        if key == "mtime":
            encoded = item.encode("utf-8")
            timestamp = re.fullmatch(
                rb"(-?)(0|[1-9][0-9]*)(?:\.([0-9]{1,9}))?", encoded
            )
            _require(
                len(encoded) <= MAX_TRACE_NUMBER_LITERAL_BYTES
                and encoded.isascii()
                and timestamp is not None
                and -(1 << 63)
                <= int(timestamp.group(1) + timestamp.group(2))
                <= (1 << 63) - 1,
                f"{label} contains an invalid tar timestamp",
            )
        if key == "hdrcharset":
            _require(
                item == "ISO-IR 10646 2000 UTF-8",
                f"{label} contains an unsupported tar character encoding",
            )
        if scan_raw:
            scan_public_bytes(
                _utf8_bytes(key, f"{label} key"),
                label=f"{label} key",
                forbidden_values=forbidden_values,
                _check_openings=False,
            )
            scan_public_bytes(
                _utf8_bytes(item, f"{label} value"),
                label=f"{label} value",
                forbidden_values=forbidden_values,
                _check_openings=False,
            )
        _require(
            key in _ALLOWED_TAR_PAX_KEYS,
            f"{label} contains an unsupported tar metadata override",
        )


def _scan_tar_metadata(value: Any, label: str, forbidden_values: Any) -> None:
    _scan_tar_metadata_fields(value, label, forbidden_values)


_TAR_PAX_HELPER_TYPES = {tarfile.XHDTYPE, tarfile.XGLTYPE}
_TAR_GNU_LONGNAME_TYPES = {tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_LONGLINK}
_TAR_HELPER_TYPES = _TAR_PAX_HELPER_TYPES | _TAR_GNU_LONGNAME_TYPES
_TAR_HELPER_SENTINEL_NAMES = {"././@PaxHeader", "././@LongLink"}


@dataclass(slots=True)
class _PaxResourceState:
    source_view_bytes: int = 0
    retained_decoded_bytes: int = 0
    record_object_bytes: int = 0
    association_references: int = 0
    association_members: int = 0
    physical_association_bytes: int = 0
    physical_associated_member_bytes: int = 0
    physical_helper_bytes: int = 0
    physical_record_count: int = 0
    physical_helper_ranges: list[tuple[int, int]] = field(default_factory=list)
    physical_helper_starts: list[int] = field(default_factory=list)
    logical_reparse_bytes: int = 0
    logical_reparse_peak_read_bytes: int = 0
    logical_reparse_raw_bytes: int = 0
    logical_reparse_decoded_bytes: int = 0
    logical_reparse_record_object_bytes: int = 0
    logical_cached_association_bytes: int = 0
    logical_cached_member_bytes: int = 0
    current_same_value_owners: int = 0
    peak_same_value_owners: int = 0

    def retain_helper_body(self, start: int, end: int) -> None:
        _require(
            type(start) is int and type(end) is int and 0 <= start <= end,
            "PAX helper resource range is invalid",
        )
        self.physical_helper_bytes += end - start
        self.physical_helper_ranges.append((start, end))
        self.physical_helper_starts.append(start)

    def retain_value(self, source_bytes: int) -> None:
        self.source_view_bytes += source_bytes
        self.retained_decoded_bytes += _PAX_DECODED_BYTES_PER_SOURCE_BYTE * source_bytes
        self.record_object_bytes += _PAX_RECORD_OBJECT_RESERVE_BYTES
        self.physical_record_count += 1
        self.current_same_value_owners = 2
        self.peak_same_value_owners = max(self.peak_same_value_owners, 2)

    def retain_association_owners(self, references: int) -> None:
        _require(
            type(references) is int and references >= 0,
            "PAX association reference count is invalid",
        )
        if not references:
            return
        self.association_references += references
        self.association_members += 1
        self.physical_association_bytes += (
            _PAX_PHYSICAL_ASSOCIATION_RESERVE_BYTES * references
        )
        self.physical_associated_member_bytes += (
            _PAX_PHYSICAL_ASSOCIATED_MEMBER_RESERVE_BYTES
        )
        self.current_same_value_owners = max(self.current_same_value_owners, 3)
        self.peak_same_value_owners = max(
            self.peak_same_value_owners, self.current_same_value_owners
        )

    def reserve_logical_reparse(self) -> None:
        self.logical_reparse_raw_bytes = (
            _PAX_LOGICAL_REPARSE_RAW_BYTES_PER_SOURCE_BYTE * self.physical_helper_bytes
        )
        self.logical_reparse_decoded_bytes = (
            _PAX_LOGICAL_REPARSE_DECODED_BYTES_PER_SOURCE_BYTE
            * self.physical_helper_bytes
        )
        self.logical_reparse_record_object_bytes = (
            _PAX_LOGICAL_REPARSE_RECORD_OBJECT_RESERVE_BYTES
            * self.physical_record_count
        )
        self.logical_cached_association_bytes = (
            _PAX_LOGICAL_ASSOCIATION_RESERVE_BYTES * self.association_references
        )
        self.logical_cached_member_bytes = (
            _PAX_LOGICAL_ASSOCIATED_MEMBER_RESERVE_BYTES * self.association_members
        )
        if self.physical_helper_bytes:
            self.current_same_value_owners = 8 if self.association_references else 6
            self.peak_same_value_owners = max(
                self.peak_same_value_owners, self.current_same_value_owners
            )

    def observe_logical_reparse_read(self, start: int, end: int) -> None:
        index = bisect_right(self.physical_helper_starts, start)
        if index and self.physical_helper_ranges[index - 1][1] > start:
            index -= 1
        while (
            index < len(self.physical_helper_ranges)
            and self.physical_helper_ranges[index][0] < end
        ):
            helper_start, helper_end = self.physical_helper_ranges[index]
            overlap = max(0, min(end, helper_end) - max(start, helper_start))
            self.logical_reparse_bytes += overlap
            self.logical_reparse_peak_read_bytes = max(
                self.logical_reparse_peak_read_bytes, overlap
            )
            index += 1

    def resident_bytes(self) -> int:
        return (
            self.retained_decoded_bytes
            + self.record_object_bytes
            + self.physical_association_bytes
            + self.physical_associated_member_bytes
            + self.logical_reparse_raw_bytes
            + self.logical_reparse_decoded_bytes
            + self.logical_reparse_record_object_bytes
            + self.logical_cached_association_bytes
            + self.logical_cached_member_bytes
        )


@dataclass(frozen=True)
class _TarPhysicalMember:
    physical_header_offset: int
    data_offset: int
    stored_size: int
    type_byte: bytes
    is_directory: bool
    physical_name: str
    expected_logical_name: str
    effective_pax_items: tuple[tuple[str, str], ...]
    effective_owner: _TarOwnerValues = field(repr=False)


@dataclass(frozen=True)
class _TarPhysicalLedger:
    trailer_offset: int
    final_global_pax_items: tuple[tuple[str, str], ...]
    members: tuple[_TarPhysicalMember, ...]
    pax_resources: _PaxResourceState = field(repr=False)


def _tar_header_text(raw: bytes, label: str) -> str:
    nul = raw.find(b"\0")
    if nul >= 0:
        _require(not any(raw[nul:]), f"{label} has hidden suffix bytes")
        raw = raw[:nul]
    text: str | None = None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    if text is None:
        raise PrivacyError(f"{label} is not UTF-8") from None
    return text


def _validate_physical_tar_path(
    value: str,
    label: str,
    forbidden_values: Sequence[str],
    *,
    is_directory: bool = False,
) -> None:
    if value in _TAR_HELPER_SENTINEL_NAMES:
        scan_public_bytes(
            value.encode("ascii"),
            label=label,
            forbidden_values=forbidden_values,
            _check_openings=False,
        )
        return
    path_value = value
    if is_directory and not path_value.endswith("/"):
        path_value += "/"
    member, encoded_directory = _archive_member_name(path_value, label)
    _require(
        encoded_directory == is_directory,
        f"{label} type differs from its physical path",
    )
    scan_public_bytes(
        _utf8_bytes(member, label),
        label=label,
        forbidden_values=forbidden_values,
        _check_openings=False,
    )
    _archive_alias(member)


def _validate_tar_helper_path(
    value: str, label: str, forbidden_values: Sequence[str]
) -> tuple[str, bool]:
    member, is_directory = _archive_member_name(value, label)
    scan_public_bytes(
        _utf8_bytes(member, label),
        label=label,
        forbidden_values=forbidden_values,
        _check_openings=False,
    )
    _archive_alias(member)
    return member, is_directory


def _view_find_byte(raw: memoryview, byte: int, start: int, stop: int) -> int:
    for index in range(start, min(stop, len(raw))):
        if raw[index] == byte:
            return index
    return -1


def _view_contains(raw: memoryview, needle: bytes) -> bool:
    if not needle:
        return True
    stop = len(raw) - len(needle) + 1
    for start in range(max(0, stop)):
        if raw[start] == needle[0] and all(
            raw[start + offset] == byte for offset, byte in enumerate(needle[1:], 1)
        ):
            return True
    return False


def _reject_tar_helper_payload(raw: memoryview, label: str) -> None:
    _require(not _view_contains(raw, b"\x93NUMPY"), f"{label} contains NPY array bytes")
    _require(
        not any(
            _view_contains(raw, signature)
            for signature in (
                b"PK\x03\x04",
                b"PK\x05\x06",
                b"PK\x07\x08",
                b"\x1f\x8b\x08",
            )
        ),
        f"{label} contains a nested archive",
    )
    _require(not _view_has_embedded_tar(raw), f"{label} contains a nested archive")


def _view_has_embedded_tar(raw: memoryview) -> bool:
    candidates = 0
    for offset in range(257, len(raw) - 4):
        if raw[offset : offset + 5] != b"ustar":
            continue
        candidates += 1
        _require(
            candidates <= MAX_ARCHIVE_MEMBERS,
            "tar helper archive detection exceeds its bound",
        )
        start = offset - 257
        if (
            start >= 0
            and start + 512 <= len(raw)
            and _tar_checksum_matches(
                bytes(raw[start : start + 512]), require_ustar=False
            )
        ):
            return True
    return False


def _scan_pax_helper_body(
    body: memoryview,
    label: str,
    forbidden_values: Any,
    *,
    owner_policy: _TarOwnerPolicy = _TarOwnerPolicy.PUBLIC_NEUTRAL,
    record_limit: int = MAX_TAR_PAX_RECORDS,
    resources: _PaxResourceState | None = None,
) -> tuple[
    tuple[tuple[str, str], ...],
    int,
    tuple[tuple[int, int], ...],
]:
    _require(
        len(body) <= MAX_TAR_PAX_HELPER_BYTES,
        f"{label} exceeds its PAX byte bound",
    )
    cursor = 0
    records = 0
    keys: set[str] = set()
    items: list[tuple[str, str]] = []
    key_mask_spans: list[tuple[int, int]] = []
    while cursor < len(body):
        space = _view_find_byte(body, ord(" "), cursor, cursor + 32)
        _require(space > cursor, f"{label} has a malformed PAX record")
        length_raw = body[cursor:space]
        _require(
            all(ord("0") <= byte <= ord("9") for byte in length_raw)
            and length_raw[0] != ord("0"),
            f"{label} has a malformed PAX record",
        )
        length = int(bytes(length_raw))
        end = cursor + length
        _require(
            end <= len(body) and end > space + 2 and body[end - 1] == ord("\n"),
            f"{label} has a malformed PAX record",
        )
        key_start = space + 1
        separator = _view_find_byte(body, ord("="), key_start, end - 1)
        key_raw = (
            body[key_start:separator] if separator > key_start else memoryview(b"")
        )
        value_raw = (
            body[separator + 1 : end - 1] if separator > key_start else memoryview(b"")
        )
        _require(
            separator > key_start
            and len(key_raw) <= MAX_PRIVATE_JSON_STRING_BYTES
            and len(value_raw) <= MAX_PUBLIC_JSON_BYTES,
            f"{label} has a malformed PAX record",
        )
        key: str | None = None
        value: str | None = None
        try:
            key = bytes(key_raw).decode("utf-8")
            value = codecs.decode(value_raw, "utf-8")
        except UnicodeDecodeError:
            pass
        if key is None or value is None:
            raise PrivacyError(f"{label} has non-UTF-8 PAX metadata") from None
        normalized_key = unicodedata.normalize("NFKC", key).casefold()
        records += 1
        _require(
            records <= min(MAX_ARCHIVE_MEMBERS, record_limit)
            and normalized_key not in keys,
            f"{label} PAX record closure differs",
        )
        keys.add(normalized_key)
        _scan_tar_metadata_fields(
            {key: value},
            label,
            forbidden_values,
            owner_policy=owner_policy,
            scan_raw=False,
        )
        if resources is not None:
            # Both decoded key and value survive in the physical ledger.  The
            # combined source interval is bounded by this helper body.
            resources.retain_value(len(key_raw) + len(value_raw))
        if key == "path":
            key_mask_spans.append((key_start, key_start + len(key_raw) + 1))
        items.append((key, value))
        cursor = end
    _require(cursor == len(body), f"{label} PAX body has trailing bytes")
    _require(bool(items), f"{label} is an empty PAX helper")
    return tuple(items), records, tuple(key_mask_spans)


def _scan_tar_helper_body(
    physical: tarfile.TarInfo,
    body: memoryview,
    label: str,
    forbidden_values: Any,
    *,
    body_offset: int | None = None,
    owner_policy: _TarOwnerPolicy = _TarOwnerPolicy.PUBLIC_NEUTRAL,
    pax_record_limit: int = MAX_TAR_PAX_RECORDS,
    resources: _PaxResourceState | None = None,
) -> (
    tuple[
        tuple[tuple[str, str], ...],
        int,
        tuple[tuple[int, int], ...],
    ]
    | str
):
    if physical.type in _TAR_PAX_HELPER_TYPES:
        _require(
            len(body) <= MAX_TAR_PAX_HELPER_BYTES,
            f"{label} exceeds its PAX byte bound",
        )
        if resources is not None:
            _require(
                type(body_offset) is int and body_offset >= 0,
                "PAX helper resource offset is invalid",
            )
            resources.retain_helper_body(body_offset, body_offset + len(body))
        result = _scan_pax_helper_body(
            body,
            label,
            forbidden_values,
            owner_policy=owner_policy,
            record_limit=pax_record_limit,
            resources=resources,
        )
        _reject_tar_helper_payload(body, label)
        return result
    _require(
        physical.type == tarfile.GNUTYPE_LONGNAME,
        f"{label} contains an unsupported GNU link helper",
    )
    _require(
        len(body) <= MAX_TAR_GNU_LONGNAME_BYTES,
        f"{label} exceeds its GNU long-name byte bound",
    )
    gnu_body = bytes(body)
    _reject_array_or_nested_archive("tar-helper.bin", gnu_body, label)
    _require(
        gnu_body.endswith(b"\0") and b"\0" not in gnu_body[:-1],
        f"{label} GNU long-name body is malformed",
    )
    path: str | None = None
    try:
        path = gnu_body[:-1].decode("utf-8")
    except UnicodeDecodeError:
        pass
    if path is None:
        raise PrivacyError(f"{label} GNU long-name body is not UTF-8") from None
    _validate_tar_helper_path(path, label, forbidden_values)
    return path


def _merge_pax_items(
    current: Mapping[str, str],
    items: Sequence[tuple[str, str]],
    label: str,
) -> dict[str, str]:
    merged = dict(current)
    normalized_keys = {
        unicodedata.normalize("NFKC", key).casefold(): key for key in merged
    }
    for key, value in items:
        normalized_key = unicodedata.normalize("NFKC", key).casefold()
        previous = normalized_keys.get(normalized_key)
        _require(
            previous is None or previous == key,
            f"{label} PAX metadata aliases a key",
        )
        merged[key] = value
        normalized_keys[normalized_key] = key
    return merged


def _scan_contiguous_tar_bytes(
    raw: bytes | bytearray,
    key_mask_spans: Sequence[tuple[int, int]],
    label: str,
    forbidden_values: Sequence[str],
) -> None:
    context = _privacy_scan_context(forbidden_values)
    previous_end = 0
    for start, end in key_mask_spans:
        _require(
            previous_end <= start < end <= len(raw) and raw[start:end] == b"path=",
            f"{label} PAX path masking differs",
        )
        previous_end = end

    _scan_pattern_view(
        raw,
        label,
        assignment_exempt_starts=frozenset(start for start, _end in key_mask_spans),
    )
    _scan_nfkc_utf8(
        raw,
        label,
        context,
        check_openings=False,
        mask_spans=key_mask_spans,
    )
    _scan_nfkc_utf8(
        raw,
        label,
        context,
        check_patterns=False,
        check_openings=True,
    )
    _scan_utf16_ascii(raw, label, context)


def _private_tar_owner_id(raw: bytes, parsed: Any, label: str) -> int:
    _require(
        bool(re.fullmatch(rb"[0-7]{1,7}[\0 ]*", raw)),
        f"{label} is malformed",
    )
    value = int(raw.rstrip(b"\0 "), 8)
    _require(
        type(parsed) is int and parsed == value and value <= (1 << 63) - 1,
        f"{label} is malformed",
    )
    return value


def _private_tar_owner_name(raw: bytes, label: str) -> str:
    value = _tar_header_text(raw, label)
    _require(
        not any(
            ord(character) < 32 or 0xD800 <= ord(character) <= 0xDFFF
            for character in value
        )
        and len(_utf8_bytes(value, label)) <= 255,
        f"{label} is malformed",
    )
    return value


def _effective_tar_owner(
    fixed: _TarOwnerValues,
    items: Mapping[str, str],
    label: str,
    owner_policy: _TarOwnerPolicy,
) -> _TarOwnerValues:
    if owner_policy is _TarOwnerPolicy.PUBLIC_NEUTRAL:
        return fixed
    uid = items.get("uid", str(fixed.uid))
    gid = items.get("gid", str(fixed.gid))
    uname = items.get("uname", fixed.uname)
    gname = items.get("gname", fixed.gname)
    _scan_tar_metadata_fields(
        {"uid": uid, "gid": gid, "uname": uname, "gname": gname},
        label,
        (),
        owner_policy=owner_policy,
        scan_raw=False,
    )
    return _TarOwnerValues(int(uid), int(gid), uname, gname)


def _scan_tar_physical_records(
    name: str,
    tar_raw: bytes | bytearray,
    forbidden_values: Any,
    *,
    owner_policy: _TarOwnerPolicy = _TarOwnerPolicy.PUBLIC_NEUTRAL,
    limits: _PrivateSdistValidationLimits | None = None,
) -> _TarPhysicalLedger:
    forbidden_values = _privacy_scan_context(forbidden_values)
    member_limit = MAX_ARCHIVE_MEMBERS if limits is None else limits.members
    member_byte_limit = (
        MAX_ARCHIVE_MEMBER_BYTES if limits is None else limits.member_bytes
    )
    helper_byte_limit = (
        MAX_TAR_HELPER_TOTAL_BYTES if limits is None else limits.helper_bytes
    )
    pax_record_limit = MAX_TAR_PAX_RECORDS if limits is None else limits.helper_records
    consecutive_helper_limit = (
        MAX_TAR_CONSECUTIVE_HELPERS if limits is None else limits.consecutive_helpers
    )
    association_limit = (
        MAX_TAR_EFFECTIVE_PAX_ASSOCIATIONS
        if limits is None
        else limits.effective_pax_associations
    )
    raw_view = memoryview(tar_raw)
    _require(
        len(tar_raw) % 512 == 0 and len(tar_raw) >= 1024,
        f"{name} tar block closure differs",
    )
    offset = 0
    record_count = 0
    helper_count = 0
    helper_bytes = 0
    pax_records = 0
    effective_pax_associations = 0
    consecutive_helpers = 0
    global_pax: dict[str, str] = {}
    pending_local_pax: tuple[tuple[str, str], ...] | None = None
    pending_gnu_longname: str | None = None
    members: list[_TarPhysicalMember] = []
    names: set[str] = set()
    aliases: set[str] = set()
    pax_key_mask_spans: list[tuple[int, int]] = []
    pax_resources = _PaxResourceState()
    while offset + 512 <= len(tar_raw):
        header = bytes(raw_view[offset : offset + 512])
        if not any(header):
            trailer = raw_view[offset:]
            _require(
                len(trailer) >= 1024 and not any(trailer),
                f"{name} tar trailer is malformed or hides bytes",
            )
            _require(
                consecutive_helpers == 0
                and pending_local_pax is None
                and pending_gnu_longname is None,
                f"{name} tar helper chain does not close",
            )
            _scan_contiguous_tar_bytes(
                tar_raw,
                pax_key_mask_spans,
                f"{name} contiguous tar bytes",
                forbidden_values,
            )
            return _TarPhysicalLedger(
                trailer_offset=offset,
                final_global_pax_items=tuple(global_pax.items()),
                members=tuple(members),
                pax_resources=pax_resources,
            )
        record_count += 1
        _require(
            record_count <= MAX_ARCHIVE_MEMBERS * 4,
            f"{name} has too many physical tar records",
        )
        _require(_tar_checksum_matches(header), f"{name} tar header is malformed")
        recursion_failure = False
        header_failure = False
        try:
            physical = tarfile.TarInfo.frombuf(
                header, encoding="utf-8", errors="surrogateescape"
            )
        except RecursionError:
            recursion_failure = True
        except tarfile.HeaderError, ValueError, OverflowError:
            header_failure = True
        if recursion_failure or header_failure:
            raise PrivacyError(f"{name} tar header is malformed") from None
        _require(
            type(physical.size) is int and 0 <= physical.size <= member_byte_limit,
            f"{name} physical tar record exceeds its byte bound",
        )
        header_name = _tar_header_text(header[:100], f"{name} physical tar path")
        header_prefix = _tar_header_text(header[345:500], f"{name} physical tar prefix")
        physical_name = (
            f"{header_prefix}/{header_name}" if header_prefix else header_name
        )
        parsed_physical_name = (
            physical_name.rstrip("/") if physical.isdir() else physical_name
        )
        _require(
            physical.name == parsed_physical_name,
            f"{name} physical tar path fields disagree",
        )
        if owner_policy is _TarOwnerPolicy.PUBLIC_NEUTRAL:
            _require(
                physical.uid == physical.gid == 0
                and all(byte in b"\0 0" for byte in header[108:124]),
                f"{name} physical tar owner identifiers are not neutral",
            )
            _require(
                physical.uname == physical.gname == "" and not any(header[265:329]),
                f"{name} physical tar owner names are not empty",
            )
            fixed_owner = _TarOwnerValues(0, 0, "", "")
        else:
            fixed_owner = _TarOwnerValues(
                _private_tar_owner_id(
                    header[108:116], physical.uid, f"{name} physical tar uid"
                ),
                _private_tar_owner_id(
                    header[116:124], physical.gid, f"{name} physical tar gid"
                ),
                _private_tar_owner_name(header[265:297], f"{name} physical tar uname"),
                _private_tar_owner_name(header[297:329], f"{name} physical tar gname"),
            )
        _require(
            physical.linkname == "" and not any(header[157:257]),
            f"{name} physical tar link field is not empty",
        )
        is_helper = physical.type in _TAR_HELPER_TYPES
        _require(
            is_helper
            or physical.type in {tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE},
            f"{name} physical tar record type is unsupported",
        )
        if is_helper:
            helper_count += 1
            consecutive_helpers += 1
            _require(
                helper_count <= member_limit,
                f"{name} has too many tar helpers",
            )
            _require(
                consecutive_helpers <= consecutive_helper_limit,
                f"{name} has too many consecutive tar helpers",
            )
            helper_bytes += physical.size
            _require(
                helper_bytes <= helper_byte_limit,
                f"{name} tar helpers exceed their byte bound",
            )
            if physical.type in _TAR_PAX_HELPER_TYPES:
                _require(
                    physical.size <= min(MAX_TAR_PAX_HELPER_BYTES, helper_byte_limit),
                    f"{name} tar helper exceeds its PAX byte bound",
                )
            elif physical.type == tarfile.GNUTYPE_LONGNAME:
                _require(
                    physical.size <= MAX_TAR_GNU_LONGNAME_BYTES,
                    f"{name} tar helper exceeds its GNU long-name byte bound",
                )
        else:
            _require(
                len(members) < member_limit,
                f"{name} has too many members",
            )
        _validate_physical_tar_path(
            physical_name,
            f"{name} physical tar path",
            forbidden_values,
            is_directory=physical.isdir(),
        )
        data_start = offset + 512
        data_end = data_start + physical.size
        padded_end = data_start + ((physical.size + 511) // 512) * 512
        _require(
            data_start <= data_end <= padded_end <= len(tar_raw),
            f"{name} physical tar record is truncated",
        )
        scan_public_bytes(
            header,
            label=f"{name} tar header",
            forbidden_values=forbidden_values,
            _check_openings=False,
        )
        body = raw_view[data_start:data_end]
        padding = raw_view[data_end:padded_end]
        _require(not any(padding), f"{name} tar physical padding is not zero")
        _require(
            not physical.isdir() or physical.size == 0,
            f"{name} physical directory record has a body",
        )
        if is_helper:
            helper = _scan_tar_helper_body(
                physical,
                body,
                f"{name} tar helper body",
                forbidden_values,
                body_offset=data_start,
                owner_policy=owner_policy,
                pax_record_limit=pax_record_limit - pax_records,
                resources=pax_resources,
            )
            if physical.type in _TAR_PAX_HELPER_TYPES:
                _require(
                    type(helper) is tuple and len(helper) == 3,
                    f"{name} tar helper parsing differs",
                )
                items, record_total, relative_mask_spans = helper
                pax_key_mask_spans.extend(
                    (data_start + start, data_start + end)
                    for start, end in relative_mask_spans
                )
                pax_records += record_total
                _require(
                    pax_records <= pax_record_limit,
                    f"{name} has too many PAX records",
                )
                if physical.type == tarfile.XGLTYPE:
                    _require(
                        pending_local_pax is None and pending_gnu_longname is None,
                        f"{name} tar helper order cannot be reconciled",
                    )
                    _require(
                        all(key != "path" for key, _value in items),
                        f"{name} contains an unsupported global PAX path",
                    )
                    global_pax = _merge_pax_items(global_pax, items, f"{name} global")
                else:
                    _require(
                        pending_local_pax is None,
                        f"{name} repeats a local PAX helper",
                    )
                    pending_local_pax = items
            else:
                _require(
                    physical.type == tarfile.GNUTYPE_LONGNAME and type(helper) is str,
                    f"{name} contains an unsupported GNU link helper",
                )
                _require(
                    pending_gnu_longname is None,
                    f"{name} repeats a GNU long-name helper",
                )
                pending_gnu_longname = helper
        else:
            local_pax = pending_local_pax or ()
            effective_pax_count = len(global_pax) + sum(
                key not in global_pax for key, _value in local_pax
            )
            _require(
                effective_pax_count <= association_limit - effective_pax_associations,
                f"{name} effective PAX associations exceed their bound",
            )
            effective_pax_associations += effective_pax_count
            effective_pax = _merge_pax_items(
                global_pax,
                local_pax,
                f"{name} member",
            )
            pax_resources.retain_association_owners(len(effective_pax))
            _require(
                pax_resources.retained_decoded_bytes
                + pax_resources.record_object_bytes
                + pax_resources.physical_association_bytes
                + pax_resources.physical_associated_member_bytes
                <= _PAX_DECODED_BYTES_PER_SOURCE_BYTE * helper_byte_limit
                + _PAX_RECORD_OBJECT_RESERVE_BYTES * pax_record_limit
                + _PAX_PHYSICAL_ASSOCIATION_RESERVE_BYTES * association_limit
                + _PAX_PHYSICAL_ASSOCIATED_MEMBER_RESERVE_BYTES * member_limit,
                f"{name} PAX resident ownership exceeds its bound",
            )
            effective_owner = _effective_tar_owner(
                fixed_owner,
                effective_pax,
                f"{name} effective tar owner",
                owner_policy,
            )
            pax_path = effective_pax.get("path")
            _require(
                pax_path is None or pending_gnu_longname is None,
                f"{name} has conflicting tar path helpers",
            )
            expected_path = pax_path or pending_gnu_longname or physical_name
            expected_name, encoded_directory = _archive_member_name(
                expected_path, f"{name} resolved tar path"
            )
            _require(
                physical.isdir() or not encoded_directory,
                f"{name} resolved tar path type differs",
            )
            physical_member, _ = _archive_member_name(
                physical_name, f"{name} physical tar path"
            )
            _require(
                pending_gnu_longname is None or expected_name != physical_member,
                f"{name} contains a no-op GNU long-name helper",
            )
            alias = _archive_alias(expected_name)
            _require(
                expected_name not in names and alias not in aliases,
                f"{name} aliases a member",
            )
            names.add(expected_name)
            aliases.add(alias)
            members.append(
                _TarPhysicalMember(
                    physical_header_offset=offset,
                    data_offset=data_start,
                    stored_size=physical.size,
                    type_byte=physical.type,
                    is_directory=physical.isdir(),
                    physical_name=physical_member,
                    expected_logical_name=expected_name,
                    effective_pax_items=tuple(effective_pax.items()),
                    effective_owner=effective_owner,
                )
            )
            pending_local_pax = None
            pending_gnu_longname = None
            consecutive_helpers = 0
        offset = padded_end
    raise PrivacyError(f"{name} tar trailer is missing")


def _reconcile_tar_member(
    name: str,
    info: tarfile.TarInfo,
    physical: _TarPhysicalMember,
    *,
    owner_policy: _TarOwnerPolicy = _TarOwnerPolicy.PUBLIC_NEUTRAL,
) -> None:
    _require(
        type(info.name) is str and info.name == physical.expected_logical_name,
        f"{name} logical tar member name differs from its physical record",
    )
    _require(
        info.type == physical.type_byte
        and info.isdir() == physical.is_directory
        and (info.isdir() or info.isfile()),
        f"{name} logical tar member type differs from its physical record",
    )
    _require(
        type(info.offset_data) is int
        and info.offset_data == physical.data_offset
        and type(info.size) is int
        and info.size == physical.stored_size,
        f"{name} logical tar member storage differs from its physical record",
    )
    _require(
        isinstance(info.pax_headers, Mapping)
        and all(
            type(key) is str and type(value) is str
            for key, value in info.pax_headers.items()
        )
        and dict(info.pax_headers) == dict(physical.effective_pax_items),
        f"{name} logical PAX metadata differs from its physical helpers",
    )
    if owner_policy is _TarOwnerPolicy.PUBLIC_NEUTRAL:
        _require(
            type(info.uid) is int
            and type(info.gid) is int
            and info.uid == info.gid == 0
            and info.uname == info.gname == info.linkname == "",
            f"{name} logical tar ownership or link metadata differs",
        )
    else:
        _require(
            type(info.uid) is int
            and type(info.gid) is int
            and info.uid == physical.effective_owner.uid
            and info.gid == physical.effective_owner.gid
            and info.uname == physical.effective_owner.uname
            and info.gname == physical.effective_owner.gname
            and info.linkname == "",
            f"{name} logical tar ownership or link metadata differs",
        )
    _require(
        info.sparse is None,
        f"{name} logical tar member contains sparse storage",
    )


def _pax_resident_reserve(
    helper_bytes: int = MAX_TAR_HELPER_TOTAL_BYTES,
    helper_records: int = MAX_TAR_PAX_RECORDS,
    association_references: int = MAX_TAR_EFFECTIVE_PAX_ASSOCIATIONS,
    association_members: int = MAX_ARCHIVE_MEMBERS,
) -> int:
    _require(
        type(helper_bytes) is int
        and type(helper_records) is int
        and type(association_references) is int
        and type(association_members) is int
        and helper_bytes >= 0
        and helper_records >= 0
        and association_references >= 0
        and association_members >= 0,
        "PAX resident-memory reserve is invalid",
    )
    reserve = (
        _PAX_DECODED_BYTES_PER_SOURCE_BYTE * helper_bytes
        + _PAX_RECORD_OBJECT_RESERVE_BYTES * helper_records
        + _PAX_LOGICAL_REPARSE_RAW_BYTES_PER_SOURCE_BYTE * helper_bytes
        + _PAX_LOGICAL_REPARSE_DECODED_BYTES_PER_SOURCE_BYTE * helper_bytes
        + _PAX_LOGICAL_REPARSE_RECORD_OBJECT_RESERVE_BYTES * helper_records
        + (
            _PAX_PHYSICAL_ASSOCIATION_RESERVE_BYTES
            + _PAX_LOGICAL_ASSOCIATION_RESERVE_BYTES
        )
        * association_references
        + (
            _PAX_PHYSICAL_ASSOCIATED_MEMBER_RESERVE_BYTES
            + _PAX_LOGICAL_ASSOCIATED_MEMBER_RESERVE_BYTES
        )
        * association_members
        + MAX_TAR_SCAN_TRANSIENT_BYTES
    )
    _require(
        reserve >= MAX_TAR_SCAN_TRANSIENT_BYTES,
        "PAX resident-memory reserve is invalid",
    )
    return reserve


def _tar_gzip_output_allowance(
    compressed_size: int, *, limits: _PrivateSdistValidationLimits | None = None
) -> int:
    helper_bytes = MAX_TAR_HELPER_TOTAL_BYTES if limits is None else limits.helper_bytes
    helper_records = MAX_TAR_PAX_RECORDS if limits is None else limits.helper_records
    association_limit = (
        MAX_TAR_EFFECTIVE_PAX_ASSOCIATIONS
        if limits is None
        else limits.effective_pax_associations
    )
    member_limit = MAX_ARCHIVE_MEMBERS if limits is None else limits.members
    fixed_resident = (
        compressed_size
        + 2 * TAR_GZIP_INPUT_CHUNK_BYTES
        + TAR_GZIP_OUTPUT_CHUNK_BYTES
        + _pax_resident_reserve(
            helper_bytes,
            helper_records,
            association_limit,
            member_limit,
        )
    )
    _require(
        fixed_resident <= MAX_TAR_RESIDENT_BYTES,
        "gzip input exceeds its resident-memory bound",
    )
    resident_allowance = (MAX_TAR_RESIDENT_BYTES - fixed_resident) // 2
    ratio_allowance = max(
        MIN_TAR_GZIP_RATIO_BYTES,
        compressed_size * MAX_COMPRESSION_RATIO,
    )
    return min(
        MAX_TAR_BYTES,
        MAX_ARCHIVE_TOTAL_BYTES,
        resident_allowance,
        ratio_allowance,
    )


def _allocate_tar_buffer(size: int) -> bytearray:
    return bytearray(size)


class _TarBufferReader:
    __slots__ = ("_pax_resources", "_position", "_raw")

    def __init__(
        self, raw: bytes, *, pax_resources: _PaxResourceState | None = None
    ) -> None:
        if type(raw) is not bytes:
            raise TypeError("tar reader requires immutable bytes")
        self._raw = raw
        self._pax_resources = pax_resources
        self._position = 0

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
            if position < 0:
                raise ValueError("negative seek position")
        elif whence == io.SEEK_CUR:
            position = max(0, self._position + offset)
        elif whence == io.SEEK_END:
            position = max(0, len(self._raw) + offset)
        else:
            raise ValueError("invalid seek mode")
        self._position = position
        return position

    def read(self, size: int = -1) -> bytes:
        if self._position >= len(self._raw):
            return b""
        if size is None or size < 0:
            end = len(self._raw)
        else:
            end = min(len(self._raw), self._position + size)
        start = min(self._position, len(self._raw))
        self._position = end
        if self._pax_resources is not None:
            self._pax_resources.observe_logical_reparse_read(start, end)
        return self._raw[start:end]


def _decompress_gzip_bounded(
    name: str, raw: bytes, *, limits: _PrivateSdistValidationLimits | None = None
) -> bytearray:
    _require(len(raw) >= 4, f"{name} is not a valid gzip payload")
    output_allowance = _tar_gzip_output_allowance(len(raw), limits=limits)
    advertised_size = int.from_bytes(raw[-4:], "little")
    _require(
        advertised_size <= output_allowance,
        f"{name} gzip stream is trailing, concatenated, or oversized",
    )
    result = _allocate_tar_buffer(advertised_size)
    decompressor = None
    gzip_failure = False
    try:
        decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    except OSError, RecursionError, ValueError, zlib.error:
        gzip_failure = True
    if gzip_failure:
        raise PrivacyError(f"{name} is not a valid gzip payload") from None
    written = 0
    compressed_consumed = 0
    cursor = 0
    pending: bytes | memoryview = b""
    while True:
        if not pending:
            _require(
                cursor < len(raw),
                f"{name} gzip stream is trailing, concatenated, or oversized",
            )
            end = min(len(raw), cursor + TAR_GZIP_INPUT_CHUNK_BYTES)
            pending = memoryview(raw)[cursor:end]
            cursor = end

        prospective_consumed = compressed_consumed + len(pending)
        prospective_limit = min(
            output_allowance,
            max(
                MIN_TAR_GZIP_RATIO_BYTES,
                prospective_consumed * MAX_COMPRESSION_RATIO,
            ),
        )
        remaining = prospective_limit - written
        _require(
            remaining >= 0,
            f"{name} gzip stream is trailing, concatenated, or oversized",
        )
        supplied = pending
        gzip_failure = False
        try:
            decoded = decompressor.decompress(
                supplied,
                min(
                    TAR_GZIP_OUTPUT_CHUNK_BYTES,
                    remaining + 1,
                    advertised_size - written + 1,
                ),
            )
        except OSError, RecursionError, ValueError, zlib.error:
            gzip_failure = True
            decoded = b""
        if gzip_failure:
            raise PrivacyError(f"{name} is not a valid gzip payload") from None

        unused_size = len(decompressor.unused_data)
        pending = decompressor.unconsumed_tail
        consumed = len(supplied) - len(pending) - unused_size
        _require(
            consumed >= 0,
            f"{name} gzip stream is trailing, concatenated, or oversized",
        )
        compressed_consumed += consumed
        _require(
            len(decoded) <= advertised_size - written,
            f"{name} gzip stream is trailing, concatenated, or oversized",
        )
        result[written : written + len(decoded)] = decoded
        written += len(decoded)
        output_limit = min(
            output_allowance,
            max(
                MIN_TAR_GZIP_RATIO_BYTES,
                compressed_consumed * MAX_COMPRESSION_RATIO,
            ),
        )
        _require(
            written <= output_limit,
            f"{name} gzip stream is trailing, concatenated, or oversized",
        )
        if decompressor.eof:
            _require(
                not decompressor.unused_data
                and not pending
                and cursor == len(raw)
                and written == advertised_size,
                f"{name} gzip stream is trailing, concatenated, or oversized",
            )
            return result
        _require(
            bool(decoded) or consumed > 0,
            f"{name} gzip stream is trailing, concatenated, or oversized",
        )


@dataclass(frozen=True, slots=True)
class _ValidatedTar:
    members: tuple[_PrivateSdistMember, ...]
    total_member_bytes: int
    physical_ordinary_count: int
    logical_member_count: int


def _scan_tar(
    name: str,
    raw: bytes,
    forbidden_values: Any,
    *,
    owner_policy: _TarOwnerPolicy = _TarOwnerPolicy.PUBLIC_NEUTRAL,
    limits: _PrivateSdistValidationLimits | None = None,
) -> _ValidatedTar:
    if _looks_like_gzip(raw):
        mutable_tar_raw = _decompress_gzip_bounded(name, raw, limits=limits)
        # During this conversion both complete tar buffers are live (C + 2D),
        # which is already the gzip preflight peak.  Drop the mutable owner
        # before the logical parser runs so no live view can pin it.
        tar_raw = bytes(mutable_tar_raw)
        del mutable_tar_raw
    else:
        tar_raw = raw
        helper_bytes = (
            MAX_TAR_HELPER_TOTAL_BYTES if limits is None else limits.helper_bytes
        )
        helper_records = (
            MAX_TAR_PAX_RECORDS if limits is None else limits.helper_records
        )
        association_limit = (
            MAX_TAR_EFFECTIVE_PAX_ASSOCIATIONS
            if limits is None
            else limits.effective_pax_associations
        )
        member_limit = MAX_ARCHIVE_MEMBERS if limits is None else limits.members
        _require(
            len(tar_raw)
            + _pax_resident_reserve(
                helper_bytes,
                helper_records,
                association_limit,
                member_limit,
            )
            <= MAX_TAR_RESIDENT_BYTES,
            f"{name} tar stream exceeds its resident-memory bound",
        )
    _require(
        len(tar_raw) <= MAX_TAR_BYTES
        and (limits is None or len(tar_raw) <= limits.total_member_bytes),
        f"{name} tar stream exceeds its byte bound",
    )
    ledger = _scan_tar_physical_records(
        name,
        tar_raw,
        forbidden_values,
        owner_policy=owner_policy,
        limits=limits,
    )
    ledger.pax_resources.reserve_logical_reparse()
    helper_bytes = MAX_TAR_HELPER_TOTAL_BYTES if limits is None else limits.helper_bytes
    helper_records = MAX_TAR_PAX_RECORDS if limits is None else limits.helper_records
    association_limit = (
        MAX_TAR_EFFECTIVE_PAX_ASSOCIATIONS
        if limits is None
        else limits.effective_pax_associations
    )
    member_limit = MAX_ARCHIVE_MEMBERS if limits is None else limits.members
    _require(
        ledger.pax_resources.resident_bytes()
        <= _pax_resident_reserve(
            helper_bytes,
            helper_records,
            association_limit,
            member_limit,
        )
        - MAX_TAR_SCAN_TRANSIENT_BYTES,
        f"{name} PAX resident ownership exceeds its bound",
    )
    archive: tarfile.TarFile | None = None
    archive_failure = False
    try:
        archive = tarfile.open(
            fileobj=_TarBufferReader(tar_raw, pax_resources=ledger.pax_resources),
            mode="r:",
        )
        total = 0
        logical_count = 0
        safe_members: list[_PrivateSdistMember] = []
        for index, info in enumerate(archive):
            _require(
                index < (MAX_ARCHIVE_MEMBERS if limits is None else limits.members)
                and index < len(ledger.members),
                f"{name} logical tar membership differs from physical records",
            )
            physical = ledger.members[index]
            _reconcile_tar_member(name, info, physical, owner_policy=owner_policy)
            logical_count += 1
            member = physical.expected_logical_name
            data_end = physical.data_offset + physical.stored_size
            padded_end = (
                physical.data_offset + ((physical.stored_size + 511) // 512) * 512
            )
            _require(
                physical.data_offset <= data_end <= padded_end <= len(tar_raw),
                f"{name} member storage is truncated",
            )
            stored_raw = memoryview(tar_raw)[physical.data_offset : data_end]
            padding = memoryview(tar_raw)[data_end:padded_end]
            _require(not any(padding), f"{name} member padding is not zero")
            _require(
                physical.stored_size
                <= (
                    MAX_ARCHIVE_TOTAL_BYTES
                    if limits is None
                    else limits.total_member_bytes
                )
                - total,
                f"{name} expands past its byte bound",
            )
            total += physical.stored_size
            _require(
                total
                <= (
                    MAX_ARCHIVE_TOTAL_BYTES
                    if limits is None
                    else limits.total_member_bytes
                ),
                f"{name} expands past its byte bound",
            )
            if physical.is_directory:
                _require(
                    physical.stored_size == 0,
                    f"{name} directory member has a body",
                )
                safe_members.append(
                    _PrivateSdistMember(
                        member,
                        "directory",
                        0,
                        physical.data_offset,
                        hashlib.sha256(b"").hexdigest(),
                    )
                )
                continue
            # The physical pass already fixed this exact storage interval, and
            # logical offset/size/type metadata was independently reconciled.
            # Materialize only that interval; ExFileObject.read(size) can
            # allocate according to the caller's much larger requested size.
            member_raw = bytes(stored_raw)
            _require(
                len(member_raw) == physical.stored_size,
                f"{name} member storage or size differs",
            )
            _scan_contextual_json(
                member, member_raw, f"{name}:{member}", forbidden_values
            )
            _reject_array_or_nested_archive(member, member_raw, f"{name}:{member}")
            safe_members.append(
                _PrivateSdistMember(
                    member,
                    "file",
                    physical.stored_size,
                    physical.data_offset,
                    hashlib.sha256(member_raw).hexdigest(),
                )
            )
            del member_raw, stored_raw, padding
        _require(
            logical_count == len(ledger.members),
            f"{name} logical tar membership differs from physical records",
        )
        _require(
            isinstance(archive.pax_headers, Mapping)
            and dict(archive.pax_headers) == dict(ledger.final_global_pax_items),
            f"{name} global PAX metadata differs from physical helpers",
        )
        _require(
            len(tar_raw) % 512 == 0
            and ledger.trailer_offset + 1024 <= len(tar_raw)
            and not any(tar_raw[ledger.trailer_offset :]),
            f"{name} tar trailer is malformed or hides bytes",
        )
        _require(
            ledger.pax_resources.logical_reparse_bytes
            == ledger.pax_resources.physical_helper_bytes,
            f"{name} logical PAX reparse differs from physical helpers",
        )
    except PrivacyError:
        raise
    except (
        EOFError,
        OSError,
        OverflowError,
        RecursionError,
        ValueError,
        tarfile.TarError,
    ):
        archive_failure = True
    finally:
        if archive is not None:
            try:
                archive.close()
            except PrivacyError:
                raise
            except (
                EOFError,
                OSError,
                OverflowError,
                RecursionError,
                ValueError,
                tarfile.TarError,
            ):
                archive_failure = True
    if archive_failure:
        raise PrivacyError(f"{name} is not a valid tar payload") from None
    return _ValidatedTar(tuple(safe_members), total, len(ledger.members), logical_count)


def _private_sdist_source_matches(
    source: _PrivateSdistReadView,
) -> _PrivateSdistFailure | None:
    try:
        current = _private_sdist_identity(os.fstat(source.fd))
        if not _private_sdist_identity_is_valid(current):
            return _PrivateSdistFailure.SOURCE_INVALID
    except Exception:
        return _PrivateSdistFailure.SOURCE_INVALID
    if current != source.identity:
        return _PrivateSdistFailure.SOURCE_CHANGED
    return None


_PRIVATE_SDIST_MAX_FD = (1 << 31) - 1
_PRIVATE_SDIST_MAX_UNSIGNED_STAT_VALUE = (1 << 64) - 1
_PRIVATE_SDIST_MAX_TIMESTAMP_NS = (1 << 63) - 1


def _private_sdist_identity_is_valid(identity: Any) -> bool:
    if type(identity) is not _PrivateSdistIdentity:
        return False
    try:
        fields = (
            identity.device,
            identity.inode,
            identity.mode,
            identity.nlink,
            identity.size,
            identity.mtime_ns,
            identity.ctime_ns,
        )
    except Exception:
        return False
    if not all(type(value) is int for value in fields):
        return False
    device, inode, mode, nlink, size, mtime_ns, ctime_ns = fields
    return (
        0 <= device <= _PRIVATE_SDIST_MAX_UNSIGNED_STAT_VALUE
        and 0 < inode <= _PRIVATE_SDIST_MAX_UNSIGNED_STAT_VALUE
        and 0 <= mode <= 0o177777
        and stat.S_ISREG(mode)
        and 0 < nlink <= _PRIVATE_SDIST_MAX_UNSIGNED_STAT_VALUE
        and 0 < size <= _PRIVATE_SDIST_MAX_UNSIGNED_STAT_VALUE
        and 0 <= mtime_ns <= _PRIVATE_SDIST_MAX_TIMESTAMP_NS
        and 0 <= ctime_ns <= _PRIVATE_SDIST_MAX_TIMESTAMP_NS
    )


def _private_sdist_source_is_valid(source: Any) -> bool:
    if type(source) is not _PrivateSdistReadView:
        return False
    try:
        return (
            source._seal is _PRIVATE_SDIST_VIEW_SEAL
            and type(source.fd) is int
            and 0 <= source.fd <= _PRIVATE_SDIST_MAX_FD
            and _private_sdist_identity_is_valid(source.identity)
        )
    except Exception:
        return False


def _validate_private_sdist_raw_first(
    source: _PrivateSdistReadView,
    forbidden_values: object,
    *,
    limits: _PrivateSdistValidationLimits,
) -> _ValidatedPrivateSdist:
    """Validate one retained gzip sdist without reopening a path or extracting it."""

    # Validate the whole sealed shape before any descriptor syscall.  The
    # module-private seal is not an unforgeable runtime boundary.
    if not _private_sdist_source_is_valid(source):
        raise _PrivateSdistValidationError(
            _PrivateSdistFailure.SOURCE_INVALID
        ) from None
    if not _private_sdist_limits_are_valid(limits):
        raise _PrivateSdistValidationError(_PrivateSdistFailure.LIMIT_INVALID) from None
    source_failure = _private_sdist_source_matches(source)
    if source_failure is not None:
        raise _PrivateSdistValidationError(source_failure) from None
    size = source.identity.size
    if size <= 0 or size > limits.archive_bytes:
        raise _PrivateSdistValidationError(
            _PrivateSdistFailure.SOURCE_INVALID
        ) from None
    reserve = _pax_resident_reserve(
        limits.helper_bytes,
        limits.helper_records,
        limits.effective_pax_associations,
        limits.members,
    )
    if 2 * size + reserve > MAX_TAR_RESIDENT_BYTES:
        raise _PrivateSdistValidationError(
            _PrivateSdistFailure.ARCHIVE_REJECTED
        ) from None

    staging: bytearray | None = None
    read_failed = False
    try:
        staging = bytearray(size)
        offset = 0
        digest = hashlib.sha256()
        while offset < size:
            chunk = os.pread(
                source.fd, min(TAR_GZIP_INPUT_CHUNK_BYTES, size - offset), offset
            )
            if not chunk:
                read_failed = True
                break
            staging[offset : offset + len(chunk)] = chunk
            digest.update(chunk)
            offset += len(chunk)
        if offset != size or os.pread(source.fd, 1, size):
            read_failed = True
    except OSError:
        read_failed = True
    if read_failed or staging is None:
        raise _PrivateSdistValidationError(_PrivateSdistFailure.READ_FAILED) from None
    source_failure = _private_sdist_source_matches(source)
    if source_failure is not None:
        raise _PrivateSdistValidationError(source_failure) from None
    raw = bytes(staging)
    del staging

    archive_failure = False
    validated: _ValidatedTar | None = None
    try:
        _require(_looks_like_gzip(raw), "private sdist is not gzip")
        context = _privacy_scan_context(
            forbidden_values,
            maximum_canonical_bytes=limits.normalization_work_bytes,
            maximum_matcher_states=limits.matcher_states,
        )
        validated = _scan_tar(
            "private sdist",
            raw,
            context,
            owner_policy=_TarOwnerPolicy.PRIVATE_STRUCTURAL,
            limits=limits,
        )
    except Exception:
        archive_failure = True
    if archive_failure or validated is None:
        raise _PrivateSdistValidationError(
            _PrivateSdistFailure.ARCHIVE_REJECTED
        ) from None
    source_failure = _private_sdist_source_matches(source)
    if source_failure is not None:
        raise _PrivateSdistValidationError(source_failure) from None
    return _ValidatedPrivateSdist(
        archive_size=size,
        archive_sha256=digest.hexdigest(),
        members=validated.members,
        total_member_bytes=validated.total_member_bytes,
        physical_ordinary_count=validated.physical_ordinary_count,
        logical_member_count=validated.logical_member_count,
    )


def scan_payload(
    name: str,
    raw: bytes,
    *,
    media_type: str | None = None,
    forbidden_values: Any = (),
) -> None:
    """Scan a wheel, sdist, ZIP, tar, or opaque binary without extracting it."""

    if media_type is None:
        matching_media = [
            candidate
            for candidate, suffixes in PAYLOAD_SUFFIXES.items()
            if name.casefold().endswith(suffixes)
        ]
        _require(len(matching_media) == 1, "payload suffix does not identify its media")
        media_type = matching_media[0]
    name = _payload_name(name, "payload name", media_type)
    context = _privacy_scan_context(forbidden_values)
    _require(type(raw) is bytes, f"payload {name} must be immutable bytes")
    _require(len(raw) <= MAX_PAYLOAD_BYTES, f"payload {name} exceeds its byte bound")
    _require(b"\x93NUMPY" not in raw, f"payload {name} contains NPY bytes")
    if media_type == "application/octet-stream":
        _require(
            not (
                _looks_like_zip(raw)
                or _looks_like_gzip(raw)
                or _looks_like_tar(raw)
                or _contains_archive(raw)
            ),
            f"payload {name} disguises an archive as opaque binary",
        )
    # The physical tar pass scans every region contextually so the canonical
    # PAX `path=` field is not mistaken for a process-environment assignment.
    if media_type != "application/x-tar":
        scan_public_bytes(raw, label=f"payload {name}", forbidden_values=context)
    _scan_contextual_json(name, raw, f"payload {name}", context)
    if media_type in {"application/zip", "application/vnd.python.wheel"}:
        _require(_looks_like_zip(raw), f"payload {name} signature differs from media")
        _scan_zip(name, raw, context)
    elif media_type == "application/gzip":
        _require(_looks_like_gzip(raw), f"payload {name} signature differs from media")
        _scan_tar(name, raw, context)
    elif media_type == "application/x-tar":
        _require(_looks_like_tar(raw), f"payload {name} signature differs from media")
        _scan_tar(name, raw, context)
    else:
        _require(
            media_type == "application/octet-stream",
            f"payload {name} media type differs",
        )


def _identity_strings(value: Any):
    if type(value) is str:
        encoded = _utf8_bytes(value, "private identity string")
        _require(
            not value
            or 3 <= len(value)
            and len(encoded) <= MAX_IDENTITY_SCAN_VALUE_BYTES,
            "private identity contains an unscannably short string or oversized value",
        )
        if value:
            yield value
    elif value is None or type(value) in {bool, int, float}:
        rendered = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
        )
        encoded = rendered.encode("ascii")
        _require(
            len(rendered) >= 3 and len(encoded) <= MAX_IDENTITY_SCAN_VALUE_BYTES,
            "private identity contains an unscannably short numeric value",
        )
        yield rendered
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _identity_strings(key)
            yield from _identity_strings(item)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        for item in value:
            yield from _identity_strings(item)


def _hmac_commit(salt: bytes, domain: str, metadata: Any, raw: bytes) -> str:
    header = json.dumps(
        {"domain": domain, "metadata": metadata},
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hmac.new(salt, header + b"\x00" + raw, hashlib.sha256).hexdigest()


def _identity_commitment(
    salt: bytes,
    scope: str,
    name: str,
    raw: bytes,
    bindings: Mapping[str, Any],
) -> str:
    return _hmac_commit(
        salt,
        "issue123-identity-v1",
        {"bindings": bindings, "scope": scope, "name": name},
        raw,
    )


class _DenseMap:
    def __init__(self) -> None:
        self._values: dict[tuple[str, Any], int] = {}

    def ordinal(self, value: Any, label: str) -> int:
        def canonical(item: Any, *, nested: bool = False) -> tuple[str, Any]:
            if item is None and nested:
                return ("none", None)
            if type(item) is int:
                _require(
                    -(1 << 63) <= item < (1 << 63),
                    f"{label} identifier is invalid",
                )
                return ("int", item)
            if type(item) is str:
                _require(
                    len(_utf8_bytes(item, f"{label} identifier")) <= 128
                    and unicodedata.normalize("NFC", item) == item
                    and not any(
                        ord(character) < 0x20 or ord(character) == 0x7F
                        for character in item
                    ),
                    f"{label} identifier is invalid",
                )
                return ("str", item)
            if type(item) is tuple:
                _require(1 <= len(item) <= 8, f"{label} context is invalid")
                return ("tuple", tuple(canonical(child, nested=True) for child in item))
            raise PrivacyError(f"{label} identifier type is invalid")

        key = canonical(value)
        if key not in self._values:
            self._values[key] = len(self._values)
        return self._values[key]


def _one_arg(args: Mapping[str, Any], names: tuple[str, ...], label: str) -> Any:
    present = [name for name in names if name in args]
    _require(len(present) <= 1, f"{label} repeats an aliased trace argument")
    return args[present[0]] if present else None


def _event_kind(name: str, category: str, phase: str) -> str:
    lowered = name.lower()
    categories = {item.strip().lower() for item in category.split(",") if item.strip()}
    _require(categories <= KNOWN_CATEGORIES, "trace event category is unknown")
    if phase == "M":
        _require(name in METADATA_NAMES, "trace metadata event is unknown")
        return "metadata"
    if name == "[memory]":
        return "allocation"
    if "graph break" in lowered or "graph_break" in lowered:
        return "graph-break"
    if "recompile" in lowered:
        return "recompile"
    if "fallback" in lowered:
        return "fallback"
    if name in HALO_ANNOTATIONS:
        return "halo-annotation"
    if lowered == "cudagraphlaunch" or "cuda graph" in lowered:
        return "cuda-graph"
    if lowered.startswith("torch-compiled region:"):
        return "compiled-region"
    if "memcpy" in lowered or "gpu_memcpy" in categories:
        if "htod" in lowered or "host to device" in lowered:
            return "copy-h2d"
        if "dtoh" in lowered or "device to host" in lowered:
            return "copy-d2h"
        return "copy-device"
    if "gpu_memset" in categories or "memset" in lowered:
        return "memset"
    if "kernel" in categories:
        return "nccl-kernel" if "nccl" in lowered else "kernel"
    if name in POLICY_WRITE_OPERATIONS:
        return "indexed-write"
    if name.startswith(("aten::", "torch::")) and "cpu_op" in categories:
        return "cpu-operation"
    if "cuda_runtime" in categories or "cuda_driver" in categories:
        _require(
            lowered.startswith(("cuda", "cu")), "trace runtime event name is unknown"
        )
        return "cuda-runtime"
    if name.startswith("ProfilerStep#") and "user_annotation" in categories:
        _require(
            name.removeprefix("ProfilerStep#").isdigit(), "profiler step is malformed"
        )
        return "profiler-step"
    if phase in {"s", "t", "f"} and "ac2g" in categories:
        return "correlation-flow"
    raise PrivacyError("trace event name/category is outside the closed taxonomy")


def _semantic_token(name: str, kind: str, phase_raw: str) -> str:
    if kind == "metadata":
        token = METADATA_SEMANTIC_TOKENS[name]
    elif kind == "halo-annotation":
        token = HALO_SEMANTIC_TOKENS[name]
    elif kind == "indexed-write":
        token = WRITE_SEMANTIC_TOKENS[name]
    else:
        token = kind
    phase = TRACE_PHASES[phase_raw]
    _require(
        token in SEMANTIC_TOKEN_PHASES and phase in SEMANTIC_TOKEN_PHASES[token],
        "trace event kind and phase are incompatible",
    )
    return _safe_name(token, "trace semantic token")


def _validate_argument_values(args: Mapping[str, Any], kind: str, label: str) -> None:
    identifier_keys = {
        "Addr",
        "Allocation Id",
        "allocation_id",
        "stream",
        "Stream",
        "Stream Id",
        "correlation",
        "Correlation",
        "Correlation ID",
        "correlation_id",
        "External id",
        "graph_id",
        "Graph Id",
    }
    vector_keys = {"grid", "block"}
    numeric_keys = ARG_KEYS - identifier_keys - vector_keys - {"Device Type"}
    for key in numeric_keys & set(args):
        _number(args[key], f"{label} argument {key}")
    for key in identifier_keys & set(args):
        _require(
            type(args[key]) in {int, str},
            f"{label} argument {key} has an invalid identifier",
        )
    if "Device Type" in args:
        device_type = args["Device Type"]
        _require(
            type(device_type) is int or device_type in {"CPU", "CUDA", "cpu", "cuda"},
            f"{label} device type is unknown",
        )
    for key in vector_keys & set(args):
        vector = args[key]
        _require(
            isinstance(vector, list)
            and 1 <= len(vector) <= 3
            and all(type(item) is int and item >= 0 for item in vector),
            f"{label} argument {key} is malformed",
        )
    allocation_keys = {"Addr", "Allocation Id", "allocation_id", "Total Allocated"}
    if set(args) & allocation_keys:
        _require(kind == "allocation", f"{label} carries allocation args on {kind}")
    if "Bytes" in args:
        _require(
            kind == "allocation" or kind.startswith("copy-"),
            f"{label} carries byte counts on {kind}",
        )
        if kind.startswith("copy-"):
            _require(
                type(args["Bytes"]) is int and args["Bytes"] >= 0,
                f"{label} copy bytes are invalid",
            )
    if kind == "allocation":
        _require("Bytes" in args, f"{label} allocation has no byte delta")


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
    first: Sequence[tuple[float, float]], second: Sequence[tuple[float, float]]
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


def normalize_trace(raw: bytes, *, label: str = "trace") -> dict[str, Any]:
    """Return an event-complete deterministic privacy normalization."""

    document = _strict_json(raw, label)
    _exact_keys(document, {"traceEvents"}, label)
    events = document["traceEvents"]
    _require(isinstance(events, list) and bool(events), f"{label} has no events")
    _require(len(events) <= MAX_TRACE_EVENTS, f"{label} has too many events")

    timestamps: list[Fraction] = []
    for index, event in enumerate(events):
        _require(isinstance(event, Mapping), f"{label} event {index} is not an object")
        if event.get("ph") != "M" and "ts" in event:
            timestamps.append(
                Fraction(_number(event["ts"], f"{label} event timestamp"))
            )
    _require(bool(timestamps), f"{label} has no timestamps")
    origin = min(timestamps)

    process_ids = _DenseMap()
    thread_ids = _DenseMap()
    stream_ids = _DenseMap()
    correlation_ids = _DenseMap()
    allocation_ids = _DenseMap()
    allocation_context_ids = _DenseMap()
    graph_ids = _DenseMap()
    normalized = []
    allocated = freed = positive_allocations = allocation_events = 0
    live_by_context: dict[tuple[Any, ...], int] = {}
    live_baseline_by_context: dict[tuple[Any, ...], int] = {}
    peak_aggregate_live = 0
    graph_breaks = recompiles = fallbacks = 0
    copies = h2d = d2h = kernels = compiled = cuda_graphs = 0
    nccl_intervals: list[tuple[float, float]] = []
    compute_intervals: list[tuple[float, float]] = []
    flow_phases: dict[int, list[tuple[str, Fraction]]] = {}

    for ordinal, event in enumerate(events):
        keys = set(event)
        _require(
            keys <= TRACE_EVENT_KEYS, f"{label} event {ordinal} has unknown fields"
        )
        _require(
            {"name", "cat", "ph", "pid", "tid", "args"} <= keys,
            f"{label} event {ordinal} is incomplete",
        )
        name = event["name"]
        category = event["cat"]
        phase_raw = event["ph"]
        _require(
            type(name) is str and type(category) is str,
            f"{label} event text is invalid",
        )
        _require(
            len(_utf8_bytes(name, f"{label} event name")) <= 4096
            and len(_utf8_bytes(category, f"{label} event category")) <= 4096
            and unicodedata.normalize("NFC", name) == name
            and unicodedata.normalize("NFC", category) == category
            and not any(
                ord(character) < 0x20 or ord(character) == 0x7F
                for character in name + category
            ),
            f"{label} event text is invalid",
        )
        _require(
            type(phase_raw) is str and phase_raw in TRACE_PHASES,
            f"{label} event phase is unknown",
        )
        scan_public_bytes(
            _utf8_bytes(name, f"{label} event name"), label=f"{label} event name"
        )
        scan_public_bytes(
            _utf8_bytes(category, f"{label} event category"),
            label=f"{label} event category",
        )
        args = event["args"]
        _require(isinstance(args, Mapping), f"{label} event args must be an object")
        allowed_args = METADATA_ARG_KEYS if phase_raw == "M" else ARG_KEYS
        _require(
            set(args) <= allowed_args, f"{label} event {ordinal} has unknown arguments"
        )
        kind = _event_kind(name, category, phase_raw)
        if phase_raw == "M":
            _require(
                "ts" not in event and "dur" not in event,
                f"{label} metadata event carries a clock",
            )
            expected_args = (
                {"name"} if name in {"process_name", "thread_name"} else {"sort_index"}
            )
            _require(set(args) == expected_args, f"{label} metadata args differ")
            if "name" in args:
                metadata_name = args["name"]
                _require(
                    type(metadata_name) is str
                    and len(_utf8_bytes(metadata_name, f"{label} metadata")) <= 128
                    and unicodedata.normalize("NFC", metadata_name) == metadata_name,
                    f"{label} metadata name is invalid",
                )
                scan_public_bytes(
                    _utf8_bytes(metadata_name, f"{label} metadata"),
                    label=f"{label} metadata",
                )
            else:
                _require(
                    type(args["sort_index"]) is int
                    and -(1 << 31) <= args["sort_index"] < (1 << 31),
                    f"{label} metadata sort index is invalid",
                )
        else:
            _validate_argument_values(args, kind, f"{label} event {ordinal}")

        if phase_raw == "M":
            start_exact = Fraction(0)
            start = 0
        else:
            _require("ts" in event, f"{label} event {ordinal} has no timestamp")
            start_delta = (
                Fraction(_number(event["ts"], f"{label} event timestamp")) - origin
            )
            _require(start_delta >= 0, f"{label} event timestamp precedes its origin")
            _require(
                start_delta <= MAX_LOCAL_TIMESTAMP_US,
                f"{label} event timestamp exceeds its local-clock bound",
            )
            start_exact = start_delta
            start = float(start_delta)
        if phase_raw == "X":
            _require("dur" in event, f"{label} complete event has no duration")
            duration_exact = Fraction(_number(event["dur"], f"{label} event duration"))
            _require(duration_exact >= 0, f"{label} event duration is negative")
            _require(
                duration_exact <= MAX_LOCAL_TIMESTAMP_US,
                f"{label} event duration exceeds its local-clock bound",
            )
            _require(
                start_exact + duration_exact <= MAX_LOCAL_TIMESTAMP_US,
                f"{label} event interval exceeds its local-clock bound",
            )
            duration: int | float = float(duration_exact)
        else:
            _require("dur" not in event, f"{label} non-complete event has a duration")
            duration = 0

        stream = _one_arg(args, ("stream", "Stream", "Stream Id"), label)
        device = _one_arg(args, ("Device Id", "device"), label)
        device_type = args.get("Device Type")
        context = args.get("context")
        correlation = _one_arg(
            args,
            (
                "correlation",
                "Correlation",
                "Correlation ID",
                "correlation_id",
                "External id",
            ),
            label,
        )
        if kind == "correlation-flow":
            _require("id" in event, f"{label} correlation flow has no id")
        if "id" in event:
            _require(correlation is None, f"{label} event repeats correlation identity")
            correlation = event["id"]
        allocation = _one_arg(args, ("Addr", "Allocation Id", "allocation_id"), label)
        graph = _one_arg(args, ("graph_id", "Graph Id"), label)
        amount: int | None = None
        if "Bytes" in args:
            _require(
                type(args["Bytes"]) is int, f"{label} allocation bytes are invalid"
            )
            amount = args["Bytes"]
        live_allocated: int | None = None
        if kind == "allocation":
            _require("Total Allocated" in args, f"{label} allocation lacks live total")
            _require(
                type(args["Total Allocated"]) is int and args["Total Allocated"] >= 0,
                f"{label} live allocation is invalid",
            )
            live_allocated = args["Total Allocated"]

        process_ordinal = process_ids.ordinal(event["pid"], f"{label} process")
        stream_context = (event["pid"], device_type, device, context, stream)
        allocation_context_ordinal = (
            allocation_context_ids.ordinal(
                (event["pid"], device_type, device, context),
                f"{label} allocation context",
            )
            if kind == "allocation"
            else None
        )
        correlation_ordinal = (
            correlation_ids.ordinal(correlation, f"{label} correlation")
            if correlation is not None
            else None
        )
        record = {
            "ordinal": ordinal,
            "kind": kind,
            "semantic_token": _semantic_token(name, kind, phase_raw),
            "phase": TRACE_PHASES[phase_raw],
            "start_us": start,
            "duration_us": duration,
            "process_ordinal": process_ordinal,
            "thread_ordinal": thread_ids.ordinal(
                (event["pid"], event["tid"]), f"{label} thread"
            ),
            "stream_ordinal": (
                stream_ids.ordinal(stream_context, f"{label} stream")
                if stream is not None
                else None
            ),
            "correlation_ordinal": correlation_ordinal,
            "allocation_ordinal": (
                allocation_ids.ordinal(
                    (*stream_context, allocation), f"{label} allocation"
                )
                if allocation is not None
                else None
            ),
            "allocation_context_ordinal": allocation_context_ordinal,
            "graph_ordinal": (
                graph_ids.ordinal((*stream_context, graph), f"{label} graph")
                if graph is not None
                else None
            ),
            "bytes": amount,
            "live_allocated_bytes": live_allocated,
        }
        normalized.append(record)

        if kind == "correlation-flow":
            assert correlation_ordinal is not None
            flow_phases.setdefault(correlation_ordinal, []).append(
                (TRACE_PHASES[phase_raw], start_exact)
            )

        if kind == "allocation":
            _require(
                amount is not None and amount != 0,
                f"{label} allocation event has no byte delta",
            )
            assert live_allocated is not None
            total_allocated = live_allocated
            assert allocation_context_ordinal is not None
            allocation_context = (allocation_context_ordinal,)
            previous = live_by_context.get(allocation_context, total_allocated - amount)
            if allocation_context not in live_baseline_by_context:
                live_baseline_by_context[allocation_context] = previous
            _require(
                previous >= 0 and total_allocated == previous + amount,
                f"{label} allocation sequence is inconsistent",
            )
            live_by_context[allocation_context] = total_allocated
            peak_aggregate_live = max(
                peak_aggregate_live, sum(live_by_context.values())
            )
            allocation_events += 1
            if amount > 0:
                positive_allocations += 1
                allocated += amount
            else:
                freed -= amount
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
        elif kind == "compiled-region" and phase_raw == "X":
            compiled += 1
        elif kind == "cuda-graph" and phase_raw == "X":
            cuda_graphs += 1
        if kind in {"kernel", "nccl-kernel"} and phase_raw == "X":
            kernels += 1
            interval = (float(start), float(start) + float(duration))
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

    baseline = sum(live_baseline_by_context.values())
    final_live = sum(live_by_context.values())
    peak_live = max(baseline, peak_aggregate_live, final_live)
    nccl_us = _interval_duration(nccl_intervals)
    compute_us = _interval_duration(compute_intervals)
    overlap_us = _intersection_duration(nccl_intervals, compute_intervals)
    summary = {
        "event_count": len(normalized),
        "allocation_events": allocation_events,
        "positive_allocation_events": positive_allocations,
        "allocated_bytes": allocated,
        "freed_bytes": freed,
        "allocation_net_bytes": allocated - freed,
        "live_allocation_baseline_bytes": baseline,
        "peak_live_allocated_bytes": peak_live,
        "final_live_allocated_bytes": final_live,
        "live_allocation_growth_bytes": final_live - baseline,
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
    return {
        "clock": LOCAL_CLOCK,
        "events": normalized,
        "summary": summary,
    }


def _validate_trace_expectation(expectation: Any, label: str) -> Mapping[str, Any]:
    fields = {
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
    value = _exact_keys(expectation, fields, label)
    _safe_name(value["name"], f"{label} name")
    semantic_signatures = value["semantic_signatures"]
    _require(
        isinstance(semantic_signatures, list)
        and len(semantic_signatures) == value["event_count"],
        f"{label} semantic inventory closure differs",
    )
    for index, signature in enumerate(semantic_signatures):
        _require(
            isinstance(signature, list) and len(signature) == 2,
            f"{label} semantic signature {index} is malformed",
        )
        token, phase = signature
        _safe_name(token, f"{label} semantic token {index}")
        _safe_name(phase, f"{label} semantic phase {index}")
        _require(
            (token, phase) in SEMANTIC_EVENT_SIGNATURES,
            f"{label} semantic signature {index} is outside the closed inventory",
        )
    for field in fields - {"name", "semantic_signatures", "require_nccl_overlap"}:
        _nonnegative_int(value[field], f"{label} {field}")
    _require(value["event_count"] > 0, f"{label} event count must be positive")
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
    _require(value["allocation_net_bytes"] == 0, f"{label} permits allocation leakage")
    _require(
        value["live_allocation_growth_bytes"] == 0,
        f"{label} permits live allocation growth",
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
        type(value["require_nccl_overlap"]) is bool, f"{label} NCCL policy is invalid"
    )
    if value["require_nccl_overlap"]:
        _require(
            value["nccl_kernel_launches"] > 0, f"{label} requires no NCCL launches"
        )
    return value


def _enforce_trace_expectation(
    trace: Mapping[str, Any], expectation: Mapping[str, Any], label: str
) -> None:
    summary = trace["summary"]
    for field in (
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
    ):
        _require(
            summary[field] == expectation[field],
            f"{label} {field} differs from trusted closure",
        )
    _require(
        [[event["semantic_token"], event["phase"]] for event in trace["events"]]
        == expectation["semantic_signatures"],
        f"{label} semantic event inventory differs from trusted closure",
    )
    if expectation["require_nccl_overlap"]:
        _require(
            summary["nccl_device_us"] > 0
            and summary["nccl_compute_overlap_us"] > 0
            and 0 < summary["overlap_fraction"] <= 1,
            f"{label} does not prove NCCL/compute overlap",
        )


def _validate_timing_expectation(value: Any, label: str) -> Mapping[str, Any]:
    record = _exact_keys(value, {"name", "sample_count", "samples_sha256"}, label)
    _safe_name(record["name"], f"{label} name")
    _require(
        type(record["sample_count"]) is int
        and 0 < record["sample_count"] <= MAX_TIMING_SAMPLES,
        f"{label} sample count is invalid",
    )
    _require(
        type(record["samples_sha256"]) is str
        and SHA256_RE.fullmatch(record["samples_sha256"]),
        f"{label} sample digest is invalid",
    )
    return record


def _project_timing(
    value: Any, expectation: Mapping[str, Any], label: str
) -> dict[str, Any]:
    record = _exact_keys(value, {"name", "unit", "samples"}, label)
    name = expectation["name"]
    _require(record["name"] == name, f"{label} name differs")
    _require(record["unit"] == "seconds", f"{label} unit differs")
    samples = record["samples"]
    _require(
        isinstance(samples, list)
        and len(samples) == expectation["sample_count"]
        and len(samples) <= MAX_TIMING_SAMPLES,
        f"{label} sample closure differs",
    )
    for index, sample in enumerate(samples):
        _number(sample, f"{label} sample {index}", positive=True)
    copied = list(samples)
    samples_sha256 = _canonical_sha256(copied)
    _require(
        samples_sha256 == expectation["samples_sha256"],
        f"{label} samples differ from trusted digest",
    )
    median = statistics.median(copied)
    mad = statistics.median(abs(value - median) for value in copied)
    return {
        "name": name,
        "unit": "seconds",
        "samples": copied,
        "sample_count": len(copied),
        "samples_sha256": samples_sha256,
        "median_seconds": median,
        "mad_seconds": mad,
        "relative_mad": mad / median,
    }


def _shape(value: Any, label: str) -> list[int]:
    _require(isinstance(value, list), f"{label} must be a list")
    _require(len(value) <= 16, f"{label} has too many dimensions")
    for index, dimension in enumerate(value):
        _require(
            type(dimension) is int and dimension >= 0, f"{label}[{index}] is invalid"
        )
    return list(value)


def _element_count(shape: Sequence[int]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _decode_array(
    raw: bytes, dtype: str, shape: Sequence[int], label: str
) -> list[Any]:
    _require(type(raw) is bytes, f"{label} must be immutable bytes")
    _require(len(raw) <= MAX_ARRAY_BYTES, f"{label} exceeds its byte bound")
    code, itemsize, floating = DTYPES[dtype]
    count = _element_count(shape)
    _require(
        len(raw) == count * itemsize, f"{label} byte length differs from dtype/shape"
    )
    if count == 0:
        return []
    if dtype.startswith("complex"):
        primitive = "f" if dtype == "complex64" else "d"
        values = list(struct.iter_unpack("<" + primitive * 2, raw))
        result = [complex(real, imaginary) for real, imaginary in values]
    else:
        result = [item[0] for item in struct.iter_unpack("<" + code, raw)]
    if floating:
        _require(
            all(
                (
                    math.isfinite(value.real) and math.isfinite(value.imag)
                    if isinstance(value, complex)
                    else math.isfinite(value)
                )
                for value in result
            ),
            f"{label} contains non-finite values",
        )
    return result


def _comparison_policy(value: Any, label: str) -> Mapping[str, Any]:
    record = _exact_keys(
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
    _semantic_name(record["name"], f"{label} name")
    _require(record["dtype"] in DTYPES, f"{label} dtype is unsupported")
    shape = _shape(record["shape"], f"{label} shape")
    element_count = _element_count(shape)
    itemsize = DTYPES[record["dtype"]][1]
    _require(element_count > 0, f"{label} array must contain an element")
    _require(
        element_count * itemsize <= MAX_ARRAY_BYTES,
        f"{label} array exceeds its byte bound",
    )
    contract = record["comparison_contract"]
    _require(
        contract in COMPARISON_CONTRACTS, f"{label} comparison contract is unknown"
    )
    rtol = _number(record["rtol"], f"{label} rtol")
    atol = _number(record["atol"], f"{label} atol")
    _require(rtol >= 0 and atol >= 0, f"{label} tolerances must be nonnegative")
    limit = record["normalized_limit"]
    if contract == "normalized-linf-l2":
        _number(limit, f"{label} normalized limit", positive=True)
        _require(rtol == 0 and atol > 0, f"{label} normalized floor contract differs")
    else:
        _require(limit is None, f"{label} has an inapplicable normalized limit")
    if contract == "exact":
        _require(rtol == 0 and atol == 0, f"{label} exact tolerances differ")
        _require(
            not DTYPES[record["dtype"]][2],
            f"{label} exact floating comparison is forbidden",
        )
    return record


def _stable_l2(values: Sequence[int | float]) -> float:
    scale = max(values, default=0.0)
    if scale == 0:
        return 0.0
    result = scale * math.sqrt(math.fsum((value / scale) ** 2 for value in values))
    _require(math.isfinite(result), "correctness norm arithmetic is non-finite")
    return result


def _compare_arrays(
    reference: Sequence[Any],
    candidate: Sequence[Any],
    policy: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    _require(len(reference) == len(candidate), f"{label} array lengths differ")
    differences = [
        abs(actual - expected)
        for actual, expected in zip(candidate, reference, strict=True)
    ]
    _require(
        all(math.isfinite(value) for value in differences),
        f"{label} error arithmetic is non-finite",
    )
    max_abs = max(differences, default=0.0)
    reference_magnitudes = [abs(value) for value in reference]
    reference_abs_max = max(reference_magnitudes, default=0.0)
    reference_l2 = _stable_l2(reference_magnitudes)
    error_l2 = _stable_l2(differences)
    reference_all_zero = all(value == 0 for value in reference)
    zero_reference_exact = not reference_all_zero or all(
        actual == expected
        for actual, expected in zip(candidate, reference, strict=True)
    )
    contract = policy["comparison_contract"]
    rtol = policy["rtol"]
    atol = policy["atol"]
    max_allowed: float | None = None
    normalized_linf: float | None = None
    normalized_l2: float | None = None
    max_tolerance_excess: float | None = None
    if contract == "exact":
        passed = all(value == 0 for value in differences)
        max_allowed = 0.0
    elif contract == "elementwise":
        allowed = [atol + rtol * abs(value) for value in reference]
        _require(
            all(math.isfinite(value) for value in allowed),
            f"{label} tolerance arithmetic is non-finite",
        )
        max_allowed = max(allowed, default=atol)
        max_tolerance_excess = max(
            (error - bound for error, bound in zip(differences, allowed, strict=True)),
            default=-atol,
        )
        _require(
            math.isfinite(max_tolerance_excess),
            f"{label} tolerance excess arithmetic is non-finite",
        )
        passed = max_tolerance_excess <= 0
    else:
        normalized_linf = max_abs / max(reference_abs_max, atol)
        normalized_l2 = error_l2 / max(reference_l2, atol * math.sqrt(len(reference)))
        passed = (
            normalized_linf <= policy["normalized_limit"]
            and normalized_l2 <= policy["normalized_limit"]
        )
    passed = passed and zero_reference_exact
    _require(passed, f"{label} correctness tolerance failed")
    return {
        "contract": contract,
        "rtol": rtol,
        "atol": atol,
        "normalized_limit": policy["normalized_limit"],
        "max_abs_error": max_abs,
        "max_allowed_error": max_allowed,
        "max_tolerance_excess": max_tolerance_excess,
        "reference_abs_max": reference_abs_max,
        "reference_l2": reference_l2,
        "error_l2": error_l2,
        "normalized_linf": normalized_linf,
        "normalized_l2": normalized_l2,
        "reference_all_zero": reference_all_zero,
        "zero_reference_exact": zero_reference_exact,
        "passed": True,
    }


def _capture_key(value: Any, label: str) -> str:
    if type(value) is int:
        _require(value >= 0, f"{label} must be nonnegative")
        return str(value)
    return _safe_name(value, label)


def _validate_bindings(value: Any, label: str) -> dict[str, Any]:
    bindings = _exact_keys(value, {"final_sha", "manifest_sha256", "jobs"}, label)
    _require(
        type(bindings["final_sha"]) is str
        and COMMIT_RE.fullmatch(bindings["final_sha"]),
        f"{label} FINAL_SHA is malformed",
    )
    _require(
        type(bindings["manifest_sha256"]) is str
        and SHA256_RE.fullmatch(bindings["manifest_sha256"]),
        f"{label} manifest digest is malformed",
    )
    jobs = bindings["jobs"]
    _require(isinstance(jobs, list) and bool(jobs), f"{label} jobs are empty")
    normalized_jobs = []
    names: set[str] = set()
    jobs_seen: set[tuple[int, int, int]] = set()
    for index, value in enumerate(jobs):
        job = _exact_keys(
            value,
            {"name", "run_id", "run_attempt", "job_id"},
            f"{label} job {index}",
        )
        name = _job_name(job["name"], f"{label} job name")
        _require(name not in names, f"{label} repeats a job name")
        _require(
            type(job["run_id"]) is int and job["run_id"] > 0,
            f"{label} run id is invalid",
        )
        _require(
            type(job["run_attempt"]) is int and job["run_attempt"] > 0,
            f"{label} run attempt is invalid",
        )
        _require(
            type(job["job_id"]) is int and job["job_id"] > 0,
            f"{label} job id is invalid",
        )
        job_binding = (job["run_id"], job["run_attempt"], job["job_id"])
        _require(
            job_binding not in jobs_seen,
            f"{label} repeats a run/attempt/job binding",
        )
        names.add(name)
        jobs_seen.add(job_binding)
        normalized_jobs.append(dict(job))
    _require(
        [job["name"] for job in normalized_jobs] == list(OPERATIONS_V2_JOB_ORDER),
        f"{label} operations-v2 job order differs",
    )
    _require(
        normalized_jobs[0]["run_id"] == normalized_jobs[1]["run_id"]
        and normalized_jobs[2]["run_id"] == normalized_jobs[3]["run_id"]
        and normalized_jobs[0]["run_id"] != normalized_jobs[2]["run_id"],
        f"{label} CI and CodeQL workflow run bindings differ",
    )
    _require(
        normalized_jobs[0]["run_attempt"] == normalized_jobs[1]["run_attempt"]
        and normalized_jobs[2]["run_attempt"] == normalized_jobs[3]["run_attempt"],
        f"{label} workflow run attempts differ",
    )
    _require(
        len({job["job_id"] for job in normalized_jobs}) == len(normalized_jobs),
        f"{label} repeats a job id",
    )
    return {
        "final_sha": bindings["final_sha"],
        "manifest_sha256": bindings["manifest_sha256"],
        "jobs": normalized_jobs,
    }


def _validate_execution_witness_policy(
    value: Any,
    scopes: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    _require(isinstance(value, list), "execution witness policy must be a list")
    _require(
        [item.get("claim") if isinstance(item, Mapping) else None for item in value]
        == list(EXECUTION_CLAIM_ORDER),
        "execution witness claim order differs",
    )
    job_names = [job["name"] for job in bindings["jobs"]]
    validated = []
    traces_seen: set[tuple[str, str]] = set()
    for index, item in enumerate(value):
        witness = _exact_keys(
            item,
            {
                "claim",
                "scope",
                "trace_name",
                "validation_workflow",
                "validator_job_name",
            },
            f"execution witness policy {index}",
        )
        claim = witness["claim"]
        _require(
            witness["scope"] == EXECUTION_CLAIM_SCOPES[claim],
            f"execution witness {claim} scope differs",
        )
        trace_name = _safe_name(
            witness["trace_name"], f"execution witness {claim} trace name"
        )
        _require(
            witness["validation_workflow"] == EXECUTION_VALIDATION_WORKFLOW,
            f"execution witness {claim} validation workflow differs",
        )
        job_name = _job_name(
            witness["validator_job_name"],
            f"execution witness {claim} validator job name",
        )
        _require(
            job_name == EXECUTION_VALIDATOR_JOB_NAME and job_name in job_names,
            f"execution witness {claim} validator job differs",
        )
        scope = scopes[TECHNICAL_SCOPE_ORDER.index(witness["scope"])]
        _require(
            trace_name in [trace["name"] for trace in scope["traces"]],
            f"execution witness {claim} references an absent trusted trace",
        )
        trace_key = (witness["scope"], trace_name)
        _require(
            trace_key not in traces_seen,
            "execution witness policy reuses a trace across claims",
        )
        traces_seen.add(trace_key)
        validated.append(witness)
    return validated


def _validate_policy(
    policy: Any,
) -> tuple[
    dict[str, Any],
    list[Mapping[str, Any]],
    Mapping[str, Any],
    list[Mapping[str, Any]],
]:
    root = _exact_keys(
        policy,
        {
            "schema_version",
            "kind",
            "bindings",
            "scopes",
            "issue115",
            "execution_witnesses",
        },
        "publication policy",
    )
    _require(
        root["schema_version"] == SCHEMA_VERSION
        and type(root["schema_version"]) is int
        and root["kind"] == POLICY_KIND,
        "publication policy identity differs",
    )
    bindings = _validate_bindings(root["bindings"], "publication policy bindings")
    scopes = root["scopes"]
    _require(
        isinstance(scopes, list)
        and [item.get("name") if isinstance(item, Mapping) else None for item in scopes]
        == list(TECHNICAL_SCOPE_ORDER),
        "publication policy scope order differs",
    )
    validated = []
    for scope_index, value in enumerate(scopes):
        scope_name = TECHNICAL_SCOPE_ORDER[scope_index]
        scope = _exact_keys(
            value,
            {"name", "identities", "timings", "traces", "payloads", "correctness"},
            f"policy scope {scope_name}",
        )
        identities = scope["identities"]
        timings = scope["timings"]
        _require(
            isinstance(identities, list)
            and all(type(item) is str for item in identities),
            f"policy scope {scope_name} identities differ",
        )
        _require(
            isinstance(timings, list),
            f"policy scope {scope_name} timings differ",
        )
        for item in identities:
            _safe_name(item, f"policy scope {scope_name} identity")
        _require(
            len(identities) == len(set(identities)),
            f"policy scope {scope_name} repeats identity names",
        )
        timing_values = [
            _validate_timing_expectation(
                item, f"policy scope {scope_name} timing {index}"
            )
            for index, item in enumerate(timings)
        ]
        _require(
            len({item["name"] for item in timing_values}) == len(timing_values),
            f"policy scope {scope_name} repeats timing names",
        )
        traces = scope["traces"]
        _require(isinstance(traces, list), f"policy scope {scope_name} traces differ")
        trace_values = [
            _validate_trace_expectation(item, f"policy scope {scope_name} trace")
            for item in traces
        ]
        _require(
            len({item["name"] for item in trace_values}) == len(trace_values),
            f"policy scope {scope_name} repeats traces",
        )
        payloads = scope["payloads"]
        _require(
            isinstance(payloads, list), f"policy scope {scope_name} payloads differ"
        )
        payload_values = []
        for payload_index, item in enumerate(payloads):
            payload = _exact_keys(
                item,
                {"name", "media_type", "size_bytes", "sha256"},
                f"policy scope {scope_name} payload {payload_index}",
            )
            _payload_name(payload["name"], "policy payload name", payload["media_type"])
            _require(
                payload["media_type"] in PAYLOAD_MEDIA_TYPES,
                "policy payload media type differs",
            )
            _nonnegative_int(payload["size_bytes"], "policy payload size")
            _require(
                payload["size_bytes"] <= MAX_PAYLOAD_BYTES,
                "policy payload size exceeds its bound",
            )
            _require(
                type(payload["sha256"]) is str
                and SHA256_RE.fullmatch(payload["sha256"]),
                "policy payload digest differs",
            )
            payload_values.append(payload)
        _require(
            len({_archive_alias(item["name"]) for item in payload_values})
            == len(payload_values),
            f"policy scope {scope_name} aliases payloads",
        )
        correctness = scope["correctness"]
        _require(
            isinstance(correctness, list),
            f"policy scope {scope_name} correctness differs",
        )
        case_names: set[str] = set()
        for case_index, case_value in enumerate(correctness):
            case = _exact_keys(
                case_value,
                {"name", "captures"},
                f"policy correctness case {scope_name}/{case_index}",
            )
            case_name = _semantic_name(case["name"], "policy correctness case name")
            _require(
                case_name not in case_names,
                f"policy scope {scope_name} repeats correctness cases",
            )
            case_names.add(case_name)
            captures = case["captures"]
            _require(
                isinstance(captures, list) and bool(captures),
                f"policy correctness case {case_name} has no captures",
            )
            capture_names: set[str] = set()
            for capture_index, capture_value in enumerate(captures):
                capture = _exact_keys(
                    capture_value,
                    {"capture", "arrays"},
                    f"policy capture {scope_name}/{case_name}/{capture_index}",
                )
                capture_name = _capture_key(capture["capture"], "policy capture")
                _require(
                    capture_name not in capture_names,
                    f"policy correctness case {case_name} repeats captures",
                )
                capture_names.add(capture_name)
                arrays = capture["arrays"]
                _require(
                    isinstance(arrays, list) and bool(arrays),
                    f"policy capture {case_name}/{capture_name} has no arrays",
                )
                array_values = [
                    _comparison_policy(
                        item, f"policy array {scope_name}/{case_name}/{capture_name}"
                    )
                    for item in arrays
                ]
                _require(
                    len({item["name"] for item in array_values}) == len(array_values),
                    f"policy capture {case_name}/{capture_name} repeats arrays",
                )
        validated.append(scope)
    issue115 = _exact_keys(
        root["issue115"], {"timings", "profilers"}, "issue115 policy"
    )
    for role in ("timings", "profilers"):
        refs = issue115[role]
        _require(
            isinstance(refs, list) and bool(refs), f"issue115 {role} closure is empty"
        )
        seen: set[tuple[str, str]] = set()
        for index, item in enumerate(refs):
            ref = _exact_keys(item, {"scope", "name"}, f"issue115 {role} ref {index}")
            _require(
                ref["scope"] in TECHNICAL_SCOPE_ORDER, f"issue115 {role} scope differs"
            )
            _safe_name(ref["name"], f"issue115 {role} name")
            key = (ref["scope"], ref["name"])
            _require(key not in seen, f"issue115 {role} repeats a record")
            seen.add(key)
            scope = validated[TECHNICAL_SCOPE_ORDER.index(ref["scope"])]
            available = (
                [timing["name"] for timing in scope["timings"]]
                if role == "timings"
                else [trace["name"] for trace in scope["traces"]]
            )
            _require(
                ref["name"] in available, f"issue115 {role} references an absent record"
            )
    execution_witnesses = _validate_execution_witness_policy(
        root["execution_witnesses"], validated, bindings
    )
    return bindings, validated, issue115, execution_witnesses


def _inventory(scope: Mapping[str, Any]) -> tuple[int, int, list[list[str]]]:
    captures = arrays = 0
    records: list[list[str]] = []
    for case in scope["correctness"]:
        for capture in case["captures"]:
            captures += 1
            capture_name = _capture_key(capture["capture"], "capture")
            for array in capture["arrays"]:
                arrays += 1
                records.append(
                    [scope["name"], case["name"], capture_name, array["name"]]
                )
    return captures, arrays, records


def _project_correctness_scope(
    private_cases: Any,
    policy_scope: Mapping[str, Any],
    salt: bytes,
    bindings: Mapping[str, Any],
    openings: PrivateOpenings | None,
) -> list[dict[str, Any]]:
    scope_name = policy_scope["name"]
    policy_cases = policy_scope["correctness"]
    _require(
        isinstance(private_cases, list) and len(private_cases) == len(policy_cases),
        f"private scope {scope_name} correctness case closure differs",
    )
    result = []
    for private_case, policy_case in zip(private_cases, policy_cases, strict=True):
        case_name = policy_case["name"]
        case = _exact_keys(
            private_case,
            {"name", "captures"},
            f"private correctness case {scope_name}/{case_name}",
        )
        _require(
            case["name"] == case_name,
            f"private correctness case order differs in {scope_name}",
        )
        private_captures = case["captures"]
        policy_captures = policy_case["captures"]
        _require(
            isinstance(private_captures, list)
            and len(private_captures) == len(policy_captures),
            f"private correctness capture closure differs for {scope_name}/{case_name}",
        )
        public_captures = []
        for private_capture, policy_capture in zip(
            private_captures, policy_captures, strict=True
        ):
            capture = _exact_keys(
                private_capture,
                {"capture", "arrays"},
                f"private correctness capture {scope_name}/{case_name}",
            )
            capture_name = _capture_key(policy_capture["capture"], "policy capture")
            _require(
                type(capture["capture"]) is type(policy_capture["capture"])
                and capture["capture"] == policy_capture["capture"],
                f"private correctness capture order differs for {scope_name}/{case_name}",
            )
            private_arrays = capture["arrays"]
            policy_arrays = policy_capture["arrays"]
            _require(
                isinstance(private_arrays, list)
                and len(private_arrays) == len(policy_arrays),
                f"private correctness array closure differs for {scope_name}/{case_name}/{capture_name}",
            )
            public_arrays = []
            for private_array, policy_array in zip(
                private_arrays, policy_arrays, strict=True
            ):
                array_name = policy_array["name"]
                array = _exact_keys(
                    private_array,
                    {"name", "dtype", "shape", "reference_bytes", "candidate_bytes"},
                    f"private correctness array {scope_name}/{case_name}/{capture_name}/{array_name}",
                )
                private_shape = _shape(
                    array["shape"],
                    f"private correctness array {scope_name}/{case_name}/{capture_name}/{array_name} shape",
                )
                _require(
                    array["name"] == array_name
                    and array["dtype"] == policy_array["dtype"]
                    and private_shape == policy_array["shape"],
                    f"private correctness array descriptor differs for {scope_name}/{case_name}/{capture_name}/{array_name}",
                )
                reference_raw = array["reference_bytes"]
                candidate_raw = array["candidate_bytes"]
                reference = _decode_array(
                    reference_raw,
                    policy_array["dtype"],
                    policy_array["shape"],
                    f"reference {scope_name}/{case_name}/{capture_name}/{array_name}",
                )
                candidate = _decode_array(
                    candidate_raw,
                    policy_array["dtype"],
                    policy_array["shape"],
                    f"candidate {scope_name}/{case_name}/{capture_name}/{array_name}",
                )
                comparison = _compare_arrays(
                    reference,
                    candidate,
                    policy_array,
                    f"correctness {scope_name}/{case_name}/{capture_name}/{array_name}",
                )
                metadata = {
                    "bindings": bindings,
                    "scope": scope_name,
                    "case": case_name,
                    "capture": capture_name,
                    "array": array_name,
                    "dtype": policy_array["dtype"],
                    "shape": policy_array["shape"],
                }
                reference_commitment = _hmac_commit(
                    salt,
                    "issue123-correctness-array-v1/reference",
                    metadata,
                    reference_raw,
                )
                candidate_commitment = _hmac_commit(
                    salt,
                    "issue123-correctness-array-v1/candidate",
                    metadata,
                    candidate_raw,
                )
                if openings is not None:
                    openings._arrays[
                        (scope_name, case_name, capture_name, array_name, "reference")
                    ] = bytes(reference_raw)
                    openings._arrays[
                        (scope_name, case_name, capture_name, array_name, "candidate")
                    ] = bytes(candidate_raw)
                public_arrays.append(
                    {
                        "name": array_name,
                        "dtype": policy_array["dtype"],
                        "shape": list(policy_array["shape"]),
                        "element_count": _element_count(policy_array["shape"]),
                        "comparison": comparison,
                        "commitments": {
                            "algorithm": COMMITMENT_ALGORITHM,
                            "reference": reference_commitment,
                            "candidate": candidate_commitment,
                        },
                    }
                )
            public_captures.append(
                {"capture": policy_capture["capture"], "arrays": public_arrays}
            )
        result.append(
            {"scope": scope_name, "name": case_name, "captures": public_captures}
        )
    return result


def _project_execution_witness(
    witness_policy: Sequence[Mapping[str, Any]],
    trace_index: Mapping[tuple[str, str], Mapping[str, Any]],
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    jobs = {job["name"]: job for job in bindings["jobs"]}
    claims = []
    for mapping in witness_policy:
        claim = mapping["claim"]
        scope = mapping["scope"]
        trace_name = mapping["trace_name"]
        trace = trace_index[(scope, trace_name)]
        summary = trace["summary"]
        signatures = [
            [event["semantic_token"], event["phase"]] for event in trace["events"]
        ]
        tokens = {token for token, _phase in signatures}
        if claim == "cpu-eager":
            _require(
                summary["kernel_launches"] == 0
                and summary["cuda_graph_launches"] == 0
                and summary["compiled_region_events"] == 0
                and bool(
                    tokens
                    & {
                        "cpu-operation",
                        *WRITE_SEMANTIC_TOKENS.values(),
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
                and "compiled-region" not in tokens
                and bool(tokens & {"kernel", "nccl-kernel"}),
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
            "summary": trace["summary"],
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
        "bindings": bindings,
        "claims": claims,
    }


def project_publication(
    private_bundle: Any,
    policy: Any,
    *,
    private_openings: PrivateOpenings | None = None,
) -> dict[str, Any]:
    """Project untrusted private evidence under a separate trusted policy.

    Every call creates a fresh 32-byte salt using ``secrets``.  The salt is
    never returned and is only copied into an explicitly supplied
    :class:`PrivateOpenings` object for private verification.
    """

    bindings, policy_scopes, issue115_policy, witness_policy = _validate_policy(policy)
    private = _exact_keys(
        private_bundle,
        {"schema_version", "kind", "bindings", "scopes"},
        "private publication bundle",
    )
    _require(
        private["schema_version"] == SCHEMA_VERSION
        and type(private["schema_version"]) is int
        and private["kind"] == PRIVATE_INPUT_KIND,
        "private publication bundle identity differs",
    )
    private_bindings = _validate_bindings(
        private["bindings"], "private publication bindings"
    )
    _require(private_bindings == bindings, "private publication bindings are stale")
    scopes = private["scopes"]
    _require(
        isinstance(scopes, list)
        and [item.get("name") if isinstance(item, Mapping) else None for item in scopes]
        == list(TECHNICAL_SCOPE_ORDER),
        "private technical scope closure differs",
    )
    private_salt = (
        private_openings._salt
        if isinstance(private_openings, PrivateOpenings)
        and private_openings._salt is not None
        else secrets.token_bytes(32)
    )
    _require(
        type(private_salt) is bytes and len(private_salt) == 32,
        "commitment salt must be exactly 32 private bytes",
    )
    _require(
        private_openings is None or isinstance(private_openings, PrivateOpenings),
        "private openings sink is invalid",
    )
    if private_openings is not None:
        _require(
            not private_openings._populated
            and not private_openings._identities
            and not private_openings._arrays,
            "private openings sink was already used",
        )
    staged_openings = PrivateOpenings() if private_openings is not None else None
    if staged_openings is not None:
        staged_openings._salt = bytes(private_salt)

    public_scopes = []
    correctness_cases = []
    timing_index: dict[tuple[str, str], dict[str, Any]] = {}
    trace_index: dict[tuple[str, str], dict[str, Any]] = {}
    forbidden_values: list[str] = []
    correctness_inventory: list[list[str]] = []
    total_captures = total_arrays = 0
    private_scopes: list[Mapping[str, Any]] = []
    identity_bytes: dict[tuple[str, str], bytes] = {}

    # Collect every private identity before scanning the first payload.  An
    # earlier scope therefore cannot smuggle a later scope's host/user/device
    # identity through its trusted payload digest.
    for scope_index, (private_value, policy_scope) in enumerate(
        zip(scopes, policy_scopes, strict=True)
    ):
        scope_name = TECHNICAL_SCOPE_ORDER[scope_index]
        private_scope = _exact_keys(
            private_value,
            {
                "name",
                "satisfied",
                "identities",
                "timings",
                "traces",
                "payloads",
                "correctness",
            },
            f"private scope {scope_name}",
        )
        _require(
            private_scope["name"] == scope_name and private_scope["satisfied"] is True,
            f"private scope {scope_name} is not satisfied",
        )
        identities = private_scope["identities"]
        _require(
            type(identities) is dict
            and list(identities) == list(policy_scope["identities"]),
            f"private scope {scope_name} identity closure differs",
        )
        for identity_name in policy_scope["identities"]:
            raw = _private_json_bytes(
                identities[identity_name],
                f"private identity {scope_name}/{identity_name}",
            )
            identity_bytes[(scope_name, identity_name)] = raw
            forbidden_values.extend(_identity_strings(json.loads(raw.decode("utf-8"))))
        private_scopes.append(private_scope)
    forbidden_values = list(dict.fromkeys(forbidden_values))
    _require(
        len(forbidden_values) <= MAX_IDENTITY_SCAN_VALUES,
        "private identities contain too many exact scan values",
    )
    forbidden_context = _privacy_scan_context(forbidden_values)

    for scope_index, (private_scope, policy_scope) in enumerate(
        zip(private_scopes, policy_scopes, strict=True)
    ):
        scope_name = TECHNICAL_SCOPE_ORDER[scope_index]
        identities = private_scope["identities"]
        identity_commitments = []
        for name in policy_scope["identities"]:
            raw = identity_bytes[(scope_name, name)]
            commitment = _identity_commitment(
                private_salt, scope_name, name, raw, bindings
            )
            identity_commitments.append(
                {
                    "name": name,
                    "algorithm": COMMITMENT_ALGORITHM,
                    "commitment": commitment,
                }
            )
            if staged_openings is not None:
                staged_openings._identities[(scope_name, name)] = bytes(raw)

        private_timings = private_scope["timings"]
        _require(
            isinstance(private_timings, list)
            and len(private_timings) == len(policy_scope["timings"]),
            f"private scope {scope_name} timing closure differs",
        )
        timings = []
        for value, expectation in zip(
            private_timings, policy_scope["timings"], strict=True
        ):
            name = expectation["name"]
            timing = _project_timing(
                value, expectation, f"private timing {scope_name}/{name}"
            )
            timings.append(timing)
            timing_index[(scope_name, name)] = timing

        private_traces = private_scope["traces"]
        trace_policies = policy_scope["traces"]
        _require(
            isinstance(private_traces, list)
            and len(private_traces) == len(trace_policies),
            f"private scope {scope_name} trace closure differs",
        )
        traces = []
        for value, expectation in zip(private_traces, trace_policies, strict=True):
            trace_value = _exact_keys(
                value, {"name", "trace_bytes"}, f"private trace {scope_name}"
            )
            trace_name = expectation["name"]
            _require(
                trace_value["name"] == trace_name,
                f"private trace order differs in {scope_name}",
            )
            normalized = normalize_trace(
                trace_value["trace_bytes"],
                label=f"private trace {scope_name}/{trace_name}",
            )
            _enforce_trace_expectation(
                normalized, expectation, f"private trace {scope_name}/{trace_name}"
            )
            trace = {"name": trace_name, **normalized}
            traces.append(trace)
            trace_index[(scope_name, trace_name)] = trace

        private_payloads = private_scope["payloads"]
        payload_policies = policy_scope["payloads"]
        _require(
            isinstance(private_payloads, list)
            and len(private_payloads) == len(payload_policies),
            f"private scope {scope_name} payload closure differs",
        )
        payloads = []
        for value, expected in zip(private_payloads, payload_policies, strict=True):
            payload = _exact_keys(
                value, {"name", "media_type", "bytes"}, f"private payload {scope_name}"
            )
            _require(
                payload["name"] == expected["name"]
                and payload["media_type"] == expected["media_type"],
                f"private payload descriptor differs in {scope_name}",
            )
            raw = payload["bytes"]
            _require(
                type(raw) is bytes,
                f"private payload {scope_name}/{expected['name']} is not bytes",
            )
            _require(
                len(raw) == expected["size_bytes"]
                and hashlib.sha256(raw).hexdigest() == expected["sha256"],
                f"private payload {scope_name}/{expected['name']} differs from trusted bytes",
            )
            scan_payload(
                expected["name"],
                raw,
                media_type=expected["media_type"],
                forbidden_values=forbidden_context,
            )
            payloads.append(
                {
                    "name": expected["name"],
                    "media_type": expected["media_type"],
                    "size_bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "scan_contract": SCAN_CONTRACT,
                }
            )

        projected_cases = _project_correctness_scope(
            private_scope["correctness"],
            policy_scope,
            private_salt,
            bindings,
            staged_openings,
        )
        correctness_cases.extend(projected_cases)
        captures, arrays, inventory = _inventory(policy_scope)
        total_captures += captures
        total_arrays += arrays
        correctness_inventory.extend(inventory)
        closure = {
            "identity_names": list(policy_scope["identities"]),
            "timing_names": [item["name"] for item in policy_scope["timings"]],
            "trace_names": [item["name"] for item in policy_scope["traces"]],
            "payload_names": [item["name"] for item in policy_scope["payloads"]],
            "correctness_case_names": [
                item["name"] for item in policy_scope["correctness"]
            ],
            "correctness_capture_count": captures,
            "correctness_array_count": arrays,
        }
        public_scopes.append(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": SCOPE_KIND,
                "scope": scope_name,
                "bindings": bindings,
                "identity_commitments": identity_commitments,
                "timings": timings,
                "traces": traces,
                "payloads": payloads,
                "closure": closure,
            }
        )

    correctness = {
        "schema_version": SCHEMA_VERSION,
        "kind": CORRECTNESS_KIND,
        "bindings": bindings,
        "algorithm": COMMITMENT_ALGORITHM,
        "cases": correctness_cases,
        "closure": {
            "scope_order": list(TECHNICAL_SCOPE_ORDER),
            "case_count": len(correctness_cases),
            "capture_count": total_captures,
            "array_count": total_arrays,
            "inventory_sha256": _canonical_sha256(correctness_inventory),
        },
    }
    execution_witness = _project_execution_witness(
        witness_policy, trace_index, bindings
    )

    raw_records = []
    for ref in issue115_policy["timings"]:
        record = timing_index[(ref["scope"], ref["name"])]
        raw_records.append({"scope": ref["scope"], **record})
    raw_timing = {
        "schema_version": SCHEMA_VERSION,
        "kind": RAW_TIMING_KIND,
        "contract_id": RAW_TIMING_CONTRACT,
        "bindings": bindings,
        "records": raw_records,
        "closure": {
            "record_count": len(raw_records),
            "inventory_sha256": _canonical_sha256(
                [[item["scope"], item["name"]] for item in raw_records]
            ),
        },
    }
    profiler_records = []
    for ref in issue115_policy["profilers"]:
        record = trace_index[(ref["scope"], ref["name"])]
        profiler_records.append({"scope": ref["scope"], **record})
    event_profiler = {
        "schema_version": SCHEMA_VERSION,
        "kind": EVENT_PROFILER_KIND,
        "contract_id": EVENT_PROFILER_CONTRACT,
        "bindings": bindings,
        "records": profiler_records,
        "closure": {
            "record_count": len(profiler_records),
            "inventory_sha256": _canonical_sha256(
                [[item["scope"], item["name"]] for item in profiler_records]
            ),
        },
    }
    projection = {
        "schema_version": SCHEMA_VERSION,
        "kind": PROJECTION_KIND,
        "bindings": bindings,
        "technical_scopes": public_scopes,
        "correctness_commitments": correctness,
        "execution_witness": execution_witness,
        "raw_timing": raw_timing,
        "event_profiler": event_profiler,
    }
    archive_json_size = 0
    for index, scope in enumerate(public_scopes):
        scope_raw = canonical_json_bytes(scope, label=f"public scope {index}")
        _require(
            len(scope_raw) <= MAX_PUBLIC_JSON_BYTES,
            f"public scope {index} exceeds its publication byte bound",
        )
        archive_json_size += len(scope_raw)
    correctness_raw = canonical_json_bytes(
        correctness, label="public correctness commitments"
    )
    _require(
        len(correctness_raw) <= MAX_PUBLIC_JSON_BYTES,
        "public correctness commitments exceed their publication byte bound",
    )
    archive_json_size += len(correctness_raw)
    witness_raw = canonical_json_bytes(
        execution_witness, label="public execution witness"
    )
    _require(
        len(witness_raw) <= MAX_PUBLIC_JSON_BYTES,
        "public execution witness exceeds its publication byte bound",
    )
    archive_json_size += len(witness_raw)
    _require(
        archive_json_size <= MAX_PUBLIC_ARCHIVE_BYTES,
        "public technical documents exceed their archive byte bound",
    )
    for document, label in (
        (raw_timing, "public raw timing"),
        (event_profiler, "public event profiler"),
    ):
        _require(
            len(canonical_json_bytes(document, label=label)) <= MAX_PUBLIC_JSON_BYTES,
            f"{label} exceeds its publication byte bound",
        )
    public_raw = canonical_json_bytes(projection, label="public projection")
    scan_public_bytes(
        public_raw, label="public projection", forbidden_values=forbidden_context
    )
    _require(
        private_salt.hex().encode("ascii") not in public_raw,
        "private salt entered public bytes",
    )
    if private_openings is not None:
        assert staged_openings is not None and staged_openings._salt is not None
        private_openings._salt = bytes(staged_openings._salt)
        private_openings._identities.update(staged_openings._identities)
        private_openings._arrays.update(staged_openings._arrays)
        private_openings._populated = True
    return projection


@dataclass(frozen=True)
class PublicationSourceSpecification:
    """Validated private source specification with its trusted policy."""

    document: dict[str, Any]
    base: Path
    policy: dict[str, Any]
    sha256: str
    evaluator_bindings: tuple["EvaluatorTargetBinding", ...]


@dataclass(frozen=True)
class MaterializedPublicationInputs:
    """Private projector input plus the sanitized transitive binding inventory."""

    private_bundle: dict[str, Any]
    technical_inventory: dict[str, Any]
    technical_input_root: str
    source_specification_sha256: str


def _binding_normalize(value: Any, label: str = "binding value") -> Any:
    if value is None or type(value) in {bool, int}:
        return value
    if isinstance(value, float):
        _require(math.isfinite(value), f"{label} contains a non-finite number")
        return value
    if isinstance(value, str):
        _require(
            unicodedata.normalize("NFC", value) == value,
            f"{label} contains non-NFC text",
        )
        return value
    if isinstance(value, list):
        return [
            _binding_normalize(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        _require(
            all(type(key) is str for key in value),
            f"{label} contains a non-string key",
        )
        items = list(value.items())
        normalized_keys = [unicodedata.normalize("NFC", key) for key, _item in items]
        _require(
            len(set(normalized_keys)) == len(normalized_keys)
            and all(
                key == normalized
                for (key, _item), normalized in zip(items, normalized_keys, strict=True)
            ),
            f"{label} contains a non-NFC or colliding key",
        )
        return {
            key: _binding_normalize(item, f"{label} member {ordinal}")
            for ordinal, (key, item) in enumerate(sorted(items))
        }
    raise PrivacyError(f"{label} is not canonical JSON data")


def binding_canonical_json_bytes(value: Any) -> bytes:
    """Canonical JSON used only by the new private binding contracts."""

    normalized = _binding_normalize(value)
    try:
        rendered = json.dumps(
            normalized,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise PrivacyError("binding value is not canonicalizable") from error
    return (rendered + "\n").encode("utf-8")


def tagged_canonical_sha256(domain: str, value: Any) -> str:
    """Return the design-specified length-framed, domain-separated digest."""

    _require(
        type(domain) is str
        and domain.isascii()
        and re.fullmatch(r"[a-z0-9.-]+", domain) is not None,
        "binding digest domain is invalid",
    )
    raw = binding_canonical_json_bytes(value)
    framed = (
        TAGGED_DIGEST_PREFIX
        + domain.encode("ascii")
        + b"\x00"
        + len(raw).to_bytes(8, "big")
        + raw
    )
    return hashlib.sha256(framed).hexdigest()


def _tagged_hmac(salt: bytes, domain: str, value: Any) -> str:
    _require(type(salt) is bytes and len(salt) == 32, "private salt is invalid")
    raw = binding_canonical_json_bytes(value)
    framed = (
        TAGGED_DIGEST_PREFIX
        + domain.encode("ascii")
        + b"\x00"
        + len(raw).to_bytes(8, "big")
        + raw
    )
    return hmac.new(salt, framed, hashlib.sha256).hexdigest()


def _permitted_system_path_alias(path: Path, metadata: os.stat_result) -> bool:
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


def _audit_absolute_path(path: Path, label: str, *, require_leaf: bool) -> Path:
    anchor = Path(path.anchor)
    try:
        anchor.lstat()
    except OSError:
        raise PrivacyError(f"{label} path root is unavailable") from None
    current = anchor
    used_alias = False
    parts = path.parts[1:]
    for ordinal, part in enumerate(parts):
        current = current / part
        leaf = ordinal == len(parts) - 1
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if leaf and not require_leaf:
                break
            raise PrivacyError(f"{label} path is unavailable") from None
        except OSError:
            raise PrivacyError(f"{label} path is unavailable") from None
        if stat.S_ISLNK(metadata.st_mode):
            _require(
                _permitted_system_path_alias(current, metadata),
                f"{label} path uses a symbolic link",
            )
            used_alias = True
        elif not leaf:
            _require(
                stat.S_ISDIR(metadata.st_mode),
                f"{label} ancestor is not a directory",
            )
    try:
        resolved = current.resolve(strict=require_leaf)
    except OSError:
        raise PrivacyError(f"{label} path is unavailable") from None
    if used_alias:
        resolved_current = Path(resolved.anchor)
        for part in resolved.parts[1:]:
            resolved_current = resolved_current / part
            try:
                resolved_metadata = resolved_current.lstat()
            except OSError:
                if resolved_current == resolved and not require_leaf:
                    break
                raise PrivacyError(f"{label} resolved path is unavailable") from None
            _require(
                not stat.S_ISLNK(resolved_metadata.st_mode),
                f"{label} resolved path uses a symbolic link",
            )
    return resolved


def _lexical_path_without_symlinks(
    path_value: Path | str,
    label: str,
    *,
    require_leaf: bool,
) -> Path:
    """Audit every lexical component before returning a canonical path."""

    try:
        raw = os.fspath(path_value)
    except TypeError:
        raise PrivacyError(f"{label} path is invalid") from None
    _require(
        type(raw) is str and bool(raw) and "\x00" not in raw,
        f"{label} path is invalid",
    )
    _require(
        all(part not in {".", ".."} for part in raw.split(os.sep)),
        f"{label} path contains a dot segment",
    )
    supplied = Path(raw)
    absolute = supplied if supplied.is_absolute() else Path.cwd() / supplied
    return _audit_absolute_path(absolute, label, require_leaf=require_leaf)


def _private_file_bytes(
    path_value: Path | str,
    label: str,
    *,
    maximum: int = MAX_PAYLOAD_BYTES,
) -> tuple[Path, bytes]:
    path = _lexical_path_without_symlinks(path_value, label, require_leaf=True)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise PrivacyError(f"{label} is unavailable") from None
    try:
        before = os.fstat(descriptor)
        _require(
            stat.S_ISREG(before.st_mode) and 0 < before.st_size <= maximum,
            f"{label} identity or byte bound differs",
        )
        chunks = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            _require(bool(chunk), f"{label} changed while being read")
            chunks.append(chunk)
            remaining -= len(chunk)
        _require(not os.read(descriptor, 1), f"{label} changed while being read")
        after = os.fstat(descriptor)
    except Exception:
        os.close(descriptor)
        raise
    os.close(descriptor)
    raw = b"".join(chunks)
    _require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        and before.st_size == len(raw)
        and path.lstat().st_ino == before.st_ino,
        f"{label} identity or byte bound differs",
    )
    return path, raw


def preflight_private_output_path(
    path_value: Path | str,
    *,
    label: str,
    forbidden_roots: Sequence[Path | str],
    prospective_forbidden_roots: Sequence[Path | str] = (),
) -> tuple[Path, Path]:
    """Resolve one output and reject protected-root overlap without mutation."""

    try:
        supplied = Path(path_value)
        supplied_name = os.fspath(supplied.name)
    except TypeError:
        raise PrivacyError("private authority output path is invalid") from None
    _require(
        type(supplied_name) is str
        and supplied_name not in {"", ".", ".."}
        and supplied_name == os.path.basename(supplied_name),
        f"{label} leaf name is invalid",
    )
    _lexical_path_without_symlinks(supplied, label, require_leaf=False)
    parent_value = supplied.parent if supplied.parent != Path(".") else Path.cwd()
    parent = _lexical_path_without_symlinks(
        parent_value, f"{label} parent", require_leaf=True
    )
    final = parent / supplied_name
    canonical_roots: list[Path] = []
    for root_value in forbidden_roots:
        root = _lexical_path_without_symlinks(
            root_value, "private authority forbidden root", require_leaf=True
        )
        try:
            root_metadata = root.lstat()
        except OSError:
            raise PrivacyError(
                "private authority forbidden root is unavailable"
            ) from None
        _require(
            stat.S_ISDIR(root_metadata.st_mode)
            and not stat.S_ISLNK(root_metadata.st_mode),
            "private authority forbidden root is not a directory",
        )
        if root not in canonical_roots:
            canonical_roots.append(root)
    for root_value in prospective_forbidden_roots:
        root = _lexical_path_without_symlinks(
            root_value,
            "prospective private authority forbidden root",
            require_leaf=False,
        )
        if root.exists():
            try:
                root_metadata = root.lstat()
            except OSError:
                raise PrivacyError(
                    "prospective private authority forbidden root is unavailable"
                ) from None
            _require(
                stat.S_ISDIR(root_metadata.st_mode)
                and not stat.S_ISLNK(root_metadata.st_mode),
                "prospective private authority forbidden root is not a directory",
            )
        if root not in canonical_roots:
            canonical_roots.append(root)
    _require(
        all(not final.is_relative_to(root) for root in canonical_roots),
        "private authority output overlaps a forbidden root",
    )
    return final, parent


@dataclass
class _PrivateTempOwnership:
    """Retained handle and name used for identity-scoped failure cleanup."""

    name: str
    descriptor: int | None
    identity: tuple[int, int] | None = None


def write_private_authority_file(
    path_value: Path | str,
    raw: bytes,
    *,
    label: str,
    forbidden_roots: Sequence[Path | str] = (),
    before_commit: Callable[[], None] | None = None,
) -> Path:
    """Atomically publish one private authority file without replacement."""

    _require(type(raw) is bytes and bool(raw), f"{label} bytes are invalid")
    final, parent = preflight_private_output_path(
        path_value,
        label=label,
        forbidden_roots=forbidden_roots,
    )
    try:
        parent_metadata = parent.lstat()
    except OSError:
        raise PrivacyError("private authority parent is unavailable") from None
    _require(
        stat.S_ISDIR(parent_metadata.st_mode)
        and not stat.S_ISLNK(parent_metadata.st_mode)
        and stat.S_IMODE(parent_metadata.st_mode) == 0o700,
        "private authority parent must be a mode-0700 directory",
    )
    required_dir_fd = {os.open, os.stat, os.link, os.unlink}
    _require(
        hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
        and required_dir_fd.issubset(os.supports_dir_fd)
        and os.stat in os.supports_fd
        and os.stat in os.supports_follow_symlinks
        and os.link in os.supports_follow_symlinks,
        "private authority atomic publication is unsupported",
    )
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        parent_fd = os.open(parent, directory_flags)
    except OSError:
        raise PrivacyError("private authority parent could not be opened") from None
    ownership: _PrivateTempOwnership | None = None
    final_linked = False
    failure_message: str | None = None
    committed_failure = False

    def identity_for(name: str) -> tuple[int, int] | None:
        try:
            metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return None
        except OSError:
            raise PrivacyError(
                "private authority identity could not be checked"
            ) from None
        return metadata.st_dev, metadata.st_ino

    def retain_created_identity() -> bool:
        if ownership is None:
            return True
        if ownership.identity is not None:
            return True
        if ownership.descriptor is None:
            return False
        try:
            metadata = os.fstat(ownership.descriptor)
        except OSError:
            try:
                metadata = os.stat(ownership.descriptor)
            except OSError:
                return False
        if not stat.S_ISREG(metadata.st_mode):
            return False
        ownership.identity = metadata.st_dev, metadata.st_ino
        return True

    def unlink_if_created(name: str | None) -> None:
        if (
            name is not None
            and ownership is not None
            and ownership.identity is not None
            and identity_for(name) == ownership.identity
        ):
            os.unlink(name, dir_fd=parent_fd)

    def verify_committed_leaf() -> None:
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        try:
            descriptor = os.open(final.name, flags, dir_fd=parent_fd)
        except OSError:
            raise PrivacyError(
                "private authority committed leaf could not be reopened"
            ) from None
        close_failure = False
        try:
            before = os.fstat(descriptor)
            chunks: list[bytes] = []
            remaining = len(raw)
            while remaining:
                chunk = os.read(descriptor, min(remaining, 1024 * 1024))
                _require(
                    bool(chunk),
                    "private authority committed leaf bytes differ",
                )
                chunks.append(chunk)
                remaining -= len(chunk)
            _require(
                not os.read(descriptor, 1),
                "private authority committed leaf bytes differ",
            )
            after = os.fstat(descriptor)
            _require(
                stat.S_ISREG(before.st_mode)
                and stat.S_IMODE(before.st_mode) == 0o600
                and before.st_size == len(raw)
                and (before.st_dev, before.st_ino) == ownership.identity
                and (
                    before.st_dev,
                    before.st_ino,
                    before.st_mode,
                    before.st_size,
                    before.st_mtime_ns,
                )
                == (
                    after.st_dev,
                    after.st_ino,
                    after.st_mode,
                    after.st_size,
                    after.st_mtime_ns,
                )
                and b"".join(chunks) == raw,
                "private authority committed leaf bytes differ",
            )
        finally:
            try:
                os.close(descriptor)
            except OSError:
                close_failure = True
        if close_failure:
            raise PrivacyError(
                "private authority committed leaf could not be closed"
            ) from None

    try:
        opened_parent = os.fstat(parent_fd)
        _require(
            stat.S_ISDIR(opened_parent.st_mode)
            and stat.S_IMODE(opened_parent.st_mode) == 0o700
            and (opened_parent.st_dev, opened_parent.st_ino)
            == (parent_metadata.st_dev, parent_metadata.st_ino),
            "private authority parent identity differs",
        )
        _require(
            identity_for(final.name) is None,
            "private authority output already exists",
        )
        temp_name = f".{final.name}.tmp-{secrets.token_hex(16)}"
        file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
        temp_fd = os.open(temp_name, file_flags, 0o600, dir_fd=parent_fd)
        ownership = _PrivateTempOwnership(temp_name, temp_fd)
        initial = os.fstat(temp_fd)
        ownership.identity = initial.st_dev, initial.st_ino
        os.fchmod(temp_fd, 0o600)
        view = memoryview(raw)
        while view:
            written = os.write(temp_fd, view)
            _require(written > 0, "private authority write made no progress")
            view = view[written:]
        os.fsync(temp_fd)
        metadata = os.fstat(temp_fd)
        _require(
            stat.S_ISREG(metadata.st_mode)
            and stat.S_IMODE(metadata.st_mode) == 0o600
            and metadata.st_size == len(raw)
            and (metadata.st_dev, metadata.st_ino) == ownership.identity,
            "private authority temporary file differs",
        )
        ownership.descriptor = None
        os.close(temp_fd)
        if before_commit is not None:
            _require(
                callable(before_commit),
                "private authority commit barrier is invalid",
            )
            before_commit()
        os.link(
            ownership.name,
            final.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        final_linked = True
        _require(
            identity_for(final.name) == ownership.identity
            and identity_for(ownership.name) == ownership.identity,
            "private authority published identity differs",
        )
        os.unlink(ownership.name, dir_fd=parent_fd)
        ownership.name = ""
        os.fsync(parent_fd)
        verify_committed_leaf()
    except Exception:
        committed_failure = final_linked
        cleanup_failed = not retain_created_identity()
        if ownership is not None and ownership.descriptor is not None:
            descriptor = ownership.descriptor
            ownership.descriptor = None
            try:
                os.close(descriptor)
            except OSError:
                cleanup_failed = True
        try:
            unlink_if_created(ownership.name if ownership is not None else None)
            os.fsync(parent_fd)
        except Exception:
            cleanup_failed = True
        try:
            os.close(parent_fd)
        except OSError:
            cleanup_failed = True
        failure_message = (
            "private authority file was committed but final verification failed"
            if committed_failure
            else (
                "private authority cleanup could not be completed"
                if cleanup_failed
                else "private authority file could not be committed"
            )
        )
    if failure_message is not None:
        if committed_failure:
            raise PrivateAuthorityCommitError(
                failure_message,
                committed=True,
            ) from None
        raise PrivacyError(failure_message) from None
    try:
        os.close(parent_fd)
    except OSError:
        raise PrivateAuthorityCommitError(
            "private authority file was committed but final verification failed",
            committed=True,
        ) from None
    return final


def _canonical_bundle_source_path(value: Any, label: str) -> str:
    _require(type(value) is str and value and "\\" not in value, f"{label} differs")
    path = PurePosixPath(value)
    _require(
        not path.is_absolute()
        and all(part not in {"", ".", ".."} for part in path.parts)
        and path.as_posix() == value,
        f"{label} is not a canonical bundle path",
    )
    return value


def _expected_source_targets(policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    _bindings, scopes, _issue115, _witness = _validate_policy(policy)
    targets: list[dict[str, Any]] = []
    for scope in scopes:
        scope_name = scope["name"]
        for name in scope["identities"]:
            targets.append(
                {"scope": scope_name, "semantic_role": "identity", "name": name}
            )
        for timing in scope["timings"]:
            targets.append(
                {
                    "scope": scope_name,
                    "semantic_role": "timing",
                    "name": timing["name"],
                }
            )
        for trace in scope["traces"]:
            targets.append(
                {
                    "scope": scope_name,
                    "semantic_role": "trace",
                    "name": trace["name"],
                }
            )
        for payload in scope["payloads"]:
            targets.append(
                {
                    "scope": scope_name,
                    "semantic_role": "payload",
                    "name": payload["name"],
                }
            )
        for case in scope["correctness"]:
            for capture in case["captures"]:
                for array in capture["arrays"]:
                    for role in ("reference", "candidate"):
                        targets.append(
                            {
                                "scope": scope_name,
                                "semantic_role": f"correctness-{role}",
                                "case": case["name"],
                                "capture": capture["capture"],
                                "name": array["name"],
                                "dtype": array["dtype"],
                                "shape": list(array["shape"]),
                            }
                        )
    return targets


@dataclass(frozen=True)
class TargetKey:
    """Canonical, caller-independent identity of one projected value."""

    scope: str
    semantic_role: str
    name: str
    case: str | None
    capture: str | None
    dtype: str | None
    shape: tuple[int, ...] | None


@dataclass(frozen=True)
class RoleSelector:
    """One exact evaluator role and its code-owned extraction selector."""

    role: str
    selector: dict[str, Any]


@dataclass(frozen=True)
class EvaluatorTargetBinding:
    """In-memory authority binding; never serialized into public data."""

    target_key: TargetKey
    primary: RoleSelector
    corroborating: tuple[RoleSelector, ...] = ()
    archive_share_group: str | None = None


def _target_key(target: Mapping[str, Any]) -> TargetKey:
    capture = (
        _capture_key(target["capture"], "publication target capture")
        if "capture" in target
        else None
    )
    return TargetKey(
        scope=target["scope"],
        semantic_role=target["semantic_role"],
        name=target["name"],
        case=target.get("case"),
        capture=capture,
        dtype=target.get("dtype"),
        shape=(
            tuple(target["shape"]) if isinstance(target.get("shape"), list) else None
        ),
    )


# A JSON, timing, trace, or payload target is enabled only by an exact
# code-reviewed entry.  The concrete production policy is intentionally not
# present in this repository, so this map stays empty rather than inferring
# authority from a caller-supplied role, path, digest, or selected value.
CODE_OWNED_LITERAL_TARGET_BINDINGS: Mapping[TargetKey, EvaluatorTargetBinding] = {}


def _validate_source_selector(
    selector: Any, target: Mapping[str, Any], label: str
) -> dict[str, Any]:
    selector = _exact_keys(
        selector, set(selector) if isinstance(selector, Mapping) else set(), label
    )
    variants = set(selector)
    _require(
        variants
        in ({"whole_bytes"}, {"json_pointer", "expected_type"}, {"npz_member"}),
        f"{label} must contain exactly one supported selector",
    )
    if variants == {"whole_bytes"}:
        _require(
            selector["whole_bytes"] is True, f"{label} whole-bytes selector differs"
        )
        _require(
            target["semantic_role"] in {"trace", "payload"},
            f"{label} whole bytes are incompatible with the semantic role",
        )
    elif "json_pointer" in selector:
        pointer = selector["json_pointer"]
        expected_type = selector["expected_type"]
        _require(
            type(pointer) is str
            and (pointer == "" or pointer.startswith("/"))
            and expected_type
            in {"object", "array", "string", "number", "integer", "boolean", "null"}
            and target["semantic_role"] in {"identity", "timing"},
            f"{label} JSON selector differs",
        )
    else:
        member = _exact_keys(
            selector["npz_member"],
            {"name", "dtype", "shape", "byte_order", "order"},
            f"{label} NPZ member",
        )
        _require(
            target["semantic_role"].startswith("correctness-")
            and member["name"] == target["name"]
            and member["dtype"] == target["dtype"]
            and member["shape"] == target["shape"]
            and member["byte_order"] == "little"
            and member["order"] == "C",
            f"{label} NPZ member contract differs from policy",
        )
    return dict(selector)


def _validate_source_record_shape(
    value: Any,
    target: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    record = _exact_keys(
        value,
        {
            "scope",
            "semantic_role",
            "completion_role",
            "source_path",
            "bundle_path",
            "sha256",
            "size_bytes",
            "media_type",
            "selector",
        },
        label,
    )
    _require(
        record["scope"] == target["scope"]
        and record["semantic_role"] == target["semantic_role"]
        and type(record["completion_role"]) is str
        and bool(record["completion_role"])
        and type(record["source_path"]) is str
        and bool(record["source_path"])
        and isinstance(record["sha256"], str)
        and SHA256_RE.fullmatch(record["sha256"]) is not None
        and type(record["size_bytes"]) is int
        and 0 < record["size_bytes"] <= MAX_PAYLOAD_BYTES
        and type(record["media_type"]) is str
        and bool(record["media_type"]),
        f"{label} identity or descriptor differs",
    )
    normalized = dict(record)
    normalized["bundle_path"] = _canonical_bundle_source_path(
        record["bundle_path"], f"{label} bundle path"
    )
    normalized["selector"] = _validate_source_selector(
        record["selector"], target, f"{label} selector"
    )
    return normalized


def _validate_source_record(
    value: Any,
    target: Mapping[str, Any],
    binding: EvaluatorTargetBinding,
    label: str,
) -> dict[str, Any]:
    record = _validate_source_record_shape(value, target, label)
    _require(
        record["completion_role"] == binding.primary.role,
        "publication source evaluator role assertion differs",
    )
    _require(
        binding_canonical_json_bytes(record["selector"])
        == binding_canonical_json_bytes(binding.primary.selector),
        "publication source selector differs from code-owned binding",
    )
    return record


def load_publication_source_spec(
    path: Path | str,
    policy: Mapping[str, Any],
    *,
    completion_index: Path | str,
    policy_sha256: str | None = None,
) -> PublicationSourceSpecification:
    """Load and strictly bind the sole production projection source format."""

    source_path, raw = _private_file_bytes(
        path, "publication source specification", maximum=MAX_PUBLIC_JSON_BYTES
    )
    document = _strict_json(raw, "publication source specification")
    _require(
        raw == binding_canonical_json_bytes(document),
        "publication source specification is not canonical JSON",
    )
    document = dict(
        _exact_keys(
            document,
            {
                "schema_version",
                "kind",
                "candidate_evidence",
                "policy_sha256",
                "scope_order",
                "scope_artifacts",
                "runtime_receipt_order",
                "runtime_receipts",
                "sources",
            },
            "publication source specification",
        )
    )
    _require(
        type(document["schema_version"]) is int
        and document["schema_version"] == PUBLICATION_SOURCE_SPECIFICATION_VERSION
        and document["kind"] == PUBLICATION_SOURCE_SPECIFICATION_KIND,
        "publication source specification identity differs",
    )
    bindings, _scopes, _issue115, _witness = _validate_policy(policy)
    canonical_policy_sha256 = hashlib.sha256(canonical_json_bytes(policy)).hexdigest()
    _require(
        document["policy_sha256"] == canonical_policy_sha256
        and (policy_sha256 is None or policy_sha256 == canonical_policy_sha256),
        "publication source specification policy digest differs",
    )
    candidate = _exact_keys(
        document["candidate_evidence"],
        {"candidate_git_commit", "candidate_git_status", "manifest_sha256"},
        "publication source candidate",
    )
    _require(
        candidate["candidate_git_commit"] == bindings["final_sha"]
        and candidate["candidate_git_status"] == ""
        and candidate["manifest_sha256"] == bindings["manifest_sha256"],
        "publication source candidate differs from policy bindings",
    )
    _require(
        document["scope_order"] == list(TECHNICAL_SCOPE_ORDER)
        and document["runtime_receipt_order"] == list(RUNTIME_RECEIPT_ORDER),
        "publication source fixed order differs",
    )
    _exact_keys(
        document["scope_artifacts"],
        set(TECHNICAL_SCOPE_ORDER),
        "publication source completion scope mappings",
    )
    _binding_normalize(
        document["scope_artifacts"], "publication source completion scope mappings"
    )
    runtime = document["runtime_receipts"]
    _require(
        isinstance(runtime, list) and len(runtime) == len(RUNTIME_RECEIPT_ORDER),
        "publication source runtime receipt closure differs",
    )
    normalized_runtime = []
    for index, (record, role) in enumerate(
        zip(runtime, RUNTIME_RECEIPT_ORDER, strict=True)
    ):
        record = dict(
            _exact_keys(
                record,
                {"role", "source_path", "bundle_path", "sha256", "size_bytes"},
                f"publication source runtime receipt {index}",
            )
        )
        _require(
            record["role"] == role
            and type(record["source_path"]) is str
            and bool(record["source_path"])
            and isinstance(record["sha256"], str)
            and SHA256_RE.fullmatch(record["sha256"]) is not None
            and type(record["size_bytes"]) is int
            and 0 < record["size_bytes"] <= MAX_PUBLIC_JSON_BYTES,
            f"publication source runtime receipt {index} differs",
        )
        record["bundle_path"] = _canonical_bundle_source_path(
            record["bundle_path"], f"publication source runtime receipt {index} path"
        )
        normalized_runtime.append(record)
    targets = _expected_source_targets(policy)
    sources = document["sources"]
    _require(
        isinstance(sources, list) and len(sources) == len(targets),
        "publication source selector closure differs",
    )
    _reader, completion_document, catalog, raw_by_path = _completion_source_state(
        completion_index,
        {
            "technical_inventory": {
                "candidate_evidence": candidate,
                "scope_artifacts": document["scope_artifacts"],
            }
        },
    )
    evaluator_bindings = _code_owned_evaluator_bindings(
        policy,
        completion_document,
        catalog,
        raw_by_path,
    )
    normalized_sources = [
        _validate_source_record(
            value,
            target,
            binding,
            f"publication source {index}",
        )
        for index, (value, target, binding) in enumerate(
            zip(sources, targets, evaluator_bindings, strict=True)
        )
    ]
    for index, (record, binding) in enumerate(
        zip(normalized_sources, evaluator_bindings, strict=True)
    ):
        descriptor = catalog[binding.primary.role]
        expected_descriptor = {
            "path": record["bundle_path"],
            "sha256": record["sha256"],
            "size_bytes": record["size_bytes"],
            "media_type": record["media_type"],
            "candidate_evidence": candidate,
        }
        _require(
            descriptor == expected_descriptor,
            "publication target descriptor differs from evaluator-consumed B1",
        )
    source_keys = [
        binding_canonical_json_bytes(_sanitized_source_record(item))
        for item in normalized_sources
    ]
    _require(
        len(set(source_keys)) == len(source_keys),
        "publication source selector entries are duplicated",
    )
    document["runtime_receipts"] = normalized_runtime
    document["sources"] = normalized_sources
    return PublicationSourceSpecification(
        document=document,
        base=source_path.parent,
        policy=json.loads(binding_canonical_json_bytes(policy)),
        sha256=hashlib.sha256(raw).hexdigest(),
        evaluator_bindings=evaluator_bindings,
    )


def _source_path(base: Path, value: str) -> Path:
    supplied = Path(value)
    return supplied if supplied.is_absolute() else base / supplied


def _json_pointer(value: Any, pointer: str, label: str) -> Any:
    current = value
    if pointer == "":
        return current
    for raw_token in pointer.removeprefix("/").split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        _require("~" not in token, f"{label} has an invalid JSON pointer escape")
        if isinstance(current, Mapping):
            _require(token in current, f"{label} does not resolve")
            current = current[token]
        elif isinstance(current, list):
            _require(
                re.fullmatch(r"0|[1-9][0-9]*", token) is not None,
                f"{label} array token differs",
            )
            ordinal = int(token)
            _require(ordinal < len(current), f"{label} does not resolve")
            current = current[ordinal]
        else:
            raise PrivacyError(f"{label} traverses a scalar")
    return current


def _require_json_type(value: Any, expected: str, label: str) -> None:
    matches = {
        "object": isinstance(value, Mapping),
        "array": isinstance(value, list),
        "string": type(value) is str,
        "number": type(value) in {int, float},
        "integer": type(value) is int,
        "boolean": type(value) is bool,
        "null": value is None,
    }
    _require(matches[expected], f"{label} JSON type differs")


def _materialized_json_value(value: Any, label: str) -> Any:
    """Convert the strict parser's exact decimals to projector JSON numbers."""

    if type(value) is Decimal:
        converted = float(value)
        _require(math.isfinite(converted), f"{label} number is outside its bound")
        return converted
    if isinstance(value, list):
        return [
            _materialized_json_value(item, f"{label}[{ordinal}]")
            for ordinal, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        return {
            key: _materialized_json_value(item, f"{label}.{key}")
            for key, item in value.items()
        }
    return value


def _select_source(raw: bytes, selector: Mapping[str, Any], label: str) -> Any:
    if "whole_bytes" in selector:
        return bytes(raw)
    if "json_pointer" in selector:
        document = _strict_json(raw, label)
        selected = _json_pointer(document, selector["json_pointer"], label)
        _require_json_type(selected, selector["expected_type"], label)
        return _materialized_json_value(selected, label)
    member = selector["npz_member"]
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            _require(
                member["name"] in archive.files,
                f"{label} NPZ member is absent",
            )
            array = np.asarray(archive[member["name"]])
    except PrivacyError:
        raise
    except (OSError, TypeError, ValueError, zipfile.BadZipFile) as error:
        raise PrivacyError(f"{label} NPZ member could not be loaded") from error
    try:
        expected_dtype = np.dtype(member["dtype"])
    except TypeError as error:
        raise PrivacyError(f"{label} NPZ dtype differs") from error
    _require(
        array.dtype == expected_dtype and list(array.shape) == member["shape"],
        f"{label} NPZ dtype or shape differs",
    )
    normalized_dtype = expected_dtype.newbyteorder("<")
    normalized = np.ascontiguousarray(array.astype(normalized_dtype, copy=False))
    return normalized.tobytes(order="C")


def _target_npz_selector(target: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "npz_member": {
            "name": target["name"],
            "dtype": target["dtype"],
            "shape": list(target["shape"]),
            "byte_order": "little",
            "order": "C",
        }
    }


def _catalog_document(
    role: str,
    catalog: Mapping[str, Mapping[str, Any]],
    raw_by_path: Mapping[str, bytes],
) -> Mapping[str, Any]:
    descriptor = catalog.get(role)
    _require(
        isinstance(descriptor, Mapping)
        and descriptor.get("media_type") == "application/json",
        "publication target evaluator role closure differs",
    )
    raw = raw_by_path.get(descriptor["path"])
    _require(
        type(raw) is bytes,
        "publication target evaluator role closure differs",
    )
    document = _strict_json(raw, "code-owned evaluator document")
    _require(
        isinstance(document, Mapping),
        "publication target evaluator role closure differs",
    )
    return document


def _cpu_correctness_binding(
    target: Mapping[str, Any],
    catalog: Mapping[str, Mapping[str, Any]],
    raw_by_path: Mapping[str, bytes],
) -> EvaluatorTargetBinding:
    root = "/artifacts/cpu/correctness_index"
    document = _catalog_document(root, catalog, raw_by_path)
    artifacts = document.get("artifacts")
    _require(
        isinstance(artifacts, list),
        "publication target evaluator role closure differs",
    )
    matches = [
        ordinal
        for ordinal, record in enumerate(artifacts)
        if isinstance(record, Mapping) and record.get("case") == target.get("case")
    ]
    _require(
        len(matches) == 1,
        "publication target has no code-owned evaluator binding",
    )
    ordinal = matches[0]
    side = target["semantic_role"].removeprefix("correctness-")
    role = f"{root}/@document/artifacts/{ordinal}/{side}"
    selector = _target_npz_selector(target)
    corroborating: tuple[RoleSelector, ...] = ()
    share_group = None
    if side == "reference":
        corroborating = tuple(
            RoleSelector(
                (
                    "/artifacts/single_gpu/cuda_gates/@document/"
                    "cuda_suite_gate/correctness_indexes/"
                    f"{mode}/source_artifact/@document/artifacts/"
                    f"{ordinal}/reference"
                ),
                copy.deepcopy(selector),
            )
            for mode in range(2)
        )
        share_group = f"native-reference-{ordinal}"
    return EvaluatorTargetBinding(
        target_key=_target_key(target),
        primary=RoleSelector(role, selector),
        corroborating=corroborating,
        archive_share_group=share_group,
    )


def _differential_correctness_binding(
    target: Mapping[str, Any],
    catalog: Mapping[str, Mapping[str, Any]],
    raw_by_path: Mapping[str, bytes],
) -> EvaluatorTargetBinding:
    roots = {
        "policy_paired_real": (
            "/artifacts/policy_paired_real/paired_real_differential"
        ),
        "single_gpu": "/artifacts/single_gpu/correctness",
    }
    root = roots[target["scope"]]
    document = _catalog_document(root, catalog, raw_by_path)
    cases = document.get("cases")
    _require(
        isinstance(cases, list),
        "publication target evaluator role closure differs",
    )
    case_matches = [
        (ordinal, record)
        for ordinal, record in enumerate(cases)
        if isinstance(record, Mapping) and record.get("case") == target.get("case")
    ]
    _require(
        len(case_matches) == 1,
        "publication target has no code-owned evaluator binding",
    )
    case_ordinal, record = case_matches[0]
    groups = record.get("projection_groups")
    _require(
        isinstance(groups, list),
        "publication target evaluator role closure differs",
    )
    capture = _target_key(target).capture
    group_matches = [
        ordinal
        for ordinal, group in enumerate(groups)
        if isinstance(group, list)
        and any(
            type(step) is int
            and _capture_key(step, "evaluator projection capture") == capture
            for step in group
        )
    ]
    _require(
        len(group_matches) == 1,
        "publication target has no code-owned evaluator binding",
    )
    side = target["semantic_role"].removeprefix("correctness-")
    role = f"{root}/@document/cases/{case_ordinal}/{side}/{group_matches[0]}"
    return EvaluatorTargetBinding(
        target_key=_target_key(target),
        primary=RoleSelector(role, _target_npz_selector(target)),
    )


def _binding_for_target(
    target: Mapping[str, Any],
    catalog: Mapping[str, Mapping[str, Any]],
    raw_by_path: Mapping[str, bytes],
) -> EvaluatorTargetBinding:
    key = _target_key(target)
    if target["semantic_role"].startswith("correctness-"):
        if target["scope"] == "cpu":
            return _cpu_correctness_binding(target, catalog, raw_by_path)
        if target["scope"] in {"policy_paired_real", "single_gpu"}:
            return _differential_correctness_binding(target, catalog, raw_by_path)
    literal = CODE_OWNED_LITERAL_TARGET_BINDINGS.get(key)
    _require(
        type(literal) is EvaluatorTargetBinding and literal.target_key == key,
        "publication target has no code-owned evaluator binding",
    )
    return literal


def _code_owned_evaluator_bindings(
    policy: Mapping[str, Any],
    index: Mapping[str, Any],
    catalog: Mapping[str, Mapping[str, Any]],
    raw_by_path: Mapping[str, bytes],
) -> tuple[EvaluatorTargetBinding, ...]:
    """Derive every target from closed evaluator semantics, never source hints."""

    targets = _expected_source_targets(policy)
    _require(
        isinstance(index.get("artifacts"), Mapping)
        and all(scope in index["artifacts"] for scope in TECHNICAL_SCOPE_ORDER),
        "publication target/evaluator binding coverage differs",
    )
    target_keys = tuple(_target_key(target) for target in targets)
    _require(
        len(set(target_keys)) == len(target_keys),
        "publication target/evaluator binding coverage differs",
    )
    bindings = tuple(
        _binding_for_target(target, catalog, raw_by_path) for target in targets
    )
    _require(
        tuple(binding.target_key for binding in bindings) == target_keys,
        "publication target/evaluator binding coverage differs",
    )
    used_assertions: set[tuple[TargetKey, bytes, str | None]] = set()
    for target, binding in zip(targets, bindings, strict=True):
        selectors = (binding.primary, *binding.corroborating)
        for role_selector in selectors:
            descriptor = catalog.get(role_selector.role)
            _require(
                isinstance(descriptor, Mapping)
                and (
                    role_selector.role.startswith(f"/artifacts/{target['scope']}/")
                    or binding.archive_share_group is not None
                    and role_selector.role.startswith("/artifacts/single_gpu/")
                ),
                "publication target evaluator role closure differs",
            )
            raw = raw_by_path.get(descriptor["path"])
            _require(
                type(raw) is bytes,
                "publication target evaluator role closure differs",
            )
            _select_source(
                raw,
                role_selector.selector,
                "code-owned evaluator source",
            )
        primary_descriptor = catalog[binding.primary.role]
        if binding.archive_share_group is not None:
            _require(
                all(
                    catalog[item.role] == primary_descriptor
                    for item in binding.corroborating
                ),
                "publication target shared archive binding differs",
            )
        else:
            primary_value = _select_source(
                raw_by_path[primary_descriptor["path"]],
                binding.primary.selector,
                "code-owned evaluator source",
            )
            _require(
                all(
                    _selected_source_equal(
                        primary_value,
                        _select_source(
                            raw_by_path[catalog[item.role]["path"]],
                            item.selector,
                            "code-owned corroborating evaluator source",
                        ),
                    )
                    for item in binding.corroborating
                ),
                "publication target evaluator role closure differs",
            )
        assertion = (
            binding.target_key,
            binding_canonical_json_bytes(binding.primary.selector),
            binding.archive_share_group,
        )
        _require(
            assertion not in used_assertions,
            "publication target/evaluator binding coverage differs",
        )
        used_assertions.add(assertion)
    return bindings


def _require_evaluator_bindings_consumed(
    bindings: Sequence[EvaluatorTargetBinding],
    catalog: Mapping[str, Mapping[str, Any]],
    descriptor_access_log: Sequence[Mapping[str, Any]],
) -> None:
    consumed = []
    for ordinal, access in enumerate(descriptor_access_log):
        access = _exact_keys(
            access,
            {"label", "descriptor"},
            f"evaluator descriptor access {ordinal}",
        )
        descriptor = access["descriptor"]
        _require(
            type(access["label"]) is str
            and bool(access["label"])
            and isinstance(descriptor, Mapping),
            "publication target evaluator role closure differs",
        )
        consumed.append(dict(descriptor))
    for binding in bindings:
        for item in (binding.primary, *binding.corroborating):
            _require(
                any(descriptor == catalog[item.role] for descriptor in consumed),
                "publication target descriptor differs from evaluator-consumed B1",
            )


def _sanitized_source_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in (
            "scope",
            "semantic_role",
            "completion_role",
            "bundle_path",
            "sha256",
            "size_bytes",
            "media_type",
            "selector",
        )
    }


def _sanitized_runtime_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: record[key] for key in ("role", "bundle_path", "sha256", "size_bytes")}


def _materialize_records(
    *,
    policy: Mapping[str, Any],
    candidate: Mapping[str, Any],
    source_records: Sequence[Mapping[str, Any]],
    raw_by_record: Sequence[bytes],
    evaluator_bindings: Sequence[EvaluatorTargetBinding],
) -> dict[str, Any]:
    targets = _expected_source_targets(policy)
    scopes = [
        {
            "name": name,
            "satisfied": True,
            "identities": {},
            "timings": [],
            "traces": [],
            "payloads": [],
            "correctness": [],
        }
        for name in TECHNICAL_SCOPE_ORDER
    ]
    by_scope = {scope["name"]: scope for scope in scopes}
    correctness: dict[tuple[str, str], dict[str, Any]] = {}
    captures: dict[tuple[str, str, str], dict[str, Any]] = {}
    arrays: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for index, (record, target, raw, binding) in enumerate(
        zip(
            source_records,
            targets,
            raw_by_record,
            evaluator_bindings,
            strict=True,
        )
    ):
        selected = _select_source(
            raw,
            binding.primary.selector,
            f"publication source {index}",
        )
        scope = by_scope[target["scope"]]
        role = target["semantic_role"]
        if role == "identity":
            scope["identities"][target["name"]] = selected
        elif role == "timing":
            scope["timings"].append(
                {"name": target["name"], "unit": "seconds", "samples": selected}
            )
        elif role == "trace":
            scope["traces"].append({"name": target["name"], "trace_bytes": selected})
        elif role == "payload":
            policy_scope = next(
                item for item in policy["scopes"] if item["name"] == target["scope"]
            )
            descriptor = next(
                item
                for item in policy_scope["payloads"]
                if item["name"] == target["name"]
            )
            scope["payloads"].append(
                {
                    "name": target["name"],
                    "media_type": descriptor["media_type"],
                    "bytes": selected,
                }
            )
        else:
            capture_key = (target["scope"], target["case"], str(target["capture"]))
            case_key = capture_key[:2]
            array_key = (*capture_key, target["name"])
            if case_key not in correctness:
                correctness[case_key] = {"name": target["case"], "captures": []}
                scope["correctness"].append(correctness[case_key])
            if capture_key not in captures:
                captures[capture_key] = {"capture": target["capture"], "arrays": []}
                correctness[case_key]["captures"].append(captures[capture_key])
            if array_key not in arrays:
                arrays[array_key] = {
                    "name": target["name"],
                    "dtype": target["dtype"],
                    "shape": target["shape"],
                }
                captures[capture_key]["arrays"].append(arrays[array_key])
            arrays[array_key][f"{role.removeprefix('correctness-')}_bytes"] = selected
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": PRIVATE_INPUT_KIND,
        "bindings": copy.deepcopy(policy["bindings"]),
        "scopes": scopes,
    }


def materialize_publication_inputs(
    specification: PublicationSourceSpecification,
    *,
    runtime_receipt_paths: Mapping[str, Path | str] | None = None,
) -> MaterializedPublicationInputs:
    """Materialize the v1 private projector mapping from exact source bytes."""

    _require(
        isinstance(specification, PublicationSourceSpecification),
        "publication source specification is not validated",
    )
    document = specification.document
    runtime_by_role = (
        {item["role"]: item["source_path"] for item in document["runtime_receipts"]}
        if runtime_receipt_paths is None
        else dict(runtime_receipt_paths)
    )
    _require(
        list(runtime_by_role) == list(RUNTIME_RECEIPT_ORDER),
        "runtime receipt path order differs",
    )
    resolved_paths: list[tuple[Path, tuple[str, int, str]]] = []
    for index, record in enumerate(document["runtime_receipts"]):
        path, raw = _private_file_bytes(
            _source_path(specification.base, str(runtime_by_role[record["role"]])),
            f"runtime receipt {record['role']}",
            maximum=MAX_PUBLIC_JSON_BYTES,
        )
        _require(
            len(raw) == record["size_bytes"]
            and hmac.compare_digest(hashlib.sha256(raw).hexdigest(), record["sha256"]),
            f"runtime receipt {record['role']} bytes differ",
        )
        resolved_paths.append(
            (path, (record["bundle_path"], record["size_bytes"], record["sha256"]))
        )
    source_raw = []
    for index, record in enumerate(document["sources"]):
        path, raw = _private_file_bytes(
            _source_path(specification.base, record["source_path"]),
            f"publication source {index}",
        )
        _require(
            len(raw) == record["size_bytes"]
            and hmac.compare_digest(hashlib.sha256(raw).hexdigest(), record["sha256"]),
            f"publication source {index} bytes differ",
        )
        resolved_paths.append(
            (path, (record["bundle_path"], record["size_bytes"], record["sha256"]))
        )
        source_raw.append(raw)
    identities: dict[tuple[int, int], tuple[Path, tuple[str, int, str]]] = {}
    for path, descriptor in resolved_paths:
        identity = (path.stat().st_dev, path.stat().st_ino)
        previous = identities.get(identity)
        _require(
            previous is None or previous == (path, descriptor),
            "publication source paths are aliased or inconsistently described",
        )
        identities[identity] = (path, descriptor)
    private = _materialize_records(
        policy=specification.policy,
        candidate=document["candidate_evidence"],
        source_records=document["sources"],
        raw_by_record=source_raw,
        evaluator_bindings=specification.evaluator_bindings,
    )
    inventory = {
        "candidate_evidence": copy.deepcopy(document["candidate_evidence"]),
        "policy_sha256": document["policy_sha256"],
        "scope_order": list(TECHNICAL_SCOPE_ORDER),
        "scope_artifacts": copy.deepcopy(document["scope_artifacts"]),
        "runtime_receipts": [
            _sanitized_runtime_record(item) for item in document["runtime_receipts"]
        ],
        "sources": [_sanitized_source_record(item) for item in document["sources"]],
    }
    return MaterializedPublicationInputs(
        private_bundle=private,
        technical_inventory=inventory,
        technical_input_root=tagged_canonical_sha256(
            TECHNICAL_INPUT_INVENTORY_DOMAIN, inventory
        ),
        source_specification_sha256=specification.sha256,
    )


def publication_binding_context(
    materialized: MaterializedPublicationInputs,
    projection: Mapping[str, Any],
    asset_ledger: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ledger = [dict(item) for item in asset_ledger]
    return {
        "candidate_evidence": copy.deepcopy(
            materialized.technical_inventory["candidate_evidence"]
        ),
        "policy_sha256": materialized.technical_inventory["policy_sha256"],
        "source_specification_sha256": materialized.source_specification_sha256,
        "technical_inventory": copy.deepcopy(materialized.technical_inventory),
        "technical_input_root": materialized.technical_input_root,
        "public_projection_sha256": tagged_canonical_sha256(
            PUBLIC_PROJECTION_DOMAIN, projection
        ),
        "public_asset_ledger": ledger,
        "public_asset_ledger_sha256": tagged_canonical_sha256(
            PUBLIC_ASSET_LEDGER_DOMAIN, ledger
        ),
    }


def serialize_private_openings(
    openings: PrivateOpenings, binding_context: Mapping[str, Any]
) -> bytes:
    """Serialize salt and binding metadata, never raw opening values."""

    _require(
        isinstance(openings, PrivateOpenings)
        and openings._populated
        and openings._salt is not None,
        "private openings are not populated",
    )
    context = _binding_normalize(dict(binding_context), "private binding context")
    document = {
        "schema_version": PROTECTED_OPENINGS_VERSION,
        "kind": PROTECTED_OPENINGS_KIND,
        "salt_hex": openings._salt.hex(),
        "binding_context": context,
        "binding": {
            "algorithm": "HMAC-SHA-256",
            "domain": PRIVATE_OPENING_BINDING_DOMAIN,
            "value": _tagged_hmac(
                openings._salt, PRIVATE_OPENING_BINDING_DOMAIN, context
            ),
        },
    }
    raw = binding_canonical_json_bytes(document)
    for private_raw in (*openings._identities.values(), *openings._arrays.values()):
        _require(
            private_raw not in raw, "raw private opening entered protected metadata"
        )
    return raw


def load_private_openings(
    source: Path | str | bytes | Mapping[str, Any],
) -> tuple[PrivateOpenings, dict[str, Any]]:
    """Load and authenticate a protected v1 openings sidecar."""

    if isinstance(source, (Path, str)):
        _path, raw = _private_file_bytes(
            source, "protected publication openings", maximum=MAX_PUBLIC_JSON_BYTES
        )
        try:
            file_metadata = _path.lstat()
            parent_metadata = _path.parent.lstat()
        except OSError as error:
            raise PrivacyError(
                "protected publication privacy metadata is unavailable"
            ) from error
        _require(
            stat.S_IMODE(file_metadata.st_mode) == 0o600
            and stat.S_ISDIR(parent_metadata.st_mode)
            and not stat.S_ISLNK(parent_metadata.st_mode)
            and stat.S_IMODE(parent_metadata.st_mode) == 0o700,
            "protected publication openings must be mode 0600 in a mode-0700 directory",
        )
        document = _strict_json(raw, "protected publication openings")
        _require(
            raw == binding_canonical_json_bytes(document),
            "protected publication openings are not canonical JSON",
        )
    elif type(source) is bytes:
        raw = source
        document = _strict_json(raw, "protected publication openings")
        _require(
            raw == binding_canonical_json_bytes(document),
            "protected publication openings are not canonical JSON",
        )
    else:
        document = dict(source)
    document = _exact_keys(
        document,
        {"schema_version", "kind", "salt_hex", "binding_context", "binding"},
        "protected publication openings",
    )
    _require(
        type(document["schema_version"]) is int
        and document["schema_version"] == PROTECTED_OPENINGS_VERSION
        and document["kind"] == PROTECTED_OPENINGS_KIND
        and isinstance(document["salt_hex"], str)
        and re.fullmatch(r"[0-9a-f]{64}", document["salt_hex"]) is not None,
        "protected publication openings identity differs",
    )
    try:
        salt = bytes.fromhex(document["salt_hex"])
    except ValueError as error:
        raise PrivacyError("protected publication salt differs") from error
    binding = _exact_keys(
        document["binding"],
        {"algorithm", "domain", "value"},
        "protected publication binding",
    )
    context = _binding_normalize(
        document["binding_context"], "protected publication binding context"
    )
    expected = _tagged_hmac(salt, PRIVATE_OPENING_BINDING_DOMAIN, context)
    _require(
        binding["algorithm"] == "HMAC-SHA-256"
        and binding["domain"] == PRIVATE_OPENING_BINDING_DOMAIN
        and isinstance(binding["value"], str)
        and hmac.compare_digest(binding["value"], expected),
        "protected publication binding authentication differs",
    )
    return PrivateOpenings(salt), dict(context)


def verify_private_openings(
    source: Path | str | bytes | Mapping[str, Any],
    expected_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _openings, context = load_private_openings(source)
    if expected_context is not None:
        _require(
            hmac.compare_digest(
                binding_canonical_json_bytes(context),
                binding_canonical_json_bytes(expected_context),
            ),
            "protected publication binding context differs",
        )
    return context


def _pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _completion_source_state(
    index_path: Path | str,
    context: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, dict[str, Any]], dict[str, bytes]]:
    try:
        from benchmarks import issue123_completion as completion
    except (ImportError, OSError) as error:
        raise PrivacyError("completion bundle adapter is unavailable") from error
    checked_index, index_raw = completion._bounded_regular_file_bytes(
        Path(index_path),
        "completion binding index",
        max_bytes=completion.MAX_INDEX_BYTES,
    )
    index = completion._strict_json_bytes(
        index_raw, "completion binding index", max_bytes=completion.MAX_INDEX_BYTES
    )
    completion._exact_keys(
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
        "completion binding index",
    )
    _require(
        type(index["schema_version"]) is int
        and index["schema_version"] == completion.INDEX_SCHEMA_VERSION
        and index["kind"] == completion.INDEX_KIND
        and index["issue"] == 123,
        "completion binding index identity differs",
    )
    inventory = context.get("technical_inventory")
    _require(isinstance(inventory, Mapping), "technical binding inventory is absent")
    _require(
        index["candidate_evidence"] == inventory.get("candidate_evidence")
        and {scope: index["artifacts"].get(scope) for scope in TECHNICAL_SCOPE_ORDER}
        == inventory.get("scope_artifacts"),
        "completion first-five scope mapping differs from protected bindings",
    )
    reader = completion.ArtifactReader(
        checked_index.parent, index["candidate_evidence"], index["payloads"]
    )
    catalog: dict[str, dict[str, Any]] = {}
    raw_by_path: dict[str, bytes] = {}
    active: set[tuple[str, str]] = set()

    def walk(value: Any, role: str) -> None:
        if isinstance(value, Mapping):
            if set(value) == {
                "path",
                "sha256",
                "size_bytes",
                "media_type",
                "candidate_evidence",
            }:
                loaded = reader.load(
                    value,
                    "completion projection source descriptor",
                    json_document=value["media_type"] == completion.MEDIA_TYPE_JSON,
                )
                _require(
                    role not in catalog, "completion descriptor role is duplicated"
                )
                catalog[role] = dict(loaded.descriptor)
                raw_by_path[loaded.descriptor["path"]] = loaded.raw
                key = (loaded.descriptor["path"], role)
                if loaded.document is not None and key not in active:
                    active.add(key)
                    walk(loaded.document, role + "/@document")
                    active.remove(key)
                return
            if set(value) == {"path", "sha256", "size_bytes", "media_type"}:
                registered = reader._registry.get(value["path"])
                _require(
                    isinstance(registered, Mapping)
                    and all(registered.get(key) == item for key, item in value.items()),
                    "completion path-independent descriptor is unregistered",
                )
                walk(registered, role)
                return
            for key in sorted(value):
                walk(value[key], role + "/" + _pointer_token(str(key)))
        elif isinstance(value, list):
            for ordinal, item in enumerate(value):
                walk(item, role + f"/{ordinal}")

    for scope in TECHNICAL_SCOPE_ORDER:
        walk(index["artifacts"][scope], f"/artifacts/{scope}")
    return reader, index, catalog, raw_by_path


def completion_source_catalog(
    index_path: Path | str,
    protected_openings: Path | str | bytes | Mapping[str, Any],
) -> list[dict[str, Any]]:
    """List private completion descriptor roles for source-spec preparation."""

    _openings, context = load_private_openings(protected_openings)
    _reader, _index, catalog, _raw = _completion_source_state(index_path, context)
    return [
        {"completion_role": role, "descriptor": catalog[role]}
        for role in sorted(catalog)
    ]


def _selected_source_equal(first: Any, second: Any) -> bool:
    if type(first) is bytes or type(second) is bytes:
        return (
            type(first) is bytes
            and type(second) is bytes
            and hmac.compare_digest(first, second)
        )
    try:
        return hmac.compare_digest(
            binding_canonical_json_bytes(first),
            binding_canonical_json_bytes(second),
        )
    except TypeError, ValueError:
        return False


def verify_publication_bundle_binding(
    *,
    index_path: Path | str,
    protected_openings: Path | str | bytes | Mapping[str, Any],
    policy: Mapping[str, Any],
    public_assets: Mapping[str, Path | str | bytes],
    runtime_receipt_paths: Mapping[str, Path | str],
    manifest_path: Path | str | None = None,
) -> dict[str, Any]:
    """Rebuild projection/assets from final B1 and reject any A/B splice."""

    try:
        from benchmarks import issue123_completion as completion
        from benchmarks import issue123_publication as publication
    except (ImportError, OSError) as error:
        raise PrivacyError("publication binding validators are unavailable") from error
    openings, context = load_private_openings(protected_openings)
    _exact_keys(
        context,
        {
            "candidate_evidence",
            "policy_sha256",
            "source_specification_sha256",
            "technical_inventory",
            "technical_input_root",
            "public_projection_sha256",
            "public_asset_ledger",
            "public_asset_ledger_sha256",
        },
        "protected publication binding context",
    )
    policy_raw = canonical_json_bytes(policy)
    _require(
        hashlib.sha256(policy_raw).hexdigest() == context["policy_sha256"],
        "protected publication policy digest differs",
    )
    _reader, index, catalog, raw_by_path = _completion_source_state(index_path, context)
    receipt_paths = dict(runtime_receipt_paths)
    _require(
        list(receipt_paths) == list(RUNTIME_RECEIPT_ORDER),
        "final runtime receipt order differs",
    )
    technical = context["technical_inventory"]
    _exact_keys(
        technical,
        {
            "candidate_evidence",
            "policy_sha256",
            "scope_order",
            "scope_artifacts",
            "runtime_receipts",
            "sources",
        },
        "protected technical inventory",
    )
    _require(
        technical["scope_order"] == list(TECHNICAL_SCOPE_ORDER)
        and [item.get("role") for item in technical["runtime_receipts"]]
        == list(RUNTIME_RECEIPT_ORDER),
        "protected technical scope or receipt order differs",
    )
    registry = {item["path"]: item for item in index["payloads"]}
    for record in technical["runtime_receipts"]:
        _exact_keys(
            record,
            {"role", "bundle_path", "sha256", "size_bytes"},
            f"protected runtime receipt {record.get('role', 'unknown')}",
        )
        registered = registry.get(record["bundle_path"])
        _require(
            isinstance(registered, Mapping)
            and registered.get("sha256") == record["sha256"]
            and registered.get("size_bytes") == record["size_bytes"]
            and registered.get("media_type") == completion.MEDIA_TYPE_JSON,
            f"protected runtime receipt {record['role']} is not bundle registered",
        )
        _path, external_raw = _private_file_bytes(
            receipt_paths[record["role"]],
            f"external runtime receipt {record['role']}",
            maximum=MAX_PUBLIC_JSON_BYTES,
        )
        bundled_path = (
            Path(index_path)
            .resolve(strict=True)
            .parent.joinpath(*PurePosixPath(record["bundle_path"]).parts)
        )
        _bundle_path, bundled_raw = _private_file_bytes(
            bundled_path,
            f"bundled runtime receipt {record['role']}",
            maximum=MAX_PUBLIC_JSON_BYTES,
        )
        _require(
            hmac.compare_digest(external_raw, bundled_raw)
            and len(external_raw) == record["size_bytes"]
            and hmac.compare_digest(
                hashlib.sha256(external_raw).hexdigest(), record["sha256"]
            ),
            f"external runtime receipt {record['role']} differs from final B1",
        )
    source_records = technical["sources"]
    targets = _expected_source_targets(policy)
    _require(
        isinstance(source_records, list) and len(source_records) == len(targets),
        "protected projection source closure differs",
    )
    descriptor_access_log: list[dict[str, Any]] = []
    structural = completion.evaluate_completion(
        index_path,
        manifest_path,
        [receipt_paths[role] for role in RUNTIME_RECEIPT_ORDER],
        descriptor_access_log=descriptor_access_log,
    )
    _require(
        all(
            structural.get("scopes", {}).get(scope, {}).get("satisfied") is True
            for scope in TECHNICAL_SCOPE_ORDER
        ),
        "final B1 first five scopes are not structurally validated",
    )
    evaluator_bindings = _code_owned_evaluator_bindings(
        policy,
        index,
        catalog=catalog,
        raw_by_path=raw_by_path,
    )
    source_raw: list[bytes] = []
    for ordinal, (record, target, binding) in enumerate(
        zip(source_records, targets, evaluator_bindings, strict=True)
    ):
        expected = dict(record)
        expected["source_path"] = "protected-source-not-serialized"
        validated = _validate_source_record(
            expected,
            target,
            binding,
            f"protected projection source {ordinal}",
        )
        descriptor = catalog[binding.primary.role]
        _require(
            descriptor
            == {
                "path": validated["bundle_path"],
                "sha256": validated["sha256"],
                "size_bytes": validated["size_bytes"],
                "media_type": validated["media_type"],
                "candidate_evidence": context["candidate_evidence"],
            },
            "publication target descriptor differs from evaluator-consumed B1",
        )
        source_raw.append(raw_by_path[descriptor["path"]])
    _require_evaluator_bindings_consumed(
        evaluator_bindings,
        catalog,
        descriptor_access_log,
    )
    recomputed_root = tagged_canonical_sha256(
        TECHNICAL_INPUT_INVENTORY_DOMAIN, technical
    )
    _require(
        hmac.compare_digest(recomputed_root, context["technical_input_root"]),
        "final B1 technical input root differs",
    )
    private_bundle = _materialize_records(
        policy=policy,
        candidate=context["candidate_evidence"],
        source_records=source_records,
        raw_by_record=source_raw,
        evaluator_bindings=evaluator_bindings,
    )
    projection = project_publication(private_bundle, policy, private_openings=openings)
    _require(
        hmac.compare_digest(
            tagged_canonical_sha256(PUBLIC_PROJECTION_DOMAIN, projection),
            context["public_projection_sha256"],
        ),
        "recomputed public projection differs from protected binding",
    )
    rebuilt = publication.build_publication_assets(
        projection,
        expected_policy=policy,
        expected_bindings=policy["bindings"],
    )
    _require(
        set(public_assets) == {role for role, _name in publication.ASSET_ORDER},
        "published asset role closure differs",
    )
    supplied: dict[str, bytes] = {}
    for role, name in publication.ASSET_ORDER:
        value = public_assets.get(role)
        _require(value is not None, "published asset role closure differs")
        if type(value) is bytes:
            supplied[name] = value
        else:
            _path, supplied[name] = _private_file_bytes(
                value, f"published asset {role}", maximum=MAX_PUBLIC_ARCHIVE_BYTES
            )
    _require(
        all(hmac.compare_digest(rebuilt[name], supplied[name]) for name in rebuilt),
        "published public bytes differ from final B1 projection",
    )
    ledger = [
        {
            "role": role,
            "name": name,
            "size_bytes": len(supplied[name]),
            "sha256": hashlib.sha256(supplied[name]).hexdigest(),
        }
        for role, name in publication.ASSET_ORDER
    ]
    _require(
        ledger == context["public_asset_ledger"]
        and hmac.compare_digest(
            tagged_canonical_sha256(PUBLIC_ASSET_LEDGER_DOMAIN, ledger),
            context["public_asset_ledger_sha256"],
        ),
        "published asset ledger differs from protected binding",
    )
    return {
        "technical_input_root": context["technical_input_root"],
        "public_projection_sha256": context["public_projection_sha256"],
        "public_asset_ledger_sha256": context["public_asset_ledger_sha256"],
        "source_count": len(source_records),
        "runtime_receipt_count": len(RUNTIME_RECEIPT_ORDER),
        "first_five_scopes_validated": True,
    }


__all__ = [
    "COMMITMENT_ALGORITHM",
    "CORRECTNESS_KIND",
    "CORRECTNESS_PATH",
    "EXECUTION_CLAIM_ORDER",
    "EXECUTION_WITNESS_KIND",
    "EXECUTION_WITNESS_PATH",
    "EVENT_PROFILER_CONTRACT",
    "EVENT_PROFILER_KIND",
    "LOCAL_CLOCK",
    "MaterializedPublicationInputs",
    "POLICY_KIND",
    "PRIVATE_INPUT_KIND",
    "PROJECTION_KIND",
    "PROTECTED_OPENINGS_KIND",
    "PROTECTED_OPENINGS_VERSION",
    "PUBLICATION_SOURCE_SPECIFICATION_KIND",
    "PUBLICATION_SOURCE_SPECIFICATION_VERSION",
    "PublicationSourceSpecification",
    "PrivateOpenings",
    "PrivacyError",
    "RAW_TIMING_CONTRACT",
    "RAW_TIMING_KIND",
    "RUNTIME_RECEIPT_ORDER",
    "SCAN_CONTRACT",
    "SCHEMA_VERSION",
    "SCOPE_KIND",
    "TECHNICAL_SCOPE_ORDER",
    "TECHNICAL_SCOPE_PATHS",
    "binding_canonical_json_bytes",
    "canonical_json_bytes",
    "completion_source_catalog",
    "load_private_openings",
    "load_publication_source_spec",
    "materialize_publication_inputs",
    "normalize_trace",
    "publication_binding_context",
    "project_publication",
    "scan_payload",
    "scan_public_bytes",
    "serialize_private_openings",
    "tagged_canonical_sha256",
    "verify_private_openings",
    "verify_publication_bundle_binding",
]
