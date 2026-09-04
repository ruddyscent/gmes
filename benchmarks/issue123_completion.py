#!/usr/bin/env python3
"""Fail-closed final evidence aggregator for GitHub issue #123."""

from __future__ import annotations

import argparse
import ast
import copy
import datetime as dt
import hashlib
import hmac
import io
import json
import math
import os
import platform
import re
import shutil
import stat
import sys
import tempfile
import zipfile
import zlib
from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from statistics import median
from typing import Any, Callable
from urllib.parse import urlsplit

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "benchmarks" / "native_oracle_workloads.json"
TRUSTED_MANIFEST_SHA256 = (
    "0766dbf932882dfec7a40abfbcd78eb67978ed8cd65e38625193a16502cc29a9"
)

INDEX_KIND = "issue-123-completion-evidence-index"
BUNDLE_SPEC_KIND = "issue-123-completion-bundle-specification"
OUTPUT_KIND = "issue-123-completion-evaluation"
OFFLINE_EVALUATION_MODE = "offline-structural"
LIVE_EVALUATION_MODE = "production-same-process-live"
OFFLINE_AUTHORITY = "same-process-live-verification-required"
LIVE_AUTHORITY = "same-process-authenticated-gh-live-verification"
LIVE_RECEIPT_NAME = "operations-live-receipt.json"
LIVE_RESULT_NAME = "completion-live-result.json"
LIVE_OPERATIONS_DIRECTORY = "operations-input"
LIVE_OPERATIONS_RESPONSE_COUNT = 22
BUNDLE_REOPEN_RECEIPT_KIND = "issue-123-completion-bundle-reopen-receipt"
BUNDLE_REOPEN_RECEIPT_VERSION = 1
PRIVATE_BUNDLE_BINDING_DOMAIN = "gmes.issue123.private-bundle-binding.v1"
COMPLETION_BUNDLE_INVENTORY_DOMAIN = "gmes.issue123.completion-bundle-inventory.v1"
DIFFERENTIAL_KIND = "issue-123-differential-evidence"
MACOS_INDEX_KIND = "issue-123-macos-evidence-index"
FAILURE_RUN_KIND = "two-gpu-failure-run"
TORCH_SOLVER_ABI = "torch-fdtd-regions-v15"
LOCAL_COMPILED_REGION_TOPOLOGY = (
    "local-two-static-half-step-regions+external-cached-two-stage-foreach-"
    "boundary-sync-v2"
)
BOUNDARY_SYNC_REPRESENTATION = "cached-two-stage-foreach-v1"
CUDA_GRAPH_EXECUTION_REPRESENTATION = (
    "external-no-inner-cudagraph-regions+dm2-raw-fixed-masked-v1"
)

INDEX_SCHEMA_VERSION = 2
BUNDLE_SPEC_SCHEMA_VERSION = 1
OUTPUT_SCHEMA_VERSION = 3
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
MAX_RETAINED_BUNDLE_ENTRIES = 100_000
MAX_ZIP_MEMBERS = 4096
MAX_ZIP_MEMBER_BYTES = 512 * 1024 * 1024
MAX_ZIP_TOTAL_BYTES = 1024 * 1024 * 1024
MAX_ZIP_COMPRESSION_RATIO = 10_000.0
MAX_NPZ_MEMBERS = 512
MAX_NPZ_ARRAY_BYTES = 256 * 1024 * 1024
MAX_NPZ_TOTAL_BYTES = 512 * 1024 * 1024
MAX_CORRECTNESS_NPZ_MEMBERS = 8192
MAX_CORRECTNESS_NPZ_ARRAY_BYTES = 256 * 1024 * 1024
MAX_CORRECTNESS_NPZ_TOTAL_BYTES = 1024 * 1024 * 1024
MAX_CORRECTNESS_NPZ_METADATA_BYTES = 16 * 1024 * 1024
MAX_RUNTIME_RECEIPT_BYTES = 16 * 1024 * 1024
MAX_NPZ_DIMENSIONS = 8
MAX_NPY_HEADER_BYTES = 64 * 1024
MAX_NPZ_ARCHIVE_BYTES = (
    MAX_NPZ_TOTAL_BYTES + MAX_NPZ_MEMBERS * (MAX_NPY_HEADER_BYTES + 1024) + 65535
)
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
CUDA_SUITE_CONTRACT_ID = "single-gpu-cuda-closure-v2"
CUDA_PERFORMANCE_PRECISION_BY_CASE = {
    "cpu-crossover-2d": "float32",
    "cpu-crossover-3d": "float32",
    "cpu-large-2d": "float32",
    "cpu-large-3d": "float32",
    "single-gpu-2d": "float32",
    "single-gpu-3d": "float64",
}
CUDA_PRECISION_LIMITATION_REVIEW = {
    "contract_id": "single-gpu-3d-float32-dynamic-range-review-v1",
    "case": "single-gpu-3d",
    "rejected_precision": "float32",
    "accepted_precision": "float64",
    "reason": "native-step-100-magnitude-exceeds-float32-range",
}
CUDA_CORRECTNESS_RUNTIME_MODES = (
    {
        "device": "cuda:0",
        "precision": "float32",
        "graph_mode": "eager",
        "compile_policy": "eager",
        "compile_mode": "default",
    },
    {
        "device": "cuda:0",
        "precision": "float32",
        "graph_mode": "graph",
        "compile_policy": "compile",
        "compile_mode": "reduce-overhead",
    },
)
CPU_CORRECTNESS_RUNTIME_MODE = {
    "device": "cpu",
    "precision": "float64",
    "graph_mode": "eager",
    "compile_policy": "eager",
    "compile_mode": "default",
}
STATE_FINITENESS_CONTRACT_ID = "dynamic-checkpoint-finite-v1"
CORRECTNESS_EVIDENCE_CONTRACT_ID = "torch-cpu-acceptance-v8"
CORRECTNESS_INDEX_SCHEMA_VERSION = 2
CORRECTNESS_INDEX_KIND = "torch-correctness-evidence-index"
CORRECTNESS_INDEX_CONTRACT_ID = "complete-field-state-and-runtime-receipt-v2"
CORRECTNESS_UNIQUE_ARCHIVE_COUNT = 136
CORRECTNESS_INDEX_KEYS = {
    "schema_version",
    "kind",
    "contract_id",
    "manifest_contract_sha256",
    "candidate_evidence",
    "runtime_mode",
    "runtime_receipt",
    "required_cases",
    "artifacts",
    "suite_acceptance",
}
RUNTIME_RECEIPT_KIND = "issue123-runtime-publication-receipt"
RUNTIME_RECEIPT_ROLES = (
    "cpu",
    "cuda-eager",
    "cuda-graph",
    "single-gpu-2d",
    "single-gpu-3d",
)
SINGLE_GPU_DIFFERENTIAL_RUNTIME_MODES = (
    {
        "device": "cuda:0",
        "precision": "float32",
        "graph_mode": "eager",
        "compile_policy": "eager",
        "compile_mode": "default",
    },
    {
        "device": "cuda:0",
        "precision": "float64",
        "graph_mode": "eager",
        "compile_policy": "eager",
        "compile_mode": "default",
    },
)
CORRECTNESS_RUNTIME_MODE_KEYS = {
    "device",
    "precision",
    "graph_mode",
    "compile_policy",
    "compile_mode",
}
CORRECTNESS_EVIDENCE_KEYS = {
    "evidence_contract_id",
    "cpu_contract_id",
    "manifest_sha256",
    "runner_sha256",
    "solver_sha256",
    "solver_abi",
    "candidate_git_commit",
    "candidate_git_status",
}
CORRECTNESS_RUNNER_INPUTS = (
    "benchmarks/native_oracle.py",
    "benchmarks/torch_cpu_baseline.py",
    "benchmarks/torch_dm2.py",
    "benchmarks/torch_tuning.py",
)
CORRECTNESS_SOLVER_INPUTS = (
    "gmes/torch_dm2.py",
    "gmes/torch_fdtd.py",
    "gmes/torch_dispersive.py",
    "gmes/torch_distributed.py",
    "gmes/torch_plan.py",
    "gmes/torch_source.py",
)
CORRECTNESS_ARTIFACT_KEYS = {
    "case",
    "group",
    "reference",
    "reference_observer_commit",
    "candidate",
    "candidate_provenance",
    "comparison",
    "tolerance_results",
}
DIFFERENTIAL_ELEMENTWISE_MODE = "elementwise-allclose-v1"
DIFFERENTIAL_NORMALIZED_MODE = "normalized-linf-l2-v1"
DIFFERENTIAL_NORMALIZED_LIMIT = 1e-6
DIFFERENTIAL_NORMALIZED_ABSOLUTE_SCALE_FLOOR = 2e-12
DIFFERENTIAL_SCHEMA_VERSION = 5
DIFFERENTIAL_NORMALIZED_CASE = (
    "single-gpu-cuda",
    "single-gpu-3d",
    "cuda:0",
)
DIFFERENTIAL_NORMALIZED_STEPS = (0, 1, 2, 5, 20, 100)
DIFFERENTIAL_NORMALIZED_GROUPS = ((0, 1), (2, 5), (20, 100))
DIFFERENTIAL_NORMALIZED_RESIDUAL_STEPS = frozenset({20, 100})
FROZEN_DIFFERENTIAL_CAPTURE_STEPS = (1, 2, 5, 20, 100)
FROZEN_DIFFERENTIAL_RECORDS_BY_SCOPE = {
    "paired-real": tuple(
        (case, device, "float64" if device == "cpu" else "float32")
        for case in (
            "bloch-2d",
            "bloch-3d",
            "upml-bloch",
            "cpml-bloch",
            "lorentz-bloch",
            "dcp-ade-bloch",
            "dcp-plrc-bloch",
            "dcp-rc-bloch",
        )
        for device in ("cpu", "cuda:0")
    ),
    "single-gpu-cuda": (
        ("single-gpu-2d", "cuda:0", "float32"),
        ("single-gpu-3d", "cuda:0", "float64"),
    ),
}
# These are duplicated deliberately: the differential acceptance contract must
# not be redefined by a manifest selected by an evidence bundle.
FROZEN_DIFFERENTIAL_TOLERANCES = {
    "dielectric": {
        "float32": {"rtol": 3e-5, "atol": 3e-6},
        "float64": {"rtol": 1e-13, "atol": 1e-14},
        "complex64": {"rtol": 3e-5, "atol": 3e-6},
        "complex128": {"rtol": 2e-13, "atol": 2e-14},
    },
    "pml": {
        "float32": {"rtol": 5e-5, "atol": 5e-6},
        "float64": {"rtol": 2e-12, "atol": 2e-13},
        "complex64": {"rtol": 8e-5, "atol": 8e-6},
        "complex128": {"rtol": 4e-12, "atol": 4e-13},
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
    "dm2": {
        "float32": {"rtol": 6e-4, "atol": 3e-6},
        "float64": {"rtol": 2e-10, "atol": 2e-12},
    },
}
DIFFERENTIAL_SOURCE_ARRAY = "persistent/source/semantic-contract.json"
DIFFERENTIAL_SOURCE_SCHEMA = "point-source-semantic-v1"
DIFFERENTIAL_SOURCE_PROOF_ARRAY = "persistent/source/raw-proof.json"
DIFFERENTIAL_SOURCE_PROOF_SCHEMA = "point-source-raw-proof-v2"
DIFFERENTIAL_POINT_SOURCE_LIVE_ARRAYS = (
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
DIFFERENTIAL_CASE_UPDATER_LABELS = {
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
FROZEN_DIFFERENTIAL_PERSISTENT_GEOMETRY_SHA256_BY_CASE = {
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
DIFFERENTIAL_STRATEGY_TOLERANCE_MODELS = {
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
DIFFERENTIAL_POINT_SOURCE_FREQUENCY = 0.35
DIFFERENTIAL_POINT_SOURCE_PHASE = 0.0
DIFFERENTIAL_POINT_SOURCE_START = 0.0
DIFFERENTIAL_POINT_SOURCE_END = math.inf
DIFFERENTIAL_POINT_SOURCE_WIDTH = 5.0 / DIFFERENTIAL_POINT_SOURCE_FREQUENCY
DIFFERENTIAL_POINT_SOURCE_VALUE_ULP_FACTOR = 256
DIFFERENTIAL_POINT_SOURCE_COURANT_RATIO = 0.99
DIFFERENTIAL_SOURCE_PREIMAGE_SCHEMA = "point-source-role-preimage-v1"
DIFFERENTIAL_GROUP_NPZ_SCHEMA = "issue-123-differential-group-npz-v1"
MAX_DIFFERENTIAL_SOURCE_PROOF_BYTES = 1024 * 1024
MAX_DIFFERENTIAL_SOURCE_PROOF_ARRAY_BYTES = 64 * 1024
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
DIFFERENTIAL_PHYSICAL_SUFFIXES = (
    *(f"physical/spectrum/{name}" for name in FIELD_ARRAYS),
    "physical/summary",
)
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


class CommittedAuthorityError(EvidenceError):
    """The no-replace authority link exists but final custody cleanup failed."""

    committed = True


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


def _trusted_manifest_bytes() -> tuple[Path, bytes, dict[str, Any]]:
    path, raw = _bounded_regular_file_bytes(
        DEFAULT_MANIFEST,
        "trusted repository manifest",
        max_bytes=MAX_MANIFEST_BYTES,
    )
    _require(
        _sha256(raw) == TRUSTED_MANIFEST_SHA256,
        "trusted repository manifest digest differs from the frozen contract",
    )
    document = _strict_json_bytes(
        raw,
        "trusted repository manifest",
        max_bytes=MAX_MANIFEST_BYTES,
    )
    _require(isinstance(document, dict), "trusted repository manifest is not an object")
    return path, raw, document


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


def _zip_local_sizes_match(
    raw: bytes,
    offset: int,
    name_size: int,
    extra_size: int,
    info: zipfile.ZipInfo,
    label: str,
) -> bool:
    compressed = int.from_bytes(raw[offset + 18 : offset + 22], "little")
    uncompressed = int.from_bytes(raw[offset + 22 : offset + 26], "little")
    if compressed != 0xFFFFFFFF and uncompressed != 0xFFFFFFFF:
        return compressed == info.compress_size and uncompressed == info.file_size
    extra = raw[offset + 30 + name_size : offset + 30 + name_size + extra_size]
    cursor = 0
    zip64 = None
    while cursor < len(extra):
        _require(cursor + 4 <= len(extra), f"{label} local extra field is truncated")
        field_id = int.from_bytes(extra[cursor : cursor + 2], "little")
        field_size = int.from_bytes(extra[cursor + 2 : cursor + 4], "little")
        cursor += 4
        _require(
            cursor + field_size <= len(extra), f"{label} local extra field is truncated"
        )
        if field_id == 0x0001:
            _require(zip64 is None, f"{label} repeats the ZIP64 local extra field")
            zip64 = extra[cursor : cursor + field_size]
        cursor += field_size
    _require(zip64 is not None, f"{label} omits the ZIP64 local size field")
    cursor = 0
    expected = []
    if uncompressed == 0xFFFFFFFF:
        expected.append(info.file_size)
    if compressed == 0xFFFFFFFF:
        expected.append(info.compress_size)
    _require(len(zip64) == 8 * len(expected), f"{label} ZIP64 local sizes differ")
    return all(
        int.from_bytes(zip64[index * 8 : (index + 1) * 8], "little") == value
        for index, value in enumerate(expected)
    )


def _validate_zip_complete_coverage(
    raw: bytes, infos: list[zipfile.ZipInfo], label: str
) -> None:
    """Require every byte to belong to an indexed local or central record."""

    minimum_eocd = max(0, len(raw) - (22 + 65535))
    eocd_offset = -1
    for candidate in range(len(raw) - 22, minimum_eocd - 1, -1):
        if raw[candidate : candidate + 4] != b"PK\x05\x06":
            continue
        comment_size = int.from_bytes(raw[candidate + 20 : candidate + 22], "little")
        if candidate + 22 + comment_size == len(raw):
            eocd_offset = candidate
            break
    _require(eocd_offset >= 0, f"{label} has no terminal ZIP directory record")
    disk = int.from_bytes(raw[eocd_offset + 4 : eocd_offset + 6], "little")
    central_disk = int.from_bytes(raw[eocd_offset + 6 : eocd_offset + 8], "little")
    disk_entries = int.from_bytes(raw[eocd_offset + 8 : eocd_offset + 10], "little")
    total_entries = int.from_bytes(raw[eocd_offset + 10 : eocd_offset + 12], "little")
    central_size = int.from_bytes(raw[eocd_offset + 12 : eocd_offset + 16], "little")
    central_offset = int.from_bytes(raw[eocd_offset + 16 : eocd_offset + 20], "little")
    _require(
        disk == central_disk == 0
        and disk_entries == total_entries == len(infos)
        and total_entries != 0xFFFF
        and central_size != 0xFFFFFFFF
        and central_offset != 0xFFFFFFFF
        and central_offset + central_size == eocd_offset,
        f"{label} central-directory coverage differs",
    )

    cursor = central_offset
    for index, info in enumerate(infos):
        member_label = f"{label} ZIP central member {index}"
        _require(
            cursor + 46 <= eocd_offset and raw[cursor : cursor + 4] == b"PK\x01\x02",
            f"{member_label} is not contiguous",
        )
        name_size = int.from_bytes(raw[cursor + 28 : cursor + 30], "little")
        extra_size = int.from_bytes(raw[cursor + 30 : cursor + 32], "little")
        comment_size = int.from_bytes(raw[cursor + 32 : cursor + 34], "little")
        record_end = cursor + 46 + name_size + extra_size + comment_size
        encoding = "utf-8" if info.flag_bits & 0x800 else "cp437"
        encoded_name = info.filename.encode(encoding)
        _require(
            record_end <= eocd_offset
            and raw[cursor + 46 : cursor + 46 + name_size] == encoded_name
            and int.from_bytes(raw[cursor + 8 : cursor + 10], "little")
            == info.flag_bits
            and int.from_bytes(raw[cursor + 10 : cursor + 12], "little")
            == info.compress_type
            and int.from_bytes(raw[cursor + 16 : cursor + 20], "little") == info.CRC
            and int.from_bytes(raw[cursor + 20 : cursor + 24], "little")
            == info.compress_size
            and int.from_bytes(raw[cursor + 24 : cursor + 28], "little")
            == info.file_size
            and int.from_bytes(raw[cursor + 34 : cursor + 36], "little") == 0
            and int.from_bytes(raw[cursor + 42 : cursor + 46], "little")
            == info.header_offset,
            f"{member_label} header differs",
        )
        cursor = record_end
    _require(cursor == eocd_offset, f"{label} central directory contains gap bytes")

    ordered = sorted(infos, key=lambda info: info.header_offset)
    expected_offset = 0
    for index, info in enumerate(ordered):
        member_label = f"{label} ZIP local member {index}"
        offset = info.header_offset
        _require(offset == expected_offset, f"{member_label} is not contiguous")
        name_size = int.from_bytes(raw[offset + 26 : offset + 28], "little")
        extra_size = int.from_bytes(raw[offset + 28 : offset + 30], "little")
        data_end = offset + 30 + name_size + extra_size + info.compress_size
        next_offset = (
            ordered[index + 1].header_offset
            if index + 1 < len(ordered)
            else central_offset
        )
        _require(data_end <= next_offset, f"{member_label} overlaps another record")
        if info.flag_bits & 0x08:
            descriptor = raw[data_end:next_offset]
            if len(descriptor) == 16:
                _require(
                    descriptor[:4] == b"PK\x07\x08",
                    f"{member_label} data descriptor signature differs",
                )
                descriptor = descriptor[4:]
            _require(
                len(descriptor) == 12
                and int.from_bytes(descriptor[0:4], "little") == info.CRC
                and int.from_bytes(descriptor[4:8], "little") == info.compress_size
                and int.from_bytes(descriptor[8:12], "little") == info.file_size,
                f"{member_label} data descriptor differs",
            )
        else:
            _require(
                data_end == next_offset
                and int.from_bytes(raw[offset + 14 : offset + 18], "little") == info.CRC
                and _zip_local_sizes_match(
                    raw, offset, name_size, extra_size, info, member_label
                ),
                f"{member_label} byte coverage differs",
            )
        expected_offset = next_offset
    _require(
        expected_offset == central_offset, f"{label} local records leave gap bytes"
    )


def _preflight_zip(
    raw: bytes,
    label: str,
    *,
    max_members: int = MAX_ZIP_MEMBERS,
    max_member_bytes: int = MAX_ZIP_MEMBER_BYTES,
    max_total_bytes: int = MAX_ZIP_TOTAL_BYTES,
    expected_files: set[str] | None = None,
    expected_comment: bytes | None = None,
) -> list[zipfile.ZipInfo]:
    """Validate ZIP structure and CRCs before any consumer extracts a member."""

    _require(type(raw) is bytes, f"{label} ZIP input must be exact bytes")
    _require(raw.startswith(b"PK"), f"{label} is not ZIP bytes")
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            if expected_comment is not None:
                _require(
                    type(expected_comment) is bytes
                    and archive.comment == expected_comment,
                    f"{label} ZIP comment binding differs",
                )
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
                offset = info.header_offset
                _require(
                    type(offset) is int
                    and 0 <= offset <= len(raw) - 30
                    and raw[offset : offset + 4] == b"PK\x03\x04",
                    f"{member_label} has no bounded local header",
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
                    encoded_name = info.filename.encode(encoding)
                except UnicodeEncodeError as error:
                    raise EvidenceError(
                        f"{member_label} local name is not canonical"
                    ) from error
                _require(
                    local_header_end <= len(raw)
                    and raw[offset + 30 : offset + 30 + local_name_size] == encoded_name
                    and local_flags == info.flag_bits
                    and not (local_flags & (0x1 | 0x40))
                    and local_compression == info.compress_type,
                    f"{member_label} local header contract differs",
                )
                mode = info.external_attr >> 16
                _require(
                    not stat.S_ISLNK(mode),
                    f"{member_label} is a symbolic link",
                )
                _require(
                    not info.flag_bits & (0x1 | 0x40),
                    f"{member_label} is encrypted",
                )
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
            _validate_zip_complete_coverage(raw, infos, label)
            if expected_files is not None:
                _require(files == expected_files, f"{label} ZIP file closure differs")
            return infos
    except EvidenceError:
        raise
    except (
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
        raise EvidenceError(f"{label} is not a bounded valid ZIP archive") from error


def _validate_media_payload(raw: bytes, media_type: str, label: str) -> None:
    if media_type == MEDIA_TYPE_JSON:
        _strict_json_bytes(raw, label)
    elif media_type == MEDIA_TYPE_NPZ:
        infos = _preflight_zip(
            raw,
            label,
            max_members=MAX_CORRECTNESS_NPZ_MEMBERS,
            max_member_bytes=MAX_CORRECTNESS_NPZ_ARRAY_BYTES,
            max_total_bytes=MAX_CORRECTNESS_NPZ_TOTAL_BYTES,
        )
        _require(
            bool(infos)
            and all(
                not info.is_dir()
                and info.filename.endswith(".npy")
                and info.compress_type in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                for info in infos
            ),
            f"{label} NPZ member closure differs",
        )
        names = [info.filename.removesuffix(".npy") for info in infos]
        _require(
            all(bool(name) for name in names),
            f"{label} NPZ contains an empty array name",
        )
        _preflight_npz_array_headers(
            raw,
            infos,
            names,
            label,
            exact_differential_group=False,
            max_array_bytes=MAX_CORRECTNESS_NPZ_ARRAY_BYTES,
            max_total_bytes=MAX_CORRECTNESS_NPZ_TOTAL_BYTES,
            allow_metadata_json=True,
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


@dataclass(frozen=True, slots=True, repr=False)
class _ArtifactDescriptorIdentity:
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True, slots=True, repr=False)
class _RetainedArtifactView:
    fd: int
    identity: _ArtifactDescriptorIdentity
    raw: bytes
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class _ValidatedSdistInventory:
    archive_sha256: str
    archive_size: int
    member_count: int
    total_member_bytes: int
    member_names: tuple[str, ...]


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    size_bytes: int
    sha256: str
    identity: tuple[int, int, int, int]


@dataclass(frozen=True)
class StagedOperationsSnapshots:
    index: FileSnapshot
    responses: tuple[tuple[str, FileSnapshot], ...]


@dataclass(frozen=True)
class RetainedBundleFile:
    relative_path: PurePosixPath
    descriptor_bytes: bytes | None
    fd: int
    identity: tuple[int, int, int, int, int]
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class RetainedBundleDirectory:
    relative_path: PurePosixPath
    fd: int
    identity: tuple[int, int, int, int]


@dataclass(frozen=True)
class RetainedBundleTree:
    root: Path
    root_directory: RetainedBundleDirectory
    directories: tuple[RetainedBundleDirectory, ...]
    index: RetainedBundleFile
    payloads: tuple[RetainedBundleFile, ...]
    expected_entry_types: tuple[tuple[str, str], ...]
    descriptor_ledger: tuple[bytes, ...]
    inventory_bytes: bytes
    inventory_root: str
    index_semantics: bytes


@dataclass(frozen=True)
class LiveAuthoritySnapshots:
    source_bundle: RetainedBundleTree
    reopened_bundle: RetainedBundleTree
    manifest: FileSnapshot
    runtime_receipts: tuple[FileSnapshot, ...]
    protected_openings: FileSnapshot
    pre_acknowledgment_receipt: FileSnapshot
    final_reopen_receipt: FileSnapshot

    @property
    def source_index(self) -> FileSnapshot:
        return _retained_file_snapshot(self.source_bundle, self.source_bundle.index)

    @property
    def reopened_index(self) -> FileSnapshot:
        return _retained_file_snapshot(self.reopened_bundle, self.reopened_bundle.index)


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _artifact_descriptor_identity(
    metadata: os.stat_result,
) -> _ArtifactDescriptorIdentity:
    return _ArtifactDescriptorIdentity(
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


class _RetainedArtifactLease:
    """Own one no-follow descriptor while an sdist is validated."""

    __slots__ = (
        "_expected_digest",
        "_expected_identity",
        "_expected_size",
        "_fd",
        "_path",
        "_view",
    )

    def __init__(
        self,
        path: Path,
        identity: _ArtifactDescriptorIdentity,
        expected_size: int,
        expected_digest: str,
    ) -> None:
        self._path = path
        self._expected_identity = identity
        self._expected_size = expected_size
        self._expected_digest = expected_digest
        self._fd = -1
        self._view: _RetainedArtifactView | None = None

    def __repr__(self) -> str:
        return "<_RetainedArtifactLease>"

    def _close_after_entry_failure(self) -> None:
        if self._fd < 0:
            return
        try:
            os.close(self._fd)
        except OSError:
            pass
        self._fd = -1

    def __enter__(self) -> _RetainedArtifactView:
        failure: str | None = None
        raw: bytes | None = None
        identity: _ArtifactDescriptorIdentity | None = None
        if self._view is not None or self._fd >= 0:
            failure = "macOS sdist source is invalid"
        if failure is None:
            flags = (
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                self._fd = os.open(self._path, flags)
            except OSError:
                failure = "macOS sdist source is invalid"
        if failure is None:
            try:
                metadata = os.fstat(self._fd)
                identity = _artifact_descriptor_identity(metadata)
            except OSError:
                failure = "macOS sdist source is invalid"
            else:
                if not stat.S_ISREG(metadata.st_mode):
                    failure = "macOS sdist source is invalid"
                elif identity != self._expected_identity:
                    failure = "macOS sdist source changed during validation"
                elif (
                    metadata.st_size != self._expected_size
                    or metadata.st_size < 0
                    or metadata.st_size > MAX_ARTIFACT_BYTES
                ):
                    failure = "macOS sdist bytes differ from their descriptor"
        if failure is None:
            chunks: list[bytes] = []
            offset = 0
            try:
                while offset < self._expected_size:
                    chunk = os.pread(
                        self._fd,
                        min(1024 * 1024, self._expected_size - offset),
                        offset,
                    )
                    if not chunk:
                        failure = "macOS sdist source changed during validation"
                        break
                    chunks.append(chunk)
                    offset += len(chunk)
                if failure is None and os.pread(self._fd, 1, self._expected_size):
                    failure = "macOS sdist source changed during validation"
                raw = b"".join(chunks)
                after = _artifact_descriptor_identity(os.fstat(self._fd))
            except OSError:
                failure = "macOS sdist source is invalid"
            else:
                if failure is None and (
                    after != self._expected_identity or len(raw) != self._expected_size
                ):
                    failure = "macOS sdist source changed during validation"
        if failure is None:
            assert raw is not None
            digest = _sha256(raw)
            if digest != self._expected_digest:
                failure = "macOS sdist bytes differ from their descriptor"
        if failure is not None:
            self._close_after_entry_failure()
            raise EvidenceError(failure) from None
        assert raw is not None and identity is not None
        self._view = _RetainedArtifactView(
            self._fd,
            identity,
            raw,
            _sha256(raw),
            len(raw),
        )
        return self._view

    def __exit__(self, exc_type: Any, _exc: Any, _traceback: Any) -> bool:
        close_failed = False
        if self._fd >= 0:
            try:
                os.close(self._fd)
            except OSError:
                close_failed = True
            self._fd = -1
        if close_failed and exc_type is None:
            raise EvidenceError("macOS sdist descriptor could not be closed") from None
        return False


def _private_sdist_inventory(
    result: Any,
    view: _RetainedArtifactView,
    descriptor: dict[str, Any],
    limits: Any,
) -> tuple[str, _ValidatedSdistInventory | None]:
    """Map the privacy-only ledger to the small completion inventory."""

    try:
        archive_size = result.archive_size
        archive_sha256 = result.archive_sha256
    except Exception:
        return "structural", None
    if (
        type(archive_size) is not int
        or archive_size != view.size_bytes
        or archive_size != descriptor["size_bytes"]
        or archive_sha256 != view.sha256
        or archive_sha256 != descriptor["sha256"]
    ):
        return "descriptor", None
    try:
        members = result.members
        total_member_bytes = result.total_member_bytes
        physical_ordinary_count = result.physical_ordinary_count
        logical_member_count = result.logical_member_count
        if (
            type(members) is not tuple
            or type(total_member_bytes) is not int
            or type(physical_ordinary_count) is not int
            or type(logical_member_count) is not int
            or archive_size > limits.archive_bytes
            or len(members) > limits.members
            or total_member_bytes < 0
            or total_member_bytes > limits.total_member_bytes
            or physical_ordinary_count != len(members)
            or logical_member_count != len(members)
        ):
            return "structural", None
        names: list[str] = []
        total = 0
        files = 0
        for member in members:
            name = member.name
            type_code = member.type_code
            size = member.size
            body_offset = member.body_offset
            digest = member.sha256
            if (
                not isinstance(name, str)
                or _canonical_bundle_path(name, "macOS sdist member name").as_posix()
                != name
                or type_code not in {"file", "directory"}
                or type(size) is not int
                or size < 0
                or size > limits.member_bytes
                or type(body_offset) is not int
                or body_offset < 0
                or not isinstance(digest, str)
                or SHA256_RE.fullmatch(digest) is None
            ):
                return "structural", None
            total += size
            if total > limits.total_member_bytes:
                return "structural", None
            names.append(name)
            files += type_code == "file"
        if len(set(names)) != len(names) or files == 0 or total != total_member_bytes:
            return "structural", None
    except Exception:
        return "structural", None
    return (
        "ok",
        _ValidatedSdistInventory(
            archive_sha256,
            archive_size,
            len(members),
            total_member_bytes,
            tuple(names),
        ),
    )


def _snapshot_regular_file(
    path_value: Path | str,
    label: str,
    *,
    max_bytes: int,
) -> tuple[FileSnapshot, bytes]:
    try:
        supplied = Path(path_value)
    except TypeError as error:
        raise EvidenceError(f"{label} path is invalid") from error
    path, metadata = _path_without_symlinks(supplied, label)
    _require(stat.S_ISREG(metadata.st_mode), f"{label} is not a regular file")
    _require(
        0 < metadata.st_size <= max_bytes,
        f"{label} exceeds the byte bound or is empty",
    )
    identity = _file_identity(metadata)
    raw = _read_opened_regular_file(
        path,
        label,
        max_bytes=max_bytes,
        expected_size=metadata.st_size,
    )
    try:
        after = path.stat()
    except OSError as error:
        raise EvidenceError(f"{label} changed after being read") from error
    _require(
        _file_identity(after) == identity,
        f"{label} changed after being read",
    )
    return FileSnapshot(path, len(raw), _sha256(raw), identity), raw


def _require_snapshot_unchanged(
    snapshot: FileSnapshot,
    label: str,
    *,
    max_bytes: int,
) -> None:
    current, _raw = _snapshot_regular_file(
        snapshot.path,
        label,
        max_bytes=max_bytes,
    )
    _require(current == snapshot, f"{label} was substituted during verification")


class ArtifactReader:
    """Read exact-byte artifacts from one relocatable, symlink-free bundle."""

    def __init__(
        self,
        base: Path,
        candidate: dict[str, str],
        registry: list[dict[str, Any]] | None = None,
        descriptor_access_log: list[dict[str, Any]] | None = None,
    ):
        self.base, metadata = _path_without_symlinks(base, "evidence bundle root")
        _require(
            stat.S_ISDIR(metadata.st_mode),
            "evidence bundle root is not a directory",
        )
        self.candidate = candidate
        self._seen: dict[Path, tuple[int, str]] = {}
        self._registry: dict[str, dict[str, Any]] | None = None
        self._descriptor_access_log = descriptor_access_log
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
        if self._descriptor_access_log is not None:
            self._descriptor_access_log.append(
                {"label": label, "descriptor": dict(descriptor)}
            )
        return LoadedArtifact(dict(descriptor), path, raw, document)

    def load_private_sdist(
        self,
        descriptor: Any,
        label: str,
    ) -> tuple[LoadedArtifact, _ValidatedSdistInventory]:
        """Load one gzip sdist from its retained descriptor, never a reopened path."""

        descriptor = _validate_descriptor(descriptor, self.candidate, label)
        if self._registry is not None:
            _require(
                self._registry.get(descriptor["path"]) == descriptor,
                f"{label} descriptor is absent from the payload registry",
            )
        _require(
            descriptor["media_type"] == MEDIA_TYPE_GZIP,
            f"{label} media type differs from its role",
        )
        path: Path | None = None
        metadata: os.stat_result | None = None
        path_failure = False
        try:
            portable = _canonical_bundle_path(descriptor["path"], f"{label} path")
            path = self.base
            for part in portable.parts:
                path = path / part
                metadata = path.lstat()
                if stat.S_ISLNK(metadata.st_mode):
                    path_failure = True
                    break
            if not path_failure:
                path, metadata = _path_without_symlinks(path, label)
                metadata = path.lstat()
        except (EvidenceError, OSError):
            path_failure = True
        if path_failure or path is None or metadata is None:
            raise EvidenceError("macOS sdist source is invalid") from None
        if not stat.S_ISREG(metadata.st_mode):
            raise EvidenceError("macOS sdist source is invalid") from None
        if metadata.st_size != descriptor["size_bytes"]:
            raise EvidenceError(
                "macOS sdist bytes differ from their descriptor"
            ) from None
        expected_identity = _artifact_descriptor_identity(metadata)
        try:
            from benchmarks import issue123_privacy as privacy
        except (ImportError, OSError):
            raise EvidenceError("macOS sdist privacy validation failed") from None
        privacy_failure = False
        result: Any = None
        limits: Any = None
        inventory: _ValidatedSdistInventory | None = None
        with _RetainedArtifactLease(
            path,
            expected_identity,
            descriptor["size_bytes"],
            descriptor["sha256"],
        ) as view:
            if not view.raw.startswith(b"\x1f\x8b"):
                raise EvidenceError("macOS sdist source is invalid") from None
            try:
                limits = privacy._default_private_sdist_validation_limits()
                with privacy._retain_private_sdist_fd(view.fd) as source:
                    result = privacy._validate_private_sdist_raw_first(
                        source,
                        (),
                        limits=limits,
                    )
            except Exception:
                privacy_failure = True
            if privacy_failure:
                raise EvidenceError("macOS sdist privacy validation failed") from None
            status, inventory = _private_sdist_inventory(
                result,
                view,
                descriptor,
                limits,
            )
            if status == "descriptor":
                raise EvidenceError(
                    "macOS sdist bytes differ from their descriptor"
                ) from None
            if status != "ok" or inventory is None:
                raise EvidenceError(
                    "macOS sdist structural inventory differs"
                ) from None
            identity_failure = False
            try:
                current_identity = _artifact_descriptor_identity(os.fstat(view.fd))
            except OSError:
                identity_failure = True
            if identity_failure or current_identity != expected_identity:
                raise EvidenceError(
                    "macOS sdist source changed during validation"
                ) from None
            artifact = LoadedArtifact(dict(descriptor), path, view.raw)
        identity = (len(artifact.raw), artifact.descriptor["sha256"])
        previous = self._seen.setdefault(path, identity)
        _require(previous == identity, f"{label} path has conflicting descriptors")
        if self._descriptor_access_log is not None:
            self._descriptor_access_log.append(
                {"label": label, "descriptor": dict(descriptor)}
            )
        return artifact, inventory

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


def _repository_inputs_sha256(paths: tuple[str, ...], label: str) -> str:
    digest = hashlib.sha256()
    for relative in sorted(paths):
        _canonical_bundle_path(relative, f"{label} input")
        _path, raw = _bounded_regular_file_bytes(
            ROOT / relative,
            f"{label} input {relative}",
            max_bytes=MAX_ARTIFACT_BYTES,
        )
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(raw)
        digest.update(b"\0")
    return digest.hexdigest()


def _expected_correctness_candidate_evidence(
    manifest: dict[str, Any], candidate: dict[str, str]
) -> dict[str, str]:
    try:
        cpu_contract_id = manifest["performance_gates"]["cpu_acceptance"]["contract_id"]
    except (KeyError, TypeError) as error:
        raise EvidenceError("manifest CPU correctness contract is absent") from error
    _require(
        isinstance(cpu_contract_id, str) and bool(cpu_contract_id),
        "manifest CPU correctness contract is invalid",
    )
    return {
        "evidence_contract_id": CORRECTNESS_EVIDENCE_CONTRACT_ID,
        "cpu_contract_id": cpu_contract_id,
        "manifest_sha256": candidate["manifest_sha256"],
        "runner_sha256": _repository_inputs_sha256(
            CORRECTNESS_RUNNER_INPUTS, "correctness runner"
        ),
        "solver_sha256": _repository_inputs_sha256(
            CORRECTNESS_SOLVER_INPUTS, "correctness solver"
        ),
        "solver_abi": TORCH_SOLVER_ABI,
        "candidate_git_commit": candidate["candidate_git_commit"],
        "candidate_git_status": candidate["candidate_git_status"],
    }


def _validate_correctness_runtime_mode(value: Any, label: str) -> None:
    _exact_keys(value, CORRECTNESS_RUNTIME_MODE_KEYS, label)
    device = value["device"]
    precision = value["precision"]
    graph_mode = value["graph_mode"]
    compile_mode = value["compile_mode"]
    canonical_cuda = (
        isinstance(device, str)
        and device.startswith("cuda:")
        and device[5:].isdigit()
        and str(int(device[5:])) == device[5:]
    )
    _require(
        (device == "cpu" or canonical_cuda)
        and precision in {"float32", "float64"}
        and graph_mode in {"eager", "graph"}
        and compile_mode in {"default", "reduce-overhead", "max-autotune"}
        and (graph_mode != "eager" or compile_mode == "default")
        and (device != "cpu" or compile_mode == "default")
        and value["compile_policy"]
        == ("compile" if graph_mode == "graph" else "eager"),
        f"{label} differs from the frozen execution contract",
    )


def _validate_runtime_receipt_document(
    value: Any,
    manifest: dict[str, Any],
    candidate: dict[str, str],
    expected_runtime_mode: dict[str, str],
    label: str,
    *,
    expected_candidate_cases: tuple[str, ...] | None = None,
) -> None:
    _exact_keys(
        value,
        {
            "schema_version",
            "kind",
            "final_sha",
            "manifest_sha256",
            "workflow",
            "profiler_witness",
            "runtime_mode",
            "candidate_archives",
        },
        label,
    )
    workflow = value["workflow"]
    witness = value["profiler_witness"]
    _exact_keys(
        workflow,
        {"repository", "run_id", "run_attempt", "job_id", "job_name"},
        f"{label} workflow",
    )
    _exact_keys(
        witness,
        {"name", "sha256", "size_bytes", "media_type"},
        f"{label} profiler witness",
    )
    _validate_correctness_runtime_mode(value["runtime_mode"], f"{label} runtime mode")
    _require(
        _is_exact_int(value["schema_version"], 1)
        and value["kind"] == RUNTIME_RECEIPT_KIND
        and value["final_sha"] == candidate["candidate_git_commit"]
        and value["manifest_sha256"]
        == candidate["manifest_sha256"]
        == TRUSTED_MANIFEST_SHA256
        and _type_exact_equal(value["runtime_mode"], expected_runtime_mode),
        f"{label} identity or runtime binding differs",
    )
    _require(
        workflow["repository"] == "ruddyscent/gmes"
        and all(
            type(workflow[name]) is int and workflow[name] > 0
            for name in ("run_id", "run_attempt", "job_id")
        )
        and isinstance(workflow["job_name"], str)
        and 0 < len(workflow["job_name"]) <= 256
        and workflow["job_name"] == workflow["job_name"].strip()
        and not any(ord(character) < 32 for character in workflow["job_name"]),
        f"{label} workflow identity differs",
    )
    _require(
        isinstance(witness["name"], str)
        and 0 < len(witness["name"]) <= 256
        and PurePosixPath(witness["name"]).name == witness["name"]
        and witness["name"] not in {".", ".."}
        and "\\" not in witness["name"]
        and "\x00" not in witness["name"]
        and isinstance(witness["sha256"], str)
        and SHA256_RE.fullmatch(witness["sha256"]) is not None
        and type(witness["size_bytes"]) is int
        and 0 < witness["size_bytes"] <= MAX_ARTIFACT_BYTES
        and isinstance(witness["media_type"], str)
        and witness["media_type"] == witness["media_type"].lower()
        and witness["media_type"].count("/") == 1
        and not any(character.isspace() for character in witness["media_type"]),
        f"{label} profiler witness differs",
    )
    archives = value["candidate_archives"]
    required_cases = (
        list(expected_candidate_cases)
        if expected_candidate_cases is not None
        else [
            case["name"]
            for group in ("correctness", "physical_checks")
            for case in manifest[group]
        ]
    )
    _require(
        isinstance(archives, list) and len(archives) == len(required_cases),
        f"{label} candidate archive closure differs",
    )
    for index, (archive, case) in enumerate(zip(archives, required_cases, strict=True)):
        _exact_keys(
            archive,
            {"case", "sha256", "size_bytes"},
            f"{label} candidate archive {index}",
        )
        _require(
            archive["case"] == case
            and isinstance(archive["sha256"], str)
            and SHA256_RE.fullmatch(archive["sha256"]) is not None
            and type(archive["size_bytes"]) is int
            and 0 < archive["size_bytes"] <= MAX_ARTIFACT_BYTES,
            f"{label} candidate archive {index} differs",
        )


def _load_trusted_runtime_receipts(
    paths: Any,
    manifest: dict[str, Any],
    candidate: dict[str, str],
    bundle_root: Path,
) -> tuple[LoadedArtifact, ...]:
    _require(
        isinstance(paths, (list, tuple)) and len(paths) == len(RUNTIME_RECEIPT_ROLES),
        "exactly five trusted runtime receipt paths are required",
    )
    modes = (
        CPU_CORRECTNESS_RUNTIME_MODE,
        *CUDA_CORRECTNESS_RUNTIME_MODES,
        *SINGLE_GPU_DIFFERENTIAL_RUNTIME_MODES,
    )
    candidate_case_closures = (
        None,
        None,
        None,
        ("single-gpu-2d",),
        ("single-gpu-3d",),
    )
    loaded = []
    for role, path_value, mode, case_closure in zip(
        RUNTIME_RECEIPT_ROLES,
        paths,
        modes,
        candidate_case_closures,
        strict=True,
    ):
        path, raw = _bounded_regular_file_bytes(
            Path(path_value),
            f"trusted {role} runtime receipt",
            max_bytes=MAX_RUNTIME_RECEIPT_BYTES,
        )
        _require(
            not path.is_relative_to(bundle_root),
            f"trusted {role} runtime receipt must be outside the evidence bundle",
        )
        document = _strict_json_bytes(
            raw,
            f"trusted {role} runtime receipt",
            max_bytes=MAX_RUNTIME_RECEIPT_BYTES,
        )
        _require(
            raw == _canonical_json_bytes(document),
            f"trusted {role} runtime receipt is not canonical JSON",
        )
        _validate_runtime_receipt_document(
            document,
            manifest,
            candidate,
            mode,
            f"trusted {role} runtime receipt",
            expected_candidate_cases=case_closure,
        )
        loaded.append(
            LoadedArtifact(
                {
                    "path": path.name,
                    "sha256": _sha256(raw),
                    "size_bytes": len(raw),
                    "media_type": MEDIA_TYPE_JSON,
                },
                path,
                raw,
                document,
            )
        )
    _require(
        len({item.path for item in loaded}) == len(loaded)
        and len({item.descriptor["sha256"] for item in loaded}) == len(loaded)
        and len({(item.path.stat().st_dev, item.path.stat().st_ino) for item in loaded})
        == len(loaded),
        "trusted runtime receipts are not five distinct files and byte sets",
    )
    return tuple(loaded)


def _load_bundled_runtime_receipt(
    reader: ArtifactReader, descriptor: Any, label: str
) -> LoadedArtifact:
    _exact_keys(
        descriptor,
        {"path", "sha256", "size_bytes", "media_type"},
        f"{label} descriptor",
    )
    _canonical_bundle_path(descriptor["path"], f"{label} path")
    _require(
        isinstance(descriptor["sha256"], str)
        and SHA256_RE.fullmatch(descriptor["sha256"]) is not None
        and type(descriptor["size_bytes"]) is int
        and 0 < descriptor["size_bytes"] <= MAX_RUNTIME_RECEIPT_BYTES
        and descriptor["media_type"] == MEDIA_TYPE_JSON,
        f"{label} descriptor differs",
    )
    registered = (
        reader._registry.get(descriptor["path"])
        if reader._registry is not None
        else {
            **descriptor,
            "candidate_evidence": reader.candidate,
        }
    )
    _require(isinstance(registered, dict), f"{label} descriptor is not registered")
    _require(
        all(registered.get(name) == descriptor[name] for name in descriptor),
        f"{label} descriptor differs from its registered artifact",
    )
    return reader.load(
        registered,
        label,
        expected_media_types={MEDIA_TYPE_JSON},
        require_embedded_candidate=False,
    )


def _correctness_archive_binding(
    artifact: LoadedArtifact, case: str, label: str
) -> dict[str, Any]:
    try:
        metadata = artifact.path.stat()
    except OSError as error:
        raise EvidenceError(f"{label} identity is unreadable") from error
    _require(
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_size == artifact.descriptor["size_bytes"]
        and metadata.st_size == len(artifact.raw),
        f"{label} payload identity differs",
    )
    return {
        "case": case,
        "path": artifact.descriptor["path"],
        "sha256": artifact.descriptor["sha256"],
        "size_bytes": artifact.descriptor["size_bytes"],
        "media_type": artifact.descriptor["media_type"],
        "payload_identity": _file_identity(metadata),
    }


def _validate_correctness_index(
    artifact: LoadedArtifact,
    manifest: dict[str, Any],
    candidate: dict[str, str],
    reader: ArtifactReader,
    *,
    include_archive_bindings: bool = False,
    expected_runtime_mode: dict[str, str] | None = None,
    trusted_runtime_receipt: LoadedArtifact | None = None,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    document = artifact.document
    _require(isinstance(document, dict), "correctness index root must be an object")
    _document_candidate_matches(document, candidate, required=True)
    evidence = document.get("candidate_evidence")
    expected_evidence = _expected_correctness_candidate_evidence(manifest, candidate)
    _exact_keys(
        evidence,
        CORRECTNESS_EVIDENCE_KEYS,
        "correctness index candidate evidence",
    )
    _require(
        _type_exact_equal(evidence, expected_evidence),
        "correctness index candidate evidence differs from the current candidate",
    )
    _exact_keys(document, CORRECTNESS_INDEX_KEYS, "correctness index")
    _require(
        _is_exact_int(document["schema_version"], CORRECTNESS_INDEX_SCHEMA_VERSION)
        and document["kind"] == CORRECTNESS_INDEX_KIND
        and document["contract_id"] == CORRECTNESS_INDEX_CONTRACT_ID,
        "correctness index identity differs",
    )
    _validate_correctness_runtime_mode(
        document["runtime_mode"], "correctness index runtime mode"
    )
    if expected_runtime_mode is not None:
        _require(
            _type_exact_equal(document["runtime_mode"], expected_runtime_mode),
            "correctness index runtime mode differs from the required scope",
        )
    _require(
        isinstance(trusted_runtime_receipt, LoadedArtifact),
        "an external trusted runtime publication receipt is required",
    )
    bundled_runtime_receipt = _load_bundled_runtime_receipt(
        reader,
        document["runtime_receipt"],
        "correctness runtime publication receipt",
    )
    _require(
        bundled_runtime_receipt.raw == trusted_runtime_receipt.raw
        and _type_exact_equal(
            bundled_runtime_receipt.document, trusted_runtime_receipt.document
        ),
        "bundled runtime publication receipt bytes differ from the external receipt",
    )
    bundled_metadata = bundled_runtime_receipt.path.stat()
    trusted_metadata = trusted_runtime_receipt.path.stat()
    _require(
        (bundled_metadata.st_dev, bundled_metadata.st_ino)
        != (trusted_metadata.st_dev, trusted_metadata.st_ino),
        "bundled runtime publication receipt is not independent from the external "
        "receipt",
    )
    _validate_runtime_receipt_document(
        trusted_runtime_receipt.document,
        manifest,
        candidate,
        document["runtime_mode"],
        "correctness external runtime publication receipt",
    )
    _require(
        document.get("manifest_contract_sha256") == _canonical_sha256(manifest),
        "correctness index canonical manifest digest differs",
    )
    required_pairs = [
        (group, case["name"])
        for group in ("correctness", "physical_checks")
        for case in manifest.get(group, ())
    ]
    required = [name for _group, name in required_pairs]
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
    archive_bindings: dict[str, list[dict[str, Any]]] = {
        "reference": [],
        "candidate": [],
    }
    used_archive_paths = set()
    used_archive_digests = set()
    for index, (record, (expected_group, expected_case)) in enumerate(
        zip(artifacts, required_pairs, strict=True)
    ):
        record_label = f"correctness artifact {index}"
        _exact_keys(record, CORRECTNESS_ARTIFACT_KEYS, record_label)
        _require(
            record["case"] == expected_case and record["group"] == expected_group,
            f"{record_label} identity differs from the manifest",
        )
        for role in ("reference", "candidate"):
            loaded = reader.load(
                record[role],
                f"correctness {index} {role} archive",
                json_document=False,
                expected_media_types={MEDIA_TYPE_NPZ},
            )
            descriptor = loaded.descriptor
            _require(
                descriptor["path"] not in used_archive_paths
                and descriptor["sha256"] not in used_archive_digests,
                "correctness archive descriptors reuse a path or digest",
            )
            used_archive_paths.add(descriptor["path"])
            used_archive_digests.add(descriptor["sha256"])
            archive_bindings[role].append(
                _correctness_archive_binding(
                    loaded,
                    expected_case,
                    f"correctness {index} {role} archive",
                )
            )
            nested.append(loaded)
    expected_receipt_archives = [
        {
            "case": case,
            "sha256": descriptor["sha256"],
            "size_bytes": descriptor["size_bytes"],
        }
        for case, descriptor in zip(
            required, archive_bindings["candidate"], strict=True
        )
    ]
    _require(
        _type_exact_equal(
            trusted_runtime_receipt.document["candidate_archives"],
            expected_receipt_archives,
        ),
        "runtime publication receipt candidate archive binding differs",
    )
    try:
        from benchmarks.torch_correctness import load_correctness_evidence_index

        rebuilt = load_correctness_evidence_index(
            artifact.path,
            manifest,
            expected_evidence,
            descriptor_root=reader.base,
            runtime_receipt=trusted_runtime_receipt.path,
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
    if include_archive_bindings:
        return rebuilt, archive_bindings
    return rebuilt


def _validate_cuda_correctness_indexes(
    records: Any,
    reader: ArtifactReader,
    manifest: dict[str, Any],
    candidate: dict[str, str],
    label: str,
    trusted_runtime_receipts: tuple[LoadedArtifact, ...] | None = None,
) -> tuple[
    list[dict[str, Any]],
    tuple[dict[str, list[dict[str, Any]]], ...],
]:
    _require(
        trusted_runtime_receipts is None
        or (
            isinstance(trusted_runtime_receipts, tuple)
            and len(trusted_runtime_receipts) == len(CUDA_CORRECTNESS_RUNTIME_MODES)
            and all(
                isinstance(receipt, LoadedArtifact)
                for receipt in trusted_runtime_receipts
            )
        ),
        f"{label} requires exactly two external correctness runtime receipts",
    )
    _require(
        isinstance(records, list)
        and len(records) == len(CUDA_CORRECTNESS_RUNTIME_MODES),
        f"{label} correctness index closure differs",
    )
    rebuilt_indexes = []
    archive_bindings_by_mode = []
    used_paths = set()
    used_digests = set()
    for index, (record, expected_mode) in enumerate(
        zip(records, CUDA_CORRECTNESS_RUNTIME_MODES, strict=True)
    ):
        record_label = f"{label} correctness index {index}"
        _exact_keys(record, {"runtime_mode", "source_artifact"}, record_label)
        _require(
            _type_exact_equal(record["runtime_mode"], expected_mode),
            f"{record_label} runtime mode differs",
        )
        artifact = reader.load(
            record["source_artifact"],
            record_label,
            expected_media_types={MEDIA_TYPE_JSON},
        )
        descriptor = artifact.descriptor
        _require(
            descriptor["path"] not in used_paths
            and descriptor["sha256"] not in used_digests,
            f"{label} correctness indexes reuse an artifact path or digest",
        )
        used_paths.add(descriptor["path"])
        used_digests.add(descriptor["sha256"])
        _require(
            isinstance(artifact.document, dict)
            and _type_exact_equal(artifact.document.get("runtime_mode"), expected_mode),
            f"{record_label} loaded runtime mode differs",
        )
        receipt_arguments = (
            {"trusted_runtime_receipt": trusted_runtime_receipts[index]}
            if trusted_runtime_receipts is not None
            else {}
        )
        rebuilt, archive_bindings = _validate_correctness_index(
            artifact,
            manifest,
            candidate,
            reader,
            include_archive_bindings=True,
            **receipt_arguments,
        )
        _require(
            _type_exact_equal(rebuilt.get("runtime_mode"), expected_mode),
            f"{record_label} recomputed runtime mode differs",
        )
        recomputed_source = rebuilt.get("source_artifact")
        _exact_keys(
            recomputed_source,
            {"path", "sha256", "size_bytes", "media_type", "candidate_evidence"},
            f"{record_label} recomputed source artifact",
        )
        _require(
            _type_exact_equal(recomputed_source, descriptor)
            and recomputed_source["sha256"] == descriptor["sha256"],
            f"{record_label} descriptor differs from the recomputed source artifact",
        )
        rebuilt_indexes.append(rebuilt)
        archive_bindings_by_mode.append(archive_bindings)
    eager_bindings, graph_bindings = archive_bindings_by_mode
    _require(
        _type_exact_equal(eager_bindings["reference"], graph_bindings["reference"]),
        f"{label} correctness native reference archives differ by runtime mode",
    )
    eager_candidate_paths = {
        descriptor["path"] for descriptor in eager_bindings["candidate"]
    }
    graph_candidate_paths = {
        descriptor["path"] for descriptor in graph_bindings["candidate"]
    }
    eager_candidate_digests = {
        descriptor["sha256"] for descriptor in eager_bindings["candidate"]
    }
    graph_candidate_digests = {
        descriptor["sha256"] for descriptor in graph_bindings["candidate"]
    }
    _require(
        eager_candidate_paths.isdisjoint(graph_candidate_paths)
        and eager_candidate_digests.isdisjoint(graph_candidate_digests),
        f"{label} correctness candidate archives overlap by runtime mode",
    )
    return rebuilt_indexes, tuple(archive_bindings_by_mode)


def _candidate_archive_bindings_by_case(
    manifest: dict[str, Any],
    archive_bindings: dict[str, list[dict[str, Any]]],
    label: str,
) -> list[dict[str, Any]]:
    required_cases = [
        case["name"]
        for group in ("correctness", "physical_checks")
        for case in manifest[group]
    ]
    candidates = archive_bindings.get("candidate")
    _require(
        isinstance(candidates, list) and len(candidates) == len(required_cases),
        f"{label} candidate archive closure differs",
    )
    return [
        {
            "case": case,
            "sha256": descriptor["sha256"],
            "size_bytes": descriptor["size_bytes"],
        }
        for case, descriptor in zip(required_cases, candidates, strict=True)
    ]


def _ordered_correctness_archive_bindings(
    value: Any, manifest: dict[str, Any], label: str
) -> dict[str, list[dict[str, Any]]]:
    _exact_keys(value, {"reference", "candidate"}, label)
    required_cases = [
        case["name"]
        for group in ("correctness", "physical_checks")
        for case in manifest[group]
    ]
    checked: dict[str, list[dict[str, Any]]] = {}
    for role in ("reference", "candidate"):
        records = value[role]
        _require(
            isinstance(records, list) and len(records) == len(required_cases),
            f"{label} {role} archive closure differs",
        )
        validated = []
        for index, (record, case) in enumerate(
            zip(records, required_cases, strict=True)
        ):
            record_label = f"{label} {role} archive {index}"
            _exact_keys(
                record,
                {
                    "case",
                    "path",
                    "sha256",
                    "size_bytes",
                    "media_type",
                    "payload_identity",
                },
                record_label,
            )
            identity = record["payload_identity"]
            _canonical_bundle_path(record["path"], f"{record_label} path")
            _require(
                record["case"] == case
                and isinstance(record["sha256"], str)
                and SHA256_RE.fullmatch(record["sha256"]) is not None
                and type(record["size_bytes"]) is int
                and 0 < record["size_bytes"] <= MAX_ARTIFACT_BYTES
                and record["media_type"] == MEDIA_TYPE_NPZ
                and isinstance(identity, tuple)
                and len(identity) == 4
                and all(type(item) is int and item >= 0 for item in identity),
                f"{record_label} differs",
            )
            validated.append(copy.deepcopy(record))
        checked[role] = validated
    return checked


def _validate_global_correctness_archive_topology(
    cpu: dict[str, Any], single: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, int]:
    cpu_bindings = _ordered_correctness_archive_bindings(
        cpu.get("_correctness_archive_bindings"),
        manifest,
        "CPU correctness",
    )
    cuda_by_mode = single.get("_correctness_archive_bindings_by_mode")
    _exact_keys(
        cuda_by_mode,
        {"eager", "graph"},
        "CUDA correctness archive runtime modes",
    )
    eager_bindings = _ordered_correctness_archive_bindings(
        cuda_by_mode["eager"], manifest, "CUDA eager correctness"
    )
    graph_bindings = _ordered_correctness_archive_bindings(
        cuda_by_mode["graph"], manifest, "CUDA graph correctness"
    )
    _require(
        _type_exact_equal(cpu_bindings["reference"], eager_bindings["reference"])
        and _type_exact_equal(cpu_bindings["reference"], graph_bindings["reference"]),
        "CPU and CUDA correctness native reference archives differ",
    )
    groups = (
        cpu_bindings["reference"],
        cpu_bindings["candidate"],
        eager_bindings["candidate"],
        graph_bindings["candidate"],
    )
    required_cases = len(cpu_bindings["reference"])
    _require(
        required_cases * len(groups) == CORRECTNESS_UNIQUE_ARCHIVE_COUNT,
        "correctness manifest does not define the frozen 136-archive topology",
    )
    identities = {
        "path": lambda record: record["path"],
        "digest": lambda record: record["sha256"],
        "payload identity": lambda record: record["payload_identity"],
    }
    for identity_label, identity in identities.items():
        values = [identity(record) for group in groups for record in group]
        _require(
            len(values) == CORRECTNESS_UNIQUE_ARCHIVE_COUNT
            and len(set(values)) == CORRECTNESS_UNIQUE_ARCHIVE_COUNT,
            "correctness native reference and CPU/CUDA candidate archives are "
            f"not exactly 136 globally unique {identity_label}s",
        )
    return {
        "case_count": required_cases,
        "shared_reference_archive_count": required_cases,
        "candidate_archive_count": required_cases * 3,
        "unique_archive_count": CORRECTNESS_UNIQUE_ARCHIVE_COUNT,
    }


def _validate_tuning_acceptance(
    result: Any,
    label: str,
    *,
    allow_allocation_override: bool = False,
) -> None:
    _require(isinstance(result, dict), f"{label} must be an object")
    diagnostics = result.get("diagnostics")
    boundaries = (
        diagnostics.get("boundaries") if isinstance(diagnostics, dict) else None
    )
    _exact_keys(
        boundaries,
        {"scheduling", "execution_representation", "paired_real_scratch_bytes"},
        f"{label} boundary execution diagnostics",
    )
    _require(
        boundaries["scheduling"] == "external"
        and boundaries.get("execution_representation") == BOUNDARY_SYNC_REPRESENTATION
        and type(boundaries.get("paired_real_scratch_bytes")) is int
        and boundaries["paired_real_scratch_bytes"] >= 0,
        f"{label} boundary execution diagnostics differ from the solver ABI",
    )
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
        from benchmarks.torch_cpu_baseline import (
            privacy_preserving_host_identity,
            timing_runtime_identity_matches,
        )

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
        and timing_runtime_identity_matches(baseline_environment, environment)
        and _type_exact_equal(
            environment.get("thread_environment"),
            baseline_thread_environment,
        ),
        f"{label} differs from the exact frozen Torch baseline host, timing runtime, "
        "or thread environment",
    )


def _public_torch_versions_match(first: Any, second: Any) -> bool:
    try:
        from benchmarks.torch_cpu_baseline import public_torch_version

        return public_torch_version(first) == public_torch_version(second)
    except (ImportError, TypeError, ValueError):
        return False


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
    trusted_runtime_receipt: LoadedArtifact,
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
    correctness, correctness_archive_bindings = _validate_correctness_index(
        correctness_artifact,
        manifest,
        candidate,
        reader,
        include_archive_bindings=True,
        expected_runtime_mode=CPU_CORRECTNESS_RUNTIME_MODE,
        trusted_runtime_receipt=trusted_runtime_receipt,
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
        and _public_torch_versions_match(
            native_torch.get("version"), cpu_runtime["torch"]
        ),
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
        "_correctness_archive_bindings": copy.deepcopy(correctness_archive_bindings),
        "correctness_candidate_archives": _candidate_archive_bindings_by_case(
            manifest,
            correctness_archive_bindings,
            "CPU correctness",
        ),
    }


def _preflight_npz_array_headers(
    raw: bytes,
    infos: list[zipfile.ZipInfo],
    names: list[str],
    label: str,
    *,
    exact_differential_group: bool,
    max_array_bytes: int = MAX_NPZ_ARRAY_BYTES,
    max_total_bytes: int = MAX_NPZ_TOTAL_BYTES,
    allow_metadata_json: bool = False,
) -> dict[str, tuple[tuple[int, ...], np.dtype[Any], int]]:
    expected_members = [f"{name}.npy" for name in names]
    _require(
        [info.filename for info in infos] == expected_members,
        f"{label} NPY member order differs",
    )
    total_payload_bytes = 0
    metadata: dict[str, tuple[tuple[int, ...], np.dtype[Any], int]] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            for info, name in zip(infos, names, strict=True):
                array_label = f"{label} array {name!r}"
                mode = info.external_attr >> 16
                _require(
                    not info.is_dir() and stat.S_IFMT(mode) in {0, stat.S_IFREG},
                    f"{array_label} is not a regular file",
                )
                if exact_differential_group:
                    _require(
                        info.compress_type == zipfile.ZIP_DEFLATED,
                        f"{array_label} is not DEFLATE-compressed",
                    )
                with archive.open(info, "r") as stream:
                    prefix = stream.read(8)
                    _require(
                        len(prefix) == 8 and prefix[:6] == b"\x93NUMPY",
                        f"{array_label} has no NPY header",
                    )
                    version = tuple(prefix[6:])
                    _require(
                        (
                            version == (1, 0)
                            if exact_differential_group
                            else version in {(1, 0), (2, 0), (3, 0)}
                        ),
                        f"{array_label} uses an unsupported NPY version",
                    )
                    length_bytes = 2 if version == (1, 0) else 4
                    encoded_length = stream.read(length_bytes)
                    _require(
                        len(encoded_length) == length_bytes,
                        f"{array_label} has a truncated NPY header length",
                    )
                    header_length = int.from_bytes(encoded_length, "little")
                    _require(
                        0 < header_length <= MAX_NPY_HEADER_BYTES,
                        f"{array_label} NPY header exceeds the bound",
                    )
                    encoded_header = stream.read(header_length)
                    _require(
                        len(encoded_header) == header_length,
                        f"{array_label} has a truncated NPY header",
                    )
                encoding = "utf-8" if version == (3, 0) else "latin1"
                header = ast.literal_eval(encoded_header.decode(encoding))
                _exact_keys(
                    header,
                    {"descr", "fortran_order", "shape"},
                    f"{array_label} NPY header",
                )
                dtype = np.dtype(header["descr"])
                shape = header["shape"]
                metadata_json = (
                    allow_metadata_json
                    and name == "metadata.json"
                    and dtype.kind == "U"
                    and isinstance(shape, tuple)
                    and shape == ()
                    and 0 < dtype.itemsize <= MAX_CORRECTNESS_NPZ_METADATA_BYTES
                )
                _require(
                    dtype.fields is None
                    and dtype.subdtype is None
                    and not dtype.hasobject
                    and (dtype.kind in {"b", "i", "u", "f", "c"} or metadata_json)
                    and dtype.itemsize > 0
                    and (not exact_differential_group or dtype.itemsize <= 16),
                    f"{array_label} NPY dtype is not a plain numeric type",
                )
                _require(
                    header["fortran_order"] is False
                    and isinstance(shape, tuple)
                    and len(shape) <= MAX_NPZ_DIMENSIONS
                    and all(
                        type(size) is int and 0 <= size <= 2**31 - 1 for size in shape
                    ),
                    f"{array_label} NPY shape or storage order exceeds the bound",
                )
                payload_bytes = math.prod(shape) * dtype.itemsize
                _require(
                    payload_bytes <= max_array_bytes,
                    f"{array_label} declared payload exceeds the bound",
                )
                header_bytes = 8 + length_bytes + header_length
                _require(
                    header_bytes + payload_bytes == info.file_size,
                    f"{array_label} NPY header and payload size differ",
                )
                total_payload_bytes += payload_bytes
                _require(
                    total_payload_bytes <= max_total_bytes,
                    f"{label} declared NPY payloads exceed the byte bound",
                )
                metadata[name] = (shape, dtype, payload_bytes)
    except EvidenceError:
        raise
    except (
        MemoryError,
        OSError,
        OverflowError,
        RecursionError,
        RuntimeError,
        SyntaxError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
        zipfile.BadZipFile,
        zlib.error,
    ) as error:
        raise EvidenceError(f"{label} has an invalid bounded NPY header") from error
    return metadata


def _npz_arrays(
    artifact: LoadedArtifact,
    names: list[str],
    label: str,
    *,
    expected_comment: bytes | None = None,
) -> dict[str, np.ndarray]:
    _require(
        type(artifact.raw) is bytes and 0 < len(artifact.raw) <= MAX_NPZ_ARCHIVE_BYTES,
        f"{label} NPZ archive exceeds the byte bound",
    )
    if expected_comment is not None:
        eocd_offset = len(artifact.raw) - 22 - len(expected_comment)
        _require(
            artifact.raw.startswith(b"PK\x03\x04")
            and eocd_offset >= 0
            and artifact.raw[eocd_offset : eocd_offset + 4] == b"PK\x05\x06"
            and int.from_bytes(
                artifact.raw[eocd_offset + 20 : eocd_offset + 22], "little"
            )
            == len(expected_comment),
            f"{label} differential NPZ has prepended or trailing bytes",
        )
    expected_files = {f"{name}.npy" for name in names}
    infos = _preflight_zip(
        artifact.raw,
        label,
        max_members=MAX_NPZ_MEMBERS,
        max_member_bytes=MAX_NPZ_ARRAY_BYTES,
        max_total_bytes=MAX_NPZ_TOTAL_BYTES,
        expected_files=expected_files,
        expected_comment=expected_comment,
    )
    header_metadata = _preflight_npz_array_headers(
        artifact.raw,
        infos,
        names,
        label,
        exact_differential_group=expected_comment is not None,
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
                expected_shape, expected_dtype, expected_nbytes = header_metadata[name]
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
                    array.dtype == expected_dtype
                    and array.shape == expected_shape
                    and array.flags.c_contiguous
                    and array.nbytes == expected_nbytes
                    and array.nbytes <= MAX_NPZ_ARRAY_BYTES,
                    f"{label} array {name!r} differs from its bounded NPY header",
                )
                total += array.nbytes
                _require(
                    total <= MAX_NPZ_TOTAL_BYTES,
                    f"{label} arrays exceed the byte bound",
                )
                result[name] = array
            return result
    except (
        EOFError,
        MemoryError,
        OSError,
        TypeError,
        ValueError,
        zlib.error,
    ) as error:
        if isinstance(error, EvidenceError):
            raise
        raise EvidenceError(f"{label} is not a readable safe NPZ archive") from error


def _differential_float_token(value: float) -> dict[str, Any]:
    value = float(value)
    if math.isinf(value):
        return {"kind": "infinity", "sign": 1 if value > 0 else -1}
    _require(math.isfinite(value), "differential source value is non-finite")
    return {"kind": "finite", "hex": value.hex()}


def _expected_differential_source_medium(
    workload: dict[str, Any],
) -> tuple[float, float]:
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
    raise EvidenceError(
        "differential source-cell medium is outside the frozen contract"
    )


def _expected_differential_source_contract(workload: dict[str, Any]) -> dict[str, Any]:
    _require(
        workload.get("source", "point") == "point"
        and workload.get("source_component", "Ex") == "Ex",
        "differential source contract is outside the frozen Ex PointSource scope",
    )
    resolution = float(workload["resolution"])
    whole_shape = [
        1 if float(length) == 0 else int(np.rint(float(length) * resolution))
        for length in workload["size"]
    ]
    target = [
        whole_shape[0] // 2,
        0 if whole_shape[1] == 1 else int(math.floor(whole_shape[1] / 2 + 0.5)),
        0 if whole_shape[2] == 1 else int(math.floor(whole_shape[2] / 2 + 0.5)),
    ]
    frequency = 0.35
    parameters = (frequency, 0.0, 0.0, math.inf, 5.0 / frequency, 0.0)
    return {
        "schema": DIFFERENTIAL_SOURCE_SCHEMA,
        "workload": workload["name"],
        "schedule": "yee-point-source-overwrite-v1",
        "sources": [
            {
                "component": "Ex",
                "native_type": "PointSourceEx",
                "target_index": target,
                "operation": "overwrite",
                "model": {"id": 0, "name": "Continuous"},
                "parameters": [
                    _differential_float_token(value) for value in parameters
                ],
                "amplitude": _differential_float_token(
                    workload.get("source_amp", 1e-3)
                ),
                "source_cell_medium": {
                    "eps_inf": _differential_float_token(
                        _expected_differential_source_medium(workload)[0]
                    ),
                    "mu_inf": _differential_float_token(
                        _expected_differential_source_medium(workload)[1]
                    ),
                },
            }
        ],
    }


def _differential_capture_steps(
    manifest: dict[str, Any], workload: dict[str, Any]
) -> list[int]:
    reference = manifest.get("reference")
    _require(isinstance(reference, dict), "manifest reference contract is absent")
    steps = workload.get("capture_steps", reference.get("capture_steps"))
    _require(
        isinstance(steps, list)
        and bool(steps)
        and all(type(step) is int and step > 0 for step in steps)
        and steps == sorted(set(steps)),
        "differential source proof capture steps are malformed",
    )
    return steps


def _differential_source_array_from_record(
    value: Any, label: str
) -> np.ndarray[Any, Any]:
    _exact_keys(value, {"dtype", "shape", "data_hex"}, label)
    dtype_name = value["dtype"]
    _require(
        isinstance(dtype_name, str) and 0 < len(dtype_name) <= 16,
        f"{label} dtype is invalid",
    )
    try:
        dtype = np.dtype(dtype_name)
    except (TypeError, ValueError) as error:
        raise EvidenceError(f"{label} dtype is invalid") from error
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
        and len(shape) <= MAX_NPZ_DIMENSIONS
        and all(type(size) is int and 0 <= size <= 2**31 - 1 for size in shape),
        f"{label} shape is invalid",
    )
    size_bytes = math.prod(shape) * dtype.itemsize
    data_hex = value["data_hex"]
    _require(
        size_bytes <= MAX_DIFFERENTIAL_SOURCE_PROOF_ARRAY_BYTES
        and isinstance(data_hex, str)
        and len(data_hex) == 2 * size_bytes
        and re.fullmatch(r"[0-9a-f]*", data_hex) is not None,
        f"{label} exact bytes are invalid",
    )
    try:
        raw = bytes.fromhex(data_hex)
        array = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    except (MemoryError, TypeError, ValueError) as error:
        raise EvidenceError(f"{label} exact bytes are invalid") from error
    _require(
        array.flags.c_contiguous and array.tobytes(order="C").hex() == data_hex,
        f"{label} exact bytes are not canonical",
    )
    return array


def _differential_continuous_point_source_value(
    time: float, *, paired_real: bool
) -> float | complex:
    since_start = time - DIFFERENTIAL_POINT_SOURCE_START
    until_end = DIFFERENTIAL_POINT_SOURCE_END - time
    if since_start < 0.0 or until_end < 0.0:
        return 0j if paired_real else 0.0
    rise = (
        math.sin(0.5 * math.pi * since_start / DIFFERENTIAL_POINT_SOURCE_WIDTH) ** 2
        if since_start < DIFFERENTIAL_POINT_SOURCE_WIDTH
        else 1.0
    )
    fall = (
        math.sin(0.5 * math.pi * until_end / DIFFERENTIAL_POINT_SOURCE_WIDTH) ** 2
        if until_end < DIFFERENTIAL_POINT_SOURCE_WIDTH
        else 1.0
    )
    angle = (
        2.0 * math.pi * DIFFERENTIAL_POINT_SOURCE_FREQUENCY * time
        + DIFFERENTIAL_POINT_SOURCE_PHASE
    )
    value = rise * fall * complex(math.cos(angle), math.sin(angle))
    return value if paired_real else value.real


def _differential_point_source_value_matches(
    actual: Any,
    expected: Any,
    dtype: np.dtype[Any],
    scale: float,
) -> bool:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    if not np.isfinite(actual_array).all() or not np.isfinite(expected_array).all():
        return False
    if np.all(expected_array == 0):
        return bool(np.array_equal(actual_array, expected_array))
    real_dtype = np.empty((), dtype=dtype).real.dtype
    tolerance = (
        DIFFERENTIAL_POINT_SOURCE_VALUE_ULP_FACTOR
        * np.finfo(real_dtype).eps
        * max(float(np.max(np.abs(expected_array))), abs(float(scale)))
    )
    return bool(np.all(np.abs(actual_array - expected_array) <= tolerance))


def _expected_differential_point_source_time_step(
    workload: dict[str, Any],
) -> float:
    resolution = float(workload["resolution"])
    eps_inf, mu_inf = _expected_differential_source_medium(workload)
    value = (
        DIFFERENTIAL_POINT_SOURCE_COURANT_RATIO
        * math.sqrt(eps_inf * mu_inf)
        / (resolution * math.sqrt(3.0))
    )
    _require(
        math.isfinite(value) and value > 0.0,
        "differential PointSource workload has no finite Courant time step",
    )
    return value


def _validate_differential_source_time(
    value: np.ndarray[Any, Any],
    manifest: dict[str, Any],
    workload: dict[str, Any],
    captured_step: int,
    label: str,
) -> None:
    expected_step = manifest["reference"]["precondition_steps"] + captured_step
    expected_time_step = _expected_differential_point_source_time_step(workload)
    _require(
        value.shape == (3,)
        and value.dtype == np.dtype("float64")
        and np.isfinite(value).all()
        and float(value[2]) == expected_time_step
        and float(value[0]) == expected_step
        and float(value[1]) == expected_step * float(value[2]),
        f"{label} differs from the relative capture clock",
    )


def _differential_source_role_preimage_sha256(
    workload: dict[str, Any], role: str, captures: list[dict[str, Any]]
) -> str:
    _require(
        role in {"reference", "candidate"},
        "differential PointSource proof role is invalid",
    )
    source_role = "native" if role == "reference" else "candidate"
    payload = {
        "schema": DIFFERENTIAL_SOURCE_PREIMAGE_SCHEMA,
        "workload": workload["name"],
        "role": role,
        "captures": [
            {"step": capture["step"], "arrays": capture[source_role]}
            for capture in captures
        ],
    }
    raw = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _differential_point_source_field_shape(
    workload: dict[str, Any],
) -> tuple[int, int, int]:
    resolution = float(workload["resolution"])
    whole_shape = [
        1 if float(length) == 0.0 else int(np.rint(float(length) * resolution))
        for length in workload["size"]
    ]
    return (whole_shape[0], whole_shape[1] + 1, whole_shape[2] + 1)


def _validate_differential_source_proof(
    reference: np.ndarray[Any, Any],
    candidate: np.ndarray[Any, Any],
    manifest: dict[str, Any],
    workload: dict[str, Any],
    precision: str,
    semantic_contract: dict[str, Any],
    label: str,
) -> None:
    _require(
        reference.dtype == candidate.dtype == np.dtype("uint8")
        and reference.ndim == candidate.ndim == 1
        and 0 < reference.size <= MAX_DIFFERENTIAL_SOURCE_PROOF_BYTES
        and np.array_equal(reference, candidate),
        f"{label} PointSource raw proof bytes differ",
    )
    raw = reference.tobytes()
    proof = _strict_json_bytes(
        raw,
        f"{label} PointSource raw proof",
        max_bytes=MAX_DIFFERENTIAL_SOURCE_PROOF_BYTES,
    )
    _exact_keys(
        proof,
        {
            "schema",
            "workload",
            "reference_preimage_sha256",
            "candidate_preimage_sha256",
            "captures",
        },
        f"{label} PointSource raw proof",
    )
    reference_digest = proof["reference_preimage_sha256"]
    candidate_digest = proof["candidate_preimage_sha256"]
    _require(
        proof["schema"] == DIFFERENTIAL_SOURCE_PROOF_SCHEMA
        and proof["workload"] == workload["name"]
        and isinstance(reference_digest, str)
        and SHA256_RE.fullmatch(reference_digest) is not None
        and isinstance(candidate_digest, str)
        and SHA256_RE.fullmatch(candidate_digest) is not None
        and reference_digest != candidate_digest
        and raw
        == json.dumps(
            proof,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode(),
        f"{label} PointSource raw proof identity differs",
    )
    capture_steps = _differential_capture_steps(manifest, workload)
    expected_steps = [0, *capture_steps]
    captures = proof["captures"]
    _require(
        isinstance(captures, list)
        and len(captures) == len(expected_steps)
        and all(isinstance(capture, dict) for capture in captures)
        and [capture.get("step") for capture in captures] == expected_steps,
        f"{label} PointSource raw proof capture closure differs",
    )
    for index, capture in enumerate(captures):
        _exact_keys(
            capture,
            {"step", "native", "candidate"},
            f"{label} PointSource raw proof capture {index}",
        )
    _require(
        reference_digest
        == _differential_source_role_preimage_sha256(workload, "reference", captures)
        and candidate_digest
        == _differential_source_role_preimage_sha256(workload, "candidate", captures),
        f"{label} PointSource raw preimage digest differs from canonical bytes",
    )
    source = semantic_contract["sources"][0]
    amplitude = float(workload.get("source_amp", 1e-3))
    medium = _expected_differential_source_medium(workload)
    paired_real = bool(workload.get("complex"))
    try:
        candidate_dtype = np.dtype(precision)
    except (TypeError, ValueError) as error:
        raise EvidenceError(f"{label} source precision is invalid") from error
    _require(
        candidate_dtype in {np.dtype("float32"), np.dtype("float64")},
        f"{label} source precision is outside the frozen contract",
    )
    channels = 2 if paired_real else 1
    expected_parameters = np.asarray(
        [
            DIFFERENTIAL_POINT_SOURCE_FREQUENCY,
            DIFFERENTIAL_POINT_SOURCE_PHASE,
            DIFFERENTIAL_POINT_SOURCE_START,
            DIFFERENTIAL_POINT_SOURCE_END,
            DIFFERENTIAL_POINT_SOURCE_WIDTH,
            0.0,
        ],
        dtype=candidate_dtype,
    )
    expected_amplitude = np.asarray(amplitude, dtype=candidate_dtype)
    field_shape = _differential_point_source_field_shape(workload)
    target = source["target_index"]
    flat_target = int(np.ravel_multi_index(tuple(target), field_shape))
    for capture, captured_step in zip(captures, expected_steps, strict=True):
        capture_label = f"{label} PointSource raw proof step {captured_step}"
        _exact_keys(capture, {"step", "native", "candidate"}, capture_label)
        _require(
            type(capture["step"]) is int and capture["step"] == captured_step,
            f"{capture_label} identity differs",
        )
        native = capture["native"]
        torch_source = capture["candidate"]
        _exact_keys(native, {"time", "indices", "values"}, f"{capture_label} native")
        _exact_keys(
            torch_source,
            {"time", "packed_indices", "packed_values", "live"},
            f"{capture_label} candidate",
        )
        live_records = torch_source["live"]
        _exact_keys(
            live_records,
            set(DIFFERENTIAL_POINT_SOURCE_LIVE_ARRAYS),
            f"{capture_label} candidate live",
        )
        native_time = _differential_source_array_from_record(
            native["time"], f"{capture_label} native time"
        )
        candidate_time = _differential_source_array_from_record(
            torch_source["time"], f"{capture_label} candidate time"
        )
        _validate_differential_source_time(
            native_time,
            manifest,
            workload,
            captured_step,
            f"{capture_label} native time",
        )
        _validate_differential_source_time(
            candidate_time,
            manifest,
            workload,
            captured_step,
            f"{capture_label} candidate time",
        )
        _require(
            np.array_equal(native_time, candidate_time),
            f"{capture_label} native and candidate clocks differ",
        )
        native_indices = _differential_source_array_from_record(
            native["indices"], f"{capture_label} native indices"
        )
        native_values = _differential_source_array_from_record(
            native["values"], f"{capture_label} native values"
        )
        oscillator = _differential_continuous_point_source_value(
            float(native_time[1]), paired_real=paired_real
        )
        _require(
            native_indices.shape == (1, 3)
            and native_indices.dtype == np.dtype(np.intc)
            and native_indices.astype(np.int64).tolist() == [target]
            and native_values.shape == (4,)
            and native_values.dtype == np.dtype("complex128")
            and _differential_point_source_value_matches(
                native_values[0], complex(amplitude), np.dtype("complex128"), amplitude
            )
            and _differential_point_source_value_matches(
                native_values[1], complex(medium[0]), np.dtype("complex128"), medium[0]
            )
            and _differential_point_source_value_matches(
                native_values[2], complex(medium[1]), np.dtype("complex128"), medium[1]
            )
            and _differential_point_source_value_matches(
                native_values[3], oscillator, np.dtype("complex128"), 1.0
            ),
            f"{capture_label} native semantics differ from the workload",
        )
        live = {
            name: _differential_source_array_from_record(
                live_records[name], f"{capture_label} candidate live {name}"
            )
            for name in DIFFERENTIAL_POINT_SOURCE_LIVE_ARRAYS
        }
        evaluated_time = float(candidate_time[1]) - 0.5 * float(candidate_time[2])
        evaluated = _differential_continuous_point_source_value(
            evaluated_time, paired_real=paired_real
        )
        expected_evaluated = expected_amplitude.item() * np.asarray(
            (
                [complex(evaluated).real, complex(evaluated).imag]
                if channels == 2
                else [float(evaluated)]
            )
        )
        _require(
            live["overwrite_targets"].dtype == np.dtype("int64")
            and live["overwrite_targets"].shape == (1,)
            and int(live["overwrite_targets"][0]) == flat_target
            and live["overwrite_models"].dtype == np.dtype("int8")
            and live["overwrite_models"].shape == (1,)
            and int(live["overwrite_models"][0]) == source["model"]["id"]
            and live["overwrite_parameters"].dtype == candidate_dtype
            and live["overwrite_parameters"].shape == (1, 6)
            and np.array_equal(live["overwrite_parameters"][0], expected_parameters)
            and live["overwrite_amplitudes"].dtype == candidate_dtype
            and live["overwrite_amplitudes"].shape == (1,)
            and np.array_equal(live["overwrite_amplitudes"][0], expected_amplitude)
            and live["_overwrite_values"].dtype == candidate_dtype
            and live["_overwrite_values"].shape == (1, channels)
            and _differential_point_source_value_matches(
                live["_overwrite_values"][0],
                expected_evaluated,
                candidate_dtype,
                expected_amplitude.item(),
            ),
            f"{capture_label} candidate overwrite semantics differ from the workload",
        )
        for name, shape, dtype in (
            ("additive_targets", (0,), np.dtype("int64")),
            ("additive_models", (0,), np.dtype("int8")),
            ("additive_parameters", (0, 6), candidate_dtype),
            ("additive_amplitudes", (0,), candidate_dtype),
            ("_additive_values", (0, channels), candidate_dtype),
        ):
            _require(
                live[name].shape == shape and live[name].dtype == dtype,
                f"{capture_label} has an unexpected additive source",
            )
        packed_indices = _differential_source_array_from_record(
            torch_source["packed_indices"],
            f"{capture_label} candidate packed indices",
        )
        packed_values = _differential_source_array_from_record(
            torch_source["packed_values"],
            f"{capture_label} candidate packed values",
        )
        expected_packed_values = (
            np.concatenate(
                (
                    np.asarray(
                        [0.0, float(live["overwrite_models"][0])], dtype=np.float64
                    ),
                    live["overwrite_parameters"][0].astype(np.float64),
                    live["overwrite_amplitudes"].astype(np.float64),
                )
            )
            .astype("<f8", copy=False)
            .view("<u8")
        )
        _require(
            packed_indices.dtype == np.dtype("int64")
            and packed_indices.shape == (1, 3)
            and packed_indices.tolist() == [target]
            and packed_values.dtype == np.dtype("<u8")
            and packed_values.shape == (9,)
            and np.array_equal(packed_values, expected_packed_values),
            f"{capture_label} candidate packed semantics differ from live buffers",
        )


def _stable_differential_l2(value: np.ndarray) -> float:
    magnitudes = np.abs(value).reshape(-1)
    scale = float(np.max(magnitudes, initial=0.0))
    if scale == 0.0:
        return 0.0
    scaled = magnitudes / scale
    return scale * math.sqrt(float(np.sum(scaled * scaled, dtype=np.float64)))


def _differential_norms(
    reference: np.ndarray,
    actual: np.ndarray,
    floor: float,
    *,
    zero_exact: bool,
) -> tuple[float, float]:
    if reference.size == 0:
        return (0.0, 0.0) if np.array_equal(reference, actual) else (math.inf,) * 2
    difference = np.abs(actual - reference)
    difference_linf = float(np.max(difference, initial=0.0))
    reference_linf = float(np.max(np.abs(reference), initial=0.0))
    if zero_exact and reference_linf == 0.0:
        return (0.0, 0.0) if np.array_equal(reference, actual) else (math.inf,) * 2
    linf_denominator = max(reference_linf, floor)
    l2_denominator = max(
        _stable_differential_l2(reference), floor * math.sqrt(reference.size)
    )
    linf = (
        0.0
        if difference_linf == 0.0
        else math.inf if linf_denominator == 0.0 else difference_linf / linf_denominator
    )
    difference_l2 = _stable_differential_l2(difference)
    l2 = (
        0.0
        if difference_l2 == 0.0
        else math.inf if l2_denominator == 0.0 else difference_l2 / l2_denominator
    )
    return linf, l2


def _recompute_differential_metrics(
    reference: dict[str, np.ndarray],
    actual: dict[str, np.ndarray],
    comparison: dict[str, Any],
    label: str,
    *,
    array_comparisons: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, float], bool]:
    _require(
        set(reference) == set(actual),
        f"{label} projection array closure differs",
    )
    if array_comparisons is not None:
        _require(
            set(array_comparisons) == set(reference),
            f"{label} per-array comparison closure differs",
        )
    maximum_abs = 0.0
    maximum_relative = 0.0
    maximum_linf = 0.0
    maximum_l2 = 0.0
    passed = True
    for name in reference:
        left = reference[name]
        right = actual[name]
        array_comparison = (
            comparison if array_comparisons is None else array_comparisons[name]
        )
        mode = array_comparison.get("mode")
        floor = (
            array_comparison.get("absolute_scale_floor")
            if mode == DIFFERENTIAL_NORMALIZED_MODE
            else array_comparison.get("atol")
        )
        _require(
            type(floor) is float and math.isfinite(floor) and floor >= 0.0,
            f"{label} array {name} comparison scale floor is invalid",
        )
        _require(
            left.shape == right.shape and left.dtype == right.dtype,
            f"{label} array {name} shape or dtype differs",
        )
        _require(
            np.isfinite(left).all() and np.isfinite(right).all(),
            f"{label} array {name} contains non-finite values",
        )
        integer_exact = left.dtype.kind in {"b", "i", "u"}
        zero_reference = not bool(np.any(left))
        difference = (
            np.abs(right.astype(np.float64) - left.astype(np.float64))
            if integer_exact
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
        if integer_exact or zero_reference:
            equal = np.array_equal(left, right)
            linf = l2 = 0.0 if equal else math.inf
            passed = passed and bool(equal)
        else:
            linf, l2 = _differential_norms(
                left,
                right,
                floor,
                zero_exact=True,
            )
            if mode == DIFFERENTIAL_ELEMENTWISE_MODE:
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


def _expected_completion_differential_records(
    manifest: dict[str, Any], scope: str
) -> list[dict[str, str]]:
    _require(
        scope in FROZEN_DIFFERENTIAL_RECORDS_BY_SCOPE,
        f"unknown differential scope {scope!r}",
    )
    frozen = FROZEN_DIFFERENTIAL_RECORDS_BY_SCOPE[scope]
    for case, _device, _precision in frozen:
        _manifest_case(manifest, case)
    if scope == "paired-real":
        manifest_names = [
            case.get("name")
            for case in manifest.get("correctness", ())
            if case.get("complex") is True
        ]
        frozen_names = list(dict.fromkeys(case for case, _device, _precision in frozen))
        _require(
            manifest_names == frozen_names,
            "paired-real manifest cases differ from the frozen closure",
        )
    return [
        {
            "case": case,
            "device": device,
            "precision": precision,
        }
        for case, device, precision in frozen
    ]


def _is_normalized_differential_record(scope: str, record: dict[str, str]) -> bool:
    return (
        scope,
        record["case"],
        record["device"],
    ) == DIFFERENTIAL_NORMALIZED_CASE


def _expected_differential_projection_steps(
    manifest: dict[str, Any], scope: str, record: dict[str, str]
) -> list[int]:
    workload = _manifest_case(manifest, record["case"])
    reference = manifest.get("reference")
    _require(isinstance(reference, dict), "manifest reference contract is absent")
    capture_steps = workload.get("capture_steps", reference.get("capture_steps"))
    _require(
        capture_steps == list(FROZEN_DIFFERENTIAL_CAPTURE_STEPS),
        "differential capture steps differ from the frozen contract",
    )
    if _is_normalized_differential_record(scope, record):
        _require(
            set(DIFFERENTIAL_NORMALIZED_STEPS) - {0} <= set(capture_steps),
            "normalized differential capture steps are absent from the manifest",
        )
        return list(DIFFERENTIAL_NORMALIZED_STEPS)
    return [FROZEN_DIFFERENTIAL_CAPTURE_STEPS[-1]]


def _expected_differential_projection_groups(
    manifest: dict[str, Any], scope: str, record: dict[str, str]
) -> list[list[int]]:
    projection_steps = _expected_differential_projection_steps(manifest, scope, record)
    groups = (
        [list(group) for group in DIFFERENTIAL_NORMALIZED_GROUPS]
        if _is_normalized_differential_record(scope, record)
        else [projection_steps]
    )
    _require(
        [step for group in groups for step in group] == projection_steps,
        "differential projection groups do not close over the frozen steps",
    )
    return groups


def _expected_differential_group_comment(
    scope: str,
    record: dict[str, str],
    role: str,
    ordinal: int,
    steps: list[int],
    candidate: dict[str, str],
) -> bytes:
    _require(role in {"reference", "candidate"}, "differential group role is invalid")
    _exact_keys(
        candidate,
        {"candidate_git_commit", "candidate_git_status", "manifest_sha256"},
        "differential group candidate evidence",
    )
    return json.dumps(
        {
            "schema": DIFFERENTIAL_GROUP_NPZ_SCHEMA,
            "scope": scope,
            "case": record["case"],
            "device": record["device"],
            "role": role,
            "ordinal": ordinal,
            "steps": steps,
            "candidate_evidence": candidate,
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _differential_projection_name(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} name is empty")
    path = PurePosixPath(value)
    _require(
        not path.is_absolute()
        and path.as_posix() == value
        and all(part not in {"", ".", ".."} for part in path.parts),
        f"{label} name is not canonical",
    )
    return value


def _expected_differential_field_arrays(steps: list[int]) -> list[str]:
    return [
        f"step/{step}/field/{component}" for step in steps for component in FIELD_ARRAYS
    ]


def _expected_differential_physical_arrays(
    steps: list[int], *, normalized: bool
) -> list[str]:
    if not normalized:
        return []
    return [
        f"step/{step}/{suffix}"
        for step in steps
        for suffix in DIFFERENTIAL_PHYSICAL_SUFFIXES
    ]


def _expected_differential_persistent_arrays(
    steps: list[int], case: str, label: str
) -> list[str]:
    try:
        updater_labels = DIFFERENTIAL_CASE_UPDATER_LABELS[case]
    except KeyError as error:
        raise EvidenceError(
            f"{label} has no frozen persistent inventory for {case!r}"
        ) from error
    expected_suffixes = sorted(
        f"state/{component}/{updater}/{kind}"
        for component in FIELD_ARRAYS
        for updater in updater_labels
        for kind in ("indices", "values")
    )
    expected = [
        f"step/{step}/{suffix}" for step in steps for suffix in expected_suffixes
    ]
    return expected


def _validate_differential_persistent_arrays(
    value: Any, steps: list[int], case: str, label: str
) -> list[str]:
    expected = _expected_differential_persistent_arrays(steps, case, label)
    _require(
        isinstance(value, list)
        and all(
            _differential_projection_name(name, f"{label} persistent array") == name
            for name in value
        )
        and value == expected,
        f"{label} persistent array closure differs from the frozen case",
    )
    return expected


def _differential_active_model_names(case: str, label: str) -> tuple[str, ...]:
    try:
        updater_labels = DIFFERENTIAL_CASE_UPDATER_LABELS[case]
    except KeyError as error:
        raise EvidenceError(
            f"{label} has no frozen updater inventory for {case!r}"
        ) from error
    try:
        models = {
            model
            for updater_label in updater_labels
            for model in DIFFERENTIAL_STRATEGY_TOLERANCE_MODELS[
                updater_label.split("-", 1)[-1]
            ]
        }
    except KeyError as error:
        raise EvidenceError(
            f"{label} has no frozen tolerance model for an active updater"
        ) from error
    return tuple(sorted(models))


def _expected_differential_comparison(
    manifest: dict[str, Any], scope: str, record: dict[str, str]
) -> dict[str, Any]:
    dtype = _differential_comparison_dtype(scope, record)
    frozen = _differential_model_tolerance(
        manifest,
        _differential_active_model_names(record["case"], "differential comparison"),
        dtype,
    )
    if _is_normalized_differential_record(scope, record):
        return {
            "mode": DIFFERENTIAL_NORMALIZED_MODE,
            "linf_limit": DIFFERENTIAL_NORMALIZED_LIMIT,
            "l2_limit": DIFFERENTIAL_NORMALIZED_LIMIT,
            "absolute_scale_floor": DIFFERENTIAL_NORMALIZED_ABSOLUTE_SCALE_FLOOR,
            "all_zero_reference": "exact",
        }
    return {
        "mode": DIFFERENTIAL_ELEMENTWISE_MODE,
        "rtol": frozen["rtol"],
        "atol": frozen["atol"],
    }


def _differential_comparison_dtype(scope: str, record: dict[str, str]) -> str:
    return (
        "complex128"
        if scope == "paired-real" and record["device"] == "cpu"
        else record["precision"]
    )


def _differential_model_tolerance(
    manifest: dict[str, Any], models: tuple[str, ...], dtype: str
) -> dict[str, float]:
    if not models:
        return {"rtol": 0.0, "atol": 0.0}
    tolerances = []
    for model in models:
        model_dtype = "float64" if model == "dm2" and dtype == "complex128" else dtype
        try:
            tolerance = FROZEN_DIFFERENTIAL_TOLERANCES[model][model_dtype]
            manifest_tolerance = manifest["tolerances"]["torch"][model][model_dtype]
        except (KeyError, TypeError) as error:
            raise EvidenceError(
                f"there is no frozen {model_dtype} tolerance for {model}"
            ) from error
        _exact_keys(
            manifest_tolerance,
            {"rtol", "atol"},
            f"manifest {model} {model_dtype} tolerance",
        )
        parsed = {name: float(tolerance[name]) for name in ("rtol", "atol")}
        observed = {name: float(manifest_tolerance[name]) for name in ("rtol", "atol")}
        _require(
            observed == parsed
            and all(math.isfinite(value) and value >= 0.0 for value in parsed.values()),
            f"manifest {model} {model_dtype} tolerance differs from the frozen contract",
        )
        tolerances.append(parsed)
    frozen = {
        name: max(tolerance[name] for tolerance in tolerances)
        for name in ("rtol", "atol")
    }
    return frozen


def _expected_differential_array_comparisons(
    manifest: dict[str, Any],
    scope: str,
    record: dict[str, str],
    names: list[str],
) -> dict[str, dict[str, Any]]:
    suite_comparison = _expected_differential_comparison(manifest, scope, record)
    dtype = _differential_comparison_dtype(scope, record)
    result = {}
    for name in names:
        parts = name.split("/")
        step = (
            int(parts[1])
            if len(parts) >= 2
            and parts[0] == "step"
            and parts[1].isdigit()
            and str(int(parts[1])) == parts[1]
            else None
        )
        if suite_comparison["mode"] == DIFFERENTIAL_NORMALIZED_MODE:
            _require(
                step in DIFFERENTIAL_NORMALIZED_STEPS
                or name in {DIFFERENTIAL_SOURCE_ARRAY, DIFFERENTIAL_SOURCE_PROOF_ARRAY},
                f"no frozen normalized differential step for {name!r}",
            )
            if step in DIFFERENTIAL_NORMALIZED_RESIDUAL_STEPS or step is None:
                result[name] = suite_comparison
                continue
        models = _differential_active_model_names(
            record["case"], "differential array comparison"
        )
        if len(parts) == 6 and parts[2] == "state" and parts[-1] == "values":
            strategy = parts[4].split("-", 1)[-1]
            try:
                models = DIFFERENTIAL_STRATEGY_TOLERANCE_MODELS[strategy]
            except KeyError as error:
                raise EvidenceError(
                    f"no frozen differential tolerance model for {strategy!r}"
                ) from error
        tolerance = _differential_model_tolerance(manifest, models, dtype)
        result[name] = {
            "mode": DIFFERENTIAL_ELEMENTWISE_MODE,
            "rtol": tolerance["rtol"],
            "atol": tolerance["atol"],
        }
    return result


def _expected_differential_field_dtype(
    scope: str, record: dict[str, str]
) -> np.dtype[Any]:
    if scope == "paired-real":
        return np.dtype("complex128" if record["device"] == "cpu" else "complex64")
    return np.dtype(record["precision"])


def _differential_whole_shape(workload: dict[str, Any]) -> tuple[int, int, int]:
    size = workload.get("size")
    resolution = workload.get("resolution")
    _require(
        isinstance(size, list)
        and len(size) == 3
        and all(
            type(length) in {int, float} and math.isfinite(float(length))
            for length in size
        )
        and type(resolution) in {int, float}
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


def _expected_differential_field_shapes(
    workload: dict[str, Any],
) -> dict[str, tuple[int, int, int]]:
    nx, ny, nz = _differential_whole_shape(workload)
    return {
        "Ex": (nx, ny + 1, nz + 1),
        "Ey": (nx + 1, ny, nz + 1),
        "Ez": (nx + 1, ny + 1, nz),
        "Hx": (nx, ny + 1, nz + 1),
        "Hy": (nx + 1, ny, nz + 1),
        "Hz": (nx + 1, ny + 1, nz),
    }


def _differential_strategy_pole_count(workload: dict[str, Any], prefix: str) -> int:
    families = workload.get("families", ())
    _require(
        isinstance(families, (list, tuple))
        and all(isinstance(name, str) for name in families),
        "differential workload material families are invalid",
    )
    names = [workload.get("material", ""), *families]
    return 4 if f"{prefix}-4" in names else 1


def _differential_persistent_state_width(
    workload: dict[str, Any], component: str, updater_label: str
) -> int:
    strategy = updater_label.split("-", 1)[-1]
    if strategy == "Cpml":
        return 2
    if strategy == "Upml":
        return 1
    if strategy in {"Dielectric", "Dummy"} or component.startswith("H"):
        return 0
    if strategy == "Drude":
        return 2 * _differential_strategy_pole_count(workload, "drude")
    if strategy == "Lorentz":
        return 2 * _differential_strategy_pole_count(workload, "lorentz")
    if strategy == "DcpAde":
        return 7
    if strategy in {"DcpPlrc", "DcpPlrc+DcpRc", "DcpRc"}:
        return 6
    if strategy == "Dm2":
        _require(
            _differential_strategy_pole_count(workload, "dm2") == 1,
            "no frozen differential state width exists for multi-transition DM2",
        )
        return 3
    raise EvidenceError(f"no frozen persistent-state width for {strategy!r}")


def _differential_array_digest(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(b"issue-123-differential-array-v1\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _differential_persistent_geometry_sha256(
    arrays: dict[str, np.ndarray],
    workload: dict[str, Any],
    case: str,
    step: int,
) -> str:
    try:
        updater_labels = DIFFERENTIAL_CASE_UPDATER_LABELS[case]
    except KeyError as error:
        raise EvidenceError(
            f"no frozen persistent geometry exists for {case!r}"
        ) from error
    digest = hashlib.sha256()
    digest.update(b"issue-123-persistent-geometry-v1\0")
    digest.update(case.encode())
    digest.update(b"\0")
    digest.update(
        json.dumps(
            workload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    )
    digest.update(b"\0")
    suffixes = sorted(
        f"state/{component}/{updater_label}/indices"
        for component in FIELD_ARRAYS
        for updater_label in updater_labels
    )
    for suffix in suffixes:
        indices = arrays[f"step/{step}/{suffix}"]
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


def _recompute_differential_physical_arrays(
    arrays: dict[str, np.ndarray], step: int
) -> dict[str, np.ndarray]:
    summary = [0.0, 0.0, 0.0, 0.0, 1.0]
    result: dict[str, np.ndarray] = {}
    for component in FIELD_ARRAYS:
        field = arrays[f"step/{step}/field/{component}"]
        magnitude = np.abs(field)
        summary[0] += float(np.sum(magnitude * magnitude))
        summary[1] = max(summary[1], float(np.max(magnitude)))
        summary[2] += float(np.sum(magnitude[0] * magnitude[0]))
        summary[3] += float(np.sum(magnitude[-1] * magnitude[-1]))
        summary[4] = float(bool(summary[4]) and np.isfinite(field).all())
        axes = tuple(range(1, field.ndim))
        line = np.mean(field, axis=axes) if axes else field
        result[f"step/{step}/physical/spectrum/{component}"] = np.ascontiguousarray(
            np.abs(np.fft.fft(line))
        )
    result[f"step/{step}/physical/summary"] = np.asarray(summary, dtype=np.float64)
    return result


def _validate_differential_projected_array_contract(
    arrays: dict[str, np.ndarray],
    workload: dict[str, Any],
    scope: str,
    record: dict[str, str],
    projection_steps: list[int],
    label: str,
    *,
    baseline_index_digests: dict[str, str] | None = None,
) -> None:
    field_shapes = _expected_differential_field_shapes(workload)
    field_dtype = _expected_differential_field_dtype(scope, record)
    physical_dtype = np.empty((), dtype=field_dtype).real.dtype
    try:
        updater_labels = DIFFERENTIAL_CASE_UPDATER_LABELS[record["case"]]
    except KeyError as error:
        raise EvidenceError(
            f"{label} has no frozen updater inventory for {record['case']!r}"
        ) from error
    if baseline_index_digests is None:
        baseline_index_digests = {}
    for step in projection_steps:
        for component, shape in field_shapes.items():
            field = arrays[f"step/{step}/field/{component}"]
            _require(
                field.shape == shape and field.dtype == field_dtype,
                f"{label} field shape or dtype differs for step/{step}/{component}",
            )
        if _is_normalized_differential_record(scope, record):
            for component, shape in field_shapes.items():
                spectrum = arrays[f"step/{step}/physical/spectrum/{component}"]
                _require(
                    spectrum.shape == (shape[0],) and spectrum.dtype == physical_dtype,
                    f"{label} physical spectrum shape or dtype differs for "
                    f"step/{step}/{component}",
                )
            summary = arrays[f"step/{step}/physical/summary"]
            _require(
                summary.shape == (5,) and summary.dtype == np.dtype("float64"),
                f"{label} physical summary shape or dtype differs for step/{step}",
            )
            recomputed_physical = _recompute_differential_physical_arrays(arrays, step)
            for name, expected in recomputed_physical.items():
                _require(
                    np.array_equal(arrays[name], expected),
                    f"{label} {name} differs from the projected fields",
                )
        for component, shape in field_shapes.items():
            covered = np.zeros(math.prod(shape), dtype=np.bool_)
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
                width = _differential_persistent_state_width(
                    workload, component, updater_label
                )
                _require(
                    values.dtype == np.dtype("complex128")
                    and values.shape == (indices.shape[0] * width,),
                    f"{label} persistent value shape or dtype differs for {prefix}",
                )
                suffix = f"state/{component}/{updater_label}/indices"
                index_digest = _differential_array_digest(indices)
                previous = baseline_index_digests.get(suffix)
                if previous is None:
                    baseline_index_digests[suffix] = index_digest
                else:
                    _require(
                        previous == index_digest,
                        f"{label} persistent indices change across capture groups "
                        f"for {suffix}",
                    )
            _require(
                bool(np.all(covered)),
                f"{label} persistent indices do not cover the complete field for "
                f"step/{step}/{component}",
            )
        try:
            expected_geometry = FROZEN_DIFFERENTIAL_PERSISTENT_GEOMETRY_SHA256_BY_CASE[
                record["case"]
            ]
        except KeyError as error:
            raise EvidenceError(
                f"{label} has no frozen persistent geometry digest for "
                f"{record['case']!r}"
            ) from error
        _require(
            _differential_persistent_geometry_sha256(
                arrays, workload, record["case"], step
            )
            == expected_geometry,
            f"{label} persistent geometry differs from the frozen case for "
            f"step/{step}",
        )


def _expected_differential_precision_limitation(
    reference: dict[str, np.ndarray],
    *,
    normalized: bool,
    label: str,
) -> dict[str, Any] | None:
    if not normalized:
        return None
    field_names = [f"step/100/field/{component}" for component in FIELD_ARRAYS]
    maximum = max(
        float(np.max(np.abs(reference[name]), initial=0.0)) for name in field_names
    )
    float32_maximum = float(np.finfo(np.float32).max)
    _require(
        math.isfinite(maximum) and maximum > float32_maximum,
        f"{label} native reference does not prove the float32 range limitation",
    )
    return {
        "contract_id": CUDA_PRECISION_LIMITATION_REVIEW["contract_id"],
        "rejected_precision": CUDA_PRECISION_LIMITATION_REVIEW["rejected_precision"],
        "accepted_precision": CUDA_PRECISION_LIMITATION_REVIEW["accepted_precision"],
        "reference_step": 100,
        "reference_field_max_abs": maximum,
        "rejected_precision_max": float32_maximum,
        "range_exceeded": True,
        "reason": CUDA_PRECISION_LIMITATION_REVIEW["reason"],
    }


def _validate_differential(
    artifact: LoadedArtifact,
    reader: ArtifactReader,
    manifest: dict[str, Any],
    candidate: dict[str, str],
    *,
    scope: str,
) -> list[dict[str, Any]]:
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
        _is_exact_int(document["schema_version"], DIFFERENTIAL_SCHEMA_VERSION)
        and document["kind"] == DIFFERENTIAL_KIND
        and document["scope"] == scope
        and document["candidate_evidence"] == candidate,
        f"{scope} differential contract differs",
    )
    expected = _expected_completion_differential_records(manifest, scope)
    _require(
        _type_exact_equal(document["required_cases"], expected),
        f"{scope} required cases differ",
    )
    cases = document["cases"]
    _require(
        isinstance(cases, list)
        and [
            {
                "case": case.get("case"),
                "device": case.get("device"),
                "precision": case.get("precision"),
            }
            for case in cases
        ]
        == expected,
        f"{scope} evaluated case closure differs",
    )
    used_projection_paths: set[str] = set()
    used_projection_digests: set[str] = set()
    source_bindings = []
    for index, (record, expected_record) in enumerate(
        zip(cases, expected, strict=True)
    ):
        label = f"{scope} case {index}"
        _exact_keys(
            record,
            {
                "case",
                "device",
                "precision",
                "projection_steps",
                "projection_groups",
                "reference_source",
                "candidate_source",
                "reference",
                "candidate",
                "field_arrays",
                "physical_arrays",
                "persistent_arrays",
                "contract_arrays",
                "comparison",
                "precision_limitation",
                "metrics",
                "passed",
            },
            label,
        )
        projection_steps = _expected_differential_projection_steps(
            manifest, scope, expected_record
        )
        projection_groups = _expected_differential_projection_groups(
            manifest, scope, expected_record
        )
        _require(
            record["case"] == expected_record["case"]
            and record["device"] == expected_record["device"]
            and record["precision"] == expected_record["precision"]
            and _type_exact_equal(record["projection_steps"], projection_steps)
            and _type_exact_equal(record["projection_groups"], projection_groups),
            f"{label} identity differs",
        )
        sources = {}
        for role in ("reference", "candidate"):
            descriptor = _validate_descriptor(
                record[f"{role}_source"], candidate, f"{label} {role} source"
            )
            _require(
                descriptor["media_type"] == MEDIA_TYPE_NPZ,
                f"{label} {role} source media type differs",
            )
            sources[role] = descriptor
        source_bindings.append(
            {
                **copy.deepcopy(expected_record),
                "reference_source": sources["reference"],
                "candidate_source": sources["candidate"],
            }
        )
        normalized = _is_normalized_differential_record(scope, expected_record)
        field_arrays = record["field_arrays"]
        physical_arrays = record["physical_arrays"]
        persistent_arrays = record["persistent_arrays"]
        contract_arrays = record["contract_arrays"]
        expected_fields = _expected_differential_field_arrays(projection_steps)
        expected_physical = _expected_differential_physical_arrays(
            projection_steps, normalized=normalized
        )
        _require(
            _type_exact_equal(field_arrays, expected_fields),
            f"{label} complete field array list differs",
        )
        _require(
            _type_exact_equal(physical_arrays, expected_physical),
            f"{label} physical array closure differs",
        )
        persistent_arrays = _validate_differential_persistent_arrays(
            persistent_arrays,
            projection_steps,
            expected_record["case"],
            label,
        )
        _require(
            _type_exact_equal(
                contract_arrays,
                [DIFFERENTIAL_SOURCE_ARRAY, DIFFERENTIAL_SOURCE_PROOF_ARRAY],
            ),
            f"{label} source contract array closure differs",
        )
        names = field_arrays + physical_arrays + persistent_arrays + contract_arrays
        _require(len(set(names)) == len(names), f"{label} repeats array names")
        reference_descriptors = record["reference"]
        candidate_descriptors = record["candidate"]
        _require(
            isinstance(reference_descriptors, list)
            and isinstance(candidate_descriptors, list)
            and len(reference_descriptors) == len(projection_groups)
            and len(candidate_descriptors) == len(projection_groups),
            f"{label} projection descriptor group closure differs",
        )
        workload = _manifest_case(manifest, expected_record["case"])
        expected_source = _expected_differential_source_contract(workload)
        expected_comparison = _expected_differential_comparison(
            manifest, scope, expected_record
        )
        _require(
            _type_exact_equal(record["comparison"], expected_comparison),
            f"{label} comparison contract differs from the manifest",
        )
        field_dtype = _expected_differential_field_dtype(scope, expected_record)
        aggregate_metrics: dict[str, float] | None = None
        aggregate_passed = True
        expected_limitation = None
        baseline_index_digests: dict[str, str] = {}
        source_contract_raw: bytes | None = None
        source_proof_raw: bytes | None = None
        for ordinal, group_steps in enumerate(projection_groups):
            group_label = f"{label} projection group {ordinal}"
            group_fields = _expected_differential_field_arrays(group_steps)
            group_physical = _expected_differential_physical_arrays(
                group_steps, normalized=normalized
            )
            group_persistent = _expected_differential_persistent_arrays(
                group_steps, expected_record["case"], group_label
            )
            group_names = (
                group_fields + group_physical + group_persistent + contract_arrays
            )
            loaded_arrays: dict[str, dict[str, np.ndarray]] = {}
            for role, descriptors in (
                ("reference", reference_descriptors),
                ("candidate", candidate_descriptors),
            ):
                loaded = reader.load(
                    descriptors[ordinal],
                    f"{group_label} {role}",
                    json_document=False,
                    expected_media_types={MEDIA_TYPE_NPZ},
                )
                descriptor = loaded.descriptor
                _require(
                    descriptor["path"] not in used_projection_paths
                    and descriptor["sha256"] not in used_projection_digests,
                    f"{group_label} projection descriptor path or digest is reused",
                )
                used_projection_paths.add(descriptor["path"])
                used_projection_digests.add(descriptor["sha256"])
                loaded_arrays[role] = _npz_arrays(
                    loaded,
                    group_names,
                    f"{group_label} {role}",
                    expected_comment=_expected_differential_group_comment(
                        scope,
                        expected_record,
                        role,
                        ordinal,
                        group_steps,
                        candidate,
                    ),
                )
            reference_arrays = loaded_arrays["reference"]
            actual_arrays = loaded_arrays["candidate"]
            _validate_differential_projected_array_contract(
                reference_arrays,
                workload,
                scope,
                expected_record,
                group_steps,
                f"{group_label} reference",
                baseline_index_digests=baseline_index_digests,
            )
            _validate_differential_projected_array_contract(
                actual_arrays,
                workload,
                scope,
                expected_record,
                group_steps,
                f"{group_label} candidate",
                baseline_index_digests=baseline_index_digests,
            )
            _require(
                all(
                    reference_arrays[name].dtype
                    == actual_arrays[name].dtype
                    == field_dtype
                    for name in group_fields
                ),
                f"{group_label} field dtype differs from the frozen scope",
            )
            source_reference = reference_arrays[DIFFERENTIAL_SOURCE_ARRAY]
            source_actual = actual_arrays[DIFFERENTIAL_SOURCE_ARRAY]
            _require(
                source_reference.dtype == source_actual.dtype == np.dtype("uint8")
                and source_reference.ndim == source_actual.ndim == 1
                and 0 < source_reference.size <= 64 * 1024
                and np.array_equal(source_reference, source_actual),
                f"{group_label} PointSource contract bytes differ",
            )
            current_source_raw = source_reference.tobytes()
            _require(
                _strict_json_bytes(
                    current_source_raw,
                    f"{group_label} PointSource contract",
                    max_bytes=64 * 1024,
                )
                == expected_source
                and current_source_raw
                == json.dumps(
                    expected_source,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode(),
                f"{group_label} PointSource contract differs from the workload",
            )
            _validate_differential_source_proof(
                reference_arrays[DIFFERENTIAL_SOURCE_PROOF_ARRAY],
                actual_arrays[DIFFERENTIAL_SOURCE_PROOF_ARRAY],
                manifest,
                workload,
                expected_record["precision"],
                expected_source,
                group_label,
            )
            current_proof_raw = reference_arrays[
                DIFFERENTIAL_SOURCE_PROOF_ARRAY
            ].tobytes()
            if source_contract_raw is None:
                source_contract_raw = current_source_raw
                source_proof_raw = current_proof_raw
            else:
                _require(
                    current_source_raw == source_contract_raw
                    and current_proof_raw == source_proof_raw,
                    f"{group_label} source contract or proof changes across groups",
                )
            if normalized and 100 in group_steps:
                expected_limitation = _expected_differential_precision_limitation(
                    reference_arrays,
                    normalized=True,
                    label=group_label,
                )
            group_metrics, group_passed = _recompute_differential_metrics(
                reference_arrays,
                actual_arrays,
                expected_comparison,
                group_label,
                array_comparisons=_expected_differential_array_comparisons(
                    manifest,
                    scope,
                    expected_record,
                    group_names,
                ),
            )
            if aggregate_metrics is None:
                aggregate_metrics = group_metrics
            else:
                aggregate_metrics = {
                    name: max(aggregate_metrics[name], value)
                    for name, value in group_metrics.items()
                }
            aggregate_passed = aggregate_passed and group_passed
            del loaded, loaded_arrays, reference_arrays, actual_arrays
        _require(
            _type_exact_equal(record["precision_limitation"], expected_limitation),
            f"{label} precision limitation proof differs",
        )
        _require(
            aggregate_metrics is not None,
            f"{label} has no projection groups",
        )
        _exact_keys(
            record["metrics"],
            set(aggregate_metrics),
            f"{label} metrics",
        )
        for name, value in aggregate_metrics.items():
            _close(record["metrics"][name], value, f"{label} {name}")
        _require(
            aggregate_passed and record["passed"] is True,
            f"{label} differential failed",
        )
    _require(document["passed"] is True, f"{scope} embedded suite pass is false")
    return source_bindings


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
        and diagnostics.get("cuda_graph_execution_representation")
        == CUDA_GRAPH_EXECUTION_REPRESENTATION
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
    graph_contract = payload[21]
    _require(
        isinstance(graph_contract, tuple)
        and len(graph_contract) == 8
        and graph_contract[-1] == CUDA_GRAPH_EXECUTION_REPRESENTATION,
        f"{label} cache preimage differs from the CUDA graph execution contract",
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
        and peak >= before
        and peak >= after
        and type(reserved) is int
        and reserved >= peak,
        f"{label} CUDA memory gate failed",
    )


def _expected_cuda_field_buffer_sizes(
    result: dict[str, Any], label: str
) -> tuple[dict[str, int], int]:
    workload = result.get("workload")
    runtime = result.get("runtime")
    _require(
        isinstance(workload, dict)
        and isinstance(workload.get("size"), list)
        and len(workload["size"]) == 3
        and all(
            not isinstance(length, bool)
            and isinstance(length, (int, float))
            and math.isfinite(float(length))
            and float(length) >= 0.0
            for length in workload["size"]
        )
        and not isinstance(workload.get("resolution"), bool)
        and isinstance(workload.get("resolution"), (int, float))
        and math.isfinite(float(workload["resolution"]))
        and float(workload["resolution"]) > 0.0,
        f"{label} workload grid is invalid",
    )
    _require(isinstance(runtime, dict), f"{label} runtime is absent")
    precision = runtime.get("precision")
    channels = runtime.get("field_storage_channels")
    try:
        element_size = np.dtype(precision).itemsize
    except (TypeError, ValueError) as error:
        raise EvidenceError(f"{label} field precision is invalid") from error
    _require(
        precision in {"float32", "float64"}
        and type(channels) is int
        and channels in {1, 2},
        f"{label} field storage contract is invalid",
    )
    resolution = float(workload["resolution"])
    whole_shape = tuple(
        1 if float(length) == 0.0 else int(np.rint(float(length) * resolution))
        for length in workload["size"]
    )
    _require(
        all(size > 0 for size in whole_shape),
        f"{label} workload grid has an empty non-collapsed dimension",
    )
    nx, ny, nz = whole_shape
    shapes = {
        "Ex": (nx, ny + 1, nz + 1),
        "Ey": (nx + 1, ny, nz + 1),
        "Ez": (nx + 1, ny + 1, nz),
        "Hx": (nx, ny + 1, nz + 1),
        "Hy": (nx + 1, ny, nz + 1),
        "Hz": (nx + 1, ny + 1, nz),
    }
    expected = {
        f"state.{component}": math.prod(shapes[component]) * channels * element_size
        for component in FIELD_ARRAYS
    }
    return expected, element_size


_MATERIAL_DIELECTRIC = (("dielectric", ()),)
_MATERIAL_PML_DIELECTRIC = (("cpml", (2,)), ("dielectric", ()))
_MATERIAL_FULL_2D = (
    ("cpml", (2,)),
    ("dcp-ade", (1, 2)),
    ("dcp-plrc", (1, 2)),
    ("dcp-rc", (1, 2)),
    ("dielectric", ()),
    ("dm2", (1,)),
    ("drude", (1,)),
    ("lorentz", (1,)),
)
_MATERIAL_FULL_3D = tuple(
    signature for signature in _MATERIAL_FULL_2D if signature[0] != "dm2"
)
_MATERIAL_CROSSOVER_3D_TRANSVERSE = tuple(
    signature for signature in _MATERIAL_FULL_3D if signature[0] != "dcp-ade"
)
_MATERIAL_BLOCH_3D_ELECTRIC = (("drude", (4,)),)
_MATERIAL_OVERLAP_ELECTRIC = (("dielectric", ()), ("drude", (1,)))

_FROZEN_CUDA_MATERIAL_TOPOLOGY_BY_CASE = {
    "coverage-1-contiguous": (
        _MATERIAL_FULL_2D,
        _MATERIAL_PML_DIELECTRIC,
        _MATERIAL_PML_DIELECTRIC,
        _MATERIAL_PML_DIELECTRIC,
        _MATERIAL_PML_DIELECTRIC,
        _MATERIAL_PML_DIELECTRIC,
    ),
    "coverage-1-fragmented": (_MATERIAL_PML_DIELECTRIC,) * len(FIELD_ARRAYS),
    **{
        name: (_MATERIAL_FULL_2D,) * 3 + (_MATERIAL_PML_DIELECTRIC,) * 3
        for name in (
            "coverage-10-contiguous",
            "coverage-10-fragmented",
            "coverage-50-contiguous",
            "coverage-50-fragmented",
            "coverage-90-contiguous",
            "coverage-90-fragmented",
            "cpu-crossover-2d",
            "cpu-large-2d",
            "single-gpu-2d",
        )
    },
    "bloch-2d": (_MATERIAL_DIELECTRIC,) * len(FIELD_ARRAYS),
    "bloch-3d": (_MATERIAL_BLOCH_3D_ELECTRIC,) * 3 + (_MATERIAL_DIELECTRIC,) * 3,
    "cpu-crossover-3d": (
        _MATERIAL_FULL_3D,
        _MATERIAL_CROSSOVER_3D_TRANSVERSE,
        _MATERIAL_CROSSOVER_3D_TRANSVERSE,
        _MATERIAL_PML_DIELECTRIC,
        _MATERIAL_PML_DIELECTRIC,
        _MATERIAL_PML_DIELECTRIC,
    ),
    **{
        name: (_MATERIAL_FULL_3D,) * 3 + (_MATERIAL_PML_DIELECTRIC,) * 3
        for name in ("cpu-large-3d", "single-gpu-3d")
    },
    **{
        name: (_MATERIAL_OVERLAP_ELECTRIC,) * 3 + (_MATERIAL_DIELECTRIC,) * 3
        for name in REGION_INVARIANCE_CASES
    },
}

# Counts are frozen from the independently built CPU eager planner. They bind
# producer diagnostics without rerunning a workload-sized planner during audit.
_FROZEN_CUDA_MATERIAL_TARGETS_BY_CASE = {
    "coverage-1-contiguous": (
        (570, 79, 79, 79, 11244, 79, 79, 79),
        (538, 11750),
        (663, 11625),
        (538, 11750),
        (570, 11718),
        (444, 11844),
    ),
    "coverage-1-fragmented": (
        (570, 11718),
        (538, 11750),
        (663, 11625),
        (538, 11750),
        (570, 11718),
        (444, 11844),
    ),
    "coverage-10-contiguous": (
        (570, 158, 158, 158, 10770, 158, 158, 158),
        (538, 156, 156, 156, 10814, 156, 156, 156),
        (663, 158, 158, 158, 10677, 158, 158, 158),
        (538, 11750),
        (570, 11718),
        (444, 11844),
    ),
    "coverage-10-fragmented": (
        (570, 38, 38, 38, 11490, 38, 38, 38),
        (538, 40, 40, 40, 11510, 40, 40, 40),
        (663, 38, 38, 38, 11397, 38, 38, 38),
        (538, 11750),
        (570, 11718),
        (444, 11844),
    ),
    "coverage-50-contiguous": (
        (570, 790, 790, 790, 6978, 790, 790, 790),
        (538, 780, 780, 702, 7226, 780, 780, 702),
        (663, 790, 790, 711, 7043, 790, 790, 711),
        (538, 11750),
        (570, 11718),
        (444, 11844),
    ),
    "coverage-50-fragmented": (
        (570, 209, 209, 190, 10502, 209, 209, 190),
        (538, 180, 180, 200, 10630, 180, 180, 200),
        (663, 171, 171, 190, 10561, 171, 171, 190),
        (538, 11750),
        (570, 11718),
        (444, 11844),
    ),
    "coverage-90-contiguous": (
        (570, 1343, 1343, 1422, 3344, 1422, 1422, 1422),
        (538, 1404, 1404, 1404, 3482, 1326, 1326, 1404),
        (663, 1422, 1422, 1422, 3251, 1343, 1343, 1422),
        (538, 11750),
        (570, 11718),
        (444, 11844),
    ),
    "coverage-90-fragmented": (
        (570, 323, 323, 342, 9704, 342, 342, 342),
        (538, 380, 380, 360, 9550, 360, 360, 360),
        (663, 361, 361, 342, 9535, 342, 342, 342),
        (538, 11750),
        (570, 11718),
        (444, 11844),
    ),
    "bloch-2d": ((768,),) * len(FIELD_ARRAYS),
    "bloch-3d": ((3240,),) * len(FIELD_ARRAYS),
    "cpu-crossover-2d": (
        (1730, 99, 99, 66, 23342, 99, 99, 66),
        (1730, 32, 32, 64, 23614, 32, 32, 64),
        (1575, 33, 33, 66, 23761, 33, 33, 66),
        (1730, 23870),
        (1730, 23870),
        (1884, 23716),
    ),
    "cpu-crossover-3d": (
        (59738, 1664, 832, 832, 197414, 832, 832),
        (59354, 896, 1792, 197414, 1792, 896),
        (59546, 832, 1664, 197606, 1664, 832),
        (62700, 199444),
        (63084, 199060),
        (63012, 199132),
    ),
    "cpu-large-2d": (
        (7010, 6288, 6288, 6288, 364862, 6288, 6288, 6288),
        (7010, 6468, 6468, 6600, 363518, 6468, 6468, 6600),
        (6375, 6419, 6419, 6550, 364449, 6419, 6419, 6550),
        (7010, 402590),
        (7010, 402590),
        (7644, 401956),
    ),
    "cpu-large-3d": (
        (455390, 41472, 38016, 38016, 1448226, 38016, 38016),
        (457670, 39936, 39936, 39936, 1439802, 39936, 39936),
        (455030, 41472, 41472, 41472, 1434762, 41472, 41472),
        (445364, 1651788),
        (443012, 1654140),
        (445476, 1651676),
    ),
    "single-gpu-2d": (
        (5114, 16093, 16093, 16302, 946486, 16093, 16093, 16302),
        (5114, 16590, 16590, 16380, 944342, 16590, 16590, 16380),
        (6135, 16511, 16511, 16302, 943793, 16511, 16511, 16302),
        (5114, 1043462),
        (5114, 1043462),
        (4092, 1044484),
    ),
    "single-gpu-3d": (
        (179168, 18240, 16416, 18240, 618016, 18240, 16416),
        (177612, 15360, 17280, 17280, 622644, 17280, 17280),
        (179064, 14592, 16416, 16416, 625416, 16416, 16416),
        (184608, 700128),
        (186112, 698624),
        (184528, 700208),
    ),
    "equivalent-region-1": (
        (100, 156),
        (100, 156),
        (87, 169),
        (256,),
        (256,),
        (256,),
    ),
    "equivalent-region-32": (
        (100, 156),
        (100, 156),
        (87, 169),
        (256,),
        (256,),
        (256,),
    ),
}

_FROZEN_CUDA_MATERIAL_PLAN_SHA256_BY_CASE = {
    "coverage-1-contiguous": "17571d7373c7d2b189fd3cc1d68c51411b60304f06e31f42168351d9d9380156",
    "coverage-1-fragmented": "e899671163916142f49b3df31ba36f15de8d184a4dc73a0d58e1dabe2ed6a40a",
    "coverage-10-contiguous": "c0815491d7ecd431a1e9496ca52ec770f952c11e691ce288d8bb3a798edd943e",
    "coverage-10-fragmented": "02de13164cd7891cfffc2ddf00ab7541cca7d15b45d08f32528504dea430a112",
    "coverage-50-contiguous": "a04192fcceb904df34c72407af52afea3d9948e9860702512469e6862042f435",
    "coverage-50-fragmented": "36660abe6f8b9b7c6f6f84255507ac6efe93ff35aeea4b097ec1fbecfa2a29e9",
    "coverage-90-contiguous": "94d9f6513555ee77532d64cab03c885bff3e7d65bc78ea944b0cc5f78942690b",
    "coverage-90-fragmented": "9621d41404470a477846be640dfd168c31c290d8dfbfe4d477b000f65b5b8654",
    "bloch-2d": "831af63fbf973a24b10c25fd1a4a61196d448d42bd79803666d168c4a1209ffe",
    "bloch-3d": "9dabf5d501df89e97b2160a816c8bee2e3acaf5b00c3db669836fb8c34c707a3",
    "cpu-crossover-2d": "0a9b738ce299f1922748df9a0271a1a2e83903c9c46af33de062d5ed418d6122",
    "cpu-crossover-3d": "dc7f1ea3f41fb5c9850d132e03e803f5fc496f9f3872a85704312b01bb7c59ef",
    "cpu-large-2d": "b195a094c40f29d40f3e1449a017af0c50f144f60365f623dacfa1016f72b837",
    "cpu-large-3d": "d943c23b32a6c3ca4a49916ff8860d30a7c26674446f7fc913821b3db119bd1b",
    "single-gpu-2d": "7f9044cd13075c9266d1e149fcb61f0af6c08d715cbf01ac986c0e428b4799f3",
    "single-gpu-3d": "fc7f1af1ea0afc0475276db42c443d7a3750dcbbfc1aa0158b2a9cbb08f8a5b9",
    "equivalent-region-1": "6f63be609961ce4bc7a5dc847c52ae103aa493bd46dbb9b92f806cea6fa33245",
    "equivalent-region-32": "6f63be609961ce4bc7a5dc847c52ae103aa493bd46dbb9b92f806cea6fa33245",
}


def _frozen_cuda_material_contract(workload: dict[str, Any], label: str) -> tuple[
    tuple[tuple[tuple[str, tuple[int, ...]], ...], ...],
    tuple[tuple[int, ...], ...],
]:
    _require(isinstance(workload, dict), f"{label} workload is absent")
    name = workload.get("name")
    topology = _FROZEN_CUDA_MATERIAL_TOPOLOGY_BY_CASE.get(name)
    targets = _FROZEN_CUDA_MATERIAL_TARGETS_BY_CASE.get(name)
    _require(
        topology is not None
        and targets is not None
        and name in _FROZEN_CUDA_MATERIAL_PLAN_SHA256_BY_CASE
        and len(topology) == len(FIELD_ARRAYS)
        and len(targets) == len(FIELD_ARRAYS)
        and all(
            signatures == tuple(sorted(signatures))
            and len(signatures) == len(component_targets)
            and all(type(count) is int and count > 0 for count in component_targets)
            for signatures, component_targets in zip(topology, targets, strict=True)
        ),
        f"{label} case has no valid frozen lowered-material contract",
    )
    return topology, targets


def _expected_cuda_persistent_material_inventory(
    workload: dict[str, Any], label: str
) -> dict[str, tuple[str, int, int]]:
    """Derive exact mutable names, widths, and targets from frozen lowering."""

    topology, target_counts = _frozen_cuda_material_contract(workload, label)
    inventory: dict[str, tuple[str, int, int]] = {}
    dm2_ordinal = 0
    for component, signatures, component_targets in zip(
        FIELD_ARRAYS, topology, target_counts, strict=True
    ):
        for bucket_index, ((model, state_shape), targets) in enumerate(
            zip(signatures, component_targets, strict=True)
        ):
            prefix = f"bucket_{component.lower()}_{bucket_index}"
            entries: tuple[tuple[str, int], ...] = ()
            family = "dispersive"
            if model in {"cpml", "upml"}:
                entries = (
                    (f"pml_{component.lower()}_{bucket_index}_state", sum(state_shape)),
                )
                family = "pml"
            elif model in {"drude", "lorentz"}:
                poles = state_shape[0]
                entries = tuple(
                    (f"{prefix}_{suffix}", poles) for suffix in ("previous", "current")
                )
            elif model == "dcp-ade":
                poles, points = state_shape
                entries = (
                    (f"{prefix}_field_old", 1),
                    (f"{prefix}_pole_old", poles),
                    (f"{prefix}_pole_now", poles),
                    (f"{prefix}_point_old", points),
                    (f"{prefix}_point_now", points),
                )
            elif model in {"dcp-plrc", "dcp-rc"}:
                poles, points = state_shape
                entries = (
                    (f"{prefix}_pole_state", poles),
                    (f"{prefix}_point_state", 2 * points),
                )
            elif model == "dm2":
                transitions = state_shape[0]
                entries = ((f"dm2_buckets.{dm2_ordinal}.u", 3 * transitions),)
                family = "dm2"
                dm2_ordinal += 1
            for name, width in entries:
                _require(
                    name not in inventory and type(width) is int and width > 0,
                    f"{label} persistent material inventory is inconsistent",
                )
                inventory[name] = (family, width, targets)
    return inventory


def _validate_frozen_cuda_material_plan(result: dict[str, Any], label: str) -> None:
    workload = result["workload"]
    topology, target_counts = _frozen_cuda_material_contract(workload, label)
    diagnostics = result.get("diagnostics")
    plan = diagnostics.get("material_plan") if isinstance(diagnostics, dict) else None
    _require(
        isinstance(plan, list)
        and len(plan) == len(FIELD_ARRAYS)
        and [record.get("component") for record in plan if isinstance(record, dict)]
        == list(FIELD_ARRAYS),
        f"{label} lowered material-plan component closure differs",
    )
    projection = []
    precision = result["runtime"]["precision"]
    for component, record, signatures, component_targets in zip(
        FIELD_ARRAYS, plan, topology, target_counts, strict=True
    ):
        buckets = record.get("buckets")
        _require(
            type(record.get("launches")) is int
            and record["launches"] == len(signatures)
            and isinstance(buckets, list)
            and len(buckets) == len(signatures),
            f"{label} lowered material-plan {component} bucket closure differs",
        )
        projected_buckets = []
        for index, (bucket, expected_signature, expected_targets) in enumerate(
            zip(buckets, signatures, component_targets, strict=True)
        ):
            signature = bucket.get("signature") if isinstance(bucket, dict) else None
            model, state_shape = expected_signature
            if model == "dielectric":
                expected_state_width = 0
            elif model in {"cpml", "dm2"}:
                expected_state_width = sum(state_shape)
            elif model in {"drude", "lorentz"}:
                expected_state_width = 2 * state_shape[0]
            elif model == "dcp-ade":
                expected_state_width = 1 + 2 * state_shape[0] + 2 * state_shape[1]
            else:
                expected_state_width = state_shape[0] + 2 * state_shape[1]
            _require(
                isinstance(signature, dict)
                and set(signature) == {"model", "component", "precision", "state_shape"}
                and signature["model"] == model
                and signature["component"] == component
                and signature["precision"] == precision
                and signature["state_shape"] == list(state_shape)
                and _is_exact_int(bucket.get("targets"), expected_targets)
                and _is_exact_int(bucket.get("state_width"), expected_state_width),
                f"{label} lowered material-plan {component} bucket {index} differs",
            )
            projected_buckets.append(
                {
                    "model": model,
                    "state_shape": list(state_shape),
                    "targets": expected_targets,
                }
            )
        projection.append({"component": component, "buckets": projected_buckets})
    _require(
        _canonical_sha256(projection)
        == _FROZEN_CUDA_MATERIAL_PLAN_SHA256_BY_CASE[workload["name"]],
        f"{label} lowered material-plan digest differs",
    )


def _validate_cuda_state_finiteness(
    result: dict[str, Any],
    reference: dict[str, Any],
    label: str,
) -> None:
    state = result.get("state_progress")
    _exact_keys(
        state,
        {
            "initial_checksum",
            "post_warmup_checksum",
            "post_one_step_checksum",
            "final_checksum",
            "changed_after_first_timed_step",
            "one_step_count",
            "expected_one_step_count",
            "timed_step_count",
            "expected_timed_step_count",
            "profiler_step_count",
            "expected_profiler_step_count",
            "changed_buffers",
            "fields_changed",
            "all_fields_changed",
            "pml_state_changed",
            "dispersive_state_changed",
            "dm2_state_changed",
        },
        f"{label} state progress",
    )
    checksum_names = (
        "initial_checksum",
        "post_warmup_checksum",
        "post_one_step_checksum",
        "final_checksum",
    )
    _require(
        all(
            not isinstance(state[name], bool)
            and isinstance(state[name], (int, float))
            and math.isfinite(float(state[name]))
            and float(state[name]) >= 0
            for name in checksum_names
        ),
        f"{label} state checksums are non-finite or malformed",
    )
    expected_counts = {
        "expected_one_step_count": reference["performance_warmup_steps"] + 1,
        "expected_timed_step_count": (
            reference["performance_warmup_steps"]
            + reference["performance_steps_per_repeat"]
        ),
        "expected_profiler_step_count": (
            reference["performance_warmup_steps"]
            + reference["performance_profile_steps"]
        ),
    }
    observed_counts = {
        "one_step_count": expected_counts["expected_one_step_count"],
        "timed_step_count": expected_counts["expected_timed_step_count"],
        "profiler_step_count": expected_counts["expected_profiler_step_count"],
    }
    _require(
        all(
            _is_exact_int(state[name], value)
            for name, value in {**expected_counts, **observed_counts}.items()
        ),
        f"{label} state step counts differ from the benchmark contract",
    )
    changed = state["changed_buffers"]
    field_names = {name.lower() for name in FIELD_ARRAYS}
    changed_names = (
        {name for name in changed if isinstance(name, str)}
        if isinstance(changed, list)
        else set()
    )
    material_inventory = _expected_cuda_persistent_material_inventory(
        result["workload"], label
    )
    _validate_frozen_cuda_material_plan(result, label)
    expected_material_flags = {
        "pml_state_changed": any(
            name in changed_names and family == "pml"
            for name, (family, _width, _targets) in material_inventory.items()
        ),
        "dispersive_state_changed": any(
            name in changed_names and family == "dispersive"
            for name, (family, _width, _targets) in material_inventory.items()
        ),
        "dm2_state_changed": any(
            name in changed_names and family == "dm2"
            for name, (family, _width, _targets) in material_inventory.items()
        ),
    }
    material_families = {
        family for family, _width, _targets in material_inventory.values()
    }
    required_material_flags = {
        "pml_state_changed": "pml" in material_families,
        "dispersive_state_changed": "dispersive" in material_families,
        "dm2_state_changed": "dm2" in material_families,
    }
    _require(
        isinstance(changed, list)
        and all(isinstance(name, str) and bool(name) for name in changed)
        and changed == sorted(set(changed))
        and state["fields_changed"] == sorted(set(changed) & field_names)
        and state["all_fields_changed"] is (field_names <= set(changed))
        and state["all_fields_changed"] is True
        and all(
            state[name] is expected
            for name, expected in expected_material_flags.items()
        )
        and all(
            not required or expected_material_flags[name]
            for name, required in required_material_flags.items()
        )
        and state["changed_after_first_timed_step"]
        is (state["post_one_step_checksum"] != state["final_checksum"]),
        f"{label} changed dynamic-state summary differs",
    )
    finiteness = result.get("state_finiteness")
    _exact_keys(
        finiteness,
        {"contract_id", "tracked_buffers", "stages", "passed"},
        f"{label} state finiteness",
    )
    tracked = finiteness["tracked_buffers"]
    tracked_names = (
        set(tracked)
        if isinstance(tracked, list) and all(isinstance(name, str) for name in tracked)
        else set()
    )
    material_prefixes = ("pml_", "bucket_", "dm2_buckets.")
    tracked_material_names = {
        name for name in tracked_names if name.startswith(material_prefixes)
    }
    _require(
        finiteness["contract_id"] == STATE_FINITENESS_CONTRACT_ID
        and isinstance(tracked, list)
        and all(isinstance(name, str) and bool(name) for name in tracked)
        and tracked == sorted(set(tracked))
        and bool(tracked)
        and not any(
            name.startswith("plan.") or ".state.plan." in name for name in tracked
        )
        and tracked_names
        == field_names
        | {"source_time", "time_step", "step_count"}
        | set(material_inventory)
        and set(changed) <= tracked_names
        and tracked_material_names == set(material_inventory),
        f"{label} finite-state buffer or material inventory closure differs",
    )
    stages = finiteness["stages"]
    expected_stages = (
        "initial",
        "post_warmup",
        "post_one_step",
        "post_timed",
        "post_profile",
    )
    _require(
        isinstance(stages, dict) and set(stages) == set(expected_stages),
        f"{label} finite-state stage closure differs",
    )
    profiler = result.get("profiler")
    field_sizes = (
        profiler.get("field_buffer_sizes_bytes") if isinstance(profiler, dict) else None
    )
    expected_field_sizes, element_size = _expected_cuda_field_buffer_sizes(
        result, label
    )
    _require(
        isinstance(field_sizes, dict)
        and all(
            type(field_sizes.get(name)) is int and field_sizes[name] == expected_size
            for name, expected_size in expected_field_sizes.items()
        )
        and type(field_sizes.get("aggregate.all-fields")) is int
        and field_sizes["aggregate.all-fields"] == sum(expected_field_sizes.values()),
        f"{label} field-size inventory differs from the workload",
    )
    expected_field_elements = sum(expected_field_sizes.values()) // element_size
    channels = result["runtime"]["field_storage_channels"]
    expected_material_elements = sum(
        width * targets * channels
        for _family, width, targets in material_inventory.values()
    )
    expected_floating_buffers = len(FIELD_ARRAYS) + len(material_inventory) + 2
    expected_floating_elements = (
        expected_field_elements + expected_material_elements + 2
    )
    sizes = set()
    for stage in expected_stages:
        record = stages[stage]
        _exact_keys(
            record,
            {
                "floating_or_complex_buffer_count",
                "floating_or_complex_element_count",
                "nonfinite_element_count",
                "finite",
            },
            f"{label} state finiteness {stage}",
        )
        _require(
            type(record["floating_or_complex_buffer_count"]) is int
            and record["floating_or_complex_buffer_count"] == expected_floating_buffers
            and type(record["floating_or_complex_element_count"]) is int
            and record["floating_or_complex_element_count"]
            == expected_floating_elements
            and _is_exact_int(record["nonfinite_element_count"], 0)
            and record["finite"] is True,
            f"{label} state finiteness {stage} failed",
        )
        sizes.add(
            (
                record["floating_or_complex_buffer_count"],
                record["floating_or_complex_element_count"],
            )
        )
    _require(
        len(sizes) == 1 and finiteness["passed"] is True,
        f"{label} finite-state shape changed or suite failed",
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
    expected_precision = (
        CUDA_PERFORMANCE_PRECISION_BY_CASE[name]
        if not paired_real and name in CUDA_PERFORMANCE_PRECISION_BY_CASE
        else "float32"
    )
    _require(
        isinstance(runtime, dict)
        and runtime.get("device") == "cuda:0"
        and runtime.get("precision") == expected_precision
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
        and runtime.get("field_storage_dtype") == f"torch.{expected_precision}",
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
    _validate_cuda_state_finiteness(result, reference, label)
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
    differential_source_bindings = _validate_differential(
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
        "differential_source_bindings": differential_source_bindings,
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
    trusted_runtime_receipts: tuple[LoadedArtifact, ...],
) -> dict[str, Any]:
    _exact_keys(
        scope,
        {"cuda_gates", "correctness", "traces"},
        "single-GPU scope",
    )
    artifact = reader.load(scope["cuda_gates"], "single-GPU CUDA gates")
    document = artifact.document
    _document_candidate_matches(document, candidate, required=True)
    gate = document.get("cuda_suite_gate")
    _exact_keys(
        gate,
        {
            "contract_id",
            "required_cases",
            "required_case_precisions",
            "reviewed_precision_limitations",
            "required_correctness_runtime_modes",
            "case_closure_complete",
            "environment_complete",
            "correctness_index_count",
            "correctness_indexes",
            "correctness_evidence_bound",
            "timing_statistics",
            "trace_contract",
            "errors",
            "passed",
        },
        "single-GPU embedded suite gate",
    )
    required_precisions = [
        {"case": name, "precision": CUDA_PERFORMANCE_PRECISION_BY_CASE[name]}
        for name in CUDA_CASES
    ]
    correctness_indexes = gate["correctness_indexes"]
    _require(
        gate["contract_id"] == CUDA_SUITE_CONTRACT_ID
        and gate["required_cases"] == list(CUDA_CASES)
        and gate["required_case_precisions"] == required_precisions
        and gate["reviewed_precision_limitations"] == [CUDA_PRECISION_LIMITATION_REVIEW]
        and gate["required_correctness_runtime_modes"]
        == list(CUDA_CORRECTNESS_RUNTIME_MODES)
        and gate["case_closure_complete"] is True
        and gate["environment_complete"] is True
        and _is_exact_int(gate["correctness_index_count"], 2)
        and isinstance(correctness_indexes, list)
        and len(correctness_indexes) == 2
        and gate["correctness_evidence_bound"] is True
        and gate["timing_statistics"] == "raw-median-relative-mad-v1"
        and gate["trace_contract"] == "sha256-bound-zero-transfer-kernel-count-v1"
        and gate["errors"] == []
        and gate["passed"] is True,
        "single-GPU embedded suite gate differs",
    )
    _rebuilt_correctness, correctness_archive_bindings = (
        _validate_cuda_correctness_indexes(
            correctness_indexes,
            reader,
            manifest,
            candidate,
            "single-GPU embedded suite gate",
            trusted_runtime_receipts,
        )
    )
    suite = document.get("suite_acceptance")
    _require(
        isinstance(suite, dict)
        and suite.get("cuda_suite_expected") is True
        and suite.get("cuda_suite_complete") is True
        and suite.get("passed") is True,
        "single-GPU suite acceptance failed",
    )
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
    differential_source_bindings = _validate_differential(
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
        "_correctness_archive_bindings_by_mode": {
            mode["graph_mode"]: copy.deepcopy(bindings)
            for mode, bindings in zip(
                CUDA_CORRECTNESS_RUNTIME_MODES,
                correctness_archive_bindings,
                strict=True,
            )
        },
        "correctness_candidate_archives_by_mode": {
            mode["graph_mode"]: _candidate_archive_bindings_by_case(
                manifest,
                bindings,
                f"CUDA {mode['graph_mode']} correctness",
            )
            for mode, bindings in zip(
                CUDA_CORRECTNESS_RUNTIME_MODES,
                correctness_archive_bindings,
                strict=True,
            )
        },
        "differential_source_bindings": differential_source_bindings,
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
        and peak >= before
        and peak >= after
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
        with zipfile.ZipFile(io.BytesIO(archive.raw)) as zipped:
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
    if role == "sdist":
        descriptor = _validate_descriptor(record["artifact"], reader.candidate, label)
        _require(
            isinstance(record["filename"], str)
            and record["filename"] == PurePosixPath(descriptor["path"]).name,
            f"{label} filename differs",
        )
        _require(
            descriptor["media_type"] == MEDIA_TYPE_GZIP,
            "macOS sdist media type differs",
        )
        _require(
            record["filename"].endswith(".tar.gz"),
            "macOS sdist filename must end in .tar.gz",
        )
        artifact, inventory = reader.load_private_sdist(
            record["artifact"], f"{label} bytes"
        )
        _require(
            inventory.member_count > 0,
            "macOS sdist structural inventory differs",
        )
    else:
        artifact = reader.load(
            record["artifact"],
            f"{label} bytes",
            json_document=False,
            expected_media_types={MEDIA_TYPE_WHEEL},
        )
        _require(record["filename"] == artifact.path.name, f"{label} filename differs")
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
            with zipfile.ZipFile(io.BytesIO(artifact.raw)) as archive:
                _require(bool(archive.namelist()), "macOS wheel is empty")
                _require(archive.testzip() is None, "macOS wheel CRC check failed")
        except (OSError, zipfile.BadZipFile) as error:
            raise EvidenceError("macOS wheel is not a readable ZIP") from error
    return role, artifact


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


def _load_operations_scope_artifacts(
    scope: Any,
    reader: ArtifactReader,
    candidate: dict[str, str],
) -> tuple[LoadedArtifact, dict[str, LoadedArtifact]]:
    _exact_keys(scope, {"index"}, "operations scope")
    artifact = reader.load(scope["index"], "operations evidence index")
    document = artifact.document
    _require(isinstance(document, dict), "operations evidence index differs")
    records = document.get("responses")
    _require(isinstance(records, dict), "operations response index differs")
    response_artifacts = {}
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
        response_artifacts[role] = loaded
    return artifact, response_artifacts


def _validate_operations_scope(
    scope: Any,
    reader: ArtifactReader,
    candidate: dict[str, str],
) -> dict[str, Any]:
    artifact, response_artifacts = _load_operations_scope_artifacts(
        scope, reader, candidate
    )
    try:
        from benchmarks.issue123_operations import evaluate_operations
    except (ImportError, OSError) as error:
        raise EvidenceError("operations schema validator is unavailable") from error
    try:
        result = evaluate_operations(
            artifact.document,
            {role: item.document for role, item in response_artifacts.items()},
            candidate,
        )
    except (ValueError, TypeError, KeyError) as error:
        raise EvidenceError("raw GitHub operational evidence differs") from error
    _require(
        isinstance(result, dict)
        and result.get("final_acceptance") is False
        and result.get("final_acceptance_authority") == OFFLINE_AUTHORITY,
        "operations evidence is not a non-authoritative structural result",
    )
    return result


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


def _correctness_archives_by_case(
    value: Any,
    manifest: dict[str, Any],
    label: str,
) -> dict[str, dict[str, Any]]:
    required_cases = [
        case["name"]
        for group in ("correctness", "physical_checks")
        for case in manifest[group]
    ]
    _require(
        isinstance(value, list)
        and [record.get("case") for record in value if isinstance(record, dict)]
        == required_cases,
        f"{label} case closure differs",
    )
    by_case = {}
    for index, (record, case) in enumerate(zip(value, required_cases, strict=True)):
        _exact_keys(
            record,
            {"case", "sha256", "size_bytes"},
            f"{label} archive {index}",
        )
        _require(
            record["case"] == case
            and isinstance(record["sha256"], str)
            and SHA256_RE.fullmatch(record["sha256"]) is not None
            and type(record["size_bytes"]) is int
            and 0 < record["size_bytes"] <= MAX_ARTIFACT_BYTES,
            f"{label} archive {index} differs",
        )
        by_case[case] = record
    return by_case


def _validate_differential_correctness_source_bindings(
    cpu: dict[str, Any],
    policy: dict[str, Any],
    single: dict[str, Any],
    manifest: dict[str, Any],
    trusted_runtime_receipts: tuple[LoadedArtifact, ...],
) -> dict[str, Any]:
    cpu_archives = _correctness_archives_by_case(
        cpu.get("correctness_candidate_archives"), manifest, "CPU correctness"
    )
    cuda_by_mode = single.get("correctness_candidate_archives_by_mode")
    _exact_keys(cuda_by_mode, {"eager", "graph"}, "CUDA correctness runtime modes")
    cuda_eager_archives = _correctness_archives_by_case(
        cuda_by_mode["eager"], manifest, "CUDA eager correctness"
    )
    _correctness_archives_by_case(
        cuda_by_mode["graph"], manifest, "CUDA graph correctness"
    )
    _require(
        isinstance(trusted_runtime_receipts, tuple)
        and len(trusted_runtime_receipts) == len(SINGLE_GPU_DIFFERENTIAL_RUNTIME_MODES)
        and all(
            isinstance(receipt, LoadedArtifact) for receipt in trusted_runtime_receipts
        ),
        "two external single-GPU differential runtime receipts are required",
    )
    single_receipts = {}
    for case, mode, receipt in zip(
        ("single-gpu-2d", "single-gpu-3d"),
        SINGLE_GPU_DIFFERENTIAL_RUNTIME_MODES,
        trusted_runtime_receipts,
        strict=True,
    ):
        _exact_keys(
            receipt.descriptor,
            {"path", "sha256", "size_bytes", "media_type"},
            f"trusted {case} runtime receipt descriptor",
        )
        checked_path, current_raw = _bounded_regular_file_bytes(
            receipt.path,
            f"trusted {case} runtime receipt",
            max_bytes=MAX_RUNTIME_RECEIPT_BYTES,
        )
        _require(
            checked_path == receipt.path
            and current_raw == receipt.raw
            and receipt.raw == _canonical_json_bytes(receipt.document)
            and receipt.descriptor.get("sha256") == _sha256(receipt.raw)
            and receipt.descriptor.get("size_bytes") == len(receipt.raw)
            and receipt.descriptor.get("media_type") == MEDIA_TYPE_JSON,
            f"trusted {case} runtime receipt byte binding differs",
        )
        _validate_runtime_receipt_document(
            receipt.document,
            manifest,
            cpu["candidate_evidence"],
            mode,
            f"trusted {case} runtime receipt",
            expected_candidate_cases=(case,),
        )
        single_receipts[case] = receipt.document["candidate_archives"][0]

    paired_sources = policy.get("differential_source_bindings")
    single_sources = single.get("differential_source_bindings")
    _require(
        isinstance(paired_sources, list)
        and [
            {
                "case": item.get("case"),
                "device": item.get("device"),
                "precision": item.get("precision"),
            }
            for item in paired_sources
            if isinstance(item, dict)
        ]
        == _expected_completion_differential_records(manifest, "paired-real"),
        "paired differential source binding closure differs",
    )
    _require(
        isinstance(single_sources, list)
        and [
            {
                "case": item.get("case"),
                "device": item.get("device"),
                "precision": item.get("precision"),
            }
            for item in single_sources
            if isinstance(item, dict)
        ]
        == _expected_completion_differential_records(manifest, "single-gpu-cuda"),
        "single-GPU differential source binding closure differs",
    )

    bound = []
    for source in (*paired_sources, *single_sources):
        _exact_keys(
            source,
            {
                "case",
                "device",
                "precision",
                "reference_source",
                "candidate_source",
            },
            "differential source binding",
        )
        reference_source = _validate_descriptor(
            source["reference_source"],
            cpu["candidate_evidence"],
            f"{source['case']} {source['device']} differential reference source",
        )
        candidate_source = _validate_descriptor(
            source["candidate_source"],
            cpu["candidate_evidence"],
            f"{source['case']} {source['device']} differential candidate source",
        )
        _require(
            reference_source["media_type"]
            == candidate_source["media_type"]
            == MEDIA_TYPE_NPZ,
            f"{source['case']} {source['device']} differential sources are not NPZ",
        )
        if source["case"] in single_receipts:
            role = source["case"]
            archive = single_receipts[source["case"]]
        elif source["device"] == "cpu" and source["precision"] == "float64":
            role = "cpu"
            archive = cpu_archives.get(source["case"])
        elif source["device"] == "cuda:0" and source["precision"] == "float32":
            role = "cuda-eager"
            archive = cuda_eager_archives.get(source["case"])
        else:
            raise EvidenceError(
                f"{source['case']} differential source has no trusted runtime role"
            )
        _require(
            isinstance(archive, dict)
            and candidate_source["sha256"] == archive["sha256"]
            and candidate_source["size_bytes"] == archive["size_bytes"],
            f"{source['case']} {source['device']} differential candidate source "
            "differs from its externally attested runtime archive",
        )
        bound.append(
            {
                "case": source["case"],
                "device": source["device"],
                "precision": source["precision"],
                "runtime_receipt_role": role,
                "sha256": archive["sha256"],
                "size_bytes": archive["size_bytes"],
            }
        )
    _require(
        len(bound) == 18,
        "differential/correctness source binding count differs",
    )
    return {
        "bound_source_count": len(bound),
        "bindings": bound,
    }


def _empty_scope_result() -> dict[str, Any]:
    return {"satisfied": False, "errors": [], "details": None}


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _compact_canonical_json_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise EvidenceError("JSON value is not canonicalizable") from error
    return (rendered + "\n").encode("utf-8")


def _private_binding_modules() -> tuple[Any, Any]:
    try:
        from benchmarks import issue123_privacy as privacy
        from benchmarks import issue123_publication as publication
    except (ImportError, OSError) as error:
        raise EvidenceError(
            "private bundle binding interfaces are unavailable"
        ) from error
    return privacy, publication


def _private_binding_hmac(salt: bytes, domain: str, value: Any) -> str:
    privacy, _publication = _private_binding_modules()
    raw = privacy.binding_canonical_json_bytes(value)
    framed = (
        privacy.TAGGED_DIGEST_PREFIX
        + domain.encode("ascii")
        + b"\x00"
        + len(raw).to_bytes(8, "big")
        + raw
    )
    return hmac.new(salt, framed, hashlib.sha256).hexdigest()


def _bundle_path_identity(path: Path) -> dict[str, Any]:
    metadata = path.stat()
    return {
        "path_sha256": hashlib.sha256(str(path).encode("utf-8")).hexdigest(),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
    }


def _load_inventory_index(index_path: Path | str) -> tuple[Path, bytes, dict[str, Any]]:
    checked, raw = _bounded_regular_file_bytes(
        Path(index_path), "completion bundle inventory index", max_bytes=MAX_INDEX_BYTES
    )
    _require(
        checked.name == "completion-index.json",
        "content-addressed bundle index must be named completion-index.json",
    )
    document = _strict_json_bytes(
        raw, "completion bundle inventory index", max_bytes=MAX_INDEX_BYTES
    )
    _exact_keys(
        document,
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
        "completion bundle inventory index",
    )
    _require(
        _is_exact_int(document["schema_version"], INDEX_SCHEMA_VERSION)
        and document["kind"] == INDEX_KIND
        and _is_exact_int(document["issue"], 123),
        "completion bundle inventory index identity differs",
    )
    _exact_keys(
        document["artifacts"],
        {"cpu", "policy_paired_real", "single_gpu", "two_gpu", "macos", "operations"},
        "completion bundle inventory scope mapping",
    )
    return checked, raw, document


def _protected_frozen_inventory(
    binding_context: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    privacy, _publication = _private_binding_modules()
    technical = binding_context.get("technical_inventory")
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
        technical["scope_order"] == list(privacy.TECHNICAL_SCOPE_ORDER),
        "protected technical scope order differs",
    )
    frozen_scopes = technical["scope_artifacts"]
    _exact_keys(
        frozen_scopes,
        set(privacy.TECHNICAL_SCOPE_ORDER),
        "protected first-five scope mappings",
    )
    _require(
        all(
            isinstance(frozen_scopes[scope], dict) and frozen_scopes[scope]
            for scope in privacy.TECHNICAL_SCOPE_ORDER
        ),
        "protected first-five scope mappings are empty",
    )
    receipts = technical["runtime_receipts"]
    _require(
        isinstance(receipts, list) and len(receipts) == len(RUNTIME_RECEIPT_ROLES),
        "protected runtime receipt closure differs",
    )
    normalized = []
    for ordinal, (record, role) in enumerate(
        zip(receipts, RUNTIME_RECEIPT_ROLES, strict=True)
    ):
        _exact_keys(
            record,
            {"role", "bundle_path", "sha256", "size_bytes"},
            f"protected runtime receipt {ordinal}",
        )
        _canonical_bundle_path(
            record["bundle_path"], f"protected runtime receipt {ordinal} path"
        )
        _require(
            record["role"] == role
            and isinstance(record["sha256"], str)
            and SHA256_RE.fullmatch(record["sha256"]) is not None
            and type(record["size_bytes"]) is int
            and 0 < record["size_bytes"] <= MAX_RUNTIME_RECEIPT_BYTES,
            f"protected runtime receipt {ordinal} differs",
        )
        normalized.append(dict(record))
    _require(
        len({item["bundle_path"] for item in normalized}) == len(normalized),
        "protected runtime receipt paths are duplicated",
    )
    expected_root = privacy.tagged_canonical_sha256(
        privacy.TECHNICAL_INPUT_INVENTORY_DOMAIN,
        technical,
    )
    _require(
        isinstance(binding_context.get("technical_input_root"), str)
        and hmac.compare_digest(binding_context["technical_input_root"], expected_root),
        "protected technical inventory root differs",
    )
    return copy.deepcopy(frozen_scopes), copy.deepcopy(normalized)


def _require_bundle_matches_frozen_inventory(
    document: Mapping[str, Any],
    registered_payloads: list[Mapping[str, Any]],
    frozen_scope_artifacts: Mapping[str, Any],
    frozen_runtime_receipts: list[Mapping[str, Any]],
) -> None:
    privacy, _publication = _private_binding_modules()
    projected = {
        scope: document["artifacts"][scope] for scope in privacy.TECHNICAL_SCOPE_ORDER
    }
    _require(
        projected == frozen_scope_artifacts,
        "completion first-five mappings differ from protected inventory",
    )
    registry = {item["path"]: item for item in registered_payloads}
    reconstructed = []
    for record in frozen_runtime_receipts:
        descriptor = registry.get(record["bundle_path"])
        _require(
            isinstance(descriptor, Mapping)
            and descriptor["sha256"] == record["sha256"]
            and descriptor["size_bytes"] == record["size_bytes"]
            and descriptor["media_type"] == MEDIA_TYPE_JSON
            and descriptor["candidate_evidence"] == document["candidate_evidence"],
            "completion runtime receipt differs from protected inventory",
        )
        reconstructed.append(dict(record))
    _require(
        reconstructed == frozen_runtime_receipts,
        "completion runtime receipt order differs from protected inventory",
    )


def _validated_bundle_inventory_value(
    index_raw: bytes,
    document: Mapping[str, Any],
    descriptor_ledger: list[Mapping[str, Any]],
    binding_context: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    privacy, _publication = _private_binding_modules()
    frozen_scopes, runtime_receipts = _protected_frozen_inventory(binding_context)
    _require_bundle_matches_frozen_inventory(
        document,
        descriptor_ledger,
        frozen_scopes,
        runtime_receipts,
    )
    value = {
        "index": {
            "sha256": _sha256(index_raw),
            "size_bytes": len(index_raw),
        },
        "scope_order": [
            "cpu",
            "policy_paired_real",
            "single_gpu",
            "two_gpu",
            "macos",
            "operations",
        ],
        "scope_artifacts": copy.deepcopy(document["artifacts"]),
        "runtime_receipts": copy.deepcopy(runtime_receipts),
        "payloads": [dict(item) for item in descriptor_ledger],
    }
    return value, privacy.tagged_canonical_sha256(
        COMPLETION_BUNDLE_INVENTORY_DOMAIN, value
    )


def _bundle_inventory_from_context(
    index_path: Path | str, binding_context: Mapping[str, Any]
) -> dict[str, Any]:
    privacy, _publication = _private_binding_modules()
    index, index_raw, document = _load_inventory_index(index_path)
    payloads = document["payloads"]
    _require(isinstance(payloads, list), "completion payload registry differs")
    reader = ArtifactReader(index.parent, document["candidate_evidence"], payloads)
    ledger = []
    identities = []
    for descriptor in sorted(payloads, key=lambda item: item.get("path", "")):
        validated = _validate_descriptor(
            descriptor,
            document["candidate_evidence"],
            "completion inventory payload",
        )
        artifact = reader.load(
            validated,
            "completion inventory payload",
            json_document=False,
        )
        ledger.append(dict(artifact.descriptor))
        identities.append(
            (artifact.path, artifact.path.stat().st_dev, artifact.path.stat().st_ino)
        )
    expected_files = {"completion-index.json", *(item["path"] for item in ledger)}
    observed_files = set()
    try:
        for candidate in index.parent.rglob("*"):
            metadata = candidate.lstat()
            _require(
                not stat.S_ISLNK(metadata.st_mode), "bundle contains a symbolic link"
            )
            if stat.S_ISREG(metadata.st_mode):
                observed_files.add(candidate.relative_to(index.parent).as_posix())
            else:
                _require(
                    stat.S_ISDIR(metadata.st_mode), "bundle contains a special file"
                )
    except OSError as error:
        raise EvidenceError(
            "bundle directory closure could not be inspected"
        ) from error
    _require(
        observed_files == expected_files,
        "reopened bundle contains missing or extra files",
    )
    value, root = _validated_bundle_inventory_value(
        index_raw,
        document,
        ledger,
        binding_context,
    )
    return {
        "root": root,
        "inventory": value,
        "index_path": index,
        "file_identities": identities,
    }


def completion_bundle_inventory(
    index_path: Path | str,
    protected_openings: Path | str | bytes | Mapping[str, Any],
) -> dict[str, Any]:
    """Reopen an exact completion directory and return its content address."""

    privacy, _publication = _private_binding_modules()
    _openings, context = privacy.load_private_openings(protected_openings)
    state = _bundle_inventory_from_context(index_path, context)
    return {
        "root": state["root"],
        "inventory": state["inventory"],
    }


def _operations_issue_response(
    index_path: Path, index_document: Mapping[str, Any]
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    reader = ArtifactReader(
        index_path.parent,
        index_document["candidate_evidence"],
        index_document["payloads"],
    )
    operations_artifact, responses = _load_operations_scope_artifacts(
        index_document["artifacts"]["operations"],
        reader,
        index_document["candidate_evidence"],
    )
    _require(
        operations_artifact.document.get("schema_version") == 2
        and "issue_123" in responses,
        "bundle operations capture cannot prove issue #123 acknowledgment",
    )
    response = responses["issue_123"]
    _require(
        isinstance(response.document, dict),
        "bundle issue #123 response is not an object",
    )
    return response.document, response.raw, response.descriptor


def _checklist_observation(issue: Mapping[str, Any], expected: str) -> dict[str, Any]:
    try:
        from benchmarks import issue123_operations as operations

        observation = operations._validate_post_bundle_checklist(issue, expected)
        transition = operations.checklist_transition_sha256(issue, expected)
    except (ImportError, TypeError, ValueError):
        raise EvidenceError("issue final checklist evidence differs") from None
    return {
        **observation,
        "checklist_transition_sha256": transition,
    }


def _load_reopen_receipt(
    source: Path | str | bytes | Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    privacy, _publication = _private_binding_modules()
    if isinstance(source, (Path, str)):
        _path, raw = _bounded_regular_file_bytes(
            Path(source), "bundle reopen receipt", max_bytes=MAX_JSON_BYTES
        )
        document = _strict_json_bytes(
            raw, "bundle reopen receipt", max_bytes=MAX_JSON_BYTES
        )
        _require(
            raw == privacy.binding_canonical_json_bytes(document),
            "bundle reopen receipt is not canonical JSON",
        )
    elif type(source) is bytes:
        raw = source
        document = _strict_json_bytes(
            raw, "bundle reopen receipt", max_bytes=MAX_JSON_BYTES
        )
        _require(
            raw == privacy.binding_canonical_json_bytes(document),
            "bundle reopen receipt is not canonical JSON",
        )
    else:
        document = dict(source)
        raw = privacy.binding_canonical_json_bytes(document)
    return document, raw


def _reopen_receipt_file_sha256(raw: bytes) -> str:
    _document, canonical = _load_reopen_receipt(raw)
    _require(
        hmac.compare_digest(raw, canonical),
        "bundle reopen receipt bytes are not canonical",
    )
    return hashlib.sha256(canonical).hexdigest()


def validate_bundle_reopen_receipt(
    receipt: Path | str | bytes | Mapping[str, Any],
    expected_stage: str,
    protected_openings: Path | str | bytes | Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate a detached B0/B1 reopen receipt against the sidecar."""

    privacy, _publication = _private_binding_modules()
    openings, context = privacy.load_private_openings(protected_openings)
    document, _raw = _load_reopen_receipt(receipt)
    _exact_keys(
        document,
        {
            "schema_version",
            "kind",
            "stage",
            "observed_at",
            "source_bundle",
            "reopened_bundle",
            "registered_payloads",
            "technical_binding",
            "issue_response",
            "pre_acknowledgment_receipt_sha256",
            "binding",
        },
        "bundle reopen receipt",
    )
    _require(
        _is_exact_int(document["schema_version"], BUNDLE_REOPEN_RECEIPT_VERSION)
        and document["kind"] == BUNDLE_REOPEN_RECEIPT_KIND
        and document["stage"] == expected_stage
        and expected_stage in {"pre_acknowledgment", "final"},
        "bundle reopen receipt identity or stage differs",
    )
    observed_at = _timestamp(document["observed_at"], "bundle reopen observation time")
    canonical_observed_at = (
        observed_at.astimezone(dt.UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    _require(
        document["observed_at"] == canonical_observed_at,
        "bundle reopen observation time is not canonical UTC",
    )
    bundles = []
    for label in ("source_bundle", "reopened_bundle"):
        bundle = document[label]
        _exact_keys(
            bundle,
            {"inventory_root", "index", "path_identity"},
            f"bundle reopen {label}",
        )
        _exact_keys(
            bundle["index"],
            {"sha256", "size_bytes"},
            f"bundle reopen {label} index",
        )
        _exact_keys(
            bundle["path_identity"],
            {"path_sha256", "device", "inode"},
            f"bundle reopen {label} path identity",
        )
        _require(
            isinstance(bundle["inventory_root"], str)
            and SHA256_RE.fullmatch(bundle["inventory_root"]) is not None
            and isinstance(bundle["index"]["sha256"], str)
            and SHA256_RE.fullmatch(bundle["index"]["sha256"]) is not None
            and type(bundle["index"]["size_bytes"]) is int
            and 0 < bundle["index"]["size_bytes"] <= MAX_INDEX_BYTES
            and isinstance(bundle["path_identity"]["path_sha256"], str)
            and SHA256_RE.fullmatch(bundle["path_identity"]["path_sha256"]) is not None
            and type(bundle["path_identity"]["device"]) is int
            and bundle["path_identity"]["device"] >= 0
            and type(bundle["path_identity"]["inode"]) is int
            and bundle["path_identity"]["inode"] > 0,
            f"bundle reopen {label} descriptor differs",
        )
        bundles.append(bundle)
    source_bundle, reopened_bundle = bundles
    _require(
        source_bundle["inventory_root"] == reopened_bundle["inventory_root"]
        and source_bundle["index"] == reopened_bundle["index"]
        and source_bundle["path_identity"]["path_sha256"]
        != reopened_bundle["path_identity"]["path_sha256"]
        and (
            source_bundle["path_identity"]["device"],
            source_bundle["path_identity"]["inode"],
        )
        != (
            reopened_bundle["path_identity"]["device"],
            reopened_bundle["path_identity"]["inode"],
        ),
        "bundle reopen source/reopened identity or inventory differs",
    )
    payloads = document["registered_payloads"]
    _require(
        isinstance(payloads, list) and bool(payloads),
        "bundle reopen registered payload closure differs",
    )
    validated_payloads = [
        _validate_descriptor(
            item,
            context["candidate_evidence"],
            f"bundle reopen registered payload {ordinal}",
        )
        for ordinal, item in enumerate(payloads)
    ]
    payload_paths = [item["path"] for item in validated_payloads]
    _require(
        payloads == validated_payloads
        and payload_paths == sorted(payload_paths)
        and len(payload_paths) == len(set(payload_paths)),
        "bundle reopen registered payload order or uniqueness differs",
    )
    issue_response = document["issue_response"]
    _exact_keys(
        issue_response,
        {
            "state",
            "lines",
            "body_sha256",
            "updated_at",
            "checklist_transition_sha256",
            "canonical_response_sha256",
            "descriptor",
        },
        "bundle reopen issue response",
    )
    expected_state = (
        "unchecked" if expected_stage == "pre_acknowledgment" else "checked"
    )
    checked_lines = (
        "- [x] publish the final bundle",
        "- [x] complete the post-bundle checklist",
    )
    expected_lines = (
        tuple(line.replace("[x]", "[ ]") for line in checked_lines)
        if expected_state == "unchecked"
        else checked_lines
    )
    issue_descriptor = _validate_descriptor(
        issue_response["descriptor"],
        context["candidate_evidence"],
        "bundle reopen issue response descriptor",
    )
    issue_updated_at = _timestamp(
        issue_response["updated_at"], "bundle reopen issue update time"
    )
    _require(
        issue_response["state"] == expected_state
        and issue_response["lines"] == list(expected_lines)
        and isinstance(issue_response["body_sha256"], str)
        and SHA256_RE.fullmatch(issue_response["body_sha256"]) is not None
        and isinstance(issue_response["checklist_transition_sha256"], str)
        and SHA256_RE.fullmatch(issue_response["checklist_transition_sha256"])
        is not None
        and isinstance(issue_response["canonical_response_sha256"], str)
        and SHA256_RE.fullmatch(issue_response["canonical_response_sha256"]) is not None
        and issue_response["descriptor"] == issue_descriptor
        and issue_descriptor in validated_payloads
        and issue_updated_at <= observed_at,
        "bundle reopen issue response authority differs",
    )
    previous_digest = document["pre_acknowledgment_receipt_sha256"]
    _require(
        (expected_stage == "pre_acknowledgment" and previous_digest is None)
        or (
            expected_stage == "final"
            and isinstance(previous_digest, str)
            and SHA256_RE.fullmatch(previous_digest) is not None
        ),
        "bundle reopen predecessor receipt binding differs",
    )
    expected_technical = {
        key: context[key]
        for key in (
            "technical_input_root",
            "public_projection_sha256",
            "public_asset_ledger_sha256",
        )
    }
    _require(
        document["technical_binding"] == expected_technical,
        "bundle reopen technical binding differs",
    )
    body = {key: value for key, value in document.items() if key != "binding"}
    binding = document["binding"]
    _exact_keys(binding, {"algorithm", "domain", "value"}, "bundle reopen binding")
    expected_hmac = _private_binding_hmac(
        openings.salt_for_private_verification(), PRIVATE_BUNDLE_BINDING_DOMAIN, body
    )
    _require(
        binding["algorithm"] == "HMAC-SHA-256"
        and binding["domain"] == PRIVATE_BUNDLE_BINDING_DOMAIN
        and isinstance(binding["value"], str)
        and hmac.compare_digest(binding["value"], expected_hmac),
        "bundle reopen receipt authentication differs",
    )
    return document


def record_bundle_reopen(
    *,
    source_index: Path | str,
    reopened_index: Path | str,
    stage: str,
    protected_openings: Path | str,
    pre_ack_response: Path | str | None = None,
    pre_ack_receipt: Path | str | None = None,
    output: Path | str | None = None,
) -> bytes:
    """Record a distinct, complete B0/B1 reopen without self-reference."""

    privacy, _publication = _private_binding_modules()
    if stage == "pre-acknowledgment":
        stage = "pre_acknowledgment"
    _require(stage in {"pre_acknowledgment", "final"}, "bundle reopen stage differs")
    openings, context = privacy.load_private_openings(protected_openings)
    source = _bundle_inventory_from_context(source_index, context)
    reopened = _bundle_inventory_from_context(reopened_index, context)
    _require(
        source["root"] == reopened["root"]
        and source["inventory"] == reopened["inventory"],
        "source and reopened bundle inventories differ",
    )
    _require(
        source["index_path"].parent != reopened["index_path"].parent
        and (source["index_path"].stat().st_dev, source["index_path"].stat().st_ino)
        != (reopened["index_path"].stat().st_dev, reopened["index_path"].stat().st_ino),
        "source and reopened bundles are not distinct path/inode copies",
    )
    source_ids = {(device, inode) for _path, device, inode in source["file_identities"]}
    reopened_ids = {
        (device, inode) for _path, device, inode in reopened["file_identities"]
    }
    _require(
        source_ids.isdisjoint(reopened_ids),
        "source and reopened bundle payloads share file identities",
    )
    issue, issue_raw, issue_descriptor = _operations_issue_response(
        reopened["index_path"],
        _load_inventory_index(reopened["index_path"])[2],
    )
    expected_state = "unchecked" if stage == "pre_acknowledgment" else "checked"
    issue_observation = {
        **_checklist_observation(issue, expected_state),
        "canonical_response_sha256": hashlib.sha256(issue_raw).hexdigest(),
        "descriptor": issue_descriptor,
    }
    if stage == "pre_acknowledgment":
        _require(
            pre_ack_response is not None and pre_ack_receipt is None,
            "pre-acknowledgment inputs differ",
        )
        _path, supplied_response = _bounded_regular_file_bytes(
            Path(pre_ack_response),
            "pre-acknowledgment issue response",
            max_bytes=MAX_ARTIFACT_BYTES,
        )
        _require(
            hmac.compare_digest(supplied_response, issue_raw),
            "pre-acknowledgment issue response differs from reopened B0",
        )
        previous_digest = None
    else:
        _require(
            pre_ack_receipt is not None and pre_ack_response is None,
            "final reopen inputs differ",
        )
        previous = validate_bundle_reopen_receipt(
            pre_ack_receipt, "pre_acknowledgment", protected_openings
        )
        _previous_document, previous_raw = _load_reopen_receipt(pre_ack_receipt)
        _require(
            previous["issue_response"]["state"] == "unchecked"
            and previous["issue_response"]["canonical_response_sha256"]
            != issue_observation["canonical_response_sha256"]
            and previous["issue_response"]["checklist_transition_sha256"]
            == issue_observation["checklist_transition_sha256"]
            and _timestamp(issue_observation["updated_at"], "final checklist update")
            >= _timestamp(previous["observed_at"], "pre-acknowledgment reopen time"),
            "final checklist recapture does not follow the B0 acknowledgment",
        )
        previous_digest = _reopen_receipt_file_sha256(previous_raw)
    observed_at = (
        dt.datetime.now(dt.UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    body = {
        "schema_version": BUNDLE_REOPEN_RECEIPT_VERSION,
        "kind": BUNDLE_REOPEN_RECEIPT_KIND,
        "stage": stage,
        "observed_at": observed_at,
        "source_bundle": {
            "inventory_root": source["root"],
            "index": source["inventory"]["index"],
            "path_identity": _bundle_path_identity(source["index_path"].parent),
        },
        "reopened_bundle": {
            "inventory_root": reopened["root"],
            "index": reopened["inventory"]["index"],
            "path_identity": _bundle_path_identity(reopened["index_path"].parent),
        },
        "registered_payloads": copy.deepcopy(reopened["inventory"]["payloads"]),
        "technical_binding": {
            key: context[key]
            for key in (
                "technical_input_root",
                "public_projection_sha256",
                "public_asset_ledger_sha256",
            )
        },
        "issue_response": issue_observation,
        "pre_acknowledgment_receipt_sha256": previous_digest,
    }
    document = {
        **body,
        "binding": {
            "algorithm": "HMAC-SHA-256",
            "domain": PRIVATE_BUNDLE_BINDING_DOMAIN,
            "value": _private_binding_hmac(
                openings.salt_for_private_verification(),
                PRIVATE_BUNDLE_BINDING_DOMAIN,
                body,
            ),
        },
    }
    raw = privacy.binding_canonical_json_bytes(document)
    validate_bundle_reopen_receipt(raw, stage, protected_openings)
    if output is not None:
        output_path = Path(output)
        _write_exclusive_private_file(
            output_path,
            raw,
            "bundle reopen receipt",
            forbidden_roots=(
                source["index_path"].parent,
                reopened["index_path"].parent,
            ),
        )
    return raw


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
        path_independent_keys = {"path", "sha256", "size_bytes", "media_type"}
        if set(value) == path_independent_keys:
            _canonical_bundle_path(value["path"], f"{label} path")
            _require(
                isinstance(value["sha256"], str)
                and SHA256_RE.fullmatch(value["sha256"]) is not None
                and type(value["size_bytes"]) is int
                and 0 < value["size_bytes"] <= MAX_RUNTIME_RECEIPT_BYTES
                and value["media_type"] == MEDIA_TYPE_JSON,
                f"{label} path-independent descriptor differs",
            )
            registered = registry.get(value["path"])
            _require(
                isinstance(registered, dict)
                and all(registered.get(name) == value[name] for name in value),
                f"{label} embedded path-independent descriptor is not registered",
            )
            used.add(value["path"])
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
    messages = {
        "evidence-resource-limit": "evidence resource limit exceeded",
        "evidence-io-error": "evidence input/output validation failed",
        "invalid-evidence": "evidence validation failed closed",
        "evidence-validation-error": "evidence validation failed closed",
    }
    return {
        "code": code,
        "phase": phase,
        "scope": scope,
        "message": messages[code],
    }


def evaluate_completion(
    index_path: Path | str,
    manifest_path: Path | str | None = None,
    runtime_receipt_paths: list[Path | str] | tuple[Path | str, ...] | None = None,
    *,
    descriptor_access_log: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate issue #123 evidence offline without granting final authority."""

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
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "kind": OUTPUT_KIND,
        "issue": 123,
        "evaluation_mode": OFFLINE_EVALUATION_MODE,
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
        "structural_validation_satisfied": False,
        "final_acceptance": False,
        "final_acceptance_authority": OFFLINE_AUTHORITY,
        "receipt_replay_authority": False,
        "candidate_bundle_binding": {
            "satisfied": False,
            "technical_input_root": None,
            "public_projection_sha256": None,
            "public_asset_ledger_sha256": None,
            "final_bundle_inventory_root": None,
        },
        "post_bundle": {
            "satisfied": False,
            "pre_acknowledgment_receipt": None,
            "final_reopen_receipt": None,
            "acknowledgment": None,
        },
        "baseline_authority": {
            "satisfied": False,
            "mode": None,
            "authority_sha256": None,
        },
        "live_verification": {
            "invocation_attempted": False,
            "invocation_succeeded": False,
            "authority": None,
            "verified_at": None,
            "operations_index": None,
            "receipt": None,
            "errors": [],
        },
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
        reader = ArtifactReader(
            index_path.parent,
            candidate,
            payloads,
            descriptor_access_log=descriptor_access_log,
        )
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
        _trusted_path, trusted_manifest_raw, trusted_manifest = (
            _trusted_manifest_bytes()
        )
        _require(
            manifest_raw == trusted_manifest_raw
            and manifest_artifact.descriptor["sha256"] == TRUSTED_MANIFEST_SHA256
            and _type_exact_equal(manifest, trusted_manifest),
            "bundled manifest bytes differ from the trusted repository manifest",
        )
        if manifest_path is not None:
            _supplied_manifest_path, supplied_manifest_raw = (
                _bounded_regular_file_bytes(
                    Path(manifest_path),
                    "supplied manifest",
                    max_bytes=MAX_MANIFEST_BYTES,
                )
            )
            _require(
                supplied_manifest_raw == trusted_manifest_raw,
                "supplied manifest bytes differ from the trusted repository manifest",
            )
        output["manifest"].update(
            path=manifest_artifact.descriptor["path"],
            size_bytes=len(manifest_raw),
            sha256=_sha256(manifest_raw),
        )
        output["candidate_evidence"] = candidate
        trusted_runtime_receipts = _load_trusted_runtime_receipts(
            runtime_receipt_paths, manifest, candidate, reader.base
        )
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
            trusted_runtime_receipts[0],
        ),
        "policy_paired_real": lambda: _validate_policy_scope(
            artifacts["policy_paired_real"], reader, manifest, candidate
        ),
        "single_gpu": lambda: _validate_single_gpu_scope(
            artifacts["single_gpu"],
            reader,
            manifest,
            candidate,
            trusted_runtime_receipts[1:3],
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
            correctness_topology = _validate_global_correctness_archive_topology(
                cpu["details"], single["details"], manifest
            )
        except Exception as error:
            record = _structured_evidence_error(
                error,
                phase="correctness-archive-topology",
            )
            output["cross_scope_errors"].append(record)
            for scoped in (cpu, single):
                scoped["satisfied"] = False
                scoped["errors"].append(record)
        else:
            output["cross_scope_details"][
                "correctness_archive_topology"
            ] = correctness_topology
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

    if cpu["satisfied"] and policy["satisfied"] and single["satisfied"]:
        try:
            source_bindings = _validate_differential_correctness_source_bindings(
                cpu["details"],
                policy["details"],
                single["details"],
                manifest,
                trusted_runtime_receipts[3:],
            )
        except Exception as error:
            record = _structured_evidence_error(
                error,
                phase="differential-runtime-binding",
            )
            output["cross_scope_errors"].append(record)
            for scoped in (cpu, policy, single):
                scoped["satisfied"] = False
                scoped["errors"].append(record)
        else:
            output["cross_scope_details"][
                "differential_runtime_bindings"
            ] = source_bindings

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
    for scoped, key in (
        (cpu, "_correctness_archive_bindings"),
        (single, "_correctness_archive_bindings_by_mode"),
    ):
        details = scoped.get("details")
        if isinstance(details, dict):
            details.pop(key, None)
    output["structural_validation_satisfied"] = not output[
        "cross_scope_errors"
    ] and all(output["scopes"][name]["satisfied"] for name in scope_names)
    return output


def _create_private_live_directory(
    output_directory: Path | str,
    forbidden_roots: tuple[Path, ...],
) -> Path:
    try:
        supplied = Path(output_directory)
    except TypeError as error:
        raise EvidenceError("live output directory path is invalid") from error
    _require(
        supplied.name not in {"", ".", ".."}
        and all(part not in {".", ".."} for part in supplied.parts),
        "live output directory name is invalid",
    )
    _require(
        not supplied.exists() and not supplied.is_symlink(),
        "live output directory already exists",
    )
    try:
        from benchmarks import issue123_privacy as privacy

        destination, _preflight_parent = privacy.preflight_private_output_path(
            supplied,
            label="live output directory",
            forbidden_roots=forbidden_roots,
        )
    except (ImportError, OSError, TypeError, ValueError):
        failure = EvidenceError("live output directory overlaps protected evidence")
    else:
        failure = None
    if failure is not None:
        raise failure from None
    parent = _ensure_directory_without_symlinks(
        supplied.parent,
        "live output parent",
    )
    _require(
        destination == parent / supplied.name,
        "live output directory identity differs",
    )
    try:
        destination.mkdir(mode=0o700)
        os.chmod(destination, 0o700)
        metadata = destination.lstat()
    except OSError as error:
        raise EvidenceError("live output directory could not be created") from error
    _require(
        stat.S_ISDIR(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and stat.S_IMODE(metadata.st_mode) == 0o700,
        "live output directory is not a private mode-0700 directory",
    )
    return destination.resolve(strict=True)


def _create_private_subdirectory(parent: Path, name: str) -> Path:
    _require(
        re.fullmatch(r"[a-z0-9][a-z0-9._-]*", name) is not None,
        "private subdirectory name is invalid",
    )
    destination = parent / name
    _require(
        not destination.exists() and not destination.is_symlink(),
        f"private subdirectory {name} already exists",
    )
    try:
        destination.mkdir(mode=0o700)
        os.chmod(destination, 0o700)
        metadata = destination.lstat()
    except OSError as error:
        raise EvidenceError(
            f"private subdirectory {name} could not be created"
        ) from error
    _require(
        stat.S_ISDIR(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and stat.S_IMODE(metadata.st_mode) == 0o700,
        f"private subdirectory {name} is not mode 0700",
    )
    return destination


def _ensure_private_relative_parent(root: Path, relative: PurePosixPath) -> Path:
    current = root
    for part in relative.parts:
        current = current / part
        try:
            current.mkdir(mode=0o700)
        except FileExistsError:
            pass
        except OSError as error:
            raise EvidenceError(
                "private staging directory could not be created"
            ) from error
        try:
            metadata = current.lstat()
        except OSError as error:
            raise EvidenceError("private staging directory is unreadable") from error
        _require(
            stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
            "private staging path is not a directory",
        )
        try:
            os.chmod(current, 0o700)
        except OSError as error:
            raise EvidenceError(
                "private staging directory mode could not be set"
            ) from error
    return current


def _write_exclusive_private_file(
    path: Path,
    raw: bytes,
    label: str,
    *,
    forbidden_roots: tuple[Path, ...] = (),
    before_commit: Callable[[], None] | None = None,
) -> Path:
    try:
        from benchmarks import issue123_privacy as privacy
    except ImportError:
        raise EvidenceError("private authority file could not be committed") from None
    try:
        return privacy.write_private_authority_file(
            path,
            raw,
            label=label,
            forbidden_roots=forbidden_roots,
            before_commit=before_commit,
        )
    except privacy.PrivateAuthorityCommitError:
        raise CommittedAuthorityError(
            "private authority file was committed but final verification failed"
        ) from None
    except (OSError, TypeError, ValueError):
        raise EvidenceError("private authority file could not be committed") from None


def _prepare_operations_live_input(
    index_snapshot: FileSnapshot,
    offline_result: dict[str, Any],
    destination: Path,
    forbidden_roots: tuple[Path, ...],
) -> tuple[Path, dict[str, Any], StagedOperationsSnapshots]:
    current_index, index_raw = _snapshot_regular_file(
        index_snapshot.path,
        "completion evidence index",
        max_bytes=MAX_INDEX_BYTES,
    )
    _require(
        current_index == index_snapshot
        and offline_result.get("evidence_index")
        == {
            "path": index_snapshot.path.name,
            "size_bytes": index_snapshot.size_bytes,
            "sha256": index_snapshot.sha256,
        },
        "completion index changed after structural validation",
    )
    index = _strict_json_bytes(
        index_raw,
        "completion evidence index",
        max_bytes=MAX_INDEX_BYTES,
    )
    _require(isinstance(index, dict), "completion evidence index differs")
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
    candidate = offline_result.get("candidate_evidence")
    _require(
        isinstance(candidate, dict) and index["candidate_evidence"] == candidate,
        "completion candidate changed after structural validation",
    )
    payloads = index["payloads"]
    _require(isinstance(payloads, list), "payload registry must be a list")
    reader = ArtifactReader(index_snapshot.path.parent, candidate, payloads)
    manifest_artifact = reader.load(index["manifest"], "bundled manifest")
    _require(
        offline_result.get("manifest")
        == {
            "path": manifest_artifact.descriptor["path"],
            "size_bytes": len(manifest_artifact.raw),
            "sha256": _sha256(manifest_artifact.raw),
        },
        "bundled manifest changed after structural validation",
    )
    artifacts = index["artifacts"]
    _require(isinstance(artifacts, dict), "completion evidence scopes differ")
    structural_operations = _validate_operations_scope(
        artifacts.get("operations"), reader, candidate
    )
    _require(
        _type_exact_equal(
            structural_operations,
            offline_result.get("scopes", {}).get("operations", {}).get("details"),
        ),
        "operations result changed after structural validation",
    )
    operations_index, response_artifacts = _load_operations_scope_artifacts(
        artifacts["operations"], reader, candidate
    )
    _require(
        len(response_artifacts) == LIVE_OPERATIONS_RESPONSE_COUNT,
        "operations staging does not contain exactly 22 responses",
    )
    operations_descriptor = {
        "size_bytes": len(operations_index.raw),
        "sha256": _sha256(operations_index.raw),
    }
    _require(
        operations_descriptor
        == {
            "size_bytes": operations_index.descriptor["size_bytes"],
            "sha256": operations_index.descriptor["sha256"],
        },
        "operations index descriptor differs",
    )
    staging = _create_private_subdirectory(destination, LIVE_OPERATIONS_DIRECTORY)
    response_snapshots = []
    for role, artifact in sorted(response_artifacts.items()):
        relative = _canonical_bundle_path(
            artifact.descriptor["path"],
            f"operations raw response {role} staging path",
        )
        _require(
            relative.as_posix() != "operations-index.json",
            "operations response collides with the staged index",
        )
        parent = _ensure_private_relative_parent(staging, relative.parent)
        response_path = parent / relative.name
        _write_exclusive_private_file(
            response_path,
            artifact.raw,
            f"staged operations raw response {role}",
            forbidden_roots=forbidden_roots,
        )
        response_snapshot, response_raw = _snapshot_regular_file(
            response_path,
            f"staged operations raw response {role}",
            max_bytes=MAX_ARTIFACT_BYTES,
        )
        _require(
            response_raw == artifact.raw
            and response_snapshot.size_bytes == artifact.descriptor["size_bytes"]
            and response_snapshot.sha256 == artifact.descriptor["sha256"],
            f"staged operations raw response {role} differs after creation",
        )
        response_snapshots.append((role, response_snapshot))
    staged_index = staging / "operations-index.json"
    _write_exclusive_private_file(
        staged_index,
        operations_index.raw,
        "staged operations evidence index",
        forbidden_roots=forbidden_roots,
    )
    staged_index_snapshot, staged_index_raw = _snapshot_regular_file(
        staged_index,
        "staged operations evidence index",
        max_bytes=MAX_JSON_BYTES,
    )
    _require(
        staged_index_raw == operations_index.raw
        and staged_index_snapshot.size_bytes == operations_descriptor["size_bytes"]
        and staged_index_snapshot.sha256 == operations_descriptor["sha256"],
        "staged operations evidence index differs after creation",
    )
    snapshots = StagedOperationsSnapshots(
        index=staged_index_snapshot,
        responses=tuple(response_snapshots),
    )
    all_snapshots = (snapshots.index, *(item for _role, item in snapshots.responses))
    _require(
        len(snapshots.responses) == LIVE_OPERATIONS_RESPONSE_COUNT
        and len({item.path for item in all_snapshots}) == len(all_snapshots)
        and len({item.identity[:2] for item in all_snapshots}) == len(all_snapshots),
        "staged operations inputs are not 23 distinct files",
    )
    return staged_index, operations_descriptor, snapshots


def _require_staged_operations_inputs_unchanged(
    snapshots: StagedOperationsSnapshots,
) -> None:
    _require(
        isinstance(snapshots, StagedOperationsSnapshots)
        and len(snapshots.responses) == LIVE_OPERATIONS_RESPONSE_COUNT,
        "staged operations snapshot closure differs",
    )
    _require_snapshot_unchanged(
        snapshots.index,
        "staged operations evidence index",
        max_bytes=MAX_JSON_BYTES,
    )
    for role, snapshot in snapshots.responses:
        _require_snapshot_unchanged(
            snapshot,
            f"staged operations raw response {role}",
            max_bytes=MAX_ARTIFACT_BYTES,
        )


def _retained_file_identity(
    metadata: os.stat_result,
) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _retained_directory_identity(
    metadata: os.stat_result,
) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_mtime_ns,
    )


def _read_retained_file(
    descriptor: int,
    expected_size: int,
    maximum: int,
    *,
    changed_message: str,
) -> bytes:
    _require(
        0 <= expected_size <= maximum,
        "live B1 registered file closure differs",
    )
    chunks = []
    offset = 0
    while offset < expected_size:
        try:
            chunk = os.pread(
                descriptor,
                min(1024 * 1024, expected_size - offset),
                offset,
            )
        except OSError:
            raise EvidenceError(changed_message) from None
        _require(bool(chunk), changed_message)
        chunks.append(chunk)
        offset += len(chunk)
    try:
        extra = os.pread(descriptor, 1, expected_size)
    except OSError:
        raise EvidenceError(changed_message) from None
    _require(not extra, changed_message)
    return b"".join(chunks)


def _enumerate_retained_tree(
    root_fd: int,
) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()

    def walk(directory_fd: int, prefix: PurePosixPath) -> None:
        try:
            with os.scandir(directory_fd) as iterator:
                entries = sorted(iterator, key=lambda item: item.name)
        except OSError:
            raise EvidenceError(
                "live B1 directory closure changed during verification"
            ) from None
        for entry in entries:
            relative = (
                PurePosixPath(entry.name)
                if prefix == PurePosixPath(".")
                else prefix / entry.name
            )
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError:
                raise EvidenceError(
                    "live B1 directory closure changed during verification"
                ) from None
            if stat.S_ISLNK(metadata.st_mode):
                raise EvidenceError(
                    "live B1 directory closure changed during verification"
                )
            if stat.S_ISDIR(metadata.st_mode):
                directories.add(relative.as_posix())
                try:
                    child_fd = os.open(
                        entry.name,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                        dir_fd=directory_fd,
                    )
                except OSError:
                    raise EvidenceError(
                        "live B1 directory closure changed during verification"
                    ) from None
                try:
                    walk(child_fd, relative)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(metadata.st_mode):
                files.add(relative.as_posix())
            else:
                raise EvidenceError(
                    "live B1 directory closure changed during verification"
                )

    walk(root_fd, PurePosixPath("."))
    return files, directories


def _capture_retained_bundle_tree(
    index_path: Path | str,
    label: str,
    binding_context: Mapping[str, Any],
) -> RetainedBundleTree:
    opened: list[int] = []
    try:
        checked_index, metadata = _path_without_symlinks(Path(index_path), label)
        _require(
            checked_index.name == "completion-index.json"
            and stat.S_ISREG(metadata.st_mode),
            "live B1 registered file closure differs",
        )
        root = checked_index.parent
        root_fd = os.open(
            root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        opened.append(root_fd)
        root_metadata = os.fstat(root_fd)
        root_directory = RetainedBundleDirectory(
            PurePosixPath("."),
            root_fd,
            _retained_directory_identity(root_metadata),
        )
        index_fd = os.open(
            "completion-index.json",
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=root_fd,
        )
        opened.append(index_fd)
        index_metadata = os.fstat(index_fd)
        _require(
            stat.S_ISREG(index_metadata.st_mode)
            and 0 < index_metadata.st_size <= MAX_INDEX_BYTES,
            "live B1 registered file closure differs",
        )
        index_raw = _read_retained_file(
            index_fd,
            index_metadata.st_size,
            MAX_INDEX_BYTES,
            changed_message="live B1 registered file bytes changed during verification",
        )
        document = _strict_json_bytes(index_raw, label, max_bytes=MAX_INDEX_BYTES)
        _exact_keys(
            document,
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
            label,
        )
        _require(
            _is_exact_int(document["schema_version"], INDEX_SCHEMA_VERSION)
            and document["kind"] == INDEX_KIND
            and _is_exact_int(document["issue"], 123),
            "live B1 registered file closure differs",
        )
        _exact_keys(
            document["bundle"],
            {"format", "path_contract", "artifact_count", "artifact_bytes"},
            "live B1 bundle contract",
        )
        _exact_keys(
            document["artifacts"],
            {
                "cpu",
                "policy_paired_real",
                "single_gpu",
                "two_gpu",
                "macos",
                "operations",
            },
            "live B1 scope mapping",
        )
        payload_values = document["payloads"]
        _require(
            isinstance(payload_values, list)
            and len(payload_values) <= MAX_RETAINED_BUNDLE_ENTRIES,
            "live B1 registered file closure differs",
        )
        candidate = _candidate_evidence(
            document["candidate_evidence"],
            (
                document["candidate_evidence"].get("manifest_sha256")
                if isinstance(document["candidate_evidence"], Mapping)
                else None
            ),
        )
        ledger = [
            _validate_descriptor(item, candidate, f"{label} payload {ordinal}")
            for ordinal, item in enumerate(payload_values)
        ]
        paths = [item["path"] for item in ledger]
        bundle = document["bundle"]
        _require(
            paths == sorted(paths)
            and len(paths) == len(set(paths))
            and bundle["format"] == BUNDLE_FORMAT
            and bundle["path_contract"] == PATH_CONTRACT
            and _is_exact_int(bundle["artifact_count"], len(ledger))
            and type(bundle["artifact_bytes"]) is int
            and bundle["artifact_bytes"] == sum(item["size_bytes"] for item in ledger),
            "live B1 registered file closure differs",
        )
        expected_files = {"completion-index.json", *paths}
        expected_directories: set[str] = set()
        for value in paths:
            parent = PurePosixPath(value).parent
            while parent != PurePosixPath("."):
                expected_directories.add(parent.as_posix())
                parent = parent.parent
        _require(
            len(expected_files) + len(expected_directories)
            <= MAX_RETAINED_BUNDLE_ENTRIES,
            "live B1 registered file closure differs",
        )
        actual_files, actual_directories = _enumerate_retained_tree(root_fd)
        _require(
            actual_files == expected_files
            and actual_directories == expected_directories,
            "live B1 registered file closure differs",
        )
        directory_by_path = {PurePosixPath("."): root_directory}
        retained_directories = []
        for relative_value in sorted(
            expected_directories, key=lambda value: (value.count("/"), value)
        ):
            relative = PurePosixPath(relative_value)
            parent = directory_by_path[relative.parent]
            directory_fd = os.open(
                relative.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent.fd,
            )
            opened.append(directory_fd)
            directory_metadata = os.fstat(directory_fd)
            retained = RetainedBundleDirectory(
                relative,
                directory_fd,
                _retained_directory_identity(directory_metadata),
            )
            directory_by_path[relative] = retained
            retained_directories.append(retained)
        retained_payloads = []
        file_identities = {(index_metadata.st_dev, index_metadata.st_ino)}
        for descriptor in ledger:
            relative = PurePosixPath(descriptor["path"])
            parent = directory_by_path[relative.parent]
            payload_fd = os.open(
                relative.name,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent.fd,
            )
            opened.append(payload_fd)
            payload_metadata = os.fstat(payload_fd)
            _require(
                stat.S_ISREG(payload_metadata.st_mode)
                and payload_metadata.st_size == descriptor["size_bytes"],
                "live B1 registered file closure differs",
            )
            identity = (payload_metadata.st_dev, payload_metadata.st_ino)
            _require(
                identity not in file_identities,
                "live B1 registered file identity changed during verification",
            )
            file_identities.add(identity)
            raw = _read_retained_file(
                payload_fd,
                descriptor["size_bytes"],
                MAX_ARTIFACT_BYTES,
                changed_message="live B1 registered file bytes changed during verification",
            )
            _require(
                _sha256(raw) == descriptor["sha256"],
                "live B1 registered file bytes changed during verification",
            )
            _validate_media_payload(raw, descriptor["media_type"], label)
            _require(
                _retained_file_identity(os.fstat(payload_fd))
                == _retained_file_identity(payload_metadata),
                "live B1 registered file identity changed during verification",
            )
            retained_payloads.append(
                RetainedBundleFile(
                    relative,
                    _canonical_json_bytes(descriptor),
                    payload_fd,
                    _retained_file_identity(payload_metadata),
                    len(raw),
                    _sha256(raw),
                )
            )
        inventory, inventory_root = _validated_bundle_inventory_value(
            index_raw,
            document,
            ledger,
            binding_context,
        )
        index_file = RetainedBundleFile(
            PurePosixPath("completion-index.json"),
            None,
            index_fd,
            _retained_file_identity(index_metadata),
            len(index_raw),
            _sha256(index_raw),
        )
        expected_entry_types = tuple(
            sorted(
                [(value, "directory") for value in expected_directories]
                + [(value, "file") for value in expected_files]
            )
        )
        return RetainedBundleTree(
            root=root,
            root_directory=root_directory,
            directories=tuple(retained_directories),
            index=index_file,
            payloads=tuple(retained_payloads),
            expected_entry_types=expected_entry_types,
            descriptor_ledger=tuple(_canonical_json_bytes(item) for item in ledger),
            inventory_bytes=_canonical_json_bytes(inventory),
            inventory_root=inventory_root,
            index_semantics=index_raw,
        )
    except Exception:
        for descriptor in reversed(opened):
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise


def _retained_file_snapshot(
    tree: RetainedBundleTree, retained: RetainedBundleFile
) -> FileSnapshot:
    identity = retained.identity
    path = tree.root.joinpath(*retained.relative_path.parts)
    return FileSnapshot(
        path,
        retained.size_bytes,
        retained.sha256,
        (identity[0], identity[1], identity[3], identity[4]),
    )


def _require_retained_bundle_tree_unchanged(tree: RetainedBundleTree) -> None:
    try:
        root_path, root_metadata = _path_without_symlinks(tree.root, "live B1 root")
    except EvidenceError:
        raise EvidenceError(
            "live B1 directory closure changed during verification"
        ) from None
    _require(
        root_path == tree.root
        and _retained_directory_identity(root_metadata) == tree.root_directory.identity
        and _retained_directory_identity(os.fstat(tree.root_directory.fd))
        == tree.root_directory.identity,
        "live B1 directory closure changed during verification",
    )
    directory_by_path = {PurePosixPath("."): tree.root_directory}
    for retained in tree.directories:
        directory_by_path[retained.relative_path] = retained
        _require(
            _retained_directory_identity(os.fstat(retained.fd)) == retained.identity,
            "live B1 directory closure changed during verification",
        )
        parent = directory_by_path[retained.relative_path.parent]
        try:
            current = os.stat(
                retained.relative_path.name,
                dir_fd=parent.fd,
                follow_symlinks=False,
            )
        except OSError:
            raise EvidenceError(
                "live B1 directory closure changed during verification"
            ) from None
        _require(
            _retained_directory_identity(current) == retained.identity,
            "live B1 directory closure changed during verification",
        )
    for retained in (tree.index, *tree.payloads):
        try:
            metadata = os.fstat(retained.fd)
        except OSError:
            raise EvidenceError(
                "live B1 registered file identity changed during verification"
            ) from None
        _require(
            _retained_file_identity(metadata) == retained.identity,
            "live B1 registered file identity changed during verification",
        )
        raw = _read_retained_file(
            retained.fd,
            retained.size_bytes,
            MAX_INDEX_BYTES if retained is tree.index else MAX_ARTIFACT_BYTES,
            changed_message="live B1 registered file bytes changed during verification",
        )
        _require(
            _sha256(raw) == retained.sha256,
            "live B1 registered file bytes changed during verification",
        )
        parent = directory_by_path[retained.relative_path.parent]
        try:
            current = os.stat(
                retained.relative_path.name,
                dir_fd=parent.fd,
                follow_symlinks=False,
            )
        except OSError:
            raise EvidenceError(
                "live B1 registered file identity changed during verification"
            ) from None
        _require(
            (current.st_dev, current.st_ino)
            == (retained.identity[0], retained.identity[1]),
            "live B1 registered file identity changed during verification",
        )
    files, directories = _enumerate_retained_tree(tree.root_directory.fd)
    current_types = tuple(
        sorted(
            [(value, "directory") for value in directories]
            + [(value, "file") for value in files]
        )
    )
    _require(
        current_types == tree.expected_entry_types,
        "live B1 directory closure changed during verification",
    )
    index_raw = _read_retained_file(
        tree.index.fd,
        tree.index.size_bytes,
        MAX_INDEX_BYTES,
        changed_message="live B1 registered file bytes changed during verification",
    )
    index = _strict_json_bytes(
        index_raw, "retained live B1 index", max_bytes=MAX_INDEX_BYTES
    )
    ledger = tuple(
        _canonical_json_bytes(
            _validate_descriptor(
                item,
                index["candidate_evidence"],
                f"retained live B1 payload {ordinal}",
            )
        )
        for ordinal, item in enumerate(index["payloads"])
    )
    _require(
        index_raw == tree.index_semantics and ledger == tree.descriptor_ledger,
        "live B1 registered file closure differs",
    )


def _close_retained_bundle_tree(tree: RetainedBundleTree) -> None:
    failed = False
    owners = [
        *tree.payloads,
        tree.index,
        *reversed(tree.directories),
        tree.root_directory,
    ]
    for owner in owners:
        descriptor = owner.fd
        if descriptor < 0:
            continue
        object.__setattr__(owner, "fd", -1)
        try:
            os.close(descriptor)
        except OSError:
            failed = True
    if failed:
        raise EvidenceError("live B1 retained descriptors could not be closed")


def _close_retained_bundle_trees(
    trees: tuple[RetainedBundleTree, ...],
    *,
    primary_error: BaseException | None = None,
) -> None:
    """Attempt every owned-tree close once without masking a primary failure."""

    cleanup_failed = False
    for tree in trees:
        try:
            _close_retained_bundle_tree(tree)
        except EvidenceError:
            cleanup_failed = True
    if cleanup_failed:
        if primary_error is not None:
            return
        raise EvidenceError(
            "live B1 retained descriptors could not be closed"
        ) from None


def _capture_detached_live_inputs(
    manifest_path: Path | str | None,
    runtime_receipt_paths: Any,
    pre_acknowledgment_receipt: Path | str,
    final_reopen_receipt: Path | str,
) -> tuple[FileSnapshot, tuple[FileSnapshot, ...], FileSnapshot, FileSnapshot]:
    manifest_snapshot, _manifest_raw = _snapshot_regular_file(
        DEFAULT_MANIFEST if manifest_path is None else manifest_path,
        "trusted completion manifest",
        max_bytes=MAX_MANIFEST_BYTES,
    )
    runtime_snapshots = tuple(
        _snapshot_regular_file(
            path,
            f"trusted {role} runtime receipt",
            max_bytes=MAX_RUNTIME_RECEIPT_BYTES,
        )[0]
        for role, path in zip(
            RUNTIME_RECEIPT_ROLES,
            runtime_receipt_paths,
            strict=True,
        )
    )
    pre_ack_snapshot, _pre_ack_raw = _snapshot_regular_file(
        pre_acknowledgment_receipt,
        "pre-acknowledgment bundle reopen receipt",
        max_bytes=MAX_JSON_BYTES,
    )
    final_reopen_snapshot, _final_reopen_raw = _snapshot_regular_file(
        final_reopen_receipt,
        "final bundle reopen receipt",
        max_bytes=MAX_JSON_BYTES,
    )
    return (
        manifest_snapshot,
        runtime_snapshots,
        pre_ack_snapshot,
        final_reopen_snapshot,
    )


def _capture_core_live_inputs(
    index_path: Path | str,
    reopened_index: Path | str,
    manifest_path: Path | str | None,
    runtime_receipt_paths: Any,
    protected_openings: Path | str,
    pre_acknowledgment_receipt: Path | str,
    final_reopen_receipt: Path | str,
) -> LiveAuthoritySnapshots:
    _require(
        isinstance(runtime_receipt_paths, (list, tuple))
        and len(runtime_receipt_paths) == len(RUNTIME_RECEIPT_ROLES),
        "exactly five trusted runtime receipt paths are required",
    )
    openings_snapshot, _openings_raw = _snapshot_regular_file(
        protected_openings,
        "protected publication openings",
        max_bytes=MAX_JSON_BYTES,
    )
    privacy, _publication = _private_binding_modules()
    _openings, binding_context = privacy.load_private_openings(openings_snapshot.path)
    source_bundle = _capture_retained_bundle_tree(
        index_path,
        "source live B1 index",
        binding_context,
    )
    reopened_bundle: RetainedBundleTree | None = None
    try:
        reopened_bundle = _capture_retained_bundle_tree(
            reopened_index,
            "reopened live B1 index",
            binding_context,
        )
        index_snapshot = _retained_file_snapshot(source_bundle, source_bundle.index)
        reopened_snapshot = _retained_file_snapshot(
            reopened_bundle, reopened_bundle.index
        )
        (
            manifest_snapshot,
            runtime_snapshots,
            pre_ack_snapshot,
            final_reopen_snapshot,
        ) = _capture_detached_live_inputs(
            manifest_path,
            runtime_receipt_paths,
            pre_acknowledgment_receipt,
            final_reopen_receipt,
        )
        source_root = source_bundle.root
        reopened_root = reopened_bundle.root
        authority_snapshots = (
            index_snapshot,
            reopened_snapshot,
            manifest_snapshot,
            *runtime_snapshots,
            openings_snapshot,
            pre_ack_snapshot,
            final_reopen_snapshot,
        )
        source_identities = {
            source_bundle.root_directory.identity[:2],
            *(item.identity[:2] for item in source_bundle.directories),
            source_bundle.index.identity[:2],
            *(item.identity[:2] for item in source_bundle.payloads),
        }
        reopened_identities = {
            reopened_bundle.root_directory.identity[:2],
            *(item.identity[:2] for item in reopened_bundle.directories),
            reopened_bundle.index.identity[:2],
            *(item.identity[:2] for item in reopened_bundle.payloads),
        }
        _require(
            source_root != reopened_root
            and source_bundle.root_directory.identity[:2]
            != reopened_bundle.root_directory.identity[:2]
            and index_snapshot.sha256 == reopened_snapshot.sha256
            and index_snapshot.size_bytes == reopened_snapshot.size_bytes
            and source_bundle.descriptor_ledger == reopened_bundle.descriptor_ledger
            and source_bundle.inventory_bytes == reopened_bundle.inventory_bytes
            and source_bundle.inventory_root == reopened_bundle.inventory_root
            and source_bundle.expected_entry_types
            == reopened_bundle.expected_entry_types
            and source_identities.isdisjoint(reopened_identities)
            and len({item.path for item in authority_snapshots})
            == len(authority_snapshots)
            and len({item.identity[:2] for item in authority_snapshots})
            == len(authority_snapshots)
            and len({item.sha256 for item in runtime_snapshots})
            == len(runtime_snapshots)
            and all(
                not snapshot.path.is_relative_to(source_root)
                and not snapshot.path.is_relative_to(reopened_root)
                for snapshot in (
                    manifest_snapshot,
                    *runtime_snapshots,
                    openings_snapshot,
                    pre_ack_snapshot,
                    final_reopen_snapshot,
                )
            ),
            "source and reopened live B1 snapshot closure differs",
        )
        return LiveAuthoritySnapshots(
            source_bundle=source_bundle,
            reopened_bundle=reopened_bundle,
            manifest=manifest_snapshot,
            runtime_receipts=runtime_snapshots,
            protected_openings=openings_snapshot,
            pre_acknowledgment_receipt=pre_ack_snapshot,
            final_reopen_receipt=final_reopen_snapshot,
        )
    except BaseException as error:
        _close_retained_bundle_trees(
            tuple(
                tree for tree in (reopened_bundle, source_bundle) if tree is not None
            ),
            primary_error=error,
        )
        raise


def _require_core_live_inputs_unchanged(
    snapshots: LiveAuthoritySnapshots,
) -> None:
    _require_retained_bundle_tree_unchanged(snapshots.source_bundle)
    _require_retained_bundle_tree_unchanged(snapshots.reopened_bundle)
    _require_snapshot_unchanged(
        snapshots.manifest,
        "trusted completion manifest",
        max_bytes=MAX_MANIFEST_BYTES,
    )
    for role, snapshot in zip(
        RUNTIME_RECEIPT_ROLES,
        snapshots.runtime_receipts,
        strict=True,
    ):
        _require_snapshot_unchanged(
            snapshot,
            f"trusted {role} runtime receipt",
            max_bytes=MAX_RUNTIME_RECEIPT_BYTES,
        )
    _require_snapshot_unchanged(
        snapshots.protected_openings,
        "protected publication openings",
        max_bytes=MAX_JSON_BYTES,
    )
    _require_snapshot_unchanged(
        snapshots.pre_acknowledgment_receipt,
        "pre-acknowledgment bundle reopen receipt",
        max_bytes=MAX_JSON_BYTES,
    )
    _require_snapshot_unchanged(
        snapshots.final_reopen_receipt,
        "final bundle reopen receipt",
        max_bytes=MAX_JSON_BYTES,
    )


def _capture_publication_live_inputs(
    *,
    operations_module: Any,
    publication_policy: Path | str,
    publication_policy_sha256: str,
    publication_assets: Mapping[str, Path | str],
    bundle_root: Path,
    core_snapshots: tuple[FileSnapshot, ...],
) -> tuple[
    FileSnapshot, dict[str, FileSnapshot], dict[str, Path], list[dict[str, Any]]
]:
    _require(
        isinstance(publication_policy_sha256, str)
        and SHA256_RE.fullmatch(publication_policy_sha256) is not None,
        "caller-owned publication policy digest is malformed",
    )
    policy_snapshot, policy_raw = _snapshot_regular_file(
        publication_policy,
        "trusted publication policy",
        max_bytes=operations_module.MAX_PUBLICATION_POLICY_BYTES,
    )
    policy_document = _strict_json_bytes(
        policy_raw,
        "trusted publication policy",
        max_bytes=operations_module.MAX_PUBLICATION_POLICY_BYTES,
    )
    _require(
        isinstance(policy_document, dict)
        and policy_raw == _compact_canonical_json_bytes(policy_document),
        "trusted publication policy is not canonical JSON",
    )
    _require(
        policy_snapshot.sha256 == publication_policy_sha256,
        "trusted publication policy differs from the caller-owned digest",
    )
    _require(
        not policy_snapshot.path.is_relative_to(bundle_root),
        "trusted publication policy must be outside the evidence bundle",
    )
    expected_assets = operations_module.TECHNICAL_RELEASE_ASSETS
    _require(
        isinstance(publication_assets, Mapping)
        and set(publication_assets) == set(expected_assets),
        "publication asset path closure differs",
    )
    asset_snapshots: dict[str, FileSnapshot] = {}
    resolved_assets: dict[str, Path] = {}
    asset_ledger = []
    for role, name in expected_assets.items():
        snapshot, _raw = _snapshot_regular_file(
            publication_assets[role],
            f"downloaded publication asset {role}",
            max_bytes=operations_module.MAX_PUBLICATION_ASSET_BYTES,
        )
        _require(
            snapshot.path.name == name,
            f"downloaded publication asset {role} has the wrong filename",
        )
        _require(
            not snapshot.path.is_relative_to(bundle_root),
            f"downloaded publication asset {role} must be outside the evidence bundle",
        )
        asset_snapshots[role] = snapshot
        resolved_assets[role] = snapshot.path
        asset_ledger.append(
            {
                "role": role,
                "name": name,
                "size_bytes": snapshot.size_bytes,
                "sha256": snapshot.sha256,
            }
        )
    all_snapshots = (*core_snapshots, policy_snapshot, *asset_snapshots.values())
    _require(
        len({item.path for item in all_snapshots}) == len(all_snapshots)
        and len({item.identity[:2] for item in all_snapshots}) == len(all_snapshots),
        "live verification inputs are not distinct files",
    )
    return policy_snapshot, asset_snapshots, resolved_assets, asset_ledger


def _require_publication_live_inputs_unchanged(
    operations_module: Any,
    policy_snapshot: FileSnapshot,
    asset_snapshots: Mapping[str, FileSnapshot],
) -> None:
    _require_snapshot_unchanged(
        policy_snapshot,
        "trusted publication policy",
        max_bytes=operations_module.MAX_PUBLICATION_POLICY_BYTES,
    )
    for role in operations_module.TECHNICAL_RELEASE_ASSETS:
        _require_snapshot_unchanged(
            asset_snapshots[role],
            f"downloaded publication asset {role}",
            max_bytes=operations_module.MAX_PUBLICATION_ASSET_BYTES,
        )


def _require_private_authority_file(snapshot: FileSnapshot, label: str) -> None:
    try:
        file_metadata = snapshot.path.lstat()
        parent_metadata = snapshot.path.parent.lstat()
    except OSError as error:
        raise EvidenceError(f"{label} privacy metadata is unavailable") from error
    _require(
        stat.S_ISREG(file_metadata.st_mode)
        and not stat.S_ISLNK(file_metadata.st_mode)
        and stat.S_IMODE(file_metadata.st_mode) == 0o600
        and stat.S_ISDIR(parent_metadata.st_mode)
        and not stat.S_ISLNK(parent_metadata.st_mode)
        and stat.S_IMODE(parent_metadata.st_mode) == 0o700,
        f"{label} must be mode 0600 in a mode-0700 directory",
    )


def _validate_final_bundle_reopen_chain(
    core_inputs: LiveAuthoritySnapshots,
) -> dict[str, Any]:
    """Authenticate O0/B0 -> O1/B1 and reopen the exact final B1 bytes."""

    for snapshot, label in (
        (core_inputs.protected_openings, "protected publication openings"),
        (
            core_inputs.pre_acknowledgment_receipt,
            "pre-acknowledgment bundle receipt",
        ),
        (core_inputs.final_reopen_receipt, "final bundle receipt"),
    ):
        _require_private_authority_file(snapshot, label)
    privacy, _publication = _private_binding_modules()
    _openings, context = privacy.load_private_openings(
        core_inputs.protected_openings.path
    )
    pre_receipt = validate_bundle_reopen_receipt(
        core_inputs.pre_acknowledgment_receipt.path,
        "pre_acknowledgment",
        core_inputs.protected_openings.path,
    )
    final_receipt = validate_bundle_reopen_receipt(
        core_inputs.final_reopen_receipt.path,
        "final",
        core_inputs.protected_openings.path,
    )
    _pre_document, pre_raw = _load_reopen_receipt(
        core_inputs.pre_acknowledgment_receipt.path
    )
    _final_document, final_raw = _load_reopen_receipt(
        core_inputs.final_reopen_receipt.path
    )
    pre_sha256 = _reopen_receipt_file_sha256(pre_raw)
    _require(
        final_receipt["pre_acknowledgment_receipt_sha256"] == pre_sha256,
        "final bundle receipt does not bind the exact B0 receipt",
    )
    source = _bundle_inventory_from_context(core_inputs.source_index.path, context)
    reopened = _bundle_inventory_from_context(core_inputs.reopened_index.path, context)
    _require(
        source["root"] == reopened["root"]
        and source["inventory"] == reopened["inventory"],
        "source and reopened final B1 inventories differ",
    )
    source_identity = _bundle_path_identity(source["index_path"].parent)
    reopened_identity = _bundle_path_identity(reopened["index_path"].parent)
    expected_source = {
        "inventory_root": source["root"],
        "index": source["inventory"]["index"],
        "path_identity": source_identity,
    }
    expected_reopened = {
        "inventory_root": reopened["root"],
        "index": reopened["inventory"]["index"],
        "path_identity": reopened_identity,
    }
    _require(
        final_receipt["source_bundle"] == expected_source
        and final_receipt["reopened_bundle"] == expected_reopened
        and final_receipt["registered_payloads"] == reopened["inventory"]["payloads"],
        "final receipt inventory does not describe the exact reopened B1",
    )
    source_file_ids = {
        (device, inode) for _path, device, inode in source["file_identities"]
    }
    reopened_file_ids = {
        (device, inode) for _path, device, inode in reopened["file_identities"]
    }
    _require(
        source_file_ids.isdisjoint(reopened_file_ids),
        "source and reopened final B1 payloads share file identities",
    )
    reopened_index_document = _load_inventory_index(reopened["index_path"])[2]
    issue, issue_raw, issue_descriptor = _operations_issue_response(
        reopened["index_path"], reopened_index_document
    )
    expected_issue = {
        **_checklist_observation(issue, "checked"),
        "canonical_response_sha256": _sha256(issue_raw),
        "descriptor": issue_descriptor,
    }
    _require(
        final_receipt["issue_response"] == expected_issue,
        "final B1 checked issue response differs from its reopen receipt",
    )
    pre_observed_at = _timestamp(
        pre_receipt["observed_at"], "pre-acknowledgment bundle observation"
    )
    final_updated_at = _timestamp(
        final_receipt["issue_response"]["updated_at"],
        "final checklist update",
    )
    final_observed_at = _timestamp(
        final_receipt["observed_at"], "final bundle observation"
    )
    _require(
        pre_receipt["source_bundle"]["inventory_root"]
        == pre_receipt["reopened_bundle"]["inventory_root"]
        and pre_receipt["source_bundle"]["inventory_root"] != source["root"]
        and pre_receipt["technical_binding"] == final_receipt["technical_binding"]
        and pre_receipt["issue_response"]["checklist_transition_sha256"]
        == final_receipt["issue_response"]["checklist_transition_sha256"]
        == expected_issue["checklist_transition_sha256"]
        and pre_receipt["issue_response"]["canonical_response_sha256"]
        != final_receipt["issue_response"]["canonical_response_sha256"]
        and pre_observed_at <= final_updated_at <= final_observed_at,
        "B0/O0 and B1/O1 fixed-point chronology differs",
    )
    try:
        from benchmarks import issue123_operations as operations

        post_bundle_expectation = operations.AuthenticatedPostBundleExpectation(
            checked_lines=tuple(final_receipt["issue_response"]["lines"]),
            o0_canonical_response_sha256=pre_receipt["issue_response"][
                "canonical_response_sha256"
            ],
            o1_canonical_response_sha256=final_receipt["issue_response"][
                "canonical_response_sha256"
            ],
            o1_body_sha256=final_receipt["issue_response"]["body_sha256"],
            o1_updated_at=final_receipt["issue_response"]["updated_at"],
            b0_inventory_root=pre_receipt["source_bundle"]["inventory_root"],
            b0_reopen_receipt_sha256=pre_sha256,
            b0_reopened_at=pre_receipt["observed_at"],
            checklist_transition_sha256=final_receipt["issue_response"][
                "checklist_transition_sha256"
            ],
        )
    except (ImportError, TypeError, ValueError):
        raise EvidenceError("authenticated post-bundle expectation differs") from None
    return {
        "final_bundle_inventory_root": source["root"],
        "post_bundle_expectation": post_bundle_expectation,
        "post_bundle_result": {
            "pre_acknowledgment_receipt": {
                "schema_version": BUNDLE_REOPEN_RECEIPT_VERSION,
                "kind": BUNDLE_REOPEN_RECEIPT_KIND,
                "size_bytes": len(pre_raw),
                "sha256": pre_sha256,
                "observed_at": pre_receipt["observed_at"],
                "bundle_inventory_root": pre_receipt["source_bundle"]["inventory_root"],
            },
            "final_reopen_receipt": {
                "schema_version": BUNDLE_REOPEN_RECEIPT_VERSION,
                "kind": BUNDLE_REOPEN_RECEIPT_KIND,
                "size_bytes": len(final_raw),
                "sha256": _reopen_receipt_file_sha256(final_raw),
                "observed_at": final_receipt["observed_at"],
                "bundle_inventory_root": source["root"],
            },
        },
    }


class AuthenticatedPostBundleLease:
    """Own retained B1 authority until the caller's final bytes are durable."""

    __slots__ = ("_snapshots", "_chain", "expectation", "_closed")

    def __init__(
        self,
        snapshots: LiveAuthoritySnapshots,
        chain: dict[str, Any],
    ) -> None:
        self._snapshots = snapshots
        self._chain = chain
        self.expectation = chain["post_bundle_expectation"]
        self._closed = False

    def require_unchanged(self) -> None:
        _require(not self._closed, "authenticated post-bundle lease is closed")
        _require_core_live_inputs_unchanged(self._snapshots)

    def _private_writer_roots(self) -> tuple[Path, Path]:
        """Return only the canonical roots owned by the authenticated lease."""

        _require(not self._closed, "authenticated post-bundle lease is closed")
        return (
            self._snapshots.source_bundle.root,
            self._snapshots.reopened_bundle.root,
        )

    def _baseline_authority_set(self, operations_module: Any) -> Any:
        """Derive the exact ordered baseline authority from retained final B1."""

        _require(not self._closed, "authenticated post-bundle lease is closed")
        self.require_unchanged()
        authority = _validate_final_b1_baseline_descriptors(
            self._snapshots.reopened_index.path,
            operations_module,
        )
        self.require_unchanged()
        return authority

    def close(self, *, primary_error: BaseException | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        _close_retained_bundle_trees(
            (
                self._snapshots.reopened_bundle,
                self._snapshots.source_bundle,
            ),
            primary_error=primary_error,
        )


@contextmanager
def open_authenticated_post_bundle_transition(
    *,
    source_index: Path | str,
    reopened_index: Path | str,
    protected_openings: Path | str,
    pre_ack_bundle_reopen_receipt: Path | str,
    final_bundle_reopen_receipt: Path | str,
    manifest_path: Path | str | None,
    runtime_receipt_paths: list[Path | str] | tuple[Path | str, ...],
):
    """Hold exact source/reopened B1 authority for one same-process decision."""

    snapshots = _capture_core_live_inputs(
        source_index,
        reopened_index,
        manifest_path,
        runtime_receipt_paths,
        protected_openings,
        pre_ack_bundle_reopen_receipt,
        final_bundle_reopen_receipt,
    )
    lease: AuthenticatedPostBundleLease | None = None
    primary_error: BaseException | None = None
    try:
        _require_core_live_inputs_unchanged(snapshots)
        chain = _validate_final_bundle_reopen_chain(snapshots)
        _require_core_live_inputs_unchanged(snapshots)
        lease = AuthenticatedPostBundleLease(snapshots, chain)
        yield lease
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if lease is None:
            _close_retained_bundle_trees(
                (snapshots.reopened_bundle, snapshots.source_bundle),
                primary_error=primary_error,
            )
        else:
            lease.close(primary_error=primary_error)


def _validate_final_b1_baseline_descriptors(
    index_path: Path,
    operations_module: Any,
) -> Any:
    """Return the two exact baseline descriptors already validated inside B1."""

    checked, _raw, document = _load_inventory_index(index_path)
    cpu_scope = document["artifacts"]["cpu"]
    _require(isinstance(cpu_scope, dict), "final B1 CPU scope mapping differs")
    descriptors = cpu_scope.get("torch_baseline_artifacts")
    _require(
        isinstance(descriptors, list) and len(descriptors) == 2,
        "final B1 baseline descriptor closure differs",
    )
    reader = ArtifactReader(
        checked.parent,
        document["candidate_evidence"],
        document["payloads"],
    )
    expected = operations_module.PRODUCTION_BASELINE_AUTHORITY_SET
    _require(
        type(expected) is operations_module.BaselineAuthoritySet,
        "operations baseline authority interface differs",
    )
    for ordinal, (descriptor, authority) in enumerate(
        zip(descriptors, expected.assets, strict=True)
    ):
        artifact = reader.load(
            descriptor,
            f"final B1 baseline artifact {ordinal}",
            json_document=False,
        )
        _require(
            authority.ordinal == ordinal
            and artifact.path.name == authority.name
            and len(artifact.raw) == authority.size_bytes
            and _sha256(artifact.raw) == authority.sha256,
            f"final B1 baseline artifact {ordinal} differs",
        )
    return expected


def _validate_current_live_receipt(
    *,
    operations_module: Any,
    receipt: Any,
    receipt_path: Path,
    candidate: dict[str, str],
    operations_descriptor: dict[str, Any],
    publication_policy_sha256: str,
    asset_ledger: list[dict[str, Any]],
    post_bundle_expectation: Any,
    baseline_descriptors: Any,
    invocation_started_at: dt.datetime,
    invocation_finished_at: dt.datetime,
) -> dict[str, Any]:
    _require(isinstance(receipt, dict), "live operations receipt is not an object")
    _exact_keys(
        receipt,
        {
            "schema_version",
            "kind",
            "authority",
            "receipt_replay_authority",
            "verified_at",
            "candidate_evidence",
            "repository",
            "pull_request_number",
            "operations_index",
            "publication_validation",
            "post_bundle_acknowledgment",
            "baseline_validation",
            "queries",
            "same_process_live_accepted",
        },
        "live operations receipt",
    )
    _require(
        _is_exact_int(
            receipt["schema_version"],
            operations_module.LIVE_VERIFICATION_RECEIPT_SCHEMA_VERSION,
        )
        and receipt["kind"] == operations_module.LIVE_VERIFICATION_RECEIPT_KIND
        and receipt["authority"] == LIVE_AUTHORITY
        and receipt["receipt_replay_authority"] is False
        and receipt["same_process_live_accepted"] is True
        and receipt["candidate_evidence"] == candidate
        and receipt["repository"] == operations_module.REPOSITORY
        and _is_exact_int(
            receipt["pull_request_number"], operations_module.PULL_REQUEST_NUMBER
        )
        and receipt["operations_index"] == operations_descriptor,
        "live operations receipt identity or authority differs",
    )
    verified_at = _timestamp(receipt["verified_at"], "live verification time")
    canonical_verified_at = (
        verified_at.astimezone(dt.UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    _require(
        receipt["verified_at"] == canonical_verified_at
        and invocation_started_at <= verified_at <= invocation_finished_at,
        "live operations receipt was not created by the current invocation",
    )
    publication = receipt["publication_validation"]
    _exact_keys(
        publication,
        {
            "strict_four_byte_validator",
            "receipt_sha256",
            "trusted_policy_sha256",
            "asset_ledger",
            "release_identity_anchor",
            "bindings",
            "execution_claims",
            "event_profiler",
        },
        "live publication validation",
    )
    _require(
        publication["strict_four_byte_validator"] == "same-process-invoked"
        and isinstance(publication["receipt_sha256"], str)
        and SHA256_RE.fullmatch(publication["receipt_sha256"]) is not None
        and publication["trusted_policy_sha256"] == publication_policy_sha256
        and publication["asset_ledger"] == asset_ledger
        and isinstance(publication["release_identity_anchor"], dict)
        and isinstance(publication["bindings"], dict)
        and isinstance(publication["execution_claims"], list)
        and isinstance(publication["event_profiler"], dict),
        "live publication validation is stale or substituted",
    )
    expected_acknowledgment = {
        "checked_lines": list(post_bundle_expectation.checked_lines),
        "o0_canonical_response_sha256": (
            post_bundle_expectation.o0_canonical_response_sha256
        ),
        "o1_canonical_response_sha256": (
            post_bundle_expectation.o1_canonical_response_sha256
        ),
        "o1_body_sha256": post_bundle_expectation.o1_body_sha256,
        "o1_updated_at": post_bundle_expectation.o1_updated_at,
        "b0_inventory_root": post_bundle_expectation.b0_inventory_root,
        "b0_reopen_receipt_sha256": (post_bundle_expectation.b0_reopen_receipt_sha256),
        "b0_reopened_at": post_bundle_expectation.b0_reopened_at,
        "fresh_response_equal": True,
    }
    _require(
        receipt["post_bundle_acknowledgment"] == expected_acknowledgment,
        "live post-bundle acknowledgment is stale or substituted",
    )
    baseline = receipt["baseline_validation"]
    _exact_keys(
        baseline,
        {
            "release_identity",
            "asset_ledger",
            "observed_at",
            "api_observations",
            "authority_sha256",
        },
        "live baseline validation",
    )
    release_identity = baseline["release_identity"]
    _exact_keys(
        release_identity,
        {
            "repository",
            "release_id",
            "tag_name",
            "api_url",
            "html_url",
            "tag_ref",
        },
        "live baseline release identity",
    )
    _exact_keys(
        release_identity["tag_ref"],
        {"ref", "object_type", "object_sha", "object_url"},
        "live baseline tag identity",
    )
    repository = operations_module.REPOSITORY
    release_id = release_identity["release_id"]
    tag_name = operations_module.BASELINE_RELEASE_TAG
    api_root = f"https://api.github.com/repos/{repository}"
    web_root = f"https://github.com/{repository}"
    _require(
        release_identity["repository"] == repository
        and type(release_id) is int
        and release_id > 0
        and release_identity["tag_name"] == tag_name
        and release_identity["api_url"] == f"{api_root}/releases/{release_id}"
        and release_identity["html_url"] == f"{web_root}/releases/tag/{tag_name}"
        and release_identity["tag_ref"]["ref"] == f"refs/tags/{tag_name}"
        and release_identity["tag_ref"]["object_type"] == "commit"
        and release_identity["tag_ref"]["object_sha"]
        == operations_module.BASELINE_V3_ROOT_COMMIT
        and isinstance(release_identity["tag_ref"]["object_url"], str)
        and release_identity["tag_ref"]["object_url"]
        == (f"{api_root}/git/commits/" f"{operations_module.BASELINE_V3_ROOT_COMMIT}"),
        "live baseline release/tag authority differs",
    )
    baseline_ledger = baseline["asset_ledger"]
    _require(
        isinstance(baseline_ledger, list)
        and type(baseline_descriptors) is operations_module.BaselineAuthoritySet
        and len(baseline_ledger) == len(baseline_descriptors.assets),
        "live baseline asset closure differs",
    )
    reduced_baseline_ledger = []
    for ordinal, item in enumerate(baseline_ledger):
        _exact_keys(
            item,
            {
                "thread_mode",
                "name",
                "asset_id",
                "release_id",
                "api_url",
                "browser_download_url",
                "size_bytes",
                "sha256",
            },
            f"live baseline asset {ordinal}",
        )
        _require(
            type(item["asset_id"]) is int
            and item["asset_id"] > 0
            and item["release_id"] == release_id
            and item["api_url"] == f"{api_root}/releases/assets/{item['asset_id']}"
            and item["browser_download_url"]
            == (f"{web_root}/releases/download/{tag_name}/" f"{item['name']}"),
            f"live baseline asset {ordinal} release identity differs",
        )
        reduced_baseline_ledger.append(
            {key: item[key] for key in ("thread_mode", "name", "size_bytes", "sha256")}
        )
    _require(
        reduced_baseline_ledger
        == [
            {
                "thread_mode": asset.thread_mode,
                "name": asset.name,
                "size_bytes": asset.size_bytes,
                "sha256": asset.sha256,
            }
            for asset in baseline_descriptors.assets
        ]
        and len({item["asset_id"] for item in baseline_ledger}) == len(baseline_ledger),
        "live baseline bytes differ from reopened final B1",
    )
    observations = baseline["api_observations"]
    expected_endpoints = [
        f"repos/{repository}/releases/tags/{tag_name}",
        f"repos/{repository}/git/ref/tags/{tag_name}",
    ]
    _require(
        isinstance(observations, list)
        and [item.get("endpoint") for item in observations if isinstance(item, dict)]
        == expected_endpoints,
        "live baseline API observation closure differs",
    )
    for ordinal, observation in enumerate(observations):
        _exact_keys(
            observation,
            {
                "endpoint",
                "canonical_response_sha256",
                "canonical_response_size_bytes",
                "page_ledger_sha256",
            },
            f"live baseline API observation {ordinal}",
        )
        _require(
            isinstance(observation["canonical_response_sha256"], str)
            and SHA256_RE.fullmatch(observation["canonical_response_sha256"])
            is not None
            and type(observation["canonical_response_size_bytes"]) is int
            and observation["canonical_response_size_bytes"] > 0
            and isinstance(observation["page_ledger_sha256"], str)
            and SHA256_RE.fullmatch(observation["page_ledger_sha256"]) is not None,
            f"live baseline API observation {ordinal} differs",
        )
    baseline_observed_at = _timestamp(
        baseline["observed_at"], "live baseline observation time"
    )
    canonical_baseline_observed_at = (
        baseline_observed_at.astimezone(dt.UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    baseline_body = {
        key: baseline[key] for key in baseline if key != "authority_sha256"
    }
    try:
        from benchmarks import issue123_privacy as privacy

        expected_baseline_sha256 = privacy.tagged_canonical_sha256(
            operations_module.BASELINE_AUTHORITY_DOMAIN,
            baseline_body,
        )
    except (ImportError, TypeError, ValueError) as error:
        raise EvidenceError("live baseline authority digest is unavailable") from error
    _require(
        baseline["observed_at"] == canonical_baseline_observed_at
        and invocation_started_at <= baseline_observed_at <= invocation_finished_at
        and isinstance(baseline["authority_sha256"], str)
        and hmac.compare_digest(baseline["authority_sha256"], expected_baseline_sha256),
        "live baseline authority is stale or substituted",
    )
    queries = receipt["queries"]
    _require(
        isinstance(queries, list)
        and [item.get("role") for item in queries if isinstance(item, dict)]
        == list(operations_module.RESPONSE_ROLE_ORDER),
        "live operations query closure or order differs",
    )
    for index, query in enumerate(queries):
        _exact_keys(
            query,
            {
                "role",
                "canonical_response_sha256",
                "canonical_response_size_bytes",
                "page_count",
                "page_ledger_sha256",
            },
            f"live operations query {index}",
        )
        _require(
            isinstance(query["canonical_response_sha256"], str)
            and SHA256_RE.fullmatch(query["canonical_response_sha256"]) is not None
            and type(query["canonical_response_size_bytes"]) is int
            and query["canonical_response_size_bytes"] > 0
            and type(query["page_count"]) is int
            and query["page_count"] > 0
            and isinstance(query["page_ledger_sha256"], str)
            and SHA256_RE.fullmatch(query["page_ledger_sha256"]) is not None,
            f"live operations query {index} metadata differs",
        )
    receipt_snapshot, receipt_raw = _snapshot_regular_file(
        receipt_path,
        "live operations receipt",
        max_bytes=MAX_JSON_BYTES,
    )
    parsed_receipt = _strict_json_bytes(
        receipt_raw,
        "live operations receipt",
        max_bytes=MAX_JSON_BYTES,
    )
    _require(
        receipt_snapshot.path == receipt_path.resolve(strict=True)
        and stat.S_IMODE(receipt_snapshot.path.stat().st_mode) == 0o600
        and receipt_raw == _compact_canonical_json_bytes(receipt)
        and _type_exact_equal(parsed_receipt, receipt),
        "live operations receipt bytes differ from the current invocation result",
    )
    return {
        "path": LIVE_RECEIPT_NAME,
        "size_bytes": receipt_snapshot.size_bytes,
        "sha256": receipt_snapshot.sha256,
        "media_type": MEDIA_TYPE_JSON,
    }


def _live_result_from_offline(offline_result: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(offline_result)
    result["evaluation_mode"] = LIVE_EVALUATION_MODE
    result["final_acceptance"] = False
    result["issue_completion_satisfied"] = False
    result["final_acceptance_authority"] = OFFLINE_AUTHORITY
    result["receipt_replay_authority"] = False
    result["live_verification"] = {
        "invocation_attempted": False,
        "invocation_succeeded": False,
        "authority": None,
        "verified_at": None,
        "operations_index": None,
        "receipt": None,
        "errors": [],
    }
    return result


def _append_live_error(result: dict[str, Any], error: Exception) -> None:
    result["live_verification"]["errors"].append(
        _structured_evidence_error(
            error,
            phase="operations-live-verification",
            scope="operations",
        )
    )


def _publish_authoritative_live_result(
    *,
    result: dict[str, Any],
    operations: Any,
    operations_lease: Any,
    post_bundle_lease: AuthenticatedPostBundleLease,
    core_inputs: LiveAuthoritySnapshots,
    policy_snapshot: FileSnapshot,
    asset_snapshots: Mapping[str, FileSnapshot],
    staged_operations_snapshots: StagedOperationsSnapshots,
    receipt_path: Path,
    result_path: Path,
    receipt: dict[str, Any],
    receipt_descriptor: dict[str, Any],
    operations_descriptor: dict[str, Any],
    bundle_binding: dict[str, Any],
    reopen_chain: dict[str, Any],
) -> dict[str, Any]:
    """Link authority only after the last retained-input barrier succeeds."""

    authoritative_result = copy.deepcopy(result)
    authoritative_result["candidate_bundle_binding"].update(
        satisfied=True,
        technical_input_root=bundle_binding["technical_input_root"],
        public_projection_sha256=bundle_binding["public_projection_sha256"],
        public_asset_ledger_sha256=bundle_binding["public_asset_ledger_sha256"],
        final_bundle_inventory_root=reopen_chain["final_bundle_inventory_root"],
    )
    authoritative_result["post_bundle"].update(
        satisfied=True,
        pre_acknowledgment_receipt=reopen_chain["post_bundle_result"][
            "pre_acknowledgment_receipt"
        ],
        final_reopen_receipt=reopen_chain["post_bundle_result"]["final_reopen_receipt"],
        acknowledgment=receipt["post_bundle_acknowledgment"],
    )
    authoritative_result["baseline_authority"].update(
        satisfied=True,
        mode="live-release",
        authority_sha256=receipt["baseline_validation"]["authority_sha256"],
    )
    _require(
        authoritative_result["candidate_bundle_binding"]["satisfied"] is True
        and authoritative_result["post_bundle"]["satisfied"] is True
        and authoritative_result["baseline_authority"]["satisfied"] is True,
        "final authority predicate closure differs",
    )
    authoritative_result["live_verification"].update(
        invocation_succeeded=True,
        authority=LIVE_AUTHORITY,
        verified_at=receipt["verified_at"],
        operations_index=operations_descriptor,
        receipt=receipt_descriptor,
    )
    authoritative_result["final_acceptance"] = True
    authoritative_result["issue_completion_satisfied"] = True
    authoritative_result["final_acceptance_authority"] = LIVE_AUTHORITY
    authoritative_raw = _canonical_json_bytes(authoritative_result)

    def final_authority_barrier() -> None:
        _require_publication_live_inputs_unchanged(
            operations,
            policy_snapshot,
            asset_snapshots,
        )
        _require_staged_operations_inputs_unchanged(staged_operations_snapshots)
        _receipt_snapshot, current_receipt_raw = _snapshot_regular_file(
            receipt_path,
            "live operations receipt",
            max_bytes=operations.MAX_PUBLICATION_RECEIPT_BYTES,
        )
        _require(
            len(current_receipt_raw) == receipt_descriptor["size_bytes"]
            and _sha256(current_receipt_raw) == receipt_descriptor["sha256"]
            and current_receipt_raw == _compact_canonical_json_bytes(receipt),
            "live operations receipt changed before final authority",
        )
        operations_lease.require_unchanged()
        # This call is deliberately last.  The shared writer performs the
        # no-replace authority link as its immediately following operation.
        post_bundle_lease.require_unchanged()

    try:
        committed_path = _write_exclusive_private_file(
            result_path,
            authoritative_raw,
            "completion live result",
            forbidden_roots=(
                core_inputs.source_bundle.root,
                core_inputs.reopened_bundle.root,
            ),
            before_commit=final_authority_barrier,
        )
        final_snapshot, final_raw = _snapshot_regular_file(
            committed_path,
            "committed completion live result",
            max_bytes=MAX_JSON_BYTES,
        )
        final_document = _strict_json_bytes(
            final_raw,
            "committed completion live result",
            max_bytes=MAX_JSON_BYTES,
        )
        _require(
            final_snapshot.path == result_path.resolve(strict=True)
            and stat.S_IMODE(final_snapshot.path.stat().st_mode) == 0o600
            and final_raw == authoritative_raw
            and _type_exact_equal(final_document, authoritative_result)
            and final_document["final_acceptance"] is True
            and final_document["issue_completion_satisfied"] is True,
            "committed completion authority differs",
        )
        operations_lease.require_unchanged()
        post_bundle_lease.require_unchanged()
    except CommittedAuthorityError:
        raise
    except Exception as error:
        if result_path.exists():
            raise CommittedAuthorityError(
                "completion authority was committed but final custody failed"
            ) from None
        _append_live_error(result, error)
        raise EvidenceError("completion live result could not be emitted") from None
    return final_document


def _verify_completion_live_with_lease(
    *,
    lease: AuthenticatedPostBundleLease,
    index_path: Path | str,
    manifest_path: Path | str | None = None,
    runtime_receipt_paths: list[Path | str] | tuple[Path | str, ...],
    publication_policy: Path | str,
    publication_policy_sha256: str,
    publication_assets: Mapping[str, Path | str],
    output_directory: Path | str,
) -> dict[str, Any]:
    """Make a final decision only through a current-process operations live check."""

    core_inputs = lease._snapshots
    offline_result = evaluate_completion(
        index_path,
        manifest_path,
        runtime_receipt_paths,
    )
    result = _live_result_from_offline(offline_result)
    if offline_result.get("structural_validation_satisfied") is not True:
        _append_live_error(
            result,
            EvidenceError("offline structural validation is not satisfied"),
        )
        return result
    index_snapshot = core_inputs.source_index
    manifest_snapshot = core_inputs.manifest
    runtime_snapshots = core_inputs.runtime_receipts
    protected_bundle_roots = lease._private_writer_roots()
    try:
        _require(
            offline_result.get("evidence_index", {}).get("size_bytes")
            == index_snapshot.size_bytes
            and offline_result.get("evidence_index", {}).get("sha256")
            == index_snapshot.sha256
            and offline_result.get("manifest", {}).get("size_bytes")
            == manifest_snapshot.size_bytes
            and offline_result.get("manifest", {}).get("sha256")
            == manifest_snapshot.sha256,
            "structural result is not bound to the snapshotted index and manifest",
        )
        lease.require_unchanged()
        destination = _create_private_live_directory(
            output_directory,
            protected_bundle_roots,
        )
    except Exception as error:
        _append_live_error(result, error)
        return result

    receipt_path = destination / LIVE_RECEIPT_NAME
    result_path = destination / LIVE_RESULT_NAME
    authority_linked = False
    authority_publication_started = False
    try:
        from benchmarks import issue123_operations as operations
        from benchmarks import issue123_privacy as privacy

        _require(
            tuple(operations.TECHNICAL_RELEASE_ASSETS)
            == (
                "technical_evidence",
                "technical_summary",
                "raw_timing",
                "event_profiler",
            ),
            "operations publication asset interface is incompatible",
        )
        policy_snapshot, asset_snapshots, resolved_assets, asset_ledger = (
            _capture_publication_live_inputs(
                operations_module=operations,
                publication_policy=publication_policy,
                publication_policy_sha256=publication_policy_sha256,
                publication_assets=publication_assets,
                bundle_root=index_snapshot.path.parent,
                core_snapshots=(
                    index_snapshot,
                    core_inputs.reopened_index,
                    manifest_snapshot,
                    *runtime_snapshots,
                    core_inputs.protected_openings,
                    core_inputs.pre_acknowledgment_receipt,
                    core_inputs.final_reopen_receipt,
                ),
            )
        )
        _require(
            all(
                core_inputs.protected_openings.path.parent != snapshot.path.parent
                for snapshot in asset_snapshots.values()
            ),
            "protected openings must remain outside public asset directories",
        )
        reopen_chain = lease._chain
        lease.require_unchanged()
        _policy_snapshot, policy_raw = _snapshot_regular_file(
            policy_snapshot.path,
            "trusted publication policy",
            max_bytes=operations.MAX_PUBLICATION_POLICY_BYTES,
        )
        policy_document = _strict_json_bytes(
            policy_raw,
            "trusted publication policy",
            max_bytes=operations.MAX_PUBLICATION_POLICY_BYTES,
        )
        _require(
            isinstance(policy_document, dict),
            "trusted publication policy is not an object",
        )
        runtime_receipt_mapping = {
            role: snapshot.path
            for role, snapshot in zip(
                RUNTIME_RECEIPT_ROLES,
                runtime_snapshots,
                strict=True,
            )
        }
        bundle_binding = privacy.verify_publication_bundle_binding(
            index_path=core_inputs.reopened_index.path,
            protected_openings=core_inputs.protected_openings.path,
            policy=policy_document,
            public_assets=resolved_assets,
            runtime_receipt_paths=runtime_receipt_mapping,
            manifest_path=manifest_snapshot.path,
        )
        _require(
            bundle_binding.get("first_five_scopes_validated") is True
            and bundle_binding.get("runtime_receipt_count")
            == len(RUNTIME_RECEIPT_ROLES),
            "final B1 publication binding is incomplete",
        )
        baseline_descriptors = lease._baseline_authority_set(operations)
        lease.require_unchanged()
        _require_publication_live_inputs_unchanged(
            operations,
            policy_snapshot,
            asset_snapshots,
        )
        (
            staged_index,
            operations_descriptor,
            staged_operations_snapshots,
        ) = _prepare_operations_live_input(
            index_snapshot,
            offline_result,
            destination,
            protected_bundle_roots,
        )
        _require_publication_live_inputs_unchanged(
            operations,
            policy_snapshot,
            asset_snapshots,
        )
        _require_staged_operations_inputs_unchanged(staged_operations_snapshots)
        lease.require_unchanged()
        invocation_started_at = dt.datetime.now(dt.UTC).replace(microsecond=0)
        result["live_verification"]["invocation_attempted"] = True
        with operations.open_verified_operations_live(
            index_path=staged_index,
            manifest=manifest_snapshot.path,
            publication_policy=policy_snapshot.path,
            publication_policy_sha256=publication_policy_sha256,
            publication_assets=resolved_assets,
            receipt_output=receipt_path,
            post_bundle_lease=lease,
            baseline_authority="live-release",
        ) as operations_lease:
            invocation_finished_at = dt.datetime.now(dt.UTC).replace(microsecond=0)
            _require_staged_operations_inputs_unchanged(staged_operations_snapshots)
            lease.require_unchanged()
            _require_publication_live_inputs_unchanged(
                operations,
                policy_snapshot,
                asset_snapshots,
            )
            receipt = operations_lease.receipt
            receipt_descriptor = _validate_current_live_receipt(
                operations_module=operations,
                receipt=receipt,
                receipt_path=receipt_path,
                candidate=offline_result["candidate_evidence"],
                operations_descriptor=operations_descriptor,
                publication_policy_sha256=publication_policy_sha256,
                asset_ledger=asset_ledger,
                post_bundle_expectation=lease.expectation,
                baseline_descriptors=baseline_descriptors,
                invocation_started_at=invocation_started_at,
                invocation_finished_at=invocation_finished_at,
            )
            operations_lease.require_unchanged()
            lease.require_unchanged()
            _require_publication_live_inputs_unchanged(
                operations,
                policy_snapshot,
                asset_snapshots,
            )
            _require_staged_operations_inputs_unchanged(staged_operations_snapshots)
            authority_publication_started = True
            authoritative_result = _publish_authoritative_live_result(
                result=result,
                operations=operations,
                operations_lease=operations_lease,
                post_bundle_lease=lease,
                core_inputs=core_inputs,
                policy_snapshot=policy_snapshot,
                asset_snapshots=asset_snapshots,
                staged_operations_snapshots=staged_operations_snapshots,
                receipt_path=receipt_path,
                result_path=result_path,
                receipt=receipt,
                receipt_descriptor=receipt_descriptor,
                operations_descriptor=operations_descriptor,
                bundle_binding=bundle_binding,
                reopen_chain=reopen_chain,
            )
            authority_linked = True
    except CommittedAuthorityError:
        raise
    except Exception as error:
        if authority_linked or result_path.exists():
            raise CommittedAuthorityError(
                "completion authority was committed but custody cleanup failed"
            ) from None
        if authority_publication_started:
            raise EvidenceError("completion live result could not be emitted") from None
        _append_live_error(result, error)
    else:
        return authoritative_result
    try:
        _write_exclusive_private_file(
            result_path,
            _canonical_json_bytes(result),
            "completion live result",
            forbidden_roots=(
                core_inputs.source_bundle.root,
                core_inputs.reopened_bundle.root,
            ),
        )
    except Exception as error:
        _append_live_error(result, error)
        raise EvidenceError("completion live result could not be emitted") from error
    return result


def verify_completion_live(
    *,
    index_path: Path | str,
    reopened_index: Path | str,
    protected_openings: Path | str,
    pre_ack_bundle_reopen_receipt: Path | str,
    final_bundle_reopen_receipt: Path | str,
    manifest_path: Path | str | None = None,
    runtime_receipt_paths: list[Path | str] | tuple[Path | str, ...],
    publication_policy: Path | str,
    publication_policy_sha256: str,
    publication_assets: Mapping[str, Path | str],
    output_directory: Path | str,
) -> dict[str, Any]:
    """Verify final authority while retaining both exact B1 trees."""

    manager = open_authenticated_post_bundle_transition(
        source_index=index_path,
        reopened_index=reopened_index,
        protected_openings=protected_openings,
        pre_ack_bundle_reopen_receipt=pre_ack_bundle_reopen_receipt,
        final_bundle_reopen_receipt=final_bundle_reopen_receipt,
        manifest_path=manifest_path,
        runtime_receipt_paths=runtime_receipt_paths,
    )
    try:
        lease = manager.__enter__()
    except Exception as error:
        offline_result = evaluate_completion(
            index_path,
            manifest_path,
            runtime_receipt_paths,
        )
        result = _live_result_from_offline(offline_result)
        _append_live_error(result, error)
        return result
    authoritative_linked = False
    try:
        result = _verify_completion_live_with_lease(
            lease=lease,
            index_path=index_path,
            manifest_path=manifest_path,
            runtime_receipt_paths=runtime_receipt_paths,
            publication_policy=publication_policy,
            publication_policy_sha256=publication_policy_sha256,
            publication_assets=publication_assets,
            output_directory=output_directory,
        )
        authoritative_linked = (
            result.get("final_acceptance") is True
            and result.get("issue_completion_satisfied") is True
        )
        return result
    finally:
        exit_arguments = sys.exc_info()
        committed_cleanup_failed = False
        try:
            manager.__exit__(*exit_arguments)
        except EvidenceError:
            if exit_arguments[0] is None and authoritative_linked:
                committed_cleanup_failed = True
            else:
                raise
        if committed_cleanup_failed:
            raise CommittedAuthorityError(
                "completion authority was committed but custody cleanup failed"
            ) from None


class _CliUsageError(ValueError):
    pass


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise _CliUsageError("completion CLI usage differs") from None


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _SafeArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    assemble = commands.add_parser("assemble", help="create a relocatable bundle")
    assemble.add_argument("--specification", "--spec", type=Path, required=True)
    assemble.add_argument("--bundle", type=Path, required=True)
    assemble.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    record = commands.add_parser(
        "record-reopen",
        help="authenticate a distinct B0 or B1 content-addressed reopen",
    )
    record.add_argument("--source-index", type=Path, required=True)
    record.add_argument("--reopened-index", type=Path, required=True)
    record.add_argument(
        "--stage",
        choices=("pre-acknowledgment", "final"),
        required=True,
    )
    record.add_argument("--private-openings", type=Path, required=True)
    record.add_argument("--pre-ack-response", type=Path)
    record.add_argument("--pre-ack-receipt", type=Path)
    record.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate", help="validate a completed bundle")
    evaluate.add_argument("--index", type=Path, required=True)
    evaluate.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    evaluate.add_argument(
        "--runtime-receipts",
        type=Path,
        nargs=5,
        metavar=(
            "CPU",
            "CUDA_EAGER",
            "CUDA_GRAPH",
            "SINGLE_GPU_2D",
            "SINGLE_GPU_3D",
        ),
        required=True,
        help=(
            "trusted publication receipts in fixed CPU/eager/graph/"
            "single-GPU-2D/single-GPU-3D order"
        ),
    )
    evaluate.add_argument("--output", type=Path)
    evaluation_enforcement = evaluate.add_mutually_exclusive_group()
    evaluation_enforcement.add_argument(
        "--enforce-structural",
        action="store_true",
        help="exit 2 unless the offline structural validation is satisfied",
    )
    evaluation_enforcement.add_argument(
        "--enforce",
        action="store_true",
        help=(
            "legacy final-acceptance gate; offline evaluation always exits 2 "
            "because it has no live authority"
        ),
    )
    verify = commands.add_parser(
        "verify-live",
        help="make the production decision through a same-process live check",
    )
    verify.add_argument("--index", type=Path, required=True)
    verify.add_argument("--reopened-index", type=Path, required=True)
    verify.add_argument("--private-openings", type=Path, required=True)
    verify.add_argument(
        "--pre-ack-bundle-reopen-receipt",
        type=Path,
        required=True,
    )
    verify.add_argument(
        "--final-bundle-reopen-receipt",
        type=Path,
        required=True,
    )
    verify.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    verify.add_argument(
        "--runtime-receipts",
        type=Path,
        nargs=5,
        metavar=(
            "CPU",
            "CUDA_EAGER",
            "CUDA_GRAPH",
            "SINGLE_GPU_2D",
            "SINGLE_GPU_3D",
        ),
        required=True,
        help=(
            "trusted publication receipts in fixed CPU/eager/graph/"
            "single-GPU-2D/single-GPU-3D order"
        ),
    )
    verify.add_argument("--publication-policy", type=Path, required=True)
    verify.add_argument("--publication-policy-sha256", required=True)
    verify.add_argument("--technical-evidence-asset", type=Path, required=True)
    verify.add_argument("--technical-summary-asset", type=Path, required=True)
    verify.add_argument("--raw-timing-asset", type=Path, required=True)
    verify.add_argument("--event-profiler-asset", type=Path, required=True)
    verify.add_argument("--output-directory", type=Path, required=True)
    verify.add_argument(
        "--enforce",
        action="store_true",
        required=True,
        help="exit 2 unless this invocation obtains production final acceptance",
    )
    return parser.parse_args(argv)


def _main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    if args.command == "assemble":
        index_path = assemble_evidence_bundle(
            args.specification,
            args.bundle,
            args.manifest,
        )
        print(index_path)
        return 0
    if args.command == "record-reopen":
        record_bundle_reopen(
            source_index=args.source_index,
            reopened_index=args.reopened_index,
            stage=args.stage,
            protected_openings=args.private_openings,
            pre_ack_response=args.pre_ack_response,
            pre_ack_receipt=args.pre_ack_receipt,
            output=args.output,
        )
        print("issue123-completion-record-reopen-ok")
        return 0
    if args.command == "evaluate":
        result = evaluate_completion(args.index, args.manifest, args.runtime_receipts)
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
        if args.enforce:
            return 2
        return (
            2
            if args.enforce_structural and not result["structural_validation_satisfied"]
            else 0
        )
    result = verify_completion_live(
        index_path=args.index,
        reopened_index=args.reopened_index,
        protected_openings=args.private_openings,
        pre_ack_bundle_reopen_receipt=args.pre_ack_bundle_reopen_receipt,
        final_bundle_reopen_receipt=args.final_bundle_reopen_receipt,
        manifest_path=args.manifest,
        runtime_receipt_paths=args.runtime_receipts,
        publication_policy=args.publication_policy,
        publication_policy_sha256=args.publication_policy_sha256,
        publication_assets={
            "technical_evidence": args.technical_evidence_asset,
            "technical_summary": args.technical_summary_asset,
            "raw_timing": args.raw_timing_asset,
            "event_profiler": args.event_profiler_asset,
        },
        output_directory=args.output_directory,
    )
    rendered = json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    return 0 if result["final_acceptance"] else 2


def main(argv: list[str] | None = None) -> int:
    """Run the fixed-token completion command boundary."""

    values = list(sys.argv[1:] if argv is None else argv)
    command = (
        values[0]
        if values
        and values[0]
        in {
            "assemble",
            "record-reopen",
            "evaluate",
            "verify-live",
        }
        else None
    )
    try:
        return _main(values)
    except _CliUsageError:
        print("issue123-completion-usage-failed", file=sys.stderr)
        return 2
    except (ImportError, OSError, EvidenceError, TypeError, ValueError):
        token = (
            f"issue123-completion-{command}-failed"
            if command is not None
            else "issue123-completion-usage-failed"
        )
        print(token, file=sys.stderr)
        return 2


def _cli(argv: list[str] | None = None) -> int:
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(_cli())
