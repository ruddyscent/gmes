#!/usr/bin/env python3
"""Load and compare the frozen legacy Torch CPU timing baseline.

The legacy artifacts predate the revised CPU acceptance contract.  This module
therefore treats them as immutable timing measurements, not as candidate gate
decisions: it revalidates their provenance, runtime controls, sampling, and
allocation accounting before exposing their timings as the Torch reference.
The revised fixed-temporary and full-field-clone rules apply only to candidates.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median
from urllib.parse import urlsplit

_EXPECTED_CASE_COUNT = 6
_FROZEN_TIMING_ROOT_COMMIT = "821c075b9328e02c3f3e5d16488a44b64ff08c04"
_RELEASE_TAG = "issue-123-torch-cpu-baseline-v2"
_RELEASE_ASSET_BASE_URL = (
    "https://github.com/ruddyscent/gmes/releases/download/" f"{_RELEASE_TAG}/"
)
_PUBLIC_BASELINE_HOSTNAME = "redacted"
_PUBLIC_HOST_IDENTITY_SCHEMA = "torch-cpu-host-identity-v1"
_FROZEN_SLICE_ARTIFACTS = {
    "one": {
        "thread_mode": "one",
        "threads": 1,
        "publication_url": (f"{_RELEASE_ASSET_BASE_URL}torch-cpu-baseline-one.json"),
        "size_bytes": 18281,
        "sha256": "ea57620653b6e96a200ffc15ba8ca9cf2309a5ada2d8ee86a2945e4787431c79",
    },
    "physical": {
        "thread_mode": "physical",
        "threads": 4,
        "publication_url": (
            f"{_RELEASE_ASSET_BASE_URL}torch-cpu-baseline-physical.json"
        ),
        "size_bytes": 18292,
        "sha256": "492b478211b5d1c32493197064393601008f1f2ca5683e261d9d103699b87ba6",
    },
}
_RUNTIME_HARD_CONTROLS = (
    "compiler_clean",
    "compiled_hot_path_complete",
    "external_indexed_writes_only_sources",
    "steady_state_transfers_zero",
    "storage_stable",
    "memory_bounded",
    "measurement_contract_matches_manifest",
    "state_progressed",
)
_RAW_HOST_IDENTITY_KEYS = (
    "platform",
    "python",
    "torch",
    "cuda_runtime",
    "devices",
    "cpu_count",
    "cpu_affinity",
    "cpu_count_physical_affinity",
    "cpu_topology",
    "cpu_model",
    "gpu_topology",
)
_PUBLIC_HOST_IDENTITY_KEYS = ("schema", "salt", "sha256")
_PUBLIC_ENVIRONMENT_KEYS = ("hostname", "host_identity", "thread_environment")
_RUNTIME_CONTRACT_KEYS = (
    "device",
    "precision",
    "compile_policy",
    "compile_mode",
    "explicit_cuda_graphs",
    "execution_policy",
    "experimental_dispersive_grouping",
    "experimental_dispersive_grouping_scope",
    "threads",
    "interop_threads",
)
_PUBLIC_CASE_KEYS = (
    "workload",
    "benchmark_contract",
    "runtime",
    "measurements",
    "memory",
    "profiler",
    "acceptance",
)
_PUBLIC_ADVANCE_KEYS = (
    "raw_seconds",
    "median_seconds",
    "relative_mad",
    "repetitions",
    "steps_per_repeat",
    "seconds_per_step",
)
_PUBLIC_PROFILER_KEYS = (
    "positive_allocation_events",
    "allocated_bytes",
    "freed_bytes",
    "allocation_net_bytes",
    "max_allocation_bytes",
    "allocation_size_histogram",
)
_LEGACY_EVIDENCE_KEYS = (
    "evidence_contract_id",
    "cpu_contract_id",
    "manifest_sha256",
    "runner_sha256",
    "solver_sha256",
    "solver_abi",
)


def _is_integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _json_equal(first, second):
    """Compare JSON values without treating booleans as integers."""
    if type(first) is not type(second):
        return False
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(
            _json_equal(first[key], second[key]) for key in first
        )
    if isinstance(first, list):
        return len(first) == len(second) and all(
            _json_equal(left, right) for left, right in zip(first, second, strict=True)
        )
    return first == second


def _read_json(path, label):
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(f"could not read {label} {path}: {error}") from error
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} {path} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} {path} must contain a JSON object")
    return value, raw


def _manifest_value(manifest):
    if isinstance(manifest, Mapping):
        return copy.deepcopy(dict(manifest))
    value, _raw = _read_json(Path(manifest), "manifest")
    return value


def _full_lower_hex(value, length):
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _release_asset_url(mode):
    return (
        _RELEASE_ASSET_BASE_URL
        + {
            "one": "torch-cpu-baseline-one.json",
            "physical": "torch-cpu-baseline-physical.json",
        }[mode]
    )


def _canonical_release_asset_url(value, mode):
    if not isinstance(value, str) or not value:
        return False
    parsed = urlsplit(value)
    return (
        parsed.scheme == "https"
        and parsed.netloc == "github.com"
        and not parsed.username
        and not parsed.password
        and parsed.query == ""
        and parsed.fragment == ""
        and value == _release_asset_url(mode)
    )


def _normalized_cpu_model(value):
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("CPU baseline CPU model identity is invalid")
    return "\n".join(
        line
        for line in value.splitlines()
        if not line.lstrip().startswith("CPU(s) scaling MHz:")
    )


def _canonical_json_bytes(value):
    try:
        return json.dumps(
            value, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("CPU baseline host identity is not canonical JSON") from error


def _raw_host_identity(environment):
    if not isinstance(environment, Mapping):
        raise ValueError("CPU baseline host identity must be an object")
    missing = [name for name in _RAW_HOST_IDENTITY_KEYS if name not in environment]
    if missing:
        raise ValueError(f"CPU baseline host identity is incomplete: {missing}")
    for name in ("platform", "python", "torch"):
        if not isinstance(environment[name], str) or not environment[name]:
            raise ValueError(f"CPU baseline host identity {name} is empty")
    if environment["cuda_runtime"] is not None and not isinstance(
        environment["cuda_runtime"], str
    ):
        raise ValueError("CPU baseline CUDA runtime identity is invalid")
    if not isinstance(environment["devices"], list):
        raise ValueError("CPU baseline device identity must be a sequence")
    if not _is_integer(environment["cpu_count"]) or environment["cpu_count"] < 1:
        raise ValueError("CPU baseline logical CPU count is invalid")
    affinity = environment["cpu_affinity"]
    if affinity is not None and (
        not isinstance(affinity, list)
        or not affinity
        or any(not _is_integer(value) or value < 0 for value in affinity)
        or len(set(affinity)) != len(affinity)
    ):
        raise ValueError("CPU baseline affinity is malformed")
    physical = environment["cpu_count_physical_affinity"]
    if not _is_integer(physical) or physical < 2:
        raise ValueError("CPU baseline physical-core count must distinguish two modes")
    for name in ("cpu_topology", "gpu_topology"):
        value = environment[name]
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"CPU baseline {name} identity is invalid")
    return {
        name: (
            _normalized_cpu_model(environment[name])
            if name == "cpu_model"
            else copy.deepcopy(environment[name])
        )
        for name in _RAW_HOST_IDENTITY_KEYS
    }


def _host_identity_token_from_raw_environment(environment, salt):
    if not _full_lower_hex(salt, 64):
        raise ValueError("CPU baseline host identity salt is invalid")
    material = _raw_host_identity(environment)
    return {
        "schema": _PUBLIC_HOST_IDENTITY_SCHEMA,
        "salt": salt,
        "sha256": hashlib.sha256(
            _canonical_json_bytes(
                {
                    "schema": _PUBLIC_HOST_IDENTITY_SCHEMA,
                    "salt": salt,
                    "identity": material,
                }
            )
        ).hexdigest(),
    }


def _public_host_identity_token(value):
    if not isinstance(value, Mapping) or set(value) != set(_PUBLIC_HOST_IDENTITY_KEYS):
        raise ValueError("CPU baseline public host identity schema is invalid")
    if (
        value["schema"] != _PUBLIC_HOST_IDENTITY_SCHEMA
        or not _full_lower_hex(value["salt"], 64)
        or not _full_lower_hex(value["sha256"], 64)
    ):
        raise ValueError("CPU baseline public host identity is invalid")
    return copy.deepcopy(dict(value))


def privacy_preserving_host_identity(environment, *, salt=None):
    """Return a release-scoped host identity commitment for comparison."""
    if not isinstance(environment, Mapping):
        raise ValueError("CPU baseline host identity must be an object")
    if "host_identity" in environment:
        allowed = {
            frozenset({"hostname", "host_identity"}),
            frozenset(_PUBLIC_ENVIRONMENT_KEYS),
        }
        if frozenset(environment) not in allowed:
            raise ValueError("CPU baseline public environment schema is invalid")
        if environment.get("hostname") != _PUBLIC_BASELINE_HOSTNAME:
            raise ValueError("CPU baseline hostname must be redacted")
        return _public_host_identity_token(environment["host_identity"])
    return _host_identity_token_from_raw_environment(environment, salt)


def _manifest_contract(manifest):
    try:
        reference = manifest["reference"]
        acceptance = manifest["performance_gates"]["cpu_acceptance"]
        timing_reference = acceptance["timing_reference"]
        slice_artifacts = timing_reference["slice_artifacts"]
        legacy_evidence = timing_reference["legacy_evidence"]
        case_names = acceptance["cases"]
        statistics = acceptance["statistics"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "manifest is missing the Torch CPU baseline contract"
        ) from error

    if acceptance.get("contract_id") != "cpu-acceptance-v2":
        raise ValueError("manifest CPU acceptance contract must be version 2")
    if timing_reference.get("backend") != "torch":
        raise ValueError("manifest CPU timing reference must use Torch")
    root_commit = timing_reference.get("root_commit")
    if root_commit != _FROZEN_TIMING_ROOT_COMMIT:
        raise ValueError("manifest timing reference must use the frozen root commit")
    if not isinstance(slice_artifacts, list) or len(slice_artifacts) != 2:
        raise ValueError("manifest must pin two Torch CPU baseline artifacts")
    expected_artifacts = {}
    for artifact, (expected_mode, expected_threads) in zip(
        slice_artifacts, (("one", 1), ("physical", 4)), strict=True
    ):
        if not isinstance(artifact, dict) or set(artifact) != {
            "thread_mode",
            "threads",
            "publication_url",
            "size_bytes",
            "sha256",
        }:
            raise ValueError("manifest Torch CPU baseline artifact schema is invalid")
        if (
            artifact["thread_mode"] != expected_mode
            or type(artifact["threads"]) is not int
            or artifact["threads"] != expected_threads
            or not _canonical_release_asset_url(
                artifact["publication_url"], expected_mode
            )
            or type(artifact["size_bytes"]) is not int
            or artifact["size_bytes"] < 1
            or not _full_lower_hex(artifact["sha256"], 64)
        ):
            raise ValueError("manifest Torch CPU baseline artifact pin is invalid")
        expected_artifacts[expected_mode] = copy.deepcopy(artifact)
    if not _json_equal(expected_artifacts, _FROZEN_SLICE_ARTIFACTS):
        raise ValueError("manifest Torch CPU baseline artifacts are not frozen")
    if not isinstance(legacy_evidence, dict) or set(legacy_evidence) != set(
        _LEGACY_EVIDENCE_KEYS
    ):
        raise ValueError("manifest legacy evidence descriptor is incomplete")
    for name in ("manifest_sha256", "runner_sha256", "solver_sha256"):
        if not _full_lower_hex(legacy_evidence.get(name), 64):
            raise ValueError(f"manifest legacy {name} is not a SHA-256 digest")
    for name in ("evidence_contract_id", "cpu_contract_id", "solver_abi"):
        if not isinstance(legacy_evidence.get(name), str) or not legacy_evidence[name]:
            raise ValueError(f"manifest legacy {name} is empty")
    expected_legacy_evidence = {
        "evidence_contract_id": "torch-cpu-acceptance-v7",
        "cpu_contract_id": "cpu-acceptance-v1",
        "manifest_sha256": (
            "6d7fe084c558cf69771f0c3928bc9be96fc6bb5b55ba777d674151fbbe6cbe19"
        ),
        "runner_sha256": (
            "fee6d418bb50729ddb26ff14e931a4f51bb8d2a92cb0ad537c2757846247a770"
        ),
        "solver_sha256": (
            "9cd8decc801a6f9d93551c6e6f427afeff1c65e3092e54b03e5abe0a3e9192d5"
        ),
        "solver_abi": "torch-fdtd-regions-v8",
    }
    if not _json_equal(legacy_evidence, expected_legacy_evidence):
        raise ValueError("manifest legacy Torch evidence is not frozen")

    if (
        not isinstance(case_names, list)
        or len(case_names) != _EXPECTED_CASE_COUNT
        or len(set(case_names)) != _EXPECTED_CASE_COUNT
        or any(not isinstance(name, str) or not name for name in case_names)
    ):
        raise ValueError("manifest CPU baseline must contain six unique ordered cases")
    if acceptance.get("thread_modes") != ["one", "physical"]:
        raise ValueError("manifest CPU thread modes must be one and physical")
    if acceptance.get("precision") != "float64":
        raise ValueError("manifest CPU baseline precision must be float64")
    if acceptance.get("native_comparison") != "informational":
        raise ValueError("manifest native comparison must be informational")
    max_ratio = acceptance.get("max_individual_ratio")
    if (
        isinstance(max_ratio, bool)
        or not isinstance(max_ratio, (int, float))
        or not math.isfinite(float(max_ratio))
        or float(max_ratio) != 1.05
    ):
        raise ValueError("manifest individual ratio limit must be exactly 1.05")
    max_relative_mad = statistics.get("max_relative_mad")
    if (
        isinstance(max_relative_mad, bool)
        or not isinstance(max_relative_mad, (int, float))
        or not math.isfinite(float(max_relative_mad))
        or not 0.0 <= float(max_relative_mad) < 1.0
    ):
        raise ValueError("manifest relative-MAD limit is invalid")
    expected_statistics = {
        "method": "independent-stratified-bootstrap-log-geomean-v1",
        "resamples": 20000,
        "seed": 123,
        "one_sided_confidence": 0.95,
        "regression_ratio": 1.0,
        "max_relative_mad": 0.05,
    }
    if not _json_equal(statistics, expected_statistics):
        raise ValueError("manifest CPU baseline statistics are not frozen")

    expected_contract = {
        "initializer": reference.get("field_initializer"),
        "seed": reference.get("seed"),
        "field_scale": reference.get("field_scale"),
        "warmup_steps": reference.get("performance_warmup_steps"),
        "steps_per_repeat": reference.get("performance_steps_per_repeat"),
        "repetitions": reference.get("performance_repetitions"),
        "profile_steps": reference.get("performance_profile_steps"),
        "timer": "time.perf_counter",
        "sample_start": "independently-restored-pre-warmup-state",
    }
    if expected_contract["steps_per_repeat"] != 100:
        raise ValueError("Torch CPU baseline requires exactly 100 steps per repeat")
    if expected_contract["repetitions"] != 15:
        raise ValueError("Torch CPU baseline requires exactly 15 timing repeats")
    for name in ("warmup_steps", "profile_steps"):
        if not _is_integer(expected_contract[name]) or expected_contract[name] < 1:
            raise ValueError(f"manifest {name} must be a positive integer")

    known_specs = {}
    for group in ("correctness", "benchmarks", "physical_checks"):
        values = manifest.get(group, ())
        if not isinstance(values, list):
            raise ValueError(f"manifest {group} workloads must be a sequence")
        for spec in values:
            if not isinstance(spec, dict) or not isinstance(spec.get("name"), str):
                raise ValueError(f"manifest {group} contains a malformed workload")
            if spec["name"] in known_specs:
                raise ValueError(f"manifest workload {spec['name']!r} is duplicated")
            known_specs[spec["name"]] = spec
    try:
        specs = [copy.deepcopy(known_specs[name]) for name in case_names]
    except KeyError as error:
        raise ValueError(
            f"manifest CPU workload {error.args[0]!r} is unknown"
        ) from error

    evidence = {
        **copy.deepcopy(legacy_evidence),
        "candidate_git_commit": root_commit,
        "candidate_git_status": "",
    }
    return {
        "acceptance": acceptance,
        "timing_reference": copy.deepcopy(timing_reference),
        "slice_artifacts": expected_artifacts,
        "evidence": evidence,
        "case_names": tuple(case_names),
        "specs": specs,
        "benchmark_contract": expected_contract,
        "max_relative_mad": float(max_relative_mad),
        "max_individual_ratio": float(max_ratio),
    }


def _validate_host_identity(environment, expected_threads):
    if not isinstance(environment, dict):
        raise ValueError("CPU baseline host identity must be an object")
    if set(environment) != set(_PUBLIC_ENVIRONMENT_KEYS):
        raise ValueError("CPU baseline public environment schema is invalid")
    if environment["hostname"] != _PUBLIC_BASELINE_HOSTNAME:
        raise ValueError("CPU baseline hostname must be redacted")
    host_identity = _public_host_identity_token(environment["host_identity"])
    thread_environment = environment["thread_environment"]
    if not isinstance(thread_environment, dict) or set(thread_environment) != {
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    }:
        raise ValueError("CPU baseline thread environment is incomplete")
    expected_thread_value = str(expected_threads)
    if any(value != expected_thread_value for value in thread_environment.values()):
        raise ValueError("CPU baseline thread environment differs from its slice")
    return {
        "hostname": _PUBLIC_BASELINE_HOSTNAME,
        "host_identity": host_identity,
        "thread_environment": copy.deepcopy(thread_environment),
    }


def _runtime_contract(runtime, expected_threads):
    if not isinstance(runtime, dict):
        raise ValueError("CPU baseline runtime contract must be an object")
    expected = {
        "device": "cpu",
        "precision": "float64",
        "compile_policy": "compile",
        "compile_mode": "default",
        "explicit_cuda_graphs": False,
        "execution_policy": "auto",
        "experimental_dispersive_grouping": False,
        "experimental_dispersive_grouping_scope": "combined",
        "threads": expected_threads,
        "interop_threads": 1,
    }
    normalized = {}
    for name in _RUNTIME_CONTRACT_KEYS:
        default = "combined" if name.endswith("grouping_scope") else None
        value = runtime.get(name, default)
        if not _json_equal(value, expected[name]):
            raise ValueError(f"CPU baseline runtime {name} differs from the contract")
        normalized[name] = copy.deepcopy(expected[name])
    return normalized


def _reported_float(value, expected):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and math.isclose(float(value), expected, rel_tol=1e-12, abs_tol=0.0)
    )


def _normalize_advance(advance, benchmark_contract, max_relative_mad):
    if not isinstance(advance, dict):
        raise ValueError("CPU baseline advance measurement must be an object")
    if set(advance) != set(_PUBLIC_ADVANCE_KEYS):
        raise ValueError("CPU baseline advance measurement schema is invalid")
    raw = advance.get("raw_seconds")
    repetitions = benchmark_contract["repetitions"]
    steps = benchmark_contract["steps_per_repeat"]
    if not isinstance(raw, list) or len(raw) != repetitions:
        raise ValueError("CPU baseline advance measurement must have 15 raw samples")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for value in raw
    ):
        raise ValueError("CPU baseline raw samples must be finite positive numbers")
    values = [float(value) for value in raw]
    middle = median(values)
    relative_mad = median(abs(value - middle) for value in values) / middle
    if advance.get("repetitions") != repetitions:
        raise ValueError("CPU baseline reported repetition count is inconsistent")
    if advance.get("steps_per_repeat") != steps:
        raise ValueError("CPU baseline reported step count is inconsistent")
    if not _reported_float(advance.get("median_seconds"), middle):
        raise ValueError("CPU baseline reported median is inconsistent")
    if not _reported_float(advance.get("seconds_per_step"), middle / steps):
        raise ValueError("CPU baseline reported seconds per step is inconsistent")
    if not _reported_float(advance.get("relative_mad"), relative_mad):
        raise ValueError("CPU baseline reported relative MAD is inconsistent")
    if relative_mad > max_relative_mad:
        raise ValueError("CPU baseline raw samples exceed the relative-MAD limit")
    return {
        "raw_seconds": values,
        "raw_seconds_per_step": [value / steps for value in values],
        "median_seconds": middle,
        "seconds_per_step": middle / steps,
        "relative_mad": relative_mad,
        "repetitions": repetitions,
        "steps_per_repeat": steps,
    }


def _nonnegative_integer(mapping, name):
    value = mapping.get(name)
    if not _is_integer(value) or value < 0:
        raise ValueError(f"CPU baseline profiler {name} is invalid")
    return value


def _normalize_profiler(profiler, profile_steps):
    if not isinstance(profiler, dict):
        raise ValueError("CPU baseline profiler record must be an object")
    if set(profiler) != set(_PUBLIC_PROFILER_KEYS):
        raise ValueError("CPU baseline profiler schema is invalid")
    histogram = profiler.get("allocation_size_histogram")
    if not isinstance(histogram, dict):
        raise ValueError("CPU baseline allocation histogram must be an object")
    normalized_histogram = {}
    for text, count in histogram.items():
        if not isinstance(text, str) or not text.isdecimal() or str(int(text)) != text:
            raise ValueError("CPU baseline allocation histogram size is invalid")
        size = int(text)
        if size <= 0 or not _is_integer(count) or count <= 0:
            raise ValueError("CPU baseline allocation histogram entry is invalid")
        if count % profile_steps:
            raise ValueError(
                "CPU baseline allocation counts must be divisible by profile steps"
            )
        normalized_histogram[text] = count
    normalized_histogram = dict(
        sorted(normalized_histogram.items(), key=lambda item: int(item[0]))
    )
    events = _nonnegative_integer(profiler, "positive_allocation_events")
    allocated = _nonnegative_integer(profiler, "allocated_bytes")
    freed = _nonnegative_integer(profiler, "freed_bytes")
    net = profiler.get("allocation_net_bytes")
    maximum = _nonnegative_integer(profiler, "max_allocation_bytes")
    if events != sum(normalized_histogram.values()):
        raise ValueError("CPU baseline allocation event histogram is inconsistent")
    if allocated != sum(
        int(size) * count for size, count in normalized_histogram.items()
    ):
        raise ValueError("CPU baseline allocation byte histogram is inconsistent")
    if not _is_integer(net) or net != allocated - freed or net != 0:
        raise ValueError("CPU baseline allocation net bytes must be exactly zero")
    expected_maximum = max((int(size) for size in normalized_histogram), default=0)
    if maximum != expected_maximum:
        raise ValueError("CPU baseline maximum allocation size is inconsistent")
    return {
        "profile_steps": profile_steps,
        "positive_allocation_events": events,
        "allocated_bytes": allocated,
        "freed_bytes": freed,
        "allocation_net_bytes": net,
        "max_allocation_bytes": maximum,
        "allocation_size_histogram": normalized_histogram,
    }


def _normalize_case(
    result,
    spec,
    benchmark_contract,
    expected_threads,
    max_relative_mad,
    source_sha256,
):
    if not isinstance(result, dict):
        raise ValueError("CPU baseline case must be an object")
    if set(result) != set(_PUBLIC_CASE_KEYS):
        raise ValueError("CPU baseline case schema is invalid")
    if not _json_equal(result.get("workload"), spec):
        raise ValueError(
            f"CPU baseline workload {spec['name']!r} differs from manifest"
        )
    if not _json_equal(result.get("benchmark_contract"), benchmark_contract):
        raise ValueError(
            f"CPU baseline benchmark contract for {spec['name']!r} differs from manifest"
        )
    if not isinstance(result.get("runtime"), dict) or set(result["runtime"]) != set(
        _RUNTIME_CONTRACT_KEYS
    ):
        raise ValueError("CPU baseline runtime schema is invalid")
    runtime = _runtime_contract(result["runtime"], expected_threads)
    controls = result.get("acceptance")
    if (
        not isinstance(controls, dict)
        or set(controls) != set(_RUNTIME_HARD_CONTROLS)
        or any(controls.get(name) is not True for name in _RUNTIME_HARD_CONTROLS)
    ):
        raise ValueError(
            f"CPU baseline runtime controls for {spec['name']!r} are not all true"
        )
    measurements = result.get("measurements")
    if not isinstance(measurements, dict) or set(measurements) != {"advance"}:
        raise ValueError(f"CPU baseline measurements for {spec['name']!r} are missing")
    advance = _normalize_advance(
        measurements.get("advance"), benchmark_contract, max_relative_mad
    )
    memory = result.get("memory")
    if (
        not isinstance(memory, dict)
        or set(memory) != {"bounded"}
        or memory.get("bounded") is not True
    ):
        raise ValueError(f"CPU baseline memory for {spec['name']!r} is not bounded")
    profiler = _normalize_profiler(
        result.get("profiler"), benchmark_contract["profile_steps"]
    )
    return {
        "name": spec["name"],
        "threads": expected_threads,
        "workload": copy.deepcopy(spec),
        "runtime": runtime,
        "benchmark_contract": copy.deepcopy(benchmark_contract),
        "measurements": {"advance": advance},
        "memory": {"bounded": True},
        "profiler": profiler,
        "source_artifact_sha256": source_sha256,
    }


def sanitize_public_baseline_artifact(document, *, host_identity_salt):
    """Project one legacy slice to the minimal public, reproducible schema.

    This intentionally omits raw host topology, process metadata, compiler cache
    keys, and full profiler traces. The host identity becomes a normalized,
    release-scoped SHA-256 commitment that a local candidate can recompute.
    """
    if not isinstance(document, Mapping):
        raise ValueError("CPU baseline artifact must contain a JSON object")
    if document.get("schema_version") != 3:
        raise ValueError("CPU baseline artifact must use schema 3")
    if document.get("kind") != "cpu-acceptance-thread-slice":
        raise ValueError("CPU baseline artifact has the wrong kind")
    evidence = document.get("evidence")
    environment = document.get("environment")
    cases = document.get("cases")
    if not isinstance(evidence, Mapping) or not isinstance(cases, list):
        raise ValueError("CPU baseline artifact is incomplete")
    if not isinstance(environment, Mapping):
        raise ValueError("CPU baseline host identity must be an object")
    thread_environment = environment.get("thread_environment")
    if not isinstance(thread_environment, Mapping):
        raise ValueError("CPU baseline thread environment is incomplete")
    public_cases = []
    for result in cases:
        if not isinstance(result, Mapping):
            raise ValueError("CPU baseline case must be an object")
        runtime = result.get("runtime")
        measurements = result.get("measurements")
        memory = result.get("memory")
        profiler = result.get("profiler")
        controls = result.get("acceptance")
        if not all(
            isinstance(value, Mapping)
            for value in (runtime, measurements, memory, profiler, controls)
        ):
            raise ValueError("CPU baseline case is incomplete")
        advance = measurements.get("advance")
        if not isinstance(advance, Mapping):
            raise ValueError("CPU baseline advance measurement is missing")
        try:
            public_runtime = {
                name: copy.deepcopy(runtime[name]) for name in _RUNTIME_CONTRACT_KEYS
            }
            public_advance = {
                name: copy.deepcopy(advance[name]) for name in _PUBLIC_ADVANCE_KEYS
            }
            public_profiler = {
                name: copy.deepcopy(profiler[name]) for name in _PUBLIC_PROFILER_KEYS
            }
            public_controls = {
                name: copy.deepcopy(controls[name]) for name in _RUNTIME_HARD_CONTROLS
            }
        except KeyError as error:
            raise ValueError(f"CPU baseline case omits {error.args[0]!r}") from error
        public_cases.append(
            {
                "workload": copy.deepcopy(result.get("workload")),
                "benchmark_contract": copy.deepcopy(result.get("benchmark_contract")),
                "runtime": public_runtime,
                "measurements": {"advance": public_advance},
                "memory": {"bounded": memory.get("bounded")},
                "profiler": public_profiler,
                "acceptance": public_controls,
            }
        )
    return {
        "schema_version": 3,
        "kind": "cpu-acceptance-thread-slice",
        "evidence": copy.deepcopy(dict(evidence)),
        "environment": {
            "hostname": _PUBLIC_BASELINE_HOSTNAME,
            "host_identity": _host_identity_token_from_raw_environment(
                environment, host_identity_salt
            ),
            "thread_environment": copy.deepcopy(dict(thread_environment)),
        },
        "cases": public_cases,
    }


def load_torch_cpu_baseline(artifacts, manifest):
    """Validate two legacy thread slices and return the frozen Torch baseline.

    ``artifacts`` must contain exactly two locally downloaded JSON paths. Their
    exact sizes and byte-level SHA-256 digests are checked against the manifest
    and retained in the result. The manifest pins GitHub Release asset URLs and
    SHA-256 digests; this loader deliberately performs no network download.
    Provenance inside each artifact must exactly identify the legacy checkout and
    clean root commit.
    """
    if (
        isinstance(artifacts, (str, bytes, Path))
        or not isinstance(artifacts, Sequence)
        or len(artifacts) != 2
    ):
        raise ValueError("Torch CPU baseline requires exactly two artifact paths")
    paths = [Path(value) for value in artifacts]
    if paths[0].resolve() == paths[1].resolve():
        raise ValueError("Torch CPU baseline artifact paths must be distinct")
    manifest_data = _manifest_value(manifest)
    contract = _manifest_contract(manifest_data)
    sources = []
    common_environment = None

    for path in paths:
        output, raw = _read_json(path, "CPU baseline artifact")
        digest = hashlib.sha256(raw).hexdigest()
        if set(output) != {
            "schema_version",
            "kind",
            "evidence",
            "environment",
            "cases",
        }:
            raise ValueError(f"CPU baseline artifact {path} public schema is invalid")
        if output.get("schema_version") != 3:
            raise ValueError(f"CPU baseline artifact {path} must use schema 3")
        if output.get("kind") != "cpu-acceptance-thread-slice":
            raise ValueError(f"CPU baseline artifact {path} has the wrong kind")
        if not _json_equal(output.get("evidence"), contract["evidence"]):
            raise ValueError(f"CPU baseline artifact {path} evidence is not exact")

        results = output.get("cases")
        if not isinstance(results, list) or len(results) != _EXPECTED_CASE_COUNT:
            raise ValueError(f"CPU baseline artifact {path} must contain six cases")
        names = [
            (
                result.get("workload", {}).get("name")
                if isinstance(result, dict) and isinstance(result.get("workload"), dict)
                else None
            )
            for result in results
        ]
        if tuple(names) != contract["case_names"]:
            raise ValueError(f"CPU baseline artifact {path} has the wrong case order")
        runtime_threads = {
            (
                result.get("runtime", {}).get("threads")
                if isinstance(result, dict) and isinstance(result.get("runtime"), dict)
                else None
            )
            for result in results
        }
        if len(runtime_threads) != 1:
            raise ValueError(f"CPU baseline artifact {path} mixes thread counts")
        threads = next(iter(runtime_threads))
        environment = _validate_host_identity(output.get("environment"), threads)
        host_identity = {
            "hostname": environment["hostname"],
            "host_identity": copy.deepcopy(environment["host_identity"]),
        }
        if common_environment is None:
            common_environment = host_identity
        elif not _json_equal(host_identity, common_environment):
            raise ValueError(
                "Torch CPU baseline artifacts have different host identity"
            )
        if threads == contract["slice_artifacts"]["one"]["threads"] and _is_integer(
            threads
        ):
            thread_mode = "one"
        elif threads == contract["slice_artifacts"]["physical"][
            "threads"
        ] and _is_integer(threads):
            thread_mode = "physical"
        else:
            raise ValueError(
                f"CPU baseline artifact {path} is neither one nor physical threads"
            )
        expected_artifact = contract["slice_artifacts"][thread_mode]
        if len(raw) != expected_artifact["size_bytes"]:
            raise ValueError(
                f"CPU baseline artifact {path} byte size does not match the manifest"
            )
        if digest != expected_artifact["sha256"]:
            raise ValueError(
                f"CPU baseline artifact {path} SHA-256 does not match the manifest"
            )
        if threads != expected_artifact["threads"]:
            raise ValueError(
                f"CPU baseline artifact {path} thread count does not match its pin"
            )
        normalized_cases = [
            _normalize_case(
                result,
                spec,
                contract["benchmark_contract"],
                threads,
                contract["max_relative_mad"],
                digest,
            )
            for result, spec in zip(results, contract["specs"], strict=True)
        ]
        sources.append(
            {
                "publication_url": expected_artifact["publication_url"],
                "size_bytes": len(raw),
                "sha256": digest,
                "thread_mode": thread_mode,
                "threads": threads,
                "thread_environment": copy.deepcopy(environment["thread_environment"]),
                "root_commit": contract["timing_reference"]["root_commit"],
                "evidence": copy.deepcopy(contract["evidence"]),
                "cpu_acceptance_contract_id": contract["acceptance"]["contract_id"],
                "benchmark_contract": copy.deepcopy(contract["benchmark_contract"]),
                "cases": normalized_cases,
            }
        )

    modes = {source["thread_mode"] for source in sources}
    if modes != {"one", "physical"}:
        raise ValueError("Torch CPU baseline requires one and physical thread slices")
    sources.sort(key=lambda source: (source["thread_mode"] != "one", source["threads"]))
    return {
        "schema_version": 1,
        "kind": "torch-cpu-baseline",
        "cpu_acceptance_contract_id": contract["acceptance"]["contract_id"],
        "timing_reference": copy.deepcopy(contract["timing_reference"]),
        "max_individual_ratio": contract["max_individual_ratio"],
        "max_relative_mad": contract["max_relative_mad"],
        "environment": common_environment,
        "source_artifacts": sources,
    }


def find_baseline_case(baseline, name, threads):
    """Return one normalized baseline case by workload name and thread count."""
    if not isinstance(name, str) or not name or not _is_integer(threads):
        raise KeyError((name, threads))
    matches = [
        case
        for source in baseline.get("source_artifacts", ())
        if isinstance(source, dict)
        for case in source.get("cases", ())
        if isinstance(case, dict)
        and case.get("name") == name
        and case.get("threads") == threads
    ]
    if len(matches) != 1:
        raise KeyError((name, threads))
    return matches[0]


def compare_candidate_to_baseline(baseline, candidate, name=None, threads=None):
    """Compare one candidate case with its same-host Torch baseline case."""
    errors = []
    reference = None
    reference_raw = []
    candidate_raw = []
    ratio = None

    if not isinstance(candidate, dict):
        errors.append("candidate must be an object")
        candidate = {}
    workload = candidate.get("workload")
    runtime = candidate.get("runtime")
    inferred_name = workload.get("name") if isinstance(workload, dict) else None
    inferred_threads = runtime.get("threads") if isinstance(runtime, dict) else None
    lookup_name = inferred_name if name is None else name
    lookup_threads = inferred_threads if threads is None else threads
    if name is not None and inferred_name != name:
        errors.append("candidate workload name differs from requested baseline")
    if threads is not None and inferred_threads != threads:
        errors.append("candidate thread count differs from requested baseline")
    try:
        reference = find_baseline_case(baseline, lookup_name, lookup_threads)
    except AttributeError, KeyError, TypeError:
        errors.append("matching Torch CPU baseline case was not found")

    if reference is not None:
        reference_raw = list(
            reference["measurements"]["advance"]["raw_seconds_per_step"]
        )
        if not _json_equal(workload, reference["workload"]):
            errors.append("candidate workload differs from the Torch baseline")
        try:
            candidate_runtime = _runtime_contract(runtime, reference["threads"])
        except (KeyError, TypeError, ValueError) as error:
            errors.append(str(error))
        else:
            if not _json_equal(candidate_runtime, reference["runtime"]):
                errors.append("candidate runtime differs from the Torch baseline")
        benchmark_contract = candidate.get("benchmark_contract")
        if not _json_equal(benchmark_contract, reference["benchmark_contract"]):
            errors.append(
                "candidate benchmark contract differs from the Torch baseline"
            )
        else:
            try:
                measurements = candidate.get("measurements")
                if not isinstance(measurements, dict):
                    raise ValueError("candidate measurements are missing")
                advance = _normalize_advance(
                    measurements.get("advance"),
                    reference["benchmark_contract"],
                    baseline["max_relative_mad"],
                )
                candidate_raw = advance["raw_seconds_per_step"]
            except (KeyError, TypeError, ValueError) as error:
                errors.append(str(error).replace("CPU baseline", "candidate"))
    if reference_raw and candidate_raw:
        ratio = median(candidate_raw) / median(reference_raw)
    limit = baseline.get("max_individual_ratio")
    valid = not errors and ratio is not None and math.isfinite(ratio)
    return {
        "comparison_valid": valid,
        "contract_errors": errors,
        "reference_source_artifact_sha256": (
            reference.get("source_artifact_sha256") if reference is not None else None
        ),
        "reference_root_commit": baseline.get("timing_reference", {}).get(
            "root_commit"
        ),
        "reference_raw_seconds_per_step": reference_raw,
        "candidate_raw_seconds_per_step": candidate_raw,
        "reference_seconds_per_step": median(reference_raw) if reference_raw else None,
        "candidate_seconds_per_step": median(candidate_raw) if candidate_raw else None,
        "candidate_to_torch_baseline_ratio": ratio,
        "individual_ratio_limit": limit,
        "within_five_percent": (
            valid
            and not isinstance(limit, bool)
            and isinstance(limit, (int, float))
            and ratio <= float(limit)
        ),
    }
