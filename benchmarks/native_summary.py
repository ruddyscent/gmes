#!/usr/bin/env python3
"""Assemble the frozen twelve-cell native CPU performance summary."""

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import median, pstdev

ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "native_oracle_workloads.json"
CASE_NAMES = (
    "cpu-crossover-2d",
    "cpu-crossover-3d",
    "cpu-large-2d",
    "cpu-large-3d",
    "bloch-2d",
    "bloch-3d",
)
THREAD_MODES = ("one", "physical")
CELL_KEYS = {
    "schema_version",
    "backend",
    "workload",
    "benchmark_contract",
    "environment",
    "measurements",
    "memory",
    "updaters",
    "profiler",
}
ENVIRONMENT_KEYS = {
    "platform",
    "hostname",
    "os",
    "python",
    "python_executable",
    "numpy",
    "gmes_version",
    "gmes_source",
    "native_extension",
    "git_commit",
    "git_status",
    "uv_lock_sha256",
    "python_compiler",
    "python_build_cflags",
    "cxx_version",
    "swig_version",
    "extension_compile_standard",
    "build_environment",
    "openmp_enabled",
    "openmp_threads",
    "omp_num_threads",
    "cpu_count_logical",
    "cpu_count_physical",
    "cpu_topology",
    "cpu_model",
    "memory_bytes",
    "gpu",
    "gpu_topology",
    "torch",
}
BUILD_ENVIRONMENT_KEYS = {
    "CC",
    "CXX",
    "CFLAGS",
    "CXXFLAGS",
    "LDFLAGS",
    "GMES_ENABLE_OPENMP",
    "GMES_OPENMP_PREFIX",
    "MACOSX_DEPLOYMENT_TARGET",
}
OS_KEYS = {"system", "node", "release", "version", "machine", "processor"}
MEASUREMENT_KEYS = {
    "construction",
    "geometry_mapping",
    "native_initialization_and_plan_lowering",
    "host_to_device_transfer",
    "eager_warmup_seconds",
    "cold_compile",
    "cached_compile",
    "one_step",
    "advance",
}
TIMING_KEYS = {
    "raw_seconds",
    "median_seconds",
    "p95_seconds",
    "population_stdev_seconds",
    "relative_mad",
    "repetitions",
}
MEMORY_KEYS = {
    "peak_rss_bytes",
    "rss_samples_bytes",
    "rss_growth_bytes",
    "live_field_bytes",
    "live_plan_bytes",
    "live_index_bytes",
    "live_parameter_bytes",
    "live_updater_bytes",
    "live_state_bytes",
    "cuda_allocated_peak_bytes",
    "cuda_reserved_peak_bytes",
}
UPDATER_KEYS = {
    "component",
    "strategy",
    "strategies",
    "native_type",
    "cells",
    "coverage",
    "fragmentation_runs",
    "fragmentation_ratio",
    "state_values",
    "state_nonzero_values",
    "state_width",
    "state_key",
    "state_bytes",
    "plan_bytes",
    "index_bytes",
    "parameter_bytes",
    "live_updater_bytes",
    "plan_runs",
    "bucket_signature",
}
COMPONENT_NAMES = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
GPU_TOPOLOGY_NORMALIZATION_RULE = "nvidia-smi-topology-underline-sgr-v1"
GPU_TOPOLOGY_NORMALIZATION_PATH = "/environment/gpu_topology"
GPU_TOPOLOGY_UNDERLINE_ON = "\x1b[4m"
GPU_TOPOLOGY_UNDERLINE_OFF = "\x1b[0m"


def _reject_json_constant(value):
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _object_without_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is not allowed: {key!r}")
        result[key] = value
    return result


def _reject_ansi(value, location="$"):
    if isinstance(value, str):
        if "\x1b" in value or "\x9b" in value:
            raise ValueError(f"ANSI escape data is not allowed at {location}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_ansi(item, f"{location}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _reject_ansi(key, f"{location}.<key>")
            _reject_ansi(item, f"{location}.{key}")


def _normalize_gpu_topology_sgr(value):
    """Remove only balanced nvidia-smi header underline SGR tokens."""

    if value is None:
        return None, {"applied": False, "removed_pair_count": 0}
    if not isinstance(value, str):
        raise ValueError("native GPU topology must be text or null")
    normalized = []
    index = 0
    underlined = False
    pair_count = 0
    while index < len(value):
        if value.startswith(GPU_TOPOLOGY_UNDERLINE_ON, index):
            if underlined:
                raise ValueError("nested GPU topology underline SGR is not allowed")
            underlined = True
            index += len(GPU_TOPOLOGY_UNDERLINE_ON)
            continue
        if value.startswith(GPU_TOPOLOGY_UNDERLINE_OFF, index):
            if not underlined:
                raise ValueError("unpaired GPU topology reset SGR is not allowed")
            underlined = False
            pair_count += 1
            index += len(GPU_TOPOLOGY_UNDERLINE_OFF)
            continue
        character = value[index]
        codepoint = ord(character)
        if character == "\x1b" or character == "\x9b":
            raise ValueError("unsupported ANSI data in native GPU topology")
        if (codepoint < 0x20 and character not in "\t\n\r") or (
            0x7F <= codepoint <= 0x9F
        ):
            raise ValueError("unsupported control data in native GPU topology")
        normalized.append(character)
        index += 1
    if underlined:
        raise ValueError("unterminated GPU topology underline SGR is not allowed")
    return "".join(normalized), {
        "applied": pair_count > 0,
        "removed_pair_count": pair_count,
    }


def _load_json(path, *, normalize_gpu_topology=False):
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path}: JSON must be UTF-8") from error
    if text.startswith("\ufeff"):
        raise ValueError(f"{path}: a UTF-8 BOM is not allowed")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error.msg}") from error
    normalization = {"applied": False, "removed_pair_count": 0}
    if normalize_gpu_topology:
        if not isinstance(value, dict) or not isinstance(
            value.get("environment"), dict
        ):
            raise ValueError(f"{path}: native benchmark environment is absent")
        topology, normalization = _normalize_gpu_topology_sgr(
            value["environment"].get("gpu_topology")
        )
        value["environment"]["gpu_topology"] = topology
    _reject_ansi(value)
    return value, raw, normalization


def _is_int(value):
    return type(value) is int


def _is_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _same_json_value(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _same_json_value(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same_json_value(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    if isinstance(expected, float):
        return actual == expected and math.copysign(1.0, actual) == math.copysign(
            1.0, expected
        )
    return actual == expected


def _require_exact_keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} schema is invalid")


def _validate_digest(value, label):
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a full lowercase Git commit")


def _load_manifest(path):
    manifest, _, _ = _load_json(path)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 2:
        raise ValueError("unsupported native oracle workload schema")
    try:
        reference = manifest["reference"]
        acceptance = manifest["performance_gates"]["cpu_acceptance"]
    except KeyError as error:
        raise ValueError("manifest lacks the native CPU acceptance contract") from error
    if acceptance.get("cases") != list(CASE_NAMES):
        raise ValueError("manifest CPU acceptance cases are not the frozen six cases")
    if acceptance.get("thread_modes") != list(THREAD_MODES):
        raise ValueError("manifest CPU thread modes are not one/physical")
    if acceptance.get("precision") != "float64":
        raise ValueError("manifest native CPU precision must be float64")
    try:
        slice_artifacts = acceptance["timing_reference"]["slice_artifacts"]
        max_relative_mad = acceptance["statistics"]["max_relative_mad"]
    except (KeyError, TypeError) as error:
        raise ValueError("manifest lacks frozen CPU evidence controls") from error
    if not isinstance(slice_artifacts, list) or len(slice_artifacts) != len(
        THREAD_MODES
    ):
        raise ValueError("manifest CPU thread artifact pins are invalid")
    expected_threads = {}
    for artifact, mode in zip(slice_artifacts, THREAD_MODES, strict=True):
        if (
            not isinstance(artifact, dict)
            or artifact.get("thread_mode") != mode
            or not _is_int(artifact.get("threads"))
            or artifact["threads"] < 1
        ):
            raise ValueError("manifest CPU thread artifact pins are invalid")
        expected_threads[mode] = artifact["threads"]
    if expected_threads != {"one": 1, "physical": 4}:
        raise ValueError("manifest CPU thread artifact pins are not frozen")
    if not _is_number(max_relative_mad) or max_relative_mad != 0.05:
        raise ValueError("manifest native relative-MAD limit is not frozen")
    _validate_digest(reference.get("commit"), "physics reference commit")
    _validate_digest(reference.get("observer_commit"), "correctness observer commit")
    _validate_digest(
        reference.get("performance_observer_commit"), "performance observer commit"
    )
    if not isinstance(reference.get("tag"), str) or not reference["tag"]:
        raise ValueError("physics reference tag is invalid")
    if (
        not isinstance(reference.get("observer_tag"), str)
        or not reference["observer_tag"]
    ):
        raise ValueError("observer tag is invalid")
    if (
        not isinstance(reference.get("performance_observer_tag"), str)
        or not reference["performance_observer_tag"]
    ):
        raise ValueError("performance observer tag is invalid")
    cases = {
        case.get("name"): case
        for group in ("correctness", "benchmarks", "physical_checks")
        for case in manifest.get(group, ())
        if isinstance(case, dict)
    }
    if any(name not in cases for name in CASE_NAMES):
        raise ValueError("manifest lacks a frozen native CPU workload")
    contract = {
        "initializer": reference.get("field_initializer"),
        "seed": reference.get("seed"),
        "field_scale": reference.get("field_scale"),
        "warmup_steps": reference.get("performance_warmup_steps"),
        "steps_per_repeat": reference.get("performance_steps_per_repeat"),
        "repetitions": reference.get("performance_repetitions"),
        "timer": "time.perf_counter",
        "sample_start": "independently-rebuilt-post-warmup-state",
    }
    if (
        not isinstance(contract["initializer"], str)
        or not contract["initializer"]
        or not _is_int(contract["seed"])
        or type(contract["field_scale"]) is not float
        or not math.isfinite(contract["field_scale"])
        or contract["field_scale"] <= 0
        or any(
            not _is_int(contract[name]) or contract[name] < 1
            for name in ("warmup_steps", "steps_per_repeat", "repetitions")
        )
    ):
        raise ValueError("manifest native benchmark contract is invalid")
    return manifest, reference, cases, contract, expected_threads, max_relative_mad


def _percentile95(values):
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.95
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _matches_reported(actual, expected):
    return _is_number(actual) and math.isclose(
        actual, expected, rel_tol=1e-12, abs_tol=0.0
    )


def _validate_timing_summary(summary, repetitions, label, *, positive=True):
    _require_exact_keys(summary, TIMING_KEYS, label)
    raw = summary["raw_seconds"]
    if (
        not isinstance(raw, list)
        or len(raw) != repetitions
        or any(
            not _is_number(value) or (value <= 0 if positive else value < 0)
            for value in raw
        )
        or not _is_int(summary["repetitions"])
        or summary["repetitions"] != repetitions
    ):
        raise ValueError(f"{label} raw sample contract is invalid")
    middle = median(raw)
    deviation = median(abs(value - middle) for value in raw)
    expected = {
        "median_seconds": middle,
        "p95_seconds": _percentile95(raw),
        "population_stdev_seconds": pstdev(raw),
        "relative_mad": deviation / middle if middle else 0.0,
    }
    if any(
        not _matches_reported(summary[name], value) for name, value in expected.items()
    ):
        raise ValueError(f"{label} reported statistics do not match raw samples")
    return expected


def _validate_measurements(measurements, contract, max_relative_mad, active_cells):
    _require_exact_keys(measurements, MEASUREMENT_KEYS, "native measurements")
    repetitions = contract["repetitions"]
    for name in (
        "construction",
        "geometry_mapping",
        "native_initialization_and_plan_lowering",
        "one_step",
    ):
        _validate_timing_summary(
            measurements[name], repetitions, f"native {name} timing"
        )
    advance = measurements["advance"]
    _require_exact_keys(
        advance,
        TIMING_KEYS | {"steps_per_repeat", "steps_per_second", "cells_per_second"},
        "native advance timing",
    )
    advance_statistics = _validate_timing_summary(
        {name: advance[name] for name in TIMING_KEYS},
        repetitions,
        "native advance timing",
    )
    if advance_statistics["relative_mad"] > max_relative_mad:
        raise ValueError("native advance samples exceed the relative-MAD limit")
    if (
        not _is_int(advance["steps_per_repeat"])
        or advance["steps_per_repeat"] != contract["steps_per_repeat"]
        or not _matches_reported(
            advance["steps_per_second"],
            contract["steps_per_repeat"] / median(advance["raw_seconds"]),
        )
        or not _matches_reported(
            advance["cells_per_second"],
            active_cells
            * contract["steps_per_repeat"]
            / median(advance["raw_seconds"]),
        )
    ):
        raise ValueError("native advance timing contract is invalid")
    transfer = measurements["host_to_device_transfer"]
    _require_exact_keys(
        transfer,
        {"raw_seconds", "median_seconds", "p95_seconds"},
        "native transfer timing",
    )
    transfer_raw = transfer["raw_seconds"]
    if (
        not isinstance(transfer_raw, list)
        or len(transfer_raw) != repetitions
        or any(type(value) is not float or value != 0.0 for value in transfer_raw)
        or type(transfer["median_seconds"]) is not float
        or transfer["median_seconds"] != 0.0
        or type(transfer["p95_seconds"]) is not float
        or transfer["p95_seconds"] != 0.0
    ):
        raise ValueError("native host-to-device transfer must be exactly zero")
    if (
        not _is_number(measurements["eager_warmup_seconds"])
        or measurements["eager_warmup_seconds"] <= 0
        or measurements["cold_compile"] is not None
        or measurements["cached_compile"] is not None
    ):
        raise ValueError("native warmup/compile measurement schema is invalid")


def _validate_updaters(updaters):
    if not isinstance(updaters, list) or not updaters:
        raise ValueError("native updater evidence is absent")
    state_keys = set()
    normalized = []
    for record in updaters:
        _require_exact_keys(record, UPDATER_KEYS, "native updater")
        component = record["component"]
        strategies = record["strategies"]
        if (
            component not in COMPONENT_NAMES
            or not isinstance(strategies, list)
            or not strategies
            or any(not isinstance(value, str) or not value for value in strategies)
            or record["strategy"] != "+".join(strategies)
            or not isinstance(record["native_type"], str)
            or not record["native_type"]
        ):
            raise ValueError("native updater identity is invalid")
        integer_names = (
            "cells",
            "fragmentation_runs",
            "state_values",
            "state_nonzero_values",
            "state_bytes",
            "plan_bytes",
            "index_bytes",
            "parameter_bytes",
            "live_updater_bytes",
            "plan_runs",
        )
        if any(not _is_int(record[name]) or record[name] < 0 for name in integer_names):
            raise ValueError("native updater integer accounting is invalid")
        cells = record["cells"]
        state_values = record["state_values"]
        runs = record["fragmentation_runs"]
        if (
            record["state_nonzero_values"] > state_values
            or runs > cells
            or not _is_number(record["coverage"])
            or not 0.0 <= record["coverage"] <= 1.0
            or not _matches_reported(
                record["fragmentation_ratio"], runs / cells if cells else 0.0
            )
            or not _matches_reported(
                record["state_width"], state_values / cells if cells else 0.0
            )
            or record["live_updater_bytes"]
            != record["plan_bytes"] + record["index_bytes"] + record["parameter_bytes"]
        ):
            raise ValueError("native updater numeric accounting is invalid")
        ordinal = sum(item["component"] == component for item in normalized)
        expected_state_key = (
            f"step/benchmark/state/{component}/{ordinal}-"
            f"{record['strategy']}/values"
        )
        signature = [
            component,
            record["strategy"],
            record["native_type"],
            cells,
            state_values,
        ]
        if (
            record["state_key"] != expected_state_key
            or record["state_key"] in state_keys
            or record["bucket_signature"] != signature
        ):
            raise ValueError("native updater state key or signature is invalid")
        state_keys.add(record["state_key"])
        normalized.append(record)
    return normalized


def _validate_memory(memory, repetitions, updaters):
    _require_exact_keys(memory, MEMORY_KEYS, "native memory")
    rss = memory["rss_samples_bytes"]
    if (
        not _is_int(memory["peak_rss_bytes"])
        or memory["peak_rss_bytes"] <= 0
        or not isinstance(rss, list)
        or len(rss) != repetitions + 1
        or any(not _is_int(value) or value <= 0 for value in rss)
        or not _is_int(memory["rss_growth_bytes"])
        or memory["rss_growth_bytes"] != rss[-1] - rss[0]
        or memory["peak_rss_bytes"] < max(rss)
    ):
        raise ValueError("native RSS evidence is invalid")
    for name in (
        "live_field_bytes",
        "live_plan_bytes",
        "live_index_bytes",
        "live_parameter_bytes",
        "live_updater_bytes",
        "live_state_bytes",
    ):
        if not _is_int(memory[name]) or memory[name] < 0:
            raise ValueError(f"native {name} is invalid")
    sums = {
        "live_plan_bytes": sum(record["plan_bytes"] for record in updaters),
        "live_index_bytes": sum(record["index_bytes"] for record in updaters),
        "live_parameter_bytes": sum(record["parameter_bytes"] for record in updaters),
        "live_updater_bytes": sum(record["live_updater_bytes"] for record in updaters),
        "live_state_bytes": sum(record["state_bytes"] for record in updaters),
    }
    if (
        memory["live_field_bytes"] <= 0
        or any(memory[name] != value for name, value in sums.items())
        or memory["live_updater_bytes"]
        != memory["live_plan_bytes"]
        + memory["live_index_bytes"]
        + memory["live_parameter_bytes"]
        or memory["live_state_bytes"] > memory["live_parameter_bytes"]
    ):
        raise ValueError("native live-memory accounting is inconsistent")
    if (
        memory["cuda_allocated_peak_bytes"] is not None
        or memory["cuda_reserved_peak_bytes"] is not None
    ):
        raise ValueError("native CPU evidence must not contain CUDA memory values")


def _normalized_cpu_model(value):
    if not isinstance(value, str) or not value:
        raise ValueError("native CPU model identity is invalid")
    return "\n".join(
        line
        for line in value.splitlines()
        if not line.lstrip().startswith("CPU(s) scaling MHz:")
    )


def _validate_environment(environment, observer_commit, expected_threads):
    _require_exact_keys(environment, ENVIRONMENT_KEYS, "native environment")
    _require_exact_keys(
        environment["build_environment"],
        BUILD_ENVIRONMENT_KEYS,
        "native build environment",
    )
    _require_exact_keys(environment["os"], OS_KEYS, "native OS environment")
    torch = environment["torch"]
    if torch is not None:
        _require_exact_keys(torch, {"version", "cuda_build"}, "native Torch metadata")
    if environment["git_commit"] != observer_commit or environment["git_status"] != "":
        raise ValueError("native observer commit is mismatched or checkout is dirty")
    for name in (
        "platform",
        "hostname",
        "python",
        "python_executable",
        "numpy",
        "gmes_version",
        "gmes_source",
        "native_extension",
        "python_compiler",
        "cxx_version",
        "swig_version",
        "extension_compile_standard",
        "cpu_topology",
        "cpu_model",
    ):
        if not isinstance(environment[name], str) or not environment[name]:
            raise ValueError(f"native environment {name} is empty")
    if (
        not isinstance(environment["uv_lock_sha256"], str)
        or len(environment["uv_lock_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in environment["uv_lock_sha256"]
        )
        or any(
            value is not None and not isinstance(value, str)
            for value in environment["build_environment"].values()
        )
        or any(not isinstance(value, str) for value in environment["os"].values())
        or not _is_int(environment["cpu_count_logical"])
        or environment["cpu_count_logical"] < 1
        or not _is_int(environment["memory_bytes"])
        or environment["memory_bytes"] < 1
        or not isinstance(environment["gpu"], list)
        or any(not isinstance(value, str) for value in environment["gpu"])
        or (
            environment["gpu_topology"] is not None
            and not isinstance(environment["gpu_topology"], str)
        )
        or (
            environment["python_build_cflags"] is not None
            and not isinstance(environment["python_build_cflags"], str)
        )
    ):
        raise ValueError("native environment identity is invalid")
    if torch is not None and (
        not isinstance(torch["version"], str)
        or not torch["version"]
        or (
            torch["cuda_build"] is not None and not isinstance(torch["cuda_build"], str)
        )
    ):
        raise ValueError("native Torch environment identity is invalid")
    if environment["openmp_enabled"] is not True:
        raise ValueError("native evidence was not built with OpenMP")
    threads = environment["openmp_threads"]
    physical = environment["cpu_count_physical"]
    if (
        not _is_int(threads)
        or threads < 1
        or not _is_int(physical)
        or physical <= 1
        or environment["omp_num_threads"] != str(threads)
    ):
        raise ValueError("native thread environment is invalid")
    if threads == 1:
        mode = "one"
    elif threads == physical:
        mode = "physical"
    else:
        raise ValueError("native thread count is neither one nor physical cores")
    if threads != expected_threads[mode]:
        raise ValueError("native thread count differs from its frozen artifact pin")
    normalized = copy.deepcopy(environment)
    del normalized["openmp_threads"]
    del normalized["omp_num_threads"]
    normalized["cpu_model"] = _normalized_cpu_model(normalized["cpu_model"])
    return mode, threads, normalized


def _validate_cell(
    cell, reference, cases, contract, expected_threads, max_relative_mad
):
    _require_exact_keys(cell, CELL_KEYS, "native benchmark cell")
    if cell["schema_version"] != 2 or cell["backend"] != "native":
        raise ValueError("native benchmark cell schema/backend is invalid")
    workload = cell["workload"]
    if not isinstance(workload, dict) or workload.get("name") not in CASE_NAMES:
        raise ValueError("native benchmark cell has an unknown CPU workload")
    name = workload["name"]
    if not _same_json_value(workload, cases[name]):
        raise ValueError(f"native workload {name!r} differs from the frozen manifest")
    if not _same_json_value(cell["benchmark_contract"], contract):
        raise ValueError("native benchmark contract differs from the frozen manifest")
    mode, threads, normalized_environment = _validate_environment(
        cell["environment"], reference["performance_observer_commit"], expected_threads
    )
    updaters = _validate_updaters(cell["updaters"])
    active_cells = sum(record["cells"] for record in updaters)
    if active_cells <= 0:
        raise ValueError("native updater evidence has no active cells")
    _validate_measurements(
        cell["measurements"], contract, max_relative_mad, active_cells
    )
    _validate_memory(cell["memory"], contract["repetitions"], updaters)
    if cell["profiler"] is not None:
        raise ValueError("native CPU benchmark profiler field must be null")
    return name, mode, threads, normalized_environment


def assemble_summary(inputs, manifest_path=DEFAULT_MANIFEST):
    """Validate and deterministically assemble twelve native benchmark cells."""
    paths = [Path(path) for path in inputs]
    if len(paths) != len(CASE_NAMES) * len(THREAD_MODES):
        raise ValueError("exactly twelve native benchmark cell inputs are required")
    (
        manifest,
        reference,
        cases,
        contract,
        expected_threads,
        max_relative_mad,
    ) = _load_manifest(manifest_path)
    del manifest
    cells = {}
    common_environment = None
    for path in paths:
        cell, raw, normalization = _load_json(path, normalize_gpu_topology=True)
        if not isinstance(cell, dict):
            raise ValueError(f"{path}: native benchmark cell must be an object")
        name, mode, threads, environment = _validate_cell(
            cell,
            reference,
            cases,
            contract,
            expected_threads,
            max_relative_mad,
        )
        key = (name, mode)
        if key in cells:
            raise ValueError(f"duplicate native benchmark cell: {name}/{mode}")
        if common_environment is None:
            common_environment = environment
        elif not _same_json_value(environment, common_environment):
            raise ValueError(
                "native benchmark cells were not captured in one environment"
            )
        cells[key] = {
            "cell": cell,
            "threads": threads,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "normalization": normalization,
        }
    expected = {(name, mode) for name in CASE_NAMES for mode in THREAD_MODES}
    if set(cells) != expected:
        missing = sorted(expected - set(cells))
        unexpected = sorted(set(cells) - expected)
        raise ValueError(
            f"native benchmark cell matrix is incomplete: "
            f"missing={missing}, unexpected={unexpected}"
        )
    samples = []
    source_artifacts = []
    for name in CASE_NAMES:
        for mode in THREAD_MODES:
            item = cells[(name, mode)]
            cell = item["cell"]
            samples.append(
                {
                    "workload": copy.deepcopy(cell["workload"]),
                    "threads": str(item["threads"]),
                    "openmp_threads": item["threads"],
                    "benchmark_contract": copy.deepcopy(cell["benchmark_contract"]),
                    "measurements": copy.deepcopy(cell["measurements"]),
                    "memory": copy.deepcopy(cell["memory"]),
                    "updaters": copy.deepcopy(cell["updaters"]),
                    "profiler": cell["profiler"],
                }
            )
            source_artifacts.append(
                {
                    "workload": name,
                    "thread_mode": mode,
                    "threads": item["threads"],
                    "sha256": item["sha256"],
                    "normalization_provenance": {
                        "rule_id": GPU_TOPOLOGY_NORMALIZATION_RULE,
                        "json_pointer": GPU_TOPOLOGY_NORMALIZATION_PATH,
                        "source_sha256": item["sha256"],
                        **item["normalization"],
                    },
                    "raw_environment": {
                        "cpu_model": cell["environment"]["cpu_model"],
                        "openmp_threads": cell["environment"]["openmp_threads"],
                        "omp_num_threads": cell["environment"]["omp_num_threads"],
                    },
                }
            )
    return {
        "schema_version": 3,
        "kind": "native-cpu-acceptance-summary",
        "physics_reference": reference["tag"],
        "observer_tag": reference["performance_observer_tag"],
        "observer_commit": reference["performance_observer_commit"],
        "benchmark_contract": copy.deepcopy(contract),
        "environment": common_environment,
        "assembly_contract": {
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
        "source_artifacts": source_artifacts,
        "samples": samples,
    }


def render_summary(summary):
    return json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    parser.add_argument("inputs", type=Path, nargs=12)
    args = parser.parse_args(argv)
    try:
        rendered = render_summary(assemble_summary(args.inputs, args.manifest))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.write_text(rendered, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
