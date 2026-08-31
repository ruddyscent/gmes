#!/usr/bin/env python3
"""Produce candidate-bound macOS package evidence for issue #123."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import tarfile
import zipfile
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib import parse

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "benchmarks" / "native_oracle_workloads.json"

RUNTIME_INDEX_KIND = "issue-123-macos-runtime-evidence"
MACOS_INDEX_KIND = "issue-123-macos-evidence-index"
COMMAND_RECORD_KIND = "issue-123-macos-command-record"
IMPORT_RESULT_KIND = "issue-123-macos-import-result"
SUITE_RESULT_KIND = "issue-123-macos-runtime-suite-result"
MACOS_REQUIRED_JOB = "Python 3.14 / macos-latest"
RUNTIME_ROLES = (
    "wheel-import",
    "wheel-default-suite",
    "wheel-serial-suite",
    "sdist-import",
    "sdist-default-suite",
    "sdist-serial-suite",
)
SUITE_MODES = {
    "wheel-default-suite": "default",
    "wheel-serial-suite": "serial",
    "sdist-default-suite": "default",
    "sdist-serial-suite": "serial",
}
FIELD_NAMES = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
COMMAND_ENVIRONMENT_KEYS = {
    "GMES_ENABLE_OPENMP",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "TORCHINDUCTOR_CACHE_DIR",
}
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
MACOS_WHEEL_RE = re.compile(r"gmes-[^-]+-cp314[^/]*-macosx_[^/]*_arm64\.whl\Z")
PYTHON_314_RE = re.compile(r"3\.14(?:\.\d+)?\Z")
MEDIA_TYPE_JSON = "application/json"
MEDIA_TYPE_ZIP = "application/zip"
MEDIA_TYPE_WHEEL = "application/vnd.python.wheel+zip"
MEDIA_TYPE_GZIP = "application/gzip"
MEDIA_TYPE_TEXT = "text/plain; charset=utf-8"


class EvidenceError(ValueError):
    """The requested evidence is missing, ambiguous, or not candidate-bound."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def _exact_keys(value: Any, required: set[str], label: str) -> None:
    _require(isinstance(value, dict), f"{label} must be an object")
    _require(set(value) == required, f"{label} has an invalid schema")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _strict_json_bytes(raw: bytes, label: str) -> Any:
    try:
        text = raw.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                EvidenceError(f"non-finite JSON number: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"{label} is not one strict UTF-8 JSON document") from error


def _strict_json(path: Path, label: str) -> Any:
    try:
        return _strict_json_bytes(path.read_bytes(), label)
    except OSError as error:
        raise EvidenceError(f"{label} cannot be read") from error


def _strict_json_text(raw: str, label: str) -> Any:
    return _strict_json_bytes(raw.encode("utf-8"), label)


def _type_exact_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _type_exact_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _type_exact_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _sha256(raw)


def _git(repository: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ("git", "-C", str(repository), *arguments),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise EvidenceError(f"git {' '.join(arguments)} failed") from error
    return completed.stdout.strip()


def _candidate_evidence(
    repository: Path,
    manifest: Path,
    candidate_commit: str,
) -> dict[str, str]:
    _require(
        COMMIT_RE.fullmatch(candidate_commit) is not None,
        "candidate commit must be one full lowercase Git object id",
    )
    repository = repository.resolve(strict=True)
    manifest = manifest.resolve(strict=True)
    _require(
        manifest == repository / "benchmarks" / "native_oracle_workloads.json",
        "candidate manifest path differs",
    )
    _require(
        _git(repository, "rev-parse", "HEAD") == candidate_commit,
        "candidate checkout HEAD differs from the PR head",
    )
    status = _git(repository, "status", "--porcelain=v1", "--untracked-files=all")
    _require(status == "", "candidate checkout is not clean")
    return {
        "candidate_git_commit": candidate_commit,
        "candidate_git_status": "",
        "manifest_sha256": _sha256(manifest.read_bytes()),
    }


def _media_type(path: Path) -> str:
    name = path.name
    if name.endswith(".tar.gz"):
        return MEDIA_TYPE_GZIP
    if name.endswith(".whl"):
        return MEDIA_TYPE_WHEEL
    if name.endswith(".json"):
        return MEDIA_TYPE_JSON
    if name.endswith(".txt"):
        return MEDIA_TYPE_TEXT
    if name.endswith(".zip"):
        return MEDIA_TYPE_ZIP
    raise EvidenceError(f"evidence path has no canonical media type: {name}")


def _descriptor(
    path: Path,
    base: Path,
    candidate: dict[str, str],
) -> dict[str, Any]:
    path = path.resolve(strict=True)
    base = base.resolve(strict=True)
    _require(path.is_file(), f"evidence path is not a regular file: {path}")
    try:
        relative = path.relative_to(base)
    except ValueError as error:
        raise EvidenceError(
            f"evidence path is outside its index directory: {path}"
        ) from error
    raw = path.read_bytes()
    return {
        "path": relative.as_posix(),
        "sha256": _sha256(raw),
        "size_bytes": len(raw),
        "media_type": _media_type(path),
        "candidate_evidence": candidate,
    }


def _validate_descriptor(
    value: Any,
    base: Path,
    candidate: dict[str, str],
    label: str,
) -> Path:
    _exact_keys(
        value,
        {"path", "sha256", "size_bytes", "media_type", "candidate_evidence"},
        f"{label} descriptor",
    )
    _require(value["candidate_evidence"] == candidate, f"{label} candidate differs")
    _require(
        isinstance(value["path"], str) and bool(value["path"]),
        f"{label} path is empty",
    )
    _require(
        isinstance(value["sha256"], str)
        and SHA256_RE.fullmatch(value["sha256"]) is not None,
        f"{label} digest differs",
    )
    _require(
        type(value["size_bytes"]) is int and value["size_bytes"] >= 0,
        f"{label} byte size differs",
    )
    supplied = Path(value["path"])
    _require(not supplied.is_absolute(), f"{label} path must be relative")
    _require(".." not in supplied.parts, f"{label} path traverses a parent")
    path = (base / supplied).resolve(strict=True)
    _require(
        path.is_relative_to(base.resolve(strict=True)),
        f"{label} path escapes its evidence directory",
    )
    _require(path.is_file(), f"{label} path is not a regular file")
    _require(value["media_type"] == _media_type(path), f"{label} media type differs")
    raw = path.read_bytes()
    _require(len(raw) == value["size_bytes"], f"{label} byte size differs")
    _require(_sha256(raw) == value["sha256"], f"{label} digest differs")
    return path


def _validate_sdist(path: Path) -> None:
    _require(
        path.name.startswith("gmes-") and path.name.endswith(".tar.gz"),
        "source distribution filename differs",
    )
    try:
        with tarfile.open(path, "r:gz") as archive:
            _require(bool(archive.getmembers()), "source distribution is empty")
    except (OSError, tarfile.TarError) as error:
        raise EvidenceError("source distribution is not a readable gzip tar") from error


def _validate_wheel(path: Path) -> None:
    _require(
        MACOS_WHEEL_RE.fullmatch(path.name) is not None,
        "wheel is not a CPython 3.14 macOS arm64 wheel",
    )
    try:
        with zipfile.ZipFile(path) as archive:
            _require(bool(archive.namelist()), "wheel is empty")
            _require(archive.testzip() is None, "wheel CRC check failed")
    except (OSError, zipfile.BadZipFile) as error:
        raise EvidenceError("wheel is not a readable ZIP") from error


def _one_match(directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(path for path in directory.glob(pattern) if path.is_file())
    _require(len(matches) == 1, f"{label} closure requires exactly one file")
    return matches[0]


def _path_is_exact(value: Any, expected: Path) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    if not path.is_absolute():
        return False

    def canonical_system_alias(candidate: Path) -> Path:
        if platform.system() != "Darwin":
            return candidate
        for alias, target in (
            (Path("/var"), Path("/private/var")),
            (Path("/tmp"), Path("/private/tmp")),
        ):
            try:
                return target / candidate.relative_to(alias)
            except ValueError:
                continue
        return candidate

    return canonical_system_alias(path) == canonical_system_alias(expected)


def _platform_record() -> dict[str, str]:
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }


def _probe_argv(
    executable: str,
    repository: Path,
    forbidden_root: Path,
    expected_package: Path,
    role: str,
    mode: str | None,
) -> list[str]:
    argv = [
        executable,
        "-I",
        "-W",
        "error",
        str(repository / "benchmarks" / "macos_ci_evidence.py"),
        "_probe",
        "--role",
        role,
        "--repository",
        str(repository),
        "--forbidden-root",
        str(forbidden_root),
        "--expected-package",
        str(expected_package),
    ]
    if mode is not None:
        argv.extend(("--mode", mode))
    return argv


def _validate_platform(value: Any, label: str) -> None:
    _exact_keys(value, {"system", "machine", "python"}, label)
    _require(
        value["system"] == "Darwin"
        and value["machine"] == "arm64"
        and isinstance(value["python"], str)
        and PYTHON_314_RE.fullmatch(value["python"]) is not None,
        f"{label} differs",
    )


def _validate_host_contract(value: Any, label: str) -> None:
    _exact_keys(value, {"schema_version", "common_identity", "runtime_identity"}, label)
    common = value["common_identity"]
    runtime = value["runtime_identity"]
    _exact_keys(
        common,
        {
            "hostname",
            "platform",
            "os",
            "python",
            "cxx_version",
            "swig_version",
            "uv_lock_sha256",
        },
        f"{label} common identity",
    )
    _exact_keys(common["os"], {"system", "release", "machine"}, f"{label} OS")
    _exact_keys(runtime, {"torch", "cuda_runtime"}, f"{label} runtime identity")
    _require(
        type(value["schema_version"]) is int
        and value["schema_version"] == 2
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
        and isinstance(common["uv_lock_sha256"], str)
        and SHA256_RE.fullmatch(common["uv_lock_sha256"]) is not None
        and all(
            isinstance(common["os"][name], str) and bool(common["os"][name])
            for name in ("system", "release", "machine")
        )
        and isinstance(runtime["torch"], str)
        and bool(runtime["torch"])
        and runtime["cuda_runtime"] is None,
        f"{label} is incomplete or not a CPU host",
    )


def _validate_number(value: Any, label: str) -> None:
    _require(
        type(value) in {int, float} and math.isfinite(value),
        f"{label} is not a finite JSON number",
    )


def _validate_array_record(value: Any, expected_name: str, label: str) -> None:
    _exact_keys(
        value,
        {
            "name",
            "shape",
            "dtype",
            "initial_values",
            "final_values",
            "initial_sha256",
            "final_sha256",
            "finite",
            "changed",
        },
        label,
    )
    _require(value["name"] == expected_name, f"{label} name differs")
    shape = value["shape"]
    _require(
        isinstance(shape, list)
        and all(type(item) is int and item >= 0 for item in shape),
        f"{label} shape differs",
    )
    count = math.prod(shape)
    _require(
        isinstance(value["dtype"], str) and bool(value["dtype"]),
        f"{label} dtype differs",
    )
    for phase in ("initial", "final"):
        values = value[f"{phase}_values"]
        _require(
            isinstance(values, list) and len(values) == count,
            f"{label} {phase} values differ",
        )
        for index, item in enumerate(values):
            _validate_number(item, f"{label} {phase} value {index}")
        expected_sha = _canonical_sha256(
            {"dtype": value["dtype"], "shape": shape, "values": values}
        )
        _require(
            value[f"{phase}_sha256"] == expected_sha,
            f"{label} {phase} digest differs",
        )
    finite = all(
        math.isfinite(item)
        for phase in ("initial_values", "final_values")
        for item in value[phase]
    )
    changed = not _type_exact_equal(value["initial_values"], value["final_values"])
    _require(
        value["finite"] is finite and value["changed"] is changed,
        f"{label} finite/change summary differs",
    )


def _records_sha256(records: list[dict[str, Any]], phase: str) -> str:
    return _canonical_sha256(
        [
            {"name": record["name"], "sha256": record[f"{phase}_sha256"]}
            for record in records
        ]
    )


def _validate_addresses(value: Any, label: str) -> None:
    _require(isinstance(value, dict) and bool(value), f"{label} must be nonempty")
    _require(
        all(
            isinstance(name, str)
            and bool(name)
            and type(address) is int
            and address > 0
            for name, address in value.items()
        ),
        f"{label} differs",
    )


def _validate_counter(value: Any, label: str) -> None:
    _exact_keys(value, {"unique_graphs", "calls_captured", "graph_breaks"}, label)
    _require(
        all(type(value[name]) is int and value[name] >= 0 for name in value),
        f"{label} differs",
    )


def _validate_native_result(value: Any, label: str) -> None:
    _exact_keys(
        value,
        {
            "openmp_enabled",
            "steps",
            "fields",
            "initial_field_sha256",
            "final_field_sha256",
            "storage_addresses",
            "storage_stable",
            "finite",
            "progressed",
            "passed",
        },
        label,
    )
    _exact_keys(value["steps"], {"initial", "final"}, f"{label} steps")
    _require(
        type(value["steps"]["initial"]) is int
        and value["steps"]["initial"] == 0
        and type(value["steps"]["final"]) is int
        and value["steps"]["final"] == 2,
        f"{label} steps differ",
    )
    fields = value["fields"]
    _require(isinstance(fields, list) and len(fields) == 6, f"{label} fields differ")
    for index, (record, name) in enumerate(zip(fields, FIELD_NAMES, strict=True)):
        _validate_array_record(record, name, f"{label} field {index}")
    addresses = value["storage_addresses"]
    _exact_keys(addresses, {"initial", "final"}, f"{label} storage addresses")
    _validate_addresses(addresses["initial"], f"{label} initial addresses")
    _validate_addresses(addresses["final"], f"{label} final addresses")
    finite = all(record["finite"] for record in fields)
    progressed = _records_sha256(fields, "initial") != _records_sha256(fields, "final")
    stable = addresses["initial"] == addresses["final"]
    _require(
        value["openmp_enabled"] is False
        and value["initial_field_sha256"] == _records_sha256(fields, "initial")
        and value["final_field_sha256"] == _records_sha256(fields, "final")
        and value["storage_stable"] is stable
        and value["finite"] is finite
        and value["progressed"] is progressed
        and value["passed"] is (finite and progressed and stable),
        f"{label} summaries differ",
    )


def _validate_torch_mode(value: Any, expected_mode: str, label: str) -> None:
    _exact_keys(
        value,
        {
            "mode",
            "runtime",
            "steps",
            "fields",
            "state_buffers",
            "initial_field_sha256",
            "final_field_sha256",
            "initial_state_sha256",
            "final_state_sha256",
            "storage_addresses",
            "storage_stable",
            "compiler",
            "finite",
            "progressed",
            "passed",
        },
        label,
    )
    runtime = value["runtime"]
    _exact_keys(
        runtime,
        {
            "device",
            "precision",
            "compile_policy",
            "compile_mode",
            "cpu_threads",
            "cpu_interop_threads",
        },
        f"{label} runtime",
    )
    expected_policy = "eager" if expected_mode == "eager" else "compile"
    _require(
        value["mode"] == expected_mode
        and runtime
        == {
            "device": "cpu",
            "precision": "float64",
            "compile_policy": expected_policy,
            "compile_mode": "default",
            "cpu_threads": 1,
            "cpu_interop_threads": 1,
        },
        f"{label} runtime differs",
    )
    _exact_keys(value["steps"], {"initial", "warmup", "final"}, f"{label} steps")
    _require(
        all(type(value["steps"][name]) is int for name in value["steps"])
        and value["steps"] == {"initial": 0, "warmup": 1, "final": 2},
        f"{label} steps differ",
    )
    fields = value["fields"]
    _require(isinstance(fields, list) and len(fields) == 6, f"{label} fields differ")
    for index, (record, name) in enumerate(zip(fields, FIELD_NAMES, strict=True)):
        _validate_array_record(record, name, f"{label} field {index}")
    states = value["state_buffers"]
    _require(isinstance(states, list) and bool(states), f"{label} state is empty")
    state_names = [record.get("name") for record in states if isinstance(record, dict)]
    _require(
        len(state_names) == len(states)
        and state_names == sorted(state_names)
        and len(set(state_names)) == len(state_names),
        f"{label} state names differ",
    )
    for index, (record, name) in enumerate(zip(states, state_names, strict=True)):
        _validate_array_record(record, name, f"{label} state {index}")
    addresses = value["storage_addresses"]
    _exact_keys(
        addresses,
        {"initial", "warmup", "final"},
        f"{label} storage addresses",
    )
    for phase in ("initial", "warmup", "final"):
        _validate_addresses(addresses[phase], f"{label} {phase} addresses")
    compiler = value["compiler"]
    _exact_keys(compiler, {"before", "warmup", "final"}, f"{label} compiler")
    for phase in ("before", "warmup", "final"):
        _validate_counter(compiler[phase], f"{label} compiler {phase}")
    hot_stable = compiler["final"] == compiler["warmup"]
    compiler_clean = (
        compiler["before"]["graph_breaks"] == 0
        and compiler["warmup"]["graph_breaks"] == 0
        and compiler["final"]["graph_breaks"] == 0
        and hot_stable
    )
    if expected_mode == "compile":
        compiler_clean = (
            compiler_clean
            and compiler["warmup"]["unique_graphs"] > 0
            and compiler["warmup"]["calls_captured"] > 0
        )
    else:
        compiler_clean = compiler_clean and all(
            counter == {"unique_graphs": 0, "calls_captured": 0, "graph_breaks": 0}
            for counter in compiler.values()
        )
    finite = all(record["finite"] for record in (*fields, *states))
    progressed = _records_sha256(fields, "initial") != _records_sha256(
        fields, "final"
    ) and _records_sha256(states, "initial") != _records_sha256(states, "final")
    stable = addresses["initial"] == addresses["warmup"] == addresses["final"]
    _require(
        value["initial_field_sha256"] == _records_sha256(fields, "initial")
        and value["final_field_sha256"] == _records_sha256(fields, "final")
        and value["initial_state_sha256"] == _records_sha256(states, "initial")
        and value["final_state_sha256"] == _records_sha256(states, "final")
        and value["storage_stable"] is stable
        and value["finite"] is finite
        and value["progressed"] is progressed
        and value["passed"] is (finite and progressed and stable and compiler_clean),
        f"{label} summaries differ",
    )


def _field_map(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {record["name"]: record for record in value["fields"]}


def _validate_comparison(
    value: Any,
    expected_left: str,
    expected_right: str,
    sources: dict[str, dict[str, Any]],
    label: str,
) -> None:
    _exact_keys(value, {"left", "right", "tolerance", "fields", "passed"}, label)
    _exact_keys(value["tolerance"], {"rtol", "atol"}, f"{label} tolerance")
    rtol = value["tolerance"]["rtol"]
    atol = value["tolerance"]["atol"]
    _validate_number(rtol, f"{label} rtol")
    _validate_number(atol, f"{label} atol")
    _require(rtol >= 0 and atol >= 0, f"{label} tolerance differs")
    _require(
        value["left"] == expected_left and value["right"] == expected_right,
        f"{label} endpoints differ",
    )
    left = _field_map(sources[expected_left])
    right = _field_map(sources[expected_right])
    fields = value["fields"]
    _require(isinstance(fields, list) and len(fields) == 6, f"{label} fields differ")
    comparison_passed = True
    for index, (record, name) in enumerate(zip(fields, FIELD_NAMES, strict=True)):
        _exact_keys(
            record, {"name", "max_abs_error", "passed"}, f"{label} field {index}"
        )
        _require(record["name"] == name, f"{label} field {index} name differs")
        left_values = left[name]["final_values"]
        right_values = right[name]["final_values"]
        _require(len(left_values) == len(right_values), f"{label} {name} shape differs")
        errors = [abs(a - b) for a, b in zip(left_values, right_values, strict=True)]
        maximum = max(errors, default=0.0)
        passed = all(
            error <= atol + rtol * abs(reference)
            for error, reference in zip(errors, right_values, strict=True)
        )
        _require(
            type(record["max_abs_error"]) is float
            and record["max_abs_error"] == float(maximum)
            and record["passed"] is passed,
            f"{label} {name} comparison differs",
        )
        comparison_passed = comparison_passed and passed
    _require(value["passed"] is comparison_passed, f"{label} pass summary differs")


def _validate_probe_result(
    value: Any,
    role: str,
    package_sha256: str,
    platform_record: dict[str, str],
) -> None:
    base_keys = {
        "kind",
        "role",
        "package_sha256",
        "platform",
        "host_contract",
        "passed",
    }
    _require(isinstance(value, dict), f"{role} result must be an object")
    _require(value.get("role") == role, f"{role} result role differs")
    _require(
        value.get("package_sha256") == package_sha256,
        f"{role} result package digest differs",
    )
    _require(
        value.get("platform") == platform_record, f"{role} result platform differs"
    )
    _validate_platform(value.get("platform"), f"{role} result platform")
    _validate_host_contract(value.get("host_contract"), f"{role} host contract")
    if role.endswith("-import"):
        _exact_keys(value, base_keys | {"distribution"}, f"{role} result")
        distribution = value["distribution"]
        _exact_keys(
            distribution,
            {"name", "version", "module_path", "native_module_paths", "outside_source"},
            f"{role} distribution",
        )
        _require(
            value["kind"] == IMPORT_RESULT_KIND
            and distribution["name"] == "gmes"
            and isinstance(distribution["version"], str)
            and bool(distribution["version"])
            and isinstance(distribution["module_path"], str)
            and bool(distribution["module_path"])
            and isinstance(distribution["native_module_paths"], list)
            and len(distribution["native_module_paths"]) == 2
            and all(
                isinstance(path, str) and bool(path)
                for path in distribution["native_module_paths"]
            )
            and distribution["outside_source"] is True
            and value["passed"] is True,
            f"{role} import evidence differs",
        )
        return
    _exact_keys(value, base_keys | {"mode", "native", "torch_cpu"}, f"{role} result")
    _require(
        value["kind"] == SUITE_RESULT_KIND and value["mode"] == SUITE_MODES[role],
        f"{role} suite identity differs",
    )
    _validate_native_result(value["native"], f"{role} native")
    torch_cpu = value["torch_cpu"]
    _exact_keys(torch_cpu, {"modes", "comparisons"}, f"{role} Torch CPU")
    modes = torch_cpu["modes"]
    _require(isinstance(modes, list) and len(modes) == 2, f"{role} Torch modes differ")
    _validate_torch_mode(modes[0], "eager", f"{role} eager")
    _validate_torch_mode(modes[1], "compile", f"{role} compile")
    sources = {"native": value["native"], "eager": modes[0], "compile": modes[1]}
    comparisons = torch_cpu["comparisons"]
    expected = (("eager", "compile"), ("eager", "native"), ("compile", "native"))
    _require(
        isinstance(comparisons, list) and len(comparisons) == len(expected),
        f"{role} comparisons differ",
    )
    for index, ((left, right), comparison) in enumerate(
        zip(expected, comparisons, strict=True)
    ):
        _validate_comparison(
            comparison,
            left,
            right,
            sources,
            f"{role} comparison {index}",
        )
    passed = (
        value["native"]["passed"]
        and all(item["passed"] for item in modes)
        and all(item["passed"] for item in comparisons)
    )
    _require(value["passed"] is passed, f"{role} suite pass summary differs")


def capture_runtime_index(
    evidence_directory: Path,
    records_directory: Path,
    repository: Path,
    manifest: Path,
    candidate_commit: str,
) -> Path:
    """Validate packages plus raw command records and emit a runtime index."""

    evidence_directory = evidence_directory.resolve(strict=True)
    records_directory = records_directory.resolve(strict=True)
    _require(
        not records_directory.is_relative_to(evidence_directory),
        "command records must remain outside the uploaded evidence closure",
    )
    candidate = _candidate_evidence(repository, manifest, candidate_commit)
    platform_record = _platform_record()
    _validate_platform(platform_record, "runtime platform")

    package_directory = evidence_directory / "packages"
    log_directory = evidence_directory / "logs"
    _require(package_directory.is_dir(), "package evidence directory is missing")
    _require(log_directory.is_dir(), "runtime log directory is missing")
    sdist = _one_match(package_directory, "*.tar.gz", "sdist")
    wheel = _one_match(package_directory, "*.whl", "wheel")
    _require(
        {path.name for path in package_directory.iterdir()} == {sdist.name, wheel.name},
        "package evidence contains unexpected entries",
    )
    _validate_sdist(sdist)
    _validate_wheel(wheel)
    package_artifacts = {
        "sdist": _descriptor(sdist, evidence_directory, candidate),
        "wheel-macos-arm64": _descriptor(wheel, evidence_directory, candidate),
    }
    packages = [
        {"role": role, "filename": path.name, "artifact": package_artifacts[role]}
        for role, path in (("sdist", sdist), ("wheel-macos-arm64", wheel))
    ]

    expected_logs = {
        f"{role}.{stream}.{suffix}"
        for role in RUNTIME_ROLES
        for stream, suffix in (("stdout", "json"), ("stderr", "txt"))
    }
    _require(
        {path.name for path in log_directory.iterdir()} == expected_logs,
        "runtime log closure differs",
    )
    runtime_checks = []
    for role in RUNTIME_ROLES:
        package_role = "wheel-macos-arm64" if role.startswith("wheel-") else "sdist"
        package_sha256 = package_artifacts[package_role]["sha256"]
        record_path = records_directory / f"{role}.json"
        record = _strict_json(record_path, f"{role} command record")
        _exact_keys(
            record,
            {
                "schema_version",
                "kind",
                "role",
                "command",
                "exit_code",
                "stdout_sha256",
                "stdout_size_bytes",
                "stderr_sha256",
                "stderr_size_bytes",
                "result",
            },
            f"{role} command record",
        )
        _require(
            type(record["schema_version"]) is int
            and record["schema_version"] == 2
            and record["kind"] == COMMAND_RECORD_KIND
            and record["role"] == role
            and type(record["exit_code"]) is int
            and record["exit_code"] == 0,
            f"{role} command did not succeed",
        )
        command = record["command"]
        _exact_keys(command, {"argv", "cwd", "environment"}, f"{role} command")
        expected_package = wheel if role.startswith("wheel-") else sdist
        expected_argv = _probe_argv(
            sys.executable,
            repository.resolve(strict=True),
            repository.resolve(strict=True),
            expected_package.resolve(strict=True),
            role,
            SUITE_MODES.get(role),
        )
        command_cwd = Path(command.get("cwd", ""))
        _require(
            command["argv"] == expected_argv
            and isinstance(command["cwd"], str)
            and command_cwd.is_absolute()
            and command_cwd.resolve(strict=True) == command_cwd
            and not command_cwd.is_relative_to(repository.resolve(strict=True))
            and not command_cwd.is_relative_to(evidence_directory),
            f"{role} command provenance differs",
        )
        environment = command["environment"]
        _exact_keys(
            environment, COMMAND_ENVIRONMENT_KEYS, f"{role} command environment"
        )
        expected_openmp = "0" if SUITE_MODES.get(role) == "serial" else "auto"
        expected_cache_directory = records_directory / "torchinductor" / role
        _require(
            environment["GMES_ENABLE_OPENMP"] == expected_openmp
            and all(
                environment[name] == "1"
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                )
            )
            and _path_is_exact(
                environment["TORCHINDUCTOR_CACHE_DIR"], expected_cache_directory
            ),
            f"{role} command environment differs",
        )
        stdout = log_directory / f"{role}.stdout.json"
        stderr = log_directory / f"{role}.stderr.txt"
        stdout_raw = stdout.read_bytes()
        stderr_raw = stderr.read_bytes()
        try:
            stderr_raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise EvidenceError(f"{role} stderr is not UTF-8") from error
        _require(
            record["stdout_sha256"] == _sha256(stdout_raw)
            and type(record["stdout_size_bytes"]) is int
            and record["stdout_size_bytes"] == len(stdout_raw)
            and record["stderr_sha256"] == _sha256(stderr_raw)
            and type(record["stderr_size_bytes"]) is int
            and record["stderr_size_bytes"] == len(stderr_raw),
            f"{role} raw command output differs",
        )
        result = _strict_json_bytes(stdout_raw, f"{role} stdout")
        _require(
            _type_exact_equal(result, record["result"]),
            f"{role} command result differs from raw stdout",
        )
        _validate_probe_result(result, role, package_sha256, platform_record)
        runtime_checks.append(
            {
                "role": role,
                "package_sha256": package_sha256,
                "platform": platform_record,
                "command": command,
                "exit_code": record["exit_code"],
                "stdout": _descriptor(stdout, evidence_directory, candidate),
                "stderr": _descriptor(stderr, evidence_directory, candidate),
                "result": result,
            }
        )

    document = {
        "schema_version": 2,
        "kind": RUNTIME_INDEX_KIND,
        "candidate_evidence": candidate,
        "packages": packages,
        "runtime_checks": runtime_checks,
        "passed": True,
    }
    output = evidence_directory / "runtime-index.json"
    output.write_text(
        json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def _candidate_from_runtime(value: Any, manifest: Path) -> dict[str, str]:
    _exact_keys(
        value,
        {"candidate_git_commit", "candidate_git_status", "manifest_sha256"},
        "runtime candidate evidence",
    )
    _require(
        isinstance(value["candidate_git_commit"], str)
        and COMMIT_RE.fullmatch(value["candidate_git_commit"]) is not None
        and value["candidate_git_status"] == ""
        and value["manifest_sha256"]
        == _sha256(manifest.resolve(strict=True).read_bytes()),
        "runtime candidate evidence differs",
    )
    return dict(value)


def _load_runtime_index(
    path: Path,
    manifest: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    path = path.resolve(strict=True)
    document = _strict_json(path, "macOS runtime index")
    _exact_keys(
        document,
        {
            "schema_version",
            "kind",
            "candidate_evidence",
            "packages",
            "runtime_checks",
            "passed",
        },
        "macOS runtime index",
    )
    _require(
        type(document["schema_version"]) is int
        and document["schema_version"] == 2
        and document["kind"] == RUNTIME_INDEX_KIND
        and document["passed"] is True,
        "macOS runtime index identity differs",
    )
    candidate = _candidate_from_runtime(document["candidate_evidence"], manifest)
    packages = document["packages"]
    _require(
        isinstance(packages, list) and len(packages) == 2,
        "macOS package closure differs",
    )
    package_digests: dict[str, str] = {}
    for index, (record, expected_role) in enumerate(
        zip(packages, ("sdist", "wheel-macos-arm64"), strict=True)
    ):
        label = f"macOS package {index}"
        _exact_keys(record, {"role", "filename", "artifact"}, label)
        _require(record["role"] == expected_role, f"{label} role differs")
        package_path = _validate_descriptor(
            record["artifact"], path.parent, candidate, label
        )
        _require(record["filename"] == package_path.name, f"{label} filename differs")
        (_validate_sdist if expected_role == "sdist" else _validate_wheel)(package_path)
        package_digests[expected_role] = record["artifact"]["sha256"]

    checks = document["runtime_checks"]
    _require(
        isinstance(checks, list) and len(checks) == len(RUNTIME_ROLES),
        "macOS runtime check closure differs",
    )
    for index, (check, role) in enumerate(zip(checks, RUNTIME_ROLES, strict=True)):
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
        _require(
            check["role"] == role
            and check["package_sha256"] == package_digests[package_role]
            and type(check["exit_code"]) is int
            and check["exit_code"] == 0,
            f"{label} identity or exit code differs",
        )
        _validate_platform(check["platform"], f"{label} platform")
        _exact_keys(
            check["command"], {"argv", "cwd", "environment"}, f"{label} command"
        )
        _exact_keys(
            check["command"]["environment"],
            COMMAND_ENVIRONMENT_KEYS,
            f"{label} command environment",
        )
        stdout = _validate_descriptor(
            check["stdout"], path.parent, candidate, f"{label} stdout"
        )
        stderr = _validate_descriptor(
            check["stderr"], path.parent, candidate, f"{label} stderr"
        )
        _require(
            check["stdout"]["media_type"] == MEDIA_TYPE_JSON,
            f"{label} stdout media differs",
        )
        _require(
            check["stderr"]["media_type"] == MEDIA_TYPE_TEXT,
            f"{label} stderr media differs",
        )
        result = _strict_json(stdout, f"{label} stdout")
        try:
            stderr.read_bytes().decode("utf-8")
        except UnicodeDecodeError as error:
            raise EvidenceError(f"{label} stderr is not UTF-8") from error
        _require(
            _type_exact_equal(result, check["result"]), f"{label} raw result differs"
        )
        _validate_probe_result(result, role, check["package_sha256"], check["platform"])
    return document, candidate


def _github_api(endpoint: str, *fields: tuple[str, str]) -> Any:
    command = ["gh", "api", endpoint]
    if fields:
        command.extend(("-X", "GET"))
        for key, value in fields:
            command.extend(("-f", f"{key}={value}"))
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise EvidenceError(f"GitHub API request failed: {endpoint}") from error
    return _strict_json_text(completed.stdout, f"GitHub API response for {endpoint}")


def _timestamp(value: str, label: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as error:
        raise EvidenceError(f"{label} is not an ISO timestamp") from error
    _require(parsed.tzinfo is not None, f"{label} must include a timezone")
    return parsed


def _macos_job_record(
    repository: str,
    run_id: int,
    candidate: dict[str, str],
) -> dict[str, Any]:
    _require(type(run_id) is int and run_id > 0, "CI run id must be positive")
    run = _github_api(f"repos/{repository}/actions/runs/{run_id}")
    _require(
        isinstance(run, dict)
        and run.get("id") == run_id
        and run.get("name") == "CI"
        and run.get("event") == "pull_request"
        and run.get("head_sha") == candidate["candidate_git_commit"]
        and run.get("status") == "completed"
        and run.get("conclusion") == "success"
        and type(run.get("run_attempt")) is int
        and run["run_attempt"] > 0,
        "CI workflow run is not a successful candidate run",
    )
    response = _github_api(f"repos/{repository}/actions/runs/{run_id}/jobs")
    _require(
        isinstance(response, dict) and isinstance(response.get("jobs"), list),
        "CI jobs response differs",
    )
    matches = [
        job
        for job in response["jobs"]
        if isinstance(job, dict) and job.get("name") == MACOS_REQUIRED_JOB
    ]
    _require(len(matches) == 1, "required macOS job closure differs")
    job = matches[0]
    _require(
        type(job.get("id")) is int
        and job["id"] > 0
        and job.get("run_id") == run_id
        and job.get("run_attempt") == run["run_attempt"]
        and job.get("status") == "completed"
        and job.get("conclusion") == "success"
        and isinstance(job.get("started_at"), str)
        and isinstance(job.get("completed_at"), str),
        "required macOS job did not complete successfully",
    )
    return {
        "run_id": run_id,
        "started_at": job["started_at"],
        "completed_at": job["completed_at"],
    }


def _validate_actions_archive(
    archive_path: Path,
    runtime_index: Path,
    runtime: dict[str, Any],
    candidate: dict[str, str],
) -> None:
    descriptors = [record["artifact"] for record in runtime["packages"]]
    for check in runtime["runtime_checks"]:
        descriptors.extend((check["stdout"], check["stderr"]))
    by_path = {record["path"]: record for record in descriptors}
    _require(
        len(by_path) == 14
        and all(record["candidate_evidence"] == candidate for record in descriptors),
        "macOS archive payload descriptor closure differs",
    )
    expected_names = {"runtime-index.json", *by_path}
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = [item for item in archive.infolist() if not item.is_dir()]
            names = [item.filename for item in members]
            _require(
                len(names) == 15
                and len(set(names)) == 15
                and set(names) == expected_names,
                "Actions archive member closure differs",
            )
            _require(archive.testzip() is None, "Actions archive CRC check failed")
            _require(
                archive.read("runtime-index.json") == runtime_index.read_bytes(),
                "Actions archive runtime index bytes differ",
            )
            for path, descriptor in by_path.items():
                raw = archive.read(path)
                _require(
                    len(raw) == descriptor["size_bytes"]
                    and _sha256(raw) == descriptor["sha256"],
                    f"Actions archive payload bytes differ: {path}",
                )
    except (OSError, zipfile.BadZipFile) as error:
        raise EvidenceError("Actions archive is not a readable ZIP") from error


def _actions_artifact_record(
    repository: str,
    ci_run_id: int,
    macos_job: dict[str, Any],
    candidate: dict[str, str],
    archive_path: Path,
) -> dict[str, Any]:
    response = _github_api(
        f"repos/{repository}/actions/runs/{ci_run_id}/artifacts",
        ("per_page", "100"),
    )
    _require(
        isinstance(response, dict) and isinstance(response.get("artifacts"), list),
        "Actions artifact response differs",
    )
    expected_name = f"issue-123-macos-{candidate['candidate_git_commit']}"
    matches = [
        item
        for item in response["artifacts"]
        if isinstance(item, dict) and item.get("name") == expected_name
    ]
    _require(len(matches) == 1, "canonical Actions artifact closure differs")
    record = matches[0]
    workflow_run = record.get("workflow_run")
    raw = archive_path.resolve(strict=True).read_bytes()
    _require(
        type(record.get("id")) is int
        and record["id"] > 0
        and record.get("size_in_bytes") == len(raw)
        and record.get("digest") == f"sha256:{_sha256(raw)}"
        and record.get("expired") is False
        and record.get("archive_download_url")
        == f"https://api.github.com/repos/{repository}/actions/artifacts/{record['id']}/zip"
        and isinstance(record.get("created_at"), str)
        and isinstance(record.get("updated_at"), str)
        and isinstance(workflow_run, dict)
        and type(workflow_run.get("repository_id")) is int
        and workflow_run["repository_id"] > 0
        and workflow_run.get("head_repository_id") == workflow_run["repository_id"]
        and isinstance(workflow_run.get("head_branch"), str)
        and bool(workflow_run["head_branch"])
        and workflow_run.get("head_sha") == candidate["candidate_git_commit"]
        and workflow_run.get("id") == ci_run_id,
        "canonical Actions artifact is incomplete or candidate-mismatched",
    )
    started = _timestamp(macos_job["started_at"], "macOS job start")
    completed = _timestamp(macos_job["completed_at"], "macOS job completion")
    created = _timestamp(record["created_at"], "Actions artifact creation")
    updated = _timestamp(record["updated_at"], "Actions artifact update")
    _require(
        started <= created <= updated <= completed,
        "Actions artifact was not created during the required macOS job",
    )
    return {
        "id": record["id"],
        "name": record["name"],
        "size_in_bytes": record["size_in_bytes"],
        "archive_download_url": record["archive_download_url"],
        "expired": record["expired"],
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
        "digest": record["digest"],
        "workflow_run": {
            "head_branch": workflow_run["head_branch"],
            "head_repository_id": workflow_run["head_repository_id"],
            "head_sha": workflow_run["head_sha"],
            "id": workflow_run["id"],
            "repository_id": workflow_run["repository_id"],
        },
    }


def assemble_macos_index(
    runtime_index: Path,
    manifest: Path,
    repository: str,
    ci_run_id: int,
    actions_archive: Path,
    output: Path,
    scope_output: Path | None = None,
) -> tuple[Path, Path]:
    """Bind exact runtime bytes to their successful candidate Actions artifact."""

    runtime_path = runtime_index.resolve(strict=True)
    runtime, candidate = _load_runtime_index(runtime_path, manifest)
    _require(
        re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) is not None,
        "GitHub repository must use owner/name syntax",
    )
    macos_job = _macos_job_record(repository, ci_run_id, candidate)
    archive_path = actions_archive.resolve(strict=True)
    _validate_actions_archive(archive_path, runtime_path, runtime, candidate)
    actions_artifact = _actions_artifact_record(
        repository,
        ci_run_id,
        macos_job,
        candidate,
        archive_path,
    )
    output = output.resolve()
    _require(
        output not in {runtime_path, archive_path},
        "macOS index must not overwrite its runtime index or Actions archive",
    )
    _require(
        output.parent == runtime_path.parent == archive_path.parent,
        "macOS index, runtime index, and Actions archive must share a directory",
    )
    if scope_output is None:
        scope_output = output.with_name("scope.json")
    scope_output = scope_output.resolve()
    _require(
        scope_output.parent == output.parent
        and scope_output not in {output, runtime_path, archive_path},
        "macOS scope output path differs",
    )
    document = {
        "schema_version": 2,
        "kind": MACOS_INDEX_KIND,
        "candidate_evidence": candidate,
        "actions_artifact": actions_artifact,
        "packages": runtime["packages"],
        "runtime_checks": runtime["runtime_checks"],
        "passed": True,
    }
    output.write_text(
        json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    scope = {
        "index": _descriptor(output, output.parent, candidate),
        "actions_archive": _descriptor(archive_path, output.parent, candidate),
    }
    scope_output.write_text(
        json.dumps(scope, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output, scope_output


def _installed_archive_sha256(expected_package: Path) -> str:
    expected_package = expected_package.resolve(strict=True)
    direct_url_raw = metadata.distribution("gmes").read_text("direct_url.json")
    _require(
        isinstance(direct_url_raw, str) and bool(direct_url_raw),
        "installed gmes has no PEP 610 archive provenance",
    )
    direct_url = _strict_json_text(direct_url_raw, "installed gmes direct URL")
    _require(isinstance(direct_url, dict), "installed gmes direct URL differs")
    url = direct_url.get("url")
    archive_info = direct_url.get("archive_info")
    _require(
        isinstance(url, str) and isinstance(archive_info, dict),
        "installed gmes archive provenance differs",
    )
    parsed = parse.urlsplit(url)
    _require(
        parsed.scheme == "file" and parsed.netloc in {"", "localhost"},
        "installed gmes did not come from a local evidence archive",
    )
    installed_archive = Path(parse.unquote(parsed.path)).resolve(strict=True)
    _require(
        installed_archive == expected_package,
        "installed gmes came from a different evidence archive",
    )
    hashes = archive_info.get("hashes")
    digest = hashes.get("sha256") if isinstance(hashes, dict) else None
    if digest is None:
        legacy_hash = archive_info.get("hash")
        if isinstance(legacy_hash, str) and legacy_hash.startswith("sha256="):
            digest = legacy_hash.removeprefix("sha256=")
    _require(
        isinstance(digest, str)
        and SHA256_RE.fullmatch(digest) is not None
        and digest == _sha256(expected_package.read_bytes()),
        "installed gmes archive digest differs",
    )
    return digest


def _command_text(repository: Path, *command: str) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise EvidenceError(f"host command {command[0]!r} could not run") from error
    _require(completed.returncode == 0, f"host command {command[0]!r} failed")
    output = completed.stdout.strip()
    _require(bool(output), f"host command {command[0]!r} returned empty output")
    return output


def _capture_host_contract(repository: Path, torch_module: Any) -> dict[str, Any]:
    version = getattr(torch_module, "__version__", None)
    cuda_runtime = getattr(getattr(torch_module, "version", None), "cuda", None)
    _require(isinstance(version, str) and bool(version), "Torch version is absent")
    _require(cuda_runtime is None, "macOS Torch runtime unexpectedly reports CUDA")
    result = {
        "schema_version": 2,
        "common_identity": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "os": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
            "python": platform.python_version(),
            "cxx_version": _command_text(repository, "c++", "--version"),
            "swig_version": _command_text(repository, "swig", "-version"),
            "uv_lock_sha256": _sha256((repository / "uv.lock").read_bytes()),
        },
        "runtime_identity": {"torch": version, "cuda_runtime": None},
    }
    _validate_host_contract(result, "captured host contract")
    return result


def _array_payload(values: Any) -> tuple[list[int], str, list[int | float]]:
    import numpy as np  # pylint: disable=import-outside-toplevel

    array = np.asarray(values)
    _require(array.dtype.kind in "fiu", "runtime array dtype is not numeric")
    flattened: list[int | float]
    if array.dtype.kind == "f":
        flattened = [float(item) for item in array.reshape(-1)]
    else:
        flattened = [int(item) for item in array.reshape(-1)]
    _require(
        all(math.isfinite(item) for item in flattened), "runtime array is non-finite"
    )
    return list(array.shape), str(array.dtype), flattened


def _array_record(name: str, initial: Any, final: Any) -> dict[str, Any]:
    initial_shape, initial_dtype, initial_values = _array_payload(initial)
    final_shape, final_dtype, final_values = _array_payload(final)
    _require(
        initial_shape == final_shape and initial_dtype == final_dtype,
        f"runtime array contract changed for {name}",
    )
    record = {
        "name": name,
        "shape": initial_shape,
        "dtype": initial_dtype,
        "initial_values": initial_values,
        "final_values": final_values,
        "initial_sha256": _canonical_sha256(
            {"dtype": initial_dtype, "shape": initial_shape, "values": initial_values}
        ),
        "final_sha256": _canonical_sha256(
            {"dtype": final_dtype, "shape": final_shape, "values": final_values}
        ),
        "finite": all(math.isfinite(item) for item in (*initial_values, *final_values)),
        "changed": not _type_exact_equal(initial_values, final_values),
    }
    _validate_array_record(record, name, f"captured array {name}")
    return record


def _native_fields(simulation: Any, gmes_module: Any) -> dict[str, Any]:
    return {
        name: simulation.field[getattr(gmes_module, name)].copy()
        for name in FIELD_NAMES
    }


def _native_result(
    gmes_module: Any, initial_fields: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np  # pylint: disable=import-outside-toplevel

    geometry = [
        gmes_module.DefaultMedium(gmes_module.Dielectric(eps_inf=1.7, mu_inf=1.05))
    ]
    simulation = gmes_module.FDTD(
        gmes_module.Cartesian(size=(2, 2, 2), resolution=2),
        geometry,
        verbose=False,
    )
    simulation.init()
    for name, values in initial_fields.items():
        simulation.field[getattr(gmes_module, name)][...] = values
    initial = _native_fields(simulation, gmes_module)
    initial_addresses = {
        name: int(simulation.field[getattr(gmes_module, name)].ctypes.data)
        for name in FIELD_NAMES
    }
    initial_step = int(simulation.time_step.n)
    simulation.step()
    simulation.step()
    final = _native_fields(simulation, gmes_module)
    final_addresses = {
        name: int(simulation.field[getattr(gmes_module, name)].ctypes.data)
        for name in FIELD_NAMES
    }
    fields = [_array_record(name, initial[name], final[name]) for name in FIELD_NAMES]
    finite = all(
        np.isfinite(values).all() for values in (*initial.values(), *final.values())
    )
    progressed = _records_sha256(fields, "initial") != _records_sha256(fields, "final")
    stable = initial_addresses == final_addresses
    result = {
        "openmp_enabled": False,
        "steps": {"initial": initial_step, "final": int(simulation.time_step.n)},
        "fields": fields,
        "initial_field_sha256": _records_sha256(fields, "initial"),
        "final_field_sha256": _records_sha256(fields, "final"),
        "storage_addresses": {"initial": initial_addresses, "final": final_addresses},
        "storage_stable": stable,
        "finite": bool(finite),
        "progressed": progressed,
        "passed": bool(finite and progressed and stable),
    }
    _validate_native_result(result, "captured native result")
    return result, final


def _torch_state(simulation: Any) -> dict[str, Any]:
    return {
        name: tensor.detach().cpu().numpy().copy()
        for name, tensor in simulation.state.named_buffers()
    }


def _counter_snapshot(torch_module: Any) -> dict[str, int]:
    counters = torch_module._dynamo.utils.counters
    return {
        "unique_graphs": int(counters["stats"].get("unique_graphs", 0)),
        "calls_captured": int(counters["stats"].get("calls_captured", 0)),
        "graph_breaks": int(sum(counters["graph_break"].values())),
    }


def _torch_result(
    gmes_module: Any,
    torch_module: Any,
    initial_fields: dict[str, Any],
    mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np  # pylint: disable=import-outside-toplevel

    policy = "eager" if mode == "eager" else "compile"
    torch_module._dynamo.reset()
    torch_module._dynamo.utils.counters.clear()
    runtime = gmes_module.TorchRuntimeConfig(
        device="cpu",
        precision="float64",
        compile_policy=policy,
        compile_mode="default",
        cpu_threads=1,
        cpu_interop_threads=1,
    )
    simulation = gmes_module.TorchSimulation(
        space=gmes_module.Cartesian(size=(2, 2, 2), resolution=2),
        geometry=[
            gmes_module.DefaultMedium(gmes_module.Dielectric(eps_inf=1.7, mu_inf=1.05))
        ],
        runtime=runtime,
    )
    simulation.load_host_fields(initial_fields)
    initial = simulation.host_snapshot()
    initial_state = _torch_state(simulation)
    initial_addresses = {
        name: int(address) for name, address in simulation.buffer_addresses().items()
    }
    before = _counter_snapshot(torch_module)
    initial_step = int(simulation.state.step_count.detach().cpu())
    simulation.advance(1)
    warmup_step = int(simulation.state.step_count.detach().cpu())
    warmup_addresses = {
        name: int(address) for name, address in simulation.buffer_addresses().items()
    }
    warmup = _counter_snapshot(torch_module)
    simulation.advance(1)
    final_step = int(simulation.state.step_count.detach().cpu())
    final = simulation.host_snapshot()
    final_state = _torch_state(simulation)
    final_addresses = {
        name: int(address) for name, address in simulation.buffer_addresses().items()
    }
    final_counters = _counter_snapshot(torch_module)
    fields = [_array_record(name, initial[name], final[name]) for name in FIELD_NAMES]
    states = [
        _array_record(name, initial_state[name], final_state[name])
        for name in sorted(initial_state)
    ]
    finite = all(
        np.isfinite(values).all() for values in (*initial.values(), *final.values())
    )
    finite = finite and all(record["finite"] for record in states)
    progressed = _records_sha256(fields, "initial") != _records_sha256(
        fields, "final"
    ) and _records_sha256(states, "initial") != _records_sha256(states, "final")
    stable = initial_addresses == warmup_addresses == final_addresses
    compiler_clean = (
        before["graph_breaks"]
        == warmup["graph_breaks"]
        == final_counters["graph_breaks"]
        == 0
        and warmup == final_counters
    )
    if mode == "compile":
        compiler_clean = (
            compiler_clean
            and warmup["unique_graphs"] > 0
            and warmup["calls_captured"] > 0
        )
    else:
        compiler_clean = compiler_clean and before == warmup == final_counters == {
            "unique_graphs": 0,
            "calls_captured": 0,
            "graph_breaks": 0,
        }
    result = {
        "mode": mode,
        "runtime": {
            "device": "cpu",
            "precision": "float64",
            "compile_policy": policy,
            "compile_mode": "default",
            "cpu_threads": 1,
            "cpu_interop_threads": 1,
        },
        "steps": {"initial": initial_step, "warmup": warmup_step, "final": final_step},
        "fields": fields,
        "state_buffers": states,
        "initial_field_sha256": _records_sha256(fields, "initial"),
        "final_field_sha256": _records_sha256(fields, "final"),
        "initial_state_sha256": _records_sha256(states, "initial"),
        "final_state_sha256": _records_sha256(states, "final"),
        "storage_addresses": {
            "initial": initial_addresses,
            "warmup": warmup_addresses,
            "final": final_addresses,
        },
        "storage_stable": stable,
        "compiler": {"before": before, "warmup": warmup, "final": final_counters},
        "finite": bool(finite),
        "progressed": progressed,
        "passed": bool(finite and progressed and stable and compiler_clean),
    }
    _validate_torch_mode(result, mode, f"captured Torch {mode}")
    return result, final


def _comparison(
    left_name: str,
    right_name: str,
    left: dict[str, Any],
    right: dict[str, Any],
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    import numpy as np  # pylint: disable=import-outside-toplevel

    fields = []
    for name in FIELD_NAMES:
        left_values = np.asarray(left[name])
        right_values = np.asarray(right[name])
        _require(
            left_values.shape == right_values.shape,
            f"comparison shape differs for {name}",
        )
        maximum = float(np.max(np.abs(left_values - right_values), initial=0.0))
        passed = bool(np.allclose(left_values, right_values, rtol=rtol, atol=atol))
        fields.append({"name": name, "max_abs_error": maximum, "passed": passed})
    return {
        "left": left_name,
        "right": right_name,
        "tolerance": {"rtol": float(rtol), "atol": float(atol)},
        "fields": fields,
        "passed": all(record["passed"] for record in fields),
    }


def probe_installed_package(
    role: str,
    mode: str | None,
    repository: Path,
    forbidden_root: Path,
    expected_package: Path,
) -> dict[str, Any]:
    """Run one installed-package import or complete native/Torch CPU contract."""

    _require(role in RUNTIME_ROLES, "package probe role differs")
    _require(SUITE_MODES.get(role) == mode, "package probe role and mode differ")
    platform_record = _platform_record()
    _validate_platform(platform_record, "package probe platform")
    repository = repository.resolve(strict=True)
    forbidden_root = forbidden_root.resolve(strict=True)
    expected_package = expected_package.resolve(strict=True)

    import numpy as np  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel

    import gmes  # pylint: disable=import-outside-toplevel

    package_sha256 = _installed_archive_sha256(expected_package)
    host_contract = _capture_host_contract(repository, torch)
    module_path = Path(gmes.__file__).resolve(strict=True)
    native_modules = [
        Path(importlib.import_module(name).__file__).resolve(strict=True)
        for name in ("gmes._constant", "gmes._pw_material")
    ]
    outside_source = all(
        not path.is_relative_to(forbidden_root)
        for path in (module_path, *native_modules)
    )
    _require(outside_source, "package probe imported from the source checkout")
    if role.endswith("-import"):
        result = {
            "kind": IMPORT_RESULT_KIND,
            "role": role,
            "package_sha256": package_sha256,
            "platform": platform_record,
            "host_contract": host_contract,
            "distribution": {
                "name": "gmes",
                "version": metadata.version("gmes"),
                "module_path": str(module_path),
                "native_module_paths": [str(path) for path in native_modules],
                "outside_source": outside_source,
            },
            "passed": True,
        }
        _validate_probe_result(result, role, package_sha256, platform_record)
        return result

    _require(
        gmes.pw_material.openmp_enabled() is False,
        "macOS native package enabled OpenMP",
    )
    expected_openmp = "auto" if mode == "default" else "0"
    _require(
        os.environ.get("GMES_ENABLE_OPENMP") == expected_openmp,
        "package probe OpenMP environment differs",
    )
    native_template = gmes.FDTD(
        gmes.Cartesian(size=(2, 2, 2), resolution=2),
        [gmes.DefaultMedium(gmes.Dielectric(eps_inf=1.7, mu_inf=1.05))],
        verbose=False,
    )
    native_template.init()
    rng = np.random.default_rng(1729)
    initial_fields = {
        name: rng.normal(size=native_template.field[getattr(gmes, name)].shape) * 1e-3
        for name in FIELD_NAMES
    }
    _require(
        all(
            np.all(values != 0) and np.isfinite(values).all()
            for values in initial_fields.values()
        ),
        "deterministic initial field contract differs",
    )
    native, native_final = _native_result(gmes, initial_fields)
    eager, eager_final = _torch_result(gmes, torch, initial_fields, "eager")
    compiled, compiled_final = _torch_result(gmes, torch, initial_fields, "compile")
    manifest = _strict_json(
        repository / "benchmarks" / "native_oracle_workloads.json", "candidate manifest"
    )
    tolerance = manifest["tolerances"]["torch"]["dielectric"]["float64"]
    rtol = float(tolerance["rtol"])
    atol = float(tolerance["atol"])
    comparisons = [
        _comparison("eager", "compile", eager_final, compiled_final, rtol, atol),
        _comparison("eager", "native", eager_final, native_final, rtol, atol),
        _comparison("compile", "native", compiled_final, native_final, rtol, atol),
    ]
    result = {
        "kind": SUITE_RESULT_KIND,
        "role": role,
        "mode": mode,
        "package_sha256": package_sha256,
        "platform": platform_record,
        "host_contract": host_contract,
        "native": native,
        "torch_cpu": {"modes": [eager, compiled], "comparisons": comparisons},
        "passed": bool(
            native["passed"]
            and eager["passed"]
            and compiled["passed"]
            and all(item["passed"] for item in comparisons)
        ),
    }
    _validate_probe_result(result, role, package_sha256, platform_record)
    return result


def record_runtime_command(
    role: str,
    mode: str | None,
    repository: Path,
    forbidden_root: Path,
    expected_package: Path,
    evidence_directory: Path,
    records_directory: Path,
    working_directory: Path,
) -> int:
    """Run a real isolated probe and persist its exact argv, exit, stdout, stderr."""

    _require(role in RUNTIME_ROLES, "runtime command role differs")
    _require(SUITE_MODES.get(role) == mode, "runtime command role and mode differ")
    repository = repository.resolve(strict=True)
    forbidden_root = forbidden_root.resolve(strict=True)
    expected_package = expected_package.resolve(strict=True)
    evidence_directory = evidence_directory.resolve(strict=True)
    records_directory = records_directory.resolve(strict=True)
    working_directory = working_directory.resolve(strict=True)
    _require(
        forbidden_root == repository,
        "forbidden source root must be the candidate checkout",
    )
    _require(
        expected_package.parent == evidence_directory / "packages",
        "expected package must belong to the evidence package directory",
    )
    _require(
        not working_directory.is_relative_to(forbidden_root)
        and not working_directory.is_relative_to(evidence_directory),
        "probe working directory must be outside source and evidence",
    )
    _require(
        not records_directory.is_relative_to(evidence_directory),
        "command records must remain outside the uploaded evidence closure",
    )
    log_directory = evidence_directory / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    records_directory.mkdir(parents=True, exist_ok=True)
    cache_directory = records_directory / "torchinductor" / role
    cache_directory.mkdir(parents=True, exist_ok=True)
    argv = _probe_argv(
        sys.executable,
        repository,
        forbidden_root,
        expected_package,
        role,
        mode,
    )
    environment_record = {
        "GMES_ENABLE_OPENMP": "0" if mode == "serial" else "auto",
        "MKL_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "TORCHINDUCTOR_CACHE_DIR": str(cache_directory),
    }
    environment = dict(os.environ)
    environment.update(environment_record)
    completed = subprocess.run(
        argv,
        cwd=working_directory,
        env=environment,
        check=False,
        capture_output=True,
    )
    stdout_raw = (
        completed.stdout
        if isinstance(completed.stdout, bytes)
        else completed.stdout.encode("utf-8")
    )
    stderr_raw = (
        completed.stderr
        if isinstance(completed.stderr, bytes)
        else completed.stderr.encode("utf-8")
    )
    stdout = log_directory / f"{role}.stdout.json"
    stderr = log_directory / f"{role}.stderr.txt"
    stdout.write_bytes(stdout_raw)
    stderr.write_bytes(stderr_raw)
    try:
        result = _strict_json_bytes(stdout_raw, f"{role} stdout")
    except EvidenceError:
        result = None
    record = {
        "schema_version": 2,
        "kind": COMMAND_RECORD_KIND,
        "role": role,
        "command": {
            "argv": argv,
            "cwd": str(working_directory),
            "environment": environment_record,
        },
        "exit_code": int(completed.returncode),
        "stdout_sha256": _sha256(stdout_raw),
        "stdout_size_bytes": len(stdout_raw),
        "stderr_sha256": _sha256(stderr_raw),
        "stderr_size_bytes": len(stderr_raw),
        "result": result,
    }
    (records_directory / f"{role}.json").write_text(
        json.dumps(record, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return int(completed.returncode)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser("record")
    record.add_argument("--role", choices=RUNTIME_ROLES, required=True)
    record.add_argument("--mode", choices=("default", "serial"))
    record.add_argument("--repository", type=Path, required=True)
    record.add_argument("--forbidden-root", type=Path, required=True)
    record.add_argument("--expected-package", type=Path, required=True)
    record.add_argument("--evidence-dir", type=Path, required=True)
    record.add_argument("--records-dir", type=Path, required=True)
    record.add_argument("--working-directory", type=Path, required=True)

    probe = subparsers.add_parser("_probe")
    probe.add_argument("--role", choices=RUNTIME_ROLES, required=True)
    probe.add_argument("--mode", choices=("default", "serial"))
    probe.add_argument("--repository", type=Path, required=True)
    probe.add_argument("--forbidden-root", type=Path, required=True)
    probe.add_argument("--expected-package", type=Path, required=True)

    capture = subparsers.add_parser("capture")
    capture.add_argument("--evidence-dir", type=Path, required=True)
    capture.add_argument("--records-dir", type=Path, required=True)
    capture.add_argument("--repository", type=Path, default=ROOT)
    capture.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    capture.add_argument("--candidate-commit", required=True)

    assemble = subparsers.add_parser("assemble")
    assemble.add_argument("--runtime-index", type=Path, required=True)
    assemble.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    assemble.add_argument("--repository", default="ruddyscent/gmes")
    assemble.add_argument("--ci-run-id", type=int, required=True)
    assemble.add_argument("--actions-archive", type=Path, required=True)
    assemble.add_argument("--output", type=Path, required=True)
    assemble.add_argument("--scope-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if args.command == "record":
        return record_runtime_command(
            role=args.role,
            mode=args.mode,
            repository=args.repository,
            forbidden_root=args.forbidden_root,
            expected_package=args.expected_package,
            evidence_directory=args.evidence_dir,
            records_directory=args.records_dir,
            working_directory=args.working_directory,
        )
    if args.command == "_probe":
        result = probe_installed_package(
            role=args.role,
            mode=args.mode,
            repository=args.repository,
            forbidden_root=args.forbidden_root,
            expected_package=args.expected_package,
        )
        print(
            json.dumps(result, allow_nan=False, separators=(",", ":"), sort_keys=True)
        )
        return 0
    if args.command == "capture":
        output = capture_runtime_index(
            args.evidence_dir,
            args.records_dir,
            args.repository,
            args.manifest,
            args.candidate_commit,
        )
        print(output)
        return 0
    index, scope = assemble_macos_index(
        runtime_index=args.runtime_index,
        manifest=args.manifest,
        repository=args.repository,
        ci_run_id=args.ci_run_id,
        actions_archive=args.actions_archive,
        output=args.output,
        scope_output=args.scope_output,
    )
    print(json.dumps({"index": str(index), "scope": str(scope)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
