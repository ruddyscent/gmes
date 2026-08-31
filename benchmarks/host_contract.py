"""Canonical host and candidate provenance shared by issue #123 producers."""

from __future__ import annotations

import hashlib
import platform
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "benchmarks" / "native_oracle_workloads.json"
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMON_IDENTITY_KEYS = {
    "hostname",
    "platform",
    "os",
    "python",
    "cxx_version",
    "swig_version",
    "uv_lock_sha256",
}
RUNTIME_IDENTITY_KEYS = {"torch", "cuda_runtime"}


def host_contract_complete(value: object) -> bool:
    """Return whether *value* is one exact schema-v2 host/runtime identity."""

    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "common_identity",
        "runtime_identity",
    }:
        return False
    common = value.get("common_identity")
    runtime = value.get("runtime_identity")
    if (
        type(value.get("schema_version")) is not int
        or value["schema_version"] != 2
        or not isinstance(common, dict)
        or set(common) != COMMON_IDENTITY_KEYS
        or not isinstance(runtime, dict)
        or set(runtime) != RUNTIME_IDENTITY_KEYS
    ):
        return False
    os_record = common.get("os")
    return (
        isinstance(os_record, dict)
        and set(os_record) == {"system", "release", "machine"}
        and all(
            isinstance(common.get(name), str) and bool(common[name])
            for name in COMMON_IDENTITY_KEYS - {"os", "uv_lock_sha256"}
        )
        and all(
            isinstance(os_record.get(name), str) and bool(os_record[name])
            for name in ("system", "release", "machine")
        )
        and isinstance(common.get("uv_lock_sha256"), str)
        and SHA256_RE.fullmatch(common["uv_lock_sha256"]) is not None
        and isinstance(runtime.get("torch"), str)
        and bool(runtime["torch"])
        and (
            runtime.get("cuda_runtime") is None
            or isinstance(runtime["cuda_runtime"], str)
            and bool(runtime["cuda_runtime"])
        )
    )


def _command_text(*command: str, allow_empty: bool = False) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise RuntimeError(
            f"host command {command[0]!r} could not be executed"
        ) from error
    if completed.returncode != 0:
        raise RuntimeError(
            f"host command {command[0]!r} exited with {completed.returncode}"
        )
    output = completed.stdout.strip()
    if not allow_empty and not output:
        raise RuntimeError(f"host command {command[0]!r} returned empty output")
    return output


def capture_host_contract(torch_module) -> dict[str, object]:
    """Return the exact Linux host/runtime/toolchain identity for one run."""

    uv_lock = ROOT / "uv.lock"
    torch_version = getattr(torch_module, "__version__", None)
    version_module = getattr(torch_module, "version", None)
    cuda_runtime = getattr(version_module, "cuda", None)
    if not isinstance(torch_version, str) or not torch_version:
        raise RuntimeError("Torch build identity is absent")
    if cuda_runtime is not None and (
        not isinstance(cuda_runtime, str) or not cuda_runtime
    ):
        raise RuntimeError("Torch CUDA runtime identity is malformed")
    common_identity = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "python": platform.python_version(),
        "cxx_version": _command_text("c++", "--version"),
        "swig_version": _command_text("swig", "-version"),
        "uv_lock_sha256": hashlib.sha256(uv_lock.read_bytes()).hexdigest(),
    }
    if any(
        not isinstance(value, str) or not value
        for value in common_identity.values()
        if not isinstance(value, dict)
    ):
        raise RuntimeError("common host identity is incomplete")
    if any(
        not isinstance(value, str) or not value
        for value in common_identity["os"].values()
    ):
        raise RuntimeError("common OS identity is incomplete")
    result = {
        "schema_version": 2,
        "common_identity": common_identity,
        "runtime_identity": {
            "torch": torch_version,
            "cuda_runtime": cuda_runtime,
        },
    }
    if not host_contract_complete(result):
        raise RuntimeError("captured host contract is incomplete")
    return result


def candidate_evidence(
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, str]:
    """Bind an artifact to one clean candidate commit and manifest bytes."""

    commit = _command_text("git", "rev-parse", "HEAD")
    status = _command_text("git", "status", "--short", allow_empty=True)
    if COMMIT_RE.fullmatch(commit) is None:
        raise RuntimeError("candidate commit is not one full lowercase Git object id")
    if status != "":
        raise RuntimeError("candidate checkout is not clean")
    return {
        "candidate_git_commit": commit,
        "candidate_git_status": status,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
