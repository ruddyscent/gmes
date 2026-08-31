"""Capture canonical, candidate-bound evidence for two-GPU failure contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch

from benchmarks.host_contract import (
    DEFAULT_MANIFEST,
    ROOT,
    candidate_evidence,
    capture_host_contract,
)

FAILURE_RUN_KIND = "two-gpu-failure-run"
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


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _descriptor(
    path: Path,
    base: Path,
    candidate: dict[str, str],
    media_type: str,
) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": path.relative_to(base).as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "media_type": media_type,
        "candidate_evidence": candidate,
    }


def _observed_failure(
    mode: str,
    exit_code: int,
    stdout: bytes,
    stderr: bytes,
) -> bool:
    try:
        stdout_text = stdout.decode("utf-8")
        stderr_text = stderr.decode("utf-8")
    except UnicodeDecodeError:
        return False
    contract = FAILURE_REASON_CONTRACTS[mode]
    required_tokens = contract["required_tokens"]
    if mode == "rank-failure":
        text = (stdout_text + "\n" + stderr_text).lower()
        return (
            exit_code != 0
            and all(token in text for token in required_tokens)
            and ("childfailederror" in text or "rank" in text)
        )

    records = []
    for line in stdout_text.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("mode") == mode:
            records.append(value)
    if len(records) != 1:
        return False
    record = records[0]
    error = record.get("rank0_error")
    return (
        exit_code == 0
        and set(record) == {"mode", "passed", "rank0_error"}
        and record.get("passed") is True
        and isinstance(error, str)
        and bool(error)
        and all(token in error.lower() for token in required_tokens)
    )


def capture_failure_evidence(
    mode: str,
    output_directory: Path,
    manifest: Path = DEFAULT_MANIFEST,
    descriptor_root: Path | None = None,
) -> Path:
    """Run one exact two-rank probe and preserve its raw canonical wrapper."""

    _require(mode in FAILURE_REASON_CONTRACTS, f"unknown failure mode: {mode}")
    candidate = candidate_evidence(manifest.resolve(strict=True))
    _require(
        candidate["candidate_git_status"] == "",
        "failure evidence requires a clean candidate checkout",
    )
    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    descriptor_root = (
        output_directory
        if descriptor_root is None
        else descriptor_root.resolve(strict=True)
    )
    output_directory.relative_to(descriptor_root)
    stdout_path = output_directory / f"two-gpu-{mode}.stdout"
    stderr_path = output_directory / f"two-gpu-{mode}.stderr"
    wrapper_path = output_directory / f"two-gpu-{mode}.json"
    _require(
        not any(path.exists() for path in (stdout_path, stderr_path, wrapper_path)),
        f"failure evidence already exists for {mode}",
    )

    command = [
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
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    stdout_path.write_bytes(completed.stdout)
    stderr_path.write_bytes(completed.stderr)
    passed = _observed_failure(
        mode,
        completed.returncode,
        completed.stdout,
        completed.stderr,
    )
    wrapper = {
        "schema_version": 1,
        "kind": FAILURE_RUN_KIND,
        "mode": mode,
        "candidate_evidence": candidate,
        "host_contract": capture_host_contract(torch),
        "command": command,
        "exit_code": completed.returncode,
        "stdout": _descriptor(
            stdout_path,
            descriptor_root,
            candidate,
            "text/plain; charset=utf-8",
        ),
        "stderr": _descriptor(
            stderr_path,
            descriptor_root,
            candidate,
            "text/plain; charset=utf-8",
        ),
        "expected_failure": FAILURE_REASON_CONTRACTS[mode],
        "passed": passed,
    }
    wrapper_path.write_text(
        json.dumps(wrapper, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return wrapper_path


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=tuple(FAILURE_REASON_CONTRACTS))
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--descriptor-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    output = capture_failure_evidence(
        args.mode,
        args.output_directory,
        args.manifest,
        args.descriptor_root,
    )
    document = json.loads(output.read_text(encoding="utf-8"))
    print(output)
    return 0 if document["passed"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
