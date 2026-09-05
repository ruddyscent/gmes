#!/usr/bin/env python3
"""Run a historical oracle capture from its pinned observer checkout."""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

_NATIVE_PROGRESS_PREFIXES = (
    "Estimated time of completion:",
    "Elapsed time:",
)


def sanitized_environment():
    """Prevent imports from the controller checkout or user site packages."""
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def load_capture_stdout(stdout):
    """Load the final JSON document after known native progress messages."""
    lines = stdout.splitlines()
    try:
        document_start = lines.index("{")
    except ValueError as error:
        raise ValueError("native capture stdout has no JSON document") from error
    unexpected = [
        line
        for line in lines[:document_start]
        if line and not line.startswith(_NATIVE_PROGRESS_PREFIXES)
    ]
    if unexpected:
        raise ValueError(f"unexpected native capture stdout: {unexpected[0]!r}")
    document = json.loads("\n".join(lines[document_start:]))
    if not isinstance(document, dict):
        raise ValueError("native capture stdout JSON is not an object")
    return document


def run_capture(checkout, python, manifest, case, output):
    """Invoke the immutable observer, never this checkout's oracle runtime."""
    checkout = Path(checkout).resolve(strict=True)
    python = Path(python).absolute()
    if not python.is_file():
        raise FileNotFoundError(f"Python executable is absent: {python}")
    manifest = Path(manifest).resolve(strict=True)
    contract = json.loads(manifest.read_text(encoding="utf-8"))
    reference = contract["reference"]
    expected_commit = reference["observer_commit"]
    actual_commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != expected_commit:
        raise ValueError(
            "historical oracle checkout does not match the pinned observer commit: "
            f"expected {expected_commit}, got {actual_commit}"
        )
    output = Path(output).resolve()
    runner = checkout / "benchmarks" / "native_oracle.py"
    if not runner.is_file():
        raise FileNotFoundError(f"historical oracle runner is absent: {runner}")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(python),
        "-I",
        str(runner),
        "--manifest",
        str(manifest),
        "capture",
        "--case",
        case,
        "--output",
        str(output),
    ]
    environment = sanitized_environment()
    environment["GMES_ORACLE_EXPECTED_CHECKOUT"] = str(checkout)
    with tempfile.TemporaryDirectory(prefix="gmes-oracle-") as directory:
        result = subprocess.run(
            command,
            cwd=directory,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
    return {
        "checkout": str(checkout),
        "python": str(python),
        "manifest": str(manifest),
        "case": case,
        "output": str(output),
        "historical_observer_tag": reference["observer_tag"],
        "historical_observer_commit": expected_commit,
        "command": command,
        "capture": load_capture_stdout(result.stdout),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_capture(
                args.checkout, args.python, args.manifest, args.case, args.output
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
