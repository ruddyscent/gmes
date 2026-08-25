#!/usr/bin/env python3
"""Run an oracle capture in a checkout-isolated process."""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


def sanitized_environment():
    """Prevent imports from the controller checkout or user site packages."""
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def run_capture(checkout, python, manifest, case, output):
    checkout = Path(checkout).resolve(strict=True)
    python = Path(python).absolute()
    if not python.is_file():
        raise FileNotFoundError(f"Python executable is absent: {python}")
    manifest = Path(manifest).resolve(strict=True)
    output = Path(output).resolve()
    runner = Path(__file__).resolve().with_name("native_oracle.py")
    if not runner.is_file():
        raise FileNotFoundError(f"oracle controller is absent: {runner}")
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
        "command": command,
        "capture": json.loads(result.stdout),
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
