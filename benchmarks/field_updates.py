#!/usr/bin/env python3
"""Read the immutable historical native field-update benchmark contract.

Current checkouts do not execute the retired native FDTD benchmark.  Historical
measurements remain identified by the manifest's native tag and observer pins.
"""

import argparse
import json
from pathlib import Path

DEFAULT_MANIFEST = Path(__file__).with_name("native_oracle_workloads.json")


def historical_contract(manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, object]:
    """Return the immutable native benchmark identity without executing it."""

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    reference = manifest["reference"]
    return {
        "status": "historical-only",
        "reference_tag": reference["tag"],
        "reference_commit": reference["commit"],
        "observer_tag": reference["performance_observer_tag"],
        "observer_commit": reference["performance_observer_commit"],
        "summary_sha256": reference["performance_summary_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    """Print the read-only historical benchmark identity."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)
    print(json.dumps(historical_contract(args.manifest), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
