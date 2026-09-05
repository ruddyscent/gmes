#!/usr/bin/env python3
"""Read the immutable historical native geometry-mapping benchmark contract.

The retired native FDTD mapper is never instantiated in a current checkout.
Its historical measurements are retained by observer tag, commit, and digest.
"""

import argparse
import json
from pathlib import Path

DEFAULT_MANIFEST = Path(__file__).with_name("native_oracle_workloads.json")


def historical_contract(manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, object]:
    """Return the frozen mapping benchmark provenance without lowering geometry."""

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
    """Print the read-only historical mapping benchmark identity."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)
    print(json.dumps(historical_contract(args.manifest), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
