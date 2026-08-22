#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate Figs. 1-12 from Ziolkowski, Arnold, and Gogny.

Reference: R. W. Ziolkowski, J. M. Arnold, and D. M. Gogny,
"Ultrafast pulse interactions with two-level atoms," Phys. Rev. A 52,
3082-3094 (1995), https://doi.org/10.1103/PhysRevA.52.3082.
"""

from argparse import ArgumentParser
from pathlib import Path

from ziolkowski1995_gain import generate as generate_gain
from ziolkowski1995_pump_probe import generate as generate_pump_probe
from ziolkowski1995_sit import generate as generate_sit
from ziolkowski1995_ultrafast import generate as generate_ultrafast


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--figures", nargs="+", type=int, default=list(range(1, 13)))
    parser.add_argument("--output-dir", type=Path, default=Path("ziolkowski1995"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    arguments = parse_args()
    invalid = set(arguments.figures).difference(range(1, 13))
    if invalid:
        raise ValueError(f"figures must be in 1-12, got {sorted(invalid)}")
    common = {
        "output_dir": arguments.output_dir,
        "quick": arguments.quick,
        "show": arguments.show,
        "verbose": arguments.verbose,
    }
    groups = (
        (range(1, 5), generate_sit),
        (range(5, 10), generate_ultrafast),
        ((10, 11), generate_gain),
        ((12,), generate_pump_probe),
    )
    requested = set(arguments.figures)
    for candidates, generator in groups:
        selected = sorted(requested.intersection(candidates))
        if selected:
            generator(selected, **common)


if __name__ == "__main__":
    main()
