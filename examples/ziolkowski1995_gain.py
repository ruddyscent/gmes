#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Reproduce Ziolkowski, Arnold, and Gogny, Figs. 10-11 (gain).

Reference: Phys. Rev. A 52, 3082-3094 (1995),
https://doi.org/10.1103/PhysRevA.52.3082. The published Fig. 10 transient
is not fully reproduced; see VERIFICATION.md for the equation audit.
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from ziolkowski1995_common import (
    UNITS,
    configure_axes,
    figure_title,
    gain_scenario,
    plot_gain,
    population_mask,
    print_intensity_summary,
    print_scenario,
    run_gain,
    save_plot,
)


def generate(
    figures=(10, 11),
    *,
    output_dir=Path("ziolkowski1995"),
    quick=False,
    show=False,
    verbose=False,
):
    requested = set(figures)
    invalid = requested.difference((10, 11))
    if invalid:
        raise ValueError(f"gain figures must be 10 or 11, got {sorted(invalid)}")
    if not requested:
        return

    scenario = gain_scenario(quick)
    print_scenario(scenario)
    result = run_gain(
        scenario,
        duration_s=1.875e-12,
        sample_stride=1 if quick else 10,
        verbose=verbose,
    )
    print_intensity_summary(
        result,
        peak_window_s=(80e-15, 550e-15),
        late_window_s=(700e-15, 850e-15),
    )
    if 10 in requested:
        plot_gain(
            result,
            output_dir / "fig10.png",
            title=figure_title("Fig. 10 - small-signal gain and saturation", quick),
            show=show,
        )
    if 11 in requested:
        figure, axis = plt.subplots(figsize=(9, 4.8))
        snapshot = result.snapshot
        axis.plot(
            snapshot.distance_um,
            snapshot.electric / UNITS.electric_field(scenario.amplitude_v_m),
            linestyle="--",
            label=r"$E_x$",
        )
        axis.plot(
            snapshot.distance_um,
            population_mask(snapshot, scenario),
            label=r"initial $\rho_3$",
        )
        axis.set_xlim(0, 15)
        axis.set_ylim(-1.2, 1.25)
        axis.set_title(
            figure_title("Fig. 11 - final gain field and initial inversion", quick)
        )
        configure_axes(axis)
        save_plot(figure, output_dir / "fig11.png", show)


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--figures", nargs="+", type=int, default=[10, 11])
    parser.add_argument("--output-dir", type=Path, default=Path("ziolkowski1995"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    arguments = parse_args()
    generate(
        arguments.figures,
        output_dir=arguments.output_dir,
        quick=arguments.quick,
        show=arguments.show,
        verbose=arguments.verbose,
    )


if __name__ == "__main__":
    main()
