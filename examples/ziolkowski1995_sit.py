#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Reproduce Figs. 1-4 (self-induced transparency) from Ziolkowski 1995."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
from ziolkowski1995_common import (
    UNITS,
    configure_axes,
    figure_title,
    make_simulation,
    plot_spatial_pair,
    population_mask,
    print_population_summary,
    print_scenario,
    run_snapshots,
    save_plot,
    sit_scenario,
)


def generate(
    figures=(1, 2, 3, 4),
    *,
    output_dir=Path("ziolkowski1995"),
    quick=False,
    show=False,
    verbose=False,
):
    requested = set(figures)
    invalid = requested.difference(range(1, 5))
    if invalid:
        raise ValueError(f"SIT figures must be in 1-4, got {sorted(invalid)}")

    if requested.intersection((1, 2)):
        scenario = sit_scenario(2, quick)
        print_scenario(scenario)
        times = []
        if 2 in requested:
            times.append(187.5e-15)
        if 1 in requested:
            times.extend((200e-15, 300e-15, 400e-15))
        snapshots = run_snapshots(make_simulation(scenario, verbose), tuple(times))
        for snapshot in snapshots.values():
            print_population_summary(snapshot, scenario)

        if 1 in requested:
            figure, axis = plt.subplots(figsize=(9, 4.8))
            first = snapshots[200e-15]
            axis.plot(
                first.distance_um,
                population_mask(first, scenario),
                label=r"initial $\rho_3$",
            )
            for time_s in (200e-15, 300e-15, 400e-15):
                snapshot = snapshots[time_s]
                axis.plot(
                    snapshot.distance_um,
                    snapshot.electric / UNITS.electric_field(scenario.amplitude_v_m),
                    label=rf"$E_x$ ({time_s / 1e-15:.0f} fs)",
                )
            axis.set_xlim(0, 150)
            axis.set_ylim(-1.15, 1.15)
            axis.set_title(
                figure_title(r"Fig. 1 - $2\pi$ self-induced transparency", quick)
            )
            configure_axes(axis)
            save_plot(figure, output_dir / "fig01.png", show)

        if 2 in requested:
            plot_spatial_pair(
                snapshots[187.5e-15],
                scenario,
                output_dir / "fig02.png",
                x_limits=(25, 55),
                title=figure_title(r"Fig. 2 - local response to a $2\pi$ pulse", quick),
                show=show,
            )

    for figure_number, area_pi in ((3, 1), (4, 4)):
        if figure_number not in requested:
            continue
        scenario = sit_scenario(area_pi, quick)
        print_scenario(scenario)
        snapshot = run_snapshots(make_simulation(scenario, verbose), (187.5e-15,))[
            187.5e-15
        ]
        print_population_summary(snapshot, scenario)
        plot_spatial_pair(
            snapshot,
            scenario,
            output_dir / f"fig{figure_number:02d}.png",
            x_limits=(25, 55),
            title=figure_title(
                rf"Fig. {figure_number} - local response to a "
                + (r"$\pi$ pulse" if area_pi == 1 else rf"${area_pi}\pi$ pulse"),
                quick,
            ),
            show=show,
        )


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--figures", nargs="+", type=int, default=[1, 2, 3, 4])
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
