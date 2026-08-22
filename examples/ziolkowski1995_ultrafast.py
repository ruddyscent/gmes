#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Reproduce Figs. 5-9 (ultrafast pulses) from Ziolkowski 1995."""

from argparse import ArgumentParser
from pathlib import Path

from ziolkowski1995_common import (
    ULTRAFAST_PREROLL_S,
    figure_title,
    make_simulation,
    plot_population,
    plot_spatial_pair,
    print_population_summary,
    print_scenario,
    run_snapshots,
    ultrafast_scenario,
)


def generate(
    figures=(5, 6, 7, 8, 9),
    *,
    output_dir=Path("ziolkowski1995"),
    quick=False,
    show=False,
    verbose=False,
):
    requested = set(figures)
    invalid = requested.difference(range(5, 10))
    if invalid:
        raise ValueError(f"ultrafast figures must be in 5-9, got {sorted(invalid)}")

    if requested.intersection((5, 6, 7)):
        scenario = ultrafast_scenario(5, quick)
        print_scenario(scenario)
        elapsed_time = 12.5e-15 + ULTRAFAST_PREROLL_S
        snapshot = run_snapshots(make_simulation(scenario, verbose), (elapsed_time,))[
            elapsed_time
        ]
        print_population_summary(snapshot, scenario)
        if 5 in requested:
            plot_spatial_pair(
                snapshot,
                scenario,
                output_dir / "fig05.png",
                x_limits=(5.5, 8.0),
                title=figure_title("Fig. 5 - complete ultrafast inversion", quick),
                show=show,
            )
        for figure_number, rho_index in ((6, 1), (7, 2)):
            if figure_number in requested:
                plot_population(
                    snapshot,
                    rho_index,
                    output_dir / f"fig{figure_number:02d}.png",
                    title=figure_title(
                        rf"Fig. {figure_number} - $\rho_{{{rho_index}}}$ after inversion",
                        quick,
                    ),
                    x_limits=(0, 10),
                    show=show,
                )

    for figure_number, time_s, limits in (
        (8, 12.5e-15 + ULTRAFAST_PREROLL_S, (5.5, 8.0)),
        (9, 62.5e-15, (12.0, 20.0)),
    ):
        if figure_number not in requested:
            continue
        scenario = ultrafast_scenario(figure_number, quick)
        print_scenario(scenario)
        snapshot = run_snapshots(make_simulation(scenario, verbose), (time_s,))[time_s]
        print_population_summary(snapshot, scenario)
        title = figure_title(
            (
                "Fig. 8 - ultrafast excitation and de-excitation"
                if figure_number == 8
                else "Fig. 9 - delayed ultrafast de-excitation"
            ),
            quick,
        )
        plot_spatial_pair(
            snapshot,
            scenario,
            output_dir / f"fig{figure_number:02d}.png",
            x_limits=limits,
            title=title,
            show=show,
        )


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--figures", nargs="+", type=int, default=[5, 6, 7, 8, 9])
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
