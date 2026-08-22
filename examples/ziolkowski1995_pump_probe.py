#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Reproduce Ziolkowski, Arnold, and Gogny, Fig. 12 (pump-probe gain).

Reference: Phys. Rev. A 52, 3082-3094 (1995),
https://doi.org/10.1103/PhysRevA.52.3082.
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
from ziolkowski1995_common import (
    figure_title,
    print_intensity_summary,
    print_population_summary,
    print_scenario,
    pump_probe_scenario,
    run_gain,
    save_plot,
)


def generate(
    figures=(12,),
    *,
    output_dir=Path("ziolkowski1995"),
    quick=False,
    show=False,
    verbose=False,
):
    requested = set(figures)
    if requested.difference((12,)):
        raise ValueError("the pump-probe script only generates Fig. 12")
    if 12 not in requested:
        return

    results = {}
    for delay_periods in (20, 40):
        scenario = pump_probe_scenario(delay_periods, quick)
        print_scenario(scenario)
        results[delay_periods] = run_gain(
            scenario,
            # Continue beyond the displayed interval so the one-period
            # envelope is not evaluated at the end of the sampled record.
            duration_s=625e-15,
            sample_stride=1 if quick else 10,
            verbose=verbose,
            normalization_amplitude_v_m=scenario.amplitude_v_m * 1.0e-4,
        )
        print_intensity_summary(
            results[delay_periods],
            peak_window_s=(delay_periods * 5e-15 + 40e-15, 350e-15),
            late_window_s=(500e-15, 600e-15),
        )
        print_population_summary(results[delay_periods].snapshot, scenario)

    figure, axis = plt.subplots(figsize=(9, 4.8))
    for delay_periods, result in results.items():
        time_fs = result.time_s / 1.0e-15
        axis.plot(
            time_fs,
            result.output_intensity,
            label=rf"output, ${delay_periods}T_p$",
        )
        axis.plot(
            time_fs,
            result.input_intensity,
            linestyle="--",
            label=rf"input, ${delay_periods}T_p$",
        )
    axis.set_xlim(100, 600)
    axis.set_ylim(0, 1.65)
    axis.set_title(figure_title("Fig. 12 - ultrafast pump-probe gain", quick))
    axis.set_xlabel("Time (fs)")
    axis.set_ylabel("Normalized probe intensity")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    save_plot(figure, output_dir / "fig12.png", show)


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("ziolkowski1995"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    arguments = parse_args()
    generate(
        output_dir=arguments.output_dir,
        quick=arguments.quick,
        show=arguments.show,
        verbose=arguments.verbose,
    )


if __name__ == "__main__":
    main()
