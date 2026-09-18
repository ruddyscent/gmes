"""Chapter 2 wave figures using GMES, with a scalar finite-difference oracle.

The magnetic field Hy obeys Eq. (2.16) on a one-dimensional line with eps=1
and spatially varying mu. Intentional instability uses faster finite material
regions; no production timestep guard or solver behavior is changed.
See TAFLOVE_CHAPTER2.md for source locations and unspecified setup choices.
"""

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import gmes

FIGURES = tuple(f"2.{number}" for number in range(1, 8))


@dataclass(frozen=True)
class WaveResult:
    courant: np.ndarray
    snapshots: dict[int, np.ndarray]
    time_step: float


def dispersion(samples_per_wavelength, courant=0.5):
    """Return vp/c and alpha*dx on the principal temporal Nyquist branch."""
    density = np.asarray(samples_per_wavelength, dtype=np.float64)
    if not 0 < courant <= 1 or np.any(~np.isfinite(density)):
        raise ValueError("finite density and 0 < courant <= 1 are required")
    if np.any(density < 2 * courant):
        raise ValueError("density must meet temporal Nyquist sampling N >= 2*S")
    xi = 1 - 2 * (np.sin(np.pi * courant / density) / courant) ** 2
    phase = np.arccos(np.clip(xi, -1, 1))
    velocity = 2 * np.pi / (density * phase)
    attenuation = np.arccosh(np.maximum(-xi, 1))
    return velocity, attenuation


def unstable_amplification(courant):
    """Highest-frequency amplitude multiplier per step, Eq. (2.49)."""
    if courant < 1:
        raise ValueError("unstable amplification requires S >= 1")
    return (courant + np.sqrt(courant**2 - 1)) ** 2


def _scalar_step(current, previous, speed_squared):
    following = torch.zeros_like(current)
    following[1:-1] = (
        speed_squared * (current[2:] - 2 * current[1:-1] + current[:-2])
        + 2 * current[1:-1]
        - previous[1:-1]
    )
    return following


def scalar_wave(
    courant, snapshots, *, pulse="gaussian", half_width=20.0, time_step=1.0, cells=512
):
    """Evaluate Eq. (2.16) on CPU in float64, without compiled kernels.

    Hard source at i=0 peaks at t=60. Zero initial fields and a distant zero
    right boundary are explicit conventions, not specified textbook details.
    Gaussian half_width is the half width at 1/e. Rectangular width is 40.
    """
    requested = tuple(sorted(set(snapshots)))
    if not requested or requested[0] < 0 or any(int(n) != n for n in requested):
        raise ValueError("snapshots must be nonnegative integer steps")
    if pulse not in {"gaussian", "rectangle"}:
        raise ValueError("pulse must be gaussian or rectangle")
    if time_step <= 0 or half_width <= 0:
        raise ValueError("time_step and half_width must be positive")
    speeds = np.broadcast_to(np.asarray(courant, dtype=np.float64), (cells + 1,)).copy()
    if not np.all(np.isfinite(speeds)) or np.any(speeds <= 0):
        raise ValueError("Courant numbers must be finite and positive")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        speed_squared = torch.tensor(
            speeds[1:-1] ** 2, dtype=torch.float64, device="cpu"
        )
        previous = torch.zeros(cells + 1, dtype=torch.float64, device="cpu")
        current = previous.clone()
        saved = {0: current.numpy().copy()} if 0 in requested else {}
        for step in range(1, requested[-1] + 1):
            following = _scalar_step(current, previous, speed_squared)
            time = step * time_step
            following[0] = (
                float(40 <= time < 80)
                if pulse == "rectangle"
                else np.exp(-(((time - 60) / half_width) ** 2))
            )
            previous, current = current, following
            if step in requested:
                saved[step] = current.numpy().copy()
        return WaveResult(speeds, saved, time_step)
    finally:
        torch.set_num_threads(previous_threads)


def gmes_wave(courant, snapshots, *, pulse="gaussian", half_width=20.0, time_step=1.0):
    """Evolve scalar u=Hy through GMES's public Maxwell solver.

    dz=1, eps=1 and mu=(dt/S)**2 give the desired local Courant number.
    A finite 1800-cell region sits inside a slower mu=4 default medium. Its
    interfaces and the periodic domain edges cannot reach the plotted region
    during these <=396-step experiments. The constructor checks the default
    medium CFL only: acceptance does not imply local stability.

    Hy at index k lives at z=k-1/2-N/2. The hard source is index 768 (i=0).
    Since GMES updates E before H, loading the prescribed magnetic source
    after each full step gives the scalar Dirichlet boundary at integer n;
    the next electric update sees that corrected H. All interior updates,
    including the intentionally unstable ones, are performed by GMES.
    """
    requested = tuple(sorted(set(snapshots)))
    if not requested or requested[0] < 0 or any(int(n) != n for n in requested):
        raise ValueError("snapshots must be nonnegative integer steps")
    if requested[-1] > 396:
        raise ValueError("this padded figure domain supports at most 396 steps")
    if pulse not in {"gaussian", "rectangle"}:
        raise ValueError("pulse must be gaussian or rectangle")
    if not np.isfinite(time_step) or time_step <= 0 or half_width <= 0:
        raise ValueError("time_step and half_width must be finite and positive")
    speeds = np.broadcast_to(np.asarray(courant, dtype=np.float64), (513,)).copy()
    if not np.all(np.isfinite(speeds)) or np.any(speeds <= 0):
        raise ValueError("Courant numbers must be finite and positive")
    source_index, domain_cells = 768, 2048
    source_z = source_index - 0.5 - domain_cells / 2
    geometry = [
        gmes.DefaultMedium(gmes.Dielectric(eps_inf=1, mu_inf=4)),
        gmes.Block(
            gmes.Dielectric(eps_inf=1, mu_inf=(time_step / speeds[0]) ** 2),
            size=(np.inf, np.inf, 1800),
        ),
    ]
    # Interfaces halfway between magnetic samples realize the scalar
    # coefficient at exactly i=140, or only i=90 for the isolated defect.
    changes = np.flatnonzero(np.diff(speeds)) + 1
    bounds = np.r_[changes, len(speeds)]
    for low, high in zip(bounds[:-1], bounds[1:]):
        lower_z = source_z + low - 0.5
        upper_z = source_z + high - 0.5
        geometry.append(
            gmes.Block(
                gmes.Dielectric(eps_inf=1, mu_inf=(time_step / speeds[low]) ** 2),
                center=(0, 0, (lower_z + upper_z) / 2),
                size=(np.inf, np.inf, upper_z - lower_z),
            )
        )
    previous_threads = torch.get_num_threads()
    try:
        simulation = gmes.TorchSimulation(
            space=gmes.Cartesian(size=(0, 0, domain_cells), resolution=(0.01, 0.01, 1)),
            geometry=geometry,
            dt=time_step,
            runtime=gmes.TorchRuntimeConfig(
                device="cpu", precision="float64", cpu_threads=1, compile_policy="eager"
            ),
        )
        saved = {0: np.zeros(513)} if 0 in requested else {}
        for step in range(1, requested[-1] + 1):
            simulation.step()
            fields = simulation.host_snapshot()
            time = step * time_step
            value = (
                float(40 <= time < 80)
                if pulse == "rectangle"
                else np.exp(-(((time - 60) / half_width) ** 2))
            )
            # Both x planes carry the same one-dimensional Hy source.
            fields["Hy"][:, 0, source_index] = value
            simulation.load_host_fields(fields)
            if step in requested:
                saved[step] = fields["Hy"][
                    1, 0, source_index : source_index + 513
                ].copy()
        return WaveResult(speeds, saved, time_step)
    finally:
        torch.set_num_threads(previous_threads)


def figure_data(figure):
    """Return numerical arrays without importing plotting dependencies."""
    if figure not in FIGURES:
        raise ValueError(f"unknown figure {figure}")
    if figure in {"2.1", "2.2"}:
        density = (
            np.linspace(1, 10, 901) if figure == "2.1" else np.linspace(3, 80, 771)
        )
        velocity, attenuation = dispersion(density)
        return {"density": density, "velocity": velocity, "attenuation": attenuation}
    if figure in {"2.3", "2.4"}:
        # 198 is a common physical snapshot time for S=1, .99, and .5.
        pulse = "rectangle" if figure == "2.3" else "gaussian"
        return {
            str(speed): gmes_wave(
                speed, [round(198 / speed)], pulse=pulse, time_step=speed
            )
            for speed in (1.0, 0.99, 0.5)
        }
    if figure == "2.5":
        speeds = np.ones(513)
        speeds[140:] = 0.25
        return {"interface": gmes_wave(speeds, [240])}
    if figure == "2.6":
        return {"uniform": gmes_wave(1.0005, [200, 210, 220])}
    speeds = np.ones(513)
    speeds[90] = 1.075
    return {"local": gmes_wave(speeds, [190, 200], half_width=5)}


def check_gmes(courant=0.5):
    """Compare a scalar Gaussian to GMES Ex on a periodic vacuum line.

    Zero Hy makes the first electric update unchanged; eliminating Hy from
    subsequent Yee steps gives Eq. (2.16). Transverse dr=100 keeps the public
    three-axis CFL bound above .99, while derivatives vanish along those axes.
    This checks stable evolution, not the intentionally unstable figures.
    """
    previous_threads = torch.get_num_threads()
    try:
        simulation = gmes.TorchSimulation(
            space=gmes.Cartesian(size=(0, 0, 256), resolution=(0.01, 0.01, 1)),
            geometry=[gmes.DefaultMedium(gmes.Dielectric())],
            dt=courant,
            runtime=gmes.TorchRuntimeConfig(
                device="cpu", precision="float64", cpu_threads=1, compile_policy="eager"
            ),
        )
        fields = simulation.host_snapshot()
        initial = np.exp(-(((np.arange(257) - 128) / 20) ** 2))
        fields["Ex"][:] = initial
        simulation.load_host_fields(fields)
        previous = torch.from_numpy(initial.copy())
        current = previous.clone()
        for _ in range(39):
            following = _scalar_step(current, previous, courant**2)
            previous, current = current, following
        simulation.advance(40)
        actual = simulation.host_snapshot()["Ex"][0, 0]
        # The distant Gaussian tails are below rounding at the boundaries.
        return float(np.max(np.abs(actual[40:-40] - current.numpy()[40:-40])))
    finally:
        torch.set_num_threads(previous_threads)


def plot_figure(figure, output):
    """Render one numbered figure (including both panels where applicable)."""
    import matplotlib.pyplot as plt

    data = figure_data(figure)
    panels = 2 if figure in {"2.3", "2.4", "2.6", "2.7"} else 1
    fig, axes = plt.subplots(panels, 1, figsize=(7, 3.5 * panels), squeeze=False)
    axes = axes[:, 0]
    if figure == "2.1":
        axes[0].plot(data["density"], data["velocity"], label="phase velocity")
        axes[0].set_ylabel("Phase velocity / c")
        right = axes[0].twinx()
        right.plot(data["density"], data["attenuation"], "--", color="tab:orange")
        right.set_ylabel("Attenuation × dx")
        axes[0].set_xlabel("Samples per free-space wavelength")
    elif figure == "2.2":
        axes[0].semilogy(data["density"], 100 * (1 - data["velocity"]))
        axes[0].set_ylabel("Phase velocity error (%)")
        axes[0].set_xlabel("Samples per free-space wavelength")
    elif figure in {"2.3", "2.4"}:
        for axis, speed in zip(axes, ("0.99", "0.5")):
            for key in ("1.0", speed):
                axis.plot(next(iter(data[key].snapshots.values())), label=f"S={key}")
            axis.set_xlim(0, 200)
            axis.legend()
    else:
        result = next(iter(data.values()))
        for axis in axes:
            for step, values in result.snapshots.items():
                axis.plot(values, label=f"n={step}")
            axis.legend()
        axes[0].set_xlim(1, {"2.5": 200, "2.6": 220, "2.7": 160}[figure])
        if figure == "2.5":
            axes[0].axvline(140, color="gray", linestyle=":", label="interface")
            axes[0].legend()
        if panels == 2:
            axes[1].set_xlim((1, 20) if figure == "2.6" else (70, 110))
            values = np.concatenate(
                [
                    v[1:21] if figure == "2.6" else v[70:111]
                    for v in result.snapshots.values()
                ]
            )
            low, high = values.min(), values.max()
            axes[1].set_ylim(low - 0.05 * (high - low), high + 0.05 * (high - low))
            if figure == "2.7":
                # The source's roundoff-dependent instability is far smaller
                # here; retain its full window and show actual noise in an inset.
                inset = axes[1].inset_axes([0.12, 0.48, 0.42, 0.38])
                for step, values in result.snapshots.items():
                    inset.plot(np.arange(85, 96), values[85:96], label=f"n={step}")
                inset.set_title("Actual local noise", fontsize=8)
                inset.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
                inset.tick_params(labelsize=7)

    if figure not in {"2.1", "2.2"}:
        for axis in axes:
            axis.set_xlabel("Position (cells)")
            axis.set_ylabel("Scalar wave amplitude")
    backend = "analytical dispersion" if figure in {"2.1", "2.2"} else "GMES"
    fig.suptitle(f"Taflove Chapter 2, Fig. {figure} ({backend})")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=FIGURES, default="2.1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-gmes", action="store_true")
    args = parser.parse_args()
    if args.check_gmes:
        for speed in (0.5, 0.99):
            print(
                f"GMES S={speed}: max absolute scalar discrepancy {check_gmes(speed):.3g}"
            )
    if args.output:
        plot_figure(args.figure, args.output)
        print(f"Saved {args.output}")
    else:
        data = figure_data(args.figure)
        if args.figure in {"2.1", "2.2"}:
            print(f"Fig. {args.figure}: {len(data['density'])} analytical samples")
        else:
            for label, result in data.items():
                for step, values in result.snapshots.items():
                    print(
                        f"Fig. {args.figure}, {label}, n={step}, t={step * result.time_step:g}: "
                        f"min={values.min():.6g}, max={values.max():.6g}"
                    )


if __name__ == "__main__":
    main()
