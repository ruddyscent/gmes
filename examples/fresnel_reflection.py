"""Oblique TE reflection/transmission through the documented thin layer."""

import math

import gmes


def make_simulation(resolution=100):
    wavelength, angle, a = 10, 0, 20e-9
    dp = gmes.DrudePole(omega=13.1839e15 * a / gmes.c0, gamma=0.109173e15 * a / gmes.c0)
    cps = (
        gmes.CriticalPoint(
            0.273222, -1.18299, 3.88123e15 * a / gmes.c0, 0.452006e15 * a / gmes.c0
        ),
        gmes.CriticalPoint(
            3.04155, -1.09115, 4.20737e15 * a / gmes.c0, 2.35409e15 * a / gmes.c0
        ),
    )
    gold = gmes.DcpPlrc(eps_inf=1.11683, mu_inf=1, dps=(dp,), cps=cps)
    return gmes.TorchSimulation(
        space=gmes.Cartesian((4, 0.5, 0), resolution),
        geometry=[
            gmes.DefaultMedium(gmes.Dielectric()),
            gmes.Cylinder(
                gold, center=(0, 0, 0), axis=(1, 0, 0), radius=1000, height=1
            ),
            gmes.Shell(gmes.Cpml(), minus_y=False, plus_y=False),
        ],
        sources=[
            gmes.GaussianBeam(
                gmes.Continuous(1 / wavelength, width=50),
                gmes.MinusX,
                (0.6, 0, 0),
                (0, 1.5, 1),
                (-math.cos(angle), math.sin(angle), 0),
                (0, 0, 1),
            )
        ],
        bloch=(0, 2 * math.pi / wavelength * math.sin(angle), 0),
        probes=[
            gmes.TorchProbeSpec("Ez", (0.7, 0, 0), coordinates="space"),
            gmes.TorchProbeSpec("Ez", (-0.7, 0, 0), coordinates="space"),
        ],
        runtime=gmes.TorchRuntimeConfig(device="cpu", cpu_threads=1),
    )


def run(until=200):
    simulation = make_simulation()
    simulation.advance(math.ceil(until / simulation.plan.dt))
    return simulation.flush_probes()


if __name__ == "__main__":
    run()
