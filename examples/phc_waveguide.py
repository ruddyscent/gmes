"""Photonic-crystal waveguide with explicit Torch execution."""

import math

import gmes


def make_simulation():
    geometry = [gmes.DefaultMedium(gmes.Dielectric())]
    geometry += [
        gmes.Cylinder(gmes.Dielectric(8.9), radius=0.38, center=(x, y, 0))
        for x in range(-8, 9)
        for y in range(-4, 5)
        if y != 0
    ]
    geometry.append(gmes.Shell(gmes.Cpml()))
    return gmes.TorchSimulation(
        space=gmes.Cartesian((16, 8, 0), 20),
        geometry=geometry,
        sources=[gmes.PointSource(gmes.Continuous(0.43), (-7, 0, 0), gmes.Ez)],
        runtime=gmes.TorchRuntimeConfig(device="cpu", cpu_threads=1),
    )


def run(until=200):
    simulation = make_simulation()
    simulation.advance(math.ceil(until / simulation.plan.dt))
    return simulation.host_snapshot()


if __name__ == "__main__":
    run()
