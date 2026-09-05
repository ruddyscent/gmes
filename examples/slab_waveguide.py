"""Ez propagation in a dielectric slab waveguide."""

import math

import gmes


def make_simulation(verbose=False):
    return gmes.TorchSimulation(
        space=gmes.Cartesian(size=(16, 8, 0), resolution=10),
        geometry=[
            gmes.DefaultMedium(gmes.Dielectric()),
            gmes.Block(gmes.Dielectric(12), size=(16, 1, 1)),
            gmes.Shell(gmes.Cpml()),
        ],
        sources=[gmes.PointSource(gmes.Continuous(0.15), (-7, 0, 0), gmes.Ez)],
        runtime=gmes.TorchRuntimeConfig(device="cpu", cpu_threads=1),
    )


def run(until=200):
    simulation = make_simulation()
    simulation.advance(math.ceil(until / simulation.plan.dt))
    return simulation.host_snapshot()


if __name__ == "__main__":
    run()
