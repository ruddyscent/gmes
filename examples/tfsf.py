"""Launch a vacuum plane wave with a bounded Torch run."""

import math

import gmes


def make_simulation():
    return gmes.TorchSimulation(
        space=gmes.Cartesian((5, 5, 0), 20),
        geometry=[
            gmes.DefaultMedium(gmes.Dielectric()),
            gmes.Shell(gmes.Cpml(), thickness=0.5),
        ],
        sources=[
            gmes.TotalFieldScatteredField(
                gmes.Continuous(0.8), (0, 0, 0), (3, 3, 1), (1, -1, 0), (0, 0, 1)
            )
        ],
        runtime=gmes.TorchRuntimeConfig(device="cpu", cpu_threads=1),
    )


def run(until=200):
    simulation = make_simulation()
    simulation.advance(math.ceil(until / simulation.plan.dt))
    return simulation.host_snapshot()


if __name__ == "__main__":
    run()
