"""Two-dimensional TMz cylindrical-wave propagation in air."""

import argparse
import math

import gmes

SIZE = (10, 10, 0)
DISPLAY_COMPONENTS = (gmes.Ez, gmes.Hx, gmes.Hy)


def make_simulation(verbose=False):
    return gmes.TorchSimulation(
        space=gmes.Cartesian(size=SIZE, resolution=20),
        geometry=[gmes.DefaultMedium(gmes.Dielectric()), gmes.Shell(gmes.Cpml())],
        sources=[gmes.PointSource(gmes.Continuous(0.8), (0, 0, 0), gmes.Ez)],
        runtime=gmes.TorchRuntimeConfig(device="cpu", cpu_threads=1),
    )


def run(until=10):
    simulation = make_simulation()
    simulation.advance(math.ceil(until / simulation.plan.dt))
    return simulation.host_snapshot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int)
    parser.add_argument("--no-plot", action="store_true")
    arguments = parser.parse_args()
    if arguments.steps is None:
        run()
    else:
        simulation = make_simulation()
        simulation.advance(arguments.steps)
