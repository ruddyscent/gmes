"""Run a small custom hemisphere simulation with public geometry opt-in."""

import numpy as np
from numpy.typing import NDArray

import gmes
from gmes.pygeom import Vector3


@gmes.vectorized_geometry
class Hemisphere(gmes.Sphere):
    """Keep the half of a sphere at or above its center's x coordinate."""

    def in_object(self, point: Vector3) -> bool:
        """Return scalar containment, including the cut plane."""
        return bool(point[0] >= self.center[0] and super().in_object(point))

    def _contains_points(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        z: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        """Apply the same inclusive cut to a bounded coordinate tile."""
        return super()._contains_points(x, y, z) & (x >= self.center[0])


def main() -> None:
    """Advance three eager CPU steps without plotting or writing output files."""
    simulation = gmes.TorchSimulation(
        space=gmes.Cartesian(size=(2, 2, 2), resolution=4),
        geometry=[
            gmes.DefaultMedium(gmes.Dielectric()),
            Hemisphere(gmes.Dielectric(2), radius=0.6),
        ],
        sources=[
            gmes.PointSource(gmes.Continuous(0.8), (0, 0, 0), gmes.Ez),
        ],
        runtime=gmes.TorchRuntimeConfig(
            device="cpu",
            precision="float64",
            cpu_threads=1,
            execution_policy="dense",
            compile_policy="eager",
        ),
    )
    simulation.advance(3)
    peak = max(
        float(np.max(np.abs(field))) for field in simulation.host_snapshot().values()
    )
    print(f"Completed 3 steps; peak field magnitude: {peak:.6g}")


if __name__ == "__main__":
    main()
