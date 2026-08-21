import unittest
from pathlib import Path
from subprocess import run
from sys import executable
from textwrap import dedent

import numpy as np

from examples.air2d import DISPLAY_COMPONENTS
from examples.air2d import make_simulation as make_air2d
from examples.slab_waveguide import make_simulation
from gmes.constant import Z

try:
    from gmes.show import Snapshot
except ModuleNotFoundError as error:
    if error.name != "matplotlib":
        raise
    Snapshot = None


class ReadmeQuickStartTest(unittest.TestCase):
    def test_air2d_base_simulation_runs_without_optional_output(self):
        script = dedent("""
            import sys

            class BlockOptionalDependencies:
                def find_spec(self, fullname, path=None, target=None):
                    package = fullname.partition(".")[0]
                    if package in {"matplotlib", "tables"}:
                        raise ModuleNotFoundError(
                            f"No module named '{package}'", name=package
                        )
                    return None

            sys.meta_path.insert(0, BlockOptionalDependencies())

            from examples.air2d import make_simulation

            simulation = make_simulation(verbose=False)
            simulation.init()
            simulation.step()
            assert simulation.time_step.n == 1
            """)

        result = run(
            [executable, "-c", script],
            cwd=Path(__file__).parents[1],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipIf(Snapshot is None, "plot extra is not installed")
    def test_air2d_plot_components_construct_with_plot_extra(self):
        simulation = make_air2d(verbose=False)
        simulation.init()

        for component in DISPLAY_COMPONENTS:
            with self.subTest(component=component.__name__):
                snapshot = Snapshot(
                    simulation,
                    component,
                    Z,
                    cut=0,
                    vrange=(-1, 1),
                    title="",
                    fig_id=0,
                )
                self.assertGreater(snapshot.data.size, 0)


class SlabWaveguideExampleTest(unittest.TestCase):
    def test_finite_slab_covers_the_intended_waveguide(self):
        simulation = make_simulation(verbose=False)
        slab = simulation.geom_list[1]

        self.assertTrue(np.isfinite(slab.size).all())
        for point in ((-6, 0, 0), (0, 0, 0), (6, 0, 0)):
            with self.subTest(point=point):
                material, _ = simulation.geom_tree.material_of_point(point)
                self.assertEqual(material.eps_inf, 12)

        for point in ((-6, 0.6, 0), (0, 0.6, 0), (6, 0.6, 0)):
            with self.subTest(point=point):
                material, _ = simulation.geom_tree.material_of_point(point)
                self.assertEqual(material.eps_inf, 1)


if __name__ == "__main__":
    unittest.main()
