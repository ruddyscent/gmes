"""Retained eager-Torch FDTD regression coverage."""

import unittest

import numpy as np
import torch

import gmes


class TorchFDTDRetentionTest(unittest.TestCase):
    """Cover retained source, field, and checkpoint behavior without FDTD."""

    def runtime(self):
        """Return the deterministic CPU runtime used by these regressions."""
        return gmes.TorchRuntimeConfig(device="cpu", cpu_threads=1)

    def test_source_free_simulation_has_empty_source_plan_and_zero_fields(self):
        simulation = gmes.TorchSimulation(
            space=gmes.Cartesian(size=(2, 2, 0), resolution=3),
            geometry=[gmes.DefaultMedium(gmes.Dielectric())],
            runtime=self.runtime(),
        )

        simulation.step()

        self.assertTrue(simulation.sources.empty)
        self.assertEqual(int(simulation.state.step_count), 1)
        for field in simulation.state.fields().values():
            self.assertFalse(bool(torch.count_nonzero(field)))

    def test_tmz_cpml_point_source_preserves_the_five_step_reference(self):
        expected_time = 0.7000357133746822
        expected_center = -0.9996801161298625
        expected_energy = 1.5780099423691636
        simulation = gmes.TorchSimulation(
            # The z extent is collapsed, but an explicit inactive-axis spacing
            # retains this historical 2D step under Torch's conservative 3D guard.
            space=gmes.Cartesian(size=(2, 2, 0), resolution=(5, 5, 1)),
            geometry=[
                gmes.DefaultMedium(gmes.Dielectric()),
                gmes.Shell(gmes.Cpml()),
            ],
            sources=[
                gmes.PointSource(
                    gmes.Continuous(freq=0.8, width=0.5),
                    center=(0, 0, 0),
                    component=gmes.Ez,
                )
            ],
            runtime=self.runtime(),
            dt=expected_time / 5,
        )

        simulation.advance(5)
        fields = simulation.host_snapshot()

        self.assertEqual(tuple(simulation.plan.shapes["Ez"]), (11, 11, 1))
        self.assertEqual(int(simulation.state.step_count), 5)
        self.assertAlmostEqual(float(simulation.state.source_time), expected_time)
        self.assertTrue(np.isfinite(fields["Ez"]).all())
        self.assertAlmostEqual(fields["Ez"][5, 5, 0], expected_center)
        self.assertAlmostEqual(np.sum(np.abs(fields["Ez"]) ** 2), expected_energy)

    def test_real_and_bloch_checkpoint_replay_preserves_fixed_buffers(self):
        for bloch in (None, (0.1, 0.2, 0.0)):
            with self.subTest(bloch=bloch):
                simulation = gmes.TorchSimulation(
                    space=gmes.Cartesian(size=(2, 2, 0), resolution=3),
                    geometry=[gmes.DefaultMedium(gmes.Dielectric())],
                    sources=[
                        gmes.PointSource(
                            gmes.Continuous(freq=0.8, width=0.5),
                            center=(0, 0, 0),
                            component=gmes.Ez,
                        )
                    ],
                    runtime=self.runtime(),
                    bloch=bloch,
                )
                addresses = simulation.buffer_addresses()
                simulation.step()
                checkpoint = simulation.checkpoint()
                simulation.advance(2)
                uninterrupted_fields = simulation.host_snapshot()
                uninterrupted_state = simulation.checkpoint()["state"]
                simulation.load_checkpoint(checkpoint).advance(2)

                self.assertEqual(addresses, simulation.buffer_addresses())
                replayed_fields = simulation.host_snapshot()
                replayed_state = simulation.checkpoint()["state"]
                for name, expected in uninterrupted_fields.items():
                    np.testing.assert_array_equal(replayed_fields[name], expected)
                self.assertEqual(set(replayed_state), set(uninterrupted_state))
                for name, expected in uninterrupted_state.items():
                    self.assertTrue(torch.equal(replayed_state[name], expected), name)


if __name__ == "__main__":
    unittest.main()
