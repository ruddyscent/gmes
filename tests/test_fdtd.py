"""Retained eager-Torch FDTD regression coverage."""

import numpy as np
import pytest
import torch

import gmes
from tests.test_torch_fdtd import restore_torch_runtime as restore_torch_runtime


class TestTorchFDTDRetention:
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

        assert simulation.sources.empty
        assert int(simulation.state.step_count) == 1
        for field in simulation.state.fields().values():
            assert not bool(torch.count_nonzero(field))

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

        assert tuple(simulation.plan.shapes["Ez"]) == (11, 11, 1)
        assert int(simulation.state.step_count) == 5
        assert round(abs(float(simulation.state.source_time) - expected_time), 7) == 0
        assert np.isfinite(fields["Ez"]).all()
        assert round(abs(fields["Ez"][5, 5, 0] - expected_center), 7) == 0
        assert round(abs(np.sum(np.abs(fields["Ez"]) ** 2) - expected_energy), 7) == 0

    @pytest.mark.parametrize("bloch", (None, (0.1, 0.2, 0.0)), ids=("real", "bloch"))
    def test_real_and_bloch_checkpoint_replay_preserves_fixed_buffers(self, bloch):
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

        assert addresses == simulation.buffer_addresses()
        replayed_fields = simulation.host_snapshot()
        replayed_state = simulation.checkpoint()["state"]
        for name, expected in uninterrupted_fields.items():
            np.testing.assert_array_equal(replayed_fields[name], expected)
        assert set(replayed_state) == set(uninterrupted_state)
        for name, expected in uninterrupted_state.items():
            assert torch.equal(replayed_state[name], expected), name
