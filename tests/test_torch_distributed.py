"""CPU contract tests for the two-GPU Torch decomposition layer."""

import inspect
import os
from unittest import mock

import numpy as np
import pytest
import torch

import gmes
from benchmarks.torch_two_gpu import (
    CASES,
    _intersection_duration,
    _interval_duration,
)
from gmes.torch_distributed import (
    TorchHaloExchange,
    choose_two_gpu_decomposition,
    rank_local_space,
)
from tests.test_torch_fdtd import restore_torch_runtime as restore_torch_runtime


class TestTwoGpuDecomposition:
    @pytest.mark.parametrize("axis", range(3), ids=("x", "y", "z"))
    def test_all_axes_support_nondivisible_rank_local_geometry(self, axis):
        global_space = gmes.Cartesian((3.5, 3.0, 2.5), 2)
        geometry = [gmes.DefaultMedium(gmes.Dielectric(eps_inf=1.7))]
        decomposition = choose_two_gpu_decomposition(
            global_space,
            geometry,
            split_axis=axis,
            device_weights=(3, 2),
        )
        rank0 = rank_local_space(global_space, decomposition, 0)
        rank1 = rank_local_space(global_space, decomposition, 1)
        assert (
            rank0.my_field_size[axis] + rank1.my_field_size[axis]
            == global_space.whole_field_size[axis]
        )
        assert rank0.global_field_offset[axis] == 0
        assert rank1.global_field_offset[axis] == decomposition.cut
        shape0 = list(global_space.whole_field_size)
        shape1 = list(global_space.whole_field_size)
        shape0[axis] = rank0.my_field_size[axis]
        shape1[axis] = rank1.my_field_size[axis]
        axes0 = rank0.component_coordinate_axes(gmes.Ex, tuple(shape0))
        axes1 = rank1.component_coordinate_axes(gmes.Ex, tuple(shape1))
        expected = global_space.component_coordinate_axes(
            gmes.Ex, tuple(global_space.whole_field_size)
        )[axis]
        np.testing.assert_allclose(np.concatenate((axes0[axis], axes1[axis])), expected)

    def test_cost_and_device_weights_move_the_cut(self):
        space = gmes.Cartesian((8, 2, 2), 2)
        geometry = [
            gmes.DefaultMedium(gmes.Dielectric()),
            gmes.Block(
                material=gmes.Dm2(
                    eps_inf=1.0,
                    mu_inf=1.0,
                    omega=(1.0,),
                    n_atom=(1.0,),
                ),
                center=(-2.5, 0, 0),
                size=(3, 2, 2),
            ),
        ]
        balanced = choose_two_gpu_decomposition(
            space, geometry, split_axis=0, device_weights=(1, 1)
        )
        rank0_faster = choose_two_gpu_decomposition(
            space, geometry, split_axis=0, device_weights=(3, 1)
        )
        assert rank0_faster.cut >= balanced.cut
        assert rank0_faster.rank_costs != rank0_faster.device_weights

    def test_surface_cost_prefers_contiguous_leading_axis(self):
        decomposition = choose_two_gpu_decomposition(
            gmes.Cartesian((4, 4, 4), 2),
            [gmes.DefaultMedium(gmes.Dielectric())],
            device_weights=(1, 1),
        )
        assert decomposition.axis == 0

    def test_source_crossing_and_metadata_are_deterministic(self):
        space = gmes.Cartesian((4, 3, 2), 2)
        source = gmes.TotalFieldScatteredField(
            gmes.Continuous(0.2),
            center=(0, 0, 0),
            size=(2, 2, 1),
            direction=(1, 0, 0),
            polarization=(0, 1, 0),
        )
        kwargs = {
            "space": space,
            "geometry": [gmes.DefaultMedium(gmes.Dielectric())],
            "sources": [source],
            "device_weights": (1, 1),
            "split_axis": 0,
            "cut": 4,
        }
        first = choose_two_gpu_decomposition(**kwargs)
        second = choose_two_gpu_decomposition(**kwargs)
        assert first == second
        assert first.identity == second.identity
        assert first.source_crossings == 1
        assert first.metadata()["axis_name"] == "x"


class TestDistributedLaunchContract:
    def test_environment_launch_requires_every_torchrun_variable(self):
        environment = {
            "RANK": "1",
            "WORLD_SIZE": "2",
            "LOCAL_RANK": "1",
            "LOCAL_WORLD_SIZE": "2",
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            launch = gmes.distributed_launch_from_environment()
        assert (
            launch.rank,
            launch.world_size,
            launch.local_rank,
            launch.local_world_size,
        ) == (1, 2, 1, 2)
        with (
            mock.patch.dict(os.environ, {"RANK": "0"}, clear=True),
            pytest.raises(gmes.TorchConfigurationError, match="torchrun.*missing"),
        ):
            gmes.distributed_launch_from_environment()

    def test_direct_world_size_two_simulation_is_rejected(self):
        runtime = gmes.TorchRuntimeConfig(
            device="cpu",
            cpu_threads=1,
            launch=gmes.DistributedLaunch(
                world_size=2, local_world_size=2, rank=0, local_rank=0
            ),
        )
        with pytest.raises(
            gmes.TorchConfigurationError, match="TorchDistributedSimulation"
        ):
            gmes.TorchSimulation(
                space=gmes.Cartesian((2, 2, 2), 2),
                geometry=[gmes.DefaultMedium(gmes.Dielectric())],
                runtime=runtime,
            )

    def test_hot_path_has_no_host_or_file_operations(self):
        forbidden = (".cpu(", ".numpy(", ".item(", "open(", "plot(")
        source = "\n".join(
            inspect.getsource(value)
            for value in (
                gmes.TorchSimulation.advance,
                TorchHaloExchange.begin,
                TorchHaloExchange.finish,
            )
        )
        for marker in forbidden:
            assert marker not in source
        assert "batch_isend_irecv" in source
        assert "work.wait()" in source
        assert "torch.cuda.synchronize" not in source

    def test_cuda_graph_capture_rejects_cpu_runtime(self):
        simulation = gmes.TorchSimulation(
            space=gmes.Cartesian((2, 2, 2), 2),
            geometry=[gmes.DefaultMedium(gmes.Dielectric())],
            runtime=gmes.TorchRuntimeConfig(device="cpu", cpu_threads=1),
        )
        with pytest.raises(
            gmes.TorchConfigurationError, match="requires a CUDA runtime"
        ):
            simulation.capture_cuda_graphs()


class TestTwoGpuBenchmarkContract:
    def test_fixed_strong_and_weak_cases_keep_expected_volume_contract(self):
        strong = CASES["strong-mixed"]
        weak = CASES["weak-mixed"]
        assert strong["serial_size"] == strong["distributed_size"]
        assert int(np.prod(weak["distributed_size"])) == 2 * int(
            np.prod(weak["serial_size"])
        )

    def test_trace_interval_math_separates_overlap_and_exposed_time(self):
        communication = [(0, 10), (8, 15), (20, 24)]
        compute = [(5, 12), (22, 30)]
        assert _interval_duration(communication) == 19
        assert _intersection_duration(communication, compute) == 9


class TestRankLocalOwnership:
    def test_point_sources_filter_nonlocal_targets_before_local_validation(self):
        global_space = gmes.Cartesian((5, 4, 4), 1)
        geometry = [gmes.DefaultMedium(gmes.Dielectric())]
        waveform = gmes.Continuous(0.2)
        sources = [
            gmes.PointSource(waveform, center=(-1, 0, 0), component=gmes.Ex),
            gmes.PointSource(waveform, center=(1, 0, 0), component=gmes.Ex, amp=2),
            gmes.PointSource(waveform, center=(-1, 0, 0), component=gmes.Ey, amp=3),
            gmes.PointSource(waveform, center=(1, 0, 0), component=gmes.Ex, amp=4),
        ]
        decomposition = choose_two_gpu_decomposition(
            global_space,
            geometry,
            sources=sources,
            split_axis=0,
            cut=2,
        )
        batches = []
        for rank in (0, 1):
            local = rank_local_space(global_space, decomposition, rank)
            runtime = gmes.TorchRuntimeConfig(
                device="cpu",
                cpu_threads=1,
                launch=gmes.DistributedLaunch(
                    rank=rank,
                    world_size=2,
                    local_rank=rank,
                    local_world_size=2,
                ),
            )
            simulation = gmes.TorchSimulation(
                space=local,
                geometry=geometry,
                sources=sources,
                runtime=runtime,
                _distributed_partition=decomposition,
            )
            batches.append(
                {batch.component: batch for batch in simulation.sources.batches}
            )

        assert decomposition.local_shape(0)[0] == 2
        assert decomposition.local_shape(1)[0] == 3
        assert set(batches[0]) == {"Ex"}
        assert set(batches[1]) == {"Ex", "Ey"}
        assert batches[0]["Ex"].overwrite_targets.numel() == 1
        assert batches[1]["Ex"].overwrite_targets.numel() == 1
        assert batches[1]["Ey"].overwrite_targets.numel() == 1
        assert batches[1]["Ex"].overwrite_amplitudes.tolist() == [4.0]

    def test_point_source_is_owned_by_exactly_one_rank(self):
        global_space = gmes.Cartesian((4, 3, 2), 2)
        geometry = [gmes.DefaultMedium(gmes.Dielectric())]
        source = gmes.PointSource(
            gmes.Continuous(0.2), center=(0, 0, 0), component=gmes.Ez
        )
        decomposition = choose_two_gpu_decomposition(
            global_space,
            geometry,
            sources=[source],
            split_axis=0,
            cut=4,
        )
        batch_counts = []
        for rank in (0, 1):
            local = rank_local_space(global_space, decomposition, rank)
            runtime = gmes.TorchRuntimeConfig(
                device="cpu",
                cpu_threads=1,
                launch=gmes.DistributedLaunch(
                    rank=rank,
                    world_size=2,
                    local_rank=rank,
                    local_world_size=2,
                ),
            )
            simulation = gmes.TorchSimulation(
                space=local,
                geometry=geometry,
                sources=[source],
                runtime=runtime,
                _distributed_partition=decomposition,
            )
            batch_counts.append(len(simulation.sources.batches))
        assert sum(batch_counts) == 1
