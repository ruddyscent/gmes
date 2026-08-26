"""CPU contract tests for the two-GPU Torch decomposition layer."""

import inspect
import os
import unittest
from unittest import mock

import numpy as np
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


class TwoGpuDecompositionTest(unittest.TestCase):
    def test_all_axes_support_nondivisible_rank_local_geometry(self):
        global_space = gmes.Cartesian((3.5, 3.0, 2.5), 2)
        geometry = [gmes.DefaultMedium(gmes.Dielectric(eps_inf=1.7))]
        for axis in range(3):
            with self.subTest(axis=axis):
                decomposition = choose_two_gpu_decomposition(
                    global_space,
                    geometry,
                    split_axis=axis,
                    device_weights=(3, 2),
                )
                rank0 = rank_local_space(global_space, decomposition, 0)
                rank1 = rank_local_space(global_space, decomposition, 1)
                self.assertEqual(
                    rank0.my_field_size[axis] + rank1.my_field_size[axis],
                    global_space.whole_field_size[axis],
                )
                self.assertEqual(rank0.global_field_offset[axis], 0)
                self.assertEqual(rank1.global_field_offset[axis], decomposition.cut)
                shape0 = list(global_space.whole_field_size)
                shape1 = list(global_space.whole_field_size)
                shape0[axis] = rank0.my_field_size[axis]
                shape1[axis] = rank1.my_field_size[axis]
                axes0 = rank0.component_coordinate_axes(gmes.Ex, tuple(shape0))
                axes1 = rank1.component_coordinate_axes(gmes.Ex, tuple(shape1))
                expected = global_space.component_coordinate_axes(
                    gmes.Ex, tuple(global_space.whole_field_size)
                )[axis]
                np.testing.assert_allclose(
                    np.concatenate((axes0[axis], axes1[axis])), expected
                )

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
        self.assertGreaterEqual(rank0_faster.cut, balanced.cut)
        self.assertNotEqual(rank0_faster.rank_costs, rank0_faster.device_weights)

    def test_surface_cost_prefers_contiguous_leading_axis(self):
        decomposition = choose_two_gpu_decomposition(
            gmes.Cartesian((4, 4, 4), 2),
            [gmes.DefaultMedium(gmes.Dielectric())],
            device_weights=(1, 1),
        )
        self.assertEqual(decomposition.axis, 0)

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
        self.assertEqual(first, second)
        self.assertEqual(first.identity, second.identity)
        self.assertEqual(first.source_crossings, 1)
        self.assertEqual(first.metadata()["axis_name"], "x")


class DistributedLaunchContractTest(unittest.TestCase):
    def test_environment_launch_requires_every_torchrun_variable(self):
        environment = {
            "RANK": "1",
            "WORLD_SIZE": "2",
            "LOCAL_RANK": "1",
            "LOCAL_WORLD_SIZE": "2",
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            launch = gmes.distributed_launch_from_environment()
        self.assertEqual(
            (
                launch.rank,
                launch.world_size,
                launch.local_rank,
                launch.local_world_size,
            ),
            (1, 2, 1, 2),
        )
        with (
            mock.patch.dict(os.environ, {"RANK": "0"}, clear=True),
            self.assertRaisesRegex(gmes.TorchConfigurationError, "torchrun.*missing"),
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
        with self.assertRaisesRegex(
            gmes.TorchConfigurationError, "TorchDistributedSimulation"
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
            self.assertNotIn(marker, source)
        self.assertIn("batch_isend_irecv", source)
        self.assertIn("work.wait()", source)
        self.assertNotIn("torch.cuda.synchronize", source)

    def test_cuda_graph_capture_rejects_cpu_runtime(self):
        simulation = gmes.TorchSimulation(
            space=gmes.Cartesian((2, 2, 2), 2),
            geometry=[gmes.DefaultMedium(gmes.Dielectric())],
            runtime=gmes.TorchRuntimeConfig(device="cpu", cpu_threads=1),
        )
        with self.assertRaisesRegex(
            gmes.TorchConfigurationError, "requires a CUDA runtime"
        ):
            simulation.capture_cuda_graphs()


class TwoGpuBenchmarkContractTest(unittest.TestCase):
    def test_fixed_strong_and_weak_cases_keep_expected_volume_contract(self):
        strong = CASES["strong-mixed"]
        weak = CASES["weak-mixed"]
        self.assertEqual(strong["serial_size"], strong["distributed_size"])
        self.assertEqual(
            int(np.prod(weak["distributed_size"])),
            2 * int(np.prod(weak["serial_size"])),
        )

    def test_trace_interval_math_separates_overlap_and_exposed_time(self):
        communication = [(0, 10), (8, 15), (20, 24)]
        compute = [(5, 12), (22, 30)]
        self.assertEqual(_interval_duration(communication), 19)
        self.assertEqual(_intersection_duration(communication, compute), 9)


class RankLocalOwnershipTest(unittest.TestCase):
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
        self.assertEqual(sum(batch_counts), 1)


if __name__ == "__main__":
    unittest.main()
