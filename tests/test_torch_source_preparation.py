"""Regression coverage for Torch source preparation before device allocation."""

import math
import unittest
from dataclasses import fields
from types import SimpleNamespace
from unittest import mock

import numpy as np

import gmes
from gmes.torch_distributed import choose_two_gpu_decomposition, rank_local_space
from gmes.torch_fdtd import TorchSimulationPlan, TorchSimulationState


def _geometry():
    return [gmes.DefaultMedium(gmes.Dielectric(eps_inf=2.0, mu_inf=1.0))]


def _runtime(**kwargs):
    return gmes.TorchRuntimeConfig(device="cpu", cpu_threads=1, **kwargs)


def _simulation(*, sources=(), space=None, runtime=None, **kwargs):
    return gmes.TorchSimulation(
        space=space or gmes.Cartesian((2, 2, 2), 1),
        geometry=_geometry(),
        sources=sources,
        runtime=runtime or _runtime(),
        **kwargs,
    )


class _Records:
    def __init__(self, records):
        self.records = records

    def lower_torch_source(self, context):
        return self.records


class TorchSourcePreparationTest(unittest.TestCase):
    def test_invalid_sources_fail_before_plan_or_state_allocation(self):
        valid = gmes.TorchPointSourceRecord(
            "Ex", (1, 1, 1), gmes.Continuous(0.2, width=1)
        )

        class NoHook:
            pass

        class NonCallableHook:
            lower_torch_source = "not callable"

        class RecordSubclass(gmes.TorchPointSourceRecord):
            pass

        class _PrivateRecord:
            component = "Ex"
            target = (1, 1, 1)
            source_time = gmes.Continuous(0.2, width=1)
            amplitude = 1.0
            current_scale = None

        invalid = {
            "filename": gmes.PointSource(
                gmes.Continuous(0.2, width=1), (0, 0, 0), gmes.Ex, filename="x"
            ),
            "missing hook": NoHook(),
            "noncallable hook": NonCallableHook(),
            "subclass record": _Records((RecordSubclass(**valid.__dict__),)),
            "private record": _Records((_PrivateRecord(),)),
            "spoof record": _Records((SimpleNamespace(**valid.__dict__),)),
            "component": _Records(
                (gmes.TorchPointSourceRecord("Jx", (1, 1, 1), valid.source_time),)
            ),
            "indices": _Records(
                (gmes.TorchPointSourceRecord("Ex", (9, 1, 1), valid.source_time),)
            ),
            "waveform": _Records(
                (gmes.TorchPointSourceRecord("Ex", (1, 1, 1), object()),)
            ),
            "nan amplitude": _Records(
                (
                    gmes.TorchPointSourceRecord(
                        "Ex", (1, 1, 1), valid.source_time, math.nan
                    ),
                )
            ),
            "infinite amplitude": _Records(
                (
                    gmes.TorchPointSourceRecord(
                        "Ex", (1, 1, 1), valid.source_time, math.inf
                    ),
                )
            ),
            "negative infinite amplitude": _Records(
                (
                    gmes.TorchPointSourceRecord(
                        "Ex", (1, 1, 1), valid.source_time, -math.inf
                    ),
                )
            ),
            "nan current": _Records(
                (
                    gmes.TorchPointSourceRecord(
                        "Ex", (1, 1, 1), valid.source_time, 1, math.nan
                    ),
                )
            ),
            "infinite current": _Records(
                (
                    gmes.TorchPointSourceRecord(
                        "Ex", (1, 1, 1), valid.source_time, 1, -math.inf
                    ),
                )
            ),
            "positive infinite current": _Records(
                (
                    gmes.TorchPointSourceRecord(
                        "Ex", (1, 1, 1), valid.source_time, 1, math.inf
                    ),
                )
            ),
        }
        plan_init = TorchSimulationPlan.__init__
        state_init = TorchSimulationState.__init__
        for name, source in invalid.items():
            with self.subTest(name=name):
                allocations = []

                def plan_spy(instance, *args, **kwargs):
                    allocations.append("plan")
                    return plan_init(instance, *args, **kwargs)

                def state_spy(instance, *args, **kwargs):
                    allocations.append("state")
                    return state_init(instance, *args, **kwargs)

                with (
                    mock.patch.object(TorchSimulationPlan, "__init__", plan_spy),
                    mock.patch.object(TorchSimulationState, "__init__", state_spy),
                    self.assertRaises((TypeError, ValueError)),
                ):
                    _simulation(sources=[source])
                self.assertEqual(allocations, [])

    def test_exact_extension_lowers_once_with_frozen_context_and_infinite_end(self):
        waveform = gmes.Continuous(0.2, width=1)
        self.assertTrue(math.isinf(waveform.end))

        class Extension:
            calls = 0

            def lower_torch_source(self, context):
                self.calls += 1
                self.context = context
                return (gmes.TorchPointSourceRecord("Ez", (1, 1, 1), waveform),)

        source = Extension()
        simulation = _simulation(sources=[source])
        self.assertEqual(source.calls, 1)
        self.assertEqual(
            tuple(field.name for field in fields(type(source.context))),
            ("space", "geometry_tree", "paired_real", "dtype", "device", "dt"),
        )
        self.assertIs(source.context.space, simulation.space)
        self.assertIs(source.context.geometry_tree, simulation.geom_tree)
        self.assertEqual(source.context.paired_real, simulation.state.paired_real)
        self.assertEqual(source.context.dtype, simulation.dtype)
        self.assertEqual(source.context.device, simulation.device)
        self.assertEqual(source.context.dt, simulation.plan.dt)
        context_values = tuple(
            getattr(source.context, name)
            for name in (
                "space",
                "geometry_tree",
                "paired_real",
                "dtype",
                "device",
                "dt",
            )
        )

        checkpoint = simulation.advance(1).checkpoint()
        simulation.load_checkpoint(checkpoint).advance(1)
        self.assertEqual(source.calls, 1)
        self.assertEqual(
            tuple(
                getattr(source.context, name)
                for name in (
                    "space",
                    "geometry_tree",
                    "paired_real",
                    "dtype",
                    "device",
                    "dt",
                )
            ),
            context_values,
        )

    def test_point_and_current_sources_apply_expected_local_values(self):
        waveform = gmes.Continuous(0.0, width=1)
        simulation = _simulation(
            sources=[
                gmes.PointSource(waveform, (0, 0, 0), gmes.Ex, amp=2.0),
                gmes.PointSource(waveform, (-1, 0, 0), gmes.Jx, amp=0.5),
            ]
        )
        waveform.init(False)
        expected_waveform = waveform.oscillator(0.5 * simulation.plan.dt)
        simulation.step()
        self.assertAlmostEqual(
            float(simulation.state.ex[1, 1, 1]), 2.0 * expected_waveform
        )
        self.assertAlmostEqual(
            float(simulation.state.ex[0, 1, 1]),
            -0.5 * simulation.plan.dt * expected_waveform / 2.0,
        )

        shared_waveform = gmes.Continuous(0.0, width=1)
        shared = _simulation(
            sources=[
                gmes.PointSource(shared_waveform, (0, 0, 0), gmes.Ex, amp=2.0),
                gmes.PointSource(shared_waveform, (0, 0, 0), gmes.Jx, amp=0.5),
            ]
        )
        shared_waveform.init(False)
        shared_expected_waveform = shared_waveform.oscillator(0.5 * shared.plan.dt)
        shared.step()
        self.assertAlmostEqual(
            float(shared.state.ex[1, 1, 1]),
            -0.5 * shared.plan.dt * shared_expected_waveform / 2.0,
        )
        shared_batch = next(
            batch for batch in shared.sources.batches if batch.component == "Ex"
        )
        self.assertEqual(shared_batch.overwrite_targets.numel(), 0)
        self.assertEqual(shared_batch.additive_targets.numel(), 1)

    def test_point_source_is_materialized_only_by_its_local_rank(self):
        global_space = gmes.Cartesian((4, 3, 2), 2)
        geometry = _geometry()
        source = gmes.PointSource(gmes.Continuous(0.2), (0, 0, 0), gmes.Ez)
        decomposition = choose_two_gpu_decomposition(
            global_space, geometry, sources=[source], split_axis=0, cut=4
        )
        batch_counts = []
        for rank in (0, 1):
            runtime = _runtime(
                launch=gmes.DistributedLaunch(
                    rank=rank, world_size=2, local_rank=rank, local_world_size=2
                )
            )
            simulation = _simulation(
                space=rank_local_space(global_space, decomposition, rank),
                runtime=runtime,
                sources=[source],
                _distributed_partition=decomposition,
            )
            batch_counts.append(len(simulation.sources.batches))
        self.assertEqual(sum(batch_counts), 1)


if __name__ == "__main__":
    unittest.main()
