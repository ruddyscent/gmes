import unittest
from math import pi, sin
from types import SimpleNamespace

import numpy as np
import torch

import gmes
from gmes import DefaultMedium, Dielectric, Sphere
from gmes import source as source_module
from gmes.constant import Ex
from gmes.geometry import Cartesian, in_range
from gmes.pygeom import GeomBoxTree
from gmes.source import (
    Bandpass,
    Continuous,
    DifferentiatedGaussian,
    TotalFieldScatteredField,
)
from gmes.torch_source import TorchTransparentBatch


class SourceTimeTest(unittest.TestCase):
    def make_tfsf(self, **kwargs):
        parameters = {
            "src_time": Continuous(freq=0.8),
            "center": (0, 0, 0),
            "size": (3, 3, 1),
            "direction": (1, 0, 0),
            "polarization": (0, 1, 0),
        }
        parameters.update(kwargs)
        return TotalFieldScatteredField(**parameters)

    def test_tfsf_accepts_finite_center_and_size(self):
        source = self.make_tfsf(center=(-1, 0.5, 2), size=(3, 4, 5))

        np.testing.assert_array_equal(source.center, (-1, 0.5, 2))
        np.testing.assert_array_equal(source.size, (3, 4, 5))

    def test_tfsf_rejects_each_non_finite_center_and_size_component(self):
        for argument in ("center", "size"):
            for axis in range(3):
                for value in (np.inf, -np.inf, np.nan):
                    with self.subTest(argument=argument, axis=axis, value=value):
                        vector = [0, 0, 0] if argument == "center" else [3, 3, 1]
                        vector[axis] = value

                        with self.assertRaisesRegex(
                            ValueError,
                            rf"{argument} must contain only finite values",
                        ):
                            self.make_tfsf(**{argument: vector})

    def test_continuous_source_default_width(self):
        source = Continuous(freq=2)

        self.assertEqual(source.width, 2.5)

    def test_continuous_source_window_and_complex_phase(self):
        source = Continuous(freq=0.5, phase=pi / 2, width=1)
        source.init(cmplx=True)

        self.assertEqual(source.oscillator(-1), 0)
        self.assertEqual(source.oscillator(0), 0j)
        self.assertAlmostEqual(source.oscillator(1).real, 0.0, places=12)
        self.assertAlmostEqual(source.oscillator(1).imag, -1.0, places=12)

    def test_continuous_source_combines_overlapping_ramps(self):
        cases = (
            (4, 1, 0.5, 0.5),
            (4, 1, 2, 1.0),
            (4, 1, 3.5, 0.5),
            (2, 1, 1, 1.0),
            (1.5, 1, 0.75, sin(0.375 * pi) ** 4),
            (1, 2, 0.5, sin(0.125 * pi) ** 4),
        )

        for end, width, time, expected_envelope in cases:
            with self.subTest(end=end, width=width, time=time):
                source = Continuous(freq=1, start=0, end=end, width=width)
                source.init(cmplx=True)

                self.assertAlmostEqual(abs(source.oscillator(time)), expected_envelope)
                self.assertEqual(source.oscillator(end), 0j)
                self.assertEqual(source.oscillator(end + 1e-12), 0)

    def test_bandpass_is_zero_outside_cutoff(self):
        source = Bandpass(freq=1, fwidth=0.5)
        source.init(cmplx=False)

        self.assertEqual(source.oscillator(source.peak_time + source.cutoff + 1), 0)

    def test_differentiated_gaussian_is_antisymmetric(self):
        source = DifferentiatedGaussian(tw=2, t0=5)
        source.init(cmplx=False)

        self.assertAlmostEqual(source.oscillator(4), -source.oscillator(6))
        self.assertEqual(source.oscillator(5), 0.0)

    def test_torch_transparent_plan_uses_integral_consolidated_sampling(self):
        simulation = gmes.TorchSimulation(
            space=Cartesian(size=(4, 4, 4), resolution=2),
            geometry=[DefaultMedium(Dielectric())],
            sources=[self.make_tfsf(src_time=Continuous(freq=0.2))],
            runtime=gmes.TorchRuntimeConfig(device="cpu", cpu_threads=1),
        )
        batches = tuple(
            batch
            for batch in simulation.sources.batches
            if isinstance(batch, TorchTransparentBatch)
        )
        self.assertTrue(batches)
        for batch in batches:
            self.assertEqual(batch.targets.dtype, torch.int64)
            self.assertEqual(batch.samples.dtype, torch.int64)
            self.assertTrue(torch.isfinite(batch.weights).all())
            self.assertEqual(torch.unique(batch.targets).numel(), batch.targets.numel())
            target_size = simulation.state.field(batch.component).numel()
            sample_size = batch.auxiliary.state.field(batch.auxiliary_component).numel()
            self.assertTrue(
                torch.all((0 <= batch.targets) & (batch.targets < target_size))
            )
            self.assertTrue(
                torch.all((0 <= batch.samples) & (batch.samples < sample_size))
            )
            for samples, weights in zip(batch.samples, batch.weights):
                active = samples[weights != 0]
                self.assertEqual(torch.unique(active).numel(), active.numel())

    def test_tfsf_batches_builtin_geometry_mapping_with_field_clipping(self):
        source = self.make_tfsf()
        source._MAPPING_TILE_SIZE = 3
        space = Cartesian(size=(2, 2, 2), resolution=2)
        default = DefaultMedium(Dielectric(1))
        sphere = Sphere(Dielectric(3), radius=0.6)
        for obj in (default, sphere):
            obj.init(space)
        geometry_tree = GeomBoxTree((default, sphere))
        field = space.get_ex_storage((Ex,))

        lowering_token = source_module._torch_tfsf_lowering.set(
            SimpleNamespace(geometry_tree=geometry_tree)
        )
        try:
            mapped = list(
                source._mapped_source_points(
                    space,
                    Ex,
                    field,
                    (-2, -2, -2),
                    tuple(value + 2 for value in field.shape),
                )
            )
        finally:
            source_module._torch_tfsf_lowering.reset(lowering_token)
        expected_indices = [
            index
            for index in np.ndindex(field.shape)
            if in_range(index, field.shape, Ex)
        ]

        self.assertEqual([index for index, *_ in mapped], expected_indices)
        for index, point, material, underneath in mapped:
            self.assertEqual(point, space.ex_index_to_space(*index))
            self.assertEqual(
                (material, underneath), geometry_tree.material_of_point(point)
            )

    def test_tfsf_custom_geometry_uses_pointwise_fallback(self):
        class CustomSphere(Sphere):
            calls = 0

            def in_object(self, point):
                type(self).calls += 1
                return super().in_object(point)

        source = self.make_tfsf()
        space = Cartesian(size=(2, 2, 2), resolution=2)
        default = DefaultMedium(Dielectric(1))
        sphere = CustomSphere(Dielectric(3), radius=0.6)
        for obj in (default, sphere):
            obj.init(space)
        geometry_tree = GeomBoxTree((default, sphere))
        field = space.get_ex_storage((Ex,))

        lowering_token = source_module._torch_tfsf_lowering.set(
            SimpleNamespace(geometry_tree=geometry_tree)
        )
        try:
            mapped = list(
                source._mapped_source_points(space, Ex, field, (0, 0, 0), field.shape)
            )
        finally:
            source_module._torch_tfsf_lowering.reset(lowering_token)

        self.assertGreater(CustomSphere.calls, 0)
        self.assertTrue(mapped)


if __name__ == "__main__":
    unittest.main()
