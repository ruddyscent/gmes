import unittest
from math import pi, sin
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from gmes import DefaultMedium, Dielectric, Sphere
from gmes.constant import Ex, PlusX, PlusY
from gmes.geometry import Cartesian, in_range
from gmes.pw_source import (
    TransparentElectricParam,
    TransparentEx,
    TransparentMagneticParam,
)
from gmes.pygeom import GeomBoxTree
from gmes.source import (
    Bandpass,
    Continuous,
    DifferentiatedGaussian,
    TotalFieldScatteredField,
)


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

    def test_transparent_source_sampling_indices_are_integral(self):
        aux_fdtd = SimpleNamespace(space=Cartesian(size=(0, 0, 2), resolution=10))
        parameters = (
            TransparentElectricParam(1, 1, aux_fdtd, (0, 0, 0.13), PlusX),
            TransparentMagneticParam(1, 1, aux_fdtd, (0, 0, 0.13), PlusX),
        )

        for parameter in parameters:
            for index in parameter.samp_idx0[PlusX] + parameter.samp_idx1[PlusX]:
                self.assertIsInstance(index, np.integer)

    def test_transparent_source_merge_preserves_shared_edge_faces(self):
        aux_fdtd = SimpleNamespace(space=Cartesian(size=(0, 0, 2), resolution=10))
        first = TransparentEx()
        second = TransparentEx()
        first.attach(
            (1, 2, 3),
            TransparentElectricParam(2, 3, aux_fdtd, (0, 0, 0.13), PlusX),
        )
        second.attach(
            (1, 2, 3),
            TransparentElectricParam(2, 5, aux_fdtd, (0, 0, 0.27), PlusY),
        )

        first.merge(second)

        parameter = first._param[(1, 2, 3)]
        self.assertEqual(parameter.face_list, [PlusX, PlusY])
        self.assertEqual(parameter.amp, {PlusX: 3, PlusY: 5})
        for values in (
            parameter.samp_idx0,
            parameter.samp_idx1,
            parameter.r0,
            parameter.r1,
        ):
            self.assertEqual(set(values), {PlusX, PlusY})

    def test_tfsf_batches_builtin_geometry_mapping_with_field_clipping(self):
        source = self.make_tfsf()
        source._MAPPING_TILE_SIZE = 3
        space = Cartesian(size=(2, 2, 2), resolution=2)
        default = DefaultMedium(Dielectric(1))
        sphere = Sphere(Dielectric(3), radius=0.6)
        for obj in (default, sphere):
            obj.init(space)
        source.geom_tree = GeomBoxTree((default, sphere))
        field = space.get_ex_storage((Ex,))

        mapped = list(
            source._mapped_source_points(
                space,
                Ex,
                field,
                (-2, -2, -2),
                tuple(value + 2 for value in field.shape),
            )
        )
        expected_indices = [
            index
            for index in np.ndindex(field.shape)
            if in_range(index, field.shape, Ex)
        ]

        self.assertEqual([index for index, *_ in mapped], expected_indices)
        for index, point, material, underneath in mapped:
            self.assertEqual(point, space.ex_index_to_space(*index))
            self.assertEqual(
                (material, underneath), source.geom_tree.material_of_point(point)
            )

    def test_tfsf_custom_geometry_uses_pointwise_fallback(self):
        class CustomSphere(Sphere):
            pass

        source = self.make_tfsf()
        space = Cartesian(size=(2, 2, 2), resolution=2)
        default = DefaultMedium(Dielectric(1))
        sphere = CustomSphere(Dielectric(3), radius=0.6)
        for obj in (default, sphere):
            obj.init(space)
        source.geom_tree = GeomBoxTree((default, sphere))
        field = space.get_ex_storage((Ex,))

        with patch.object(
            space,
            "component_coordinate_axes",
            side_effect=AssertionError("batch path must not be used"),
        ):
            mapped = list(
                source._mapped_source_points(space, Ex, field, (0, 0, 0), field.shape)
            )

        self.assertTrue(mapped)


if __name__ == "__main__":
    unittest.main()
