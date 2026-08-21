import unittest
from math import pi, sin
from types import SimpleNamespace

import numpy as np

from gmes.constant import PlusX, PlusY
from gmes.geometry import Cartesian
from gmes.pw_source import (
    TransparentElectricParam,
    TransparentEx,
    TransparentMagneticParam,
)
from gmes.source import Bandpass, Continuous, DifferentiatedGaussian


class SourceTimeTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
