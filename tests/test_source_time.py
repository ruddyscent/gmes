import unittest
from math import pi
from types import SimpleNamespace

import numpy as np

from gmes.constant import PlusX
from gmes.geometry import Cartesian
from gmes.pw_source import TransparentElectricParam, TransparentMagneticParam
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


if __name__ == "__main__":
    unittest.main()
