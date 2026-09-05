"""Tensor-native Ziolkowski et al. (1995) source waveform coverage."""

import unittest
from math import cosh, sin

import numpy as np
import torch

import gmes
from gmes.source import (
    PumpProbe,
    SechSinePulse,
    SmoothSine,
    UltrafastPulse,
    UltrafastPulseTrain,
)
from gmes.torch_source import _evaluate_time, _time_parameters


def _ultrafast(time, width):
    if not 0 <= time <= width:
        return 0.0
    x_value = 2 * time / width - 1
    return -4.201355 * x_value * (1 - x_value**2) ** 3


def _smooth(time, omega, period):
    if time < 0:
        return 0.0
    rise_time = 5 * period
    envelope = 1.0 if time >= rise_time else (1 - (time / rise_time - 1) ** 2) ** 4
    return envelope * sin(omega * time)


def _explicit_value(waveform, time):
    if isinstance(waveform, SechSinePulse):
        if not 0 <= time <= waveform.pulse_width:
            return 0.0
        gamma = (time - waveform.pulse_width / 2) / (waveform.pulse_width / 2)
        return sin(waveform.omega * time) / cosh(10 * gamma)
    if isinstance(waveform, UltrafastPulse):
        return _ultrafast(time, waveform.pulse_width)
    if isinstance(waveform, UltrafastPulseTrain):
        return _ultrafast(time, waveform.pulse_width) + waveform.alpha * _ultrafast(
            time - waveform.delay, waveform.pulse_width
        )
    if isinstance(waveform, SmoothSine):
        return _smooth(time, waveform.omega, waveform.period)
    if isinstance(waveform, PumpProbe):
        return _ultrafast(time, waveform.pump.pulse_width) + waveform.beta * _smooth(
            time - waveform.delay, waveform.probe.omega, waveform.probe.period
        )
    raise AssertionError(type(waveform))


def _tensor_value(waveform, time, dtype):
    model, parameters = _time_parameters(waveform)
    output = torch.zeros((1, 1), dtype=dtype)
    _evaluate_time(
        torch.tensor([model], dtype=torch.int8),
        torch.tensor([parameters], dtype=dtype),
        torch.tensor(time, dtype=dtype),
        False,
        output,
    )
    return float(output[0, 0])


class TorchPaperWaveformTest(unittest.TestCase):
    def test_scalar_and_tensor_formulas_match_at_boundaries(self):
        omega = 1.7
        width = 2.5
        waveforms = (
            (SechSinePulse(omega, width), (-0.1, 0, width / 2, width, width + 0.1)),
            (UltrafastPulse(width), (-0.1, 0, width / 2, width, width + 0.1)),
            (
                UltrafastPulseTrain(width, alpha=-0.4, delay_periods=1.5),
                (-0.1, 0, width, 1.5 * width, 2.5 * width, 2.5 * width + 0.1),
            ),
            (
                SmoothSine(omega, width),
                (-0.1, 0, 2.5 * width, 5 * width, 5 * width + 0.1),
            ),
            (
                PumpProbe(omega, width, beta=0.03, delay=3 * width),
                (-0.1, 0, width, 3 * width, 5.5 * width, 8 * width),
            ),
        )
        for waveform, times in waveforms:
            for time in times:
                expected = _explicit_value(waveform, time)
                with self.subTest(waveform=type(waveform).__name__, time=time):
                    self.assertAlmostEqual(
                        waveform.oscillator(time), expected, places=14
                    )
                    self.assertAlmostEqual(
                        _tensor_value(waveform, time, torch.float64),
                        expected,
                        places=13,
                    )
                    self.assertAlmostEqual(
                        _tensor_value(waveform, time, torch.float32),
                        expected,
                        delta=2e-6,
                    )

    def test_composition_delays_and_zero_area(self):
        width = 3.0
        train = UltrafastPulseTrain(width, alpha=0.25, delay_periods=2)
        probe = PumpProbe(1.3, width, beta=0.02, delay=4 * width)
        self.assertEqual(train.oscillator(-1), 0.0)
        self.assertAlmostEqual(
            train.oscillator(train.delay + width / 2),
            0.25 * _ultrafast(width / 2, width),
        )
        self.assertEqual(probe.oscillator(2 * width), 0.0)
        self.assertAlmostEqual(
            probe.oscillator(probe.delay + width / 4),
            probe.beta * _smooth(width / 4, 1.3, width),
        )
        times = np.linspace(0, width, 100_001)
        self.assertAlmostEqual(
            np.trapezoid([_ultrafast(time, width) for time in times], times),
            0.0,
            places=12,
        )
        train_times = np.linspace(0, train.delay + width, 200_001)
        self.assertAlmostEqual(
            np.trapezoid([train.oscillator(time) for time in train_times], train_times),
            0.0,
            places=11,
        )

    def test_malformed_parameters_fail_before_tensor_allocation(self):
        invalid = (
            SechSinePulse(1.0, 0),
            UltrafastPulse(0),
            UltrafastPulseTrain(1.0, alpha=float("nan")),
            SmoothSine(1.0, float("inf")),
            PumpProbe(1.0, 1.0, beta=float("inf"), delay=1.0),
        )
        for waveform in invalid:
            with self.subTest(waveform=type(waveform).__name__):
                with self.assertRaises((TypeError, ValueError)):
                    _time_parameters(waveform)

    def test_paper_waveform_runs_one_real_torch_step(self):
        waveform = SechSinePulse(1.2, 2.0)
        simulation = gmes.TorchSimulation(
            space=gmes.Cartesian((2, 2, 2), 1),
            geometry=(gmes.DefaultMedium(gmes.Dielectric()),),
            sources=(gmes.PointSource(waveform, (0, 0, 0), gmes.Ex),),
            runtime=gmes.TorchRuntimeConfig(device="cpu", cpu_threads=1),
        )
        simulation.step()
        self.assertTrue(
            all(
                np.isfinite(value).all()
                for value in simulation.host_snapshot().values()
            )
        )


if __name__ == "__main__":
    unittest.main()
