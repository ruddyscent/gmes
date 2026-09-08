"""Tensor-native Ziolkowski et al. (1995) source waveform coverage."""

from math import cosh, sin

import numpy as np
import pytest
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
from tests.test_torch_fdtd import restore_torch_runtime as restore_torch_runtime


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


class TestTorchPaperWaveform:
    @pytest.mark.parametrize(
        ("waveform_index", "time_index"),
        (
            pytest.param(0, 0, id="sech-before"),
            pytest.param(0, 1, id="sech-start"),
            pytest.param(0, 2, id="sech-center"),
            pytest.param(0, 3, id="sech-end"),
            pytest.param(0, 4, id="sech-after"),
            pytest.param(1, 0, id="ultrafast-before"),
            pytest.param(1, 1, id="ultrafast-start"),
            pytest.param(1, 2, id="ultrafast-center"),
            pytest.param(1, 3, id="ultrafast-end"),
            pytest.param(1, 4, id="ultrafast-after"),
            pytest.param(2, 0, id="train-before"),
            pytest.param(2, 1, id="train-start"),
            pytest.param(2, 2, id="train-first-end"),
            pytest.param(2, 3, id="train-second-start"),
            pytest.param(2, 4, id="train-second-end"),
            pytest.param(2, 5, id="train-after"),
            pytest.param(3, 0, id="smooth-before"),
            pytest.param(3, 1, id="smooth-start"),
            pytest.param(3, 2, id="smooth-mid-rise"),
            pytest.param(3, 3, id="smooth-end-rise"),
            pytest.param(3, 4, id="smooth-plateau"),
            pytest.param(4, 0, id="pump-probe-before"),
            pytest.param(4, 1, id="pump-probe-start"),
            pytest.param(4, 2, id="pump-probe-pump-end"),
            pytest.param(4, 3, id="pump-probe-probe-start"),
            pytest.param(4, 4, id="pump-probe-probe-mid-rise"),
            pytest.param(4, 5, id="pump-probe-probe-end-rise"),
        ),
    )
    def test_scalar_and_tensor_formulas_match_at_boundaries(
        self, waveform_index, time_index
    ):
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
        waveform, times = waveforms[waveform_index]
        time = times[time_index]
        expected = _explicit_value(waveform, time)
        assert round(abs(waveform.oscillator(time) - expected), 14) == 0
        assert (
            round(abs(_tensor_value(waveform, time, torch.float64) - expected), 13) == 0
        )
        assert abs(_tensor_value(waveform, time, torch.float32) - expected) <= 2e-6

    def test_composition_delays_and_zero_area(self):
        width = 3.0
        train = UltrafastPulseTrain(width, alpha=0.25, delay_periods=2)
        probe = PumpProbe(1.3, width, beta=0.02, delay=4 * width)
        assert train.oscillator(-1) == 0.0
        assert (
            round(
                abs(
                    train.oscillator(train.delay + width / 2)
                    - 0.25 * _ultrafast(width / 2, width)
                ),
                7,
            )
            == 0
        )
        assert probe.oscillator(2 * width) == 0.0
        assert (
            round(
                abs(
                    probe.oscillator(probe.delay + width / 4)
                    - probe.beta * _smooth(width / 4, 1.3, width)
                ),
                7,
            )
            == 0
        )
        times = np.linspace(0, width, 100_001)
        assert (
            round(
                abs(
                    np.trapezoid([_ultrafast(time, width) for time in times], times)
                    - 0.0
                ),
                12,
            )
            == 0
        )
        train_times = np.linspace(0, train.delay + width, 200_001)
        assert (
            round(
                abs(
                    np.trapezoid(
                        [train.oscillator(time) for time in train_times], train_times
                    )
                    - 0.0
                ),
                11,
            )
            == 0
        )

    @pytest.mark.parametrize(
        "waveform",
        (
            SechSinePulse(1.0, 0),
            UltrafastPulse(0),
            UltrafastPulseTrain(1.0, alpha=float("nan")),
            SmoothSine(1.0, float("inf")),
            PumpProbe(1.0, 1.0, beta=float("inf"), delay=1.0),
        ),
        ids=(
            "sech-zero-width",
            "ultrafast-zero-width",
            "train-nan-alpha",
            "smooth-infinite-period",
            "probe-infinite-beta",
        ),
    )
    def test_malformed_parameters_fail_before_tensor_allocation(self, waveform):
        with pytest.raises((TypeError, ValueError)):
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
        assert all(
            np.isfinite(value).all() for value in simulation.host_snapshot().values()
        )
