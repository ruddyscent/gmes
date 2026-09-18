"""Compact analytical and CPU float64 checks, independent of plotting."""

import numpy as np
import pytest
import torch

from examples.taflove_chapter2 import (
    _scalar_step,
    check_gmes,
    dispersion,
    figure_data,
    gmes_wave,
    scalar_wave,
    unstable_amplification,
)
from tests.test_torch_fdtd import restore_torch_runtime as restore_torch_runtime


@pytest.mark.parametrize(
    ("density", "velocity", "attenuation"),
    [(1, 2, np.arccosh(7)), (3, 2 / 3, 0), (10, 0.9873, 0), (20, 0.99689, 0)],
    ids=["evanescent", "cutoff", "ten-samples", "twenty-samples"],
)
def test_dispersion_landmarks(density, velocity, attenuation):
    actual_velocity, actual_attenuation = dispersion(density)
    assert actual_velocity == pytest.approx(velocity, abs=5e-5)
    assert actual_attenuation == pytest.approx(attenuation, abs=1e-7)


def test_phase_error_converges_quadratically():
    velocity, _ = dispersion([20, 40, 80])
    errors = 1 - velocity
    assert np.all(errors > 0)
    assert np.allclose(errors[:-1] / errors[1:], 4, rtol=0.01)


@pytest.mark.parametrize("pulse", ["rectangle", "gaussian"])
def test_unit_courant_translates_source_without_distortion(pulse):
    result = gmes_wave(1, [198], pulse=pulse)
    retarded_time = 198 - np.arange(201)
    expected = (
        np.asarray((retarded_time >= 40) & (retarded_time < 80), dtype=float)
        if pulse == "rectangle"
        else np.exp(-(((retarded_time - 60) / 20) ** 2))
    )
    expected[retarded_time <= 0] = 0
    assert np.allclose(result.snapshots[198][:201], expected, atol=2e-14, rtol=0)


@pytest.mark.parametrize("figure", ["2.3", "2.4"])
def test_comparison_snapshots_share_physical_time(figure):
    for result in figure_data(figure).values():
        step = next(iter(result.snapshots))
        assert step * result.time_step == pytest.approx(198, abs=1e-12)
        assert np.isfinite(result.snapshots[step]).all()


def _interpolated_peak(values):
    index = np.argmax(values)
    left, middle, right = values[index - 1 : index + 2]
    return middle - (right - left) ** 2 / (8 * (right - 2 * middle + left))


def _width_at_one_over_e(values):
    threshold = values.max() / np.e
    above = np.flatnonzero(values >= threshold)
    left, right = above[0], above[-1]
    left_crossing = (
        left - 1 + ((threshold - values[left - 1]) / (values[left] - values[left - 1]))
    )
    right_crossing = right + (
        (threshold - values[right]) / (values[right + 1] - values[right])
    )
    return right_crossing - left_crossing


def test_interface_reflection_transmission_and_compressed_width():
    values = figure_data("2.5")["interface"].snapshots[240]
    reflected, transmitted = -values[30:135], values[140:200]
    # Source reports -0.603 rounded. This setup is 0.512% from exact -0.6;
    # it does not quite reproduce the source's claimed <0.5% reflection error.
    assert reflected.max() == pytest.approx(0.603, abs=2e-4)
    # Grid sampling misses the transmitted peak; quadratic subcell estimate
    # resolves a ten-cell-wide Gaussian and is disclosed separately.
    assert _interpolated_peak(transmitted) == pytest.approx(0.4, rel=0.005)
    assert _width_at_one_over_e(reflected) == pytest.approx(40, abs=0.3)
    assert _width_at_one_over_e(transmitted) == pytest.approx(10, abs=0.3)


def test_uniform_instability_growth_and_alternating_noise():
    snapshots = figure_data("2.6")["uniform"].snapshots
    amplitudes = [np.max(np.abs(values[1:21])) for values in snapshots.values()]
    assert np.all((np.diff(amplitudes) > 0))
    ratios = np.asarray(amplitudes[1:]) / amplitudes[:-1]
    assert np.all((1.75 < ratios) & (ratios < 2.0))
    assert unstable_amplification(1.0005) ** 10 == pytest.approx(1.8822, abs=1e-4)
    noise = snapshots[220][1:21]
    assert np.all(noise[:-1] * noise[1:] < 0)


def test_local_instability_preserves_main_packet_and_grows_at_defect():
    snapshots = figure_data("2.7")["local"].snapshots
    # Absolute noise is arithmetic-sensitive: check the mechanism, not its
    # tiny plotted magnitude or a bit-for-bit rendering of the textbook.
    before, after = snapshots[190], snapshots[200]
    assert np.argmax(after) == 140
    assert after[140] == pytest.approx(1, abs=0.005)
    assert abs(after[90]) > abs(before[90])
    assert after[89] * after[90] < 0
    assert after[90] * after[91] < 0
    assert abs(after[90]) > abs(after[85])


@pytest.mark.parametrize("courant", [0.5, 0.99], ids=["half-step", "near-limit"])
def test_supported_gmes_matches_scalar_recurrence(courant):
    assert check_gmes(courant) < 1e-12


@pytest.mark.parametrize("courant", [0.99, 0.5], ids=["near-limit", "half-step"])
def test_rectangular_ringing_and_gaussian_low_distortion(courant):
    rectangle = figure_data("2.3")
    gaussian = figure_data("2.4")
    key = str(courant)
    actual_rectangle = next(iter(rectangle[key].snapshots.values()))
    actual_gaussian = next(iter(gaussian[key].snapshots.values()))
    exact_gaussian = next(iter(gaussian["1.0"].snapshots.values()))
    assert actual_rectangle.max() > 1.1
    assert actual_rectangle.min() < -0.1
    assert actual_rectangle[159:175].max() > 0.01
    assert np.max(np.abs(actual_gaussian - exact_gaussian)) < 0.003


def test_highest_frequency_mode_matches_amplification_relation():
    courant = 1.0005
    amplification = unstable_amplification(courant)
    current = torch.tensor([1, -1] * 16, dtype=torch.float64, device="cpu")
    following = _scalar_step(current, -current / amplification, courant**2)
    assert torch.allclose(
        following[1:-1], -amplification * current[1:-1], atol=1e-14, rtol=0
    )


@pytest.mark.parametrize("figure", ["2.3", "2.4", "2.5", "2.6", "2.7"])
def test_gmes_figure_fields_match_independent_scalar_oracle(figure):
    for result in figure_data(figure).values():
        oracle = scalar_wave(
            result.courant,
            result.snapshots,
            time_step=result.time_step,
            pulse="rectangle" if figure == "2.3" else "gaussian",
            half_width=5 if figure == "2.7" else 20,
        )
        # Roundoff grows exponentially in unstable runs; compare the full
        # plotted/source region, while allowing the documented tiny absolute
        # local-noise differences between the two arithmetic orderings.
        tolerance = 1e-8 if figure == "2.7" else 1e-11
        for step, values in result.snapshots.items():
            assert np.max(np.abs(values - oracle.snapshots[step])) < tolerance
