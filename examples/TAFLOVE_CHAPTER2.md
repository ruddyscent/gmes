# Taflove Chapter 2 numerical examples

Source: Allen Taflove and Susan C. Hagness, *Computational Electrodynamics:
The Finite-Difference Time-Domain Method*, third edition (2005), Chapter 2.
This inventory was checked against the supplied third-edition PDF. Printed
page numbers are authoritative; PDF page numbers below are 1-based and depend
on the scan.

## Scope and solver coverage

Chapter 2 contains **seven numbered figures and eleven plotted panels**:
1 + 1 + 2 + 2 + 1 + 2 + 2 = 11. Issue #136's original description of eleven
figures is interpreted here as eleven panels. None is a conceptual schematic.
Figures 2.1–2.2 evaluate the analytical numerical-dispersion relation;
figures 2.3–2.7 demonstrate the scalar wave equation.

Figures 2.3–2.7 evolve the magnetic component Hᵧ through GMES's public
Maxwell solver. With constant permittivity and a transversely uniform field,
eliminating the companion Eₓ component gives the chapter's scalar recurrence.
The separate scalar implementation serves as an independent comparison oracle;
it does not generate the time-stepped figure data. Figures 2.1–2.2 remain
analytical evaluations of that recurrence's dispersion relation.

The deliberately unstable cases use material regions with local S>1. The
constructor's Courant check uses the default medium, so accepted construction
does not establish stability in faster material regions. These examples keep
the existing check intact and deliberately demonstrate that local instability.

## Source-checked targets

Here S=cΔt/Δx and Nλ=λ₀/Δx. A Gaussian width means the full width between its
1/e amplitude points, not its standard deviation or full width at half maximum.

| Figure | Printed / PDF page | Parameters and numerical reference | Required capability |
| --- | --- | --- | --- |
| 2.1 | 35 / 64 | S=0.5; Nλ=1…10. Normalized phase velocity and attenuation per cell. At Nλ=3, vₚ/c=2/3 and αΔx=0; at Nλ=1, vₚ/c=2 and αΔx≈2.639. | Analytical dispersion relation; two vertical axes. |
| 2.2 | 35 / 64 | S=0.5; Nλ=3…80. Percent phase-velocity error on a logarithmic axis; asymptotic error scales as Nλ⁻². Text examples give vₚ/c≈0.9873 at Nλ=10 and ≈0.99689 at Nλ=20. | Analytical dispersion relation. |
| 2.3a,b | 36 / 65 | Unit rectangular pulse, spatial width 40 cells. Panel (a): S=1 and 0.99; panel (b): S=1 and 0.5. Snapshots share an absolute physical time. Display i=0…200; exact pulse edges are approximately 120 and 160. Observe ringing and a leading precursor for S<1. | Scalar time stepping and pulse excitation. |
| 2.4a,b | 37 / 66 | Unit Gaussian, spatial width 40Δx. Same S pairs, common physical time, and display range as figure 2.3. Smooth pulses show substantially less distortion. | Scalar time stepping and Gaussian excitation. |
| 2.5 | 39 / 68; setup p. 38 | Unit Gaussian, spatial width 40Δx. Uniform Δx and Δt; S=1 at i=1…139 and S=0.25 at i=140…200. Reflection coefficient −0.6 and transmission coefficient 0.4; the text reports each within 0.5%. Reflected width 40Δx; transmitted width 10Δx. | Spatially varying scalar update coefficient. |
| 2.6a,b | 44 / 73; setup p. 43 | S=1.0005 everywhere. Gaussian temporal width 40Δt, unit peak at n=60. Snapshots n=200, 210, 220. Panel (a): i=1…220; panel (b): i=1…20. Alternating two-cell noise; reported growth 1.75…2.0 per ten steps, compared with theoretical 1.8822. | Intentionally unstable scalar time stepping. |
| 2.7a,b | 46 / 75; setup p. 45 | S=1 except S=1.075 at i=90. Gaussian temporal width 10Δt, unit peak at n=60. Snapshots n=190, 200. Panel (a): i=1…160; panel (b): i=70…110. Alternating noise originates near i=90 and propagates in both directions. Reported growth ≈15 per ten steps; uniform-grid theory would predict ≈2200. | Local intentional instability; roundoff-sensitive growth. |

## Equations and comparison limits

Equation (2.16) defines the educational scalar recurrence:

```text
u[i,n+1] = S[i]² (u[i+1,n] − 2u[i,n] + u[i−1,n])
           + 2u[i,n] − u[i,n−1].
```

For density N and Courant factor S, let ξ=1−2sin²(πS/N)/S².
On the real branch, kΔx=acos(ξ), vₚ/c=2π/[N acos(ξ)], and attenuation
is zero. For ξ<−1, the real part of kΔx is π, vₚ/c=2/N, and attenuation
per cell is acosh(−ξ). The transition density is πS/asin(S); the temporal
Nyquist minimum is N=2S. These expressions use the principal branch within
that sampling domain.

Equation (2.49) gives the uniform unstable highest-frequency amplification
q=(S+√(S²−1))² per step. At S=1.0005, q¹⁰≈1.8822. This prediction concerns
an individual Fourier mode; it is not an exact amplification factor for every
sample of a transient noisy pulse.

The book does not specify the complete source injection, initialization,
boundary algorithm, floating-point precision, rectangular launch timing, or
exact snapshot times for figures 2.3–2.5. Those choices must be stated when
comparing a reproduction. The figures provide no sampled reference arrays,
so image similarity alone cannot establish a quantitative waveform tolerance.

The footnote on printed page 45 explicitly warns that figure 2.7's instability
growth depends on word length, arithmetic operations, and roundoff. It says
replication may require different S values. The example retains S=1.075;
its noise amplitude must not be presented as a portable reproduction of the
book's precise curve or its reported growth factor.

## Implementation conventions

Both GMES and the scalar oracle use CPU float64, one CPU thread, and eager
execution, restoring the previous thread count afterward. The GMES domain has
2,048 z cells with Δz=1 and collapsed transverse dimensions whose cell spacing
is 100. Permittivity is one everywhere. A finite 1,800-cell block lies in a
slower default medium with μ=4. Within the block, μᵢ=(Δt/Sᵢ)² gives the desired
local Courant coefficient. Figures 2.3–2.4 use Δt=S and μ=1; figure 2.5 uses
Δt=1 with μ=1 before i=140 and μ=16 afterward. Figures 2.6–2.7 use Δt=1 and
μᵢ=1/Sᵢ². This is a magnetic-component representation of the scalar equation,
not electric-field reflection at a conventional nonmagnetic dielectric.

The plotted coordinate i=0 is magnetic index 768, at z=−256.5. Material faces
lie halfway between magnetic samples, assigning the requested coefficient
exactly at i=140 for the interface and only i=90 for the local defect. All
fields initially vanish. After each complete public `step()`, the example
sets the prescribed Hᵧ source on both transverse planes using `host_snapshot()`
and `load_host_fields()`. GMES updates E before H, so the next electric update
uses that corrected magnetic boundary value. All interior evolution uses the
GMES update kernels. The slower exterior and periodic domain edges lie outside
the discrete one-cell-per-step dependency cone for these runs of at most
396 steps. The scalar oracle instead has a distant fixed-zero boundary at
coordinate 512; neither remote boundary affects the selected snapshots.

The Gaussian source is exp[−((t−60)/20)²]; figure 2.7 uses denominator 5.
The rectangular source is one for 40≤t<80 and zero otherwise. Units are
normalized so the reference wave speed and cell width are one. Figures 2.3
and 2.4 use t=198, corresponding to exactly 198, 200, and 396 steps for
S=1, 0.99, and 0.5. This preserves the book's common-time comparison;
the chosen pulse position is two cells earlier than the approximate position
in the book's plot. Figure 2.5 uses step 240. Figures 2.6 and 2.7 use the
source-specified step numbers and widths in time-step units.

## Observed discrepancies

The GMES CPU float64 run gives figure 2.5 a reflected sample minimum of
−0.6030700, a 0.512% error relative to −0.6. This narrowly misses the book's
reported error below 0.5%; that quantitative target remains unmet. The
transmitted sampled maximum is about 0.396995 (0.751% error relative to 0.4).
A three-point parabolic estimate of the subcell peak gives approximately
0.400448 (0.112% error). Sampled maxima and interpolated peaks are distinct
observables; interpolation does not make the sampled-maximum target pass.

For figure 2.7, the GMES run reproduces roughly fifteenfold local growth
between steps 190 and 200, but its central noise magnitude is approximately
2.25×10⁻¹⁰ at step 200, far below the order-one oscillation in the book.
This is a mechanism reproduction with a substantial amplitude discrepancy,
consistent with the source's precision caveat. It is not a quantitative match
to the plotted unstable waveform. A labeled inset shows the actual small
noise on its own vertical scale while retaining the source's main plot limits.

## Running an example

Install the locked environment, including the plotting extra, from the repository root:

```sh
uv sync --locked --extra torch-cpu --extra hdf5 --extra plot
```

Select any numbered figure independently. Omitting `--output` runs its numerical
calculation and prints a summary without generating a plot:

```sh
uv run --no-sync python -m examples.taflove_chapter2 --figure 2.1
uv run --no-sync python -m examples.taflove_chapter2 --figure 2.2
uv run --no-sync python -m examples.taflove_chapter2 --figure 2.3
uv run --no-sync python -m examples.taflove_chapter2 --figure 2.4
uv run --no-sync python -m examples.taflove_chapter2 --figure 2.5
uv run --no-sync python -m examples.taflove_chapter2 --figure 2.6
uv run --no-sync python -m examples.taflove_chapter2 --figure 2.7
```

Generate the selected figure, including both panels where applicable:

```sh
uv run --no-sync python -m examples.taflove_chapter2 \
    --figure 2.3 --output /tmp/taflove-2.3.png
```

Run an additional stable scalar/GMES cross-check:

```sh
uv run --no-sync python -m examples.taflove_chapter2 --check-gmes
```

This additional check covers stable Gaussian propagation at S=0.5 and S=0.99.
The figure regressions separately compare every time-stepped GMES configuration
with the independent scalar oracle, including both unstable experiments.
Generated figures are local artifacts and should not be committed.

## Verification tracking

Every figure is implemented by `figure_data` in `taflove_chapter2.py`, selected
by the corresponding `--figure` value. These statuses distinguish available
calculations from verified outcomes. Single-run numerical calculation times,
excluding imports and plotting, were approximately 0.00024, 0.00012, 0.433,
0.407, 0.144, 0.120, and 0.112 seconds for figures 2.1 through 2.7. These
are local observations, not performance guarantees. Seven sequential plotting
CLI processes took 23.18 seconds in total, including
imports and plotting, with peak RSS 346,548 KiB (about 338 MiB). The peak is a
high-water mark for the measured command and its children, not their sum.
The plotting run overlapped the focused tests, so it is not an isolated
benchmark. These measurements used Linux x86_64, Python 3.14.7, Torch
2.14.0+cpu, CPU float64, eager execution, and one requested CPU thread;
they do not establish CUDA or macOS results. All seven final PNG files
(all eleven panels) were independently inspected.

Tests below are in `tests/test_taflove_chapter2.py`. Passing a regression with
an explicitly chosen implementation tolerance does not establish every stricter
or different textbook observable. Test execution is recorded separately from
the source inventory. `test_rectangular_ringing_and_gaussian_low_distortion`
also checks both stable nonunit Courant values: rectangular overshoot above
1.1, undershoot below −0.1, precursor above 0.01, and Gaussian maximum error
below 0.003 against the unit-Courant result. These are qualitative regression
criteria, not digitized textbook error bounds.
`test_highest_frequency_mode_matches_amplification_relation` applies the
recurrence to an alternating mode and checks its signed amplification with
absolute tolerance 10⁻¹⁴.

| Figure | Example / test | Numerical criterion | Status | Runtime / peak memory |
| --- | --- | --- | --- | --- |
| 2.1 | `--figure 2.1`; `test_dispersion_landmarks` | Phase velocity absolute tolerance 5×10⁻⁵ for rounded source values; attenuation 10⁻⁷. | Focused regressions pass. | ≤0.44 s numerical calculation; aggregate process measurement above. |
| 2.2 | `--figure 2.2`; `test_phase_error_converges_quadratically` | Error ratio for doubled density equals four within 1%. | Focused regressions pass. | ≤0.44 s numerical calculation; aggregate process measurement above. |
| 2.3a,b | `--figure 2.3`; `test_unit_courant_translates_source_without_distortion`, `test_comparison_snapshots_share_physical_time` | S=1 translation absolute tolerance 2×10⁻¹⁴; common time absolute tolerance 10⁻¹². | Focused regressions pass. | ≤0.44 s numerical calculation; aggregate process measurement above. |
| 2.4a,b | `--figure 2.4`; same translation and common-time tests | Gaussian S=1 translation absolute tolerance 2×10⁻¹⁴. | Focused regressions pass. | ≤0.44 s numerical calculation; aggregate process measurement above. |
| 2.5 | `--figure 2.5`; `test_interface_reflection_transmission_and_compressed_width` | Reflected peak matches rounded 0.603 within 0.0002; interpolated transmission matches 0.4 within 0.5%; widths 40 and 10 within 0.3 cell. | Regression criteria pass; source sampled-amplitude thresholds remain unmet. | ≤0.44 s numerical calculation; aggregate process measurement above. |
| 2.6a,b | `--figure 2.6`; `test_uniform_instability_growth_and_alternating_noise` | Noise alternates; growth per ten steps 1.75…2.0; modal prediction 1.8822 within 0.0001. | Focused regressions pass. | ≤0.44 s numerical calculation; aggregate process measurement above. |
| 2.7a,b | `--figure 2.7`; `test_local_instability_preserves_main_packet_and_grows_at_defect` | Packet peak at i=140 within 0.005 of unity; local alternating noise grows. | Mechanism regression passes; source plotted noise amplitude not reproduced. | ≤0.44 s numerical calculation; aggregate process measurement above. |

`test_gmes_figure_fields_match_independent_scalar_oracle` compares all 513
returned field samples at every requested snapshot for figures 2.3–2.7.
Its absolute bound is 10⁻¹¹ except for the roundoff-sensitive local instability,
where it is 10⁻⁸. Observed maximum differences for figures 2.3 through 2.7
were 9.99×10⁻¹⁵, 4.89×10⁻¹⁵, 3.22×10⁻¹⁵, 6.40×10⁻¹³, and 1.06×10⁻⁹.
The final bound does not establish agreement of the tiny local-noise waveform;
separate assertions check its alternating, localized growth.

`test_supported_gmes_matches_scalar_recurrence` checks maximum absolute field
error below 10⁻¹² at S=0.5 and S=0.99; the independent CLI check measured
2.22×10⁻¹⁵ and 1.33×10⁻¹⁵. Independent verification passed all 22 final
focused cases in 6.13 seconds
(8.61 seconds process wall time; peak RSS 397,652 KiB). This is a focused-suite
result, not a full repository-suite pass. Run the focused regressions with:

```sh
uv run --no-sync python -m pytest -q tests/test_taflove_chapter2.py
```
