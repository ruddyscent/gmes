# Example verification

Verification was performed on an Apple silicon MacBook with Python 3.14.6. Interactive plots were exercised with Matplotlib's non-interactive `Agg` backend so runs could complete unattended. Generated simulation outputs were written to temporary directories outside the repository.

| Example | Verification | Result | Runtime / peak RSS | Notes |
| --- | --- | --- | --- | --- |
| `air2d.py` | Full run to `t=10` | Pass | 3.20 s / 143 MB | Produced a finite `(200, 200, 1)` Ez field with peak magnitude `0.999382775451675`. |
| `fresnel_reflection.py` | Full run to `t=200` | Pass | 21.67 s / 88 MB | Completed 28,570 timesteps. Six reflection/transmission probe files each contained 28,570 finite samples. |
| `man.py` | Reduced construction run (`--quick`) | Pass (reduced) | 2.17 s / 130 MB | Mapped all six field components for the eight-object 3D geometry and rendered three permittivity cuts. The historical full-resolution run was not attempted. |
| `metal_array.py` | Reduced run to `t=1` (`--quick`) | Pass (reduced) | 3.52 s / 127 MB | Completed 28 timesteps with six DCP silver spheres, CPML, a `Jy` source, and an Ey visualization. The 1.1 GB full run was not attempted. |
| `phc_slab.py` | Reduced run to `t=1` (`--quick`) | Pass (reduced) | 2.34 s / 129 MB | Completed 9 timesteps for a reduced 3D silicon-on-insulator lattice with a line defect, CPML, an `Hz` source, and three visualizations. The 1.3 GB full run was not attempted. |
| `phc_waveguide.py` | Full run to `t=200` | Pass | 9.93 s / 139 MB | Completed 5,714 timesteps. The `(321, 161, 1)` Ez field was finite with peak magnitude `0.9989081615432893`. |
| `slab_waveguide.py` | Full run to `t=200` | Pass | 2.23 s / 130 MB | Completed 2,857 timesteps. The `(161, 81, 1)` Ez field was finite with peak magnitude `0.9994622335025213`. |
| `tfsf.py` | Full run to `t=200` | Pass | 4.50 s / 125 MB | Completed 5,714 timesteps. Ez, Hx, and Hy were finite and nonzero after fixing CPML grading at rounded outer boundaries. |
| `tfsf_with_scatterer.py` | Full run to `t=200` | Pass | 4.96 s / 122 MB | Completed 5,714 timesteps with the dielectric cylinder. Ez, Hx, and Hy were finite and nonzero, with peak magnitudes `2.0727`, `1.9535`, and `1.9615`. |
| `ziolkowski1995_sit.py` | Full Figs. 1-4 | Pass | 270.5 s total | Used 20,000 cells. The pi, 2-pi, and 4-pi cases reached the expected inversion cycles; maximum `rho3` ranged from `0.9954` to `0.9985`, with minimum Bloch norm at least `0.9955`. |
| `ziolkowski1995_ultrafast.py` | Full Figs. 5-9 | Pass | 8.2 s total | Used the paper cell counts. Figs. 5 and 9 reached essentially complete inversion and de-excitation, and the lossless Bloch norm remained near one. |
| `ziolkowski1995_gain.py` | Full Figs. 10-11 | Partial | 56.3 s | Completed 150,000 steps over 2,000 cells. The late normalized intensity and final field reproduce the paper's approximately `1.223` and `1.106` values, but the reported early `1.483` intensity peak was not recovered. |
| `ziolkowski1995_pump_probe.py` | Full Fig. 12 | Pass | 150.2 s for both delays | At `lambda0/400`, a three-period Hann intensity envelope recovered the smooth pump free-induction decay and an output peak of `1.5185`. Late intensities are `1.2093` and `1.2086`, with residual inversion near `0.938`. The reported late targets are `1.2073` and `0.9317`. |

High-cost examples may receive construction or reduced-size checks instead of full simulation runs. Such cases are explicitly identified rather than reported as full passes.

## Ziolkowski 1995 reproduction notes

All figure numbers and equation numbers in this section refer to
R. W. Ziolkowski, J. M. Arnold, and D. M. Gogny, "Ultrafast pulse
interactions with two-level atoms," Phys. Rev. A 52, 3082-3094 (1995),
https://doi.org/10.1103/PhysRevA.52.3082.

### Dm2 implementation citation audit

The two references recorded in the [pre-cutover Dm2 header](https://github.com/ruddyscent/gmes/blob/66a0a1aa8d6f163134967e8b8a7e9dc46530717b/src/pw_dm2.hh) are appropriate, but they
support different parts of the implementation. The Ziolkowski paper is the
primary source for the homogeneous two-level Maxwell-Bloch equations,
exponential variables, and predictor-corrector updates in Eqs. (11)-(12) and
Appendix Eqs. (A1)-(A4). F. Schlottau, M. Piket-May, and K. Wagner,
"Modeling of femtosecond pulse interaction with inhomogeneously broadened
media using an iterative predictor corrector FDTD method," Opt. Express 13,
182-194 (2005), https://doi.org/10.1364/OPEX.13.000182, is the appropriate
source for extending that scheme to discrete `omega[j]` and `N_atom[j]`
transition bins. It is supporting, rather than primary, evidence for the
single-transition homogeneous core. The header now states these roles and no
longer claims that the implementation is limited to one-dimensional Ex/Hy;
the current code supplies Ex, Ey, and Ez electric-update classes.

### Appendix Eq. (A3e)

Appendix Eq. (A3e) prints `D(t) = rho30 / T1 exp(t / T1)`. Direct
substitution of Eqs. (A1) into Bloch Eq. (12b) instead gives the
additive coefficient

```text
D(t) = 2 gamma rho30 / hbar exp(t / T2),
```

which is the expression implemented by `Dm2`. A test-only A/B build
with the extra dimensionless `1/T1` factor removed the Fig. 10 gain
(approximately `1.22` became `1.00`), so the printed form is not a
viable reproduction fix. Focused tests also verify that the initial
Bloch drive is independent of `T1` and that the lossless Bloch-sphere
norm is conserved.

### Fig. 10 smooth turn-on

Equation (29) describes a resonant sinusoid multiplied during the first five
periods by `(1 - x^2)^4`, followed by a continuous-wave branch, but its
printed definition of `x` is inconsistent with that five-period interval. An
earlier implementation used the full symmetric window
`x = 2t / (5 Tp) - 1`. That envelope rises from zero, returns to zero at
`5 Tp`, and then jumps to one, creating artificial preceding peaks in both
the input and output envelopes.

A high-resolution inspection of the published Fig. 10 shows no such peaks.
The dotted `I_left` curve rises monotonically from zero to one and remains
there; its steep leading edge is not a separate peak. The source now uses the
monotone half-window `x = t / (5 Tp) - 1`, which rises smoothly from zero to
one and joins the continuous-wave branch with zero slope. This interpretation
is consistent with both the paper's phrase "smooth turn-on over 5 periods"
and the plotted `I_left` trace.

### Fig. 10 gain transient

The published Fig. 10 reports an early output-intensity gain of `1.483` that
falls near `0.55 ps` to `1.223`. The GMES result approaches `1.223` directly
and does not contain the decrease. The discrepancy is associated with a chain
of factor-of-two inconsistencies in the paper's small-signal derivation, not
with the turn-on envelope.

Let

```text
G = N_atom gamma^2 omega0 T2 / (hbar epsilon0 c).
```

For the reported parameters, `G = 0.02245 micrometer^-1`. Direct solution of
Bloch Eqs. (12a)-(12b) at resonance gives the polarization amplitude
`rho1 = i gamma E T2 rho30 / hbar`. Combining this with Maxwell Eq. (11b) and
expanding the resulting square root gives the electric-field gain coefficient
`G/2`; the normalized intensity after a length `L` is therefore
`exp(G L) = 1.224`.

In contrast, Eq. (17b) retains only `1/T2` in the second-order damping term;
direct differentiation of Eqs. (12a)-(12b) gives `2/T2`. Consequently,
Eq. (18b) doubles the resonant polarization, and Eq. (19) carries twice the
susceptibility obtained from the original Bloch equations. Equation (20) has
an additional notational inconsistency: its printed coefficient contains
`2 N_atom`, while the quoted numerical value `g = 0.0225 micrometer^-1`
corresponds to `G`, not `2G`. The paper then uses `exp(2 g L) = 1.498` for the
early intensity and describes a later halving of the coefficient, which gives
`exp(g L) = 1.224`. The latter is the value obtained directly by GMES from the
stated Maxwell-Bloch equations.

A controlled run with only `N_atom` doubled from `1.0e24` to `2.0e24 m^-3`
produced a peak intensity of `1.49720` and a `0.70-0.84 ps` median of
`1.49726`, with no late decrease. This confirms that the missing peak is a
coupling-factor issue and that the reported `1.483 -> 1.223` transition does
not follow from the stated constant-parameter Maxwell-Bloch system. The
published information is insufficient to determine whether the transition
came from the original implementation or an undocumented envelope-processing
choice. The test-only Appendix `1/T1` variant suppresses the gain and does not
recover this transient.

### Fig. 12 probe turn-on

The weak probe in Fig. 12 is the delayed version of the same resonant source
defined by Eq. (29), so it must use the corrected monotone half-window from
the Fig. 10 audit as well. Full-resolution reruns at delays of $20T_p$ and
$40T_p$ show monotone input turn-ons. Their late normalized output intensities
are `1.209263` and `1.208560`, respectively, and the final inversion $\rho_3$
ranges are `0.937946-0.938031` and `0.938006-0.938106`. These remain close to
the paper's reported `1.2073` intensity and `0.9317` inversion; the turn-on
correction does not require a change to the Dm2 material parameters.

The small structure near 120 fs on the reconstructed $I_{\mathrm{right},40}$
curve cannot be caused by the $20T_p$ input: the $20T_p$ and $40T_p$ traces
come from separate simulations. A full-resolution diagnostic found that the
two output histories agree to within `1.01e-4` from 100 to 140 fs, before the
$20T_p$ probe can reach the output plane. The structure is in the
pump-induced free-induction decay and remained visible with a one-period
boxcar average. Direct two- and three-period boxcar averages retained smaller
carrier ripples. A three-period Hann-weighted RMS removed the local extrema,
reduced the $I_{\mathrm{right},20}$ peak from `1.536595` to `1.518470` (close
to the approximately `1.52` shown in Fig. 12), and left the late intensities
at `1.209263` and `1.208560`. The input 10%-90% rise interval changed only
from `121.69-132.06 fs` to `121.37-133.25 fs`, and the output peak shifted by
`1.44 fs`.

The paper identifies the traces only as intensity envelopes and does not
state its extraction or smoothing algorithm. The three-period Hann window is
therefore a documented reproduction choice, not a claim about the original
detector. Its agreement with the smoother published curve shows that the
short transient does not require a change to Dm2.

The tiny mark near the start of the published $I_{\mathrm{left},20}$ trace is
also not sufficient evidence for a physical prepeak. That trace is printed
with a long-dashed line, and its first visible dash lies on the low-amplitude
part of the monotone turn-on. The corrected run rises continuously from
`0.0595` at 120 fs to `0.9587` at 135 fs. By contrast, the literal symmetric
interpretation of the printed Eq. (29) creates an order-one lobe and returns
to zero before jumping to the continuous-wave branch, which is qualitatively
larger than the small published mark. The mark could additionally contain a
small residual pump field at the input plane, but the paper provides neither
the sampled data nor the envelope algorithm needed to separate that effect
from dashed-line rendering. It therefore does not justify restoring the
discontinuous source.

### Other reproduction notes

For Figs. 5-8, the caption says `t=12.5 fs`, but a source at `z=0`
cannot reach the plotted `6-7.5 micrometer` interval by that time.
The examples use a documented 12.5 fs pre-roll and preserve the
paper's displayed time label. This produces the reported spatial
profiles without changing material or pulse parameters.

Intensity envelopes use local averages of `2 E^2`: a one-period boxcar for
Figs. 10-11 and a three-period Hann window for Fig. 12. A whole-record Hilbert
transform is unsuitable for Fig. 12 because its nonlocal edge ringing leaks
the pump, which is 10,000 times larger in amplitude, into the later probe
interval.
