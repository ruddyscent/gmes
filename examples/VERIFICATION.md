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
| `ziolkowski1995_pump_probe.py` | Full Fig. 12 | Pass | 147.1 s for both delays | At `lambda0/400`, recovered the 20- and 40-period probe turn-ons, free-induction decay, late intensities `1.2093` and `1.2086`, and residual inversion near `0.938`. The reported targets are `1.2073` and `0.9317`. |

High-cost examples may receive construction or reduced-size checks instead of full simulation runs. Such cases are explicitly identified rather than reported as full passes.

## Ziolkowski 1995 reproduction notes

All figure numbers and equation numbers in this section refer to
R. W. Ziolkowski, J. M. Arnold, and D. M. Gogny, "Ultrafast pulse
interactions with two-level atoms," Phys. Rev. A 52, 3082-3094 (1995),
https://doi.org/10.1103/PhysRevA.52.3082.

### Dm2 implementation citation audit

The two references at the start of `src/pw_dm2.hh` are appropriate, but they
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

### Fig. 10 turn-on precursor

Equation (29) describes a resonant sinusoid multiplied during the first five
periods by `(1 - x^2)^4`, followed by a continuous-wave branch. Interpreting
the five-period interval as `x = 2t / (5 Tp) - 1` makes this envelope rise
from zero, return to zero at `5 Tp`, and then jump to one. It therefore
produces a separate turn-on lobe before the main continuous-wave envelope.
The narrow onset feature is also visible in the published Fig. 10, although
the one-carrier-period average of `2 E^2` used here makes it more prominent.
The equation and its description as a "smooth turn-on" are internally
ambiguous; replacing it with a monotone ramp would remove the precursor but
would no longer be a literal reproduction of Eq. (29).

### Fig. 10 gain transient

The published Fig. 10 reports an early output-intensity gain of `1.483` that
falls near `0.55 ps` to `1.223`. The GMES result approaches `1.223` directly
and does not contain the decrease. This difference follows an internal
factor-of-two inconsistency between Eqs. (19) and (20). Equation (19) has the
form `k^2 = (omega0/c)^2 (1 - i delta)`, so its small-signal square root is
`k = (omega0/c) (1 - i delta/2)`. Equation (20) omits this factor of one-half
when defining the gain coefficient. The correctly expanded coefficient gives
the normalized intensity `exp(g L) = 1.224`, consistent with the GMES value;
the doubled coefficient used by Eq. (20) gives `exp(2 g L) = 1.498`.

A controlled run with only `N_atom` doubled from `1.0e24` to `2.0e24 m^-3`
produced a peak intensity of `1.49720` and a `0.70-0.84 ps` median of
`1.49726`, with no late decrease. This confirms that the missing peak is a
coupling-factor issue and that the reported `1.483 -> 1.223` transition does
not follow from the stated constant-parameter Maxwell-Bloch system. The
published information is insufficient to determine whether the transition
came from the original implementation or an undocumented envelope-processing
choice. The test-only Appendix `1/T1` variant suppresses the gain and does not
recover this transient.

### Other reproduction notes

For Figs. 5-8, the caption says `t=12.5 fs`, but a source at `z=0`
cannot reach the plotted `6-7.5 micrometer` interval by that time.
The examples use a documented 12.5 fs pre-roll and preserve the
paper's displayed time label. This produces the reported spatial
profiles without changing material or pulse parameters.

Intensity envelopes in Figs. 10 and 12 use a one-carrier-period local
average of `2 E^2`. A whole-record Hilbert transform is unsuitable for
Fig. 12 because its nonlocal edge ringing leaks the pump, which is
10,000 times larger in amplitude, into the later probe interval.
