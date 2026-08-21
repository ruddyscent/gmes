# Example verification

Verification was performed on an Apple silicon MacBook with Python 3.14.6. Interactive plots were exercised with Matplotlib's non-interactive `Agg` backend so runs could complete unattended. Generated simulation outputs were written to temporary directories outside the repository.

| Example | Verification | Result | Runtime / peak RSS | Notes |
| --- | --- | --- | --- | --- |
| `air2d.py` | Full run to `t=10` | Pass | 3.20 s / 143 MB | Produced a finite `(200, 200, 1)` Ez field with peak magnitude `0.999382775451675`. |
| `fresnel_reflection.py` | Full run to `t=200` | Pass | 21.67 s / 88 MB | Completed 28,570 timesteps. Six reflection/transmission probe files each contained 28,570 finite samples. |
| `man.py` | Reduced construction run (`--quick`) | Pass (reduced) | 2.17 s / 130 MB | Mapped all six field components for the eight-object 3D geometry and rendered three permittivity cuts. The historical full-resolution run was not attempted. |
| `metal_array.py` | Reduced run to `t=1` (`--quick`) | Pass (reduced) | 3.52 s / 127 MB | Completed 28 timesteps with six DCP silver spheres, CPML, a `Jy` source, and an Ey visualization. The 1.1 GB full run was not attempted. |
| `phc_slab.py` | Reduced run to `t=1` (`--quick`) | Pass (reduced) | 2.34 s / 129 MB | Completed 9 timesteps for a reduced 3D silicon-on-insulator lattice with a line defect, CPML, an `Hz` source, and three visualizations. The 1.3 GB full run was not attempted. |

High-cost examples may receive construction or reduced-size checks instead of full simulation runs. Such cases are explicitly identified rather than reported as full passes.
