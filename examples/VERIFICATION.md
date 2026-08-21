# Example verification

Verification was performed on an Apple silicon MacBook with Python 3.14.6. Interactive plots were exercised with Matplotlib's non-interactive `Agg` backend so runs could complete unattended. Generated simulation outputs were written to temporary directories outside the repository.

| Example | Verification | Result | Runtime / peak RSS | Notes |
| --- | --- | --- | --- | --- |
| `air2d.py` | Full run to `t=10` | Pass | 3.20 s / 143 MB | Produced a finite `(200, 200, 1)` Ez field with peak magnitude `0.999382775451675`. |
| `fresnel_reflection.py` | Full run to `t=200` | Pass | 21.67 s / 88 MB | Completed 28,570 timesteps. Six reflection/transmission probe files each contained 28,570 finite samples. |

High-cost examples may receive construction or reduced-size checks instead of full simulation runs. Such cases are explicitly identified rather than reported as full passes.
