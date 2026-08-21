# OpenMP field-update benchmark

The initial benchmark used an Apple M1 Pro with 10 CPU cores, Apple Clang 21,
and Homebrew libomp 22.1.8. Each value is the median seconds per complete FDTD
step after five warm-up steps, measured over three runs of 100 steps. No
visualization or simulation output was enabled.

| Threads | 2D dielectric | 3D dielectric | 2D Drude dispersive |
| ---: | ---: | ---: | ---: |
| 1 | 0.000610 | 0.006013 | 0.002102 |
| 2 | 0.000610 | 0.003632 | 0.001430 |
| 4 | 0.000584 | 0.002875 | 0.001130 |
| 8 | 0.000597 | 0.001852 | 0.001352 |

All thread counts produced identical checksums in each case. The 3D case was
3.25 times faster with eight threads than with one. The Drude case performed
best at four threads, where it was 1.86 times faster; adding threads after that
increased coordination and memory-bandwidth pressure. The smaller 2D case
stayed below the default 32,768-cell threshold and its timing remained broadly
flat.

These results motivated parallelizing the native C++ material-update loops
with static OpenMP scheduling while retaining the existing serial loop below a
configurable threshold. Releasing Python's global interpreter lock around each
individual update call was also tested, but increased the small 2D step time by
about 50%, so that change was not retained.

The values above are a local reference, not a performance guarantee. Run
`benchmarks/field_updates.py` on the target machine to select an appropriate
thread count and, if necessary, tune `GMES_OPENMP_THRESHOLD`.
