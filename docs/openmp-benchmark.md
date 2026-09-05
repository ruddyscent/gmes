# OpenMP field-update benchmark (historical pre-cutover evidence)

This is a preserved record of the former native implementation, not an active
GMES installation, tuning, or benchmark procedure. Reproduce it only from the
recorded historical native reference/archive associated with the measurement,
not from the current Torch-only checkout. The timings and checksums below are
retained for historical differential context.

The reference benchmark ran on an Apple M1 Pro with 10 CPU cores using Python
3.14.2, Apple Clang 21, and Homebrew libomp 22.1.8. Each main result is the
median seconds per complete FDTD step after five warm-up steps, measured over
three runs of 100 steps. Visualization and simulation output were disabled.

The serial reference used the OpenMP-linked extension with a threshold of one
billion cells, so no eligible loop entered a parallel region. The other rows
forced all eligible loops through OpenMP with a threshold of zero.

| Threads | 2D dielectric | 3D dielectric | 2D Drude dispersive |
| ---: | ---: | ---: | ---: |
| Serial reference | 0.000753 | 0.006473 | 0.002574 |
| 2 | 0.000624 | 0.003689 | 0.001566 |
| 4 | 0.000558 | 0.002292 | 0.001466 |
| 8 | 0.000626 | 0.001955 | 0.001811 |

Four threads provided the best balanced result: 1.35 times faster for the 2D
case, 2.82 times faster for 3D, and 1.76 times faster for the dispersive case.
Eight threads improved the 3D result further but lost performance on the
smaller and more memory-intensive cases.

## Threshold selection

The benchmark records every native updater's cell count. The larger cases had
eligible loops from 14,560 to 99,770 cells. With four threads, an 8,192-cell
threshold completed the 2D, 3D, and dispersive cases in 0.000560, 0.002275,
and 0.001232 seconds per step, respectively. A small control case used
1,600-cell CPML loops: forcing those loops through OpenMP took 0.000086 seconds
per step, while retaining the serial path took about 0.000060 seconds. The
parallel startup cost made the small case roughly 43 percent slower.

The default threshold is therefore 8,192 cells. It lies between the measured
unprofitable 1,600-cell loops and the profitable 14,560-cell loops, preserving
the serial path for small work while parallelizing every substantial loop in
the representative cases. It is a conservative crossover, not a hardware
guarantee; users should rerun the benchmark when tuning another system.

All tested thread counts and thresholds produced identical checksums for each
workload. This confirms numerical equivalence for the benchmark cases; the
unit suite provides broader material-model regression coverage in default,
forced-parallel, and OpenMP-disabled configurations.

The former command recipe and `GMES_OPENMP_THRESHOLD` controls apply only to
that recorded native reference environment; they are not supported by the
current package.
