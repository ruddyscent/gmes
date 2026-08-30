"""Checker-only assertions for representative supported public types."""

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import assert_type

import numpy as np
import torch

import gmes

runtime = gmes.TorchRuntimeConfig(device="cpu", cpu_threads=1)
assert_type(runtime.dtype, torch.dtype)
assert_type(runtime.validate_static(), None)

launch = gmes.DistributedLaunch()
assert_type(launch.validate(), None)

probe = gmes.TorchProbeSpec(component=gmes.Ex, location=(0.0, 0.0, 0.0))
assert_type(probe.component, str | type[object])
samples = gmes.TorchProbeSamples(
    component="Ex",
    index=(0, 0, 0),
    times=np.array((0.0, 1.0), dtype=np.float64),
    values=np.array((1.0, 0.0), dtype=np.float64),
    dropped=0,
    total=2,
)
assert_type(gmes.probe_spectrum(samples), gmes.TorchProbeSpectrum)
assert_type(gmes.write_probe_text((samples,), "."), tuple[Path, ...])


class CustomSource:
    """Example structural Torch source extension."""

    def lower_torch_source(
        self, context: gmes.TorchSourceLoweringContext
    ) -> Iterable[gmes.TorchPointSourceRecord]:
        del context
        yield gmes.TorchPointSourceRecord(
            component="Ex",
            target=(0, 0, 0),
            source_time=gmes.Continuous(1.0),
        )


callback: Callable[
    [gmes.TorchSourceLoweringContext], Iterable[gmes.TorchPointSourceRecord]
] = CustomSource().lower_torch_source
assert_type(
    callback,
    Callable[[gmes.TorchSourceLoweringContext], Iterable[gmes.TorchPointSourceRecord]],
)
