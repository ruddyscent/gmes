"""Deliberately invalid calls used to verify negative type-check coverage."""

import gmes

gmes.TorchRuntimeConfig(device=1)
gmes.TorchProbeSpec(component="Ex", location=(0.0, 0.0))
gmes.TorchPointSourceRecord(
    component="Ex",
    target=(0, 0),
    source_time=gmes.Continuous(1.0),
)
