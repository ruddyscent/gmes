# Torch-native execution

`TorchSimulation` is GMES's intentionally breaking execution path. PyTorch is
the destination, not one selectable backend, so this API has no
`backend="cpp"` switch and does not expose NumPy fields or the legacy stepping
surface. The native classes remain temporarily available only while the frozen
oracle is needed by the migration.

The first slice supports serial periodic 1-D, 2-D, and 3-D Cartesian domains,
simple nondispersive `Dielectric` geometry, eager or compiled electric and
magnetic phases, and real or Bloch-periodic fields. Sources, PML, dispersive
materials, distributed domain decomposition, plotting, and solver-owned I/O
remain outside this slice and fail during construction instead of silently
falling back.

## Install a wheel variant

GMES pins the stable PyTorch 2.13 line as `torch>=2.13,<2.14`. The universal
package requirement does not contain a local CUDA suffix. The uv lock records
the CPU, CUDA 12.6, and CUDA 13.0 variants as mutually exclusive deployment
choices using PyTorch's explicit indexes:

```sh
# Linux and macOS CPU CI/development
uv sync --locked --extra torch-cpu --extra hdf5

# Linux target with the CUDA 12.6 runtime
uv sync --locked --extra torch-cu126 --extra hdf5

# Linux target with the CUDA 13.0 runtime
uv sync --locked --extra torch-cu130 --extra hdf5
```

Do not enable more than one `torch-*` extra. Plain package installation keeps
the standard platform selection: PyPI supplies CPU wheels on macOS and the
CUDA 13.0 wheel on Linux. CI explicitly requests `torch-cpu` so GPU runtime
packages are not mistaken for GPU test coverage.

The 2.13.0 probe used Python 3.14.7. CPU plus CUDA 12.6 and CUDA 13.0 eager/compiled tests ran on Linux with
NVIDIA compute capabilities 8.6 and 7.5. The required macOS CPU
job verifies the arm64 wheel and CPU execution. See the
[PyTorch release matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md)
and [uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/)
for the upstream compatibility and index contracts.

## Construct and advance

Runtime choices are explicit and validated before fields or compiler state are
created:

```python
from gmes import Cartesian, DefaultMedium, Dielectric
from gmes import TorchRuntimeConfig, TorchSimulation

runtime = TorchRuntimeConfig(
    device="cuda:0",       # use "cpu" explicitly for CPU execution
    precision="float32",   # "float32" or "float64"
    compile_policy="compile",  # "eager" or fullgraph Torch compilation
    cpu_threads=4,
)
simulation = TorchSimulation(
    space=Cartesian(size=(8, 6, 0), resolution=20),
    geometry=[DefaultMedium(Dielectric(eps_inf=1.7))],
    runtime=runtime,
    bloch=(0.07, 0.11, 0.0),
)
simulation.advance(100)
```

A requested CUDA device is never changed to CPU. An unavailable device, invalid
dtype, autograd request, oversubscribed local launch, or unsupported
distributed launch raises `TorchConfigurationError` with an actionable message.
`torch_runtime_diagnostics(runtime)` reports only the requested precision,
PyTorch/CUDA versions, CUDA/NCCL availability, visible device count, and the
requested device's name and capability.

`advance(steps)` is the throughput API; `step()` is only a one-step
convenience. Both run under `torch.inference_mode()`. The timestep, source
time, fields, dielectric coefficients, region IDs, and preallocated curl
scratch are registered non-trainable buffers. The electric and magnetic
functions are compiled independently with `fullgraph=True` and
`dynamic=False`; geometry, boundary communication, snapshots, and I/O stay
outside those graphs.

Bloch fields have a final length-two real plane holding real and imaginary
parts in the requested real dtype. Complex conversion occurs only at an
explicit host boundary:

```python
device_copy = simulation.state.snapshot()      # cloned tensors, same device
checkpoint = simulation.state.checkpoint()     # cloned persistent state
host_fields = simulation.state.host_snapshot() # cloned NumPy arrays by default

simulation.state.load_checkpoint(checkpoint)   # in-place; buffer addresses stay fixed
```

Normal advancement does not call `.cpu()`, `.numpy()`, or `.item()`. Use
`buffer_addresses()` only as an explicit diagnostic when checking fixed
storage for CUDA graph capture.
