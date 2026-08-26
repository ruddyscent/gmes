# Torch-native execution

`TorchSimulation` is GMES's intentionally breaking execution path. PyTorch is
the destination, not one selectable backend, so this API has no
`backend="cpp"` switch and does not expose NumPy fields or the legacy stepping
surface. The native classes remain temporarily available only while the frozen
oracle is needed by the migration.

The executable slice supports serial periodic 1-D, 2-D, and 3-D Cartesian
domains, simple nondispersive `Dielectric`, `Const`, and `Dummy` geometry,
real-field `Dm2` Maxwell--Bloch media, UPML and CPML absorbing layers, Drude,
Lorentz, and every DCP strategy, and eager or compiled phases. Bloch-periodic
fields support every family except DM2. Sources, distributed domain
decomposition, plotting, and solver-owned I/O remain outside this slice and
fail during construction instead of silently falling back.

## Mixed-material execution planning

Construction lowers every Yee component through an immutable
`ComponentPlan`. Before any tensor is finalized, the planner validates bounds,
complete active-cell coverage, unique destinations, and exact material and
underlying-region IDs. Each component records flattened stencil metadata, a
dense inverse-coefficient plane, compact target arrays, and a bounded
tile-dense candidate as appropriate. Host arrays are built in bounded geometry
tiles; each selected array is converted to one contiguous device tensor.

Execution buckets use the normalized signature `(model, component, real
precision, state shape)`. Geometry object identity and coefficient values are
not part of that signature: equivalent regions share one launch bucket and
refer to shared coefficient rows through stable region indirection. Drude,
Lorentz, DCP, and DM2 magnetic cells normalize to the same simple magnetic
signature because their magnetic behavior is physically identical.

`TorchRuntimeConfig.execution_policy` accepts `"auto"`, `"dense"`,
`"compact"`, or `"tiled"`. The default `auto` decision uses a fixed
CPU/CUDA cost model over occupancy, active-target fragmentation, state width,
tile coverage, memory, and launch cost; it never inserts material autotuning
into a timestep. Forced policies exist for differential benchmarks and
debugging, and do not silently replace an unsupported state equation.
`simulation.diagnostics()["material_plan"]` explains every decision and its
candidate costs.

UPML and CPML use active-cell-only contiguous tensor buckets. Coordinate-
dependent coefficients and four flattened gather indices are finalized once;
the mutable UPML `b/d` or CPML `psi1/psi2` state and three fixed scratch
arrays stay on the requested device. The base permittivity/permeability comes
from each shell cell's lowered underlying-region ID, including corners and
overlap. Buckets are keyed by model, component, precision, and state width, so
geometry object count does not create per-region launches.

`simulation.state.pml_state_snapshot()` is an explicit oracle/debug adapter.
`simulation.diagnostics()["pml"]` reports active component cells, mutable
state bytes, and launches per step. Normal stepping uses gather/update/scatter
over unique targets and performs no host conversion or native fallback.

## DM2 Maxwell--Bloch execution

Electric DM2 cells are bucketed by component, real precision, and exact
transition count. Each bucket expands immutable coefficients once into
contiguous structure-of-arrays tensors and allocates transformed Bloch state
only for its active targets. Mutable state and predictor--corrector scratch
remain device-resident at fixed addresses; magnetic DM2 behavior shares the
ordinary dielectric path.

The corrector uses a device bool mask and a fixed schedule of ten compiled
chunks with ten iterations each. Converged targets retain their field and
state while unconverged targets continue, with the native 100-iteration limit,
zero-reference relative error, NaN handling, and tolerance semantics
preserved. There is no `.item()` or host decision inside this schedule. After
all DM2 buckets finish, one compact status tensor crosses the diagnostic
boundary; failures identify the component, transition width, and flattened
targets without overwriting failed state. Successful runs expose per-bucket
iteration distributions through `simulation.diagnostics()["dm2"]`.

DM2 supports real fields only. Construction rejects a DM2 geometry combined
with a Bloch vector instead of silently changing the field representation.
Explicit state I/O is cell-major:

```python
state = simulation.dm2_state_snapshot()
# Each record contains targets, transformed u/v/w, physical rho, and time.

simulation.load_host_dm2_state(
    [record["u"] for record in state],
    step_count=simulation.state.step_count.cpu().item(),
)
```
PML cells may resolve a dispersive underlying region's base permittivity or
permeability while non-PML cells retain their exact-width dispersive state.
Target ownership keeps the two update families disjoint.

## Dispersive state buckets

Drude, Lorentz, DCP ADE, DCP PLRC, and DCP RC electric updates execute from
the same component plans. Buckets are exact-width by
`(model, component, real precision, pole count, critical-point count)`; the
runtime does not merge widths or pad to a grid-wide maximum. Planner
diagnostics report the actual scalar state width, while the benchmark reports
`state_width_policy="exact"`, zero padded elements, device state bytes, and
the elements that a same-family bounded-padding alternative would add. The
decision record compares exact signature launches/state elements with a
single-launch bounded max-width merge for later tuning.

Each bucket finalizes recurrence coefficients once as contiguous
structure-of-arrays tensors. The planner retains shared coefficient rows for
memory and diagnostics; execution expands those rows once per active target to
avoid a coefficient-indirection gather on every step. Mutable pole and
critical-point state is allocated only for the bucket's active cells and stays
registered in `TorchSimulationState`, so checkpoints include it and normal
advancement performs no host transfer or per-cell allocation.

Bloch fields and DCP complex accumulators use paired-real tensors. Complex
multiplication is written explicitly over the final real/imaginary plane, so
the recurrences do not depend on native complex Inductor support. Dispersive
magnetic cells continue through the shared simple Dielectric path. Mixed PML
geometry consumes lowered underlying-region IDs and can coexist with every
dispersive family without a native fallback.

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
    execution_policy="auto",  # or force dense/compact/tiled for measurements
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
checkpoint = simulation.state.checkpoint()     # fields, clocks, and PML state
host_fields = simulation.state.host_snapshot() # cloned NumPy arrays by default
pml_state = simulation.state.pml_state_snapshot() # active PML cells only

simulation.state.load_checkpoint(checkpoint)   # in-place; buffer addresses stay fixed
```

Nondispersive advancement does not call `.cpu()`, `.numpy()`, or `.item()`.
DM2 performs only the single post-electric status transfer described above;
its predictor--corrector itself has no host synchronization. Use
`buffer_addresses()` only as an explicit diagnostic when checking fixed
storage for CUDA graph capture.
