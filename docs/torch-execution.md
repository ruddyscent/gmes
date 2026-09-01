# Torch-native execution

`TorchSimulation` is GMES's intentionally breaking execution path. PyTorch is
the destination, not one selectable backend, so this API has no
`backend="cpp"` switch and does not expose NumPy fields or the legacy stepping
surface. The native classes remain temporarily available only while the frozen
oracle is needed by the migration.

The executable slice supports periodic 1-D, 2-D, and 3-D Cartesian
domains, simple nondispersive `Dielectric`, `Const`, and `Dummy` geometry,
real-field `Dm2` Maxwell--Bloch media, UPML and CPML absorbing layers, Drude,
Lorentz, and every DCP strategy, and eager or compiled phases. Bloch-periodic
fields support every family except DM2. Point/current, TFSF, Transparent, and
Gaussian-beam sources are device-resident. Plotting and solver-owned hot-path
I/O remain outside this slice and fail during construction instead of silently
falling back. On Linux, one physical domain may be split across exactly two
local CUDA ranks through `TorchDistributedSimulation` and NCCL.

## Two-GPU spatial decomposition

Start exactly one process per visible GPU. Each process resolves the four
`torchrun` rank variables, binds exclusively to `cuda:LOCAL_RANK`, sets the
current CUDA device, and initializes NCCL with that device before creating a
simulation:

```python
import gmes

launch = gmes.distributed_launch_from_environment()
runtime = gmes.TorchRuntimeConfig(
    device=f"cuda:{launch.local_rank}",
    precision="float32",
    compile_policy="compile",
    cpu_threads=1,
    launch=launch,
)
simulation = gmes.TorchDistributedSimulation(
    space=gmes.Cartesian((128, 96, 96), 1),
    geometry=[gmes.DefaultMedium(gmes.Dielectric(eps_inf=1.7))],
    runtime=runtime,
)
simulation.capture_cuda_graphs()  # optional fixed-storage throughput setup
simulation.advance(100)
global_fields = simulation.global_field_snapshot()  # rank 0 only by default
simulation.close()
```

The decomposition scores every possible Cartesian cut using local material
and mutable-state cost, measured device weights, source crossings,
communication surface, and whether halo planes require strided packing. Odd
and non-divisible sizes use explicit global offsets. Yee ownership is unique:
rank 0 owns the low-side electric interface and rank 1 owns the high-side
magnetic interface. Point sources, TFSF faces, and probes are filtered by the
same ownership maps. A bounded TFSF incident-field auxiliary solver is
replicated on the same rank-local device because it is not the physical main
domain and its global sample coordinates must remain identical.

Each half step packs persistent CUDA send buffers, starts paired NCCL sends and
receives with `batch_isend_irecv`, runs the independent dense update, waits at
the first halo consumer, applies the interface correction, and then runs PML,
dispersive, DM2, source, and probe updates. No halo uses CPU staging or Python
object serialization. `capture_cuda_graphs()` captures only fixed rank-local
compute regions; NCCL, boundary exchange, sources, diagnostics, and I/O stay
outside the graphs.

The default permits a topology-qualified NCCL transport when CUDA peer access
is unavailable, as on a PCIe PHB path. Pass `require_peer_access=True` to make
direct peer access mandatory. Construction validates two visible devices, one
unique local rank per device, NCCL, matching precision, and identical
decomposition metadata. `diagnostics()` records rank binding, device model and
memory, CUDA/PyTorch/NCCL versions, peer access, local shape, and persistent
halo bytes. A distributed checkpoint is rank-local and validates all ranks
collectively before any state is restored.

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
into a timestep. These choices currently affect planner decisions and optional
storage only. Execution normally uses a dense dielectric base plus compact
indexed material updates; local compiled CPU CPML additionally selects the
dense-base, axis-sparse residual representation described below. Forced-policy
timings are therefore exploratory and the automatic-policy performance gate
remains fail-closed until the policies select distinct executable
representations.
`simulation.diagnostics()["material_plan"]` explains every decision and its
candidate costs.

UPML and eager, CUDA, and distributed CPML use active-cell-only contiguous
tensor buckets. Coordinate-dependent coefficients and four flattened gather
indices are finalized once; the mutable UPML `b/d` or CPML `psi1/psi2` state
and three fixed scratch arrays stay on the requested device. The base
permittivity/permeability comes from each shell cell's lowered underlying-region
ID, including corners and overlap. Buckets are keyed by model, component,
precision, and state width, so geometry object count does not create per-region
launches.

Local compiled CPU CPML applies the ordinary curl in the dense field phase
using that same underlying inverse coefficient. The planner then lowers only
structurally active curl axes, where `c != 0` or `kappa != 1`, into contiguous
target, two-index stencil, and parameter tensors. Each active axis keeps one
one-dimensional `psi` state and two fixed scratch buffers. Its indexed update
is the residual

```text
psi = b * psi + c * derivative
field += dt * direction * inv_base * axis_sign
         * ((1 / kappa - 1) * derivative + psi)
```

This avoids gathering both complete curl directions and materializing both
logical state lanes for every CPML target. Empty axes do not create residual
updates. Before selecting this decomposition, the planner verifies in the
requested dtype that adding the residual coefficient back to one reproduces
the direct reciprocal-kappa coefficient within a small machine-precision
bound. A bucket that would suffer cancellation, such as extreme float32
`kappa`, retains the numerically stable compact full-curl path. UPML and other
non-local-compiled CPML also retain that representation.

`simulation.state.pml_state_snapshot()` is an explicit oracle/debug adapter.
For sparse CPU CPML it reconstructs the canonical target-order `(N, 2)` state,
or `(N, 2, 2)` paired-real state, with zeros in structurally inactive lanes.
`simulation.diagnostics()["pml"]` reports active component cells, physical
mutable-state bytes, active-axis states, the selected representation, and
logical updates per step. The material-planner benchmark derives indexed
gather/scatter traffic from the selected full-curl or two-index axis stencil
instead of applying one fixed per-cell estimate. Normal stepping performs no
host conversion or native fallback.

## DM2 Maxwell--Bloch execution

Electric DM2 cells are bucketed by component, real precision, and exact
transition count. Each bucket expands immutable coefficients once into
contiguous structure-of-arrays tensors and allocates transformed Bloch state
only for its active targets. Mutable state and predictor--corrector scratch
remain device-resident at fixed addresses; magnetic DM2 behavior shares the
ordinary dielectric path.

The corrector uses a device bool mask. Eager execution performs ten chunks of
ten masked iterations; compiled CPU execution packs the carry and performs
three masked iterations per `torch.while_loop` body, amortizing each lowered
condition evaluation. Both paths preserve the exact native maximum of 100
iterations. Converged targets retain their field and state while unconverged
targets continue, with zero-reference relative error, NaN handling, and
tolerance semantics preserved. The solver does not call `.item()` or branch
on convergence in Python. Current CPU Inductor lowers each `while_loop`
condition through a scalar conversion, which the three-iteration body
amortizes.
Device-side asynchronous assertions validate each bucket's status without
transferring a status tensor during successful advancement; failures identify
the component and transition-width bucket without overwriting failed state.
The explicit `simulation.diagnostics()["dm2"]` boundary transfers iteration
counts and reports their per-bucket distributions.

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

## Sources, probes, and checkpoints

Pass legacy built-in sources through the explicit `sources` argument. Point
and current targets are normalized once with the native last-source-wins
overlap rule, then grouped by Yee component. Continuous, bandpass, and
differentiated-Gaussian oscillators execute from tensor parameters at the
native electric and magnetic half-step times. TFSF faces are consolidated by
unique target; their auxiliary Torch solver uses the parent's device,
timestep, and paired-real layout, but retains the native incident solver's
float64 state and arithmetic even for a float32 outer field. Interpolation is
performed in float64 and cast once immediately before the outer field update.
Gaussian modes are lowered during construction, and their envelope derives
from the exact post-prewarm auxiliary integer step and float64 timestep.

A third-party source must implement `lower_torch_source(context)` and return
`TorchPointSourceRecord` values. The lowering hook runs once during
construction. Arbitrary legacy callbacks and `PointSource(filename=...)`
are rejected rather than entering or graph-breaking `advance()`.

`TorchProbeSpec` creates a fixed-capacity device ring. When a producer
outpaces explicit flushing, the ring overwrites its oldest values and reports
the dropped count; it never grows. `flush_probes()`, `host_snapshot()`,
`probe_spectrum()`, and `write_probe_text()` are explicit synchronization/output
boundaries and are not called by the solver phases.

Use the simulation-level versioned checkpoint API for restartable work:

```python
checkpoint = simulation.checkpoint()
simulation.advance(100)
simulation.load_checkpoint(checkpoint)
simulation.save_checkpoint("restart.npz")  # tensor arrays plus JSON metadata
simulation.load_checkpoint_file("restart.npz")

samples = simulation.flush_probes()
write_probe_text(samples, "/tmp/gmes-probes")
```

The in-memory and pickle-free NPZ schemas contain only metadata and tensors:
fields, material state, source
clock, auxiliary solver state, probe rings, and time state. Loading verifies
the schema version, plan identity, device, dtype, and paired-real layout before
copying into the fixed live buffers. Unsupported versions or a different
execution plan fail without partial restoration. The lower-level
`simulation.state.checkpoint()` remains useful for material-state debugging
but intentionally does not include auxiliary solvers or probes.

The compiled CPU sparse CPML layout does not change checkpoint version 1.
Checkpoint state uses the same canonical `pml_<component>_<bucket>_state` keys
and logical shapes as the full-curl representation. Restore scatters active
lanes back into the fixed one-dimensional buffers without replacing their
storage and rejects nonzero values in structurally inactive lanes rather than
silently discarding them. Standard `TorchSimulationState.state_dict()` and
`load_state_dict()` use the same virtual canonical entries, so PyTorch module
serialization does not expose or lose the physical sparse layout.

Periodic/Bloch boundary synchronization remains outside compiled material
kernels and runs immediately before its next static electric or magnetic
compute region. Boundary views are cached and grouped into two ordered stages:
the first periodic axis of each field is updated before its second axis, so
corner values retain the composed Bloch phase while independent components can
use batched `torch._foreach_copy_` and `torch._foreach_mul_` calls. Paired-real
phase scalars live in fixed registered, non-checkpoint boundary storage. Empty
fields and axes with at most one plane are skipped, so a boundary update never
uses aliased source and destination views. The diagnostic compatibility name
`paired_real_scratch_bytes` reports this reserved boundary workspace even though
the cached implementation only uses its first paired-real element for phase
storage. State buffers cannot be replaced with `load_state_dict(assign=True)` or
moved/converted after construction; create a new simulation for another device
or dtype.

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
    compile_mode="default",  # CUDA: default/reduce-overhead/max-autotune
    execution_policy="auto",  # forced values currently inspect planner choices
    cpu_threads=4,
    cpu_interop_threads=1,
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
scratch are registered non-trainable buffers. For a local simulation, the
electric and magnetic compute half-steps are compiled with `fullgraph=True` and
`dynamic=False`, reducing steady execution to two static regions. Boundary
synchronization, distributed communication, geometry, callbacks, snapshots,
and I/O stay outside those graphs; fused local source updates stay inside the
compute half-steps. The compilation cache key includes the solver ABI, PyTorch
version, device capability, dtype, eager/compiled policy, compile mode, actual
compiled-region and material representation, DM2 algorithm constants, grid
spacing, timestep, Bloch vector, local/distributed topology, field shapes, and
every compiled material/state tensor layout. It is exposed in
`simulation.diagnostics()` as a specialization fingerprint for evidence and
cache attribution. It is not a cross-instance callable cache: compiled bound
methods retain their owning simulation, so wrappers must not be shared between
instances solely because their fingerprints match.

Bloch fields have a final length-two real plane holding real and imaginary
parts in the requested real dtype. Complex conversion occurs only at an
explicit host boundary:

```python
device_copy = simulation.state.snapshot()  # cloned tensors, same device
checkpoint = simulation.checkpoint()        # versioned full execution state
host_fields = simulation.host_snapshot()    # cloned NumPy arrays by default
pml_state = simulation.state.pml_state_snapshot() # active PML cells only

simulation.load_checkpoint(checkpoint)  # in-place; buffer addresses stay fixed
```

Nondispersive advancement does not call `.cpu()`, `.numpy()`, or `.item()`.
DM2 status validation also stays on-device; only explicit diagnostics transfer
its iteration counts. Use
`buffer_addresses()` only as an explicit diagnostic when checking fixed
storage for CUDA graph capture.
