"""One-versus-two GPU field, source, material, and restart matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch.distributed as dist

import gmes
from gmes.torch_plan import COMPONENTS

CAPTURE_STEPS = (1, 2, 5, 20, 100)


def _drude():
    return gmes.Drude(
        eps_inf=1.2,
        sigma=0.01,
        dps=(gmes.DrudePole(omega=0.7, gamma=0.03),),
    )


def _lorentz():
    return gmes.Lorentz(
        eps_inf=1.2,
        sigma=0.01,
        lps=(gmes.LorentzPole(amp=0.05, omega=0.8, gamma=0.03),),
    )


def _dcp(model):
    return model(
        eps_inf=1.2,
        sigma=0.01,
        dps=(gmes.DrudePole(omega=0.7, gamma=0.03),),
        cps=(
            gmes.CriticalPoint(
                amp=0.04,
                phi=0.2,
                omega=0.9,
                gamma=0.03,
            ),
        ),
    )


def _dm2():
    return gmes.Dm2(
        eps_inf=1.4,
        mu_inf=1.1,
        omega=(0.7, 1.1),
        n_atom=(0.2, 0.4),
        rho30=-0.8,
        gamma=0.15,
        t1=2.5,
        t2=1.7,
        hbar=1.2,
        rtol=1e-10,
    )


def _tfsf():
    return gmes.TotalFieldScatteredField(
        gmes.Continuous(0.2, phase=0.2, width=1),
        center=(0, 0, 0),
        size=(2, 2, 2),
        direction=(1, 0.2, 0.1),
        polarization=(0, 1, 0),
        amp=0.3,
    )


def _gaussian():
    return gmes.GaussianBeam(
        gmes.Continuous(0.2, width=0.2),
        directivity=gmes.PlusX,
        center=(0, 0, 0),
        size=(1, 1, 1),
        direction=(1, 0, 0),
        polarization=(0, 1, 0),
        waist=0.7,
        amp=0.3,
    )


def _default_geometry():
    return [gmes.DefaultMedium(gmes.Dielectric(eps_inf=1.7, mu_inf=1.05))]


def _material_geometry(factory, *, pml=None):
    geometry = _default_geometry()
    if factory is not None:
        geometry.append(gmes.Block(factory(), center=(0, 0, 0), size=(2.4, 2.4, 2.4)))
    if pml is not None:
        geometry.append(gmes.Shell(pml(), thickness=0.5))
    return geometry


def _cases():
    result = []
    for axis in range(3):
        for bloch in (None, (0.07, 0.05, 0.03)):
            result.append(
                {
                    "name": f"axis-{axis}-{'bloch' if bloch else 'real'}",
                    "size": (3.5, 3.0, 2.5),
                    "resolution": 2,
                    "axis": axis,
                    "bloch": bloch,
                    "geometry": _default_geometry,
                    "sources": lambda: (),
                    "probes": lambda: (),
                }
            )
    for name, size in (
        ("collapsed-1d", (4, 0, 0)),
        ("collapsed-2d", (4, 3, 0)),
    ):
        result.append(
            {
                "name": name,
                "size": size,
                "resolution": 2,
                "axis": 0,
                "bloch": None,
                "geometry": _default_geometry,
                "sources": lambda: (),
                "probes": lambda: (),
            }
        )
    materials = (
        ("upml", None, gmes.Upml),
        ("cpml", None, gmes.Cpml),
        ("drude", _drude, None),
        ("lorentz", _lorentz, None),
        ("dcp-ade", lambda: _dcp(gmes.DcpAde), None),
        ("dcp-plrc", lambda: _dcp(gmes.DcpPlrc), None),
        ("dcp-rc", lambda: _dcp(gmes.DcpRc), None),
        ("dm2", _dm2, None),
    )
    for name, factory, pml in materials:
        result.append(
            {
                "name": name,
                "size": (4, 4, 4),
                "resolution": 2,
                "axis": 0,
                "bloch": None,
                "geometry": lambda factory=factory, pml=pml: _material_geometry(
                    factory, pml=pml
                ),
                "sources": lambda: (),
                "probes": lambda: (),
            }
        )
    for name, source in (("tfsf", _tfsf), ("gaussian", _gaussian)):
        result.append(
            {
                "name": name,
                "size": (4, 4, 4),
                "resolution": 2,
                "axis": 0,
                "bloch": None,
                "geometry": _default_geometry,
                "sources": lambda source=source: (source(),),
                "probes": lambda: (
                    gmes.TorchProbeSpec("Ez", (3, 3, 3), capacity=256),
                    gmes.TorchProbeSpec("Hy", (5, 4, 4), capacity=256),
                ),
            }
        )
    return result


def _global_shapes(space):
    nx, ny, nz = (int(value) for value in space.whole_field_size)
    return {
        "Ex": (nx, ny + 1, nz + 1),
        "Ey": (nx + 1, ny, nz + 1),
        "Ez": (nx + 1, ny + 1, nz),
        "Hx": (nx, ny + 1, nz + 1),
        "Hy": (nx + 1, ny, nz + 1),
        "Hz": (nx + 1, ny + 1, nz),
    }


def _seed_fields(space, *, complex_fields, seed):
    rng = np.random.default_rng(seed)
    result = {}
    for name, shape in _global_shapes(space).items():
        values = rng.normal(size=shape) * 1e-3
        if complex_fields:
            values = values + 1j * rng.normal(size=shape) * 1e-3
        result[name] = values
    return result


def _maximum_error(actual, expected):
    return max(
        float(np.max(np.abs(actual[name] - expected[name]))) for name in COMPONENTS
    )


def _run_case(case, launch, *, capture_graphs):
    runtime = gmes.TorchRuntimeConfig(
        device=f"cuda:{launch.local_rank}",
        precision="float64",
        compile_policy="compile" if capture_graphs else "eager",
        cpu_threads=1,
        launch=launch,
    )
    global_space = gmes.Cartesian(case["size"], case["resolution"])
    fields = _seed_fields(
        global_space,
        complex_fields=case["bloch"] is not None,
        seed=122,
    )
    simulation = gmes.TorchDistributedSimulation(
        space=global_space,
        geometry=case["geometry"](),
        runtime=runtime,
        dt=0.025,
        bloch=case["bloch"],
        sources=case["sources"](),
        probes=case["probes"](),
        split_axis=case["axis"],
    )
    simulation.load_host_fields(fields)
    if launch.rank == 0:
        reference = gmes.TorchSimulation(
            space=gmes.Cartesian(case["size"], case["resolution"]),
            geometry=case["geometry"](),
            runtime=gmes.TorchRuntimeConfig(
                device="cuda:0",
                precision="float64",
                compile_policy="compile" if capture_graphs else "eager",
                cpu_threads=1,
            ),
            dt=0.025,
            bloch=case["bloch"],
            sources=case["sources"](),
            probes=case["probes"](),
        )
        reference.load_host_fields(fields)
    if capture_graphs:
        simulation.capture_cuda_graphs()
        if launch.rank == 0:
            reference.capture_cuda_graphs()
    errors = {}
    completed = 0
    for target in CAPTURE_STEPS:
        simulation.advance(target - completed)
        if launch.rank == 0:
            reference.advance(target - completed)
        actual = simulation.global_field_snapshot()
        if launch.rank == 0:
            errors[str(target)] = _maximum_error(actual, reference.host_snapshot())
        completed = target
    checkpoint = simulation.checkpoint()
    if launch.rank == 0:
        reference_checkpoint = reference.checkpoint()
    simulation.advance(5)
    if launch.rank == 0:
        reference.advance(5)
    expected_replay = simulation.global_field_snapshot()
    simulation.load_checkpoint(checkpoint).advance(5)
    if launch.rank == 0:
        reference.load_checkpoint(reference_checkpoint).advance(5)
    replay = simulation.global_field_snapshot()
    local_probes = simulation.flush_probes()
    result = None
    if launch.rank == 0:
        result = {
            "name": case["name"],
            "axis": case["axis"],
            "cut": simulation.decomposition.cut,
            "capture_errors": errors,
            "checkpoint_determinism_error": _maximum_error(replay, expected_replay),
            "checkpoint_reference_error": _maximum_error(
                replay, reference.host_snapshot()
            ),
            "rank0_probe_count": len(local_probes["samples"]),
        }
    del simulation
    if launch.rank == 0:
        del reference
    dist.barrier()
    return result


def _run_long_stability(launch, steps):
    size = (8, 6, 4)
    space = gmes.Cartesian(size, 2)
    fields = _seed_fields(space, complex_fields=False, seed=123)
    initial_energy = sum(float(np.square(values).sum()) for values in fields.values())
    simulation = gmes.TorchDistributedSimulation(
        space=space,
        geometry=_default_geometry(),
        runtime=gmes.TorchRuntimeConfig(
            device=f"cuda:{launch.local_rank}",
            precision="float64",
            cpu_threads=1,
            launch=launch,
        ),
        dt=0.025,
        split_axis=0,
    ).load_host_fields(fields)
    if launch.rank == 0:
        reference = gmes.TorchSimulation(
            space=gmes.Cartesian(size, 2),
            geometry=_default_geometry(),
            runtime=gmes.TorchRuntimeConfig(
                device="cuda:0", precision="float64", cpu_threads=1
            ),
            dt=0.025,
        ).load_host_fields(fields)
    simulation.advance(steps)
    if launch.rank == 0:
        reference.advance(steps)
    actual = simulation.global_field_snapshot()
    result = None
    if launch.rank == 0:
        final_energy = sum(
            float(np.square(np.abs(values)).sum()) for values in actual.values()
        )
        result = {
            "steps": steps,
            "maximum_error": _maximum_error(actual, reference.host_snapshot()),
            "finite": all(np.isfinite(values).all() for values in actual.values()),
            "initial_energy": initial_energy,
            "final_energy": final_energy,
            "energy_ratio": final_energy / initial_energy,
        }
    dist.barrier()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append")
    parser.add_argument("--capture-graphs", action="store_true")
    parser.add_argument("--long-steps", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/gmes-two-gpu-correctness.json")
    )
    args = parser.parse_args()
    launch = gmes.distributed_launch_from_environment()
    cases = _cases()
    if args.case:
        requested = set(args.case)
        cases = [case for case in cases if case["name"] in requested]
        missing = requested - {case["name"] for case in cases}
        if missing:
            raise ValueError("unknown cases: " + ", ".join(sorted(missing)))
    results = []
    for case in cases:
        result = _run_case(case, launch, capture_graphs=args.capture_graphs)
        if result is not None:
            results.append(result)
    long_stability = (
        _run_long_stability(launch, args.long_steps) if args.long_steps else None
    )
    if launch.rank == 0:
        output = {
            "schema_version": 1,
            "capture_steps": list(CAPTURE_STEPS),
            "capture_graphs": args.capture_graphs,
            "maximum_error": max(
                max(record["capture_errors"].values()) for record in results
            ),
            "passed": all(
                max(record["capture_errors"].values()) <= 2e-10
                and record["checkpoint_determinism_error"] == 0.0
                and record["checkpoint_reference_error"] <= 2e-10
                for record in results
            ),
            "cases": results,
            "long_stability": long_stability,
        }
        if long_stability is not None:
            output["passed"] = (
                output["passed"]
                and long_stability["finite"]
                and long_stability["maximum_error"] <= 2e-10
                and long_stability["energy_ratio"] < 100.0
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        print(json.dumps(output, sort_keys=True))
    gmes.TorchDistributedSimulation.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
