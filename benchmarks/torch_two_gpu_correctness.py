"""One-versus-two GPU field, source, material, and restart matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
import torch.distributed as dist

import gmes
from benchmarks.host_contract import candidate_evidence, host_contract_complete
from benchmarks.torch_two_gpu import _environment as _two_gpu_environment
from gmes.torch_plan import COMPONENTS

CAPTURE_STEPS = (1, 2, 5, 20, 100)
SCHEMA_VERSION = 3
CORRECTNESS_CONTRACT_ID = "two-gpu-full-field-replay-v3"
MEDIA_TYPE_NPZ = "application/x-npz"


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


def _storage_digest(names, values):
    hasher = hashlib.sha256()
    hasher.update(json.dumps(names, separators=(",", ":")).encode())
    hasher.update(np.asarray(values, dtype=np.uint64).tobytes(order="C"))
    return hasher.hexdigest()


def _storage_record(rank, initial, final):
    names = sorted(initial)
    if names != sorted(final):
        raise ValueError("storage buffer names changed during correctness replay")
    initial_values = [initial[name] for name in names]
    final_values = [final[name] for name in names]
    record = {
        "address_names": names,
        "address_count": len(names),
        "initial_sha256": _storage_digest(names, initial_values),
        "final_sha256": _storage_digest(names, final_values),
        "addresses_stable": initial_values == final_values,
    }
    if rank is not None:
        record["rank"] = rank
    return (
        record,
        np.asarray(initial_values, dtype=np.uint64),
        np.asarray(final_values, dtype=np.uint64),
    )


def _raw_descriptor(path, root, candidate):
    path = Path(path)
    root = Path(root).resolve(strict=True)
    if path.is_symlink():
        raise ValueError("raw evidence artifact cannot be a symlink")
    path = path.resolve(strict=True)
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError("raw evidence artifact escapes descriptor root") from error
    raw = path.read_bytes()
    return {
        "path": relative.as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "media_type": MEDIA_TYPE_NPZ,
        "candidate_evidence": candidate,
    }


def _field_shapes(size, resolution):
    shapes = _global_shapes(gmes.Cartesian(size, resolution))
    return {name: list(shapes[name]) for name in COMPONENTS}


def _case_raw_array_names():
    return [
        *(
            f"capture/{step}/{role}/{name}"
            for step in CAPTURE_STEPS
            for name in COMPONENTS
            for role in ("distributed", "serial")
        ),
        *(
            f"checkpoint/{phase}/{name}"
            for phase in ("expected", "replay", "serial")
            for name in COMPONENTS
        ),
        *(
            f"storage/rank/{rank}/{phase}"
            for rank in range(2)
            for phase in ("initial", "final")
        ),
        *(f"storage/serial/{phase}" for phase in ("initial", "final")),
    ]


def _long_raw_array_names():
    return [
        f"{phase}/{name}"
        for phase in ("initial", "distributed", "serial")
        for name in COMPONENTS
    ]


def _write_raw_evidence(
    path,
    arrays,
    root,
    candidate,
    field_shapes,
    *,
    expected_names,
    field_dtype,
):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise ValueError(f"raw evidence output already exists: {path}")
    names = list(arrays)
    if names != expected_names:
        raise ValueError("raw evidence array closure or order differs")
    if set(field_shapes) != set(COMPONENTS):
        raise ValueError("raw evidence field-shape closure differs")
    expected_dtype = np.dtype(field_dtype)
    normalized = {}
    for name, value in arrays.items():
        array = np.asarray(value)
        component = name.rsplit("/", 1)[-1]
        is_storage = name.startswith("storage/")
        if (
            not isinstance(name, str)
            or not name
            or array.dtype.fields is not None
            or array.dtype.subdtype is not None
            or array.dtype.kind not in {"b", "i", "u", "f", "c"}
            or not array.flags.c_contiguous
            or not np.isfinite(array).all()
            or (is_storage and (array.dtype != np.dtype("uint64") or array.ndim != 1))
            or (
                not is_storage
                and (
                    component not in COMPONENTS
                    or array.dtype != expected_dtype
                    or list(array.shape) != field_shapes[component]
                )
            )
        ):
            raise ValueError(f"raw evidence array is invalid: {name!r}")
        normalized[name] = np.ascontiguousarray(array)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **normalized)
    return {
        "artifact": _raw_descriptor(path, root, candidate),
        "array_names": expected_names,
        "field_shapes": field_shapes,
    }


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
    initial_addresses = simulation.buffer_addresses()
    raw_arrays = {}
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
        reference_initial_addresses = reference.buffer_addresses()
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
            serial_snapshot = reference.host_snapshot()
            errors[str(target)] = _maximum_error(actual, serial_snapshot)
            for name in COMPONENTS:
                raw_arrays[f"capture/{target}/distributed/{name}"] = np.asarray(
                    actual[name]
                ).copy()
                raw_arrays[f"capture/{target}/serial/{name}"] = np.asarray(
                    serial_snapshot[name]
                ).copy()
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
    final_addresses = simulation.buffer_addresses()
    local_probes = simulation.flush_probes()
    rank_storage_raw = [None, None]
    dist.all_gather_object(
        rank_storage_raw,
        {
            "rank": launch.rank,
            "initial": initial_addresses,
            "final": final_addresses,
        },
        group=simulation.group,
    )
    result = None
    if launch.rank == 0:
        serial_snapshot = reference.host_snapshot()
        serial_final_addresses = reference.buffer_addresses()
        for phase, snapshot in (
            ("expected", expected_replay),
            ("replay", replay),
            ("serial", serial_snapshot),
        ):
            for name in COMPONENTS:
                raw_arrays[f"checkpoint/{phase}/{name}"] = np.asarray(
                    snapshot[name]
                ).copy()
        rank_storage = []
        for item in rank_storage_raw:
            storage, initial_values, final_values = _storage_record(
                item["rank"], item["initial"], item["final"]
            )
            rank_storage.append(storage)
            raw_arrays[f"storage/rank/{item['rank']}/initial"] = initial_values
            raw_arrays[f"storage/rank/{item['rank']}/final"] = final_values
        serial_storage, serial_initial, serial_final = _storage_record(
            None, reference_initial_addresses, serial_final_addresses
        )
        raw_arrays["storage/serial/initial"] = serial_initial
        raw_arrays["storage/serial/final"] = serial_final
        result = {
            "name": case["name"],
            "axis": case["axis"],
            "cut": simulation.decomposition.cut,
            "capture_errors": errors,
            "checkpoint_determinism_error": _maximum_error(replay, expected_replay),
            "checkpoint_reference_error": _maximum_error(
                replay, reference.host_snapshot()
            ),
            "checkpoint_replay_steps": 5,
            "checkpoint_replay_fields": list(COMPONENTS),
            "rank_storage": rank_storage,
            "serial_storage": serial_storage,
            "rank0_probe_count": len(local_probes["samples"]),
        }
    del simulation
    if launch.rank == 0:
        del reference
    dist.barrier()
    return result, raw_arrays if launch.rank == 0 else None


def _run_long_stability(launch, steps):
    size = (8, 6, 4)
    space = gmes.Cartesian(size, 2)
    fields = _seed_fields(space, complex_fields=False, seed=123)
    raw_arrays = {
        f"initial/{name}": np.asarray(fields[name]).copy() for name in COMPONENTS
    }
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
        serial_snapshot = reference.host_snapshot()
        for phase, snapshot in (
            ("distributed", actual),
            ("serial", serial_snapshot),
        ):
            for name in COMPONENTS:
                raw_arrays[f"{phase}/{name}"] = np.asarray(snapshot[name]).copy()
        final_energy = sum(
            float(np.square(np.abs(values)).sum()) for values in actual.values()
        )
        result = {
            "steps": steps,
            "maximum_error": _maximum_error(actual, serial_snapshot),
            "finite": all(np.isfinite(values).all() for values in actual.values()),
            "initial_energy": initial_energy,
            "final_energy": final_energy,
            "energy_ratio": final_energy / initial_energy,
        }
    del simulation
    if launch.rank == 0:
        del reference
    dist.barrier()
    return result, raw_arrays if launch.rank == 0 else None


def _environment_complete(environment):
    if not isinstance(environment, dict) or set(environment) != {
        "host_contract",
        "hostname",
        "platform",
        "python",
        "torch",
        "cuda_runtime",
        "nccl",
        "devices",
        "topology",
        "topology_command",
        "topology_command_status",
    }:
        return False
    devices = environment["devices"]
    nccl = environment["nccl"]
    statuses = {
        name: value
        for name, value in environment.items()
        if name.endswith("_command_status") and value is not None
    }
    return (
        host_contract_complete(environment["host_contract"])
        and all(
            isinstance(environment[name], str) and bool(environment[name])
            for name in ("hostname", "platform", "python", "torch", "cuda_runtime")
        )
        and (
            (type(nccl) is int and nccl > 0)
            or (
                isinstance(nccl, list)
                and bool(nccl)
                and all(type(value) is int and value >= 0 for value in nccl)
            )
        )
        and isinstance(environment["topology"], str)
        and bool(environment["topology"].strip())
        and environment["topology_command"] == ["nvidia-smi", "topo", "-m"]
        and type(environment["topology_command_status"]) is int
        and environment["topology_command_status"] == 0
        and bool(statuses)
        and all(type(value) is int and value == 0 for value in statuses.values())
        and isinstance(devices, list)
        and len(devices) >= 2
        and all(isinstance(device, dict) for device in devices)
        and [device.get("index") for device in devices[:2]] == [0, 1]
        and all(
            set(device)
            == {
                "index",
                "name",
                "memory_bytes",
                "capability",
                "multiprocessors",
            }
            and type(device["index"]) is int
            and isinstance(device["name"], str)
            and bool(device["name"])
            and type(device["memory_bytes"]) is int
            and device["memory_bytes"] > 0
            and isinstance(device["capability"], list)
            and len(device["capability"]) == 2
            and all(type(value) is int and value >= 0 for value in device["capability"])
            and device["capability"][0] > 0
            and type(device["multiprocessors"]) is int
            and device["multiprocessors"] > 0
            for device in devices
        )
    )


def _lower_hex(value, width):
    return (
        isinstance(value, str)
        and len(value) == width
        and all(character in "0123456789abcdef" for character in value)
    )


def _candidate_binding_complete(value):
    return (
        isinstance(value, dict)
        and set(value)
        == {
            "candidate_git_commit",
            "candidate_git_status",
            "manifest_sha256",
        }
        and _lower_hex(value["candidate_git_commit"], 40)
        and value["candidate_git_status"] == ""
        and _lower_hex(value["manifest_sha256"], 64)
    )


def _raw_descriptor_complete(value):
    if not isinstance(value, dict) or set(value) != {
        "path",
        "sha256",
        "size_bytes",
        "media_type",
        "candidate_evidence",
    }:
        return False
    path = value["path"]
    if not isinstance(path, str) or not path or "\\" in path:
        return False
    portable = PurePosixPath(path)
    return (
        not portable.is_absolute()
        and portable.as_posix() == path
        and all(part not in {"", ".", ".."} for part in portable.parts)
        and _lower_hex(value["sha256"], 64)
        and type(value["size_bytes"]) is int
        and value["size_bytes"] > 0
        and value["media_type"] == MEDIA_TYPE_NPZ
        and _candidate_binding_complete(value["candidate_evidence"])
    )


def _raw_evidence_metadata_complete(value, expected_names, expected_shapes):
    return (
        isinstance(value, dict)
        and set(value) == {"artifact", "array_names", "field_shapes"}
        and _raw_descriptor_complete(value["artifact"])
        and value["array_names"] == expected_names
        and value["field_shapes"] == expected_shapes
    )


def _storage_summary_complete(value, rank=None):
    expected_keys = {
        "address_names",
        "address_count",
        "initial_sha256",
        "final_sha256",
        "addresses_stable",
    }
    if rank is not None:
        expected_keys.add("rank")
    return (
        isinstance(value, dict)
        and set(value) == expected_keys
        and (rank is None or value["rank"] == rank)
        and isinstance(value["address_names"], list)
        and value["address_names"] == sorted(value["address_names"])
        and len(value["address_names"]) == len(set(value["address_names"]))
        and type(value["address_count"]) is int
        and value["address_count"] == len(value["address_names"]) > 0
        and _lower_hex(value["initial_sha256"], 64)
        and _lower_hex(value["final_sha256"], 64)
        and value["addresses_stable"] is True
    )


def _suite_acceptance(results, long_stability, numerical_passed, environment):
    case_specs = _cases()
    expected_cases = [case["name"] for case in case_specs]
    checks = {
        "environment_complete": _environment_complete(environment),
        "case_closure_complete": [record.get("name") for record in results]
        == expected_cases,
        "capture_steps_complete": all(
            set(record.get("capture_errors", ()))
            == {str(step) for step in CAPTURE_STEPS}
            for record in results
        ),
        "complete_field_replay": all(
            record.get("checkpoint_replay_fields") == list(COMPONENTS)
            and record.get("checkpoint_replay_steps") == 5
            and record.get("checkpoint_determinism_error") == 0.0
            and record.get("checkpoint_reference_error", float("inf")) <= 2e-10
            for record in results
        ),
        "rank_storage_stable": all(
            isinstance(record.get("rank_storage"), list)
            and len(record["rank_storage"]) == 2
            and all(
                _storage_summary_complete(item, rank)
                for rank, item in enumerate(record["rank_storage"])
            )
            and _storage_summary_complete(record.get("serial_storage"))
            for record in results
        ),
        "raw_full_fields_bound": (
            len(results) == len(case_specs)
            and all(
                record.get("name") == case["name"]
                and _raw_evidence_metadata_complete(
                    record.get("raw_evidence"),
                    _case_raw_array_names(),
                    _field_shapes(case["size"], case["resolution"]),
                )
                for record, case in zip(results, case_specs, strict=True)
            )
            and isinstance(long_stability, dict)
            and _raw_evidence_metadata_complete(
                long_stability.get("raw_evidence"),
                _long_raw_array_names(),
                _field_shapes((8, 6, 4), 2),
            )
        ),
        "long_stability_complete": (
            isinstance(long_stability, dict)
            and long_stability.get("steps", 0) >= 1000
            and long_stability.get("finite") is True
            and long_stability.get("maximum_error", float("inf")) <= 2e-10
            and 0 < long_stability.get("energy_ratio", float("inf")) < 100.0
        ),
        "numerical_acceptance": numerical_passed is True,
    }
    return {
        "required_cases": expected_cases,
        "required_capture_steps": list(CAPTURE_STEPS),
        "required_long_steps": 1000,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _prepare_output_paths(output, descriptor_root):
    root = Path(descriptor_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("descriptor root must be a directory")
    supplied = Path(output)
    if supplied.exists() or supplied.is_symlink():
        raise ValueError("two-GPU correctness output already exists")
    parent = supplied.parent.resolve()
    try:
        parent.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "two-GPU correctness output escapes descriptor root"
        ) from error
    parent.mkdir(parents=True, exist_ok=True)
    parent = parent.resolve(strict=True)
    output_path = parent / supplied.name
    raw_directory = parent / f"{output_path.stem}-raw"
    if raw_directory.exists() or raw_directory.is_symlink():
        raise ValueError("two-GPU correctness raw directory already exists")
    raw_directory.mkdir()
    return output_path, raw_directory, root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append")
    parser.add_argument("--capture-graphs", action="store_true")
    parser.add_argument("--long-steps", type=int, default=0)
    parser.add_argument("--enforce", action="store_true")
    parser.add_argument("--descriptor-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/gmes-two-gpu-correctness.json")
    )
    args = parser.parse_args()
    if args.long_steps < 0:
        raise ValueError("--long-steps must be non-negative")
    launch = gmes.distributed_launch_from_environment()
    cases = _cases()
    if args.case:
        requested = set(args.case)
        cases = [case for case in cases if case["name"] in requested]
        missing = requested - {case["name"] for case in cases}
        if missing:
            raise ValueError("unknown cases: " + ", ".join(sorted(missing)))
    results = []
    case_raw_arrays = {}
    for case in cases:
        result, raw_arrays = _run_case(case, launch, capture_graphs=args.capture_graphs)
        if result is not None:
            results.append(result)
            case_raw_arrays[result["name"]] = raw_arrays
    if args.long_steps:
        long_stability, long_raw_arrays = _run_long_stability(launch, args.long_steps)
    else:
        long_stability, long_raw_arrays = None, None
    if launch.rank == 0:
        output_path, raw_directory, descriptor_root = _prepare_output_paths(
            args.output, args.descriptor_root
        )
        candidate = candidate_evidence()
        case_specs = {case["name"]: case for case in cases}
        for record in results:
            case = case_specs[record["name"]]
            field_shapes = _field_shapes(case["size"], case["resolution"])
            record["raw_evidence"] = _write_raw_evidence(
                raw_directory / f"{record['name']}.npz",
                case_raw_arrays[record["name"]],
                descriptor_root,
                candidate,
                field_shapes,
                expected_names=_case_raw_array_names(),
                field_dtype="complex128" if case["bloch"] is not None else "float64",
            )
        if long_stability is not None:
            long_stability["raw_evidence"] = _write_raw_evidence(
                raw_directory / "long-stability.npz",
                long_raw_arrays,
                descriptor_root,
                candidate,
                _field_shapes((8, 6, 4), 2),
                expected_names=_long_raw_array_names(),
                field_dtype="float64",
            )
        environment = _two_gpu_environment()
        output = {
            "candidate_evidence": candidate,
            "environment": environment,
            "schema_version": SCHEMA_VERSION,
            "contract_id": CORRECTNESS_CONTRACT_ID,
            "capture_steps": list(CAPTURE_STEPS),
            "capture_graphs": args.capture_graphs,
            "execution_mode": "graph" if args.capture_graphs else "eager",
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
        output["suite_acceptance"] = _suite_acceptance(
            results,
            long_stability,
            output["passed"],
            environment,
        )
        output_path.write_text(
            json.dumps(output, allow_nan=False, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(output, allow_nan=False, sort_keys=True))
        exit_status = (
            0 if not args.enforce or output["suite_acceptance"]["passed"] else 2
        )
    else:
        exit_status = 0
    status = [exit_status if launch.rank == 0 else None]
    dist.broadcast_object_list(status, src=0, group=dist.group.WORLD)
    gmes.TorchDistributedSimulation.close()
    return int(status[0])


if __name__ == "__main__":
    raise SystemExit(main())
