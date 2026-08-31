"""Fixed two-GPU strong/weak scaling and overlap evidence runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from statistics import median

import torch
import torch.distributed as dist

import gmes
from benchmarks.host_contract import (
    ROOT,
    candidate_evidence,
    capture_host_contract,
    host_contract_complete,
)
from benchmarks.native_oracle import _coverage_geometry

SCHEMA_VERSION = 2
WORKER_KINDS = {
    "serial": "two-gpu-serial-worker",
    "distributed": "two-gpu-distributed-worker",
}
HALO_ANNOTATIONS = tuple(
    f"gmes::halo_{phase}_{operation}"
    for phase in ("magnetic", "electric")
    for operation in ("pack_launch", "exposed_wait", "boundary_unpack")
)
DESCRIPTOR_KEYS = {
    "path",
    "sha256",
    "size_bytes",
    "media_type",
    "candidate_evidence",
}
SUBPROCESS_RECORD_KEYS = {
    "role",
    "command",
    "exit_code",
    "stdout_sha256",
    "stdout_size_bytes",
    "stderr_sha256",
    "stderr_size_bytes",
    "artifact_sha256",
    "artifact_size_bytes",
    "artifact",
    "stdout",
    "stderr",
}

CASES = {
    "strong-mixed": {
        "serial_size": (128, 96, 96),
        "distributed_size": (128, 96, 96),
        "geometry": "mixed",
        "gate": "strong",
    },
    "weak-mixed": {
        "serial_size": (96, 96, 96),
        "distributed_size": (192, 96, 96),
        "geometry": "mixed",
        "gate": "weak",
    },
    "strong-homogeneous": {
        "serial_size": (128, 96, 96),
        "distributed_size": (128, 96, 96),
        "geometry": "homogeneous",
        "gate": "informational",
    },
    "strong-imbalanced": {
        "serial_size": (128, 96, 96),
        "distributed_size": (128, 96, 96),
        "geometry": "imbalanced",
        "gate": "strong",
    },
}


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, default="strong-mixed")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--profile-steps", type=int, default=10)
    parser.add_argument("--threads-per-rank", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace-directory", type=Path, default=Path("/tmp"))
    parser.add_argument("--descriptor-root", type=Path)
    parser.add_argument("--enforce", action="store_true")
    parser.add_argument(
        "--worker",
        choices=tuple(WORKER_KINDS),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def _mixed_geometry(size):
    return _coverage_geometry(
        {
            "size": list(size),
            "resolution": 1,
            "coverage_percent": 50,
            "layout": "fragmented",
            "include_pml": True,
        },
        gmes,
    )


def _imbalanced_geometry(size):
    width = float(size[0]) * 0.68
    return [
        gmes.DefaultMedium(gmes.Dielectric(eps_inf=1.7, mu_inf=1.05)),
        gmes.Shell(gmes.Cpml(), thickness=max(0.25, min(size) * 0.04)),
        gmes.Block(
            gmes.DcpAde(
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
            ),
            center=(-0.5 * float(size[0]) + 0.5 * width, 0, 0),
            size=(width, float(size[1]), float(size[2])),
        ),
    ]


def _geometry(kind, size):
    if kind == "mixed":
        return _mixed_geometry(size)
    if kind == "imbalanced":
        return _imbalanced_geometry(size)
    return [gmes.DefaultMedium(gmes.Dielectric(eps_inf=1.7, mu_inf=1.05))]


def _runtime(device, *, threads, launch=None):
    return gmes.TorchRuntimeConfig(
        device=device,
        precision="float32",
        compile_policy="compile",
        execution_policy="auto",
        cpu_threads=threads,
        launch=gmes.DistributedLaunch() if launch is None else launch,
    )


def _synchronize(device):
    torch.cuda.synchronize(device)


def _canonical_sha256(value):
    raw = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _tensor_bytes(tensor):
    return int(tensor.numel() * tensor.element_size())


def _tracked_tensors(simulation):
    local = getattr(simulation, "local", simulation)
    modules = (
        ("state", local.state),
        ("plan", local.plan),
        ("sources", local.sources),
        ("probes", local.probes),
    )
    records = []
    for prefix, module in modules:
        records.extend(
            (f"{prefix}.{name}", tensor) for name, tensor in module.named_buffers()
        )
    overlay = getattr(local, "_dispersive_overlay", None)
    if overlay is not None:
        records.extend(
            (f"dispersive_overlay.{name}", tensor)
            for name, tensor in overlay.named_buffers()
        )
    exchange = getattr(simulation, "exchange", None)
    if exchange is not None:
        records.extend(
            (f"halo.{index}", tensor) for index, tensor in enumerate(exchange.buffers())
        )
    return records


def _storage_snapshot(simulation):
    addresses = dict(simulation.buffer_addresses())
    exchange = getattr(simulation, "exchange", None)
    halo = list(exchange.buffers()) if exchange is not None else []
    addresses.update(
        {f"halo.{index}": tensor.data_ptr() for index, tensor in enumerate(halo)}
    )
    tracked = _tracked_tensors(simulation)
    category_bytes = {}
    for name, tensor in tracked:
        category = name.partition(".")[0]
        category_bytes[category] = category_bytes.get(category, 0) + _tensor_bytes(
            tensor
        )
    return {
        "address_count": len(addresses),
        "address_sha256": _canonical_sha256(addresses),
        "alias_count": len(addresses) - len(set(addresses.values())),
        "tracked_tensor_count": len(tracked),
        "device_resident": all(
            tensor.device == simulation.device for _, tensor in tracked
        ),
        "resident_bytes": sum(_tensor_bytes(tensor) for _, tensor in tracked),
        "category_bytes": dict(sorted(category_bytes.items())),
    }


def _halo_snapshot(simulation):
    buffers = list(simulation.exchange.buffers())
    addresses = [tensor.data_ptr() for tensor in buffers]
    return {
        "buffer_count": len(buffers),
        "bytes": sum(_tensor_bytes(tensor) for tensor in buffers),
        "address_sha256": _canonical_sha256(addresses),
        "alias_count": len(addresses) - len(set(addresses)),
        "device": str(simulation.device),
        "device_resident": all(
            tensor.device == simulation.device for tensor in buffers
        ),
    }


def _storage_evidence(initial, final):
    return {
        "address_count": initial["address_count"],
        "initial_address_sha256": initial["address_sha256"],
        "final_address_sha256": final["address_sha256"],
        "addresses_stable": initial == final,
        "alias_count": initial["alias_count"],
        "tracked_tensor_count": initial["tracked_tensor_count"],
        "device_resident": initial["device_resident"] and final["device_resident"],
        "resident_bytes": initial["resident_bytes"],
        "category_bytes": initial["category_bytes"],
    }


def _halo_evidence(initial, final):
    return {
        "buffer_count": initial["buffer_count"],
        "bytes": initial["bytes"],
        "initial_address_sha256": initial["address_sha256"],
        "final_address_sha256": final["address_sha256"],
        "addresses_stable": initial == final,
        "alias_count": initial["alias_count"],
        "device": initial["device"],
        "device_resident": initial["device_resident"] and final["device_resident"],
    }


def _memory_evidence(device, before):
    after = int(torch.cuda.memory_allocated(device))
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    growth = after - before
    return {
        "allocated_before_bytes": before,
        "allocated_after_bytes": after,
        "allocated_growth_bytes": growth,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "bounded": growth <= 1024 * 1024 and peak_reserved >= peak_allocated > 0,
    }


def _serial_samples(simulation, checkpoint, *, warmup, steps, repeats):
    samples = []
    for _ in range(repeats):
        simulation.load_checkpoint(checkpoint)
        simulation.advance(warmup)
        _synchronize(simulation.device)
        start = time.perf_counter()
        simulation.advance(steps)
        _synchronize(simulation.device)
        samples.append(time.perf_counter() - start)
    return samples


def _distributed_samples(simulation, checkpoint, *, warmup, steps, repeats):
    samples = []
    rank_samples = []
    local_samples = []
    for _ in range(repeats):
        simulation.load_checkpoint(checkpoint)
        simulation.advance(warmup)
        _synchronize(simulation.device)
        dist.barrier(group=simulation.group)
        start = time.perf_counter()
        simulation.advance(steps)
        _synchronize(simulation.device)
        elapsed = torch.tensor(
            [time.perf_counter() - start],
            device=simulation.device,
            dtype=torch.float64,
        )
        gathered = [torch.empty_like(elapsed) for _ in range(2)]
        dist.all_gather(gathered, elapsed, group=simulation.group)
        rank_samples.append([float(item.cpu()) for item in gathered])
        local_samples.append(float(elapsed.cpu()))
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=simulation.group)
        samples.append(float(elapsed.cpu()))
    return samples, rank_samples, local_samples


def _interval_union(intervals):
    merged = []
    for start, stop in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)
    return merged


def _interval_duration(intervals):
    return sum(stop - start for start, stop in _interval_union(intervals))


def _intersection_duration(first, second):
    left = _interval_union(first)
    right = _interval_union(second)
    total = 0.0
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        start = max(left[left_index][0], right[right_index][0])
        stop = min(left[left_index][1], right[right_index][1])
        total += max(0.0, stop - start)
        if left[left_index][1] < right[right_index][1]:
            left_index += 1
        else:
            right_index += 1
    return total


def _trace_summary(path, descriptor_root=None):
    raw = path.read_bytes()
    trace = json.loads(raw)
    kernels = [
        event
        for event in trace["traceEvents"]
        if event.get("ph") == "X" and event.get("cat", "").lower() == "kernel"
    ]
    nccl = [
        (float(event["ts"]), float(event["ts"] + event["dur"]))
        for event in kernels
        if "nccl" in event.get("name", "").lower()
    ]
    compute = [
        (float(event["ts"]), float(event["ts"] + event["dur"]))
        for event in kernels
        if "nccl" not in event.get("name", "").lower()
        and "memcpy" not in event.get("name", "").lower()
        and "memset" not in event.get("name", "").lower()
    ]
    nccl_duration = _interval_duration(nccl)
    overlap = _intersection_duration(nccl, compute)
    h2d = 0
    d2h = 0
    for event in trace["traceEvents"]:
        if event.get("ph") != "X":
            continue
        name = event.get("name", "").lower()
        category = event.get("cat", "").lower()
        if "memcpy" not in name and "memcpy" not in category:
            continue
        if "htod" in name or "host to device" in name:
            h2d += 1
        if "dtoh" in name or "device to host" in name:
            d2h += 1
    annotations = {name: {"count": 0, "duration_us": 0.0} for name in HALO_ANNOTATIONS}
    for event in trace["traceEvents"]:
        name = event.get("name", "")
        if event.get("ph") == "X" and name in annotations:
            annotations[name]["count"] += 1
            annotations[name]["duration_us"] += float(event["dur"])
    return {
        "trace": (
            path.resolve().relative_to(descriptor_root.resolve()).as_posix()
            if descriptor_root is not None
            else path.name
        ),
        "trace_size_bytes": len(raw),
        "trace_sha256": hashlib.sha256(raw).hexdigest(),
        "kernel_launches": len(kernels),
        "host_to_device_events": h2d,
        "device_to_host_events": d2h,
        "nccl_kernel_launches": len(nccl),
        "nccl_device_us": nccl_duration,
        "nccl_compute_overlap_us": overlap,
        "nccl_exposed_us": max(0.0, nccl_duration - overlap),
        "overlap_fraction": overlap / nccl_duration if nccl_duration else 0.0,
        "halo_annotations": annotations,
    }


def _profile(simulation, *, steps, directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"gmes-two-gpu-{simulation.rank}.json"
    simulation.exchange.profile_annotations = True
    with torch.profiler.profile(
        activities=(
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ),
        profile_memory=True,
        record_shapes=False,
    ) as profile:
        simulation.advance(steps)
        _synchronize(simulation.device)
    simulation.exchange.profile_annotations = False
    profile.export_chrome_trace(str(path))
    dist.barrier(group=simulation.group)
    return path


def _environment():
    topology_command = ["nvidia-smi", "topo", "-m"]
    try:
        command = subprocess.run(
            topology_command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        topology = ""
        topology_status = 127
    else:
        topology = command.stdout.strip()
        topology_status = int(command.returncode)
    inventory = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        inventory.append(
            {
                "index": index,
                "name": props.name,
                "memory_bytes": props.total_memory,
                "capability": [props.major, props.minor],
                "multiprocessors": props.multi_processor_count,
            }
        )
    return {
        "host_contract": capture_host_contract(torch),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "nccl": torch.cuda.nccl.version(),
        "devices": inventory,
        "topology": topology,
        "topology_command": topology_command,
        "topology_command_status": topology_status,
    }


def _summary(samples, *, steps, cells):
    middle = median(samples)
    return {
        "raw_seconds": samples,
        "median_seconds": middle,
        "steps_per_second": steps / middle,
        "cells_per_second": cells * steps / middle,
    }


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _measurement(args):
    return {
        "warmup": args.warmup,
        "steps": args.steps,
        "repeats": args.repeats,
        "profile_steps": args.profile_steps,
        "threads_per_rank": args.threads_per_rank,
    }


def _run_serial_worker(args):
    spec = CASES[args.case]
    size = spec["serial_size"]
    start = time.perf_counter()
    simulation = gmes.TorchSimulation(
        space=gmes.Cartesian(size, 1),
        geometry=_geometry(spec["geometry"], size),
        runtime=_runtime("cuda:0", threads=args.threads_per_rank),
    )
    construction = time.perf_counter() - start
    start = time.perf_counter()
    simulation.capture_cuda_graphs()
    capture = time.perf_counter() - start
    checkpoint = simulation.checkpoint()
    initial_storage = _storage_snapshot(simulation)
    before = int(torch.cuda.memory_allocated(simulation.device))
    torch.cuda.reset_peak_memory_stats(simulation.device)
    samples = _serial_samples(
        simulation,
        checkpoint,
        warmup=args.warmup,
        steps=args.steps,
        repeats=args.repeats,
    )
    memory = _memory_evidence(simulation.device, before)
    final_storage = _storage_snapshot(simulation)
    serial = _summary(samples, steps=args.steps, cells=math.prod(size))
    serial.update(
        {
            "construction_seconds": construction,
            "capture_seconds": capture,
            "memory": memory,
            "storage": _storage_evidence(initial_storage, final_storage),
        }
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": WORKER_KINDS["serial"],
        "candidate_evidence": candidate_evidence(),
        "environment": _environment(),
        "case": args.case,
        "size": list(size),
        "measurement": _measurement(args),
        "serial": serial,
    }
    _write_json(args.worker_output, result)
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


def _run_distributed_worker(args):
    launch = gmes.distributed_launch_from_environment()
    spec = CASES[args.case]
    size = spec["distributed_size"]
    start = time.perf_counter()
    simulation = gmes.TorchDistributedSimulation(
        space=gmes.Cartesian(size, 1),
        geometry=_geometry(spec["geometry"], size),
        runtime=_runtime(
            f"cuda:{launch.local_rank}",
            threads=args.threads_per_rank,
            launch=launch,
        ),
    )
    local_construction = time.perf_counter() - start
    start = time.perf_counter()
    simulation.capture_cuda_graphs()
    local_capture = time.perf_counter() - start
    checkpoint = simulation.checkpoint()
    initial_storage = _storage_snapshot(simulation)
    initial_halo = _halo_snapshot(simulation)
    before = int(torch.cuda.memory_allocated(simulation.device))
    torch.cuda.reset_peak_memory_stats(simulation.device)
    distributed_samples, rank_samples, local_samples = _distributed_samples(
        simulation,
        checkpoint,
        warmup=args.warmup,
        steps=args.steps,
        repeats=args.repeats,
    )
    memory = _memory_evidence(simulation.device, before)
    _profile(
        simulation,
        steps=args.profile_steps,
        directory=args.trace_directory,
    )
    final_storage = _storage_snapshot(simulation)
    final_halo = _halo_snapshot(simulation)
    diagnostics = simulation.diagnostics()
    rank_record = {
        "rank": launch.rank,
        "local_rank": launch.local_rank,
        "device": str(simulation.device),
        "peer_rank": 1 - launch.rank,
        "peer_access": diagnostics["peer_access"],
        "construction_seconds": local_construction,
        "capture_seconds": local_capture,
        "raw_seconds": local_samples,
        "memory": memory,
        "storage": _storage_evidence(initial_storage, final_storage),
        "halo": _halo_evidence(initial_halo, final_halo),
        "decomposition_identity": simulation.decomposition.identity,
        "local_field_shape": list(diagnostics["local_field_shape"]),
        "global_offset": list(simulation.decomposition.offset(launch.rank)),
    }
    rank_evidence = [None, None]
    dist.all_gather_object(rank_evidence, rank_record, group=simulation.group)
    dist.barrier(group=simulation.group)
    if launch.rank == 0:
        construction = max(record["construction_seconds"] for record in rank_evidence)
        capture = max(record["capture_seconds"] for record in rank_evidence)
        distributed = _summary(
            distributed_samples,
            steps=args.steps,
            cells=math.prod(size),
        )
        distributed.update(
            {
                "rank_raw_seconds": rank_samples,
                "construction_seconds": construction,
                "capture_seconds": capture,
                "peak_allocated_bytes_rank0": rank_evidence[0]["memory"][
                    "peak_allocated_bytes"
                ],
                "halo_bytes_rank0": rank_evidence[0]["halo"]["bytes"],
                "storage_addresses_stable": all(
                    record["storage"]["addresses_stable"] for record in rank_evidence
                ),
            }
        )
        result = {
            "schema_version": SCHEMA_VERSION,
            "kind": WORKER_KINDS["distributed"],
            "candidate_evidence": candidate_evidence(),
            "environment": _environment(),
            "case": args.case,
            "size": list(size),
            "measurement": _measurement(args),
            "distributed": distributed,
            "decomposition": simulation.decomposition.metadata(),
            "rank_evidence": rank_evidence,
            "profiles": [
                _trace_summary(
                    args.trace_directory / f"gmes-two-gpu-{rank}.json",
                    args.descriptor_root,
                )
                for rank in range(2)
            ],
        }
        _write_json(args.worker_output, result)
        print(json.dumps(result, allow_nan=False, sort_keys=True))
    gmes.TorchDistributedSimulation.close()
    return 0


def _worker_options(args, role, output):
    options = [
        "--worker",
        role,
        "--worker-output",
        os.path.relpath(output.resolve(), ROOT),
        "--case",
        args.case,
        "--warmup",
        str(args.warmup),
        "--steps",
        str(args.steps),
        "--repeats",
        str(args.repeats),
        "--profile-steps",
        str(args.profile_steps),
        "--threads-per-rank",
        str(args.threads_per_rank),
        "--trace-directory",
        os.path.relpath(args.trace_directory.resolve(), ROOT),
    ]
    if args.descriptor_root is not None:
        options.extend(
            (
                "--descriptor-root",
                os.path.relpath(args.descriptor_root.resolve(), ROOT),
            )
        )
    return options


def _worker_commands(args, directory):
    serial_output = directory / f"{args.case}.serial-child.json"
    distributed_output = directory / f"{args.case}.distributed-child.json"
    python = os.path.relpath(Path(sys.executable).absolute(), ROOT)
    serial = [
        python,
        "-m",
        "benchmarks.torch_two_gpu",
        *_worker_options(args, "serial", serial_output),
    ]
    distributed = [
        python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        "--module",
        "benchmarks.torch_two_gpu",
        *_worker_options(args, "distributed", distributed_output),
    ]
    return {
        "serial": (serial, serial_output),
        "distributed": (distributed, distributed_output),
    }


def _object_without_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json_bytes(raw):
    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(
        raw,
        object_pairs_hook=_object_without_duplicate_keys,
        parse_constant=reject_constant,
    )


def _strict_json(path):
    return _strict_json_bytes(path.read_bytes())


def _run_worker(role, command, output, descriptor_root):
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    stdout_path = output.with_suffix(".stdout")
    stderr_path = output.with_suffix(".stderr")
    stdout_path.write_bytes(completed.stdout)
    stderr_path.write_bytes(completed.stderr)
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"{role} worker exited with {completed.returncode}: {stderr[-2000:]}"
        )
    if not output.is_file():
        raise RuntimeError(f"{role} worker did not create its result")
    raw = output.read_bytes()
    document = _strict_json(output)
    stdout_document = _strict_json_bytes(completed.stdout)
    if _canonical_sha256(stdout_document) != _canonical_sha256(document):
        raise RuntimeError(f"{role} worker stdout differs from its result artifact")
    candidate = document.get("candidate_evidence")

    def descriptor(path, media_type):
        content = path.read_bytes()
        return {
            "path": path.resolve().relative_to(descriptor_root).as_posix(),
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
            "media_type": media_type,
            "candidate_evidence": candidate,
        }

    record = {
        "role": role,
        "command": command,
        "exit_code": int(completed.returncode),
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "stdout_size_bytes": len(completed.stdout),
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        "stderr_size_bytes": len(completed.stderr),
        "artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "artifact_size_bytes": len(raw),
        "artifact": descriptor(output, "application/json"),
        "stdout": descriptor(stdout_path, "application/json"),
        "stderr": descriptor(stderr_path, "text/plain; charset=utf-8"),
    }
    return document, record


def _positive_finite(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value > 0
    )


def _hex_string(value, width):
    return (
        isinstance(value, str)
        and len(value) == width
        and all(character in "0123456789abcdef" for character in value)
    )


def _portable_relative_path(value):
    if not isinstance(value, str) or not value:
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and path.as_posix() == value
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


def _descriptor_valid(descriptor, candidate, media_type):
    return (
        isinstance(descriptor, dict)
        and set(descriptor) == DESCRIPTOR_KEYS
        and _portable_relative_path(descriptor.get("path"))
        and _hex_string(descriptor.get("sha256"), 64)
        and type(descriptor.get("size_bytes")) is int
        and descriptor["size_bytes"] >= 0
        and descriptor.get("media_type") == media_type
        and descriptor.get("candidate_evidence") == candidate
    )


def _subprocess_record_valid(record, role, candidate):
    if not isinstance(record, dict) or set(record) != SUBPROCESS_RECORD_KEYS:
        return False
    artifact = record.get("artifact")
    stdout = record.get("stdout")
    stderr = record.get("stderr")
    return (
        record.get("role") == role
        and record.get("exit_code") == 0
        and isinstance(record.get("command"), list)
        and bool(record["command"])
        and all(isinstance(value, str) and value for value in record["command"])
        and _descriptor_valid(artifact, candidate, "application/json")
        and _descriptor_valid(stdout, candidate, "application/json")
        and _descriptor_valid(stderr, candidate, "text/plain; charset=utf-8")
        and record.get("artifact_sha256") == artifact.get("sha256")
        and record.get("artifact_size_bytes") == artifact.get("size_bytes")
        and record.get("stdout_sha256") == stdout.get("sha256")
        and record.get("stdout_size_bytes") == stdout.get("size_bytes")
        and record.get("stderr_sha256") == stderr.get("sha256")
        and record.get("stderr_size_bytes") == stderr.get("size_bytes")
    )


def _ratio(values):
    if not values or not all(_positive_finite(value) for value in values):
        raise ValueError("imbalance values must be finite and positive")
    return max(values) / min(values)


def _memory_valid(memory):
    return (
        isinstance(memory, dict)
        and set(memory)
        == {
            "allocated_before_bytes",
            "allocated_after_bytes",
            "allocated_growth_bytes",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "bounded",
        }
        and type(memory.get("allocated_before_bytes")) is int
        and memory["allocated_before_bytes"] >= 0
        and type(memory.get("allocated_after_bytes")) is int
        and memory["allocated_after_bytes"] >= 0
        and memory.get("allocated_growth_bytes")
        == memory["allocated_after_bytes"] - memory["allocated_before_bytes"]
        and memory["allocated_growth_bytes"] <= 1024 * 1024
        and type(memory.get("peak_allocated_bytes")) is int
        and memory["peak_allocated_bytes"] > 0
        and type(memory.get("peak_reserved_bytes")) is int
        and memory["peak_reserved_bytes"] >= memory["peak_allocated_bytes"]
        and memory.get("bounded") is True
    )


def _storage_valid(storage):
    return (
        isinstance(storage, dict)
        and set(storage)
        == {
            "address_count",
            "initial_address_sha256",
            "final_address_sha256",
            "addresses_stable",
            "alias_count",
            "tracked_tensor_count",
            "device_resident",
            "resident_bytes",
            "category_bytes",
        }
        and type(storage.get("address_count")) is int
        and storage["address_count"] > 0
        and type(storage.get("tracked_tensor_count")) is int
        and storage["tracked_tensor_count"] > 0
        and type(storage.get("resident_bytes")) is int
        and storage["resident_bytes"] > 0
        and type(storage.get("alias_count")) is int
        and 0 <= storage["alias_count"] <= storage["address_count"]
        and storage.get("addresses_stable") is True
        and storage.get("device_resident") is True
        and _hex_string(storage.get("initial_address_sha256"), 64)
        and storage["initial_address_sha256"] == storage.get("final_address_sha256")
        and isinstance(storage.get("category_bytes"), dict)
        and bool(storage["category_bytes"])
        and all(
            isinstance(name, str) and bool(name) and type(value) is int and value >= 0
            for name, value in storage["category_bytes"].items()
        )
        and sum(storage["category_bytes"].values()) == storage["resident_bytes"]
    )


def _halo_valid(halo, device):
    return (
        isinstance(halo, dict)
        and set(halo)
        == {
            "buffer_count",
            "bytes",
            "initial_address_sha256",
            "final_address_sha256",
            "addresses_stable",
            "alias_count",
            "device",
            "device_resident",
        }
        and type(halo.get("buffer_count")) is int
        and halo["buffer_count"] > 0
        and type(halo.get("bytes")) is int
        and halo["bytes"] > 0
        and type(halo.get("alias_count")) is int
        and 0 <= halo["alias_count"] <= halo["buffer_count"]
        and halo.get("device") == device
        and halo.get("addresses_stable") is True
        and halo.get("device_resident") is True
        and _hex_string(halo.get("initial_address_sha256"), 64)
        and halo.get("initial_address_sha256") == halo.get("final_address_sha256")
    )


def _profile_valid(profile, profile_steps):
    annotations = profile.get("halo_annotations") if isinstance(profile, dict) else None
    return (
        isinstance(profile, dict)
        and set(profile)
        == {
            "trace",
            "trace_size_bytes",
            "trace_sha256",
            "kernel_launches",
            "host_to_device_events",
            "device_to_host_events",
            "nccl_kernel_launches",
            "nccl_device_us",
            "nccl_compute_overlap_us",
            "nccl_exposed_us",
            "overlap_fraction",
            "halo_annotations",
        }
        and _portable_relative_path(profile.get("trace"))
        and type(profile.get("trace_size_bytes")) is int
        and profile["trace_size_bytes"] > 0
        and _hex_string(profile.get("trace_sha256"), 64)
        and type(profile.get("kernel_launches")) is int
        and profile["kernel_launches"] > 0
        and profile.get("host_to_device_events") == 0
        and profile.get("device_to_host_events") == 0
        and type(profile.get("nccl_kernel_launches")) is int
        and profile["nccl_kernel_launches"] > 0
        and _positive_finite(profile.get("nccl_device_us"))
        and _positive_finite(profile.get("nccl_compute_overlap_us"))
        and profile["nccl_compute_overlap_us"] <= profile["nccl_device_us"]
        and isinstance(profile.get("nccl_exposed_us"), (int, float))
        and not isinstance(profile["nccl_exposed_us"], bool)
        and math.isfinite(profile["nccl_exposed_us"])
        and profile["nccl_exposed_us"] >= 0
        and _positive_finite(profile.get("overlap_fraction"))
        and profile["overlap_fraction"] <= 1.0
        and math.isclose(
            profile["nccl_exposed_us"],
            profile["nccl_device_us"] - profile["nccl_compute_overlap_us"],
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
        and math.isclose(
            profile["overlap_fraction"],
            profile["nccl_compute_overlap_us"] / profile["nccl_device_us"],
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        and isinstance(annotations, dict)
        and set(annotations) == set(HALO_ANNOTATIONS)
        and all(
            isinstance(record, dict)
            and set(record) == {"count", "duration_us"}
            and type(record.get("count")) is int
            and record["count"] >= profile_steps
            and _positive_finite(record.get("duration_us"))
            for record in annotations.values()
        )
    )


def _summary_valid(summary, *, repeats, steps, cells):
    if not isinstance(summary, dict):
        return False
    raw = summary.get("raw_seconds")
    if (
        not isinstance(raw, list)
        or len(raw) != repeats
        or not all(_positive_finite(value) for value in raw)
    ):
        return False
    middle = median(raw)
    return all(
        math.isclose(
            float(summary.get(name, -1)), expected, rel_tol=1e-12, abs_tol=1e-15
        )
        for name, expected in (
            ("median_seconds", middle),
            ("steps_per_second", steps / middle),
            ("cells_per_second", cells * steps / middle),
        )
    )


def _candidate_valid(candidate):
    return (
        isinstance(candidate, dict)
        and set(candidate)
        == {"candidate_git_commit", "candidate_git_status", "manifest_sha256"}
        and _hex_string(candidate.get("candidate_git_commit"), 40)
        and candidate.get("candidate_git_status") == ""
        and _hex_string(candidate.get("manifest_sha256"), 64)
    )


def _decomposition_valid(decomposition, expected_shape):
    if not isinstance(decomposition, dict) or set(decomposition) != {
        "global_shape",
        "axis",
        "cut",
        "rank_costs",
        "device_weights",
        "communication_cells",
        "source_crossings",
        "identity",
        "axis_name",
    }:
        return False
    shape = decomposition["global_shape"]
    axis = decomposition["axis"]
    cut = decomposition["cut"]
    costs = decomposition["rank_costs"]
    weights = decomposition["device_weights"]
    if (
        shape != list(expected_shape)
        or type(axis) is not int
        or axis not in (0, 1, 2)
        or type(cut) is not int
        or not 1 < cut < shape[axis] - 1
        or not isinstance(costs, list)
        or len(costs) != 2
        or not all(_positive_finite(value) for value in costs)
        or not isinstance(weights, list)
        or len(weights) != 2
        or not all(_positive_finite(value) for value in weights)
        or not math.isclose(sum(weights), 1.0, rel_tol=1e-12, abs_tol=1e-12)
        or type(decomposition["communication_cells"]) is not int
        or decomposition["communication_cells"] <= 0
        or type(decomposition["source_crossings"]) is not int
        or decomposition["source_crossings"] < 0
        or decomposition["axis_name"] != "xyz"[axis]
    ):
        return False
    payload = (
        tuple(shape),
        axis,
        cut,
        tuple(round(value, 12) for value in weights),
    )
    identity = hashlib.sha256(repr(payload).encode()).hexdigest()
    return decomposition["identity"] == identity


def _imbalance_valid(record, repeats):
    if not isinstance(record, dict) or set(record) != {
        "rank_seconds_ratio_per_repeat",
        "rank_seconds_ratio_median",
        "rank_seconds_ratio_maximum",
        "material_cost_ratio",
        "device_adjusted_cost_ratio",
        "resident_storage_ratio",
        "peak_allocated_ratio",
        "halo_bytes_ratio",
    }:
        return False
    raw = record["rank_seconds_ratio_per_repeat"]
    return (
        isinstance(raw, list)
        and len(raw) == repeats
        and all(_positive_finite(value) and value >= 1.0 for value in raw)
        and math.isclose(
            record["rank_seconds_ratio_median"],
            median(raw),
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        and math.isclose(
            record["rank_seconds_ratio_maximum"],
            max(raw),
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        and all(
            _positive_finite(record[name]) and record[name] >= 1.0
            for name in (
                "material_cost_ratio",
                "device_adjusted_cost_ratio",
                "resident_storage_ratio",
                "peak_allocated_ratio",
                "halo_bytes_ratio",
            )
        )
    )


def _imbalance(rank_evidence, decomposition, rank_samples):
    timing_ratios = [_ratio(values) for values in rank_samples]
    costs = decomposition["rank_costs"]
    weights = decomposition["device_weights"]
    return {
        "rank_seconds_ratio_per_repeat": timing_ratios,
        "rank_seconds_ratio_median": median(timing_ratios),
        "rank_seconds_ratio_maximum": max(timing_ratios),
        "material_cost_ratio": _ratio(costs),
        "device_adjusted_cost_ratio": _ratio(
            [cost / weight for cost, weight in zip(costs, weights, strict=True)]
        ),
        "resident_storage_ratio": _ratio(
            [record["storage"]["resident_bytes"] for record in rank_evidence]
        ),
        "peak_allocated_ratio": _ratio(
            [record["memory"]["peak_allocated_bytes"] for record in rank_evidence]
        ),
        "halo_bytes_ratio": _ratio(
            [record["halo"]["bytes"] for record in rank_evidence]
        ),
    }


def _combine_worker_results(serial_worker, distributed_worker, subprocesses, args):
    spec = CASES[args.case]
    serial_cells = math.prod(spec["serial_size"])
    distributed_cells = math.prod(spec["distributed_size"])
    measurement = _measurement(args)
    serial = serial_worker.get("serial", {})
    distributed = distributed_worker.get("distributed", {})
    rank_evidence = distributed_worker.get("rank_evidence")
    profiles = distributed_worker.get("profiles")
    decomposition = distributed_worker.get("decomposition")
    candidate = distributed_worker.get("candidate_evidence")
    subprocess_contract = (
        isinstance(subprocesses, dict)
        and set(subprocesses) == {"serial", "distributed"}
        and all(
            _subprocess_record_valid(record, role, candidate)
            for role, record in subprocesses.items()
        )
        and subprocesses["serial"]["command"] != subprocesses["distributed"]["command"]
        and "--worker" in subprocesses["serial"]["command"]
        and "serial" in subprocesses["serial"]["command"]
        and "torch.distributed.run" in subprocesses["distributed"]["command"]
        and "distributed" in subprocesses["distributed"]["command"]
        and not any(
            Path(value).is_absolute()
            for record in subprocesses.values()
            for value in record["command"]
            if isinstance(value, str)
        )
        and len(
            {
                record[name]["path"]
                for record in subprocesses.values()
                for name in ("artifact", "stdout", "stderr")
            }
        )
        == 6
    )
    worker_contract = (
        isinstance(serial_worker, dict)
        and set(serial_worker)
        == {
            "schema_version",
            "kind",
            "candidate_evidence",
            "environment",
            "case",
            "size",
            "measurement",
            "serial",
        }
        and isinstance(distributed_worker, dict)
        and set(distributed_worker)
        == {
            "schema_version",
            "kind",
            "candidate_evidence",
            "environment",
            "case",
            "size",
            "measurement",
            "distributed",
            "decomposition",
            "rank_evidence",
            "profiles",
        }
        and serial_worker.get("schema_version") == SCHEMA_VERSION
        and serial_worker.get("kind") == WORKER_KINDS["serial"]
        and distributed_worker.get("schema_version") == SCHEMA_VERSION
        and distributed_worker.get("kind") == WORKER_KINDS["distributed"]
        and serial_worker.get("case") == distributed_worker.get("case") == args.case
        and serial_worker.get("size") == list(spec["serial_size"])
        and distributed_worker.get("size") == list(spec["distributed_size"])
        and serial_worker.get("measurement")
        == distributed_worker.get("measurement")
        == measurement
        and serial_worker.get("candidate_evidence") == candidate
        and _candidate_valid(candidate)
        and serial_worker.get("environment") == distributed_worker.get("environment")
    )
    rank_contract = (
        isinstance(rank_evidence, list)
        and len(rank_evidence) == 2
        and all(
            isinstance(record, dict)
            and set(record)
            == {
                "rank",
                "local_rank",
                "device",
                "peer_rank",
                "peer_access",
                "construction_seconds",
                "capture_seconds",
                "raw_seconds",
                "memory",
                "storage",
                "halo",
                "decomposition_identity",
                "local_field_shape",
                "global_offset",
            }
            for record in rank_evidence
        )
        and [record.get("rank") for record in rank_evidence] == [0, 1]
        and [record.get("local_rank") for record in rank_evidence] == [0, 1]
        and [record.get("device") for record in rank_evidence] == ["cuda:0", "cuda:1"]
        and all(type(record.get("peer_access")) is bool for record in rank_evidence)
        and all(
            record.get("peer_rank") == 1 - record["rank"]
            and _positive_finite(record.get("construction_seconds"))
            and _positive_finite(record.get("capture_seconds"))
            and isinstance(record.get("raw_seconds"), list)
            and len(record["raw_seconds"]) == args.repeats
            and all(_positive_finite(value) for value in record["raw_seconds"])
            and _hex_string(record.get("decomposition_identity"), 64)
            and isinstance(record.get("local_field_shape"), list)
            and len(record["local_field_shape"]) == 3
            and all(
                type(value) is int and value > 0
                for value in record["local_field_shape"]
            )
            and isinstance(record.get("global_offset"), list)
            and len(record["global_offset"]) == 3
            and all(
                type(value) is int and value >= 0 for value in record["global_offset"]
            )
            for record in rank_evidence
        )
        and all(_memory_valid(record.get("memory")) for record in rank_evidence)
        and all(_storage_valid(record.get("storage")) for record in rank_evidence)
        and all(
            _halo_valid(record.get("halo"), record["device"])
            for record in rank_evidence
        )
    )
    serial_auxiliary_contract = (
        isinstance(serial, dict)
        and set(serial)
        == {
            "raw_seconds",
            "median_seconds",
            "steps_per_second",
            "cells_per_second",
            "construction_seconds",
            "capture_seconds",
            "memory",
            "storage",
        }
        and _positive_finite(serial.get("construction_seconds"))
        and _positive_finite(serial.get("capture_seconds"))
        and _memory_valid(serial.get("memory"))
        and _storage_valid(serial.get("storage"))
    )
    distributed_auxiliary_contract = (
        isinstance(distributed, dict)
        and set(distributed)
        == {
            "raw_seconds",
            "median_seconds",
            "steps_per_second",
            "cells_per_second",
            "rank_raw_seconds",
            "construction_seconds",
            "capture_seconds",
            "peak_allocated_bytes_rank0",
            "halo_bytes_rank0",
            "storage_addresses_stable",
        }
        and rank_contract
        and math.isclose(
            distributed.get("construction_seconds", -1),
            max(record["construction_seconds"] for record in rank_evidence),
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        and math.isclose(
            distributed.get("capture_seconds", -1),
            max(record["capture_seconds"] for record in rank_evidence),
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        and distributed.get("peak_allocated_bytes_rank0")
        == rank_evidence[0]["memory"]["peak_allocated_bytes"]
        and distributed.get("halo_bytes_rank0") == rank_evidence[0]["halo"]["bytes"]
        and distributed.get("storage_addresses_stable") is True
        and all(record["storage"]["addresses_stable"] for record in rank_evidence)
    )
    rank_samples = distributed.get("rank_raw_seconds")
    timing_contract = (
        serial_auxiliary_contract
        and distributed_auxiliary_contract
        and _summary_valid(
            serial,
            repeats=args.repeats,
            steps=args.steps,
            cells=serial_cells,
        )
        and _summary_valid(
            distributed,
            repeats=args.repeats,
            steps=args.steps,
            cells=distributed_cells,
        )
        and isinstance(rank_samples, list)
        and len(rank_samples) == args.repeats
        and rank_contract
        and all(
            isinstance(values, list)
            and len(values) == 2
            and all(_positive_finite(value) for value in values)
            and math.isclose(
                reduced,
                max(values),
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            for values, reduced in zip(
                rank_samples,
                distributed.get("raw_seconds", ()),
                strict=True,
            )
        )
        and all(
            rank_evidence[rank]["raw_seconds"]
            == [values[rank] for values in rank_samples]
            for rank in range(2)
        )
    )
    decomposition_contract = (
        _decomposition_valid(decomposition, spec["distributed_size"])
        and rank_contract
        and all(
            record.get("decomposition_identity") == decomposition.get("identity")
            for record in rank_evidence
        )
        and decomposition.get("global_shape") == list(spec["distributed_size"])
        and rank_evidence[0].get("local_field_shape")
        == [
            decomposition["cut"] if axis == decomposition["axis"] else length
            for axis, length in enumerate(decomposition["global_shape"])
        ]
        and rank_evidence[1].get("local_field_shape")
        == [
            length - decomposition["cut"] if axis == decomposition["axis"] else length
            for axis, length in enumerate(decomposition["global_shape"])
        ]
        and rank_evidence[0].get("global_offset") == [0, 0, 0]
        and rank_evidence[1].get("global_offset")
        == [
            decomposition["cut"] if axis == decomposition["axis"] else 0
            for axis in range(3)
        ]
    )
    profile_contract = (
        isinstance(profiles, list)
        and len(profiles) == 2
        and all(_profile_valid(profile, args.profile_steps) for profile in profiles)
        and len({profile["trace"] for profile in profiles}) == 2
        and [PurePosixPath(profile["trace"]).name for profile in profiles]
        == ["gmes-two-gpu-0.json", "gmes-two-gpu-1.json"]
    )
    environment = distributed_worker.get("environment", {})
    devices = environment.get("devices") if isinstance(environment, dict) else None
    nccl = environment.get("nccl") if isinstance(environment, dict) else None
    environment_contract = (
        isinstance(environment, dict)
        and set(environment)
        == {
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
        }
        and host_contract_complete(environment.get("host_contract"))
        and all(
            isinstance(environment.get(name), str) and bool(environment[name])
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
        and type(environment.get("topology_command_status")) is int
        and environment["topology_command_status"] == 0
        and environment.get("topology_command") == ["nvidia-smi", "topo", "-m"]
        and isinstance(environment.get("topology"), str)
        and bool(environment["topology"].strip())
        and isinstance(devices, list)
        and len(devices) >= 2
        and [device.get("index") for device in devices[:2]] == [0, 1]
        and all(
            isinstance(device, dict)
            and set(device)
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
    ratio = (
        distributed["cells_per_second"] / (2.0 * serial["cells_per_second"])
        if spec["gate"] == "weak" and timing_contract
        else (
            serial["median_seconds"] / distributed["median_seconds"]
            if timing_contract
            else 0.0
        )
    )
    threshold = (
        0.8 if spec["gate"] == "weak" else 1.6 if spec["gate"] == "strong" else None
    )
    imbalance = (
        _imbalance(rank_evidence, decomposition, rank_samples)
        if rank_contract and timing_contract and decomposition_contract
        else None
    )
    checks = {
        "independent_subprocesses": subprocess_contract,
        "worker_contract": worker_contract,
        "environment_complete": environment_contract,
        "timing_reduction_complete": timing_contract,
        "rank_evidence_complete": rank_contract,
        "decomposition_complete": decomposition_contract,
        "imbalance_evidence_complete": _imbalance_valid(imbalance, args.repeats),
        "device_memory_bounded": (
            _memory_valid(serial.get("memory"))
            and rank_contract
            and all(record["memory"]["bounded"] for record in rank_evidence)
        ),
        "storage_addresses_stable": (
            _storage_valid(serial.get("storage"))
            and rank_contract
            and all(record["storage"]["addresses_stable"] for record in rank_evidence)
        ),
        "halos_device_resident": (
            rank_contract
            and all(
                record["halo"]["device_resident"] and record["halo"]["addresses_stable"]
                for record in rank_evidence
            )
        ),
        "peer_access_reported": (
            rank_contract
            and all(type(record["peer_access"]) is bool for record in rank_evidence)
        ),
        "steady_state_transfers_zero": (
            profile_contract
            and all(
                profile["host_to_device_events"] == 0
                and profile["device_to_host_events"] == 0
                for profile in profiles
            )
        ),
        "nccl_phases_complete": profile_contract,
        "interior_halo_overlap_observed": (
            profile_contract
            and all(profile["nccl_compute_overlap_us"] > 0 for profile in profiles)
        ),
        "performance_threshold": threshold is None or ratio >= threshold,
    }
    return {
        "candidate_evidence": candidate,
        "schema_version": SCHEMA_VERSION,
        "case": args.case,
        "gate": spec["gate"],
        "sizes": {
            "serial": list(spec["serial_size"]),
            "distributed": list(spec["distributed_size"]),
            "serial_cells": serial_cells,
            "distributed_cells": distributed_cells,
        },
        "measurement": measurement,
        "subprocesses": subprocesses,
        "serial": serial,
        "distributed": distributed,
        "decomposition": decomposition,
        "rank_evidence": rank_evidence,
        "imbalance": imbalance,
        "profiles": profiles,
        "environment": environment,
        "acceptance": {
            "ratio": ratio,
            "threshold": threshold,
            "checks": checks,
            "passed": all(checks.values()),
        },
    }


def _run_parent(args):
    if any(name in os.environ for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK")):
        raise ValueError(
            "launch the parent runner directly; it creates independent serial and "
            "two-rank subprocesses"
        )
    output = (args.output or Path(f"/tmp/gmes-{args.case}.json")).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.descriptor_root is None:
        if args.enforce:
            raise ValueError("--enforce requires an explicit --descriptor-root")
        args.descriptor_root = output.parent
    args.descriptor_root = args.descriptor_root.resolve(strict=True)
    output.relative_to(args.descriptor_root)
    args.trace_directory = args.trace_directory.resolve()
    args.trace_directory.relative_to(args.descriptor_root)
    args.trace_directory.mkdir(parents=True, exist_ok=True)
    commands = _worker_commands(args, output.parent)
    workers = {}
    subprocesses = {}
    for role in ("serial", "distributed"):
        command, worker_output = commands[role]
        workers[role], subprocesses[role] = _run_worker(
            role,
            command,
            worker_output,
            args.descriptor_root,
        )
    result = _combine_worker_results(
        workers["serial"],
        workers["distributed"],
        subprocesses,
        args,
    )
    _write_json(output, result)
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0 if not args.enforce or result["acceptance"]["passed"] else 2


def main():
    args = _arguments()
    if (
        args.warmup < 0
        or min(
            args.steps,
            args.repeats,
            args.profile_steps,
            args.threads_per_rank,
        )
        < 1
    ):
        raise ValueError("warmup, step, repeat, profile, and thread counts are invalid")
    if args.worker is not None:
        if args.worker_output is None:
            raise ValueError("--worker requires --worker-output")
        if args.worker == "serial":
            return _run_serial_worker(args)
        return _run_distributed_worker(args)
    if args.worker_output is not None:
        raise ValueError("--worker-output is internal and requires --worker")
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
