"""Fixed two-GPU strong/weak scaling and overlap evidence runner."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path
from statistics import median

import torch
import torch.distributed as dist

import gmes
from benchmarks.native_oracle import _coverage_geometry

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
    parser.add_argument("--enforce", action="store_true")
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


def _serial_samples(simulation, *, warmup, steps, repeats):
    checkpoint = simulation.checkpoint()
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


def _distributed_samples(simulation, *, warmup, steps, repeats):
    checkpoint = simulation.checkpoint()
    samples = []
    rank_samples = []
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
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=simulation.group)
        samples.append(float(elapsed.cpu()))
    return samples, rank_samples


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


def _trace_summary(path):
    trace = json.loads(path.read_text())
    kernels = [
        event
        for event in trace["traceEvents"]
        if event.get("ph") == "X" and event.get("cat") == "kernel"
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
    annotations = {}
    for event in trace["traceEvents"]:
        name = event.get("name", "")
        if event.get("ph") == "X" and name.startswith("gmes::halo_"):
            annotations[name] = annotations.get(name, 0.0) + float(event["dur"])
    return {
        "trace": str(path),
        "nccl_device_us": nccl_duration,
        "nccl_compute_overlap_us": overlap,
        "nccl_exposed_us": max(0.0, nccl_duration - overlap),
        "overlap_fraction": overlap / nccl_duration if nccl_duration else 0.0,
        "host_annotation_us": annotations,
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
    command = subprocess.run(
        ("nvidia-smi", "topo", "-m"),
        check=False,
        capture_output=True,
        text=True,
    )
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
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "nccl": torch.cuda.nccl.version(),
        "devices": inventory,
        "topology": command.stdout,
        "topology_command_status": command.returncode,
    }


def _summary(samples, *, steps, cells):
    middle = median(samples)
    return {
        "raw_seconds": samples,
        "median_seconds": middle,
        "steps_per_second": steps / middle,
        "cells_per_second": cells * steps / middle,
    }


def main():
    args = _arguments()
    if args.warmup < 0 or args.steps < 1 or args.repeats < 1:
        raise ValueError("warmup, steps, and repeats must be positive as applicable")
    launch = gmes.distributed_launch_from_environment()
    spec = CASES[args.case]
    distributed_size = spec["distributed_size"]
    construction_start = time.perf_counter()
    simulation = gmes.TorchDistributedSimulation(
        space=gmes.Cartesian(distributed_size, 1),
        geometry=_geometry(spec["geometry"], distributed_size),
        runtime=_runtime(
            f"cuda:{launch.local_rank}",
            threads=args.threads_per_rank,
            launch=launch,
        ),
    )
    construction = torch.tensor(
        [time.perf_counter() - construction_start],
        device=simulation.device,
        dtype=torch.float64,
    )
    dist.all_reduce(construction, op=dist.ReduceOp.MAX, group=simulation.group)
    capture_start = time.perf_counter()
    simulation.capture_cuda_graphs()
    capture = torch.tensor(
        [time.perf_counter() - capture_start],
        device=simulation.device,
        dtype=torch.float64,
    )
    dist.all_reduce(capture, op=dist.ReduceOp.MAX, group=simulation.group)
    addresses = simulation.buffer_addresses()

    serial_result = None
    if launch.rank == 0:
        serial_size = spec["serial_size"]
        start = time.perf_counter()
        serial = gmes.TorchSimulation(
            space=gmes.Cartesian(serial_size, 1),
            geometry=_geometry(spec["geometry"], serial_size),
            runtime=_runtime("cuda:0", threads=args.threads_per_rank),
        )
        serial_construction = time.perf_counter() - start
        start = time.perf_counter()
        serial.capture_cuda_graphs()
        serial_capture = time.perf_counter() - start
        torch.cuda.reset_peak_memory_stats(serial.device)
        serial_samples = _serial_samples(
            serial,
            warmup=args.warmup,
            steps=args.steps,
            repeats=args.repeats,
        )
        serial_result = _summary(
            serial_samples,
            steps=args.steps,
            cells=int(torch.tensor(serial_size).prod()),
        )
        serial_result.update(
            {
                "construction_seconds": serial_construction,
                "capture_seconds": serial_capture,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(serial.device),
            }
        )

    dist.barrier(group=simulation.group)
    torch.cuda.reset_peak_memory_stats(simulation.device)
    distributed_samples, rank_samples = _distributed_samples(
        simulation,
        warmup=args.warmup,
        steps=args.steps,
        repeats=args.repeats,
    )
    trace_path = _profile(
        simulation,
        steps=args.profile_steps,
        directory=args.trace_directory,
    )
    storage_stable = addresses == simulation.buffer_addresses()

    exit_status = 0
    if launch.rank == 0:
        serial_cells = int(torch.tensor(spec["serial_size"]).prod())
        distributed_cells = int(torch.tensor(distributed_size).prod())
        distributed_result = _summary(
            distributed_samples,
            steps=args.steps,
            cells=distributed_cells,
        )
        distributed_result.update(
            {
                "rank_raw_seconds": rank_samples,
                "construction_seconds": float(construction.cpu()),
                "capture_seconds": float(capture.cpu()),
                "peak_allocated_bytes_rank0": torch.cuda.max_memory_allocated(
                    simulation.device
                ),
                "halo_bytes_rank0": simulation.diagnostics()["halo_bytes"],
                "storage_addresses_stable": storage_stable,
            }
        )
        ratio = (
            distributed_result["cells_per_second"]
            / (2.0 * serial_result["cells_per_second"])
            if spec["gate"] == "weak"
            else serial_result["median_seconds"] / distributed_result["median_seconds"]
        )
        threshold = (
            0.8 if spec["gate"] == "weak" else 1.6 if spec["gate"] == "strong" else None
        )
        traces = [
            _trace_summary(args.trace_directory / f"gmes-two-gpu-{rank}.json")
            for rank in range(2)
        ]
        result = {
            "schema_version": 1,
            "case": args.case,
            "gate": spec["gate"],
            "sizes": {
                "serial": list(spec["serial_size"]),
                "distributed": list(distributed_size),
                "serial_cells": serial_cells,
                "distributed_cells": distributed_cells,
            },
            "measurement": {
                "warmup": args.warmup,
                "steps": args.steps,
                "repeats": args.repeats,
                "profile_steps": args.profile_steps,
                "threads_per_rank": args.threads_per_rank,
            },
            "serial": serial_result,
            "distributed": distributed_result,
            "decomposition": simulation.decomposition.metadata(),
            "profiles": traces,
            "environment": _environment(),
            "acceptance": {
                "ratio": ratio,
                "threshold": threshold,
                "passed": (
                    storage_stable and (threshold is None or ratio >= threshold)
                ),
            },
        }
        output = args.output or Path(f"/tmp/gmes-{args.case}.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, sort_keys=True))
        if args.enforce and not result["acceptance"]["passed"]:
            exit_status = 2

    status = torch.tensor([exit_status], device=simulation.device, dtype=torch.int32)
    dist.broadcast(status, src=0, group=simulation.group)
    gmes.TorchDistributedSimulation.close()
    return int(status.cpu())


if __name__ == "__main__":
    raise SystemExit(main())
