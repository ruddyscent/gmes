#!/usr/bin/env python3
"""Strict Torch compiler, runtime benchmark, and profiler evidence runner."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import subprocess
import time
from collections import Counter
from pathlib import Path
from statistics import median

import numpy as np
import torch
from torch.utils import benchmark as torch_benchmark

import gmes
from benchmarks.native_oracle import (
    _build_sources,
    _coverage_geometry,
    _heterogeneous_geometry,
    _mixed_geometry,
    find_case,
    load_manifest,
    material_from_name,
)
from benchmarks.torch_dm2 import build_case as build_dm2_case

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "native_oracle_workloads.json"
CPU_GATES = ("cpu-crossover-2d", "cpu-crossover-3d")
CUDA_GATES = (
    "cpu-crossover-2d",
    "cpu-crossover-3d",
    "cpu-large-2d",
    "cpu-large-3d",
    "single-gpu-2d",
    "single-gpu-3d",
)
POLICY_GATES = (
    "coverage-1-contiguous",
    "coverage-1-fragmented",
    "coverage-10-contiguous",
    "coverage-10-fragmented",
    "coverage-50-contiguous",
    "coverage-50-fragmented",
    "coverage-90-contiguous",
    "coverage-90-fragmented",
)
SPECIAL_CASES = ("all-material-2d", "all-material-3d")
COMPILE_VARIANTS = (
    ("default-no-graph", "default", False),
    ("default-explicit-graph", "default", True),
    ("reduce-overhead", "reduce-overhead", False),
    ("max-autotune", "max-autotune", False),
)


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def _percentile(values, value):
    return float(np.percentile(np.asarray(values, dtype=np.float64), value))


def _timing_summary(values, *, steps=1):
    values = [float(value) for value in values]
    middle = median(values)
    return {
        "raw_seconds": values,
        "median_seconds": middle,
        "p95_seconds": _percentile(values, 95),
        "steps_per_repeat": steps,
        "seconds_per_step": middle / steps,
        "steps_per_second": steps / middle if middle else None,
    }


def _counter_snapshot():
    counters = torch._dynamo.utils.counters
    graph_break = counters.get("graph_break", Counter())
    stats = counters.get("stats", Counter())
    frames = counters.get("frames", Counter())
    inductor = counters.get("inductor", Counter())
    return {
        "graph_breaks": int(sum(graph_break.values())),
        "unique_graphs": int(stats.get("unique_graphs", 0)),
        "calls_captured": int(stats.get("calls_captured", 0)),
        "frames_total": int(frames.get("total", 0)),
        "frames_ok": int(frames.get("ok", 0)),
        "fxgraph_cache_hit": int(inductor.get("fxgraph_cache_hit", 0)),
        "fxgraph_cache_miss": int(inductor.get("fxgraph_cache_miss", 0)),
    }


def _counter_delta(first, second):
    return {name: int(second[name] - first[name]) for name in first}


def _environment():
    cpu = subprocess.run(
        ("lscpu",),
        check=False,
        capture_output=True,
        text=True,
    )
    topology = subprocess.run(
        ("nvidia-smi", "topo", "-m"),
        check=False,
        capture_output=True,
        text=True,
    )
    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "memory_bytes": props.total_memory,
                    "capability": [props.major, props.minor],
                    "multiprocessors": props.multi_processor_count,
                }
            )
    affinity = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "devices": devices,
        "cpu_count": os.cpu_count(),
        "cpu_affinity": affinity,
        "cpu_topology": cpu.stdout,
        "gpu_topology": topology.stdout,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
    }


def _build_manifest_case(name, manifest):
    spec = find_case(manifest, name)
    space = gmes.Cartesian(tuple(spec["size"]), spec["resolution"])
    if spec["recipe"] == "mixed":
        geometry = _mixed_geometry(spec, gmes)
    elif spec["recipe"] == "coverage":
        geometry = _coverage_geometry(spec, gmes)
    elif spec["recipe"] == "heterogeneous":
        geometry = _heterogeneous_geometry(spec, gmes)
    else:
        material_name = spec["material"]
        geometry = [gmes.DefaultMedium(material_from_name("dielectric", gmes))]
        if material_name in {"upml", "cpml"}:
            geometry.append(
                gmes.Shell(
                    material_from_name(material_name, gmes),
                    thickness=spec.get("pml_thickness", 1),
                )
            )
        elif material_name != "dielectric":
            size = tuple(max(float(value), 1.0) for value in spec["size"])
            geometry.append(
                gmes.Block(
                    material_from_name(material_name, gmes),
                    center=(0, 0, 0),
                    size=size,
                )
            )
    sources = _build_sources(spec, gmes)
    bloch = (0.07, 0.11, 0.13) if spec.get("complex") else None
    return spec, space, geometry, sources, bloch


def _build_case(name, manifest):
    if name in SPECIAL_CASES:
        space, geometry = build_dm2_case(name, gmes)
        spec = {
            "name": name,
            "recipe": "all-material",
            "size": list(space.size),
            "resolution": space.resolution,
            "complex": False,
        }
        return spec, space, geometry, (), None
    return _build_manifest_case(name, manifest)


def _initialize_fields(simulation, seed, scale):
    rng = np.random.default_rng(seed)
    fields = {
        name: rng.normal(size=tuple(field.shape)) * scale
        for name, field in simulation.state.fields().items()
    }
    start = time.perf_counter()
    simulation.load_host_fields(fields)
    _synchronize(simulation.device)
    return time.perf_counter() - start


def _checksum(simulation):
    value = torch.zeros((), dtype=torch.float64, device=simulation.device)
    for field in simulation.state.fields().values():
        value.add_(field.detach().to(dtype=torch.float64).abs().sum())
    return float(value.cpu())


def _timer_samples(simulation, steps, repeats, threads, checkpoint):
    timer = torch_benchmark.Timer(
        stmt="simulation.advance(steps)",
        globals={"simulation": simulation, "steps": steps},
        num_threads=threads,
        label="GMES Torch advance",
        sub_label=f"{steps} step(s)",
        description=str(simulation.device),
    )
    samples = []
    for _ in range(repeats):
        simulation.load_checkpoint(checkpoint)
        measurement = timer.timeit(1)
        samples.append(float(measurement.median))
    _synchronize(simulation.device)
    return samples


def _trace_summary(path):
    trace = json.loads(path.read_text())
    events = trace.get("traceEvents", ())
    kernels = 0
    h2d = 0
    d2h = 0
    device_copies = 0
    for event in events:
        if event.get("ph") != "X":
            continue
        name = event.get("name", "").lower()
        category = event.get("cat", "").lower()
        if category == "kernel":
            kernels += 1
        if "memcpy" in category or "memcpy" in name:
            device_copies += 1
            if "htod" in name or "host to device" in name:
                h2d += 1
            if "dtoh" in name or "device to host" in name:
                d2h += 1
    return {
        "chrome_trace": str(path),
        "kernel_launches": kernels,
        "device_copy_events": device_copies,
        "host_to_device_events": h2d,
        "device_to_host_events": d2h,
    }


def _profile(simulation, steps, path):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if simulation.device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(
        activities=activities,
        profile_memory=True,
        record_shapes=False,
    ) as profile:
        simulation.advance(steps)
        _synchronize(simulation.device)
    profile.export_chrome_trace(str(path))
    events = profile.key_averages()
    result = _trace_summary(path)
    result.update(
        {
            "profile_steps": steps,
            "positive_allocation_events": sum(
                event.count
                for event in events
                if event.self_cpu_memory_usage > 0
                or getattr(event, "self_device_memory_usage", 0) > 0
            ),
            "top_operations": [
                {
                    "name": event.key,
                    "count": event.count,
                    "self_cpu_time_us": event.self_cpu_time_total,
                }
                for event in sorted(
                    events,
                    key=lambda event: event.self_cpu_time_total,
                    reverse=True,
                )[:15]
            ],
        }
    )
    return result


def _native_gate(reference, name, threads, candidate):
    if reference is None:
        return None
    data = json.loads(reference.read_text())
    matches = [
        item
        for item in data["samples"]
        if item["workload"]["name"] == name and int(item["threads"]) == threads
    ]
    if not matches:
        return None
    sample = matches[0]["measurements"]["advance"]
    reference_seconds = sample["median_seconds"] / sample["steps_per_repeat"]
    candidate_seconds = candidate["measurements"]["advance"]["seconds_per_step"]
    ratio = candidate_seconds / reference_seconds
    return {
        "reference_observer_tag": data["observer_tag"],
        "reference_observer_commit": data["observer_commit"],
        "reference_seconds_per_step": reference_seconds,
        "candidate_seconds_per_step": candidate_seconds,
        "torch_to_native_ratio": ratio,
        "within_five_percent": ratio <= 1.05,
    }


def run_case(
    name,
    *,
    device,
    precision,
    compile_mode,
    capture_graphs,
    execution_policy,
    threads,
    interop_threads,
    warmup,
    steps,
    repeats,
    profile_steps,
    trace_directory,
    manifest,
):
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    spec, space, geometry, sources, bloch = _build_case(name, manifest)
    runtime = gmes.TorchRuntimeConfig(
        device=device,
        precision=precision,
        compile_policy="compile",
        compile_mode=compile_mode,
        cpu_threads=threads,
        cpu_interop_threads=interop_threads,
        execution_policy=execution_policy,
    )
    benchmark_device = torch.device(device)
    if benchmark_device.type == "cuda":
        with torch.cuda.device(benchmark_device):
            torch.cuda.reset_peak_memory_stats()
    construction_start = time.perf_counter()
    simulation = gmes.TorchSimulation(
        space=space,
        geometry=geometry,
        sources=sources,
        bloch=bloch,
        runtime=runtime,
    )
    construction_seconds = time.perf_counter() - construction_start
    transfer_seconds = _initialize_fields(
        simulation,
        manifest["reference"]["seed"],
        manifest["reference"]["field_scale"],
    )
    checkpoint = simulation.checkpoint()
    initial_checksum = _checksum(simulation)

    start = time.perf_counter()
    simulation.advance(1)
    _synchronize(simulation.device)
    cold_compile_seconds = time.perf_counter() - start
    after_cold = _counter_snapshot()

    simulation.load_checkpoint(checkpoint)
    start = time.perf_counter()
    simulation.advance(1)
    _synchronize(simulation.device)
    cached_compile_seconds = time.perf_counter() - start

    simulation.load_checkpoint(checkpoint)
    simulation.advance(warmup)
    _synchronize(simulation.device)
    capture_seconds = 0.0
    if capture_graphs:
        start = time.perf_counter()
        simulation.capture_cuda_graphs()
        capture_seconds = time.perf_counter() - start
    after_warmup = _counter_snapshot()
    addresses = simulation.buffer_addresses()
    allocated_before = (
        torch.cuda.memory_allocated(simulation.device)
        if simulation.device.type == "cuda"
        else None
    )
    one_step = _timer_samples(
        simulation,
        1,
        repeats,
        threads,
        checkpoint,
    )
    advance = _timer_samples(
        simulation,
        steps,
        repeats,
        threads,
        checkpoint,
    )
    _synchronize(simulation.device)
    allocated_after = (
        torch.cuda.memory_allocated(simulation.device)
        if simulation.device.type == "cuda"
        else None
    )
    final_checksum = _checksum(simulation)
    trace_path = trace_directory / (
        f"{name}-{str(device).replace(':', '-')}-{compile_mode}-"
        f"{'graph' if capture_graphs else 'no-graph'}.json"
    )
    profiler = _profile(simulation, profile_steps, trace_path)
    after_steady = _counter_snapshot()
    counter_growth = _counter_delta(after_warmup, after_steady)
    storage_stable = addresses == simulation.buffer_addresses()
    memory_growth = (
        allocated_after - allocated_before if allocated_before is not None else None
    )
    compiler_clean = (
        after_steady["graph_breaks"] == 0
        and counter_growth["unique_graphs"] == 0
        and counter_growth["frames_total"] == 0
    )
    transfers_clean = (
        profiler["host_to_device_events"] == 0
        and profiler["device_to_host_events"] == 0
    )
    allocations_clean = (
        simulation.device.type != "cpu" or profiler["positive_allocation_events"] == 0
    )
    memory_bounded = memory_growth is None or memory_growth <= 1024 * 1024
    step_count = int(simulation.state.step_count.cpu())
    result = {
        "schema_version": 1,
        "backend": "torch",
        "workload": spec,
        "runtime": {
            "device": str(simulation.device),
            "precision": precision,
            "compile_policy": "compile",
            "compile_mode": compile_mode,
            "explicit_cuda_graphs": capture_graphs,
            "execution_policy": execution_policy,
            "threads": threads,
            "interop_threads": interop_threads,
            "compile_cache_key": simulation.compile_cache_key,
        },
        "measurements": {
            "construction": _timing_summary([construction_seconds]),
            "host_to_device_transfer": _timing_summary([transfer_seconds]),
            "cold_compile_and_step": _timing_summary([cold_compile_seconds]),
            "cached_compile_and_step": _timing_summary([cached_compile_seconds]),
            "cuda_graph_capture": _timing_summary([capture_seconds]),
            "one_step": _timing_summary(one_step),
            "advance": _timing_summary(advance, steps=steps),
        },
        "compiler": {
            "after_cold": after_cold,
            "after_warmup": after_warmup,
            "after_steady": after_steady,
            "steady_state_delta": counter_growth,
            "fullgraph_clean": compiler_clean,
        },
        "memory": {
            "peak_rss_bytes": _rss_bytes(),
            "cuda_allocated_before_bytes": allocated_before,
            "cuda_allocated_after_bytes": allocated_after,
            "cuda_allocated_growth_bytes": memory_growth,
            "cuda_peak_allocated_bytes": (
                torch.cuda.max_memory_allocated(simulation.device)
                if simulation.device.type == "cuda"
                else None
            ),
            "cuda_peak_reserved_bytes": (
                torch.cuda.max_memory_reserved(simulation.device)
                if simulation.device.type == "cuda"
                else None
            ),
            "storage_addresses_stable": storage_stable,
            "bounded": memory_bounded,
        },
        "profiler": profiler,
        "state_progress": {
            "initial_checksum": initial_checksum,
            "final_checksum": final_checksum,
            "changed": initial_checksum != final_checksum,
            "step_count": step_count,
        },
        "diagnostics": simulation.diagnostics(),
        "acceptance": {
            "compiler_clean": compiler_clean,
            "steady_state_transfers_zero": transfers_clean,
            "storage_stable": storage_stable,
            "memory_bounded": memory_bounded,
            "recurring_allocations_zero": allocations_clean,
            "state_progressed": initial_checksum != final_checksum and step_count > 0,
        },
    }
    result["acceptance"]["passed"] = all(result["acceptance"].values())
    return result


def _variant_matrix(args, name, manifest):
    results = {}
    for variant, compile_mode, capture_graphs in COMPILE_VARIANTS:
        results[variant] = run_case(
            name,
            device=args.device,
            precision=args.precision,
            compile_mode=compile_mode,
            capture_graphs=capture_graphs,
            execution_policy=args.policy,
            threads=args.threads,
            interop_threads=args.interop_threads,
            warmup=args.warmup,
            steps=args.steps,
            repeats=args.repeats,
            profile_steps=args.profile_steps,
            trace_directory=args.trace_directory,
            manifest=manifest,
        )
    fastest = min(
        results,
        key=lambda key: results[key]["measurements"]["advance"]["seconds_per_step"],
    )
    return {
        "case": name,
        "fastest_variant": fastest,
        "results": results,
        "passed": all(item["acceptance"]["passed"] for item in results.values()),
    }


def _policy_matrix(args, name, manifest):
    results = {}
    for policy in ("auto", "dense", "compact", "tiled"):
        results[policy] = run_case(
            name,
            device=args.device,
            precision=args.precision,
            compile_mode=args.compile_mode,
            capture_graphs=args.capture_graphs,
            execution_policy=policy,
            threads=args.threads,
            interop_threads=args.interop_threads,
            warmup=args.warmup,
            steps=args.steps,
            repeats=args.repeats,
            profile_steps=args.profile_steps,
            trace_directory=args.trace_directory,
            manifest=manifest,
        )
    fastest = min(
        results[policy]["measurements"]["advance"]["seconds_per_step"]
        for policy in ("dense", "compact", "tiled")
    )
    automatic = results["auto"]["measurements"]["advance"]["seconds_per_step"]
    ratio = automatic / fastest
    return {
        "case": name,
        "auto_to_fastest_forced_ratio": ratio,
        "within_ten_percent": ratio <= 1.10,
        "results": results,
        "passed": ratio <= 1.10
        and all(item["acceptance"]["passed"] for item in results.values()),
    }


def _arguments():
    manifest = load_manifest(MANIFEST)
    names = tuple(item["name"] for item in manifest["benchmarks"])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=names + SPECIAL_CASES + ("cpu-gates", "cuda-gates", "policy-gates"),
        default="cpu-crossover-2d",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--precision",
        choices=("float32", "float64"),
        default="float64",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune", "matrix"),
        default="default",
    )
    parser.add_argument(
        "--policy",
        choices=("auto", "dense", "compact", "tiled", "matrix"),
        default="auto",
    )
    parser.add_argument("--capture-graphs", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--profile-steps", type=int, default=5)
    parser.add_argument(
        "--trace-directory",
        type=Path,
        default=Path("/tmp/gmes-torch-tuning-traces"),
    )
    parser.add_argument("--native-summary", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--enforce", action="store_true")
    return parser.parse_args(), manifest


def main():
    args, manifest = _arguments()
    if (
        min(
            args.threads,
            args.interop_threads,
            args.steps,
            args.repeats,
            args.profile_steps,
        )
        < 1
        or args.warmup < 0
    ):
        raise ValueError("thread, step, repeat, and profile counts must be positive")
    if args.capture_graphs and torch.device(args.device).type != "cuda":
        raise ValueError("--capture-graphs requires CUDA")
    if args.compile_mode == "matrix" and torch.device(args.device).type != "cuda":
        raise ValueError("compile mode matrix requires CUDA")
    if args.compile_mode == "matrix" and args.policy == "matrix":
        raise ValueError("run compile mode and planner policy matrices separately")

    torch.set_num_threads(args.threads)
    if torch.get_num_interop_threads() != args.interop_threads:
        torch.set_num_interop_threads(args.interop_threads)

    if args.case == "cpu-gates":
        cases = CPU_GATES
    elif args.case == "cuda-gates":
        cases = CUDA_GATES
    elif args.case == "policy-gates":
        cases = POLICY_GATES
    else:
        cases = (args.case,)

    results = []
    for name in cases:
        if args.compile_mode == "matrix":
            result = _variant_matrix(args, name, manifest)
        elif args.policy == "matrix":
            result = _policy_matrix(args, name, manifest)
        else:
            result = run_case(
                name,
                device=args.device,
                precision=args.precision,
                compile_mode=args.compile_mode,
                capture_graphs=args.capture_graphs,
                execution_policy=args.policy,
                threads=args.threads,
                interop_threads=args.interop_threads,
                warmup=args.warmup,
                steps=args.steps,
                repeats=args.repeats,
                profile_steps=args.profile_steps,
                trace_directory=args.trace_directory,
                manifest=manifest,
            )
        if (
            args.native_summary is not None
            and args.compile_mode != "matrix"
            and args.policy != "matrix"
        ):
            result["native_gate"] = _native_gate(
                args.native_summary,
                name,
                args.threads,
                result,
            )
        results.append(result)

    native_gates = [
        result.get("native_gate")
        for result in results
        if result.get("native_gate") is not None
    ]
    geometric_ratio = (
        math.exp(
            sum(math.log(item["torch_to_native_ratio"]) for item in native_gates)
            / len(native_gates)
        )
        if native_gates
        else None
    )
    passed = all(
        result.get("passed", result.get("acceptance", {}).get("passed", False))
        for result in results
    )
    if native_gates:
        passed = (
            passed
            and all(item["within_five_percent"] for item in native_gates)
            and geometric_ratio <= 1.05
        )
    output = {
        "schema_version": 1,
        "environment": _environment(),
        "cases": results,
        "suite_acceptance": {
            "native_geometric_mean_ratio": geometric_ratio,
            "native_individual_within_five_percent": (
                all(item["within_five_percent"] for item in native_gates)
                if native_gates
                else None
            ),
            "passed": passed,
        },
    }
    rendered = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if not args.enforce or passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
