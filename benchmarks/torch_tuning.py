#!/usr/bin/env python3
"""Strict Torch compiler, runtime benchmark, and profiler evidence runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import time
from collections import Counter
from pathlib import Path
from statistics import median, pstdev

import numpy as np
import torch
from torch.utils import benchmark as torch_benchmark

import gmes
from benchmarks.native_oracle import (
    FIELD_INITIALIZER,
    _build_sources,
    _coverage_geometry,
    _heterogeneous_geometry,
    _mixed_geometry,
    find_case,
    initial_field_values,
    load_manifest,
    material_from_name,
)
from benchmarks.torch_dm2 import build_case as build_dm2_case

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "native_oracle_workloads.json"
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
EVIDENCE_CONTRACT_ID = "torch-cpu-acceptance-v3"
RUNTIME_ACCEPTANCE_KEYS = (
    "compiler_clean",
    "compiled_hot_path_complete",
    "external_indexed_writes_only_sources",
    "steady_state_transfers_zero",
    "storage_stable",
    "memory_bounded",
    "recurring_allocations_zero",
    "measurement_contract_matches_manifest",
    "state_progressed",
    "passed",
)
RUNNER_INPUTS = (
    "benchmarks/native_oracle.py",
    "benchmarks/torch_dm2.py",
    "benchmarks/torch_tuning.py",
)
SOLVER_INPUTS = (
    "gmes/torch_dm2.py",
    "gmes/torch_fdtd.py",
    "gmes/torch_dispersive.py",
    "gmes/torch_distributed.py",
    "gmes/torch_plan.py",
    "gmes/torch_source.py",
)


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def _current_rss_bytes():
    """Return current resident memory on supported CPU acceptance hosts."""
    system = platform.system()
    if system == "Linux":
        try:
            fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
            return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
        except (IndexError, OSError, ValueError):
            return None
    if system == "Darwin":
        value = _command_text("ps", "-o", "rss=", "-p", str(os.getpid()))
        try:
            return int(value) * 1024 if value is not None else None
        except ValueError:
            return None
    return None


def _cpu_memory_probe(simulation, checkpoint, steps):
    """Measure current RSS across a compiled, post-warmup CPU advance."""
    probe_steps = max(1, int(steps))
    simulation.load_checkpoint(checkpoint)
    before = _current_rss_bytes()
    try:
        simulation.advance(probe_steps)
        _synchronize(simulation.device)
        after = _current_rss_bytes()
    finally:
        simulation.load_checkpoint(checkpoint)
    return {
        "probe_steps": probe_steps,
        "before_bytes": before,
        "after_bytes": after,
        "growth_bytes": (
            after - before if before is not None and after is not None else None
        ),
    }


def _memory_growth_bounded(device, cuda_growth, cpu_rss_growth):
    if device.type == "cuda":
        return cuda_growth is None or cuda_growth <= 1024 * 1024
    if device.type == "cpu":
        return cpu_rss_growth is not None and cpu_rss_growth <= 1024 * 1024
    return False


def _percentile(values, value):
    return float(np.percentile(np.asarray(values, dtype=np.float64), value))


def _timing_summary(values, *, steps=1):
    values = [float(value) for value in values]
    middle = median(values)
    absolute_deviations = [abs(value - middle) for value in values]
    return {
        "raw_seconds": values,
        "median_seconds": middle,
        "p95_seconds": _percentile(values, 95),
        "population_stdev_seconds": pstdev(values),
        "relative_mad": median(absolute_deviations) / middle if middle else 0.0,
        "repetitions": len(values),
        "steps_per_repeat": steps,
        "seconds_per_step": middle / steps,
        "steps_per_second": steps / middle if middle else None,
    }


def _cpu_gate_cases(manifest):
    return tuple(manifest["performance_gates"]["cpu_acceptance"]["cases"])


def _files_sha256(paths):
    digest = hashlib.sha256()
    for relative in sorted(paths):
        path = ROOT / relative
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _current_evidence(manifest):
    return {
        "evidence_contract_id": EVIDENCE_CONTRACT_ID,
        "cpu_contract_id": manifest["performance_gates"]["cpu_acceptance"][
            "contract_id"
        ],
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "runner_sha256": _files_sha256(RUNNER_INPUTS),
        "solver_sha256": _files_sha256(SOLVER_INPUTS),
        "solver_abi": gmes.torch_fdtd.TORCH_SOLVER_ABI,
        "candidate_git_commit": _command_text("git", "rev-parse", "HEAD"),
        "candidate_git_status": _command_text(
            "git", "status", "--short", "--untracked-files=normal"
        ),
    }


def _command_text(*command):
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _cpu_contract_environment():
    affinity = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    parsable = _command_text("lscpu", "-p=CPU,CORE,SOCKET")
    rows = []
    if parsable:
        for line in parsable.splitlines():
            if not line or line.startswith("#"):
                continue
            cpu, core, socket = (int(value) for value in line.split(",")[:3])
            if affinity is None or cpu in affinity:
                rows.append((cpu, core, socket))
    physical = len({(core, socket) for _cpu, core, socket in rows}) or None
    if physical is None:
        value = _command_text("sysctl", "-n", "hw.physicalcpu")
        try:
            physical = int(value.strip()) if value else None
        except ValueError:
            physical = None
    return {
        "cpu_affinity": affinity,
        "cpu_count_physical_affinity": physical,
        "cpu_topology": _command_text("lscpu", "-p=CORE,SOCKET"),
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
    cpu = _command_text("lscpu")
    topology = _command_text("nvidia-smi", "topo", "-m")
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
    cpu_contract = _cpu_contract_environment()
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "devices": devices,
        "cpu_count": os.cpu_count(),
        **cpu_contract,
        "cpu_model": cpu,
        "gpu_topology": topology,
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
    fields = initial_field_values(
        simulation.plan.shapes,
        seed,
        scale,
        complex_fields=simulation.state.paired_real,
    )
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
    """Collect exploratory Timer samples; timeit adds two hidden warmup calls."""
    timer = torch_benchmark.Timer(
        stmt="simulation.advance(steps)",
        setup="simulation.load_checkpoint(checkpoint)",
        globals={
            "simulation": simulation,
            "steps": steps,
            "checkpoint": checkpoint,
        },
        num_threads=threads,
        label="GMES Torch advance",
        sub_label=f"{steps} step(s)",
        description=str(simulation.device),
    )
    samples = [float(timer.timeit(1).median) for _ in range(repeats)]
    _synchronize(simulation.device)
    return samples


def _perf_counter_samples(
    simulation, steps, repeats, initial_checkpoint, warmup
):
    """Measure repeats after independently restoring and warming initial state."""
    samples = []
    for _ in range(repeats):
        simulation.load_checkpoint(initial_checkpoint)
        simulation.advance(warmup)
        _synchronize(simulation.device)
        start = time.perf_counter()
        simulation.advance(steps)
        _synchronize(simulation.device)
        samples.append(time.perf_counter() - start)
    return samples


def _state_change_summary(first, second):
    if set(first) != set(second):
        raise ValueError("state checkpoint keys changed during timed execution")
    changed = {
        name for name in first if not torch.equal(first[name], second[name])
    }
    field_names = {name.lower() for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
    pml_names = {name for name in first if name.startswith("pml_")}
    dispersive_names = {name for name in first if name.startswith("bucket_")}
    dm2_names = {name for name in first if name.startswith("dm2_buckets.")}
    return {
        "changed_buffers": sorted(changed),
        "fields_changed": sorted(changed & field_names),
        "all_fields_changed": field_names <= changed,
        "pml_state_changed": bool(changed & pml_names),
        "dispersive_state_changed": bool(changed & dispersive_names),
        "dm2_state_changed": bool(changed & dm2_names),
    }


def _trace_filename(
    name,
    *,
    device,
    precision,
    compile_mode,
    capture_graphs,
    execution_policy,
    threads,
    interop_threads,
    experimental_dispersive_grouping=False,
):
    values = (
        name,
        str(device),
        precision,
        compile_mode,
        "graph" if capture_graphs else "no-graph",
        execution_policy,
        f"threads-{threads}",
        f"interop-{interop_threads}",
    )
    if experimental_dispersive_grouping:
        values += ("exact-schema-dispersive",)
    return "__".join(
        "".join(character if character.isalnum() else "-" for character in value)
        for value in values
    ) + ".json"


def _trace_summary(path):
    trace_bytes = path.read_bytes()
    trace = json.loads(trace_bytes)
    events = trace.get("traceEvents", ())
    kernels = 0
    h2d = 0
    d2h = 0
    device_copies = 0
    allocation_events = 0
    allocated_bytes = 0
    freed_bytes = 0
    max_allocation_bytes = 0
    allocation_size_histogram = Counter()
    compiled_regions = Counter()
    compiled_intervals = []
    cuda_graph_launches = 0
    for event in events:
        if event.get("name") == "[memory]":
            size = int(event.get("args", {}).get("Bytes", 0))
            if size > 0:
                allocation_events += 1
                allocated_bytes += size
                max_allocation_bytes = max(max_allocation_bytes, size)
                allocation_size_histogram[size] += 1
            elif size < 0:
                freed_bytes -= size
        if event.get("ph") != "X":
            continue
        name = event.get("name", "").lower()
        category = event.get("cat", "").lower()
        if name.startswith("torch-compiled region:"):
            compiled_regions[event["name"]] += 1
            if all(key in event for key in ("ts", "dur", "pid", "tid")):
                compiled_intervals.append(
                    (
                        event["pid"],
                        event["tid"],
                        float(event["ts"]),
                        float(event["ts"]) + float(event["dur"]),
                    )
                )
        if name == "cudagraphlaunch":
            cuda_graph_launches += 1
        if category == "kernel":
            kernels += 1
        if "memcpy" in category or "memcpy" in name:
            device_copies += 1
            if "htod" in name or "host to device" in name:
                h2d += 1
            if "dtoh" in name or "device to host" in name:
                d2h += 1
    indexed_writes_outside_regions = Counter()
    for event in events:
        if event.get("ph") != "X" or event.get("name") not in {
            "aten::index_add_",
            "aten::index_copy_",
            "aten::index_put_",
        }:
            continue
        timestamp = float(event.get("ts", -1))
        inside = any(
            event.get("pid") == pid
            and event.get("tid") == tid
            and start <= timestamp < stop
            for pid, tid, start, stop in compiled_intervals
        )
        if not inside:
            indexed_writes_outside_regions[event["name"]] += 1
    return {
        "chrome_trace": str(path),
        "chrome_trace_size_bytes": len(trace_bytes),
        "chrome_trace_sha256": hashlib.sha256(trace_bytes).hexdigest(),
        "kernel_launches": kernels,
        "device_copy_events": device_copies,
        "host_to_device_events": h2d,
        "device_to_host_events": d2h,
        "positive_allocation_events": allocation_events,
        "allocated_bytes": allocated_bytes,
        "freed_bytes": freed_bytes,
        "allocation_net_bytes": allocated_bytes - freed_bytes,
        "max_allocation_bytes": max_allocation_bytes,
        "allocation_size_histogram": {
            str(size): count
            for size, count in sorted(allocation_size_histogram.items())
        },
        "compiled_region_events": sum(compiled_regions.values()),
        "compiled_region_names": dict(sorted(compiled_regions.items())),
        "cuda_graph_launches": cuda_graph_launches,
        "indexed_write_operations_outside_compiled_regions": sum(
            indexed_writes_outside_regions.values()
        ),
        "indexed_write_names_outside_compiled_regions": dict(
            sorted(indexed_writes_outside_regions.items())
        ),
    }


def _recurring_allocations_zero(device, profiler):
    if device.type != "cpu":
        return True
    return all(
        profiler.get(key) == 0
        for key in (
            "positive_allocation_events",
            "allocated_bytes",
            "positive_allocation_operations",
        )
    )


def _source_index_operations_per_step(simulation):
    operations = Counter()
    if not simulation._fused_source_updates:
        for batch in simulation.sources.batches:
            point_source = False
            for prefix, operation in (
                ("additive", "aten::index_add_"),
                ("overwrite", "aten::index_copy_"),
            ):
                targets = getattr(batch, f"{prefix}_targets", None)
                if targets is not None:
                    point_source = True
                    operations[operation] += int(targets.numel() > 0)
            if not point_source and hasattr(batch, "targets"):
                operations["aten::index_add_"] += int(batch.targets.numel() > 0)
    for auxiliary in simulation.sources.auxiliaries:
        operations.update(_source_index_operations_per_step(auxiliary))
    return operations


def _compiled_local_simulation_count(simulation):
    count = int(
        simulation._electric_half is not None
        and simulation._magnetic_half is not None
    )
    return count + sum(
        _compiled_local_simulation_count(auxiliary)
        for auxiliary in simulation.sources.auxiliaries
    )


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
            "positive_allocation_operations": sum(
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


def _native_gate(reference, name, threads, candidate, manifest):
    if reference is None:
        return None
    reference_bytes = reference.read_bytes()
    reference_sha256 = hashlib.sha256(reference_bytes).hexdigest()
    data = json.loads(reference_bytes)
    matches = [
        item
        for item in data["samples"]
        if item["workload"]["name"] == name and int(item["threads"]) == threads
    ]
    errors = []
    if not matches:
        return {
            "comparison_valid": False,
            "contract_errors": [
                f"native summary has no {name!r} sample at {threads} thread(s)"
            ],
            "reference_seconds_per_step": None,
            "candidate_seconds_per_step": candidate["measurements"]["advance"][
                "seconds_per_step"
            ],
            "torch_to_native_ratio": None,
            "within_five_percent": False,
        }
    if len(matches) != 1:
        errors.append("native summary has duplicate workload/thread samples")
    sample = matches[0]["measurements"]["advance"]
    candidate_sample = candidate["measurements"]["advance"]
    contract = candidate["benchmark_contract"]
    expected = manifest["reference"]
    acceptance = manifest["performance_gates"]["cpu_acceptance"]
    expected_native_contract = {
        "initializer": expected["field_initializer"],
        "seed": expected["seed"],
        "field_scale": expected["field_scale"],
        "warmup_steps": expected["performance_warmup_steps"],
        "steps_per_repeat": expected["performance_steps_per_repeat"],
        "repetitions": expected["performance_repetitions"],
        "timer": "time.perf_counter",
        "sample_start": "independently-rebuilt-post-warmup-state",
    }
    expected_candidate_contract = {
        "initializer": expected["field_initializer"],
        "seed": expected["seed"],
        "field_scale": expected["field_scale"],
        "warmup_steps": expected["performance_warmup_steps"],
        "steps_per_repeat": expected["performance_steps_per_repeat"],
        "repetitions": expected["performance_repetitions"],
        "profile_steps": expected["performance_profile_steps"],
        "timer": "time.perf_counter",
        "sample_start": "independently-restored-pre-warmup-state",
    }
    native_contract = matches[0].get(
        "benchmark_contract", data.get("benchmark_contract")
    )
    if native_contract is None:
        contract_provenance = "exact SHA-256-pinned #115 artifact record"
        native_contract = expected_native_contract
    else:
        contract_provenance = "exact SHA-256-pinned embedded contract"
    checks = (
        (
            data.get("observer_tag") == expected["observer_tag"],
            "native observer tag does not match the frozen manifest",
        ),
        (
            data.get("observer_commit") == expected["observer_commit"]
            and data.get("environment", {}).get("git_commit")
            == expected["observer_commit"]
            and not data.get("environment", {}).get("git_status"),
            "native observer commit or checkout cleanliness does not match",
        ),
        (
            data.get("physics_reference") == expected["tag"],
            "native physics reference does not match the frozen manifest",
        ),
        (
            matches[0]["workload"] == candidate["workload"],
            "native and Torch workload specifications differ",
        ),
        (
            int(matches[0]["openmp_threads"]) == threads,
            "native OpenMP thread count differs from Torch intra-op threads",
        ),
        (
            sample.get("steps_per_repeat")
            == expected["performance_steps_per_repeat"]
            == contract["steps_per_repeat"],
            "steps per repeat do not match the frozen performance contract",
        ),
        (
            sample.get("repetitions")
            == expected["performance_repetitions"]
            == contract["repetitions"],
            "replicate count does not match the frozen performance contract",
        ),
        (
            len(sample.get("raw_seconds", ()))
            == expected["performance_repetitions"],
            "native raw replicate count does not match the frozen contract",
        ),
        (
            len(candidate_sample.get("raw_seconds", ()))
            == expected["performance_repetitions"],
            "Torch raw replicate count does not match the frozen contract",
        ),
        (
            native_contract == expected_native_contract,
            "native embedded benchmark contract differs from the frozen manifest",
        ),
        (
            contract == expected_candidate_contract,
            "Torch benchmark contract differs from the frozen performance contract",
        ),
        (
            candidate["runtime"]["device"] == "cpu"
            and candidate["runtime"]["precision"] == acceptance["precision"],
            "CPU native gate device or precision differs from the manifest",
        ),
        (
            candidate["runtime"]["threads"] == threads
            and candidate["runtime"]["interop_threads"] == 1,
            "Torch thread policy does not match the CPU gate contract",
        ),
        (
            data["environment"].get("hostname") == platform.node(),
            "native and Torch measurements were not captured on the same host",
        ),
        (
            data["environment"].get("platform") == platform.platform(),
            "native and Torch platform identities differ",
        ),
        (
            data["environment"].get("cpu_count_physical")
            == candidate["runtime"].get("cpu_count_physical_affinity"),
            "native and affinity-aware Torch physical-core counts differ",
        ),
        (
            data["environment"].get("cpu_topology")
            == candidate["runtime"].get("cpu_topology"),
            "native and Torch CPU topologies differ",
        ),
        (
            data["environment"].get("openmp_enabled") is True,
            "frozen native reference was not built with OpenMP",
        ),
    )
    if reference_sha256 != expected["performance_summary_sha256"]:
        errors.append("native summary SHA-256 does not match the frozen artifact")
    errors.extend(message for passed, message in checks if not passed)
    try:
        native_values = tuple(sample.get("raw_seconds", ()))
        candidate_values = tuple(candidate_sample.get("raw_seconds", ()))
        if any(
            isinstance(value, bool) for value in native_values + candidate_values
        ):
            raise TypeError
        native_raw = tuple(float(value) for value in native_values)
        candidate_raw = tuple(float(value) for value in candidate_values)
    except (TypeError, ValueError):
        native_raw = ()
        candidate_raw = ()
        errors.append("raw timing samples must be numeric")
    if not native_raw or not all(
        math.isfinite(value) and value > 0 for value in native_raw
    ):
        errors.append("native raw timing samples must be finite and positive")
    if not candidate_raw or not all(
        math.isfinite(value) and value > 0 for value in candidate_raw
    ):
        errors.append("Torch raw timing samples must be finite and positive")
    max_relative_mad = manifest["performance_gates"]["cpu_acceptance"][
        "statistics"
    ]["max_relative_mad"]
    for label, values in (("native", native_raw), ("Torch", candidate_raw)):
        if values:
            middle = median(values)
            relative_mad = median(abs(value - middle) for value in values) / middle
            if relative_mad > max_relative_mad:
                errors.append(
                    f"{label} raw timing samples exceed the relative-MAD limit"
                )
    def reported_value_matches(value, expected_value):
        try:
            return math.isclose(
                float(value), expected_value, rel_tol=1e-12, abs_tol=0.0
            )
        except (TypeError, ValueError):
            return False

    if native_raw and not reported_value_matches(
        sample.get("median_seconds"), median(native_raw)
    ):
        errors.append("native reported median does not match its raw timing samples")
    if candidate_raw and (
        not reported_value_matches(
            candidate_sample.get("median_seconds"), median(candidate_raw)
        )
        or not reported_value_matches(
            candidate_sample.get("seconds_per_step"),
            median(candidate_raw) / contract["steps_per_repeat"],
        )
    ):
        errors.append("Torch reported timing summary does not match its raw samples")
    reference_seconds = (
        median(native_raw) / sample["steps_per_repeat"] if native_raw else None
    )
    candidate_seconds = (
        median(candidate_raw) / contract["steps_per_repeat"]
        if candidate_raw
        else None
    )
    ratio = (
        candidate_seconds / reference_seconds
        if reference_seconds is not None and candidate_seconds is not None
        else None
    )
    return {
        "reference_observer_tag": data["observer_tag"],
        "reference_observer_commit": data["observer_commit"],
        "reference_precision": acceptance["precision"],
        "reference_contract": native_contract,
        "reference_contract_provenance": contract_provenance,
        "reference_sha256": reference_sha256,
        "reference_seconds_per_step": reference_seconds,
        "candidate_seconds_per_step": candidate_seconds,
        "reference_raw_seconds_per_step": [
            value / sample["steps_per_repeat"] for value in native_raw
        ],
        "candidate_raw_seconds_per_step": [
            value / contract["steps_per_repeat"] for value in candidate_raw
        ],
        "torch_to_native_ratio": ratio,
        "comparison_valid": not errors,
        "contract_errors": errors,
        "individual_ratio_limit": acceptance["max_individual_ratio"],
        "within_five_percent": (
            not errors
            and ratio is not None
            and ratio <= acceptance["max_individual_ratio"]
        ),
    }


def _bootstrap_geomean_regression(gates, statistics):
    """Test a same-thread case slice without accepting malformed evidence."""
    invalid = {
        "method": statistics["method"],
        "evaluated": False,
        "geometric_mean_ratio": None,
        "one_sided_lower_bound": None,
        "significant_regression": None,
        "passed": False,
    }
    if not gates or any(gate.get("comparison_valid") is not True for gate in gates):
        return invalid
    repetitions = None
    validated = []
    for gate in gates:
        try:
            native_values = gate["reference_raw_seconds_per_step"]
            candidate_values = gate["candidate_raw_seconds_per_step"]
            if any(
                isinstance(value, bool)
                for value in tuple(native_values) + tuple(candidate_values)
            ):
                return invalid
            native = np.asarray(native_values, dtype=np.float64)
            candidate = np.asarray(candidate_values, dtype=np.float64)
            if isinstance(gate["torch_to_native_ratio"], bool):
                return invalid
            point_ratio = float(gate["torch_to_native_ratio"])
        except (KeyError, TypeError, ValueError):
            return invalid
        if (
            native.ndim != 1
            or candidate.ndim != 1
            or native.size == 0
            or candidate.size != native.size
            or not np.all(np.isfinite(native))
            or not np.all(native > 0)
            or not np.all(np.isfinite(candidate))
            or not np.all(candidate > 0)
            or not math.isfinite(point_ratio)
            or point_ratio <= 0
        ):
            return invalid
        if repetitions is None:
            repetitions = native.size
        elif native.size != repetitions:
            return invalid
        computed_ratio = float(np.median(candidate) / np.median(native))
        if not math.isclose(point_ratio, computed_ratio, rel_tol=1e-12):
            return invalid
        validated.append((native, candidate, point_ratio))
    try:
        resamples = int(statistics["resamples"])
        seed = int(statistics["seed"])
        confidence = float(statistics["one_sided_confidence"])
        regression_ratio = float(statistics["regression_ratio"])
    except (KeyError, TypeError, ValueError):
        return invalid
    if (
        resamples < 1
        or not 0 < confidence < 1
        or not math.isfinite(regression_ratio)
        or regression_ratio <= 0
    ):
        return invalid
    rng = np.random.default_rng(seed)
    log_ratios = []
    point_ratios = []
    for native, candidate, point_ratio in validated:
        native_indices = rng.integers(0, len(native), size=(resamples, len(native)))
        candidate_indices = rng.integers(
            0, len(candidate), size=(resamples, len(candidate))
        )
        native_medians = np.median(native[native_indices], axis=1)
        candidate_medians = np.median(candidate[candidate_indices], axis=1)
        log_ratios.append(np.log(candidate_medians / native_medians))
        point_ratios.append(point_ratio)
    distribution = np.exp(np.mean(np.stack(log_ratios), axis=0))
    if not np.all(np.isfinite(distribution)):
        return invalid
    lower_bound = float(np.quantile(distribution, 1.0 - confidence))
    significant_regression = lower_bound > regression_ratio
    return {
        "method": statistics["method"],
        "resamples": resamples,
        "seed": seed,
        "one_sided_confidence": confidence,
        "regression_ratio": regression_ratio,
        "evaluated": True,
        "geometric_mean_ratio": math.exp(
            sum(math.log(value) for value in point_ratios) / len(point_ratios)
        ),
        "one_sided_lower_bound": lower_bound,
        "significant_regression": significant_regression,
        "passed": not significant_regression,
    }


def _evaluate_cpu_slice(
    output, manifest, native_summary=None, expected_evidence=None
):
    """Recompute one CPU slice from raw measurements and pinned provenance."""
    acceptance = manifest["performance_gates"]["cpu_acceptance"]
    expected_cases = tuple(acceptance["cases"])
    expected_evidence = expected_evidence or _current_evidence(manifest)
    errors = []
    results = output.get("cases", ())
    if not isinstance(results, (list, tuple)):
        results = ()
        errors.append("CPU slice cases must be a sequence")
    names = tuple(result.get("workload", {}).get("name") for result in results)
    if output.get("schema_version") != 3:
        errors.append("CPU slice schema must be version 3")
    if output.get("kind") != "cpu-acceptance-thread-slice":
        errors.append("CPU slice kind does not match the evidence contract")
    evidence = output.get("evidence")
    if evidence != expected_evidence:
        errors.append("CPU slice provenance does not match the current checkout")
    if not isinstance(evidence, dict) or evidence.get("candidate_git_status") != "":
        errors.append("CPU slice candidate checkout was not clean")
    if names != expected_cases:
        errors.append("CPU slice cases do not match the ordered manifest contract")
    else:
        for result, name in zip(results, expected_cases, strict=True):
            if result.get("workload") != find_case(manifest, name):
                errors.append(f"CPU slice workload {name!r} differs from the manifest")
    runtimes = [result.get("runtime", {}) for result in results]
    threads = {runtime.get("threads") for runtime in runtimes}
    interop_threads = {runtime.get("interop_threads") for runtime in runtimes}
    precisions = {runtime.get("precision") for runtime in runtimes}
    devices = {runtime.get("device") for runtime in runtimes}
    environment = output.get("environment", {})
    physical = environment.get("cpu_count_physical_affinity")
    if threads == {1}:
        mode = "one"
    elif physical is not None and threads == {physical}:
        mode = "physical"
    else:
        mode = None
        errors.append("CPU slice thread count is neither one nor physical cores")
    if interop_threads != {1}:
        errors.append("CPU slice must use exactly one inter-op thread")
    if precisions != {acceptance["precision"]} or devices != {"cpu"}:
        errors.append("CPU slice device or precision differs from the manifest")
    for runtime in runtimes:
        if (
            runtime.get("compile_policy") != "compile"
            or runtime.get("compile_mode") != "default"
            or runtime.get("explicit_cuda_graphs") is not False
            or runtime.get("execution_policy") != "auto"
            or runtime.get("experimental_dispersive_grouping", False) is not False
        ):
            errors.append("CPU slice runtime execution policy differs from the contract")
            break
    for name in ("cpu_affinity", "cpu_count_physical_affinity", "cpu_topology"):
        if any(runtime.get(name) != environment.get(name) for runtime in runtimes):
            errors.append(f"CPU slice runtime {name} differs from root metadata")
    if any(
        not isinstance(result.get("acceptance"), dict)
        or set(result["acceptance"]) != set(RUNTIME_ACCEPTANCE_KEYS)
        or any(
            result["acceptance"].get(name) is not True
            for name in RUNTIME_ACCEPTANCE_KEYS
        )
        for result in results
    ):
        errors.append("CPU slice contains a failing or malformed runtime record")
    native_gates = []
    thread_value = next(iter(threads)) if len(threads) == 1 else None
    if native_summary is None:
        errors.append("CPU slice aggregation requires the pinned native summary")
    elif thread_value is None:
        errors.append("CPU native comparison requires one consistent thread count")
    else:
        for result, name in zip(results, expected_cases):
            try:
                gate = _native_gate(
                    native_summary, name, thread_value, result, manifest
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                errors.append(f"CPU native comparison could not be recomputed: {error}")
                continue
            native_gates.append(gate)
            if result.get("native_gate") != gate:
                errors.append(
                    f"CPU slice embedded native comparison for {name!r} was not exact"
                )
    if len(native_gates) != len(expected_cases) or any(
        gate.get("comparison_valid") is not True
        or gate.get("within_five_percent") is not True
        for gate in native_gates
    ):
        errors.append("CPU slice contains an invalid or failing native comparison")
    statistics = _bootstrap_geomean_regression(
        native_gates, acceptance["statistics"]
    )
    if statistics["passed"] is not True:
        errors.append("CPU slice geometric-mean regression gate failed")
    return {
        "thread_mode": mode,
        "threads": next(iter(threads)) if len(threads) == 1 else None,
        "native_geomean_statistics": statistics,
        "errors": errors,
        "passed": not errors,
    }


def _aggregate_cpu_slice_outputs(
    outputs, manifest, native_summary=None, expected_evidence=None
):
    """Accept only two complete slices from one clean candidate checkout."""
    expected_evidence = expected_evidence or _current_evidence(manifest)
    slices = [
        _evaluate_cpu_slice(
            output, manifest, native_summary, expected_evidence
        )
        for output in outputs
    ]
    errors = []
    if len(outputs) != 2:
        errors.append("exactly two CPU slice artifacts are required")
    modes = [item["thread_mode"] for item in slices]
    if sorted(mode for mode in modes if mode is not None) != ["one", "physical"]:
        errors.append("CPU artifacts must contain one and physical thread modes")
    identity_names = (
        "hostname",
        "platform",
        "python",
        "torch",
        "cpu_affinity",
        "cpu_count_physical_affinity",
        "cpu_topology",
        "cpu_model",
    )
    identities = [
        tuple(output.get("environment", {}).get(name) for name in identity_names)
        for output in outputs
    ]
    if any(value is None for identity in identities for value in identity):
        errors.append("CPU slice host identity metadata is incomplete")
    if identities and any(identity != identities[0] for identity in identities[1:]):
        errors.append("CPU slice host, runtime, topology, or affinity differs")
    evidences = [output.get("evidence") for output in outputs]
    if evidences and any(evidence != evidences[0] for evidence in evidences[1:]):
        errors.append("CPU slices were not produced from the same candidate")
    if any(item["passed"] is not True for item in slices):
        errors.append("at least one CPU thread slice failed")
    return {
        "schema_version": 3,
        "kind": "cpu-acceptance-aggregate",
        "evidence": expected_evidence,
        "environment": outputs[0].get("environment", {}) if outputs else {},
        "cpu_slices": slices,
        "suite_acceptance": {
            "cpu_contract_id": _cpu_contract_id(manifest),
            "cpu_required_cases": list(_cpu_gate_cases(manifest)),
            "cpu_required_thread_modes": ["one", "physical"],
            "cpu_all_thread_modes_complete": not errors,
            "errors": errors,
            "passed": not errors,
        },
    }


def _cpu_contract_id(manifest):
    return manifest["performance_gates"]["cpu_acceptance"]["contract_id"]


def _aggregate_cpu_slice_files(paths, manifest, native_summary):
    artifacts = []
    outputs = []
    for path in paths:
        content = path.read_bytes()
        artifacts.append(
            {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}
        )
        outputs.append(json.loads(content))
    result = _aggregate_cpu_slice_outputs(outputs, manifest, native_summary)
    result["artifacts"] = artifacts
    return result


def run_case(
    name,
    *,
    device,
    precision,
    compile_mode,
    capture_graphs,
    execution_policy,
    experimental_dispersive_grouping,
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
        experimental_dispersive_grouping=experimental_dispersive_grouping,
    )
    cpu_contract = _cpu_contract_environment()
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
    warm_checkpoint = simulation.checkpoint()
    warm_checksum = _checksum(simulation)
    capture_seconds = 0.0
    if capture_graphs:
        start = time.perf_counter()
        simulation.capture_cuda_graphs()
        capture_seconds = time.perf_counter() - start
    after_warmup = _counter_snapshot()
    addresses = simulation.buffer_addresses()
    cpu_memory = (
        _cpu_memory_probe(simulation, warm_checkpoint, profile_steps)
        if simulation.device.type == "cpu"
        else {
            "probe_steps": None,
            "before_bytes": None,
            "after_bytes": None,
            "growth_bytes": None,
        }
    )
    allocated_before = (
        torch.cuda.memory_allocated(simulation.device)
        if simulation.device.type == "cuda"
        else None
    )
    one_step = _perf_counter_samples(
        simulation,
        1,
        repeats,
        checkpoint,
        warmup,
    )
    one_step_checkpoint = simulation.checkpoint()
    one_step_checksum = _checksum(simulation)
    advance = _perf_counter_samples(
        simulation,
        steps,
        repeats,
        checkpoint,
        warmup,
    )
    _synchronize(simulation.device)
    allocated_after = (
        torch.cuda.memory_allocated(simulation.device)
        if simulation.device.type == "cuda"
        else None
    )
    final_checksum = _checksum(simulation)
    timed_step_count = int(simulation.state.step_count.cpu())
    final_checkpoint = simulation.checkpoint()
    state_changes = _state_change_summary(
        one_step_checkpoint["state"], final_checkpoint["state"]
    )
    exploratory_one_step = _timer_samples(
        simulation,
        1,
        repeats,
        threads,
        warm_checkpoint,
    )
    exploratory_advance = _timer_samples(
        simulation,
        steps,
        repeats,
        threads,
        warm_checkpoint,
    )
    simulation.load_checkpoint(warm_checkpoint)
    trace_path = trace_directory / _trace_filename(
        name,
        device=device,
        precision=precision,
        compile_mode=compile_mode,
        capture_graphs=capture_graphs,
        execution_policy=execution_policy,
        threads=threads,
        interop_threads=interop_threads,
        experimental_dispersive_grouping=experimental_dispersive_grouping,
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
    allocations_clean = _recurring_allocations_zero(simulation.device, profiler)
    memory_bounded = _memory_growth_bounded(
        simulation.device,
        memory_growth,
        cpu_memory["growth_bytes"],
    )
    profiler_step_count = int(simulation.state.step_count.cpu())
    expected_one_step_count = warmup + 1
    expected_timed_step_count = warmup + steps
    expected_profiler_step_count = warmup + profile_steps
    contract_matches_manifest = (
        warmup == manifest["reference"]["performance_warmup_steps"]
        and steps == manifest["reference"]["performance_steps_per_repeat"]
        and repeats == manifest["reference"]["performance_repetitions"]
        and profile_steps == manifest["reference"]["performance_profile_steps"]
    )
    compiled_simulations = _compiled_local_simulation_count(simulation)
    expected_compiled_region_events = (
        2 * profile_steps * compiled_simulations
    )
    compiled_hot_path_complete = (
        len(simulation._cuda_graphs) == 2
        and profiler["cuda_graph_launches"] == 2 * profile_steps
        if capture_graphs
        else profiler["compiled_region_events"]
        == expected_compiled_region_events
        and bool(profiler["compiled_region_names"])
        and all(
            count > 0 and count % profile_steps == 0
            for count in profiler["compiled_region_names"].values()
        )
    )
    expected_external_indexed_writes = Counter(
        {
            name: count * profile_steps
            for name, count in _source_index_operations_per_step(simulation).items()
            if count
        }
    )
    profiler["expected_source_indexed_write_names_outside_compiled_regions"] = (
        dict(sorted(expected_external_indexed_writes.items()))
    )
    external_indexed_writes_clean = (
        profiler["indexed_write_names_outside_compiled_regions"]
        == dict(sorted(expected_external_indexed_writes.items()))
    )
    material_state_progressed = (
        (not simulation._has_pml or state_changes["pml_state_changed"])
        and (
            not simulation._dispersive_buckets
            or state_changes["dispersive_state_changed"]
        )
        and (not simulation._has_dm2 or state_changes["dm2_state_changed"])
    )
    result = {
        "schema_version": 2,
        "backend": "torch",
        "workload": spec,
        "benchmark_contract": {
            "initializer": FIELD_INITIALIZER,
            "seed": manifest["reference"]["seed"],
            "field_scale": manifest["reference"]["field_scale"],
            "warmup_steps": warmup,
            "steps_per_repeat": steps,
            "repetitions": repeats,
            "profile_steps": profile_steps,
            "timer": "time.perf_counter",
            "sample_start": "independently-restored-pre-warmup-state",
        },
        "runtime": {
            "device": str(simulation.device),
            "precision": precision,
            "compile_policy": "compile",
            "compile_mode": compile_mode,
            "explicit_cuda_graphs": capture_graphs,
            "execution_policy": execution_policy,
            "experimental_dispersive_grouping": (
                experimental_dispersive_grouping
            ),
            "threads": threads,
            "interop_threads": interop_threads,
            **cpu_contract,
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
            "exploratory_torch_utils_benchmark": {
                "authoritative_for_native_gate": False,
                "hidden_warmup_calls_per_timeit": 2,
                "one_step": _timing_summary(exploratory_one_step),
                "advance": _timing_summary(exploratory_advance, steps=steps),
            },
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
            "cpu_rss_probe_steps": cpu_memory["probe_steps"],
            "cpu_rss_before_bytes": cpu_memory["before_bytes"],
            "cpu_rss_after_bytes": cpu_memory["after_bytes"],
            "cpu_rss_growth_bytes": cpu_memory["growth_bytes"],
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
            "post_warmup_checksum": warm_checksum,
            "post_one_step_checksum": one_step_checksum,
            "final_checksum": final_checksum,
            "changed_after_first_timed_step": one_step_checksum != final_checksum,
            "one_step_count": int(one_step_checkpoint["state"]["step_count"].cpu()),
            "expected_one_step_count": expected_one_step_count,
            "timed_step_count": timed_step_count,
            "expected_timed_step_count": expected_timed_step_count,
            "profiler_step_count": profiler_step_count,
            "expected_profiler_step_count": expected_profiler_step_count,
            **state_changes,
        },
        "diagnostics": simulation.diagnostics(),
        "acceptance": {
            "compiler_clean": compiler_clean,
            "compiled_hot_path_complete": compiled_hot_path_complete,
            "external_indexed_writes_only_sources": external_indexed_writes_clean,
            "steady_state_transfers_zero": transfers_clean,
            "storage_stable": storage_stable,
            "memory_bounded": memory_bounded,
            "recurring_allocations_zero": allocations_clean,
            "measurement_contract_matches_manifest": contract_matches_manifest,
            "state_progressed": (
                state_changes["all_fields_changed"]
                and material_state_progressed
                and int(one_step_checkpoint["state"]["step_count"].cpu())
                == expected_one_step_count
                and timed_step_count == expected_timed_step_count
                and profiler_step_count == expected_profiler_step_count
            ),
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
            experimental_dispersive_grouping=getattr(
                args, "experimental_dispersive_grouping", False
            ),
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
            experimental_dispersive_grouping=getattr(
                args, "experimental_dispersive_grouping", False
            ),
            threads=args.threads,
            interop_threads=args.interop_threads,
            warmup=args.warmup,
            steps=args.steps,
            repeats=args.repeats,
            profile_steps=args.profile_steps,
            trace_directory=args.trace_directory,
            manifest=manifest,
        )
    return {
        "case": name,
        "comparison_valid": False,
        "invalid_reason": (
            "execution_policy currently changes planner/storage metadata only; "
            "runtime uses dense dielectric and compact indexed material updates"
        ),
        "auto_to_fastest_forced_ratio": None,
        "within_ten_percent": None,
        "results": results,
        "passed": False,
    }


def _arguments():
    manifest = load_manifest(MANIFEST)
    reference = manifest["reference"]
    cpu_acceptance = manifest["performance_gates"]["cpu_acceptance"]
    names = tuple(
        item["name"]
        for item in manifest["benchmarks"] + manifest["correctness"]
    )
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
        default=cpu_acceptance["precision"],
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
    parser.add_argument(
        "--experimental-dispersive-grouping",
        action="store_true",
        help="enable the unselected CPU exact-schema dispersive prototype",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument(
        "--warmup", type=int, default=reference["performance_warmup_steps"]
    )
    parser.add_argument(
        "--steps", type=int, default=reference["performance_steps_per_repeat"]
    )
    parser.add_argument(
        "--repeats", type=int, default=reference["performance_repetitions"]
    )
    parser.add_argument(
        "--profile-steps", type=int, default=reference["performance_profile_steps"]
    )
    parser.add_argument(
        "--trace-directory",
        type=Path,
        default=Path("/tmp/gmes-torch-tuning-traces"),
    )
    parser.add_argument("--native-summary", type=Path)
    parser.add_argument(
        "--cpu-slice-artifacts",
        type=Path,
        nargs=2,
        metavar=("ONE_THREAD_JSON", "PHYSICAL_THREAD_JSON"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--enforce", action="store_true")
    return parser.parse_args(), manifest


def main():
    args, manifest = _arguments()
    cpu_gate_cases = _cpu_gate_cases(manifest)
    cpu_slice_artifacts = getattr(args, "cpu_slice_artifacts", None)
    if cpu_slice_artifacts is not None:
        if args.case != "cpu-gates":
            raise ValueError("--cpu-slice-artifacts requires --case cpu-gates")
        if args.native_summary is None:
            raise ValueError(
                "--cpu-slice-artifacts requires --native-summary"
            )
        output = _aggregate_cpu_slice_files(
            cpu_slice_artifacts, manifest, args.native_summary
        )
        rendered = json.dumps(output, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered)
        print(rendered, end="")
        passed = output["suite_acceptance"]["passed"]
        return 0 if not args.enforce or passed else 2
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
        cases = cpu_gate_cases
    elif args.case == "cuda-gates":
        cases = CUDA_GATES
    elif args.case == "policy-gates":
        cases = POLICY_GATES
    else:
        cases = (args.case,)
    native_comparisons_expected = (
        torch.device(args.device).type == "cpu"
        and args.compile_mode != "matrix"
        and args.policy != "matrix"
        and any(name in cpu_gate_cases for name in cases)
    )

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
                experimental_dispersive_grouping=getattr(
                    args, "experimental_dispersive_grouping", False
                ),
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
                manifest,
            )
        results.append(result)

    native_gates = [
        result.get("native_gate")
        for result in results
        if result.get("native_gate") is not None
    ]
    valid_native_gates = [
        item for item in native_gates if item["comparison_valid"]
    ]
    native_comparisons_present = bool(native_gates)
    complete_native_slice = (
        native_comparisons_present
        and len(native_gates) == len(cases)
        and len(valid_native_gates) == len(cases)
    )
    cpu_full_suite_requested = args.case == "cpu-gates"
    statistics = manifest["performance_gates"]["cpu_acceptance"]["statistics"]
    native_statistics = _bootstrap_geomean_regression(
        native_gates if complete_native_slice and cpu_full_suite_requested else (),
        statistics,
    )
    geometric_ratio = native_statistics["geometric_mean_ratio"]
    runtime_passed = all(
        result.get("passed", result.get("acceptance", {}).get("passed", False))
        for result in results
    )
    diagnostic_passed = runtime_passed
    if native_comparisons_expected:
        diagnostic_passed = (
            diagnostic_passed
            and native_comparisons_present
            and complete_native_slice
            and all(item["within_five_percent"] for item in native_gates)
            and (not cpu_full_suite_requested or native_statistics["passed"])
        )
    elif args.native_summary is not None:
        diagnostic_passed = (
            diagnostic_passed
            and len(native_gates) == len(cases)
            and all(item["comparison_valid"] for item in native_gates)
        )
    cpu_thread_modes = manifest["performance_gates"]["cpu_acceptance"][
        "thread_modes"
    ]
    environment = _environment()
    physical_threads = environment.get("cpu_count_physical_affinity")
    if args.threads == 1:
        evaluated_thread_mode = "one"
    elif physical_threads is not None and args.threads == physical_threads:
        evaluated_thread_mode = "physical"
    else:
        evaluated_thread_mode = "unsupported"
    if cpu_full_suite_requested and evaluated_thread_mode == "unsupported":
        diagnostic_passed = False
    is_cpu_diagnostic = (
        torch.device(args.device).type == "cpu"
        and any(name in cpu_gate_cases for name in cases)
    )
    named_non_cpu_suite = args.case in {"cuda-gates", "policy-gates"}
    suite_passed = False
    evidence = _current_evidence(manifest)
    output = {
        "schema_version": 3,
        "kind": (
            "cpu-acceptance-thread-slice"
            if cpu_full_suite_requested
            else "torch-tuning-diagnostic"
        ),
        "evidence": evidence,
        "environment": environment,
        "cases": results,
        "diagnostic_acceptance": {
            "scope": "cpu-thread-slice" if cpu_full_suite_requested else "requested-run",
            "passed": diagnostic_passed,
        },
        "suite_acceptance": {
            "native_geometric_mean_ratio": geometric_ratio,
            "native_comparisons_expected": native_comparisons_expected,
            "native_comparisons_present": native_comparisons_present,
            "native_comparisons_valid": (
                complete_native_slice
                if native_comparisons_expected or args.native_summary is not None
                else None
            ),
            "native_individual_within_five_percent": (
                all(item["within_five_percent"] for item in native_gates)
                if native_gates
                else None
            ),
            "native_geomean_statistics": native_statistics,
            "cpu_contract_id": manifest["performance_gates"]["cpu_acceptance"][
                "contract_id"
            ],
            "cpu_required_cases": list(cpu_gate_cases),
            "cpu_required_thread_modes": cpu_thread_modes,
            "cpu_evaluated_thread_mode": evaluated_thread_mode,
            "cpu_all_thread_modes_complete": (
                False if cpu_full_suite_requested else None
            ),
            "cpu_thread_slice_passed": None,
            "cpu_suite_status": (
                "thread-slice"
                if cpu_full_suite_requested
                else "diagnostic-only"
                if is_cpu_diagnostic
                else "not-applicable"
            ),
            "cpu_incomplete_reason": (
                "cpu-gates records one isolated thread slice; both one-thread and "
                "physical-core artifacts must be combined with "
                "--cpu-slice-artifacts for epic acceptance"
                if cpu_full_suite_requested
                else "a single CPU case cannot satisfy the epic CPU suite"
                if is_cpu_diagnostic
                else None
            ),
            "passed": suite_passed,
        },
    }
    if cpu_full_suite_requested:
        slice_evaluation = _evaluate_cpu_slice(
            output, manifest, args.native_summary, evidence
        )
        output["cpu_slice_evaluation"] = slice_evaluation
        output["diagnostic_acceptance"]["passed"] = slice_evaluation["passed"]
        output["suite_acceptance"]["cpu_thread_slice_passed"] = slice_evaluation[
            "passed"
        ]
    rendered = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    enforced_passed = (
        output["suite_acceptance"]["passed"]
        if is_cpu_diagnostic or cpu_full_suite_requested or named_non_cpu_suite
        else output["diagnostic_acceptance"]["passed"]
    )
    return 0 if not args.enforce or enforced_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
