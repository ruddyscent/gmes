#!/usr/bin/env python3
"""Strict Torch compiler, runtime benchmark, and profiler evidence runner."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import functools
import hashlib
import itertools
import json
import math
import os
import platform
import re
import resource
import subprocess
import sys
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
from benchmarks.torch_cpu_baseline import (
    compare_candidate_to_baseline,
    load_torch_cpu_baseline,
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
EVIDENCE_CONTRACT_ID = "torch-cpu-acceptance-v8"
ALLOCATION_PROVENANCE_METHOD = "reviewed-fixed-temporary-provenance-v1"
ALLOCATION_PROVENANCE_KIND = "torch-cpu-allocation-provenance"
PYTORCH_ISSUE_URL = re.compile(
    r"https://github\.com/pytorch/pytorch/issues/[1-9][0-9]*\Z"
)
CPU_RSS_LIMIT_BYTES = 1024 * 1024
CPU_RSS_STABILIZATION_WINDOWS = 16
CPU_RSS_EVALUATION_BLOCK_WINDOWS = 6
CPU_RSS_EVALUATION_BLOCKS = 2
RUNTIME_ACCEPTANCE_KEYS = (
    "compiler_clean",
    "compiled_hot_path_complete",
    "external_indexed_writes_only_sources",
    "steady_state_transfers_zero",
    "storage_stable",
    "memory_bounded",
    "fixed_temporary_contract_satisfied",
    "measurement_contract_matches_manifest",
    "state_progressed",
    "passed",
)
COUNTER_FIELDS = (
    "graph_breaks",
    "unique_graphs",
    "calls_captured",
    "frames_total",
    "frames_ok",
    "fxgraph_cache_hit",
    "fxgraph_cache_miss",
)
STATE_FIELD_NAMES = frozenset(("ex", "ey", "ez", "hx", "hy", "hz"))
RUNNER_INPUTS = (
    "benchmarks/native_oracle.py",
    "benchmarks/torch_cpu_baseline.py",
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


class _DarwinRusageInfoV0(ctypes.Structure):
    """Public ``rusage_info_v0`` layout from Darwin's ``libproc.h``."""

    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
    ]


@functools.cache
def _darwin_proc_pid_rusage_function():
    """Load the Darwin self-process resource function once per child."""
    library_name = ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib"
    try:
        library = ctypes.CDLL(library_name, use_errno=True)
        proc_pid_rusage = library.proc_pid_rusage
        proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        proc_pid_rusage.restype = ctypes.c_int
    except AttributeError, OSError:
        return None
    return library, proc_pid_rusage


def _darwin_proc_pid_rusage_bytes():
    """Return current Darwin RSS directly from the self-process kernel API."""
    loaded = _darwin_proc_pid_rusage_function()
    if loaded is None:
        return None
    _library, proc_pid_rusage = loaded
    usage = _DarwinRusageInfoV0()
    if proc_pid_rusage(os.getpid(), 0, ctypes.byref(usage)) != 0:
        return None
    return int(usage.ri_resident_size)


def _ps_current_rss_bytes():
    value = _command_text("ps", "-o", "rss=", "-p", str(os.getpid()))
    try:
        return int(value) * 1024 if value is not None else None
    except ValueError:
        return None


def _current_rss_bytes():
    """Return current resident memory on supported CPU acceptance hosts."""
    system = platform.system()
    if system == "Linux":
        try:
            fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
            return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
        except IndexError, OSError, ValueError:
            return None
    if system == "Darwin":
        return _darwin_proc_pid_rusage_bytes()
    return None


def _current_rss_provider():
    """Select and document a current-RSS provider for one probe process."""
    system = platform.system()
    if system == "Linux":
        return _current_rss_bytes, {
            "name": "proc-self-statm",
            "units": "bytes",
            "validated": True,
        }
    if system == "Darwin":
        direct_before = _darwin_proc_pid_rusage_bytes()
        reference = _ps_current_rss_bytes()
        direct_after = _darwin_proc_pid_rusage_bytes()
        available = all(
            value is not None for value in (direct_before, reference, direct_after)
        )
        lower = (
            min(direct_before, direct_after) - CPU_RSS_LIMIT_BYTES
            if available
            else None
        )
        upper = (
            max(direct_before, direct_after) + CPU_RSS_LIMIT_BYTES
            if available
            else None
        )
        validated = bool(available and lower <= reference <= upper)
        metadata = {
            "name": "proc-pid-rusage-v0",
            "units": "bytes",
            "validated": validated,
            "validation": {
                "reference_provider": "ps-rss",
                "direct_before_bytes": direct_before,
                "reference_bytes": reference,
                "direct_after_bytes": direct_after,
                "tolerance_bytes": CPU_RSS_LIMIT_BYTES,
            },
        }
        return (
            _darwin_proc_pid_rusage_bytes if validated else lambda: None,
            metadata,
        )
    return lambda: None, {
        "name": "unavailable",
        "units": "bytes",
        "validated": False,
        "system": system,
    }


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


def _positive_order_permutation_pvalue(values):
    """Return the exact one-sided p-value for a positive six-point trend."""
    values = tuple(int(value) for value in values)
    if len(values) != CPU_RSS_EVALUATION_BLOCK_WINDOWS:
        raise ValueError("RSS trend blocks must contain exactly six values")
    positions = tuple(range(len(values)))

    def order_score(items):
        return sum(position * value for position, value in zip(positions, items))

    observed = order_score(values)
    at_least_observed = sum(
        order_score(permutation) >= observed
        for permutation in itertools.permutations(values)
    )
    return at_least_observed / math.factorial(len(values))


def _evaluate_cpu_rss_plateau(samples):
    """Fail closed unless a 28-window RSS series reaches a bounded plateau."""
    expected = CPU_RSS_STABILIZATION_WINDOWS + (
        CPU_RSS_EVALUATION_BLOCK_WINDOWS * CPU_RSS_EVALUATION_BLOCKS
    )
    if not isinstance(samples, (list, tuple)) or len(samples) != expected:
        return {
            "schema_version": 2,
            "bounded": False,
            "error": f"expected exactly {expected} RSS windows",
        }
    try:
        before = tuple(int(sample["before_bytes"]) for sample in samples)
        after = tuple(int(sample["after_bytes"]) for sample in samples)
    except KeyError, TypeError, ValueError:
        return {
            "schema_version": 2,
            "bounded": False,
            "error": "RSS windows contain unavailable or malformed measurements",
        }
    if any(value < 0 for value in before + after):
        return {
            "schema_version": 2,
            "bounded": False,
            "error": "RSS measurements must be non-negative",
        }

    stable = after[CPU_RSS_STABILIZATION_WINDOWS:]
    blocks = tuple(
        stable[
            index
            * CPU_RSS_EVALUATION_BLOCK_WINDOWS : (index + 1)
            * CPU_RSS_EVALUATION_BLOCK_WINDOWS
        ]
        for index in range(CPU_RSS_EVALUATION_BLOCKS)
    )
    pvalues = tuple(_positive_order_permutation_pvalue(block) for block in blocks)
    positions = tuple(range(CPU_RSS_EVALUATION_BLOCK_WINDOWS))
    position_mean = sum(positions) / len(positions)
    position_variance = sum((position - position_mean) ** 2 for position in positions)
    block_slopes = tuple(
        sum(
            (position - position_mean) * (value - (sum(block) / len(block)))
            for position, value in zip(positions, block)
        )
        / position_variance
        for block in blocks
    )
    persistent_positive_order_trend = all(value <= 0.05 for value in pvalues)
    upward_excursion = max(0, max(stable) - stable[0])
    bounded = (
        upward_excursion <= CPU_RSS_LIMIT_BYTES and not persistent_positive_order_trend
    )
    return {
        "schema_version": 2,
        "bounded": bounded,
        "error": None,
        "window_count": expected,
        "stabilization_window_count": CPU_RSS_STABILIZATION_WINDOWS,
        "stabilization_boundary_index": CPU_RSS_STABILIZATION_WINDOWS,
        "evaluation_block_window_count": CPU_RSS_EVALUATION_BLOCK_WINDOWS,
        "evaluation_block_count": CPU_RSS_EVALUATION_BLOCKS,
        "limit_bytes": CPU_RSS_LIMIT_BYTES,
        "stable_start_bytes": stable[0],
        "stable_start_upward_excursion_bytes": upward_excursion,
        "absolute_envelope_bytes": max(stable) - min(stable),
        "evaluation_block_pvalues": pvalues,
        "evaluation_block_slopes_bytes_per_window": block_slopes,
        "persistent_positive_order_trend": persistent_positive_order_trend,
        "peak_rss_bytes": max(before + after),
        "before_bytes": before,
        "after_bytes": after,
        "deltas_bytes": tuple(
            after_value - before_value
            for before_value, after_value in zip(before, after)
        ),
    }


def _cpu_memory_plateau_probe(simulation, steps):
    """Record the fixed RSS window vector without measured-interval bookkeeping."""
    probe_steps = max(1, int(steps))
    window_count = CPU_RSS_STABILIZATION_WINDOWS + (
        CPU_RSS_EVALUATION_BLOCK_WINDOWS * CPU_RSS_EVALUATION_BLOCKS
    )
    readings = np.empty((window_count, 2), dtype=np.int64)
    read_rss, provider = _current_rss_provider()
    for index in range(window_count):
        before = read_rss()
        if before is None:
            result = _evaluate_cpu_rss_plateau(())
            result["measurement_provider"] = provider
            return result
        simulation.advance(probe_steps)
        _synchronize(simulation.device)
        after = read_rss()
        if after is None:
            result = _evaluate_cpu_rss_plateau(())
            result["measurement_provider"] = provider
            return result
        readings[index, 0] = before
        readings[index, 1] = after
    samples = [
        {"before_bytes": int(before), "after_bytes": int(after)}
        for before, after in readings
    ]
    result = _evaluate_cpu_rss_plateau(samples)
    result["probe_steps_per_window"] = probe_steps
    result["measurement_provider"] = provider
    return result


def _cpu_rss_request(
    name,
    *,
    precision,
    compile_mode,
    execution_policy,
    experimental_dispersive_grouping,
    experimental_dispersive_grouping_scope,
    threads,
    interop_threads,
    warmup,
    profile_steps,
):
    return {
        "case": name,
        "device": "cpu",
        "precision": precision,
        "compile_mode": compile_mode,
        "execution_policy": execution_policy,
        "experimental_dispersive_grouping": experimental_dispersive_grouping,
        "experimental_dispersive_grouping_scope": (
            experimental_dispersive_grouping_scope
        ),
        "threads": threads,
        "interop_threads": interop_threads,
        "warmup": warmup,
        "profile_steps": profile_steps,
    }


def _run_cpu_rss_child(request, manifest):
    """Build and probe one CPU case in the current fresh child process."""
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    torch.set_num_threads(request["threads"])
    if torch.get_num_interop_threads() != request["interop_threads"]:
        torch.set_num_interop_threads(request["interop_threads"])
    _spec, space, geometry, sources, bloch = _build_case(request["case"], manifest)
    runtime = gmes.TorchRuntimeConfig(
        device="cpu",
        precision=request["precision"],
        compile_policy="compile",
        compile_mode=request["compile_mode"],
        cpu_threads=request["threads"],
        cpu_interop_threads=request["interop_threads"],
        execution_policy=request["execution_policy"],
        experimental_dispersive_grouping=request["experimental_dispersive_grouping"],
        experimental_dispersive_grouping_scope=request[
            "experimental_dispersive_grouping_scope"
        ],
    )
    simulation = gmes.TorchSimulation(
        space=space,
        geometry=geometry,
        sources=sources,
        bloch=bloch,
        runtime=runtime,
    )
    _initialize_fields(
        simulation,
        manifest["reference"]["seed"],
        manifest["reference"]["field_scale"],
    )
    checkpoint = simulation.checkpoint()
    simulation.advance(1)
    simulation.load_checkpoint(checkpoint).advance(1)
    simulation.load_checkpoint(checkpoint).advance(request["warmup"])
    counters_before = _counter_snapshot()
    addresses = simulation.buffer_addresses()
    plateau = _cpu_memory_plateau_probe(simulation, request["profile_steps"])
    counters_after = _counter_snapshot()
    counter_growth = _counter_delta(counters_before, counters_after)
    final_addresses = simulation.buffer_addresses()
    storage_stable = addresses == final_addresses
    compiler_clean = (
        counter_growth["unique_graphs"] == 0
        and counter_growth["frames_total"] == 0
        and counter_growth["graph_breaks"] == 0
    )
    plateau["bounded"] = bool(
        plateau.get("bounded") and compiler_clean and storage_stable
    )
    return {
        "schema_version": 1,
        "kind": "cpu-rss-fresh-process",
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "request": request,
        "evidence": _current_evidence(manifest),
        "compile_cache_key": simulation.compile_cache_key,
        "counter_growth": counter_growth,
        "compiler_clean": compiler_clean,
        "storage_addresses_before": addresses,
        "storage_addresses_after": final_addresses,
        "storage_addresses_stable": storage_stable,
        "plateau": plateau,
    }


def _fresh_cpu_memory_probe(request, manifest):
    """Launch and validate the fixed RSS probe in one new Python process."""
    expected_evidence = _current_evidence(manifest)
    command = [
        sys.executable,
        "-m",
        "benchmarks.torch_tuning",
        "--cpu-rss-child",
        "--case",
        request["case"],
        "--precision",
        request["precision"],
        "--compile-mode",
        request["compile_mode"],
        "--policy",
        request["execution_policy"],
        "--experimental-dispersive-grouping-scope",
        request["experimental_dispersive_grouping_scope"],
        "--threads",
        str(request["threads"]),
        "--interop-threads",
        str(request["interop_threads"]),
        "--warmup",
        str(request["warmup"]),
        "--profile-steps",
        str(request["profile_steps"]),
    ]
    if request["experimental_dispersive_grouping"]:
        command.append("--experimental-dispersive-grouping")
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    if completed.returncode != 0:
        return {
            "schema_version": 1,
            "kind": "cpu-rss-fresh-process",
            "request": request,
            "plateau": {
                "schema_version": 2,
                "bounded": False,
                "error": "fresh RSS child failed",
            },
            "child_returncode": completed.returncode,
            "child_stderr": completed.stderr[-4000:],
        }
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        result = {
            "schema_version": 1,
            "kind": "cpu-rss-fresh-process",
            "request": request,
            "plateau": {
                "schema_version": 2,
                "bounded": False,
                "error": "fresh RSS child returned malformed JSON",
            },
        }
    if (
        result.get("request") != request
        or result.get("parent_pid") != os.getpid()
        or not isinstance(result.get("pid"), int)
        or result.get("pid") <= 0
        or result.get("evidence") != expected_evidence
    ):
        result.setdefault("plateau", {})["bounded"] = False
        result["binding_error"] = (
            "fresh RSS child request, PID, or checkout binding failed"
        )
    return result


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


def _load_allocation_provenance(path):
    """Load one reviewed provenance collection and reject ambiguous selectors."""
    raw = path.read_bytes()
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("allocation provenance is not valid JSON") from error
    required_top_level = {"schema_version", "kind", "method", "records"}
    if not isinstance(data, dict) or set(data) != required_top_level:
        raise ValueError("allocation provenance has an invalid top-level schema")
    if (
        data["schema_version"] != 1
        or data["kind"] != ALLOCATION_PROVENANCE_KIND
        or data["method"] != ALLOCATION_PROVENANCE_METHOD
        or not isinstance(data["records"], list)
    ):
        raise ValueError("allocation provenance contract does not match the runner")
    selector_names = (
        "workload",
        "device",
        "precision",
        "compile_mode",
        "execution_policy",
        "threads",
    )
    selectors = set()
    records = []
    for record in data["records"]:
        if not isinstance(record, dict):
            raise ValueError("allocation provenance records must be objects")
        selector = tuple(record.get(name) for name in selector_names)
        if (
            any(not isinstance(value, str) or not value for value in selector[:-1])
            or type(selector[-1]) is not int
            or selector[-1] < 1
        ):
            raise ValueError("allocation provenance record selector is malformed")
        if record.get("method") != ALLOCATION_PROVENANCE_METHOD:
            raise ValueError("allocation provenance record method is invalid")
        if selector in selectors:
            raise ValueError("allocation provenance contains a duplicate selector")
        selectors.add(selector)
        records.append(record)
    return {
        "schema_version": data["schema_version"],
        "kind": data["kind"],
        "method": data["method"],
        "source_artifact": {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        "records": records,
    }


def _select_allocation_provenance(
    document,
    *,
    workload,
    device,
    precision,
    compile_mode,
    execution_policy,
    threads,
):
    if document is None:
        return None
    selector = (
        workload,
        device,
        precision,
        compile_mode,
        execution_policy,
        threads,
    )
    matches = [
        record
        for record in document["records"]
        if tuple(
            record[name]
            for name in (
                "workload",
                "device",
                "precision",
                "compile_mode",
                "execution_policy",
                "threads",
            )
        )
        == selector
    ]
    if len(matches) > 1:
        raise ValueError("allocation provenance selector is ambiguous")
    return matches[0] if matches else None


def _torch_baseline_provenance(baseline):
    if baseline is None:
        return None
    return {
        "kind": baseline.get("kind"),
        "cpu_acceptance_contract_id": baseline.get("cpu_acceptance_contract_id"),
        "timing_reference": baseline.get("timing_reference"),
        "source_artifacts": [
            {
                name: source.get(name)
                for name in (
                    "path",
                    "sha256",
                    "thread_mode",
                    "threads",
                    "thread_environment",
                    "root_commit",
                )
            }
            for source in baseline.get("source_artifacts", ())
        ],
    }


def _normalized_cpu_model_identity(value):
    if not isinstance(value, str):
        return value
    return "\n".join(
        line
        for line in value.splitlines()
        if not line.lstrip().startswith("CPU(s) scaling MHz:")
    )


def _torch_baseline_environment_matches(baseline, environment):
    reference = baseline.get("environment") if isinstance(baseline, dict) else None
    if not isinstance(reference, dict) or not isinstance(environment, dict):
        return False
    candidate = {name: environment.get(name) for name in reference}
    if "cpu_model" in candidate:
        candidate["cpu_model"] = _normalized_cpu_model_identity(candidate["cpu_model"])
    return candidate == reference


def _torch_baseline_thread_environment_matches(baseline, environment, threads):
    if not isinstance(baseline, dict) or not isinstance(environment, dict):
        return False
    matches = [
        source
        for source in baseline.get("source_artifacts", ())
        if isinstance(source, dict) and source.get("threads") == threads
    ]
    return (
        len(matches) == 1
        and isinstance(environment.get("thread_environment"), dict)
        and environment["thread_environment"] == matches[0].get("thread_environment")
    )


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


def _perf_counter_samples(simulation, steps, repeats, initial_checkpoint, warmup):
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
    changed = {name for name in first if not torch.equal(first[name], second[name])}
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
    experimental_dispersive_grouping_scope="combined",
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
        values += (f"exact-schema-dispersive-{experimental_dispersive_grouping_scope}",)
    return (
        "__".join(
            "".join(character if character.isalnum() else "-" for character in value)
            for value in values
        )
        + ".json"
    )


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
    memory_events = 0
    live_allocation_metrics_complete = True
    live_allocation_baseline_bytes = None
    live_allocation_totals = []
    compiled_regions = Counter()
    compiled_intervals = []
    cuda_graph_launches = 0
    for event in events:
        if event.get("name") == "[memory]":
            memory_events += 1
            args = event.get("args", {})
            try:
                size = int(args["Bytes"])
                total_allocated = int(args["Total Allocated"])
            except KeyError, TypeError, ValueError:
                live_allocation_metrics_complete = False
                try:
                    size = int(args.get("Bytes", 0))
                except TypeError, ValueError:
                    size = 0
            else:
                if live_allocation_baseline_bytes is None:
                    live_allocation_baseline_bytes = total_allocated - size
                previous_live = (
                    live_allocation_totals[-1]
                    if live_allocation_totals
                    else live_allocation_baseline_bytes
                )
                if (
                    live_allocation_baseline_bytes < 0
                    or total_allocated < 0
                    or total_allocated != previous_live + size
                ):
                    live_allocation_metrics_complete = False
                live_allocation_totals.append(total_allocated)
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
    if memory_events == 0:
        live_allocation_baseline_bytes = 0
        peak_live_allocated_bytes = 0
        final_live_allocated_bytes = 0
        live_allocation_growth_bytes = 0
    elif live_allocation_metrics_complete:
        peak_live_allocated_bytes = max(
            live_allocation_baseline_bytes, *live_allocation_totals
        )
        final_live_allocated_bytes = live_allocation_totals[-1]
        live_allocation_growth_bytes = (
            final_live_allocated_bytes - live_allocation_baseline_bytes
        )
    else:
        live_allocation_baseline_bytes = None
        peak_live_allocated_bytes = None
        final_live_allocated_bytes = None
        live_allocation_growth_bytes = None
    return {
        "chrome_trace": str(path.resolve()),
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
        "live_allocation_baseline_bytes": live_allocation_baseline_bytes,
        "peak_live_allocated_bytes": peak_live_allocated_bytes,
        "final_live_allocated_bytes": final_live_allocated_bytes,
        "live_allocation_growth_bytes": live_allocation_growth_bytes,
        "live_allocation_metrics_complete": live_allocation_metrics_complete,
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


def _profiler_trace_matches(profiler):
    """Re-read a saved trace and bind every trace-derived profiler field."""
    if not isinstance(profiler, dict) or not isinstance(
        profiler.get("chrome_trace"), str
    ):
        return False
    try:
        summary = _trace_summary(Path(profiler["chrome_trace"]))
    except OSError, TypeError, ValueError, json.JSONDecodeError:
        return False
    return all(profiler.get(name) == value for name, value in summary.items())


def _fixed_temporary_allocation_contract(
    device,
    profiler,
    *,
    compile_cache_key,
    allocation_provenance=None,
    public_upstream_issue_required=True,
):
    result = {
        "method": ALLOCATION_PROVENANCE_METHOD,
        "applied": device.type == "cpu",
        "satisfied": False,
        "status": "failed",
        "zero_allocation": False,
        "checks": {},
        "errors": [],
        "provenance": allocation_provenance,
        "verified_generated_sources": [],
    }
    if device.type != "cpu":
        result.update(satisfied=True, status="not-applied")
        return result

    def check(name, condition, message):
        result["checks"][name] = bool(condition)
        if not condition:
            result["errors"].append(message)

    required_metrics = (
        "positive_allocation_events",
        "allocated_bytes",
        "freed_bytes",
        "allocation_net_bytes",
        "allocation_size_histogram",
        "profile_steps",
        "positive_allocation_operations",
        "live_allocation_baseline_bytes",
        "peak_live_allocated_bytes",
        "final_live_allocated_bytes",
        "live_allocation_growth_bytes",
        "live_allocation_metrics_complete",
    )
    missing_metrics = [key for key in required_metrics if key not in profiler]
    check(
        "profiler_metrics_present",
        not missing_metrics,
        f"missing profiler allocation metrics: {missing_metrics}",
    )
    if missing_metrics:
        return result

    histogram = profiler["allocation_size_histogram"]
    histogram_valid = isinstance(histogram, dict) and all(
        isinstance(size, str)
        and size.isdecimal()
        and int(size) > 0
        and type(count) is int
        and count > 0
        for size, count in histogram.items()
    )
    check(
        "allocation_histogram_valid",
        histogram_valid,
        "allocation histogram must contain positive decimal byte sizes and counts",
    )
    integer_metrics = (
        "positive_allocation_events",
        "allocated_bytes",
        "freed_bytes",
        "allocation_net_bytes",
        "profile_steps",
        "positive_allocation_operations",
    )
    integer_metrics_valid = all(
        type(profiler[key]) is int for key in integer_metrics
    ) and all(
        profiler[key] >= 0
        for key in (
            "positive_allocation_events",
            "allocated_bytes",
            "freed_bytes",
            "positive_allocation_operations",
        )
    )
    check(
        "integer_metrics_valid",
        integer_metrics_valid,
        "allocation counts, bytes, and profile steps must be integers",
    )
    if not histogram_valid or not integer_metrics_valid:
        return result

    profile_steps = profiler["profile_steps"]
    check(
        "profile_steps_positive",
        profile_steps > 0,
        "profile_steps must be positive",
    )
    if profile_steps <= 0:
        return result

    observed_events = sum(histogram.values())
    observed_bytes = sum(int(size) * count for size, count in histogram.items())
    trace_integrity = (
        profiler["positive_allocation_events"] == observed_events
        and profiler["allocated_bytes"] == observed_bytes
        and profiler["allocation_net_bytes"]
        == profiler["allocated_bytes"] - profiler["freed_bytes"]
        and (
            (profiler["positive_allocation_events"] == 0)
            == (profiler["positive_allocation_operations"] == 0)
        )
    )
    check(
        "trace_allocation_integrity",
        trace_integrity,
        "profiler allocation totals do not match the raw event histogram",
    )

    zero_allocation = (
        profiler["positive_allocation_events"] == 0
        and profiler["allocated_bytes"] == 0
        and profiler["freed_bytes"] == 0
        and profiler["allocation_net_bytes"] == 0
        and profiler["positive_allocation_operations"] == 0
        and not histogram
    )
    result["zero_allocation"] = zero_allocation
    if zero_allocation:
        live_zero = (
            profiler["live_allocation_metrics_complete"] is True
            and profiler["live_allocation_baseline_bytes"] == 0
            and profiler["peak_live_allocated_bytes"] == 0
            and profiler["final_live_allocated_bytes"] == 0
            and profiler["live_allocation_growth_bytes"] == 0
        )
        check(
            "zero_allocation_live_metrics",
            live_zero,
            "zero-allocation traces must report complete zero live metrics",
        )
        result["satisfied"] = all(result["checks"].values())
        result["status"] = "zero-allocation" if result["satisfied"] else "failed"
        return result

    balanced = (
        profiler["allocation_net_bytes"] == 0
        and profiler["allocated_bytes"] == profiler["freed_bytes"]
    )
    check(
        "allocation_bytes_balanced",
        balanced,
        "fixed temporary allocations must be completely freed in the trace",
    )
    live_metric_names = (
        "live_allocation_baseline_bytes",
        "peak_live_allocated_bytes",
        "final_live_allocated_bytes",
        "live_allocation_growth_bytes",
    )
    live_metrics_complete = (
        profiler["live_allocation_metrics_complete"] is True
        and all(type(profiler[key]) is int for key in live_metric_names)
        and profiler["peak_live_allocated_bytes"]
        >= max(
            profiler["live_allocation_baseline_bytes"],
            profiler["final_live_allocated_bytes"],
        )
    )
    check(
        "live_allocation_metrics_complete",
        live_metrics_complete,
        "every memory event must contain Bytes and Total Allocated",
    )
    live_growth_zero = (
        live_metrics_complete
        and profiler["live_allocation_growth_bytes"] == 0
        and profiler["final_live_allocated_bytes"]
        == profiler["live_allocation_baseline_bytes"]
    )
    check(
        "live_allocation_growth_zero",
        live_growth_zero,
        "final live allocation must equal the pre-trace baseline",
    )
    histogram_step_stable = all(
        count % profile_steps == 0 for count in histogram.values()
    )
    check(
        "allocation_histogram_step_stable",
        histogram_step_stable,
        "every allocation-size count must be divisible by profile_steps",
    )

    field_buffer_sizes = profiler.get("field_buffer_sizes_bytes")
    field_buffer_sizes_valid = (
        isinstance(field_buffer_sizes, dict)
        and bool(field_buffer_sizes)
        and all(
            isinstance(name, str) and name and type(size) is int and size > 0
            for name, size in field_buffer_sizes.items()
        )
    )
    check(
        "field_buffer_sizes_present",
        field_buffer_sizes_valid,
        "nonzero allocation traces require the measured field buffer byte sizes",
    )
    histogram_sizes = {int(size) for size in histogram}
    no_field_sized_allocations = (
        field_buffer_sizes_valid
        and histogram_sizes.isdisjoint(field_buffer_sizes.values())
    )
    check(
        "no_field_buffer_sized_allocations",
        no_field_sized_allocations,
        "an allocation size matches a live field buffer size",
    )

    provenance_present = isinstance(allocation_provenance, dict)
    check(
        "reviewed_provenance_present",
        provenance_present,
        "nonzero CPU allocations require an explicit reviewed provenance record",
    )
    if not provenance_present:
        return result

    binding_checks = {
        "provenance_method_matches": (
            allocation_provenance.get("method") == ALLOCATION_PROVENANCE_METHOD
        ),
        "provenance_reviewed": allocation_provenance.get("reviewed") is True,
        "provenance_trace_matches": (
            isinstance(profiler.get("chrome_trace_sha256"), str)
            and bool(profiler["chrome_trace_sha256"])
            and allocation_provenance.get("trace_sha256")
            == profiler["chrome_trace_sha256"]
        ),
        "provenance_compile_cache_matches": (
            isinstance(compile_cache_key, str)
            and bool(compile_cache_key)
            and allocation_provenance.get("compile_cache_key") == compile_cache_key
        ),
        "provenance_profile_steps_match": (
            allocation_provenance.get("profile_steps") == profile_steps
        ),
        "provenance_histogram_matches": (
            allocation_provenance.get("allocation_size_histogram") == histogram
        ),
        "full_field_or_domain_clone_events_zero": (
            type(allocation_provenance.get("full_field_or_domain_clone_events")) is int
            and allocation_provenance["full_field_or_domain_clone_events"] == 0
        ),
    }
    for name, condition in binding_checks.items():
        check(name, condition, f"allocation provenance check failed: {name}")

    upstream_issue_urls = allocation_provenance.get("upstream_issue_urls")
    upstream_issues_valid = (
        isinstance(upstream_issue_urls, list)
        and bool(upstream_issue_urls)
        and all(
            isinstance(url, str) and PYTORCH_ISSUE_URL.fullmatch(url) is not None
            for url in upstream_issue_urls
        )
        and len(upstream_issue_urls) == len(set(upstream_issue_urls))
    )
    check(
        "public_upstream_issues_valid",
        not public_upstream_issue_required or upstream_issues_valid,
        "nonzero reviewed allocations require unique canonical public PyTorch "
        "issue URLs",
    )

    allocations = allocation_provenance.get("allocations")
    allocations_valid = isinstance(allocations, list) and bool(allocations)
    accounted_per_step = Counter()
    if allocations_valid:
        for item in allocations:
            item_valid = (
                isinstance(item, dict)
                and type(item.get("size_bytes")) is int
                and item["size_bytes"] > 0
                and type(item.get("events_per_step")) is int
                and item["events_per_step"] > 0
                and item.get("classification") == "allowed-plan-bounded-temporary"
                and isinstance(item.get("generated_operation"), str)
                and bool(item["generated_operation"].strip())
            )
            if not item_valid:
                allocations_valid = False
                break
            accounted_per_step[item["size_bytes"]] += item["events_per_step"]
    check(
        "provenance_allocations_valid",
        allocations_valid,
        "every provenance allocation needs size, per-step count, allowed "
        "classification, and generated operation",
    )
    observed_per_step = Counter(
        {int(size): count // profile_steps for size, count in histogram.items()}
    )
    check(
        "provenance_allocations_fully_accounted",
        allocations_valid and accounted_per_step == observed_per_step,
        "provenance allocations do not exactly account for every per-size event",
    )

    generated_sources = allocation_provenance.get("generated_sources")
    sources_valid = isinstance(generated_sources, list) and bool(generated_sources)
    if sources_valid:
        for item in generated_sources:
            if not (
                isinstance(item, dict)
                and isinstance(item.get("path"), str)
                and bool(item["path"])
                and isinstance(item.get("sha256"), str)
                and bool(item["sha256"])
            ):
                sources_valid = False
                continue
            source_path = Path(item["path"])
            if not source_path.is_absolute():
                source_path = ROOT / source_path
            try:
                source_bytes = source_path.read_bytes()
            except OSError:
                sources_valid = False
                continue
            actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
            source_valid = actual_sha256 == item["sha256"]
            sources_valid = sources_valid and source_valid
            result["verified_generated_sources"].append(
                {
                    "path": str(source_path),
                    "sha256": actual_sha256,
                    "matches_provenance": source_valid,
                }
            )
    check(
        "generated_sources_verified",
        sources_valid,
        "at least one generated source must exist and match its recorded SHA-256",
    )

    result["satisfied"] = all(result["checks"].values())
    if result["satisfied"]:
        result["status"] = "reviewed-fixed-temporary"
    return result


def _json_contract_equal(first, second):
    try:
        return json.dumps(
            first, sort_keys=True, separators=(",", ":"), allow_nan=False
        ) == json.dumps(second, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except TypeError, ValueError:
        return False


def _counter_record_valid(record):
    return (
        isinstance(record, dict)
        and set(record) == set(COUNTER_FIELDS)
        and all(
            type(record[name]) is int and record[name] >= 0 for name in COUNTER_FIELDS
        )
    )


def _storage_address_record_valid(record):
    return (
        isinstance(record, dict)
        and bool(record)
        and all(
            isinstance(name, str)
            and bool(name)
            and type(address) is int
            and address >= 0
            for name, address in record.items()
        )
    )


def _rss_provider_valid(provider):
    if not isinstance(provider, dict):
        return False
    common = provider.get("units") == "bytes" and provider.get("validated") is True
    if provider.get("name") == "proc-self-statm":
        return common and set(provider) == {"name", "units", "validated"}
    if provider.get("name") != "proc-pid-rusage-v0" or not common:
        return False
    validation = provider.get("validation")
    if not isinstance(validation, dict) or set(validation) != {
        "reference_provider",
        "direct_before_bytes",
        "reference_bytes",
        "direct_after_bytes",
        "tolerance_bytes",
    }:
        return False
    values = tuple(
        validation.get(name)
        for name in (
            "direct_before_bytes",
            "reference_bytes",
            "direct_after_bytes",
        )
    )
    if (
        validation.get("reference_provider") != "ps-rss"
        or validation.get("tolerance_bytes") != CPU_RSS_LIMIT_BYTES
        or any(type(value) is not int or value < 0 for value in values)
    ):
        return False
    direct_before, reference, direct_after = values
    return (
        min(direct_before, direct_after) - CPU_RSS_LIMIT_BYTES
        <= reference
        <= max(direct_before, direct_after) + CPU_RSS_LIMIT_BYTES
    )


def _cpu_case_material_state_requirements(workload):
    recipe = workload.get("recipe")
    material = workload.get("material")
    pml = material in {"upml", "cpml"}
    dispersive = isinstance(material, str) and material.startswith(
        ("drude-", "lorentz-", "dcp-")
    )
    dm2 = isinstance(material, str) and material.startswith("dm2-")
    if recipe == "coverage":
        families = workload.get(
            "families",
            ["drude-1", "lorentz-1", "dcp-ade", "dcp-plrc", "dcp-rc", "dm2-1"],
        )
        size = workload.get("size", ())
        try:
            is_3d = len(size) == 3 and float(size[2]) > 0
        except TypeError, ValueError:
            is_3d = False
        if is_3d:
            families = [name for name in families if not name.startswith("dm2-")]
        pml = workload.get("include_pml", True) is True
        dispersive = any(
            isinstance(name, str) and name.startswith(("drude-", "lorentz-", "dcp-"))
            for name in families
        )
        dm2 = any(
            isinstance(name, str) and name.startswith("dm2-") for name in families
        )
    return {"pml": pml, "dispersive": dispersive, "dm2": dm2}


def _cpu_rss_memory_gate(candidate, manifest, expected_evidence):
    errors = []
    memory = candidate.get("memory")
    runtime = candidate.get("runtime")
    contract = candidate.get("benchmark_contract")
    workload = candidate.get("workload")
    if not all(
        isinstance(value, dict) for value in (memory, runtime, contract, workload)
    ):
        return False, ["CPU RSS evidence records are missing"]
    fresh = memory.get("cpu_rss_fresh_process")
    if not isinstance(fresh, dict):
        return False, ["fresh-process CPU RSS evidence is missing"]
    required_fresh_keys = {
        "schema_version",
        "kind",
        "pid",
        "parent_pid",
        "request",
        "evidence",
        "compile_cache_key",
        "counter_growth",
        "compiler_clean",
        "storage_addresses_before",
        "storage_addresses_after",
        "storage_addresses_stable",
        "plateau",
    }
    if set(fresh) != required_fresh_keys:
        errors.append("fresh-process CPU RSS schema is not exact")
    if (
        type(fresh.get("schema_version")) is not int
        or fresh["schema_version"] != 1
        or fresh.get("kind") != "cpu-rss-fresh-process"
    ):
        errors.append("fresh-process CPU RSS identity is invalid")
    expected_request = _cpu_rss_request(
        workload.get("name"),
        precision=runtime.get("precision"),
        compile_mode=runtime.get("compile_mode"),
        execution_policy=runtime.get("execution_policy"),
        experimental_dispersive_grouping=runtime.get(
            "experimental_dispersive_grouping", False
        ),
        experimental_dispersive_grouping_scope=runtime.get(
            "experimental_dispersive_grouping_scope"
        ),
        threads=runtime.get("threads"),
        interop_threads=runtime.get("interop_threads"),
        warmup=contract.get("warmup_steps"),
        profile_steps=contract.get("profile_steps"),
    )
    if not _json_contract_equal(fresh.get("request"), expected_request):
        errors.append("fresh-process CPU RSS request is not exact")
    if not _json_contract_equal(fresh.get("evidence"), expected_evidence):
        errors.append("fresh-process CPU RSS checkout evidence is not exact")
    if fresh.get("compile_cache_key") != runtime.get("compile_cache_key"):
        errors.append("fresh-process CPU RSS compile cache key is not exact")
    if not (
        type(fresh.get("pid")) is int
        and fresh["pid"] > 0
        and type(fresh.get("parent_pid")) is int
        and fresh["parent_pid"] > 0
        and fresh["pid"] != fresh["parent_pid"]
    ):
        errors.append("fresh-process CPU RSS process identity is malformed")

    child_counters = fresh.get("counter_growth")
    child_counters_valid = _counter_record_valid(child_counters)
    child_compiler_clean = child_counters_valid and all(
        child_counters[name] == 0
        for name in ("graph_breaks", "unique_graphs", "frames_total")
    )
    if not child_counters_valid:
        errors.append("fresh-process CPU RSS compiler counters are malformed")
    if fresh.get("compiler_clean") is not child_compiler_clean:
        errors.append("fresh-process CPU RSS compiler result is not exact")

    child_addresses_before = fresh.get("storage_addresses_before")
    child_addresses_after = fresh.get("storage_addresses_after")
    child_storage_stable = (
        _storage_address_record_valid(child_addresses_before)
        and _storage_address_record_valid(child_addresses_after)
        and child_addresses_before == child_addresses_after
    )
    if fresh.get("storage_addresses_stable") is not child_storage_stable:
        errors.append("fresh-process CPU RSS storage result is not exact")

    plateau = fresh.get("plateau")
    plateau_exact = False
    recomputed_bounded = False
    if isinstance(plateau, dict):
        before = plateau.get("before_bytes")
        after = plateau.get("after_bytes")
        if isinstance(before, (list, tuple)) and isinstance(after, (list, tuple)):
            samples = [
                {"before_bytes": first, "after_bytes": second}
                for first, second in zip(before, after, strict=False)
            ]
            recomputed_plateau = _evaluate_cpu_rss_plateau(samples)
            recomputed_plateau["probe_steps_per_window"] = contract.get("profile_steps")
            recomputed_plateau["measurement_provider"] = plateau.get(
                "measurement_provider"
            )
            recomputed_plateau["bounded"] = bool(
                recomputed_plateau.get("bounded")
                and child_compiler_clean
                and child_storage_stable
            )
            plateau_exact = _json_contract_equal(plateau, recomputed_plateau)
            recomputed_bounded = recomputed_plateau["bounded"]
        if not _rss_provider_valid(plateau.get("measurement_provider")):
            errors.append("fresh-process CPU RSS provider is not validated")
    if not plateau_exact:
        errors.append("fresh-process CPU RSS plateau does not match its raw windows")

    addresses_before = memory.get("storage_addresses_before")
    addresses_after = memory.get("storage_addresses_after")
    storage_stable = (
        _storage_address_record_valid(addresses_before)
        and _storage_address_record_valid(addresses_after)
        and addresses_before == addresses_after
    )
    if memory.get("storage_addresses_stable") is not storage_stable:
        errors.append("parent CPU storage result is not exact")
    before_values = plateau.get("before_bytes", ()) if isinstance(plateau, dict) else ()
    after_values = plateau.get("after_bytes", ()) if isinstance(plateau, dict) else ()
    first_before = before_values[0] if before_values else None
    final_after = after_values[-1] if after_values else None
    growth = (
        final_after - first_before
        if type(first_before) is int and type(final_after) is int
        else None
    )
    parent_summary = (
        memory.get("cpu_rss_probe_steps"),
        memory.get("cpu_rss_before_bytes"),
        memory.get("cpu_rss_after_bytes"),
        memory.get("cpu_rss_growth_bytes"),
    )
    expected_parent_summary = (
        contract.get("profile_steps"),
        first_before,
        final_after,
        growth,
    )
    if (
        any(type(value) is not int for value in parent_summary)
        or parent_summary != expected_parent_summary
    ):
        errors.append("parent CPU RSS summary does not match the child raw windows")
    if memory.get("bounded") is not recomputed_bounded:
        errors.append("parent CPU RSS bounded result is not exact")
    if any(
        memory.get(name) is not None
        for name in (
            "cuda_allocated_before_bytes",
            "cuda_allocated_after_bytes",
            "cuda_allocated_growth_bytes",
            "cuda_peak_allocated_bytes",
            "cuda_peak_reserved_bytes",
        )
    ):
        errors.append("CPU memory evidence contains CUDA allocation values")
    return not errors and recomputed_bounded and storage_stable, errors


def _recompute_cpu_runtime_acceptance(candidate, manifest, expected_evidence):
    """Recompute every non-allocation CPU runtime gate from raw evidence."""
    errors = []
    compiler = candidate.get("compiler")
    profiler = candidate.get("profiler")
    memory = candidate.get("memory")
    state = candidate.get("state_progress")
    diagnostics = candidate.get("diagnostics")
    contract = candidate.get("benchmark_contract")
    workload = candidate.get("workload")
    records = (compiler, profiler, memory, state, diagnostics, contract, workload)
    if not all(isinstance(value, dict) for value in records):
        return {}, ["CPU runtime raw evidence records are missing"]

    expected_contract = {
        "initializer": FIELD_INITIALIZER,
        "seed": manifest["reference"]["seed"],
        "field_scale": manifest["reference"]["field_scale"],
        "warmup_steps": manifest["reference"]["performance_warmup_steps"],
        "steps_per_repeat": manifest["reference"]["performance_steps_per_repeat"],
        "repetitions": manifest["reference"]["performance_repetitions"],
        "profile_steps": manifest["reference"]["performance_profile_steps"],
        "timer": "time.perf_counter",
        "sample_start": "independently-restored-pre-warmup-state",
    }
    measurement_contract_matches = _json_contract_equal(contract, expected_contract)

    expected_compiler_keys = {
        "after_cold",
        "after_warmup",
        "after_steady",
        "steady_state_delta",
        "fullgraph_clean",
    }
    snapshots_valid = set(compiler) == expected_compiler_keys and all(
        _counter_record_valid(compiler.get(name))
        for name in ("after_cold", "after_warmup", "after_steady")
    )
    compiler_clean = False
    if snapshots_valid:
        expected_delta = _counter_delta(
            compiler["after_warmup"], compiler["after_steady"]
        )
        monotonic = all(value >= 0 for value in expected_delta.values()) and all(
            compiler["after_warmup"][name] >= compiler["after_cold"][name]
            for name in COUNTER_FIELDS
        )
        if compiler.get("steady_state_delta") != expected_delta or not monotonic:
            errors.append("CPU compiler snapshots and steady delta are inconsistent")
        compiler_clean = (
            monotonic
            and compiler["after_steady"]["graph_breaks"] == 0
            and expected_delta["unique_graphs"] == 0
            and expected_delta["frames_total"] == 0
        )
    else:
        errors.append("CPU compiler counter schema is malformed")
    if compiler.get("fullgraph_clean") is not compiler_clean:
        errors.append("CPU compiler embedded gate is not exact")

    profile_steps = contract.get("profile_steps")
    compiled_names = profiler.get("compiled_region_names")
    compiled_hot_path_complete = (
        type(profile_steps) is int
        and profile_steps > 0
        and profiler.get("profile_steps") == profile_steps
        and type(profiler.get("compiled_region_events")) is int
        and profiler["compiled_region_events"] == 2 * profile_steps
        and isinstance(compiled_names, dict)
        and len(compiled_names) == 2
        and all(
            isinstance(name, str)
            and bool(name)
            and type(count) is int
            and count == profile_steps
            for name, count in compiled_names.items()
        )
    )
    transfer_counts = (
        profiler.get("host_to_device_events"),
        profiler.get("device_to_host_events"),
    )
    transfers_zero = all(type(value) is int and value == 0 for value in transfer_counts)
    source_diagnostics = diagnostics.get("sources")
    expected_external_writes = {}
    external_indexed_writes_clean = (
        isinstance(source_diagnostics, dict)
        and source_diagnostics.get("execution_representation")
        == gmes.torch_fdtd.FUSED_SOURCE_REPRESENTATION
        and profiler.get("expected_source_indexed_write_names_outside_compiled_regions")
        == expected_external_writes
        and profiler.get("indexed_write_names_outside_compiled_regions")
        == expected_external_writes
        and type(profiler.get("indexed_write_operations_outside_compiled_regions"))
        is int
        and profiler["indexed_write_operations_outside_compiled_regions"] == 0
    )

    memory_bounded, memory_errors = _cpu_rss_memory_gate(
        candidate, manifest, expected_evidence
    )
    errors.extend(memory_errors)
    storage_stable = (
        _storage_address_record_valid(memory.get("storage_addresses_before"))
        and memory.get("storage_addresses_before")
        == memory.get("storage_addresses_after")
        and memory.get("storage_addresses_stable") is True
    )

    changed_buffers = state.get("changed_buffers")
    changed_buffers_valid = (
        isinstance(changed_buffers, list)
        and all(isinstance(name, str) and bool(name) for name in changed_buffers)
        and changed_buffers == sorted(set(changed_buffers))
    )
    changed = set(changed_buffers) if changed_buffers_valid else set()
    fields_changed = sorted(changed & STATE_FIELD_NAMES)
    derived_state = {
        "fields_changed": fields_changed,
        "all_fields_changed": STATE_FIELD_NAMES <= changed,
        "pml_state_changed": any(name.startswith("pml_") for name in changed),
        "dispersive_state_changed": any(name.startswith("bucket_") for name in changed),
        "dm2_state_changed": any(name.startswith("dm2_buckets.") for name in changed),
    }
    if not changed_buffers_valid or any(
        state.get(name) != value for name, value in derived_state.items()
    ):
        errors.append("CPU state changed-buffer summaries are inconsistent")
    requirements = _cpu_case_material_state_requirements(workload)
    pml_diagnostics = diagnostics.get("pml")
    dispersive_diagnostics = diagnostics.get("dispersive")
    diagnostic_material_flags = {
        "pml": (
            isinstance(pml_diagnostics, dict)
            and type(pml_diagnostics.get("active_cells")) is int
            and pml_diagnostics["active_cells"] > 0
        ),
        "dispersive": (
            isinstance(dispersive_diagnostics, dict)
            and type(dispersive_diagnostics.get("active_cells")) is int
            and dispersive_diagnostics["active_cells"] > 0
        ),
        "dm2": isinstance(diagnostics.get("dm2"), (list, tuple))
        and bool(diagnostics["dm2"]),
    }
    if diagnostic_material_flags != requirements:
        errors.append("CPU material diagnostics differ from the manifest workload")

    numeric_checksums = all(
        not isinstance(state.get(name), bool)
        and isinstance(state.get(name), (int, float))
        and math.isfinite(float(state[name]))
        and float(state[name]) >= 0
        for name in (
            "initial_checksum",
            "post_warmup_checksum",
            "post_one_step_checksum",
            "final_checksum",
        )
    )
    changed_after_first = (
        numeric_checksums and state["post_one_step_checksum"] != state["final_checksum"]
    )
    if state.get("changed_after_first_timed_step") is not changed_after_first:
        errors.append("CPU state checksum-change summary is inconsistent")
    if not numeric_checksums:
        errors.append("CPU state checksums are malformed")
    expected_counts = {
        "expected_one_step_count": expected_contract["warmup_steps"] + 1,
        "expected_timed_step_count": expected_contract["warmup_steps"]
        + expected_contract["steps_per_repeat"],
        "expected_profiler_step_count": expected_contract["warmup_steps"]
        + expected_contract["profile_steps"],
    }
    counts_valid = all(
        type(state.get(name)) is int
        for name in (
            "one_step_count",
            "expected_one_step_count",
            "timed_step_count",
            "expected_timed_step_count",
            "profiler_step_count",
            "expected_profiler_step_count",
        )
    ) and all(state.get(name) == value for name, value in expected_counts.items())
    counts_match = counts_valid and all(
        state[actual] == state[expected]
        for actual, expected in (
            ("one_step_count", "expected_one_step_count"),
            ("timed_step_count", "expected_timed_step_count"),
            ("profiler_step_count", "expected_profiler_step_count"),
        )
    )
    if not counts_match:
        errors.append("CPU state step counts are inconsistent")
    material_progressed = all(
        not required or derived_state[f"{name}_state_changed"]
        for name, required in requirements.items()
    )
    state_progressed = (
        changed_buffers_valid
        and numeric_checksums
        and derived_state["all_fields_changed"]
        and material_progressed
        and counts_match
    )

    recomputed = {
        "compiler_clean": compiler_clean,
        "compiled_hot_path_complete": compiled_hot_path_complete,
        "external_indexed_writes_only_sources": external_indexed_writes_clean,
        "steady_state_transfers_zero": transfers_zero,
        "storage_stable": storage_stable,
        "memory_bounded": memory_bounded,
        "measurement_contract_matches_manifest": measurement_contract_matches,
        "state_progressed": state_progressed,
    }
    recorded = candidate.get("acceptance")
    if not isinstance(recorded, dict):
        errors.append("CPU runtime acceptance record is missing")
    else:
        for name, value in recomputed.items():
            if recorded.get(name) is not value:
                errors.append(f"CPU runtime acceptance {name!r} is not exact")
            if value is not True:
                errors.append(f"CPU runtime gate {name!r} failed")
    return recomputed, errors


def _recompute_cpu_allocation_contract(
    candidate,
    allocation_document,
    expected_method,
    public_upstream_issue_required=True,
):
    """Re-evaluate only allocation acceptance from a saved candidate trace."""
    errors = []
    runtime = candidate.get("runtime")
    profiler = candidate.get("profiler")
    recorded_acceptance = candidate.get("acceptance")
    embedded_contract = candidate.get("allocation_contract")
    if not isinstance(runtime, dict):
        errors.append("CPU allocation runtime is missing")
        runtime = {}
    if (
        not isinstance(recorded_acceptance, dict)
        or set(recorded_acceptance) != set(RUNTIME_ACCEPTANCE_KEYS)
        or any(type(value) is not bool for value in recorded_acceptance.values())
    ):
        errors.append("CPU runtime acceptance record is malformed")
        recorded_acceptance = {}
    else:
        recorded_passed = all(
            recorded_acceptance[name]
            for name in RUNTIME_ACCEPTANCE_KEYS
            if name != "passed"
        )
        if recorded_acceptance["passed"] is not recorded_passed:
            errors.append("CPU runtime aggregate acceptance is inconsistent")
        if any(
            recorded_acceptance[name] is not True
            for name in RUNTIME_ACCEPTANCE_KEYS
            if name not in {"fixed_temporary_contract_satisfied", "passed"}
        ):
            errors.append("CPU runtime contains a non-allocation gate failure")
    if not _profiler_trace_matches(profiler):
        errors.append("saved CPU profiler trace does not match its embedded summary")
    try:
        provenance = _select_allocation_provenance(
            allocation_document,
            workload=candidate.get("workload", {}).get("name"),
            device=runtime.get("device"),
            precision=runtime.get("precision"),
            compile_mode=runtime.get("compile_mode"),
            execution_policy=runtime.get("execution_policy"),
            threads=runtime.get("threads"),
        )
        device = torch.device(runtime.get("device"))
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"CPU allocation provenance selection failed: {error}")
        provenance = None
        device = torch.device("cpu")
    recomputed = _fixed_temporary_allocation_contract(
        device,
        profiler if isinstance(profiler, dict) else {},
        compile_cache_key=runtime.get("compile_cache_key"),
        allocation_provenance=provenance,
        public_upstream_issue_required=public_upstream_issue_required,
    )
    original_recomputed = _fixed_temporary_allocation_contract(
        device,
        profiler if isinstance(profiler, dict) else {},
        compile_cache_key=runtime.get("compile_cache_key"),
        allocation_provenance=(
            embedded_contract.get("provenance")
            if isinstance(embedded_contract, dict)
            else None
        ),
        public_upstream_issue_required=public_upstream_issue_required,
    )
    if recomputed.get("method") != expected_method:
        errors.append("recomputed CPU allocation method differs from the manifest")
    if recomputed.get("applied") is not True or recomputed.get("satisfied") is not True:
        errors.append("recomputed CPU allocation contract failed")
    if not isinstance(embedded_contract, dict):
        errors.append("embedded CPU allocation contract is missing")
    elif embedded_contract != original_recomputed:
        errors.append("embedded CPU allocation contract was not originally exact")
    elif embedded_contract.get("satisfied") is True:
        if embedded_contract != recomputed:
            errors.append("embedded successful allocation contract is not exact")
        if recorded_acceptance.get("fixed_temporary_contract_satisfied") is not True:
            errors.append("embedded allocation acceptance is inconsistent")
    elif not (
        embedded_contract.get("method") == expected_method
        and embedded_contract.get("applied") is True
        and embedded_contract.get("satisfied") is False
        and embedded_contract.get("status") == "failed"
        and recorded_acceptance.get("fixed_temporary_contract_satisfied") is False
        and recorded_acceptance.get("passed") is False
    ):
        errors.append("embedded failed allocation draft is malformed")
    return recomputed, errors


def _field_buffer_sizes_bytes(simulation):
    result = {}

    def visit(item, prefix):
        fields = item.state.fields()
        for name, tensor in fields.items():
            result[f"{prefix}state.{name}"] = tensor.numel() * tensor.element_size()
        groups = {
            "all-fields": tuple(fields.values()),
            "state-domain": tuple(item.state.buffers()),
            "plan-domain": tuple(item.plan.buffers()),
            "source-domain": tuple(item.sources.buffers()),
            "probe-domain": tuple(item.probes.buffers()),
        }
        live_domain_bytes = 0
        for name, tensors in groups.items():
            size = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
            if size > 0:
                result[f"{prefix}aggregate.{name}"] = size
                if name != "all-fields":
                    live_domain_bytes += size
        if live_domain_bytes > 0:
            result[f"{prefix}aggregate.live-domain"] = live_domain_bytes
        for index, auxiliary in enumerate(item.sources.auxiliaries):
            visit(auxiliary, f"{prefix}sources.auxiliaries[{index}].")

    visit(simulation, "")
    return result


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
        simulation._electric_half is not None and simulation._magnetic_half is not None
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
            "comparison_role": "informational",
            "comparison_valid": False,
            "contract_errors": [
                f"native summary has no {name!r} sample at {threads} thread(s)"
            ],
            "reference_seconds_per_step": None,
            "candidate_seconds_per_step": candidate["measurements"]["advance"][
                "seconds_per_step"
            ],
            "torch_to_native_ratio": None,
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
            len(sample.get("raw_seconds", ())) == expected["performance_repetitions"],
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
        if any(isinstance(value, bool) for value in native_values + candidate_values):
            raise TypeError
        native_raw = tuple(float(value) for value in native_values)
        candidate_raw = tuple(float(value) for value in candidate_values)
    except TypeError, ValueError:
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
    max_relative_mad = manifest["performance_gates"]["cpu_acceptance"]["statistics"][
        "max_relative_mad"
    ]
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
        except TypeError, ValueError:
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
        median(candidate_raw) / contract["steps_per_repeat"] if candidate_raw else None
    )
    ratio = (
        candidate_seconds / reference_seconds
        if reference_seconds is not None and candidate_seconds is not None
        else None
    )
    return {
        "comparison_role": "informational",
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
    }


def _bootstrap_geomean_regression(gates, statistics, *, ratio_key):
    """Bootstrap a case collection without accepting malformed evidence."""
    invalid = {
        "method": statistics["method"],
        "ratio_key": ratio_key,
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
            reference_values = gate["reference_raw_seconds_per_step"]
            candidate_values = gate["candidate_raw_seconds_per_step"]
            if any(
                isinstance(value, bool)
                for value in tuple(reference_values) + tuple(candidate_values)
            ):
                return invalid
            reference = np.asarray(reference_values, dtype=np.float64)
            candidate = np.asarray(candidate_values, dtype=np.float64)
            if isinstance(gate[ratio_key], bool):
                return invalid
            point_ratio = float(gate[ratio_key])
        except KeyError, TypeError, ValueError:
            return invalid
        if (
            reference.ndim != 1
            or candidate.ndim != 1
            or reference.size == 0
            or candidate.size != reference.size
            or not np.all(np.isfinite(reference))
            or not np.all(reference > 0)
            or not np.all(np.isfinite(candidate))
            or not np.all(candidate > 0)
            or not math.isfinite(point_ratio)
            or point_ratio <= 0
        ):
            return invalid
        if repetitions is None:
            repetitions = reference.size
        elif reference.size != repetitions:
            return invalid
        computed_ratio = float(np.median(candidate) / np.median(reference))
        if not math.isclose(point_ratio, computed_ratio, rel_tol=1e-12):
            return invalid
        validated.append((reference, candidate, point_ratio))
    try:
        resamples = int(statistics["resamples"])
        seed = int(statistics["seed"])
        confidence = float(statistics["one_sided_confidence"])
        regression_ratio = float(statistics["regression_ratio"])
    except KeyError, TypeError, ValueError:
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
    for reference, candidate, point_ratio in validated:
        reference_indices = rng.integers(
            0, len(reference), size=(resamples, len(reference))
        )
        candidate_indices = rng.integers(
            0, len(candidate), size=(resamples, len(candidate))
        )
        reference_medians = np.median(reference[reference_indices], axis=1)
        candidate_medians = np.median(candidate[candidate_indices], axis=1)
        log_ratios.append(np.log(candidate_medians / reference_medians))
        point_ratios.append(point_ratio)
    distribution = np.exp(np.mean(np.stack(log_ratios), axis=0))
    if not np.all(np.isfinite(distribution)):
        return invalid
    lower_bound = float(np.quantile(distribution, 1.0 - confidence))
    significant_regression = lower_bound > regression_ratio
    return {
        "method": statistics["method"],
        "ratio_key": ratio_key,
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
    output,
    manifest,
    native_summary=None,
    torch_baseline=None,
    allocation_document=None,
    expected_evidence=None,
):
    """Recompute one CPU slice from raw measurements and pinned references."""
    acceptance = manifest["performance_gates"]["cpu_acceptance"]
    expected_cases = tuple(acceptance["cases"])
    expected_evidence = expected_evidence or _current_evidence(manifest)
    errors = []
    if not isinstance(output, dict):
        output = {}
        errors.append("CPU slice root must be an object")
    results = output.get("cases", ())
    if not isinstance(results, (list, tuple)):
        results = ()
        errors.append("CPU slice cases must be a sequence")
    elif any(not isinstance(result, dict) for result in results):
        results = ()
        errors.append("CPU slice cases must contain objects")
    names = tuple(result.get("workload", {}).get("name") for result in results)
    if output.get("schema_version") != 4:
        errors.append("CPU slice schema must be version 4")
    if output.get("kind") != "cpu-acceptance-thread-slice":
        errors.append("CPU slice kind does not match the evidence contract")
    evidence = output.get("evidence")
    if not _json_contract_equal(evidence, expected_evidence):
        errors.append("CPU slice provenance does not match the current checkout")
    if not isinstance(evidence, dict) or evidence.get("candidate_git_status") != "":
        errors.append("CPU slice candidate checkout was not clean")
    if not _json_contract_equal(
        output.get("torch_baseline"), _torch_baseline_provenance(torch_baseline)
    ):
        errors.append("CPU slice Torch baseline provenance is not exact")
    if names != expected_cases:
        errors.append("CPU slice cases do not match the ordered manifest contract")
    else:
        for result, name in zip(results, expected_cases, strict=True):
            if not _json_contract_equal(
                result.get("workload"), find_case(manifest, name)
            ):
                errors.append(f"CPU slice workload {name!r} differs from the manifest")
    runtimes = [result.get("runtime", {}) for result in results]
    if any(not isinstance(runtime, dict) for runtime in runtimes):
        errors.append("CPU slice runtimes must be objects")
        runtimes = [
            runtime if isinstance(runtime, dict) else {} for runtime in runtimes
        ]
    threads = {runtime.get("threads") for runtime in runtimes}
    interop_threads = {runtime.get("interop_threads") for runtime in runtimes}
    precisions = {runtime.get("precision") for runtime in runtimes}
    devices = {runtime.get("device") for runtime in runtimes}
    environment = output.get("environment", {})
    if not isinstance(environment, dict):
        environment = {}
        errors.append("CPU slice environment must be an object")
    physical = environment.get("cpu_count_physical_affinity")
    thread_counts_valid = bool(runtimes) and all(
        type(runtime.get("threads")) is int and runtime["threads"] > 0
        for runtime in runtimes
    )
    if not thread_counts_valid:
        errors.append("CPU slice intra-op thread counts are malformed")
    if thread_counts_valid and threads == {1}:
        mode = "one"
    elif (
        thread_counts_valid
        and type(physical) is int
        and physical > 0
        and threads == {physical}
    ):
        mode = "physical"
    else:
        mode = None
        errors.append("CPU slice thread count is neither one nor physical cores")
    thread_value = next(iter(threads)) if len(threads) == 1 else None
    if interop_threads != {1} or any(
        type(runtime.get("interop_threads")) is not int for runtime in runtimes
    ):
        errors.append("CPU slice must use exactly one inter-op thread")
    if precisions != {acceptance["precision"]} or devices != {"cpu"}:
        errors.append("CPU slice device or precision differs from the manifest")
    if torch_baseline is not None and not _torch_baseline_environment_matches(
        torch_baseline, environment
    ):
        errors.append("CPU slice host identity differs from the Torch baseline")
    if torch_baseline is not None and not _torch_baseline_thread_environment_matches(
        torch_baseline, environment, thread_value
    ):
        errors.append("CPU slice thread environment differs from the Torch baseline")
    for runtime in runtimes:
        if (
            runtime.get("compile_policy") != "compile"
            or runtime.get("compile_mode") != "default"
            or runtime.get("explicit_cuda_graphs") is not False
            or runtime.get("execution_policy") != "auto"
            or runtime.get("experimental_dispersive_grouping", False) is not False
        ):
            errors.append(
                "CPU slice runtime execution policy differs from the contract"
            )
            break
    for name in ("cpu_affinity", "cpu_count_physical_affinity", "cpu_topology"):
        if any(runtime.get(name) != environment.get(name) for runtime in runtimes):
            errors.append(f"CPU slice runtime {name} differs from root metadata")
    runtime_acceptance = []
    for result in results:
        recomputed, runtime_errors = _recompute_cpu_runtime_acceptance(
            result, manifest, expected_evidence
        )
        runtime_acceptance.append(recomputed)
        errors.extend(f"CPU runtime evidence: {message}" for message in runtime_errors)
    allocation_definition = acceptance["allocation_contract"]
    allocation_method = allocation_definition["method"]
    allocation_contracts = []
    for result in results:
        recomputed, allocation_errors = _recompute_cpu_allocation_contract(
            result,
            allocation_document,
            allocation_method,
            allocation_definition["public_upstream_issue_required"],
        )
        allocation_contracts.append(recomputed)
        errors.extend(
            f"CPU allocation evidence: {message}" for message in allocation_errors
        )
    native_gates = []
    baseline_gates = []
    if native_summary is None:
        errors.append("CPU slice requires the pinned native summary")
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
        gate.get("comparison_valid") is not True for gate in native_gates
    ):
        errors.append("CPU slice contains an invalid native comparison")
    if torch_baseline is None:
        errors.append("CPU slice requires the pinned Torch baseline artifacts")
    elif thread_value is None:
        errors.append("Torch baseline comparison requires one thread count")
    else:
        for result, name in zip(results, expected_cases):
            try:
                gate = compare_candidate_to_baseline(
                    torch_baseline,
                    result,
                    name=name,
                    threads=thread_value,
                )
            except (KeyError, TypeError, ValueError) as error:
                errors.append(
                    f"Torch baseline comparison could not be recomputed: {error}"
                )
                continue
            baseline_gates.append(gate)
            if result.get("torch_baseline_gate") != gate:
                errors.append(
                    "CPU slice embedded Torch baseline comparison for "
                    f"{name!r} was not exact"
                )
    if len(baseline_gates) != len(expected_cases) or any(
        gate.get("comparison_valid") is not True
        or gate.get("within_five_percent") is not True
        for gate in baseline_gates
    ):
        errors.append(
            "CPU slice contains an invalid or failing Torch baseline comparison"
        )
    return {
        "thread_mode": mode,
        "threads": next(iter(threads)) if len(threads) == 1 else None,
        "native_comparison_role": "informational",
        "native_comparisons": native_gates,
        "torch_baseline_comparisons": baseline_gates,
        "recomputed_runtime_acceptance": runtime_acceptance,
        "recomputed_allocation_contracts": allocation_contracts,
        "errors": errors,
        "passed": not errors,
    }


def _aggregate_cpu_slice_outputs(
    outputs,
    manifest,
    native_summary=None,
    torch_baseline=None,
    allocation_document=None,
    expected_evidence=None,
):
    """Accept two complete slices after a twelve-cell baseline bootstrap."""
    expected_evidence = expected_evidence or _current_evidence(manifest)
    slices = [
        _evaluate_cpu_slice(
            output,
            manifest,
            native_summary,
            torch_baseline,
            allocation_document,
            expected_evidence,
        )
        for output in outputs
    ]
    errors = []
    if len(outputs) != 2:
        errors.append("exactly two CPU slice artifacts are required")
    for index, item in enumerate(slices):
        errors.extend(
            f"CPU slice {index}: {message}" for message in item.get("errors", ())
        )
    modes = [item["thread_mode"] for item in slices]
    modes_complete = sorted(mode for mode in modes if mode is not None) == [
        "one",
        "physical",
    ]
    if not modes_complete:
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
    identities = []
    for output in outputs:
        environment = output.get("environment", {})
        identity = []
        for name in identity_names:
            value = environment.get(name)
            if name == "cpu_model":
                value = _normalized_cpu_model_identity(value)
            identity.append(value)
        identities.append(tuple(identity))
    if any(value is None for identity in identities for value in identity):
        errors.append("CPU slice host identity metadata is incomplete")
    if identities and any(identity != identities[0] for identity in identities[1:]):
        errors.append("CPU slice host, runtime, topology, or affinity differs")
    evidences = [output.get("evidence") for output in outputs]
    if evidences and any(evidence != evidences[0] for evidence in evidences[1:]):
        errors.append("CPU slices were not produced from the same candidate")
    if any(item["passed"] is not True for item in slices):
        errors.append("at least one CPU thread slice failed")
    baseline_gates = [
        gate for item in slices for gate in item["torch_baseline_comparisons"]
    ]
    expected_cell_count = 2 * len(_cpu_gate_cases(manifest))
    native_gates = [gate for item in slices for gate in item["native_comparisons"]]
    native_comparisons_valid = len(native_gates) == expected_cell_count and all(
        gate.get("comparison_valid") is True for gate in native_gates
    )
    native_statistics = _bootstrap_geomean_regression(
        native_gates,
        manifest["performance_gates"]["cpu_acceptance"]["statistics"],
        ratio_key="torch_to_native_ratio",
    )
    individual_comparisons_passed = len(baseline_gates) == expected_cell_count and all(
        gate.get("comparison_valid") is True and gate.get("within_five_percent") is True
        for gate in baseline_gates
    )
    if not individual_comparisons_passed:
        errors.append("the twelve Torch baseline comparisons are incomplete or failing")
    statistics = _bootstrap_geomean_regression(
        baseline_gates,
        manifest["performance_gates"]["cpu_acceptance"]["statistics"],
        ratio_key="candidate_to_torch_baseline_ratio",
    )
    if len(baseline_gates) != expected_cell_count or statistics["passed"] is not True:
        errors.append("the twelve-cell Torch baseline bootstrap gate failed")
    return {
        "schema_version": 4,
        "kind": "cpu-acceptance-aggregate",
        "acceptance_scope": "cpu-performance-only",
        "issue_completion_satisfied": False,
        "issue_completion_blockers": [
            "complete-field-and-persistent-state-correctness-not-bound"
        ],
        "evidence": expected_evidence,
        "environment": outputs[0].get("environment", {}) if outputs else {},
        "torch_baseline": _torch_baseline_provenance(torch_baseline),
        "allocation_provenance_artifact": (
            allocation_document["source_artifact"]
            if allocation_document is not None
            else None
        ),
        "cpu_slices": slices,
        "suite_acceptance": {
            "cpu_contract_id": _cpu_contract_id(manifest),
            "cpu_required_cases": list(_cpu_gate_cases(manifest)),
            "cpu_required_thread_modes": ["one", "physical"],
            "cpu_all_thread_modes_complete": modes_complete,
            "cpu_evaluated_cell_count": len(baseline_gates),
            "native_comparison_role": "informational",
            "native_comparisons_valid": native_comparisons_valid,
            "native_geomean_statistics": native_statistics,
            "torch_baseline_individual_within_five_percent": (
                individual_comparisons_passed
            ),
            "torch_baseline_geomean_statistics": statistics,
            "errors": errors,
            "passed": not errors,
        },
    }


def _cpu_contract_id(manifest):
    return manifest["performance_gates"]["cpu_acceptance"]["contract_id"]


def _aggregate_cpu_slice_files(
    paths,
    manifest,
    native_summary,
    torch_baseline_artifacts,
    allocation_document=None,
):
    artifacts = []
    outputs = []
    for path in paths:
        content = path.read_bytes()
        artifacts.append(
            {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}
        )
        outputs.append(json.loads(content))
    torch_baseline = load_torch_cpu_baseline(torch_baseline_artifacts, manifest)
    result = _aggregate_cpu_slice_outputs(
        outputs,
        manifest,
        native_summary,
        torch_baseline,
        allocation_document,
    )
    result["candidate_slice_artifacts"] = artifacts
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
    experimental_dispersive_grouping_scope,
    threads,
    interop_threads,
    warmup,
    steps,
    repeats,
    profile_steps,
    trace_directory,
    manifest,
    allocation_provenance=None,
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
        experimental_dispersive_grouping_scope=(experimental_dispersive_grouping_scope),
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
    cpu_memory = {
        "probe_steps": None,
        "before_bytes": None,
        "after_bytes": None,
        "growth_bytes": None,
        "fresh_process": None,
    }
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
        experimental_dispersive_grouping_scope=(experimental_dispersive_grouping_scope),
    )
    profiler = _profile(simulation, profile_steps, trace_path)
    profiler["field_buffer_sizes_bytes"] = _field_buffer_sizes_bytes(simulation)
    after_steady = _counter_snapshot()
    if simulation.device.type == "cpu":
        rss_request = _cpu_rss_request(
            name,
            precision=precision,
            compile_mode=compile_mode,
            execution_policy=execution_policy,
            experimental_dispersive_grouping=experimental_dispersive_grouping,
            experimental_dispersive_grouping_scope=(
                experimental_dispersive_grouping_scope
            ),
            threads=threads,
            interop_threads=interop_threads,
            warmup=warmup,
            profile_steps=profile_steps,
        )
        rss_artifact = _fresh_cpu_memory_probe(rss_request, manifest)
        plateau = rss_artifact.get("plateau", {})
        if rss_artifact.get("compile_cache_key") != simulation.compile_cache_key:
            plateau["bounded"] = False
            rss_artifact["compile_key_binding_error"] = (
                "fresh RSS child compile key differs from the measured parent"
            )
        after_values = plateau.get("after_bytes", ())
        before_values = plateau.get("before_bytes", ())
        cpu_memory = {
            "probe_steps": plateau.get("probe_steps_per_window"),
            "before_bytes": before_values[0] if before_values else None,
            "after_bytes": after_values[-1] if after_values else None,
            "growth_bytes": (
                after_values[-1] - before_values[0]
                if after_values and before_values
                else None
            ),
            "fresh_process": rss_artifact,
        }
    counter_growth = _counter_delta(after_warmup, after_steady)
    final_addresses = simulation.buffer_addresses()
    storage_stable = addresses == final_addresses
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
    allocation_contract = _fixed_temporary_allocation_contract(
        simulation.device,
        profiler,
        compile_cache_key=simulation.compile_cache_key,
        allocation_provenance=allocation_provenance,
        public_upstream_issue_required=manifest["performance_gates"]["cpu_acceptance"][
            "allocation_contract"
        ]["public_upstream_issue_required"],
    )
    memory_bounded = (
        cpu_memory["fresh_process"].get("plateau", {}).get("bounded") is True
        if simulation.device.type == "cpu"
        else _memory_growth_bounded(
            simulation.device,
            memory_growth,
            cpu_memory["growth_bytes"],
        )
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
    expected_compiled_region_events = 2 * profile_steps * compiled_simulations
    compiled_hot_path_complete = (
        len(simulation._cuda_graphs) == 2
        and profiler["cuda_graph_launches"] == 2 * profile_steps
        if capture_graphs
        else profiler["compiled_region_events"] == expected_compiled_region_events
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
    profiler["expected_source_indexed_write_names_outside_compiled_regions"] = dict(
        sorted(expected_external_indexed_writes.items())
    )
    external_indexed_writes_clean = profiler[
        "indexed_write_names_outside_compiled_regions"
    ] == dict(sorted(expected_external_indexed_writes.items()))
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
            "experimental_dispersive_grouping": (experimental_dispersive_grouping),
            "experimental_dispersive_grouping_scope": (
                experimental_dispersive_grouping_scope
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
            "cpu_rss_fresh_process": cpu_memory["fresh_process"],
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
            "storage_addresses_before": addresses,
            "storage_addresses_after": final_addresses,
            "storage_addresses_stable": storage_stable,
            "bounded": memory_bounded,
        },
        "profiler": profiler,
        "allocation_contract": allocation_contract,
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
            "fixed_temporary_contract_satisfied": allocation_contract["satisfied"],
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


def _allocation_provenance_for_run(
    document, args, name, *, compile_mode=None, execution_policy=None
):
    return _select_allocation_provenance(
        document,
        workload=name,
        device=str(torch.device(args.device)),
        precision=args.precision,
        compile_mode=compile_mode or args.compile_mode,
        execution_policy=execution_policy or args.policy,
        threads=args.threads,
    )


def _variant_matrix(args, name, manifest, allocation_document=None):
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
            experimental_dispersive_grouping_scope=getattr(
                args, "experimental_dispersive_grouping_scope", "combined"
            ),
            threads=args.threads,
            interop_threads=args.interop_threads,
            warmup=args.warmup,
            steps=args.steps,
            repeats=args.repeats,
            profile_steps=args.profile_steps,
            trace_directory=args.trace_directory,
            manifest=manifest,
            allocation_provenance=_allocation_provenance_for_run(
                allocation_document,
                args,
                name,
                compile_mode=compile_mode,
            ),
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


def _policy_matrix(args, name, manifest, allocation_document=None):
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
            experimental_dispersive_grouping_scope=getattr(
                args, "experimental_dispersive_grouping_scope", "combined"
            ),
            threads=args.threads,
            interop_threads=args.interop_threads,
            warmup=args.warmup,
            steps=args.steps,
            repeats=args.repeats,
            profile_steps=args.profile_steps,
            trace_directory=args.trace_directory,
            manifest=manifest,
            allocation_provenance=_allocation_provenance_for_run(
                allocation_document,
                args,
                name,
                execution_policy=policy,
            ),
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
        item["name"] for item in manifest["benchmarks"] + manifest["correctness"]
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
    parser.add_argument("--cpu-rss-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--experimental-dispersive-grouping",
        action="store_true",
        help="enable the unselected CPU exact-schema dispersive prototype",
    )
    parser.add_argument(
        "--experimental-dispersive-grouping-scope",
        choices=("combined", "two-level", "dcp-convolution"),
        default="combined",
        help="select the exact-schema recurrence family used by the experiment",
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
        "--torch-baseline-slice-artifacts",
        type=Path,
        nargs=2,
        metavar=("ONE_THREAD_JSON", "PHYSICAL_THREAD_JSON"),
    )
    parser.add_argument("--allocation-provenance", type=Path)
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
    if getattr(args, "cpu_rss_child", False):
        request = _cpu_rss_request(
            args.case,
            precision=args.precision,
            compile_mode=args.compile_mode,
            execution_policy=args.policy,
            experimental_dispersive_grouping=args.experimental_dispersive_grouping,
            experimental_dispersive_grouping_scope=(
                args.experimental_dispersive_grouping_scope
            ),
            threads=args.threads,
            interop_threads=args.interop_threads,
            warmup=args.warmup,
            profile_steps=args.profile_steps,
        )
        output = _run_cpu_rss_child(request, manifest)
        rendered = json.dumps(output, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered)
        print(rendered, end="")
        # An adverse but structurally valid plateau is evidence, not a child crash.
        return 0
    cpu_slice_artifacts = getattr(args, "cpu_slice_artifacts", None)
    torch_baseline_artifacts = getattr(args, "torch_baseline_slice_artifacts", None)
    allocation_document = (
        _load_allocation_provenance(args.allocation_provenance)
        if getattr(args, "allocation_provenance", None) is not None
        else None
    )
    if cpu_slice_artifacts is not None:
        if args.case != "cpu-gates":
            raise ValueError("--cpu-slice-artifacts requires --case cpu-gates")
        if args.native_summary is None:
            raise ValueError("--cpu-slice-artifacts requires --native-summary")
        if torch_baseline_artifacts is None:
            raise ValueError(
                "--cpu-slice-artifacts requires " "--torch-baseline-slice-artifacts"
            )
        output = _aggregate_cpu_slice_files(
            cpu_slice_artifacts,
            manifest,
            args.native_summary,
            torch_baseline_artifacts,
            allocation_document,
        )
        rendered = json.dumps(output, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered)
        print(rendered, end="")
        passed = (
            output["suite_acceptance"]["passed"]
            and output.get("issue_completion_satisfied") is True
        )
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

    torch_baseline = (
        load_torch_cpu_baseline(torch_baseline_artifacts, manifest)
        if torch_baseline_artifacts is not None
        else None
    )

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
            result = _variant_matrix(args, name, manifest, allocation_document)
        elif args.policy == "matrix":
            result = _policy_matrix(args, name, manifest, allocation_document)
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
                experimental_dispersive_grouping_scope=getattr(
                    args, "experimental_dispersive_grouping_scope", "combined"
                ),
                threads=args.threads,
                interop_threads=args.interop_threads,
                warmup=args.warmup,
                steps=args.steps,
                repeats=args.repeats,
                profile_steps=args.profile_steps,
                trace_directory=args.trace_directory,
                manifest=manifest,
                allocation_provenance=_allocation_provenance_for_run(
                    allocation_document,
                    args,
                    name,
                ),
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
        if (
            torch_baseline is not None
            and torch.device(args.device).type == "cpu"
            and name in cpu_gate_cases
            and args.compile_mode != "matrix"
            and args.policy != "matrix"
        ):
            result["torch_baseline_gate"] = compare_candidate_to_baseline(
                torch_baseline,
                result,
                name=name,
                threads=args.threads,
            )
        results.append(result)

    native_gates = [
        result.get("native_gate")
        for result in results
        if result.get("native_gate") is not None
    ]
    valid_native_gates = [item for item in native_gates if item["comparison_valid"]]
    native_comparisons_present = bool(native_gates)
    complete_native_slice = (
        native_comparisons_present
        and len(native_gates) == len(cases)
        and len(valid_native_gates) == len(cases)
    )
    baseline_gates = [
        result.get("torch_baseline_gate")
        for result in results
        if result.get("torch_baseline_gate") is not None
    ]
    valid_baseline_gates = [
        item for item in baseline_gates if item.get("comparison_valid") is True
    ]
    baseline_comparisons_present = bool(baseline_gates)
    complete_baseline_slice = (
        baseline_comparisons_present
        and len(baseline_gates) == len(cases)
        and len(valid_baseline_gates) == len(cases)
    )
    cpu_full_suite_requested = args.case == "cpu-gates"
    statistics = manifest["performance_gates"]["cpu_acceptance"]["statistics"]
    baseline_statistics = _bootstrap_geomean_regression(
        (),
        statistics,
        ratio_key="candidate_to_torch_baseline_ratio",
    )
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
            and baseline_comparisons_present
            and complete_baseline_slice
            and all(item.get("within_five_percent") is True for item in baseline_gates)
        )
    elif args.native_summary is not None:
        diagnostic_passed = (
            diagnostic_passed
            and len(native_gates) == len(cases)
            and all(item["comparison_valid"] for item in native_gates)
        )
    cpu_thread_modes = manifest["performance_gates"]["cpu_acceptance"]["thread_modes"]
    environment = _environment()
    baseline_host_matches = (
        _torch_baseline_environment_matches(torch_baseline, environment)
        and _torch_baseline_thread_environment_matches(
            torch_baseline, environment, args.threads
        )
        if torch_baseline is not None
        else False
    )
    if native_comparisons_expected and not baseline_host_matches:
        diagnostic_passed = False
    physical_threads = environment.get("cpu_count_physical_affinity")
    if args.threads == 1:
        evaluated_thread_mode = "one"
    elif physical_threads is not None and args.threads == physical_threads:
        evaluated_thread_mode = "physical"
    else:
        evaluated_thread_mode = "unsupported"
    if cpu_full_suite_requested and evaluated_thread_mode == "unsupported":
        diagnostic_passed = False
    is_cpu_diagnostic = torch.device(args.device).type == "cpu" and any(
        name in cpu_gate_cases for name in cases
    )
    named_non_cpu_suite = args.case in {"cuda-gates", "policy-gates"}
    suite_passed = False
    evidence = _current_evidence(manifest)
    output = {
        "schema_version": 4,
        "kind": (
            "cpu-acceptance-thread-slice"
            if cpu_full_suite_requested
            else "torch-tuning-diagnostic"
        ),
        "evidence": evidence,
        "environment": environment,
        "torch_baseline": _torch_baseline_provenance(torch_baseline),
        "allocation_provenance_artifact": (
            allocation_document["source_artifact"]
            if allocation_document is not None
            else None
        ),
        "cases": results,
        "diagnostic_acceptance": {
            "scope": (
                "cpu-thread-slice" if cpu_full_suite_requested else "requested-run"
            ),
            "passed": diagnostic_passed,
        },
        "suite_acceptance": {
            "native_comparison_role": "informational",
            "native_comparisons_expected": native_comparisons_expected,
            "native_comparisons_present": native_comparisons_present,
            "native_comparisons_valid": (
                complete_native_slice
                if native_comparisons_expected or args.native_summary is not None
                else None
            ),
            "torch_baseline_comparisons_expected": native_comparisons_expected,
            "torch_baseline_comparisons_present": baseline_comparisons_present,
            "torch_baseline_comparisons_valid": (
                complete_baseline_slice
                if native_comparisons_expected or torch_baseline is not None
                else None
            ),
            "torch_baseline_host_matches": (
                baseline_host_matches if torch_baseline is not None else None
            ),
            "torch_baseline_individual_within_five_percent": (
                all(item.get("within_five_percent") is True for item in baseline_gates)
                if baseline_gates
                else None
            ),
            "torch_baseline_geomean_statistics": baseline_statistics,
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
                else "diagnostic-only" if is_cpu_diagnostic else "not-applicable"
            ),
            "cpu_incomplete_reason": (
                "cpu-gates records one isolated thread slice; both one-thread and "
                "physical-core artifacts must be combined with "
                "--cpu-slice-artifacts for epic acceptance"
                if cpu_full_suite_requested
                else (
                    "a single CPU case cannot satisfy the epic CPU suite"
                    if is_cpu_diagnostic
                    else None
                )
            ),
            "passed": suite_passed,
        },
    }
    if cpu_full_suite_requested:
        slice_evaluation = _evaluate_cpu_slice(
            output,
            manifest,
            args.native_summary,
            torch_baseline,
            allocation_document,
            evidence,
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
