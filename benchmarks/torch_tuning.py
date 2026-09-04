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
from pathlib import Path, PurePosixPath
from statistics import median, pstdev
from threading import Lock

import numpy as np
import torch
from torch.utils import benchmark as torch_benchmark

import gmes
from benchmarks.host_contract import capture_host_contract, host_contract_complete
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
from benchmarks.torch_correctness import (
    TRUSTED_MANIFEST_SHA256,
    _runtime_receipt_candidates,
    correctness_binding_complete,
    load_correctness_evidence_index,
    load_runtime_publication_receipt,
)
from benchmarks.torch_cpu_baseline import (
    compare_candidate_to_baseline,
    load_torch_cpu_baseline,
    privacy_preserving_host_identity,
    timing_runtime_identity_matches,
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
CUDA_SUITE_CONTRACT_ID = "single-gpu-cuda-closure-v2"
CUDA_PERFORMANCE_PRECISION_BY_CASE = {
    "cpu-crossover-2d": "float32",
    "cpu-crossover-3d": "float32",
    "cpu-large-2d": "float32",
    "cpu-large-3d": "float32",
    "single-gpu-2d": "float32",
    "single-gpu-3d": "float64",
}
CUDA_PRECISION_LIMITATION_REVIEW = {
    "contract_id": "single-gpu-3d-float32-dynamic-range-review-v1",
    "case": "single-gpu-3d",
    "rejected_precision": "float32",
    "accepted_precision": "float64",
    "reason": "native-step-100-magnitude-exceeds-float32-range",
}
STATE_FINITENESS_CONTRACT_ID = "dynamic-checkpoint-finite-v1"
STATE_FINITENESS_STAGES = (
    "initial",
    "post_warmup",
    "post_one_step",
    "post_timed",
    "post_profile",
)
SIMULATION_CHECKPOINT_KEYS = {
    "format",
    "version",
    "metadata",
    "state",
    "auxiliaries",
    "probes",
}
CPU_CORRECTNESS_RUNTIME_MODE = {
    "device": "cpu",
    "precision": "float64",
    "graph_mode": "eager",
    "compile_policy": "eager",
    "compile_mode": "default",
}
CUDA_CORRECTNESS_RUNTIME_MODES = (
    {
        "device": "cuda:0",
        "precision": "float32",
        "graph_mode": "eager",
        "compile_policy": "eager",
        "compile_mode": "default",
    },
    {
        "device": "cuda:0",
        "precision": "float32",
        "graph_mode": "graph",
        "compile_policy": "compile",
        "compile_mode": "reduce-overhead",
    },
)
CUDA_CORRECTNESS_SOURCE_DESCRIPTOR_KEYS = (
    "path",
    "sha256",
    "size_bytes",
    "media_type",
    "candidate_evidence",
)
CUDA_CORRECTNESS_SOURCE_CANDIDATE_KEYS = (
    "candidate_git_commit",
    "candidate_git_status",
    "manifest_sha256",
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
PAIRED_REAL_GATES = ("bloch-2d", "bloch-3d")
REGION_INVARIANCE_GATES = (
    "equivalent-region-1",
    "equivalent-region-32",
)
SPECIAL_CASES = (
    "all-material-2d",
    "all-material-3d",
    *REGION_INVARIANCE_GATES,
)
COMPILE_VARIANTS = (
    ("default-no-graph", "default", False),
    ("default-explicit-graph", "default", True),
    ("reduce-overhead", "reduce-overhead", False),
    ("max-autotune", "max-autotune", False),
)
POLICY_EXECUTION_REPRESENTATIONS = {
    "dense": "dense-mask-scatter-v1",
    "compact": "compact-index-copy-v1",
    "tiled": "tiled-index-scatter-v1",
}
POLICY_WRITE_OPERATIONS = {
    "dense": "aten::masked_scatter_",
    "compact": "aten::index_copy_",
    "tiled": "aten::scatter_",
}
POLICY_DIAGNOSTIC_KIND = "torch-policy-uncompiled-operation-trace"
POLICY_DIAGNOSTIC_CONTRACT = "forced-dispersive-write-op-v1"
COMPILE_CACHE_PREIMAGE_ALGORITHM = "sha256-python-repr-nested-tuples-v1"
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
LINUX_PROC_STATM_BUFFER_BYTES = 256
INT64_MAX = (1 << 63) - 1
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
        reader = _LinuxProcStatmReader()
        try:
            return reader()
        finally:
            reader.close()
    if system == "Darwin":
        return _darwin_proc_pid_rusage_bytes()
    return None


class _LinuxProcStatmReader:
    """Read Linux RSS with probe-scoped, allocation-stable storage."""

    _WHITESPACE = (9, 10, 11, 12, 13, 32)

    def __init__(self):
        self._buffer = bytearray(LINUX_PROC_STATM_BUFFER_BYTES)
        self._view = memoryview(self._buffer)
        self._iov = (self._view,)
        self._lock = Lock()
        self._pid = os.getpid()
        self._fd = None
        self._closed = False
        preadv = getattr(os, "preadv", None)
        self._preadv = preadv if callable(preadv) else None
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
        except OSError, ValueError:
            page_size = None
        self._page_size = (
            page_size if type(page_size) is int and page_size > 0 else None
        )
        maximum_resident_pages = (
            INT64_MAX // self._page_size if self._page_size is not None else None
        )
        if maximum_resident_pages is None:
            self._maximum_resident_pages_tens = None
            self._maximum_resident_pages_ones = None
        else:
            (
                self._maximum_resident_pages_tens,
                self._maximum_resident_pages_ones,
            ) = divmod(maximum_resident_pages, 10)
        if self._page_size is not None and self._preadv is not None:
            self._open()

    def _open(self):
        read_only = getattr(os, "O_RDONLY", None)
        close_on_exec = getattr(os, "O_CLOEXEC", None)
        if type(read_only) is not int or type(close_on_exec) is not int:
            self._fd = None
            return
        try:
            self._fd = os.open(
                "/proc/self/statm",
                read_only | close_on_exec,
            )
        except AttributeError, OSError:
            self._fd = None

    def _close_fd(self):
        fd = self._fd
        self._fd = None
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass

    def _reset_for_pid(self, pid):
        self._pid = pid
        self._lock = Lock()
        self._close_fd()
        if (
            not self._closed
            and self._page_size is not None
            and self._preadv is not None
        ):
            self._open()

    @staticmethod
    def _resident_pages(buffer, length, maximum_tens, maximum_ones):
        index = 0
        while index < length and buffer[index] in _LinuxProcStatmReader._WHITESPACE:
            index += 1
        first_start = index
        while index < length and 48 <= buffer[index] <= 57:
            index += 1
        if (
            index == first_start
            or index == length
            or buffer[index] not in _LinuxProcStatmReader._WHITESPACE
        ):
            return None
        while index < length and buffer[index] in _LinuxProcStatmReader._WHITESPACE:
            index += 1
        second_start = index
        value = 0
        while index < length and 48 <= buffer[index] <= 57:
            digit = buffer[index] - 48
            if value > maximum_tens or (value == maximum_tens and digit > maximum_ones):
                return None
            value = value * 10 + digit
            index += 1
        if index == second_start:
            return None
        if index == length or buffer[index] not in _LinuxProcStatmReader._WHITESPACE:
            return None
        return value

    def __call__(self):
        pid = os.getpid()
        if pid != self._pid:
            self._reset_for_pid(pid)
        with self._lock:
            if (
                self._closed
                or self._fd is None
                or self._page_size is None
                or self._preadv is None
            ):
                return None
            try:
                length = self._preadv(self._fd, self._iov, 0)
            except AttributeError, OSError:
                self._close_fd()
                return None
            if type(length) is not int or length <= 0 or length >= len(self._buffer):
                self._close_fd()
                return None
            resident_pages = _LinuxProcStatmReader._resident_pages(
                self._buffer,
                length,
                self._maximum_resident_pages_tens,
                self._maximum_resident_pages_ones,
            )
            if resident_pages is None:
                self._close_fd()
                return None
            return resident_pages * self._page_size

    def close(self):
        """Close the probe-scoped descriptor once without raising."""
        pid = os.getpid()
        if pid != self._pid:
            self._pid = pid
            self._lock = Lock()
        with self._lock:
            self._closed = True
            self._close_fd()


def _current_rss_provider():
    """Select and document a current-RSS provider for one probe process."""
    system = platform.system()
    if system == "Linux":
        reader = _LinuxProcStatmReader()
        validated = reader() is not None
        if not validated:
            reader.close()
        return reader, {
            "name": "proc-self-statm-preadv-v1",
            "units": "bytes",
            "validated": validated,
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
    try:
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
    finally:
        close = getattr(read_rss, "close", None)
        if close is not None:
            close()
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


def _cuda_memory_bounded(
    device,
    memory_growth,
    allocated_before,
    allocated_after,
    peak_allocated,
):
    return (
        _memory_growth_bounded(device, memory_growth, None)
        and type(allocated_before) is int
        and type(allocated_after) is int
        and type(peak_allocated) is int
        and peak_allocated >= max(allocated_before, allocated_after)
    )


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
                    "publication_url",
                    "size_bytes",
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
    reference_environment = (
        baseline.get("environment") if isinstance(baseline, dict) else None
    )
    if not isinstance(reference_environment, dict) or not isinstance(environment, dict):
        return False
    try:
        reference = privacy_preserving_host_identity(reference_environment)
        candidate = privacy_preserving_host_identity(
            environment, salt=reference["salt"]
        )
    except KeyError, TypeError, ValueError:
        return False
    return candidate == reference and timing_runtime_identity_matches(
        reference_environment, environment
    )


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


def _command_with_status(*command):
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None, 127
    return (
        result.stdout.strip() if result.returncode == 0 else None,
        int(result.returncode),
    )


def _command_text(*command):
    return _command_with_status(*command)[0]


def _cpu_contract_environment():
    affinity = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    parsable, affinity_topology_status = _command_with_status(
        "lscpu",
        "-p=CPU,CORE,SOCKET",
    )
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
        value, physical_fallback_status = _command_with_status(
            "sysctl",
            "-n",
            "hw.physicalcpu",
        )
        try:
            physical = int(value.strip()) if value else None
        except ValueError:
            physical = None
    else:
        physical_fallback_status = None
    cpu_topology, cpu_topology_status = _command_with_status(
        "lscpu",
        "-p=CORE,SOCKET",
    )
    return {
        "cpu_affinity": affinity,
        "cpu_count_physical_affinity": physical,
        "cpu_topology": cpu_topology,
        "cpu_affinity_topology_command_status": affinity_topology_status,
        "cpu_topology_command_status": cpu_topology_status,
        "cpu_physical_fallback_command_status": physical_fallback_status,
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
    cpu, cpu_status = _command_with_status("lscpu")
    devices = []
    if torch.cuda.is_available():
        topology, topology_status = _command_with_status("nvidia-smi", "topo", "-m")
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
    else:
        topology = None
        topology_status = None
    cpu_contract = _cpu_contract_environment()
    return {
        "host_contract": capture_host_contract(torch),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "devices": devices,
        "cpu_count": os.cpu_count(),
        **cpu_contract,
        "cpu_model": cpu,
        "cpu_model_command_status": cpu_status,
        "gpu_topology": topology,
        "gpu_topology_command_status": topology_status,
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
    if name in REGION_INVARIANCE_GATES:
        region_count = 1 if name.endswith("-1") else 32
        size = (4, 4, 0)
        resolution = 4
        space = gmes.Cartesian(size, resolution)
        geometry = [
            gmes.DefaultMedium(material_from_name("dielectric", gmes)),
            *(
                gmes.Block(
                    material_from_name("drude-1", gmes),
                    center=(0, 0, 0),
                    size=(3, 3, 1),
                )
                for _ in range(region_count)
            ),
        ]
        spec = {
            "name": name,
            "recipe": "equivalent-overlap",
            "size": list(size),
            "resolution": resolution,
            "material": "drude-1",
            "complex": False,
            "equivalence_contract_id": "material-region-launch-invariance-v1",
            "equivalence_group": "overlapping-identical-drude-block-v1",
            "geometry_region_count": region_count,
            "geometry_object_count": len(geometry),
        }
        return spec, space, geometry, (), None
    if name in SPECIAL_CASES:
        space, geometry = build_dm2_case(name, gmes)
        three_dimensional = name.endswith("3d")
        spec = {
            "name": name,
            "recipe": "all-material",
            "size": [12, 6, 4 if three_dimensional else 0],
            "resolution": 2 if three_dimensional else 4,
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
    result = float(value.cpu())
    if not math.isfinite(result):
        raise ValueError("simulation fields contain non-finite values")
    return result


def _add_checkpoint_values(value, prefix, tensors):
    if isinstance(value, torch.Tensor):
        if prefix in tensors:
            raise ValueError("checkpoint tensor path is duplicated")
        tensors[prefix] = value
    elif isinstance(value, dict):
        for name, item in value.items():
            if not isinstance(name, str) or not name:
                raise ValueError("checkpoint tensor path is malformed")
            _add_checkpoint_values(
                item, f"{prefix}.{name}" if prefix else str(name), tensors
            )
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _add_checkpoint_values(item, f"{prefix}[{index}]", tensors)
    else:
        raise ValueError("checkpoint dynamic state contains a non-tensor leaf")


def _add_simulation_checkpoint_tensors(value, prefix, tensors):
    if (
        not isinstance(value, dict)
        or set(value) != SIMULATION_CHECKPOINT_KEYS
        or value.get("format") != "gmes.torch.simulation"
        or value.get("version") != 1
        or not isinstance(value.get("metadata"), dict)
        or not isinstance(value.get("state"), dict)
        or not isinstance(value.get("auxiliaries"), tuple)
        or not isinstance(value.get("probes"), dict)
    ):
        raise ValueError("checkpoint dynamic state closure differs")
    dynamic_state_names = {
        name
        for name in value["state"]
        if isinstance(name, str) and not name.startswith("plan.")
    }
    if not STATE_FIELD_NAMES <= dynamic_state_names:
        raise ValueError("checkpoint field closure differs")
    for name, tensor in value["state"].items():
        if not isinstance(name, str) or not name:
            raise ValueError("checkpoint state name is malformed")
        if name.startswith("plan."):
            continue
        qualified = name if not prefix else f"{prefix}.state.{name}"
        count = len(tensors)
        _add_checkpoint_values(tensor, qualified, tensors)
        if len(tensors) == count:
            raise ValueError("checkpoint dynamic state entry has no tensors")
    auxiliaries = value["auxiliaries"]
    for index, auxiliary in enumerate(auxiliaries):
        qualified = f"{prefix}.auxiliaries" if prefix else "auxiliaries"
        _add_simulation_checkpoint_tensors(auxiliary, f"{qualified}[{index}]", tensors)
    qualified = f"{prefix}.probes" if prefix else "probes"
    _add_checkpoint_values(value["probes"], qualified, tensors)


def _add_live_simulation_tensor_names(simulation, prefix, names):
    state = simulation.state.state_dict()
    probes = simulation.probes.state_dict()
    auxiliaries = simulation.sources.auxiliaries
    if (
        not isinstance(state, dict)
        or not isinstance(probes, dict)
        or not isinstance(auxiliaries, tuple)
    ):
        raise ValueError("live simulation dynamic state closure differs")
    dynamic_state_names = {
        name for name in state if isinstance(name, str) and not name.startswith("plan.")
    }
    if not STATE_FIELD_NAMES <= dynamic_state_names:
        raise ValueError("live simulation field closure differs")
    for name, tensor in state.items():
        if not isinstance(name, str) or not name:
            raise ValueError("live simulation state name is malformed")
        if name.startswith("plan."):
            continue
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("live simulation state contains a non-tensor")
        qualified = name if not prefix else f"{prefix}.state.{name}"
        if qualified in names:
            raise ValueError("live simulation tensor path is duplicated")
        names.add(qualified)
    probe_prefix = f"{prefix}.probes" if prefix else "probes"
    for name, tensor in probes.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(tensor, torch.Tensor)
        ):
            raise ValueError("live simulation probe state is malformed")
        qualified = f"{probe_prefix}.{name}"
        if qualified in names:
            raise ValueError("live simulation tensor path is duplicated")
        names.add(qualified)
    for index, auxiliary in enumerate(auxiliaries):
        auxiliary_prefix = f"{prefix}.auxiliaries" if prefix else "auxiliaries"
        _add_live_simulation_tensor_names(
            auxiliary, f"{auxiliary_prefix}[{index}]", names
        )


def _simulation_dynamic_tensor_names(simulation):
    names = set()
    _add_live_simulation_tensor_names(simulation, "", names)
    return sorted(names)


def _checkpoint_dynamic_tensors(checkpoint):
    tensors = {}
    _add_simulation_checkpoint_tensors(checkpoint, "", tensors)
    return tensors


def _dynamic_state_finiteness(checkpoints, changed_buffers, expected_buffers):
    field_names = {name.lower() for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
    if (
        not isinstance(expected_buffers, list)
        or not all(isinstance(name, str) and bool(name) for name in expected_buffers)
        or expected_buffers != sorted(set(expected_buffers))
    ):
        raise ValueError("expected dynamic tensor inventory is malformed")
    states = {
        stage: _checkpoint_dynamic_tensors(checkpoint)
        for stage, checkpoint in checkpoints.items()
    }
    if not states:
        raise ValueError("checkpoint dynamic state closure differs")
    state_values = list(states.values())
    state_keys = set(state_values[0])
    if state_keys != set(expected_buffers):
        raise ValueError("checkpoint dynamic tensor inventory differs")
    if any(set(state) != state_keys for state in state_values[1:]):
        raise ValueError("checkpoint dynamic tensor keys changed")
    first = state_values[0]
    observed_changed = {
        name
        for name in state_keys
        if any(not torch.equal(first[name], state[name]) for state in state_values[1:])
    }
    if not set(changed_buffers) <= observed_changed:
        raise ValueError(
            "timed changed-buffer summary is outside observed dynamic state"
        )
    tracked = sorted(state_keys)
    if not field_names <= set(tracked):
        raise ValueError("checkpoint field closure differs")
    stages = {}
    for stage, state in states.items():
        floating_buffers = 0
        floating_elements = 0
        nonfinite_elements = 0
        for name in tracked:
            value = state[name]
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"{stage} checkpoint buffer {name!r} is not a tensor")
            if value.is_floating_point() or value.is_complex():
                floating_buffers += 1
                floating_elements += value.numel()
                nonfinite_elements += int(
                    torch.count_nonzero(~torch.isfinite(value)).cpu()
                )
        stages[stage] = {
            "floating_or_complex_buffer_count": floating_buffers,
            "floating_or_complex_element_count": floating_elements,
            "nonfinite_element_count": nonfinite_elements,
            "finite": nonfinite_elements == 0,
        }
    return {
        "contract_id": STATE_FINITENESS_CONTRACT_ID,
        "tracked_buffers": tracked,
        "stages": stages,
        "passed": all(record["finite"] for record in stages.values()),
    }


def _cuda_state_finiteness_valid(result):
    state_progress = result.get("state_progress")
    finiteness = result.get("state_finiteness")
    if not isinstance(state_progress, dict) or not isinstance(finiteness, dict):
        return False
    changed = state_progress.get("changed_buffers")
    fields_changed = state_progress.get("fields_changed")
    if (
        not isinstance(changed, list)
        or not all(isinstance(name, str) and bool(name) for name in changed)
        or changed != sorted(set(changed))
        or fields_changed != sorted(set(changed) & STATE_FIELD_NAMES)
        or state_progress.get("all_fields_changed")
        is not (STATE_FIELD_NAMES <= set(changed))
        or state_progress.get("all_fields_changed") is not True
        or set(finiteness) != {"contract_id", "tracked_buffers", "stages", "passed"}
    ):
        return False
    tracked = finiteness["tracked_buffers"]
    if (
        finiteness["contract_id"] != STATE_FINITENESS_CONTRACT_ID
        or not isinstance(tracked, list)
        or not all(isinstance(name, str) and bool(name) for name in tracked)
        or tracked != sorted(set(tracked))
        or not tracked
        or any(name.startswith("plan.") or ".state.plan." in name for name in tracked)
        or not (set(changed) | STATE_FIELD_NAMES) <= set(tracked)
        or finiteness["passed"] is not True
    ):
        return False
    stages = finiteness["stages"]
    if not isinstance(stages, dict) or set(stages) != set(STATE_FINITENESS_STAGES):
        return False
    runtime = result.get("runtime")
    profiler = result.get("profiler")
    field_sizes = (
        profiler.get("field_buffer_sizes_bytes") if isinstance(profiler, dict) else None
    )
    precision = runtime.get("precision") if isinstance(runtime, dict) else None
    try:
        element_size = np.dtype(precision).itemsize
    except TypeError, ValueError:
        return False
    field_size_keys = tuple(
        f"state.{name}" for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    )
    if not isinstance(field_sizes, dict) or any(
        type(field_sizes.get(name)) is not int
        or field_sizes[name] <= 0
        or field_sizes[name] % element_size != 0
        for name in field_size_keys
    ):
        return False
    minimum_field_elements = (
        sum(field_sizes[name] for name in field_size_keys) // element_size
    )
    sizes = set()
    for stage in STATE_FINITENESS_STAGES:
        record = stages[stage]
        if (
            not isinstance(record, dict)
            or set(record)
            != {
                "floating_or_complex_buffer_count",
                "floating_or_complex_element_count",
                "nonfinite_element_count",
                "finite",
            }
            or type(record["floating_or_complex_buffer_count"]) is not int
            or not len(STATE_FIELD_NAMES)
            <= record["floating_or_complex_buffer_count"]
            <= len(tracked)
            or type(record["floating_or_complex_element_count"]) is not int
            or record["floating_or_complex_element_count"] < minimum_field_elements
            or type(record["nonfinite_element_count"]) is not int
            or record["nonfinite_element_count"] != 0
            or record["finite"] is not True
        ):
            return False
        sizes.add(
            (
                record["floating_or_complex_buffer_count"],
                record["floating_or_complex_element_count"],
            )
        )
    return len(sizes) == 1


def _canonical_sha256(value):
    rendered = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(rendered).hexdigest()


def _effective_material_plan(simulation):
    """Return raw effective targets and coefficients independent of region IDs."""
    records = []
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        component = simulation.plan.components[name]
        buckets = []
        for bucket in component.buckets:
            coefficient_rows = bucket.region_coefficient_indices[
                bucket.target_region_indices
            ]
            buckets.append(
                {
                    "signature": {
                        "component": bucket.signature.component,
                        "model": bucket.signature.model,
                        "precision": bucket.signature.precision,
                        "state_shape": list(bucket.signature.state_shape),
                    },
                    "coefficient_names": list(bucket.coefficient_names),
                    "targets": bucket.targets.tolist(),
                    "target_coefficients": bucket.coefficient_table[
                        coefficient_rows
                    ].tolist(),
                    "cell_coefficient_names": list(bucket.cell_coefficient_names),
                    "cell_coefficients": bucket.cell_coefficients.tolist(),
                }
            )
        records.append(
            {
                "component": name,
                "shape": list(component.shape),
                "dense_inverse": component.dense_inverse.tolist(),
                "constant_targets": component.constant_targets.tolist(),
                "constant_values": component.constant_values.tolist(),
                "buckets": buckets,
            }
        )
    return records


def _region_equivalence_record(simulation, spec, geometry):
    raw_plan = _effective_material_plan(simulation)
    return {
        "contract_id": spec["equivalence_contract_id"],
        "equivalence_group": spec["equivalence_group"],
        "geometry_region_count": spec["geometry_region_count"],
        "geometry_object_count": len(geometry),
        "material_compute_launches_per_step": sum(
            component.launch_count for component in simulation.plan.components.values()
        ),
        "effective_material_plan": raw_plan,
        "effective_material_plan_sha256": _canonical_sha256(raw_plan),
    }


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
    policy_write_operations = Counter()
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
        exact_name = event.get("name", "")
        if exact_name in POLICY_WRITE_OPERATIONS.values():
            policy_write_operations[exact_name] += 1
        name = exact_name.lower()
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
        "policy_write_operations": {
            name: policy_write_operations[name]
            for name in POLICY_WRITE_OPERATIONS.values()
        },
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


def _positive_finite(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value > 0
    )


def _boundary_execution_diagnostics_match(result):
    diagnostics = result.get("diagnostics")
    boundaries = (
        diagnostics.get("boundaries") if isinstance(diagnostics, dict) else None
    )
    return (
        isinstance(boundaries, dict)
        and set(boundaries)
        == {"scheduling", "execution_representation", "paired_real_scratch_bytes"}
        and boundaries["scheduling"] == "external"
        and boundaries["execution_representation"]
        == gmes.torch_fdtd.BOUNDARY_SYNC_REPRESENTATION
        and type(boundaries["paired_real_scratch_bytes"]) is int
        and boundaries["paired_real_scratch_bytes"] >= 0
    )


def _tuning_timing_errors(result, label):
    errors = []
    if not _boundary_execution_diagnostics_match(result):
        errors.append(
            f"{label} boundary execution diagnostics differ from the solver ABI"
        )
    contract = result.get("benchmark_contract")
    measurements = result.get("measurements")
    summary = measurements.get("advance") if isinstance(measurements, dict) else None
    if not isinstance(contract, dict) or not isinstance(summary, dict):
        errors.append(f"{label} timing contract is absent")
        return errors
    raw = summary.get("raw_seconds")
    repetitions = contract.get("repetitions")
    steps = contract.get("steps_per_repeat")
    if (
        type(repetitions) is not int
        or repetitions < 1
        or type(steps) is not int
        or steps < 1
        or not isinstance(raw, list)
        or len(raw) != repetitions
        or not all(_positive_finite(value) for value in raw)
    ):
        errors.append(f"{label} raw timing samples are invalid")
        return errors
    middle = median(raw)
    relative_mad = median(abs(value - middle) for value in raw) / middle
    expected = {
        "median_seconds": middle,
        "relative_mad": relative_mad,
        "repetitions": repetitions,
        "steps_per_repeat": steps,
        "seconds_per_step": middle / steps,
    }
    for name, value in expected.items():
        observed = summary.get(name)
        if type(value) is int:
            matches = observed == value and type(observed) is int
        else:
            matches = _positive_finite(observed) or (
                name == "relative_mad"
                and not isinstance(observed, bool)
                and isinstance(observed, (int, float))
                and math.isfinite(observed)
                and observed == 0
            )
            matches = matches and math.isclose(
                float(observed), value, rel_tol=1e-12, abs_tol=1e-15
            )
        if not matches:
            errors.append(f"{label} {name} does not match raw timing samples")
    return errors


def _cuda_trace_memory_errors(result, label):
    errors = []
    profiler = result.get("profiler")
    memory = result.get("memory")
    contract = result.get("benchmark_contract")
    if not isinstance(profiler, dict):
        errors.append(f"{label} profiler summary is absent")
    else:
        if not _profiler_trace_matches(profiler):
            errors.append(f"{label} saved trace does not match its summary")
        if (
            type(profiler.get("profile_steps")) is not int
            or not isinstance(contract, dict)
            or profiler["profile_steps"] != contract.get("profile_steps")
        ):
            errors.append(f"{label} profiler step count differs")
        if (
            type(profiler.get("kernel_launches")) is not int
            or profiler["kernel_launches"] <= 0
        ):
            errors.append(f"{label} CUDA kernel launch count is absent")
        if (
            profiler.get("host_to_device_events") != 0
            or profiler.get("device_to_host_events") != 0
        ):
            errors.append(f"{label} steady-state host/device transfers are nonzero")
    if not isinstance(memory, dict) or memory.get("bounded") is not True:
        errors.append(f"{label} CUDA memory is not bounded")
    else:
        before = memory.get("cuda_allocated_before_bytes")
        after = memory.get("cuda_allocated_after_bytes")
        growth = memory.get("cuda_allocated_growth_bytes")
        peak = memory.get("cuda_peak_allocated_bytes")
        reserved = memory.get("cuda_peak_reserved_bytes")
        if (
            type(before) is not int
            or before < 0
            or type(after) is not int
            or after < 0
            or type(growth) is not int
            or growth != after - before
            or growth > 1024 * 1024
            or type(peak) is not int
            or peak <= 0
            or peak < max(before, after)
            or type(reserved) is not int
            or reserved < peak
        ):
            errors.append(f"{label} CUDA allocation metrics are invalid")
    return errors


def _cuda_environment_errors(environment):
    errors = []
    if not isinstance(environment, dict):
        return ["single-GPU CUDA environment is absent"]
    for name in ("platform", "python", "torch", "cuda_runtime"):
        if not isinstance(environment.get(name), str) or not environment[name]:
            errors.append(f"single-GPU CUDA environment {name} is absent")
    if not host_contract_complete(environment.get("host_contract")):
        errors.append("single-GPU CUDA host contract is absent")
    topology = environment.get("gpu_topology")
    if not isinstance(topology, str) or not topology.strip():
        errors.append("single-GPU CUDA topology is absent")
    statuses = {
        name: value
        for name, value in environment.items()
        if name.endswith("_command_status") and value is not None
    }
    if (
        type(environment.get("gpu_topology_command_status")) is not int
        or environment["gpu_topology_command_status"] != 0
        or not statuses
        or any(type(value) is not int or value != 0 for value in statuses.values())
    ):
        errors.append("single-GPU CUDA environment command status is incomplete")
    devices = environment.get("devices")
    if (
        not isinstance(devices, list)
        or not devices
        or not isinstance(devices[0], dict)
        or devices[0].get("index") != 0
        or any(
            not isinstance(device, dict)
            or set(device)
            != {
                "index",
                "name",
                "memory_bytes",
                "capability",
                "multiprocessors",
            }
            or type(device["index"]) is not int
            or not isinstance(device["name"], str)
            or not device["name"]
            or type(device["memory_bytes"]) is not int
            or device["memory_bytes"] <= 0
            or not isinstance(device["capability"], list)
            or len(device["capability"]) != 2
            or any(
                type(value) is not int or value < 0 for value in device["capability"]
            )
            or device["capability"][0] <= 0
            or type(device["multiprocessors"]) is not int
            or device["multiprocessors"] <= 0
            for device in devices
        )
    ):
        errors.append("single-GPU CUDA device inventory is incomplete")
    return errors


def _cuda_correctness_source_descriptor(index):
    if not isinstance(index, dict):
        return None
    source = index.get("source_artifact")
    evidence = index.get("candidate_evidence")
    if (
        not isinstance(source, dict)
        or set(source) != set(CUDA_CORRECTNESS_SOURCE_DESCRIPTOR_KEYS)
        or not isinstance(evidence, dict)
    ):
        return None
    candidate = source.get("candidate_evidence")
    expected_candidate = {
        name: evidence.get(name) for name in CUDA_CORRECTNESS_SOURCE_CANDIDATE_KEYS
    }
    if (
        not isinstance(candidate, dict)
        or set(candidate) != set(CUDA_CORRECTNESS_SOURCE_CANDIDATE_KEYS)
        or candidate != expected_candidate
        or not isinstance(candidate["candidate_git_commit"], str)
        or re.fullmatch(r"[0-9a-f]{40}", candidate["candidate_git_commit"]) is None
        or candidate["candidate_git_status"] != ""
        or not isinstance(candidate["manifest_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", candidate["manifest_sha256"]) is None
    ):
        return None
    path = source.get("path")
    if (
        not isinstance(path, str)
        or not path
        or "\\" in path
        or "\x00" in path
        or (len(path) >= 2 and path[0].isalpha() and path[1] == ":")
        or any(part in {"", ".", ".."} for part in path.split("/"))
        or PurePosixPath(path).is_absolute()
        or PurePosixPath(path).as_posix() != path
        or not isinstance(source.get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", source["sha256"]) is None
        or type(source.get("size_bytes")) is not int
        or source["size_bytes"] <= 0
        or source.get("media_type") != "application/json"
    ):
        return None
    try:
        index_document = dict(index)
        del index_document["source_artifact"]
        source_bytes = (
            json.dumps(
                index_document,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode()
    except KeyError, TypeError, ValueError:
        return None
    if (
        source["size_bytes"] != len(source_bytes)
        or source["sha256"] != hashlib.sha256(source_bytes).hexdigest()
    ):
        return None
    return {
        "path": path,
        "sha256": source["sha256"],
        "size_bytes": source["size_bytes"],
        "media_type": source["media_type"],
        "candidate_evidence": dict(candidate),
    }


def _cuda_correctness_descriptor_records(indexes):
    records = []
    errors = []
    descriptors = []
    if not isinstance(indexes, (list, tuple)):
        return records, ["CUDA correctness descriptor index closure is invalid"]
    for position, index in enumerate(indexes):
        runtime_mode = index.get("runtime_mode") if isinstance(index, dict) else None
        descriptor = _cuda_correctness_source_descriptor(index)
        records.append(
            {
                "runtime_mode": runtime_mode,
                "source_artifact": descriptor,
            }
        )
        if descriptor is None:
            errors.append(
                f"CUDA correctness source artifact descriptor {position} is invalid"
            )
        else:
            descriptors.append(descriptor)
    if len(indexes) != len(CUDA_CORRECTNESS_RUNTIME_MODES):
        errors.append("CUDA correctness descriptor index closure is invalid")
    if len(descriptors) == len(CUDA_CORRECTNESS_RUNTIME_MODES):
        if len({descriptor["path"] for descriptor in descriptors}) != len(descriptors):
            errors.append("CUDA correctness source artifact paths are not distinct")
        if len({descriptor["sha256"] for descriptor in descriptors}) != len(
            descriptors
        ):
            errors.append("CUDA correctness source artifact digests are not distinct")
    return records, errors


def _load_cuda_correctness_indexes(paths, receipt_paths, manifest, expected_evidence):
    indexes = []
    errors = []
    if paths is None:
        return [], ["CUDA correctness eager/graph indexes are absent"]
    if not isinstance(paths, (list, tuple)) or len(paths) != 2:
        return [], ["CUDA correctness requires exactly eager and graph indexes"]
    if not isinstance(receipt_paths, (list, tuple)) or len(receipt_paths) != 2:
        return [], ["CUDA correctness requires exactly eager and graph receipts"]
    for expected_mode, path, receipt_path in zip(
        CUDA_CORRECTNESS_RUNTIME_MODES,
        paths,
        receipt_paths,
        strict=True,
    ):
        try:
            index = load_correctness_evidence_index(
                path,
                manifest,
                expected_evidence,
                descriptor_root=Path(path).resolve().parent,
                runtime_receipt=receipt_path,
            )
        except (KeyError, OSError, TypeError, ValueError) as error:
            errors.append(f"{expected_mode['graph_mode']} CUDA correctness: {error}")
            continue
        if index.get("runtime_mode") != expected_mode:
            errors.append(
                f"{expected_mode['graph_mode']} CUDA correctness runtime mode differs"
            )
        try:
            receipt = load_runtime_publication_receipt(
                receipt_path,
                manifest,
                expected_evidence,
                index.get("runtime_mode"),
                _runtime_receipt_candidates(index.get("artifacts", ())),
            )
        except (KeyError, OSError, TypeError, ValueError) as error:
            errors.append(
                f"{expected_mode['graph_mode']} CUDA correctness receipt: {error}"
            )
            continue
        if not correctness_binding_complete(
            index,
            manifest,
            expected_evidence,
            runtime_receipt=receipt,
            require_source_artifact=True,
        ):
            errors.append(
                f"{expected_mode['graph_mode']} CUDA correctness binding is incomplete"
            )
        indexes.append(index)
    if len(indexes) != len(CUDA_CORRECTNESS_RUNTIME_MODES):
        errors.append("CUDA correctness eager/graph index closure differs")
    _records, descriptor_errors = _cuda_correctness_descriptor_records(indexes)
    for error in descriptor_errors:
        if error not in errors:
            errors.append(error)
    return indexes, errors


def _cuda_suite_gate(
    results,
    correctness_indexes,
    correctness_errors,
    manifest,
    environment,
):
    environment_errors = _cuda_environment_errors(environment)
    errors = [*correctness_errors, *environment_errors]
    names = [result.get("workload", {}).get("name") for result in results]
    if names != list(CUDA_GATES):
        errors.append("single-GPU CUDA case closure differs")
    maximum_relative_mad = manifest["performance_gates"]["cpu_acceptance"][
        "statistics"
    ]["max_relative_mad"]
    reference = manifest["reference"]
    expected_benchmark_contract = {
        "initializer": FIELD_INITIALIZER,
        "seed": reference["seed"],
        "field_scale": reference["field_scale"],
        "warmup_steps": reference["performance_warmup_steps"],
        "steps_per_repeat": reference["performance_steps_per_repeat"],
        "repetitions": reference["performance_repetitions"],
        "profile_steps": reference["performance_profile_steps"],
        "timer": "time.perf_counter",
        "sample_start": "independently-restored-pre-warmup-state",
    }
    for result in results:
        workload = result.get("workload")
        runtime = result.get("runtime")
        name = workload.get("name") if isinstance(workload, dict) else None
        label = f"single-GPU CUDA {name or 'case'}"
        try:
            expected_workload = find_case(manifest, name)
        except KeyError, TypeError, ValueError:
            expected_workload = None
        if expected_workload is None or not _json_contract_equal(
            workload, expected_workload
        ):
            errors.append(f"{label} manifest workload differs")
        if result.get("schema_version") != 2 or result.get("backend") != "torch":
            errors.append(f"{label} schema/backend contract differs")
        if not _json_contract_equal(
            result.get("benchmark_contract"), expected_benchmark_contract
        ):
            errors.append(f"{label} frozen benchmark contract differs")
        runtime_statuses = (
            {
                key: value
                for key, value in runtime.items()
                if key.endswith("_command_status") and value is not None
            }
            if isinstance(runtime, dict)
            else {}
        )
        compile_key = (
            runtime.get("compile_cache_key") if isinstance(runtime, dict) else None
        )
        expected_precision = CUDA_PERFORMANCE_PRECISION_BY_CASE.get(name)
        expected_storage_dtype = (
            f"torch.{expected_precision}" if expected_precision is not None else None
        )
        if (
            not isinstance(runtime, dict)
            or runtime.get("device") != "cuda:0"
            or runtime.get("precision") != expected_precision
            or runtime.get("compile_policy") != "compile"
            or runtime.get("compile_mode") != "default"
            or runtime.get("explicit_cuda_graphs") is not False
            or runtime.get("execution_policy") != "auto"
            or runtime.get("experimental_dispersive_grouping") is not False
            or runtime.get("experimental_dispersive_grouping_scope") != "combined"
            or runtime.get("threads") != 1
            or runtime.get("interop_threads") != 1
            or runtime.get("paired_real") is not False
            or runtime.get("field_storage_representation") != "real-v1"
            or runtime.get("field_storage_channels") != 1
            or runtime.get("field_storage_dtype") != expected_storage_dtype
            or not isinstance(compile_key, str)
            or len(compile_key) != 64
            or any(character not in "0123456789abcdef" for character in compile_key)
            or not runtime_statuses
            or any(
                type(value) is not int or value != 0
                for value in runtime_statuses.values()
            )
        ):
            errors.append(f"{label} official runtime/storage contract differs")
        if not _cuda_state_finiteness_valid(result):
            errors.append(f"{label} dynamic state finiteness contract failed")
        timing_errors = _tuning_timing_errors(result, label)
        errors.extend(timing_errors)
        summary = result.get("measurements", {}).get("advance", {})
        relative_mad = summary.get("relative_mad")
        if (
            isinstance(relative_mad, bool)
            or not isinstance(relative_mad, (int, float))
            or not math.isfinite(relative_mad)
            or relative_mad < 0
            or relative_mad > maximum_relative_mad
        ):
            errors.append(f"{label} relative MAD exceeds the frozen limit")
        errors.extend(_cuda_trace_memory_errors(result, label))
        acceptance = result.get("acceptance")
        if (
            not isinstance(acceptance, dict)
            or set(acceptance) != set(RUNTIME_ACCEPTANCE_KEYS)
            or any(acceptance.get(name) is not True for name in RUNTIME_ACCEPTANCE_KEYS)
        ):
            errors.append(f"{label} runtime acceptance failed")
    correctness_records, descriptor_errors = _cuda_correctness_descriptor_records(
        correctness_indexes
    )
    for error in descriptor_errors:
        if error not in errors:
            errors.append(error)
    correctness_modes = [record["runtime_mode"] for record in correctness_records]
    if correctness_modes != list(CUDA_CORRECTNESS_RUNTIME_MODES):
        errors.append("CUDA correctness runtime mode closure differs")
    correctness_bound = (
        not correctness_errors
        and not descriptor_errors
        and len(correctness_records) == len(CUDA_CORRECTNESS_RUNTIME_MODES)
        and correctness_modes == list(CUDA_CORRECTNESS_RUNTIME_MODES)
    )
    return {
        "contract_id": CUDA_SUITE_CONTRACT_ID,
        "required_cases": list(CUDA_GATES),
        "required_case_precisions": [
            {"case": name, "precision": CUDA_PERFORMANCE_PRECISION_BY_CASE[name]}
            for name in CUDA_GATES
        ],
        "reviewed_precision_limitations": [dict(CUDA_PRECISION_LIMITATION_REVIEW)],
        "required_correctness_runtime_modes": list(CUDA_CORRECTNESS_RUNTIME_MODES),
        "case_closure_complete": names == list(CUDA_GATES),
        "environment_complete": not environment_errors,
        "correctness_index_count": len(correctness_indexes),
        "correctness_indexes": correctness_records,
        "correctness_evidence_bound": correctness_bound,
        "timing_statistics": "raw-median-relative-mad-v1",
        "trace_contract": "sha256-bound-zero-transfer-kernel-count-v1",
        "errors": errors,
        "passed": not errors and correctness_bound,
    }


def _paired_real_cuda_gate(results):
    errors = []
    cases = [result.get("workload", {}).get("name") for result in results]
    if cases != list(PAIRED_REAL_GATES):
        errors.append("paired-real case closure differs")
    for result in results:
        workload = result.get("workload")
        runtime = result.get("runtime")
        label = (
            f"paired-real {workload.get('name')}"
            if isinstance(workload, dict)
            else "paired-real case"
        )
        if (
            not isinstance(workload, dict)
            or workload.get("complex") is not True
            or not isinstance(runtime, dict)
            or not str(runtime.get("device", "")).startswith("cuda")
            or runtime.get("precision") != "float32"
            or runtime.get("paired_real") is not True
            or runtime.get("field_storage_representation") != "paired-real-v1"
            or runtime.get("field_storage_channels") != 2
            or runtime.get("field_storage_dtype") != "torch.float32"
        ):
            errors.append(f"{label} does not use CUDA float32 paired-real storage")
        errors.extend(_tuning_timing_errors(result, label))
        errors.extend(_cuda_trace_memory_errors(result, label))
        if result.get("acceptance", {}).get("passed") is not True:
            errors.append(f"{label} runtime acceptance failed")
    return {
        "contract_id": "cuda-float32-paired-real-tuning-v1",
        "required_cases": list(PAIRED_REAL_GATES),
        "timing_statistics": "raw-median-relative-mad-v1",
        "trace_contract": "sha256-bound-zero-transfer-kernel-count-v1",
        "errors": errors,
        "passed": not errors,
    }


def _region_invariance_gate(results):
    errors = []
    names = [result.get("workload", {}).get("name") for result in results]
    if names != list(REGION_INVARIANCE_GATES):
        errors.append("equivalent-region case closure differs")
    records = []
    for result in results:
        workload = result.get("workload")
        evidence = result.get("region_equivalence")
        diagnostics = result.get("diagnostics")
        profiler = result.get("profiler")
        label = (
            f"equivalent-region {workload.get('name')}"
            if isinstance(workload, dict)
            else "equivalent-region case"
        )
        if not all(
            isinstance(value, dict)
            for value in (workload, evidence, diagnostics, profiler)
        ):
            errors.append(f"{label} evidence records are absent")
            continue
        plan = evidence.get("effective_material_plan")
        region_count = evidence.get("geometry_region_count")
        object_count = evidence.get("geometry_object_count")
        launches = evidence.get("material_compute_launches_per_step")
        material_plan = diagnostics.get("material_plan")
        diagnostic_launches = (
            sum(item.get("launches", -1) for item in material_plan)
            if isinstance(material_plan, (list, tuple))
            and all(isinstance(item, dict) for item in material_plan)
            else None
        )
        if (
            evidence.get("contract_id") != "material-region-launch-invariance-v1"
            or evidence.get("equivalence_group")
            != "overlapping-identical-drude-block-v1"
            or type(region_count) is not int
            or region_count < 1
            or workload.get("geometry_region_count") != region_count
            or type(object_count) is not int
            or object_count != region_count + 1
            or workload.get("geometry_object_count") != object_count
            or not isinstance(plan, list)
            or not plan
            or evidence.get("effective_material_plan_sha256") != _canonical_sha256(plan)
            or type(launches) is not int
            or launches < 1
            or launches != diagnostic_launches
        ):
            errors.append(f"{label} raw material plan binding is invalid")
        errors.extend(_tuning_timing_errors(result, label))
        errors.extend(_cuda_trace_memory_errors(result, label))
        if result.get("acceptance", {}).get("passed") is not True:
            errors.append(f"{label} runtime acceptance failed")
        records.append(
            {
                "region_count": region_count,
                "plan": plan,
                "plan_sha256": evidence.get("effective_material_plan_sha256"),
                "material_launches": launches,
                "profile_steps": profiler.get("profile_steps"),
                "kernel_launches": profiler.get("kernel_launches"),
            }
        )
    if len(records) == 2:
        baseline, expanded = records
        if not (
            type(baseline["region_count"]) is int
            and type(expanded["region_count"]) is int
            and expanded["region_count"] > baseline["region_count"]
        ):
            errors.append("equivalent-region input region count did not increase")
        if baseline["plan"] != expanded["plan"]:
            errors.append("equivalent-region effective material plans differ")
        if baseline["material_launches"] != expanded["material_launches"]:
            errors.append("equivalent-region material compute launch counts differ")
        if (
            baseline["profile_steps"] != expanded["profile_steps"]
            or baseline["kernel_launches"] != expanded["kernel_launches"]
        ):
            errors.append("equivalent-region profiled CUDA launch counts differ")
    return {
        "contract_id": "material-region-launch-invariance-v1",
        "required_cases": list(REGION_INVARIANCE_GATES),
        "errors": errors,
        "passed": not errors,
    }


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
    fixed_boundary_buffer_sizes = profiler.get("fixed_boundary_buffer_sizes_bytes")
    fixed_boundary_buffer_sizes_valid = isinstance(
        fixed_boundary_buffer_sizes, dict
    ) and all(
        isinstance(name, str) and name and type(size) is int and size > 0
        for name, size in fixed_boundary_buffer_sizes.items()
    )
    check(
        "fixed_boundary_buffer_sizes_present",
        fixed_boundary_buffer_sizes_valid,
        "nonzero allocation traces require the registered fixed boundary "
        "buffer byte sizes",
    )
    histogram_sizes = {int(size) for size in histogram}
    no_field_sized_allocations = (
        field_buffer_sizes_valid
        and histogram_sizes.isdisjoint(field_buffer_sizes.values())
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
        "provenance_fixed_boundary_buffers_match": (
            allocation_provenance.get("fixed_boundary_buffer_sizes_bytes")
            == fixed_boundary_buffer_sizes
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
                and "fixed_temporary_buffer" not in item
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
    check(
        "no_field_buffer_sized_allocations",
        field_buffer_sizes_valid
        and fixed_boundary_buffer_sizes_valid
        and no_field_sized_allocations,
        "an allocation size matches a live field or domain buffer; preallocated "
        "boundary scratch never authorizes a dynamic allocation",
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
    if provider.get("name") == "proc-self-statm-preadv-v1":
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
    if not _boundary_execution_diagnostics_match(candidate):
        errors.append("CPU boundary execution diagnostics differ from the solver ABI")
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


def _fixed_boundary_buffer_sizes_bytes(simulation):
    """Inventory direct, registered nonpersistent boundary scratch buffers."""
    result = {}
    visited = set()

    def visit_state(state, prefix):
        if not isinstance(state, torch.nn.Module):
            return
        for name, tensor in state.named_buffers(recurse=False):
            if (
                not name.startswith("_boundary_")
                or name not in state._non_persistent_buffers_set
                or not isinstance(tensor, torch.Tensor)
                or tensor.numel() == 0
            ):
                continue
            result[f"{prefix}state.{name}"] = tensor.numel() * tensor.element_size()

    def visit(item, prefix):
        if id(item) in visited:
            return
        visited.add(id(item))
        visit_state(getattr(item, "state", None), prefix)
        sources = getattr(item, "sources", None)
        for index, auxiliary in enumerate(getattr(sources, "auxiliaries", ())):
            visit(auxiliary, f"{prefix}sources.auxiliaries[{index}].")

    visit(simulation, "")
    return dict(sorted(result.items()))


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


def _json_cache_preimage(value):
    """Convert the runtime tuple preimage to one lossless JSON tree."""
    if isinstance(value, tuple):
        return [_json_cache_preimage(item) for item in value]
    if value is None or type(value) in {bool, int, float, str}:
        return value
    raise TypeError(f"unsupported compile-cache preimage value: {type(value)!r}")


def _tuple_cache_preimage(value):
    if isinstance(value, list):
        return tuple(_tuple_cache_preimage(item) for item in value)
    return value


def _canonical_json_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _policy_config_preimage(result, name, policy):
    runtime = result["runtime"]
    executions = result["diagnostics"]["dispersive"]["policy_executions"]
    return {
        "workload": name,
        "device": runtime["device"],
        "precision": runtime["precision"],
        "compile_policy": runtime["compile_policy"],
        "compile_mode": runtime["compile_mode"],
        "explicit_cuda_graphs": runtime["explicit_cuda_graphs"],
        "requested_policy": policy,
        "resolved_policies": sorted({item["policy"] for item in executions}),
        "execution_representations": sorted(
            {item["execution_representation"] for item in executions}
        ),
        "execution_topology": sorted(
            [item["component"], item["model"], item["targets"]] for item in executions
        ),
    }


def _compile_cache_key_evidence(simulation, result, name, policy):
    runtime_preimage = _json_cache_preimage(simulation._compile_cache_key_preimage)
    recomputed = hashlib.sha256(
        repr(_tuple_cache_preimage(runtime_preimage)).encode()
    ).hexdigest()
    if recomputed != simulation.compile_cache_key:
        raise RuntimeError("compile cache preimage does not reproduce the runtime key")
    policy_config = _policy_config_preimage(result, name, policy)
    return {
        "schema_version": 1,
        "algorithm": COMPILE_CACHE_PREIMAGE_ALGORITHM,
        "runtime_preimage": runtime_preimage,
        "policy_config": policy_config,
        "policy_config_sha256": _canonical_json_sha256(policy_config),
    }


def _compile_cache_key_evidence_matches(result, name, policy):
    try:
        evidence = result["compile_cache_key_evidence"]
        if set(evidence) != {
            "schema_version",
            "algorithm",
            "runtime_preimage",
            "policy_config",
            "policy_config_sha256",
        }:
            return False
        if (
            type(evidence["schema_version"]) is not int
            or evidence["schema_version"] != 1
            or evidence["algorithm"] != COMPILE_CACHE_PREIMAGE_ALGORITHM
            or not isinstance(evidence["runtime_preimage"], list)
        ):
            return False
        runtime_key = hashlib.sha256(
            repr(_tuple_cache_preimage(evidence["runtime_preimage"])).encode()
        ).hexdigest()
        expected_config = _policy_config_preimage(result, name, policy)
        return (
            runtime_key == result["runtime"]["compile_cache_key"]
            and evidence["policy_config"] == expected_config
            and evidence["policy_config_sha256"]
            == _canonical_json_sha256(expected_config)
        )
    except KeyError, TypeError, ValueError:
        return False


def _policy_trace_descriptor(path, root, candidate_evidence):
    root = Path(root).resolve(strict=True)
    path = Path(path).resolve(strict=True)
    if not root.is_dir() or not path.is_file():
        raise ValueError(
            "policy trace descriptor requires a directory and regular file"
        )
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "policy diagnostic trace escapes the descriptor root"
        ) from error
    portable = relative.as_posix()
    if not portable or any(part in {"", ".", ".."} for part in portable.split("/")):
        raise ValueError("policy diagnostic trace path is not canonical")
    candidate = {
        key: candidate_evidence.get(key)
        for key in (
            "candidate_git_commit",
            "candidate_git_status",
            "manifest_sha256",
        )
    }
    if (
        not isinstance(candidate["candidate_git_commit"], str)
        or re.fullmatch(r"[0-9a-f]{40}", candidate["candidate_git_commit"]) is None
        or candidate["candidate_git_status"] != ""
        or not isinstance(candidate["manifest_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", candidate["manifest_sha256"]) is None
    ):
        raise ValueError("policy diagnostic candidate evidence is incomplete")
    raw = path.read_bytes()
    return {
        "path": portable,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "media_type": "application/json",
        "candidate_evidence": candidate,
    }


def _policy_execution_diagnostic(args, name, manifest, policy):
    if policy not in POLICY_WRITE_OPERATIONS:
        raise ValueError("policy operation diagnostics require one forced policy")
    descriptor_root = getattr(args, "descriptor_root", None)
    candidate_evidence = getattr(args, "candidate_evidence", None)
    if descriptor_root is None or not isinstance(candidate_evidence, dict):
        raise ValueError(
            "policy operation diagnostics require --descriptor-root and candidate evidence"
        )
    _spec, space, geometry, sources, bloch = _build_case(name, manifest)
    simulation = gmes.TorchSimulation(
        space=space,
        geometry=geometry,
        sources=sources,
        bloch=bloch,
        runtime=gmes.TorchRuntimeConfig(
            device=args.device,
            precision=args.precision,
            compile_policy="eager",
            compile_mode=args.compile_mode,
            cpu_threads=args.threads,
            cpu_interop_threads=args.interop_threads,
            execution_policy=policy,
            experimental_dispersive_grouping=False,
        ),
    )
    _initialize_fields(
        simulation,
        manifest["reference"]["seed"],
        manifest["reference"]["field_scale"],
    )
    executions = simulation.diagnostics()["dispersive"]["policy_executions"]
    if not executions or {item["policy"] for item in executions} != {policy}:
        raise RuntimeError(
            "forced policy diagnostic did not resolve the requested path"
        )
    trace_path = args.trace_directory / _trace_filename(
        name,
        device=args.device,
        precision=args.precision,
        compile_mode="uncompiled-policy-operation",
        capture_graphs=False,
        execution_policy=policy,
        threads=args.threads,
        interop_threads=args.interop_threads,
    )
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(
        activities=(
            [torch.profiler.ProfilerActivity.CPU]
            + (
                [torch.profiler.ProfilerActivity.CUDA]
                if simulation.device.type == "cuda"
                else []
            )
        ),
        profile_memory=False,
        record_shapes=False,
    ) as profile:
        for _ in range(args.profile_steps):
            simulation._apply_dispersive()
        _synchronize(simulation.device)
    profile.export_chrome_trace(str(trace_path))
    summary = _trace_summary(trace_path)
    expected_operation = POLICY_WRITE_OPERATIONS[policy]
    expected_count = len(executions) * args.profile_steps
    expected_counts = {
        operation: expected_count if operation == expected_operation else 0
        for operation in POLICY_WRITE_OPERATIONS.values()
    }
    if (
        summary["compiled_region_events"] != 0
        or summary["cuda_graph_launches"] != 0
        or summary["policy_write_operations"] != expected_counts
    ):
        raise RuntimeError(
            "uncompiled policy trace did not isolate the forced write op"
        )
    return {
        "schema_version": 1,
        "kind": POLICY_DIAGNOSTIC_KIND,
        "contract_id": POLICY_DIAGNOSTIC_CONTRACT,
        "execution_policy": policy,
        "compile_policy": "eager",
        "profile_steps": args.profile_steps,
        "execution_records_per_step": len(executions),
        "expected_operation": expected_operation,
        "expected_operation_count": expected_count,
        "observed_operation_counts": expected_counts,
        "trace": _policy_trace_descriptor(
            trace_path,
            descriptor_root,
            candidate_evidence,
        ),
    }


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
            data.get("observer_tag") == expected["performance_observer_tag"],
            "native observer tag does not match the frozen manifest",
        ),
        (
            data.get("observer_commit") == expected["performance_observer_commit"]
            and data.get("environment", {}).get("git_commit")
            == expected["performance_observer_commit"]
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


def _cpu_correctness_binding_complete(
    index, manifest, expected_evidence, runtime_receipt
):
    return correctness_binding_complete(
        index,
        manifest,
        expected_evidence,
        runtime_receipt=runtime_receipt,
        require_source_artifact=True,
    ) and _json_contract_equal(
        index.get("runtime_mode") if isinstance(index, dict) else None,
        CPU_CORRECTNESS_RUNTIME_MODE,
    )


def _aggregate_cpu_slice_outputs(
    outputs,
    manifest,
    native_summary=None,
    torch_baseline=None,
    allocation_document=None,
    expected_evidence=None,
    correctness_evidence=None,
    correctness_runtime_receipt=None,
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
    correctness_evidence_bound = _cpu_correctness_binding_complete(
        correctness_evidence,
        manifest,
        expected_evidence,
        correctness_runtime_receipt,
    )
    cpu_correctness_satisfied = correctness_evidence_bound and not errors
    if correctness_evidence_bound:
        acceptance_scope = "cpu-performance-and-correctness"
        blockers = ["gpu-policy-macos-evidence-not-bound"]
        if errors:
            blockers.insert(0, "cpu-performance-acceptance-failed")
    else:
        acceptance_scope = "cpu-performance-only"
        blockers = ["complete-field-and-persistent-state-correctness-not-bound"]
        if errors:
            blockers.insert(0, "cpu-performance-acceptance-failed")
    return {
        "schema_version": 4,
        "kind": "cpu-acceptance-aggregate",
        "acceptance_scope": acceptance_scope,
        "cpu_correctness_satisfied": cpu_correctness_satisfied,
        "issue_completion_satisfied": False,
        "issue_completion_blockers": blockers,
        "evidence": expected_evidence,
        "correctness_evidence": correctness_evidence,
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
            "correctness_evidence_bound": correctness_evidence_bound,
            "cpu_correctness_satisfied": cpu_correctness_satisfied,
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
    correctness_evidence_index=None,
    correctness_runtime_receipt=None,
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
    expected_evidence = _current_evidence(manifest)
    if (correctness_evidence_index is None) is not (
        correctness_runtime_receipt is None
    ):
        raise ValueError(
            "CPU correctness index and runtime publication receipt are both required"
        )
    correctness_evidence = (
        load_correctness_evidence_index(
            correctness_evidence_index,
            manifest,
            expected_evidence,
            descriptor_root=Path(correctness_evidence_index).resolve().parent,
            runtime_receipt=correctness_runtime_receipt,
        )
        if correctness_evidence_index is not None
        else None
    )
    correctness_receipt = (
        load_runtime_publication_receipt(
            correctness_runtime_receipt,
            manifest,
            expected_evidence,
            correctness_evidence.get("runtime_mode"),
            _runtime_receipt_candidates(correctness_evidence.get("artifacts", ())),
        )
        if correctness_evidence is not None
        else None
    )
    if correctness_evidence is not None and not _cpu_correctness_binding_complete(
        correctness_evidence,
        manifest,
        expected_evidence,
        correctness_receipt,
    ):
        raise ValueError(
            "CPU correctness evidence must use CPU float64 eager execution "
            "with default compile mode"
        )
    result = _aggregate_cpu_slice_outputs(
        outputs,
        manifest,
        native_summary,
        torch_baseline,
        allocation_document,
        expected_evidence,
        correctness_evidence,
        correctness_receipt,
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
    include_compile_cache_preimage=False,
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
    final_checksum = _checksum(simulation)
    timed_step_count = int(simulation.state.step_count.cpu())
    final_checkpoint = simulation.checkpoint()
    one_step_count = int(one_step_checkpoint["state"]["step_count"].cpu())
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
    profile_checkpoint = simulation.checkpoint()
    expected_dynamic_buffers = _simulation_dynamic_tensor_names(simulation)
    state_finiteness = _dynamic_state_finiteness(
        {
            "initial": checkpoint,
            "post_warmup": warm_checkpoint,
            "post_one_step": one_step_checkpoint,
            "post_timed": final_checkpoint,
            "post_profile": profile_checkpoint,
        },
        state_changes["changed_buffers"],
        expected_dynamic_buffers,
    )
    if state_finiteness["passed"] is not True:
        raise ValueError("simulation dynamic state contains non-finite values")
    # Evidence checkpoints are device clones, not persistent solver storage. Drop
    # the clones created after the baseline before measuring steady-state growth.
    del one_step_checkpoint, final_checkpoint, profile_checkpoint
    _synchronize(simulation.device)
    allocated_after = (
        torch.cuda.memory_allocated(simulation.device)
        if simulation.device.type == "cuda"
        else None
    )
    cuda_peak_allocated = (
        int(torch.cuda.max_memory_allocated(simulation.device))
        if simulation.device.type == "cuda"
        else None
    )
    cuda_peak_reserved = (
        int(torch.cuda.max_memory_reserved(simulation.device))
        if simulation.device.type == "cuda"
        else None
    )
    profiler["field_buffer_sizes_bytes"] = _field_buffer_sizes_bytes(simulation)
    profiler["fixed_boundary_buffer_sizes_bytes"] = _fixed_boundary_buffer_sizes_bytes(
        simulation
    )
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
    if simulation.device.type == "cpu":
        memory_bounded = (
            cpu_memory["fresh_process"].get("plateau", {}).get("bounded") is True
        )
    else:
        memory_bounded = _cuda_memory_bounded(
            simulation.device,
            memory_growth,
            allocated_before,
            allocated_after,
            cuda_peak_allocated,
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
            "paired_real": simulation.state.paired_real,
            "field_storage_representation": (
                "paired-real-v1" if simulation.state.paired_real else "real-v1"
            ),
            "field_storage_channels": 2 if simulation.state.paired_real else 1,
            "field_storage_dtype": str(simulation.dtype),
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
            "cuda_peak_allocated_bytes": cuda_peak_allocated,
            "cuda_peak_reserved_bytes": cuda_peak_reserved,
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
            "one_step_count": one_step_count,
            "expected_one_step_count": expected_one_step_count,
            "timed_step_count": timed_step_count,
            "expected_timed_step_count": expected_timed_step_count,
            "profiler_step_count": profiler_step_count,
            "expected_profiler_step_count": expected_profiler_step_count,
            **state_changes,
        },
        "state_finiteness": state_finiteness,
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
                and one_step_count == expected_one_step_count
                and timed_step_count == expected_timed_step_count
                and profiler_step_count == expected_profiler_step_count
            ),
        },
    }
    if name in REGION_INVARIANCE_GATES:
        result["region_equivalence"] = _region_equivalence_record(
            simulation, spec, geometry
        )
    if include_compile_cache_preimage:
        result["compile_cache_key_evidence"] = _compile_cache_key_evidence(
            simulation,
            result,
            name,
            execution_policy,
        )
    result["acceptance"]["passed"] = all(result["acceptance"].values())
    return result


def _allocation_provenance_for_run(
    document,
    args,
    name,
    *,
    precision=None,
    compile_mode=None,
    execution_policy=None,
):
    return _select_allocation_provenance(
        document,
        workload=name,
        device=str(torch.device(args.device)),
        precision=precision or args.precision,
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
            include_compile_cache_preimage=True,
        )
        if policy in POLICY_WRITE_OPERATIONS:
            results[policy]["policy_execution_diagnostic"] = (
                _policy_execution_diagnostic(args, name, manifest, policy)
            )
    errors = []
    forced_representations = {}
    forced_compile_cache_keys = {}
    forced_topologies = {}
    for policy, result in results.items():
        if not _boundary_execution_diagnostics_match(result):
            errors.append(
                f"{policy} boundary execution diagnostics differ from the solver ABI"
            )
        if not _compile_cache_key_evidence_matches(result, name, policy):
            errors.append(f"{policy} compile cache preimage evidence is invalid")
    for policy, expected_representation in POLICY_EXECUTION_REPRESENTATIONS.items():
        result = results[policy]
        dispersive = result.get("diagnostics", {}).get("dispersive", {})
        executions = dispersive.get("policy_executions")
        if not isinstance(executions, (list, tuple)) or not executions:
            errors.append(f"{policy} has no dispersive policy execution evidence")
            continue
        valid_records = all(
            isinstance(execution, dict)
            and isinstance(execution.get("component"), str)
            and bool(execution["component"])
            and isinstance(execution.get("model"), str)
            and bool(execution["model"])
            and type(execution.get("targets")) is int
            and execution["targets"] > 0
            and isinstance(execution.get("policy"), str)
            and isinstance(execution.get("execution_representation"), str)
            for execution in executions
        )
        if valid_records:
            forced_topologies[policy] = sorted(
                (item["component"], item["model"], item["targets"])
                for item in executions
            )
            representations = {
                execution["execution_representation"] for execution in executions
            }
            policies = {execution["policy"] for execution in executions}
        else:
            forced_topologies[policy] = []
            representations = set()
            policies = set()
        forced_representations[policy] = sorted(representations)
        if (
            not valid_records
            or policies != {policy}
            or representations != {expected_representation}
        ):
            errors.append(f"{policy} execution representation is not exact")
        if result.get("runtime", {}).get("execution_policy") != policy:
            errors.append(f"{policy} runtime policy is not exact")
        expected_top = f"policy-dispatched-bucket-io-v2[{expected_representation}]"
        if dispersive.get("execution_representation") != expected_top:
            errors.append(f"{policy} top-level execution representation is not exact")
        compile_key = result.get("runtime", {}).get("compile_cache_key")
        forced_compile_cache_keys[policy] = compile_key
        if (
            not isinstance(compile_key, str)
            or len(compile_key) != 64
            or any(character not in "0123456789abcdef" for character in compile_key)
        ):
            errors.append(f"{policy} compile cache key is invalid")
    representation_values = [
        tuple(forced_representations.get(policy, ()))
        for policy in POLICY_EXECUTION_REPRESENTATIONS
    ]
    if len(set(representation_values)) != len(POLICY_EXECUTION_REPRESENTATIONS):
        errors.append("forced execution representations are not distinct")
    topology_values = [
        tuple(forced_topologies.get(policy, ()))
        for policy in POLICY_EXECUTION_REPRESENTATIONS
    ]
    if len(set(topology_values)) != 1:
        errors.append("forced execution topologies do not match")
    compile_values = [
        forced_compile_cache_keys.get(policy)
        for policy in POLICY_EXECUTION_REPRESENTATIONS
    ]
    if any(not isinstance(value, str) or not value for value in compile_values) or len(
        set(compile_values)
    ) != len(POLICY_EXECUTION_REPRESENTATIONS):
        errors.append("forced compile cache keys are not distinct")

    auto_dispersive = results["auto"].get("diagnostics", {}).get("dispersive", {})
    auto_executions = auto_dispersive.get("policy_executions")
    auto_valid = isinstance(auto_executions, (list, tuple)) and bool(auto_executions)
    if auto_valid:
        auto_valid = all(
            isinstance(execution, dict)
            and isinstance(execution.get("policy"), str)
            and execution["policy"] in POLICY_EXECUTION_REPRESENTATIONS
            and isinstance(execution.get("execution_representation"), str)
            and execution["execution_representation"]
            == POLICY_EXECUTION_REPRESENTATIONS[execution["policy"]]
            and isinstance(execution.get("component"), str)
            and bool(execution["component"])
            and isinstance(execution.get("model"), str)
            and bool(execution["model"])
            and type(execution.get("targets")) is int
            and execution["targets"] > 0
            for execution in auto_executions
        )
    if not auto_valid:
        errors.append("auto policy did not report resolved execution evidence")
    else:
        auto_policies = {item["policy"] for item in auto_executions}
        auto_representations = {
            item["execution_representation"] for item in auto_executions
        }
        auto_topology = sorted(
            (item["component"], item["model"], item["targets"])
            for item in auto_executions
        )
        if topology_values and auto_topology != list(topology_values[0]):
            errors.append("auto execution topology differs from forced executions")
        expected_auto_top = (
            "policy-dispatched-bucket-io-v2["
            + ",".join(sorted(auto_representations))
            + "]"
        )
        if auto_dispersive.get("execution_representation") != expected_auto_top:
            errors.append("auto top-level execution representation is not exact")
        auto_compile_key = results["auto"].get("runtime", {}).get("compile_cache_key")
        if results["auto"].get("runtime", {}).get("execution_policy") != "auto":
            errors.append("auto runtime policy is not exact")
        if (
            not isinstance(auto_compile_key, str)
            or len(auto_compile_key) != 64
            or any(
                character not in "0123456789abcdef" for character in auto_compile_key
            )
        ):
            errors.append("auto compile cache key is invalid")
        elif len(auto_policies) == 1:
            selected_policy = next(iter(auto_policies))
            if auto_compile_key != forced_compile_cache_keys.get(selected_policy):
                errors.append(
                    "auto compile cache key differs from its resolved forced path"
                )

    seconds = {}
    for policy, result in results.items():
        value = (
            result.get("measurements", {}).get("advance", {}).get("seconds_per_step")
        )
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"{policy} timing is invalid")
        elif not math.isfinite(value) or value <= 0:
            errors.append(f"{policy} timing is invalid")
        else:
            seconds[policy] = float(value)
    ratio = (
        seconds["auto"]
        / min(seconds[policy] for policy in POLICY_EXECUTION_REPRESENTATIONS)
        if len(seconds) == 4
        else None
    )
    comparison_valid = not errors
    within_ten_percent = ratio <= 1.10 if comparison_valid else None
    all_acceptance_passed = all(
        result.get("acceptance", {}).get("passed") is True
        for result in results.values()
    )
    return {
        "case": name,
        "comparison_valid": comparison_valid,
        "invalid_reason": "; ".join(errors) if errors else None,
        "forced_execution_representations": forced_representations,
        "forced_compile_cache_keys": forced_compile_cache_keys,
        "forced_execution_topologies": forced_topologies,
        "auto_compile_cache_key": results["auto"]
        .get("runtime", {})
        .get("compile_cache_key"),
        "auto_policy_executions": auto_executions,
        "all_acceptance_passed": all_acceptance_passed,
        "auto_to_fastest_forced_ratio": ratio,
        "within_ten_percent": within_ten_percent,
        "results": results,
        "passed": (
            comparison_valid and within_ten_percent is True and all_acceptance_passed
        ),
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
        choices=names
        + SPECIAL_CASES
        + (
            "cpu-gates",
            "cuda-gates",
            "policy-gates",
            "paired-real-gates",
            "region-invariance-gates",
        ),
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
    parser.add_argument(
        "--descriptor-root",
        type=Path,
        help=(
            "root used for candidate-bound policy diagnostic trace descriptors; "
            "required with --policy matrix"
        ),
    )
    parser.add_argument("--native-summary", type=Path)
    parser.add_argument(
        "--torch-baseline-slice-artifacts",
        type=Path,
        nargs=2,
        metavar=("ONE_THREAD_JSON", "PHYSICAL_THREAD_JSON"),
    )
    parser.add_argument("--allocation-provenance", type=Path)
    parser.add_argument("--correctness-evidence-index", type=Path)
    parser.add_argument("--correctness-runtime-receipt", type=Path)
    parser.add_argument(
        "--cuda-correctness-index",
        type=Path,
        nargs=2,
        metavar=("EAGER_INDEX", "GRAPH_INDEX"),
    )
    parser.add_argument(
        "--cuda-correctness-receipt",
        type=Path,
        nargs=2,
        metavar=("EAGER_RECEIPT", "GRAPH_RECEIPT"),
    )
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
        rendered = json.dumps(output, allow_nan=False, indent=2, sort_keys=True) + "\n"
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
            getattr(args, "correctness_evidence_index", None),
            getattr(args, "correctness_runtime_receipt", None),
        )
        rendered = json.dumps(output, allow_nan=False, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered)
        print(rendered, end="")
        passed = (
            output["suite_acceptance"]["passed"] is True
            and output.get("acceptance_scope") == "cpu-performance-and-correctness"
            and output.get("cpu_correctness_satisfied") is True
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
    if args.policy == "matrix" and getattr(args, "descriptor_root", None) is None:
        raise ValueError("policy matrix evidence requires --descriptor-root")
    cuda_correctness_paths = getattr(args, "cuda_correctness_index", None)
    cuda_correctness_receipts = getattr(args, "cuda_correctness_receipt", None)
    if cuda_correctness_paths is not None and args.case != "cuda-gates":
        raise ValueError("--cuda-correctness-index requires --case cuda-gates")
    if args.case in {"paired-real-gates", "region-invariance-gates"} and (
        torch.device(args.device).type != "cuda"
        or args.precision != "float32"
        or args.compile_mode != "default"
        or args.policy != "auto"
        or args.capture_graphs
        or args.experimental_dispersive_grouping
    ):
        raise ValueError(
            "official paired-real and region-invariance suites require CUDA "
            "float32, default compilation, auto policy, and no experimental "
            "grouping or explicit CUDA graphs"
        )
    if args.case == "cuda-gates" and (
        args.device != "cuda:0"
        or args.precision != "float32"
        or args.compile_mode != "default"
        or args.policy != "auto"
        or args.capture_graphs
        or args.experimental_dispersive_grouping
        or args.threads != 1
        or args.interop_threads != 1
        or args.warmup != manifest["reference"]["performance_warmup_steps"]
        or args.steps != manifest["reference"]["performance_steps_per_repeat"]
        or args.repeats != manifest["reference"]["performance_repetitions"]
        or args.profile_steps != manifest["reference"]["performance_profile_steps"]
        or cuda_correctness_paths is None
        or cuda_correctness_receipts is None
    ):
        raise ValueError(
            "the official CUDA suite requires cuda:0, the float32 suite "
            "selector with its pinned per-case precision contract, default "
            "compilation, auto policy, one intra/inter-op thread, frozen "
            "measurement settings, eager/graph correctness indexes, and no "
            "experimental grouping or explicit CUDA graphs"
        )

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
    elif args.case == "paired-real-gates":
        cases = PAIRED_REAL_GATES
    elif args.case == "region-invariance-gates":
        cases = REGION_INVARIANCE_GATES
    else:
        cases = (args.case,)
    native_comparisons_expected = (
        torch.device(args.device).type == "cpu"
        and args.compile_mode != "matrix"
        and args.policy != "matrix"
        and any(name in cpu_gate_cases for name in cases)
    )

    evidence = _current_evidence(manifest)
    args.candidate_evidence = evidence
    cuda_correctness_indexes, cuda_correctness_errors = (
        _load_cuda_correctness_indexes(
            cuda_correctness_paths,
            cuda_correctness_receipts,
            manifest,
            evidence,
        )
        if args.case == "cuda-gates"
        else ([], [])
    )
    results = []
    for name in cases:
        effective_precision = (
            CUDA_PERFORMANCE_PRECISION_BY_CASE[name]
            if args.case == "cuda-gates"
            else args.precision
        )
        if args.compile_mode == "matrix":
            result = _variant_matrix(args, name, manifest, allocation_document)
        elif args.policy == "matrix":
            result = _policy_matrix(args, name, manifest, allocation_document)
        else:
            result = run_case(
                name,
                device=args.device,
                precision=effective_precision,
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
                    precision=effective_precision,
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
    paired_real_gate = (
        _paired_real_cuda_gate(results) if args.case == "paired-real-gates" else None
    )
    region_invariance_gate = (
        _region_invariance_gate(results)
        if args.case == "region-invariance-gates"
        else None
    )
    environment = _environment()
    cuda_suite_gate = (
        _cuda_suite_gate(
            results,
            cuda_correctness_indexes,
            cuda_correctness_errors,
            manifest,
            environment,
        )
        if args.case == "cuda-gates"
        else None
    )
    diagnostic_passed = (
        runtime_passed
        and (cuda_suite_gate is None or cuda_suite_gate["passed"])
        and (paired_real_gate is None or paired_real_gate["passed"])
        and (region_invariance_gate is None or region_invariance_gate["passed"])
    )
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
    named_non_cpu_suite = args.case in {
        "cuda-gates",
        "policy-gates",
        "paired-real-gates",
        "region-invariance-gates",
    }
    policy_matrix_expected = args.case == "policy-gates"
    policy_matrix_complete = (
        policy_matrix_expected
        and args.policy == "matrix"
        and tuple(result["case"] for result in results) == POLICY_GATES
    )
    paired_real_suite_expected = args.case == "paired-real-gates"
    paired_real_suite_complete = paired_real_suite_expected and [
        result.get("workload", {}).get("name") for result in results
    ] == list(PAIRED_REAL_GATES)
    region_invariance_suite_expected = args.case == "region-invariance-gates"
    region_invariance_suite_complete = region_invariance_suite_expected and [
        result.get("workload", {}).get("name") for result in results
    ] == list(REGION_INVARIANCE_GATES)
    cuda_suite_expected = args.case == "cuda-gates"
    cuda_suite_complete = (
        cuda_suite_expected
        and cuda_suite_gate is not None
        and cuda_suite_gate["case_closure_complete"] is True
        and cuda_suite_gate["correctness_evidence_bound"] is True
        and cuda_suite_gate["passed"] is True
    )
    suite_passed = (
        cuda_suite_complete
        or policy_matrix_complete
        or paired_real_suite_complete
        or region_invariance_suite_complete
    ) and diagnostic_passed
    output = {
        "schema_version": 4,
        "kind": (
            "cpu-acceptance-thread-slice"
            if cpu_full_suite_requested
            else "torch-tuning-diagnostic"
        ),
        "evidence": evidence,
        "candidate_evidence": evidence,
        "environment": environment,
        "torch_baseline": _torch_baseline_provenance(torch_baseline),
        "allocation_provenance_artifact": (
            allocation_document["source_artifact"]
            if allocation_document is not None
            else None
        ),
        "cuda_suite_gate": cuda_suite_gate,
        "paired_real_cuda_gate": paired_real_gate,
        "region_invariance_gate": region_invariance_gate,
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
            "policy_matrix_expected": policy_matrix_expected,
            "policy_matrix_complete": policy_matrix_complete,
            "cuda_suite_expected": cuda_suite_expected,
            "cuda_suite_complete": cuda_suite_complete,
            "paired_real_suite_expected": paired_real_suite_expected,
            "paired_real_suite_complete": paired_real_suite_complete,
            "region_invariance_suite_expected": region_invariance_suite_expected,
            "region_invariance_suite_complete": region_invariance_suite_complete,
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
    rendered = json.dumps(output, allow_nan=False, indent=2, sort_keys=True) + "\n"
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
