"""Reproduce an Inductor allocation caused by composed tensor mutation.

This script intentionally has no GMES imports.  It compares the affected
slice-based graph with a public ``torch.as_strided`` formulation that exposes
the same interior field view without cloning the full field.  The optional
reinplace override is a private, diagnostic experiment, not a supported fix.
"""

import argparse
import collections
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


class ReproducerAssertionError(RuntimeError):
    """Raised when a requested fail-closed reproducer check does not hold."""


def _require(condition, message):
    if not condition:
        raise ReproducerAssertionError(message)


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-reinplace",
        action="store_true",
        help=(
            "diagnostically monkeypatch a private Inductor reinplace check; "
            "this is not a supported workaround"
        ),
    )
    parser.add_argument(
        "--assert-affected",
        action="store_true",
        help="fail unless all equivalence, allocation, and provenance checks pass",
    )
    parser.add_argument("--profile-calls", type=int, default=5)
    parser.add_argument("--timing-repeats", type=int, default=9)
    parser.add_argument("--timing-calls", type=int, default=200)
    parser.add_argument("--equivalence-calls", type=int, default=5)
    parser.add_argument("--cache-directory", type=Path)
    parser.add_argument("--trace-directory", type=Path)
    return parser.parse_args()


ARGS = _arguments()
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_SHA256 = hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest()
_temporary_cache = ARGS.cache_directory is None
cache_directory = ARGS.cache_directory or Path(
    tempfile.mkdtemp(prefix="gmes-inductor-repro-cache-")
)
cache_initially_empty = not cache_directory.exists() or not any(
    cache_directory.iterdir()
)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_directory)

import torch

if ARGS.force_reinplace:
    # Private monkeypatch retained only to distinguish the profitability check
    # from the supported public-as-strided formulation below.
    from torch._inductor.fx_passes import reinplace

    reinplace.inplaceable_ops[reinplace._generalized_scatter] = reinplace.InplaceableOp(
        reinplace._inplace_generalized_scatter,
        0,
        extra_check=lambda _node: True,
    )


FIELD_ELEMENTS = 51_520
INDEXED_ELEMENTS = 99
PROFILE_CALLS = ARGS.profile_calls
TIMING_REPEATS = ARGS.timing_repeats
TIMING_CALLS = ARGS.timing_calls
EQUIVALENCE_CALLS = ARGS.equivalence_calls


def stencil(field, delta):
    """Apply the regular mutation that precedes indexed material work."""
    field[1:-1].add_(delta)


def material(field, indices, gain):
    """Apply a compact gather/update/scatter in isolation."""
    values = torch.index_select(field, 0, indices) * gain
    field.index_copy_(0, indices, values)


def composed(field, delta, indices, gain):
    """Compose both mutations in the graph shape that triggers the clone."""
    field[1:-1].add_(delta)
    values = torch.index_select(field, 0, indices) * gain
    field.index_copy_(0, indices, values)


def composed_as_strided(field, delta, indices, gain):
    """Express the same composed update through a public storage view."""
    strides = field.stride()
    interior = torch.as_strided(
        field,
        size=(field.shape[0] - 2,),
        stride=strides,
        storage_offset=strides[0],
    )
    interior.add_(delta)
    values = torch.index_select(field, 0, indices) * gain
    field.index_copy_(0, indices, values)


def _counter_snapshot():
    return {
        str(group): {
            str(key): int(value)
            for key, value in sorted(values.items(), key=lambda item: str(item[0]))
        }
        for group, values in sorted(
            torch._dynamo.utils.counters.items(), key=lambda item: str(item[0])
        )
        if values
    }


def _counter_delta(before, after):
    delta = {}
    for group in sorted(set(before) | set(after)):
        group_delta = {}
        before_group = before.get(group, {})
        after_group = after.get(group, {})
        for key in sorted(set(before_group) | set(after_group)):
            difference = after_group.get(key, 0) - before_group.get(key, 0)
            if difference:
                group_delta[key] = difference
        if group_delta:
            delta[group] = group_delta
    return delta


def _inside_scope(event, scopes):
    timestamp = event.get("ts")
    if timestamp is None:
        return False
    return any(
        event.get("pid") == scope.get("pid")
        and event.get("tid") == scope.get("tid")
        and scope["ts"] <= timestamp <= scope["ts"] + scope["dur"]
        for scope in scopes
    )


def _allocation_summary(name, call, reset, trace_directory):
    mode = "forced" if ARGS.force_reinplace else "baseline"
    trace_path = trace_directory / f"{mode}-{name}.json"
    scope_name = f"gmes-repro::{name}"
    reset()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        profile_memory=True,
    ) as profile:
        for _ in range(PROFILE_CALLS):
            with torch.profiler.record_function(scope_name):
                call()
    profile.export_chrome_trace(str(trace_path))
    trace_bytes = trace_path.read_bytes()
    trace = json.loads(trace_bytes)
    events = trace.get("traceEvents", ())
    scopes = [
        event
        for event in events
        if event.get("name") == scope_name
        and event.get("ph") == "X"
        and event.get("dur") is not None
    ]
    _require(
        len(scopes) == PROFILE_CALLS,
        f"{name}: expected {PROFILE_CALLS} profiler scopes, found {len(scopes)}",
    )
    positive_memory_events = [
        event
        for event in events
        if event.get("name") == "[memory]"
        and int(event.get("args", {}).get("Bytes", 0)) > 0
    ]
    scoped_events = [
        event for event in positive_memory_events if _inside_scope(event, scopes)
    ]
    unscoped_events = [
        event for event in positive_memory_events if not _inside_scope(event, scopes)
    ]
    sizes = [int(event["args"]["Bytes"]) for event in scoped_events]
    full_field_bytes = (
        FIELD_ELEMENTS * torch.empty((), dtype=torch.float64).element_size()
    )
    return {
        "positive_allocation_events": len(sizes),
        "allocated_bytes": sum(sizes),
        "full_field_allocation_bytes": full_field_bytes,
        "full_field_allocation_events": sizes.count(full_field_bytes),
        "allocation_histogram": {
            str(size): count
            for size, count in sorted(collections.Counter(sizes).items())
        },
        "unscoped_positive_allocation_events": len(unscoped_events),
        "unscoped_allocated_bytes": sum(
            int(event["args"]["Bytes"]) for event in unscoped_events
        ),
        "profile_scopes": len(scopes),
        "trace": {
            "path": str(trace_path),
            "bytes": len(trace_bytes),
            "sha256": hashlib.sha256(trace_bytes).hexdigest(),
        },
    }


def _timing_summary(call, reset):
    samples = []
    for _ in range(TIMING_REPEATS):
        reset()
        start = time.perf_counter_ns()
        for _ in range(TIMING_CALLS):
            call()
        samples.append((time.perf_counter_ns() - start) / TIMING_CALLS / 1e3)
    return {
        "unit": "microseconds_per_call",
        "raw_samples": samples,
        "median": statistics.median(samples),
        "minimum": min(samples),
        "maximum": max(samples),
    }


def _tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


def _verify_exact_equivalence(name, eager, compiled, arguments, base):
    eager_field = base.clone()
    compiled_field = base.clone()
    for iteration in range(1, EQUIVALENCE_CALLS + 1):
        eager(eager_field, *arguments)
        compiled(compiled_field, *arguments)
        if not torch.equal(compiled_field, eager_field):
            mismatches = int(torch.count_nonzero(compiled_field != eager_field).item())
            max_absolute_error = float(
                torch.max(torch.abs(compiled_field - eager_field)).item()
            )
            raise ReproducerAssertionError(
                f"{name}: eager/compiled mismatch after iteration {iteration}: "
                f"{mismatches} elements, max abs error {max_absolute_error}"
            )
    return {
        "calls": EQUIVALENCE_CALLS,
        "exact": True,
        "final_tensor_sha256": _tensor_sha256(compiled_field),
    }


def _cpu_model():
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.partition(":")[2].strip()
    try:
        return subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except FileNotFoundError, subprocess.CalledProcessError:
        return platform.processor() or "unknown"


def _cpu_affinity():
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return None


def _assert_expected_behavior(results, steady_state_counter_delta):
    for name, result in results.items():
        _require(
            result["allocation"]["unscoped_positive_allocation_events"] == 0,
            f"{name}: positive allocations escaped the measured call scopes",
        )
        _require(
            result["equivalence"]["exact"],
            f"{name}: exact eager/compiled equivalence was not established",
        )
    _require(
        results["stencil"]["allocation"]["positive_allocation_events"] == 0,
        "stencil: isolated compiled mutation unexpectedly allocated",
    )
    _require(
        results["material"]["allocation"]["full_field_allocation_events"] == 0,
        "material: isolated indexed mutation allocated a full-field buffer",
    )
    _require(
        results["composed_as_strided"]["allocation"]["full_field_allocation_events"]
        == 0,
        "composed_as_strided: public view still allocated a full-field buffer",
    )
    expected_composed = 0 if ARGS.force_reinplace else PROFILE_CALLS
    _require(
        results["composed"]["allocation"]["full_field_allocation_events"]
        == expected_composed,
        "composed: expected "
        f"{expected_composed} full-field allocations, found "
        f"{results['composed']['allocation']['full_field_allocation_events']}",
    )
    _require(
        not steady_state_counter_delta,
        "Dynamo/Inductor counters changed during profiling or timing: "
        f"{steady_state_counter_delta}",
    )


def main():
    for option, value in (
        ("--profile-calls", PROFILE_CALLS),
        ("--timing-repeats", TIMING_REPEATS),
        ("--timing-calls", TIMING_CALLS),
        ("--equivalence-calls", EQUIVALENCE_CALLS),
    ):
        _require(value > 0, f"{option} must be positive")
    if ARGS.assert_affected:
        _require(
            cache_initially_empty,
            "--assert-affected requires a new or empty Inductor cache directory",
        )

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    counters_before_compile = _counter_snapshot()

    trace_directory = ARGS.trace_directory or Path(
        tempfile.mkdtemp(prefix="gmes-inductor-repro-traces-")
    )
    trace_directory.mkdir(parents=True, exist_ok=True)

    base = torch.linspace(0.25, 1.25, FIELD_ELEMENTS, dtype=torch.float64)
    delta = torch.linspace(-0.01, 0.01, FIELD_ELEMENTS - 2, dtype=torch.float64)
    indices = torch.arange(0, 2 * INDEXED_ELEMENTS, 2, dtype=torch.int64)
    gain = torch.linspace(0.99, 1.01, INDEXED_ELEMENTS, dtype=torch.float64)
    definitions = {
        "stencil": (stencil, (delta,)),
        "material": (material, (indices, gain)),
        "composed": (composed, (delta, indices, gain)),
        "composed_as_strided": (
            composed_as_strided,
            (delta, indices, gain),
        ),
    }
    compiled = {
        name: torch.compile(function, fullgraph=True)
        for name, (function, _arguments) in definitions.items()
    }

    equivalence = {
        name: _verify_exact_equivalence(
            name,
            function,
            compiled[name],
            arguments,
            base,
        )
        for name, (function, arguments) in definitions.items()
    }

    fields = {name: base.clone() for name in definitions}
    calls = {
        name: (
            lambda name=name, arguments=arguments: compiled[name](
                fields[name], *arguments
            )
        )
        for name, (_function, arguments) in definitions.items()
    }
    resets = {
        name: (lambda name=name: fields[name].copy_(base)) for name in definitions
    }
    for name in definitions:
        resets[name]()
        calls[name]()
        calls[name]()
    counters_after_warmup = _counter_snapshot()

    results = {}
    for name in definitions:
        results[name] = {
            "equivalence": equivalence[name],
            "allocation": _allocation_summary(
                name, calls[name], resets[name], trace_directory
            ),
            "timing": _timing_summary(calls[name], resets[name]),
        }

    counters_after_measurement = _counter_snapshot()
    compile_counter_delta = _counter_delta(
        counters_before_compile, counters_after_warmup
    )
    steady_state_counter_delta = _counter_delta(
        counters_after_warmup, counters_after_measurement
    )
    _require(
        hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest() == SCRIPT_SHA256,
        "the reproducer source changed while the measurement was running",
    )
    if ARGS.assert_affected:
        _assert_expected_behavior(results, steady_state_counter_delta)

    print(
        json.dumps(
            {
                "schema": 2,
                "kind": "torch_inductor_composed_mutation",
                "environment": {
                    "platform": platform.platform(),
                    "machine": platform.machine(),
                    "cpu_model": _cpu_model(),
                    "logical_cpu_count": os.cpu_count(),
                    "cpu_affinity": _cpu_affinity(),
                    "python": platform.python_version(),
                    "python_optimization_level": sys.flags.optimize,
                    "torch": torch.__version__,
                    "torch_git_version": torch.version.git_version,
                    "torch_build_config": torch.__config__.show(),
                    "threads": torch.get_num_threads(),
                    "interop_threads": torch.get_num_interop_threads(),
                    "thread_environment": {
                        name: os.environ.get(name)
                        for name in (
                            "OMP_NUM_THREADS",
                            "MKL_NUM_THREADS",
                            "OPENBLAS_NUM_THREADS",
                        )
                    },
                    "force_reinplace_private_diagnostic": ARGS.force_reinplace,
                    "inductor_cache_directory": str(cache_directory),
                    "inductor_cache_initially_empty": cache_initially_empty,
                },
                "provenance": {
                    "script": str(SCRIPT_PATH),
                    "script_sha256": SCRIPT_SHA256,
                    "argv": sys.argv,
                },
                "shape": {
                    "field_elements": FIELD_ELEMENTS,
                    "indexed_elements": INDEXED_ELEMENTS,
                    "dtype": "float64",
                    "profile_calls": PROFILE_CALLS,
                    "timing_repeats": TIMING_REPEATS,
                    "timing_calls": TIMING_CALLS,
                    "equivalence_calls": EQUIVALENCE_CALLS,
                },
                "compiler_counters": {
                    "before_compile": counters_before_compile,
                    "after_warmup": counters_after_warmup,
                    "compile_delta": compile_counter_delta,
                    "after_measurement": counters_after_measurement,
                    "steady_state_delta": steady_state_counter_delta,
                },
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        if _temporary_cache:
            shutil.rmtree(cache_directory, ignore_errors=True)
