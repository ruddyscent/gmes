#!/usr/bin/env python3
"""Reproduce recurring CPU allocations from an Inductor ``torch.while_loop``.

This script intentionally has no GMES imports.  It compares the original
multi-carry topology with the best compact representation used by GMES: one
exact-size, caller-owned floating-point workspace containing the field
iterate, history state, and a per-cell completion code.  Only steady-state
compiled calls are recorded in separate raw Chrome allocation traces.
After that evidence is captured, it also demonstrates why mutating the packed
carry in place is unavailable through the public higher-order-op contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import sys
import tempfile
import time
from typing import Any, Callable

CELLS = 99
WIDTH = 4
LOOP_ITERATIONS = 4
MULTI_CARRY = "multi_carry"
SINGLE_PACKED_CARRY = "single_packed_carry"
INPLACE_PACKED_CARRY = "inplace_packed_carry"
PACKED_FIELD_OFFSET = 0
PACKED_HISTORY_OFFSET = PACKED_FIELD_OFFSET + CELLS
PACKED_CODE_OFFSET = PACKED_HISTORY_OFFSET + CELLS * WIDTH
PACKED_ELEMENTS = PACKED_CODE_OFFSET + CELLS


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assert-affected",
        action="store_true",
        help=(
            "fail unless both multi-carry and single-packed-carry allocations "
            "recur in their raw traces and in-place carry mutation is rejected "
            "during higher-order-op tracing"
        ),
    )
    parser.add_argument("--profile-calls", type=_positive_integer, default=5)
    parser.add_argument("--warmup-calls", type=_positive_integer, default=2)
    parser.add_argument("--equivalence-calls", type=_positive_integer, default=5)
    parser.add_argument("--timing-samples", type=_positive_integer, default=9)
    parser.add_argument("--calls-per-sample", type=_positive_integer, default=100)
    parser.add_argument(
        "--cache-directory",
        type=Path,
        help="fresh empty directory for TORCHINDUCTOR_CACHE_DIR",
    )
    parser.add_argument(
        "--trace-directory",
        type=Path,
        help="fresh empty directory for the raw Chrome allocation traces",
    )
    return parser.parse_args()


def _fresh_directory(requested: Path | None, prefix: str) -> tuple[Path, bool]:
    if requested is None:
        return Path(tempfile.mkdtemp(prefix=prefix)), True

    directory = requested.expanduser().resolve()
    if directory.exists():
        if not directory.is_dir():
            raise ValueError(f"not a directory: {directory}")
        if any(directory.iterdir()):
            raise ValueError(f"directory must be empty: {directory}")
    else:
        directory.mkdir(parents=True)
    return directory, False


ARGS = _parse_args()
CACHE_DIRECTORY, CACHE_DIRECTORY_IS_TEMPORARY = _fresh_directory(
    ARGS.cache_directory, "torch-while-loop-cache-"
)
TRACE_DIRECTORY, TRACE_DIRECTORY_IS_TEMPORARY = _fresh_directory(
    ARGS.trace_directory, "torch-while-loop-trace-"
)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(CACHE_DIRECTORY)

import torch


class ReproductionError(RuntimeError):
    """Raised when a fail-closed reproduction check does not hold."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReproductionError(message)


def _counter_snapshot() -> dict[str, dict[str, int]]:
    return {
        str(group): {str(key): int(value) for key, value in values.items()}
        for group, values in torch._dynamo.utils.counters.items()
    }


def _counter_value(snapshot: dict[str, dict[str, int]], group: str, key: str) -> int:
    return snapshot.get(group, {}).get(key, 0)


def _counter_delta(
    before: dict[str, dict[str, int]], after: dict[str, dict[str, int]]
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for group in sorted(set(before) | set(after)):
        keys = set(before.get(group, {})) | set(after.get(group, {}))
        changes = {
            key: after.get(group, {}).get(key, 0) - before.get(group, {}).get(key, 0)
            for key in sorted(keys)
        }
        nonzero = {key: value for key, value in changes.items() if value != 0}
        if nonzero:
            result[group] = nonzero
    return result


def _graph_break_count(snapshot: dict[str, dict[str, int]]) -> int:
    return sum(
        snapshot.get(group, {}).get(key, 0)
        for group in ("graph_break", "graph_breaks")
        for key in snapshot.get(group, {})
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _allocation_summary(
    trace_path: Path,
    expected_sizes: dict[str, int],
    minimum_events: dict[str, int],
) -> dict[str, Any]:
    with trace_path.open(encoding="utf-8") as stream:
        trace = json.load(stream)

    events = [
        event
        for event in trace.get("traceEvents", [])
        if event.get("name") == "[memory]"
    ]
    increments = [int(event.get("args", {}).get("Bytes", 0)) for event in events]
    positive = [increment for increment in increments if increment > 0]
    negative = [increment for increment in increments if increment < 0]
    histogram: dict[int, int] = {}
    for allocation_bytes in positive:
        histogram[allocation_bytes] = histogram.get(allocation_bytes, 0) + 1

    recurring_sizes = {
        name: {
            "bytes": allocation_bytes,
            "events": histogram.get(allocation_bytes, 0),
            "minimum_events": minimum_events[name],
        }
        for name, allocation_bytes in expected_sizes.items()
        if histogram.get(allocation_bytes, 0) >= minimum_events[name]
    }
    compiled_region_events = sum(
        1
        for event in trace.get("traceEvents", [])
        if str(event.get("name", "")).startswith("Torch-Compiled Region")
    )
    return {
        "memory_events": len(events),
        "positive_allocation_events": len(positive),
        "positive_allocation_bytes": sum(positive),
        "negative_allocation_events": len(negative),
        "negative_allocation_bytes": sum(negative),
        "net_allocation_bytes": sum(increments),
        "allocation_size_histogram": {
            str(key): histogram[key] for key in sorted(histogram)
        },
        "expected_recurring_sizes": expected_sizes,
        "minimum_recurring_events": minimum_events,
        "recurring_expected_sizes": recurring_sizes,
        "compiled_region_events": compiled_region_events,
    }


def _eager_reference(
    field: torch.Tensor, history: torch.Tensor, weights: torch.Tensor
) -> None:
    field_current = field
    history_current = history
    for _ in range(LOOP_ITERATIONS):
        weighted = torch.sum(history_current * weights, dim=1)
        field_next = field_current * 0.5 + weighted * 0.25
        history_next = history_current * 0.5 + field_next[:, None] * 0.25
        field_current = field_next
        history_current = history_next
    field.copy_(field_current)
    history.copy_(history_current)


def _multi_carry_update(
    field: torch.Tensor,
    history: torch.Tensor,
    weights: torch.Tensor,
    initial_iteration: torch.Tensor,
) -> None:
    def cond_fn(
        iteration: torch.Tensor,
        field_current: torch.Tensor,
        history_current: torch.Tensor,
    ) -> torch.Tensor:
        del field_current, history_current
        return iteration < LOOP_ITERATIONS

    def body_fn(
        iteration: torch.Tensor,
        field_current: torch.Tensor,
        history_current: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weighted = torch.sum(history_current * weights, dim=1)
        field_next = field_current * 0.5 + weighted * 0.25
        history_next = history_current * 0.5 + field_next[:, None] * 0.25
        return iteration + 1, field_next, history_next

    _, field_result, history_result = torch.while_loop(
        cond_fn,
        body_fn,
        (initial_iteration, field, history),
    )
    field.copy_(field_result)
    history.copy_(history_result)


def _packed_views(
    workspace: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    field = workspace.narrow(0, PACKED_FIELD_OFFSET, CELLS)
    history = workspace.narrow(0, PACKED_HISTORY_OFFSET, CELLS * WIDTH).view(
        CELLS, WIDTH
    )
    code = workspace.narrow(0, PACKED_CODE_OFFSET, CELLS)
    return field, history, code


def _single_packed_carry_update(
    field: torch.Tensor,
    history: torch.Tensor,
    weights: torch.Tensor,
    workspace: torch.Tensor,
) -> None:
    workspace_field, workspace_history, workspace_code = _packed_views(workspace)
    workspace_field.copy_(field)
    workspace_history.copy_(history)
    workspace_code.zero_()

    def cond_fn(packed: torch.Tensor) -> torch.Tensor:
        _, _, code = _packed_views(packed)
        active = torch.logical_and(code >= 0, code < LOOP_ITERATIONS)
        return torch.any(active)

    def body_fn(packed: torch.Tensor) -> tuple[torch.Tensor]:
        field_current, history_current, code = _packed_views(packed)
        active = torch.logical_and(code >= 0, code < LOOP_ITERATIONS)
        weighted = torch.sum(history_current * weights, dim=1)
        field_candidate = field_current * 0.5 + weighted * 0.25
        history_candidate = history_current * 0.5 + field_candidate[:, None] * 0.25
        field_next = torch.where(active, field_candidate, field_current)
        history_next = torch.where(
            active[:, None], history_candidate, history_current
        )
        code_next = code + active.to(dtype=code.dtype)
        return (
            torch.cat(
                (field_next, history_next.reshape(-1), code_next),
                dim=0,
            ),
        )

    (result,) = torch.while_loop(cond_fn, body_fn, (workspace,))
    workspace.copy_(result)
    result_field, result_history, _ = _packed_views(result)
    field.copy_(result_field)
    history.copy_(result_history)


def _inplace_packed_carry_update(workspace: torch.Tensor) -> None:
    """Attempt the tempting allocation-free carry mutation rejected by HOP."""

    def cond_fn(packed: torch.Tensor) -> torch.Tensor:
        return packed[-1] < LOOP_ITERATIONS

    def body_fn(packed: torch.Tensor) -> tuple[torch.Tensor]:
        packed.add_(1.0)
        return (packed,)

    (result,) = torch.while_loop(cond_fn, body_fn, (workspace,))
    workspace.copy_(result)


def _normalize_rejection(error: BaseException) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in str(error).splitlines()]
    lines = [line for line in lines if line]
    reason = next(
        (
            line
            for line in lines
            if "mutat" in line.lower() and "input" in line.lower()
        ),
        lines[0] if lines else type(error).__qualname__,
    )
    return reason[:500]


def _inplace_rejection_evidence() -> dict[str, Any]:
    compiled = torch.compile(
        _inplace_packed_carry_update, fullgraph=True, dynamic=False
    )
    workspace = torch.zeros(PACKED_ELEMENTS, dtype=torch.float64)
    try:
        compiled(workspace)
    except Exception as error:
        exception_type = f"{type(error).__module__}.{type(error).__qualname__}"
        reason = _normalize_rejection(error)
        reason_lower = reason.lower()
        input_mutation_during_hop_tracing = (
            "higherorderop" in exception_type.lower()
            and "mutat" in reason_lower
            and "input" in reason_lower
        )
        evidence = {
            "attempted_after_functional_evidence_and_counter_capture": True,
            "rejected": True,
            "exception_type": exception_type,
            "reason": reason,
            "input_mutation_during_higher_order_op_tracing": (
                input_mutation_during_hop_tracing
            ),
        }
    else:
        evidence = {
            "attempted_after_functional_evidence_and_counter_capture": True,
            "rejected": False,
            "exception_type": None,
            "reason": None,
            "input_mutation_during_higher_order_op_tracing": False,
        }

    if ARGS.assert_affected:
        _require(
            evidence["input_mutation_during_higher_order_op_tracing"],
            "in-place packed carry was not rejected specifically for input "
            "mutation during higher-order-op tracing: "
            f"{evidence}",
        )
    return evidence


def _new_common_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    field = torch.arange(CELLS, dtype=torch.float64) / 128.0
    history = field[:, None].expand(-1, WIDTH).clone()
    weights = torch.full((WIDTH,), 0.125, dtype=torch.float64)
    return field, history, weights


def _new_multi_carry_inputs() -> tuple[torch.Tensor, ...]:
    field, history, weights = _new_common_inputs()
    initial_iteration = torch.zeros((), dtype=torch.int64)
    return field, history, weights, initial_iteration


def _new_single_packed_inputs() -> tuple[torch.Tensor, ...]:
    field, history, weights = _new_common_inputs()
    workspace = torch.empty(PACKED_ELEMENTS, dtype=torch.float64)
    return field, history, weights, workspace


def _check_exact_repeated_equivalence(
    name: str,
    compiled: Any,
    new_inputs: Callable[[], tuple[torch.Tensor, ...]],
) -> list[dict[str, Any]]:
    reference_field, reference_history, weights = _new_common_inputs()
    inputs = new_inputs()
    compiled_field, compiled_history = inputs[:2]
    workspace = inputs[3] if name == SINGLE_PACKED_CARRY else None
    workspace_data_ptr = workspace.data_ptr() if workspace is not None else None
    results: list[dict[str, Any]] = []
    for call_index in range(ARGS.equivalence_calls):
        _eager_reference(reference_field, reference_history, weights)
        compiled(*inputs)
        field_equal = torch.equal(reference_field, compiled_field)
        history_equal = torch.equal(reference_history, compiled_history)
        field_max_difference = float(
            torch.max(torch.abs(reference_field - compiled_field)).item()
        )
        history_max_difference = float(
            torch.max(torch.abs(reference_history - compiled_history)).item()
        )
        workspace_storage_stable = None
        workspace_contents_equal = None
        completion_codes_equal = None
        if workspace is not None:
            workspace_storage_stable = workspace.data_ptr() == workspace_data_ptr
            packed_field, packed_history, packed_code = _packed_views(workspace)
            workspace_contents_equal = torch.equal(
                compiled_field, packed_field
            ) and torch.equal(compiled_history, packed_history)
            completion_codes_equal = torch.equal(
                packed_code,
                torch.full_like(packed_code, float(LOOP_ITERATIONS)),
            )
        results.append(
            {
                "call": call_index + 1,
                "field_equal": field_equal,
                "history_equal": history_equal,
                "field_max_absolute_difference": field_max_difference,
                "history_max_absolute_difference": history_max_difference,
                "caller_owned_workspace_storage_stable": workspace_storage_stable,
                "workspace_contents_equal_outputs": workspace_contents_equal,
                "completion_codes_equal_loop_iterations": completion_codes_equal,
            }
        )
        _require(
            field_equal
            and history_equal
            and workspace_storage_stable is not False
            and workspace_contents_equal is not False
            and completion_codes_equal is not False,
            f"compiled {name} while_loop diverged from the eager Python-loop "
            f"reference at repeated call {call_index + 1}: "
            f"field_max_difference={field_max_difference}, "
            f"history_max_difference={history_max_difference}, "
            f"workspace_storage_stable={workspace_storage_stable}, "
            f"workspace_contents_equal={workspace_contents_equal}, "
            f"completion_codes_equal={completion_codes_equal}",
        )
    return results


def _profile_allocations(
    compiled: Any,
    new_inputs: Callable[[], tuple[torch.Tensor, ...]],
    trace_path: Path,
    expected_sizes: dict[str, int],
    minimum_events: dict[str, int],
) -> dict[str, Any]:
    inputs = new_inputs()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        profile_memory=True,
        record_shapes=True,
    ) as profile:
        for _ in range(ARGS.profile_calls):
            compiled(*inputs)
    profile.export_chrome_trace(str(trace_path))
    return _allocation_summary(trace_path, expected_sizes, minimum_events)


def _time_compiled(
    compiled: Any,
    new_inputs: Callable[[], tuple[torch.Tensor, ...]],
) -> list[float]:
    inputs = new_inputs()
    base_field = inputs[0].clone()
    base_history = inputs[1].clone()
    samples: list[float] = []
    for _ in range(ARGS.timing_samples):
        inputs[0].copy_(base_field)
        inputs[1].copy_(base_history)
        started = time.perf_counter_ns()
        for _ in range(ARGS.calls_per_sample):
            compiled(*inputs)
        elapsed = time.perf_counter_ns() - started
        samples.append(elapsed / ARGS.calls_per_sample / 1_000.0)
    return samples


def _variant_evidence(
    name: str,
    compiled: Any,
    new_inputs: Callable[[], tuple[torch.Tensor, ...]],
    expected_sizes: dict[str, int],
    minimum_events: dict[str, int],
) -> dict[str, Any]:
    equivalence = _check_exact_repeated_equivalence(name, compiled, new_inputs)
    trace_path = TRACE_DIRECTORY / f"{name}_allocation_trace.json"
    allocations = _profile_allocations(
        compiled, new_inputs, trace_path, expected_sizes, minimum_events
    )
    timing_samples = _time_compiled(compiled, new_inputs)
    _require(trace_path.is_file(), f"allocation trace was not written: {trace_path}")
    _require(trace_path.stat().st_size > 0, f"allocation trace is empty: {trace_path}")

    recurring = allocations["recurring_expected_sizes"]
    affected = len(recurring) == len(expected_sizes)
    if ARGS.assert_affected:
        _require(
            affected,
            f"{name} did not exhibit every expected recurring allocation; "
            f"expected_sizes={expected_sizes}, minimum_events={minimum_events}, "
            f"histogram={allocations['allocation_size_histogram']}",
        )
    return {
        "affected": affected,
        "exact_repeated_equivalence": equivalence,
        "timing_microseconds_per_call": {
            "raw_samples": timing_samples,
            "median": statistics.median(timing_samples),
            "minimum": min(timing_samples),
            "maximum": max(timing_samples),
        },
        "allocation_trace": {
            "path": str(trace_path),
            "size_bytes": trace_path.stat().st_size,
            "sha256": _sha256(trace_path),
            **allocations,
        },
    }


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.compiler.reset()
    torch._dynamo.utils.counters.clear()

    compiled_multi = torch.compile(
        _multi_carry_update, fullgraph=True, dynamic=False
    )
    compiled_packed = torch.compile(
        _single_packed_carry_update, fullgraph=True, dynamic=False
    )
    variants = (
        (compiled_multi, _new_multi_carry_inputs),
        (compiled_packed, _new_single_packed_inputs),
    )
    for compiled, new_inputs in variants:
        warmup_inputs = new_inputs()
        input_devices = {tensor.device.type for tensor in warmup_inputs}
        _require(input_devices == {"cpu"}, f"expected CPU inputs, got {input_devices}")
        for _ in range(ARGS.warmup_calls):
            compiled(*warmup_inputs)
    counters_after_warmup = _counter_snapshot()

    element_size = torch.empty((), dtype=torch.float64).element_size()
    predicate_events = ARGS.profile_calls * (LOOP_ITERATIONS + 1)
    carry_events = ARGS.profile_calls * LOOP_ITERATIONS
    evidence = {
        MULTI_CARRY: _variant_evidence(
            MULTI_CARRY,
            compiled_multi,
            _new_multi_carry_inputs,
            {
                "predicate": torch.empty((), dtype=torch.bool).element_size(),
                "iteration_carry": torch.empty((), dtype=torch.int64).element_size(),
                "field_carry": CELLS * element_size,
                "history_carry": CELLS * WIDTH * element_size,
            },
            {
                "predicate": predicate_events,
                "iteration_carry": carry_events,
                "field_carry": carry_events,
                "history_carry": carry_events,
            },
        ),
        SINGLE_PACKED_CARRY: _variant_evidence(
            SINGLE_PACKED_CARRY,
            compiled_packed,
            _new_single_packed_inputs,
            {
                "predicate": torch.empty((), dtype=torch.bool).element_size(),
                "packed_carry": PACKED_ELEMENTS * element_size,
            },
            {
                "predicate": predicate_events,
                "packed_carry": carry_events,
            },
        ),
    }
    counters_after_steady_state = _counter_snapshot()
    steady_state_counter_delta = _counter_delta(
        counters_after_warmup, counters_after_steady_state
    )

    graph_breaks = _graph_break_count(counters_after_steady_state)
    new_frames = _counter_value(
        counters_after_steady_state, "frames", "total"
    ) - _counter_value(counters_after_warmup, "frames", "total")
    new_unique_graphs = _counter_value(
        counters_after_steady_state, "stats", "unique_graphs"
    ) - _counter_value(counters_after_warmup, "stats", "unique_graphs")
    _require(graph_breaks == 0, f"unexpected graph breaks: {graph_breaks}")
    _require(
        new_frames == 0, f"unexpected steady-state frame compilations: {new_frames}"
    )
    _require(
        new_unique_graphs == 0,
        f"unexpected steady-state unique graphs: {new_unique_graphs}",
    )

    inplace_rejection = _inplace_rejection_evidence()

    cpu_capability = None
    if hasattr(torch.backends, "cpu") and hasattr(
        torch.backends.cpu, "get_cpu_capability"
    ):
        cpu_capability = torch.backends.cpu.get_cpu_capability()

    script_path = Path(__file__).resolve()
    report = {
        "reproducer": "torch_inductor_while_loop_allocation",
        "affected": all(variant["affected"] for variant in evidence.values())
        and inplace_rejection["input_mutation_during_higher_order_op_tracing"],
        "assert_affected_requested": ARGS.assert_affected,
        "configuration": {
            "cells": CELLS,
            "width": WIDTH,
            "loop_iterations": LOOP_ITERATIONS,
            "dtype": str(torch.float64),
            "packed_workspace_elements": PACKED_ELEMENTS,
            "packed_workspace_bytes": PACKED_ELEMENTS * element_size,
            "packed_workspace_caller_owned": True,
            "warmup_calls": ARGS.warmup_calls,
            "equivalence_calls": ARGS.equivalence_calls,
            "profile_calls": ARGS.profile_calls,
            "timing_samples": ARGS.timing_samples,
            "calls_per_sample": ARGS.calls_per_sample,
        },
        "variants": evidence,
        INPLACE_PACKED_CARRY: inplace_rejection,
        "compilation": {
            "fullgraph": True,
            "dynamic": False,
            "graph_break_count": graph_breaks,
            "steady_state_new_frames": new_frames,
            "steady_state_new_unique_graphs": new_unique_graphs,
            "counters_after_warmup": counters_after_warmup,
            "counters_after_steady_state": counters_after_steady_state,
            "steady_state_counter_delta": steady_state_counter_delta,
        },
        "provenance": {
            "script_path": str(script_path),
            "script_size_bytes": script_path.stat().st_size,
            "script_sha256": _sha256(script_path),
        },
        "environment": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "torch_version": torch.__version__,
            "torch_git_version": getattr(torch.version, "git_version", None),
            "torch_debug_build": getattr(torch.version, "debug", None),
            "torch_cuda_version": torch.version.cuda,
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_cpu_capability": cpu_capability,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "torch_parallel_info": torch.__config__.parallel_info(),
            "inductor_cache_directory": str(CACHE_DIRECTORY),
            "inductor_cache_directory_is_temporary": CACHE_DIRECTORY_IS_TEMPORARY,
            "trace_directory": str(TRACE_DIRECTORY),
            "trace_directory_is_temporary": TRACE_DIRECTORY_IS_TEMPORARY,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
