#!/usr/bin/env python3
"""Fail-closed repeated-advance stability evidence for a mixed Torch case.

This is a correctness/stability runner, not a throughput benchmark.  It keeps
the measurement loop free of profilers and checkpoint clones; tracing occurs
only afterwards on separate warmed advances.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import gmes
from benchmarks.torch_tuning import (
    MANIFEST,
    _build_case,
    _counter_delta,
    _counter_snapshot,
    _current_rss_provider,
    _initialize_fields,
    _synchronize,
    load_manifest,
)

MIN_QUALIFICATION_BATCHES = 15
MIN_QUALIFICATION_STEPS = 100
PROFILE_EXECUTIONS = 3
COPY_OR_MATERIALIZE_OPERATORS = {
    "aten::clone",
    "aten::copy_",
    "aten::_to_copy",
    "aten::contiguous",
}
_PHASE_MESSAGES = {
    "warmup-start": b"torch-memory-stability phase=warmup-start\n",
    "warmup-end": b"torch-memory-stability phase=warmup-end\n",
    "observer-prime-start": b"torch-memory-stability phase=observer-prime-start\n",
    "observer-primed": b"torch-memory-stability phase=observer-primed\n",
    "measurement-start": b"torch-memory-stability phase=measurement-start\n",
    "measurement-end": b"torch-memory-stability phase=measurement-end\n",
    "profiler-start": b"torch-memory-stability phase=profiler-start\n",
    "profiler-end": b"torch-memory-stability phase=profiler-end\n",
}


def _phase_progress(enabled: bool, phase: str) -> None:
    """Emit one optional fixed diagnostic marker outside sample collection."""
    if enabled:
        os.write(2, _PHASE_MESSAGES[phase])


@dataclass
class _Telemetry:
    """Fixed storage for one complete observer series."""

    rss: np.ndarray
    candidate_tensors: np.ndarray
    reference_tensors: np.ndarray
    candidate_finite: np.ndarray
    reference_finite: np.ndarray
    field_error: np.ndarray
    storage_stable: np.ndarray
    cuda_allocated: np.ndarray | None
    cuda_reserved: np.ndarray | None


def _new_telemetry(samples: int, *, cuda: bool) -> _Telemetry:
    """Allocate every retained record before collecting an RSS sample."""
    if samples < 1:
        raise ValueError("telemetry requires at least one sample")
    return _Telemetry(
        rss=np.empty(samples, dtype=np.int64),
        candidate_tensors=np.empty(samples, dtype=np.int64),
        reference_tensors=np.empty(samples, dtype=np.int64),
        candidate_finite=np.empty(samples, dtype=np.bool_),
        reference_finite=np.empty(samples, dtype=np.bool_),
        field_error=np.empty(samples, dtype=np.float64),
        storage_stable=np.empty(samples, dtype=np.bool_),
        cuda_allocated=np.empty(samples, dtype=np.int64) if cuda else None,
        cuda_reserved=np.empty(samples, dtype=np.int64) if cuda else None,
    )


def _telemetry_report(telemetry: _Telemetry) -> dict[str, object]:
    """Convert fixed records after, never during, collection."""
    result: dict[str, object] = {
        "rss_samples_bytes": telemetry.rss.tolist(),
        "candidate_tensors": telemetry.candidate_tensors.tolist(),
        "reference_tensors": telemetry.reference_tensors.tolist(),
        "candidate_finite": telemetry.candidate_finite.tolist(),
        "reference_finite": telemetry.reference_finite.tolist(),
        "field_error_samples": telemetry.field_error.tolist(),
        "storage_stable": telemetry.storage_stable.tolist(),
    }
    if telemetry.cuda_allocated is not None and telemetry.cuda_reserved is not None:
        result["cuda_samples"] = [
            {"allocated": int(allocated), "reserved": int(reserved)}
            for allocated, reserved in zip(
                telemetry.cuda_allocated, telemetry.cuda_reserved, strict=True
            )
        ]
    else:
        result["cuda_samples"] = []
    return result


@dataclass(frozen=True)
class _BufferSlot:
    """A prebound module buffer and the storage it owned after warmup."""

    owner: torch.nn.Module
    name: str
    tensor: torch.Tensor
    data_ptr: int


@dataclass(frozen=True)
class _ModuleBufferKeys:
    """The complete registered-buffer key set of one observed module."""

    owner: torch.nn.Module
    names: tuple[str, ...]


@dataclass(frozen=True)
class _BufferObserver:
    """Check prebound buffers without rebuilding named-buffer dictionaries."""

    slots: tuple[_BufferSlot, ...]
    module_buffers: tuple[_ModuleBufferKeys, ...]

    def stable(self) -> bool:
        """Return whether every observed module slot still owns its storage."""
        return all(
            len(module.owner._buffers) == len(module.names)
            and all(name in module.owner._buffers for name in module.names)
            for module in self.module_buffers
        ) and all(
            slot.owner._buffers.get(slot.name) is slot.tensor
            and slot.tensor.data_ptr() == slot.data_ptr
            for slot in self.slots
        )

    def finite(self) -> bool:
        """Return whether every observed live tensor remains finite."""
        return all(
            isinstance(current := slot.owner._buffers.get(slot.name), torch.Tensor)
            and bool(torch.isfinite(current).all().item())
            for slot in self.slots
        )


def _buffer_observer(
    roots: tuple[torch.nn.Module | None, ...], *, skip_packed_loop_state: bool = False
) -> _BufferObserver:
    """Bind buffer slots once, preserving module-recursive coverage."""
    slots: list[_BufferSlot] = []
    modules: list[_ModuleBufferKeys] = []
    seen_modules: set[int] = set()
    for root in roots:
        if root is None:
            continue
        for module in root.modules():
            if id(module) in seen_modules:
                continue
            seen_modules.add(id(module))
            modules.append(_ModuleBufferKeys(module, tuple(module._buffers)))
            for name, tensor in module._buffers.items():
                if tensor is None or (
                    skip_packed_loop_state and name == "_packed_loop_state"
                ):
                    continue
                slots.append(_BufferSlot(module, name, tensor, tensor.data_ptr()))
    return _BufferObserver(tuple(slots), tuple(modules))


def _storage_observer(simulation: gmes.TorchSimulation) -> _BufferObserver:
    """Bind permanent simulation and transparent-auxiliary storage after warmup."""
    roots: list[torch.nn.Module | None] = []
    visited: set[int] = set()

    def visit(current: gmes.TorchSimulation) -> None:
        if id(current) in visited:
            return
        visited.add(id(current))
        roots.extend(
            (
                current.state,
                current.plan,
                current.sources,
                current.probes,
                current._dispersive_overlay,
            )
        )
        for auxiliary in current.sources.auxiliaries:
            if not isinstance(auxiliary, gmes.TorchSimulation):
                raise RuntimeError(
                    "storage observation requires TorchSimulation transparent auxiliaries"
                )
            visit(auxiliary)

    visit(simulation)
    return _buffer_observer(tuple(roots))


def _live_observer(simulation: gmes.TorchSimulation) -> _BufferObserver:
    """Bind field, material, PML, and source buffers without clones."""
    return _buffer_observer(
        (simulation.state, simulation.sources), skip_packed_loop_state=True
    )


def _field_shapes(simulation: gmes.TorchSimulation) -> tuple[tuple[int, ...], ...]:
    """Return the six live field shapes used to classify full-domain clones."""
    return tuple(
        tuple(int(size) for size in getattr(simulation.state, name).shape)
        for name in ("ex", "ey", "ez", "hx", "hy", "hz")
    )


def _field_error(
    candidate: gmes.TorchSimulation, reference: gmes.TorchSimulation
) -> float:
    """Return a synchronized maximum live-field difference without snapshots."""
    errors = []
    for name in ("ex", "ey", "ez", "hx", "hy", "hz"):
        difference = torch.amax(
            torch.abs(getattr(candidate.state, name) - getattr(reference.state, name))
        )
        errors.append(float(difference.item()))
    return max(errors)


def _cuda_current_memory(device: torch.device) -> tuple[int, int]:
    """Read both current CUDA allocator totals from one validated snapshot."""
    snapshot = torch.cuda.memory_stats_as_nested_dict(device)
    if not isinstance(snapshot, dict):
        raise RuntimeError("CUDA allocator snapshot is not a mapping")

    def current(kind: str) -> int:
        section = snapshot.get(kind)
        all_devices = section.get("all") if isinstance(section, dict) else None
        value = all_devices.get("current") if isinstance(all_devices, dict) else None
        if type(value) is not int or value < 0:
            raise RuntimeError(
                "CUDA allocator snapshot has no non-negative integer "
                f"{kind}.all.current value"
            )
        return value

    return current("allocated_bytes"), current("reserved_bytes")


def _observe_into(
    telemetry: _Telemetry,
    index: int,
    *,
    candidate: gmes.TorchSimulation,
    reference: gmes.TorchSimulation,
    read_rss,
    candidate_live: _BufferObserver,
    reference_live: _BufferObserver,
    candidate_storage: _BufferObserver,
    reference_storage: _BufferObserver,
    compare_fields: bool,
) -> None:
    """Record one complete live observation into already allocated storage."""
    _synchronize(candidate.device)
    candidate_count, candidate_finite = (
        len(candidate_live.slots),
        candidate_live.finite(),
    )
    reference_count, reference_finite = (
        len(reference_live.slots),
        reference_live.finite(),
    )
    error = _field_error(candidate, reference) if compare_fields else 0.0
    storage_stable = candidate_storage.stable() and reference_storage.stable()
    if candidate.device.type == "cuda":
        assert telemetry.cuda_allocated is not None
        assert telemetry.cuda_reserved is not None
        allocated, reserved = _cuda_current_memory(candidate.device)
        telemetry.cuda_allocated[index] = allocated
        telemetry.cuda_reserved[index] = reserved
    # RSS is deliberately last: every observer operation is part of this
    # sample, and no retained Python record is allocated after it.
    rss = read_rss()
    if rss is None:
        raise RuntimeError("current RSS is unavailable")
    telemetry.rss[index] = rss
    telemetry.candidate_tensors[index] = candidate_count
    telemetry.reference_tensors[index] = reference_count
    telemetry.candidate_finite[index] = candidate_finite
    telemetry.reference_finite[index] = reference_finite
    telemetry.field_error[index] = error
    telemetry.storage_stable[index] = storage_stable


def _growth_assessment(samples: list[int]) -> dict[str, object]:
    """Classify post-warmup sustained growth while retaining raw samples.

    The first, middle, and final thirds are compared as windows.  A cache
    plateau may rise during the first window, but it does not keep raising both
    subsequent windows.  The least-squares slope additionally catches an
    oscillatory retained-growth sequence rather than requiring monotonic RSS.
    """
    if len(samples) < 3:
        return {
            "adequate": False,
            "sustained_growth": None,
            "slope_bytes_per_sample": None,
        }
    split = len(samples) // 2
    first = float(np.median(samples[:split]))
    final = float(np.median(samples[split:]))
    slope = float(np.polyfit(np.arange(len(samples)), samples, 1)[0])
    sustained = bool(slope > 0.0 and final > first)
    return {
        "adequate": len(samples) >= MIN_QUALIFICATION_BATCHES,
        "sustained_growth": sustained,
        "slope_bytes_per_sample": slope,
        "first_window_median_bytes": first,
        "final_window_median_bytes": final,
    }


def _copy_record(
    events, field_shapes: tuple[tuple[int, ...], ...]
) -> list[dict[str, object]]:
    """Classify dispatcher-visible copy/materialization events by field shape."""
    result = []
    field_shape_set = set(field_shapes)
    largest_field = max((int(np.prod(shape)) for shape in field_shapes), default=0)
    for event in events:
        if event.key not in COPY_OR_MATERIALIZE_OPERATORS:
            continue
        shapes = tuple(
            tuple(int(item) for item in shape) for shape in event.input_shapes
        )
        elements = max((int(np.prod(shape)) for shape in shapes if shape), default=0)
        full_domain = (
            any(shape in field_shape_set for shape in shapes)
            or elements >= largest_field
        )
        result.append(
            {
                "operator": event.key,
                "input_shapes": shapes,
                "full_domain": full_domain,
            }
        )
    return result


def _profile_warmed_advances(
    candidate: gmes.TorchSimulation,
    reference: gmes.TorchSimulation,
) -> list[dict[str, object]]:
    """Trace separate warmed candidate advances outside the measured hot path."""
    activities = [torch.profiler.ProfilerActivity.CPU]
    if candidate.device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    records = []
    shapes = _field_shapes(candidate)
    for index in range(PROFILE_EXECUTIONS):
        with torch.profiler.profile(
            activities=activities, record_shapes=True
        ) as profiler:
            candidate.advance(1)
            _synchronize(candidate.device)
        reference.advance(1)
        _synchronize(reference.device)
        copies = _copy_record(profiler.events(), shapes)
        records.append(
            {
                "execution": index,
                "operator_copy_events": copies,
                "full_domain_copy_count": sum(
                    int(item["full_domain"]) for item in copies
                ),
                "plan_bounded_copy_count": sum(
                    int(not item["full_domain"]) for item in copies
                ),
            }
        )
    return records


def _git(candidate_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run Git bound to the verified candidate root, never ambient cwd."""
    return subprocess.run(
        ("git", "-C", str(candidate_root), *args),
        capture_output=True,
        text=True,
        check=False,
    )


def _sha256(path: Path) -> str:
    """Hash one local evidence input without exposing its absolute path."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_provenance() -> dict[str, object]:
    """Bind evidence to the checkout that supplied both runtime and harness."""
    harness = Path(__file__).resolve()
    candidate_root = harness.parents[1]
    completed = _git(candidate_root, "rev-parse", "--show-toplevel")
    root = (
        Path(completed.stdout.strip()).resolve() if completed.returncode == 0 else None
    )
    module = Path(gmes.__file__).resolve()
    valid = (
        root is not None
        and module.is_relative_to(root)
        and harness.is_relative_to(root)
    )
    commit = _git(candidate_root, "rev-parse", "HEAD")
    dirty = _git(candidate_root, "status", "--porcelain")
    source_files = (
        "gmes/torch_fdtd.py",
        "gmes/torch_source.py",
        "gmes/torch_plan.py",
    )
    source_hashes = (
        {name: _sha256(root / name) for name in source_files}
        if valid and all((root / name).is_file() for name in source_files)
        else None
    )
    source_digest = None
    if source_hashes is not None:
        digest = hashlib.sha256()
        for name, value in source_hashes.items():
            digest.update(name.encode())
            digest.update(b"\0")
            digest.update(value.encode())
            digest.update(b"\0")
        source_digest = digest.hexdigest()
    harness_hash = _sha256(harness)
    return {
        "checkout_module_origin_valid": valid,
        "candidate_commit": (
            commit.stdout.strip() if commit.returncode == 0 else "unavailable"
        ),
        "candidate_dirty": (
            bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
        ),
        "harness_sha256": harness_hash,
        "collector_harness_sha256": harness_hash,
        "runtime_source_sha256": source_digest,
        "runtime_source_files_sha256": source_hashes,
    }


def reevaluate(raw: dict[str, object], raw_bytes: bytes) -> dict[str, object]:
    """Evaluate old evidence without relabeling it as a new collection."""
    provenance = _candidate_provenance()
    normalized = dict(raw)
    normalizations = []
    samples = raw.get("cuda_samples")
    if (
        raw.get("device_type") == "cuda"
        and isinstance(samples, list)
        and samples
        and all(
            isinstance(item, dict)
            and isinstance(item.get("allocated"), int)
            and isinstance(item.get("reserved"), int)
            for item in samples
        )
    ):
        normalized["cuda_assessment"] = {
            "allocated": _growth_assessment([item["allocated"] for item in samples]),
            "reserved": _growth_assessment([item["reserved"] for item in samples]),
        }
        normalizations.append("derived-v3-cuda-assessments-from-raw-samples")
    current_diagnostic = _evaluate(normalized)
    original_qualified = raw.get("qualified") is True
    original_failed = raw.get("diagnostic_ok") is False or bool(
        raw.get("hard_failures")
    )
    non_promotion_reasons = []
    if "qualified" not in raw:
        non_promotion_reasons.append("raw qualification status is missing")
    elif not original_qualified:
        non_promotion_reasons.append("raw collection was not qualified")
    if original_failed:
        non_promotion_reasons.append("raw collection recorded a failure")
    if current_diagnostic["qualified"] is not True:
        non_promotion_reasons.append("current evaluator diagnostic is not qualified")
    effective_decision = dict(current_diagnostic)
    effective_decision["qualified"] = bool(
        original_qualified
        and not original_failed
        and current_diagnostic["qualified"] is True
    )
    effective_decision["diagnostic_ok"] = bool(
        raw.get("diagnostic_ok") is True and current_diagnostic["diagnostic_ok"] is True
    )
    return {
        "schema": "torch-memory-stability-reevaluation-v2",
        "raw_result_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "raw_collector_harness_sha256": raw.get(
            "collector_harness_sha256", raw.get("harness_sha256")
        ),
        "raw_contract_status_preserved": {
            name: raw.get(name)
            for name in (
                "diagnostic_ok",
                "qualified",
                "hard_failures",
                "qualification_errors",
            )
        },
        "reevaluator_harness_sha256": provenance["collector_harness_sha256"],
        "reevaluator_runtime_source_sha256": provenance["runtime_source_sha256"],
        "evaluation_normalizations": normalizations,
        "current_evaluator_diagnostic": current_diagnostic,
        "effective_decision": effective_decision,
        "non_promotion_reasons": non_promotion_reasons,
        "new_collection": False,
    }


def _sha256_text(value: object, length: int = 64) -> bool:
    """Return whether one portable provenance digest has the expected shape."""
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _complete_observer_warmup(
    value: object, *, cuda: bool, require_healthy: bool
) -> bool:
    """Require the primed full observer record before qualification is possible."""
    if not isinstance(value, dict):
        return False
    if (
        value.get("complete") is not True
        or value.get("compiler_counter_collector_primed") is not True
        or not isinstance(value.get("field_error_checked"), bool)
    ):
        return False
    for name, predicate in (
        ("rss_samples_bytes", lambda item: type(item) is int),
        ("candidate_tensors", lambda item: type(item) is int),
        ("reference_tensors", lambda item: type(item) is int),
        ("candidate_finite", lambda item: type(item) is bool),
        ("reference_finite", lambda item: type(item) is bool),
        (
            "field_error_samples",
            lambda item: isinstance(item, float) and np.isfinite(item),
        ),
        ("storage_stable", lambda item: type(item) is bool),
    ):
        items = value.get(name)
        if not isinstance(items, list) or len(items) != 1 or not predicate(items[0]):
            return False
    cuda_samples = value.get("cuda_samples")
    cuda_complete = (
        cuda_samples == []
        if not cuda
        else (
            isinstance(cuda_samples, list)
            and len(cuda_samples) == 1
            and isinstance(cuda_samples[0], dict)
            and type(cuda_samples[0].get("allocated")) is int
            and type(cuda_samples[0].get("reserved")) is int
        )
    )
    if not cuda_complete:
        return False
    if not require_healthy:
        return True
    return (
        value["candidate_finite"] == [True]
        and value["reference_finite"] == [True]
        and value["storage_stable"] == [True]
        and value["field_error_checked"] is True
    )


def _collection_provenance_valid(result: dict[str, object]) -> bool:
    """Require explicit candidate and runtime binding for a new qualification."""
    return (
        _sha256_text(result.get("collector_harness_sha256"))
        and _sha256_text(result.get("runtime_source_sha256"))
        and _sha256_text(result.get("candidate_commit"), length=40)
        and type(result.get("candidate_dirty")) is bool
    )


def _evaluate(result: dict[str, object]) -> dict[str, object]:
    """Return explicit diagnostic and fail-closed qualification decisions."""
    hard_failures = []
    qualification_errors = []
    advance_role = result.get("advance_role", "both")
    if result.get("checkout_module_origin_valid") is not True:
        hard_failures.append(
            "runtime or harness origin is outside the candidate checkout"
        )
    if not _collection_provenance_valid(result):
        hard_failures.append("new collection provenance is incomplete")
    if not _complete_observer_warmup(
        result.get("observer_warmup"),
        cuda=result.get("device_type") == "cuda",
        require_healthy=advance_role == "both",
    ):
        hard_failures.append(
            "complete healthy primed observer warmup evidence is missing"
        )
    if result.get("storage_stable") is not True:
        hard_failures.append("live simulation storage was replaced")
    if result.get("all_batches_finite") is not True:
        hard_failures.append("candidate or eager reference state became non-finite")
    if result.get("field_error_checked", True) is True:
        errors = result.get("field_error_samples")
        if (
            not isinstance(errors, list)
            or not errors
            or not all(
                isinstance(value, float) and np.isfinite(value) for value in errors
            )
        ):
            hard_failures.append("field-error samples are missing or non-finite")
        elif max(errors) > float(result["field_error_tolerance"]):
            hard_failures.append("compiled and eager fields diverged beyond tolerance")
    if advance_role == "both":
        copy_records = result.get("operator_copy_records")
        if (
            not isinstance(copy_records, list)
            or len(copy_records) != PROFILE_EXECUTIONS
        ):
            hard_failures.append("warmed dispatcher copy observations are incomplete")
        elif any(record.get("full_domain_copy_count", 0) for record in copy_records):
            hard_failures.append(
                "a warmed advance observed a full-domain copy/materialization"
            )
    compiler = result.get("compiler_counter_delta")
    if result.get("compile_policy") == "compile" and advance_role == "both":
        if result.get("compiled_region_executed") is not True:
            hard_failures.append("compiled execution region was not established")
        if not isinstance(compiler, dict) or any(
            compiler.get(name) != 0
            for name in ("graph_breaks", "unique_graphs", "frames_total")
        ):
            hard_failures.append(
                "post-warmup compiler graph/recompile/fallback counters changed"
            )
    rss = result.get("rss_assessment")
    if not isinstance(rss, dict):
        hard_failures.append("current RSS evidence is unavailable")
    elif rss.get("adequate") is True and rss.get("sustained_growth") is True:
        hard_failures.append("RSS shows sustained post-warmup retained growth")
    if result.get("device_type") == "cuda":
        cuda = result.get("cuda_assessment")
        if not isinstance(cuda, dict):
            hard_failures.append(
                "CUDA allocation and reservation evidence is unavailable"
            )
        else:
            for name in ("allocated", "reserved"):
                assessment = cuda.get(name)
                if not isinstance(assessment, dict) or (
                    assessment.get("adequate") is True
                    and assessment.get("sustained_growth") is True
                ):
                    hard_failures.append(
                        f"CUDA live {name} memory shows sustained retained growth"
                    )
    if result.get("compile_policy") != "compile":
        qualification_errors.append(
            "eager evidence is diagnostic and cannot qualify compiled stability"
        )
    if advance_role != "both":
        qualification_errors.append(
            "single-runtime or observation control cannot qualify advance stability"
        )
    if int(result.get("batches", 0)) < MIN_QUALIFICATION_BATCHES:
        qualification_errors.append("fewer than 15 post-warmup batches were observed")
    if int(result.get("steps_per_batch", 0)) < MIN_QUALIFICATION_STEPS:
        qualification_errors.append(
            "fewer than 100 steps per measured batch were observed"
        )
    if isinstance(rss, dict) and rss.get("adequate") is not True:
        qualification_errors.append("RSS sample series is too short for qualification")
    if result.get("lowered_copy_materialization_verified") is not True:
        qualification_errors.append(
            "lowered copy/materialization evidence is not established by dispatcher tracing"
        )
    if result.get("device_type") == "cuda" and (
        not isinstance(result.get("cuda_assessment"), dict)
        or any(
            not isinstance(result["cuda_assessment"].get(name), dict)
            or result["cuda_assessment"][name].get("adequate") is not True
            for name in ("allocated", "reserved")
        )
    ):
        qualification_errors.append(
            "CUDA allocation sample series is too short for qualification"
        )
    if result.get("observation_only") is True:
        qualification_errors.append(
            "observation-only control cannot qualify advance stability"
        )
    qualified = not hard_failures and not qualification_errors
    return {
        "diagnostic_ok": not hard_failures,
        "qualified": qualified,
        "hard_failures": hard_failures,
        "qualification_errors": qualification_errors,
    }


def observe(
    *,
    case: str,
    device: str,
    precision: str,
    compile_policy: str,
    warmup: int,
    steps: int,
    batches: int,
    observation_only: bool = False,
    advance_role: str | None = None,
    phase_progress: bool = False,
) -> dict[str, object]:
    """Run one fixed mixed case and collect stability evidence."""
    if batches < 1 or steps < 1 or warmup < 1:
        raise ValueError("warmup, steps, and batches must all be positive")
    if advance_role is None:
        advance_role = "none" if observation_only else "both"
    if advance_role not in {"both", "candidate", "reference", "none"}:
        raise ValueError("advance role is invalid")
    if observation_only != (advance_role == "none"):
        raise ValueError("observation-only mode requires the none advance role")
    manifest = load_manifest(MANIFEST)
    spec, space, geometry, sources, bloch = _build_case(case, manifest)
    if not sources:
        sources = (
            gmes.PointSource(
                gmes.Continuous(0.2, phase=0.3, width=1), (0, 0, 0), gmes.Ex, amp=0.25
            ),
            gmes.PointSource(gmes.Bandpass(0.3, 0.1), (0, 0, 0), gmes.Jx, amp=0.1),
        )
    runtime = gmes.TorchRuntimeConfig(
        device=device,
        precision=precision,
        compile_policy=compile_policy,
        cpu_threads=1,
        cpu_interop_threads=1,
    )
    candidate = gmes.TorchSimulation(
        space=space, geometry=geometry, sources=sources, bloch=bloch, runtime=runtime
    )
    reference = gmes.TorchSimulation(
        space=space,
        geometry=geometry,
        sources=sources,
        bloch=bloch,
        runtime=gmes.TorchRuntimeConfig(
            device=device,
            precision=precision,
            compile_policy="eager",
            cpu_threads=1,
            cpu_interop_threads=1,
        ),
    )
    seed, scale = manifest["reference"]["seed"], manifest["reference"]["field_scale"]
    _initialize_fields(candidate, seed, scale)
    _initialize_fields(reference, seed, scale)
    _phase_progress(phase_progress, "warmup-start")
    if advance_role in {"both", "candidate"}:
        candidate.advance(warmup)
    if advance_role in {"both", "reference"}:
        reference.advance(warmup)
    _phase_progress(phase_progress, "warmup-end")
    _synchronize(candidate.device)
    candidate_live, reference_live = _live_observer(candidate), _live_observer(
        reference
    )
    candidate_storage, reference_storage = _storage_observer(
        candidate
    ), _storage_observer(reference)
    read_rss, provider = _current_rss_provider()
    # Allocate both series before the observer's first allocation-prone pass.
    observer_warmup = _new_telemetry(1, cuda=candidate.device.type == "cuda")
    measured = _new_telemetry(batches, cuda=candidate.device.type == "cuda")
    field_error_checked = advance_role in {"both", "none"}
    try:
        # Counter access is part of collector setup and can initialize Torch
        # bookkeeping. Prime it before the recorded complete observer pass.
        _phase_progress(phase_progress, "observer-prime-start")
        _counter_snapshot()
        _observe_into(
            observer_warmup,
            0,
            candidate=candidate,
            reference=reference,
            read_rss=read_rss,
            candidate_live=candidate_live,
            reference_live=reference_live,
            candidate_storage=candidate_storage,
            reference_storage=reference_storage,
            compare_fields=field_error_checked,
        )
        _phase_progress(phase_progress, "observer-primed")
        # The measured compiler counter window starts after complete observer
        # warmup, not merely after solver warmup.
        counters_before = _counter_snapshot()
        _phase_progress(phase_progress, "measurement-start")
        for index in range(batches):
            if advance_role in {"both", "candidate"}:
                candidate.advance(steps)
            if advance_role in {"both", "reference"}:
                reference.advance(steps)
            _observe_into(
                measured,
                index,
                candidate=candidate,
                reference=reference,
                read_rss=read_rss,
                candidate_live=candidate_live,
                reference_live=reference_live,
                candidate_storage=candidate_storage,
                reference_storage=reference_storage,
                compare_fields=field_error_checked,
            )
        _phase_progress(phase_progress, "measurement-end")
    finally:
        close = getattr(read_rss, "close", None)
        if callable(close):
            close()
    if advance_role == "both":
        _phase_progress(phase_progress, "profiler-start")
        copy_records = _profile_warmed_advances(candidate, reference)
        _phase_progress(phase_progress, "profiler-end")
    else:
        copy_records = []
    _synchronize(candidate.device)
    counters = _counter_delta(counters_before, _counter_snapshot())
    observer_report = _telemetry_report(observer_warmup)
    measured_report = _telemetry_report(measured)
    all_finite = bool(
        np.all(measured.candidate_finite) and np.all(measured.reference_finite)
    )
    if field_error_checked:
        all_finite = all_finite and bool(np.all(np.isfinite(measured.field_error)))
    storage_stable = bool(
        np.all(observer_warmup.storage_stable) and np.all(measured.storage_stable)
    )
    completed_cuda = measured_report["cuda_samples"]
    result = {
        "schema": "torch-memory-stability-v3",
        "case": spec["name"],
        "device": str(candidate.device),
        "device_type": candidate.device.type,
        "precision": precision,
        "compile_policy": compile_policy,
        "warmup_steps": warmup,
        "steps_per_batch": 0 if advance_role == "none" else steps,
        "batches": batches,
        "observation_only": observation_only,
        "advance_role": advance_role,
        "field_error_checked": field_error_checked,
        "rss_provider": provider,
        "observer_warmup": {
            "complete": True,
            "compiler_counter_collector_primed": True,
            "field_error_checked": field_error_checked,
            **observer_report,
        },
        "rss_samples_bytes": measured_report["rss_samples_bytes"],
        "rss_assessment": _growth_assessment(measured_report["rss_samples_bytes"]),
        "cuda_samples": completed_cuda,
        "cuda_assessment": (
            {
                "allocated": _growth_assessment(
                    [item["allocated"] for item in completed_cuda]
                ),
                "reserved": _growth_assessment(
                    [item["reserved"] for item in completed_cuda]
                ),
            }
            if completed_cuda
            else None
        ),
        "batch_health": [
            {
                "batch": index,
                "candidate_tensors": int(measured.candidate_tensors[index]),
                "candidate_finite": bool(measured.candidate_finite[index]),
                "reference_tensors": int(measured.reference_tensors[index]),
                "reference_finite": bool(measured.reference_finite[index]),
                "field_error_finite": (
                    bool(np.isfinite(measured.field_error[index]))
                    if field_error_checked
                    else None
                ),
            }
            for index in range(batches)
        ],
        "all_batches_finite": all_finite,
        "field_error_samples": measured_report["field_error_samples"],
        "field_error_tolerance": 1e-10 if precision == "float64" else 1e-5,
        "storage_stable": storage_stable
        and candidate_storage.stable()
        and reference_storage.stable(),
        "compiler_counter_delta": counters,
        "compiled_region_topology": getattr(
            candidate, "_compiled_region_topology", None
        ),
        "compiled_region_executed": compile_policy == "compile"
        and getattr(
            candidate,
            "_compiled_region_topology",
            "local-eager-stencil-and-material-phases",
        )
        != "local-eager-stencil-and-material-phases",
        "operator_copy_records": copy_records,
        "observed_operator_full_domain_copy": any(
            record["full_domain_copy_count"] for record in copy_records
        ),
        "copy_observation_scope": (
            "dispatcher-visible clone/copy/materialization operators only; "
            "compiled lowering internals are not observed"
        ),
        "lowered_copy_materialization_verified": False,
        **_candidate_provenance(),
    }
    result.update(_evaluate(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="all-material-2d")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--precision", choices=("float32", "float64"), default="float64"
    )
    parser.add_argument(
        "--compile-policy", choices=("eager", "compile"), default="compile"
    )
    parser.add_argument(
        "--mode",
        choices=("diagnostic", "qualify", "observation-control"),
        default="qualify",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batches", type=int, default=15)
    parser.add_argument(
        "--advance-role",
        choices=("both", "candidate", "reference", "none"),
        default="both",
        help="diagnostic control role; only both can qualify stability",
    )
    parser.add_argument(
        "--phase-progress",
        action="store_true",
        help="emit fixed pre/post-phase markers to stderr outside measured samples",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reevaluate",
        type=Path,
        help="evaluate an existing JSON record without collecting or recertifying it",
    )
    args = parser.parse_args()
    if args.reevaluate is not None:
        raw_bytes = args.reevaluate.read_bytes()
        result = reevaluate(json.loads(raw_bytes), raw_bytes)
    else:
        advance_role = (
            "none" if args.mode == "observation-control" else args.advance_role
        )
        result = observe(
            case=args.case,
            device=args.device,
            precision=args.precision,
            compile_policy=args.compile_policy,
            warmup=args.warmup,
            steps=args.steps,
            batches=args.batches,
            observation_only=args.mode == "observation-control",
            advance_role=advance_role,
            phase_progress=args.phase_progress,
        )
        result["mode"] = args.mode
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    if args.reevaluate is None and (
        not result["diagnostic_ok"]
        or (args.mode == "qualify" and not result["qualified"])
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
