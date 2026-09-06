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
import subprocess
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


def _storage_digest(simulation: gmes.TorchSimulation) -> str:
    """Hash fixed-storage identity without recording process addresses."""
    digest = hashlib.sha256()
    for name, address in sorted(simulation.buffer_addresses().items()):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(address).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _live_finite(simulation: gmes.TorchSimulation) -> tuple[int, bool]:
    """Check live field, material, PML, and source buffers without a clone."""
    tensors = tuple(simulation.state.named_buffers()) + tuple(
        simulation.sources.named_buffers()
    )
    # Eager DM2 never enters the packed CPU solve, so this intentionally
    # uninitialized non-persistent carry is not physical material state.
    tensors = tuple(
        (name, value)
        for name, value in tensors
        if not name.endswith("._packed_loop_state")
    )
    return len(tensors), all(
        bool(torch.isfinite(value).all().item()) for _, value in tensors
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


def _evaluate(result: dict[str, object]) -> dict[str, object]:
    """Return explicit diagnostic and fail-closed qualification decisions."""
    hard_failures = []
    qualification_errors = []
    if result.get("checkout_module_origin_valid") is not True:
        hard_failures.append(
            "runtime or harness origin is outside the candidate checkout"
        )
    if result.get("storage_stable") is not True:
        hard_failures.append("live simulation storage was replaced")
    if result.get("all_batches_finite") is not True:
        hard_failures.append("candidate or eager reference state became non-finite")
    errors = result.get("field_error_samples")
    if (
        not isinstance(errors, list)
        or not errors
        or not all(isinstance(value, float) and np.isfinite(value) for value in errors)
    ):
        hard_failures.append("field-error samples are missing or non-finite")
    elif max(errors) > float(result["field_error_tolerance"]):
        hard_failures.append("compiled and eager fields diverged beyond tolerance")
    if result.get("observation_only") is not True:
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
    if result.get("compile_policy") == "compile":
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
) -> dict[str, object]:
    """Run one fixed mixed case and collect stability evidence."""
    if batches < 3 or steps < 1 or warmup < 1:
        raise ValueError("warmup >= 1, steps >= 1, and batches >= 3 are required")
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
    candidate.advance(warmup)
    reference.advance(warmup)
    _synchronize(candidate.device)
    candidate_storage, reference_storage = _storage_digest(candidate), _storage_digest(
        reference
    )
    counters_before = _counter_snapshot()
    read_rss, provider = _current_rss_provider()
    rss: list[int | None] = [None] * batches
    cuda: list[dict[str, int] | None] = [None] * batches
    health: list[dict[str, object] | None] = [None] * batches
    errors: list[float | None] = [None] * batches
    storage_stable = True
    try:
        for index in range(batches):
            if not observation_only:
                candidate.advance(steps)
                reference.advance(steps)
            _synchronize(candidate.device)
            rss_value = read_rss()
            if rss_value is None:
                raise RuntimeError("current RSS is unavailable")
            candidate_count, candidate_finite = _live_finite(candidate)
            reference_count, reference_finite = _live_finite(reference)
            error = _field_error(candidate, reference)
            rss[index] = int(rss_value)
            errors[index] = error
            health[index] = {
                "batch": index,
                "candidate_tensors": candidate_count,
                "candidate_finite": candidate_finite,
                "reference_tensors": reference_count,
                "reference_finite": reference_finite,
                "field_error_finite": bool(np.isfinite(error)),
            }
            if candidate.device.type == "cuda":
                cuda[index] = {
                    "allocated": int(torch.cuda.memory_allocated(candidate.device)),
                    "reserved": int(torch.cuda.memory_reserved(candidate.device)),
                }
            if (
                _storage_digest(candidate) != candidate_storage
                or _storage_digest(reference) != reference_storage
            ):
                storage_stable = False
    finally:
        close = getattr(read_rss, "close", None)
        if callable(close):
            close()
    copy_records = (
        _profile_warmed_advances(candidate, reference) if not observation_only else []
    )
    _synchronize(candidate.device)
    counters = _counter_delta(counters_before, _counter_snapshot())
    all_finite = all(
        isinstance(item, dict)
        and item["candidate_finite"]
        and item["reference_finite"]
        and item["field_error_finite"]
        for item in health
    )
    completed_rss = [value for value in rss if isinstance(value, int)]
    completed_errors = [value for value in errors if isinstance(value, float)]
    completed_health = [value for value in health if isinstance(value, dict)]
    completed_cuda = [value for value in cuda if isinstance(value, dict)]
    result = {
        "schema": "torch-memory-stability-v3",
        "case": spec["name"],
        "device": str(candidate.device),
        "device_type": candidate.device.type,
        "precision": precision,
        "compile_policy": compile_policy,
        "warmup_steps": warmup,
        "steps_per_batch": 0 if observation_only else steps,
        "batches": batches,
        "observation_only": observation_only,
        "rss_provider": provider,
        "rss_samples_bytes": completed_rss,
        "rss_assessment": _growth_assessment(completed_rss),
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
        "batch_health": completed_health,
        "all_batches_finite": all_finite,
        "field_error_samples": completed_errors,
        "field_error_tolerance": 1e-10 if precision == "float64" else 1e-5,
        "storage_stable": storage_stable
        and _storage_digest(candidate) == candidate_storage
        and _storage_digest(reference) == reference_storage,
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
        result = observe(
            case=args.case,
            device=args.device,
            precision=args.precision,
            compile_policy=args.compile_policy,
            warmup=args.warmup,
            steps=args.steps,
            batches=args.batches,
            observation_only=args.mode == "observation-control",
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
