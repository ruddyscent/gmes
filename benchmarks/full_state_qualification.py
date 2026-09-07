"""Local #114 full-array qualification; never production publication authority.

Native inputs must be complete archives from the pinned observer, not sampled
probes.  Candidate provenance permits an uncommitted qualification harness but
records its exact bytes.  It does not reuse or override #169 completion flags.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from benchmarks import native_oracle, torch_correctness

ROOT = Path(__file__).resolve().parents[1]
CAPTURES = (0, 1, 2, 5, 20, 100)
COMPONENTS = native_oracle.COMPONENT_NAMES
KIND = "issue114-local-full-array-qualification-v1"
COMPATIBILITY_KIND = "issue114-native-capture-schema-compatibility-v1"
NATIVE_EXECUTION_SCOPE = "eager-full-state-native-correctness"
TWO_GPU_EXECUTION_SCOPE = "eager-two-gpu-full-state-serial-torch-comparison"
TWO_GPU_COMPILED_EXECUTION_SCOPE = "compiled-two-gpu-full-state-serial-torch-comparison"
TWO_GPU_COMPILE_POLICIES = ("eager", "compile")
LEGACY_OBSERVER_MANIFEST_SHA256 = (
    "1646db1a9d7b8d1f15a6336527e63c669deca188cf6429d9c21b4fed8c93bb29"
)
HISTORICAL_CPU_ARTIFACTS = (
    ("one", "torch-cpu-baseline-one.json"),
    ("physical", "torch-cpu-baseline-physical.json"),
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_json(value):
    return json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"


def _non_performance_manifest(value):
    """Return exactly the manifest portion that controls scientific capture."""
    result = copy.deepcopy(value)
    result.pop("performance_gates", None)
    for key in tuple(result["reference"]):
        if key.startswith("performance_"):
            result["reference"].pop(key)
    return result


def _legacy_scientific_capture_inputs(value):
    """Select legacy data that must match before its schema descriptor is reused."""
    reference = {
        key: item
        for key, item in value["reference"].items()
        if not key.startswith("performance_")
        and key not in {"observer_tag", "observer_commit"}
    }
    return {
        "reference": reference,
        "correctness": value["correctness"],
        "benchmarks": value["benchmarks"],
        "physical_checks": value["physical_checks"],
    }


def _compatibility_payload(current, legacy):
    """Replace only the CPU descriptor legacy observer code still requires."""
    if _legacy_scientific_capture_inputs(current) != _legacy_scientific_capture_inputs(
        legacy
    ):
        raise ValueError("legacy manifest changes a non-performance capture input")
    compatibility = copy.deepcopy(current)
    # The v6 observer serializes its reference dict verbatim, whereas the
    # current correctness validator excludes these performance-only identities.
    # Its older manifest loader does not require either field.
    compatibility["reference"].pop("performance_observer_tag")
    compatibility["reference"].pop("performance_observer_commit")
    try:
        compatibility["performance_gates"]["cpu_acceptance"] = copy.deepcopy(
            legacy["performance_gates"]["cpu_acceptance"]
        )
    except (KeyError, TypeError) as error:
        raise ValueError("legacy CPU descriptor is absent") from error
    if _non_performance_manifest(compatibility) != _non_performance_manifest(current):
        raise ValueError("capture compatibility changes a scientific input")
    return compatibility


def _historical_cpu_artifacts(legacy, artifact_root):
    """Verify historical descriptors without treating either slice as accepted."""
    root = Path(artifact_root).resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("historical artifact root is not a real directory")
    try:
        descriptors = legacy["performance_gates"]["cpu_acceptance"]["timing_reference"][
            "slice_artifacts"
        ]
    except (KeyError, TypeError) as error:
        raise ValueError("legacy CPU artifact descriptors are absent") from error
    if not isinstance(descriptors, list) or len(descriptors) != len(
        HISTORICAL_CPU_ARTIFACTS
    ):
        raise ValueError("legacy CPU descriptor count differs")
    records = []
    for descriptor, (mode, filename) in zip(
        descriptors, HISTORICAL_CPU_ARTIFACTS, strict=True
    ):
        if (
            not isinstance(descriptor, dict)
            or descriptor.get("thread_mode") != mode
            or descriptor.get("repository_path")
            != f"benchmarks/evidence/issue-123/{filename}"
        ):
            raise ValueError("legacy CPU descriptor identity differs")
        path = root / descriptor["repository_path"]
        if path.is_symlink() or not path.is_file():
            raise ValueError("historical CPU artifact is not a regular file")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != descriptor.get("sha256") or len(raw) != descriptor.get(
            "size_bytes"
        ):
            raise ValueError("historical CPU artifact digest or size differs")
        try:
            artifact = json.loads(raw)
            accepted = artifact["diagnostic_acceptance"]["passed"]
            suite_accepted = artifact["suite_acceptance"]["passed"]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(
                "historical CPU artifact acceptance record is invalid"
            ) from error
        if (
            artifact.get("schema_version") != 3
            or accepted is not False
            or suite_accepted is not False
        ):
            raise ValueError("historical CPU artifact must remain an unaccepted slice")
        records.append(
            {
                "repository_path": descriptor["repository_path"],
                "size_bytes": len(raw),
                "sha256": digest,
                "schema_version": 3,
                "diagnostic_accepted": False,
                "suite_accepted": False,
            }
        )
    return records


def build_v6_capture_compatibility_manifest(
    *, legacy_manifest, historical_artifact_root, output
):
    """Write a temporary v6-native-capture manifest with no performance claim."""
    legacy_manifest = Path(legacy_manifest).resolve(strict=True)
    output = Path(output).resolve()
    if legacy_manifest.is_symlink() or output.exists() or output.is_symlink():
        raise ValueError("compatibility manifest path is unsafe")
    if output.is_relative_to(ROOT):
        raise ValueError("compatibility manifest must remain outside the checkout")
    raw = legacy_manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != LEGACY_OBSERVER_MANIFEST_SHA256:
        raise ValueError("legacy observer manifest digest differs")
    try:
        legacy = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("legacy observer manifest is invalid JSON") from error
    current = native_oracle.load_manifest()
    compatibility = _compatibility_payload(current, legacy)
    artifacts = _historical_cpu_artifacts(legacy, historical_artifact_root)
    rendered = _canonical_json(compatibility)
    output.write_text(rendered)
    return {
        "kind": COMPATIBILITY_KIND,
        "purpose": "native-capture-only; no-performance-evaluation-or-publication",
        "parent_manifest_sha256": _sha(native_oracle.DEFAULT_MANIFEST),
        "legacy_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "compatibility_manifest_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "non_performance_contract_sha256": hashlib.sha256(
            _canonical_json(_non_performance_manifest(current)).encode()
        ).hexdigest(),
        "capture_observer": {
            "tag": compatibility["reference"]["observer_tag"],
            "commit": compatibility["reference"]["observer_commit"],
        },
        "physics_reference": {
            "tag": compatibility["reference"]["tag"],
            "commit": compatibility["reference"]["commit"],
        },
        "historical_cpu_artifacts": artifacts,
        "changed_paths": [
            "reference.performance_observer_tag",
            "reference.performance_observer_commit",
            "performance_gates.cpu_acceptance",
        ],
    }


def load_v6_capture_compatibility_manifest(
    *, legacy_manifest, historical_artifact_root, compatibility_manifest
):
    """Fail closed unless the supplied temporary manifest is exactly regenerated."""
    compatibility_manifest = Path(compatibility_manifest).resolve(strict=True)
    if compatibility_manifest.is_symlink() or not compatibility_manifest.is_file():
        raise ValueError("compatibility manifest is not a regular file")
    temporary = compatibility_manifest.with_name(
        compatibility_manifest.name + ".expected"
    )
    if temporary.exists():
        raise ValueError("compatibility expected path already exists")
    try:
        evidence = build_v6_capture_compatibility_manifest(
            legacy_manifest=legacy_manifest,
            historical_artifact_root=historical_artifact_root,
            output=temporary,
        )
        expected = temporary.read_bytes()
    finally:
        if temporary.exists():
            temporary.unlink()
    actual = compatibility_manifest.read_bytes()
    if actual != expected:
        raise ValueError("compatibility manifest does not exactly match its transform")
    return json.loads(actual), evidence


def _git(*args):
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], stderr=subprocess.PIPE
    )


def candidate_provenance():
    """Bind actual imported code and harness bytes without machine identifiers."""
    import torch

    import gmes

    imported = Path(gmes.__file__).resolve()
    if imported.parent != ROOT / "gmes":
        raise ValueError("candidate import is outside the qualification checkout")
    paths = sorted((ROOT / "gmes").glob("*.py"))
    paths += [
        Path(__file__),
        ROOT / "benchmarks/torch_correctness.py",
        ROOT / "benchmarks/native_oracle.py",
        ROOT / "benchmarks/native_oracle_workloads.json",
        ROOT / "tests/test_full_state_qualification.py",
        ROOT / "tests/test_torch_fdtd.py",
    ]
    return {
        "commit": _git("rev-parse", "HEAD").decode().strip(),
        "tree": _git("rev-parse", "HEAD^{tree}").decode().strip(),
        "tracked_clean": not bool(_git("diff", "HEAD", "--name-only").strip()),
        "clean_including_untracked": not bool(
            _git("status", "--porcelain=v1", "--untracked-files=all").strip()
        ),
        "worktree_status_sha256": hashlib.sha256(
            _git("status", "--porcelain=v1", "--untracked-files=all")
        ).hexdigest(),
        "files": {str(p.relative_to(ROOT)): _sha(p) for p in paths},
        "runtime": {"numpy": np.__version__, "torch": torch.__version__},
        "module_origin": "candidate-checkout/gmes/__init__.py",
    }


def compare_arrays(expected, actual, tolerances):
    """Compare exact keysets/shapes and every numeric element, without broadcasting.

    The caller supplies a tolerance for every floating array. Integer topology
    is exact. This low-level result contains no assertion of reference authority.
    """
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    failures = [{"key": k, "reason": "missing"} for k in missing]
    failures += [{"key": k, "reason": "unexpected"} for k in unexpected]
    compared = []
    for key in sorted(set(expected) & set(actual)):
        left, right = np.asarray(expected[key]), np.asarray(actual[key])
        record = {"key": key, "shape": list(left.shape), "elements": left.size}
        if left.shape != right.shape:
            failures.append({"key": key, "reason": "shape"})
            continue
        if left.dtype.kind not in "biufc" or right.dtype.kind not in "biufc":
            failures.append({"key": key, "reason": "non-numeric-dtype"})
            continue
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            failures.append({"key": key, "reason": "nonfinite"})
            continue
        if left.dtype.kind in "biu":
            if right.dtype.kind not in "biu":
                failures.append({"key": key, "reason": "integer-dtype"})
                continue
            equal = np.array_equal(left, right)
            rtol = atol = 0.0
        else:
            if right.dtype.kind not in "fc" or key not in tolerances:
                failures.append({"key": key, "reason": "missing-tolerance-or-dtype"})
                continue
            rtol, atol = (tolerances[key][k] for k in ("rtol", "atol"))
            if any(
                isinstance(v, bool) or not np.isfinite(v) or v < 0 for v in (rtol, atol)
            ):
                raise ValueError("invalid comparison tolerance")
            equal = np.allclose(right, left, rtol=rtol, atol=atol, equal_nan=False)
        delta = np.abs(right.astype(np.complex128) - left.astype(np.complex128))
        record.update(
            maximum_absolute_error=float(delta.max(initial=0)),
            rtol=float(rtol),
            atol=float(atol),
        )
        compared.append(record)
        if not equal:
            failures.append({"key": key, "reason": "values"})
    return {"passed": not failures, "failures": failures, "arrays": compared}


def validate_capture_contract(arrays, captures, shapes, *, precision="float64"):
    """Require every requested full field and its corresponding simulation clock."""
    if tuple(captures) != tuple(sorted(set(captures))) or captures[0] != 0:
        raise ValueError("capture sequence is not canonical")
    seen = {
        int(key.split("/")[1])
        for key in arrays
        if key.startswith("step/") and key.split("/")[1].isdigit()
    }
    if seen != set(captures):
        raise ValueError("capture provenance differs")
    for step in captures:
        for name in COMPONENTS:
            key = f"step/{step}/field/{name}"
            if key not in arrays or arrays[key].shape != tuple(shapes[name]):
                raise ValueError("missing or malformed full field")
        clock = np.asarray(arrays.get(f"step/{step}/time"))
        if clock.shape != (3,) or not np.isfinite(clock).all() or clock[2] <= 0:
            raise ValueError("missing or malformed capture clock")
        initial = np.asarray(arrays["step/0/time"])
        if not float(clock[0]).is_integer():
            raise ValueError("capture step is not integral")
        step_count = int(clock[0])
        scalar_dtype = np.dtype(precision)
        # The runtime explicitly recomputes source_time as integer n * dt in
        # its scalar dtype after every step. Validate that exact representable
        # product rather than allowing an arbitrary float32-sized discrepancy.
        expected_time = float(
            np.multiply(
                np.asarray(step_count, dtype=scalar_dtype),
                np.asarray(clock[2], dtype=scalar_dtype),
                dtype=scalar_dtype,
            )
        )
        if clock[0] != initial[0] + step or float(clock[1]) != expected_time:
            raise ValueError("capture clock differs")
        if clock[2] != initial[2]:
            raise ValueError("capture dt differs")


def read_native_reference(path, manifest):
    """Validate complete observer archives and bind their provenance to Git bytes."""
    pinned = manifest["reference"]["observer_commit"]
    tag = manifest["reference"]["observer_tag"]
    if _git("rev-parse", f"{tag}^{{commit}}").decode().strip() != pinned:
        raise ValueError("observer tag differs from the pinned commit")
    with torch_correctness._open_bounded_npz(path) as archive:
        metadata = native_oracle._validate_archive(archive, manifest, "reference")
        provenance = {}
        for role, source_path in (
            ("source", "gmes/__init__.py"),
            ("controller", "benchmarks/native_oracle.py"),
        ):
            record = metadata["provenance"][role]
            relative = Path(record["source"]).relative_to(record["checkout"])
            if record["commit"] != pinned or relative.as_posix() != source_path:
                raise ValueError("native source/controller identity differs")
            expected_hash = hashlib.sha256(
                _git("show", f"{pinned}:{source_path}")
            ).hexdigest()
            if record["source_sha256"] != expected_hash:
                raise ValueError("native source/controller bytes differ")
            provenance[role] = {
                "commit": pinned,
                "file": source_path,
                "sha256": expected_hash,
            }
        arrays = {k: archive[k].copy() for k in archive.files if k != "metadata.json"}
    return (
        metadata,
        arrays,
        {
            "kind": "full-native-observer-archive",
            "observer_tag": tag,
            "sha256": _sha(path),
            "size_bytes": Path(path).stat().st_size,
            "provenance": provenance,
        },
    )


def _native_tfsf_auxiliary_drive(simulation, batch, captured, step):
    from gmes.torch_source import TorchPointSourceBatch

    matches = [
        ordinal
        for ordinal, auxiliary in enumerate(simulation.sources.auxiliaries)
        if auxiliary is batch.auxiliary
    ]
    if len(matches) != 1:
        raise ValueError("live TFSF batch auxiliary ownership is ambiguous")
    auxiliary_batches = [
        item
        for item in batch.auxiliary.sources.batches
        if isinstance(item, TorchPointSourceBatch)
    ]
    if (
        len(auxiliary_batches) != 1
        or auxiliary_batches[0].component != "Ex"
        or auxiliary_batches[0].paired_real
    ):
        raise ValueError("live TFSF auxiliary source layout is unsupported")
    root = f"torch/step/{step}/auxiliary/{matches[0]}/sources/batches/0"
    try:
        targets = np.asarray(captured[f"{root}/overwrite_targets"])
        models = np.asarray(captured[f"{root}/overwrite_models"])
        parameters = np.asarray(captured[f"{root}/overwrite_parameters"])
        amplitudes = np.asarray(captured[f"{root}/overwrite_amplitudes"])
        additive_targets = np.asarray(captured[f"{root}/additive_targets"])
    except KeyError as error:
        raise ValueError(
            "captured TFSF auxiliary source state is incomplete"
        ) from error
    if (
        targets.dtype != np.dtype(np.int64)
        or targets.shape != (1,)
        or models.dtype != np.dtype(np.int8)
        or models.shape != (1,)
        or parameters.dtype != np.dtype(np.float64)
        or parameters.shape != (1, 6)
        or amplitudes.dtype != np.dtype(np.float64)
        or amplitudes.shape != (1,)
        or additive_targets.dtype != np.dtype(np.int64)
        or additive_targets.shape != (0,)
        or not bool(np.isfinite(amplitudes).all())
        or not np.array_equal(amplitudes, np.ones(1, dtype=np.float64))
    ):
        raise ValueError("captured TFSF auxiliary drive is unsupported")


def _native_tfsf_capture_state(simulation, captured, step):
    from gmes.torch_source import TorchTransparentBatch

    batches = [
        (ordinal, batch)
        for ordinal, batch in enumerate(simulation.sources.batches)
        if isinstance(batch, TorchTransparentBatch)
    ]
    if not batches or len({id(batch.auxiliary) for _ordinal, batch in batches}) != 1:
        raise ValueError("live TFSF auxiliary plan is unsupported")
    state = {}
    for ordinal, batch in batches:
        root = f"torch/step/{step}/sources/batches/{ordinal}"
        try:
            targets = np.asarray(captured[f"{root}/targets"])
            samples = np.asarray(captured[f"{root}/samples"])
            weights = np.asarray(captured[f"{root}/weights"])
        except KeyError as error:
            raise ValueError("captured TFSF batch state is incomplete") from error
        if (
            targets.dtype != np.dtype(np.int64)
            or samples.dtype != np.dtype(np.int64)
            or weights.dtype != np.dtype(np.float64)
            or targets.ndim != 1
            or samples.shape != (len(targets), 2)
            or weights.shape != samples.shape
            or len(set(int(target) for target in targets)) != len(targets)
            or not bool(np.isfinite(weights).all())
        ):
            raise ValueError("captured TFSF batch layout is unsupported")
        _native_tfsf_auxiliary_drive(simulation, batch, captured, step)
        for name, value in (
            ("targets", targets),
            ("samples", samples),
            ("weights", weights),
        ):
            state[f"{ordinal}/{name}"] = value.copy()
    return state


def _validate_native_tfsf_capture_stability(baseline, candidate):
    if set(baseline) != set(candidate):
        raise ValueError("captured TFSF source schema changed")
    for key in baseline:
        before = baseline[key]
        after = candidate[key]
        if (
            before.dtype != after.dtype
            or before.shape != after.shape
            or not np.array_equal(before, after)
        ):
            raise ValueError("captured TFSF source state changed")


def _point_observables(simulation, step, arrays):
    """Project actual packed point waveforms to the observer's amp/value layout."""
    import torch

    from gmes.torch_source import TorchPointSourceBatch, _evaluate_time

    clock = torch.tensor(
        float(arrays[f"step/{step}/time"][1]),
        dtype=simulation.dtype,
        device=simulation.device,
    )
    for batch in simulation.sources.batches:
        if not isinstance(batch, TorchPointSourceBatch):
            continue
        if batch.additive_targets.numel():
            raise ValueError("native current-source projection is not implemented")
        values = torch.empty_like(batch._overwrite_values)
        _evaluate_time(
            batch.overwrite_models,
            batch.overwrite_parameters,
            clock,
            batch.paired_real,
            values,
        )
        numeric = values.detach().cpu().numpy()
        wave = numeric[:, 0].astype(np.complex128)
        if batch.paired_real:
            wave += 1j * numeric[:, 1]
        order = np.argsort(
            batch.overwrite_targets.detach().cpu().numpy(), kind="stable"
        )
        targets = batch.overwrite_targets.detach().cpu().numpy()
        amplitudes = batch.overwrite_amplitudes.detach().cpu().numpy()
        component_plan = simulation.plan.components[batch.component]
        material_ids = np.asarray(component_plan.material_ids).reshape(-1)
        try:
            materials = [
                simulation.geometry[int(material_ids[int(target)])].material
                for target in targets
            ]
            epsilons = np.asarray([material.eps_inf for material in materials])
            permeabilities = np.asarray([material.mu_inf for material in materials])
        except (AttributeError, IndexError, TypeError, ValueError) as error:
            raise ValueError(
                "point source material descriptor is not representable"
            ) from error
        key = f"step/{step}/source/{batch.component}/0-PointSource{batch.component}/values"
        arrays[key] = np.column_stack(
            (
                amplitudes[order],
                epsilons[order],
                permeabilities[order],
                wave[order],
            )
        ).reshape(-1)


def _native_execution_record(simulation):
    """Describe this harness run without suggesting compiler qualification."""
    if simulation.runtime.compile_policy != "eager":
        raise ValueError("native full-state qualification must run eagerly")
    return {
        "scope": NATIVE_EXECUTION_SCOPE,
        "compile_policy": simulation.runtime.compile_policy,
        "execution_policy": simulation.runtime.execution_policy,
    }


def run_native_case(reference, manifest, *, precision, output, device="cpu"):
    """Run complete canonical fields/state; expose unresolved source-plan coverage."""
    metadata, expected, provenance = read_native_reference(reference, manifest)
    spec = metadata["workload"]
    captures = (0, *metadata["capture_steps"])
    shapes = {name: record["shape"] for name, record in metadata["maps"].items()}
    validate_capture_contract(expected, captures, shapes)
    dt = float(expected["step/0/time"][2])
    simulation = torch_correctness._build_torch_simulation(
        spec,
        dt=dt,
        threads=1,
        device=device,
        precision=precision,
        graph_mode="eager",
        compile_mode="default",
    )
    simulation.load_host_fields(
        native_oracle.initial_field_values(
            simulation.plan.shapes,
            manifest["reference"]["seed"],
            manifest["reference"]["field_scale"],
            complex_fields=spec["complex"],
        )
    )
    simulation.advance(manifest["reference"]["precondition_steps"])
    actual = {}
    torch_correctness._component_maps(
        simulation,
        simulation.host_snapshot(),
        actual,
        torch_correctness._geometry_metadata(simulation.geometry),
        torch_correctness._logical_geometry_metadata(spec, dt),
    )
    completed = 0
    native_tfsf_baseline = None
    for step in captures:
        simulation.advance(step - completed)
        torch_correctness._independent_snapshot(simulation, step, actual)
        if spec.get("name") == "tfsf-transparent":
            captured_tfsf = _native_tfsf_capture_state(simulation, actual, step)
            if native_tfsf_baseline is None:
                native_tfsf_baseline = captured_tfsf
            else:
                _validate_native_tfsf_capture_stability(
                    native_tfsf_baseline, captured_tfsf
                )
        _point_observables(simulation, step, actual)
        completed = step
    raw_keys = [key for key in actual if key.startswith("torch/")]
    raw = {key: actual.pop(key) for key in raw_keys}
    validate_capture_contract(actual, captures, shapes, precision=precision)
    unresolved = []
    # Transparent packed parameters and native face descriptors use different
    # layouts, so native transparent value arrays remain unqualified.
    omitted = [
        key
        for key in expected
        if "/source/" in key and "Transparent" in key and key.endswith("/values")
    ]
    if omitted:
        unresolved.append("transparent-source-parameter-projection")
    expected_numeric = {k: v for k, v in expected.items() if k not in omitted}
    actual_numeric = {k: v for k, v in actual.items() if k not in omitted}
    strategies = torch_correctness._reference_strategies(metadata)
    tolerances = {}
    for key, array in expected_numeric.items():
        if array.dtype.kind in "biu":
            continue
        effective = "float64" if "/source_aux" in key else precision
        complex_value = spec["complex"] and "/source_aux" not in key
        dtype = (
            ("complex128" if effective == "float64" else "complex64")
            if complex_value
            else effective
        )
        tolerances[key] = torch_correctness._manifest_tolerance(
            manifest, key, dtype, strategies, spec["name"]
        )
    comparison = compare_arrays(expected_numeric, actual_numeric, tolerances)
    np.savez_compressed(output / "candidate-arrays.npz", **actual, **raw)
    result = {
        "case": spec["name"],
        "precision": precision,
        "device": device,
        "execution": _native_execution_record(simulation),
        "reference": provenance,
        "capture_steps": list(captures),
        "dimensions": spec["size"],
        "paired_real": spec["complex"],
        "comparison": comparison,
        "unverified": unresolved,
        "excluded_parameter_keys": omitted,
        "status": (
            "failed"
            if not comparison["passed"]
            else "incomplete" if unresolved else "passed"
        ),
        "candidate_arrays_sha256": _sha(output / "candidate-arrays.npz"),
    }
    return result


def run_analytic_case(*, size, precision, paired_real, output):
    """Supplement native evidence with independent NumPy Maxwell/Yee equations."""
    import gmes
    from tests.test_torch_fdtd import _numpy_dielectric_step

    bloch = (0.07, 0.11, 0.13) if paired_real else None
    simulation = gmes.TorchSimulation(
        space=gmes.Cartesian(size, 2),
        geometry=[gmes.DefaultMedium(gmes.Dielectric(eps_inf=1.7, mu_inf=1.05))],
        bloch=bloch,
        runtime=gmes.TorchRuntimeConfig(
            device="cpu", precision=precision, cpu_threads=1
        ),
    )
    rng = np.random.default_rng(114)
    fields = {}
    for name, shape in simulation.plan.shapes.items():
        value = rng.normal(size=shape) * 1e-3
        if paired_real:
            value = value + 1j * rng.normal(size=shape) * 1e-3
        fields[name] = value
    simulation.load_host_fields(fields)
    expected, actual = {}, {}
    for step in range(101):
        if step in CAPTURES:
            for name, value in fields.items():
                key = f"step/{step}/field/{name}"
                expected[key] = value.copy()
                actual[key] = simulation.host_snapshot()[name]
            key = f"step/{step}/time"
            expected[key] = np.asarray(
                [step, step * simulation.plan.dt, simulation.plan.dt]
            )
            actual[key] = np.asarray(
                [
                    int(simulation.state.step_count),
                    float(simulation.state.source_time),
                    float(simulation.state.time_step),
                ]
            )
        if step < 100:
            fields, reference_dt = _numpy_dielectric_step(
                fields, resolution=2, bloch=bloch
            )
            if reference_dt != simulation.plan.dt:
                raise ValueError("independent equation and candidate dt differ")
            simulation.advance(1)
    validate_capture_contract(
        actual, CAPTURES, simulation.plan.shapes, precision=precision
    )
    manifest = native_oracle.load_manifest()
    dtype = (
        ("complex128" if precision == "float64" else "complex64")
        if paired_real
        else precision
    )
    tolerance = manifest["tolerances"]["torch"]["dielectric"][dtype]
    comparison = compare_arrays(expected, actual, dict.fromkeys(expected, tolerance))
    if output is not None:
        np.savez_compressed(output / "reference-arrays.npz", **expected)
        np.savez_compressed(output / "candidate-arrays.npz", **actual)
    return {
        "case": "dielectric-numpy-yee",
        "dimensions": list(size),
        "precision": precision,
        "paired_real": paired_real,
        "reference": {
            "kind": "independent-numpy-equations",
            "file": "tests/test_torch_fdtd.py",
            "sha256": _sha(ROOT / "tests/test_torch_fdtd.py"),
        },
        "capture_steps": list(CAPTURES),
        "comparison": comparison,
        "status": "passed" if comparison["passed"] else "failed",
        "native_qualification": False,
    }


def run_dm2_invariant(*, precision, output, final_step=500):
    """Check uncoupled Bloch rotation/relaxation against its closed-form solution.

    With gamma=0, the implicit midpoint rotation is exactly the Cayley rotation
    with angle 2*atan(omega*dt/2). Its transverse norm and transformed inversion
    are invariant; physical coherence/inversion relax with T2/T1. This does not
    qualify the coupled Maxwell--Bloch feedback or the published paper pulse.
    """
    import gmes

    if final_step not in (100, 500):
        raise ValueError("DM2 invariant final step must be 100 or 500")
    omega = np.asarray([0.7, 1.1])
    dt, t1, t2, equilibrium = 0.01, 2.5, 1.7, -0.7
    simulation = gmes.TorchSimulation(
        space=gmes.Cartesian((1, 1, 0), 2),
        geometry=[
            gmes.DefaultMedium(gmes.Dielectric(eps_inf=1.4)),
            gmes.Block(
                gmes.Dm2(
                    eps_inf=1.4,
                    omega=tuple(omega),
                    n_atom=(0.2, 0.4),
                    rho30=equilibrium,
                    gamma=0.0,
                    t1=t1,
                    t2=t2,
                    rtol=1e-12,
                ),
                center=(0, 0, 0),
                size=(2, 2, 2),
            ),
        ],
        dt=dt,
        runtime=gmes.TorchRuntimeConfig(
            device="cpu", precision=precision, cpu_threads=1
        ),
    )
    fields = {
        name: np.full(shape, (i + 1) * 0.001)
        for i, (name, shape) in enumerate(simulation.plan.shapes.items())
    }
    simulation.load_host_fields(fields)
    rng = np.random.default_rng(114)
    records = simulation.dm2_state_snapshot()
    if {record["component"] for record in records} != {"Ex", "Ey", "Ez"} or any(
        record["u"].size == 0 for record in records
    ):
        raise ValueError(
            "DM2 invariant requires nonempty state for all electric components"
        )
    initial = [rng.uniform(0.01, 0.05, size=record["u"].shape) for record in records]
    simulation.load_host_dm2_state(initial)
    captures = (*CAPTURES, 500) if final_step == 500 else CAPTURES
    expected, actual = {}, {}
    completed = 0
    for step in captures:
        simulation.advance(step - completed)
        completed = step
        snapshot = simulation.host_snapshot()
        for name, value in fields.items():
            key = f"step/{step}/field/{name}"
            expected[key], actual[key] = value, snapshot[name]
        key = f"step/{step}/time"
        expected[key] = np.asarray([step, step * dt, dt])
        actual[key] = np.asarray(
            [
                int(simulation.state.step_count),
                float(simulation.state.source_time),
                float(simulation.state.time_step),
            ]
        )
        angle = step * 2 * np.arctan(omega * dt / 2)
        for ordinal, (seed, record) in enumerate(
            zip(initial, simulation.dm2_state_snapshot(), strict=True)
        ):
            rotated = seed.copy()
            rotated[..., 0] = seed[..., 0] * np.cos(angle) + seed[..., 1] * np.sin(
                angle
            )
            rotated[..., 1] = seed[..., 1] * np.cos(angle) - seed[..., 0] * np.sin(
                angle
            )
            rho = rotated.copy()
            rho[..., :2] *= np.exp(-step * dt / t2)
            rho[..., 2] = rho[..., 2] * np.exp(-step * dt / t1) + equilibrium
            prefix = f"step/{step}/state/{record['component']}/{ordinal}-Dm2/"
            for suffix, reference_value, candidate_value in (
                ("u", rotated, record["u"]),
                ("rho", rho, record["rho"]),
                (
                    "transverse_norm",
                    np.sum(seed[..., :2] ** 2, axis=-1),
                    np.sum(record["u"][..., :2] ** 2, axis=-1),
                ),
            ):
                expected[prefix + suffix] = reference_value
                actual[prefix + suffix] = candidate_value
    validate_capture_contract(
        actual, captures, simulation.plan.shapes, precision=precision
    )
    tolerance = native_oracle.load_manifest()["tolerances"]["torch"]["dm2"][precision]
    comparison = compare_arrays(expected, actual, dict.fromkeys(expected, tolerance))
    if output is not None:
        np.savez_compressed(output / "reference-arrays.npz", **expected)
        np.savez_compressed(output / "candidate-arrays.npz", **actual)
    return {
        "case": "dm2-uncoupled-cayley-relaxation",
        "precision": precision,
        "reference": {
            "kind": "independent-closed-form",
            "file": "benchmarks/full_state_qualification.py",
            "sha256": _sha(__file__),
        },
        "capture_steps": list(captures),
        "comparison": comparison,
        "status": "passed" if comparison["passed"] else "failed",
        "native_qualification": False,
        "unverified": ["nonzero-gamma-coupled-feedback", "paper-pulse-propagation"],
    }


def _distributed_field_shapes(space):
    """Return global Yee shapes without constructing a second reference solver."""
    nx, ny, nz = (int(value) for value in space.whole_field_size)
    return {
        "Ex": (nx, ny + 1, nz + 1),
        "Ey": (nx + 1, ny, nz + 1),
        "Ez": (nx + 1, ny + 1, nz),
        "Hx": (nx, ny + 1, nz + 1),
        "Hy": (nx + 1, ny, nz + 1),
        "Hz": (nx + 1, ny + 1, nz),
    }


def _partition_geometry(gmes):
    """Return a Drude block that crosses the forced uneven x-partition."""
    return [
        gmes.DefaultMedium(gmes.Dielectric(eps_inf=1.7, mu_inf=1.05)),
        gmes.Block(
            gmes.Drude(
                eps_inf=1.2,
                sigma=0.01,
                dps=(gmes.DrudePole(omega=0.7, gamma=0.03),),
            ),
            center=(0, 0, 0),
            size=(3, 3, 3),
        ),
    ]


def _partition_sources(gmes):
    """Return point and TFSF sources owned on both sides of the forced cut."""
    waveform = gmes.DifferentiatedGaussian(0.7, 0.3)
    return (
        gmes.PointSource(waveform, center=(-2, 0, 0), component=gmes.Ex, amp=1),
        gmes.PointSource(waveform, center=(-1, 0, 0), component=gmes.Ex, amp=2),
        gmes.PointSource(waveform, center=(-1, 0, 0), component=gmes.Ey, amp=3),
        # Same Ex target: the final record must be the one surviving on rank 0.
        gmes.PointSource(waveform, center=(-1, 0, 0), component=gmes.Ex, amp=4),
        # This supported direct TFSF face region straddles the forced x cut.
        gmes.TotalFieldScatteredField(
            gmes.Continuous(0.2, phase=0.2, width=1),
            center=(0, 0, 0),
            size=(3, 3, 3),
            direction=(1, 0, 0),
            polarization=(0, 1, 0),
            amp=0.3,
        ),
    )


def _canonical_rows(parts):
    """Combine uniquely owned coordinate rows and reject overlap or shape drift."""
    grouped = {}
    for name, indices, values in parts:
        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values)
        if indices.ndim != 2 or indices.shape[1] != 3 or values.ndim != 2:
            raise ValueError("distributed state rows are malformed")
        if len(indices) != len(values):
            raise ValueError("distributed state row counts differ")
        grouped.setdefault(name, []).append((indices, values))
    arrays = {}
    for name, rows in grouped.items():
        indices = np.concatenate([item[0] for item in rows], axis=0)
        values = np.concatenate([item[1] for item in rows], axis=0)
        order = np.lexsort((indices[:, 2], indices[:, 1], indices[:, 0]))
        indices, values = indices[order], values[order]
        if len(indices) > 1 and np.any(np.all(np.diff(indices, axis=0) == 0, axis=1)):
            raise ValueError("distributed state has duplicate global ownership")
        arrays[f"{name}/indices"] = indices
        arrays[f"{name}/values"] = values
    return arrays


def _owned_material_rows(indices, component, *, rank, split_axis, local_shape):
    """Drop only the Yee halo rows that are not globally owned by this rank."""
    if rank is None:
        return np.ones(len(indices), dtype=bool)
    if split_axis not in (0, 1, 2) or rank not in (0, 1):
        raise ValueError("distributed material ownership is malformed")
    perpendicular = component[1].lower() != "xyz"[split_axis]
    if not perpendicular:
        return np.ones(len(indices), dtype=bool)
    coordinate = indices[:, split_axis]
    if component.startswith("E") and rank == 0:
        return coordinate < local_shape[split_axis] - 1
    if component.startswith("H") and rank == 1:
        return coordinate > 0
    return np.ones(len(indices), dtype=bool)


def _material_rows(simulation, *, offset=(0, 0, 0), rank=None, split_axis=None):
    """Canonicalize every owned live material-state row from candidate buffers."""
    arrays = {}
    records = torch_correctness._independent_material_records(simulation, 0, arrays)
    parts = []
    for record in records:
        state_key = record["state_key"]
        index_key = state_key.removesuffix("/values") + "/indices"
        indices = np.asarray(arrays[index_key], dtype=np.int64).copy()
        owned = _owned_material_rows(
            indices,
            record["component"],
            rank=rank,
            split_axis=split_axis,
            local_shape=simulation.plan.shapes[record["component"]],
        )
        indices = indices[owned]
        indices += np.asarray(offset, dtype=np.int64)
        width = int(record["state_width"])
        values = np.asarray(arrays[state_key]).reshape(len(owned), width)[owned]
        name = "/".join(
            (
                "material",
                record["component"],
                record["strategy"],
                record["native_type"],
            )
        )
        parts.append((name, indices, values))
    return _canonical_rows(parts)


def _source_rows(simulation, *, offset=(0, 0, 0)):
    """Canonicalize owned point and transparent source batch payloads."""
    parts = []
    for batch in simulation.sources.batches:
        native_type, representation, indices, values = (
            torch_correctness._source_batch_payload(simulation, batch)
        )
        if not len(indices):
            continue
        indices = np.asarray(indices, dtype=np.int64).copy()
        indices += np.asarray(offset, dtype=np.int64)
        values = np.asarray(values)
        if values.ndim != 1 or values.size % len(indices):
            raise ValueError("source batch payload cannot be partitioned by target")
        parts.append(
            (
                f"source/{batch.component}/{native_type}/{representation}",
                indices,
                values.reshape(len(indices), -1),
            )
        )
    return _canonical_rows(parts)


def _source_auxiliary_arrays(simulation):
    """Capture live transparent-source auxiliary state without an oracle archive."""
    arrays = {}
    records = torch_correctness._independent_source_records(simulation, 0, arrays)
    if not records["auxiliary"]:
        raise ValueError("partition-crossing source requires a transparent auxiliary")
    prefix = "step/0/"
    result = {
        key.removeprefix(prefix): value
        for key, value in arrays.items()
        if key.startswith(f"{prefix}source_aux/")
        or key.startswith(f"{prefix}source_aux_material/")
    }
    if not result:
        raise ValueError("transparent auxiliary state capture is empty")
    for ordinal, (record, auxiliary) in enumerate(
        zip(records["auxiliary"], simulation.sources.auxiliaries, strict=True)
    ):
        checkpoint_prefix = f"source_aux/{ordinal}-{record['source']}/checkpoint/state"
        for name, value in auxiliary.state.checkpoint().items():
            result[f"{checkpoint_prefix}/{name}"] = torch_correctness._host(value)
        clock = _source_clock(auxiliary)
        prefix = f"source_aux/{ordinal}-{record['source']}/live_clock"
        result[f"{prefix}/step_count"] = clock["source/step_count"]
        result[f"{prefix}/time"] = clock["source/time"]
    return result


def _transparent_source_ownership(rows):
    """Count owned direct TFSF/Gaussian face targets from live source rows."""
    return sum(
        len(value)
        for key, value in rows.items()
        if key.startswith("source/")
        and "/Transparent" in key
        and key.endswith("/indices")
    )


def _point_source_ownership(rows):
    """Count only direct point-source targets, never transparent face rows."""
    return sum(
        len(value)
        for key, value in rows.items()
        if key.startswith("source/")
        and "/PointSource" in key
        and key.endswith("/indices")
    )


def _source_clock(simulation):
    """Return live source scheduling state without using an expected archive."""
    state = simulation.state
    return {
        "source/step_count": np.asarray(
            [int(state.step_count.detach().cpu().item())], dtype=np.int64
        ),
        "source/time": np.asarray(
            [
                state.source_time.detach().cpu().item(),
                state.time_step.detach().cpu().item(),
            ],
            dtype=state.source_time.detach().cpu().numpy().dtype,
        ),
    }


def _canonical_distributed_rows(gathered_rows):
    """Merge exact owned rows from every rank and reject duplicate ownership."""
    return _canonical_rows(
        (
            (
                key.removesuffix("/indices"),
                value,
                gathered[key.replace("/indices", "/values")],
            )
            for gathered in gathered_rows
            for key, value in gathered.items()
            if key.endswith("/indices")
        )
    )


def _distributed_live_state(distributed, launch, dist):
    """Collect fields plus all rank-local persistent source/material state."""
    global_fields = distributed.global_field_snapshot()
    local_material = _material_rows(
        distributed.local,
        offset=distributed.decomposition.offset(launch.rank),
        rank=launch.rank,
        split_axis=distributed.decomposition.axis,
    )
    local_sources = _source_rows(
        distributed.local, offset=distributed.decomposition.offset(launch.rank)
    )
    local_auxiliary = _source_auxiliary_arrays(distributed.local)
    local_clock = _source_clock(distributed.local)
    gathered_material = [None, None]
    gathered_sources = [None, None]
    gathered_auxiliary = [None, None]
    gathered_clocks = [None, None]
    for gathered, local in (
        (gathered_material, local_material),
        (gathered_sources, local_sources),
        (gathered_auxiliary, local_auxiliary),
        (gathered_clocks, local_clock),
    ):
        dist.all_gather_object(gathered, local, group=distributed.group)
    if launch.rank != 0:
        return None
    return {
        "fields": global_fields,
        "material": _canonical_distributed_rows(gathered_material),
        "sources": _canonical_distributed_rows(gathered_sources),
        "auxiliaries": gathered_auxiliary,
        "clocks": gathered_clocks,
        "source_ownership": [
            sum(len(value) for key, value in rows.items() if key.endswith("/indices"))
            for rows in gathered_sources
        ],
        "point_source_ownership": [
            _point_source_ownership(rows) for rows in gathered_sources
        ],
        "transparent_source_ownership": [
            _transparent_source_ownership(rows) for rows in gathered_sources
        ],
    }


def _persistent_replay_arrays(state):
    """Flatten complete distributed state for exact post-checkpoint replay checks."""
    arrays = {f"field/{key}": value for key, value in state["fields"].items()}
    arrays.update(
        {f"material/{key}": value for key, value in state["material"].items()}
    )
    arrays.update({f"source/{key}": value for key, value in state["sources"].items()})
    for rank, auxiliary in enumerate(state["auxiliaries"]):
        arrays.update(
            {
                f"source_auxiliary/rank-{rank}/{key}": value
                for key, value in auxiliary.items()
            }
        )
    for rank, clock in enumerate(state["clocks"]):
        arrays.update(
            {f"source_clock/rank-{rank}/{key}": value for key, value in clock.items()}
        )
    return arrays


def _comparison_tolerances(arrays, tolerance):
    """Use exact topology and one frozen tolerance for each numeric state row."""
    return {
        key: tolerance
        for key, value in arrays.items()
        if np.asarray(value).dtype.kind not in "biu"
    }


def _live_clock_tolerances(arrays, tolerance):
    """Keep live scheduling clocks exact while retaining physical tolerances."""
    result = _comparison_tolerances(arrays, tolerance)
    for key, value in arrays.items():
        if "/live_clock/" in key or key.startswith("source_clock/"):
            if np.asarray(value).dtype.kind not in "biu":
                result[key] = {"rtol": 0.0, "atol": 0.0}
    return result


def _public_device_metadata(logical, properties):
    """Return report-safe device facts without a stable hardware identifier."""
    return {"logical": int(logical), "name": str(properties.name)}


def _two_gpu_report_passed(
    captures,
    source_comparison,
    source_auxiliary_comparisons,
    source_clock_comparisons,
    checkpoint_comparison,
    source_ownership,
    point_source_ownership,
    transparent_source_ownership,
    source_crossings,
):
    """Combine every required two-rank comparison without hiding a failure."""
    return (
        all(
            item["field_comparison"]["passed"]
            and item["material_comparison"]["passed"]
            and item["source_comparison"]["passed"]
            and all(
                comparison["comparison"]["passed"]
                for comparison in item["source_auxiliary_comparisons"]
            )
            and all(
                comparison["comparison"]["passed"]
                for comparison in item["source_clock_comparisons"]
            )
            for item in captures
        )
        and source_comparison["passed"]
        and all(item["comparison"]["passed"] for item in source_auxiliary_comparisons)
        and all(item["comparison"]["passed"] for item in source_clock_comparisons)
        and checkpoint_comparison["passed"]
        and all(source_ownership)
        and all(point_source_ownership)
        and all(transparent_source_ownership)
        and source_crossings > 0
    )


def _two_gpu_report_exit_code(report):
    """Translate the persisted root-rank report into a process exit status."""
    passed = report.get("passed") if isinstance(report, dict) else None
    if not isinstance(passed, bool):
        raise ValueError("two-GPU report has no boolean passed status")
    return 0 if passed else 1


def _two_gpu_execution_record(compile_policy):
    """Describe the selected local execution policy without production authority."""
    if compile_policy not in TWO_GPU_COMPILE_POLICIES:
        raise ValueError("two-GPU compile policy must be 'eager' or 'compile'")
    if compile_policy == "compile":
        return {
            "scope": TWO_GPU_COMPILED_EXECUTION_SCOPE,
            "compile_policy": compile_policy,
            "execution_mode": "graph",
        }
    return {
        "scope": TWO_GPU_EXECUTION_SCOPE,
        "compile_policy": compile_policy,
        "execution_mode": "eager",
    }


def _two_gpu_runtime_options(compile_policy, *, launch=None):
    """Keep the serial and distributed runtime policies identical."""
    _two_gpu_execution_record(compile_policy)
    options = {
        "precision": "float64",
        "compile_policy": compile_policy,
        "execution_policy": "auto",
        "cpu_threads": 1,
        "cpu_interop_threads": 1,
    }
    if launch is not None:
        options["launch"] = launch
    return options


def _capture_two_gpu_compute_regions(distributed, serial, *, rank, compile_policy):
    """Capture both post-load compute regions only for the explicit graph policy."""
    if compile_policy == "eager":
        return
    _two_gpu_execution_record(compile_policy)
    distributed.capture_cuda_graphs()
    if rank == 0:
        if serial is None:
            raise ValueError("rank zero serial runtime is required for graph capture")
        serial.capture_cuda_graphs()


def run_two_gpu_partition_case(output, *, compile_policy="eager"):
    """Run one serial-versus-two-GPU full field/state partition case.

    This is a distributed Torch correctness check, not native qualification and
    not production qualification. ``compile_policy='compile'`` explicitly
    captures the same post-load compute regions for both runtimes. It must be
    started under exactly two torchrun ranks and writes private evidence only
    on rank zero.
    """
    execution = _two_gpu_execution_record(compile_policy)
    import torch
    import torch.distributed as dist

    import gmes

    launch = gmes.distributed_launch_from_environment()
    if (launch.world_size, launch.local_world_size) != (2, 2):
        raise ValueError("two-GPU qualification requires exactly two local ranks")
    output = Path(output)
    if launch.rank == 0:
        if output.exists() or output.is_symlink() or output.is_relative_to(ROOT):
            raise ValueError("two-GPU output must be a new directory outside checkout")
        output.mkdir(mode=0o700, parents=False)
    # ``TorchDistributedSimulation`` creates the NCCL group.  Do not call a
    # collective while rank zero is preparing the private evidence directory.
    space = gmes.Cartesian((5, 4, 4), 1)
    geometry = _partition_geometry(gmes)
    sources = _partition_sources(gmes)
    fields = native_oracle.initial_field_values(
        _distributed_field_shapes(space), 115, 1e-3, complex_fields=False
    )
    runtime = gmes.TorchRuntimeConfig(
        device=f"cuda:{launch.local_rank}",
        **_two_gpu_runtime_options(compile_policy, launch=launch),
    )
    distributed = gmes.TorchDistributedSimulation(
        space=space,
        geometry=geometry,
        sources=sources,
        runtime=runtime,
        dt=0.025,
        split_axis=0,
        cut=2,
    ).load_host_fields(fields)
    if launch.rank == 0:
        serial = gmes.TorchSimulation(
            space=space,
            geometry=_partition_geometry(gmes),
            sources=_partition_sources(gmes),
            runtime=gmes.TorchRuntimeConfig(
                device="cuda:0",
                **_two_gpu_runtime_options(compile_policy),
            ),
            dt=0.025,
        ).load_host_fields(fields)
    _capture_two_gpu_compute_regions(
        distributed,
        serial if launch.rank == 0 else None,
        rank=launch.rank,
        compile_policy=compile_policy,
    )
    distributed.advance(2)
    if launch.rank == 0:
        serial.advance(2)
    captures = (0, *CAPTURES[1:])
    completed = 0
    raw = {}
    capture_results = []
    tolerance = native_oracle.load_manifest()["tolerances"]["torch"]["drude"]["float64"]
    source_ownership = None
    point_source_ownership = None
    transparent_source_ownership = None
    for relative_step in captures:
        distributed.advance(relative_step - completed)
        if launch.rank == 0:
            serial.advance(relative_step - completed)
        completed = relative_step
        distributed_state = _distributed_live_state(distributed, launch, dist)
        if launch.rank != 0:
            continue
        serial_fields = serial.host_snapshot()
        serial_material = _material_rows(serial)
        serial_sources = _source_rows(serial)
        serial_source_clock = _source_clock(serial)
        serial_source_auxiliary = _source_auxiliary_arrays(serial)
        field_comparison = compare_arrays(
            serial_fields,
            distributed_state["fields"],
            dict.fromkeys(serial_fields, tolerance),
        )
        material_comparison = compare_arrays(
            serial_material,
            distributed_state["material"],
            _comparison_tolerances(serial_material, tolerance),
        )
        source_comparison = compare_arrays(
            serial_sources,
            distributed_state["sources"],
            dict.fromkeys(serial_sources, {"rtol": 0.0, "atol": 0.0}),
        )
        source_auxiliary_comparisons = [
            {
                "rank": rank,
                "comparison": compare_arrays(
                    serial_source_auxiliary,
                    remote_auxiliary,
                    _live_clock_tolerances(serial_source_auxiliary, tolerance),
                ),
            }
            for rank, remote_auxiliary in enumerate(distributed_state["auxiliaries"])
        ]
        source_clock_comparisons = [
            {
                "rank": rank,
                "comparison": compare_arrays(
                    serial_source_clock,
                    remote_clock,
                    {"source/time": {"rtol": 0.0, "atol": 0.0}},
                ),
            }
            for rank, remote_clock in enumerate(distributed_state["clocks"])
        ]
        if relative_step == captures[0]:
            source_ownership = distributed_state["source_ownership"]
            point_source_ownership = distributed_state["point_source_ownership"]
            transparent_source_ownership = distributed_state[
                "transparent_source_ownership"
            ]
        for name in COMPONENTS:
            raw[f"capture/{relative_step}/serial/{name}"] = serial_fields[name]
            raw[f"capture/{relative_step}/distributed/{name}"] = distributed_state[
                "fields"
            ][name]
        for key, value in serial_material.items():
            raw[f"capture/{relative_step}/serial/{key}"] = value
        for key, value in distributed_state["material"].items():
            raw[f"capture/{relative_step}/distributed/{key}"] = value
        for prefix, arrays in (
            ("serial/source", serial_sources),
            ("distributed/source", distributed_state["sources"]),
            ("serial/source_clock", serial_source_clock),
            ("serial/source_auxiliary", serial_source_auxiliary),
        ):
            for key, value in arrays.items():
                raw[f"capture/{relative_step}/{prefix}/{key}"] = value
        for rank, values in enumerate(distributed_state["clocks"]):
            for key, value in values.items():
                raw[f"capture/{relative_step}/distributed/rank-{rank}/clock/{key}"] = (
                    value
                )
        for rank, values in enumerate(distributed_state["auxiliaries"]):
            for key, value in values.items():
                raw[
                    f"capture/{relative_step}/distributed/rank-{rank}/auxiliary/{key}"
                ] = value
        capture_results.append(
            {
                "relative_step": relative_step,
                "field_comparison": field_comparison,
                "material_comparison": material_comparison,
                "source_comparison": source_comparison,
                "source_auxiliary_comparisons": source_auxiliary_comparisons,
                "source_clock_comparisons": source_clock_comparisons,
            }
        )
    checkpoint = distributed.checkpoint()
    distributed.advance(5)
    checkpoint_expected = _distributed_live_state(distributed, launch, dist)
    distributed.load_checkpoint(checkpoint).advance(5)
    checkpoint_replay = _distributed_live_state(distributed, launch, dist)
    devices = [None, None]
    local_device = torch.cuda.get_device_properties(launch.local_rank)
    dist.all_gather_object(
        devices,
        _public_device_metadata(launch.local_rank, local_device),
        group=distributed.group,
    )
    if launch.rank == 0:
        checkpoint_expected_arrays = _persistent_replay_arrays(checkpoint_expected)
        checkpoint_replay_arrays = _persistent_replay_arrays(checkpoint_replay)
        replay_comparison = compare_arrays(
            checkpoint_expected_arrays,
            checkpoint_replay_arrays,
            _live_clock_tolerances(checkpoint_expected_arrays, tolerance),
        )
        for label, arrays in (
            ("expected", checkpoint_expected_arrays),
            ("replay", checkpoint_replay_arrays),
        ):
            for key, value in arrays.items():
                raw[f"checkpoint/{label}/{key}"] = value
        source_comparison = capture_results[-1]["source_comparison"]
        source_auxiliary_comparisons = capture_results[-1][
            "source_auxiliary_comparisons"
        ]
        source_clock_comparisons = capture_results[-1]["source_clock_comparisons"]
        raw_path = output / "raw-arrays.npz"
        np.savez_compressed(raw_path, **raw)
        report = {
            "kind": KIND,
            **execution,
            "native_qualification": False,
            "execution_policy": "auto",
            "case": {
                "space": [5, 4, 4],
                "resolution": 1,
                "split_axis": 0,
                "cut": 2,
                "precondition_steps": 2,
                "capture_steps": list(captures),
                "drude_block_crosses_partition": True,
                "ordered_point_sources": 4,
                "point_source_ownership": point_source_ownership,
                "source_ownership": source_ownership,
                "source_crossings": distributed.decomposition.source_crossings,
                "transparent_source_ownership": transparent_source_ownership,
                "source_coverage": (
                    "point-source ownership and same-target last-wins, plus one direct "
                    "TFSF face region with owned targets on both sides of the forced cut"
                ),
            },
            "devices": sorted(devices, key=lambda value: value["logical"]),
            "captures": capture_results,
            "source_comparison": source_comparison,
            "source_auxiliary_comparisons": source_auxiliary_comparisons,
            "source_clock_comparisons": source_clock_comparisons,
            "source_ownership_passed": all(source_ownership),
            "point_source_ownership_passed": all(point_source_ownership),
            "transparent_source_ownership_passed": all(transparent_source_ownership),
            "checkpoint_comparison": replay_comparison,
            "candidate": candidate_provenance(),
            "raw_arrays": {
                "name": raw_path.name,
                "sha256": _sha(raw_path),
                "size_bytes": raw_path.stat().st_size,
            },
        }
        report["passed"] = _two_gpu_report_passed(
            capture_results,
            source_comparison,
            source_auxiliary_comparisons,
            source_clock_comparisons,
            replay_comparison,
            source_ownership,
            point_source_ownership,
            transparent_source_ownership,
            distributed.decomposition.source_crossings,
        )
        (output / "result.json").write_text(_canonical_json(report))
        print(json.dumps({"scope": report["scope"], "passed": report["passed"]}))
        exit_code = _two_gpu_report_exit_code(report)
    else:
        exit_code = 1
    distributed_exit_code = torch.tensor(
        exit_code, device=distributed.device, dtype=torch.int32
    )
    dist.broadcast(distributed_exit_code, src=0, group=distributed.group)
    dist.barrier(group=distributed.group)
    gmes.TorchDistributedSimulation.close()
    return int(distributed_exit_code.cpu())


def main():
    """Execute a bounded explicit selection and write portable local evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "native",
            "analytic",
            "dm2-invariant",
            "prepare-native-manifest",
            "two-gpu",
        ),
        required=True,
    )
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument(
        "--precision", choices=("float64", "float32"), default="float64"
    )
    parser.add_argument(
        "--compile-policy", choices=TWO_GPU_COMPILE_POLICIES, default="eager"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--legacy-manifest", type=Path)
    parser.add_argument("--historical-artifact-root", type=Path)
    parser.add_argument("--compatibility-manifest", type=Path)
    args = parser.parse_args()
    if args.mode == "two-gpu":
        return run_two_gpu_partition_case(
            args.output_dir, compile_policy=args.compile_policy
        )
    if args.compile_policy != "eager":
        parser.error("--compile-policy is supported only with --mode two-gpu")
    if args.output_dir.exists():
        parser.error("output directory must be new")
    current_manifest = torch_correctness._load_trusted_manifest()[0]
    if args.mode == "native" and (not args.reference_dir or not args.cases):
        parser.error("native mode requires --reference-dir and --cases")
    requires_compatibility = args.mode in {"native", "prepare-native-manifest"}
    if requires_compatibility and not all(
        (
            args.legacy_manifest,
            args.historical_artifact_root,
            args.compatibility_manifest,
        )
    ):
        parser.error(
            "native compatibility requires --legacy-manifest, "
            "--historical-artifact-root, and --compatibility-manifest"
        )
    compatibility_evidence = None
    if args.mode == "native":
        manifest, compatibility_evidence = load_v6_capture_compatibility_manifest(
            legacy_manifest=args.legacy_manifest,
            historical_artifact_root=args.historical_artifact_root,
            compatibility_manifest=args.compatibility_manifest,
        )
    else:
        manifest = current_manifest
    cases = args.cases or []
    for name in cases:
        if not re.fullmatch(r"[a-z0-9-]+", name):
            parser.error("invalid case name")
        native_oracle.find_case(manifest, name)
    args.output_dir.mkdir(mode=0o700, parents=False)
    before = candidate_provenance()
    results = []
    started = time.monotonic()
    if args.mode == "prepare-native-manifest":
        compatibility_evidence = build_v6_capture_compatibility_manifest(
            legacy_manifest=args.legacy_manifest,
            historical_artifact_root=args.historical_artifact_root,
            output=args.compatibility_manifest,
        )
        results = [
            {
                "case": "native-capture-manifest",
                "status": "prepared",
                "native_qualification": False,
            }
        ]
        selections = []
    elif args.mode == "analytic":
        selections = [
            (size, paired)
            for size in ((2, 0, 0), (2, 2, 0), (2, 2, 2))
            for paired in (False, True)
        ]
    elif args.mode == "dm2-invariant":
        selections = ["dm2-uncoupled-cayley-relaxation"]
    else:
        selections = cases
    for ordinal, selection in enumerate(selections):
        output = args.output_dir / str(ordinal)
        output.mkdir(mode=0o700)
        try:
            if args.mode == "analytic":
                size, paired = selection
                result = run_analytic_case(
                    size=size,
                    precision=args.precision,
                    paired_real=paired,
                    output=output,
                )
            elif args.mode == "dm2-invariant":
                result = run_dm2_invariant(precision=args.precision, output=output)
            else:
                reference = args.reference_dir / f"{selection}.npz"
                if not reference.is_file():
                    result = {
                        "case": selection,
                        "status": "unavailable",
                        "reason": "missing-full-native-archive",
                    }
                else:
                    result = run_native_case(
                        reference, manifest, precision=args.precision, output=output
                    )
        except (ValueError, RuntimeError, KeyError, OSError) as error:
            # Exception text may contain private paths. Preserve its type here;
            # callers can debug locally without publishing machine identifiers.
            result = {
                "case": str(selection),
                "status": "error",
                "error_type": type(error).__name__,
            }
        results.append(result)
        print(
            json.dumps({"case": result["case"], "status": result["status"]}), flush=True
        )
    after = candidate_provenance()
    stable = before == after
    report = {
        "kind": KIND,
        "candidate": before,
        "candidate_unchanged": stable,
        "parent_manifest_sha256": _sha(native_oracle.DEFAULT_MANIFEST),
        "capture_compatibility": compatibility_evidence,
        "results": results,
        "elapsed_seconds": time.monotonic() - started,
        "remaining": [
            (
                "full-native-matrix"
                if args.mode == "analytic"
                else "unselected-native-cases"
            ),
            "longer-physical-matrix",
            "partition-crossing-gpu-window",
        ],
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    successful = {"passed", "prepared"}
    return 0 if stable and all(r["status"] in successful for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
