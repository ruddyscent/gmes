#!/usr/bin/env python3
"""Fail-closed wrapper-level CUDA Inductor output-candidate evidence.

This module is deliberately narrow.  It captures actual Torch 2.13 CUDA
Inductor wrapper text in an isolated child process, hashes it, and structurally
audits known allocation and Triton launch forms. Exact field-layout outputs are
diagnostic candidates, not clone findings on their own. It does not claim
anything about PTX, opaque external kernels, CUDA Graph workspaces, arbitrary
shapes, or allocator retention. Those conditions remain unverified rather than
passing.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

import torch

import gmes
from benchmarks.torch_tuning import (
    MANIFEST,
    _build_case,
    _initialize_fields,
    _synchronize,
    load_manifest,
)

SCHEMA = "gmes.torch.lowered-materialization.v1"
SCOPE = "pinned-torch-2.13-cuda-wrapper-output-allocation-v1"
RUNTIME_FILES = ("torch_fdtd.py", "torch_plan.py", "torch_source.py")
FIELD_NAMES = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
PINNED_CUDA_TOPOLOGY = (
    "local-two-static-half-step-regions+external-cached-two-stage-foreach-"
    "boundary-sync-v2"
)
COMPILED_REGIONS = ("electric_half", "magnetic_half")
KNOWN_ALLOCATORS = {
    "empty_strided_cuda",
    "torch.empty_strided",
    "alloc_from_pool",
}
KNOWN_ALIAS_FACTORIES = {"reinterpret_tensor", "torch.as_strided", "as_strided"}
COPY_METHODS = {"clone", "copy_", "contiguous", "_to_copy"}
DIRECT_ARGUMENT = re.compile(r"arg[0-9]+(?:_[0-9]+)?$")


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _hash_pairs(values: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(value.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return None if parent is None else f"{parent}.{node.attr}"
    return None


def _shape_literal(node: ast.AST | None) -> tuple[int, ...] | None:
    if node is None:
        return None
    try:
        value = ast.literal_eval(node)
    except TypeError, ValueError:
        return None
    if not isinstance(value, (tuple, list)) or not value:
        return None
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in value
    ):
        return None
    return tuple(value)


def _numel(shape: tuple[int, ...] | None) -> int | None:
    if shape is None:
        return None
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _field_shapes(contract: Mapping[str, object]) -> tuple[tuple[int, ...], ...]:
    raw = contract.get("field_buffers")
    if not isinstance(raw, list) or len(raw) != len(FIELD_NAMES):
        raise ValueError("field-buffer contract is incomplete")
    shapes = []
    names = set()
    for record in raw:
        if not isinstance(record, dict):
            raise ValueError("field-buffer contract record is malformed")
        name, shape = record.get("name"), record.get("shape")
        dtype, stride = record.get("dtype"), record.get("stride")
        if name not in FIELD_NAMES or name in names:
            raise ValueError("field-buffer contract names differ")
        if (
            not isinstance(shape, list)
            or not shape
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 1
                for item in shape
            )
        ):
            raise ValueError("field-buffer contract shape is malformed")
        if dtype != f"torch.{contract.get('precision')}" or not isinstance(
            stride, list
        ):
            raise ValueError("field-buffer contract layout is malformed")
        if len(stride) != len(shape) or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 1
            for item in stride
        ):
            raise ValueError("field-buffer contract layout is malformed")
        names.add(name)
        shapes.append(tuple(shape))
    if names != set(FIELD_NAMES):
        raise ValueError("field-buffer contract names differ")
    return tuple(shapes)


def _field_layouts(
    contract: Mapping[str, object],
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Return the validated field shape/stride layouts from a contract."""
    raw = contract.get("field_buffers")
    if not isinstance(raw, list):
        raise ValueError("field-buffer contract is incomplete")
    layouts = []
    for record in raw:
        if not isinstance(record, dict):
            raise ValueError("field-buffer contract record is malformed")
        shape = record.get("shape")
        stride = record.get("stride")
        if not isinstance(shape, list) or not isinstance(stride, list):
            raise ValueError("field-buffer contract layout is malformed")
        layouts.append((tuple(shape), tuple(stride)))
    return tuple(layouts)


def _validate_contract(contract: Mapping[str, object]) -> list[str]:
    required = {
        "case",
        "case_descriptor_sha256",
        "compile_cache_key",
        "compiled_region_topology",
        "device",
        "field_buffers",
        "plan_identity",
        "precision",
        "required_regions",
        "runtime_source_files_sha256",
        "runtime_source_sha256",
        "source_buffer_sha256",
        "torch_version",
    }
    missing = sorted(required - set(contract))
    reasons = [f"contract-missing:{name}" for name in missing]
    if not isinstance(contract.get("case"), str) or not contract["case"]:
        reasons.append("contract-case")
    if contract.get("compiled_region_topology") != PINNED_CUDA_TOPOLOGY:
        reasons.append("contract-compiled-region-topology")
    if contract.get("precision") not in {"float32", "float64"}:
        reasons.append("contract-precision")
    device = contract.get("device")
    if not isinstance(device, str) or not device.startswith("cuda"):
        reasons.append("contract-device-is-not-cuda")
    version = contract.get("torch_version")
    if (
        not isinstance(version, str)
        or not version.startswith("2.13.")
        or "+cu" not in version
    ):
        reasons.append("contract-torch-build-is-not-pinned-cuda-2.13")
    try:
        _field_shapes(contract)
    except ValueError:
        reasons.append("contract-field-buffers")
    regions = contract.get("required_regions")
    if (
        not isinstance(regions, list)
        or not regions
        or any(not isinstance(item, str) or not item for item in regions)
        or len(set(regions)) != len(regions)
    ):
        reasons.append("contract-required-regions")
    source_hashes = contract.get("runtime_source_files_sha256")
    expected_source_names = {f"gmes/{name}" for name in RUNTIME_FILES}
    if (
        not isinstance(source_hashes, dict)
        or set(source_hashes) != expected_source_names
    ):
        reasons.append("contract-runtime-source-files")
    elif any(not _is_sha256(value) for value in source_hashes.values()):
        reasons.append("contract-runtime-source-files")
    elif contract.get("runtime_source_sha256") != _hash_pairs(source_hashes):
        reasons.append("contract-runtime-source-digest")
    for key in (
        "case_descriptor_sha256",
        "compile_cache_key",
        "plan_identity",
        "runtime_source_sha256",
        "source_buffer_sha256",
    ):
        if not _is_sha256(contract.get(key)):
            reasons.append(f"contract-{key}")
    return reasons


def _kernel_definitions(
    tree: ast.AST, reasons: list[str]
) -> dict[str, tuple[str, ...]]:
    kernels = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and _call_name(node.value.func) == "async_compile.triton"
        ):
            continue
        if len(node.value.args) < 2 or not isinstance(node.value.args[1], ast.Constant):
            reasons.append("triton-definition-is-not-literal")
            continue
        source = node.value.args[1].value
        if not isinstance(source, str):
            reasons.append("triton-definition-is-not-text")
            continue
        try:
            inner = ast.parse(source)
        except SyntaxError:
            reasons.append("triton-definition-is-unparseable")
            continue
        functions = [
            item for item in ast.walk(inner) if isinstance(item, ast.FunctionDef)
        ]
        if len(functions) != 1:
            reasons.append("triton-definition-function-count")
            continue
        kernels[node.targets[0].id] = tuple(
            argument.arg for argument in functions[0].args.args
        )
    return kernels


def _argument_shapes(tree: ast.AST) -> dict[str, tuple[int, ...]]:
    result = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and _call_name(node.func) == "assert_size_stride"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
        ):
            continue
        shape = _shape_literal(node.args[1])
        if shape is not None:
            result[node.args[0].id] = shape
    return result


def _allocation_from_call(call: ast.Call) -> dict[str, object] | None:
    name = _call_name(call.func)
    if name not in KNOWN_ALLOCATORS:
        return None
    shape = _shape_literal(call.args[0] if call.args else None)
    stride = _shape_literal(call.args[1] if len(call.args) >= 2 else None)
    dtype_node = call.args[2] if len(call.args) >= 3 else None
    if dtype_node is None:
        dtype_node = next(
            (keyword.value for keyword in call.keywords if keyword.arg == "dtype"),
            None,
        )
    return {
        "dtype": _call_name(dtype_node) if dtype_node is not None else None,
        "factory": name,
        "shape": shape,
        "stride": stride,
        "reused_storage": name == "alloc_from_pool",
    }


def _is_unknown_allocator(call: ast.Call) -> bool:
    name = _call_name(call.func)
    if name is None:
        return False
    terminal = name.rsplit(".", maxsplit=1)[-1]
    return (
        "alloc" in terminal or terminal.startswith("empty") or terminal == "new_empty"
    )


def _is_field_output_candidate(
    shape: tuple[int, ...] | None,
    stride: tuple[int, ...] | None,
    dtype: object,
    field_layouts: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
    field_dtype: str,
) -> bool | None:
    if shape is None or stride is None:
        return None
    if dtype != field_dtype:
        return False
    return (shape, stride) in field_layouts or (
        len(shape) == 1
        and stride == (1,)
        and shape[0] in {_numel(item[0]) for item in field_layouts}
    )


def _pointer_role(name: str) -> str | None:
    if name.startswith("in_out_ptr"):
        return "in-out"
    if name.startswith("out_ptr"):
        return "out"
    if name.startswith("in_ptr"):
        return "in"
    if "ptr" in name:
        return "unknown"
    return None


def _resolved_value(
    node: ast.AST,
    *,
    allocations: Mapping[str, Mapping[str, object]],
    aliases: Mapping[str, ast.AST],
    argument_shapes: Mapping[str, tuple[int, ...]],
    depth: int = 0,
) -> dict[str, object]:
    if depth > 16:
        return {"kind": "unknown"}
    if isinstance(node, ast.Name):
        if node.id in allocations:
            return {"kind": "allocation", **allocations[node.id]}
        if node.id in aliases:
            return _resolved_value(
                aliases[node.id],
                allocations=allocations,
                aliases=aliases,
                argument_shapes=argument_shapes,
                depth=depth + 1,
            )
        if DIRECT_ARGUMENT.fullmatch(node.id):
            return {
                "kind": "direct-input",
                "shape": argument_shapes.get(node.id),
                "name": node.id,
            }
    if isinstance(node, ast.Call) and _call_name(node.func) in KNOWN_ALIAS_FACTORIES:
        if node.args:
            return _resolved_value(
                node.args[0],
                allocations=allocations,
                aliases=aliases,
                argument_shapes=argument_shapes,
                depth=depth + 1,
            )
    return {"kind": "unknown"}


def _audit_wrapper(
    *,
    region: str | None,
    source: str,
    field_layouts: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
    field_dtype: str,
) -> dict[str, object]:
    raw = source.encode()
    result = {
        "region": region,
        "sha256": _sha256_bytes(raw),
        "size_bytes": len(raw),
        "launches": [],
        "full_domain_output_candidates": [],
        "reasons": [],
    }
    reasons = result["reasons"]
    candidates = result["full_domain_output_candidates"]
    assert isinstance(reasons, list) and isinstance(candidates, list)
    if not isinstance(region, str) or not region:
        reasons.append("missing-region-label")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        reasons.append("wrapper-is-unparseable")
        return result
    kernels = _kernel_definitions(tree, reasons)
    argument_shapes = _argument_shapes(tree)
    allocations: dict[str, Mapping[str, object]] = {}
    aliases: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            continue
        target = node.targets[0].id
        if isinstance(node.value, ast.Call):
            allocation = _allocation_from_call(node.value)
            if allocation is not None:
                allocations[target] = allocation
                if allocation["shape"] is None:
                    reasons.append(f"allocation-shape-unknown:{target}")
                continue
            if _is_unknown_allocator(node.value):
                reasons.append(f"allocation-factory-unknown:{target}")
                continue
        if isinstance(node.value, (ast.Name, ast.Call)):
            aliases[target] = node.value
    launch_count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function_name = _call_name(node.func)
        if function_name is not None and function_name.startswith("extern_kernels."):
            reasons.append("opaque-external-kernel")
            continue
        if function_name in {"async_compile.cpp", "async_compile.cpp_pybinding"}:
            reasons.append("opaque-cpp-kernel")
            continue
        if not (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "run"
            and isinstance(node.func.value, ast.Name)
        ):
            continue
        kernel_name = node.func.value.id
        parameters = kernels.get(kernel_name)
        if parameters is None:
            reasons.append(f"launch-kernel-unknown:{kernel_name}")
            continue
        launch_count += 1
        launch = {"kernel": kernel_name, "outputs": []}
        outputs = launch["outputs"]
        assert isinstance(outputs, list)
        for index, parameter in enumerate(parameters):
            role = _pointer_role(parameter)
            if role is None or role == "in":
                continue
            if role == "unknown" or index >= len(node.args):
                reasons.append(f"launch-output-unmapped:{kernel_name}:{parameter}")
                continue
            resolved = _resolved_value(
                node.args[index],
                allocations=allocations,
                aliases=aliases,
                argument_shapes=argument_shapes,
            )
            outputs.append({"parameter": parameter, "role": role, **resolved})
            if resolved["kind"] == "allocation":
                candidate = _is_field_output_candidate(
                    resolved.get("shape"),
                    resolved.get("stride"),
                    resolved.get("dtype"),
                    field_layouts,
                    field_dtype,
                )
                if candidate is None:
                    reasons.append(
                        f"allocation-shape-unknown:{kernel_name}:{parameter}"
                    )
                elif candidate:
                    candidates.append(
                        {
                            "classification": "exact-field-layout-output-candidate",
                            "kernel": kernel_name,
                            "parameter": parameter,
                            "factory": resolved["factory"],
                            "reused_storage": resolved["reused_storage"],
                            "shape": list(resolved["shape"]),
                            "stride": list(resolved["stride"]),
                        }
                    )
            elif resolved["kind"] != "direct-input":
                reasons.append(f"launch-output-unmapped:{kernel_name}:{parameter}")
        result["launches"].append(launch)
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in COPY_METHODS
        ):
            continue
        values = (node.func.value, *node.args)
        resolved = [
            _resolved_value(
                value,
                allocations=allocations,
                aliases=aliases,
                argument_shapes=argument_shapes,
            )
            for value in values
        ]
        if any(item["kind"] == "unknown" for item in resolved):
            reasons.append(f"copy-call-unmapped:{node.func.attr}")
            continue
    if launch_count == 0:
        reasons.append("wrapper-has-no-recognized-launch")
    return result


def audit_compiled_wrapper_sources(
    *, sources: Sequence[Mapping[str, object]], contract: Mapping[str, object]
) -> dict[str, object]:
    """Audit hash-bound wrappers; candidates are diagnostic and unknowns fail closed."""
    reasons = _validate_contract(contract)
    try:
        field_layouts = _field_layouts(contract)
    except ValueError:
        field_layouts = ()
    field_dtype = f"torch.{contract.get('precision')}"
    wrappers = []
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        reasons.append("source-records-malformed")
        sources = ()
    for record in sources:
        if not isinstance(record, Mapping):
            reasons.append("source-record-malformed")
            continue
        source, region = record.get("source"), record.get("region")
        if not isinstance(source, str):
            reasons.append("source-record-text-missing")
            continue
        wrapper = _audit_wrapper(
            region=region,
            source=source,
            field_layouts=field_layouts,
            field_dtype=field_dtype,
        )
        wrappers.append(wrapper)
        reasons.extend(wrapper["reasons"])
    required = contract.get("required_regions", [])
    covered = sorted(
        {item["region"] for item in wrappers if isinstance(item.get("region"), str)}
    )
    missing = sorted(set(required) - set(covered)) if isinstance(required, list) else []
    if not wrappers:
        reasons.append("no-emitted-wrapper-sources")
    reasons.extend(f"missing-region-coverage:{name}" for name in missing)
    candidates = [
        event
        for wrapper in wrappers
        for event in wrapper["full_domain_output_candidates"]
    ]
    unique_reasons = sorted(set(reasons))
    status = "unverified" if unique_reasons else "verified"
    return {
        "schema": SCHEMA,
        "scope": SCOPE,
        "contract": dict(contract),
        "wrappers": wrappers,
        "region_coverage": {
            "required": required,
            "covered": covered,
            "missing": missing,
        },
        "full_domain_output_candidates": candidates,
        "reasons": unique_reasons,
        "status": status,
        "verified": status == "verified",
    }


def _regular_child(root: Path, relative: str) -> Path:
    candidate = PurePosixPath(relative)
    if (
        candidate.is_absolute()
        or len(candidate.parts) != 2
        or candidate.parts[0] != "generated"
    ):
        raise ValueError("wrapper descriptor path is unsafe")
    output = root / candidate
    if output.is_symlink() or not output.is_file() or not output.is_relative_to(root):
        raise ValueError("wrapper descriptor is not a regular child")
    return output


def _write_bundle(
    output_directory: Path, audit: Mapping[str, object]
) -> dict[str, object]:
    if output_directory.exists() or output_directory.is_symlink():
        raise ValueError("output directory must be new")
    output_directory.mkdir(mode=0o700)
    generated = output_directory / "generated"
    generated.mkdir(mode=0o700)
    descriptors = []
    for wrapper in audit["wrappers"]:
        raw = wrapper.pop("_source_bytes")
        assert isinstance(raw, bytes)
        digest = _sha256_bytes(raw)
        filename = f"{digest}.py"
        target = generated / filename
        if not target.exists():
            target.write_bytes(raw)
        descriptors.append(
            {
                "region": wrapper["region"],
                "path": f"generated/{filename}",
                "sha256": digest,
                "size_bytes": len(raw),
            }
        )
    manifest = {
        "schema": SCHEMA,
        "scope": SCOPE,
        "contract": audit["contract"],
        "wrappers": descriptors,
    }
    (output_directory / "lowering-manifest.json").write_text(
        _canonical_json(manifest) + "\n"
    )
    return manifest


def verify_compiled_wrapper_audit(
    *, output_directory: Path, expected_contract: Mapping[str, object]
) -> dict[str, object]:
    """Re-hash and re-audit an emitted bundle; never trust its stored status."""
    root = Path(output_directory).resolve(strict=True)
    manifest_path = root / "lowering-manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("lowering manifest is not a regular file")
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except json.JSONDecodeError as error:
        raise ValueError("lowering manifest is invalid JSON") from error
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ValueError("lowering manifest schema differs")
    contract = manifest.get("contract")
    if contract != dict(expected_contract):
        raise ValueError("lowering manifest contract differs")
    descriptors = manifest.get("wrappers")
    if not isinstance(descriptors, list):
        raise ValueError("lowering manifest wrappers differ")
    sources = []
    for descriptor in descriptors:
        if not isinstance(descriptor, dict):
            raise ValueError("wrapper descriptor is malformed")
        relative, digest, size = (
            descriptor.get("path"),
            descriptor.get("sha256"),
            descriptor.get("size_bytes"),
        )
        if (
            not isinstance(relative, str)
            or not isinstance(digest, str)
            or not isinstance(size, int)
        ):
            raise ValueError("wrapper descriptor types differ")
        if not _is_sha256(digest) or size < 0:
            raise ValueError("wrapper descriptor digest or size differs")
        path = _regular_child(root, relative)
        raw = path.read_bytes()
        if len(raw) != size or _sha256_bytes(raw) != digest:
            raise ValueError("wrapper descriptor digest or size differs")
        try:
            source = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("wrapper source is not UTF-8") from error
        sources.append({"region": descriptor.get("region"), "source": source})
    return audit_compiled_wrapper_sources(sources=sources, contract=expected_contract)


def _runtime_source_contract() -> dict[str, object]:
    root = Path(gmes.__file__).resolve().parent
    hashes = {}
    for name in RUNTIME_FILES:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("runtime source is not a regular file")
        hashes[f"gmes/{name}"] = _sha256_file(path)
    return {
        "runtime_source_files_sha256": hashes,
        "runtime_source_sha256": _hash_pairs(hashes),
    }


def _source_buffer_digest(simulation: gmes.TorchSimulation) -> str:
    digest = hashlib.sha256()
    for name, value in simulation.sources.named_buffers():
        host = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(repr(tuple(value.shape)).encode())
        digest.update(host.numpy().tobytes())
    return digest.hexdigest()


def _field_buffer_contract(simulation: gmes.TorchSimulation) -> list[dict[str, object]]:
    records = []
    for name in FIELD_NAMES:
        value = simulation.state.field(name)
        records.append(
            {
                "name": name,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "stride": list(value.stride()),
            }
        )
    return records


def _memory_sources(
    spec: Mapping[str, object], sources: Sequence[object]
) -> tuple[object, ...]:
    if sources:
        return tuple(sources)
    del spec
    return (
        gmes.PointSource(
            gmes.Continuous(0.2, phase=0.3, width=1), (0, 0, 0), gmes.Ex, amp=0.25
        ),
        gmes.PointSource(gmes.Bandpass(0.3, 0.1), (0, 0, 0), gmes.Jx, amp=0.1),
    )


@contextmanager
def _capturing_output_code(callback: Any):
    from torch._inductor.graph import GraphLowering

    original = GraphLowering.save_output_code
    GraphLowering.save_output_code = callback
    try:
        yield
    finally:
        GraphLowering.save_output_code = original


def capture_compiled_wrapper_audit(
    *,
    case: str,
    device: str,
    precision: str,
    warmup_steps: int,
    cache_directory: Path,
    output_directory: Path,
) -> dict[str, object]:
    """Capture actual child-process CUDA compilation into a new audit bundle.

    The caller must create a new process and set ``TORCHINDUCTOR_CACHE_DIR`` to
    the supplied empty private directory before importing or constructing any
    compiled simulation.  This function intentionally never resets Dynamo.
    """
    cache = Path(cache_directory).resolve(strict=True)
    output = Path(output_directory).resolve()
    if (
        cache.is_symlink()
        or not cache.is_dir()
        or any(cache.iterdir())
        or os.environ.get("TORCHINDUCTOR_CACHE_DIR") != str(cache)
    ):
        raise ValueError("capture requires an empty explicit Inductor cache directory")
    if warmup_steps < 1 or precision not in {"float32", "float64"}:
        raise ValueError("capture warmup or precision is invalid")
    requested = torch.device(device)
    if (
        requested.type != "cuda"
        or not torch.cuda.is_available()
        or torch.version.cuda is None
        or not torch.__version__.startswith("2.13.")
        or "+cu" not in torch.__version__
    ):
        raise ValueError(
            "capture supports only an available pinned Torch 2.13 CUDA build"
        )
    manifest = load_manifest(MANIFEST)
    spec, space, geometry, sources, bloch = _build_case(case, manifest)
    sources = _memory_sources(spec, sources)
    simulation = gmes.TorchSimulation(
        space=space,
        geometry=geometry,
        sources=sources,
        bloch=bloch,
        runtime=gmes.TorchRuntimeConfig(
            device=device,
            precision=precision,
            compile_policy="compile",
            cpu_threads=1,
            cpu_interop_threads=1,
        ),
    )
    _initialize_fields(
        simulation, manifest["reference"]["seed"], manifest["reference"]["field_scale"]
    )
    active_region: list[str | None] = [None]
    seen_regions: set[str] = set()
    sources_by_region: list[dict[str, object]] = []
    original_dispatch = simulation._run_compute_region

    def dispatch(name: Any, function: Any) -> None:
        active_region[0] = str(name)
        seen_regions.add(str(name))
        try:
            original_dispatch(name, function)
        finally:
            active_region[0] = None

    def save_output_code(source: str) -> None:
        sources_by_region.append({"region": active_region[0], "source": source})

    simulation._run_compute_region = dispatch  # type: ignore[method-assign]
    try:
        with _capturing_output_code(save_output_code):
            simulation.advance(warmup_steps)
        _synchronize(simulation.device)
    finally:
        simulation._run_compute_region = original_dispatch  # type: ignore[method-assign]
    diagnostics = simulation.diagnostics()
    if diagnostics["compiled_region_topology"] != PINNED_CUDA_TOPOLOGY:
        raise ValueError("capture does not have the pinned CUDA region topology")
    missing_dispatches = sorted(set(COMPILED_REGIONS) - seen_regions)
    if missing_dispatches:
        raise ValueError(
            "capture did not execute compiled regions: " + ", ".join(missing_dispatches)
        )
    contract = {
        "case": spec["name"],
        "case_descriptor_sha256": _sha256_bytes(_canonical_json(spec).encode()),
        "compile_cache_key": simulation.compile_cache_key,
        "compiled_region_topology": diagnostics["compiled_region_topology"],
        "device": str(simulation.device),
        "field_buffers": _field_buffer_contract(simulation),
        "plan_identity": simulation.plan_identity,
        "precision": precision,
        "required_regions": list(COMPILED_REGIONS),
        "source_buffer_sha256": _source_buffer_digest(simulation),
        "torch_version": torch.__version__,
        **_runtime_source_contract(),
    }
    audit = audit_compiled_wrapper_sources(sources=sources_by_region, contract=contract)
    for wrapper, source in zip(audit["wrappers"], sources_by_region, strict=True):
        wrapper["_source_bytes"] = str(source["source"]).encode()
    bundle = _write_bundle(output, audit)
    return {
        "schema": SCHEMA,
        "scope": SCOPE,
        "status": audit["status"],
        "verified": audit["verified"],
        "full_domain_output_candidates": audit["full_domain_output_candidates"],
        "reasons": audit["reasons"],
        "contract": contract,
        "bundle": bundle,
    }


def main() -> None:
    """Run only inside an externally isolated CUDA child process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--precision", choices=("float32", "float64"), required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--cache-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    result = capture_compiled_wrapper_audit(
        case=args.case,
        device=args.device,
        precision=args.precision,
        warmup_steps=args.warmup_steps,
        cache_directory=args.cache_directory,
        output_directory=args.output_directory,
    )
    print(_canonical_json(result))
    raise SystemExit(0 if result["verified"] else 1)


if __name__ == "__main__":
    main()
