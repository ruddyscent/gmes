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
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path, PurePosixPath
from types import CodeType, MethodType, ModuleType
from typing import Any

import torch

import gmes
import gmes.torch_fdtd as _torch_fdtd
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


def _pointer_name(node: ast.AST, parameters: set[str]) -> str | None:
    """Return the one pointer parameter used by a Triton pointer expression."""
    names = {
        item.id
        for item in ast.walk(node)
        if isinstance(item, ast.Name) and item.id in parameters
    }
    return next(iter(names)) if len(names) == 1 else None


def _assignment_values(function: ast.FunctionDef) -> dict[str, ast.AST]:
    """Return simple local assignments in one emitted Triton function."""
    result = {}
    repeated = set()
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name = node.targets[0].id
            if name in result:
                repeated.add(name)
            else:
                result[name] = node.value
    for name in repeated:
        result.pop(name, None)
    return result


def _direct_load_pointer(
    node: ast.AST,
    assignments: Mapping[str, ast.AST],
    parameters: set[str],
    depth: int = 0,
) -> str | None:
    """Resolve only an untransformed ``tl.load`` value through local aliases."""
    if depth > 32:
        return None
    if isinstance(node, ast.Name) and node.id in assignments:
        return _direct_load_pointer(
            assignments[node.id], assignments, parameters, depth + 1
        )
    if (
        isinstance(node, ast.Call)
        and _call_name(node.func) in {"tl.load", "triton.language.load"}
        and node.args
    ):
        return _pointer_name(node.args[0], parameters)
    return None


def _load_count_and_compute(
    node: ast.AST,
    assignments: Mapping[str, ast.AST],
    parameters: set[str],
    depth: int = 0,
) -> tuple[int, bool, frozenset[str]] | None:
    """Conservatively identify real work in one emitted Triton store value."""
    if depth > 32:
        return None
    if isinstance(node, ast.IfExp):
        # A ternary can select an unchanged input.  It is not arithmetic
        # provenance without an explicit arithmetic payload on every path.
        return None
    if isinstance(node, ast.Name):
        if node.id in assignments:
            return _load_count_and_compute(
                assignments[node.id], assignments, parameters, depth + 1
            )
        return (0, False, frozenset())
    if isinstance(node, ast.Constant):
        return (0, False, frozenset())
    if isinstance(node, ast.Call) and _call_name(node.func) in {
        "tl.load",
        "triton.language.load",
    }:
        if not node.args or _pointer_name(node.args[0], parameters) is None:
            return None
        return (1, False, frozenset({_pointer_name(node.args[0], parameters)}))
    values = [
        child for child in ast.iter_child_nodes(node) if isinstance(child, ast.expr)
    ]
    child_results = [
        _load_count_and_compute(child, assignments, parameters, depth + 1)
        for child in values
    ]
    if any(item is None for item in child_results):
        return None
    loads = sum(item[0] for item in child_results if item is not None)
    compute = any(item[1] for item in child_results if item is not None)
    pointers = frozenset().union(
        *(item[2] for item in child_results if item is not None)
    )
    if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare)):
        compute = True
    elif isinstance(node, ast.Call):
        # A selector can inspect both inputs while storing either one unchanged;
        # an opaque call likewise does not establish arithmetic provenance.  Keep
        # both fail-closed unless the stored payload's arithmetic is explicit in
        # the supported AST grammar above.
        name = _call_name(node.func)
        if name not in {
            "tl.load",
            "triton.language.load",
            "tl.store",
            "triton.language.store",
        }:
            return None
    return (loads, compute, pointers)


def _kernel_output_provenance(
    function: ast.FunctionDef, parameters: tuple[str, ...]
) -> dict[str, dict[str, str]]:
    """Classify only direct, arithmetic, or unknown emitted Triton stores."""
    parameter_set = set(parameters)
    assignments = _assignment_values(function)
    outputs: dict[str, dict[str, str]] = {}

    def visit(node: ast.AST, conditional: bool = False) -> None:
        branch = conditional or isinstance(
            node, (ast.If, ast.For, ast.While, ast.Try, ast.Match)
        )
        if (
            isinstance(node, ast.Call)
            and _call_name(node.func) in {"tl.store", "triton.language.store"}
            and len(node.args) >= 2
        ):
            pointer = _pointer_name(node.args[0], parameter_set)
            if pointer is not None:
                direct = _direct_load_pointer(node.args[1], assignments, parameter_set)
                summary = _load_count_and_compute(
                    node.args[1], assignments, parameter_set
                )
                if branch:
                    provenance = {"kind": "unknown"}
                elif direct is not None:
                    provenance = {"kind": "identity-input", "input": direct}
                elif (
                    summary is not None
                    and summary[0] >= 2
                    and summary[1]
                    and len(summary[2]) >= 2
                ):
                    provenance = {"kind": "computed"}
                else:
                    provenance = {"kind": "unknown"}
                previous = outputs.get(pointer)
                outputs[pointer] = (
                    provenance
                    if previous is None or previous == provenance
                    else {"kind": "unknown"}
                )
        for child in ast.iter_child_nodes(node):
            visit(child, branch)

    visit(function)
    return outputs


def _kernel_definitions(
    tree: ast.AST, reasons: list[str]
) -> dict[str, dict[str, object]]:
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
        parameters = tuple(argument.arg for argument in functions[0].args.args)
        kernels[node.targets[0].id] = {
            "parameters": parameters,
            "outputs": _kernel_output_provenance(functions[0], parameters),
        }
    return kernels


def _argument_layouts(
    nodes: Sequence[ast.AST],
) -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    result = {}
    for node in nodes:
        if not (
            isinstance(node, ast.Call)
            and _call_name(node.func) == "assert_size_stride"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
        ):
            continue
        shape = _shape_literal(node.args[1])
        stride = _shape_literal(node.args[2] if len(node.args) >= 3 else None)
        if shape is not None and stride is not None and len(shape) == len(stride):
            result[node.args[0].id] = (shape, stride)
    return result


def _execution_scope_nodes(nodes: Sequence[ast.AST]) -> tuple[ast.AST, ...]:
    """Return nodes in one execution scope without descending into nested scopes."""
    result = []
    pending = list(reversed(nodes))
    scope_nodes = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
    while pending:
        node = pending.pop()
        if isinstance(node, scope_nodes):
            continue
        result.append(node)
        children = [
            child
            for child in ast.iter_child_nodes(node)
            if not isinstance(child, scope_nodes)
        ]
        pending.extend(reversed(children))
    return tuple(result)


def _execution_scope(
    tree: ast.Module, reasons: list[str]
) -> tuple[tuple[ast.AST, ...], bool] | None:
    """Select a bound production entrypoint or an explicit legacy flat call.

    A production entrypoint receives only scoped diagnostics: its whole-wrapper
    coverage remains incomplete even when the selected body has no other reason.
    """
    runners = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Runner"
    ]
    calls = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "call"
    ]
    if runners:
        reasons.append("wrapper-execution-scope-incomplete")
        if len(runners) != 1 or calls:
            reasons.append("wrapper-execution-scope-ambiguous")
            return None
        runner_calls = [
            node
            for node in runners[0].body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "call"
        ]
        if len(runner_calls) != 1:
            reasons.append("wrapper-execution-scope-ambiguous")
            return None
        bindings = [
            (index, node)
            for index, node in enumerate(tree.body)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"runner", "call"}
        ]
        runner_bindings = [
            item for item in bindings if item[1].targets[0].id == "runner"
        ]
        call_bindings = [item for item in bindings if item[1].targets[0].id == "call"]
        if not runner_bindings or not call_bindings:
            reasons.append("wrapper-execution-entry-missing")
            return None
        if len(runner_bindings) != 1 or len(call_bindings) != 1:
            reasons.append("wrapper-execution-entry-rebound")
            return None
        runner_index, runner_binding = runner_bindings[0]
        call_index, call_binding = call_bindings[0]
        runner_value = runner_binding.value
        call_value = call_binding.value
        if (
            runner_index >= call_index
            or not isinstance(runner_value, ast.Call)
            or not isinstance(runner_value.func, ast.Name)
            or runner_value.func.id != "Runner"
            or not isinstance(call_value, ast.Attribute)
            or not isinstance(call_value.value, ast.Name)
            or call_value.value.id != "runner"
            or call_value.attr != "call"
        ):
            reasons.append("wrapper-execution-entry-unresolved")
            return None
        return _execution_scope_nodes(runner_calls[0].body), True
    if calls:
        if len(calls) != 1:
            reasons.append("wrapper-execution-scope-ambiguous")
            return None
        return tuple(ast.walk(tree)), False
    return tuple(ast.walk(tree)), False


def _has_unresolved_execution_call(
    nodes: Sequence[ast.AST], kernels: Mapping[str, object]
) -> bool:
    """Reject reachable call paths that this bounded reader does not analyze."""
    known_calls = {
        *KNOWN_ALLOCATORS,
        *KNOWN_ALIAS_FACTORIES,
        "assert_alignment",
        "assert_size_stride",
        "copy_if_misaligned",
        "get_raw_stream",
        "grid",
        "torch.cuda._DeviceGuard",
        "torch.cuda.set_device",
    }
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name in known_calls or (
            name is not None and name.startswith("extern_kernels.")
        ):
            continue
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in {*COPY_METHODS, "clear"}:
                continue
            if (
                node.func.attr == "run"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in kernels
            ):
                continue
        return True
    return False


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


def _is_possible_field_layout(
    shape: object,
    stride: object,
    field_layouts: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
) -> bool | None:
    """Return whether a resolved source can be a contracted field layout."""
    if not isinstance(shape, tuple) or not isinstance(stride, tuple):
        return None
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
    argument_layouts: Mapping[str, tuple[tuple[int, ...], tuple[int, ...]]],
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
                argument_layouts=argument_layouts,
                depth=depth + 1,
            )
        if DIRECT_ARGUMENT.fullmatch(node.id):
            layout = argument_layouts.get(node.id)
            return {
                "kind": "direct-input",
                "shape": None if layout is None else layout[0],
                "stride": None if layout is None else layout[1],
                "name": node.id,
            }
    if isinstance(node, ast.Call) and _call_name(node.func) in KNOWN_ALIAS_FACTORIES:
        if node.args:
            return _resolved_value(
                node.args[0],
                allocations=allocations,
                aliases=aliases,
                argument_layouts=argument_layouts,
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
        "non_field_output_exclusions": [],
        "reasons": [],
    }
    reasons = result["reasons"]
    candidates = result["full_domain_output_candidates"]
    exclusions = result["non_field_output_exclusions"]
    assert isinstance(reasons, list) and isinstance(candidates, list)
    assert isinstance(exclusions, list)
    if not isinstance(region, str) or not region:
        reasons.append("missing-region-label")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        reasons.append("wrapper-is-unparseable")
        return result
    kernels = _kernel_definitions(tree, reasons)
    scope = _execution_scope(tree, reasons)
    if scope is None:
        return result
    scope_nodes, restricted_scope = scope
    if restricted_scope and _has_unresolved_execution_call(scope_nodes, kernels):
        reasons.append("wrapper-execution-call-unresolved")
    argument_layouts = _argument_layouts(scope_nodes)
    allocations: dict[str, Mapping[str, object]] = {}
    aliases: dict[str, ast.AST] = {}
    for node in scope_nodes:
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
                allocations[target] = {"name": target, **allocation}
                if allocation["shape"] is None:
                    reasons.append(f"allocation-shape-unknown:{target}")
                continue
            if _is_unknown_allocator(node.value):
                reasons.append(f"allocation-factory-unknown:{target}")
                continue
        if isinstance(node.value, (ast.Name, ast.Call)):
            aliases[target] = node.value
    launch_count = 0
    for node in scope_nodes:
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
        kernel = kernels.get(kernel_name)
        if kernel is None:
            reasons.append(f"launch-kernel-unknown:{kernel_name}")
            continue
        parameters = kernel.get("parameters")
        provenance_by_pointer = kernel.get("outputs")
        if not isinstance(parameters, tuple) or not isinstance(
            provenance_by_pointer, dict
        ):
            reasons.append(f"launch-kernel-metadata-invalid:{kernel_name}")
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
                argument_layouts=argument_layouts,
            )
            provenance = provenance_by_pointer.get(parameter, {"kind": "unknown"})
            if not isinstance(provenance, dict):
                provenance = {"kind": "unknown"}
            outputs.append(
                {
                    "parameter": parameter,
                    "role": role,
                    "payload_provenance": provenance,
                    **resolved,
                }
            )
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
                    event = {
                        "kernel": kernel_name,
                        "parameter": parameter,
                        "factory": resolved["factory"],
                        "reused_storage": resolved["reused_storage"],
                        "shape": list(resolved["shape"]),
                        "stride": list(resolved["stride"]),
                    }
                    kind = provenance.get("kind")
                    if kind == "computed":
                        candidates.append(
                            {
                                "classification": "arithmetic-produced-field-workspace",
                                **event,
                            }
                        )
                    elif kind == "identity-input":
                        input_pointer = provenance.get("input")
                        try:
                            input_index = parameters.index(input_pointer)
                        except TypeError, ValueError:
                            input_index = -1
                        source_value = (
                            _resolved_value(
                                node.args[input_index],
                                allocations=allocations,
                                aliases=aliases,
                                argument_layouts=argument_layouts,
                            )
                            if 0 <= input_index < len(node.args)
                            else {"kind": "unknown"}
                        )
                        field_source = _is_possible_field_layout(
                            source_value.get("shape"),
                            source_value.get("stride"),
                            field_layouts,
                        )
                        candidates.append(
                            {
                                "classification": (
                                    "field-identity-copy"
                                    if field_source
                                    else "field-identity-source-unproven"
                                ),
                                "source": source_value,
                                **event,
                            }
                        )
                        reasons.append(
                            f"field-identity-copy:{kernel_name}:{parameter}"
                            if field_source
                            else f"field-identity-source-unproven:{kernel_name}:{parameter}"
                        )
                    else:
                        candidates.append(
                            {
                                "classification": "field-producer-unknown",
                                **event,
                            }
                        )
                        reasons.append(
                            f"field-producer-unknown:{kernel_name}:{parameter}"
                        )
                else:
                    exclusions.append(
                        {
                            "classification": "non-field-layout-output",
                            "kernel": kernel_name,
                            "parameter": parameter,
                            "factory": resolved["factory"],
                            "reused_storage": resolved["reused_storage"],
                            "shape": resolved["shape"],
                            "stride": resolved["stride"],
                            "dtype": resolved["dtype"],
                        }
                    )
            elif resolved["kind"] != "direct-input":
                reasons.append(f"launch-output-unmapped:{kernel_name}:{parameter}")
        result["launches"].append(launch)
    for node in scope_nodes:
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
                argument_layouts=argument_layouts,
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
    exclusions = [
        event
        for wrapper in wrappers
        for event in wrapper["non_field_output_exclusions"]
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
        "non_field_output_exclusions": exclusions,
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


def _code_filenames_match(code: CodeType, path: str) -> bool:
    return code.co_filename == path and all(
        _code_filenames_match(child, path)
        for child in code.co_consts
        if isinstance(child, CodeType)
    )


def _returned_module_identity(
    graph: Any, module: Any, source: str
) -> tuple[dict[str, object], list[object]]:
    """Inspect a returned module; this does not establish a subsequent invocation."""
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    if not isinstance(module, ModuleType):
        raise ValueError("returned wrapper is not a module")
    path = Path(graph.cache_path)
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or path.resolve(strict=True) != path
        or str(path) != module.__file__
        or not isinstance(graph.cache_key, str)
        or not graph.cache_key
        or graph.cache_key != module.key
        or path.read_bytes() != source.encode("utf-8")
    ):
        raise ValueError("returned module source/cache identity differs")
    code = compile(source, str(path), "exec", dont_inherit=True)
    expected_codes: dict[str, CodeType] = {}

    def collect(parent: CodeType) -> None:
        for child in parent.co_consts:
            if isinstance(child, CodeType):
                if child.co_qualname in expected_codes:
                    raise ValueError("ambiguous wrapper code definition")
                expected_codes[child.co_qualname] = child
                collect(child)

    collect(code)
    call = module.call
    if (
        not isinstance(call, MethodType)
        or call.__self__ is not module.runner
        or type(module.runner) is not module.Runner
        or call.__func__ is not module.Runner.call
        or type(module.runner.partitions) is not list
        or module.runner.partitions
        or call.__func__.__globals__ is not vars(module)
        or call.__func__.__code__ != expected_codes.get("Runner.call")
        or not _code_filenames_match(call.__func__.__code__, str(path))
    ):
        raise ValueError("unsupported or substituted wrapper Runner.call")
    keepalive = [
        graph,
        module,
        call,
        call.__self__,
        call.__func__,
        call.__func__.__code__,
    ]
    kernels = []
    functions = []
    names: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef):
            function = vars(module).get(node.name)
            if (
                getattr(function, "__globals__", None) is not vars(module)
                or getattr(function, "__code__", None) != expected_codes.get(node.name)
                or not _code_filenames_match(function.__code__, str(path))
            ):
                raise ValueError("module function differs from emitted code")
            keepalive.extend((function, function.__code__))
            functions.append(
                {
                    "name": node.name,
                    "function_id": id(function),
                    "code_id": id(function.__code__),
                }
            )
        if not (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and _call_name(node.value.func) == "async_compile.triton"
        ):
            continue
        args = node.value.args
        if (
            len(node.targets) != 1
            or not isinstance(node.targets[0], ast.Name)
            or len(args) < 2
            or not isinstance(args[0], ast.Constant)
            or not isinstance(args[1], ast.Constant)
            or not isinstance(args[1].value, str)
            or args[0].value != node.targets[0].id
            or node.targets[0].id in names
        ):
            raise ValueError("ambiguous or unsupported Triton declaration")
        name = node.targets[0].id
        names.add(name)
        tuner = vars(module).get(name)
        if not isinstance(tuner, CachingAutotuner) or tuner.fn.__name__ != name:
            raise ValueError("Triton module global is not the resolved autotuner")
        keepalive.extend((tuner, tuner.fn))
        kernels.append(
            {
                "name": name,
                "autotuner_id": id(tuner),
                "function_id": id(tuner.fn),
                "embedded_source_sha256": _sha256_bytes(args[1].value.encode()),
            }
        )
    if not kernels:
        raise ValueError("returned module has no Triton declarations")
    return {
        "source_sha256": _sha256_bytes(source.encode()),
        "cache_path": str(path),
        "module_cache_key": graph.cache_key,
        "graph_id": id(graph),
        "module_id": id(module),
        "globals_id": id(vars(module)),
        "call_id": id(call),
        "receiver_id": id(call.__self__),
        "call_function_id": id(call.__func__),
        "call_code_id": id(call.__func__.__code__),
        "kernels": kernels,
        "module_functions": functions,
    }, keepalive


def _project_returned_modules(
    records: list[dict[str, Any]], labels: dict[int, str] | None = None
) -> list[dict[str, object]]:
    """Export role labels, preserving aliases without publishing process identities."""
    if labels is None:
        labels = {}
    projected = []
    for record in sorted(
        records, key=lambda item: item["construction_context"]["region"]
    ):
        context = record["construction_context"]
        region = context["region"]

        def label(identity: int, role: str, ordinal: int = 0) -> str:
            if identity not in labels:
                labels[identity] = f"{region}.{role}.{ordinal}"
            return labels[identity]

        module = {"source_sha256": record["source_sha256"], "region": region}
        for role in (
            "graph",
            "module",
            "globals",
            "call",
            "receiver",
            "call_function",
            "call_code",
        ):
            module[role] = label(record[role + "_id"], role)
        module["construction_context"] = {
            "simulation": label(context["simulation_id"], "simulation"),
            "dispatch_function": label(
                context["dispatch_function_id"], "dispatch_function"
            ),
        }
        module["kernels"] = [
            {
                "declaration": f"{region}.declaration.{index}",
                "autotuner": label(kernel["autotuner_id"], "autotuner", index),
                "function": label(kernel["function_id"], "kernel_function", index),
                "embedded_source_sha256": kernel["embedded_source_sha256"],
            }
            for index, kernel in enumerate(record["kernels"])
        ]
        module["module_functions"] = [
            {
                "function": label(function["function_id"], "module_function", index),
                "code": label(function["code_id"], "module_function_code", index),
            }
            for index, function in enumerate(record["module_functions"])
        ]
        projected.append(module)
    return projected


class _CaseProducerReceipt:
    """Retain local constructor receipts without authenticating an Inductor cache."""

    def __init__(
        self,
        spec: Mapping[str, object],
        space: object,
        geometry: Sequence[object],
        sources: Sequence[object],
        bloch: Sequence[float] | None,
        runtime: object,
    ):
        self.spec = _canonical_json(spec)
        self.space = space
        self.geometry = tuple(geometry)
        self.sources = sources
        self.bloch = None if bloch is None else tuple(bloch)
        self.runtime = runtime
        self.simulation: object | None = None
        self.plan: object | None = None
        self.plan_components: tuple[object, ...] | None = None
        self.plan_digest: str | None = None
        self.key: str | None = None
        self.preimage: tuple[object, ...] | None = None
        self.preimage_bytes: bytes | None = None
        self.halves: dict[str, tuple[object, object, object, object]] = {}
        self.reasons: list[str] = []
        self._patches: list[tuple[object, str, object]] = []
        self._completed = False

    def _patch(self, owner: object, name: str, factory: Any) -> None:
        original = getattr(owner, name)
        self._patches.append((owner, name, original))
        setattr(owner, name, factory(original))

    def _input_matches(self, kwargs: Mapping[str, object]) -> bool:
        geometry = tuple(kwargs.get("geometry", ()))
        return (
            kwargs.get("space") is self.space
            and len(geometry) == len(self.geometry)
            and all(
                observed is expected
                for observed, expected in zip(geometry, self.geometry, strict=True)
            )
            and kwargs.get("sources") is self.sources
            and (None if kwargs.get("bloch") is None else tuple(kwargs["bloch"]))
            == self.bloch
            and kwargs.get("runtime") is self.runtime
        )

    def __enter__(self) -> _CaseProducerReceipt:
        receipt = self

        def simulation_init(original: Any) -> Any:
            def wrapped(instance: object, *args: object, **kwargs: object) -> object:
                if receipt.simulation is None:
                    receipt.simulation = instance
                    if not receipt._input_matches(kwargs):
                        receipt.reasons.append("producer-constructor-input-differs")
                elif instance is not receipt.simulation:
                    receipt.reasons.append("producer-nested-simulation")
                try:
                    result = original(instance, *args, **kwargs)
                except BaseException:
                    if instance is receipt.simulation:
                        receipt.reasons.append("producer-construction-raised")
                    raise
                if instance is receipt.simulation:
                    receipt._completed = True
                return result

            return wrapped

        def plan_init(original: Any) -> Any:
            def wrapped(
                instance: object,
                component_plans: object,
                *args: object,
                **kwargs: object,
            ) -> object:
                components = tuple(component_plans)  # Preserve the real iterable once.
                result = original(instance, components, *args, **kwargs)
                if receipt.simulation is not None and receipt.plan is None:
                    receipt.plan = instance
                    receipt.plan_components = components
                elif receipt.simulation is not None:
                    receipt.reasons.append("producer-duplicate-plan")
                return result

            return wrapped

        def plan_digest(original: Any) -> Any:
            def wrapped(instance: object) -> object:
                result = original(instance)
                if instance is receipt.simulation:
                    receipt.plan_digest = result
                return result

            return wrapped

        def key_digest(original: Any) -> Any:
            def wrapped(instance: object) -> object:
                result = original(instance)
                if instance is receipt.simulation:
                    preimage = getattr(instance, "_compile_cache_key_preimage", None)
                    if not isinstance(preimage, tuple):
                        receipt.reasons.append("producer-preimage-is-not-tuple")
                    else:
                        raw = repr(preimage).encode()
                        if hashlib.sha256(raw).hexdigest() != result:
                            receipt.reasons.append("producer-preimage-digest-differs")
                        else:
                            receipt.preimage = preimage
                            receipt.preimage_bytes = raw
                            receipt.key = result
                return result

            return wrapped

        def compile_half(original: Any) -> Any:
            def wrapped(
                function: Any,
                runtime: Any,
                device: Any,
                *,
                dynamic: Any,
                disable_cuda_graphs: bool = False,
            ) -> object:
                result = original(
                    function,
                    runtime,
                    device,
                    dynamic=dynamic,
                    disable_cuda_graphs=disable_cuda_graphs,
                )
                receiver = getattr(function, "__self__", None)
                method = getattr(function, "__func__", None)
                region = (
                    "electric_half"
                    if method is _torch_fdtd.TorchSimulation._electric_half_update
                    else (
                        "magnetic_half"
                        if method is _torch_fdtd.TorchSimulation._magnetic_half_update
                        else None
                    )
                )
                if receiver is receipt.simulation and region is not None:
                    if disable_cuda_graphs:
                        return result
                    if region in receipt.halves:
                        receipt.reasons.append("producer-duplicate-half-compile")
                    elif runtime is not receipt.runtime or dynamic is not False:
                        receipt.reasons.append("producer-half-compile-input-differs")
                    else:
                        receipt.halves[region] = (receiver, method, result, runtime)
                return result

            return wrapped

        self._patch(_torch_fdtd.TorchSimulation, "__init__", simulation_init)
        self._patch(_torch_fdtd.TorchSimulationPlan, "__init__", plan_init)
        self._patch(_torch_fdtd.TorchSimulation, "_compute_plan_identity", plan_digest)
        self._patch(
            _torch_fdtd.TorchSimulation, "_compute_compile_cache_key", key_digest
        )
        self._patch(_torch_fdtd, "_compile_fullgraph", compile_half)
        return self

    def __exit__(self, *unused: object) -> bool:
        while self._patches:
            owner, name, original = self._patches.pop()
            setattr(owner, name, original)
        return False

    def finalize(self, simulation: object) -> None:
        if not self._completed or simulation is not self.simulation:
            self.reasons.append("producer-simulation-association-differs")
        if (
            self.plan is None
            or getattr(simulation, "plan", None) is not self.plan
            or getattr(getattr(simulation, "state", None), "plan", None)
            is not self.plan
            or self.plan_components is None
            or len(tuple(getattr(self.plan, "components", {}).values()))
            != len(self.plan_components)
            or any(
                observed is not expected
                for observed, expected in zip(
                    tuple(getattr(self.plan, "components", {}).values()),
                    self.plan_components,
                    strict=True,
                )
            )
        ):
            self.reasons.append("producer-plan-association-differs")
        if (
            self.plan_digest is None
            or getattr(simulation, "plan_identity", None) != self.plan_digest
        ):
            self.reasons.append("producer-plan-digest-differs")
        if (
            self.key is None
            or self.preimage is None
            or self.preimage_bytes is None
            or getattr(simulation, "compile_cache_key", None) != self.key
            or getattr(simulation, "_compile_cache_key_preimage", None)
            is not self.preimage
            or repr(self.preimage).encode() != self.preimage_bytes
        ):
            self.reasons.append("producer-specialization-digest-differs")
        for region in COMPILED_REGIONS:
            item = self.halves.get(region)
            if (
                item is None
                or item[0] is not simulation
                or getattr(simulation, "_" + region, None) is not item[2]
            ):
                self.reasons.append("producer-half-callable-differs:" + region)

    def valid_for(self, simulation: object) -> bool:
        if simulation is not self.simulation:
            return False
        if (
            getattr(simulation, "plan", None) is not self.plan
            or getattr(getattr(simulation, "state", None), "plan", None)
            is not self.plan
        ):
            self.reasons.append("producer-plan-association-differs")
        if (
            getattr(simulation, "compile_cache_key", None) != self.key
            or getattr(simulation, "_compile_cache_key_preimage", None)
            is not self.preimage
            or self.preimage is None
            or self.preimage_bytes is None
            or repr(self.preimage).encode() != self.preimage_bytes
        ):
            self.reasons.append("producer-specialization-digest-differs")
        for region, item in self.halves.items():
            if getattr(simulation, "_" + region, None) is not item[2]:
                self.reasons.append("producer-half-callable-differs:" + region)
        return not self.reasons

    def diagnostic(self) -> dict[str, object]:
        return {
            "status": "incomplete",
            "scope": "constructor-object-plan-and-gmes-specialization-producers-v1",
            "aliases": {
                "constructor": "constructor.0",
                "simulation": "simulation.0",
                "plan": "plan.0",
                "electric_half": "electric_half.0",
                "magnetic_half": "magnetic_half.0",
            },
            "caller_descriptor_sha256": _sha256_bytes(self.spec.encode()),
            "plan_identity": self.plan_digest,
            "gmes_specialization_sha256": self.key,
            "specialization_algorithm": "sha256(repr(tuple).encode())",
            "reasons": sorted(set(self.reasons)),
            "unverified": [
                "inductor-cache-input-authentication",
                "descriptor-to-builder-output-binding",
                "effective-source-transform-provenance",
                "effective-constructor-parameter-binding",
                "producer-callback-cardinality-order",
                "plan-content-at-event",
                "source-file-read-provenance",
                "compiled-argument-mapping",
                "device-execution",
                "historical-execution",
                "global-clone-absence",
            ],
        }


class _ReturnedModuleEvidence:
    """Retain construction observations separately from selection/qualification."""

    def __init__(self, producer_receipt: _CaseProducerReceipt | None = None):
        self.entries: list[tuple[Any, ...]] = []
        self.reasons: list[str] = []
        self.producer_receipt = producer_receipt

    def capture(
        self, graph: Any, module: Any, emissions: list[str], context: tuple[Any, ...]
    ) -> None:
        from benchmarks.torch_selected_launcher_attestation import PINNED_WRAPPER_SHA256

        try:
            if len(emissions) != 1:
                raise ValueError("module lacks one associated source emission")
            record, refs = _returned_module_identity(graph, module, emissions[0])
            simulation, region, function = context
            if (
                self.producer_receipt is not None
                and not self.producer_receipt.valid_for(simulation)
            ):
                raise ValueError("constructor producer receipt is unavailable")
            if region not in COMPILED_REGIONS or function is not getattr(
                simulation, "_" + region
            ):
                raise ValueError(
                    "module construction has no matching simulation dispatcher context"
                )
            if simulation._cuda_graphs:
                raise ValueError("graph replay context is unsupported")
            if any(entry[4][1] == region for entry in self.entries):
                raise ValueError(
                    "multiple returned modules for one region are unsupported"
                )
            record["construction_context"] = {
                "simulation_id": id(simulation),
                "region": region,
                "dispatch_function_id": id(function),
            }
            if record["source_sha256"] != PINNED_WRAPPER_SHA256[region]:
                self.reasons.append("wrapper-pin-mismatch:" + region)
            self.entries.append((graph, module, emissions[0], record, context, refs))
        except AttributeError, TypeError, ValueError, OSError:
            self.reasons.append("returned-module-capture-rejected")

    def diagnostic(self) -> dict[str, object]:
        records = []
        reasons = list(self.reasons)
        producer_valid = True
        if self.producer_receipt is not None:
            simulations = {entry[4][0] for entry in self.entries}
            if len(simulations) != 1:
                producer_valid = False
                self.producer_receipt.reasons.append(
                    "producer-simulation-association-differs"
                )
            else:
                producer_valid = self.producer_receipt.valid_for(
                    next(iter(simulations))
                )
            reasons.extend(self.producer_receipt.reasons)
            if not producer_valid:
                reasons.append("producer-receipt-revalidation-failed")
        for graph, module, source, record, context, _refs in self.entries:
            try:
                current, _ = _returned_module_identity(graph, module, source)
                current["construction_context"] = record["construction_context"]
                simulation, region, function = context
                if (
                    current != record
                    or function is not getattr(simulation, "_" + region)
                    or simulation._cuda_graphs
                ):
                    raise ValueError("returned module or dispatcher identity changed")
                if producer_valid:
                    records.append(record)
            except AttributeError, TypeError, ValueError, OSError:
                reasons.append("returned-module-revalidation-failed")
        if {record["construction_context"]["region"] for record in records} != set(
            COMPILED_REGIONS
        ):
            reasons.append("returned-module-coverage-incomplete")
        return json.loads(
            _canonical_json(
                {
                    "status": "incomplete",
                    "scope": "returned-wrapper-construction-objects-v1",
                    "python_code_matching": "structural-equivalence-and-module-globals-receiver-association",
                    "tuner_inventory": "observed-named-module-globals",
                    "identity_labels": "report-local-region-role-ordinal-with-alias-preservation",
                    "modules": _project_returned_modules(records),
                    "reasons": sorted(set(reasons)),
                    "unverified": [
                        "actual-wrapper-invocation-and-selected-launcher-join",
                        "embedded-source-to-compiled-kernel",
                        "code-object-loader-provenance",
                        "pre-capture-substitution-absence",
                        "source-declaration-to-autotuner-provenance",
                        "case-plan-cache-input-binding",
                        "whole-wrapper-non-target-ownership",
                        "device-execution",
                        "historical-execution",
                        "global-clone-absence",
                    ],
                    **(
                        {"case_producer_receipt": self.producer_receipt.diagnostic()}
                        if self.producer_receipt is not None
                        else {}
                    ),
                }
            )
        )


class _WrapperCallJoin:
    """Join live runner/dispatcher frames to host events without replacing runners."""

    def __init__(
        self, evidence: _ReturnedModuleEvidence, observer: Any, dispatcher: Any
    ):
        self.evidence, self.observer, self.dispatcher = evidence, observer, dispatcher
        self.active: Any = None
        self.recording = False
        self.frames: dict[str, Any] = {}
        self.completed: list[str] = []
        self.events: list[tuple[Any, ...]] = []
        self.reasons: list[str] = []

    def _entry(self, region: str) -> tuple[Any, ...]:
        from benchmarks.torch_selected_launcher_attestation import AttestationError

        entries = [entry for entry in self.evidence.entries if entry[4][1] == region]
        if len(entries) != 1:
            raise AttestationError("wrapper-call-entry-missing-or-ambiguous")
        entry = entries[0]
        try:
            if (
                self.evidence.producer_receipt is not None
                and not self.evidence.producer_receipt.valid_for(entry[4][0])
            ):
                raise ValueError("producer receipt changed")
            current, _ = _returned_module_identity(*entry[:3])
            current["construction_context"] = entry[3]["construction_context"]
            if current != entry[3]:
                raise ValueError("changed")
        except AttributeError, TypeError, ValueError, OSError:
            raise AttestationError("wrapper-call-entry-changed") from None
        return entry

    @contextmanager
    def region(self, name: str, function: Any):
        from benchmarks.torch_selected_launcher_attestation import AttestationError

        try:
            if self.recording:
                if self.active is not None or name in self.completed:
                    raise AttestationError("duplicate-or-nested-wrapper-region")
                entry = self._entry(name)
                simulation, region, captured = entry[4]
                if (
                    simulation is not self.dispatcher.__self__
                    or function is not captured
                    or function is not getattr(simulation, "_" + region)
                    or simulation._cuda_graphs
                ):
                    raise AttestationError("wrapper-dispatch-context-changed")
                self.active = entry
            with self.observer.region(name):
                yield
            if self.recording:
                self._entry(name)
                if name not in self.frames:
                    raise AttestationError("wrapper-frame-not-observed")
                self.completed.append(name)
        except BaseException:
            self.reasons.append("wrapper-region-failed")
            raise
        finally:
            self.active = None

    @contextmanager
    def advance(self):
        from benchmarks.torch_selected_launcher_attestation import AttestationError

        if self.recording or self.completed or self.observer._call_join is not None:
            raise AttestationError("wrapper-call-join-is-single-use")
        self.observer._call_join = self
        self.recording = True
        try:
            with self.observer.attested_interval():
                yield
        except BaseException:
            self.reasons.append("wrapper-advance-failed")
            raise
        finally:
            self.recording = False
            self.observer._call_join = None

    def before_run(self, tuner: Any, caller: Any):
        from benchmarks.torch_selected_launcher_attestation import (
            ELECTRIC_KERNEL,
            MAGNETIC_KERNEL,
            AttestationError,
        )

        try:
            if self.active is None:
                raise ValueError("outside region")
            _graph, module, _source, record, context, _refs = self.active
            simulation, region, function = context
            kernel_records = [
                item for item in record["kernels"] if item["autotuner_id"] == id(tuner)
            ]
            if len(kernel_records) != 1:
                raise ValueError("unowned tuner")
            kernel = kernel_records[0]
            if (
                vars(module).get(kernel["name"]) is not tuner
                or id(tuner.fn) != kernel["function_id"]
                or tuner.fn.__name__ != kernel["name"]
            ):
                raise ValueError("changed tuner")
            allowed_codes = {
                record["call_code_id"],
                *(item["code_id"] for item in record["module_functions"]),
            }
            if (
                caller.f_globals is not vars(module)
                or id(caller.f_code) not in allowed_codes
            ):
                raise ValueError("unowned caller")
            frame = caller
            runner_frame = None
            while frame is not None:
                if (
                    id(frame.f_code) == record["call_code_id"]
                    and frame.f_globals is vars(module)
                    and frame.f_locals.get("self") is module.runner
                ):
                    if runner_frame is not None:
                        raise ValueError("recursive runner")
                    runner_frame = frame
                if (
                    frame.f_code is self.dispatcher.__func__.__code__
                    and frame.f_globals is self.dispatcher.__func__.__globals__
                ):
                    if (
                        frame.f_locals.get("self") is not simulation
                        or frame.f_locals.get("name") != region
                        or frame.f_locals.get("function") is not function
                    ):
                        raise ValueError("foreign dispatcher")
                    break
                frame = frame.f_back
            if frame is None or runner_frame is None:
                raise ValueError("missing live invocation")
            if region in self.frames and self.frames[region] is not runner_frame:
                raise ValueError("multiple runner invocations")
            self.frames[region] = runner_frame
            if kernel["name"] not in {MAGNETIC_KERNEL, ELECTRIC_KERNEL}:
                launchers = [
                    *getattr(tuner, "launchers", ()),
                    getattr(tuner, "_cached_launcher", None),
                ]
                if any(
                    id(item) in self.observer._constructors
                    or id(item) in self.observer._fast_derivations
                    for item in launchers
                ):
                    raise ValueError("target launcher in non-target global")
                return None
            expected_region = (
                "magnetic_half"
                if kernel["name"] == MAGNETIC_KERNEL
                else "electric_half"
            )
            if region != expected_region:
                raise ValueError("cross-region target")
            return (self.active, kernel, runner_frame)
        except AttributeError, TypeError, ValueError:
            self.reasons.append("wrapper-run-ownership-rejected")
            raise AttestationError("wrapper-run-ownership-rejected") from None

    def selected(self, joined: tuple[Any, ...], event: dict[str, Any]) -> None:
        self.events.append((joined, dict(event)))

    def diagnostic(self) -> dict[str, object]:
        reasons = list(self.reasons)
        module_report = self.evidence.diagnostic()
        reasons.extend(module_report["reasons"])
        try:
            self.observer.diagnostic()
        except ValueError:
            reasons.append("selected-launcher-observation-incomplete")
        if set(self.completed) != set(COMPILED_REGIONS) or len(self.events) != 5:
            reasons.append("wrapper-call-join-coverage-incomplete")
        labels: dict[int, str] = {}
        valid_regions = {item["region"] for item in module_report["modules"]}
        modules = _project_returned_modules(
            [
                entry[3]
                for entry in self.evidence.entries
                if entry[4][1] in valid_regions
            ],
            labels,
        )
        events = []
        valid = not any(
            not reason.startswith("wrapper-pin-mismatch:") for reason in reasons
        )
        for (entry, kernel, _frame), event in self.events if valid else ():
            region = entry[4][1]
            projected = {
                "region": region,
                "ordinal": event["ordinal"],
                "branch": event["branch"],
                "cubin_sha256": event["cubin_sha256"],
                "host_returned_normally": True,
                "runner": labels[entry[3]["receiver_id"]],
                "autotuner": labels[kernel["autotuner_id"]],
            }
            for role in ("selected_callable", "parent_callable"):
                identity = event[role + "_id"]
                if identity is not None:
                    projected[role] = labels.setdefault(
                        identity, f"{region}.{role}.{event['ordinal']}"
                    )
            events.append(projected)
        return {
            "status": "incomplete",
            "scope": "live-wrapper-frame-selected-host-event-join-v1",
            "python_code_matching": module_report["python_code_matching"],
            "tuner_inventory": module_report["tuner_inventory"],
            "identity_labels": module_report["identity_labels"],
            "modules": modules,
            "events": events,
            "reasons": sorted(set(reasons)),
            "unverified": [
                *module_report["unverified"],
                "bound-method-alias-invocation-provenance",
                "silent-or-additional-runner-invocations",
                "total-runner-entry-exit-cardinality",
            ],
            "claim": "one-observed-event-bearing-runner-frame-per-region-for-participating-host-calls",
            "instrumentation": "host timing and frame retention changed; no device or memory qualification",
        }


@contextmanager
def _capturing_output_code(callback: Any, module_callback: Any = None):
    from torch._inductor.graph import GraphLowering

    original = GraphLowering.save_output_code
    original_compile = GraphLowering._compile_to_module_lines
    pending: list[list[str]] = []

    def save(source: str) -> None:
        if pending:
            pending[-1].append(source)
        callback(source)

    def compile_module(graph: Any, wrapper_code: Any):
        emissions: list[str] = []
        pending.append(emissions)
        try:
            module = original_compile(graph, wrapper_code)
        finally:
            pending.pop()
        module_callback(graph, module, emissions)
        return module

    GraphLowering.save_output_code = save
    if module_callback is not None:
        GraphLowering._compile_to_module_lines = compile_module
    try:
        yield
    finally:
        GraphLowering.save_output_code = original
        if module_callback is not None:
            GraphLowering._compile_to_module_lines = original_compile


def capture_compiled_wrapper_audit(
    *,
    case: str,
    device: str,
    precision: str,
    warmup_steps: int,
    cache_directory: Path,
    output_directory: Path,
    observe_returned_modules: bool = False,
    observe_launcher_join: bool = False,
) -> dict[str, object]:
    """Capture actual child-process CUDA compilation into a new audit bundle.

    The caller must create a new process and set ``TORCHINDUCTOR_CACHE_DIR`` to
    the supplied empty private directory before importing or constructing any
    compiled simulation.  This function intentionally never resets Dynamo.
    ``observe_returned_modules`` adds incomplete construction-object evidence;
    it does not change allocation-audit qualification or observe kernel selection.
    ``observe_launcher_join`` also requests one additional instrumented advance
    after warmup, joining live Python frames to selected host-launcher events.
    Its frame retention and host overhead are not memory/performance evidence.
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
    runtime = gmes.TorchRuntimeConfig(
        device=device,
        precision=precision,
        compile_policy="compile",
        cpu_threads=1,
        cpu_interop_threads=1,
    )
    producer_receipt = (
        _CaseProducerReceipt(spec, space, geometry, sources, bloch, runtime)
        if observe_returned_modules or observe_launcher_join
        else None
    )
    with producer_receipt if producer_receipt is not None else nullcontext():
        simulation = gmes.TorchSimulation(
            space=space,
            geometry=geometry,
            sources=sources,
            bloch=bloch,
            runtime=runtime,
        )
    if producer_receipt is not None:
        producer_receipt.finalize(simulation)
    _initialize_fields(
        simulation, manifest["reference"]["seed"], manifest["reference"]["field_scale"]
    )
    active_region: list[str | None] = [None]
    active_function: list[Any] = [None]
    module_evidence = (
        _ReturnedModuleEvidence(producer_receipt)
        if observe_returned_modules or observe_launcher_join
        else None
    )
    seen_regions: set[str] = set()
    sources_by_region: list[dict[str, object]] = []
    original_dispatch = simulation._run_compute_region
    from benchmarks.torch_selected_launcher_attestation import pinned_observer

    observer = pinned_observer() if observe_launcher_join else None
    call_join = (
        _WrapperCallJoin(module_evidence, observer, original_dispatch)
        if observer is not None
        else None
    )

    def dispatch(name: Any, function: Any) -> None:
        active_region[0] = str(name)
        active_function[0] = function
        seen_regions.add(str(name))
        try:
            with (
                call_join.region(name, function)
                if call_join is not None
                else nullcontext()
            ):
                original_dispatch(name, function)
        finally:
            active_region[0] = None
            active_function[0] = None

    def save_output_code(source: str) -> None:
        sources_by_region.append({"region": active_region[0], "source": source})

    def save_module(graph: Any, module: Any, emissions: list[str]) -> None:
        module_evidence.capture(
            graph, module, emissions, (simulation, active_region[0], active_function[0])
        )

    simulation._run_compute_region = dispatch  # type: ignore[method-assign]
    try:
        with ExitStack() as stack:
            if observer is not None:
                stack.enter_context(observer)
            stack.enter_context(
                _capturing_output_code(
                    save_output_code,
                    save_module if module_evidence is not None else None,
                )
            )
            simulation.advance(warmup_steps)
            _synchronize(simulation.device)
            if call_join is not None:
                with call_join.advance():
                    simulation.advance(1)
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
        **(
            {"returned_module_evidence": module_evidence.diagnostic()}
            if module_evidence is not None
            else {}
        ),
        **(
            {"wrapper_call_join": call_join.diagnostic()}
            if call_join is not None
            else {}
        ),
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
