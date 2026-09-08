"""Verify Python 3.14 annotations for the supported Python-owned API."""

import annotationlib
import importlib
import inspect
from pathlib import Path
from typing import Any

import pytest

import gmes
from gmes.pygeom import Material
from gmes.source import Src, SrcTime

PACKAGE_ROOT = Path(gmes.__file__).resolve().parent
SOURCE_MODULES = (
    "__init__",
    "constant",
    "file_io",
    "geometry",
    "material",
    "pygeom",
    "source",
    "torch_dispersive",
    "torch_distributed",
    "torch_dm2",
    "torch_fdtd",
    "torch_output",
    "torch_plan",
    "torch_source",
)
SOURCE_FILES = {(PACKAGE_ROOT / f"{name}.py").resolve() for name in SOURCE_MODULES}
EXPORT_MODULES = (
    "torch_dispersive",
    "torch_distributed",
    "torch_dm2",
    "torch_fdtd",
    "torch_output",
    "torch_plan",
    "torch_source",
)


def _source_file(value):
    """Return a resolved source path when inspect can locate one."""
    try:
        filename = inspect.getsourcefile(value)
    except OSError, TypeError:
        return None
    return Path(filename).resolve() if filename else None


def _callable_targets(label, value):
    """Yield Python-owned public callables covered by one export."""
    if inspect.isroutine(value):
        yield label, value
        return
    if not inspect.isclass(value):
        return
    for name, member in inspect.getmembers_static(value):
        if name.startswith("_") and name != "__init__":
            continue
        targets = []
        if isinstance(member, (classmethod, staticmethod)):
            targets.append(member.__func__)
        elif isinstance(member, property):
            targets.extend(target for target in (member.fget, member.fset) if target)
        elif inspect.isroutine(member):
            targets.append(member)
        for target in targets:
            target_label = f"{label}.{name}"
            if isinstance(member, property):
                accessor = "get" if target is member.fget else "set"
                target_label += f".{accessor}"
            target = inspect.unwrap(target)
            if _source_file(target) in SOURCE_FILES:
                yield target_label, target


def _annotation_cases():
    exports = [(f"gmes.{name}", getattr(gmes, name)) for name in gmes.__all__]
    for module_name in EXPORT_MODULES:
        module = importlib.import_module(f"gmes.{module_name}")
        exports.extend(
            (f"gmes.{module_name}.{name}", getattr(module, name))
            for name in module.__all__
        )
    exports.extend(
        (f"{value.__module__}.{value.__qualname__}", value)
        for value in (Material, Src, SrcTime)
    )

    checked = set()
    for label, value in exports:
        if _source_file(value) not in SOURCE_FILES:
            continue
        for target_label, target in _callable_targets(label, value):
            identity = id(target)
            if identity in checked:
                continue
            checked.add(identity)
            yield target_label, target


_ANNOTATION_CASES = tuple(_annotation_cases())


class TestAnnotationCoverage:
    """Check completeness, resolvability, and Python 3.14 semantics."""

    @pytest.mark.parametrize(
        ("target_label", "target"),
        _ANNOTATION_CASES,
        ids=[label for label, _ in _ANNOTATION_CASES],
    )
    def test_public_annotations_are_complete_and_resolvable(self, target_label, target):
        forward = annotationlib.get_annotations(
            target, format=annotationlib.Format.FORWARDREF
        )
        values = annotationlib.get_annotations(
            target, format=annotationlib.Format.VALUE
        )
        assert set(forward) == set(values), target_label
        signature = inspect.signature(target)
        for parameter in signature.parameters.values():
            if parameter.name in {"self", "cls"}:
                continue
            assert (
                parameter.annotation is not inspect.Signature.empty
            ), f"{target_label}.{parameter.name} is untyped"
            assert (
                values[parameter.name] is not Any
            ), f"{target_label}.{parameter.name} exposes Any"
        assert (
            signature.return_annotation is not inspect.Signature.empty
        ), f"{target_label} has no return annotation"
        assert values["return"] is not Any, f"{target_label} returns Any"

    def test_public_annotation_inventory_is_complete(self):
        assert len(_ANNOTATION_CASES) > 100

    @pytest.mark.parametrize("module_name", SOURCE_MODULES, ids=SOURCE_MODULES)
    def test_source_modules_use_python_314_deferred_annotations(self, module_name):
        path = PACKAGE_ROOT / f"{module_name}.py"
        assert "from __future__ import annotations" not in path.read_text(
            encoding="utf-8"
        )
