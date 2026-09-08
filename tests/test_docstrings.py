"""Verify documentation coverage for Python-owned GMES APIs."""

import ast
import importlib
import inspect
import pydoc
from pathlib import Path

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
SOURCE_FILES = {(PACKAGE_ROOT / f"{module}.py").resolve() for module in SOURCE_MODULES}
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


def _member_target(member):
    """Return the callable that owns a class-member implementation."""
    if isinstance(member, (classmethod, staticmethod)):
        return member.__func__
    if isinstance(member, property):
        return member.fget
    return member


def _documented_targets(label, value):
    yield label, value
    if not inspect.isclass(value):
        return
    for name, member in inspect.getmembers_static(value):
        if name.startswith("_") and name != "__init__":
            continue
        target = _member_target(member)
        if not (inspect.isroutine(target) or isinstance(member, property)):
            continue
        if _source_file(target) in SOURCE_FILES:
            yield f"{label}.{name}", member


def _export_cases():
    exports = [(f"gmes.{name}", getattr(gmes, name)) for name in gmes.__all__]
    for module_name in EXPORT_MODULES:
        module = importlib.import_module(f"gmes.{module_name}")
        exports.extend(
            (f"gmes.{module_name}.{name}", getattr(module, name))
            for name in module.__all__
        )
    for label, value in exports:
        if _source_file(value) in SOURCE_FILES:
            yield from _documented_targets(label, value)


_EXPORT_CASES = tuple(_export_cases())
_HOOK_CASES = tuple(
    case
    for value in (Material, Src, SrcTime)
    for case in _documented_targets(f"{value.__module__}.{value.__qualname__}", value)
)


class TestDocstringCoverage:
    """Check tracked modules and the supported public API boundary."""

    @pytest.mark.parametrize("module", SOURCE_MODULES, ids=SOURCE_MODULES)
    def test_source_modules_have_docstrings(self, module):
        """Require a module docstring in every tracked Python source module."""
        path = PACKAGE_ROOT / f"{module}.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        assert (
            ast.get_docstring(tree) is not None
        ), f"gmes.{module} has no module docstring"

    @pytest.mark.parametrize(
        ("label", "value"), _EXPORT_CASES, ids=[label for label, _ in _EXPORT_CASES]
    )
    def test_public_exports_have_docstrings(self, label, value):
        """Require docs for package exports, members, and module-level exports."""
        assert inspect.getdoc(value) is not None, f"{label} has no docstring"

    @pytest.mark.parametrize(
        ("label", "value"), _HOOK_CASES, ids=[label for label, _ in _HOOK_CASES]
    )
    def test_supported_extension_hooks_have_docstrings(self, label, value):
        """Require docs for legacy subclass hooks used by custom extensions."""
        assert inspect.getdoc(value) is not None, f"{label} has no docstring"

    @pytest.mark.parametrize(
        "value",
        (
            gmes,
            gmes.constant,
            importlib.import_module("gmes.torch_fdtd"),
            gmes.TorchSimulation,
        ),
        ids=("package", "constants", "torch-fdtd", "simulation"),
    )
    def test_pydoc_smoke(self, value):
        """Render representative modules and primary entry points with pydoc."""
        rendered = pydoc.render_doc(value, renderer=pydoc.plaintext)
        summary = inspect.getdoc(value).splitlines()[0]
        assert summary in rendered
