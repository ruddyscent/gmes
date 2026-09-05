"""Verify documentation coverage for Python-owned GMES APIs."""

import ast
import importlib
import inspect
import pydoc
import unittest
from pathlib import Path

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


class DocstringCoverageTest(unittest.TestCase):
    """Check tracked modules and the supported public API boundary."""

    def assert_documented(self, label, value):
        """Assert that inspect resolves a nonempty docstring for a value."""
        self.assertIsNotNone(inspect.getdoc(value), f"{label} has no docstring")

    def test_source_modules_have_docstrings(self):
        """Require a module docstring in every tracked Python source module."""
        for module in SOURCE_MODULES:
            path = PACKAGE_ROOT / f"{module}.py"
            with self.subTest(module=module):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                self.assertIsNotNone(
                    ast.get_docstring(tree),
                    f"gmes.{module} has no module docstring",
                )

    def test_public_exports_have_docstrings(self):
        """Require docs for package exports and module-level __all__ exports."""
        exports = [(f"gmes.{name}", getattr(gmes, name)) for name in gmes.__all__]
        for module_name in EXPORT_MODULES:
            module = importlib.import_module(f"gmes.{module_name}")
            exports.extend(
                (f"gmes.{module_name}.{name}", getattr(module, name))
                for name in module.__all__
            )

        for label, value in exports:
            if _source_file(value) not in SOURCE_FILES:
                # Third-party objects are outside the Python-owned
                # documentation boundary.  Canonical constants are covered by
                # the tracked module-docstring check above.
                continue
            with self.subTest(export=label):
                self.assert_documented(label, value)
            if inspect.isclass(value):
                self._assert_public_members_documented(label, value)

    def test_supported_extension_hooks_have_docstrings(self):
        """Require docs for legacy subclass hooks used by custom extensions."""
        for value in (Material, Src, SrcTime):
            label = f"{value.__module__}.{value.__qualname__}"
            with self.subTest(hook=label):
                self.assert_documented(label, value)
                self._assert_public_members_documented(label, value)

    def test_pydoc_smoke(self):
        """Render representative modules and primary entry points with pydoc."""
        values = (
            gmes,
            gmes.constant,
            importlib.import_module("gmes.torch_fdtd"),
            gmes.TorchSimulation,
        )
        for value in values:
            with self.subTest(value=value):
                rendered = pydoc.render_doc(value, renderer=pydoc.plaintext)
                summary = inspect.getdoc(value).splitlines()[0]
                self.assertIn(summary, rendered)

    def _assert_public_members_documented(self, label, cls):
        for name, member in inspect.getmembers_static(cls):
            if name.startswith("_") and name != "__init__":
                continue
            target = _member_target(member)
            if not (inspect.isroutine(target) or isinstance(member, property)):
                continue
            if _source_file(target) not in SOURCE_FILES:
                continue
            with self.subTest(member=f"{label}.{name}"):
                self.assert_documented(f"{label}.{name}", member)


if __name__ == "__main__":
    unittest.main()
