"""Verify Python 3.14 annotations for the supported Python-owned API."""

import annotationlib
import importlib
import inspect
import unittest
from pathlib import Path
from typing import Any

import gmes
from gmes.pygeom import Material
from gmes.source import Src, SrcTime

PACKAGE_ROOT = Path(gmes.__file__).resolve().parent
SOURCE_MODULES = (
    "__init__",
    "fdtd",
    "file_io",
    "geometry",
    "material",
    "pw_source",
    "pygeom",
    "show",
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
            target = inspect.unwrap(target)
            if _source_file(target) in SOURCE_FILES:
                yield f"{label}.{name}", target


class AnnotationCoverageTest(unittest.TestCase):
    """Check completeness, resolvability, and Python 3.14 semantics."""

    def test_public_annotations_are_complete_and_resolvable(self):
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
                with self.subTest(callable=target_label):
                    forward = annotationlib.get_annotations(
                        target, format=annotationlib.Format.FORWARDREF
                    )
                    values = annotationlib.get_annotations(
                        target, format=annotationlib.Format.VALUE
                    )
                    self.assertEqual(set(forward), set(values))
                    signature = inspect.signature(target)
                    for parameter in signature.parameters.values():
                        if parameter.name in {"self", "cls"}:
                            continue
                        self.assertIsNot(
                            parameter.annotation,
                            inspect.Signature.empty,
                            f"{target_label}.{parameter.name} is untyped",
                        )
                        self.assertIsNot(
                            values[parameter.name],
                            Any,
                            f"{target_label}.{parameter.name} exposes Any",
                        )
                    self.assertIsNot(
                        signature.return_annotation,
                        inspect.Signature.empty,
                        f"{target_label} has no return annotation",
                    )
                    self.assertIsNot(
                        values["return"], Any, f"{target_label} returns Any"
                    )
        self.assertGreater(len(checked), 100)

    def test_source_modules_use_python_314_deferred_annotations(self):
        for module_name in SOURCE_MODULES:
            path = PACKAGE_ROOT / f"{module_name}.py"
            with self.subTest(module=module_name):
                self.assertNotIn(
                    "from __future__ import annotations",
                    path.read_text(encoding="utf-8"),
                )


if __name__ == "__main__":
    unittest.main()
