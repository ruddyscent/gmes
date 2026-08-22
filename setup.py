#!/usr/bin/env python
# -*- coding: utf-8 -*-

# System imports
import os
import sys
from glob import glob
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.macos_build import (
    MINIMUM_MACOS_VERSION,
    verify_extension_targets,
    verify_wheel_platform_tag,
)

if sys.platform == "darwin":
    os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", MINIMUM_MACOS_VERSION)

# Third-party modules - we depend on numpy for everything
import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_ext import build_ext


class BuildExt(build_ext):
    """Configure native extensions and copy generated SWIG proxies."""

    def build_extensions(self):
        self._configure_openmp()
        super().build_extensions()

    def _configure_openmp(self):
        setting, options = openmp_options()
        if setting == "disabled" or options is None:
            return

        compile_args, link_args = options
        try:
            self._probe_openmp(compile_args, link_args)
        except Exception as error:
            if setting == "required":
                raise RuntimeError(
                    "OpenMP was requested but the compiler or runtime probe failed"
                ) from error
            self.announce(
                f"OpenMP probe failed; building the serial fallback: {error}",
                level=2,
            )
            return

        extension = next(
            item for item in self.extensions if item.name == "gmes._pw_material"
        )
        extension.extra_compile_args.extend(compile_args)
        extension.extra_link_args.extend(link_args)

    def _probe_openmp(self, compile_args, link_args):
        with TemporaryDirectory(prefix="gmes-openmp-") as directory:
            source = Path(directory) / "probe.cc"
            source.write_text(
                "#include <omp.h>\n"
                'extern "C" int gmes_openmp_probe() {\n'
                "  return omp_get_max_threads();\n"
                "}\n"
            )
            objects = self.compiler.compile(
                [str(source)],
                output_dir=directory,
                extra_postargs=compile_args,
            )
            self.compiler.link_shared_object(
                objects,
                str(Path(directory) / "probe.so"),
                extra_postargs=link_args,
            )

    def run(self):
        super().run()
        package_dir = Path(self.build_lib) / "gmes"
        package_dir.mkdir(parents=True, exist_ok=True)
        for module_name in ("constant.py", "pw_material.py"):
            self.copy_file(
                str(Path("gmes") / module_name), str(package_dir / module_name)
            )


def openmp_options():
    """Return the requested OpenMP mode and platform-specific flags."""
    value = os.environ.get("GMES_ENABLE_OPENMP", "auto").strip().lower()
    if value in {"0", "false", "no", "off"}:
        return "disabled", None
    if value in {"1", "true", "yes", "on"}:
        setting = "required"
    elif value == "auto":
        setting = "auto"
    else:
        raise RuntimeError("GMES_ENABLE_OPENMP must be auto, 1/true/on, or 0/false/off")

    if sys.platform.startswith("linux"):
        return setting, (["-fopenmp"], ["-fopenmp"])

    if sys.platform == "darwin":
        configured_prefix = os.environ.get("GMES_OPENMP_PREFIX")
        prefixes = (
            [Path(configured_prefix)]
            if configured_prefix
            else [
                Path("/opt/homebrew/opt/libomp"),
                Path("/usr/local/opt/libomp"),
            ]
        )
        prefix = next(
            (
                candidate
                for candidate in prefixes
                if (candidate / "include" / "omp.h").is_file()
                and (candidate / "lib").is_dir()
            ),
            None,
        )
        if prefix is not None:
            include_directory = prefix / "include"
            library_directory = prefix / "lib"
            return setting, (
                ["-Xpreprocessor", "-fopenmp", f"-I{include_directory}"],
                [
                    f"-L{library_directory}",
                    "-lomp",
                    f"-Wl,-rpath,{library_directory}",
                ],
            )

    if setting == "required":
        raise RuntimeError(
            "OpenMP was requested but no supported runtime was found; "
            "install libomp on macOS or use an OpenMP compiler on Linux"
        )
    return setting, None


class BdistWheel(bdist_wheel):
    """Reject macOS wheels that do not honor the configured target."""

    def run(self):
        if sys.platform == "darwin":
            target = os.environ["MACOSX_DEPLOYMENT_TARGET"]
            verify_wheel_platform_tag(self.get_tag()[2], target)

        super().run()

        if sys.platform == "darwin":
            build_command = self.get_finalized_command("build_ext")
            verify_extension_targets(build_command.get_outputs(), target)


# Obtain the numpy include directory. This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

pw_src_lst = glob("src/pw_*.cc")
pw_src_lst.extend(glob("src/pw_*.i"))
pw_dep_lst = glob("src/pw_*.hh")

# pw_material module
pw_material = Extension(
    name="gmes._pw_material",
    sources=pw_src_lst,
    depends=pw_dep_lst,
    include_dirs=[numpy_include],
    swig_opts=["-c++", "-outdir", "gmes"],
    language="c++",
    extra_compile_args=["-std=c++23"],
    extra_link_args=[],
)

# constant module
constant = Extension(
    name="gmes._constant",
    sources=["src/constant.i", "src/constant.cc"],
    depends=["src/constant.hh"],
    include_dirs=[numpy_include],
    swig_opts=["-c++", "-outdir", "gmes"],
    language="c++",
    extra_compile_args=["-std=c++23"],
)

# pygeom module
pygeom = Extension(
    name="gmes.pygeom", sources=["src/pygeom.pyx"], include_dirs=[numpy_include]
)

# material module
material = Extension(
    name="gmes.material", sources=["src/material.pyx"], include_dirs=[numpy_include]
)

setup(
    ext_modules=cythonize(
        [pw_material, constant, pygeom, material],
        compiler_directives={"language_level": 3},
    ),
    cmdclass={"bdist_wheel": BdistWheel, "build_ext": BuildExt},
)
