#!/usr/bin/env python
# -*- coding: utf-8 -*-

# System imports
from glob import glob
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# Third-party modules - we depend on numpy for everything
import numpy


class BuildExt(build_ext):
    """Copy SWIG's generated proxy modules into the wheel build tree."""

    def run(self):
        super().run()
        package_dir = Path(self.build_lib) / 'gmes'
        package_dir.mkdir(parents=True, exist_ok=True)
        for module_name in ('constant.py', 'pw_material.py'):
            self.copy_file(str(Path('gmes') / module_name), str(package_dir / module_name))

# Obtain the numpy include directory. This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

pw_src_lst = glob('src/pw_*.cc')
pw_src_lst.extend(glob('src/pw_*.i'))
pw_dep_lst = glob('src/pw_*.hh')

# pw_material module
pw_material = Extension(name = 'gmes._pw_material',
                        sources = pw_src_lst,
                        depends = pw_dep_lst,
                        include_dirs = [numpy_include],
                        swig_opts = ['-c++', '-outdir', 'gmes'],
                        language = 'c++',
                        extra_compile_args=['-std=c++17'])

# constant module
constant = Extension(name = 'gmes._constant',
                     sources = ['src/constant.i', 'src/constant.cc'],
                     depends = ['src/constant.hh'],
                     include_dirs = [numpy_include],
                     swig_opts = ['-c++', '-outdir', 'gmes'],
                     language = 'c++',
                     extra_compile_args=['-std=c++17'])

# pygeom module
pygeom = Extension(name = 'gmes.pygeom',
                   sources = ['src/pygeom.pyx'],
                   include_dirs = [numpy_include])

# material module
material = Extension(name = 'gmes.material',
                     sources = ['src/material.pyx'],
                     include_dirs = [numpy_include])

setup(
    ext_modules=cythonize(
        [pw_material, constant, pygeom, material],
        compiler_directives={'language_level': 3},
    ),
    cmdclass={'build_ext': BuildExt},
)
