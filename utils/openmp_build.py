"""Resolve native OpenMP build modes without importing the build backend."""

import os
import sys
from pathlib import Path

from utils.macos_build import select_macos_openmp_prefix


def openmp_options(environment=None, platform=None):
    """Return the requested OpenMP mode, flags, and fallback diagnostic."""
    environment = os.environ if environment is None else environment
    platform = sys.platform if platform is None else platform
    value = environment.get("GMES_ENABLE_OPENMP", "auto").strip().lower()
    if value in {"0", "false", "no", "off"}:
        return "disabled", None, None
    if value in {"1", "true", "yes", "on"}:
        setting = "required"
    elif value == "auto":
        setting = "auto"
    else:
        raise RuntimeError("GMES_ENABLE_OPENMP must be auto, 1/true/on, or 0/false/off")

    if platform.startswith("linux"):
        return setting, (["-fopenmp"], ["-fopenmp"]), None

    if platform == "darwin":
        configured_prefix = environment.get("GMES_OPENMP_PREFIX")
        prefixes = (
            [Path(configured_prefix)]
            if configured_prefix
            else [
                Path("/opt/homebrew/opt/libomp"),
                Path("/usr/local/opt/libomp"),
            ]
        )
        target = environment.get("MACOSX_DEPLOYMENT_TARGET")
        if not target:
            raise RuntimeError("MACOSX_DEPLOYMENT_TARGET must be set on macOS")
        prefix, incompatibilities = select_macos_openmp_prefix(prefixes, target)

        if prefix is not None:
            include_directory = prefix / "include"
            library_directory = prefix / "lib"
            return (
                setting,
                (
                    ["-Xpreprocessor", "-fopenmp", f"-I{include_directory}"],
                    [
                        f"-L{library_directory}",
                        "-lomp",
                        f"-Wl,-rpath,{library_directory}",
                    ],
                ),
                None,
            )

        if incompatibilities:
            diagnostic = "; ".join(incompatibilities)
            if setting == "required":
                raise RuntimeError(f"OpenMP was requested but {diagnostic}")
            return setting, None, diagnostic

    if setting == "required":
        raise RuntimeError(
            "OpenMP was requested but no supported runtime was found; "
            "install libomp on macOS or use an OpenMP compiler on Linux"
        )
    return setting, None, None
